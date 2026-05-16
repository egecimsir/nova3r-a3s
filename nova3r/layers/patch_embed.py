# Copyright (c) 2026 Weirong Chen
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union
from .mlp import Mlp

import torch
from torch import Tensor
import torch.nn as nn

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Token3DEmbedMLP(nn.Module):
    def __init__(
        self, dim, mlp_ratio=1.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop=0., embed_channels=None, use_cond_embed=False, use_input_norm=False,
    ):
        super().__init__()
        
        # Optional input normalization (can disable if initialization handles it)
        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.input_norm = norm_layer(dim)
        
        # Use LayerNorm with eps=1e-6 to prevent division by zero
        self.norm = norm_layer(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            # bias=False,
            bias=True
        )

        self.emb_out_channels = 3 * dim
        if embed_channels is None:
            embed_channels = dim
        self.use_cond_embed = use_cond_embed
        if use_cond_embed:
            self.emb_layer = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(
                        embed_channels,
                        self.emb_out_channels,
                        bias=True,
                    ),
                )
        else:
            self.emb_layer = None

    def forward(self, x, cond=None):
        if self.use_cond_embed:
            if self.use_input_norm:
                x = self.input_norm(x)
            cond_embed = self.emb_layer(cond).type(x.dtype)
            shift, scale, gate = cond_embed.chunk(3, dim=-1)
            modulated = modulate(self.norm(x), shift, scale)
            x = x + gate.unsqueeze(1) * self.mlp(modulated)
        else:
            x = x + self.mlp(self.norm(x))
        return x
        