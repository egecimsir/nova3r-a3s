from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from PIL import Image
from omegaconf import OmegaConf
from typing import List, Dict
from pathlib import Path
from torchvision.transforms import transforms as T

from nova3r.modules import AggregatorPts3D
from nova3r.heads.pts3d_decoder import PointJointFMDecoderV2
from nova3r.utils.device import autocast
from nova3r.flow_matching.solver import ODESolver


# Architecture defaults
EMBED_DIM = 1024
TOKEN_DIM = 128
NUM_3D_TOKENS = 768
IMG_NORM = T.Compose([
    T.ToTensor(), 
    T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])
ENC_KW: dict = dict(
    img_size=518, 
    patch_size=14, 
    embed_dim=EMBED_DIM,
    depth=16, 
    num_heads=16, 
    mlp_ratio=4.0,
    num_register_tokens=4,
    qkv_bias=True, 
    proj_bias=True, 
    ffn_bias=True,
    patch_embed="dinov2_vitl14_reg",
    aa_order=["frame", "global", "pts3d"], 
    aa_block_size=1,
    qk_norm=True, 
    rope_freq=100, 
    init_values=0.01,
    num_3d_tokens=NUM_3D_TOKENS, 
    token_dim_3d=EMBED_DIM,
    pos_3d_embed_type="None",
    # OmegaConf wrapper: AggregatorPts3D does attribute access on this config.
    token_3d_embed_config=OmegaConf.create(dict(
        name="Token3DEmbedMLP",
        params=dict(
            dim=EMBED_DIM, 
            mlp_ratio=2.0, 
            use_cond_embed="first"
        ),
    )),
    share_3d_attn=True, 
    depth_3d_stride=2,
    token_dim=TOKEN_DIM,
    detach_vit_token=True, 
    use_image_token=True,
)
DEC_KW: dict = dict(
    dim_in=TOKEN_DIM, 
    dim_model=TOKEN_DIM, 
    output_dim=3,
    num_3d_tokens=NUM_3D_TOKENS,
    has_conf=True, 
    conf_activation="expp1",
    rgb_activation="sigmoid", 
    pts_activation="linear",
    use_sdpa=True, 
    cross_depth=3, 
    self_depth=3,
    num_virtual_tracks=512,
    token_reduce_dim=TOKEN_DIM, 
    share_t=True, 
    norm_mode="median_3",
    target_source="src_complete", 
    query_source="src_complete",
    sample_size=8192, 
    down_resolution=224,
    target_sampling="fps_edge_fast", 
    use_filter=False,
)


class Nova3r(nn.Module):
    """Standalone Stage-2 NOVA3R image-to-3D-points model.

    Re-implements ``nova3r.models.Nova3rImgCond`` so released ``scene_n1`` / ``scene_n2`` checkpoints load via a simple state-dict prefix filter.
    Submodule names mirror upstream (``vggt_aggregator``, ``img_token_proj``,``pts3d_head``) so weights drop in without key remapping. 
    Zero-config: all architecture knobs are the released-checkpoint defaults (``ENC_KW`` / ``DEC_KW`` above).
    """

    ENC_PREFIXES = ("vggt_aggregator.", "img_token_proj.", "token_norm.")
    DEC_PREFIXES = ("pts3d_head.",)

    def __init__(self, device: torch.device | str | None = None):
        super().__init__()
        self.vggt_aggregator = AggregatorPts3D(**ENC_KW)
        # AggregatorPts3D concats pts3d + global tokens -> 2 * embed_dim wide.
        self.img_token_proj = nn.Linear(EMBED_DIM * 2, TOKEN_DIM)
        self.pts3d_head = PointJointFMDecoderV2(**DEC_KW)
        self.detach_vit_token: bool = ENC_KW["detach_vit_token"]

        if device is not None:
            self.to(device)

    def forward(self, images: Tensor, **kwargs) -> dict:
        """
        End-to-end forward. ``images``: ``(B, S, 3, H, W)`` in ``[0, 1]``.
        """
        latents = self.encode(images)["tokens"]
        points = self.decode(latents, images, **kwargs)        
        return points

    def encode(self, images: Tensor, **_) -> dict:
        """
        ``images``: ``(B, S, 3, H, W)`` in ``[0, 1]``.
        """
        _, agg3d_list, _, _ = self.vggt_aggregator(images, detach_vit_token=self.detach_vit_token)
        tokens = self.img_token_proj(agg3d_list[-1])
        return {
            "pointmaps": None, 
            "tokens": tokens
        }

    def decode(
        self,
        tokens: Tensor,
        images: Tensor,
        query_points: Tensor | None = None,
        timestep: Tensor | None = None,
        token_mask: Tensor | None = None,
        **_,
    ) -> dict:
        B, S = images.shape[:2]
        num_views = torch.full((B,), S, device=images.device, dtype=images.dtype)
        pts3d_xyz = self.pts3d_head(
            [tokens], 
            mask=token_mask,
            query_points=query_points, 
            timestep=timestep, 
            num_views=num_views,
        )
        return {"pts3d_xyz": pts3d_xyz}

    @staticmethod
    def _read_state(ckpt_path: Path | str) -> dict:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        return ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    def load_checkpoints(self, enc_ckpt: Path | str, dec_ckpt: Path | str | None = None) -> "Nova3r":
        """
        Load pretrained weights.

        ``enc_ckpt``: Stage-2 checkpoint (``scene_n1`` / ``scene_n2``) — supplies the image encoder, projection, 
            and (when ``dec_ckpt`` is ``None``) the FM decoder too.
            
        ``dec_ckpt`` (optional): Stage-1 AE checkpoint (``scene_ae``) to source the FM decoder from instead.
        """
        enc_state = self._read_state(enc_ckpt)
        dec_state = self._read_state(dec_ckpt) if dec_ckpt is not None else enc_state

        merged = {k: v for k, v in enc_state.items() if k.startswith(self.ENC_PREFIXES)}
        merged.update({k: v for k, v in dec_state.items() if k.startswith(self.DEC_PREFIXES)})

        result = self.load_state_dict(merged, strict=False)
        own = set(self.state_dict())
        missing = [k for k in result.missing_keys if k in own]
        if missing:
            raise RuntimeError(
                f"Nova3r.load_checkpoints: {len(missing)} missing key(s); first few: {missing[:5]}"
            )

        enc_n = sum(1 for k in merged if k.startswith(self.ENC_PREFIXES))
        dec_n = sum(1 for k in merged if k.startswith(self.DEC_PREFIXES))
        enc_name = Path(str(enc_ckpt)).parent.name
        dec_name = Path(str(dec_ckpt)).parent.name if dec_ckpt is not None else enc_name
        print(f"Loaded encoder={enc_n} tensors from {enc_name}, "
              f"decoder={dec_n} tensors from {dec_name}")
        
        return self



def preprocess(image_paths: list[str], resolution: tuple[int, int]) -> list[dict]:
    """Load and resize images to a network-friendly ``(W, H)`` for VGGT (multiples of 14)."""
    target_W, target_H = resolution
    return [
        dict(
            img=IMG_NORM(Image.open(p).convert("RGB").resize((target_W, target_H), Image.LANCZOS))[None],
            true_shape=np.int32([target_H, target_W]), # pyright: ignore[reportArgumentType]
            idx=i, instance=str(i), view_label=f"input_{i}",
        )
        for i, p in enumerate(image_paths)
    ]


@torch.no_grad()
def predict(
    model: Nova3r,
    image_paths: list[str | Path],
    num_queries: int = 20_000,
    fm_step_size: float = 0.04,
    resolution: tuple[int, int] = (518, 392),
    seed: int | None = None,
) -> np.ndarray:
    """End-to-end inference: image paths -> ``(num_queries, 3)`` numpy point cloud.

    ``resolution`` is ``(W, H)``; both axes must be multiples of 14.
    Accepts 1 or 2 paths; a single image is duplicated to ``S = 2`` (the
    released checkpoints always see 2 views). When ``seed`` is given,
    ``torch.manual_seed`` is called immediately before sampling the flow-matching
    initial noise to make the result reproducible.
    """
    assert len(image_paths) in (1, 2), "expected 1 or 2 image paths"
    if len(image_paths) == 1:
        image_paths = list(image_paths) * 2

    W, H = resolution
    device = next(model.parameters()).device

    # Match baseline preprocessing exactly: ToTensor + Normalize(0.5, 0.5)
    # gives [-1, 1]; baseline then does `img * 0.5 + 0.5` to recover [0, 1].
    imgs = torch.stack(
        [IMG_NORM(Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)) for p in image_paths], # pyright: ignore[reportArgumentType]
        dim=0,
    ).unsqueeze(0).to(device)            # (1, S, 3, H, W) in [-1, 1]
    images = imgs * 0.5 + 0.5            # -> [0, 1]

    model.eval()
    num_steps = int(1 // fm_step_size)
    time_grid = torch.linspace(0, 1, num_steps, device=device)

    with autocast(device):
        encoder_data = model.encode(images=images, pointmaps=None)

        # Per-step velocity field consumed by ODESolver. Captures `images` and
        # `encoder_data` from the enclosing scope so we don't depend on upstream
        # `BatchModelWrapper` glue. Scalar `t` is broadcast to (B, N) — mirroring
        # what `model.decode` expects via `pts3d_head`.
        def velocity_field(x: Tensor, t: Tensor, **_) -> Tensor:
            if t.dim() == 0:
                t = t.expand(x.shape[0], x.shape[1])
            return model.decode(
                tokens=encoder_data["tokens"],
                images=images,
                query_points=x,
                timestep=t,
            )["pts3d_xyz"]

        solver = ODESolver(velocity_model=velocity_field)
        if seed is not None:
            torch.manual_seed(seed)
        x_init = torch.rand((1, num_queries, 3), dtype=torch.float32, device=device) * 2 - 1
        pts = solver.sample(
            time_grid=time_grid,
            x_init=x_init,
            method="euler",
            step_size=fm_step_size,
            return_intermediates=False,
        )

    return pts[0].detach().cpu().numpy()
