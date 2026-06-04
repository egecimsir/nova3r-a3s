# Copyright (c) 2026 Weirong Chen
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
"""Nova3r-compatible inference utilities.

Contains the data utilities that are model-agnostic (``get_all_pts3d``,
``get_complete_pts3d``, ``normalize_input``, ``check_if_same_size``,
``amp_dtype_mapping``) plus an adapted :func:`inference_nova3r` runner that
takes a :class:`nova3r.Nova3r` instance directly (no Hydra ``args``, no
``BatchModelWrapper``, no ``model._encode``).

The legacy ``inference_nova3r`` runner (and the ``loss_of_one_batch_*``
helpers) live in :mod:`nova3r._legacy.inference`.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from torch import Tensor

from nova3r.flow_matching.solver import ODESolver
from nova3r.model import Nova3r
from nova3r.utils.device import autocast
from nova3r.utils.geometry import geotrf, inv
from nova3r.utils.misc import invalid_to_zeros
from nova3r.utils.sampling import sampling_train_gen_target

__all__ = [
    "amp_dtype_mapping",
    "get_all_pts3d",
    "get_complete_pts3d",
    "normalize_input",
    "check_if_same_size",
    "inference_nova3r",
]

amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    'tf32': torch.float32,
}


def get_all_pts3d(gt_list, mode=None, down_resolution=112):
    """Extract and optionally downsample/FPS-sample ground truth 3D points from a batch."""
    if mode == 'cube':
        pts_xyz = gt_list[0]['global_center_xyz']

        valid = torch.ones_like(pts_xyz[..., 0]).bool()  # B, N
        in_camera1 = inv(gt_list[0]['camera_pose'])
        gt_pts = geotrf(in_camera1, pts_xyz)

    elif mode == 'src_complete':
        gt_pts, valid = get_complete_pts3d(gt_list)

    elif 'src_complete_fps' in mode:
        batch_size = int(mode.split('_')[-1])
        gt_pts, valid = get_complete_pts3d(gt_list)
        gt_pts, valid = sampling_train_gen_target(gt_pts, valid, None, target_sampling='fps_fast', batch_size=batch_size)

    elif 'src_complete_fps_edge' in mode:
        batch_size = int(mode.split('_')[-1])
        gt_pts, valid = get_complete_pts3d(gt_list)
        gt_pts, valid = sampling_train_gen_target(gt_pts, valid, None, target_sampling='fps_edge_fast', batch_size=batch_size)

    elif mode == 'cube_global':
        pts_xyz = gt_list[0]['global_center_xyz']
        valid = torch.ones_like(pts_xyz[..., 0]).bool()  # B, N
        gt_pts = pts_xyz

    elif mode == 'src_view':
        gt_pts_list = [gt['pts3d'] for gt in gt_list]

        in_camera1 = inv(gt_list[0]['camera_pose'])
        gt_pts_list = [geotrf(in_camera1, gt['pts3d']) for gt in gt_list]

        gt_pts = torch.stack(gt_pts_list, dim=1)
        B, H, W, C = gt_pts_list[0].shape
        gt_pts = rearrange(gt_pts, 'b s h w c -> (b s) c h w')
        gt_pts = F.interpolate(gt_pts, size=down_resolution, mode='nearest')

        gt_pts = rearrange(gt_pts, '(b s) c h w -> b (s h w) c', b=B)

        valid_list = [gt['valid_mask'].clone() for gt in gt_list]
        valid = torch.stack(valid_list, dim=1).float()
        valid = rearrange(valid, 'b s h w -> (b s) 1 h w')
        valid = F.interpolate(valid, size=down_resolution, mode='nearest')
        valid = rearrange(valid, '(b s) 1 h w -> b (s h w)', b=B).bool()

    elif 'src_view_fps' in mode:
        batch_size = int(mode.split('_')[-1])
        gt_pts_list = [gt['pts3d'] for gt in gt_list]

        in_camera1 = inv(gt_list[0]['camera_pose'])
        gt_pts_list = [geotrf(in_camera1, gt['pts3d']) for gt in gt_list]

        gt_pts = torch.stack(gt_pts_list, dim=1)
        B, H, W, C = gt_pts_list[0].shape
        gt_pts = rearrange(gt_pts, 'b s h w c -> (b s) c h w')
        gt_pts = F.interpolate(gt_pts, size=down_resolution, mode='nearest')

        gt_pts = rearrange(gt_pts, '(b s) c h w -> b (s h w) c', b=B)

        valid_list = [gt['valid_mask'].clone() for gt in gt_list]
        valid = torch.stack(valid_list, dim=1).float()
        valid = rearrange(valid, 'b s h w -> (b s) 1 h w')
        valid = F.interpolate(valid, size=down_resolution, mode='nearest')
        valid = rearrange(valid, '(b s) 1 h w -> b (s h w)', b=B).bool()

        gt_pts, valid = sampling_train_gen_target(gt_pts, valid, None, target_sampling='fps_fast', batch_size=batch_size)

    else:
        raise NotImplementedError
    return gt_pts, valid


def get_complete_pts3d(gt_list, valid_front=False):
    """Get complete (amodal) 3D point clouds from all views, transformed to camera 1 coordinates."""
    pts_xyz = [gt['pts3d_complete'] for gt in gt_list]
    in_camera1 = inv(gt_list[0]['camera_pose'])
    pts_xyz = [geotrf(in_camera1, pts) for pts in pts_xyz]  # B, N, 3
    gt_pts = torch.stack(pts_xyz, dim=1)  # B, S, N, 3

    valid_num_list = [gt['pts3d_complete_valid_num'] for gt in gt_list]  # B, S
    valid = torch.zeros_like(gt_pts[..., 0]).bool()  # B, S, N
    for i in range(len(gt_list)):
        for j in range(valid_num_list[i].shape[0]):
            valid[j, i, :valid_num_list[i][j]] = True

    gt_pts = rearrange(gt_pts, 'b s n c -> b (s n) c')  # B, S*N, 3
    valid = rearrange(valid, 'b s n -> b (s n)')  # B, S*N

    if valid_front:
        reordered_pts = []
        reordered_valid = []
        valid_counts = []

        for b in range(gt_pts.shape[0]):
            valid_mask = valid[b]
            valid_indices = torch.where(valid_mask)[0]
            invalid_indices = torch.where(~valid_mask)[0]

            reorder_indices = torch.cat([valid_indices, invalid_indices])

            reordered_pts.append(gt_pts[b][reorder_indices])
            reordered_valid.append(valid[b][reorder_indices])
            valid_counts.append(len(valid_indices))

        gt_pts = torch.stack(reordered_pts, dim=0)
        valid = torch.stack(reordered_valid, dim=0)
        valid_counts = torch.tensor(valid_counts, device=gt_pts.device)

        return gt_pts, valid, valid_counts
    else:
        return gt_pts, valid


def normalize_input(pts3d_src, valid_src, pts3d_trg, valid_trg, mode='none'):
    """Normalize the input points."""
    if mode == 'none':
        return pts3d_src, pts3d_trg

    elif 'median' in mode:
        if mode == 'median':
            target_median = 1.0
        else:
            target_median = float(mode.split('_')[-1])

        pts3d_src_new = []
        pts3d_trg_new = []

        for b in range(pts3d_src.shape[0]):
            src_xyz = pts3d_src[b]
            trg_xyz = pts3d_trg[b]
            src_valid = valid_src[b]
            trg_valid = valid_trg[b]

            nan_pts, nnz = invalid_to_zeros(trg_xyz, trg_valid, ndim=3)

            all_dis = nan_pts.norm(dim=-1)

            mean_factor = all_dis.sum() / (nnz.sum() + 1e-8)

            valid_dis = all_dis[trg_valid]
            norm_factor = valid_dis.median() if valid_dis.numel() > 0 else torch.tensor(1.0, device=all_dis.device)

            norm_factor = norm_factor.clip(min=0.01, max=100.0)

            src_xyz_norm = src_xyz / norm_factor * target_median
            trg_xyz_norm = trg_xyz / norm_factor * target_median

            src_xyz_norm = torch.clamp(src_xyz_norm, min=-1000.0, max=1000.0)
            trg_xyz_norm = torch.clamp(trg_xyz_norm, min=-1000.0, max=1000.0)

            pts3d_src_new.append(src_xyz_norm)
            pts3d_trg_new.append(trg_xyz_norm)

        pts3d_src_new = torch.stack(pts3d_src_new, dim=0)  # B, N, 3
        pts3d_trg_new = torch.stack(pts3d_trg_new, dim=0)  # B, N, 3
        return pts3d_src_new, pts3d_trg_new

    elif 'cube' in mode:
        if mode == 'cube':
            target_scale = 1.0
        else:
            target_scale = float(mode.split('_')[-1])

        pts3d_src_new = []
        pts3d_trg_new = []

        for b in range(pts3d_src.shape[0]):
            src_xyz = pts3d_src[b]
            trg_xyz = pts3d_trg[b]
            src_valid = valid_src[b]
            trg_valid = valid_trg[b]

            center_trg = trg_xyz[trg_valid].mean(dim=0)

            src_xyz_centered = src_xyz - center_trg
            trg_xyz_centered = trg_xyz - center_trg

            dist_trg = torch.norm(trg_xyz_centered[trg_valid], dim=1)
            max_dist_trg = torch.quantile(dist_trg, 0.9)

            src_xyz_norm = src_xyz_centered / max_dist_trg * target_scale
            trg_xyz_norm = trg_xyz_centered / max_dist_trg * target_scale

            pts3d_src_new.append(src_xyz_norm)
            pts3d_trg_new.append(trg_xyz_norm)

        pts3d_src = torch.stack(pts3d_src_new, dim=0)
        pts3d_trg = torch.stack(pts3d_trg_new, dim=0)

        return pts3d_src, pts3d_trg


def check_if_same_size(pairs) -> bool:
    """Return ``True`` iff every pair shares the same per-view image shape."""
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


@torch.no_grad()
def inference_nova3r(
    model: Nova3r,
    pairs: Sequence[tuple[dict, dict]],
    *,
    num_queries: int = 20_000,
    fm_step_size: float = 0.04,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Batched Nova3r-compatible inference over a list of image pairs.

    Takes a :class:`nova3r.Nova3r` instance and a list of pairs produced by
    :func:`nova3r.make_pairs` over views from :func:`nova3r.preprocess` (each
    view's ``img`` is ``(1, 3, H, W)`` in ``[-1, 1]``). Internally mirrors the
    encode + flow-matching ODE sampling of :func:`nova3r.model.predict`.

    Returns a dict shaped like the legacy runner::

        {"view": pairs,
         "pred": {"pts3d_xyz": Tensor[len(pairs), num_queries, 3]}}

    with the point cloud tensor on CPU.
    """
    if verbose:
        print(f">> Nova3r inference on {len(pairs)} image pairs")

    device = next(model.parameters()).device
    model.eval()
    num_steps = int(1 // fm_step_size)
    time_grid = torch.linspace(0, 1, num_steps, device=device)

    per_pair: list[Tensor] = []
    iterator = tqdm.tqdm(pairs, disable=not verbose, desc=">> Nova3r inference")
    for pair in iterator:
        # Each view's `img` is (1, 3, H, W) in [-1, 1] (from `preprocess`/IMG_NORM).
        # Stack the views into (1, S, 3, H, W) and rescale to [0, 1] to match
        # the convention `model.encode` expects (same as `nova3r.model.predict`).
        imgs = torch.stack([v['img'].to(device) for v in pair], dim=1)
        images = imgs * 0.5 + 0.5

        with autocast(device):
            encoder_data = model.encode(images=images, pointmaps=None)

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
            x_init = torch.rand(
                (images.shape[0], num_queries, 3),
                dtype=torch.float32, device=device,
            ) * 2 - 1
            pts = solver.sample(
                time_grid=time_grid,
                x_init=x_init,
                method="euler",
                step_size=fm_step_size,
                return_intermediates=False,
            )

        per_pair.append(pts.detach().cpu())

    pts3d_xyz = torch.cat(per_pair, dim=0)
    return {
        "view": list(pairs),
        "pred": {"pts3d_xyz": pts3d_xyz},
    }
