# Copyright (c) 2026 Weirong Chen
"""Nova3r-compatible I/O helpers.

Exposes the user-facing entry points re-exported at the package root: image
preprocessing, pairwise view-graph construction, PLY export, a zero-config
``load_model`` returning a :class:`nova3r.Nova3r`, and the high-level
:func:`predict` convenience function from :mod:`nova3r.model`.

The legacy Hydra-driven ``load_model`` / ``predict`` (returning
``Nova3r{Img,Pts}Cond``) live in :mod:`nova3r._legacy.io`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np
import torch

from nova3r.model import Nova3r, predict
from nova3r.utils.device import get_default_device, resolve_device
from nova3r.utils.image import load_images
from nova3r.utils.image_pairs import make_pairs

__all__ = [
    "load_model",
    "predict",
    "save_pointcloud_ply",
    "load_images",
    "make_pairs",
    "get_default_device",
    "resolve_device",
]


def load_model(
    ckpt_path: Union[str, Path],
    dec_ckpt: Union[str, Path, None] = None,
    device=None,
) -> Nova3r:
    """Zero-config Nova3r loader. Returns an eval-mode :class:`Nova3r`.

    ``ckpt_path``: Stage-2 checkpoint (``scene_n1`` / ``scene_n2``) supplying
    the image encoder, projection, and (when ``dec_ckpt`` is ``None``) the FM
    decoder.

    ``dec_ckpt``: optional Stage-1 AE checkpoint (``scene_ae``) to source the
    FM decoder from instead.

    ``device``: target device; ``None`` auto-selects via
    :func:`nova3r.utils.device.resolve_device`.
    """
    device = resolve_device(device)
    return Nova3r(device=device).load_checkpoints(ckpt_path, dec_ckpt).eval()


def save_pointcloud_ply(
    pts3d: Union[np.ndarray, torch.Tensor],
    path: str,
) -> str:
    """Save an ``(N, 3)`` point set to a PLY file.

    Lazily imports ``open3d``; the ``[io]`` extra installs it.
    """
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "save_pointcloud_ply requires `open3d`. Install with `pip install open3d`."
        ) from e

    if isinstance(pts3d, torch.Tensor):
        pts3d = pts3d.detach().cpu().numpy()
    pts3d = np.asarray(pts3d).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    o3d.io.write_point_cloud(path, pcd)
    return path
