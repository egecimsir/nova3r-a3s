# Copyright (c) 2026 Weirong Chen
"""Public I/O helpers for NOVA3R.

This module exposes the user-facing entry points re-exported at the package
root: image preprocessing, pairwise view-graph construction, checkpoint
loading, PLY export, and the high-level :func:`predict` convenience function
that wraps the full ``inference_nova3r`` pipeline.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as transforms

from nova3r.inference import inference_nova3r
from nova3r.models import Nova3rImgCond, Nova3rPtsCond
from nova3r.utils.device import get_default_device, resolve_device
from nova3r.utils.image import load_images
from nova3r.utils.image_pairs import make_pairs

__all__ = [
    "load_images",
    "make_pairs",
    "save_pointcloud_ply",
    "load_model",
    "predict",
    "get_default_device",
    "resolve_device",
]

_MODEL_REGISTRY = {
    "Nova3rImgCond": Nova3rImgCond,
    "Nova3rPtsCond": Nova3rPtsCond,
}


def save_pointcloud_ply(
    pts3d: Union[np.ndarray, torch.Tensor],
    path: str,
) -> str:
    """Save an ``(N, 3)`` point set to a PLY file.

    Lazily imports ``open3d``; the ``[io]`` extra installs it.

    Parameters
    ----------
    pts3d
        Point cloud as a NumPy array or torch tensor of shape ``(N, 3)``
        (or any shape that reshapes to ``(-1, 3)``).
    path
        Destination PLY file path. Parent directories are created if missing.

    Returns
    -------
    str
        The ``path`` argument, for convenience.
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


def load_model(ckpt_path: str, device=None):
    """Load a NOVA3R checkpoint together with its Hydra sidecar config.

    Expects ``<ckpt_dir>/.hydra/config.yaml`` next to the checkpoint, matching
    the layout produced by upstream training and the ``nova3r-download`` CLI.

    Parameters
    ----------
    ckpt_path
        Path to a ``.pth`` checkpoint file.
    device
        Target device; ``None`` (default) auto-selects via
        :func:`nova3r.utils.device.resolve_device`. May also be a string,
        ``torch.device``, or tensor.

    Returns
    -------
    tuple
        ``(model, cfg)`` where ``model`` is a loaded ``nn.Module`` in eval-ready
        state and ``cfg`` is the OmegaConf ``experiment`` node parsed from the
        Hydra sidecar.

    Raises
    ------
    FileNotFoundError
        If the ``.hydra/config.yaml`` sidecar is missing.
    KeyError
        If the checkpoint references a model class not in ``_MODEL_REGISTRY``.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError(
            "load_model requires `omegaconf`. Install with `pip install omegaconf`."
        ) from e

    device = resolve_device(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    config_dir = os.path.join(os.path.dirname(ckpt_path), ".hydra")
    config_path = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No .hydra/config.yaml found at {config_dir}. "
            "Ensure the checkpoint directory contains the Hydra config."
        )

    cfg = OmegaConf.load(config_path)
    cfg = cfg.experiment

    model_name = cfg.model["name"]
    if model_name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model class '{model_name}'. "
            f"Known classes: {sorted(_MODEL_REGISTRY)}"
        )
    model = _MODEL_REGISTRY[model_name](**cfg.model["params"])
    model.to(device)

    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    del ckpt
    cfg = _apply_inference_defaults(cfg)
    return model, cfg


def _apply_inference_defaults(cfg):
    """Populate Hydra defaults expected by :func:`inference_nova3r` if missing."""
    try:
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError("omegaconf is required") from e
    OmegaConf.set_struct(cfg, False)
    if "fm_step_size" not in cfg:
        cfg.fm_step_size = 0.04
    if "fm_sampling" not in cfg:
        cfg.fm_sampling = "euler"
    OmegaConf.set_struct(cfg, True)
    return cfg


def predict(
    ckpt_path: str,
    image_paths: Sequence[str],
    device=None,
    resolution: Tuple[int, int] = (518, 392),
    num_queries: int = 20000,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Run end-to-end inference from image paths to a point cloud.

    Loads ``ckpt_path``, preprocesses the input images, runs
    :func:`nova3r.inference.inference_nova3r`, and returns the predicted point
    cloud as a NumPy array. Optionally writes the result to a PLY file.

    Parameters
    ----------
    ckpt_path
        Path to a ``.pth`` checkpoint (e.g.
        ``checkpoints/scene_n1/checkpoint-last.pth``).
    image_paths
        One or two image paths. A single path is duplicated and the pair graph
        is symmetrized to emulate the single-view setting.
    device
        Target device; ``None`` auto-selects (CUDA > MPS > CPU). May also be a
        string, ``torch.device``, or tensor.
    resolution
        ``(width, height)`` to which inputs are resized. The released
        checkpoints expect ``(518, 392)``.
    num_queries
        Number of query points sampled by the flow-matching decoder.
    output_path
        Optional path to a ``.ply`` file. When provided, the result is also
        written to disk; requires the ``[io]`` extra.

    Returns
    -------
    numpy.ndarray
        Predicted point cloud with shape ``(num_queries, 3)``.

    Raises
    ------
    ValueError
        If ``image_paths`` does not contain one or two paths.
    """
    if not (1 <= len(image_paths) <= 2):
        raise ValueError("predict expects 1 or 2 image paths")

    device = resolve_device(device)
    model, cfg = load_model(ckpt_path, device)

    target_W, target_H = resolution
    img_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    paths = list(image_paths) if len(image_paths) == 2 else [image_paths[0], image_paths[0]]
    images = []
    for i, p in enumerate(paths):
        img = PIL.Image.open(p).convert("RGB").resize((target_W, target_H), PIL.Image.LANCZOS)
        images.append(dict(
            img=img_norm(img)[None],
            true_shape=np.int32([target_H, target_W]),
            idx=i, instance=str(i),
            view_label=f"input_{i}",
        ))

    symmetrize = len(image_paths) == 1
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=symmetrize)

    with torch.no_grad():
        output = inference_nova3r(
            cfg, pairs, model, device,
            batch_size=1, num_queries=num_queries,
            method=cfg.get("fm_sampling", "euler"),
        )

    pts3d = output["pred"]["pts3d_xyz"][0].numpy()

    if output_path is not None:
        save_pointcloud_ply(pts3d, output_path)

    return pts3d
