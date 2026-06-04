# Copyright (c) 2026 Weirong Chen
"""Legacy Hydra-driven loader + ``predict`` for ``Nova3r{Img,Pts}Cond``.

Verbatim move of the original :mod:`nova3r.io` symbols that depend on the
Hydra ``.hydra/config.yaml`` sidecar and the legacy ``inference_nova3r``
runner. Behavior is unchanged from the pre-refactor module.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as transforms

from nova3r._legacy.inference import inference_nova3r
from nova3r.io import save_pointcloud_ply
from nova3r.modules import Nova3rImgCond, Nova3rPtsCond
from nova3r.utils.device import resolve_device
from nova3r.utils.image_pairs import make_pairs

__all__ = ["load_model", "predict"]

_MODEL_REGISTRY = {
    "Nova3rImgCond": Nova3rImgCond,
    "Nova3rPtsCond": Nova3rPtsCond,
}


def load_model(ckpt_path: str, device=None):
    """Load a NOVA3R checkpoint together with its Hydra sidecar config.

    Expects ``<ckpt_dir>/.hydra/config.yaml`` next to the checkpoint, matching
    the layout produced by upstream training and the ``nova3r-download`` CLI.

    Returns ``(model, cfg)`` where ``model`` is a loaded ``nn.Module`` in
    eval-ready state and ``cfg`` is the OmegaConf ``experiment`` node parsed
    from the Hydra sidecar.
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
    """Legacy end-to-end inference from image paths to a point cloud.

    Loads ``ckpt_path`` via the Hydra-sidecar :func:`load_model`, preprocesses
    inputs, runs :func:`nova3r._legacy.inference.inference_nova3r`, and returns
    the predicted point cloud as a NumPy array.
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
