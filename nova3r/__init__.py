# Copyright (c) 2026 Weirong Chen
"""NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction."""

from nova3r.inference import inference_nova3r
from nova3r.io import (
    load_images,
    make_pairs,
    save_pointcloud_ply,
    load_model,
)
from nova3r.utils.device import get_default_device, resolve_device, autocast
from nova3r.scripts.download_checkpoints import download_checkpoints
from nova3r.model import Nova3r, predict, preprocess


__all__ = [
    # Base
    "Nova3r",
    "predict", 
    "preprocess",
    # Functions
    "inference_nova3r",
    "load_images",
    "make_pairs",
    "save_pointcloud_ply",
    "load_model",
    "get_default_device",
    "resolve_device",
    "autocast",
    "download_checkpoints",
]
