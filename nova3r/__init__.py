# Copyright (c) 2026 Weirong Chen
"""NOVA3R - Nova3r-compatible public API.

Only the standalone :class:`Nova3r` model and its compatible helpers are
exposed at the package root. The legacy Hydra-driven ``Nova3r{Img,Pts}Cond``
pipeline lives under :mod:`nova3r._legacy`.
"""
from nova3r.inference import inference_nova3r
from nova3r.io import (
    load_images,
    load_model,
    make_pairs,
    predict,
    save_pointcloud_ply,
)
from nova3r.model import Nova3r, preprocess
from nova3r.scripts.download_checkpoints import download_checkpoints
from nova3r.utils.device import autocast, get_default_device, resolve_device

__all__ = [
    "Nova3r",
    "predict",
    "preprocess",
    "load_model",
    "save_pointcloud_ply",
    "load_images",
    "make_pairs",
    "inference_nova3r",
    "get_default_device",
    "resolve_device",
    "autocast",
    "download_checkpoints",
]
