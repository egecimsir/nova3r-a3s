# Copyright (c) 2026 Weirong Chen
"""Legacy NOVA3R pipeline (Hydra-driven, ``Nova3r{Img,Pts}Cond``-based).

Kept verbatim for backward compatibility with checkpoints that ship a
``.hydra/config.yaml`` sidecar and the original ``inference_nova3r`` runner.
The Nova3r-compatible API lives in :mod:`nova3r.io` and :mod:`nova3r.inference`.
"""

from nova3r._legacy.io import load_model, predict
from nova3r._legacy.inference import (
    inference_nova3r,
    loss_of_one_batch_demo,
    loss_of_one_batch_lari,
)

__all__ = [
    "load_model",
    "predict",
    "inference_nova3r",
    "loss_of_one_batch_demo",
    "loss_of_one_batch_lari",
]
