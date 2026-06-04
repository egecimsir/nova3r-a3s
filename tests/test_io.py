"""Tests for :mod:`nova3r.io` — public I/O helpers (Nova3r-compatible surface).

Coverage:
* public API contract: ``__all__`` is complete and members are callable
* re-export wiring: ``predict``/``load_images``/``make_pairs`` originate from
  the documented source modules (no shadow copies)
* ``save_pointcloud_ply`` round-trips numpy and torch inputs, reshapes flat
  arrays, and creates missing parent directories
* ``load_model`` builds an eval-mode :class:`Nova3r`, honours the requested
  device, and accepts both ``str`` and ``Path`` checkpoint arguments
* ``load_model`` with a separate decoder checkpoint (``scene_ae``)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import nova3r
import nova3r.io
import nova3r.model
import nova3r.utils.image
import nova3r.utils.image_pairs
from nova3r import Nova3r
from nova3r.io import (
    get_default_device,
    load_images,
    load_model,
    make_pairs,
    predict,
    resolve_device,
    save_pointcloud_ply,
)

open3d = pytest.importorskip("open3d", reason="open3d not installed; install with `pip install open3d`")


# --------------------------------------------------------------------------- #
# Public API contract                                                         #
# --------------------------------------------------------------------------- #

def test_public_api_surface():
    """``nova3r.io.__all__`` matches the documented Nova3r-compatible surface."""
    expected = {
        "load_model",
        "predict",
        "save_pointcloud_ply",
        "load_images",
        "make_pairs",
        "get_default_device",
        "resolve_device",
    }
    assert set(nova3r.io.__all__) == expected
    for name in expected:
        assert callable(getattr(nova3r.io, name)), f"{name} is not callable"


def test_predict_is_reexport_from_model():
    """``nova3r.io.predict`` is the same callable as ``nova3r.model.predict``."""
    assert nova3r.io.predict is nova3r.model.predict
    # And the package root re-exports the same object too.
    assert nova3r.predict is nova3r.model.predict


def test_load_images_and_make_pairs_are_reexports():
    """Image helpers are re-exported verbatim from their utility modules."""
    assert nova3r.io.load_images is nova3r.utils.image.load_images
    assert nova3r.io.make_pairs is nova3r.utils.image_pairs.make_pairs


def test_device_helpers_are_reexports():
    """Device helpers are re-exported verbatim from ``nova3r.utils.device``."""
    import nova3r.utils.device as dev

    assert nova3r.io.get_default_device is dev.get_default_device
    assert nova3r.io.resolve_device is dev.resolve_device


# --------------------------------------------------------------------------- #
# save_pointcloud_ply                                                         #
# --------------------------------------------------------------------------- #

def _read_ply_points(path: str) -> np.ndarray:
    """Read a PLY back via open3d for round-trip comparison."""
    pcd = open3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)


def test_save_pointcloud_ply_numpy_roundtrip(tmp_path: Path):
    """Numpy ``(N, 3)`` input is saved and read back identically."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-0.5, 0.25, 0.75]], dtype=np.float32)
    out = tmp_path / "cloud.ply"

    returned = save_pointcloud_ply(pts, str(out))

    assert returned == str(out)
    assert out.exists() and out.stat().st_size > 0
    np.testing.assert_allclose(_read_ply_points(str(out)), pts, atol=1e-6)


def test_save_pointcloud_ply_accepts_torch_tensor(tmp_path: Path):
    """Torch tensors (including non-contiguous / on autograd graph) are detached and saved."""
    pts = torch.tensor([[0.1, 0.2, 0.3], [4.0, 5.0, 6.0]], dtype=torch.float32, requires_grad=True)
    out = tmp_path / "torch.ply"

    save_pointcloud_ply(pts, str(out))

    np.testing.assert_allclose(_read_ply_points(str(out)), pts.detach().numpy(), atol=1e-6)


def test_save_pointcloud_ply_reshapes_flat_input(tmp_path: Path):
    """A flat ``(3*N,)`` array is reshaped to ``(N, 3)`` before saving."""
    flat = np.arange(9, dtype=np.float32)
    out = tmp_path / "flat.ply"

    save_pointcloud_ply(flat, str(out))

    np.testing.assert_allclose(_read_ply_points(str(out)), flat.reshape(-1, 3), atol=1e-6)


def test_save_pointcloud_ply_creates_parent_dir(tmp_path: Path):
    """Missing parent directories are created on demand."""
    pts = np.zeros((5, 3), dtype=np.float32)
    out = tmp_path / "nested" / "subdir" / "cloud.ply"

    save_pointcloud_ply(pts, str(out))

    assert out.exists()


# --------------------------------------------------------------------------- #
# load_model — checkpoint-driven (slow)                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_load_model_returns_eval_mode_nova3r(n1_ckpt, device):
    """``load_model`` returns a :class:`Nova3r` in eval mode on the requested device."""
    model = load_model(n1_ckpt, device=device)

    assert isinstance(model, Nova3r)
    assert not model.training
    # All parameters live on the requested device (compare ``.type`` so
    # cuda:0 ↔ cuda matches).
    target_type = resolve_device(device).type
    assert next(model.parameters()).device.type == target_type


@pytest.mark.slow
def test_load_model_accepts_str_path(n1_ckpt, device):
    """``ckpt_path`` accepts a plain string."""
    model = load_model(str(n1_ckpt), device=device)
    assert isinstance(model, Nova3r)


@pytest.mark.slow
def test_load_model_default_device_resolves(n1_ckpt):
    """``device=None`` falls back to :func:`get_default_device`."""
    model = load_model(n1_ckpt)
    expected = get_default_device().type
    assert next(model.parameters()).device.type == expected


@pytest.mark.slow
def test_load_model_with_separate_decoder(n1_ckpt, device):
    """An explicit ``dec_ckpt`` overrides the decoder source.

    Skips when the AE checkpoint isn't present.
    """
    dec_ckpt = n1_ckpt.parent.parent / "scene_ae" / "checkpoint-last.pth"
    if not dec_ckpt.exists():
        pytest.skip(f"missing decoder checkpoint: {dec_ckpt}")

    model = load_model(n1_ckpt, dec_ckpt=dec_ckpt, device=device)

    assert isinstance(model, Nova3r)
    assert not model.training
    # Sanity: encoder + decoder children both populated.
    assert sum(p.numel() for p in model.vggt_aggregator.parameters()) > 0
    assert sum(p.numel() for p in model.pts3d_head.parameters()) > 0
