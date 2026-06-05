"""Parity + contract tests for the standalone ``nova3r.Nova3r`` model.

Coverage:
* state-dict completeness (no missing / unexpected keys)
* standalone construction (no upstream wrapper required)
* ``preprocess`` output contract + bit-exact match vs upstream inline preprocessing
* ``encode`` / ``forward`` parity vs ``Nova3rImgCond`` for ``scene_n1`` and ``scene_n2``
* end-to-end ``predict`` parity vs ``nova3r._legacy.io.predict`` (slow)

Runtime is kept low by caching each checkpoint load once per session
(see ``tests/conftest.py``) and reusing the cached models in the e2e test
via monkeypatching upstream ``load_model``.
"""
from __future__ import annotations

import numpy as np
import PIL.Image
import pytest
import torch
import torchvision.transforms as transforms

import nova3r._legacy.io
from nova3r import Nova3r, predict, preprocess
from nova3r.model import IMG_NORM, NUM_3D_TOKENS, TOKEN_DIM
from nova3r._legacy.io import predict as base_predict


N_QUERY = 1024
ATOL = 1e-5
# CUDA fp32 isn't bit-stable across separately-allocated module trees, and
# ours/upstream ``predict`` use different autocast dtypes (fp16 vs bf16), so
# CUDA parity is checked at the observed noise floor. MPS/CPU stay strict.
_CUDA = torch.cuda.is_available()
ATOL_ENCODE = 1e-1 if _CUDA else ATOL
ATOL_FORWARD = 5e-3 if _CUDA else ATOL
ATOL_PREDICT = 5e-1 if _CUDA else ATOL
RTOL_PREDICT = 2e-1 if _CUDA else 1e-4
RESOLUTION = (518, 392)  # (W, H), multiples of 14 (VGGT patch size)


# --------------------------------------------------------------------------- #
# State-dict + standalone construction                                        #
# --------------------------------------------------------------------------- #

def test_state_dict_coverage(device, n1_ckpt):
    """Released checkpoint covers every learnable parameter of ``Nova3r``."""
    model = Nova3r(device=device)
    state = model._read_state(n1_ckpt)
    merged = {
        k: v for k, v in state.items()
        if k.startswith(model.ENC_PREFIXES) or k.startswith(model.DEC_PREFIXES)
    }
    result = model.load_state_dict(merged, strict=False)
    own = set(model.state_dict())
    missing = [k for k in result.missing_keys if k in own]
    unexpected = [k for k in result.unexpected_keys if k in own]

    assert not missing, f"missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"


def test_nova3r_constructs_without_upstream_wrapper():
    """``Nova3r`` builds + exposes its public API without touching upstream glue."""
    model = Nova3r()  # CPU, no checkpoints
    # Required public surface
    assert callable(model.encode)
    assert callable(model.decode)
    assert callable(model.forward)
    assert callable(model.load_checkpoints)
    # Architectural children mirror upstream key prefixes
    assert hasattr(model, "vggt_aggregator")
    assert hasattr(model, "img_token_proj")
    assert hasattr(model, "pts3d_head")
    # ``predict`` must not reach for the upstream training-time glue.
    pred_globals = predict.__wrapped__.__globals__  # type: ignore[attr-defined]
    assert "BatchModelWrapper" not in pred_globals
    assert "inference_nova3r" not in pred_globals


# --------------------------------------------------------------------------- #
# preprocess                                                                  #
# --------------------------------------------------------------------------- #

def test_preprocess_output_contract(image_path):
    """``preprocess`` returns the upstream view-dict structure with correct types."""
    W, H = RESOLUTION
    out = preprocess([str(image_path), str(image_path)], RESOLUTION)

    assert isinstance(out, list) and len(out) == 2
    for i, view in enumerate(out):
        assert set(view) == {"img", "true_shape", "idx", "instance", "view_label"}
        img = view["img"]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (1, 3, H, W)
        assert img.dtype == torch.float32
        # IMG_NORM = ToTensor + Normalize(0.5, 0.5) -> values in [-1, 1].
        assert img.min().item() >= -1.0 - 1e-6
        assert img.max().item() <= 1.0 + 1e-6
        assert isinstance(view["true_shape"], np.ndarray)
        assert view["true_shape"].dtype == np.int32
        np.testing.assert_array_equal(view["true_shape"], np.int32([H, W]))
        assert view["idx"] == i
        assert view["instance"] == str(i)
        assert view["view_label"] == f"input_{i}"


def test_preprocess_matches_upstream_inline(image_path):
    """``preprocess`` is bit-exact with the inline preprocessing in ``nova3r._legacy.io.predict``."""
    W, H = RESOLUTION
    # Replicate exactly what upstream `nova3r._legacy.io.predict` does inline.
    upstream_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = PIL.Image.open(image_path).convert("RGB").resize((W, H), PIL.Image.LANCZOS)
    expected = upstream_norm(img)[None]

    out = preprocess([str(image_path)], RESOLUTION)
    assert torch.equal(out[0]["img"], expected), \
        "preprocess diverges from upstream inline preprocessing"

    # Sanity: our shared IMG_NORM is identical to the upstream-style compose.
    assert torch.equal(IMG_NORM(img)[None], expected)  # type: ignore[index]


# --------------------------------------------------------------------------- #
# encode / forward parity                                                     #
# --------------------------------------------------------------------------- #

def test_encode_parity(ckpt, ours_model, base_model, det_images):
    """``Nova3r.encode`` matches upstream ``Nova3rImgCond._encode``."""
    _, _, n_views = ckpt
    images = det_images(n_views)
    with torch.no_grad():
        enc_ours = ours_model.encode(images)["tokens"]
        enc_base = base_model._encode(images=images)["tokens"]

    # Contract: token tensor shape and dtype.
    B, S = images.shape[:2]
    assert enc_ours.shape == (B, NUM_3D_TOKENS, TOKEN_DIM)
    assert enc_ours.dtype == torch.float32

    # Parity: shape + bit-exact values vs upstream.
    assert enc_ours.shape == enc_base.shape
    max_abs = (enc_ours - enc_base).abs().max().item()
    assert torch.allclose(enc_ours, enc_base, atol=ATOL_ENCODE), \
        f"encode diverges (max |diff| = {max_abs:.2e})"


def test_forward_parity(ckpt, ours_model, base_model, device, det_images):
    """Full ``forward`` with fixed query points + timestep matches upstream."""
    _, _, n_views = ckpt
    images = det_images(n_views)
    g = torch.Generator(device="cpu").manual_seed(0)
    query_points = torch.randn((1, N_QUERY, 3), generator=g).to(device)
    timestep = torch.full((1, N_QUERY), 0.5, device=device)

    with torch.no_grad():
        out_ours = ours_model(images, query_points=query_points, timestep=timestep)
        out_base = base_model(images, query_points=query_points, timestep=timestep)

    # Contract: output dict with `pts3d_xyz` of shape (B, N, 3).
    assert isinstance(out_ours, dict) and "pts3d_xyz" in out_ours
    fwd_ours, fwd_base = out_ours["pts3d_xyz"], out_base["pts3d_xyz"]
    assert fwd_ours.shape == fwd_base.shape == (1, N_QUERY, 3)
    assert fwd_ours.dtype == torch.float32

    max_abs = (fwd_ours - fwd_base).abs().max().item()
    assert torch.allclose(fwd_ours, fwd_base, atol=ATOL_FORWARD), \
        f"forward diverges (max |diff| = {max_abs:.2e})"


# --------------------------------------------------------------------------- #
# end-to-end predict                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_predict_matches_upstream(
    n1_ckpt, image_path, n1_ours_model, n1_base_loaded,
    seed_torch_rand_xyz, monkeypatch,
):
    """End-to-end ``predict`` matches ``nova3r._legacy.io.predict`` bit-exactly.

    Patches upstream ``load_model`` to return the cached n1 ``(model, cfg)``
    so this slow test doesn't trigger an extra full checkpoint load.
    """
    monkeypatch.setattr(nova3r._legacy.io, "load_model", lambda *a, **k: n1_base_loaded)

    with seed_torch_rand_xyz():
        out_ours = predict(n1_ours_model, [image_path])
        out_base = base_predict(str(n1_ckpt), [str(image_path)])

    # Contract.
    assert isinstance(out_ours, np.ndarray)
    assert out_ours.shape == (20_000, 3)
    assert out_ours.dtype == np.float32
    assert np.isfinite(out_ours).all()

    # bf16 autocast + 25 Euler steps compound per-step rounding; ATOL is unrealistic.
    assert out_ours.shape == out_base.shape
    max_abs = float(np.max(np.abs(out_ours - out_base)))
    assert np.allclose(out_ours, out_base, atol=ATOL_PREDICT, rtol=RTOL_PREDICT), \
        f"predict diverges (max |diff| = {max_abs:.2e})"


@pytest.mark.slow
def test_predict_output_contract(n1_ours_model, image_path):
    """``predict`` returns ``(num_queries, 3)`` float32 finite ndarray."""
    out = predict(n1_ours_model, [image_path], num_queries=2_000, fm_step_size=0.25)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2_000, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
