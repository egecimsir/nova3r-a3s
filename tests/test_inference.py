"""Tests for :mod:`nova3r.inference` — Nova3r-compatible inference utilities.

Coverage:
* public API contract: ``__all__`` is complete and members are callable
* ``amp_dtype_mapping`` covers every dtype string the upstream config uses
* ``check_if_same_size`` true when every pair shares per-view shapes, false
  when any pair diverges
* ``get_complete_pts3d`` transforms per-view amodal points into camera-1 frame
  and produces a valid mask that respects ``pts3d_complete_valid_num``
* ``get_all_pts3d`` dispatches modes correctly and raises for unknown modes
* ``normalize_input`` is a no-op for ``mode='none'`` and produces normalized
  output for ``median`` / ``cube`` modes
* ``inference_nova3r`` (slow) — output contract on real pairs, reproducible
  with a fixed seed, single-pair output matches :func:`nova3r.predict`
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import nova3r.inference
from nova3r import Nova3r, make_pairs, predict, preprocess
from nova3r.inference import (
    amp_dtype_mapping,
    check_if_same_size,
    get_all_pts3d,
    get_complete_pts3d,
    inference_nova3r,
    normalize_input,
)


SEED = 0
RESOLUTION = (518, 392)  # (W, H), matches conftest's `det_images` default.


# --------------------------------------------------------------------------- #
# Public API contract                                                         #
# --------------------------------------------------------------------------- #

def test_public_api_surface():
    """``nova3r.inference.__all__`` matches the documented Nova3r-compatible surface."""
    expected = {
        "amp_dtype_mapping",
        "get_all_pts3d",
        "get_complete_pts3d",
        "normalize_input",
        "check_if_same_size",
        "inference_nova3r",
    }
    assert set(nova3r.inference.__all__) == expected


def test_no_upstream_wrapper_globals():
    """``inference_nova3r`` doesn't reach into the legacy training-time glue."""
    g = inference_nova3r.__wrapped__.__globals__  # type: ignore[attr-defined]
    assert "BatchModelWrapper" not in g
    assert "loss_of_one_batch_demo" not in g
    assert "loss_of_one_batch_lari" not in g


# --------------------------------------------------------------------------- #
# amp_dtype_mapping                                                           #
# --------------------------------------------------------------------------- #

def test_amp_dtype_mapping_values():
    """All upstream dtype keys map to the correct ``torch`` dtypes."""
    assert amp_dtype_mapping["fp16"] is torch.float16
    assert amp_dtype_mapping["bf16"] is torch.bfloat16
    assert amp_dtype_mapping["fp32"] is torch.float32
    # `tf32` is an alias for fp32 storage (kernels select TF32 separately).
    assert amp_dtype_mapping["tf32"] is torch.float32


# --------------------------------------------------------------------------- #
# check_if_same_size                                                          #
# --------------------------------------------------------------------------- #

def _view(h: int, w: int) -> dict:
    return {"img": torch.zeros((1, 3, h, w))}


def test_check_if_same_size_uniform():
    """Returns True when every pair shares per-view image shapes."""
    pairs = [(_view(64, 96), _view(64, 96)) for _ in range(3)]
    assert check_if_same_size(pairs) is True


def test_check_if_same_size_mismatched_view1():
    """Returns False when view-1 shapes diverge across pairs."""
    pairs = [
        (_view(64, 96), _view(64, 96)),
        (_view(32, 96), _view(64, 96)),
    ]
    assert check_if_same_size(pairs) is False


def test_check_if_same_size_mismatched_view2():
    """Returns False when view-2 shapes diverge across pairs."""
    pairs = [
        (_view(64, 96), _view(64, 96)),
        (_view(64, 96), _view(64, 48)),
    ]
    assert check_if_same_size(pairs) is False


# --------------------------------------------------------------------------- #
# get_complete_pts3d                                                          #
# --------------------------------------------------------------------------- #

def _identity_gt(B: int, N: int, *, valid_num: list[int]) -> dict:
    """Construct a minimal gt dict with identity camera pose and arbitrary points."""
    assert len(valid_num) == B
    g = torch.Generator().manual_seed(SEED)
    return {
        "pts3d_complete": torch.randn((B, N, 3), generator=g),
        "pts3d_complete_valid_num": torch.tensor(valid_num, dtype=torch.long),
        "camera_pose": torch.eye(4).expand(B, 4, 4).contiguous(),
    }


def test_get_complete_pts3d_shapes_and_valid_mask():
    """Output shapes are ``(B, S*N, 3)`` / ``(B, S*N)`` and the mask honours valid_num."""
    B, N = 2, 5
    gt_list = [
        _identity_gt(B, N, valid_num=[3, 5]),
        _identity_gt(B, N, valid_num=[1, 4]),
    ]
    S = len(gt_list)

    pts, valid = get_complete_pts3d(gt_list)  # type: ignore[misc]

    assert pts.shape == (B, S * N, 3)
    assert valid.shape == (B, S * N)
    assert valid.dtype == torch.bool

    # Per-batch valid count is the sum of per-view valid_num.
    assert int(valid[0].sum()) == 3 + 1
    assert int(valid[1].sum()) == 5 + 4

    # Within each view block, the first valid_num entries are True, the rest False.
    for b in range(B):
        for s, gt in enumerate(gt_list):
            block = valid[b, s * N:(s + 1) * N]
            k = int(gt["pts3d_complete_valid_num"][b])
            assert block[:k].all()
            assert not block[k:].any()


def test_get_complete_pts3d_camera1_identity_preserves_points():
    """Identity camera1 leaves all points unchanged after the inverse transform."""
    B, N = 1, 4
    gt_list = [_identity_gt(B, N, valid_num=[2]), _identity_gt(B, N, valid_num=[3])]

    pts, _ = get_complete_pts3d(gt_list)  # type: ignore[misc]

    expected = torch.cat([gt["pts3d_complete"] for gt in gt_list], dim=1)
    torch.testing.assert_close(pts, expected, atol=1e-5, rtol=1e-5)


def test_get_complete_pts3d_valid_front_reorder():
    """``valid_front=True`` packs valid points first and returns per-batch counts."""
    B, N = 1, 4
    gt_list = [_identity_gt(B, N, valid_num=[2]), _identity_gt(B, N, valid_num=[1])]

    pts, valid, counts = get_complete_pts3d(gt_list, valid_front=True)  # type: ignore[misc]

    assert pts.shape == (B, 2 * N, 3)
    assert counts.tolist() == [2 + 1]
    # Front block is all-valid, rear block all-invalid.
    assert valid[0, :counts[0]].all()
    assert not valid[0, counts[0]:].any()


# --------------------------------------------------------------------------- #
# get_all_pts3d                                                               #
# --------------------------------------------------------------------------- #

def test_get_all_pts3d_src_complete_matches_helper():
    """``mode='src_complete'`` delegates to :func:`get_complete_pts3d`."""
    B, N = 2, 4
    gt_list = [_identity_gt(B, N, valid_num=[2, 3]), _identity_gt(B, N, valid_num=[1, 4])]

    pts_dispatch, valid_dispatch = get_all_pts3d(gt_list, mode="src_complete")
    pts_direct, valid_direct = get_complete_pts3d(gt_list)  # type: ignore[misc]

    torch.testing.assert_close(pts_dispatch, pts_direct)
    torch.testing.assert_close(valid_dispatch, valid_direct)


def test_get_all_pts3d_unknown_mode_raises():
    """Unknown ``mode`` values raise :class:`NotImplementedError`."""
    gt_list = [_identity_gt(1, 4, valid_num=[2])]
    with pytest.raises(NotImplementedError):
        get_all_pts3d(gt_list, mode="not_a_real_mode")


# --------------------------------------------------------------------------- #
# normalize_input                                                             #
# --------------------------------------------------------------------------- #

def _norm_inputs(B: int = 1, N: int = 32) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(SEED)
    pts_src = torch.randn((B, N, 3), generator=g) * 10.0
    pts_trg = torch.randn((B, N, 3), generator=g) * 10.0
    valid_src = torch.ones((B, N), dtype=torch.bool)
    valid_trg = torch.ones((B, N), dtype=torch.bool)
    return pts_src, valid_src, pts_trg, valid_trg


def test_normalize_input_none_is_identity():
    """``mode='none'`` returns the inputs unchanged."""
    pts_src, valid_src, pts_trg, valid_trg = _norm_inputs()

    result = normalize_input(pts_src, valid_src, pts_trg, valid_trg, mode="none")
    assert result is not None
    out_src, out_trg = result

    assert out_src is pts_src
    assert out_trg is pts_trg


def test_normalize_input_median_rescales_to_unit_median():
    """``mode='median'`` makes the target's distance-from-origin median ~= 1."""
    pts_src, valid_src, pts_trg, valid_trg = _norm_inputs()

    result = normalize_input(pts_src, valid_src, pts_trg, valid_trg, mode="median")
    assert result is not None
    out_src, out_trg = result

    assert out_src.shape == pts_src.shape
    assert out_trg.shape == pts_trg.shape
    # Target's median ‖p‖ after normalization should be ~1.0.
    median = out_trg[0].norm(dim=-1).median().item()
    assert abs(median - 1.0) < 1e-4


def test_normalize_input_median_with_scale_factor():
    """``mode='median_5'`` rescales so the target median ~= 5."""
    pts_src, valid_src, pts_trg, valid_trg = _norm_inputs()

    result = normalize_input(pts_src, valid_src, pts_trg, valid_trg, mode="median_5")
    assert result is not None
    _, out_trg = result

    median = out_trg[0].norm(dim=-1).median().item()
    assert abs(median - 5.0) < 1e-3


def test_normalize_input_cube_centers_and_bounds_target():
    """``mode='cube'`` centres the target and rescales so ~90% of points lie inside the unit ball."""
    pts_src, valid_src, pts_trg, valid_trg = _norm_inputs()

    result = normalize_input(pts_src, valid_src, pts_trg, valid_trg, mode="cube")
    assert result is not None
    _, out_trg = result

    # After centering on the target mean and dividing by the 90th-percentile
    # radius, the 90th-percentile radius of the result is ~= 1.
    radii = out_trg[0].norm(dim=-1)
    q90 = torch.quantile(radii, 0.9).item()
    assert abs(q90 - 1.0) < 1e-4


# --------------------------------------------------------------------------- #
# inference_nova3r — end-to-end (slow)                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_inference_nova3r_output_contract(n1_ours_model, image_path):
    """Output is ``{'view', 'pred': {'pts3d_xyz': Tensor[N_pairs, Q, 3]}}`` on CPU."""
    views = preprocess([str(image_path), str(image_path)], RESOLUTION)
    pairs = make_pairs(views, scene_graph="complete", symmetrize=False)
    assert len(pairs) == 1

    Q = 2_000
    out = inference_nova3r(
        n1_ours_model, pairs,
        num_queries=Q, fm_step_size=0.25, seed=SEED, verbose=False,
    )

    assert set(out) == {"view", "pred"}
    assert out["view"] == list(pairs)
    pts = out["pred"]["pts3d_xyz"]
    assert isinstance(pts, torch.Tensor)
    assert pts.shape == (len(pairs), Q, 3)
    assert pts.dtype == torch.float32
    assert pts.device.type == "cpu"
    assert torch.isfinite(pts).all()


@pytest.mark.slow
def test_inference_nova3r_is_reproducible_with_seed(n1_ours_model, image_path):
    """Two runs with the same seed produce bit-identical point clouds."""
    views = preprocess([str(image_path), str(image_path)], RESOLUTION)
    pairs = make_pairs(views, scene_graph="complete", symmetrize=False)

    out1 = inference_nova3r(
        n1_ours_model, pairs,
        num_queries=1_000, fm_step_size=0.25, seed=SEED, verbose=False,
    )
    out2 = inference_nova3r(
        n1_ours_model, pairs,
        num_queries=1_000, fm_step_size=0.25, seed=SEED, verbose=False,
    )

    torch.testing.assert_close(out1["pred"]["pts3d_xyz"], out2["pred"]["pts3d_xyz"])


@pytest.mark.slow
def test_inference_nova3r_single_pair_matches_predict(
    n1_ours_model, image_path, seed_torch_rand_xyz,
):
    """For a single pair, ``inference_nova3r`` matches :func:`nova3r.predict`.

    Both share the same encode + ODE-solver scaffold; aligning the initial
    noise via ``seed_torch_rand_xyz`` should yield bit-identical results.
    """
    views = preprocess([str(image_path), str(image_path)], RESOLUTION)
    pairs = make_pairs(views, scene_graph="complete", symmetrize=False)
    assert len(pairs) == 1

    Q = 2_000
    step = 0.25

    with seed_torch_rand_xyz():
        out = inference_nova3r(
            n1_ours_model, pairs,
            num_queries=Q, fm_step_size=step, verbose=False,
        )
        pts_predict = predict(
            n1_ours_model, [image_path],
            num_queries=Q, fm_step_size=step, resolution=RESOLUTION,
        )

    pts_inf = out["pred"]["pts3d_xyz"][0].numpy()
    np.testing.assert_allclose(pts_inf, pts_predict, atol=1e-5, rtol=1e-4)
