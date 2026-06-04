"""Shared pytest fixtures for the nova3r-a3s test suite."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

# MPS lacks a few ops (e.g. `aten::_upsample_bicubic2d_aa.out`); fall back to
# CPU for unsupported ops. Must be set before `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pytest
import torch

from nova3r import Nova3r
from nova3r.io import load_model

SEED = 0
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CKPT_ROOT = Path(os.environ.get("NOVA3R_CKPT_DIR", _REPO_ROOT / "checkpoints" / "nova3r"))
_DEMO_IMAGE = Path(os.environ.get("NOVA3R_DEMO_IMAGE", _REPO_ROOT / "demo" / "examples" / "scene_1.png"))


# Process-level caches so the same checkpoint is loaded at most once per
# session, no matter how many fixtures or tests request it.
_OURS_CACHE: dict[str, Nova3r] = {}
_BASE_CACHE: dict[str, tuple] = {}


def _load_ours(path: Path, device: torch.device) -> Nova3r:
    key = str(path)
    if key not in _OURS_CACHE:
        _OURS_CACHE[key] = Nova3r(device=device).load_checkpoints(path).eval()
    return _OURS_CACHE[key]


def _load_base(path: Path, device: torch.device) -> tuple:
    """Load upstream ``(model, cfg)`` once; reused by parity + predict tests."""
    key = str(path)
    if key not in _BASE_CACHE:
        model, cfg = load_model(str(path), device=device)
        _BASE_CACHE[key] = (model.eval(), cfg)
    return _BASE_CACHE[key]


@pytest.fixture(scope="session")
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def image_path() -> Path:
    if not _DEMO_IMAGE.exists():
        pytest.skip(f"missing demo image: {_DEMO_IMAGE}")
    return _DEMO_IMAGE


@pytest.fixture(scope="session", params=["n1", "n2"])
def ckpt(request) -> tuple[str, Path, int]:
    """Parametrized: ``(scene_tag, .pth path, expected #views)``."""
    name: str = request.param
    path = _CKPT_ROOT / f"scene_{name}" / "checkpoint-last.pth"
    if not path.exists():
        pytest.skip(f"missing checkpoint: {path}")
    return name, path, 1 if name == "n1" else 2


@pytest.fixture(scope="session")
def n1_ckpt() -> Path:
    path = _CKPT_ROOT / "scene_n1" / "checkpoint-last.pth"
    if not path.exists():
        pytest.skip(f"missing checkpoint: {path}")
    return path


@pytest.fixture(scope="session")
def ours_model(ckpt, device) -> Nova3r:
    _, path, _ = ckpt
    return _load_ours(path, device)


@pytest.fixture(scope="session")
def base_loaded(ckpt, device) -> tuple:
    """Upstream ``(Nova3rImgCond, cfg)`` for the parametrized checkpoint."""
    _, path, _ = ckpt
    return _load_base(path, device)


@pytest.fixture(scope="session")
def base_model(base_loaded):
    return base_loaded[0]


@pytest.fixture(scope="session")
def n1_ours_model(n1_ckpt, device) -> Nova3r:
    return _load_ours(n1_ckpt, device)


@pytest.fixture(scope="session")
def n1_base_loaded(n1_ckpt, device) -> tuple:
    return _load_base(n1_ckpt, device)


@pytest.fixture
def det_images(device):
    """Factory: deterministic ``(1, S, 3, H, W)`` batch in ``[0, 1]``."""
    def _make(n_views: int, W: int = 518, H: int = 392) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(SEED)
        return torch.rand((1, n_views, 3, H, W), generator=g).to(device)
    return _make


@pytest.fixture
def seed_torch_rand_xyz():
    """Context manager that re-seeds RNG on every ``torch.rand((*, *, 3))`` call.

    Both ``predict`` pipelines draw the FM initial noise via exactly one such
    call; reseeding on that signature aligns ``x_init`` across runs.
    """
    @contextlib.contextmanager
    def _cm():
        orig = torch.rand

        def patched(*args, **kwargs):
            shape = args[0] if args else kwargs.get("size")
            if isinstance(shape, tuple) and len(shape) == 3 and shape[-1] == 3:
                torch.manual_seed(SEED)
            return orig(*args, **kwargs)

        torch.rand = patched
        try:
            yield
        finally:
            torch.rand = orig

    return _cm

