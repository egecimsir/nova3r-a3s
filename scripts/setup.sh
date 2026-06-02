#!/usr/bin/env bash
# setup.sh — device-aware setup for the nova3r-a3s package.
#
# Detects the host:
#   Linux + nvcc -> CUDA wheels (cu121) + torch-cluster from the PyG index
#   macOS        -> default PyPI wheels (MPS-enabled on arm64)
#   else         -> default PyPI wheels (CPU)
#
# Steps:
#   1. venv (skipped if VIRTUAL_ENV is already set)
#   2. torch + torchvision matching the platform (skipped if torch is present)
#   3. CUDA-only: torch-cluster from https://data.pyg.org/whl/
#   4. editable install of this package with extras
#   5. import-surface verification
#
# Overrides:
#   PYTHON_BIN=python3.10   interpreter for a fresh venv
#   VENV_DIR=.venv          venv path (relative to this package)
#   SKIP_TORCH=1            reuse torch already in the env
#   USE_UV=0                force plain pip even if uv is available
#   TORCH_VERSION=2.5.1
#   TORCHVISION_VERSION=0.20.1
#   CUDA_TAG=cu121          torch wheel index tag (CUDA only)
#   EXTRAS=io               comma-separated nova3r extras (io,sampling,cuda,all)
#                           [cuda] is auto-appended on CUDA hosts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PKG_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
CUDA_TAG="${CUDA_TAG:-cu121}"
SKIP_TORCH="${SKIP_TORCH:-0}"
USE_UV="${USE_UV:-1}"
EXTRAS="${EXTRAS:-io}"

if [ "$USE_UV" = "1" ] && command -v uv >/dev/null 2>&1; then HAVE_UV=1; else HAVE_UV=0; fi

UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"
IS_CUDA=0
if [ "$UNAME_S" = "Linux" ] && command -v nvcc >/dev/null 2>&1; then IS_CUDA=1; fi
case "$UNAME_S" in
    Linux)  PLATFORM=$([ $IS_CUDA -eq 1 ] && echo "cuda" || echo "cpu") ;;
    Darwin) PLATFORM="mps" ;;
    *)      PLATFORM="cpu" ;;
esac

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
log()  { printf "%s\n" "$*"; }
ok()   { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}!${NC} %s\n" "$*"; }
err()  { printf "${RED}✗${NC} %s\n" "$*" >&2; }

pip_install() {
    if [ $HAVE_UV -eq 1 ]; then uv pip install "$@"
    else python -m pip install "$@"; fi
}

log "================================================="
log "nova3r-a3s setup — ${PLATFORM} (${UNAME_S} ${UNAME_M})"
log "================================================="
log "Package root : $PKG_ROOT"
log "Installer    : $([ $HAVE_UV -eq 1 ] && echo "uv $(uv --version | awk '{print $2}')" || echo pip)"
log "Extras       : $EXTRAS"
log ""

# 1. venv
log "─── 1. Virtual environment ─────────────────────────────────────"
if [ -n "${VIRTUAL_ENV:-}" ]; then
    ok "Reusing active venv: $VIRTUAL_ENV"
else
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        err "$PYTHON_BIN not found. Install Python 3.10 or set PYTHON_BIN."
        exit 1
    fi
    if [ ! -d "$VENV_DIR" ]; then
        if [ $HAVE_UV -eq 1 ]; then uv venv --python "$PYTHON_BIN" "$VENV_DIR"
        else "$PYTHON_BIN" -m venv "$VENV_DIR"; fi
        ok "Created venv: $PKG_ROOT/$VENV_DIR"
    else
        ok "Reusing venv: $PKG_ROOT/$VENV_DIR"
    fi
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"
fi

# setuptools<82: legacy setup.py files (curope etc.) need the pre-PEP-632
# distutils path that setuptools 82 removed.
pip_install "pip>=24.0" "wheel" "setuptools>=69,<82"
ok "pip / setuptools / wheel in place"

# 2. PyTorch
log ""
log "─── 2. PyTorch ─────────────────────────────────────────────────"
if [ "$SKIP_TORCH" = "1" ]; then
    warn "SKIP_TORCH=1 — not touching torch"
elif python -c "import torch" >/dev/null 2>&1; then
    ok "torch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    if [ $IS_CUDA -eq 1 ]; then
        pip_install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
            "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
    else
        pip_install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
    fi
    ok "torch installed"
fi

# 3. torch-cluster (CUDA only)
if [ $IS_CUDA -eq 1 ]; then
    log ""
    log "─── 3. torch-cluster (PyG wheel) ───────────────────────────────"
    if python -c "import torch_cluster" >/dev/null 2>&1; then
        ok "torch_cluster already installed"
    else
        TORCH_FULL="$(python -c 'import torch; print(torch.__version__)')"
        TORCH_BASE="${TORCH_FULL%%+*}"
        pip_install torch-cluster \
            -f "https://data.pyg.org/whl/torch-${TORCH_BASE}+${CUDA_TAG}.html" \
            || warn "PyG wheel not found; sdist build will run on next install"
    fi
fi

# 4. nova3r-a3s (editable)
log ""
log "─── 4. nova3r-a3s (editable) ───────────────────────────────────"
# Auto-append [cuda] extra on CUDA hosts so torch-cluster is declared.
INSTALL_EXTRAS="$EXTRAS"
if [ $IS_CUDA -eq 1 ]; then
    case ",$INSTALL_EXTRAS," in
        *,cuda,*) ;;
        *) INSTALL_EXTRAS="${INSTALL_EXTRAS},cuda" ;;
    esac
fi
pip_install --no-build-isolation -e ".[${INSTALL_EXTRAS}]"
ok "nova3r-a3s [${INSTALL_EXTRAS}] (editable)"

# 5. Verification
log ""
log "─── 5. Verification ────────────────────────────────────────────"
python -c "
import torch
mps = getattr(torch.backends, 'mps', None)
print('torch', torch.__version__,
      '| cuda', torch.cuda.is_available(),
      '| mps', bool(mps and mps.is_available()))
"
if [ $IS_CUDA -eq 1 ]; then
    python -c "import torch_cluster; print('torch_cluster OK')" \
        || { err "torch_cluster import failed"; exit 1; }
    python -c "import nova3r; print('nova3r OK, device =', nova3r.get_default_device())" \
        || { err "nova3r import failed"; exit 1; }
else
    # On non-CUDA hosts, model classes that top-level-import torch_cluster
    # will fail; only assert that the package itself imports.
    python -c "import nova3r.io, nova3r.utils.device as d; print('nova3r OK, device =', d.get_default_device())" \
        || { err "nova3r import failed"; exit 1; }
fi

log ""
ok "nova3r-a3s setup complete."
if [ $IS_CUDA -ne 1 ]; then
    log ""
    warn "Non-CUDA host: torch-cluster has no MPS/CPU wheel."
    warn "Top-level imports of Nova3rImgCond / Nova3rPtsCond / TripoSG AE will fail."
    warn "Use lower-level building blocks, or run on a CUDA host for full inference."
fi
