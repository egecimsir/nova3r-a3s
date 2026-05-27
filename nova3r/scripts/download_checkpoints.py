# Copyright (c) 2026 Weirong Chen
"""Download NOVA3R checkpoints from HuggingFace.

CLI:
    nova3r-download                            # all models -> ./checkpoints
    nova3r-download --model scene_n1 --dest ./assets/ckpts
    nova3r-download --repo other/repo --force

Programmatic:
    from nova3r.scripts.download_checkpoints import download_checkpoints
    download_checkpoints("scene_n1", dest="./checkpoints")
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_REPO = "wrchen530/nova3r"
DEFAULT_DEST = "./checkpoints"
KNOWN_MODELS = ("scene_ae", "scene_n1", "scene_n2")


def download_checkpoints(
    model: str = "all",
    dest: str | os.PathLike = DEFAULT_DEST,
    repo: str = DEFAULT_REPO,
    force: bool = False,
) -> list[Path]:
    """Download one or all NOVA3R checkpoints into ``dest``.

    Each model lands under ``<dest>/<model>/`` together with its ``.hydra/``
    sidecar (``config.yaml``, ``hydra.yaml``, ``overrides.yaml``).

    Parameters
    ----------
    model
        ``"all"`` or one of ``KNOWN_MODELS``.
    dest
        Destination directory (created if missing). Relative paths resolve
        against the current working directory of the caller.
    repo
        HuggingFace repo id.
    force
        If ``True``, redownload files that already exist locally.

    Returns
    -------
    list[Path]
        Paths to each downloaded model directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "download_checkpoints requires `huggingface_hub>=0.22`."
        ) from e

    if model == "all":
        targets: Iterable[str] = KNOWN_MODELS
    elif model in KNOWN_MODELS:
        targets = (model,)
    else:
        raise ValueError(
            f"Unknown model '{model}'. Expected 'all' or one of {KNOWN_MODELS}."
        )

    dest = Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    out: list[Path] = []
    for name in targets:
        target_dir = dest / name
        ckpt_path = target_dir / "checkpoint-last.pth"
        if ckpt_path.exists() and not force:
            print(f"[skip] {name}: already at {target_dir}")
            out.append(target_dir)
            continue

        print(f"[download] {name} -> {target_dir}")
        snapshot_download(
            repo_id=repo,
            allow_patterns=[f"{name}/*", f"{name}/.hydra/*"],
            local_dir=str(dest),
            force_download=force,
        )
        out.append(target_dir)
        print(f"[done] {name}")

    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nova3r-download",
        description="Download NOVA3R checkpoints from HuggingFace.",
    )
    p.add_argument(
        "--model",
        default="all",
        choices=("all", *KNOWN_MODELS),
        help="Which model to download (default: all).",
    )
    p.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST}).",
    )
    p.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HuggingFace repo id (default: {DEFAULT_REPO}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``nova3r-download`` console script.

    Parses CLI arguments and delegates to :func:`download_checkpoints`. Returns
    ``0`` on success, ``1`` on failure; a hint about ``huggingface-cli login``
    is printed for typical auth errors.
    """
    args = _build_parser().parse_args(argv)
    try:
        download_checkpoints(
            model=args.model,
            dest=args.dest,
            repo=args.repo,
            force=args.force,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        if "401" in str(e) or "gated" in str(e).lower():
            print(
                "hint: run `huggingface-cli login` or export HF_TOKEN=... for gated repos.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
