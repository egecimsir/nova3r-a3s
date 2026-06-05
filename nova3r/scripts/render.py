# Copyright (c) 2026 Weirong Chen
"""Render a turntable video from a NOVA3R ``.ply`` point cloud.

CLI:
    nova3r-render --ply path/to/cloud.ply --out out.mp4
    nova3r-render --ply-folder dir/ --save-folder vids/ --color-type plasma

Programmatic:
    from nova3r.scripts.render import main
    main(["--ply", "x.ply", "--out", "x.mp4"])
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from nova3r.render import (
    add_bbox,
    align_obb,
    align_pca,
    center,
    colorize,
    flip_axis,
    load_pointcloud,
    place_on_floor,
    render_turntable_frames,
    save_video,
)
from nova3r.utils.device import get_default_device

DEFAULT_OUT = "./demo/outputs/pointcloud_360_normals.mp4"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nova3r-render",
        description="Render a 360 point-cloud video from a .ply file.",
    )
    p.add_argument("--ply", help="Path to input .ply file")
    p.add_argument("--ply-folder", help="Folder containing .ply files")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output video path")
    p.add_argument("--save-folder", default=None, help="Output folder for batch rendering")
    p.add_argument("--num-frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--distance", type=float, default=20.0)
    p.add_argument("--elevation", type=float, default=10.0)
    p.add_argument("--azim-start", type=float, default=0.0,
                   help="Starting azimuth angle (degrees)")
    p.add_argument("--azim-end", type=float, default=None,
                   help="Ending azimuth (degrees). If unset, uses azim_start + 360.")
    p.add_argument("--radius", type=float, default=0.005)
    p.add_argument("--points-per-pixel", type=int, default=10)
    p.add_argument("--normal-neighbors", type=int, default=60)
    p.add_argument("--compositor", choices=["alpha", "normweighted"], default="alpha",
                   help="Compositor type for point blending")
    p.add_argument("--color-type", choices=["normal", "plasma", "viridis", "xyz"],
                   default="viridis", help="Point color scheme")
    p.add_argument("--bbox", action="store_true", help="Draw oriented bounding box")
    p.add_argument("--bbox-steps", type=int, default=100, help="Samples per bbox edge")
    p.add_argument("--bbox-color", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                   help="BBox color as three floats in [0,1]")
    p.add_argument("--bbox-pca-clip", type=float, default=0.0,
                   help="Drop top/bottom quantile along first PCA axis for OBB (e.g. 0.01)")
    p.add_argument("--pca", action="store_true",
                   help="Align point cloud to PCA axes before rendering")
    p.add_argument("--obb", action="store_true",
                   help="Align point cloud to Open3D OBB axes before rendering")
    p.add_argument("--floor", action="store_true", help="Shift point cloud so min z is at 0")
    p.add_argument("--flip-axis", choices=["x", "y", "z"], default=None,
                   help="Flip point cloud along axis (useful if upside-down)")
    p.add_argument("--center", action="store_true", help="Center point cloud at origin")
    return p


def _render_one(args: argparse.Namespace, ply_path: str, out_path: str, device) -> None:
    pc = load_pointcloud(ply_path, device=device, remove_outlier=True)
    if args.pca:
        pc = align_pca(pc)
    if args.obb:
        pc = align_obb(pc)
    if args.flip_axis:
        pc = flip_axis(pc, axis=args.flip_axis)
    if args.floor:
        pc = place_on_floor(pc, axis="z")
    if args.center:
        pc = center(pc)
    pc = colorize(
        pc, device=device, mode=args.color_type, normal_neighbors=args.normal_neighbors
    )
    if args.bbox:
        pc = add_bbox(
            pc,
            device=device,
            color=tuple(args.bbox_color),
            steps=args.bbox_steps,
            pca_clip=args.bbox_pca_clip,
        )
    azim_end = args.azim_end if args.azim_end is not None else args.azim_start + 360.0
    frames = render_turntable_frames(
        pc,
        num_frames=args.num_frames,
        distance=args.distance,
        elevation=args.elevation,
        azim_start=args.azim_start,
        azim_end=azim_end,
        image_size=args.image_size,
        radius=args.radius,
        points_per_pixel=args.points_per_pixel,
        compositor=args.compositor,
    )
    written = save_video(frames, out_path, fps=args.fps)
    print(f"Saved video to {written}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``nova3r-render`` console script."""
    import torch  # local: heavy import only on actual invocation
    args = _build_parser().parse_args(argv)

    device = get_default_device()
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if not args.ply and not args.ply_folder:
        print("error: provide --ply or --ply-folder.", file=sys.stderr)
        return 1
    if args.ply and args.ply_folder:
        print("error: provide only one of --ply or --ply-folder.", file=sys.stderr)
        return 1
    if args.ply_folder and not args.save_folder:
        print("error: --ply-folder requires --save-folder.", file=sys.stderr)
        return 1

    if args.ply:
        _render_one(args, args.ply, args.out, device)
    else:
        os.makedirs(args.save_folder, exist_ok=True)
        ply_files = sorted(
            f for f in os.listdir(args.ply_folder) if f.lower().endswith(".ply")
        )
        if not ply_files:
            print(f"error: no .ply files in {args.ply_folder}", file=sys.stderr)
            return 1
        for name in ply_files:
            ply_path = os.path.join(args.ply_folder, name)
            out_path = os.path.join(args.save_folder, os.path.splitext(name)[0] + ".mp4")
            _render_one(args, ply_path, out_path, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
