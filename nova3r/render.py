# Copyright (c) 2026 Weirong Chen
"""Nova3r point-cloud rendering helpers.

Composable building blocks to load a ``.ply``, align/recolor/decorate it, and
render a turntable video. All heavy dependencies (``pytorch3d``, ``open3d``,
``matplotlib``, ``imageio``) are lazy-imported; install them via the
``[render]`` extra::

    pip install nova3r[render]

Programmatic usage mirrors the legacy script
:file:`demo/visualization/render_points.py`; a CLI is provided as
``nova3r-render`` (see :mod:`nova3r.scripts.render`).
"""
from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from nova3r.utils.device import resolve_device

if TYPE_CHECKING:
    from pytorch3d.structures import Pointclouds

__all__ = [
    "load_pointcloud",
    "align_pca",
    "align_obb",
    "center",
    "place_on_floor",
    "flip_axis",
    "smooth_surface",
    "colorize",
    "add_bbox",
    "render_turntable_frames",
    "save_video",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require(pkg: str, extra: str = "render"):
    """Lazy-import ``pkg``; raise ``ImportError`` pointing to the extra."""
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        top = pkg.split(".")[0]
        raise ImportError(
            f"This functionality requires `{top}`. "
            f"Install with `pip install nova3r[{extra}]`."
        ) from e


def _o3d_pcd_from_tensor(pts: torch.Tensor):
    """Build an Open3D ``PointCloud`` from a tensor (CPU-numpy roundtrip)."""
    o3d = _require("open3d")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy())
    return pcd


def _pca_eigh(pts: torch.Tensor):
    """Centered PCA eigen-decomposition. Returns ``(evals_asc, evecs_asc, centered)``."""
    mean = pts.mean(dim=0, keepdim=True)
    centered = pts - mean
    cov = centered.t().mm(centered) / max(centered.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    return evals, evecs, centered


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pointcloud(
    ply_path: str,
    *,
    device=None,
    remove_outlier: bool = False,
    normalize: bool = True,
) -> "Pointclouds":
    """Load a ``.ply`` into a :class:`pytorch3d.structures.Pointclouds`.

    ``remove_outlier``: statistical-outlier removal via Open3D.
    ``normalize``: center and fit into the box ``[-0.7, 0.7]^3``.
    """
    p3d_io = _require("pytorch3d.io")
    p3d_struct = _require("pytorch3d.structures")
    device = resolve_device(device)

    verts, _faces = p3d_io.load_ply(ply_path)
    if remove_outlier:
        o3d = _require("open3d")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts.cpu().numpy())
        _cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        verts = torch.from_numpy(np.asarray(pcd.points)[ind]).to(verts.dtype)

    rgb = torch.full((verts.shape[0], 3), 0.7).to(verts.dtype)
    verts = verts.to(device)
    rgb = rgb.to(device)

    if normalize:
        verts = verts - verts.mean(dim=0, keepdim=True)
        max_abs = verts.abs().max(dim=0).values.max()
        verts = verts / (max_abs + 1e-8) * 0.7

    return p3d_struct.Pointclouds(points=[verts], features=[rgb])


# ---------------------------------------------------------------------------
# Alignment / centering
# ---------------------------------------------------------------------------

def align_pca(pc: "Pointclouds") -> "Pointclouds":
    """Center and rotate so axes align with the principal components (descending eigenvalue)."""
    p3d_struct = _require("pytorch3d.structures")
    pts = pc.points_list()[0]
    feats = pc.features_list()[0]
    evals, evecs, centered = _pca_eigh(pts)
    order = torch.argsort(evals, descending=True)
    rotated = centered @ evecs[:, order]
    return p3d_struct.Pointclouds(points=[rotated], features=[feats])


def align_obb(pc: "Pointclouds") -> "Pointclouds":
    """Align point cloud to Open3D's oriented bounding box axes."""
    p3d_struct = _require("pytorch3d.structures")
    pts = pc.points_list()[0]
    feats = pc.features_list()[0]

    pcd = _o3d_pcd_from_tensor(pts)
    obb = pcd.get_oriented_bounding_box(robust=True)
    R = torch.from_numpy(np.array(obb.R, copy=True)).to(pts.dtype).to(pts.device)
    obb_center = torch.from_numpy(np.array(obb.center, copy=True)).to(pts.dtype).to(pts.device)
    aligned = (pts - obb_center) @ R
    return p3d_struct.Pointclouds(points=[aligned], features=[feats])


def center(pc: "Pointclouds") -> "Pointclouds":
    """Shift point cloud so its mean is at the origin."""
    p3d_struct = _require("pytorch3d.structures")
    pts = pc.points_list()[0]
    feats = pc.features_list()[0]
    return p3d_struct.Pointclouds(points=[pts - pts.mean(dim=0, keepdim=True)], features=[feats])


def place_on_floor(pc: "Pointclouds", axis: str = "z") -> "Pointclouds":
    """Shift so the minimum along ``axis`` lands at ``0``."""
    p3d_struct = _require("pytorch3d.structures")
    pts = pc.points_list()[0]
    feats = pc.features_list()[0]
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    shifted = pts.clone()
    shifted[:, idx] = shifted[:, idx] - shifted[:, idx].min()
    return p3d_struct.Pointclouds(points=[shifted], features=[feats])


def flip_axis(pc: "Pointclouds", axis: str = "y") -> "Pointclouds":
    """Negate coordinates along ``axis``."""
    p3d_struct = _require("pytorch3d.structures")
    pts = pc.points_list()[0]
    feats = pc.features_list()[0]
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    flipped = pts.clone()
    flipped[:, idx] = -flipped[:, idx]
    return p3d_struct.Pointclouds(points=[flipped], features=[feats])


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_surface(
    pc: "Pointclouds",
    *,
    device=None,
    num_iterations: int = 3,
    neighbor_size: int = 30,
) -> "Pointclouds":
    """Iterative Laplacian smoothing in point space via kNN averaging; features preserved."""
    p3d_struct = _require("pytorch3d.structures")
    p3d_ops = _require("pytorch3d.ops")
    device = resolve_device(device)

    pts = pc.points_list()[0].to(device)
    feats = pc.features_list()[0].to(device)
    points = pts.unsqueeze(0)  # [1, N, 3]

    for _ in range(num_iterations):
        knn = p3d_ops.knn_points(points, points, K=neighbor_size + 1)
        knn_idx = knn.idx[0, :, 1:]  # exclude self
        smoothed = points[0, knn_idx].mean(dim=1)
        points = 0.5 * points + 0.5 * smoothed.unsqueeze(0)

    return p3d_struct.Pointclouds(points=[points[0]], features=[feats])


# ---------------------------------------------------------------------------
# Coloring
# ---------------------------------------------------------------------------

def colorize(
    pc: "Pointclouds",
    *,
    device=None,
    mode: str = "normal",
    normal_neighbors: int = 60,
) -> "Pointclouds":
    """Recolor a point cloud. ``mode`` is one of ``normal``, ``plasma``, ``viridis``, ``xyz``."""
    p3d_struct = _require("pytorch3d.structures")
    device = resolve_device(device)

    if mode == "normal":
        p3d_ops = _require("pytorch3d.ops")
        # Estimate normals on CPU to avoid GPU OOM for large point clouds.
        pc_cpu = pc.to("cpu")
        normals = p3d_ops.estimate_pointcloud_normals(pc_cpu, neighborhood_size=normal_neighbors)
        normal_colors = (normals + 1.0) * 0.5
        if normal_colors.dim() == 3:
            normal_colors = normal_colors[0]
        features = [normal_colors.to(device)]
    elif mode in ("plasma", "viridis"):
        mpl = _require("matplotlib")
        pts = pc.points_list()[0]
        evals, evecs, centered = _pca_eigh(pts)
        order = torch.argsort(evals, descending=False)
        pca_coords = centered @ evecs[:, order[0]]
        q_low = torch.quantile(pca_coords, 0.02)
        q_high = torch.quantile(pca_coords, 0.98)
        z = (pca_coords - q_low) / (q_high - q_low + 1e-8)
        cmap = mpl.colormaps[mode]
        colors = torch.from_numpy(cmap(z.cpu().numpy())[:, :3]).float()
        features = [colors.to(device)]
    elif mode == "xyz":
        pts = pc.points_list()[0]
        xyz_min = pts.min(dim=0).values
        xyz_max = pts.max(dim=0).values
        xyz_norm = (pts - xyz_min) / (xyz_max - xyz_min + 1e-8)
        features = [xyz_norm.clamp(0.0, 1.0).to(device)]
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'normal', 'plasma', 'viridis', or 'xyz'."
        )

    return p3d_struct.Pointclouds(points=pc.points_list(), features=features)


# ---------------------------------------------------------------------------
# Bounding box decoration
# ---------------------------------------------------------------------------

def add_bbox(
    pc: "Pointclouds",
    *,
    device=None,
    color: Sequence[float] = (1.0, 0.0, 0.0),
    steps: int = 100,
    pca_clip: float = 0.0,
) -> "Pointclouds":
    """Densify an Open3D oriented bounding box's edges and merge them into ``pc``.

    ``pca_clip > 0`` drops top/bottom quantile along the first PCA axis before
    fitting the OBB (original points are unaffected).
    """
    p3d_struct = _require("pytorch3d.structures")
    o3d = _require("open3d")
    device = resolve_device(device)

    pts = pc.points_list()[0]
    feats = pc.features_list()[0]

    obb_pts = pts
    if pca_clip > 0.0:
        evals, evecs, centered = _pca_eigh(obb_pts)
        order = torch.argsort(evals, descending=False)
        pca_coords = centered @ evecs[:, order[0]]
        q_low = torch.quantile(pca_coords, pca_clip)
        q_high = torch.quantile(pca_coords, 1.0 - pca_clip)
        mask = (pca_coords >= q_low) & (pca_coords <= q_high)
        if mask.any():
            obb_pts = obb_pts[mask]

    pcd = _o3d_pcd_from_tensor(obb_pts)
    obb = pcd.get_oriented_bounding_box()
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    box_pts_np = np.asarray(lineset.points, dtype=np.float32)
    edges = np.asarray(lineset.lines, dtype=np.int64)

    t = torch.linspace(0.0, 1.0, steps=steps, device=pts.device, dtype=pts.dtype)
    box_pts_tensor = torch.from_numpy(box_pts_np).to(device=pts.device, dtype=pts.dtype)
    box_segments = []
    for i, j in edges:
        start = box_pts_tensor[i]
        end = box_pts_tensor[j]
        seg = start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]
        box_segments.append(seg)
    box_pts = torch.cat(box_segments, dim=0)

    box_color = torch.tensor(color, device=pts.device, dtype=feats.dtype).view(1, 3)
    box_feats = box_color.repeat(box_pts.shape[0], 1)

    merged_pts = torch.cat([pts, box_pts], dim=0).to(device)
    merged_feats = torch.cat([feats, box_feats], dim=0).to(device)
    return p3d_struct.Pointclouds(points=[merged_pts], features=[merged_feats])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_turntable_frames(
    pc: "Pointclouds",
    *,
    num_frames: int = 120,
    distance: float = 20.0,
    elevation: float = 10.0,
    azim_start: float = 0.0,
    azim_end: float = 360.0,
    image_size: int = 512,
    radius: float = 0.003,
    points_per_pixel: int = 10,
    background_color: Sequence[float] = (1.0, 1.0, 1.0),
    compositor: str = "alpha",
) -> list[np.ndarray]:
    """Render ``num_frames`` turntable views as ``H x W x 3`` uint8 arrays."""
    p3d_render = _require("pytorch3d.renderer")
    raster_settings = p3d_render.PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel,
    )

    if compositor == "alpha":
        compositor_impl = p3d_render.AlphaCompositor(background_color=background_color)
    elif compositor == "normweighted":
        compositor_impl = p3d_render.NormWeightedCompositor(background_color=background_color)
    else:
        raise ValueError(
            f"Unknown compositor '{compositor}'. Use 'alpha' or 'normweighted'."
        )

    frames: list[np.ndarray] = []
    azims = np.linspace(azim_start, azim_end, num_frames, endpoint=False)
    for azim in azims:
        R, T = p3d_render.look_at_view_transform(distance, elevation, azim)
        cameras = p3d_render.FoVOrthographicCameras(device=pc.device, R=R, T=T, znear=0.01)
        rasterizer = p3d_render.PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = p3d_render.PointsRenderer(rasterizer=rasterizer, compositor=compositor_impl)
        images = renderer(pc)
        img = images[0, ..., :3].detach().cpu().numpy()
        frames.append((img * 255).clip(0, 255).astype(np.uint8))
    return frames


def save_video(
    frames: Sequence[np.ndarray],
    out_path: str,
    *,
    fps: int = 30,
) -> str:
    """Write ``frames`` to ``out_path`` (mp4); falls back to ``.gif`` on failure.

    Returns the actual file path written.
    """
    imageio = _require("imageio.v2")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        imageio.mimsave(out_path, list(frames), fps=fps)
        return out_path
    except Exception as e:
        fallback = out_path.rsplit(".", 1)[0] + ".gif"
        imageio.mimsave(fallback, list(frames), fps=fps)
        print(f"MP4 save failed ({e}). Saved GIF to {fallback}")
        return fallback
