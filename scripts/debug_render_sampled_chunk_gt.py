"""Render a sampled ASE chunk GT image from the original 3DGS PLY."""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from gaussiangpt_ae.data.ase import load_ase_camera_cache
from gaussiangpt_ae.data.ply_io import read_gaussian_ply
from gaussiangpt_ae.data.sampler import ASEOnlineChunkSampler

SH_DEGREE = 0
Z_NEAR = 0.01
Z_FAR = 1000.0


def _shape(value: Any) -> list:
    return list(np.asarray(value).shape)


def _array_list(value: Any) -> list:
    return np.asarray(value).tolist()


def _camera_to_json(camera: Dict) -> Dict:
    keys = (
        "camera_id",
        "frame_index",
        "frame_id",
        "file_path",
        "image_coverage",
        "visible_ratio",
        "valid_projection",
        "selection_mode",
    )
    return {key: camera.get(key) for key in keys}


def _print_sample(sample: Dict) -> None:
    print("sample:")
    print(f"  coords shape: {tuple(sample['coords'].shape)}")
    print(f"  feats shape: {tuple(sample['feats'].shape)}")
    print(f"  target_feats shape: {tuple(sample['target_feats'].shape)}")
    print(f"  z_mode: {sample['z_mode']}")
    print(f"  chunk_min_voxel: {_array_list(sample['chunk_min_voxel'])}")
    print(f"  chunk_max_voxel: {_array_list(sample['chunk_max_voxel'])}")
    print(f"  chunk_world_min: {_array_list(sample['chunk_world_min'])}")
    print(f"  chunk_world_max: {_array_list(sample['chunk_world_max'])}")
    print("  top_cameras:")
    for camera in sample.get("top_cameras", []):
        print(f"    - {_camera_to_json(camera)}")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return (value / (np.linalg.norm(value, axis=1, keepdims=True) + 1e-8)).astype(
        np.float32
    )


def _make_projection_matrix(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    znear: float = Z_NEAR,
    zfar: float = Z_FAR,
) -> np.ndarray:
    # 3DGS rasterizer expects the same perspective form used by the original
    # renderer. Principal-point offsets come from K; no pose flip is applied.
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = 2.0 * float(fx) / float(width)
    projection[1, 1] = 2.0 * float(fy) / float(height)
    projection[0, 2] = 2.0 * float(cx) / float(width) - 1.0
    projection[1, 2] = 2.0 * float(cy) / float(height) - 1.0
    projection[2, 2] = float(zfar) / float(zfar - znear)
    projection[2, 3] = -float(zfar * znear) / float(zfar - znear)
    projection[3, 2] = 1.0
    return projection


def _import_render_deps():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for debug GT rendering.") from exc

    try:
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
    except ImportError as exc:
        raise ImportError(
            "diff_gaussian_rasterization is required for this debug render script. "
            "Install/run it in the CUDA AutoDL environment."
        ) from exc

    return torch, GaussianRasterizationSettings, GaussianRasterizer


def _write_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    image_u8 = (image * 255.0 + 0.5).astype(np.uint8)
    if image_u8.ndim != 3 or image_u8.shape[2] != 3:
        raise ValueError(f"PNG image must have shape [H, W, 3], got {image_u8.shape}")

    height, width, _ = image_u8.shape
    raw_rows = [b"\x00" + image_u8[row].tobytes() for row in range(height)]
    raw = b"".join(raw_rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(raw, level=6)),
        chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(payload))


def _require_file(path: Path, label: str) -> None:
    try:
        exists = path.is_file()
    except PermissionError as exc:
        raise PermissionError(f"cannot access {label}: {path}") from exc
    if not exists:
        raise FileNotFoundError(f"missing {label}: {path}")


def _render_chunk(
    xyz: np.ndarray,
    shs: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    c2w: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, Dict]:
    torch, GaussianRasterizationSettings, GaussianRasterizer = _import_render_deps()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for diff_gaussian_rasterization debug render.")

    device = torch.device("cuda")
    means3d = torch.as_tensor(xyz, dtype=torch.float32, device=device)
    means2d = torch.zeros_like(
        means3d, dtype=torch.float32, device=device, requires_grad=True
    )
    shs_tensor = torch.as_tensor(shs, dtype=torch.float32, device=device)
    opacities = torch.as_tensor(opacity, dtype=torch.float32, device=device)
    scales = torch.as_tensor(scale, dtype=torch.float32, device=device)
    rotations = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    bg = torch.zeros(3, dtype=torch.float32, device=device)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    tanfovx = float(width) / (2.0 * fx)
    tanfovy = float(height) / (2.0 * fy)
    projection = _make_projection_matrix(width, height, fx, fy, cx, cy)

    viewmatrix = torch.as_tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
    projmatrix = torch.as_tensor(projection, dtype=torch.float32, device=device).transpose(0, 1)
    full_proj = viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0)).squeeze(0)
    campos = torch.as_tensor(c2w[:3, 3], dtype=torch.float32, device=device)

    settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj,
        sh_degree=SH_DEGREE,
        campos=campos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    rendered, radii = rasterizer(
        means3D=means3d,
        means2D=means2d,
        shs=shs_tensor,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    image = rendered.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    stats = {
        "radii_positive": int((radii.detach() > 0).sum().item()),
        "image_min": float(image.min()) if image.size else 0.0,
        "image_max": float(image.max()) if image.size else 0.0,
        "image_mean": float(image.mean()) if image.size else 0.0,
    }
    return image.astype(np.float32), stats


def _prepare_chunk_gaussians(
    scene,
    chunk_world_min: np.ndarray,
    chunk_world_max: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    inside = np.all(
        (scene.xyz >= chunk_world_min.astype(np.float32))
        & (scene.xyz < chunk_world_max.astype(np.float32)),
        axis=1,
    )
    indices = np.nonzero(inside)[0].astype(np.int64)
    raw_count = int(scene.xyz.shape[0])
    chunk_count = int(indices.shape[0])
    if max_points > 0 and chunk_count > max_points:
        rng = np.random.RandomState(seed)
        indices = np.sort(rng.choice(indices, size=int(max_points), replace=False))

    xyz = scene.xyz[indices].astype(np.float32, copy=False)
    shs = scene.color[indices].astype(np.float32, copy=False)[:, None, :]
    opacity = _sigmoid(scene.opacity[indices]).astype(np.float32, copy=False)
    # Raw 3DGS render convention uses exp(raw_scale). AE softplus scale is not used here.
    scale = np.exp(scene.scale[indices]).astype(np.float32, copy=False)
    rotation = _normalize_quaternion(scene.rotation[indices])
    stats = {
        "raw_gaussian_count": raw_count,
        "chunk_gaussian_count": chunk_count,
        "render_gaussian_count": int(indices.shape[0]),
        "chunk_gaussian_ratio": float(chunk_count / raw_count) if raw_count else 0.0,
        "max_points": int(max_points),
    }
    return {
        "xyz": xyz,
        "shs": shs,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
    }, stats


def _sample_chunk(args) -> Dict:
    sampler = ASEOnlineChunkSampler(
        cache_root=args.cache_root,
        scene_id=args.scene_id,
        z_mode=args.z_mode,
        top_k_cameras=args.top_k_cameras,
        seed=args.seed,
    )
    sample = None
    for _ in range(int(args.sample_index) + 1):
        sample = sampler.sample()
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample one ASE chunk and render chunk-only GT from the original PLY."
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--z-mode", choices=["fixed_160", "full_height"], required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-k-cameras", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=0)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    ply_path = raw_root / args.scene_id / "ckpts" / "point_cloud_30000.ply"
    camera_cache_path = cache_root / "cameras" / f"{args.scene_id}_cameras.npz"
    _require_file(ply_path, "original PLY")
    _require_file(camera_cache_path, "camera cache")

    sample = _sample_chunk(args)
    _print_sample(sample)
    top_cameras = sample.get("top_cameras") or []
    if not top_cameras:
        raise RuntimeError("sampler returned no top_cameras")
    selected_camera = top_cameras[0]
    frame_index = int(selected_camera["frame_index"])

    scene = read_gaussian_ply(ply_path, scene_id=args.scene_id)
    chunk_gaussians, gaussian_stats = _prepare_chunk_gaussians(
        scene=scene,
        chunk_world_min=np.asarray(sample["chunk_world_min"], dtype=np.float32),
        chunk_world_max=np.asarray(sample["chunk_world_max"], dtype=np.float32),
        max_points=int(args.max_points),
        seed=int(args.seed) + int(args.sample_index),
    )
    print("gaussians:")
    print(f"  raw gaussian count: {gaussian_stats['raw_gaussian_count']}")
    print(f"  chunk gaussian count: {gaussian_stats['chunk_gaussian_count']}")
    print(f"  chunk gaussian ratio: {gaussian_stats['chunk_gaussian_ratio']:.8f}")
    print(f"  render gaussian count: {gaussian_stats['render_gaussian_count']}")
    if gaussian_stats["render_gaussian_count"] <= 0:
        raise RuntimeError("sampled chunk contains no original PLY Gaussians to render")

    cameras = load_ase_camera_cache(camera_cache_path, scene_id=args.scene_id)
    c2w = np.asarray(cameras.c2w[frame_index], dtype=np.float32)
    w2c = np.asarray(cameras.w2c[frame_index], dtype=np.float32)
    K = np.asarray(cameras.K, dtype=np.float32)
    image, render_stats = _render_chunk(
        xyz=chunk_gaussians["xyz"],
        shs=chunk_gaussians["shs"],
        opacity=chunk_gaussians["opacity"],
        scale=chunk_gaussians["scale"],
        rotation=chunk_gaussians["rotation"],
        c2w=c2w,
        w2c=w2c,
        K=K,
        width=int(cameras.width),
        height=int(cameras.height),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "gt_chunk.png"
    debug_path = out_dir / "sample_debug.json"
    _write_png(image_path, image)

    debug = {
        "scene_id": args.scene_id,
        "cache_root": str(cache_root),
        "raw_root": str(raw_root),
        "ply_path": str(ply_path),
        "camera_cache_path": str(camera_cache_path),
        "z_mode": sample["z_mode"],
        "sample_index": int(args.sample_index),
        "seed": int(args.seed),
        "coords_shape": _shape(sample["coords"]),
        "feats_shape": _shape(sample["feats"]),
        "target_feats_shape": _shape(sample["target_feats"]),
        "chunk_min_voxel": _array_list(sample["chunk_min_voxel"]),
        "chunk_max_voxel": _array_list(sample["chunk_max_voxel"]),
        "chunk_world_min": _array_list(sample["chunk_world_min"]),
        "chunk_world_max": _array_list(sample["chunk_world_max"]),
        "selected_camera": _camera_to_json(selected_camera),
        "top_cameras": [_camera_to_json(camera) for camera in top_cameras],
        "pose_convention": cameras.pose_convention,
        "uses_transform_device_camera": cameras.uses_transform_device_camera,
        "gaussian_stats": gaussian_stats,
        "render_stats": render_stats,
        "output_image": str(image_path),
    }
    debug_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")
    print(f"saved image: {image_path}")
    print(f"saved debug: {debug_path}")


if __name__ == "__main__":
    main()
