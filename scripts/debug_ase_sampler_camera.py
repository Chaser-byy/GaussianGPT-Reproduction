"""Experimental smoke test for ASE sampler camera selection.

Prefer debug_ase_dataloader.py when validating the training DataLoader path.
This script is useful when inspecting per-sample camera scoring details only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from gaussiangpt_ae.data.sampler import ASEOnlineChunkSampler


def _shape(value: Any) -> str:
    return str(tuple(np.asarray(value).shape))


def _array_list(value: Any) -> list:
    return np.asarray(value).tolist()


def _format_float(value: Any) -> str:
    return f"{float(value):.6f}"


def _print_camera(camera: dict, indent: str = "    ") -> None:
    print(f"{indent}- camera_id: {camera.get('camera_id')}")
    print(f"{indent}  frame_index: {camera.get('frame_index')}")
    print(f"{indent}  frame_id: {camera.get('frame_id')}")
    print(f"{indent}  file_path: {camera.get('file_path')}")
    print(f"{indent}  chunk_coverage: {_format_float(camera.get('chunk_coverage', 0.0))}")
    print(f"{indent}  image_coverage: {_format_float(camera.get('image_coverage', 0.0))}")
    print(f"{indent}  visible_ratio: {_format_float(camera.get('visible_ratio', 0.0))}")
    print(f"{indent}  visible_corners: {camera.get('visible_corners')}")
    print(f"{indent}  total_corners: {camera.get('total_corners')}")
    print(f"{indent}  valid_projection: {bool(camera.get('valid_projection'))}")
    print(f"{indent}  projected_bbox: {camera.get('projected_bbox')}")
    print(
        f"{indent}  projected_bbox_area: "
        f"{_format_float(camera.get('projected_bbox_area', 0.0))}"
    )
    print(f"{indent}  intersection_area: {_format_float(camera.get('intersection_area', 0.0))}")
    print(f"{indent}  depth_min: {camera.get('depth_min')}")
    print(f"{indent}  depth_max: {camera.get('depth_max')}")
    print(f"{indent}  near_plane_crossing: {bool(camera.get('near_plane_crossing'))}")
    print(f"{indent}  selection_mode: {camera.get('selection_mode')}")


def _print_sample(index: int, sample: dict) -> None:
    print(f"\n=== sample {index} ===")
    print(f"scene_id: {sample.get('scene_id')}")
    print(f"coords shape: {_shape(sample['coords'])}")
    print(f"feats shape: {_shape(sample['feats'])}")
    print(f"target_feats shape: {_shape(sample['target_feats'])}")
    print(f"z_mode: {sample.get('z_mode')}")
    print(f"chunk_min_voxel: {_array_list(sample['chunk_min_voxel'])}")
    print(f"chunk_max_voxel: {_array_list(sample['chunk_max_voxel'])}")
    print(f"chunk_shape_voxels: {_array_list(sample['chunk_shape_voxels'])}")
    print(f"occupancy: {_format_float(sample.get('occupancy', 0.0))}")
    print(f"num_occupied_voxels: {sample.get('num_occupied_voxels')}")

    camera_debug = sample.get("camera_debug", {})
    if camera_debug:
        print(f"camera_cache_path: {camera_debug.get('camera_cache_path')}")
        print(f"pose_convention: {camera_debug.get('pose_convention')}")
        print(
            "uses_transform_device_camera: "
            f"{camera_debug.get('uses_transform_device_camera')}"
        )

    top_cameras = sample.get("top_cameras") or camera_debug.get("top_cameras") or []
    print(f"top_cameras: {len(top_cameras)}")
    for camera in top_cameras:
        _print_camera(camera)


def _require_file(path: Path, label: str) -> None:
    try:
        exists = path.is_file()
    except PermissionError as exc:
        raise PermissionError(f"cannot access {label}: {path}") from exc
    if not exists:
        raise FileNotFoundError(f"missing {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test existing ASE voxel cache + camera cache + sampler. "
            "This does not train or render."
        )
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--z-mode",
        choices=["fixed_160", "full_height"],
        required=True,
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--top-k-cameras", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    scene_cache = cache_root / "scenes" / f"{args.scene_id}.npz"
    camera_cache = cache_root / "cameras" / f"{args.scene_id}_cameras.npz"
    _require_file(scene_cache, "scene voxel cache")
    _require_file(camera_cache, "camera cache")

    sampler = ASEOnlineChunkSampler(
        cache_root=cache_root,
        scene_id=args.scene_id,
        z_mode=args.z_mode,
        top_k_cameras=args.top_k_cameras,
        seed=args.seed,
    )

    print(f"cache_root: {cache_root}")
    print(f"scene_id: {args.scene_id}")
    print(f"scene_cache: {scene_cache}")
    print(f"camera_cache: {camera_cache}")
    print(f"z_mode: {args.z_mode}")
    print(f"num_samples: {args.num_samples}")
    print(f"top_k_cameras: {args.top_k_cameras}")
    print(f"seed: {args.seed}")

    for index in range(args.num_samples):
        _print_sample(index, sampler.sample())


if __name__ == "__main__":
    main()
