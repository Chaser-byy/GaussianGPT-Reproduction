"""Inspect one ASE autoencoder batch without starting training.

Usage:
    python scripts/debug_ase_autoencoder_batch.py --config configs/autoencoder_ase.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pprint import pformat

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAUSSIANGPT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(GAUSSIANGPT_ROOT)
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for path in (SCRIPT_DIR, GAUSSIANGPT_ROOT, SRC_ROOT):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from train_autoencoder import build_ase_dataset, load_config  # noqa: E402
from gaussiangpt_ae.data.collate import ase_sparse_collate  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/autoencoder_ase.yaml")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--scene_id", type=str, default=None)
    return parser.parse_args()


def _tensor_range(name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach()
    finite = torch.isfinite(tensor)
    has_bad = not bool(finite.all().item())
    if tensor.numel() == 0:
        print(f"{name}: empty")
        return
    if finite.any():
        finite_values = tensor[finite]
        print(
            f"{name}: min={finite_values.min().item():.6g} "
            f"max={finite_values.max().item():.6g} "
            f"nan_or_inf={has_bad}"
        )
    else:
        print(f"{name}: no finite values nan_or_inf={has_bad}")


def _print_meta(meta: dict) -> None:
    print(f"scene_id: {meta.get('scene_id')}")
    for key in (
        "scene_origin",
        "chunk_min_voxel",
        "chunk_max_voxel",
        "chunk_world_min",
        "chunk_world_max",
        "voxel_size",
    ):
        print(f"{key}: {meta.get(key)}")

    top_cameras = meta.get("top_cameras") or []
    print(f"top_cameras count: {len(top_cameras)}")
    if top_cameras:
        print(f"top_cameras[0]: {pformat(top_cameras[0])}")

    camera_debug = meta.get("camera_debug") or {}
    camera_debug_summary = {
        "camera_cache_path": camera_debug.get("camera_cache_path"),
        "pose_convention": camera_debug.get("pose_convention"),
        "uses_transform_device_camera": camera_debug.get("uses_transform_device_camera"),
        "top_cameras_count": len(camera_debug.get("top_cameras") or []),
    }
    print(f"camera_debug: {pformat(camera_debug_summary)}")


def _world_centers(coords_xyz: torch.Tensor, meta: dict) -> torch.Tensor:
    scene_origin = torch.as_tensor(meta["scene_origin"], dtype=torch.float32)
    chunk_min = torch.as_tensor(meta["chunk_min_voxel"], dtype=torch.float32)
    voxel_size = float(meta["voxel_size"])
    return scene_origin + (chunk_min + coords_xyz.to(torch.float32) + 0.5) * voxel_size


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("data", {})["dataset"] = "ase"
    if args.cache_root is not None:
        cfg["data"]["cache_root"] = args.cache_root
    if args.scene_id is not None:
        cfg["data"][f"{args.split}_scene_ids"] = [args.scene_id]

    if cfg["data"].get("cache_root") is None:
        print(
            "data.cache_root is null. Set it in the config or pass "
            "--cache_root /path/to/ase_cache."
        )
        return 2

    dataset = build_ase_dataset(cfg, split=args.split)
    batch_size = int(args.batch_size or cfg.get("training", {}).get("batch_size", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ase_sparse_collate,
    )
    batch = next(iter(loader))

    coords = batch["coords"]
    feats = batch["feats"]
    target_feats = batch["target_feats"]
    metas = batch["metas"]
    coords_xyz = coords[:, 1:4]

    print(f"coords shape: {tuple(coords.shape)}")
    if coords.numel() > 0:
        print(f"coords xyz min: {coords_xyz.min(dim=0).values.tolist()}")
        print(f"coords xyz max: {coords_xyz.max(dim=0).values.tolist()}")
    print(f"feats shape: {tuple(feats.shape)}")
    print(f"target_feats shape: {tuple(target_feats.shape)}")
    print(f"coords has NaN/Inf: {not torch.isfinite(coords.to(torch.float32)).all().item()}")
    print(f"feats has NaN/Inf: {not torch.isfinite(feats).all().item()}")
    print(f"target_feats has NaN/Inf: {not torch.isfinite(target_feats).all().item()}")

    _tensor_range("offset feats[:, 0:3]", feats[:, 0:3])
    _tensor_range("color feats[:, 3:6]", feats[:, 3:6])
    _tensor_range("opacity feats[:, 6:7]", feats[:, 6:7])
    _tensor_range("scale feats[:, 7:10]", feats[:, 7:10])
    _tensor_range("rotation feats[:, 10:14]", feats[:, 10:14])

    if not metas:
        print("metas: empty")
        return 1

    meta = metas[0]
    print("\nFirst sample metadata")
    _print_meta(meta)

    first_mask = coords[:, 0] == 0
    first_coords = coords[first_mask, 1:4]
    world_center = _world_centers(first_coords, meta)
    print(f"world_center min: {world_center.min(dim=0).values.tolist()}")
    print(f"world_center max: {world_center.max(dim=0).values.tolist()}")

    chunk_shape = torch.as_tensor(meta["chunk_shape_voxels"], dtype=torch.long)
    local_min_ok = bool((first_coords >= 0).all().item())
    local_max_ok = bool((first_coords < chunk_shape).all().item())
    print(f"coords chunk-local min>=0: {local_min_ok}")
    print(f"coords chunk-local max<chunk_shape_voxels: {local_max_ok}")
    if not (local_min_ok and local_max_ok):
        print("WARNING: coords do not look chunk-local; check for double origin handling.")
        return 1
    print("coords appear chunk-local; do not subtract chunk_min_voxel before AE input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
