"""Smoke test ASEChunkDataset and DataLoader on prebuilt ASE caches."""

from __future__ import annotations

import argparse
from typing import Any, Optional

from gaussiangpt_ae.data.collate import ase_sparse_collate
from gaussiangpt_ae.data.dataset import ASEChunkDataset


def _as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _camera_brief(camera: Optional[dict]) -> dict:
    if not camera:
        return {}
    keys = (
        "camera_id",
        "frame_index",
        "frame_id",
        "file_path",
        "chunk_coverage",
        "image_coverage",
        "visible_ratio",
        "valid_projection",
        "selection_mode",
    )
    return {key: camera.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ASE voxel/camera caches can be read by Dataset/DataLoader."
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--z-mode", choices=["fixed_160", "full_height"], required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--top-k-cameras", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=float, default=4.0)
    parser.add_argument("--occupancy-threshold", type=float, default=0.2)
    parser.add_argument("--max-candidate-chunks", type=int, default=10)
    args = parser.parse_args()

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required for debug_ase_dataloader.py.") from exc

    num_samples = max(1, int(args.batch_size) * int(args.num_batches))
    dataset = ASEChunkDataset(
        cache_root=args.cache_root,
        num_samples_per_epoch=num_samples,
        chunk_size=args.chunk_size,
        occupancy_threshold=args.occupancy_threshold,
        max_candidate_chunks=args.max_candidate_chunks,
        top_k_cameras=args.top_k_cameras,
        seed=args.seed,
        z_mode=args.z_mode,
        scene_ids=args.scene_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=ase_sparse_collate,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"cache_root: {args.cache_root}")
    print(f"scene_ids: {args.scene_ids if args.scene_ids else 'ALL'}")
    print(f"z_mode: {args.z_mode}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"num_batches: {args.num_batches}")
    print(f"top_k_cameras: {args.top_k_cameras}")
    print(f"seed: {args.seed}")

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.num_batches:
            break
        metas = batch["metas"]
        print(f"\n=== batch {batch_index} ===")
        print(f"coords shape: {tuple(batch['coords'].shape)}")
        print(f"feats shape: {tuple(batch['feats'].shape)}")
        print(f"target_feats shape: {tuple(batch['target_feats'].shape)}")
        print(f"batch scene_ids: {[meta.get('scene_id') for meta in metas]}")
        print(f"metadata keys: {sorted(metas[0].keys()) if metas else []}")
        for sample_index, meta in enumerate(metas):
            top_cameras = meta.get("top_cameras") or []
            print(f"  sample {sample_index}:")
            print(f"    scene_id: {meta.get('scene_id')}")
            print(f"    z_mode: {meta.get('z_mode')}")
            print(f"    chunk_min_voxel: {_as_list(meta.get('chunk_min_voxel'))}")
            print(f"    chunk_max_voxel: {_as_list(meta.get('chunk_max_voxel'))}")
            print(f"    chunk_world_min: {_as_list(meta.get('chunk_world_min'))}")
            print(f"    chunk_world_max: {_as_list(meta.get('chunk_world_max'))}")
            print(f"    top_camera[0]: {_camera_brief(top_cameras[0] if top_cameras else None)}")


if __name__ == "__main__":
    main()
