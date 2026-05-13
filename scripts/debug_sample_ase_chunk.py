"""Debug online ASE chunk sampling."""

from __future__ import annotations

import argparse

from gaussiangpt_ae.data.ase_online_sampler import ASEOnlineChunkSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--chunk-size", type=float, default=4.0)
    parser.add_argument("--occupancy-threshold", type=float, default=0.2)
    parser.add_argument("--top-k-cameras", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sampler = ASEOnlineChunkSampler(
        cache_root=args.cache_root,
        chunk_size=args.chunk_size,
        occupancy_threshold=args.occupancy_threshold,
        top_k_cameras=args.top_k_cameras,
        seed=args.seed,
    )

    for sample_index in range(args.num_samples):
        sample = sampler.sample()
        top_camera_ids = [camera["camera_id"] for camera in sample["top_cameras"]]
        print(f"sample: {sample_index}")
        print(f"scene_id: {sample['scene_id']}")
        print(f"coords shape: {sample['coords'].shape}")
        print(f"feats shape: {sample['feats'].shape}")
        print(f"occupancy: {sample['occupancy']}")
        print(f"num_occupied_voxels: {sample['num_occupied_voxels']}")
        print(f"chunk_min_voxel: {sample['chunk_min_voxel'].tolist()}")
        print(f"chunk_max_voxel: {sample['chunk_max_voxel'].tolist()}")
        print(f"chunk_world_min: {sample['chunk_world_min'].tolist()}")
        print(f"chunk_world_max: {sample['chunk_world_max'].tolist()}")
        print(f"top camera ids: {top_camera_ids}")


if __name__ == "__main__":
    main()
