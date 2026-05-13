import json
from pathlib import Path

import numpy as np

from gaussiangpt_ae.data.sampler import ASEOnlineChunkSampler


def write_transforms(path: Path) -> None:
    payload = {
        "share_intrinsics": True,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
        "zipped": False,
        "crop_edge": 0,
        "transform_device_camera": np.eye(4, dtype=np.float32).tolist(),
        "frames_num": 2,
        "frames": [
            {
                "file_path": "rgb_undistorted/frame0000000.jpg",
                "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
            },
            {
                "file_path": "rgb_undistorted/frame0000001.jpg",
                "transform_matrix": np.asarray(
                    [
                        [1.0, 0.0, 0.0, 100.0],
                        [0.0, 1.0, 0.0, 100.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ).tolist(),
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_fake_scene_cache(cache_root: Path) -> None:
    transforms_path = cache_root / "raw" / "00000" / "transforms_train.json"
    write_transforms(transforms_path)
    scenes_dir = cache_root / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "scene_id": "00000",
        "scene_dir": str(cache_root / "raw" / "00000"),
        "ply_path": str(cache_root / "raw" / "00000" / "ckpts" / "point_cloud_30000.ply"),
        "transforms_path": str(transforms_path),
        "num_input_gaussians": 4,
        "num_voxels": 6,
        "voxel_size": 1.0,
    }
    np.savez_compressed(
        scenes_dir / "00000.npz",
        scene_coords=np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 1],
                [1, 1, 1],
                [0, 0, 2],
                [1, 1, 2],
            ],
            dtype=np.int32,
        ),
        scene_feats=np.ones((6, 14), dtype=np.float32),
        selected_global_indices=np.arange(6, dtype=np.int64),
        scene_origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.asarray(1.0, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
    )


def test_ase_online_chunk_sampler_sample(tmp_path: Path) -> None:
    make_fake_scene_cache(tmp_path)

    sampler = ASEOnlineChunkSampler(
        cache_root=tmp_path,
        chunk_size=2.0,
        occupancy_threshold=0.0,
        top_k_cameras=1,
        seed=5,
        z_mode="fixed_160",
    )

    sample = sampler.sample()

    assert sample["coords"].ndim == 2
    assert sample["coords"].shape[1] == 3
    assert sample["feats"].shape == (sample["coords"].shape[0], 14)
    assert sample["target_feats"].shape == (sample["coords"].shape[0], 14)
    assert sample["chunk_min_voxel"][2] == 0
    assert sample["chunk_shape_voxels"][2] == 2
    assert np.all(sample["coords"] >= 0)
    assert np.all(sample["coords"] < sample["chunk_shape_voxels"])
    assert sample["top_cameras"]
    assert sample["z_mode"] == "fixed_160"
    assert sample["scene_z_min_voxel"] == 0
    assert sample["scene_z_max_voxel"] == 3
    assert sample["scene_z_voxels"] == 3
    assert "accepted_by_threshold" in sample
    assert "candidate_occupancies" in sample
    assert "best_candidate_occupancy" in sample
    assert "image_coverage" in sample["top_cameras"][0]
    assert "visible_ratio" in sample["top_cameras"][0]
    assert "valid_projection" in sample["top_cameras"][0]
    assert "selection_mode" in sample["top_cameras"][0]


def test_ase_online_chunk_sampler_full_height_z_mode(tmp_path: Path) -> None:
    make_fake_scene_cache(tmp_path)

    sampler = ASEOnlineChunkSampler(
        cache_root=tmp_path,
        chunk_size=2.0,
        occupancy_threshold=0.0,
        top_k_cameras=1,
        seed=5,
        z_mode="full_height",
    )

    sample = sampler.sample()

    assert sample["z_mode"] == "full_height"
    assert sample["chunk_min_voxel"][2] == 0
    assert sample["chunk_max_voxel"][2] == 3
    assert sample["chunk_shape_voxels"][2] == 3
    assert np.all(sample["coords"] >= 0)
    assert np.all(sample["coords"] < sample["chunk_shape_voxels"])
