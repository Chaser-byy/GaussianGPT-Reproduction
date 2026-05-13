import json
import sys
import types
from pathlib import Path

import numpy as np

from gaussiangpt_ae.data.ase_dataset import ASEChunkDataset
from gaussiangpt_ae.data.collate import ase_sparse_collate


def install_fake_torch(monkeypatch):
    class FakeTorch:
        long = np.int64
        float32 = np.float32

        @staticmethod
        def as_tensor(array, dtype=None):
            return np.asarray(array, dtype=dtype)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)


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
        "frames_num": 1,
        "frames": [
            {
                "file_path": "rgb_undistorted/frame0000000.jpg",
                "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
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
        "num_voxels": 4,
        "voxel_size": 1.0,
    }
    np.savez_compressed(
        scenes_dir / "00000.npz",
        scene_coords=np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.int32
        ),
        scene_feats=np.ones((4, 14), dtype=np.float32),
        selected_global_indices=np.arange(4, dtype=np.int64),
        scene_origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.asarray(1.0, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata), dtype=np.str_),
    )


def test_ase_chunk_dataset_and_sparse_collate(monkeypatch, tmp_path: Path) -> None:
    install_fake_torch(monkeypatch)
    make_fake_scene_cache(tmp_path)

    dataset = ASEChunkDataset(
        cache_root=tmp_path,
        num_samples_per_epoch=2,
        chunk_size=2.0,
        occupancy_threshold=0.0,
        top_k_cameras=1,
        seed=5,
    )

    sample = dataset[0]
    assert len(dataset) == 2
    assert sample["coords"].shape[1] == 3
    assert sample["feats"].shape[1] == 14
    assert "metadata" in sample

    batch = ase_sparse_collate([sample, dataset[1]])
    assert batch["coords"].shape[1] == 4
    assert batch["feats"].shape[1] == 14
    assert batch["target_feats"].shape[1] == 14
    assert len(batch["metas"]) == 2
