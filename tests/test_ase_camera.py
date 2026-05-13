import json
from pathlib import Path

import numpy as np

from gaussiangpt_ae.data.ase import (
    camera_summary,
    load_ase_camera_cache,
    read_ase_cameras,
    save_ase_camera_cache,
)


def write_transforms(path: Path) -> None:
    payload = {
        "share_intrinsics": True,
        "fx": "100.5",
        "fy": 101,
        "cx": 50,
        "cy": "51.5",
        "width": "640",
        "height": 480,
        "zipped": False,
        "crop_edge": 2,
        "transform_device_camera": np.eye(4, dtype=np.float32).tolist(),
        "bbox_min": [-1.0, -2.0, -3.0],
        "bbox_max": [1.0, 2.0, 3.0],
        "frames_num": 1,
        "frames": [
            {
                "file_path": "rgb_undistorted/frame0000000.jpg",
                "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_read_ase_cameras_reads_intrinsics_and_frame_ids(tmp_path: Path) -> None:
    path = tmp_path / "transforms_train.json"
    write_transforms(path)

    cameras = read_ase_cameras(path, scene_id="00000")

    assert cameras.scene_id == "00000"
    assert cameras.fx == 100.5
    assert cameras.width == 640
    assert cameras.transform_device_camera.dtype == np.float32
    assert cameras.frames[0]["camera_id"] == "frame0000000"
    assert cameras.frames[0]["frame_id"] == 0
    assert cameras.frames[0]["transform_matrix"].shape == (4, 4)
    assert cameras.K.shape == (3, 3)
    assert cameras.c2w.shape == (1, 4, 4)
    assert cameras.w2c.shape == (1, 4, 4)
    assert np.allclose(cameras.w2c[0] @ cameras.c2w[0], np.eye(4))
    assert cameras.uses_transform_device_camera is False

    summary = camera_summary(cameras)
    assert summary["num_frames"] == 1
    assert summary["has_transform_device_camera"] is True
    assert summary["pose_convention"] == "c2w=frame_transform_matrix"


def test_save_and_load_ase_camera_cache(tmp_path: Path) -> None:
    path = tmp_path / "transforms_train.json"
    cache_path = tmp_path / "00000_cameras.npz"
    write_transforms(path)

    cameras = read_ase_cameras(path, scene_id="00000")
    save_ase_camera_cache(cameras, cache_path)
    loaded = load_ase_camera_cache(cache_path, scene_id="00000")

    assert loaded.scene_id == "00000"
    assert loaded.K.dtype == np.float32
    assert loaded.c2w.shape == (1, 4, 4)
    assert loaded.frame_ids.dtype == np.int64
    assert str(loaded.file_paths[0]) == "rgb_undistorted/frame0000000.jpg"
    assert loaded.pose_convention == "c2w=frame_transform_matrix"
    assert loaded.uses_transform_device_camera is False
