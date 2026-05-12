import json
from pathlib import Path

import numpy as np

from gaussiangpt_ae.data.ase_camera import camera_summary, read_ase_cameras


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
    assert cameras.frames[0]["transform_matrix"].shape == (4, 4)

    summary = camera_summary(cameras)
    assert summary["num_frames"] == 1
    assert summary["has_transform_device_camera"] is True
