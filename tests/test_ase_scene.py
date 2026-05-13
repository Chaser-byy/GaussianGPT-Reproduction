import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from gaussiangpt_ae.data.ase import (
    build_ase_scene_manifest,
    discover_ase_scenes,
    find_ase_ply,
    load_ase_scene_gaussians,
)


def write_fake_gaussian_ply(path: Path, n: int = 3) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    data = np.zeros(n, dtype=dtype)
    data["x"] = np.arange(n, dtype=np.float32)
    data["rot_0"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def write_transforms(path: Path) -> None:
    payload = {
        "share_intrinsics": True,
        "fx": 100.0,
        "fy": 101.0,
        "cx": 50.0,
        "cy": 51.0,
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
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_fake_ase_scene(root: Path, scene_id: str = "00000") -> Path:
    scene_dir = root / scene_id
    write_fake_gaussian_ply(scene_dir / "ckpts" / "point_cloud_30000.ply", n=4)
    (scene_dir / "stats").mkdir(parents=True, exist_ok=True)
    (scene_dir / "stats" / "stats.json").write_text(
        json.dumps({"psnr": 30.1, "ssim": 0.95, "num_GS": 4}),
        encoding="utf-8",
    )
    write_transforms(scene_dir / "transforms_train.json")
    return scene_dir


def test_find_ase_ply_prefers_30000(tmp_path: Path) -> None:
    ckpts_dir = tmp_path / "00000" / "ckpts"
    write_fake_gaussian_ply(ckpts_dir / "point_cloud_12000.ply")
    write_fake_gaussian_ply(ckpts_dir / "point_cloud_30000.ply")

    assert find_ase_ply(ckpts_dir).name == "point_cloud_30000.ply"


def test_discover_manifest_and_load_ase_scene(tmp_path: Path) -> None:
    scene_dir = make_fake_ase_scene(tmp_path)
    (tmp_path / ".cache").mkdir()

    records = discover_ase_scenes(tmp_path)

    assert len(records) == 1
    assert records[0].scene_id == "00000"
    assert records[0].scene_dir == scene_dir
    assert records[0].valid is True

    manifest = build_ase_scene_manifest(tmp_path)
    item = manifest[0]
    assert item["scene_id"] == "00000"
    assert item["ckpts_dir"].endswith("00000/ckpts")
    assert item["stats"]["psnr"] == 30.1
    assert item["stats"]["ssim"] == 0.95
    assert item["stats"]["num_GS"] == 4
    assert item["camera_summary"]["width"] == 640
    assert item["num_gaussians"] == 4
    assert item["valid"] is True
    assert item["warnings"] == []

    scene = load_ase_scene_gaussians(records[0])
    assert scene.scene_id == "00000"
    assert scene.xyz.shape == (4, 3)
    assert scene.metadata["dataset"] == "ASE"
