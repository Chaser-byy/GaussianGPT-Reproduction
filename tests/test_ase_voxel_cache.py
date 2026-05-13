import json
import sys
import types
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from gaussiangpt_ae.data.voxelize import (
    build_ase_voxel_cache,
    load_ase_voxel_cache,
)


def install_fake_minkowski(monkeypatch):
    def sparse_quantize(coordinates, return_index=False):
        _, index = np.unique(coordinates, axis=0, return_index=True)
        index = np.sort(index).astype(np.int64)
        if return_index:
            return coordinates[index], index
        return coordinates[index]

    fake_me = types.SimpleNamespace(utils=types.SimpleNamespace(sparse_quantize=sparse_quantize))
    monkeypatch.setitem(sys.modules, "MinkowskiEngine", fake_me)


def write_fake_gaussian_ply(path: Path) -> None:
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
    data = np.zeros(4, dtype=dtype)
    data["x"] = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    data["y"] = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    data["f_dc_0"] = 0.1
    data["opacity"] = 0.5
    data["scale_0"] = 0.01
    data["scale_1"] = 0.02
    data["scale_2"] = 0.03
    data["rot_0"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


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
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_fake_ase_root(root: Path) -> None:
    scene_dir = root / "00000"
    write_fake_gaussian_ply(scene_dir / "ckpts" / "point_cloud_30000.ply")
    write_transforms(scene_dir / "transforms_train.json")
    (scene_dir / "stats").mkdir(parents=True, exist_ok=True)
    (scene_dir / "stats" / "stats.json").write_text('{"num_GS": 4}', encoding="utf-8")


def test_build_and_load_ase_voxel_cache(monkeypatch, tmp_path: Path) -> None:
    install_fake_minkowski(monkeypatch)
    root = tmp_path / "ase"
    cache_root = tmp_path / "cache"
    make_fake_ase_root(root)

    summary = build_ase_voxel_cache(root, cache_root, voxel_size=1.0, seed=3)
    cache_path = cache_root / "scenes" / "00000.npz"

    assert cache_path.exists()
    assert summary["written_scenes"] == 1

    cache = load_ase_voxel_cache(cache_path)
    assert cache["scene_id"] == "00000"
    assert cache["scene_coords"].shape[1] == 3
    assert cache["scene_feats"].shape[1] == 14
    assert cache["metadata"]["transforms_path"].endswith("transforms_train.json")
    assert cache["metadata"]["feature_representation"] == "gaussiangpt_ae_v1"
    assert cache["metadata"]["color_representation"] == "rgb_from_sh_dc_clamped_0_1"
    assert cache["metadata"]["opacity_representation"] == "logit_clamped_-10_10"
    assert (
        cache["metadata"]["scale_representation"]
        == "world_size_from_exp_raw_scale"
    )
    assert (
        cache["metadata"]["rotation_representation"]
        == "unit_quaternion_w_positive"
    )
    assert (
        cache["metadata"]["offset_representation"]
        == "world_offset_from_voxel_center"
    )
    assert cache["metadata"]["feature_scales"]["offset"] == 1.0
