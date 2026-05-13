import json
import sys
import types
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from gaussiangpt_ae.data.ase_online_sampler import ASEOnlineChunkSampler


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
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    data = np.zeros(xyz.shape[0], dtype=dtype)
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["f_dc_0"] = 0.1
    data["f_dc_1"] = 0.2
    data["f_dc_2"] = 0.3
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


def test_ase_online_chunk_sampler_sample(monkeypatch, tmp_path: Path) -> None:
    install_fake_minkowski(monkeypatch)
    make_fake_ase_root(tmp_path)

    sampler = ASEOnlineChunkSampler(
        root=tmp_path,
        voxel_size=1.0,
        chunk_size=2.0,
        occupancy_threshold=0.0,
        top_k_cameras=1,
        seed=5,
    )

    sample = sampler.sample()

    assert sample["coords"].ndim == 2
    assert sample["coords"].shape[1] == 3
    assert sample["feats"].shape == (sample["coords"].shape[0], 14)
    assert sample["target_feats"].shape == (sample["coords"].shape[0], 14)
    assert sample["chunk_min_voxel"][2] == 0
    assert np.all(sample["coords"] >= 0)
    assert np.all(sample["coords"] < sample["chunk_shape_voxels"])
    assert sample["top_cameras"]
