from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from gaussiangpt_ae.data.ply_io import read_gaussian_ply


def write_fake_gaussian_ply(path: Path, n: int = 3) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
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


def test_read_gaussian_ply(tmp_path: Path) -> None:
    path = tmp_path / "point_cloud.ply"
    write_fake_gaussian_ply(path, n=4)

    scene = read_gaussian_ply(path)

    assert scene.scene_id == "point_cloud"
    assert scene.xyz.shape == (4, 3)
    assert scene.color.shape == (4, 3)
    assert scene.opacity.shape == (4, 1)
    assert scene.scale.shape == (4, 3)
    assert scene.rotation.shape == (4, 4)
    assert scene.xyz.dtype == np.float32

