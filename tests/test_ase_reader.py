from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from gaussiangpt_ae.data.ase_reader import (
    build_ase_manifest,
    discover_ase_gaussian_items,
    iter_ase_root,
    read_ase_item,
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
    data["rot_0"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def write_bad_ply(path: Path) -> None:
    data = np.zeros(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=True).write(path)


def write_fake_gaussian_npz(path: Path, n: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        xyz=np.zeros((n, 3), dtype=np.float32),
        color=np.zeros((n, 3), dtype=np.float32),
        opacity=np.zeros((n, 1), dtype=np.float32),
        scale=np.zeros((n, 3), dtype=np.float32),
        rotation=np.zeros((n, 4), dtype=np.float32),
    )


def test_discover_and_read_ase_iteration_ply(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_0001"
    write_fake_gaussian_ply(scene_dir / "point_cloud/iteration_30000/point_cloud.ply", n=4)
    (scene_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (scene_dir / "images").mkdir()
    (scene_dir / "depth").mkdir()

    items = discover_ase_gaussian_items(tmp_path)

    assert len(items) == 1
    assert items[0].scene_id == "scene_0001"
    assert items[0].gaussian_format == "ply"

    scene = read_ase_item(items[0])
    assert scene.scene_id == "scene_0001"
    assert scene.xyz.shape == (4, 3)
    assert scene.metadata["dataset"] == "ASE"

    manifest = build_ase_manifest(tmp_path)
    assert manifest[0]["scene_id"] == "scene_0001"
    assert manifest[0]["num_gaussians"] == 4

    scenes = list(iter_ase_root(tmp_path))
    assert len(scenes) == 1
    assert scenes[0].scene_id == "scene_0001"


def test_discover_and_read_ase_npz(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_0002"
    write_fake_gaussian_npz(scene_dir / "gaussians.npz", n=3)
    (scene_dir / "cameras.json").write_text("{}", encoding="utf-8")

    items = discover_ase_gaussian_items(tmp_path)

    assert len(items) == 1
    assert items[0].scene_id == "scene_0002"
    assert items[0].gaussian_format == "npz"
    assert [path.name for path in items[0].metadata_paths] == ["cameras.json"]

    scene = read_ase_item(items[0])
    assert scene.xyz.shape == (3, 3)
    assert scene.metadata["dataset"] == "ASE"


def test_bad_ply_is_skipped(tmp_path: Path) -> None:
    write_bad_ply(tmp_path / "scene_bad" / "bad.ply")

    items = discover_ase_gaussian_items(tmp_path)

    assert items == []

