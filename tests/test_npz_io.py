from pathlib import Path

import numpy as np

from gaussiangpt_ae.data.npz_io import read_gaussian_npz


def write_fake_gaussian_npz(path: Path, n: int = 3, scene_id: str = "npz_scene") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        scene_id=np.array(scene_id),
        metadata_json=np.array('{"source": "fake"}'),
        xyz=np.zeros((n, 3), dtype=np.float64),
        color=np.zeros((n, 3), dtype=np.float64),
        opacity=np.zeros((n, 1), dtype=np.float64),
        scale=np.zeros((n, 3), dtype=np.float64),
        rotation=np.zeros((n, 4), dtype=np.float64),
    )


def test_read_gaussian_npz(tmp_path: Path) -> None:
    path = tmp_path / "gaussians.npz"
    write_fake_gaussian_npz(path, n=5, scene_id="inside_scene")

    scene = read_gaussian_npz(path)

    assert scene.scene_id == "inside_scene"
    assert scene.xyz.shape == (5, 3)
    assert scene.rotation.shape == (5, 4)
    assert scene.xyz.dtype == np.float32
    assert scene.metadata["source"] == "fake"


def test_read_gaussian_npz_argument_scene_id_wins(tmp_path: Path) -> None:
    path = tmp_path / "gaussians.npz"
    write_fake_gaussian_npz(path, scene_id="inside_scene")

    scene = read_gaussian_npz(path, scene_id="outside_scene", metadata={"source": "arg"})

    assert scene.scene_id == "outside_scene"
    assert scene.metadata["source"] == "arg"

