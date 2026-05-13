import sys
import types

import numpy as np

from gaussiangpt_ae.data.me_voxelize import (
    build_scene_voxel_features,
    quantize_scene_with_minkowski,
)
from gaussiangpt_ae.data.schema import GaussianScene


def install_fake_minkowski(monkeypatch):
    def sparse_quantize(coordinates, return_index=False):
        _, index = np.unique(coordinates, axis=0, return_index=True)
        index = np.sort(index).astype(np.int64)
        if return_index:
            return coordinates[index], index
        return coordinates[index]

    fake_me = types.SimpleNamespace(utils=types.SimpleNamespace(sparse_quantize=sparse_quantize))
    monkeypatch.setitem(sys.modules, "MinkowskiEngine", fake_me)


def make_scene() -> GaussianScene:
    xyz = np.asarray(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [1.10, 0.10, 0.10],
            [1.20, 1.20, 0.10],
        ],
        dtype=np.float32,
    )
    n = xyz.shape[0]
    return GaussianScene(
        scene_id="fake",
        xyz=xyz,
        color=np.asarray(
            [
                [-4.0, 0.0, 4.0],
                [2.0, -2.0, 0.5],
                [0.25, 10.0, -10.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        opacity=np.asarray([[-20.0], [20.0], [0.5], [-0.5]], dtype=np.float32),
        scale=np.asarray(
            [
                [-4.0, -2.0, -1.0],
                [0.0, 0.5, 1.0],
                [-8.0, -7.0, -6.0],
                [2.0, -3.0, 0.25],
            ],
            dtype=np.float32,
        ),
        rotation=np.asarray(
            [
                [-2.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        metadata={},
    )


def test_quantize_scene_with_minkowski_returns_coords_and_indices(monkeypatch) -> None:
    install_fake_minkowski(monkeypatch)
    scene = make_scene()

    quantized = quantize_scene_with_minkowski(scene, voxel_size=1.0, seed=7)

    assert quantized["scene_origin"].shape == (3,)
    assert quantized["scene_coords"].dtype == np.int32
    assert quantized["scene_coords"].shape[1] == 3
    assert quantized["selected_indices"].dtype == np.int64
    assert quantized["selected_indices"].shape[0] == quantized["scene_coords"].shape[0]


def test_build_scene_voxel_features_outputs_scene_features(monkeypatch) -> None:
    install_fake_minkowski(monkeypatch)
    scene = make_scene()
    quantized = quantize_scene_with_minkowski(scene, voxel_size=1.0, seed=7)

    features = build_scene_voxel_features(
        scene,
        quantized["scene_origin"],
        quantized["scene_coords"],
        quantized["selected_indices"],
        voxel_size=1.0,
    )

    assert features["coords"].ndim == 2
    assert features["coords"].shape[1] == 3
    assert features["feats"].shape == (features["coords"].shape[0], 14)
    assert features["selected_global_indices"].dtype == np.int64
    assert features["coords"].dtype == np.int32
    assert features["feats"].dtype == np.float32
    feats = features["feats"]
    assert np.all(feats[:, 3:6] >= 0.0)
    assert np.all(feats[:, 3:6] <= 1.0)
    assert np.all(feats[:, 6:7] >= -10.0)
    assert np.all(feats[:, 6:7] <= 10.0)
    assert np.all(feats[:, 7:10] > 0.0)
    assert np.allclose(np.linalg.norm(feats[:, 10:14], axis=1), 1.0, atol=1e-5)
    assert np.all(feats[:, 10] >= 0.0)
