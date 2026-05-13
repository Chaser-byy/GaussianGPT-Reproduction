import sys
import types

import numpy as np

from gaussiangpt_ae.data.me_voxelize import (
    build_chunk_features_from_quantized_scene,
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
        color=np.ones((n, 3), dtype=np.float32),
        opacity=np.ones((n, 1), dtype=np.float32) * 0.5,
        scale=np.ones((n, 3), dtype=np.float32) * 0.1,
        rotation=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
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


def test_build_chunk_features_from_quantized_scene_outputs_local_coords(monkeypatch) -> None:
    install_fake_minkowski(monkeypatch)
    scene = make_scene()
    quantized = quantize_scene_with_minkowski(scene, voxel_size=1.0, seed=7)

    chunk = build_chunk_features_from_quantized_scene(
        scene,
        quantized["scene_origin"],
        quantized["scene_coords"],
        quantized["selected_indices"],
        chunk_min_voxel=np.asarray([1, 0, 0], dtype=np.int32),
        chunk_shape_voxels=np.asarray([2, 2, 2], dtype=np.int32),
        voxel_size=1.0,
    )

    assert chunk["coords"].ndim == 2
    assert chunk["coords"].shape[1] == 3
    assert chunk["feats"].shape == (chunk["coords"].shape[0], 14)
    assert chunk["target_feats"].shape == (chunk["coords"].shape[0], 14)
    assert chunk["selected_global_indices"].dtype == np.int64
    assert chunk["coords"].dtype == np.int32
    assert chunk["feats"].dtype == np.float32
    assert np.all(chunk["coords"] >= 0)
    assert np.all(chunk["coords"] < np.asarray([2, 2, 2], dtype=np.int32))
