"""MinkowskiEngine-based voxelization helpers for ASE chunks."""

from __future__ import annotations

from typing import Dict

import numpy as np

from gaussiangpt_ae.data.schema import GaussianScene, validate_gaussian_scene


def _import_minkowski_engine():
    try:
        import MinkowskiEngine as ME
    except ImportError as exc:
        raise ImportError(
            "MinkowskiEngine is required for voxelization in the AutoDL environment."
        ) from exc
    return ME


def _extract_sparse_quantize_result(result, shuffled_coords: np.ndarray) -> tuple:
    if isinstance(result, tuple):
        index = None
        coords = None
        for item in result:
            array = np.asarray(item)
            if array.ndim == 1 and np.issubdtype(array.dtype, np.integer):
                index = array.astype(np.int64, copy=False)
            elif array.ndim == 2 and array.shape[1] == 3:
                coords = array.astype(np.int32, copy=False)
        if index is None:
            raise ValueError("MinkowskiEngine sparse_quantize did not return unique indices")
        if coords is None:
            coords = shuffled_coords[index]
        return coords, index

    index = np.asarray(result, dtype=np.int64)
    if index.ndim != 1:
        raise ValueError("MinkowskiEngine sparse_quantize return value is not a 1D index array")
    return shuffled_coords[index], index


def quantize_scene_with_minkowski(
    scene: GaussianScene,
    voxel_size: float = 0.025,
    seed: int = 42,
) -> Dict:
    """Quantize a Gaussian scene with MinkowskiEngine sparse_quantize."""

    validate_gaussian_scene(scene)
    ME = _import_minkowski_engine()

    scene_origin = scene.xyz.min(axis=0).astype(np.float32)
    scene_coords_all = np.floor((scene.xyz - scene_origin) / voxel_size).astype(np.int32)

    rng = np.random.RandomState(seed)
    shuffled_original_indices = rng.permutation(scene.xyz.shape[0]).astype(np.int64)
    shuffled_coords = scene_coords_all[shuffled_original_indices]

    result = ME.utils.sparse_quantize(shuffled_coords, return_index=True)
    quantized_coords, selected_shuffled_indices = _extract_sparse_quantize_result(
        result, shuffled_coords
    )
    selected_indices = shuffled_original_indices[selected_shuffled_indices].astype(np.int64)

    return {
        "scene_origin": scene_origin,
        "scene_coords": quantized_coords.astype(np.int32, copy=False),
        "selected_indices": selected_indices,
    }


def build_chunk_features_from_quantized_scene(
    scene: GaussianScene,
    scene_origin: np.ndarray,
    scene_coords: np.ndarray,
    selected_indices: np.ndarray,
    chunk_min_voxel: np.ndarray,
    chunk_shape_voxels: np.ndarray,
    voxel_size: float = 0.025,
    seed: int = 42,
) -> Dict:
    """Build chunk-local coordinates and 14D Gaussian features from quantized scene data."""

    del seed
    validate_gaussian_scene(scene)

    scene_origin = np.asarray(scene_origin, dtype=np.float32)
    scene_coords = np.asarray(scene_coords, dtype=np.int32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    chunk_min_voxel = np.asarray(chunk_min_voxel, dtype=np.int32)
    chunk_shape_voxels = np.asarray(chunk_shape_voxels, dtype=np.int32)
    chunk_max_voxel = chunk_min_voxel + chunk_shape_voxels

    inside_mask = np.all(
        (scene_coords >= chunk_min_voxel) & (scene_coords < chunk_max_voxel),
        axis=1,
    )
    scene_coords_inside = scene_coords[inside_mask]
    selected_global_indices = selected_indices[inside_mask].astype(np.int64, copy=False)
    local_coords = (scene_coords_inside - chunk_min_voxel).astype(np.int32, copy=False)

    if selected_global_indices.size == 0:
        feats = np.zeros((0, 14), dtype=np.float32)
        return {
            "coords": local_coords.reshape(0, 3),
            "feats": feats,
            "target_feats": feats.copy(),
            "selected_global_indices": selected_global_indices,
            "num_occupied_voxels": 0,
        }

    xyz_selected = scene.xyz[selected_global_indices]
    voxel_center_world = scene_origin + (scene_coords_inside.astype(np.float32) + 0.5) * float(
        voxel_size
    )
    relative_xyz = (xyz_selected - voxel_center_world).astype(np.float32)
    feats = np.concatenate(
        [
            relative_xyz,
            scene.color[selected_global_indices],
            scene.opacity[selected_global_indices],
            scene.scale[selected_global_indices],
            scene.rotation[selected_global_indices],
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    return {
        "coords": local_coords,
        "feats": feats,
        "target_feats": feats.copy(),
        "selected_global_indices": selected_global_indices,
        "num_occupied_voxels": int(local_coords.shape[0]),
    }
