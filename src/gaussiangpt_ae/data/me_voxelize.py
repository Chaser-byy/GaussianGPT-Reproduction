"""MinkowskiEngine-based voxelization helpers for ASE chunks."""

from __future__ import annotations

from typing import Dict

import numpy as np

from gaussiangpt_ae.data.schema import GaussianScene, validate_gaussian_scene

SH_C0 = 0.28209479177387814
DEFAULT_FEATURE_SCALES = {
    "offset": 1.0,
    "color": 1.0,
    "opacity": 1.0,
    "scale": 1.0,
    "rotation": 1.0,
}


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


def _feature_scales(feature_scales: Dict = None) -> Dict:
    scales = dict(DEFAULT_FEATURE_SCALES)
    if feature_scales is not None:
        scales.update(feature_scales)
    return scales


def encode_gaussian_features_for_ae(
    relative_xyz: np.ndarray,
    color_raw: np.ndarray,
    opacity_raw: np.ndarray,
    scale_raw: np.ndarray,
    rotation_raw: np.ndarray,
    voxel_size: float,
    feature_scales: Dict = None,
) -> np.ndarray:
    """Encode Gaussian attributes using the GaussianGPT AE feature representation."""

    del voxel_size
    scales = _feature_scales(feature_scales)

    relative_xyz = np.asarray(relative_xyz, dtype=np.float32) * float(scales["offset"])
    rgb = np.asarray(color_raw, dtype=np.float32) * SH_C0 + 0.5
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32) * float(scales["color"])

    opacity = np.clip(np.asarray(opacity_raw, dtype=np.float32), -10.0, 10.0)
    opacity = opacity.astype(np.float32) * float(scales["opacity"])

    scale = np.exp(np.asarray(scale_raw, dtype=np.float32))
    scale = np.maximum(scale, 1e-8).astype(np.float32) * float(scales["scale"])

    quat = np.asarray(rotation_raw, dtype=np.float32)
    quat = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-8)
    negative_w = quat[:, 0] < 0
    quat[negative_w] *= -1.0
    quat = quat.astype(np.float32) * float(scales["rotation"])

    return np.concatenate([relative_xyz, rgb, opacity, scale, quat], axis=1).astype(
        np.float32, copy=False
    )


def build_scene_voxel_features(
    scene: GaussianScene,
    scene_origin: np.ndarray,
    scene_coords: np.ndarray,
    selected_indices: np.ndarray,
    voxel_size: float = 0.025,
    feature_scales: Dict = None,
) -> Dict:
    """Build whole-scene 14D voxel features after sparse quantization."""

    validate_gaussian_scene(scene)
    scene_origin = np.asarray(scene_origin, dtype=np.float32)
    scene_coords = np.asarray(scene_coords, dtype=np.int32)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)

    voxel_center_world = scene_origin + (scene_coords.astype(np.float32) + 0.5) * float(
        voxel_size
    )
    xyz_selected = scene.xyz[selected_indices]
    relative_xyz = (xyz_selected - voxel_center_world).astype(np.float32)
    feats = encode_gaussian_features_for_ae(
        relative_xyz=relative_xyz,
        color_raw=scene.color[selected_indices],
        opacity_raw=scene.opacity[selected_indices],
        scale_raw=scene.scale[selected_indices],
        rotation_raw=scene.rotation[selected_indices],
        voxel_size=voxel_size,
        feature_scales=feature_scales,
    )

    return {
        "coords": scene_coords.astype(np.int32, copy=False),
        "feats": feats,
        "selected_global_indices": selected_indices.astype(np.int64, copy=False),
    }
