"""Voxelization, feature encoding, and ASE voxel cache utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from gaussiangpt_ae.data.ase import discover_ase_scenes, load_ase_scene_gaussians
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


def _metadata_json(metadata: Dict) -> np.ndarray:
    return np.asarray(json.dumps(metadata), dtype=np.str_)


def build_ase_voxel_cache(
    root: Union[str, Path],
    output_root: Union[str, Path],
    voxel_size: float = 0.025,
    overwrite: bool = False,
    seed: int = 42,
) -> Dict:
    """Voxelize ASE scenes once and save compact scene-level npz caches."""

    root = Path(root)
    output_root = Path(output_root)
    scenes_dir = output_root / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    for record in discover_ase_scenes(root):
        cache_path = scenes_dir / f"{record.scene_id}.npz"
        item = {
            "scene_id": record.scene_id,
            "cache_path": str(cache_path),
            "written": False,
            "skipped": False,
            "error": None,
        }
        if not record.valid:
            item["error"] = "; ".join(record.warnings)
            results.append(item)
            continue
        if cache_path.exists() and not overwrite:
            item["skipped"] = True
            results.append(item)
            continue

        try:
            scene = load_ase_scene_gaussians(record)
            quantized = quantize_scene_with_minkowski(
                scene, voxel_size=voxel_size, seed=seed
            )
            voxel_features = build_scene_voxel_features(
                scene,
                quantized["scene_origin"],
                quantized["scene_coords"],
                quantized["selected_indices"],
                voxel_size=voxel_size,
            )
            metadata = {
                "scene_id": record.scene_id,
                "scene_dir": str(record.scene_dir),
                "ply_path": str(record.ply_path),
                "transforms_path": str(record.transforms_path),
                "num_input_gaussians": int(scene.xyz.shape[0]),
                "num_voxels": int(voxel_features["coords"].shape[0]),
                "voxel_size": float(voxel_size),
                "feature_representation": "gaussiangpt_ae_v1",
                "color_representation": "rgb_from_sh_dc_clamped_0_1",
                "opacity_representation": "logit_clamped_-10_10",
                "scale_representation": "world_size_from_exp_raw_scale",
                "rotation_representation": "unit_quaternion_w_positive",
                "offset_representation": "world_offset_from_voxel_center",
                "feature_scales": dict(DEFAULT_FEATURE_SCALES),
            }
            np.savez_compressed(
                cache_path,
                scene_coords=voxel_features["coords"],
                scene_feats=voxel_features["feats"],
                selected_global_indices=voxel_features["selected_global_indices"],
                scene_origin=quantized["scene_origin"],
                voxel_size=np.asarray(float(voxel_size), dtype=np.float32),
                metadata_json=_metadata_json(metadata),
            )
            item["written"] = True
        except Exception as exc:
            item["error"] = str(exc)
        results.append(item)

    summary = build_ase_voxel_cache_summary(output_root)
    summary["failed_scenes"] = sum(1 for item in results if item["error"] is not None)
    summary["build_results"] = results
    return summary


def load_ase_voxel_cache(scene_cache_path: Union[str, Path]) -> Dict:
    """Load one ASE scene-level voxel cache npz."""

    scene_cache_path = Path(scene_cache_path)
    with np.load(scene_cache_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return {
            "scene_id": metadata["scene_id"],
            "scene_origin": np.asarray(data["scene_origin"], dtype=np.float32),
            "scene_coords": np.asarray(data["scene_coords"], dtype=np.int32),
            "scene_feats": np.asarray(data["scene_feats"], dtype=np.float32),
            "selected_global_indices": np.asarray(
                data["selected_global_indices"], dtype=np.int64
            ),
            "voxel_size": float(np.asarray(data["voxel_size"]).item()),
            "metadata": metadata,
        }


def build_ase_voxel_cache_summary(output_root: Union[str, Path]) -> Dict:
    """Summarize an ASE voxel cache output directory."""

    output_root = Path(output_root)
    cache_paths = sorted((output_root / "scenes").glob("*.npz"))
    num_voxels: List[int] = []
    voxel_sizes: List[float] = []
    for path in cache_paths:
        try:
            cache = load_ase_voxel_cache(path)
        except Exception:
            continue
        num_voxels.append(int(cache["scene_coords"].shape[0]))
        voxel_sizes.append(float(cache["voxel_size"]))

    return {
        "total_scenes": len(cache_paths),
        "written_scenes": len(num_voxels),
        "failed_scenes": 0,
        "voxel_size": voxel_sizes[0] if voxel_sizes else None,
        "min_num_voxels": int(min(num_voxels)) if num_voxels else None,
        "max_num_voxels": int(max(num_voxels)) if num_voxels else None,
        "mean_num_voxels": float(sum(num_voxels) / len(num_voxels))
        if num_voxels
        else None,
    }
