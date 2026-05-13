"""Build and load scene-level ASE voxel feature caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from gaussiangpt_ae.data.ase_scene import discover_ase_scenes, load_ase_scene_gaussians
from gaussiangpt_ae.data.me_voxelize import (
    DEFAULT_FEATURE_SCALES,
    build_scene_voxel_features,
    quantize_scene_with_minkowski,
)


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
