"""Online ASE chunk sampler for voxelized autoencoder training data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from gaussiangpt_ae.data.ase_camera import read_ase_cameras
from gaussiangpt_ae.data.ase_scene import (
    ASESceneRecord,
    discover_ase_scenes,
    load_ase_scene_gaussians,
)
from gaussiangpt_ae.data.camera_scoring import score_cameras_for_chunk
from gaussiangpt_ae.data.me_voxelize import (
    build_chunk_features_from_quantized_scene,
    quantize_scene_with_minkowski,
)


class ASEOnlineChunkSampler:
    """Randomly sample occupied ASE chunks and nearby camera scores online."""

    def __init__(
        self,
        root: Union[str, Path],
        voxel_size: float = 0.025,
        chunk_size: float = 4.0,
        occupancy_threshold: float = 0.2,
        max_candidate_chunks: int = 10,
        top_k_cameras: int = 12,
        seed: int = 42,
        cache_scenes: bool = True,
    ) -> None:
        self.root = Path(root)
        self.voxel_size = float(voxel_size)
        self.chunk_size = float(chunk_size)
        self.occupancy_threshold = float(occupancy_threshold)
        self.max_candidate_chunks = int(max_candidate_chunks)
        self.top_k_cameras = int(top_k_cameras)
        self.seed = int(seed)
        self.cache_scenes = bool(cache_scenes)
        self.rng = np.random.RandomState(self.seed)

        self.records: List[ASESceneRecord] = [
            record
            for record in discover_ase_scenes(self.root)
            if record.valid and record.ply_path is not None and record.transforms_path.is_file()
        ]
        if not self.records:
            raise ValueError(f"no valid ASE scenes found under {self.root}")

        side = int(round(self.chunk_size / self.voxel_size))
        if side <= 0:
            raise ValueError("chunk_size / voxel_size must round to a positive integer")
        self.chunk_shape_voxels = np.asarray([side, side, side], dtype=np.int32)
        self._cache: Dict[str, Dict] = {}

    def _load_record(self, record: ASESceneRecord) -> Dict:
        cached = self._cache.get(record.scene_id)
        if cached is not None and self.cache_scenes:
            return cached

        scene = load_ase_scene_gaussians(record)
        cameras = read_ase_cameras(record.transforms_path, scene_id=record.scene_id)
        quantized = quantize_scene_with_minkowski(
            scene,
            voxel_size=self.voxel_size,
            seed=int(self.rng.randint(0, np.iinfo(np.int32).max)),
        )
        loaded = {"scene": scene, "cameras": cameras, "quantized": quantized}
        if self.cache_scenes:
            self._cache[record.scene_id] = loaded
        return loaded

    def _sample_chunk_min(self, scene_coords: np.ndarray) -> np.ndarray:
        coord_min = scene_coords.min(axis=0).astype(np.int32)
        coord_max = scene_coords.max(axis=0).astype(np.int32)
        chunk_min = np.zeros(3, dtype=np.int32)
        for axis in (0, 1):
            low = int(coord_min[axis])
            high = int(coord_max[axis] - self.chunk_shape_voxels[axis] + 1)
            if high < low:
                high = low
            chunk_min[axis] = int(self.rng.randint(low, high + 1))
        chunk_min[2] = int(coord_min[2])
        return chunk_min

    def _chunk_occupancy(self, scene_coords: np.ndarray, chunk_min_voxel: np.ndarray) -> float:
        chunk_max_voxel = chunk_min_voxel + self.chunk_shape_voxels
        inside = np.all(
            (scene_coords >= chunk_min_voxel) & (scene_coords < chunk_max_voxel),
            axis=1,
        )
        total_voxels = int(np.prod(self.chunk_shape_voxels))
        if total_voxels <= 0:
            return 0.0
        return float(np.count_nonzero(inside) / total_voxels)

    def _choose_chunk(self, scene_coords: np.ndarray) -> tuple:
        best_chunk_min: Optional[np.ndarray] = None
        best_occupancy = -1.0
        for _ in range(self.max_candidate_chunks):
            candidate = self._sample_chunk_min(scene_coords)
            occupancy = self._chunk_occupancy(scene_coords, candidate)
            if occupancy > best_occupancy:
                best_chunk_min = candidate
                best_occupancy = occupancy
            if occupancy >= self.occupancy_threshold:
                return candidate, occupancy
        if best_chunk_min is None:
            best_chunk_min = scene_coords.min(axis=0).astype(np.int32)
            best_occupancy = self._chunk_occupancy(scene_coords, best_chunk_min)
        return best_chunk_min, best_occupancy

    def sample(self) -> Dict:
        """Sample one random ASE chunk."""

        record = self.records[int(self.rng.randint(0, len(self.records)))]
        loaded = self._load_record(record)
        scene = loaded["scene"]
        cameras = loaded["cameras"]
        quantized = loaded["quantized"]

        scene_origin = quantized["scene_origin"]
        scene_coords = quantized["scene_coords"]
        selected_indices = quantized["selected_indices"]
        chunk_min_voxel, occupancy = self._choose_chunk(scene_coords)
        chunk_max_voxel = chunk_min_voxel + self.chunk_shape_voxels

        chunk = build_chunk_features_from_quantized_scene(
            scene,
            scene_origin,
            scene_coords,
            selected_indices,
            chunk_min_voxel,
            self.chunk_shape_voxels,
            voxel_size=self.voxel_size,
            seed=self.seed,
        )

        chunk_world_min = scene_origin + chunk_min_voxel.astype(np.float32) * self.voxel_size
        chunk_world_max = scene_origin + chunk_max_voxel.astype(np.float32) * self.voxel_size
        top_cameras = score_cameras_for_chunk(
            cameras,
            chunk_world_min,
            chunk_world_max,
            top_k=self.top_k_cameras,
        )

        return {
            "scene_id": record.scene_id,
            "ply_path": str(record.ply_path),
            "transforms_path": str(record.transforms_path),
            "coords": chunk["coords"],
            "feats": chunk["feats"],
            "target_feats": chunk["target_feats"],
            "selected_global_indices": chunk["selected_global_indices"],
            "scene_origin": scene_origin,
            "chunk_min_voxel": chunk_min_voxel.astype(np.int32, copy=False),
            "chunk_max_voxel": chunk_max_voxel.astype(np.int32, copy=False),
            "chunk_world_min": chunk_world_min.astype(np.float32, copy=False),
            "chunk_world_max": chunk_world_max.astype(np.float32, copy=False),
            "voxel_size": self.voxel_size,
            "chunk_shape_voxels": self.chunk_shape_voxels.copy(),
            "occupancy": float(occupancy),
            "num_occupied_voxels": int(chunk["num_occupied_voxels"]),
            "num_gaussians_after_voxel_dedup": int(scene_coords.shape[0]),
            "top_cameras": top_cameras,
        }
