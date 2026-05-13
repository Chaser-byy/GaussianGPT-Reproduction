"""Online ASE chunk sampler backed by precomputed voxel caches."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from gaussiangpt_ae.data.ase_camera import ASECameras
from gaussiangpt_ae.data.ase_camera import read_ase_cameras
from gaussiangpt_ae.data.ase_voxel_cache import load_ase_voxel_cache


def _bbox_corners(chunk_world_min: np.ndarray, chunk_world_max: np.ndarray) -> np.ndarray:
    lo = np.asarray(chunk_world_min, dtype=np.float32)
    hi = np.asarray(chunk_world_max, dtype=np.float32)
    corners = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                corners.append([x, y, z])
    return np.asarray(corners, dtype=np.float32)


def _zero_camera_score(frame: Dict) -> Dict:
    return {
        "camera_id": frame.get("camera_id"),
        "file_path": frame.get("file_path"),
        "image_coverage": 0.0,
        "visible_ratio": 0.0,
        "valid_projection": False,
    }


def score_cameras_for_chunk(
    cameras: ASECameras,
    chunk_world_min: np.ndarray,
    chunk_world_max: np.ndarray,
    top_k: int = 12,
) -> List[Dict]:
    """Score cameras by projected chunk bbox overlap with the image plane."""

    corners = _bbox_corners(chunk_world_min, chunk_world_max)
    corners_h = np.concatenate([corners, np.ones((8, 1), dtype=np.float32)], axis=1)
    image_area = float(cameras.width * cameras.height)
    scores: List[Dict] = []

    for frame in cameras.frames:
        try:
            camera_to_world = np.asarray(frame["transform_matrix"], dtype=np.float32)
            world_to_camera = np.linalg.inv(camera_to_world)
            camera_points = (world_to_camera @ corners_h.T).T[:, :3]
            front = camera_points[:, 2] > 1e-6
            if not np.any(front):
                scores.append(_zero_camera_score(frame))
                continue

            visible_points = camera_points[front]
            u = cameras.fx * (visible_points[:, 0] / visible_points[:, 2]) + cameras.cx
            v = cameras.fy * (visible_points[:, 1] / visible_points[:, 2]) + cameras.cy

            x_min = float(np.min(u))
            x_max = float(np.max(u))
            y_min = float(np.min(v))
            y_max = float(np.max(v))
            projected_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
            if projected_area <= 0.0:
                scores.append(_zero_camera_score(frame))
                continue

            inter_x_min = max(0.0, x_min)
            inter_x_max = min(float(cameras.width), x_max)
            inter_y_min = max(0.0, y_min)
            inter_y_max = min(float(cameras.height), y_max)
            intersection_area = max(0.0, inter_x_max - inter_x_min) * max(
                0.0, inter_y_max - inter_y_min
            )

            scores.append(
                {
                    "camera_id": frame.get("camera_id"),
                    "file_path": frame.get("file_path"),
                    "image_coverage": float(intersection_area / image_area)
                    if image_area > 0.0
                    else 0.0,
                    "visible_ratio": float(intersection_area / projected_area),
                    "valid_projection": True,
                }
            )
        except Exception:
            scores.append(_zero_camera_score(frame))

    return sorted(scores, key=lambda item: item["image_coverage"], reverse=True)[:top_k]


def build_chunk_from_scene_cache(
    scene_cache: Dict,
    chunk_min_voxel: np.ndarray,
    chunk_shape_voxels: np.ndarray,
) -> Dict:
    """Build chunk-local sparse tensors from a scene-level voxel cache."""

    scene_coords = np.asarray(scene_cache["scene_coords"], dtype=np.int32)
    scene_feats = np.asarray(scene_cache["scene_feats"], dtype=np.float32)
    selected_global_indices = np.asarray(
        scene_cache["selected_global_indices"], dtype=np.int64
    )
    chunk_min_voxel = np.asarray(chunk_min_voxel, dtype=np.int32)
    chunk_shape_voxels = np.asarray(chunk_shape_voxels, dtype=np.int32)
    chunk_max_voxel = chunk_min_voxel + chunk_shape_voxels
    inside = np.all(
        (scene_coords >= chunk_min_voxel) & (scene_coords < chunk_max_voxel),
        axis=1,
    )

    coords = (scene_coords[inside] - chunk_min_voxel).astype(np.int32, copy=False)
    feats = scene_feats[inside].astype(np.float32, copy=False)
    selected = selected_global_indices[inside].astype(np.int64, copy=False)
    return {
        "coords": coords,
        "feats": feats,
        "target_feats": feats.copy(),
        "selected_global_indices": selected,
        "num_occupied_voxels": int(coords.shape[0]),
    }


class ASEOnlineChunkSampler:
    """Randomly sample occupied chunks from scene-level voxel caches."""

    def __init__(
        self,
        cache_root: Union[str, Path],
        chunk_size: float = 4.0,
        occupancy_threshold: float = 0.2,
        max_candidate_chunks: int = 10,
        top_k_cameras: int = 12,
        seed: int = 42,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.scene_cache_paths = sorted((self.cache_root / "scenes").glob("*.npz"))
        if not self.scene_cache_paths:
            raise ValueError(f"no ASE voxel cache files found under {self.cache_root}/scenes")

        first_cache = load_ase_voxel_cache(self.scene_cache_paths[0])
        self.voxel_size = float(first_cache["voxel_size"])
        self.chunk_size = float(chunk_size)
        self.occupancy_threshold = float(occupancy_threshold)
        self.max_candidate_chunks = int(max_candidate_chunks)
        self.top_k_cameras = int(top_k_cameras)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        side = int(round(self.chunk_size / self.voxel_size))
        if side <= 0:
            raise ValueError("chunk_size / voxel_size must round to a positive integer")
        self.chunk_shape_voxels = np.asarray([side, side, side], dtype=np.int32)
        self._scene_cache: Dict[str, Dict] = {str(self.scene_cache_paths[0]): first_cache}
        self._camera_cache: Dict[str, object] = {}

    def _load_scene_cache(self, path: Path) -> Dict:
        key = str(path)
        if key not in self._scene_cache:
            self._scene_cache[key] = load_ase_voxel_cache(path)
        return self._scene_cache[key]

    def _load_cameras(self, transforms_path: str, scene_id: str):
        if transforms_path not in self._camera_cache:
            self._camera_cache[transforms_path] = read_ase_cameras(
                transforms_path, scene_id=scene_id
            )
        return self._camera_cache[transforms_path]

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
        return float(np.count_nonzero(inside) / total_voxels) if total_voxels > 0 else 0.0

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
        """Sample one random chunk from a random scene cache."""

        cache_path = self.scene_cache_paths[int(self.rng.randint(0, len(self.scene_cache_paths)))]
        scene_cache = self._load_scene_cache(cache_path)
        scene_coords = scene_cache["scene_coords"]
        scene_origin = scene_cache["scene_origin"]
        metadata = scene_cache["metadata"]

        chunk_min_voxel, occupancy = self._choose_chunk(scene_coords)
        chunk_max_voxel = chunk_min_voxel + self.chunk_shape_voxels
        chunk = build_chunk_from_scene_cache(
            scene_cache, chunk_min_voxel, self.chunk_shape_voxels
        )

        voxel_size = float(scene_cache["voxel_size"])
        chunk_world_min = scene_origin + chunk_min_voxel.astype(np.float32) * voxel_size
        chunk_world_max = scene_origin + chunk_max_voxel.astype(np.float32) * voxel_size

        transforms_path = metadata["transforms_path"]
        cameras = self._load_cameras(transforms_path, metadata["scene_id"])
        top_cameras = score_cameras_for_chunk(
            cameras, chunk_world_min, chunk_world_max, top_k=self.top_k_cameras
        )

        return {
            "scene_id": metadata["scene_id"],
            "ply_path": metadata["ply_path"],
            "transforms_path": transforms_path,
            "coords": chunk["coords"],
            "feats": chunk["feats"],
            "target_feats": chunk["target_feats"],
            "selected_global_indices": chunk["selected_global_indices"],
            "scene_origin": scene_origin,
            "chunk_min_voxel": chunk_min_voxel.astype(np.int32, copy=False),
            "chunk_max_voxel": chunk_max_voxel.astype(np.int32, copy=False),
            "chunk_world_min": chunk_world_min.astype(np.float32, copy=False),
            "chunk_world_max": chunk_world_max.astype(np.float32, copy=False),
            "voxel_size": voxel_size,
            "chunk_shape_voxels": self.chunk_shape_voxels.copy(),
            "occupancy": float(occupancy),
            "num_occupied_voxels": int(chunk["num_occupied_voxels"]),
            "num_gaussians_after_voxel_dedup": int(scene_coords.shape[0]),
            "top_cameras": top_cameras,
        }
