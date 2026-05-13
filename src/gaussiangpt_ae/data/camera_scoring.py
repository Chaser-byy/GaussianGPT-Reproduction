"""Camera visibility scoring for ASE training chunks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from gaussiangpt_ae.data.ase_camera import ASECameras


def _bbox_corners(chunk_world_min: np.ndarray, chunk_world_max: np.ndarray) -> np.ndarray:
    lo = np.asarray(chunk_world_min, dtype=np.float32)
    hi = np.asarray(chunk_world_max, dtype=np.float32)
    corners = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                corners.append([x, y, z])
    return np.asarray(corners, dtype=np.float32)


def _zero_score(frame: Dict) -> Dict:
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
                scores.append(_zero_score(frame))
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
                scores.append(_zero_score(frame))
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
            scores.append(_zero_score(frame))

    return sorted(scores, key=lambda item: item["image_coverage"], reverse=True)[:top_k]
