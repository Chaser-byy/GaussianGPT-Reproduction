"""Readers for ASE camera transforms."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass
class ASECameras:
    """ASE camera metadata parsed from transforms_train.json."""

    scene_id: Optional[str]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    share_intrinsics: bool
    zipped: bool
    crop_edge: int
    frames_num: int
    frames: list[dict]
    transform_device_camera: Optional[np.ndarray] = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _as_matrix4(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape [4, 4], got {array.shape}")
    return array


def _camera_id_from_file_path(file_path: str, fallback: str) -> str:
    if not file_path:
        return fallback
    return Path(file_path).stem


def read_ase_cameras(
    transforms_path: Union[str, Path],
    scene_id: Optional[str] = None,
) -> ASECameras:
    """Read ASE transforms_train.json into typed camera metadata."""

    transforms_path = Path(transforms_path)
    raw = json.loads(transforms_path.read_text(encoding="utf-8"))
    warnings: list[str] = []

    transform_device_camera = None
    if raw.get("transform_device_camera") is not None:
        transform_device_camera = _as_matrix4(
            raw["transform_device_camera"], "transform_device_camera"
        )

    frames = []
    for index, frame in enumerate(raw.get("frames", [])):
        file_path = str(frame.get("file_path", ""))
        camera_id = _camera_id_from_file_path(file_path, f"frame{index:07d}")
        frames.append(
            {
                "frame_id": camera_id,
                "camera_id": camera_id,
                "file_path": file_path,
                "transform_matrix": _as_matrix4(
                    frame.get("transform_matrix"), f"frames[{index}].transform_matrix"
                ),
            }
        )

    frames_num = int(raw.get("frames_num", len(frames)))
    if frames_num != len(frames):
        warnings.append(f"frames_num={frames_num} but found {len(frames)} frames")

    return ASECameras(
        scene_id=scene_id,
        width=int(raw["width"]),
        height=int(raw["height"]),
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
        share_intrinsics=bool(raw.get("share_intrinsics", True)),
        zipped=bool(raw.get("zipped", False)),
        crop_edge=int(raw.get("crop_edge", 0)),
        frames_num=frames_num,
        frames=frames,
        transform_device_camera=transform_device_camera,
        warnings=warnings,
        metadata={"transforms_path": str(transforms_path)},
    )


def camera_summary(cameras: ASECameras) -> dict:
    """Return a compact JSON-serializable camera summary."""

    return {
        "scene_id": cameras.scene_id,
        "width": cameras.width,
        "height": cameras.height,
        "fx": cameras.fx,
        "fy": cameras.fy,
        "cx": cameras.cx,
        "cy": cameras.cy,
        "num_frames": len(cameras.frames),
        "frames_num": cameras.frames_num,
        "share_intrinsics": cameras.share_intrinsics,
        "zipped": cameras.zipped,
        "crop_edge": cameras.crop_edge,
        "has_transform_device_camera": cameras.transform_device_camera is not None,
        "warnings": list(cameras.warnings),
    }
