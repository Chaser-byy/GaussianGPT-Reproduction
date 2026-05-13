"""ASE scene, stats, and camera readers."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from gaussiangpt_ae.data.ply_io import get_gaussian_ply_num_vertices, read_gaussian_ply
from gaussiangpt_ae.data.schema import GaussianScene


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


@dataclass
class ASESceneRecord:
    """Filesystem record for one fixed-layout ASE scene."""

    scene_id: str
    scene_dir: Path
    ckpts_dir: Path
    stats_dir: Path
    ply_path: Optional[Path]
    transforms_path: Path
    valid: bool
    warnings: list[str] = field(default_factory=list)


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


def _parse_scalar(value: str):
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_text_stats(text: str) -> dict:
    parsed: dict = {}
    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line or line.startswith("#"):
            continue
        separator = ":" if ":" in line else "=" if "=" in line else None
        if separator is None:
            continue
        key, value = line.split(separator, 1)
        key = key.strip().strip('"').strip("'")
        if key:
            parsed[key] = _parse_scalar(value)
    if not parsed:
        raise ValueError("no key/value stats found")
    return parsed


def _parse_stats_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
        raise ValueError("JSON root is not an object")
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    return _parse_text_stats(text)


def read_ase_stats(stats_dir: Union[str, Path]) -> dict:
    """Read and merge all parseable ASE stats files from a stats directory."""

    stats_dir = Path(stats_dir)
    merged: dict = {"_source_files": [], "_parse_errors": {}}
    if not stats_dir.exists() or not stats_dir.is_dir():
        merged["_parse_errors"][str(stats_dir)] = "stats directory does not exist"
        return merged

    for path in sorted(child for child in stats_dir.iterdir() if child.is_file()):
        try:
            parsed = _parse_stats_file(path)
        except Exception as exc:
            merged["_parse_errors"][path.name] = str(exc)
            continue
        merged.update(parsed)
        merged["_source_files"].append(path.name)

    return merged


def _iteration_from_point_cloud_name(path: Path) -> int:
    match = re.fullmatch(r"point_cloud_(\d+)\.ply", path.name)
    return int(match.group(1)) if match else -1


def find_ase_ply(ckpts_dir: Union[str, Path]) -> Optional[Path]:
    """Select the Gaussian PLY in an ASE ckpts directory."""

    ckpts_dir = Path(ckpts_dir)
    preferred = ckpts_dir / "point_cloud_30000.ply"
    if preferred.is_file():
        return preferred

    point_cloud_candidates = sorted(ckpts_dir.glob("point_cloud_*.ply"))
    if point_cloud_candidates:
        return max(
            point_cloud_candidates,
            key=lambda path: (_iteration_from_point_cloud_name(path), path.name),
        )

    ply_candidates = sorted(ckpts_dir.glob("*.ply"))
    return ply_candidates[0] if ply_candidates else None


def _record_for_scene_dir(scene_dir: Path) -> ASESceneRecord:
    scene_id = scene_dir.name
    ckpts_dir = scene_dir / "ckpts"
    stats_dir = scene_dir / "stats"
    transforms_path = scene_dir / "transforms_train.json"
    warnings: list[str] = []

    if not ckpts_dir.is_dir():
        warnings.append("missing ckpts directory")
    if not stats_dir.is_dir():
        warnings.append("missing stats directory")
    if not transforms_path.is_file():
        warnings.append("missing transforms_train.json")

    ply_path = find_ase_ply(ckpts_dir) if ckpts_dir.is_dir() else None
    if ply_path is None:
        warnings.append("missing Gaussian PLY in ckpts")

    return ASESceneRecord(
        scene_id=scene_id,
        scene_dir=scene_dir,
        ckpts_dir=ckpts_dir,
        stats_dir=stats_dir,
        ply_path=ply_path,
        transforms_path=transforms_path,
        valid=not warnings,
        warnings=warnings,
    )


def discover_ase_scenes(root: Union[str, Path]) -> list[ASESceneRecord]:
    """Discover fixed-layout ASE scenes from direct child directories of root."""

    root = Path(root)
    if not root.exists():
        return []
    return [
        _record_for_scene_dir(scene_dir)
        for scene_dir in sorted(child for child in root.iterdir() if child.is_dir())
        if not scene_dir.name.startswith(".")
    ]


def _safe_num_gaussians(record: ASESceneRecord) -> tuple[Optional[int], Optional[str]]:
    if record.ply_path is None:
        return None, "missing PLY"
    try:
        return get_gaussian_ply_num_vertices(record.ply_path), None
    except Exception as exc:
        return None, str(exc)


def build_ase_scene_manifest(root: Union[str, Path]) -> list[dict]:
    """Build a JSON-serializable manifest for fixed-layout ASE scenes."""

    manifest: list[dict] = []
    for record in discover_ase_scenes(root):
        warnings = list(record.warnings)
        num_gaussians, num_error = _safe_num_gaussians(record)
        if num_error is not None:
            warnings.append(f"failed to count gaussians: {num_error}")

        stats = read_ase_stats(record.stats_dir)

        camera_info = None
        if record.transforms_path.is_file():
            try:
                camera_info = camera_summary(
                    read_ase_cameras(record.transforms_path, scene_id=record.scene_id)
                )
                warnings.extend(camera_info.get("warnings", []))
            except Exception as exc:
                camera_info = {"scene_id": record.scene_id, "warnings": [str(exc)]}
                warnings.append(f"failed to read cameras: {exc}")

        manifest.append(
            {
                "scene_id": record.scene_id,
                "scene_dir": str(record.scene_dir),
                "ckpts_dir": str(record.ckpts_dir),
                "stats_dir": str(record.stats_dir),
                "ply_path": str(record.ply_path) if record.ply_path is not None else None,
                "transforms_path": str(record.transforms_path),
                "num_gaussians": num_gaussians,
                "stats": stats,
                "camera_summary": camera_info,
                "valid": record.valid and num_error is None and camera_info is not None,
                "warnings": warnings,
            }
        )

    return manifest


def load_ase_scene_gaussians(record: ASESceneRecord) -> GaussianScene:
    """Load the selected Gaussian PLY for one ASE scene record."""

    if record.ply_path is None:
        raise ValueError(f"scene {record.scene_id} has no Gaussian PLY")
    return read_gaussian_ply(
        record.ply_path,
        scene_id=record.scene_id,
        metadata={
            "dataset": "ASE",
            "scene_dir": str(record.scene_dir),
            "ckpts_dir": str(record.ckpts_dir),
            "stats_dir": str(record.stats_dir),
            "ply_path": str(record.ply_path),
            "transforms_path": str(record.transforms_path),
        },
    )
