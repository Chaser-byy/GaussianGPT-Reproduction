"""ASE scene discovery and manifest construction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from gaussiangpt_ae.data.ase_camera import camera_summary, read_ase_cameras
from gaussiangpt_ae.data.ase_stats import read_ase_stats
from gaussiangpt_ae.data.ply_io import get_gaussian_ply_num_vertices, read_gaussian_ply
from gaussiangpt_ae.data.schema import GaussianScene


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
