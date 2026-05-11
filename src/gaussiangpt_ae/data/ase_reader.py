"""ASE Gaussian data discovery and reading interfaces."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from plyfile import PlyData

from gaussiangpt_ae.data.npz_io import is_gaussian_npz, read_gaussian_npz
from gaussiangpt_ae.data.ply_io import is_gaussian_ply, read_gaussian_ply
from gaussiangpt_ae.data.schema import GaussianScene, validate_gaussian_scene

LOGGER = logging.getLogger(__name__)

METADATA_FILENAMES = (
    "metadata.json",
    "transforms.json",
    "cameras.json",
    "scene.json",
    "config.json",
    "dataset_info.json",
)
EXTRA_DIR_NAMES = ("images", "image", "depth", "camera", "cameras", "trajectory")


@dataclass(slots=True)
class ASEGaussianItem:
    """A discovered ASE Gaussian asset and nearby context files."""

    scene_id: str
    root: Path
    scene_dir: Path
    gaussian_path: Path
    gaussian_format: str
    metadata_paths: list[Path]
    extra_files: dict


def _scene_dir_for_path(root: Path, gaussian_path: Path) -> Path:
    relative = gaussian_path.relative_to(root)
    if len(relative.parts) <= 1:
        return gaussian_path.parent
    return root / relative.parts[0]


def _metadata_paths(scene_dir: Path, gaussian_path: Path) -> list[Path]:
    candidates: list[Path] = []
    current = gaussian_path.parent
    while True:
        for filename in METADATA_FILENAMES:
            path = current / filename
            if path.exists() and path not in candidates:
                candidates.append(path)
        if current == scene_dir or current == current.parent:
            break
        current = current.parent
    return sorted(candidates)


def _extra_files(scene_dir: Path) -> dict:
    extras: dict[str, str | list[str]] = {}
    if not scene_dir.exists():
        return extras

    for child in scene_dir.iterdir():
        if not child.is_dir():
            continue
        lowered = child.name.lower()
        if any(name in lowered for name in EXTRA_DIR_NAMES):
            if child.name in extras:
                existing = extras[child.name]
                if isinstance(existing, list):
                    existing.append(str(child))
                else:
                    extras[child.name] = [existing, str(child)]
            else:
                extras[child.name] = str(child)
    return extras


def _iteration_number(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"iteration_(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def _candidate_priority(path: Path, gaussian_format: str) -> tuple[int, int, str]:
    name = path.name.lower()
    parent = path.parent.name.lower()
    grandparent = path.parent.parent.name.lower() if len(path.parents) > 1 else ""

    if (
        gaussian_format == "ply"
        and name == "point_cloud.ply"
        and parent.startswith("iteration_")
        and grandparent == "point_cloud"
    ):
        return (0, -_iteration_number(path), str(path))

    if gaussian_format == "ply":
        named_priority = {
            "point_cloud.ply": 1,
            "gaussian.ply": 2,
            "gaussians.ply": 3,
            "splats.ply": 4,
            "3dgs.ply": 5,
        }
        return (named_priority.get(name, 7), 0, str(path))

    return (6, 0, str(path))


def _candidate_paths(root: Path) -> Iterator[Path]:
    yield from root.rglob("*.ply")
    yield from root.rglob("*.npz")


def discover_ase_gaussian_items(root: str | Path) -> list[ASEGaussianItem]:
    """Recursively discover readable ASE Gaussian files under a HF download root."""

    root = Path(root)
    grouped: dict[Path, list[tuple[tuple[int, int, str], Path, str]]] = {}

    for path in _candidate_paths(root):
        gaussian_format = path.suffix.lower().lstrip(".")
        try:
            valid = is_gaussian_ply(path) if gaussian_format == "ply" else is_gaussian_npz(path)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable candidate %s: %s", path, exc)
            continue
        if not valid:
            continue

        scene_dir = _scene_dir_for_path(root, path)
        grouped.setdefault(scene_dir, []).append(
            (_candidate_priority(path, gaussian_format), path, gaussian_format)
        )

    items: list[ASEGaussianItem] = []
    for scene_dir, candidates in sorted(grouped.items(), key=lambda item: str(item[0])):
        _, gaussian_path, gaussian_format = sorted(candidates, key=lambda item: item[0])[0]
        scene_id = scene_dir.name or gaussian_path.stem
        items.append(
            ASEGaussianItem(
                scene_id=scene_id,
                root=root,
                scene_dir=scene_dir,
                gaussian_path=gaussian_path,
                gaussian_format=gaussian_format,
                metadata_paths=_metadata_paths(scene_dir, gaussian_path),
                extra_files=_extra_files(scene_dir),
            )
        )

    return items


def _item_metadata(item: ASEGaussianItem) -> dict:
    return {
        "dataset": "ASE",
        "source_root": str(item.root),
        "scene_dir": str(item.scene_dir),
        "gaussian_path": str(item.gaussian_path),
        "gaussian_format": item.gaussian_format,
        "metadata_paths": [str(path) for path in item.metadata_paths],
        "extra_files": item.extra_files,
    }


def read_ase_item(item: ASEGaussianItem) -> GaussianScene:
    """Read one discovered ASE item into a GaussianScene."""

    metadata = _item_metadata(item)
    if item.gaussian_format == "ply":
        scene = read_gaussian_ply(item.gaussian_path, scene_id=item.scene_id, metadata=metadata)
    elif item.gaussian_format == "npz":
        scene = read_gaussian_npz(item.gaussian_path, scene_id=item.scene_id, metadata=metadata)
    else:
        raise ValueError(f"unsupported gaussian_format: {item.gaussian_format}")

    validate_gaussian_scene(scene)
    return scene


def iter_ase_root(
    root: str | Path,
    scene_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> Iterator[GaussianScene]:
    """Yield GaussianScene objects one by one from an ASE root."""

    selected = set(scene_ids) if scene_ids is not None else None
    for item in discover_ase_gaussian_items(root):
        if selected is not None and item.scene_id not in selected:
            continue
        yield read_ase_item(item)


def read_ase_root(
    root: str | Path,
    limit: int | None = None,
    scene_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[GaussianScene]:
    """Read ASE Gaussian scenes from a local root."""

    scenes: list[GaussianScene] = []
    for scene in iter_ase_root(root, scene_ids=scene_ids):
        scenes.append(scene)
        if limit is not None and len(scenes) >= limit:
            break
    return scenes


def _safe_num_gaussians(path: Path, gaussian_format: str) -> tuple[int | None, str | None]:
    try:
        if gaussian_format == "ply":
            return len(PlyData.read(str(path))["vertex"].data), None
        with np.load(path, allow_pickle=False) as data:
            return int(data["xyz"].shape[0]), None
    except Exception as exc:
        return None, str(exc)


def build_ase_manifest(root: str | Path) -> list[dict]:
    """Build a JSON-serializable manifest for discovered ASE Gaussian items."""

    manifest: list[dict] = []
    for item in discover_ase_gaussian_items(root):
        num_gaussians, error = _safe_num_gaussians(item.gaussian_path, item.gaussian_format)
        manifest.append(
            {
                "scene_id": item.scene_id,
                "scene_dir": str(item.scene_dir),
                "gaussian_path": str(item.gaussian_path),
                "gaussian_format": item.gaussian_format,
                "has_gaussian": True,
                "metadata_paths": [str(path) for path in item.metadata_paths],
                "extra_files": item.extra_files,
                "num_gaussians": num_gaussians,
                "error": error,
            }
        )
    return manifest


def write_ase_manifest(manifest: list[dict], path: str | Path) -> None:
    """Write an ASE manifest as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

