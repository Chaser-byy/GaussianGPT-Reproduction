"""NPZ readers for standard Gaussian attributes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gaussiangpt_ae.data.schema import GaussianScene, validate_gaussian_scene

GAUSSIAN_NPZ_FIELDS = ("xyz", "color", "opacity", "scale", "rotation")


def gaussian_npz_missing_fields(path: str | Path) -> list[str]:
    """Return missing standard Gaussian NPZ fields."""

    with np.load(Path(path), allow_pickle=False) as data:
        present = set(data.files)
    return [field for field in GAUSSIAN_NPZ_FIELDS if field not in present]


def is_gaussian_npz(path: str | Path) -> bool:
    """Return True when an NPZ contains the standard Gaussian fields."""

    try:
        return not gaussian_npz_missing_fields(path)
    except Exception:
        return False


def _read_optional_string(value: Any) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    return str(array.tolist())


def read_gaussian_npz(
    path: str | Path,
    scene_id: str | None = None,
    metadata: dict | None = None,
) -> GaussianScene:
    """Read one standard Gaussian NPZ into a GaussianScene."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in GAUSSIAN_NPZ_FIELDS if field not in data.files]
        if missing:
            raise ValueError(f"missing fields: {missing}")

        npz_scene_id = _read_optional_string(data["scene_id"]) if "scene_id" in data.files else None
        merged_metadata: dict = {}
        if "metadata_json" in data.files:
            parsed = json.loads(_read_optional_string(data["metadata_json"]))
            if not isinstance(parsed, dict):
                raise ValueError("metadata_json must decode to dict")
            merged_metadata.update(parsed)
        merged_metadata.update(metadata or {})

        scene = GaussianScene(
            scene_id=scene_id if scene_id is not None else npz_scene_id or path.stem,
            xyz=np.asarray(data["xyz"], dtype=np.float32),
            color=np.asarray(data["color"], dtype=np.float32),
            opacity=np.asarray(data["opacity"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
            rotation=np.asarray(data["rotation"], dtype=np.float32),
            metadata=merged_metadata,
        )

    validate_gaussian_scene(scene)
    return scene

