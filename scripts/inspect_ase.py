"""Inspect a fixed-layout ASE dataset root and write a scene manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gaussiangpt_ae.data.ase_scene import (
    build_ase_scene_manifest,
    discover_ase_scenes,
    load_ase_scene_gaussians,
)
from gaussiangpt_ae.data.schema import compute_gaussian_scene_stats


def _summary(manifest: list[dict]) -> dict:
    counts = [item["num_gaussians"] for item in manifest if item["num_gaussians"] is not None]
    return {
        "total_scene_dirs": len(manifest),
        "valid_scenes": sum(1 for item in manifest if item["valid"]),
        "scene_ids": [item["scene_id"] for item in manifest],
        "total_gaussians": int(sum(counts)) if counts else 0,
        "min_num_gaussians": int(min(counts)) if counts else None,
        "max_num_gaussians": int(max(counts)) if counts else None,
        "mean_num_gaussians": float(sum(counts) / len(counts)) if counts else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--load-gaussians", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = build_ase_scene_manifest(root)

    if args.load_gaussians:
        records_by_id = {record.scene_id: record for record in discover_ase_scenes(root)}
        for item in manifest:
            record = records_by_id[item["scene_id"]]
            try:
                scene = load_ase_scene_gaussians(record)
            except Exception as exc:
                item["loaded"] = False
                item["load_error"] = str(exc)
                item["gaussian_scene_stats"] = None
            else:
                item["loaded"] = True
                item["load_error"] = None
                item["gaussian_scene_stats"] = compute_gaussian_scene_stats(scene)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = _summary(manifest)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
