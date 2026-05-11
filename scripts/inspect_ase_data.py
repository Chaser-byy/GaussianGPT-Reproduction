"""Inspect a local ASE Hugging Face download and write a Gaussian manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from gaussiangpt_ae.data.ase_reader import build_ase_manifest, write_ase_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = build_ase_manifest(Path(args.root))
    if args.limit is not None:
        manifest = manifest[: args.limit]
    write_ase_manifest(manifest, Path(args.out))

    total_items = len(manifest)
    valid_gaussian_items = sum(1 for item in manifest if item["has_gaussian"] and not item["error"])
    ply_items = sum(1 for item in manifest if item["gaussian_format"] == "ply")
    npz_items = sum(1 for item in manifest if item["gaussian_format"] == "npz")
    failed_items = sum(1 for item in manifest if item["error"])

    print(f"total_items: {total_items}")
    print(f"valid_gaussian_items: {valid_gaussian_items}")
    print(f"ply_items: {ply_items}")
    print(f"npz_items: {npz_items}")
    print(f"failed_items: {failed_items}")


if __name__ == "__main__":
    main()

