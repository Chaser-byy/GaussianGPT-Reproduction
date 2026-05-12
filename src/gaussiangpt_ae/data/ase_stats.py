"""Readers for ASE per-scene stats folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union


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
