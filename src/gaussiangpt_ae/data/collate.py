"""Collate helpers for ASE sparse chunk batches."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def ase_sparse_collate(batch: List[Dict]) -> Dict:
    """Collate ASE chunk samples into sparse coordinates with batch indices."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to collate ASE sparse batches.") from exc

    coords_parts = []
    feats_parts = []
    target_parts = []
    metas = []
    for batch_index, sample in enumerate(batch):
        coords = np.asarray(sample["coords"], dtype=np.int64)
        batch_column = np.full((coords.shape[0], 1), batch_index, dtype=np.int64)
        coords_parts.append(np.concatenate([batch_column, coords], axis=1))
        feats_parts.append(np.asarray(sample["feats"], dtype=np.float32))
        target_parts.append(np.asarray(sample["target_feats"], dtype=np.float32))
        metas.append(sample.get("metadata", {}))

    if coords_parts:
        coords_np = np.concatenate(coords_parts, axis=0)
        feats_np = np.concatenate(feats_parts, axis=0)
        target_np = np.concatenate(target_parts, axis=0)
    else:
        coords_np = np.zeros((0, 4), dtype=np.int64)
        feats_np = np.zeros((0, 14), dtype=np.float32)
        target_np = np.zeros((0, 14), dtype=np.float32)

    return {
        "coords": torch.as_tensor(coords_np, dtype=torch.long),
        "feats": torch.as_tensor(feats_np, dtype=torch.float32),
        "target_feats": torch.as_tensor(target_np, dtype=torch.float32),
        "metas": metas,
    }
