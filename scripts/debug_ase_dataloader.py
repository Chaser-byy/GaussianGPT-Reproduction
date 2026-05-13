"""Debug ASEChunkDataset and sparse collate output shapes."""

from __future__ import annotations

import argparse

from gaussiangpt_ae.data.ase_dataset import ASEChunkDataset
from gaussiangpt_ae.data.collate import ase_sparse_collate


def main() -> None:
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("PyTorch is required for debug_ase_dataloader.py.") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--num-samples-per-epoch", type=int, default=1000)
    parser.add_argument("--chunk-size", type=float, default=4.0)
    parser.add_argument("--occupancy-threshold", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = ASEChunkDataset(
        cache_root=args.cache_root,
        num_samples_per_epoch=args.num_samples_per_epoch,
        chunk_size=args.chunk_size,
        occupancy_threshold=args.occupancy_threshold,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=ase_sparse_collate,
        shuffle=False,
    )

    for batch_index, batch in enumerate(loader):
        if batch_index >= args.num_batches:
            break
        metas = batch["metas"]
        print(f"batch: {batch_index}")
        print(f"coords shape: {tuple(batch['coords'].shape)}")
        print(f"feats shape: {tuple(batch['feats'].shape)}")
        print(f"target_feats shape: {tuple(batch['target_feats'].shape)}")
        print(f"batch scene_ids: {[meta.get('scene_id') for meta in metas]}")
        print(f"occupancy: {[meta.get('occupancy') for meta in metas]}")


if __name__ == "__main__":
    main()
