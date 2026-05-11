# AGENTS.md

## Project

This repository is a paper-level reproduction of the Autoencoder part of GaussianGPT.

The official GaussianGPT code is not available, so do not assume access to implementation details beyond the paper.

## Local development environment

This machine is Windows + WSL2.

Local development is only for:
- writing code
- running CPU tests
- checking data processing logic
- checking shape logic
- checking Python syntax

Do not require CUDA, spconv, diff-gaussian-rasterization, simple-knn, or 3DGS renderer for normal imports or CPU tests.

CUDA-dependent modules must use lazy imports and clear error messages.

## Target Autoencoder pipeline

The AE pipeline is:

3DGS point_cloud.ply
-> Gaussian primitive dict
-> voxelized sparse grid
-> sparse 3D CNN encoder
-> LFQ / VQ latent indices
-> sparse 3D CNN decoder
-> reconstructed Gaussian attributes

Gaussian primitive fields:

- xyz: [N, 3]
- color: [N, 3], from f_dc_0, f_dc_1, f_dc_2
- opacity: [N, 1]
- scale: [N, 3]
- rotation: [N, 4]

Feature vector order:

relative_xyz: 3
color: 3
opacity: 1
scale: 3
rotation: 4

Total feature dimension: 14.

## Engineering rules

Work module by module.

Each module should have tests.

Do not implement the full model unless explicitly requested.

Prefer simple, readable code over clever abstractions.

Do not put data, checkpoints, outputs, or large ply files into git.

## Expected test command

Use:

pytest -q

## Local vs AutoDL split

Local WSL:
- PLY IO
- voxelization
- dataset
- collate
- Gaussian feature heads
- LFQ
- loss functions
- config parsing
- CPU tests

AutoDL:
- CUDA PyTorch
- spconv
- sparse AE training
- 3DGS renderer
- rendering loss