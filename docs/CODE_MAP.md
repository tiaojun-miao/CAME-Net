# Method-to-Code Map

This repository is organized so the method implementation can be inspected without reading the full experiment stack.

## Core algebra and geometry

- `method/pga_algebra.py`
  - PGA basis definitions
  - multivector utilities
  - motor construction and sandwich actions

## Core network components

- `method/mpe.py`
  - point-anchored multivector embedding
  - multivector attribute construction for points

- `method/gca.py`
  - geometric Clifford attention
  - scalar geometric-product score

- `method/gln.py`
  - geometric layer normalization
  - grade-wise normalization behavior

- `method/equiv_loss.py`
  - residual equivariance regularization
  - hidden-state consistency under sampled rigid motions

- `method/came_net.py`
  - full CAME-Net assembly
  - feature fusion and classifier head

## Training entrypoint

- `training/train.py`
  - main point-cloud training loop
  - checkpointing and optimization utilities

- `training/data_utils.py`
  - ModelNet40 dataset loading
  - point sampling and collation helpers

- `training/torch_runtime_compat.py`
  - torch runtime compatibility shims

## Experiment stack

All benchmark, ablation, ScanNet, and visualization workflows live under `experiments/`.
This separation is intentional: the repository root is documentation-first, while the actual method stack lives in `method/` and `training/`.
