# CAME-Net Project Structure

## Overview

This repository implements the current CAME-Net code path: a PGA-based network with
structured point-cloud embedding, grade-preserving geometric attention, grade-wise
normalization, and soft equivariance regularization.

The current training pipeline is point-cloud focused. Image and text branches are
implemented at the encoder level and can be fused through the shared attention stack,
but there is not yet a production multimodal dataset pipeline in this repository.

## Core Files

### `pga_algebra.py`
- Clifford / PGA basis definitions and multiplication tables
- OPNS point construction for Euclidean coordinates
- Motor construction for sampled rigid motions
- Multivector wrapper and sandwich action helpers

### `mpe.py`
- Point-cloud MPE uses four channels:
  - scalar semantic channel from rigid-motion invariants
  - grade-1 local tangent plane
  - grade-2 Euclidean bivector restricted to `e23, e31, e12`
  - grade-3 OPNS PGA point
- Image and text MPE branches only use scalar and pseudoscalar channels
- Multimodal fusion is token concatenation; cross-modal interaction happens in GCA

### `gca.py`
- Grade-preserving query / key / value projections
- Attention score from the scalar part of the geometric product
- Non-scalar learned maps avoid fixed bias terms
- Scalar-part scoring is the strict geometric core; learned maps are approximation sources

### `gln.py`
- Grade-wise normalization using the scalar part of `X_r * reverse(X_r)`
- Learnable per-grade scale
- Bias applied only to scalar and pseudoscalar grades

### `came_net.py`
- Full CAME-Net model definition
- Residual stack of GCA + grade-wise feed-forward blocks
- Global mean pooling and task head

### `equiv_loss.py`
- Soft equivariance regularization utilities
- Grade-wise multivector distance for equivariance error measurement
- Random motor sampling for rotation and translation perturbations

### `train.py`
- Training and validation loops
- Linear warmup for equivariance regularization weight
- Equivariance loss evaluated under `eval()` mode to avoid dropout noise during the consistency measurement

### `test_came_net.py`
- Regression tests for the current implementation contract
- Verifies MPE structure, GCA / GLN behavior, forward pass, and training step

### `test_method_alignment.py`
- Focused contract tests added during the method-alignment pass
- Guards Euclidean bivector usage, non-scalar bias removal, GLN bias behavior, geometric distance, and deterministic equivariance-loss evaluation

## Current Method Positioning

- The network is not claimed to be analytically strict end-to-end `SE(3)`-equivariant.
- The scalar scoring rule in GCA is the strict geometric core.
- GLN and generic grade-wise learned maps are treated as approximation sources.
- Soft equivariance regularization reduces residual equivariance error empirically rather than restoring exact analytical equivariance.
