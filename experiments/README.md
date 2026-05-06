# Experiments

This directory contains the benchmark, ablation, and visualization workflows that sit on top of the core CAME-Net method implementation.

## Main groups

- `small_modelnet_experiment.py`
  - lightweight ModelNet40 protocol
- `comparison_experiment.py`
  - method comparisons and ablations
- `robustness_benchmark.py`
  - controlled spatial robustness evaluation
- `scannet_*.py`
  - ScanNet multimodal training, benchmarking, and paper figures
- `run_*.py`
  - user-facing CLI entrypoints

The repository root is intentionally reserved for the method stack itself:

- `method/pga_algebra.py`
- `method/mpe.py`
- `method/gca.py`
- `method/gln.py`
- `method/equiv_loss.py`
- `method/came_net.py`
- `training/train.py`
