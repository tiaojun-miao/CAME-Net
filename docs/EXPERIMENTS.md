# Experiments

This document maps the repository entrypoints to the paper-facing experiments.

## 1. ModelNet40

### Small balanced experiment

```bash
python -m experiments.run_small_modelnet_experiment
```

### Full training entrypoint

```bash
python -m training.train --data_root ./ModelNet40 --epochs 300 --batch_size 32 --device cuda
```

## 2. Model comparison / ablation

```bash
python -m experiments.run_comparison_experiment --method came --device cuda
python -m experiments.run_comparison_experiment --method pointnet --device cuda
python -m experiments.run_comparison_experiment --method came_non_geometric_fusion --device cuda
```

## 3. Robustness benchmark

```bash
python -m experiments.run_robustness_benchmark --method came --device cuda
python -m experiments.run_attention_score_search --device cuda
```

## 4. ScanNet multimodal experiments

### Main multimodal training loop

```bash
python -m experiments.run_scannet_multimodal_experiment --data-root ./ScanNet-small --device cuda
```

### Comparison / ablation

```bash
python -m experiments.run_scannet_comparison_experiment --method came --data-root ./ScanNet-small --device cuda
python -m experiments.run_scannet_comparison_experiment --method pointnet --data-root ./ScanNet-small --device cuda
```

### Rigid robustness benchmark

```bash
python -m experiments.run_scannet_rigid_benchmark --method came --data-root ./ScanNet-small --device cuda
python -m experiments.run_scannet_rigid_benchmark --method pointnet --data-root ./ScanNet-small --device cuda
```

## 5. Qualitative visualizations

### Indoor scene qualitative comparison

```bash
python -m experiments.run_scannet_qualitative_figure \
  --data-root ./ScanNet-small \
  --came-ckpt /path/to/came/checkpoints/best_model.pth \
  --baseline-ckpt /path/to/baseline/checkpoints/best_model.pth \
  --baseline-method pointnet \
  --scene-ids scene0014_00 scene0015_00 \
  --output ./artifacts/figures/qualitative_scannet
```

### Robustness under spatial transformations

```bash
python -m experiments.run_scannet_spatial_robustness_figure \
  --data-root ./ScanNet-small \
  --came-ckpt /path/to/came/checkpoints/best_model.pth \
  --baseline-ckpt /path/to/baseline/checkpoints/best_model.pth \
  --baseline-method pointnet \
  --scene-ids scene0014_00 scene0015_00 \
  --transform-variants rot_z_45 tz_0p3 \
  --output ./artifacts/figures/spatial_robustness_scannet
```

### Point-wise relevance heatmaps

```bash
python -m experiments.run_scannet_point_relevance_figure \
  --data-root ./ScanNet-small \
  --came-ckpt /path/to/came/checkpoints/best_model.pth \
  --baseline-ckpt /path/to/baseline/checkpoints/best_model.pth \
  --baseline-method pointnet \
  --scene-ids scene0014_00 scene0015_00 \
  --queries doorframe table \
  --output ./artifacts/figures/point_relevance_scannet
```

## 6. ScanNet download helpers

### Original ScanNet downloader

```bash
python scripts/download-scannet.py -o ./ScanNet
```

### Subset-oriented downloader

```bash
python -m experiments.download_scannet_subset --output ./ScanNet-small
```

## 7. Notes

- The visualization scripts are paper-oriented and assume you already have trained checkpoints.
- The ScanNet `small` subset is useful for smoke runs and layout debugging, but not for final quantitative claims.
- MatterPort3D is part of the paper protocol, but a dedicated packaged public adapter is not yet included in the main release.
