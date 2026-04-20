# Geometry-Balanced Experiment Suite Design

## Context

The current repository already contains:

- a method statement that positions CAME-Net as an approximately `SE(3)`-equivariant multimodal network with a strict geometric core
- a lightweight 5-class `ModelNet40` experiment path
- internal components that are directly testable at the geometry level: `MPE`, `GCA`, `GLN`, and the soft equivariance regularizer

What is missing is an experiment structure that matches the paper's honest evidence scope. The present training path still pushes attention toward task accuracy alone, while the paper's strongest contributions are actually geometric:

1. a strict geometric scoring core inside `GCA`
2. measurable approximate equivariance at end-to-end level
3. a useful performance-geometry-efficiency tradeoff under small compute

The redesign should therefore make the experiment layer serve these three claims directly instead of behaving like a generic benchmark script.

## Narrowed Paper Claims

The experiment suite must align the implementation and all generated summaries to the following three claims:

1. **CAME-Net is a multimodal network with a strict geometric core and approximately equivariant end-to-end behavior.**
2. **Under controlled rigid transformations, CAME-Net is more stable than non-geometric or de-geometrized fusion baselines.**
3. **Under small-compute constraints, CAME-Net provides a better performance-geometric-consistency-efficiency tradeoff than internal ablated baselines.**

These claims deliberately avoid any "best overall multimodal model" framing and deliberately avoid requiring exhaustive comparison against large external models.

## Goals

Add a unified experiment suite that is managed from the current small-experiment entrypoint but is reoriented around four verifiable propositions:

1. `GCA` scalar attention scores are invariant under a shared motor action.
2. End-to-end equivariance error `epsilon_Phi(X, M)` can be measured and plotted as a function of rotation angle and translation magnitude.
3. The main approximation sources can be isolated through focused ablations.
4. The method retains practical task value on a lightweight public task setting without requiring large compute.

## Non-Goals

This design does **not** aim to:

- prove strict end-to-end `SE(3)` equivariance
- compete with large external multimodal models
- run a full `ModelNet40` large-budget benchmark by default
- add new datasets or external pretrained baselines
- materially change the mathematical purpose of `MPE`, `GCA`, `GLN`, or the equivariance regularizer

## User-Facing Experiment Modes

The unified experiment entrypoint remains `run_small_modelnet_experiment.py`, but it becomes a suite launcher instead of a single fixed demo.

Supported suite modes:

- `geometry` (default): geometry-first evidence package
- `balanced`: geometry package plus lightweight task experiments
- `all`: everything in `balanced` plus extended ablation coverage

The default mode must be `geometry`, because it best matches the narrowed claims and the user's `RTX 3080` compute budget.

## Recommended Internal Structure

Externally, the suite should remain unified. Internally, it should be split for clarity:

- `small_modelnet_experiment.py`
  - experiment orchestration
  - shared config objects
  - dataset subset preparation
  - artifact directory creation
  - summary and claim-to-evidence aggregation
- `controlled_geometry_experiments.py`
  - `GCA` sanity checks
  - equivariance error curve generation
  - ablation execution and aggregation
  - controlled robustness evaluation
- existing training/model files remain method-focused rather than becoming experiment orchestration code

This keeps the public experiment workflow unified while preventing the existing small-experiment helper from becoming an unreadable single-file grab bag.

## Core Experimental Propositions

### 1. GCA Invariance Sanity Check

This experiment directly targets the strict geometric core claim.

Procedure:

- construct a small batch of multivector tokens from point-cloud inputs
- compute raw `GCA` attention scores before softmax
- sample one or more shared motors
- apply the same motor to the whole token set
- recompute the raw `GCA` attention scores
- measure absolute and relative score differences

Expected result:

- for the strict scalar-score path, score differences should be numerically near zero up to floating-point tolerance

Outputs:

- `gca_score_invariance.json`
- a short markdown interpretation block included in the suite summary

### 2. Approximate Equivariance Curves

This experiment targets the approximate end-to-end equivariance claim.

Procedure:

- choose a fixed set of representative point-cloud samples
- define a rotation-angle grid and a translation-magnitude grid
- for each grid point, sample one or more motors with that magnitude
- compute
  `epsilon_Phi(X, M) = D(Phi(M · X), M · Phi(X))`
- aggregate mean and dispersion
- optionally break the error down by grade using the existing grade-wise distance utilities

The curves should be generated for:

- the full model
- the no-regularizer ablation
- the no-GLN ablation when it can run stably
- the non-geometric fusion baseline

Outputs:

- `equivariance_rotation_curve.csv`
- `equivariance_translation_curve.csv`
- `equivariance_rotation_curve.png`
- `equivariance_translation_curve.png`
- optional `gradewise_equivariance_curve.json`

The summary should explicitly interpret the curves as **approximate equivariance measurements**, not exact proofs.

### 3. Error-Source Ablation Suite

This experiment targets the approximation-source attribution claim.

Minimum ablations:

- `full`: current CAME-Net configuration
- `no_gln`: remove `GLN` from the stack or replace it with identity
- `no_soft_equiv_reg`: train/evaluate without the regularizer
- `unconstrained_bivector`: point branch no longer restricts grade-2 output to Euclidean bivectors
- `scalar_only`: keep only scalar channels as a deliberately de-geometrized semantic baseline
- `non_geometric_fusion`: replace the geometric score rule with a coefficient-space baseline that does not use the scalar part of the geometric product

The suite should capture for each variant:

- task accuracy
- mean equivariance error
- runtime
- parameter count
- notes on training stability or failure

The output table is central to the paper's revised narrative because it turns vague approximation talk into measured attribution.

Outputs:

- `ablation_metrics.json`
- `ablation_table.csv`
- `ablation_tradeoff_plot.png`

### 4. Lightweight Task-Level Evidence

This experiment targets the practical usefulness claim without turning the paper back into a benchmark race.

Task settings:

- public 5-class `ModelNet40` subset classification already supported by the repository
- controlled transformed-input robustness evaluation on the same task

Required comparisons:

- full model
- non-geometric fusion baseline
- at least one de-geometrized ablation such as `scalar_only`

Required task outputs:

- clean classification accuracy
- transformed-input robustness accuracy or accuracy drop
- average equivariance error on the same evaluation set
- runtime and parameter count

This gives a compact but defensible "performance-geometric-consistency-efficiency" story without requiring full-dataset large-budget training.

Outputs:

- `task_metrics.json`
- `task_comparison_table.csv`
- `robustness_vs_error_plot.png`

## Default Compute Budget and Runtime Philosophy

The suite must be explicitly designed for a single remote `RTX 3080` server run.

Default expectations:

- `geometry` suite should complete without any large training job
- `balanced` suite may include one lightweight training run plus controlled evaluations
- default models remain small: low layer count, low head count, low hidden dimension, moderate point count
- the suite should prefer deterministic or low-stochastic settings whenever the experiment is measuring geometry rather than augmentation robustness

The design should minimize repeated retraining by:

- reusing trained checkpoints across sub-experiments when the comparison is post-hoc and fair
- separating "training-required" experiments from "forward-only" geometry diagnostics

## Baseline Policy

Baselines must stay internal and controlled.

Acceptable baselines:

- de-geometrized variants of CAME-Net
- non-geometric fusion variants implemented inside the repo
- disabled-module ablations

Avoid:

- pulling in external large architectures
- comparing against heavyweight multimodal models that the available hardware cannot evaluate fairly

This keeps the comparison fair, reproducible, and aligned with the narrowed claims.

## Artifact Layout

Each run should produce a timestamped root with consistent subdirectories:

- `geometry/`
- `ablation/`
- `task/`
- `tables/`
- `summary.md`
- `claim_to_evidence.md`
- `config.json`

`claim_to_evidence.md` is required. It should explicitly map each narrowed claim to:

- the experiment that tests it
- the artifact files containing the evidence
- the caveats that limit what can be concluded

This file is intended to make the paper's checklist compliance auditable.

## Method Text Alignment

The suite redesign must be accompanied by a manuscript-positioning update in `METHOD_SECTION.tex` and any summary-facing documentation that currently overstates the paper.

The text should align to the narrowed claims above and should explicitly say:

- the strict part is the geometric scoring core
- the end-to-end network is only approximately equivariant
- the evidence is centered on stability under controlled rigid transformations and tradeoff under small compute

## Testing Requirements

The implementation must add lightweight tests for the new experiment layer:

- `GCA` invariance sanity check returns near-zero score change on a toy case
- equivariance-curve generation produces monotone, well-formed outputs and artifact files
- ablation registry enumerates the required variants
- default suite selection is `geometry`
- artifact roots contain the required claim-to-evidence and summary files

Tests should avoid full training whenever possible and instead validate structure, determinism, and artifact generation.

## Success Criteria

The redesign is successful when:

1. the default experiment path is no longer "train a small benchmark demo" but "produce geometry-first evidence"
2. the suite can run on the user's remote `RTX 3080` without requiring large-budget training
3. each narrowed paper claim is mapped to a concrete experiment and output artifact
4. at least one task-level result remains in the suite, but the suite no longer depends on benchmark dominance
5. the generated summaries and documentation use claim language that matches the actual evidence scope

