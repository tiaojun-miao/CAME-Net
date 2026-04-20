# Geometry-Balanced Experiment Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current benchmark-like small experiment flow with a unified geometry-balanced experiment suite whose default mode is a geometry-first evidence package aligned to the narrowed paper claims.

**Architecture:** Keep a single user-facing experiment entrypoint, but split the internals into orchestration and controlled-geometry helpers. Extend the current small-experiment artifact pipeline so every run produces claim-oriented evidence files, ablation tables, and lightweight task results without turning the core model files into experiment glue.

**Tech Stack:** Python, PyTorch, existing CAME-Net modules, existing small-experiment artifact helpers, lightweight plotting utilities already used in the repository.

---

### Task 1: Add failing tests for the new suite contract

**Files:**
- Modify: `F:\CAME-Net\test_small_modelnet_experiment.py`
- Test: `F:\CAME-Net\test_small_modelnet_experiment.py`

- [ ] **Step 1: Write a failing test for the default suite mode**

Add a test that expects the default config to expose a `geometry` suite mode and to write a claim-to-evidence artifact:

```python
def test_default_suite_mode_is_geometry():
    config = SmallExperimentConfig()
    assert config.suite == "geometry"


def test_create_experiment_artifacts_writes_claim_to_evidence_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = create_experiment_artifacts(
            artifact_root=tmpdir,
            config={"suite": "geometry"},
            history={"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
            metrics={"overall_accuracy": 0.0, "mean_class_accuracy": 0.0, "per_class_accuracy": {}},
            selected_classes=["airplane", "chair"],
            confusion_matrix=np.eye(2, dtype=np.int64),
            sample_predictions=[],
            runtime_seconds=1.0,
            extra_artifacts={"claim_to_evidence": "# Claim to Evidence\n"},
        )
        assert (Path(artifact_dir) / "claim_to_evidence.md").exists()
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: failure because the config does not yet expose suite selection and artifact writing does not yet require `claim_to_evidence.md`.

- [ ] **Step 3: Commit after the failing test is captured**

```powershell
git add test_small_modelnet_experiment.py
git commit -m "test: add geometry-suite contract coverage"
```

### Task 2: Add suite configuration and mode selection

**Files:**
- Modify: `F:\CAME-Net\small_modelnet_experiment.py`
- Modify: `F:\CAME-Net\run_small_modelnet_experiment.py`
- Test: `F:\CAME-Net\test_small_modelnet_experiment.py`

- [ ] **Step 1: Extend the config object with suite selection**

Update the existing experiment config so it contains at least:

```python
@dataclass
class SmallExperimentConfig:
    suite: str = "geometry"
    artifact_root: str = "artifacts/small_modelnet_experiment"
    ...
```

Accepted values should be `geometry`, `balanced`, and `all`.

- [ ] **Step 2: Route the entrypoint through the suite value**

Update `run_small_modelnet_experiment.py` so it accepts an optional CLI suite selection and passes it through:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["geometry", "balanced", "all"], default="geometry")
    args = parser.parse_args()

    config = SmallExperimentConfig(suite=args.suite)
    result = run_small_experiment(config)
    print(f"Suite: {result['suite']}")
```

- [ ] **Step 3: Run the targeted test and verify it passes**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: the new suite-default checks pass, but later tests may still fail because geometry-specific experiment artifacts are not implemented yet.

- [ ] **Step 4: Commit**

```powershell
git add small_modelnet_experiment.py run_small_modelnet_experiment.py test_small_modelnet_experiment.py
git commit -m "feat: add experiment suite selection"
```

### Task 3: Introduce controlled geometry experiment helpers

**Files:**
- Create: `F:\CAME-Net\controlled_geometry_experiments.py`
- Modify: `F:\CAME-Net\small_modelnet_experiment.py`
- Test: `F:\CAME-Net\test_small_modelnet_experiment.py`

- [ ] **Step 1: Write failing tests for geometry helper outputs**

Add tests that expect helper functions for:

- GCA score invariance sanity output
- equivariance curve generation output
- ablation registry enumeration

Example:

```python
def test_geometry_suite_exports_required_helper_results():
    from controlled_geometry_experiments import (
        run_gca_score_invariance_check,
        run_equivariance_curve_experiment,
        list_ablation_variants,
    )

    assert "full" in list_ablation_variants()
    assert "scalar_only" in list_ablation_variants()

    result = run_gca_score_invariance_check(...)
    assert "max_abs_score_delta" in result

    curves = run_equivariance_curve_experiment(...)
    assert "rotation_curve" in curves
    assert "translation_curve" in curves
```

- [ ] **Step 2: Create the helper module with narrow responsibilities**

`controlled_geometry_experiments.py` should define focused functions such as:

```python
def run_gca_score_invariance_check(...): ...
def run_equivariance_curve_experiment(...): ...
def list_ablation_variants() -> list[str]: ...
def build_ablated_model(...): ...
```

Do not turn this file into a second training entrypoint. It should only contain controlled experiment logic.

- [ ] **Step 3: Re-run the geometry helper tests**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: helper-shape tests pass, while artifact aggregation may still fail until the suite is wired into the main experiment flow.

- [ ] **Step 4: Commit**

```powershell
git add controlled_geometry_experiments.py small_modelnet_experiment.py test_small_modelnet_experiment.py
git commit -m "feat: add controlled geometry experiment helpers"
```

### Task 4: Wire the geometry suite into the unified experiment flow

**Files:**
- Modify: `F:\CAME-Net\small_modelnet_experiment.py`
- Test: `F:\CAME-Net\test_small_modelnet_experiment.py`

- [ ] **Step 1: Dispatch geometry-only runs without mandatory training**

Update `run_small_experiment(config)` so:

- `geometry` mode runs the invariance check, equivariance curves, and ablation aggregation first
- default geometry mode does not require a full training loop unless a specific sub-experiment needs a trained checkpoint
- post-hoc evaluations reuse checkpoints when fair

- [ ] **Step 2: Persist geometry artifacts in dedicated subdirectories**

Use a layout like:

```python
geometry_dir = artifact_dir / "geometry"
ablation_dir = artifact_dir / "ablation"
tables_dir = artifact_dir / "tables"
```

Store JSON/CSV/PNG outputs in those directories instead of mixing everything at the run root.

- [ ] **Step 3: Generate `claim_to_evidence.md`**

Add a function that writes a markdown file mapping:

- claim 1 -> GCA invariance artifacts
- claim 2 -> equivariance curves and robustness comparisons
- claim 3 -> tradeoff tables and task metrics

- [ ] **Step 4: Run the artifact-generation tests**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: artifact structure tests pass and summaries reference the claim-to-evidence file.

- [ ] **Step 5: Commit**

```powershell
git add small_modelnet_experiment.py test_small_modelnet_experiment.py
git commit -m "feat: wire geometry suite artifacts and summaries"
```

### Task 5: Add the internal non-geometric and de-geometrized baselines

**Files:**
- Modify: `F:\CAME-Net\controlled_geometry_experiments.py`
- Modify: `F:\CAME-Net\small_modelnet_experiment.py`
- Test: `F:\CAME-Net\test_small_modelnet_experiment.py`

- [ ] **Step 1: Write failing tests for required ablation names**

Add a test that verifies the ablation registry includes:

```python
expected = {
    "full",
    "no_gln",
    "no_soft_equiv_reg",
    "unconstrained_bivector",
    "scalar_only",
    "non_geometric_fusion",
}
assert expected.issubset(set(list_ablation_variants()))
```

- [ ] **Step 2: Implement the ablation registry and model-construction hooks**

Implement controlled ablation construction in the experiment layer rather than scattering ad hoc flags through the core model files. The experiment layer may wrap or patch components, but should keep method files readable.

- [ ] **Step 3: Capture runtime, parameter count, task accuracy, and equivariance error per ablation**

Produce a single normalized metrics structure:

```python
{
    "variant": "full",
    "task_accuracy": ...,
    "mean_equivariance_error": ...,
    "runtime_seconds": ...,
    "parameter_count": ...,
    "status": "ok",
}
```

- [ ] **Step 4: Re-run tests**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: ablation-registry tests pass and no required variant name is missing.

- [ ] **Step 5: Commit**

```powershell
git add controlled_geometry_experiments.py small_modelnet_experiment.py test_small_modelnet_experiment.py
git commit -m "feat: add internal geometry ablation baselines"
```

### Task 6: Add the balanced task-evidence mode

**Files:**
- Modify: `F:\CAME-Net\small_modelnet_experiment.py`
- Modify: `F:\CAME-Net\controlled_geometry_experiments.py`
- Test: `F:\CAME-Net\test_small_modelnet_experiment.py`

- [ ] **Step 1: Add a failing test for `balanced` mode**

Add a test that expects `balanced` runs to include both geometry outputs and task metrics:

```python
def test_balanced_suite_writes_task_and_geometry_outputs():
    result = run_small_experiment(SmallExperimentConfig(suite="balanced", ...))
    root = Path(result["artifact_dir"])
    assert (root / "geometry").exists()
    assert (root / "task").exists()
    assert (root / "tables" / "task_comparison_table.csv").exists()
```

- [ ] **Step 2: Implement lightweight task evidence**

Balanced mode should add:

- one small subset classification result
- one controlled transformed-input robustness comparison
- one tradeoff table that includes accuracy, equivariance error, runtime, and parameter count

- [ ] **Step 3: Re-run the suite tests**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: balanced-mode contract tests pass without requiring long training.

- [ ] **Step 4: Commit**

```powershell
git add small_modelnet_experiment.py controlled_geometry_experiments.py test_small_modelnet_experiment.py
git commit -m "feat: add balanced geometry-task suite"
```

### Task 7: Align manuscript and repository-facing wording

**Files:**
- Modify: `F:\CAME-Net\METHOD_SECTION.tex`
- Modify: `F:\CAME-Net\PROJECT_STRUCTURE.md`

- [ ] **Step 1: Update method wording to the narrowed claims**

Revise the method-positioning text so it explicitly states:

- strict geometric core in `GCA`
- approximate end-to-end equivariance
- controlled rigid-transformation stability rather than benchmark dominance
- performance-geometry-efficiency tradeoff under small compute

- [ ] **Step 2: Update project structure documentation**

Add a short section describing the new suite modes and the claim-to-evidence artifact.

- [ ] **Step 3: Verify the wording does not reintroduce overclaiming**

Run:

```powershell
rg -n "strict end-to-end|state-of-the-art|best overall|fully equivariant" F:\CAME-Net\METHOD_SECTION.tex F:\CAME-Net\PROJECT_STRUCTURE.md
```

Expected: no stale overclaiming language remains.

- [ ] **Step 4: Commit**

```powershell
git add METHOD_SECTION.tex PROJECT_STRUCTURE.md
git commit -m "docs: align claims with geometry-balanced evidence suite"
```

### Task 8: Run verification and a geometry-default smoke suite

**Files:**
- Modify: `F:\CAME-Net\test_small_modelnet_experiment.py` if needed for final stabilization
- Test: `F:\CAME-Net\test_small_modelnet_experiment.py`
- Test: `F:\CAME-Net\test_method_alignment.py`
- Test: `F:\CAME-Net\test_came_net.py`

- [ ] **Step 1: Run the experiment-layer tests**

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: `PASS`, with any data-dependent smoke paths skipped cleanly when necessary.

- [ ] **Step 2: Re-run the existing method regressions**

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_method_alignment.py
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_came_net.py
```

Expected: both pass; no geometry-suite change regresses the method-alignment contract.

- [ ] **Step 3: Run one geometry-default smoke experiment**

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' run_small_modelnet_experiment.py --suite geometry
```

Expected: a run directory is created with `geometry/`, `ablation/`, `tables/`, `summary.md`, and `claim_to_evidence.md`.

- [ ] **Step 4: Commit**

```powershell
git add small_modelnet_experiment.py controlled_geometry_experiments.py test_small_modelnet_experiment.py METHOD_SECTION.tex PROJECT_STRUCTURE.md run_small_modelnet_experiment.py
git commit -m "feat: add geometry-balanced experiment suite"
```

## Self-Review

- Spec coverage: the plan covers unified suite management, geometry sanity checks, equivariance curves, internal ablations, lightweight task evidence, artifact design, and manuscript claim alignment.
- Placeholder scan: each task names exact files, commands, and required outputs; no `TBD` or "appropriate handling" placeholders remain.
- Type consistency: the plan consistently uses `SmallExperimentConfig`, `run_small_experiment`, `claim_to_evidence.md`, `geometry|balanced|all`, and the ablation names defined in the spec.

