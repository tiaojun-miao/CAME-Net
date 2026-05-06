# Python Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository so the root no longer contains scattered Python modules, with method code, training code, experiments, and helper scripts separated into clear folders.

**Architecture:** Move the method primitives into a `method/` package, move training and dataset utilities into a `training/` package, keep benchmark and figure workflows in `experiments/`, and keep standalone data-preparation helpers in `scripts/`. Update imports, packaging metadata, and docs to match the new structure.

**Tech Stack:** Python, PyTorch, setuptools editable packaging, PowerShell file moves

---

### Task 1: Create method and training packages

**Files:**
- Create: `method/__init__.py`
- Create: `training/__init__.py`
- Move: `came_net.py`
- Move: `pga_algebra.py`
- Move: `mpe.py`
- Move: `gca.py`
- Move: `gln.py`
- Move: `equiv_loss.py`
- Move: `train.py`
- Move: `data_utils.py`
- Move: `torch_runtime_compat.py`

- [ ] Create package directories and package markers
- [ ] Move method files into `method/`
- [ ] Move training files into `training/`

### Task 2: Update imports to new package layout

**Files:**
- Modify: `method/*.py`
- Modify: `training/*.py`
- Modify: `experiments/*.py`
- Modify: `scripts/prepare_modelnet40.py` if needed

- [ ] Update method-internal imports to use `method.*`
- [ ] Update training imports to use `method.*` and `training.*`
- [ ] Update experiment imports to use `method.*` and `training.*`
- [ ] Verify no old root-level imports remain

### Task 3: Update packaging and docs

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/CODE_MAP.md`
- Modify: `docs/EXPERIMENTS.md`

- [ ] Update console entrypoints to the new package paths
- [ ] Update setuptools package/module declarations
- [ ] Update repository layout documentation
- [ ] Update method-reading guidance to point at `method/` and `training/`

### Task 4: Verify the refactor

**Files:**
- Verify: `method/*.py`
- Verify: `training/*.py`
- Verify: `experiments/*.py`

- [ ] Run `python -m py_compile` across the moved packages
- [ ] Run `python -m experiments.run_scannet_rigid_benchmark --help`
- [ ] Run a pure import check for the main experiment entrypoints
- [ ] Confirm the root no longer contains scattered method/training `.py` files
