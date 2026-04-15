# CAME-Net Method Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the codebase and paper agree on one mathematically coherent CAME-Net definition.

**Architecture:** Freeze the PGA representation contract first, then refactor point-cloud MPE, reduce avoidable symmetry breaking in GCA/GLN, replace raw-MSE equivariance loss with a grade-wise geometric distance, and finally align docs/tests.

**Tech Stack:** Python, PyTorch, custom PGA algebra, root-level regression scripts.

---

## Prerequisite Decision

Recommended for this repo:
- Keep Euclidean points as standard OPNS grade-3 PGA objects via `create_point_pga(coords)`.
- Fix the manuscript instead of trying to force `P(p_i) in <Cl>_1` into the current algebra stack.
- Keep the revised manuscript's Euclidean bivector restriction: only `e23, e31, e12`.
- Keep the revised manuscript's honest claim: strict scalar-score core, approximate end-to-end `SE(3)` equivariance.

Do **not** implement the current `grade-1 point` manuscript statement on top of the existing PGA conventions. That needs a separate algebra redesign.

## File Map

- `mpe.py`: replace plane + ideal-bivector embedding with fixed point object + Euclidean bivector channels.
- `gca.py`: disable non-scalar bias and make docs honest about approximation sources.
- `gln.py`: enable only scalar/pseudoscalar bias.
- `equiv_loss.py`: add grade-wise multivector distance and use it everywhere.
- `train.py`: compute equivariance regularization with dropout disabled.
- `METHOD_SECTION.tex`, `PROJECT_STRUCTURE.md`: align text to final implementation.
- `test_method_alignment.py`: focused contract tests.
- `test_came_net.py`: remove assertions that enforce the obsolete design.

### Task 1: Add failing contract tests

**Files:**
- Create: `test_method_alignment.py`
- Modify: `test_came_net.py`

- [ ] Write a failing Euclidean-bivector test.

```python
def test_point_mpe_uses_euclidean_bivectors_only():
    mv = PointCloudMPE(feature_dim=8, hidden_dim=16)(coords, features).data
    assert torch.allclose(mv[..., [8, 9, 10]], torch.zeros_like(mv[..., [8, 9, 10]]), atol=1e-6)
    assert mv[..., [5, 6, 7]].abs().sum().item() > 0.0
```

- [ ] Write a failing deterministic-regularizer test.

```python
def test_equivariance_regularizer_is_repeatable_for_fixed_inputs():
    torch.manual_seed(123)
    loss_a = equivariance_loss_efficient(model, coords, None, num_samples=1)
    torch.manual_seed(123)
    loss_b = equivariance_loss_efficient(model, coords, None, num_samples=1)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)
```

- [ ] Run: `python test_method_alignment.py`
  Expected: FAIL on current code.

### Task 2: Refactor point-cloud MPE

**Files:**
- Modify: `mpe.py`
- Modify: `test_method_alignment.py`
- Modify: `test_came_net.py`

- [ ] Replace `grade-1 plane` and `ideal bivector` heads with `alpha_proj`, `grade2_proj`, `grade3_proj`.

```python
self.grade0_proj = nn.Linear(hidden_dim, 1)
self.alpha_proj = nn.Linear(hidden_dim, 1)
self.grade2_proj = nn.Linear(hidden_dim, 3)
self.grade3_proj = nn.Linear(hidden_dim, 4)
```

- [ ] Emit only Euclidean bivectors and use the fixed OPNS point object as the explicit geometric carrier.

```python
point_object = create_point_pga(coords).data[..., GRADE_INDICES[3]]
output[..., [5, 6, 7]] = self.grade2_proj(hidden)
output[..., [8, 9, 10]] = 0.0
output[..., GRADE_INDICES[3]] = self.alpha_proj(hidden).sigmoid() * point_object + self.grade3_proj(hidden)
```

- [ ] Run: `python test_method_alignment.py`
  Expected: Euclidean-bivector test passes.

### Task 3: Reduce avoidable symmetry breaking in GCA and GLN

**Files:**
- Modify: `gca.py`
- Modify: `gln.py`
- Modify: `came_net.py`

- [ ] Add a grade-aware linear helper with bias only for grades `0` and `4`.

```python
def _grade_linear(in_dim: int, out_dim: int, grade: int) -> nn.Linear:
    return nn.Linear(in_dim, out_dim, bias=(grade in (0, 4)))
```

- [ ] Use that helper for Q/K/V/out projections and grade-wise MLP blocks.
- [ ] Set `GradewiseLayerNorm(..., learnable_bias=True)` and keep bias application restricted to grades `0` and `4`.
- [ ] Rewrite docstrings so they stop implying strict end-to-end equivariance.

- [ ] Run: `python test_came_net.py`
  Expected: shape/regression checks pass after updating stale assertions.

### Task 4: Replace raw coefficient MSE with a grade-wise geometric distance

**Files:**
- Modify: `equiv_loss.py`
- Modify: `test_method_alignment.py`

- [ ] Add `multivector_distance(a, b, eta)`.

```python
def multivector_distance(a: Multivector, b: Multivector, eta=None) -> torch.Tensor:
    if eta is None:
        eta = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    diff = a.data - b.data
    total = diff.new_zeros(())
    for grade, indices in GRADE_INDICES.items():
        grade_diff = torch.zeros_like(diff)
        grade_diff[..., indices] = diff[..., indices]
        rev = grade_diff * REVERSION_SIGNS.to(device=diff.device, dtype=diff.dtype)
        total = total + eta.get(grade, 0.0) * geometric_product(grade_diff, rev)[..., 0].abs().mean()
    return total
```

- [ ] Use `multivector_distance()` in `equivariance_loss_efficient()`, `rotational_equivariance_loss()`, and `translational_equivariance_loss()`.
- [ ] Run: `python test_method_alignment.py`
  Expected: distance tests pass.

### Task 5: Make equivariance regularization deterministic inside training

**Files:**
- Modify: `train.py`
- Modify: `test_method_alignment.py`

- [ ] Temporarily switch the model to `eval()` inside `_equivariance_loss_from_batch()` while keeping gradients enabled.

```python
was_training = model.training
model.eval()
try:
    return equiv_loss_fn(...)
finally:
    if was_training:
        model.train()
```

- [ ] Add a repeatability test that calls the training helper twice with the same seed.
- [ ] Run: `python test_method_alignment.py`
  Expected: repeatability test passes even when model dropout is nonzero.

### Task 6: Align manuscript and project docs

**Files:**
- Modify: `METHOD_SECTION.tex`
- Modify: `PROJECT_STRUCTURE.md`

- [ ] Replace the old point-cloud description with the final code contract:

```tex
\mathcal{X}_i^{\mathrm{pc}} = f_0 + \alpha_3 P(\mathbf{p}_i) + f_2 + f_3,
\qquad
f_2 \in \mathrm{span}\{e_{23}, e_{31}, e_{12}\},
\qquad
P(\mathbf{p}_i) \in \langle \mathrm{Cl} \rangle_3.
```

- [ ] Rewrite GCA and regularization sections so they explicitly distinguish strict scalar-score geometry from approximate learned parameterization.
- [ ] Narrow training claims to point-cloud-only unless a real multimodal dataset pipeline is added.
- [ ] Run: `python test_came_net.py`
  Expected: PASS with no stale references to ideal bivectors or strict end-to-end equivariance.

## Self-Review

- Spec coverage: MPE contract, GCA/GLN claims, equivariance loss math, training-time regularizer semantics, and doc alignment are all covered.
- Placeholder scan: no `TODO` or `TBD` remains.
- Type consistency: plan consistently uses `Multivector`, `GRADE_INDICES`, `equivariance_loss_efficient`, and `multivector_distance`.

## Notes

- This workspace is not a Git repository right now, so the plan omits commit steps.
- Highest-risk mistake: trying to preserve the manuscript's current `grade-1 point` statement while keeping the existing PGA conventions.
