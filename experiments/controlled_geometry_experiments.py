"""
controlled_geometry_experiments.py - Lightweight geometry-first experiment helpers.
"""

from __future__ import annotations

import copy
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from method.came_net import count_parameters
from method.equiv_loss import multivector_distance
from method.gca import GeometricCliffordAttention
from method.pga_algebra import (
    GRADE_INDICES,
    Multivector,
    SCALAR_PART_TABLE,
    REVERSION_SIGNS,
    create_point_pga,
    extract_point_coordinates,
    random_motor,
    random_rotation,
    random_translation,
)


def list_ablation_variants() -> List[str]:
    return [
        "full",
        "no_gln",
        "no_soft_equiv_reg",
        "unconstrained_bivector",
        "scalar_only",
        "non_geometric_fusion",
    ]


class IdentityMultivectorModule(nn.Module):
    def forward(self, x: Multivector, *args, **kwargs) -> Multivector:
        del args, kwargs
        return x


class ScalarOnlyMPEWrapper(nn.Module):
    def __init__(self, base_mpe: nn.Module):
        super().__init__()
        self.base_mpe = base_mpe

    def forward(self, *args, **kwargs):
        encoded = self.base_mpe(*args, **kwargs)
        if isinstance(encoded, tuple):
            multivector, splits = encoded
            return _keep_scalar_only(multivector), splits
        return _keep_scalar_only(encoded)


class UnconstrainedBivectorPointMPEWrapper(nn.Module):
    def __init__(self, base_point_mpe: nn.Module):
        super().__init__()
        self.base_point_mpe = base_point_mpe

    def forward(self, coords: torch.Tensor, features: Optional[torch.Tensor] = None) -> Multivector:
        output = self.base_point_mpe(coords, features)
        data = output.data.clone()
        euclidean_bivectors = data[..., 5:8]
        data[..., 8:11] = euclidean_bivectors
        return Multivector(data)


class CoefficientDotProductAttention(nn.Module):
    def __init__(self, base_attention: GeometricCliffordAttention):
        super().__init__()
        self.multivector_dim = base_attention.multivector_dim
        self.num_heads = base_attention.num_heads
        self.scale = base_attention.scale
        self.dropout = base_attention.dropout
        self.grade_query_projs = base_attention.grade_query_projs
        self.grade_key_projs = base_attention.grade_key_projs
        self.grade_value_projs = base_attention.grade_value_projs
        self.grade_out_projs = base_attention.grade_out_projs

    def _project(self, x: Multivector, projections: nn.ModuleDict) -> torch.Tensor:
        batch_size, num_tokens, _ = x.data.shape
        projected = torch.zeros(
            batch_size,
            self.num_heads,
            num_tokens,
            self.multivector_dim,
            device=x.data.device,
            dtype=x.data.dtype,
        )

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            grade_data = x.data[..., indices]
            dim = len(indices)
            grade_proj = projections[str(grade)](grade_data)
            grade_proj = grade_proj.view(batch_size, num_tokens, self.num_heads, dim)
            projected[..., indices] = grade_proj.permute(0, 2, 1, 3)

        return projected

    def forward(
        self,
        x: Multivector,
        context: Optional[Multivector] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Multivector:
        context = x if context is None else context
        q_heads = self._project(x, self.grade_query_projs)
        k_heads = self._project(context, self.grade_key_projs)
        v_heads = self._project(context, self.grade_value_projs)

        attn_scores = torch.einsum("bhqi,bhki->bhqk", q_heads, k_heads)
        attn_scores = attn_scores * self.scale

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        head_outputs = torch.einsum("bhqk,bhki->bhqi", attn_weights, v_heads)

        batch_size, num_queries = x.data.shape[:2]
        output = torch.zeros(
            batch_size,
            num_queries,
            self.multivector_dim,
            device=x.data.device,
            dtype=x.data.dtype,
        )

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            dim = len(indices)
            grade_head = head_outputs[..., indices]
            grade_head = grade_head.permute(0, 2, 1, 3).reshape(batch_size, num_queries, dim * self.num_heads)
            output[..., indices] = self.grade_out_projs[str(grade)](grade_head)

        return Multivector(self.dropout(output))


class NormalizedGeometricAttention(nn.Module):
    def __init__(self, base_attention: GeometricCliffordAttention, eps: float = 1e-6):
        super().__init__()
        self.multivector_dim = base_attention.multivector_dim
        self.num_heads = base_attention.num_heads
        self.scale = base_attention.scale
        self.dropout = base_attention.dropout
        self.grade_query_projs = base_attention.grade_query_projs
        self.grade_key_projs = base_attention.grade_key_projs
        self.grade_value_projs = base_attention.grade_value_projs
        self.grade_out_projs = base_attention.grade_out_projs
        self.eps = eps

    def _project(self, x: Multivector, projections: nn.ModuleDict) -> torch.Tensor:
        batch_size, num_tokens, _ = x.data.shape
        projected = torch.zeros(
            batch_size,
            self.num_heads,
            num_tokens,
            self.multivector_dim,
            device=x.data.device,
            dtype=x.data.dtype,
        )

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            grade_data = x.data[..., indices]
            dim = len(indices)
            grade_proj = projections[str(grade)](grade_data)
            grade_proj = grade_proj.view(batch_size, num_tokens, self.num_heads, dim)
            projected[..., indices] = grade_proj.permute(0, 2, 1, 3)

        return projected

    def forward(
        self,
        x: Multivector,
        context: Optional[Multivector] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Multivector:
        context = x if context is None else context
        q_heads = self._project(x, self.grade_query_projs)
        k_heads = self._project(context, self.grade_key_projs)
        v_heads = self._project(context, self.grade_value_projs)

        reversion = REVERSION_SIGNS.to(device=x.data.device, dtype=x.data.dtype).view(1, 1, 1, -1)
        k_reversed = k_heads * reversion

        scalar_table = SCALAR_PART_TABLE.to(device=x.data.device, dtype=x.data.dtype)
        attn_scores = torch.einsum("bhqi,bhkj,ij->bhqk", q_heads, k_reversed, scalar_table)

        q_norm = torch.sqrt(torch.sum(q_heads ** 2, dim=-1).clamp_min(self.eps))
        k_norm = torch.sqrt(torch.sum(k_heads ** 2, dim=-1).clamp_min(self.eps))
        denom = q_norm.unsqueeze(-1) * k_norm.unsqueeze(-2)
        attn_scores = (attn_scores / denom.clamp_min(self.eps)) * self.scale

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        head_outputs = torch.einsum("bhqk,bhki->bhqi", attn_weights, v_heads)

        batch_size, num_queries = x.data.shape[:2]
        output = torch.zeros(
            batch_size,
            num_queries,
            self.multivector_dim,
            device=x.data.device,
            dtype=x.data.dtype,
        )

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            dim = len(indices)
            grade_head = head_outputs[..., indices]
            grade_head = grade_head.permute(0, 2, 1, 3).reshape(batch_size, num_queries, dim * self.num_heads)
            output[..., indices] = self.grade_out_projs[str(grade)](grade_head)

        return Multivector(self.dropout(output))


class MixedGeometricCoefficientAttention(nn.Module):
    def __init__(self, base_attention: GeometricCliffordAttention, alpha: float = 0.5):
        super().__init__()
        self.multivector_dim = base_attention.multivector_dim
        self.num_heads = base_attention.num_heads
        self.scale = base_attention.scale
        self.dropout = base_attention.dropout
        self.grade_query_projs = base_attention.grade_query_projs
        self.grade_key_projs = base_attention.grade_key_projs
        self.grade_value_projs = base_attention.grade_value_projs
        self.grade_out_projs = base_attention.grade_out_projs
        self.alpha = alpha

    def _project(self, x: Multivector, projections: nn.ModuleDict) -> torch.Tensor:
        batch_size, num_tokens, _ = x.data.shape
        projected = torch.zeros(
            batch_size,
            self.num_heads,
            num_tokens,
            self.multivector_dim,
            device=x.data.device,
            dtype=x.data.dtype,
        )

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            grade_data = x.data[..., indices]
            dim = len(indices)
            grade_proj = projections[str(grade)](grade_data)
            grade_proj = grade_proj.view(batch_size, num_tokens, self.num_heads, dim)
            projected[..., indices] = grade_proj.permute(0, 2, 1, 3)

        return projected

    def forward(
        self,
        x: Multivector,
        context: Optional[Multivector] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Multivector:
        context = x if context is None else context
        q_heads = self._project(x, self.grade_query_projs)
        k_heads = self._project(context, self.grade_key_projs)
        v_heads = self._project(context, self.grade_value_projs)

        reversion = REVERSION_SIGNS.to(device=x.data.device, dtype=x.data.dtype).view(1, 1, 1, -1)
        k_reversed = k_heads * reversion
        scalar_table = SCALAR_PART_TABLE.to(device=x.data.device, dtype=x.data.dtype)

        geometric_scores = torch.einsum("bhqi,bhkj,ij->bhqk", q_heads, k_reversed, scalar_table)
        coefficient_scores = torch.einsum("bhqi,bhki->bhqk", q_heads, k_heads)
        attn_scores = (self.alpha * geometric_scores + (1.0 - self.alpha) * coefficient_scores) * self.scale

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        head_outputs = torch.einsum("bhqk,bhki->bhqi", attn_weights, v_heads)

        batch_size, num_queries = x.data.shape[:2]
        output = torch.zeros(
            batch_size,
            num_queries,
            self.multivector_dim,
            device=x.data.device,
            dtype=x.data.dtype,
        )

        for grade, indices in GRADE_INDICES.items():
            if not indices:
                continue
            dim = len(indices)
            grade_head = head_outputs[..., indices]
            grade_head = grade_head.permute(0, 2, 1, 3).reshape(batch_size, num_queries, dim * self.num_heads)
            output[..., indices] = self.grade_out_projs[str(grade)](grade_head)

        return Multivector(self.dropout(output))


def _keep_scalar_only(multivector: Multivector) -> Multivector:
    data = torch.zeros_like(multivector.data)
    scalar_indices = GRADE_INDICES[0]
    data[..., scalar_indices] = multivector.data[..., scalar_indices]
    return Multivector(data)


def _scalar_score_matrix(tokens: Multivector) -> torch.Tensor:
    token_data = tokens.data
    reverse_signs = REVERSION_SIGNS.to(device=token_data.device, dtype=token_data.dtype).view(1, 1, -1)
    reversed_tokens = token_data * reverse_signs
    scalar_table = SCALAR_PART_TABLE.to(device=token_data.device, dtype=token_data.dtype)
    return torch.einsum("bqi,bkj,ij->bqk", token_data, reversed_tokens, scalar_table)


def run_gca_score_invariance_check(
    token_multivectors: Multivector,
    motor: Optional[Multivector] = None,
) -> Dict[str, float]:
    if token_multivectors.data.ndim != 3:
        raise ValueError("run_gca_score_invariance_check expects token multivectors of shape (B, N, 16)")

    batch_size = token_multivectors.data.shape[0]
    motor = motor or random_motor(
        batch_size=batch_size,
        sigma_rot=0.3,
        sigma_trans=0.3,
        device=token_multivectors.data.device,
        dtype=token_multivectors.data.dtype,
    )

    original_scores = _scalar_score_matrix(token_multivectors)
    transformed_scores = _scalar_score_matrix(token_multivectors.apply_motor(motor))
    deltas = (original_scores - transformed_scores).abs()

    return {
        "max_abs_score_delta": float(deltas.max().item()),
        "mean_abs_score_delta": float(deltas.mean().item()),
    }


def _equivariance_error_for_motor(model, point_coords: torch.Tensor, motor: Multivector) -> torch.Tensor:
    original_latent = model.get_latent_multivector(point_coords=point_coords)
    transformed_coords = extract_point_coordinates(create_point_pga(point_coords).apply_motor(motor))
    transformed_latent = model.get_latent_multivector(point_coords=transformed_coords)
    target_latent = original_latent.apply_motor(motor)
    return multivector_distance(transformed_latent, target_latent)


def run_equivariance_curve_experiment(
    *,
    model,
    point_coords: torch.Tensor,
    rotation_values: Sequence[float],
    translation_values: Sequence[float],
) -> Dict[str, List[Dict[str, float]]]:
    if point_coords.ndim != 3:
        raise ValueError("point_coords must have shape (B, N, 3)")

    device = point_coords.device
    dtype = point_coords.dtype
    batch_size = point_coords.shape[0]

    model_was_training = model.training
    model.eval()
    try:
        rotation_curve = []
        for value in rotation_values:
            motor = random_rotation(batch_size=batch_size, sigma=float(value), device=device, dtype=dtype)
            error = _equivariance_error_for_motor(model, point_coords, motor)
            rotation_curve.append({"value": float(value), "error": float(error.item())})

        translation_curve = []
        for value in translation_values:
            motor = random_translation(batch_size=batch_size, sigma=float(value), device=device, dtype=dtype)
            error = _equivariance_error_for_motor(model, point_coords, motor)
            translation_curve.append({"value": float(value), "error": float(error.item())})
    finally:
        if model_was_training:
            model.train()

    return {
        "rotation_curve": rotation_curve,
        "translation_curve": translation_curve,
    }


def build_ablated_model(model: nn.Module, variant: str) -> tuple[nn.Module, str]:
    if variant not in list_ablation_variants():
        raise ValueError(f"Unknown ablation variant: {variant}")

    variant_model = copy.deepcopy(model)
    notes = {
        "full": "Reference CAME-Net configuration.",
        "no_gln": "Replaces grade-wise normalization with identity maps.",
        "no_soft_equiv_reg": "Training-only ablation; forward metrics proxy the full model unless retrained.",
        "unconstrained_bivector": "Copies Euclidean bivector coefficients into ideal bivector slots.",
        "scalar_only": "Zeros all non-scalar multivector components before the geometric core.",
        "non_geometric_fusion": "Replaces geometric scalar scoring with coefficient-space dot-product attention.",
    }[variant]

    if variant == "no_gln":
        for layer in variant_model.came_layers:
            layer.norm1 = IdentityMultivectorModule()
            layer.norm2 = IdentityMultivectorModule()
    elif variant == "unconstrained_bivector":
        variant_model.mpe.point_mpe = UnconstrainedBivectorPointMPEWrapper(variant_model.mpe.point_mpe)
    elif variant == "scalar_only":
        variant_model.mpe = ScalarOnlyMPEWrapper(variant_model.mpe)
    elif variant == "non_geometric_fusion":
        for layer in variant_model.came_layers:
            layer.attention = CoefficientDotProductAttention(layer.attention)

    variant_model.eval()
    return variant_model, notes


def _estimate_runtime_seconds(model: nn.Module, point_coords: torch.Tensor, repeats: int = 2) -> float:
    if point_coords.ndim != 3:
        raise ValueError("point_coords must have shape (B, N, 3)")

    runtimes: List[float] = []
    with torch.no_grad():
        for _ in range(max(1, repeats)):
            if point_coords.device.type == "cuda":
                torch.cuda.synchronize(point_coords.device)
            start_time = time.perf_counter()
            _ = model.get_latent_multivector(point_coords=point_coords)
            if point_coords.device.type == "cuda":
                torch.cuda.synchronize(point_coords.device)
            runtimes.append(time.perf_counter() - start_time)
    return float(sum(runtimes) / max(1, len(runtimes)))


def _compute_batch_task_accuracy(
    model: nn.Module,
    point_coords: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    with torch.no_grad():
        logits = model(point_coords=point_coords)
        predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item() * 100.0)


def _compute_loader_task_accuracy(
    model: nn.Module,
    evaluation_loader,
    device: torch.device,
    max_eval_batches: int = 2,
) -> Optional[float]:
    if evaluation_loader is None:
        return None

    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(evaluation_loader):
            if batch_index >= max(1, max_eval_batches):
                break
            point_coords = batch["point_coords"].to(device)
            labels = batch["labels"].to(device)
            logits = model(point_coords=point_coords)
            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_examples += int(labels.numel())

    if total_examples == 0:
        return None
    return float(100.0 * total_correct / total_examples)


def _mean_equivariance_error(
    model: nn.Module,
    point_coords: torch.Tensor,
    num_motors: int = 2,
    sigma_rot: float = 0.2,
    sigma_trans: float = 0.2,
) -> float:
    device = point_coords.device
    dtype = point_coords.dtype
    batch_size = point_coords.shape[0]

    errors: List[float] = []
    with torch.no_grad():
        for _ in range(max(1, num_motors)):
            motor = random_motor(
                batch_size=batch_size,
                sigma_rot=sigma_rot,
                sigma_trans=sigma_trans,
                device=device,
                dtype=dtype,
            )
            error = _equivariance_error_for_motor(model, point_coords, motor)
            errors.append(float(error.item()))
    return float(sum(errors) / max(1, len(errors)))


def run_ablation_suite(
    *,
    model: nn.Module,
    point_coords: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    variants: Optional[Sequence[str]] = None,
    evaluation_loader=None,
    device: Optional[torch.device] = None,
    max_eval_batches: int = 2,
) -> List[Dict[str, object]]:
    device = device or point_coords.device
    variant_names = list(variants) if variants is not None else list_ablation_variants()
    results: List[Dict[str, object]] = []

    for variant in variant_names:
        try:
            variant_model, notes = build_ablated_model(model, variant)
            variant_model.to(device)
            variant_model.eval()

            task_accuracy: Optional[float]
            if evaluation_loader is not None:
                task_accuracy = _compute_loader_task_accuracy(
                    variant_model,
                    evaluation_loader=evaluation_loader,
                    device=device,
                    max_eval_batches=max_eval_batches,
                )
            elif labels is not None:
                task_accuracy = _compute_batch_task_accuracy(variant_model, point_coords, labels)
            else:
                task_accuracy = None

            status = "proxy" if variant == "no_soft_equiv_reg" else "ok"
            result = {
                "variant": variant,
                "status": status,
                "task_accuracy": task_accuracy,
                "mean_equivariance_error": _mean_equivariance_error(variant_model, point_coords),
                "runtime_seconds": _estimate_runtime_seconds(variant_model, point_coords),
                "parameter_count": int(count_parameters(variant_model)),
                "notes": notes,
            }
        except Exception as exc:
            result = {
                "variant": variant,
                "status": "error",
                "task_accuracy": None,
                "mean_equivariance_error": None,
                "runtime_seconds": None,
                "parameter_count": None,
                "notes": str(exc),
            }

        results.append(result)

    return results


def write_geometry_suite_outputs(
    *,
    artifact_dir: Path,
    invariance_result: Dict[str, float],
    curve_result: Dict[str, List[Dict[str, float]]],
    ablation_variants: Sequence[str],
    ablation_metrics: Optional[Sequence[Dict[str, object]]] = None,
) -> None:
    geometry_dir = artifact_dir / "geometry"
    ablation_dir = artifact_dir / "ablation"
    tables_dir = artifact_dir / "tables"
    geometry_dir.mkdir(exist_ok=True)
    ablation_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    if ablation_metrics is None:
        ablation_metrics = [
            {
                "variant": variant,
                "status": "registered",
                "task_accuracy": None,
                "mean_equivariance_error": None,
                "runtime_seconds": None,
                "parameter_count": None,
                "notes": "Variant registered but not evaluated in this run.",
            }
            for variant in ablation_variants
        ]

    (geometry_dir / "gca_score_invariance.json").write_text(
        json.dumps(invariance_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (geometry_dir / "equivariance_curves.json").write_text(
        json.dumps(curve_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_curve_csv(geometry_dir / "equivariance_rotation_curve.csv", curve_result["rotation_curve"])
    _write_curve_csv(geometry_dir / "equivariance_translation_curve.csv", curve_result["translation_curve"])

    (ablation_dir / "ablation_variants.json").write_text(
        json.dumps({"variants": list(ablation_variants)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (ablation_dir / "ablation_metrics.json").write_text(
        json.dumps({"variants": list(ablation_metrics)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rotation_path = geometry_dir / "equivariance_rotation_curve.png"
    _plot_curve(
        curve_result["rotation_curve"],
        output_path=rotation_path,
        title="Equivariance Error vs Rotation Scale",
        x_label="Rotation scale",
    )
    translation_path = geometry_dir / "equivariance_translation_curve.png"
    _plot_curve(
        curve_result["translation_curve"],
        output_path=translation_path,
        title="Equivariance Error vs Translation Scale",
        x_label="Translation scale",
    )

    _write_ablation_table(tables_dir / "ablation_table.csv", ablation_metrics)
    _plot_ablation_tradeoff(ablation_metrics, ablation_dir / "ablation_tradeoff_plot.png")


def _plot_curve(curve: Sequence[Dict[str, float]], *, output_path: Path, title: str, x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [entry["value"] for entry in curve]
    ys = [entry["error"] for entry in curve]
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Equivariance error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_curve_csv(output_path: Path, curve: Sequence[Dict[str, float]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["value", "error"])
        for entry in curve:
            writer.writerow([entry["value"], entry["error"]])


def _write_ablation_table(output_path: Path, ablation_metrics: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "variant",
        "status",
        "task_accuracy",
        "mean_equivariance_error",
        "runtime_seconds",
        "parameter_count",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in ablation_metrics:
            writer.writerow({field: _csv_value(metric.get(field)) for field in fieldnames})


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _plot_ablation_tradeoff(ablation_metrics: Sequence[Dict[str, object]], output_path: Path) -> None:
    valid_metrics = [
        metric
        for metric in ablation_metrics
        if metric.get("runtime_seconds") is not None and metric.get("mean_equivariance_error") is not None
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))

    if not valid_metrics:
        ax.text(0.5, 0.5, "No ablation metrics available", ha="center", va="center")
        ax.set_axis_off()
    else:
        xs = [float(metric["runtime_seconds"]) for metric in valid_metrics]
        ys = [float(metric["mean_equivariance_error"]) for metric in valid_metrics]
        colors = []
        for metric in valid_metrics:
            task_accuracy = metric.get("task_accuracy")
            colors.append(float(task_accuracy) if task_accuracy is not None else 0.0)

        scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", s=50)
        fig.colorbar(scatter, ax=ax, label="Task accuracy")
        for metric, x_coord, y_coord in zip(valid_metrics, xs, ys):
            ax.annotate(str(metric["variant"]), (x_coord, y_coord), textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel("Mean equivariance error")
        ax.set_title("Ablation Tradeoff")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
