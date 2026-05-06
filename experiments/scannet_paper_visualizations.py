from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .comparison_baselines import LabelPriorBaseline
from .scannet_comparison_experiment import (
    ScanNetComparisonConfig,
    _build_scannet_model,
    _compute_train_label_priors,
    apply_scannet_comparison_defaults,
)
from .scannet_multimodal_data import ScanNetSceneConfig, ScanNetSceneDataset
from .scannet_multimodal_experiment import _make_collate_fn, _resolve_device, _write_json
from .scannet_rigid_benchmark import (
    ScanNetRigidBenchmarkConfig,
    _apply_transform,
    _split_scannet_scene_indices,
    get_default_scannet_rigid_conditions,
)
from training.torch_runtime_compat import configure_torch_runtime_compat

configure_torch_runtime_compat()


@dataclass
class ScanNetPaperFigureConfig:
    primary_artifact_dir: str
    data_root: str
    output_root: Optional[str] = None
    comparison_artifact_dir: Optional[str] = None
    ablation_artifact_dir: Optional[str] = None
    scene_index: int = 0


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
        }
    )


def _condition_lookup() -> Dict[str, Dict[str, object]]:
    return {condition["name"]: condition for condition in get_default_scannet_rigid_conditions()}


def _condition_order_from_metrics(condition_metrics: Dict[str, Dict[str, float]]) -> List[str]:
    ordered = ["clean", "rot15", "rot30", "rot45", "trans0p1", "trans0p2", "trans0p3", "se3_mix"]
    return [name for name in ordered if name in condition_metrics]


def _format_labels(labels: Sequence[str], max_width: int = 28) -> str:
    if not labels:
        return "none"
    return textwrap.fill(", ".join(labels), width=max_width)


def _select_positive_labels(logits: torch.Tensor, label_vocabulary: Sequence[str]) -> List[str]:
    probs = torch.sigmoid(logits)
    mask = probs >= 0.5
    labels = [label_vocabulary[index] for index, active in enumerate(mask.tolist()) if active]
    if labels:
        return labels
    topk = min(3, probs.numel())
    top_indices = torch.topk(probs, k=topk).indices.tolist()
    return [label_vocabulary[index] for index in top_indices]


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> Dict[str, str]:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _artifact_output_dir(config: ScanNetPaperFigureConfig) -> Path:
    if config.output_root is not None:
        output_dir = Path(config.output_root)
    else:
        output_dir = Path(config.primary_artifact_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _build_runtime_defaults(payload: Dict[str, object]) -> Tuple[ScanNetRigidBenchmarkConfig, ScanNetComparisonConfig]:
    runtime_defaults = payload["config"]["runtime_defaults"]
    rigid_defaults = payload["config"]
    rigid_config = ScanNetRigidBenchmarkConfig(
        method=rigid_defaults["method"],
        data_root=rigid_defaults["data_root"],
        artifact_root=rigid_defaults["artifact_root"],
        num_points=rigid_defaults["num_points"],
        max_frames=rigid_defaults["max_frames"],
        frame_resize=rigid_defaults["frame_resize"],
        image_feature_size=rigid_defaults["image_feature_size"],
        max_text_tokens=rigid_defaults["max_text_tokens"],
        batch_size=rigid_defaults["batch_size"],
        num_epochs=rigid_defaults["num_epochs"],
        hidden_dim=rigid_defaults["hidden_dim"],
        num_layers=rigid_defaults["num_layers"],
        num_heads=rigid_defaults["num_heads"],
        learning_rate=rigid_defaults["learning_rate"],
        weight_decay=rigid_defaults["weight_decay"],
        aux_loss_weight=rigid_defaults["aux_loss_weight"],
        equiv_loss_weight=rigid_defaults["equiv_loss_weight"],
        equiv_warmup_steps=rigid_defaults["equiv_warmup_steps"],
        dropout=rigid_defaults["dropout"],
        device=rigid_defaults["device"],
        min_label_frequency=rigid_defaults["min_label_frequency"],
        top_k_labels=rigid_defaults["top_k_labels"],
        blind_holdout_fraction=rigid_defaults["blind_holdout_fraction"],
        use_blind_holdout=rigid_defaults["use_blind_holdout"],
        print_interval=rigid_defaults["print_interval"],
    )
    comparison_config = apply_scannet_comparison_defaults(ScanNetComparisonConfig(**runtime_defaults))
    return rigid_config, comparison_config


def _load_artifact_bundle(artifact_dir: str, data_root: str, device: torch.device) -> Dict[str, object]:
    artifact_path = Path(artifact_dir)
    payload = {
        "artifact_dir": artifact_path,
        "config": _load_json(artifact_path / "config.json"),
        "metrics": _load_json(artifact_path / "metrics.json"),
        "condition_metrics": _load_json(artifact_path / "condition_metrics.json"),
        "dataset_report": _load_json(artifact_path / "dataset_report.json"),
        "figure_metadata": _load_json(artifact_path / "figure_metadata.json"),
        "label_vocabulary": json.loads((artifact_path / "label_vocabulary.json").read_text(encoding="utf-8")),
        "test_scene_ids": json.loads((artifact_path / "test_scene_ids.json").read_text(encoding="utf-8")),
    }
    rigid_config, comparison_config = _build_runtime_defaults(payload)
    dataset = ScanNetSceneDataset(
        ScanNetSceneConfig(
            data_root=data_root,
            num_points=comparison_config.num_points,
            max_frames=comparison_config.max_frames,
            frame_resize=comparison_config.frame_resize,
            min_label_frequency=comparison_config.min_label_frequency,
            top_k_labels=rigid_config.top_k_labels,
            require_all_modalities=True,
            vocabulary_scene_ids=payload["dataset_report"]["vocabulary_source_scene_ids"],
        )
    )
    if comparison_config.method == "label_prior":
        train_indices, _, _, _ = _split_scannet_scene_indices(
            dataset,
            use_blind_holdout=rigid_config.use_blind_holdout,
            blind_holdout_fraction=rigid_config.blind_holdout_fraction,
        )
        label_priors = _compute_train_label_priors(dataset, train_indices)
        model = LabelPriorBaseline(label_priors).to(device)
    else:
        model = _build_scannet_model(comparison_config, label_count=len(dataset.label_vocabulary)).to(device)
        state_dict = torch.load(artifact_path / "checkpoints" / "best_model.pth", map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    payload.update(
        {
            "rigid_config": rigid_config,
            "comparison_config": comparison_config,
            "dataset": dataset,
            "model": model,
        }
    )
    return payload


def _load_scene_batch(bundle: Dict[str, object], scene_id: str, device: torch.device) -> Dict[str, object]:
    dataset: ScanNetSceneDataset = bundle["dataset"]
    comparison_config: ScanNetComparisonConfig = bundle["comparison_config"]
    scene_index = next(index for index, entry in enumerate(dataset.scenes) if entry["scene_id"] == scene_id)
    loader = DataLoader(
        Subset(dataset, [scene_index]),
        batch_size=1,
        shuffle=False,
        collate_fn=_make_collate_fn(comparison_config),
    )
    batch = next(iter(loader))
    return {
        "scene_id": scene_id,
        "point_coords": batch["point_coords"].to(device),
        "image_patches": batch["image_patches"].to(device),
        "text_tokens": batch["text_tokens"].to(device),
        "label_targets": batch["label_targets"].to(device),
        "labels": batch["labels"][0],
    }


def _predict_scene_conditions(
    bundle: Dict[str, object],
    scene_id: str,
    condition_names: Sequence[str],
    device: torch.device,
) -> Dict[str, Dict[str, object]]:
    model = bundle["model"]
    label_vocabulary = bundle["label_vocabulary"]
    batch = _load_scene_batch(bundle, scene_id, device)
    lookup = _condition_lookup()
    results = {}
    clean_logits = None
    for condition_name in condition_names:
        variant = lookup[condition_name]["variants"][0]
        transformed_coords = _apply_transform(batch["point_coords"], variant.get("rotation"), variant.get("translation"))
        with torch.no_grad():
            logits = model(
                point_coords=transformed_coords,
                image_patches=batch["image_patches"],
                text_tokens=batch["text_tokens"],
            ).detach().cpu()[0]
        if clean_logits is None:
            clean_logits = logits
        results[condition_name] = {
            "point_coords": transformed_coords.detach().cpu()[0],
            "logits": logits,
            "predicted_labels": _select_positive_labels(logits, label_vocabulary),
            "ground_truth": batch["labels"],
        }
    for condition_name in condition_names:
        logits = results[condition_name]["logits"]
        results[condition_name]["logit_drift"] = float(((logits - clean_logits) ** 2).mean().item())
        clean_pred = set(results["clean"]["predicted_labels"])
        cond_pred = set(results[condition_name]["predicted_labels"])
        results[condition_name]["flip"] = int(cond_pred != clean_pred)
        results[condition_name]["confidence"] = float(torch.sigmoid(logits).max().item())
    return results


def _plot_point_cloud(ax, point_coords: torch.Tensor, title: str, detail_text: str) -> None:
    coords = point_coords.numpy()
    colors = coords[:, 2]
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, cmap="viridis", s=7, alpha=0.9, linewidths=0)
    ax.view_init(elev=22, azim=35)
    ax.set_title(title, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.text2D(0.02, -0.18, detail_text, transform=ax.transAxes, fontsize=8, va="top")


def _render_main_hero_figure(
    output_dir: Path,
    primary_bundle: Dict[str, object],
    comparison_bundle: Optional[Dict[str, object]],
    ablation_bundle: Optional[Dict[str, object]],
    scene_id: str,
) -> Dict[str, str]:
    _paper_style()
    condition_names = ["clean", "rot30", "trans0p2", "se3_mix"]
    primary_predictions = _predict_scene_conditions(primary_bundle, scene_id, condition_names, torch.device("cpu"))
    comparison_predictions = (
        _predict_scene_conditions(comparison_bundle, scene_id, condition_names, torch.device("cpu"))
        if comparison_bundle is not None
        else None
    )

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.2], height_ratios=[1, 1], wspace=0.28, hspace=0.34)

    scatter_axes = [
        fig.add_subplot(gs[0, 0], projection="3d"),
        fig.add_subplot(gs[0, 1], projection="3d"),
        fig.add_subplot(gs[1, 0], projection="3d"),
        fig.add_subplot(gs[1, 1], projection="3d"),
    ]
    for axis, condition_name in zip(scatter_axes, condition_names):
        comparison_labels = comparison_predictions[condition_name]["predicted_labels"] if comparison_predictions is not None else []
        detail = (
            f"GT: {_format_labels(primary_predictions[condition_name]['ground_truth'])}\n"
            f"CAME-MM: {_format_labels(primary_predictions[condition_name]['predicted_labels'])}\n"
            f"{comparison_bundle['metrics']['method'] if comparison_bundle else 'Comparison'}: {_format_labels(comparison_labels)}"
        )
        _plot_point_cloud(
            axis,
            primary_predictions[condition_name]["point_coords"],
            condition_name.replace("trans", "trans ").replace("se3_mix", "SE(3) mix"),
            detail,
        )

    panel_b = fig.add_subplot(gs[0, 2])
    x_positions = np.arange(len(condition_names))
    primary_drifts = [primary_predictions[name]["logit_drift"] for name in condition_names]
    panel_b.plot(x_positions, primary_drifts, marker="o", linewidth=2.0, color="#1f77b4", label="CAME-MM drift")
    if comparison_predictions is not None:
        comparison_drifts = [comparison_predictions[name]["logit_drift"] for name in condition_names]
        panel_b.plot(x_positions, comparison_drifts, marker="s", linewidth=2.0, color="#d62728", label="Comparison drift")
    panel_b.set_xticks(x_positions)
    panel_b.set_xticklabels(condition_names, rotation=25, ha="right")
    panel_b.set_ylabel("Logit drift")
    panel_b.grid(True, axis="y", alpha=0.25)
    panel_b.set_title("Per-scene stability diagnostics")
    panel_b_twin = panel_b.twinx()
    panel_b_twin.plot(
        x_positions,
        [primary_predictions[name]["flip"] for name in condition_names],
        marker="^",
        linestyle="--",
        color="#2ca02c",
        label="CAME-MM flip",
    )
    if comparison_predictions is not None:
        panel_b_twin.plot(
            x_positions,
            [comparison_predictions[name]["flip"] for name in condition_names],
            marker="v",
            linestyle="--",
            color="#9467bd",
            label="Comparison flip",
        )
    panel_b_twin.set_ylim(-0.05, 1.05)
    panel_b_twin.set_ylabel("Flip status")
    handles_left, labels_left = panel_b.get_legend_handles_labels()
    handles_right, labels_right = panel_b_twin.get_legend_handles_labels()
    panel_b.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", frameon=False)

    panel_c_outer = gs[1, 2].subgridspec(3, 1, hspace=0.35)
    aggregate_methods = [primary_bundle["metrics"]["method"]]
    aggregate_metrics = [primary_bundle["metrics"]]
    colors = ["#1f77b4"]
    if comparison_bundle is not None:
        aggregate_methods.append(comparison_bundle["metrics"]["method"])
        aggregate_metrics.append(comparison_bundle["metrics"])
        colors.append("#d62728")
    if ablation_bundle is not None:
        aggregate_methods.append(ablation_bundle["metrics"]["method"])
        aggregate_metrics.append(ablation_bundle["metrics"])
        colors.append("#ff7f0e")
    metric_specs = [
        ("conditional_rigid_micro_f1", "Conditional rigid micro-F1"),
        ("clean_correct_flip_rate", "Clean-correct flip rate"),
        ("logit_drift", "Logit drift"),
    ]
    for row_index, (metric_key, metric_title) in enumerate(metric_specs):
        ax = fig.add_subplot(panel_c_outer[row_index, 0])
        values = [metrics[metric_key] for metrics in aggregate_metrics]
        ax.bar(np.arange(len(values)), values, color=colors, width=0.6)
        ax.set_title(metric_title, loc="left", fontsize=10)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(aggregate_methods, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.2)

    fig.text(0.02, 0.98, "A", fontsize=15, fontweight="bold", va="top")
    fig.text(0.69, 0.98, "B", fontsize=15, fontweight="bold", va="top")
    fig.text(0.69, 0.47, "C", fontsize=15, fontweight="bold", va="top")
    fig.suptitle(f"Rigid stability of multimodal predictions on scene {scene_id}", y=0.995)
    return _save_figure(fig, output_dir, "main_hero_equivariance")


def _render_stability_curves(
    output_dir: Path,
    primary_bundle: Dict[str, object],
    comparison_bundle: Optional[Dict[str, object]],
    ablation_bundle: Optional[Dict[str, object]],
) -> Dict[str, str]:
    _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    bundles = [primary_bundle]
    colors = ["#1f77b4"]
    if comparison_bundle is not None:
        bundles.append(comparison_bundle)
        colors.append("#d62728")
    if ablation_bundle is not None:
        bundles.append(ablation_bundle)
        colors.append("#ff7f0e")
    metric_axes = [
        ("micro_f1", "Condition micro-F1"),
        ("logit_drift", "Condition logit drift"),
    ]
    for axis, (metric_key, title) in zip(axes, metric_axes):
        for bundle, color in zip(bundles, colors):
            condition_metrics = bundle["condition_metrics"]
            ordered_names = _condition_order_from_metrics(condition_metrics)
            values = [condition_metrics[name][metric_key] for name in ordered_names]
            axis.plot(ordered_names, values, marker="o", linewidth=2.0, color=color, label=bundle["metrics"]["method"])
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "rigid_stability_curves")


def _render_ablation_mechanism(
    output_dir: Path,
    primary_bundle: Dict[str, object],
    ablation_bundle: Optional[Dict[str, object]],
    comparison_bundle: Optional[Dict[str, object]],
) -> Dict[str, str]:
    _paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    bundles = [primary_bundle]
    colors = ["#1f77b4"]
    if ablation_bundle is not None:
        bundles.append(ablation_bundle)
        colors.append("#ff7f0e")
    if comparison_bundle is not None:
        bundles.append(comparison_bundle)
        colors.append("#d62728")
    metric_specs = [
        ("conditional_rigid_micro_f1", "Conditional rigid micro-F1"),
        ("clean_correct_flip_rate", "Clean-correct flip rate"),
        ("logit_drift", "Logit drift"),
    ]
    method_names = [bundle["metrics"]["method"] for bundle in bundles]
    x_positions = np.arange(len(method_names))
    for axis, (metric_key, title) in zip(axes, metric_specs):
        values = [bundle["metrics"][metric_key] for bundle in bundles]
        axis.bar(x_positions, values, color=colors, width=0.62)
        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(method_names, rotation=20, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "ablation_mechanism")


def generate_scannet_paper_figures(config: ScanNetPaperFigureConfig) -> Dict[str, object]:
    device = _resolve_device("cpu")
    output_dir = _artifact_output_dir(config)
    primary_bundle = _load_artifact_bundle(config.primary_artifact_dir, config.data_root, device)
    comparison_bundle = (
        _load_artifact_bundle(config.comparison_artifact_dir, config.data_root, device)
        if config.comparison_artifact_dir is not None
        else None
    )
    ablation_bundle = (
        _load_artifact_bundle(config.ablation_artifact_dir, config.data_root, device)
        if config.ablation_artifact_dir is not None
        else None
    )

    test_scene_ids = primary_bundle["test_scene_ids"]
    if not test_scene_ids:
        raise ValueError("Primary artifact contains no test scene ids for paper visualization.")
    scene_id = test_scene_ids[min(config.scene_index, len(test_scene_ids) - 1)]

    outputs = {
        "main_hero_equivariance": _render_main_hero_figure(
            output_dir,
            primary_bundle,
            comparison_bundle,
            ablation_bundle,
            scene_id,
        ),
        "rigid_stability_curves": _render_stability_curves(
            output_dir,
            primary_bundle,
            comparison_bundle,
            ablation_bundle,
        ),
        "ablation_mechanism": _render_ablation_mechanism(
            output_dir,
            primary_bundle,
            ablation_bundle,
            comparison_bundle,
        ),
    }
    manifest = {
        "primary_artifact_dir": config.primary_artifact_dir,
        "comparison_artifact_dir": config.comparison_artifact_dir,
        "ablation_artifact_dir": config.ablation_artifact_dir,
        "scene_id": scene_id,
        "outputs": outputs,
    }
    _write_json(output_dir / "figure_manifest.json", manifest)
    return {"output_dir": output_dir, "scene_id": scene_id, "outputs": outputs}
