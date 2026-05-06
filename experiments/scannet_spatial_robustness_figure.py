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
from matplotlib import colors as mcolors

from .scannet_qualitative_figure import (
    _build_render_scene,
    _get_scene_entry,
    _object_semantic_colors,
    _paper_style,
    _prepare_prediction_bundle,
    _pretty_method_name,
    _run_single_scene_prediction,
    _sample_model_points,
    _wrap_label_text,
)
from .scannet_rigid_benchmark import _apply_transform, get_default_scannet_rigid_conditions
from .scannet_multimodal_experiment import _resolve_device, _write_json


@dataclass
class ScanNetSpatialRobustnessFigureConfig:
    data_root: str
    came_ckpt: str
    baseline_ckpt: str
    baseline_method: str
    scene_ids: Sequence[str]
    transform_variants: Sequence[str]
    output: str
    came_method: str = "came"
    device: Optional[str] = None
    render_num_points: int = 12000
    top_k_predictions: int = 6
    camera_azim: float = 35.0
    camera_elev: float = 20.0


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_lookup() -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    for condition in get_default_scannet_rigid_conditions():
        for variant in condition["variants"]:
            lookup[str(variant["name"])] = {
                "condition_name": condition["name"],
                "variant_name": variant["name"],
                "rotation": variant.get("rotation"),
                "translation": variant.get("translation"),
            }
    return lookup


def _variant_pretty_name(name: str) -> str:
    if name.startswith("rot_"):
        _, axis, degrees = name.split("_")
        return f"Rotate {axis.upper()} {degrees.replace('p', '.')}°"
    if name.startswith("t"):
        axis = name[1]
        magnitude = name.split("_", 1)[1].replace("p", ".")
        return f"Translate {axis.upper()} +{magnitude}"
    if name.startswith("se3_"):
        axis = name.split("_", 1)[1].upper()
        return f"SE(3) mix on {axis}"
    return name.replace("_", " ")


def _project_points(coords: np.ndarray, azim: float, elev: float, *, recenter: bool) -> Tuple[np.ndarray, np.ndarray]:
    azim_rad = np.deg2rad(azim)
    elev_rad = np.deg2rad(elev)
    rz = np.asarray(
        [
            [np.cos(azim_rad), -np.sin(azim_rad), 0.0],
            [np.sin(azim_rad), np.cos(azim_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elev_rad), -np.sin(elev_rad)],
            [0.0, np.sin(elev_rad), np.cos(elev_rad)],
        ],
        dtype=np.float32,
    )
    rotated = coords @ rz.T @ rx.T
    xy = rotated[:, :2]
    if recenter:
        xy = xy - xy.mean(axis=0, keepdims=True)
    scale = float(np.abs(xy).max())
    if scale > 0:
        xy = xy / max(scale, 1.0)
    return xy, rotated[:, 2]


def _set_cloud_axis_style(ax, title: str, lim: float = 1.15) -> None:
    ax.set_title(title, loc="left", pad=4, fontsize=9.5, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    ax.patch.set_edgecolor("#E3E7EE")
    ax.patch.set_linewidth(0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_cloud(ax, coords: np.ndarray, colors: np.ndarray, title: str, azim: float, elev: float, *, recenter: bool) -> None:
    projected, depth = _project_points(coords, azim=azim, elev=elev, recenter=recenter)
    order = np.argsort(depth)
    ax.scatter(
        projected[order, 0],
        projected[order, 1],
        c=colors[order],
        s=4.8,
        alpha=1.0,
        linewidths=0,
        rasterized=True,
    )
    _set_cloud_axis_style(ax, title)


def _uniform_cloud_colors(count: int, hex_color: str = "#A7B0C1") -> np.ndarray:
    color = np.asarray(mcolors.to_rgb(hex_color), dtype=np.float32)
    return np.tile(color[None, :], (count, 1))


def _apply_transform_np(
    coords: np.ndarray,
    rotation: Optional[torch.Tensor],
    translation: Optional[torch.Tensor],
) -> np.ndarray:
    tensor = torch.tensor(coords, dtype=torch.float32)
    transformed = _apply_transform(tensor, rotation, translation)
    return transformed.detach().cpu().numpy()


def _compute_prediction_stats(original_logits: torch.Tensor, transformed_logits: torch.Tensor) -> Dict[str, float]:
    original_probs = torch.sigmoid(original_logits)
    transformed_probs = torch.sigmoid(transformed_logits)
    original_preds = original_probs >= 0.5
    transformed_preds = transformed_probs >= 0.5
    prediction_drift = float(((transformed_probs - original_probs) ** 2).mean().item())
    logit_drift = float(((transformed_logits - original_logits) ** 2).mean().item())
    prediction_changed = bool((original_preds != transformed_preds).any().item())
    agreement = float((original_preds == transformed_preds).float().mean().item())
    return {
        "prediction_drift": prediction_drift,
        "logit_drift": logit_drift,
        "prediction_changed": prediction_changed,
        "prediction_agreement": agreement,
    }


def _format_drift(value: float) -> str:
    if value == 0.0:
        return "0.0"
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def _draw_label_footer(
    ax,
    labels: Sequence[str],
    ground_truth: Sequence[str],
    *,
    drift: Optional[float] = None,
    drift_state: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> None:
    ax.axis("off")
    ax.set_facecolor("white")
    predicted = set(labels)
    truth = set(ground_truth)
    blocks = [
        ("Correct", sorted(predicted & truth), "#1B7837"),
        ("Missed", sorted(truth - predicted), "#B2182B"),
        ("Extra", sorted(predicted - truth), "#E08214"),
    ]
    x_positions = [0.02, 0.35, 0.68]
    for (title, values, color), x in zip(blocks, x_positions):
        ax.text(
            x,
            0.05,
            f"{title}\n{_wrap_label_text(values, width=18)}",
            transform=ax.transAxes,
            fontsize=7.0,
            color=color,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.9,
                "alpha": 0.98,
            },
        )
    if drift is not None:
        drift_color = "#B2182B" if drift_state == "up" else "#1B7837"
        drift_prefix = "Drift ↑" if drift_state == "up" else "Drift ↓"
        ax.text(
            0.02,
            0.96,
            f"{drift_prefix} {_format_drift(drift)}",
            transform=ax.transAxes,
            fontsize=7.2,
            color=drift_color,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "white",
                "edgecolor": drift_color,
                "linewidth": 0.9,
            },
        )
    elif subtitle:
        ax.text(
            0.02,
            0.96,
            subtitle,
            transform=ax.transAxes,
            fontsize=7.2,
            color="#4B5563",
            va="top",
            ha="left",
        )


def _draw_input_footer(ax, scene_id: str, labels: Sequence[str]) -> None:
    ax.axis("off")
    ax.set_facecolor("white")
    ax.text(
        0.02,
        0.92,
        f"Scene: {scene_id}",
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight="bold",
        va="top",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#D0D4DD", "linewidth": 0.85},
    )
    ax.text(
        0.02,
        0.08,
        "GT: " + _wrap_label_text(labels, width=24),
        transform=ax.transAxes,
        fontsize=7.0,
        color="#1B7837",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#F7FBF7", "edgecolor": "#1B7837", "linewidth": 0.9},
    )


def _draw_transform_footer(ax, variant_name: str, condition_name: str) -> None:
    ax.axis("off")
    ax.set_facecolor("white")
    ax.text(
        0.02,
        0.90,
        f"Condition: {condition_name}",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#4B5563",
        va="top",
    )
    ax.text(
        0.02,
        0.10,
        _variant_pretty_name(variant_name),
        transform=ax.transAxes,
        fontsize=7.4,
        color="#2F3B52",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "#FBFBFD", "edgecolor": "#D0D4DD", "linewidth": 0.9},
    )


def generate_scannet_spatial_robustness_figure(config: ScanNetSpatialRobustnessFigureConfig) -> Dict[str, object]:
    _paper_style()
    if not config.scene_ids:
        raise ValueError("At least one scene id is required.")
    if len(config.scene_ids) != len(config.transform_variants):
        raise ValueError("scene_ids and transform_variants must have the same length.")

    variant_lookup = _variant_lookup()
    device = _resolve_device(config.device or "cpu")
    came_bundle = _prepare_prediction_bundle(config.came_ckpt, config.came_method, config.data_root, device)
    baseline_bundle = _prepare_prediction_bundle(config.baseline_ckpt, config.baseline_method, config.data_root, device)

    rows: List[Dict[str, object]] = []
    for scene_id, variant_name in zip(config.scene_ids, config.transform_variants):
        if variant_name not in variant_lookup:
            raise ValueError(f"Unknown transform variant: {variant_name}")
        variant = variant_lookup[variant_name]
        came_scene = _get_scene_entry(came_bundle["dataset"], scene_id)
        baseline_scene = _get_scene_entry(baseline_bundle["dataset"], scene_id)
        render_scene = _build_render_scene(came_scene, config.render_num_points)

        render_coords = render_scene["render_coords"]
        transformed_render_coords = _apply_transform_np(render_coords, variant.get("rotation"), variant.get("translation"))
        render_labels = render_scene["render_object_labels"]

        came_model_coords, _ = _sample_model_points(render_scene, came_bundle["config"].num_points)
        baseline_model_coords, _ = _sample_model_points(render_scene, baseline_bundle["config"].num_points)
        transformed_came_model_coords = _apply_transform_np(came_model_coords, variant.get("rotation"), variant.get("translation"))
        transformed_baseline_model_coords = _apply_transform_np(baseline_model_coords, variant.get("rotation"), variant.get("translation"))

        came_original = _run_single_scene_prediction(
            came_bundle,
            came_scene,
            torch.tensor(came_model_coords, dtype=torch.float32),
            device,
            config.top_k_predictions,
        )
        came_transformed = _run_single_scene_prediction(
            came_bundle,
            came_scene,
            torch.tensor(transformed_came_model_coords, dtype=torch.float32),
            device,
            config.top_k_predictions,
        )
        baseline_original = _run_single_scene_prediction(
            baseline_bundle,
            baseline_scene,
            torch.tensor(baseline_model_coords, dtype=torch.float32),
            device,
            config.top_k_predictions,
        )
        baseline_transformed = _run_single_scene_prediction(
            baseline_bundle,
            baseline_scene,
            torch.tensor(transformed_baseline_model_coords, dtype=torch.float32),
            device,
            config.top_k_predictions,
        )

        came_stats = _compute_prediction_stats(came_original["logits"], came_transformed["logits"])
        baseline_stats = _compute_prediction_stats(baseline_original["logits"], baseline_transformed["logits"])
        baseline_drift_state = "up" if baseline_stats["prediction_drift"] >= came_stats["prediction_drift"] else "down"
        came_drift_state = "down" if came_stats["prediction_drift"] <= baseline_stats["prediction_drift"] else "up"

        rows.append(
            {
                "scene_id": scene_id,
                "ground_truth": list(came_original["ground_truth"]),
                "variant_name": variant_name,
                "condition_name": str(variant["condition_name"]),
                "render_coords": render_coords,
                "transformed_render_coords": transformed_render_coords,
                "render_labels": render_labels,
                "baseline_original": baseline_original,
                "baseline_transformed": baseline_transformed,
                "came_original": came_original,
                "came_transformed": came_transformed,
                "baseline_stats": baseline_stats,
                "came_stats": came_stats,
                "baseline_drift_state": baseline_drift_state,
                "came_drift_state": came_drift_state,
            }
        )

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(20.0, max(1, len(rows)) * 4.0))
    outer = fig.add_gridspec(len(rows), 1, hspace=0.15)

    for row_index, row in enumerate(rows):
        row_gs = outer[row_index].subgridspec(2, 6, height_ratios=[0.83, 0.17], wspace=0.06, hspace=0.02)
        orig_ax = fig.add_subplot(row_gs[0, 0])
        trans_ax = fig.add_subplot(row_gs[0, 1])
        base_orig_ax = fig.add_subplot(row_gs[0, 2])
        base_trans_ax = fig.add_subplot(row_gs[0, 3])
        came_orig_ax = fig.add_subplot(row_gs[0, 4])
        came_trans_ax = fig.add_subplot(row_gs[0, 5])

        orig_footer = fig.add_subplot(row_gs[1, 0])
        trans_footer = fig.add_subplot(row_gs[1, 1])
        base_orig_footer = fig.add_subplot(row_gs[1, 2])
        base_trans_footer = fig.add_subplot(row_gs[1, 3])
        came_orig_footer = fig.add_subplot(row_gs[1, 4])
        came_trans_footer = fig.add_subplot(row_gs[1, 5])

        _plot_cloud(
            orig_ax,
            row["render_coords"],
            _object_semantic_colors(row["render_labels"]),
            "Original point cloud",
            config.camera_azim,
            config.camera_elev,
            recenter=False,
        )
        _plot_cloud(
            trans_ax,
            row["transformed_render_coords"],
            _object_semantic_colors(row["render_labels"]),
            "Transformed point cloud",
            config.camera_azim,
            config.camera_elev,
            recenter=False,
        )
        neutral_colors = _uniform_cloud_colors(row["render_coords"].shape[0])
        _plot_cloud(
            base_orig_ax,
            row["render_coords"],
            neutral_colors,
            f"{_pretty_method_name(config.baseline_method)} / original",
            config.camera_azim,
            config.camera_elev,
            recenter=False,
        )
        _plot_cloud(
            base_trans_ax,
            row["transformed_render_coords"],
            neutral_colors,
            f"{_pretty_method_name(config.baseline_method)} / transformed",
            config.camera_azim,
            config.camera_elev,
            recenter=False,
        )
        _plot_cloud(
            came_orig_ax,
            row["render_coords"],
            neutral_colors,
            "CAME-Net / original",
            config.camera_azim,
            config.camera_elev,
            recenter=False,
        )
        _plot_cloud(
            came_trans_ax,
            row["transformed_render_coords"],
            neutral_colors,
            "CAME-Net / transformed",
            config.camera_azim,
            config.camera_elev,
            recenter=False,
        )

        _draw_input_footer(orig_footer, row["scene_id"], row["ground_truth"])
        _draw_transform_footer(trans_footer, row["variant_name"], row["condition_name"])
        _draw_label_footer(base_orig_footer, row["baseline_original"]["predicted_labels"], row["ground_truth"], subtitle="Prediction on original")
        _draw_label_footer(
            base_trans_footer,
            row["baseline_transformed"]["predicted_labels"],
            row["ground_truth"],
            drift=row["baseline_stats"]["prediction_drift"],
            drift_state=row["baseline_drift_state"],
        )
        _draw_label_footer(came_orig_footer, row["came_original"]["predicted_labels"], row["ground_truth"], subtitle="Prediction on original")
        _draw_label_footer(
            came_trans_footer,
            row["came_transformed"]["predicted_labels"],
            row["ground_truth"],
            drift=row["came_stats"]["prediction_drift"],
            drift_state=row["came_drift_state"],
        )

    fig.suptitle("Robustness Under Spatial Transformations", y=0.992, x=0.5, fontsize=21, fontweight="bold")
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=260, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=260, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "data_root": config.data_root,
        "came_ckpt": config.came_ckpt,
        "baseline_ckpt": config.baseline_ckpt,
        "came_method": config.came_method,
        "baseline_method": config.baseline_method,
        "scene_ids": list(config.scene_ids),
        "transform_variants": list(config.transform_variants),
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    _write_json(output_path.with_name(output_path.stem + "_manifest.json"), manifest)
    return {"png": str(png_path), "pdf": str(pdf_path), "manifest": manifest}
