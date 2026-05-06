from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors

from .scannet_qualitative_figure import (
    _build_render_scene,
    _extract_frame_image,
    _get_scene_entry,
    _object_semantic_colors,
    _paper_style,
    _prepare_prediction_bundle,
    _project_points,
    _sample_indices,
    _wrap_label_text,
)
from .scannet_multimodal_experiment import _resolve_device, _write_json


@dataclass
class ScanNetPointRelevanceFigureConfig:
    data_root: str
    came_ckpt: str
    baseline_ckpt: str
    baseline_method: str
    scene_ids: Sequence[str]
    queries: Sequence[str]
    output: str
    came_method: str = "came"
    device: Optional[str] = None
    render_num_points: int = 12000
    frame_index: int = 0
    camera_azim: float = 35.0
    camera_elev: float = 20.0


def _set_panel_style(ax, title: str) -> None:
    ax.set_title(title, loc="left", pad=4, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    ax.patch.set_edgecolor("#E3E7EE")
    ax.patch.set_linewidth(0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_cloud(ax, coords: np.ndarray, colors: np.ndarray, title: str, azim: float, elev: float) -> None:
    projected, depth = _project_points(coords, azim=azim, elev=elev)
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
    _set_panel_style(ax, title)


def _blend_relevance_colors(weights: np.ndarray) -> np.ndarray:
    base = np.tile(np.asarray(mcolors.to_rgb("#C5CBD6"), dtype=np.float32)[None, :], (weights.shape[0], 1))
    cmap = matplotlib.colormaps["magma"]
    mapped = np.asarray([cmap(float(weight))[:3] for weight in weights], dtype=np.float32)
    blend = 0.12 + 0.88 * weights[:, None]
    return base * (1.0 - blend) + mapped * blend


def _normalize_saliency(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmax = float(np.quantile(values, 0.98))
    if vmax <= 0:
        return np.zeros_like(values, dtype=np.float32)
    normalized = np.clip(values / vmax, 0.0, 1.0)
    return normalized.astype(np.float32)


def _build_single_scene_batch(bundle: Dict[str, object], scene_entry: Dict[str, object], point_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
    dataset = bundle["dataset"]
    example = {
        "scene_id": scene_entry["scene_id"],
        "point_coords": point_coords.detach().cpu(),
        "image_tensor": dataset._load_image_tensor(scene_entry["scene_dir"]),
        "text_prompt": scene_entry["text_prompt"],
        "label_targets": scene_entry["label_targets"].clone(),
        "labels": list(scene_entry["labels"]),
    }
    return bundle["collate_fn"]([example])


def _compute_point_saliency(
    bundle: Dict[str, object],
    scene_entry: Dict[str, object],
    point_coords: torch.Tensor,
    query_label: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    label_to_index = {label: idx for idx, label in enumerate(bundle["dataset"].label_vocabulary)}
    if query_label not in label_to_index:
        raise ValueError(f"Query label {query_label} is not in the current label vocabulary.")
    batch = _build_single_scene_batch(bundle, scene_entry, point_coords)
    coords = batch["point_coords"].to(device).clone().detach().requires_grad_(True)
    model = bundle["model"]
    model.zero_grad(set_to_none=True)
    logits = model(
        point_coords=coords,
        image_patches=batch["image_patches"].to(device),
        text_tokens=batch["text_tokens"].to(device),
    )
    target_logit = logits[0, label_to_index[query_label]]
    target_logit.backward()
    grads = coords.grad.detach()[0]
    saliency = grads.norm(dim=-1)
    return saliency.cpu(), logits.detach().cpu()[0]


def _map_saliency_to_render_points(render_count: int, sampled_indices: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    weights = np.zeros(render_count, dtype=np.float32)
    for index, value in zip(sampled_indices.tolist(), saliency.tolist()):
        weights[index] = max(weights[index], float(value))
    return _normalize_saliency(weights)


def _draw_metadata(ax, scene_id: str, frame_image: np.ndarray, ground_truth: Sequence[str], query: str) -> None:
    ax.axis("off")
    ax.set_title("GT / RGB / Query", loc="left", pad=4, fontsize=10, fontweight="bold")
    ax.patch.set_facecolor("white")
    ax.patch.set_edgecolor("#E3E7EE")
    ax.patch.set_linewidth(0.9)
    inset = ax.inset_axes([0.07, 0.52, 0.86, 0.40])
    inset.imshow(frame_image)
    inset.axis("off")
    inset.set_title("RGB view", fontsize=8, pad=2, loc="left")
    ax.text(
        0.03,
        0.46,
        f"Scene: {scene_id}",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#D0D4DD", "linewidth": 0.9},
    )
    ax.text(
        0.03,
        0.31,
        "GT labels\n" + _wrap_label_text(ground_truth, width=30),
        transform=ax.transAxes,
        fontsize=8,
        color="#1B7837",
        va="top",
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "#F7FBF7", "edgecolor": "#1B7837", "linewidth": 0.95},
    )
    ax.text(
        0.03,
        0.12,
        f"Query = {query}",
        transform=ax.transAxes,
        fontsize=8.3,
        color="#2F3B52",
        va="top",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "#FBFBFD", "edgecolor": "#D0D4DD", "linewidth": 0.9},
    )


def _draw_relevance_caption(ax, method_name: str, query: str) -> None:
    ax.text(
        0.02,
        0.03,
        f"{method_name} relevance for '{query}'",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#4B5563",
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#D0D4DD", "linewidth": 0.8},
    )


def generate_scannet_point_relevance_figure(config: ScanNetPointRelevanceFigureConfig) -> Dict[str, object]:
    _paper_style()
    if not config.scene_ids:
        raise ValueError("At least one scene id is required.")
    if len(config.scene_ids) != len(config.queries):
        raise ValueError("scene_ids and queries must have the same length.")

    device = _resolve_device(config.device or "cpu")
    came_bundle = _prepare_prediction_bundle(config.came_ckpt, config.came_method, config.data_root, device)
    baseline_bundle = _prepare_prediction_bundle(config.baseline_ckpt, config.baseline_method, config.data_root, device)

    rows: List[Dict[str, object]] = []
    for scene_id, query in zip(config.scene_ids, config.queries):
        came_scene = _get_scene_entry(came_bundle["dataset"], scene_id)
        baseline_scene = _get_scene_entry(baseline_bundle["dataset"], scene_id)
        render_scene = _build_render_scene(came_scene, config.render_num_points)
        render_coords = render_scene["render_coords"]
        render_colors = _object_semantic_colors(render_scene["render_object_labels"])

        came_indices = _sample_indices(render_coords.shape[0], came_bundle["config"].num_points, repeat=True)
        baseline_indices = _sample_indices(render_coords.shape[0], baseline_bundle["config"].num_points, repeat=True)
        came_points = torch.tensor(render_coords[came_indices], dtype=torch.float32)
        baseline_points = torch.tensor(render_coords[baseline_indices], dtype=torch.float32)

        came_saliency, came_logits = _compute_point_saliency(came_bundle, came_scene, came_points, query, device)
        baseline_saliency, baseline_logits = _compute_point_saliency(baseline_bundle, baseline_scene, baseline_points, query, device)

        came_weights = _map_saliency_to_render_points(render_coords.shape[0], came_indices, came_saliency.numpy())
        baseline_weights = _map_saliency_to_render_points(render_coords.shape[0], baseline_indices, baseline_saliency.numpy())

        rows.append(
            {
                "scene_id": scene_id,
                "query": query,
                "ground_truth": list(came_scene["labels"]),
                "frame_image": _extract_frame_image(came_bundle, came_scene, config.frame_index),
                "render_coords": render_coords,
                "render_colors": render_colors,
                "baseline_colors": _blend_relevance_colors(baseline_weights),
                "came_colors": _blend_relevance_colors(came_weights),
                "baseline_method_name": baseline_bundle["method"],
                "came_probs": torch.sigmoid(came_logits).numpy(),
                "baseline_probs": torch.sigmoid(baseline_logits).numpy(),
            }
        )

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14.8, max(1, len(rows)) * 3.7))
    gs = fig.add_gridspec(len(rows), 4, width_ratios=[1.0, 1.0, 1.0, 0.92], wspace=0.05, hspace=0.12)

    for row_index, row in enumerate(rows):
        input_ax = fig.add_subplot(gs[row_index, 0])
        baseline_ax = fig.add_subplot(gs[row_index, 1])
        came_ax = fig.add_subplot(gs[row_index, 2])
        meta_ax = fig.add_subplot(gs[row_index, 3])

        _plot_cloud(input_ax, row["render_coords"], row["render_colors"], "Input point cloud", config.camera_azim, config.camera_elev)
        _plot_cloud(
            baseline_ax,
            row["render_coords"],
            row["baseline_colors"],
            f"{row['baseline_method_name']} relevance",
            config.camera_azim,
            config.camera_elev,
        )
        _plot_cloud(came_ax, row["render_coords"], row["came_colors"], "CAME-Net relevance", config.camera_azim, config.camera_elev)
        _draw_relevance_caption(baseline_ax, row["baseline_method_name"], row["query"])
        _draw_relevance_caption(came_ax, "CAME-Net", row["query"])
        _draw_metadata(meta_ax, row["scene_id"], row["frame_image"], row["ground_truth"], row["query"])

    fig.suptitle("Point-wise Relevance Visualization on Indoor Point Clouds", y=0.992, x=0.5, fontsize=20, fontweight="bold")
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
        "queries": list(config.queries),
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    _write_json(output_path.with_name(output_path.stem + "_manifest.json"), manifest)
    return {"png": str(png_path), "pdf": str(pdf_path), "manifest": manifest}
