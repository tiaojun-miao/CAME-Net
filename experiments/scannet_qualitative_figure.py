from __future__ import annotations

from dataclasses import asdict, dataclass
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

from .scannet_comparison_experiment import (
    ScanNetComparisonConfig,
    _build_scannet_model,
    apply_scannet_comparison_defaults,
)
from .scannet_multimodal_data import ScanNetSceneConfig, ScanNetSceneDataset
from .scannet_multimodal_experiment import _make_collate_fn, _resolve_device, _write_json
from training.torch_runtime_compat import configure_torch_runtime_compat

configure_torch_runtime_compat()


@dataclass
class ScanNetQualitativeFigureConfig:
    data_root: str
    came_ckpt: str
    baseline_ckpt: str
    baseline_method: str
    scene_ids: Sequence[str]
    output: str
    came_method: str = "came"
    device: Optional[str] = None
    render_num_points: int = 12000
    frame_index: int = 0
    top_k_predictions: int = 6
    camera_azim: float = 35.0
    camera_elev: float = 20.0
    max_mask_objects: int = 16


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.titlesize": 13,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _resolve_artifact_dir_from_checkpoint(ckpt_path: str) -> Path:
    path = Path(ckpt_path)
    if not path.exists():
        raise ValueError(f"Checkpoint does not exist: {ckpt_path}")
    if path.parent.name == "checkpoints":
        return path.parent.parent
    return path.parent


def _build_runtime_defaults(payload: Dict[str, object]) -> ScanNetComparisonConfig:
    config_payload = payload["config"]
    if "runtime_defaults" in config_payload:
        config_payload = config_payload["runtime_defaults"]
    return apply_scannet_comparison_defaults(ScanNetComparisonConfig(**config_payload))


def _select_positive_labels(logits: torch.Tensor, label_vocabulary: Sequence[str], top_k: int) -> List[str]:
    probs = torch.sigmoid(logits)
    active = [label_vocabulary[index] for index, value in enumerate((probs >= 0.5).tolist()) if value]
    if active:
        return active
    count = min(top_k, probs.numel())
    indices = torch.topk(probs, k=count).indices.tolist()
    return [label_vocabulary[index] for index in indices]


def _normalize_points(coords: np.ndarray) -> np.ndarray:
    normalized = coords.astype(np.float32, copy=True)
    normalized = normalized - normalized.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(normalized, axis=1).max()
    if scale > 0:
        normalized = normalized / scale
    return normalized


def _sample_indices(length: int, count: int, repeat: bool) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot sample from an empty point set.")
    if repeat:
        if length >= count:
            return np.linspace(0, length - 1, count, dtype=int)
        repeats = int(np.ceil(count / float(length)))
        tiled = np.tile(np.arange(length, dtype=int), repeats)
        return tiled[:count]
    if length <= count:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, count, dtype=int)


def _known_object_palette() -> Dict[str, str]:
    return {
        "chair": "#2166AC",
        "table": "#E08214",
        "sofa": "#1B7837",
        "bed": "#B2182B",
        "cabinet": "#7B3294",
    }


def _load_ply_vertices_with_rgb(scene_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply_path = scene_dir / f"{scene_dir.name}_vh_clean_2.ply"
    with ply_path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Malformed PLY header in {ply_path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        vertex_line = next((line for line in header_lines if line.startswith("element vertex")), None)
        if vertex_line is None:
            raise ValueError(f"PLY file missing vertex declaration: {ply_path}")
        vertex_count = int(vertex_line.split()[2])
        vertex_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("alpha", "u1"),
            ]
        )
        vertices = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
    coords = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
    colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.float32) / 255.0
    return coords, colors


def _load_scene_objects(scene_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    aggregation_path = scene_dir / f"{scene_dir.name}.aggregation.json"
    payload = _load_json(aggregation_path)
    segs_file = payload.get("segmentsFile", f"scannet.{scene_dir.name}_vh_clean_2.0.010000.segs.json")
    segs_path = scene_dir / str(segs_file).replace(f"scannet.{scene_dir.name}_", f"{scene_dir.name}_")
    if not segs_path.exists():
        segs_path = scene_dir / f"{scene_dir.name}_vh_clean_2.0.010000.segs.json"
    seg_payload = _load_json(segs_path)
    seg_indices = np.asarray(seg_payload["segIndices"], dtype=np.int32)

    object_ids = np.full(seg_indices.shape[0], -1, dtype=np.int32)
    object_labels = np.full(seg_indices.shape[0], "other", dtype=object)
    objects: List[Dict[str, object]] = []
    segment_to_object: Dict[int, int] = {}
    for object_index, group in enumerate(payload.get("segGroups", [])):
        label = str(group.get("label", "")).strip().lower()
        segments = [int(segment) for segment in group.get("segments", [])]
        if not label or not segments:
            continue
        objects.append({"object_id": object_index, "label": label, "segments": segments})
        for segment in segments:
            segment_to_object[segment] = object_index

    for vertex_index, segment in enumerate(seg_indices.tolist()):
        object_index = segment_to_object.get(int(segment), -1)
        if object_index >= 0:
            object_ids[vertex_index] = object_index
            object_labels[vertex_index] = str(objects[object_index]["label"])
    return object_ids, object_labels, objects


def _object_semantic_colors(labels: np.ndarray) -> np.ndarray:
    palette = _known_object_palette()
    default_color = np.asarray(mcolors.to_rgb("#C8CEDA"), dtype=np.float32)
    colors = np.tile(default_color[None, :], (labels.shape[0], 1))
    for label, hex_color in palette.items():
        mask = labels == label
        if np.any(mask):
            colors[mask] = np.asarray(mcolors.to_rgb(hex_color), dtype=np.float32)
    return colors


def _blend_heat_colors(weights: np.ndarray) -> np.ndarray:
    base = np.tile(np.asarray(mcolors.to_rgb("#B9C0CD"), dtype=np.float32)[None, :], (weights.shape[0], 1))
    cmap = matplotlib.colormaps["magma"]
    mapped = np.asarray([cmap(float(weight))[:3] for weight in weights], dtype=np.float32)
    blend = 0.20 + 0.80 * weights[:, None]
    return base * (1.0 - blend) + mapped * blend


def _draw_label_rows(ax, labels: Sequence[str], ground_truth: Sequence[str], anchor_y: float) -> None:
    predicted = set(labels)
    truth = set(ground_truth)
    correct = sorted(predicted & truth)
    missed = sorted(truth - predicted)
    extras = sorted(predicted - truth)
    blocks = [
        ("Correct", correct, "#1B7837"),
        ("Missed", missed, "#B2182B"),
        ("Extra", extras, "#E08214"),
    ]
    y = anchor_y
    for title, values, color in blocks:
        text = ", ".join(values) if values else "none"
        ax.text2D(
            0.02,
            y,
            f"{title}: {text}",
            transform=ax.transAxes,
            fontsize=8,
            color=color,
            va="top",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color, "linewidth": 0.8, "alpha": 0.92},
        )
        y -= 0.085


def _wrap_label_text(labels: Sequence[str], width: int = 28) -> str:
    if not labels:
        return "none"
    return textwrap.fill(", ".join(labels), width=width)


def _pretty_method_name(method: str) -> str:
    names = {
        "pointnet": "PointNet",
        "pointclip_style": "PointCLIP-style",
        "gatr_style": "GATr-style",
        "equiformer_v2_style": "EquiformerV2-style",
        "came": "CAME-Net",
        "came_no_gln": "CAME w/o GLN",
        "came_no_equiv_reg": "CAME w/o equiv. reg.",
        "came_non_geometric_fusion": "CAME w/o geometric fusion",
        "came_scalar_only": "CAME scalar-only",
    }
    return names.get(method, method.replace("_", " "))


def _project_points(coords: np.ndarray, azim: float, elev: float) -> Tuple[np.ndarray, np.ndarray]:
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
    xy = xy - xy.mean(axis=0, keepdims=True)
    scale = float(np.abs(xy).max())
    if scale > 0:
        xy = xy / scale
    return xy, rotated[:, 2]


def _set_projection_axis_style(ax, title: str) -> None:
    ax.set_title(title, loc="left", pad=5, fontsize=10, fontweight="bold")
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


def _plot_projected_cloud(
    ax,
    coords: np.ndarray,
    colors: np.ndarray,
    title: str,
    azim: float,
    elev: float,
) -> None:
    projected, depth = _project_points(coords, azim=azim, elev=elev)
    order = np.argsort(depth)
    ax.scatter(
        projected[order, 0],
        projected[order, 1],
        c=colors[order],
        s=5.0,
        alpha=1.0,
        linewidths=0,
        marker="o",
        rasterized=True,
    )
    _set_projection_axis_style(ax, title)


def _prepare_prediction_bundle(
    ckpt_path: str,
    method_override: Optional[str],
    data_root: str,
    device: torch.device,
) -> Dict[str, object]:
    artifact_dir = _resolve_artifact_dir_from_checkpoint(ckpt_path)
    config_payload = _load_json(artifact_dir / "config.json")
    dataset_report = _load_json(artifact_dir / "dataset_report.json")
    label_vocabulary = _load_json(artifact_dir / "label_vocabulary.json")
    comparison_config = _build_runtime_defaults({"config": config_payload})
    if method_override is not None:
        comparison_config = apply_scannet_comparison_defaults(
            ScanNetComparisonConfig(**{**asdict(comparison_config), "method": method_override})
        )

    dataset = ScanNetSceneDataset(
        ScanNetSceneConfig(
            data_root=data_root,
            num_points=comparison_config.num_points,
            max_frames=comparison_config.max_frames,
            frame_resize=comparison_config.frame_resize,
            min_label_frequency=comparison_config.min_label_frequency,
            top_k_labels=config_payload.get("top_k_labels", config_payload.get("top_k_labels", None)),
            require_all_modalities=True,
            vocabulary_scene_ids=dataset_report.get("vocabulary_source_scene_ids"),
        )
    )
    model = _build_scannet_model(comparison_config, label_count=len(dataset.label_vocabulary)).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    collate_fn = _make_collate_fn(comparison_config)
    return {
        "artifact_dir": artifact_dir,
        "config": comparison_config,
        "dataset": dataset,
        "model": model,
        "label_vocabulary": label_vocabulary,
        "collate_fn": collate_fn,
        "method": comparison_config.method,
    }


def _get_scene_entry(dataset: ScanNetSceneDataset, scene_id: str) -> Dict[str, object]:
    for entry in dataset.scenes:
        if entry["scene_id"] == scene_id:
            return entry
    raise ValueError(f"Scene id {scene_id} is not retained in the current dataset protocol.")


def _build_prompt_from_labels(labels: Sequence[str]) -> str:
    return "A room containing " + ", ".join(labels) + "."


def _build_single_scene_batch(
    bundle: Dict[str, object],
    scene_entry: Dict[str, object],
    point_coords: torch.Tensor,
) -> Dict[str, object]:
    dataset: ScanNetSceneDataset = bundle["dataset"]
    scene_dir = scene_entry["scene_dir"]
    example = {
        "scene_id": scene_entry["scene_id"],
        "point_coords": point_coords,
        "image_tensor": dataset._load_image_tensor(scene_dir),
        "text_prompt": scene_entry["text_prompt"],
        "label_targets": scene_entry["label_targets"].clone(),
        "labels": list(scene_entry["labels"]),
    }
    return bundle["collate_fn"]([example])


def _run_single_scene_prediction(
    bundle: Dict[str, object],
    scene_entry: Dict[str, object],
    point_coords: torch.Tensor,
    device: torch.device,
    top_k: int,
) -> Dict[str, object]:
    batch = _build_single_scene_batch(bundle, scene_entry, point_coords)
    with torch.no_grad():
        logits = bundle["model"](
            point_coords=batch["point_coords"].to(device),
            image_patches=batch["image_patches"].to(device),
            text_tokens=batch["text_tokens"].to(device),
        ).detach().cpu()[0]
    predicted_labels = _select_positive_labels(logits, bundle["dataset"].label_vocabulary, top_k)
    return {
        "logits": logits,
        "predicted_labels": predicted_labels,
        "ground_truth": list(scene_entry["labels"]),
    }


def _compute_object_relevance(
    bundle: Dict[str, object],
    scene_entry: Dict[str, object],
    model_coords: np.ndarray,
    model_object_ids: np.ndarray,
    objects: Sequence[Dict[str, object]],
    target_labels: Sequence[str],
    device: torch.device,
    top_k: int,
    max_mask_objects: int,
) -> Dict[int, float]:
    if not target_labels:
        return {}
    label_to_index = {label: idx for idx, label in enumerate(bundle["dataset"].label_vocabulary)}
    target_indices = [label_to_index[label] for label in target_labels if label in label_to_index]
    if not target_indices:
        return {}

    base_prediction = _run_single_scene_prediction(
        bundle,
        scene_entry,
        torch.tensor(model_coords, dtype=torch.float32),
        device,
        top_k,
    )
    base_probs = torch.sigmoid(base_prediction["logits"])
    object_scores: Dict[int, float] = {}

    candidate_objects = [obj for obj in objects if obj["label"] in set(target_labels)]
    candidate_objects = candidate_objects[:max_mask_objects]
    for obj in candidate_objects:
        object_id = int(obj["object_id"])
        keep_mask = model_object_ids != object_id
        if int(keep_mask.sum()) < 16:
            continue
        masked_prediction = _run_single_scene_prediction(
            bundle,
            scene_entry,
            torch.tensor(model_coords[keep_mask], dtype=torch.float32),
            device,
            top_k,
        )
        masked_probs = torch.sigmoid(masked_prediction["logits"])
        score = float(torch.clamp(base_probs[target_indices] - masked_probs[target_indices], min=0.0).mean().item())
        object_scores[object_id] = score
    return object_scores


def _build_render_scene(scene_entry: Dict[str, object], render_num_points: int) -> Dict[str, object]:
    scene_dir: Path = scene_entry["scene_dir"]
    coords, _ = _load_ply_vertices_with_rgb(scene_dir)
    object_ids, object_labels, objects = _load_scene_objects(scene_dir)

    render_indices = _sample_indices(coords.shape[0], render_num_points, repeat=False)
    render_coords = _normalize_points(coords[render_indices])
    render_object_ids = object_ids[render_indices]
    render_object_labels = object_labels[render_indices]

    return {
        "render_coords": render_coords,
        "render_object_ids": render_object_ids,
        "render_object_labels": render_object_labels,
        "objects": objects,
        "scene_dir": scene_dir,
    }


def _sample_model_points(render_scene: Dict[str, object], model_num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    render_coords = render_scene["render_coords"]
    render_object_ids = render_scene["render_object_ids"]
    indices = _sample_indices(render_coords.shape[0], model_num_points, repeat=True)
    return render_coords[indices], render_object_ids[indices]


def _extract_frame_image(bundle: Dict[str, object], scene_entry: Dict[str, object], frame_index: int) -> np.ndarray:
    dataset: ScanNetSceneDataset = bundle["dataset"]
    image_tensor = dataset._load_image_tensor(scene_entry["scene_dir"])
    index = min(max(frame_index, 0), image_tensor.shape[0] - 1)
    image = image_tensor[index].permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def _plot_input_cloud(ax, coords: np.ndarray, labels: np.ndarray, title: str, azim: float, elev: float) -> None:
    colors = _object_semantic_colors(labels)
    _plot_projected_cloud(ax, coords, colors, title, azim, elev)


def _plot_heat_cloud(
    ax,
    coords: np.ndarray,
    object_ids: np.ndarray,
    object_scores: Dict[int, float],
    title: str,
    azim: float,
    elev: float,
) -> None:
    weights = np.zeros(coords.shape[0], dtype=np.float32)
    if object_scores:
        max_value = max(max(object_scores.values()), 1e-8)
        for object_id, score in object_scores.items():
            weights[object_ids == object_id] = float(score / max_value)
    colors = _blend_heat_colors(weights)
    _plot_projected_cloud(ax, coords, colors, title, azim, elev)


def _plot_metadata_panel(
    ax,
    scene_id: str,
    frame_image: np.ndarray,
    ground_truth: Sequence[str],
    prompt: str,
) -> None:
    ax.axis("off")
    ax.set_title("GT / RGB / Prompt", loc="left", pad=5, fontsize=10, fontweight="bold")
    ax.patch.set_facecolor("white")
    ax.patch.set_edgecolor("#E3E7EE")
    ax.patch.set_linewidth(0.9)
    inset = ax.inset_axes([0.06, 0.54, 0.88, 0.38])
    inset.imshow(frame_image)
    inset.axis("off")
    inset.set_title("RGB view", fontsize=8, pad=2, loc="left")
    ax.text(
        0.02,
        0.48,
        f"Scene: {scene_id}",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#D0D4DD", "linewidth": 0.9},
    )
    ax.text(
        0.02,
        0.34,
        "GT labels\n" + _wrap_label_text(ground_truth, width=30),
        transform=ax.transAxes,
        fontsize=8,
        color="#1B7837",
        va="top",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F7FBF7", "edgecolor": "#1B7837", "linewidth": 1.0},
    )
    ax.text(
        0.02,
        0.14,
        "Semantic prompt\n" + textwrap.fill(prompt, width=34),
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        va="top",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#FBFBFD", "edgecolor": "#D0D4DD", "linewidth": 0.9},
    )


def _draw_prediction_cards(ax, labels: Sequence[str], ground_truth: Sequence[str]) -> None:
    ax.axis("off")
    ax.set_facecolor("white")
    predicted = set(labels)
    truth = set(ground_truth)
    blocks = [
        ("Correct", sorted(predicted & truth), "#1B7837"),
        ("Missed", sorted(truth - predicted), "#B2182B"),
        ("Extra", sorted(predicted - truth), "#E08214"),
    ]
    positions = [
        (0.03, 0.10),
        (0.36, 0.10),
        (0.69, 0.10),
    ]
    for (title, values, color), (x, y) in zip(blocks, positions):
        ax.text(
            x,
            y,
            f"{title}\n{_wrap_label_text(values, width=22)}",
            transform=ax.transAxes,
            fontsize=7.3,
            color=color,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.95,
                "alpha": 0.98,
            },
        )


def _render_scene_row(
    fig: plt.Figure,
    gs,
    row_index: int,
    scene_payload: Dict[str, object],
    config: ScanNetQualitativeFigureConfig,
) -> None:
    coords = scene_payload["render_coords"]
    object_ids = scene_payload["render_object_ids"]
    object_labels = scene_payload["render_object_labels"]
    ground_truth = scene_payload["ground_truth"]

    input_ax = fig.add_subplot(gs[row_index, 0])
    metadata_ax = fig.add_subplot(gs[row_index, 1])
    baseline_gs = gs[row_index, 2].subgridspec(2, 1, height_ratios=[0.82, 0.18], hspace=0.02)
    came_gs = gs[row_index, 3].subgridspec(2, 1, height_ratios=[0.82, 0.18], hspace=0.02)
    baseline_ax = fig.add_subplot(baseline_gs[0, 0])
    baseline_cards_ax = fig.add_subplot(baseline_gs[1, 0])
    came_ax = fig.add_subplot(came_gs[0, 0])
    came_cards_ax = fig.add_subplot(came_gs[1, 0])

    _plot_input_cloud(
        input_ax,
        coords,
        object_labels,
        "Input point cloud",
        config.camera_azim,
        config.camera_elev,
    )
    _plot_metadata_panel(
        metadata_ax,
        scene_payload["scene_id"],
        scene_payload["frame_image"],
        ground_truth,
        scene_payload["prompt"],
    )
    _plot_heat_cloud(
        baseline_ax,
        coords,
        object_ids,
        scene_payload["baseline_scores"],
        _pretty_method_name(scene_payload["baseline_method"]),
        config.camera_azim,
        config.camera_elev,
    )
    _plot_heat_cloud(
        came_ax,
        coords,
        object_ids,
        scene_payload["came_scores"],
        "CAME-Net",
        config.camera_azim,
        config.camera_elev,
    )

    _draw_prediction_cards(baseline_cards_ax, scene_payload["baseline_prediction"]["predicted_labels"], ground_truth)
    _draw_prediction_cards(came_cards_ax, scene_payload["came_prediction"]["predicted_labels"], ground_truth)


def generate_scannet_qualitative_figure(config: ScanNetQualitativeFigureConfig) -> Dict[str, object]:
    _paper_style()
    if not config.scene_ids:
        raise ValueError("At least one scene id is required.")

    device = _resolve_device(config.device or "cpu")
    came_bundle = _prepare_prediction_bundle(config.came_ckpt, config.came_method, config.data_root, device)
    baseline_bundle = _prepare_prediction_bundle(config.baseline_ckpt, config.baseline_method, config.data_root, device)

    rows = []
    for scene_id in config.scene_ids:
        came_scene = _get_scene_entry(came_bundle["dataset"], scene_id)
        baseline_scene = _get_scene_entry(baseline_bundle["dataset"], scene_id)
        render_scene = _build_render_scene(came_scene, config.render_num_points)

        came_model_coords, came_model_object_ids = _sample_model_points(render_scene, came_bundle["config"].num_points)
        baseline_model_coords, baseline_model_object_ids = _sample_model_points(render_scene, baseline_bundle["config"].num_points)

        came_prediction = _run_single_scene_prediction(
            came_bundle,
            came_scene,
            torch.tensor(came_model_coords, dtype=torch.float32),
            device,
            config.top_k_predictions,
        )
        baseline_prediction = _run_single_scene_prediction(
            baseline_bundle,
            baseline_scene,
            torch.tensor(baseline_model_coords, dtype=torch.float32),
            device,
            config.top_k_predictions,
        )

        target_labels = list(dict.fromkeys(came_prediction["ground_truth"]))
        came_scores = _compute_object_relevance(
            came_bundle,
            came_scene,
            came_model_coords,
            came_model_object_ids,
            render_scene["objects"],
            target_labels,
            device,
            config.top_k_predictions,
            config.max_mask_objects,
        )
        baseline_scores = _compute_object_relevance(
            baseline_bundle,
            baseline_scene,
            baseline_model_coords,
            baseline_model_object_ids,
            render_scene["objects"],
            target_labels,
            device,
            config.top_k_predictions,
            config.max_mask_objects,
        )

        rows.append(
            {
                "scene_id": scene_id,
                "render_coords": render_scene["render_coords"],
                "render_object_ids": render_scene["render_object_ids"],
                "render_object_labels": render_scene["render_object_labels"],
                "ground_truth": target_labels,
                "prompt": _build_prompt_from_labels(target_labels),
                "frame_image": _extract_frame_image(came_bundle, came_scene, config.frame_index),
                "came_prediction": came_prediction,
                "baseline_prediction": baseline_prediction,
                "came_scores": came_scores,
                "baseline_scores": baseline_scores,
                "baseline_method": baseline_bundle["method"],
            }
        )

    figure_path = Path(config.output)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15.2, max(1, len(rows)) * 3.32))
    gs = fig.add_gridspec(len(rows), 4, width_ratios=[1.08, 0.94, 1.08, 1.08], wspace=0.04, hspace=0.10)
    for row_index, payload in enumerate(rows):
        _render_scene_row(fig, gs, row_index, payload, config)

    fig.suptitle("Qualitative Comparison on Indoor Point Clouds", y=0.988, x=0.5, fontweight="bold", fontsize=20)
    png_path = figure_path.with_suffix(".png")
    pdf_path = figure_path.with_suffix(".pdf")
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
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }
    _write_json(figure_path.with_name(figure_path.stem + "_manifest.json"), manifest)
    return {"png": str(png_path), "pdf": str(pdf_path), "manifest": manifest}
