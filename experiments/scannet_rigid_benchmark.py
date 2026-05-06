from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .comparison_baselines import describe_comparison_method, get_comparison_method_specs
from .scannet_comparison_experiment import (
    LabelPriorBaseline,
    ScanNetComparisonConfig,
    _build_model_audit,
    _build_scannet_model,
    _compute_train_label_priors,
    _evaluate,
    _forward_scannet_batch,
    _subset_indices,
    _train_one_epoch,
    apply_scannet_comparison_defaults,
)
from .scannet_multimodal_data import ScanNetSceneConfig, ScanNetSceneDataset, split_public_holdout_scene_ids
from .scannet_multimodal_experiment import (
    _compute_multilabel_metrics,
    _create_artifact_dir,
    _make_collate_fn,
    _plot_training_curves,
    _resolve_device,
    _split_indices,
    _write_json,
)
from training.torch_runtime_compat import configure_torch_runtime_compat

configure_torch_runtime_compat()


@dataclass
class ScanNetRigidBenchmarkConfig:
    method: str = "came"
    data_root: str = "F:/CAME-Net/ScanNet-small"
    artifact_root: str = "artifacts/scannet_rigid_benchmark"
    num_points: int = 256
    max_frames: int = 3
    frame_resize: int = 32
    image_feature_size: int = 8
    max_text_tokens: int = 48
    batch_size: int = 2
    num_epochs: int = 12
    hidden_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    aux_loss_weight: float = 0.0
    equiv_loss_weight: float = 1e-4
    equiv_warmup_steps: int = 200
    dropout: float = 0.0
    device: Optional[str] = None
    min_label_frequency: int = 1
    top_k_labels: int = 12
    blind_holdout_fraction: float = 0.2
    use_blind_holdout: bool = False
    print_interval: int = 1


def _axis_rotation_matrix(axis: str, degrees: float) -> torch.Tensor:
    radians = float(np.deg2rad(degrees))
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    if axis == "x":
        matrix = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    elif axis == "y":
        matrix = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    elif axis == "z":
        matrix = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    else:
        raise ValueError(f"Unsupported rotation axis: {axis}")
    return torch.tensor(matrix, dtype=torch.float32)


def _translation_vector(axis: str, magnitude: float) -> torch.Tensor:
    if axis == "x":
        vector = [magnitude, 0.0, 0.0]
    elif axis == "y":
        vector = [0.0, magnitude, 0.0]
    elif axis == "z":
        vector = [0.0, 0.0, magnitude]
    else:
        raise ValueError(f"Unsupported translation axis: {axis}")
    return torch.tensor(vector, dtype=torch.float32)


def get_default_scannet_rigid_conditions() -> List[Dict[str, object]]:
    return [
        {"name": "clean", "variants": [{"name": "identity", "rotation": None, "translation": None}]},
        {
            "name": "rot15",
            "variants": [
                {"name": "rot_x_15", "rotation": _axis_rotation_matrix("x", 15.0), "translation": None},
                {"name": "rot_y_15", "rotation": _axis_rotation_matrix("y", 15.0), "translation": None},
                {"name": "rot_z_15", "rotation": _axis_rotation_matrix("z", 15.0), "translation": None},
            ],
        },
        {
            "name": "rot30",
            "variants": [
                {"name": "rot_x_30", "rotation": _axis_rotation_matrix("x", 30.0), "translation": None},
                {"name": "rot_y_30", "rotation": _axis_rotation_matrix("y", 30.0), "translation": None},
                {"name": "rot_z_30", "rotation": _axis_rotation_matrix("z", 30.0), "translation": None},
            ],
        },
        {
            "name": "rot45",
            "variants": [
                {"name": "rot_x_45", "rotation": _axis_rotation_matrix("x", 45.0), "translation": None},
                {"name": "rot_y_45", "rotation": _axis_rotation_matrix("y", 45.0), "translation": None},
                {"name": "rot_z_45", "rotation": _axis_rotation_matrix("z", 45.0), "translation": None},
            ],
        },
        {
            "name": "trans0p1",
            "variants": [
                {"name": "tx_0p1", "rotation": None, "translation": _translation_vector("x", 0.1)},
                {"name": "ty_0p1", "rotation": None, "translation": _translation_vector("y", 0.1)},
                {"name": "tz_0p1", "rotation": None, "translation": _translation_vector("z", 0.1)},
            ],
        },
        {
            "name": "trans0p2",
            "variants": [
                {"name": "tx_0p2", "rotation": None, "translation": _translation_vector("x", 0.2)},
                {"name": "ty_0p2", "rotation": None, "translation": _translation_vector("y", 0.2)},
                {"name": "tz_0p2", "rotation": None, "translation": _translation_vector("z", 0.2)},
            ],
        },
        {
            "name": "trans0p3",
            "variants": [
                {"name": "tx_0p3", "rotation": None, "translation": _translation_vector("x", 0.3)},
                {"name": "ty_0p3", "rotation": None, "translation": _translation_vector("y", 0.3)},
                {"name": "tz_0p3", "rotation": None, "translation": _translation_vector("z", 0.3)},
            ],
        },
        {
            "name": "se3_mix",
            "variants": [
                {"name": "se3_x", "rotation": _axis_rotation_matrix("x", 30.0), "translation": _translation_vector("x", 0.1)},
                {"name": "se3_y", "rotation": _axis_rotation_matrix("y", 30.0), "translation": _translation_vector("y", 0.1)},
                {"name": "se3_z", "rotation": _axis_rotation_matrix("z", 30.0), "translation": _translation_vector("z", 0.1)},
            ],
        },
    ]


def _apply_transform(point_coords: torch.Tensor, rotation: Optional[torch.Tensor], translation: Optional[torch.Tensor]) -> torch.Tensor:
    transformed = point_coords
    if rotation is not None:
        transformed = transformed @ rotation.to(point_coords.device).T
    if translation is not None:
        transformed = transformed + translation.to(point_coords.device)
    return transformed


def _clone_batch_with_coords(batch: Dict[str, torch.Tensor], point_coords: torch.Tensor) -> Dict[str, torch.Tensor]:
    cloned = dict(batch)
    cloned["point_coords"] = point_coords
    return cloned


def _split_scannet_scene_indices(dataset: ScanNetSceneDataset, use_blind_holdout: bool, blind_holdout_fraction: float) -> Tuple[List[int], List[int], List[int], Dict[str, List[str]]]:
    scene_ids = dataset.dataset_report["scene_ids"]
    scene_to_index = {entry["scene_id"]: idx for idx, entry in enumerate(dataset.scenes)}
    public_scene_ids, holdout_scene_ids = split_public_holdout_scene_ids(scene_ids, blind_holdout_fraction=blind_holdout_fraction)

    if use_blind_holdout and holdout_scene_ids:
        public_indices = [scene_to_index[scene_id] for scene_id in public_scene_ids]
        train_rel, val_rel, _ = _split_indices(len(public_indices))
        train_indices = [public_indices[index] for index in train_rel]
        val_indices = [public_indices[index] for index in val_rel]
        test_indices = [scene_to_index[scene_id] for scene_id in holdout_scene_ids]
    else:
        ordered_scene_ids = sorted(scene_ids)
        ordered_indices = [scene_to_index[scene_id] for scene_id in ordered_scene_ids]
        train_rel, val_rel, test_rel = _split_indices(len(ordered_indices))
        train_indices = [ordered_indices[index] for index in train_rel]
        val_indices = [ordered_indices[index] for index in val_rel]
        test_indices = [ordered_indices[index] for index in test_rel]
        public_scene_ids = ordered_scene_ids
        holdout_scene_ids = []

    return train_indices, val_indices, test_indices, {"public": public_scene_ids, "holdout": holdout_scene_ids}


def _build_scannet_rigid_dataset(config: ScanNetRigidBenchmarkConfig) -> ScanNetSceneDataset:
    bootstrap_dataset = ScanNetSceneDataset(
        ScanNetSceneConfig(
            data_root=config.data_root,
            num_points=config.num_points,
            max_frames=config.max_frames,
            frame_resize=config.frame_resize,
            min_label_frequency=config.min_label_frequency,
            top_k_labels=config.top_k_labels,
            require_all_modalities=True,
        )
    )
    train_indices, _, _, _ = _split_scannet_scene_indices(
        bootstrap_dataset,
        use_blind_holdout=config.use_blind_holdout,
        blind_holdout_fraction=config.blind_holdout_fraction,
    )
    vocabulary_scene_ids = sorted(bootstrap_dataset.scenes[index]["scene_id"] for index in train_indices)
    return ScanNetSceneDataset(
        ScanNetSceneConfig(
            data_root=config.data_root,
            num_points=config.num_points,
            max_frames=config.max_frames,
            frame_resize=config.frame_resize,
            min_label_frequency=config.min_label_frequency,
            top_k_labels=config.top_k_labels,
            require_all_modalities=True,
            vocabulary_scene_ids=vocabulary_scene_ids,
        )
    )


def _build_scannet_rigid_loaders(dataset: ScanNetSceneDataset, config: ScanNetRigidBenchmarkConfig):
    train_indices, val_indices, test_indices, scene_protocol = _split_scannet_scene_indices(
        dataset,
        use_blind_holdout=config.use_blind_holdout,
        blind_holdout_fraction=config.blind_holdout_fraction,
    )
    collate_fn = _make_collate_fn(config)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    split_sizes = {"train": len(train_indices), "val": len(val_indices), "test": len(test_indices)}
    return train_loader, val_loader, test_loader, split_sizes, scene_protocol


def _collect_clean_predictions(model: nn.Module, dataloader, device: torch.device) -> Dict[str, torch.Tensor]:
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            logits = _forward_scannet_batch(model, batch, device)
            all_logits.append(logits.detach().cpu())
            all_targets.append(batch["label_targets"].detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return {"logits": logits, "targets": targets, "probs": probs, "preds": preds}


def _evaluate_condition(
    *,
    model: nn.Module,
    dataloader,
    device: torch.device,
    condition: Dict[str, object],
    clean_logits: torch.Tensor,
    clean_probs: torch.Tensor,
    clean_preds: torch.Tensor,
    targets: torch.Tensor,
    clean_micro_f1: float,
) -> Dict[str, float]:
    clean_correct_mask = (clean_preds == targets).all(dim=1)
    clean_correct_count = int(clean_correct_mask.sum().item())

    if condition["name"] == "clean":
        clean_metrics = _compute_multilabel_metrics(torch.logit(clean_probs.clamp(1e-6, 1 - 1e-6)), targets)
        return {
            "micro_f1": clean_metrics["micro_f1"],
            "macro_f1": clean_metrics["macro_f1"],
            "exact_match_accuracy": clean_metrics["exact_match_accuracy"],
            "conditional_rigid_micro_f1": clean_metrics["micro_f1"],
            "clean_correct_flip_rate": 0.0,
            "logit_drift": 0.0,
            "clean_correct_count": clean_correct_count,
            "micro_f1_drop": 0.0,
            "prediction_drift": 0.0,
            "prediction_agreement": 1.0,
        }

    variant_metrics: List[Dict[str, float]] = []
    for variant in condition["variants"]:
        all_logits = []
        offset = 0
        with torch.no_grad():
            for batch in dataloader:
                batch_size = int(batch["label_targets"].shape[0])
                transformed_coords = _apply_transform(
                    batch["point_coords"].to(device),
                    variant.get("rotation"),
                    variant.get("translation"),
                )
                logits = _forward_scannet_batch(model, _clone_batch_with_coords(batch, transformed_coords), device)
                all_logits.append(logits.detach().cpu())
                offset += batch_size
        logits = torch.cat(all_logits, dim=0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        metrics = _compute_multilabel_metrics(logits, targets)
        if clean_correct_mask.any():
            conditional_metrics = _compute_multilabel_metrics(logits[clean_correct_mask], targets[clean_correct_mask])
            conditional_micro_f1 = conditional_metrics["micro_f1"]
            flip_rate = float((~(preds[clean_correct_mask] == targets[clean_correct_mask]).all(dim=1)).float().mean().item())
        else:
            conditional_micro_f1 = 0.0
            flip_rate = 0.0
        drift = float(((probs - clean_probs) ** 2).mean(dim=1).mean().item())
        logit_drift = float(((logits - clean_logits) ** 2).mean(dim=1).mean().item())
        agreement = float((preds == clean_preds).all(dim=1).float().mean().item())
        variant_metrics.append(
            {
                "micro_f1": metrics["micro_f1"],
                "macro_f1": metrics["macro_f1"],
                "exact_match_accuracy": metrics["exact_match_accuracy"],
                "conditional_rigid_micro_f1": conditional_micro_f1,
                "clean_correct_flip_rate": flip_rate,
                "logit_drift": logit_drift,
                "clean_correct_count": float(clean_correct_count),
                "micro_f1_drop": clean_micro_f1 - metrics["micro_f1"],
                "prediction_drift": drift,
                "prediction_agreement": agreement,
            }
        )
    return {key: float(np.mean([metrics[key] for metrics in variant_metrics])) for key in variant_metrics[0]}


def _plot_condition_metric(condition_metrics: Dict[str, Dict[str, float]], metric_key: str, title: str, output_path: Path) -> None:
    names = list(condition_metrics.keys())
    values = [condition_metrics[name][metric_key] for name in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, values, color="#4E79A7")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_scannet_rigid_benchmark(config: ScanNetRigidBenchmarkConfig) -> Dict[str, object]:
    comparison_config = apply_scannet_comparison_defaults(
        ScanNetComparisonConfig(
            method=config.method,
            data_root=config.data_root,
            artifact_root=config.artifact_root,
            num_points=config.num_points,
            max_frames=config.max_frames,
            frame_resize=config.frame_resize,
            image_feature_size=config.image_feature_size,
            max_text_tokens=config.max_text_tokens,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            aux_loss_weight=config.aux_loss_weight,
            equiv_loss_weight=config.equiv_loss_weight,
            equiv_warmup_steps=config.equiv_warmup_steps,
            dropout=config.dropout,
            device=config.device,
            min_label_frequency=config.min_label_frequency,
            print_interval=config.print_interval,
        )
    )
    if comparison_config.method not in get_comparison_method_specs():
        raise ValueError(f"Unknown comparison method: {comparison_config.method}")

    device = _resolve_device(comparison_config.device)
    dataset = _build_scannet_rigid_dataset(config)
    train_loader, val_loader, test_loader, split_sizes, scene_protocol = _build_scannet_rigid_loaders(dataset, config)
    conditions = get_default_scannet_rigid_conditions()

    artifact_dir = _create_artifact_dir(comparison_config.artifact_root)
    tables_dir = artifact_dir / "tables"
    plots_dir = artifact_dir / "plots"
    checkpoints_dir = artifact_dir / "checkpoints"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_indices = _subset_indices(train_loader.dataset)
    if comparison_config.method == "label_prior":
        train_label_priors = _compute_train_label_priors(dataset, train_indices)
        model = LabelPriorBaseline(train_label_priors).to(device)
    else:
        model = _build_scannet_model(comparison_config, label_count=len(dataset.label_vocabulary)).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = (
        optim.AdamW(trainable_parameters, lr=comparison_config.learning_rate, weight_decay=comparison_config.weight_decay)
        if trainable_parameters
        else None
    )
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_micro_f1": [],
        "val_micro_f1": [],
    }
    best_val_f1 = -1.0
    best_state = None
    start_time = time.time()

    for epoch in range(comparison_config.num_epochs):
        train_metrics = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = _evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_micro_f1"].append(train_metrics["micro_f1"])
        history["val_micro_f1"].append(val_metrics["micro_f1"])

        if (epoch + 1) % comparison_config.print_interval == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{comparison_config.num_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train micro-F1: {train_metrics['micro_f1']:.4f} "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val micro-F1: {val_metrics['micro_f1']:.4f}"
            )

        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without a best checkpoint.")

    torch.save(best_state, checkpoints_dir / "best_model.pth")
    model.load_state_dict(best_state)
    clean = _collect_clean_predictions(model, test_loader, device)
    clean_metrics = _compute_multilabel_metrics(clean["logits"], clean["targets"])
    condition_metrics = {}
    for condition in conditions:
        condition_metrics[condition["name"]] = _evaluate_condition(
            model=model,
            dataloader=test_loader,
            device=device,
            condition=condition,
            clean_logits=clean["logits"],
            clean_probs=clean["probs"],
            clean_preds=clean["preds"],
            targets=clean["targets"],
            clean_micro_f1=clean_metrics["micro_f1"],
        )

    transformed_names = [name for name in condition_metrics if name != "clean"]
    metrics = {
        "clean_micro_f1": clean_metrics["micro_f1"],
        "clean_macro_f1": clean_metrics["macro_f1"],
        "exact_match_accuracy": clean_metrics["exact_match_accuracy"],
        "mean_rigid_micro_f1": float(np.mean([condition_metrics[name]["micro_f1"] for name in transformed_names])) if transformed_names else clean_metrics["micro_f1"],
        "worst_rigid_micro_f1": float(np.min([condition_metrics[name]["micro_f1"] for name in transformed_names])) if transformed_names else clean_metrics["micro_f1"],
        "conditional_rigid_micro_f1": float(np.mean([condition_metrics[name]["conditional_rigid_micro_f1"] for name in transformed_names])) if transformed_names else clean_metrics["micro_f1"],
        "clean_correct_flip_rate": float(np.mean([condition_metrics[name]["clean_correct_flip_rate"] for name in transformed_names])) if transformed_names else 0.0,
        "logit_drift": float(np.mean([condition_metrics[name]["logit_drift"] for name in transformed_names])) if transformed_names else 0.0,
        "micro_f1_drop": float(np.mean([condition_metrics[name]["micro_f1_drop"] for name in transformed_names])) if transformed_names else 0.0,
        "prediction_drift": float(np.mean([condition_metrics[name]["prediction_drift"] for name in transformed_names])) if transformed_names else 0.0,
        "prediction_agreement": float(np.mean([condition_metrics[name]["prediction_agreement"] for name in transformed_names])) if transformed_names else 1.0,
        "condition_metrics": condition_metrics,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_runtime_seconds": time.time() - start_time,
        "method": comparison_config.method,
    }

    summary_lines = [
        "# ScanNet Rigid Benchmark Summary",
        "",
        f"- Method: {comparison_config.method}",
        f"- Scenes retained: {dataset.dataset_report['retained_scenes']}",
        f"- Public scenes: {len(scene_protocol['public'])}",
        f"- Holdout scenes: {len(scene_protocol['holdout'])}",
        f"- Train/Val/Test split sizes: {split_sizes}",
        f"- Label vocabulary size: {len(dataset.label_vocabulary)}",
        f"- Clean micro-F1: {metrics['clean_micro_f1']:.4f}",
        f"- Mean rigid micro-F1: {metrics['mean_rigid_micro_f1']:.4f}",
        f"- Worst rigid micro-F1: {metrics['worst_rigid_micro_f1']:.4f}",
        f"- Conditional rigid micro-F1: {metrics['conditional_rigid_micro_f1']:.4f}",
        f"- Clean-correct flip rate: {metrics['clean_correct_flip_rate']:.4f}",
        f"- Micro-F1 drop: {metrics['micro_f1_drop']:.4f}",
        f"- Logit drift: {metrics['logit_drift']:.6f}",
        f"- Prediction drift: {metrics['prediction_drift']:.6f}",
        f"- Prediction agreement: {metrics['prediction_agreement']:.4f}",
    ]
    (artifact_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (artifact_dir / "comparison_method.md").write_text(describe_comparison_method(comparison_config.method), encoding="utf-8")
    _write_json(artifact_dir / "config.json", {**asdict(config), "runtime_defaults": asdict(comparison_config)})
    _write_json(artifact_dir / "dataset_report.json", {**dataset.dataset_report, "scene_protocol": scene_protocol})
    _write_json(artifact_dir / "label_vocabulary.json", dataset.label_vocabulary)
    _write_json(artifact_dir / "label_frequencies.json", dataset.dataset_report["label_frequencies"])
    _write_json(artifact_dir / "train_label_frequencies.json", dataset.dataset_report["vocabulary_label_frequencies"])
    test_scene_ids = [dataset.scenes[index]["scene_id"] for index in _subset_indices(test_loader.dataset)]
    _write_json(artifact_dir / "test_scene_ids.json", test_scene_ids)
    _write_json(
        artifact_dir / "figure_metadata.json",
        {
            "method": comparison_config.method,
            "condition_names": [condition["name"] for condition in conditions],
            "label_vocabulary": dataset.label_vocabulary,
            "test_scene_ids": test_scene_ids,
            "split_sizes": split_sizes,
            "scene_protocol": scene_protocol,
        },
    )
    _write_json(artifact_dir / "audit.json", _build_model_audit(model))
    _write_json(artifact_dir / "metrics.json", {key: value for key, value in metrics.items() if key != "condition_metrics"})
    _write_json(artifact_dir / "condition_metrics.json", condition_metrics)
    _plot_training_curves(history, plots_dir / "training_curves.png")
    _plot_condition_metric(condition_metrics, "micro_f1", "Micro-F1 Under Rigid Changes", plots_dir / "rigid_micro_f1.png")
    _plot_condition_metric(condition_metrics, "prediction_drift", "Prediction Drift Under Rigid Changes", plots_dir / "prediction_drift.png")

    benchmark_row = {
        "method": comparison_config.method,
        "clean_micro_f1": metrics["clean_micro_f1"],
        "mean_rigid_micro_f1": metrics["mean_rigid_micro_f1"],
        "worst_rigid_micro_f1": metrics["worst_rigid_micro_f1"],
        "conditional_rigid_micro_f1": metrics["conditional_rigid_micro_f1"],
        "clean_correct_flip_rate": metrics["clean_correct_flip_rate"],
        "logit_drift": metrics["logit_drift"],
        "micro_f1_drop": metrics["micro_f1_drop"],
        "prediction_drift": metrics["prediction_drift"],
        "prediction_agreement": metrics["prediction_agreement"],
        "parameter_count": metrics["parameter_count"],
        "train_runtime_seconds": metrics["train_runtime_seconds"],
    }
    _write_json(tables_dir / "benchmark_row.json", benchmark_row)

    return {
        "artifact_dir": artifact_dir,
        "metrics": metrics,
        "split_sizes": split_sizes,
        "scene_protocol": scene_protocol,
        "dataset_report": dataset.dataset_report,
    }
