"""
robustness_benchmark.py - Rigid-robustness benchmark runner with a default 5-class protocol.
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from training.torch_runtime_compat import configure_torch_runtime_compat

configure_torch_runtime_compat()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .comparison_baselines import (
    build_comparison_model,
    describe_comparison_method,
    list_comparison_methods,
)
from .comparison_experiment import (
    ComparisonExperimentConfig,
    _forward_batch,
    _load_best_checkpoint,
    _save_checkpoint,
    _train_one_epoch,
    _validate,
    apply_comparison_runtime_defaults,
)
from training.data_utils import ModelNetDataset
from method.equiv_loss import equivariance_loss_efficient
from .small_modelnet_experiment import (
    DEFAULT_CLASS_PROTOCOL,
    FilteredModelNetSubset,
    collect_sample_predictions,
    create_experiment_artifacts,
    resolve_experiment_class_names,
    resolve_modelnet_root,
)


@dataclass
class RobustnessBenchmarkConfig:
    method: str = "came"
    data_root: Optional[str] = None
    class_protocol: str = DEFAULT_CLASS_PROTOCOL
    class_names: Optional[Sequence[str]] = None
    val_samples_per_class: int = 10
    train_samples_per_class: Optional[int] = None
    num_points: int = 256
    image_size: int = 32
    hidden_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    batch_size: int = 8
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    equiv_loss_weight: float = 1e-4
    equiv_warmup_steps: int = 200
    aux_loss_weight: float = 0.1
    dropout: float = 0.0
    device: Optional[str] = None
    artifact_root: str = "artifacts/robustness_benchmark"
    sample_visualization_count: int = 6
    checkpoint_interval: int = 100
    print_interval: int = 1
    disable_train_rigid_augmentation: bool = True

    @property
    def class_count(self) -> int:
        if self.class_names is not None:
            return len(self.class_names)
        if self.class_protocol == "small5":
            return 5
        return 40


def benchmark_collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    collated = {
        "point_coords": torch.stack([item["point_coords"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }
    if "sample_index" in batch[0]:
        collated["sample_index"] = torch.stack([item["sample_index"] for item in batch], dim=0)
    return collated


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


def get_default_robustness_conditions() -> List[Dict[str, object]]:
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


def apply_robustness_runtime_defaults(config: RobustnessBenchmarkConfig) -> RobustnessBenchmarkConfig:
    comparison_defaults = apply_comparison_runtime_defaults(
        ComparisonExperimentConfig(
            method=config.method,
            num_points=config.num_points,
            image_size=config.image_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            equiv_loss_weight=config.equiv_loss_weight,
            equiv_warmup_steps=config.equiv_warmup_steps,
            aux_loss_weight=config.aux_loss_weight,
            dropout=config.dropout,
            device=config.device,
            artifact_root=config.artifact_root,
        )
    )
    adjusted = asdict(config)
    adjusted["learning_rate"] = comparison_defaults.learning_rate
    adjusted["equiv_loss_weight"] = comparison_defaults.equiv_loss_weight
    adjusted["equiv_warmup_steps"] = comparison_defaults.equiv_warmup_steps
    adjusted["aux_loss_weight"] = comparison_defaults.aux_loss_weight
    adjusted["dropout"] = comparison_defaults.dropout
    return RobustnessBenchmarkConfig(**adjusted)


def build_robustness_benchmark_loaders(
    config: RobustnessBenchmarkConfig,
    *,
    resolved_data_root: Optional[Path] = None,
) -> tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    root = resolved_data_root or resolve_modelnet_root(config.data_root)
    allowed_classes = list(
        resolve_experiment_class_names(
            class_protocol=config.class_protocol,
            class_names=config.class_names,
            resolved_data_root=root,
            data_root=config.data_root,
        )
    )

    train_base_dataset = ModelNetDataset(
        data_dir=str(root),
        split="train",
        num_points=config.num_points,
        data_augmentation=not config.disable_train_rigid_augmentation,
    )
    val_base_dataset = ModelNetDataset(
        data_dir=str(root),
        split="train",
        num_points=config.num_points,
        data_augmentation=False,
    )
    test_base_dataset = ModelNetDataset(
        data_dir=str(root),
        split="test",
        num_points=config.num_points,
        data_augmentation=False,
    )

    if not allowed_classes:
        raise ValueError("Robustness benchmark requires at least one class name.")

    train_max = config.train_samples_per_class
    if train_max is None:
        train_by_class: Dict[str, int] = defaultdict(int)
        for _, label in train_base_dataset.samples:
            class_name = train_base_dataset.class_names[int(label)]
            if class_name in allowed_classes:
                train_by_class[class_name] += 1
        shortages = {
            class_name: count
            for class_name, count in train_by_class.items()
            if count <= config.val_samples_per_class
        }
        if shortages:
            raise ValueError(f"Not enough train samples after validation split for classes: {shortages}")
        train_max = min(train_by_class[class_name] - config.val_samples_per_class for class_name in allowed_classes)

    train_dataset = FilteredModelNetSubset(
        base_dataset=train_base_dataset,
        allowed_classes=allowed_classes,
        skip_samples_per_class=config.val_samples_per_class,
        max_samples_per_class=train_max,
    )
    val_dataset = FilteredModelNetSubset(
        base_dataset=val_base_dataset,
        allowed_classes=allowed_classes,
        max_samples_per_class=config.val_samples_per_class,
    )
    test_dataset = FilteredModelNetSubset(
        base_dataset=test_base_dataset,
        allowed_classes=allowed_classes,
        max_samples_per_class=min(
            sum(
                1
                for _, label in test_base_dataset.samples
                if test_base_dataset.class_names[int(label)] == class_name
            )
            for class_name in allowed_classes
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=benchmark_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=benchmark_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=benchmark_collate_fn,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


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


def _collect_clean_predictions(
    model: nn.Module,
    dataloader,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, object]:
    model.eval()
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    probabilities: List[torch.Tensor] = []
    predictions: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            logits = _forward_batch(model, batch, device)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            preds = torch.argmax(logits, dim=1).detach().cpu()
            labels = batch["labels"].detach().cpu()

            probabilities.append(probs)
            predictions.append(preds)
            labels_all.append(labels)

            for true_label, predicted_label in zip(labels.tolist(), preds.tolist()):
                confusion_matrix[int(true_label), int(predicted_label)] += 1

    labels_tensor = torch.cat(labels_all, dim=0) if labels_all else torch.empty(0, dtype=torch.long)
    preds_tensor = torch.cat(predictions, dim=0) if predictions else torch.empty(0, dtype=torch.long)
    probs_tensor = torch.cat(probabilities, dim=0) if probabilities else torch.empty((0, num_classes), dtype=torch.float32)

    total_samples = int(confusion_matrix.sum())
    clean_accuracy = 100.0 * int(np.trace(confusion_matrix)) / max(1, total_samples)

    return {
        "probabilities": probs_tensor,
        "predictions": preds_tensor,
        "labels": labels_tensor,
        "confusion_matrix": confusion_matrix.tolist(),
        "clean_accuracy": float(clean_accuracy),
    }


def _evaluate_condition(
    *,
    model: nn.Module,
    dataloader,
    device: torch.device,
    condition: Dict[str, object],
    clean_probs: torch.Tensor,
    clean_preds: torch.Tensor,
    labels: torch.Tensor,
    clean_accuracy: float,
) -> Dict[str, float]:
    if condition["name"] == "clean":
        return {
            "accuracy": float(clean_accuracy),
            "accuracy_drop": 0.0,
            "prediction_drift_mse": 0.0,
            "prediction_agreement": 100.0,
        }

    variant_metrics: List[Dict[str, float]] = []
    model.eval()
    for variant in condition["variants"]:
        total = 0
        correct = 0
        drift_total = 0.0
        agreement_total = 0
        offset = 0
        with torch.no_grad():
            for batch in dataloader:
                batch_size = int(batch["labels"].shape[0])
                transformed_coords = _apply_transform(
                    batch["point_coords"].to(device),
                    variant.get("rotation"),
                    variant.get("translation"),
                )
                logits = _forward_batch(model, _clone_batch_with_coords(batch, transformed_coords), device)
                probs = torch.softmax(logits, dim=1).detach().cpu()
                preds = torch.argmax(logits, dim=1).detach().cpu()
                batch_labels = labels[offset : offset + batch_size]
                batch_clean_probs = clean_probs[offset : offset + batch_size]
                batch_clean_preds = clean_preds[offset : offset + batch_size]

                correct += int((preds == batch_labels).sum().item())
                total += batch_size
                agreement_total += int((preds == batch_clean_preds).sum().item())
                drift_total += float(((probs - batch_clean_probs) ** 2).mean(dim=1).sum().item())
                offset += batch_size

        accuracy = 100.0 * correct / max(1, total)
        agreement = 100.0 * agreement_total / max(1, total)
        drift = drift_total / max(1, total)
        variant_metrics.append(
            {
                "accuracy": float(accuracy),
                "accuracy_drop": float(clean_accuracy - accuracy),
                "prediction_drift_mse": float(drift),
                "prediction_agreement": float(agreement),
            }
        )

    return {
        key: float(np.mean([metrics[key] for metrics in variant_metrics]))
        for key in ["accuracy", "accuracy_drop", "prediction_drift_mse", "prediction_agreement"]
    }


def evaluate_robustness_benchmark(
    *,
    model: nn.Module,
    dataloader,
    device: torch.device,
    class_names: Sequence[str],
    conditions: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    clean = _collect_clean_predictions(model, dataloader, device, class_names)
    clean_accuracy = float(clean["clean_accuracy"])
    condition_metrics: Dict[str, Dict[str, float]] = {}
    for condition in conditions:
        condition_metrics[str(condition["name"])] = _evaluate_condition(
            model=model,
            dataloader=dataloader,
            device=device,
            condition=condition,
            clean_probs=clean["probabilities"],
            clean_preds=clean["predictions"],
            labels=clean["labels"],
            clean_accuracy=clean_accuracy,
        )

    transformed_names = [name for name in condition_metrics if name != "clean"]
    mean_shift_accuracy = float(np.mean([condition_metrics[name]["accuracy"] for name in transformed_names])) if transformed_names else clean_accuracy
    worst_shift_accuracy = float(np.min([condition_metrics[name]["accuracy"] for name in transformed_names])) if transformed_names else clean_accuracy
    mean_accuracy_drop = float(np.mean([condition_metrics[name]["accuracy_drop"] for name in transformed_names])) if transformed_names else 0.0
    mean_prediction_drift = float(np.mean([condition_metrics[name]["prediction_drift_mse"] for name in transformed_names])) if transformed_names else 0.0
    mean_prediction_agreement = float(np.mean([condition_metrics[name]["prediction_agreement"] for name in transformed_names])) if transformed_names else 100.0

    per_class_accuracy: Dict[str, float] = {}
    confusion_matrix = np.asarray(clean["confusion_matrix"], dtype=np.int64)
    for class_index, class_name in enumerate(class_names):
        class_total = int(confusion_matrix[class_index].sum())
        per_class_accuracy[class_name] = float(100.0 * confusion_matrix[class_index, class_index] / max(1, class_total))

    return {
        "overall_accuracy": clean_accuracy,
        "mean_class_accuracy": float(np.mean(list(per_class_accuracy.values()))) if per_class_accuracy else 0.0,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": clean["confusion_matrix"],
        "condition_metrics": condition_metrics,
        "clean_accuracy": clean_accuracy,
        "mean_shift_accuracy": mean_shift_accuracy,
        "worst_shift_accuracy": worst_shift_accuracy,
        "mean_accuracy_drop": mean_accuracy_drop,
        "mean_prediction_drift": mean_prediction_drift,
        "mean_prediction_agreement": mean_prediction_agreement,
        "class_names": list(class_names),
    }


def _plot_condition_bar(condition_metrics: Dict[str, Dict[str, float]], metric_key: str, title: str, output_path: Path) -> None:
    names = list(condition_metrics.keys())
    values = [condition_metrics[name][metric_key] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, values, color="#4E79A7")
    ax.set_title(title)
    ax.set_ylabel(metric_key)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _append_robustness_summary(summary_path: Path, metrics: Dict[str, object]) -> None:
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Robustness Benchmark\n\n")
        handle.write(f"- Clean accuracy: {metrics['clean_accuracy']:.2f}\n")
        handle.write(f"- Mean shift accuracy: {metrics['mean_shift_accuracy']:.2f}\n")
        handle.write(f"- Worst shift accuracy: {metrics['worst_shift_accuracy']:.2f}\n")
        handle.write(f"- Mean accuracy drop: {metrics['mean_accuracy_drop']:.2f}\n")
        handle.write(f"- Mean prediction drift: {metrics['mean_prediction_drift']:.6f}\n")
        handle.write(f"- Mean prediction agreement: {metrics['mean_prediction_agreement']:.2f}\n")


def _write_condition_metrics_csv(path: Path, condition_metrics: Dict[str, Dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "accuracy", "accuracy_drop", "prediction_drift_mse", "prediction_agreement"],
        )
        writer.writeheader()
        for condition_name, metrics in condition_metrics.items():
            writer.writerow({"condition": condition_name, **metrics})


def run_robustness_benchmark(config: RobustnessBenchmarkConfig) -> Dict[str, object]:
    config = apply_robustness_runtime_defaults(config)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("Warning: CUDA is unavailable; running on CPU may exceed the intended runtime budget.")

    resolved_data_root = resolve_modelnet_root(config.data_root)
    run_root = Path(config.artifact_root) / "runs" / f"{config.method}-{time.strftime('%Y%m%d-%H%M%S')}"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_robustness_benchmark_loaders(
        config,
        resolved_data_root=resolved_data_root,
    )
    conditions = get_default_robustness_conditions()

    model = build_comparison_model(
        method=config.method,
        class_names=train_dataset.class_names,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        image_size=config.image_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.learning_rate / 100)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = float("-inf")
    best_checkpoint = checkpoint_dir / "best_model.pth"
    global_step = 0
    start_time = time.time()

    print("Starting robustness benchmark training...")
    print(f"Method: {config.method}")
    print(f"Device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Auxiliary loss weight: {config.aux_loss_weight}")
    print(f"Equivariance loss weight: {config.equiv_loss_weight}")
    print("-" * 60)

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        train_metrics, global_step = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            method=config.method,
            aux_loss_weight=config.aux_loss_weight,
            equiv_loss_weight=config.equiv_loss_weight,
            equiv_warmup_steps=config.equiv_warmup_steps,
            global_step=global_step,
        )
        scheduler.step()
        val_metrics = _validate(model=model, dataloader=val_loader, criterion=criterion, device=device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_accuracy"])
        history["lr"].append(scheduler.get_last_lr()[0])

        if (epoch + 1) % config.print_interval == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{config.num_epochs}] ({time.time() - epoch_start:.1f}s) "
                f"Train Loss: {train_metrics['loss']:.4f} Train Acc: {train_metrics['accuracy']:.2f}%"
            )
            print(
                f"                    Val Loss: {val_metrics['val_loss']:.4f} "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
            )

        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_accuracy=val_metrics["val_accuracy"],
                filepath=best_checkpoint,
            )
            print(f"Checkpoint saved to {best_checkpoint}")

    _load_best_checkpoint(model, best_checkpoint, device)

    evaluation_start = time.time()
    metrics = evaluate_robustness_benchmark(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=train_dataset.class_names,
        conditions=conditions,
    )
    evaluation_runtime_seconds = time.time() - evaluation_start
    train_runtime_seconds = time.time() - start_time

    metrics["parameter_count"] = sum(p.numel() for p in model.parameters())
    metrics["method"] = config.method
    metrics["train_runtime_seconds"] = train_runtime_seconds
    metrics["evaluation_runtime_seconds"] = evaluation_runtime_seconds

    sample_predictions = collect_sample_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        sample_count=config.sample_visualization_count,
    )

    artifact_dir = create_experiment_artifacts(
        artifact_root=str(run_root),
        config={**asdict(config), "resolved_data_root": str(resolved_data_root), "conditions": [condition["name"] for condition in conditions]},
        history=history,
        metrics=metrics,
        selected_classes=train_dataset.class_names,
        dataset_sizes={"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
        runtime_seconds=train_runtime_seconds,
        sample_predictions=sample_predictions,
        confusion_matrix=metrics["confusion_matrix"],
        extra_artifacts={"comparison_method": describe_comparison_method(config.method)},
    )

    plots_dir = artifact_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    _plot_condition_bar(metrics["condition_metrics"], "accuracy", "Accuracy Under Rigid Changes", plots_dir / "robustness_accuracy.png")
    _plot_condition_bar(metrics["condition_metrics"], "prediction_drift_mse", "Prediction Drift Under Rigid Changes", plots_dir / "robustness_drift.png")

    (artifact_dir / "robustness_conditions.json").write_text(
        json.dumps(
            [
                {
                    "name": condition["name"],
                    "variants": [variant["name"] for variant in condition["variants"]],
                }
                for condition in conditions
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "condition_metrics.json").write_text(
        json.dumps(metrics["condition_metrics"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    tables_dir = artifact_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    robustness_row = {
        "method": config.method,
        "clean_accuracy": metrics["clean_accuracy"],
        "mean_shift_accuracy": metrics["mean_shift_accuracy"],
        "worst_shift_accuracy": metrics["worst_shift_accuracy"],
        "mean_accuracy_drop": metrics["mean_accuracy_drop"],
        "mean_prediction_drift": metrics["mean_prediction_drift"],
        "mean_prediction_agreement": metrics["mean_prediction_agreement"],
        "parameter_count": metrics["parameter_count"],
        "train_runtime_seconds": train_runtime_seconds,
        "evaluation_runtime_seconds": evaluation_runtime_seconds,
    }
    (tables_dir / "robustness_row.json").write_text(json.dumps(robustness_row, indent=2, sort_keys=True), encoding="utf-8")
    _write_condition_metrics_csv(tables_dir / "condition_metrics.csv", metrics["condition_metrics"])
    _append_robustness_summary(artifact_dir / "summary.md", metrics)

    return {
        "artifact_dir": str(artifact_dir),
        "history": history,
        "metrics": metrics,
        "method": config.method,
    }


__all__ = [
    "RobustnessBenchmarkConfig",
    "apply_robustness_runtime_defaults",
    "benchmark_collate_fn",
    "build_robustness_benchmark_loaders",
    "get_default_robustness_conditions",
    "list_comparison_methods",
    "run_robustness_benchmark",
]
