"""
small_modelnet_experiment.py - Lightweight ModelNet40 experiment helpers.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils import ModelNetDataset


DEFAULT_SMALL_EXPERIMENT_CLASSES = [
    "airplane",
    "chair",
    "lamp",
    "sofa",
    "toilet",
]


@dataclass
class SmallExperimentConfig:
    data_root: Optional[str] = None
    class_names: Sequence[str] = tuple(DEFAULT_SMALL_EXPERIMENT_CLASSES)
    train_samples_per_class: int = 100
    test_samples_per_class: int = 30
    num_points: int = 256
    hidden_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    equiv_loss_weight: float = 0.1
    equiv_warmup_steps: int = 0
    dropout: float = 0.1
    device: Optional[str] = None
    artifact_root: str = "artifacts/small_modelnet_experiment"
    sample_visualization_count: int = 6
    checkpoint_interval: int = 100
    print_interval: int = 1


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _plot_training_curves(history: Dict[str, list], output_path: Path) -> None:
    epochs = list(range(1, max(len(history.get("train_loss", [])), len(history.get("val_loss", [])), 1) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(epochs[: len(history.get("train_loss", []))], history.get("train_loss", []), label="Train")
    if history.get("val_loss"):
        axes[0].plot(epochs[: len(history.get("val_loss", []))], history.get("val_loss", []), label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs[: len(history.get("train_acc", []))], history.get("train_acc", []), label="Train")
    if history.get("val_acc"):
        axes[1].plot(epochs[: len(history.get("val_acc", []))], history.get("val_acc", []), label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    if history.get("lr"):
        axes[2].plot(epochs[: len(history.get("lr", []))], history.get("lr", []), label="LR")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(confusion_matrix, class_names: Sequence[str], output_path: Path) -> None:
    matrix = np.asarray(confusion_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("confusion_matrix must be a 2D array-like structure")

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * matrix.shape[1]), max(4, 0.7 * matrix.shape[0])))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(list(class_names)[: matrix.shape[1]], rotation=45, ha="right")
    ax.set_yticklabels(list(class_names)[: matrix.shape[0]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{int(value)}" if float(value).is_integer() else f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_predictions(sample_predictions: Sequence[Dict], output_path: Path) -> None:
    num_samples = len(sample_predictions)
    if num_samples == 0:
        fig = plt.figure(figsize=(5, 4))
        fig.text(0.5, 0.5, "No sample predictions provided", ha="center", va="center")
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    num_cols = min(3, num_samples)
    num_rows = int(np.ceil(num_samples / num_cols))
    fig = plt.figure(figsize=(5 * num_cols, 4.5 * num_rows))

    for index, sample in enumerate(sample_predictions, start=1):
        ax = fig.add_subplot(num_rows, num_cols, index, projection="3d")
        point_coords = torch.as_tensor(sample["point_coords"]).detach().cpu().numpy()
        labels = sample.get("class_names", [])
        predicted_label = int(sample.get("predicted_label", -1))
        true_label = int(sample.get("true_label", -1))
        title_parts = [f"True: {labels[true_label] if 0 <= true_label < len(labels) else true_label}"]
        title_parts.append(f"Pred: {labels[predicted_label] if 0 <= predicted_label < len(labels) else predicted_label}")
        ax.set_title("\n".join(title_parts), fontsize=9)
        ax.scatter(point_coords[:, 0], point_coords[:, 1], point_coords[:, 2], c=point_coords[:, 2], cmap="viridis", s=5)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _format_summary(
    *,
    selected_classes: Sequence[str],
    dataset_sizes: Dict[str, int],
    runtime_seconds: float,
    metrics: Dict[str, object],
    artifact_files: Sequence[str],
) -> str:
    per_class_accuracy = metrics.get("per_class_accuracy", [])
    lines = [
        "# Small ModelNet Experiment Summary",
        "",
        "## Selected Classes",
        "",
        ", ".join(selected_classes) if selected_classes else "None",
        "",
        "## Dataset Sizes",
        "",
    ]
    lines.extend(f"- {name}: {count}" for name, count in dataset_sizes.items())
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"{runtime_seconds:.2f} seconds",
            "",
            "## Metrics",
            "",
            f"- Overall accuracy: {metrics.get('overall_accuracy', 'n/a')}",
            f"- Mean class accuracy: {metrics.get('mean_class_accuracy', 'n/a')}",
            "- Per-class accuracy:",
        ]
    )
    lines.extend(f"  - Class {index}: {value}" for index, value in enumerate(per_class_accuracy))
    lines.extend(
        [
            "",
            "## Artifact Files",
            "",
        ]
    )
    lines.extend(f"- {name}" for name in artifact_files)
    lines.append("")
    return "\n".join(lines)


def create_experiment_artifacts(
    *,
    artifact_root: str,
    config: Dict[str, object],
    history: Dict[str, list],
    metrics: Dict[str, object],
    selected_classes: Sequence[str],
    dataset_sizes: Dict[str, int],
    runtime_seconds: float,
    sample_predictions: Sequence[Dict],
    confusion_matrix=None,
) -> Path:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    artifact_dir = root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    artifact_dir.mkdir(parents=False, exist_ok=False)

    _write_json(artifact_dir / "config.json", config)
    _write_json(artifact_dir / "history.json", history)
    _write_json(artifact_dir / "metrics.json", metrics)

    _plot_training_curves(history, artifact_dir / "training_curves.png")

    matrix = confusion_matrix if confusion_matrix is not None else metrics.get("confusion_matrix")
    if matrix is None:
        matrix = np.zeros((len(selected_classes), len(selected_classes)), dtype=int)
    _plot_confusion_matrix(matrix, selected_classes, artifact_dir / "confusion_matrix.png")

    _plot_sample_predictions(sample_predictions, artifact_dir / "sample_predictions.png")

    artifact_files = sorted(path.name for path in artifact_dir.iterdir() if path.is_file())
    summary = _format_summary(
        selected_classes=selected_classes,
        dataset_sizes=dataset_sizes,
        runtime_seconds=runtime_seconds,
        metrics=metrics,
        artifact_files=artifact_files + ["summary.md"],
    )
    (artifact_dir / "summary.md").write_text(summary, encoding="utf-8")

    return artifact_dir


def resolve_modelnet_root(data_root: Optional[str] = None) -> Path:
    if data_root is not None:
        root = Path(data_root)
    else:
        root = Path(__file__).resolve().parent / "ModelNet40" / "ModelNet40"

    if not root.exists():
        raise FileNotFoundError(f"ModelNet40 root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"ModelNet40 root is not a directory: {root}")
    return root


class FilteredModelNetSubset(Dataset):
    def __init__(
        self,
        base_dataset: ModelNetDataset,
        allowed_classes: Sequence[str],
        max_samples_per_class: int,
    ) -> None:
        if max_samples_per_class <= 0:
            raise ValueError("max_samples_per_class must be positive")

        self.base_dataset = base_dataset
        self.class_names = list(allowed_classes)
        if not self.class_names:
            raise ValueError("allowed_classes must contain at least one class")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("allowed_classes must not contain duplicate class names")
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        base_class_to_idx = getattr(base_dataset, "class_to_idx", {})
        missing_classes = [class_name for class_name in self.class_names if class_name not in base_class_to_idx]
        if missing_classes:
            raise ValueError(f"Missing classes in dataset: {missing_classes}")

        self.indices: List[int] = []
        selected_counts: Counter[str] = Counter()

        for sample_idx, (_, original_label) in enumerate(base_dataset.samples):
            class_name = base_dataset.class_names[int(original_label)]
            if class_name not in self.class_to_idx:
                continue
            if selected_counts[class_name] >= max_samples_per_class:
                continue
            self.indices.append(sample_idx)
            selected_counts[class_name] += 1

        shortages = {
            class_name: max_samples_per_class - selected_counts[class_name]
            for class_name in self.class_names
            if selected_counts[class_name] < max_samples_per_class
        }
        if shortages:
            raise ValueError(f"Not enough samples for requested subset: {shortages}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[self.indices[idx]]
        original_label = int(sample["labels"].item())
        class_name = self.base_dataset.class_names[original_label]
        remapped_label = self.class_to_idx[class_name]
        return {
            "point_coords": sample["point_coords"],
            "labels": torch.tensor(remapped_label, dtype=torch.long),
        }
