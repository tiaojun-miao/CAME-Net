"""
small_modelnet_experiment.py - Lightweight ModelNet40 experiment helpers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
