# Small ModelNet Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight, reproducible 5-class `ModelNet40` experiment that trains a small `CAMENet`, saves accuracy metrics and visual artifacts, and stays within the user's small-scale demo constraints.

**Architecture:** Keep the core method unchanged. Build a thin experiment layer around the existing `ModelNetDataset`, `CAMENet`, `train_came_net`, and equivariance loss utilities. Use a wrapper dataset to filter and remap a fixed 5-class subset, then add one helper module for experiment orchestration, plotting, artifact writing, and one script entrypoint that runs the demo with safe defaults.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, existing CAME-Net training utilities, local `ModelNet40` `OFF` meshes

---

### Task 1: Add Filtered Small-Experiment Dataset Wrappers

**Files:**
- Create: `small_modelnet_experiment.py`
- Create: `test_small_modelnet_experiment.py`
- Test: `test_small_modelnet_experiment.py`

- [ ] **Step 1: Write the failing test**

Create `test_small_modelnet_experiment.py` with the filtering/remapping check:

```python
"""
test_small_modelnet_experiment.py - Lightweight experiment checks for ModelNet40.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import ModelNetDataset
from small_modelnet_experiment import FilteredModelNetSubset


def get_modelnet_root() -> str:
    env_root = os.environ.get("MODELNET40_ROOT")
    if env_root:
        return env_root
    return str(Path(__file__).resolve().parent / "ModelNet40" / "ModelNet40")


def require_modelnet_root() -> str:
    root = get_modelnet_root()
    if not Path(root).exists():
        raise SystemExit(f"SKIP: ModelNet40 root not found at {root}")
    return root


def test_filtered_subset_remaps_labels_and_caps_per_class():
    base_dataset = ModelNetDataset(
        data_dir=require_modelnet_root(),
        split="train",
        num_points=32,
        data_augmentation=False,
    )
    subset = FilteredModelNetSubset(
        base_dataset=base_dataset,
        allowed_classes=["airplane", "chair"],
        max_samples_per_class=2,
    )

    assert subset.class_names == ["airplane", "chair"]
    assert subset.num_classes == 2
    assert len(subset) == 4

    label_values = sorted({subset[i]["labels"].item() for i in range(len(subset))})
    assert label_values == [0, 1]


if __name__ == "__main__":
    test_filtered_subset_remaps_labels_and_caps_per_class()
    print("test_small_modelnet_experiment.py: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: `FAIL` with `ModuleNotFoundError` because `small_modelnet_experiment.py` and `FilteredModelNetSubset` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `small_modelnet_experiment.py` with the subset wrapper and fixed-class defaults:

```python
"""
small_modelnet_experiment.py - Lightweight ModelNet40 experiment helpers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

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
    ):
        if max_samples_per_class <= 0:
            raise ValueError("max_samples_per_class must be positive")

        self.base_dataset = base_dataset
        self.class_names = list(allowed_classes)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        missing_classes = [
            class_name for class_name in self.class_names
            if class_name not in base_dataset.class_to_idx
        ]
        if missing_classes:
            raise ValueError(f"Missing classes in dataset: {missing_classes}")

        self.indices: List[int] = []
        selected_counts: Counter[str] = Counter()

        for sample_idx, (_, original_label) in enumerate(base_dataset.samples):
            class_name = base_dataset.class_names[original_label]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: `PASS` and the remapped labels are `[0, 1]` instead of sparse 40-class indices.

- [ ] **Step 5: Commit**

Run:

```powershell
git add small_modelnet_experiment.py test_small_modelnet_experiment.py
git commit -m "feat: add filtered ModelNet subset wrapper"
```

### Task 2: Add Artifact Writers and Visualization Helpers

**Files:**
- Modify: `small_modelnet_experiment.py`
- Modify: `test_small_modelnet_experiment.py`
- Test: `test_small_modelnet_experiment.py`

- [ ] **Step 1: Write the failing test**

Extend `test_small_modelnet_experiment.py` with an artifact creation check:

```python
from tempfile import TemporaryDirectory

import numpy as np

from small_modelnet_experiment import SmallExperimentConfig, write_experiment_artifacts
```

```python
def test_write_experiment_artifacts_creates_expected_files():
    config = SmallExperimentConfig(
        data_root=require_modelnet_root(),
        class_names=("airplane", "chair"),
        train_samples_per_class=2,
        test_samples_per_class=1,
        num_points=32,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        batch_size=2,
        num_epochs=1,
        equiv_loss_weight=0.0,
        device="cpu",
        artifact_root="unused",
        sample_visualization_count=1,
    )
    history = {
        "train_loss": [1.2, 0.8],
        "train_acc": [50.0, 75.0],
        "val_loss": [1.1, 0.7],
        "val_acc": [40.0, 80.0],
        "lr": [1e-3, 5e-4],
    }
    metrics = {
        "overall_accuracy": 80.0,
        "mean_class_accuracy": 75.0,
        "per_class_accuracy": {"airplane": 100.0, "chair": 50.0},
        "confusion_matrix": [[1, 0], [1, 1]],
        "class_names": ["airplane", "chair"],
        "runtime_seconds": 12.5,
    }
    sample_predictions = [
        {
            "points": np.zeros((8, 3), dtype=np.float32),
            "true_label": "airplane",
            "predicted_label": "chair",
            "correct": False,
        }
    ]

    with TemporaryDirectory() as tmpdir:
        artifact_dir = write_experiment_artifacts(
            output_root=Path(tmpdir),
            config=config,
            history=history,
            metrics=metrics,
            sample_predictions=sample_predictions,
        )

        for filename in [
            "config.json",
            "history.json",
            "metrics.json",
            "summary.md",
            "training_curves.png",
            "confusion_matrix.png",
            "sample_predictions.png",
        ]:
            assert (artifact_dir / filename).exists(), filename
```

Update the script footer:

```python
if __name__ == "__main__":
    test_filtered_subset_remaps_labels_and_caps_per_class()
    test_write_experiment_artifacts_creates_expected_files()
    print("test_small_modelnet_experiment.py: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: `FAIL` because `write_experiment_artifacts` and the plotting utilities do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Extend `small_modelnet_experiment.py` with artifact writing, JSON serialization, and plots:

```python
import json
from dataclasses import asdict
from datetime import datetime

import matplotlib.pyplot as plt
```

```python
def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train loss")
    if history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(epochs, history["train_acc"], label="train acc")
    if history["val_acc"]:
        axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: Sequence[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            ax.text(col, row, int(confusion_matrix[row, col]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sample_predictions(sample_predictions: List[Dict[str, object]], output_path: Path) -> None:
    columns = min(3, max(1, len(sample_predictions)))
    rows = int(np.ceil(len(sample_predictions) / columns))
    fig = plt.figure(figsize=(4 * columns, 4 * rows))

    for index, sample in enumerate(sample_predictions, start=1):
        ax = fig.add_subplot(rows, columns, index, projection="3d")
        points = np.asarray(sample["points"], dtype=np.float32)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=6, c=points[:, 2], cmap="viridis")
        title = f"T: {sample['true_label']} | P: {sample['predicted_label']}"
        if not sample["correct"]:
            title += " | wrong"
        ax.set_title(title)
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_experiment_artifacts(
    output_root: Path,
    config: SmallExperimentConfig,
    history: Dict[str, List[float]],
    metrics: Dict[str, object],
    sample_predictions: List[Dict[str, object]],
) -> Path:
    artifact_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_dir.mkdir(parents=True, exist_ok=False)

    (artifact_dir / "config.json").write_text(json.dumps(_to_serializable(asdict(config)), indent=2), encoding="utf-8")
    (artifact_dir / "history.json").write_text(json.dumps(_to_serializable(history), indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(json.dumps(_to_serializable(metrics), indent=2), encoding="utf-8")

    plot_training_curves(history, artifact_dir / "training_curves.png")
    plot_confusion_matrix(np.asarray(metrics["confusion_matrix"]), metrics["class_names"], artifact_dir / "confusion_matrix.png")
    plot_sample_predictions(sample_predictions, artifact_dir / "sample_predictions.png")

    per_class_lines = [
        f"  - {class_name}: {accuracy:.2f}"
        for class_name, accuracy in metrics["per_class_accuracy"].items()
    ]
    artifact_files = [
        "config.json",
        "history.json",
        "metrics.json",
        "summary.md",
        "training_curves.png",
        "confusion_matrix.png",
        "sample_predictions.png",
    ]
    summary_lines = [
        "# Small ModelNet Experiment Summary",
        "",
        f"- Classes: {', '.join(metrics['class_names'])}",
        f"- Train samples: {config.train_samples_per_class} per class",
        f"- Test samples: {config.test_samples_per_class} per class",
        f"- Runtime (s): {metrics['runtime_seconds']:.2f}",
        f"- Overall accuracy: {metrics['overall_accuracy']:.2f}",
        f"- Mean class accuracy: {metrics['mean_class_accuracy']:.2f}",
        "- Per-class accuracy:",
        *per_class_lines,
        "- Artifact files:",
        *[f"  - {filename}" for filename in artifact_files],
        f"- Artifact directory: {artifact_dir}",
    ]
    (artifact_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return artifact_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: `PASS` and the temp artifact directory contains all required `.json`, `.md`, and `.png` outputs.

- [ ] **Step 5: Commit**

Run:

```powershell
git add small_modelnet_experiment.py test_small_modelnet_experiment.py
git commit -m "feat: add small experiment artifact writers"
```

### Task 3: Add End-to-End Small Experiment Runner

**Files:**
- Modify: `small_modelnet_experiment.py`
- Modify: `test_small_modelnet_experiment.py`
- Test: `test_small_modelnet_experiment.py`

- [ ] **Step 1: Write the failing test**

Extend `test_small_modelnet_experiment.py` with a tiny smoke run:

```python
from small_modelnet_experiment import run_small_experiment
```

```python
def test_small_experiment_smoke_run():
    with TemporaryDirectory() as tmpdir:
        config = SmallExperimentConfig(
            data_root=require_modelnet_root(),
            class_names=("airplane", "chair", "lamp", "sofa", "toilet"),
            train_samples_per_class=2,
            test_samples_per_class=1,
            num_points=32,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            batch_size=2,
            num_epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            equiv_loss_weight=0.0,
            device="cpu",
            artifact_root=tmpdir,
            sample_visualization_count=2,
            checkpoint_interval=100,
            print_interval=1,
        )

        result = run_small_experiment(config)

        assert Path(result["artifact_dir"]).exists()
        assert result["metrics"]["overall_accuracy"] >= 0.0
        assert len(result["metrics"]["class_names"]) == 5
        assert len(result["history"]["train_loss"]) == 1
```

Update the script footer:

```python
if __name__ == "__main__":
    test_filtered_subset_remaps_labels_and_caps_per_class()
    test_write_experiment_artifacts_creates_expected_files()
    test_small_experiment_smoke_run()
    print("test_small_modelnet_experiment.py: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: `FAIL` because there is no training/evaluation orchestration function yet.

- [ ] **Step 3: Write minimal implementation**

Extend `small_modelnet_experiment.py` with dataloader creation, evaluation, sample visualization collection, and the orchestration function:

```python
import time

from torch.utils.data import DataLoader

from came_net import CAMENet, count_parameters
from data_utils import collate_fn
from equiv_loss import equivariance_loss_efficient
from train import load_checkpoint, train_came_net
```

```python
def build_small_experiment_dataloaders(config: SmallExperimentConfig):
    root = resolve_modelnet_root(config.data_root)

    train_base = ModelNetDataset(
        data_dir=str(root),
        split="train",
        num_points=config.num_points,
        data_augmentation=True,
    )
    test_base = ModelNetDataset(
        data_dir=str(root),
        split="test",
        num_points=config.num_points,
        data_augmentation=False,
    )

    train_dataset = FilteredModelNetSubset(
        base_dataset=train_base,
        allowed_classes=config.class_names,
        max_samples_per_class=config.train_samples_per_class,
    )
    test_dataset = FilteredModelNetSubset(
        base_dataset=test_base,
        allowed_classes=config.class_names,
        max_samples_per_class=config.test_samples_per_class,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def evaluate_subset_model(
    model: CAMENet,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, object]:
    model.eval()
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in dataloader:
            logits = model(point_coords=batch["point_coords"].to(device))
            predicted = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            for true_label, predicted_label in zip(labels, predicted):
                confusion[int(true_label), int(predicted_label)] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    overall_accuracy = 100.0 * correct / max(1, total)

    per_class_accuracy = {}
    per_class_values = []
    for index, class_name in enumerate(class_names):
        class_total = int(confusion[index].sum())
        class_accuracy = 100.0 * confusion[index, index] / max(1, class_total)
        per_class_accuracy[class_name] = class_accuracy
        per_class_values.append(class_accuracy)

    return {
        "overall_accuracy": overall_accuracy,
        "mean_class_accuracy": float(np.mean(per_class_values)),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion.tolist(),
        "class_names": list(class_names),
    }


def collect_sample_predictions(
    model: CAMENet,
    dataset: FilteredModelNetSubset,
    device: torch.device,
    sample_count: int,
) -> List[Dict[str, object]]:
    model.eval()
    samples: List[Dict[str, object]] = []
    limit = min(sample_count, len(dataset))

    with torch.no_grad():
        for index in range(limit):
            sample = dataset[index]
            logits = model(point_coords=sample["point_coords"].unsqueeze(0).to(device))
            predicted_label = int(torch.argmax(logits, dim=1).item())
            true_label = int(sample["labels"].item())
            samples.append(
                {
                    "points": sample["point_coords"].cpu().numpy(),
                    "true_label": dataset.class_names[true_label],
                    "predicted_label": dataset.class_names[predicted_label],
                    "correct": predicted_label == true_label,
                }
            )

    return samples


def run_small_experiment(config: SmallExperimentConfig) -> Dict[str, object]:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    artifact_root = Path(config.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    if device.type != "cuda":
        print("Warning: CUDA not available. Running on CPU; runtime may exceed the intended 15-minute budget.")

    train_dataset, test_dataset, train_loader, test_loader = build_small_experiment_dataloaders(config)

    model = CAMENet(
        num_classes=len(config.class_names),
        point_feature_dim=0,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        multimodal=False,
    ).to(device)

    start_time = time.time()
    history = train_came_net(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        device=device,
        equiv_loss_weight=config.equiv_loss_weight,
        equiv_loss_fn=equivariance_loss_efficient,
        equiv_warmup_steps=config.equiv_warmup_steps,
        checkpoint_dir=str(artifact_root / "checkpoints"),
        checkpoint_interval=config.checkpoint_interval,
        print_interval=config.print_interval,
    )

    best_checkpoint = artifact_root / "checkpoints" / "best_model.pth"
    if best_checkpoint.exists():
        load_checkpoint(model, optimizer=None, scheduler=None, filepath=str(best_checkpoint), device=device)

    metrics = evaluate_subset_model(model, test_loader, device, train_dataset.class_names)
    metrics["runtime_seconds"] = time.time() - start_time
    metrics["parameter_count"] = count_parameters(model)

    sample_predictions = collect_sample_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        sample_count=config.sample_visualization_count,
    )
    artifact_dir = write_experiment_artifacts(
        output_root=artifact_root,
        config=config,
        history=history,
        metrics=metrics,
        sample_predictions=sample_predictions,
    )

    return {
        "artifact_dir": str(artifact_dir),
        "history": history,
        "metrics": metrics,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: `PASS` after a short CPU smoke training run and artifact emission into a temp directory.

- [ ] **Step 5: Commit**

Run:

```powershell
git add small_modelnet_experiment.py test_small_modelnet_experiment.py
git commit -m "feat: add small ModelNet experiment runner"
```

### Task 4: Add User Entrypoint and Ignore Generated Artifacts

**Files:**
- Create: `run_small_modelnet_experiment.py`
- Modify: `.gitignore`
- Test: `test_small_modelnet_experiment.py`

- [ ] **Step 1: Write the failing test**

Extend `test_small_modelnet_experiment.py` with an entrypoint and ignore-file check:

```python
def test_entrypoint_script_and_gitignore_contract():
    repo_root = Path(__file__).resolve().parent
    gitignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")

    assert "artifacts/" in gitignore_text.splitlines()

    script_path = repo_root / "run_small_modelnet_experiment.py"
    assert script_path.exists()
    script_text = script_path.read_text(encoding="utf-8")
    assert "SmallExperimentConfig()" in script_text
    assert "run_small_experiment(config)" in script_text
```

Update the script footer:

```python
if __name__ == "__main__":
    test_filtered_subset_remaps_labels_and_caps_per_class()
    test_write_experiment_artifacts_creates_expected_files()
    test_small_experiment_smoke_run()
    test_entrypoint_script_and_gitignore_contract()
    print("test_small_modelnet_experiment.py: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
```

Expected: `FAIL` because `run_small_modelnet_experiment.py` does not exist yet and `.gitignore` does not contain `artifacts/`.

- [ ] **Step 3: Write minimal implementation**

Create `run_small_modelnet_experiment.py`:

```python
"""
run_small_modelnet_experiment.py - User entrypoint for the lightweight 5-class demo.
"""

from __future__ import annotations

from small_modelnet_experiment import SmallExperimentConfig, run_small_experiment


def main() -> None:
    config = SmallExperimentConfig()
    result = run_small_experiment(config)

    print("Small ModelNet experiment finished.")
    print(f"Artifacts: {result['artifact_dir']}")
    print(f"Overall accuracy: {result['metrics']['overall_accuracy']:.2f}%")
    print(f"Mean class accuracy: {result['metrics']['mean_class_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
```

Update `.gitignore`:

```gitignore
artifacts/
```

- [ ] **Step 4: Run tests to verify the feature**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_small_modelnet_experiment.py
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_modelnet40_data.py
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_method_alignment.py
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_came_net.py
```

Expected:

- `test_small_modelnet_experiment.py: PASS`
- `test_modelnet40_data.py: PASS`
- `test_method_alignment.py: PASS`
- `test_came_net.py: PASS`

- [ ] **Step 5: Commit**

Run:

```powershell
git add .gitignore run_small_modelnet_experiment.py small_modelnet_experiment.py test_small_modelnet_experiment.py
git commit -m "feat: add lightweight ModelNet experiment entrypoint"
```
