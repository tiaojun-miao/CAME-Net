"""
test_small_modelnet_experiment.py - Lightweight experiment checks for ModelNet40.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from small_modelnet_experiment import FilteredModelNetSubset, create_experiment_artifacts


class DummyModelNetDataset:
    def __init__(self, samples, class_names):
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, label = self.samples[idx]
        return {
            "point_coords": torch.tensor([[float(idx), 0.0, 0.0]], dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def test_filtered_subset_remaps_labels_and_caps_per_class():
    base_dataset = DummyModelNetDataset(
        samples=[
            ("sample_0.off", 0),
            ("sample_1.off", 3),
            ("sample_2.off", 0),
            ("sample_3.off", 1),
            ("sample_4.off", 1),
            ("sample_5.off", 0),
            ("sample_6.off", 4),
        ],
        class_names=["airplane", "chair", "lamp", "sofa", "toilet"],
    )

    subset = FilteredModelNetSubset(
        base_dataset=base_dataset,
        allowed_classes=["airplane", "chair"],
        max_samples_per_class=2,
    )

    assert subset.class_names == ["airplane", "chair"]
    assert subset.num_classes == 2
    assert len(subset) == 4
    assert [int(subset[i]["point_coords"][0, 0].item()) for i in range(len(subset))] == [0, 2, 3, 4]
    assert [int(subset[i]["labels"].item()) for i in range(len(subset))] == [0, 0, 1, 1]


def test_filtered_subset_reports_missing_and_short_classes():
    base_dataset = DummyModelNetDataset(
        samples=[
            ("sample_0.off", 0),
            ("sample_1.off", 0),
            ("sample_2.off", 1),
        ],
        class_names=["airplane", "chair"],
    )

    try:
        FilteredModelNetSubset(
            base_dataset=base_dataset,
            allowed_classes=["airplane", "table"],
            max_samples_per_class=1,
        )
    except ValueError as exc:
        assert "Missing classes" in str(exc)
        assert "table" in str(exc)
    else:
        raise AssertionError("Expected missing class error")

    try:
        FilteredModelNetSubset(
            base_dataset=base_dataset,
            allowed_classes=["airplane", "chair"],
            max_samples_per_class=2,
        )
    except ValueError as exc:
        assert "Not enough samples" in str(exc)
        assert "chair" in str(exc)
    else:
        raise AssertionError("Expected insufficient samples error")


def test_filtered_subset_rejects_duplicate_allowed_classes():
    base_dataset = DummyModelNetDataset(
        samples=[
            ("sample_0.off", 0),
            ("sample_1.off", 1),
        ],
        class_names=["airplane", "chair"],
    )

    try:
        FilteredModelNetSubset(
            base_dataset=base_dataset,
            allowed_classes=["airplane", "airplane"],
            max_samples_per_class=1,
        )
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate class error")


def test_filtered_subset_rejects_empty_allowed_classes():
    base_dataset = DummyModelNetDataset(
        samples=[
            ("sample_0.off", 0),
            ("sample_1.off", 1),
        ],
        class_names=["airplane", "chair"],
    )

    try:
        FilteredModelNetSubset(
            base_dataset=base_dataset,
            allowed_classes=[],
            max_samples_per_class=1,
        )
    except ValueError as exc:
        assert "empty" in str(exc).lower() or "at least one" in str(exc).lower()
    else:
        raise AssertionError("Expected empty class list error")


def test_create_experiment_artifacts_writes_expected_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = create_experiment_artifacts(
            artifact_root=tmpdir,
            config={
                "class_names": ["airplane", "chair"],
                "num_epochs": 2,
            },
            history={
                "train_loss": [1.0, 0.5],
                "train_acc": [25.0, 75.0],
                "val_loss": [1.2, 0.7],
                "val_acc": [20.0, 70.0],
                "lr": [0.001, 0.0005],
            },
            metrics={
                "overall_accuracy": 72.5,
                "per_class_accuracy": [80.0, 65.0],
            },
            selected_classes=["airplane", "chair"],
            dataset_sizes={"train": 4, "val": 2, "test": 2},
            runtime_seconds=12.3,
            sample_predictions=[
                {
                    "point_coords": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                    "predicted_label": 0,
                    "true_label": 1,
                    "class_names": ["airplane", "chair"],
                }
            ],
        )

        artifact_path = Path(artifact_dir)
        assert artifact_path.exists()
        expected_files = {
            "config.json",
            "history.json",
            "metrics.json",
            "training_curves.png",
            "confusion_matrix.png",
            "sample_predictions.png",
            "summary.md",
        }
        assert expected_files.issubset({path.name for path in artifact_path.iterdir()})

        summary_text = (artifact_path / "summary.md").read_text(encoding="utf-8").lower()
        assert "selected classes" in summary_text
        assert "airplane" in summary_text
        assert "chair" in summary_text
        assert "dataset sizes" in summary_text
        assert "runtime" in summary_text
        assert "overall accuracy" in summary_text
        assert "per-class accuracy" in summary_text
        assert "training_curves.png" in summary_text


if __name__ == "__main__":
    test_filtered_subset_remaps_labels_and_caps_per_class()
    test_filtered_subset_reports_missing_and_short_classes()
    test_filtered_subset_rejects_duplicate_allowed_classes()
    test_filtered_subset_rejects_empty_allowed_classes()
    test_create_experiment_artifacts_writes_expected_files()
    print("test_small_modelnet_experiment.py: PASS")
