"""
test_small_modelnet_experiment.py - Lightweight experiment checks for ModelNet40.
"""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from small_modelnet_experiment import (
    FilteredModelNetSubset,
    SmallExperimentConfig,
    create_experiment_artifacts,
    run_small_experiment,
)


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


def require_modelnet_root(start_path: Path | None = None) -> Path:
    base_path = start_path or Path(__file__).resolve()
    parents = list(base_path.parents)
    worktree_root = parents[1] if len(parents) > 1 else base_path.parent
    repo_root = parents[2] if len(parents) > 2 else worktree_root.parent
    candidates = [
        worktree_root / "ModelNet40" / "ModelNet40",
        worktree_root / "ModelNet40",
        repo_root / "ModelNet40" / "ModelNet40",
        repo_root / "ModelNet40",
    ]
    for candidate in candidates:
        if candidate.exists() and any(path.is_dir() for path in candidate.iterdir()):
            return candidate
    raise unittest.SkipTest("Real ModelNet40 data not available; skipping smoke run.")


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


def test_create_experiment_artifacts_formats_dict_per_class_accuracy():
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
                "per_class_accuracy": {
                    "airplane": 80.0,
                    "chair": 65.0,
                },
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

        summary_text = (Path(artifact_dir) / "summary.md").read_text(encoding="utf-8")
        assert "- airplane: 80.0" in summary_text
        assert "- chair: 65.0" in summary_text
        assert "Class 0: airplane" not in summary_text
        assert "Class 1: chair" not in summary_text


def test_create_experiment_artifacts_rejects_mismatched_confusion_matrix():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            create_experiment_artifacts(
                artifact_root=tmpdir,
                config={
                    "class_names": ["airplane", "chair", "lamp", "sofa", "toilet"],
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
                    "per_class_accuracy": [80.0, 65.0, 70.0, 75.0, 72.0],
                },
                selected_classes=["airplane", "chair", "lamp", "sofa", "toilet"],
                dataset_sizes={"train": 10, "val": 5, "test": 5},
                runtime_seconds=12.3,
                sample_predictions=[
                    {
                        "point_coords": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                        "predicted_label": 0,
                        "true_label": 1,
                        "class_names": ["airplane", "chair", "lamp", "sofa", "toilet"],
                    }
                ],
                confusion_matrix=[[0 for _ in range(40)] for _ in range(40)],
            )
        except ValueError as exc:
            error_text = str(exc).lower()
            assert "confusion" in error_text
            assert "selected_classes" in error_text or "selected classes" in error_text
        else:
            raise AssertionError("Expected mismatched confusion matrix error")


def test_require_modelnet_root_finds_outer_repo_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        worktree_dir = repo_root / ".worktrees" / "small-modelnet-experiment"
        worktree_dir.mkdir(parents=True)
        fake_test_path = worktree_dir / "test_small_modelnet_experiment.py"
        fake_test_path.write_text("", encoding="utf-8")

        outer_modelnet_root = repo_root / "ModelNet40"
        (outer_modelnet_root / "airplane" / "train").mkdir(parents=True)

        resolved_root = require_modelnet_root(fake_test_path)

        assert resolved_root == outer_modelnet_root


def test_small_experiment_smoke_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SmallExperimentConfig(
            data_root=str(require_modelnet_root()),
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


if __name__ == "__main__":
    test_filtered_subset_remaps_labels_and_caps_per_class()
    test_filtered_subset_reports_missing_and_short_classes()
    test_filtered_subset_rejects_duplicate_allowed_classes()
    test_filtered_subset_rejects_empty_allowed_classes()
    test_create_experiment_artifacts_writes_expected_files()
    test_create_experiment_artifacts_formats_dict_per_class_accuracy()
    test_create_experiment_artifacts_rejects_mismatched_confusion_matrix()
    test_require_modelnet_root_finds_outer_repo_dataset()
    try:
        test_small_experiment_smoke_run()
    except unittest.SkipTest as exc:
        print(f"test_small_experiment_smoke_run: SKIPPED ({exc})")
    print("test_small_modelnet_experiment.py: PASS")
