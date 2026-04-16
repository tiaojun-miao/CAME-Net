"""
test_small_modelnet_experiment.py - Lightweight experiment checks for ModelNet40.
"""

from __future__ import annotations

import tempfile
import sys
import unittest
from unittest import mock
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import small_modelnet_experiment as sme
from small_modelnet_experiment import (
    FilteredModelNetSubset,
    SmallExperimentConfig,
    create_experiment_artifacts,
    resolve_modelnet_root,
    run_small_experiment,
)


def test_entrypoint_script_and_gitignore_are_present():
    repo_root = Path(__file__).resolve().parent
    assert (repo_root / "run_small_modelnet_experiment.py").exists()
    gitignore_lines = [line.strip() for line in (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "/artifacts/" in gitignore_lines


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
    try:
        return resolve_modelnet_root(start_path=start_path or Path(__file__).resolve())
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise unittest.SkipTest(f"Real ModelNet40 data not available; skipping smoke run. ({exc})") from exc


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


def test_resolve_modelnet_root_finds_outer_repo_dataset_from_worktree():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        worktree_dir = repo_root / ".worktrees" / "small-modelnet-experiment"
        worktree_dir.mkdir(parents=True)
        fake_test_path = worktree_dir / "test_small_modelnet_experiment.py"
        fake_test_path.write_text("", encoding="utf-8")

        outer_modelnet_root = repo_root / "ModelNet40"
        (outer_modelnet_root / "airplane" / "train").mkdir(parents=True)

        resolved_root = resolve_modelnet_root(start_path=fake_test_path)

        assert resolved_root == outer_modelnet_root


def test_resolve_modelnet_root_rejects_missing_dataset_with_clear_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir) / "repo"
        worktree_dir = repo_root / ".worktrees" / "small-modelnet-experiment"
        worktree_dir.mkdir(parents=True)
        fake_test_path = worktree_dir / "test_small_modelnet_experiment.py"
        fake_test_path.write_text("", encoding="utf-8")

        try:
            resolve_modelnet_root(start_path=fake_test_path)
        except FileNotFoundError as exc:
            assert "modelnet40 root not found" in str(exc).lower()
        else:
            raise AssertionError("Expected missing dataset root error")


def test_run_small_experiment_ignores_stale_shared_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_root = Path(tmpdir)
        dataset_root = Path(tmpdir) / "ModelNet40"
        (dataset_root / "airplane" / "train").mkdir(parents=True)
        stale_checkpoint = artifact_root / "checkpoints" / "best_model.pth"
        stale_checkpoint.parent.mkdir(parents=True)
        stale_checkpoint.write_bytes(b"stale-checkpoint")

        class DummyRunModel:
            def to(self, device):
                return self

        train_dataset = type("TrainDataset", (), {"class_names": ["airplane", "chair"], "__len__": lambda self: 4})()
        test_dataset = type("TestDataset", (), {"class_names": ["airplane", "chair"], "__len__": lambda self: 2})()

        config = SmallExperimentConfig(
            data_root=str(dataset_root),
            class_names=("airplane", "chair"),
            num_epochs=1,
            artifact_root=str(artifact_root),
            device="cpu",
        )

        with mock.patch.object(sme, "build_small_experiment_datasets_and_loaders", return_value=(train_dataset, test_dataset, object(), object())) as build_mock, \
            mock.patch.object(sme, "CAMENet", return_value=DummyRunModel()), \
            mock.patch.object(sme, "train_came_net", return_value={"train_loss": [1.0], "train_acc": [50.0], "val_loss": [], "val_acc": [], "lr": [0.001]}) as train_mock, \
            mock.patch.object(sme, "load_checkpoint") as load_checkpoint_mock, \
            mock.patch.object(sme, "evaluate_subset_model", return_value={"overall_accuracy": 50.0, "mean_class_accuracy": 50.0, "per_class_accuracy": {"airplane": 50.0, "chair": 50.0}, "confusion_matrix": [[1, 0], [0, 1]], "class_names": ["airplane", "chair"]}), \
            mock.patch.object(sme, "collect_sample_predictions", return_value=[]), \
            mock.patch.object(sme, "create_experiment_artifacts", return_value=artifact_root / "artifacts" / "run"), \
            mock.patch.object(sme, "count_parameters", return_value=123):
            run_small_experiment(config)

        assert build_mock.call_args.kwargs["resolved_data_root"] == dataset_root
        checkpoint_dir = Path(train_mock.call_args.kwargs["checkpoint_dir"])
        assert checkpoint_dir.name == "checkpoints"
        assert checkpoint_dir.parent.parent.name == "runs"
        assert checkpoint_dir.parent.parent.parent == artifact_root
        assert not load_checkpoint_mock.called


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
    test_entrypoint_script_and_gitignore_are_present()
    test_filtered_subset_remaps_labels_and_caps_per_class()
    test_filtered_subset_reports_missing_and_short_classes()
    test_filtered_subset_rejects_duplicate_allowed_classes()
    test_filtered_subset_rejects_empty_allowed_classes()
    test_create_experiment_artifacts_writes_expected_files()
    test_create_experiment_artifacts_formats_dict_per_class_accuracy()
    test_create_experiment_artifacts_rejects_mismatched_confusion_matrix()
    test_resolve_modelnet_root_finds_outer_repo_dataset_from_worktree()
    test_resolve_modelnet_root_rejects_missing_dataset_with_clear_error()
    test_run_small_experiment_ignores_stale_shared_checkpoint()
    try:
        test_small_experiment_smoke_run()
    except unittest.SkipTest as exc:
        print(f"test_small_experiment_smoke_run: SKIPPED ({exc})")
    print("test_small_modelnet_experiment.py: PASS")
