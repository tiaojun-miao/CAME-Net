"""
test_small_modelnet_experiment.py - Lightweight experiment checks for ModelNet40.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from small_modelnet_experiment import FilteredModelNetSubset


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


if __name__ == "__main__":
    test_filtered_subset_remaps_labels_and_caps_per_class()
    test_filtered_subset_reports_missing_and_short_classes()
    print("test_small_modelnet_experiment.py: PASS")
