"""
test_modelnet40_data.py - Dataset and loader checks for local ModelNet40 OFF meshes.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from data_utils import ModelNetDataset, collate_fn


def get_modelnet_root() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ModelNet40", "ModelNet40")


def test_modelnet_indexing():
    dataset = ModelNetDataset(
        data_dir=get_modelnet_root(),
        split="train",
        num_points=32,
        data_augmentation=False,
    )

    assert len(dataset) == 9843
    assert dataset.num_classes == 40
    assert dataset.class_names[0] == "airplane"
    assert dataset.samples[0][0].endswith(".off")


def test_modelnet_getitem_and_collate():
    dataset = ModelNetDataset(
        data_dir=get_modelnet_root(),
        split="train",
        num_points=16,
        data_augmentation=False,
    )

    sample = dataset[0]
    assert sample["point_coords"].shape == (16, 3)
    assert sample["point_coords"].dtype == torch.float32
    assert sample["labels"].dtype == torch.long

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    assert batch["point_coords"].shape == (2, 16, 3)
    assert batch["point_coords"].dtype == torch.float32
    assert batch["labels"].shape == (2,)
    assert batch["labels"].dtype == torch.long


def test_modelnet_validation_errors():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        missing_root = tmp_path / "missing"
        try:
            ModelNetDataset(data_dir=str(missing_root), split="train")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing root should raise FileNotFoundError")

        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("not a directory", encoding="utf-8")
        try:
            ModelNetDataset(data_dir=str(file_path), split="train")
        except NotADirectoryError:
            pass
        else:
            raise AssertionError("File path should raise NotADirectoryError")

        empty_root = tmp_path / "empty_root"
        (empty_root / "airplane").mkdir(parents=True)
        try:
            ModelNetDataset(data_dir=str(empty_root), split="train")
        except ValueError as exc:
            assert "No OFF files found" in str(exc)
        else:
            raise AssertionError("Empty split should raise ValueError")

    try:
        ModelNetDataset(data_dir=get_modelnet_root(), split="validation")
    except ValueError as exc:
        assert "Unsupported split" in str(exc)
    else:
        raise AssertionError("Unsupported split should raise ValueError")


if __name__ == "__main__":
    test_modelnet_indexing()
    test_modelnet_getitem_and_collate()
    test_modelnet_validation_errors()
    print("test_modelnet40_data.py: PASS")
