"""
test_modelnet40_data.py - Dataset and loader checks for local ModelNet40 OFF meshes.
"""

import os
import sys
import tempfile
from unittest import SkipTest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from came_net import CAMENet
from data_utils import ModelNetDataset, collate_fn
from train import create_default_modelnet_dataloaders


def get_modelnet_root() -> str:
    return os.environ.get(
        "MODELNET40_ROOT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ModelNet40", "ModelNet40"),
    )


def require_modelnet_root(root: str) -> str:
    if not Path(root).exists():
        raise SkipTest(f"ModelNet40 dataset not found at {root}")
    return root


def test_modelnet_indexing():
    root = require_modelnet_root(get_modelnet_root())
    dataset = ModelNetDataset(
        data_dir=root,
        split="train",
        num_points=32,
        data_augmentation=False,
    )

    assert len(dataset) == 9843
    assert dataset.num_classes == 40
    assert dataset.class_names[0] == "airplane"
    assert dataset.samples[0][0].endswith(".off")


def test_modelnet_getitem_and_collate():
    root = require_modelnet_root(get_modelnet_root())
    dataset = ModelNetDataset(
        data_dir=root,
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


def test_modelnet_train_sample_is_deterministic_without_augmentation():
    root = require_modelnet_root(get_modelnet_root())
    dataset = ModelNetDataset(
        data_dir=root,
        split="train",
        num_points=128,
        data_augmentation=False,
    )

    sample_a = dataset[0]
    sample_b = dataset[0]

    assert torch.allclose(sample_a["point_coords"], sample_b["point_coords"])
    assert sample_a["labels"].item() == sample_b["labels"].item()


def test_modelnet_test_sample_is_sampled_and_normalized():
    root = require_modelnet_root(get_modelnet_root())
    dataset = ModelNetDataset(
        data_dir=root,
        split="test",
        num_points=128,
        data_augmentation=False,
    )

    sample_a = dataset[0]
    sample_b = dataset[0]

    assert sample_a["point_coords"].shape == (128, 3)
    assert sample_a["point_coords"].dtype == torch.float32
    assert sample_a["labels"].dtype == torch.long
    assert torch.allclose(sample_a["point_coords"], sample_b["point_coords"])
    assert sample_a["point_coords"].mean(dim=0).abs().max().item() < 1e-3
    assert torch.linalg.norm(sample_a["point_coords"], dim=1).max().item() <= 1.0001


def test_modelnet_loads_inline_header_off_mesh():
    root = Path(require_modelnet_root(get_modelnet_root()))
    off_path = root / "bathtub" / "test" / "bathtub_0107.off"

    vertices, triangles = ModelNetDataset._load_off_mesh(str(off_path))

    assert vertices.shape == (1568, 3)
    assert triangles.shape == (1820, 3)
    assert vertices.dtype == np.float32
    assert triangles.dtype == np.int64
    assert triangles.min().item() >= 0
    assert triangles.max().item() < vertices.shape[0]


def test_modelnet_rejects_malformed_face_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        off_path = Path(tmpdir) / "bad_face.off"
        off_path.write_text(
            "\n".join(
                [
                    "OFF",
                    "4 1 0",
                    "0 0 0",
                    "1 0 0",
                    "0 1 0",
                    "0 0 1",
                    "3 0 1",
                ]
            ),
            encoding="utf-8",
        )

        try:
            ModelNetDataset._load_off_mesh(str(off_path))
        except ValueError as exc:
            assert "Malformed OFF face row" in str(exc)
        else:
            raise AssertionError("Malformed face row should raise ValueError")


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
            ModelNetDataset(data_dir=str(tmp_path), split="validation")
        except ValueError as exc:
            assert "Unsupported split" in str(exc)
        else:
            raise AssertionError("Unsupported split should raise ValueError")


def test_default_modelnet_loaders_and_forward_smoke():
    root = require_modelnet_root(get_modelnet_root())
    train_loader, val_loader = create_default_modelnet_dataloaders(
        data_root=root,
        num_points=64,
        batch_size=2,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    model = CAMENet(num_classes=40, point_feature_dim=0, num_layers=2, num_heads=4)

    train_logits = model(point_coords=train_batch["point_coords"])
    val_logits = model(point_coords=val_batch["point_coords"])

    assert train_batch["point_coords"].shape == (2, 64, 3)
    assert val_batch["point_coords"].shape == (2, 64, 3)
    assert train_logits.shape == (2, 40)
    assert val_logits.shape == (2, 40)


def test_default_modelnet_loader_default_root_is_independent_of_cwd():
    require_modelnet_root(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ModelNet40", "ModelNet40"))
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            train_loader, val_loader = create_default_modelnet_dataloaders(
                num_points=64,
                batch_size=2,
            )
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
        finally:
            os.chdir(original_cwd)

    assert train_batch["point_coords"].shape == (2, 64, 3)
    assert val_batch["point_coords"].shape == (2, 64, 3)


if __name__ == "__main__":
    tests = [
        test_modelnet_indexing,
        test_modelnet_getitem_and_collate,
        test_modelnet_train_sample_is_deterministic_without_augmentation,
        test_modelnet_test_sample_is_sampled_and_normalized,
        test_modelnet_loads_inline_header_off_mesh,
        test_modelnet_rejects_malformed_face_rows,
        test_modelnet_validation_errors,
        test_default_modelnet_loaders_and_forward_smoke,
        test_default_modelnet_loader_default_root_is_independent_of_cwd,
    ]

    skipped = 0
    for test in tests:
        try:
            test()
        except SkipTest as exc:
            skipped += 1
            print(f"{test.__name__}: SKIP ({exc})")
        else:
            print(f"{test.__name__}: PASS")

    if skipped:
        print(f"test_modelnet40_data.py: PASS ({skipped} skipped)")
    else:
        print("test_modelnet40_data.py: PASS")
