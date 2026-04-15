"""
test_modelnet40_data.py - Dataset and loader checks for local ModelNet40 OFF meshes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import ModelNetDataset


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


if __name__ == "__main__":
    test_modelnet_indexing()
    print("test_modelnet40_data.py: PASS")
