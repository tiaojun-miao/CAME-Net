# ModelNet40 Data Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local `ModelNet40` `OFF` mesh dataset trainable by the existing CAME-Net point-cloud pipeline without changing the paper method purpose.

**Architecture:** Keep the model and method unchanged. Implement a real `ModelNetDataset` that indexes local `OFF` files, parses meshes, samples fixed-size point clouds from triangle surfaces, normalizes them, and returns the existing `point_coords + labels` batch contract. Add one focused test script for dataset and loader behavior, then switch the example training entry point to the real dataset.

**Tech Stack:** Python, PyTorch, NumPy, local `OFF` mesh parsing, existing CAME-Net training utilities

---

### Task 1: Index Local ModelNet40 OFF Files

**Files:**
- Create: `test_modelnet40_data.py`
- Modify: `data_utils.py`
- Test: `test_modelnet40_data.py`

- [ ] **Step 1: Write the failing test**

Add this new test script:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_modelnet40_data.py
```

Expected: `FAIL` because `ModelNetDataset` is still a placeholder, so `len(dataset)` is `0` and metadata like `class_names` / `samples` do not exist.

- [ ] **Step 3: Write minimal implementation**

Update `data_utils.py` to replace the placeholder indexing logic with real file discovery:

```python
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
```

```python
class ModelNetDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_points: int = 1024,
        data_augmentation: bool = True,
        rotation_range: float = 0.5,
        translation_range: float = 0.3,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.data_augmentation = data_augmentation
        self.rotation_range = rotation_range
        self.translation_range = translation_range

        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"ModelNet root not found: {self.data_dir}")

        self.class_names = sorted(
            entry.name for entry in self.data_dir.iterdir() if entry.is_dir()
        )
        self.class_to_idx = {
            class_name: class_idx for class_idx, class_name in enumerate(self.class_names)
        }
        self.num_classes = len(self.class_names)
        self.samples: List[Tuple[str, int]] = self._index_samples()

        if not self.samples:
            raise ValueError(
                f"No OFF files found for split '{self.split}' under {self.data_dir}"
            )

    def _index_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for class_name in self.class_names:
            split_dir = self.data_dir / class_name / self.split
            if not split_dir.exists():
                continue
            for off_path in sorted(split_dir.glob("*.off")):
                samples.append((str(off_path), self.class_to_idx[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_modelnet40_data.py
```

Expected: `PASS` with `len(dataset) == 9843` and `num_classes == 40`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add data_utils.py test_modelnet40_data.py
git commit -m "feat: index local ModelNet40 OFF files"
```

### Task 2: Parse OFF Meshes and Sample Surface Point Clouds

**Files:**
- Modify: `data_utils.py`
- Modify: `test_modelnet40_data.py`
- Test: `test_modelnet40_data.py`

- [ ] **Step 1: Write the failing test**

Extend `test_modelnet40_data.py` with a sample-quality check:

```python
import torch
```

```python
def test_modelnet_test_sample_is_sampled_and_normalized():
    dataset = ModelNetDataset(
        data_dir=get_modelnet_root(),
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
```

Update the script footer:

```python
if __name__ == "__main__":
    test_modelnet_indexing()
    test_modelnet_test_sample_is_sampled_and_normalized()
    print("test_modelnet40_data.py: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_modelnet40_data.py
```

Expected: `FAIL` because `__getitem__` still tries to read nonexistent in-memory `point_clouds` instead of parsing and sampling from `OFF` meshes.

- [ ] **Step 3: Write minimal implementation**

Add mesh parsing, triangulation, surface sampling, deterministic test sampling, and normalized output in `data_utils.py`:

```python
class ModelNetDataset(Dataset):
    def _augment(self, points: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            points = PointCloudProcessor.random_rotation(points, self.rotation_range)
        if np.random.rand() < 0.5:
            points = PointCloudProcessor.random_translation(points, self.translation_range)
        return points

    @staticmethod
    def _load_off_mesh(off_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(off_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        if not lines or lines[0] != "OFF":
            raise ValueError(f"Invalid OFF header in {off_path}")

        try:
            num_vertices, num_faces, _ = map(int, lines[1].split())
        except ValueError as exc:
            raise ValueError(f"Invalid OFF counts line in {off_path}") from exc

        vertex_start = 2
        vertex_end = vertex_start + num_vertices
        face_end = vertex_end + num_faces

        vertices = np.asarray(
            [list(map(float, line.split())) for line in lines[vertex_start:vertex_end]],
            dtype=np.float32,
        )

        triangles: List[List[int]] = []
        for line in lines[vertex_end:face_end]:
            parts = list(map(int, line.split()))
            if not parts:
                continue
            face_degree = parts[0]
            face_indices = parts[1:1 + face_degree]
            if face_degree < 3:
                continue
            for offset in range(1, face_degree - 1):
                triangles.append(
                    [face_indices[0], face_indices[offset], face_indices[offset + 1]]
                )

        return vertices, np.asarray(triangles, dtype=np.int64)

    @staticmethod
    def _sample_vertices(vertices: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
        if len(vertices) == 0:
            raise ValueError("Cannot sample from an empty vertex array")
        indices = rng.choice(len(vertices), size=num_points, replace=len(vertices) < num_points)
        return vertices[indices].astype(np.float32)

    @staticmethod
    def _sample_surface_points(
        vertices: np.ndarray,
        triangles: np.ndarray,
        num_points: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if len(triangles) == 0:
            return ModelNetDataset._sample_vertices(vertices, num_points, rng)

        tri_vertices = vertices[triangles]
        edge_a = tri_vertices[:, 1] - tri_vertices[:, 0]
        edge_b = tri_vertices[:, 2] - tri_vertices[:, 0]
        areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
        valid_mask = areas > 1e-12

        if not np.any(valid_mask):
            return ModelNetDataset._sample_vertices(vertices, num_points, rng)

        tri_vertices = tri_vertices[valid_mask]
        areas = areas[valid_mask]
        triangle_indices = rng.choice(
            len(tri_vertices),
            size=num_points,
            p=areas / areas.sum(),
            replace=True,
        )
        chosen = tri_vertices[triangle_indices]

        u = rng.random(num_points, dtype=np.float32)
        v = rng.random(num_points, dtype=np.float32)
        sqrt_u = np.sqrt(u)
        bary_a = 1.0 - sqrt_u
        bary_b = sqrt_u * (1.0 - v)
        bary_c = sqrt_u * v

        points = (
            bary_a[:, None] * chosen[:, 0]
            + bary_b[:, None] * chosen[:, 1]
            + bary_c[:, None] * chosen[:, 2]
        )
        return points.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        off_path, label = self.samples[idx]
        rng_seed = idx if self.split == "test" else None
        rng = np.random.default_rng(rng_seed)

        vertices, triangles = self._load_off_mesh(off_path)
        points = self._sample_surface_points(vertices, triangles, self.num_points, rng)
        points = PointCloudProcessor.normalize(points)

        if self.data_augmentation and self.split == "train":
            points = self._augment(points)

        return {
            "point_coords": torch.from_numpy(points.astype(np.float32)),
            "labels": torch.tensor(label, dtype=torch.long),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_modelnet40_data.py
```

Expected: `PASS`, including deterministic repeated access on the `test` split and normalized point radii within `1.0001`.

- [ ] **Step 5: Commit**

Run:

```powershell
git add data_utils.py test_modelnet40_data.py
git commit -m "feat: sample point clouds from ModelNet40 meshes"
```

### Task 3: Wire Real ModelNet40 Loaders Into the Training Entry Point

**Files:**
- Modify: `train.py`
- Modify: `test_modelnet40_data.py`
- Test: `test_modelnet40_data.py`

- [ ] **Step 1: Write the failing test**

Extend `test_modelnet40_data.py` with a loader and model smoke test:

```python
from came_net import CAMENet
from train import create_default_modelnet_dataloaders
```

```python
def test_default_modelnet_loaders_and_forward_smoke():
    train_loader, val_loader = create_default_modelnet_dataloaders(
        data_root=get_modelnet_root(),
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
```

Update the script footer:

```python
if __name__ == "__main__":
    test_modelnet_indexing()
    test_modelnet_test_sample_is_sampled_and_normalized()
    test_default_modelnet_loaders_and_forward_smoke()
    print("test_modelnet40_data.py: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_modelnet40_data.py
```

Expected: `FAIL` with `ImportError` or `AttributeError` because `create_default_modelnet_dataloaders` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a small loader factory in `train.py` and switch the example entry point to it:

```python
def create_default_modelnet_dataloaders(
    data_root: str = "ModelNet40/ModelNet40",
    num_points: int = 1024,
    batch_size: int = 8,
):
    from data_utils import ModelNetDataset, collate_fn

    train_dataset = ModelNetDataset(
        data_dir=data_root,
        split="train",
        num_points=num_points,
        data_augmentation=True,
    )
    val_dataset = ModelNetDataset(
        data_dir=data_root,
        split="test",
        num_points=num_points,
        data_augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
```

Replace the `__main__` dataset block with:

```python
if __name__ == "__main__":
    from came_net import CAMENet, count_parameters
    from equiv_loss import equivariance_loss_efficient

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CAMENet(
        num_classes=40,
        point_feature_dim=0,
        num_layers=4,
        num_heads=8,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    train_loader, val_loader = create_default_modelnet_dataloaders(
        data_root="ModelNet40/ModelNet40",
        num_points=1024,
        batch_size=8,
    )

    history = train_came_net(
        model,
        train_loader,
        val_loader,
        num_epochs=20,
        learning_rate=1e-3,
        device=device,
        equiv_loss_weight=0.1,
        equiv_loss_fn=equivariance_loss_efficient,
        equiv_warmup_steps=1000,
        print_interval=2,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_modelnet40_data.py
```

Expected: `PASS`, with two real-data batches producing `(2, 40)` logits from the current model.

- [ ] **Step 5: Commit**

Run:

```powershell
git add train.py test_modelnet40_data.py
git commit -m "feat: wire ModelNet40 loaders into training entry point"
```

### Task 4: Final Verification on the Isolated Branch

**Files:**
- Modify: none
- Test: `test_modelnet40_data.py`
- Test: `test_method_alignment.py`
- Test: `test_came_net.py`

- [ ] **Step 1: Run the new dataset compatibility test**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_modelnet40_data.py
```

Expected: `PASS`

- [ ] **Step 2: Run the method-alignment regression suite**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_method_alignment.py
```

Expected: `method alignment tests passed`

- [ ] **Step 3: Run the broader regression suite**

Run:

```powershell
& 'D:\projectcreating\anaconda\envs\pytorch\python.exe' test_came_net.py
```

Expected: `ALL TESTS PASSED SUCCESSFULLY!`

- [ ] **Step 4: Check branch state**

Run:

```powershell
git status --short
```

Expected: clean working tree on branch `modelnet40-data-adapter`
