"""
data_utils.py - Data Loading and Preprocessing Utilities

This module provides data loading and preprocessing utilities for the CAME-Net model,
including point cloud datasets, augmentation, and collation functions.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class RandomPointCloudDataset(Dataset):
    """
    Random Point Cloud Dataset for testing CAME-Net.

    Generates random point clouds with optional labels for classification tasks.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_points: int = 1024,
        num_classes: int = 40,
        data_augmentation: bool = True,
        rotation_range: float = 0.5,
        translation_range: float = 0.3
    ):
        """
        Initialize RandomPointCloudDataset.

        Args:
            num_samples: Number of samples in the dataset
            num_points: Number of points per point cloud
            num_classes: Number of classes for classification
            data_augmentation: Whether to apply data augmentation
            rotation_range: Range for random rotation (in radians)
            translation_range: Range for random translation
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes
        self.data_augmentation = data_augmentation
        self.rotation_range = rotation_range
        self.translation_range = translation_range

        self.point_clouds = []
        self.labels = []

        self._generate_data()

    def _generate_data(self):
        """Generate random point cloud data."""
        for i in range(self.num_samples):
            label = i % self.num_classes

            points = np.random.randn(self.num_points, 3).astype(np.float32)

            points = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-8)

            scale = np.random.uniform(0.8, 1.2)
            points = points * scale

            self.point_clouds.append(points)
            self.labels.append(label)

    def _augment(self, points: np.ndarray) -> np.ndarray:
        """Apply random rotation and translation augmentation."""
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            points = points @ rotation_matrix.T

        if np.random.rand() < 0.5:
            translation = np.random.uniform(
                -self.translation_range,
                self.translation_range,
                size=3
            ).astype(np.float32)
            points = points + translation

        return points

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing 'point_coords' and 'labels'
        """
        points = self.point_clouds[idx].copy()
        label = self.labels[idx]

        if self.data_augmentation:
            points = self._augment(points)

        return {
            'point_coords': torch.from_numpy(points),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModelNetDataset(Dataset):
    """
    ModelNet Dataset for point cloud classification.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_points: int = 1024,
        data_augmentation: bool = True,
        rotation_range: float = 0.5,
        translation_range: float = 0.3
    ):
        """
        Initialize ModelNetDataset.

        Args:
            data_dir: Directory containing ModelNet data
            split: 'train' or 'test' split
            num_points: Number of points to sample per object
            data_augmentation: Whether to apply data augmentation
        """
        if split not in {'train', 'test'}:
            raise ValueError(f"Unsupported split: {split}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.data_augmentation = data_augmentation
        self.rotation_range = rotation_range
        self.translation_range = translation_range

        if not self.data_dir.exists():
            raise FileNotFoundError(f"ModelNet root not found: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"ModelNet root is not a directory: {self.data_dir}")

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
            for off_path in sorted(split_dir.glob('*.off')):
                samples.append((str(off_path), self.class_to_idx[class_name]))
        return samples

    def _augment(self, points: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            points = PointCloudProcessor.random_rotation(points, self.rotation_range)
        if np.random.rand() < 0.5:
            points = PointCloudProcessor.random_translation(points, self.translation_range)
        return points

    @staticmethod
    def _load_off_mesh(off_path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(off_path, 'r', encoding='utf-8') as handle:
            lines = [line.strip() for line in handle if line.strip()]

        if not lines or not lines[0].startswith('OFF'):
            raise ValueError(f"Invalid OFF header in {off_path}")

        first_line = lines[0]
        counts_text = first_line[3:].strip()
        vertex_start = 1
        if not counts_text:
            if len(lines) < 2:
                raise ValueError(f"Invalid OFF counts line in {off_path}")
            counts_text = lines[1]
            vertex_start = 2

        try:
            num_vertices, num_faces, _ = map(int, counts_text.split()[:3])
        except ValueError as exc:
            raise ValueError(f"Invalid OFF counts line in {off_path}") from exc

        vertex_end = vertex_start + num_vertices
        face_end = vertex_end + num_faces

        if len(lines) < face_end:
            raise ValueError(f"OFF file is missing mesh data: {off_path}")

        vertices: List[List[float]] = []
        for vertex_line in lines[vertex_start:vertex_end]:
            parts = vertex_line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed OFF vertex row in {off_path}: {vertex_line}")
            try:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError as exc:
                raise ValueError(f"Malformed OFF vertex row in {off_path}: {vertex_line}") from exc

        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.shape != (num_vertices, 3):
            raise ValueError(f"OFF vertices must have shape (N, 3): {off_path}")

        triangles: List[List[int]] = []
        for face_line in lines[vertex_end:face_end]:
            parts = face_line.split()
            if not parts:
                raise ValueError(f"Malformed OFF face row in {off_path}: {face_line}")

            try:
                face_values = [int(value) for value in parts]
            except ValueError as exc:
                raise ValueError(f"Malformed OFF face row in {off_path}: {face_line}") from exc

            face_degree = face_values[0]
            if face_degree < 3:
                raise ValueError(f"OFF face must have at least 3 vertices in {off_path}: {face_line}")
            if len(face_values) < face_degree + 1:
                raise ValueError(f"Malformed OFF face row in {off_path}: {face_line}")

            face_indices = face_values[1:1 + face_degree]
            if min(face_indices) < 0 or max(face_indices) >= num_vertices:
                raise ValueError(f"OFF face index out of bounds in {off_path}: {face_line}")

            for offset in range(1, face_degree - 1):
                triangles.append(
                    [face_indices[0], face_indices[offset], face_indices[offset + 1]]
                )

        return vertices, np.asarray(triangles, dtype=np.int64).reshape(-1, 3)

    @staticmethod
    def _sample_vertices(vertices: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
        if len(vertices) == 0:
            raise ValueError("Cannot sample from an empty vertex array")

        indices = rng.choice(len(vertices), size=num_points, replace=len(vertices) < num_points)
        return vertices[indices].astype(np.float32, copy=False)

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

        u = rng.random(num_points).astype(np.float32)
        v = rng.random(num_points).astype(np.float32)
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

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        off_path, label = self.samples[idx]
        rng_seed = idx if self.split == 'test' else None
        rng = np.random.default_rng(rng_seed)

        vertices, triangles = self._load_off_mesh(off_path)
        points = self._sample_surface_points(vertices, triangles, self.num_points, rng)
        points = PointCloudProcessor.normalize(points)

        if self.data_augmentation and self.split == 'train':
            points = self._augment(points)

        return {
            'point_coords': torch.from_numpy(points.astype(np.float32)),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    """
    Custom collate function for batching point cloud samples.

    Args:
        batch: List of samples from the dataset

    Returns:
        Batched dictionary containing 'point_coords' and 'labels'
    """
    point_coords = torch.stack([item['point_coords'] for item in batch], dim=0)
    labels = torch.stack([item['labels'] for item in batch], dim=0)

    return {
        'point_coords': point_coords,
        'labels': labels
    }


class PointCloudProcessor:
    """
    Utility class for point cloud preprocessing and augmentation.
    """

    @staticmethod
    def normalize(points: np.ndarray) -> np.ndarray:
        """
        Normalize point cloud to unit sphere.

        Args:
            points: Point cloud of shape (N, 3)

        Returns:
            Normalized point cloud
        """
        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        points = points / (max_dist + 1e-8)
        return points

    @staticmethod
    def random_rotation(points: np.ndarray, angle_range: float = 0.5) -> np.ndarray:
        """
        Apply random rotation around z-axis.

        Args:
            points: Point cloud of shape (N, 3)
            angle_range: Range for random rotation

        Returns:
            Rotated point cloud
        """
        angle = np.random.uniform(-angle_range, angle_range)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        return points @ rotation_matrix.T

    @staticmethod
    def random_translation(points: np.ndarray, translation_range: float = 0.3) -> np.ndarray:
        """
        Apply random translation.

        Args:
            points: Point cloud of shape (N, 3)
            translation_range: Maximum translation distance

        Returns:
            Translated point cloud
        """
        translation = np.random.uniform(
            -translation_range,
            translation_range,
            size=3
        ).astype(np.float32)
        return points + translation

    @staticmethod
    def random_jitter(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
        """
        Apply random jitter to point positions.

        Args:
            points: Point cloud of shape (N, 3)
            sigma: Standard deviation of Gaussian noise
            clip: Maximum value for clipping

        Returns:
            Jittered point cloud
        """
        jitter = np.clip(
            sigma * np.random.randn(*points.shape),
            -clip,
            clip
        ).astype(np.float32)
        return points + jitter

    @staticmethod
    def shuffle_points(points: np.ndarray) -> np.ndarray:
        """
        Randomly shuffle point order.

        Args:
            points: Point cloud of shape (N, 3)

        Returns:
            Shuffled point cloud
        """
        indices = np.random.permutation(points.shape[0])
        return points[indices]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for point cloud datasets.

    Args:
        dataset: Dataset to load
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
