from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import io
import json
import struct

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_SCANNET_FILE_SUFFIXES = (
    ".aggregation.json",
    ".sens",
    ".txt",
    "_vh_clean_2.ply",
)


@dataclass
class ScanNetSceneConfig:
    data_root: str
    num_points: int = 256
    max_frames: int = 3
    frame_resize: int = 32
    min_label_frequency: int = 1
    top_k_labels: Optional[int] = None
    require_all_modalities: bool = True
    vocabulary_scene_ids: Optional[Sequence[str]] = None


def _resolve_scannet_root(data_root: str) -> Path:
    root = Path(data_root)
    if not root.exists():
        raise ValueError(f"ScanNet data root does not exist: {data_root}")
    if (root / "scans").is_dir():
        return root
    if root.name == "scans" and root.is_dir():
        return root.parent
    raise ValueError(f"ScanNet data root must contain a 'scans' directory: {data_root}")


def count_scene_label_frequencies(entries: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in entries:
        for label in entry["labels"]:
            counts[label] = counts.get(label, 0) + 1
    return counts


def count_scene_label_frequencies_for_scene_ids(
    entries: Sequence[Dict[str, object]],
    scene_ids: Sequence[str],
) -> Dict[str, int]:
    selected = set(scene_ids)
    filtered_entries = [entry for entry in entries if entry["scene_id"] in selected]
    return count_scene_label_frequencies(filtered_entries)


def select_label_vocabulary(
    label_counts: Dict[str, int],
    *,
    min_label_frequency: int,
    top_k_labels: Optional[int],
) -> List[str]:
    ranked = [
        label
        for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_label_frequency
    ]
    if top_k_labels is not None:
        ranked = ranked[:top_k_labels]
    return sorted(ranked)


def split_public_holdout_scene_ids(
    scene_ids: Sequence[str],
    *,
    blind_holdout_fraction: float = 0.2,
) -> Tuple[List[str], List[str]]:
    ordered = sorted(scene_ids)
    if len(ordered) < 5:
        return ordered, []
    holdout_count = max(1, int(round(len(ordered) * blind_holdout_fraction)))
    holdout_count = min(holdout_count, max(1, len(ordered) - 3))
    public = ordered[:-holdout_count]
    holdout = ordered[-holdout_count:]
    return public, holdout


class ScanNetSceneDataset(Dataset):
    def __init__(self, config: ScanNetSceneConfig):
        self.config = config
        self.root = _resolve_scannet_root(config.data_root)
        self.scans_dir = self.root / "scans"
        self.dataset_report = {
            "discovered_scenes": 0,
            "retained_scenes": 0,
            "skipped_scenes": 0,
            "skip_reasons": {},
            "scene_ids": [],
            "label_frequencies": {},
            "vocabulary_source_scene_ids": [],
            "vocabulary_label_frequencies": {},
        }
        self._image_cache: Dict[str, torch.Tensor] = {}
        self._point_cache: Dict[str, torch.Tensor] = {}

        raw_entries = self._discover_scene_entries()
        self.vocabulary_source_scene_ids = self._resolve_vocabulary_source_scene_ids(raw_entries)
        self.dataset_report["vocabulary_source_scene_ids"] = self.vocabulary_source_scene_ids
        self.dataset_report["vocabulary_label_frequencies"] = count_scene_label_frequencies_for_scene_ids(
            raw_entries,
            self.vocabulary_source_scene_ids,
        )
        self.label_vocabulary = self._build_label_vocabulary(raw_entries, self.vocabulary_source_scene_ids)
        self.scenes = self._finalize_scene_entries(raw_entries)
        self.dataset_report["retained_scenes"] = len(self.scenes)
        self.dataset_report["scene_ids"] = [entry["scene_id"] for entry in self.scenes]
        self.dataset_report["label_frequencies"] = count_scene_label_frequencies(self.scenes)

        if not self.scenes:
            raise ValueError("No valid ScanNet scenes found after filtering.")

    def _record_skip(self, reason: str) -> None:
        self.dataset_report["skipped_scenes"] += 1
        self.dataset_report["skip_reasons"][reason] = self.dataset_report["skip_reasons"].get(reason, 0) + 1

    def _required_paths(self, scene_dir: Path) -> Dict[str, Path]:
        scene_id = scene_dir.name
        suffix_to_key = {
            ".aggregation.json": "aggregation",
            ".txt": "metadata",
            ".sens": "sens",
            "_vh_clean_2.ply": "mesh",
        }
        return {suffix_to_key[suffix]: scene_dir / f"{scene_id}{suffix}" for suffix in REQUIRED_SCANNET_FILE_SUFFIXES}

    def _discover_scene_entries(self) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        for scene_dir in sorted(path for path in self.scans_dir.iterdir() if path.is_dir()):
            self.dataset_report["discovered_scenes"] += 1
            required = self._required_paths(scene_dir)
            missing = [name for name, path in required.items() if not path.exists()]
            if missing:
                self._record_skip("missing_required_files")
                continue

            labels = self._load_scene_labels(scene_dir)
            if not labels:
                self._record_skip("empty_label_set")
                continue

            try:
                self._load_point_tensor(scene_dir)
            except Exception:
                self._record_skip("mesh_decode_failed")
                continue

            try:
                self._load_image_tensor(scene_dir)
            except Exception:
                self._record_skip("sens_decode_failed")
                continue

            entries.append(
                {
                    "scene_id": scene_dir.name,
                    "scene_dir": scene_dir,
                    "labels": labels,
                    "text_prompt": self._build_text_prompt(scene_dir, labels),
                }
            )
        return entries

    def _resolve_vocabulary_source_scene_ids(self, entries: List[Dict[str, object]]) -> List[str]:
        discovered_scene_ids = sorted(entry["scene_id"] for entry in entries)
        if self.config.vocabulary_scene_ids is None:
            return discovered_scene_ids

        allowed = set(discovered_scene_ids)
        filtered = sorted(scene_id for scene_id in self.config.vocabulary_scene_ids if scene_id in allowed)
        if not filtered:
            raise ValueError("No valid ScanNet scenes remain after applying vocabulary_scene_ids.")
        return filtered

    def _build_label_vocabulary(self, entries: List[Dict[str, object]], vocabulary_scene_ids: Sequence[str]) -> List[str]:
        counts = count_scene_label_frequencies_for_scene_ids(entries, vocabulary_scene_ids)
        return select_label_vocabulary(
            counts,
            min_label_frequency=self.config.min_label_frequency,
            top_k_labels=self.config.top_k_labels,
        )

    def _finalize_scene_entries(self, entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
        vocabulary = set(self.label_vocabulary)
        finalized: List[Dict[str, object]] = []
        for entry in entries:
            filtered = [label for label in entry["labels"] if label in vocabulary]
            if not filtered:
                self._record_skip("labels_filtered_out")
                continue
            updated = dict(entry)
            updated["labels"] = filtered
            updated["label_targets"] = self._encode_targets(filtered)
            finalized.append(updated)
        return finalized

    def _load_scene_labels(self, scene_dir: Path) -> List[str]:
        aggregation_path = scene_dir / f"{scene_dir.name}.aggregation.json"
        payload = json.loads(aggregation_path.read_text(encoding="utf-8"))
        labels = sorted(
            {
                group["label"].strip().lower()
                for group in payload.get("segGroups", [])
                if isinstance(group.get("label"), str) and group.get("label").strip()
            }
        )
        return labels

    def _build_text_prompt(self, scene_dir: Path, labels: List[str]) -> str:
        metadata_path = scene_dir / f"{scene_dir.name}.txt"
        scene_type = ""
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("sceneType ="):
                scene_type = line.split("=", 1)[1].strip().lower()
                break
        if scene_type:
            return f"scene type {scene_type}; objects {' '.join(labels)}"
        return "objects " + " ".join(labels)

    def _encode_targets(self, labels: List[str]) -> torch.Tensor:
        target = torch.zeros(len(self.label_vocabulary), dtype=torch.float32)
        vocab_index = {label: idx for idx, label in enumerate(self.label_vocabulary)}
        for label in labels:
            if label in vocab_index:
                target[vocab_index[label]] = 1.0
        return target

    def _load_mesh_vertices(self, scene_dir: Path) -> np.ndarray:
        ply_path = scene_dir / f"{scene_dir.name}_vh_clean_2.ply"
        with ply_path.open("rb") as handle:
            header_lines = []
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"Malformed PLY header in {ply_path}")
                decoded = line.decode("ascii").strip()
                header_lines.append(decoded)
                if decoded == "end_header":
                    break
            vertex_line = next((line for line in header_lines if line.startswith("element vertex")), None)
            if vertex_line is None:
                raise ValueError(f"PLY file missing vertex declaration: {ply_path}")
            vertex_count = int(vertex_line.split()[2])
            vertex_dtype = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                    ("alpha", "u1"),
                ]
            )
            vertices = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
        return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)

    def _sample_points(self, coords: np.ndarray) -> torch.Tensor:
        if coords.shape[0] == 0:
            raise ValueError("Mesh contains no vertices.")

        if coords.shape[0] >= self.config.num_points:
            indices = np.linspace(0, coords.shape[0] - 1, self.config.num_points, dtype=int)
            sampled = coords[indices]
        else:
            repeats = int(np.ceil(self.config.num_points / float(coords.shape[0])))
            sampled = np.tile(coords, (repeats, 1))[: self.config.num_points]

        sampled = sampled - sampled.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(sampled, axis=1).max()
        if scale > 0:
            sampled = sampled / scale
        return torch.tensor(sampled, dtype=torch.float32)

    def _selected_frame_indices(self, num_frames: int) -> List[int]:
        if num_frames <= 0:
            return []
        if self.config.max_frames <= 1:
            return [0]
        indices = np.linspace(0, num_frames - 1, self.config.max_frames, dtype=int)
        return indices.tolist()

    def _decode_selected_frames(self, scene_dir: Path) -> torch.Tensor:
        sens_path = scene_dir / f"{scene_dir.name}.sens"
        with sens_path.open("rb") as handle:
            version = struct.unpack("<I", handle.read(4))[0]
            if version != 4:
                raise ValueError(f"Unsupported .sens version {version} in {sens_path}")
            name_len = struct.unpack("<Q", handle.read(8))[0]
            handle.read(name_len)
            handle.read(16 * 4 * 4)
            handle.read(4 + 4)  # color/depth compression
            color_width = struct.unpack("<I", handle.read(4))[0]
            color_height = struct.unpack("<I", handle.read(4))[0]
            handle.read(4 + 4)  # depth width/height
            handle.read(4)  # depth shift
            num_frames = struct.unpack("<Q", handle.read(8))[0]
            selected = set(self._selected_frame_indices(num_frames))
            frames: List[np.ndarray] = []
            for frame_index in range(num_frames):
                handle.read(16 * 4)  # camera_to_world
                handle.read(8 + 8)  # timestamps
                color_size = struct.unpack("<Q", handle.read(8))[0]
                depth_size = struct.unpack("<Q", handle.read(8))[0]
                color_bytes = handle.read(color_size)
                handle.seek(depth_size, 1)

                if frame_index not in selected:
                    continue

                image = Image.open(io.BytesIO(color_bytes)).convert("RGB")
                if image.size != (self.config.frame_resize, self.config.frame_resize):
                    image = image.resize((self.config.frame_resize, self.config.frame_resize))
                array = np.asarray(image, dtype=np.float32) / 255.0
                array = np.transpose(array, (2, 0, 1))
                frames.append(array)

                if len(frames) == len(selected):
                    break

        if not frames:
            raise ValueError(f"No RGB frames decoded from {sens_path}")

        while len(frames) < self.config.max_frames:
            frames.append(frames[-1].copy())

        return torch.tensor(np.stack(frames, axis=0), dtype=torch.float32)

    def _load_image_tensor(self, scene_dir: Path) -> torch.Tensor:
        scene_id = scene_dir.name
        if scene_id not in self._image_cache:
            self._image_cache[scene_id] = self._decode_selected_frames(scene_dir)
        return self._image_cache[scene_id]

    def _load_point_tensor(self, scene_dir: Path) -> torch.Tensor:
        scene_id = scene_dir.name
        if scene_id not in self._point_cache:
            coords = self._load_mesh_vertices(scene_dir)
            self._point_cache[scene_id] = self._sample_points(coords)
        return self._point_cache[scene_id]

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> Dict[str, object]:
        entry = self.scenes[index]
        scene_dir = entry["scene_dir"]
        return {
            "scene_id": entry["scene_id"],
            "point_coords": self._load_point_tensor(scene_dir),
            "image_tensor": self._load_image_tensor(scene_dir),
            "text_prompt": entry["text_prompt"],
            "label_targets": entry["label_targets"].clone(),
            "labels": list(entry["labels"]),
        }
