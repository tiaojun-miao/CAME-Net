from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from method.came_net import CAMENet, count_parameters
from .scannet_multimodal_data import ScanNetSceneConfig, ScanNetSceneDataset
from training.torch_runtime_compat import configure_torch_runtime_compat

configure_torch_runtime_compat()


@dataclass
class ScanNetMultimodalConfig:
    data_root: str
    artifact_root: str = "artifacts/scannet_multimodal"
    num_points: int = 256
    max_frames: int = 3
    frame_resize: int = 32
    image_feature_size: int = 8
    max_text_tokens: int = 48
    batch_size: int = 2
    num_epochs: int = 10
    hidden_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.0
    device: str | None = None
    min_label_frequency: int = 1
    print_interval: int = 1


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _create_artifact_dir(root: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = Path(root) / "runs" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tokenize_prompt(prompt: str, max_tokens: int) -> torch.Tensor:
    values = [min(ord(char), 255) / 255.0 for char in prompt[:max_tokens]]
    if len(values) < max_tokens:
        values.extend([0.0] * (max_tokens - len(values)))
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1)


def _frames_to_patch_features(image_tensor: torch.Tensor, image_feature_size: int) -> torch.Tensor:
    batch_size, num_frames, channels, height, width = image_tensor.shape
    merged = image_tensor.reshape(batch_size * num_frames, channels, height, width)
    pooled = F.adaptive_avg_pool2d(merged, output_size=(image_feature_size, image_feature_size))
    pooled = pooled.reshape(batch_size, num_frames, channels, image_feature_size, image_feature_size)
    return pooled.flatten(start_dim=2)


def _make_collate_fn(config: ScanNetMultimodalConfig):
    def _collate(batch):
        point_coords = torch.stack([item["point_coords"] for item in batch])
        image_tensor = torch.stack([item["image_tensor"] for item in batch])
        image_patches = _frames_to_patch_features(image_tensor, config.image_feature_size)
        text_tokens = torch.stack([_tokenize_prompt(item["text_prompt"], config.max_text_tokens) for item in batch])
        label_targets = torch.stack([item["label_targets"] for item in batch])
        return {
            "scene_id": [item["scene_id"] for item in batch],
            "point_coords": point_coords,
            "image_patches": image_patches,
            "text_tokens": text_tokens,
            "label_targets": label_targets,
            "labels": [item["labels"] for item in batch],
        }

    return _collate


def _split_indices(num_items: int) -> Tuple[List[int], List[int], List[int]]:
    if num_items < 1:
        raise ValueError("ScanNet multimodal experiment requires at least 1 valid scene.")
    if num_items == 1:
        return [0], [0], [0]
    if num_items == 2:
        return [0], [1], [1]
    indices = list(range(num_items))
    train_end = max(1, int(round(num_items * 0.5)))
    val_end = max(train_end + 1, int(round(num_items * 0.75)))
    train = indices[:train_end]
    val = indices[train_end:val_end]
    test = indices[val_end:]
    if not val:
        if len(train) > 1:
            val = [train.pop()]
        elif len(test) > 1:
            val = [test.pop(0)]
        else:
            val = [indices[-1]]
    if not test:
        if len(val) > 1:
            test = [val.pop()]
        elif len(train) > 1:
            test = [train.pop()]
        else:
            test = [val[-1]]
    return train, val, test


def _build_dataloaders(dataset: ScanNetSceneDataset, config: ScanNetMultimodalConfig):
    train_idx, val_idx, test_idx = _split_indices(len(dataset))
    collate_fn = _make_collate_fn(config)
    return (
        DataLoader(Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn),
        DataLoader(Subset(dataset, test_idx), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn),
        {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    )


def _build_model(config: ScanNetMultimodalConfig, num_labels: int) -> CAMENet:
    patch_dim = 3 * config.image_feature_size * config.image_feature_size
    return CAMENet(
        num_classes=num_labels,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        multimodal=True,
        image_patch_dim=patch_dim,
        text_token_dim=1,
    )


def _compute_multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    micro_precision = tp / max(tp + fp, 1.0)
    micro_recall = tp / max(tp + fn, 1.0)
    micro_f1 = 2.0 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-8)

    per_label_f1 = []
    for label_idx in range(targets.shape[1]):
        label_pred = preds[:, label_idx]
        label_true = targets[:, label_idx]
        label_tp = (label_pred * label_true).sum().item()
        label_fp = (label_pred * (1.0 - label_true)).sum().item()
        label_fn = ((1.0 - label_pred) * label_true).sum().item()
        label_precision = label_tp / max(label_tp + label_fp, 1.0)
        label_recall = label_tp / max(label_tp + label_fn, 1.0)
        label_f1 = 2.0 * label_precision * label_recall / max(label_precision + label_recall, 1e-8)
        per_label_f1.append(label_f1)

    exact_match = (preds == targets).all(dim=1).float().mean().item()
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        "exact_match_accuracy": exact_match,
    }


def _evaluate(model: CAMENet, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                point_coords=batch["point_coords"].to(device),
                image_patches=batch["image_patches"].to(device),
                text_tokens=batch["text_tokens"].to(device),
            )
            targets = batch["label_targets"].to(device)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.shape[0]
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
    if not all_logits:
        raise ValueError("Evaluation dataloader is empty; dataset split produced no validation/test samples.")
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = _compute_multilabel_metrics(logits, targets)
    metrics["loss"] = total_loss / max(targets.shape[0], 1)
    return metrics


def _plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("BCE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(epochs, history["train_micro_f1"], label="Train micro-F1")
    axes[1].plot(epochs, history["val_micro_f1"], label="Val micro-F1")
    axes[1].set_title("Micro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_scannet_multimodal_experiment(config: ScanNetMultimodalConfig) -> Dict[str, object]:
    device = _resolve_device(config.device)
    dataset = ScanNetSceneDataset(
        ScanNetSceneConfig(
            data_root=config.data_root,
            num_points=config.num_points,
            max_frames=config.max_frames,
            frame_resize=config.frame_resize,
            min_label_frequency=config.min_label_frequency,
            require_all_modalities=True,
        )
    )

    train_loader, val_loader, test_loader, split_sizes = _build_dataloaders(dataset, config)
    artifact_dir = _create_artifact_dir(config.artifact_root)
    examples_dir = artifact_dir / "examples"
    plots_dir = artifact_dir / "plots"
    examples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(config, num_labels=len(dataset.label_vocabulary)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_micro_f1": [],
        "val_micro_f1": [],
    }
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        train_logits = []
        train_targets = []
        for batch in train_loader:
            logits = model(
                point_coords=batch["point_coords"].to(device),
                image_patches=batch["image_patches"].to(device),
                text_tokens=batch["text_tokens"].to(device),
            )
            targets = batch["label_targets"].to(device)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * targets.shape[0]
            train_logits.append(logits.detach().cpu())
            train_targets.append(targets.detach().cpu())

        train_logits_tensor = torch.cat(train_logits, dim=0)
        train_targets_tensor = torch.cat(train_targets, dim=0)
        train_metrics = _compute_multilabel_metrics(train_logits_tensor, train_targets_tensor)
        train_metrics["loss"] = epoch_loss / max(train_targets_tensor.shape[0], 1)
        val_metrics = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_micro_f1"].append(train_metrics["micro_f1"])
        history["val_micro_f1"].append(val_metrics["micro_f1"])

        if (epoch + 1) % config.print_interval == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{config.num_epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train micro-F1: {train_metrics['micro_f1']:.4f} "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val micro-F1: {val_metrics['micro_f1']:.4f}"
            )

        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without a best checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test_loader, criterion, device)

    summary_lines = [
        "# ScanNet Multimodal Summary",
        "",
        f"- Device: {device}",
        f"- Scenes retained: {dataset.dataset_report['retained_scenes']}",
        f"- Scenes skipped: {dataset.dataset_report['skipped_scenes']}",
        f"- Train/Val/Test split sizes: {split_sizes}",
        f"- Label vocabulary size: {len(dataset.label_vocabulary)}",
        f"- Test micro-F1: {test_metrics['micro_f1']:.4f}",
        f"- Test macro-F1: {test_metrics['macro_f1']:.4f}",
        f"- Test exact-match accuracy: {test_metrics['exact_match_accuracy']:.4f}",
        f"- Parameter count: {count_parameters(model)}",
    ]
    (artifact_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    _write_json(artifact_dir / "config.json", asdict(config))
    _write_json(artifact_dir / "metrics.json", test_metrics)
    _write_json(artifact_dir / "dataset_report.json", dataset.dataset_report)
    _write_json(artifact_dir / "label_vocabulary.json", dataset.label_vocabulary)
    _plot_training_curves(history, plots_dir / "training_curves.png")

    example_rows = []
    for batch in test_loader:
        logits = model(
            point_coords=batch["point_coords"].to(device),
            image_patches=batch["image_patches"].to(device),
            text_tokens=batch["text_tokens"].to(device),
        ).detach().cpu()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        for row_idx, scene_id in enumerate(batch["scene_id"]):
            predicted = [
                dataset.label_vocabulary[label_idx]
                for label_idx, active in enumerate(preds[row_idx].tolist())
                if active
            ]
            truth = [
                dataset.label_vocabulary[label_idx]
                for label_idx, active in enumerate(batch["label_targets"][row_idx].tolist())
                if active
            ]
            example_rows.append({"scene_id": scene_id, "predicted": predicted, "ground_truth": truth})
    _write_json(examples_dir / "predictions.json", example_rows)

    return {
        "artifact_dir": artifact_dir,
        "metrics": test_metrics,
        "label_vocabulary": dataset.label_vocabulary,
        "dataset_report": dataset.dataset_report,
        "parameter_count": count_parameters(model),
    }
