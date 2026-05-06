from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

from method.came_net import CAMENet, count_parameters
from .comparison_baselines import (
    LabelPriorBaseline,
    apply_came_variant,
    build_comparison_model,
    describe_comparison_method,
    get_comparison_method_specs,
)
from .scannet_multimodal_data import ScanNetSceneConfig, ScanNetSceneDataset
from .scannet_multimodal_experiment import (
    _build_dataloaders,
    _compute_multilabel_metrics,
    _create_artifact_dir,
    _make_collate_fn,
    _plot_training_curves,
    _resolve_device,
    _write_json,
)
from training.torch_runtime_compat import configure_torch_runtime_compat

configure_torch_runtime_compat()


REQUIRED_SCANNET_COMPARISON_METHODS = [
    "came",
    "came_no_gln",
    "came_no_equiv_reg",
    "came_non_geometric_fusion",
    "came_scalar_only",
    "label_prior",
    "pointnet",
    "dgcnn_style",
    "gatr_style",
    "equiformer_v2_style",
    "pointclip_style",
    "ulip_style",
]


@dataclass
class ScanNetComparisonConfig:
    method: str = "came"
    data_root: str = "F:/CAME-Net/ScanNet-small"
    artifact_root: str = "artifacts/scannet_comparison"
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
    aux_loss_weight: float = 0.0
    equiv_loss_weight: float = 1e-4
    equiv_warmup_steps: int = 200
    dropout: float = 0.0
    device: Optional[str] = None
    min_label_frequency: int = 1
    print_interval: int = 1


def apply_scannet_comparison_defaults(config: ScanNetComparisonConfig) -> ScanNetComparisonConfig:
    adjusted = asdict(config)
    if config.method in {"came_no_equiv_reg", "came_scalar_only", "came_non_geometric_fusion"}:
        adjusted["equiv_loss_weight"] = 0.0
        adjusted["equiv_warmup_steps"] = 0
    if config.method == "pointclip_style":
        adjusted["learning_rate"] = 1e-3
    if config.method == "ulip_style":
        adjusted["learning_rate"] = 5e-4
        adjusted["aux_loss_weight"] = 0.0
    return ScanNetComparisonConfig(**adjusted)


def _build_scannet_dataset_config(
    config: ScanNetComparisonConfig,
    *,
    vocabulary_scene_ids: Optional[List[str]] = None,
) -> ScanNetSceneConfig:
    return ScanNetSceneConfig(
        data_root=config.data_root,
        num_points=config.num_points,
        max_frames=config.max_frames,
        frame_resize=config.frame_resize,
        min_label_frequency=config.min_label_frequency,
        require_all_modalities=True,
        vocabulary_scene_ids=vocabulary_scene_ids,
    )


def _subset_indices(subset) -> List[int]:
    if isinstance(subset, Subset):
        return list(subset.indices)
    return list(range(len(subset)))


def _compute_train_label_priors(dataset: ScanNetSceneDataset, train_indices: List[int]) -> torch.Tensor:
    if not train_indices:
        raise ValueError("Label-prior baseline requires at least one training scene.")
    targets = torch.stack([dataset.scenes[index]["label_targets"] for index in train_indices], dim=0)
    return targets.float().mean(dim=0)


def _build_dataset_with_train_split_vocabulary(config: ScanNetComparisonConfig) -> ScanNetSceneDataset:
    bootstrap_dataset = ScanNetSceneDataset(_build_scannet_dataset_config(config))
    bootstrap_train_loader, _, _, _ = _build_dataloaders(bootstrap_dataset, config)
    vocabulary_scene_ids = [
        bootstrap_dataset.scenes[index]["scene_id"]
        for index in _subset_indices(bootstrap_train_loader.dataset)
    ]
    return ScanNetSceneDataset(
        _build_scannet_dataset_config(config, vocabulary_scene_ids=sorted(vocabulary_scene_ids))
    )


def _build_model_audit(model: nn.Module) -> Dict[str, object]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    module_report = {}
    for name, module in model.named_children():
        module_total = sum(parameter.numel() for parameter in module.parameters())
        module_trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        module_report[name] = {
            "parameter_count_total": module_total,
            "parameter_count_trainable": module_trainable,
        }
    return {
        "parameter_count_total": total_parameters,
        "parameter_count_trainable": trainable_parameters,
        "parameter_count_frozen": total_parameters - trainable_parameters,
        "module_report": module_report,
    }


def _build_scannet_came_model(config: ScanNetComparisonConfig, label_count: int) -> CAMENet:
    image_patch_dim = 3 * config.image_feature_size * config.image_feature_size
    model = CAMENet(
        num_classes=label_count,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        multimodal=True,
        image_patch_dim=image_patch_dim,
        text_token_dim=1,
    )
    return apply_came_variant(model, config.method)


def _build_scannet_model(config: ScanNetComparisonConfig, label_count: int) -> nn.Module:
    if config.method == "label_prior":
        raise ValueError("Label-prior model construction requires train-split label priors.")
    if config.method.startswith("came"):
        return _build_scannet_came_model(config, label_count)
    return build_comparison_model(
        method=config.method,
        class_names=[str(index) for index in range(label_count)],
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        image_size=config.frame_resize,
    )


def _forward_scannet_batch(model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    point_features = batch.get("point_features")
    image_patches = batch.get("image_patches")
    text_tokens = batch.get("text_tokens")
    return model(
        point_coords=batch["point_coords"].to(device),
        point_features=point_features.to(device) if point_features is not None else None,
        image_patches=image_patches.to(device) if image_patches is not None else None,
        text_tokens=text_tokens.to(device) if text_tokens is not None else None,
    )


def _train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: Optional[optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    for batch in dataloader:
        logits = _forward_scannet_batch(model, batch, device)
        targets = batch["label_targets"].to(device)
        loss = criterion(logits, targets)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * targets.shape[0]
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = _compute_multilabel_metrics(logits, targets)
    metrics["loss"] = total_loss / max(targets.shape[0], 1)
    return metrics


def _evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            logits = _forward_scannet_batch(model, batch, device)
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


def run_scannet_comparison_experiment(config: ScanNetComparisonConfig) -> Dict[str, object]:
    config = apply_scannet_comparison_defaults(config)
    if config.method not in get_comparison_method_specs():
        raise ValueError(f"Unknown comparison method: {config.method}")

    device = _resolve_device(config.device)
    dataset = _build_dataset_with_train_split_vocabulary(config)

    train_loader, val_loader, test_loader, split_sizes = _build_dataloaders(dataset, config)
    artifact_dir = _create_artifact_dir(config.artifact_root)
    tables_dir = artifact_dir / "tables"
    examples_dir = artifact_dir / "examples"
    plots_dir = artifact_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_indices = _subset_indices(train_loader.dataset)
    if config.method == "label_prior":
        train_label_priors = _compute_train_label_priors(dataset, train_indices)
        model = LabelPriorBaseline(train_label_priors).to(device)
    else:
        model = _build_scannet_model(config, label_count=len(dataset.label_vocabulary)).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = (
        optim.AdamW(trainable_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
        if trainable_parameters
        else None
    )
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_micro_f1": [],
        "val_micro_f1": [],
    }
    best_val_f1 = -1.0
    best_state = None
    start_time = time.time()

    for epoch in range(config.num_epochs):
        train_metrics = _train_one_epoch(model, train_loader, optimizer, criterion, device)
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
    runtime_seconds = time.time() - start_time

    comparison_row = {
        "method": config.method,
        "display_name": get_comparison_method_specs()[config.method].description,
        "micro_f1": test_metrics["micro_f1"],
        "macro_f1": test_metrics["macro_f1"],
        "micro_precision": test_metrics["micro_precision"],
        "micro_recall": test_metrics["micro_recall"],
        "exact_match_accuracy": test_metrics["exact_match_accuracy"],
        "parameter_count": count_parameters(model),
        "runtime_seconds": runtime_seconds,
        "label_vocabulary_size": len(dataset.label_vocabulary),
        "scene_count": len(dataset),
    }

    summary_lines = [
        "# ScanNet Comparison Summary",
        "",
        f"- Method: {config.method}",
        f"- Scenes retained: {dataset.dataset_report['retained_scenes']}",
        f"- Train/Val/Test split sizes: {split_sizes}",
        f"- Micro-F1: {test_metrics['micro_f1']:.4f}",
        f"- Macro-F1: {test_metrics['macro_f1']:.4f}",
        f"- Exact-match accuracy: {test_metrics['exact_match_accuracy']:.4f}",
        f"- Parameter count: {count_parameters(model)}",
        f"- Runtime seconds: {runtime_seconds:.2f}",
    ]
    (artifact_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (artifact_dir / "comparison_method.md").write_text(
        describe_comparison_method(config.method),
        encoding="utf-8",
    )
    _write_json(artifact_dir / "config.json", asdict(config))
    _write_json(artifact_dir / "metrics.json", test_metrics)
    _write_json(artifact_dir / "dataset_report.json", dataset.dataset_report)
    _write_json(artifact_dir / "label_vocabulary.json", dataset.label_vocabulary)
    _write_json(artifact_dir / "train_label_frequencies.json", dataset.dataset_report["vocabulary_label_frequencies"])
    _write_json(artifact_dir / "audit.json", _build_model_audit(model))
    _write_json(tables_dir / "comparison_row.json", comparison_row)
    _plot_training_curves(history, plots_dir / "training_curves.png")

    return {
        "artifact_dir": artifact_dir,
        "metrics": test_metrics,
        "comparison_row": comparison_row,
        "label_vocabulary": dataset.label_vocabulary,
        "dataset_report": dataset.dataset_report,
    }

