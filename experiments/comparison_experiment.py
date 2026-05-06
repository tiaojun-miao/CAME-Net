"""
comparison_experiment.py - Lightweight training/evaluation runner for comparison baselines and trainable ablations.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from training.torch_runtime_compat import configure_torch_runtime_compat

configure_torch_runtime_compat()

import torch
import torch.nn as nn
import torch.optim as optim

from .comparison_baselines import (
    build_comparison_model,
    describe_comparison_method,
    get_comparison_method_specs,
)
from method.equiv_loss import equivariance_loss_efficient
from .small_modelnet_experiment import (
    DEFAULT_CLASS_PROTOCOL,
    build_small_experiment_datasets_and_loaders,
    collect_sample_predictions,
    create_experiment_artifacts,
    evaluate_subset_model,
    resolve_modelnet_root,
)


@dataclass
class ComparisonExperimentConfig:
    method: str = "came"
    data_root: Optional[str] = None
    class_protocol: str = DEFAULT_CLASS_PROTOCOL
    class_names: Optional[Sequence[str]] = None
    train_samples_per_class: int = 100
    val_samples_per_class: int = 10
    test_samples_per_class: int = 20
    num_points: int = 256
    image_size: int = 32
    hidden_dim: int = 32
    num_layers: int = 2
    num_heads: int = 4
    batch_size: int = 8
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    equiv_loss_weight: float = 1e-4
    equiv_warmup_steps: int = 200
    aux_loss_weight: float = 0.1
    dropout: float = 0.0
    device: Optional[str] = None
    artifact_root: str = "artifacts/comparison_experiments"
    sample_visualization_count: int = 6
    checkpoint_interval: int = 100
    print_interval: int = 1


def apply_comparison_runtime_defaults(config: ComparisonExperimentConfig) -> ComparisonExperimentConfig:
    specs = get_comparison_method_specs()
    if config.method not in specs:
        raise ValueError(f"Unknown comparison method: {config.method}")

    base = ComparisonExperimentConfig(method=config.method)
    adjusted = asdict(config)

    if not specs[config.method].uses_equivariance_regularizer:
        if config.equiv_loss_weight == base.equiv_loss_weight:
            adjusted["equiv_loss_weight"] = 0.0
        if config.equiv_warmup_steps == base.equiv_warmup_steps:
            adjusted["equiv_warmup_steps"] = 0

    if not specs[config.method].uses_auxiliary_loss and config.aux_loss_weight == base.aux_loss_weight:
        adjusted["aux_loss_weight"] = 0.0

    if config.method == "pointclip_style":
        if config.learning_rate == base.learning_rate:
            adjusted["learning_rate"] = 1e-3
        if config.aux_loss_weight == base.aux_loss_weight:
            adjusted["aux_loss_weight"] = 0.0

    if config.method == "ulip_style":
        if config.learning_rate == base.learning_rate:
            adjusted["learning_rate"] = 5e-4

    return ComparisonExperimentConfig(**adjusted)


def _forward_batch(model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return model(
        point_coords=batch["point_coords"].to(device),
        point_features=batch.get("point_features").to(device) if batch.get("point_features") is not None else None,
        image_patches=batch.get("image_patches").to(device) if batch.get("image_patches") is not None else None,
        text_tokens=batch.get("text_tokens").to(device) if batch.get("text_tokens") is not None else None,
    )


def _save_checkpoint(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    val_accuracy: float,
    filepath: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "accuracy": val_accuracy,
        },
        filepath,
    )


def _load_best_checkpoint(model: nn.Module, filepath: Path, device: torch.device) -> None:
    if not filepath.exists():
        return
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def _train_one_epoch(
    *,
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    method: str,
    aux_loss_weight: float,
    equiv_loss_weight: float,
    equiv_warmup_steps: int,
    global_step: int,
) -> tuple[Dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_aux_loss = 0.0
    total_equiv_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        labels = batch["labels"].to(device)
        point_coords = batch["point_coords"].to(device)

        optimizer.zero_grad()
        logits = _forward_batch(model, batch, device)
        task_loss = criterion(logits, labels)
        loss = task_loss

        if aux_loss_weight > 0 and hasattr(model, "compute_auxiliary_loss"):
            auxiliary_loss = model.compute_auxiliary_loss(point_coords, labels)
            loss = loss + aux_loss_weight * auxiliary_loss
            total_aux_loss += auxiliary_loss.item()

        if equiv_loss_weight > 0 and method.startswith("came") and hasattr(model, "get_latent_multivector"):
            if equiv_warmup_steps > 0:
                warmup_progress = min(1.0, global_step / float(max(1, equiv_warmup_steps)))
            else:
                warmup_progress = 1.0
            current_lambda = equiv_loss_weight * warmup_progress
            if current_lambda > 0:
                equiv_loss = equivariance_loss_efficient(
                    model,
                    point_coords=point_coords,
                    labels=labels,
                    num_samples=1,
                )
                loss = loss + current_lambda * equiv_loss
                total_equiv_loss += equiv_loss.item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        predictions = torch.argmax(logits, dim=1)
        total += labels.numel()
        correct += int((predictions == labels).sum().item())
        global_step += 1

    num_batches = max(1, len(dataloader))
    return {
        "loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "aux_loss": total_aux_loss / num_batches,
        "equiv_loss": total_equiv_loss / num_batches,
        "accuracy": 100.0 * correct / max(1, total),
    }, global_step


def _validate(
    *,
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            logits = _forward_batch(model, batch, device)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total += labels.numel()
            correct += int((predictions == labels).sum().item())

    num_batches = max(1, len(dataloader))
    return {
        "val_loss": total_loss / num_batches,
        "val_accuracy": 100.0 * correct / max(1, total),
    }


def run_comparison_experiment(config: ComparisonExperimentConfig) -> Dict[str, object]:
    config = apply_comparison_runtime_defaults(config)
    specs = get_comparison_method_specs()
    if config.method not in specs:
        raise ValueError(f"Unknown comparison method: {config.method}")

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("Warning: CUDA is unavailable; running on CPU may exceed the intended runtime budget.")

    resolved_data_root = resolve_modelnet_root(config.data_root)
    run_root = Path(config.artifact_root) / "runs" / f"{config.method}-{time.strftime('%Y%m%d-%H%M%S')}"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_small_experiment_datasets_and_loaders(
        config,
        resolved_data_root=resolved_data_root,
    )

    model = build_comparison_model(
        method=config.method,
        class_names=train_dataset.class_names,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        image_size=config.image_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.learning_rate / 100)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = float("-inf")
    best_checkpoint = checkpoint_dir / "best_model.pth"
    global_step = 0
    start_time = time.time()

    print("Starting comparison training...")
    print(f"Method: {config.method}")
    print(f"Device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Auxiliary loss weight: {config.aux_loss_weight}")
    print(f"Equivariance loss weight: {config.equiv_loss_weight}")
    print("-" * 60)

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        train_metrics, global_step = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            method=config.method,
            aux_loss_weight=config.aux_loss_weight,
            equiv_loss_weight=config.equiv_loss_weight,
            equiv_warmup_steps=config.equiv_warmup_steps,
            global_step=global_step,
        )
        scheduler.step()
        val_metrics = _validate(model=model, dataloader=val_loader, criterion=criterion, device=device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_accuracy"])
        history["lr"].append(scheduler.get_last_lr()[0])

        if (epoch + 1) % config.print_interval == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{config.num_epochs}] ({time.time() - epoch_start:.1f}s) "
                f"Train Loss: {train_metrics['loss']:.4f} Train Acc: {train_metrics['accuracy']:.2f}%"
            )
            print(
                f"                    Val Loss: {val_metrics['val_loss']:.4f} "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
            )

        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_accuracy=val_metrics["val_accuracy"],
                filepath=best_checkpoint,
            )
            print(f"Checkpoint saved to {best_checkpoint}")

    _load_best_checkpoint(model, best_checkpoint, device)

    metrics = evaluate_subset_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=train_dataset.class_names,
    )
    metrics["parameter_count"] = sum(p.numel() for p in model.parameters())
    metrics["method"] = config.method

    sample_predictions = collect_sample_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        sample_count=config.sample_visualization_count,
    )

    artifact_dir = create_experiment_artifacts(
        artifact_root=str(run_root),
        config={
            **asdict(config),
            "resolved_data_root": str(resolved_data_root),
        },
        history=history,
        metrics=metrics,
        selected_classes=train_dataset.class_names,
        dataset_sizes={"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)},
        runtime_seconds=time.time() - start_time,
        sample_predictions=sample_predictions,
        confusion_matrix=metrics["confusion_matrix"],
        extra_artifacts={
            "comparison_method": describe_comparison_method(config.method),
        },
    )

    tables_dir = artifact_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    comparison_row = {
        "method": config.method,
        "overall_accuracy": metrics["overall_accuracy"],
        "mean_class_accuracy": metrics["mean_class_accuracy"],
        "parameter_count": metrics["parameter_count"],
        "runtime_seconds": time.time() - start_time,
        "equivariance_regularizer": specs[config.method].uses_equivariance_regularizer,
        "auxiliary_loss": specs[config.method].uses_auxiliary_loss,
    }
    (tables_dir / "comparison_row.json").write_text(json.dumps(comparison_row, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "artifact_dir": str(artifact_dir),
        "history": history,
        "metrics": metrics,
        "method": config.method,
    }
