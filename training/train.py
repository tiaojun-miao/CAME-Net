"""
train.py - Training Script for CAME-Net

This module provides the training and evaluation functions for the CAME-Net model,
including the training loop, validation, and checkpointing utilities.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.amp import GradScaler as TorchGradScaler
except ImportError:
    from torch.cuda.amp import GradScaler as TorchGradScaler


def _model_forward_from_batch(model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    point_coords = batch.get('point_coords')
    point_features = batch.get('point_features')
    image_patches = batch.get('image_patches')
    text_tokens = batch.get('text_tokens')

    return model(
        point_coords=point_coords.to(device, non_blocking=True) if point_coords is not None else None,
        point_features=point_features.to(device, non_blocking=True) if point_features is not None else None,
        image_patches=image_patches.to(device, non_blocking=True) if image_patches is not None else None,
        text_tokens=text_tokens.to(device, non_blocking=True) if text_tokens is not None else None,
    )


def _equivariance_loss_from_batch(equiv_loss_fn, model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device):
    if equiv_loss_fn is None:
        return None

    point_coords = batch['point_coords'].to(device, non_blocking=True)
    labels = batch.get('labels')
    point_features = batch.get('point_features')
    image_patches = batch.get('image_patches')
    text_tokens = batch.get('text_tokens')

    was_training = model.training
    model.eval()
    try:
        return equiv_loss_fn(
            model,
            point_coords,
            labels.to(device, non_blocking=True) if labels is not None else None,
            point_features=point_features.to(device, non_blocking=True) if point_features is not None else None,
            image_patches=image_patches.to(device, non_blocking=True) if image_patches is not None else None,
            text_tokens=text_tokens.to(device, non_blocking=True) if text_tokens is not None else None,
        )
    finally:
        if was_training:
            model.train()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding='utf-8',
    )


def _create_grad_scaler(device: torch.device) -> Optional[TorchGradScaler]:
    if device.type != 'cuda':
        return None
    try:
        return TorchGradScaler('cuda')
    except TypeError:
        return TorchGradScaler()


def _torch_load_trusted_checkpoint(filepath: str, device: torch.device):
    # Training checkpoints in this project intentionally store optimizer, scheduler,
    # AMP scaler, and history state. On PyTorch 2.6+, torch.load defaults to
    # weights_only=True, which rejects these richer checkpoint objects.
    try:
        return torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(filepath, map_location=device)


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return 'out of memory' in message and 'cuda' in message


def _batch_size_from_batch(batch: Dict[str, torch.Tensor]) -> int:
    labels = batch.get('labels')
    if isinstance(labels, torch.Tensor):
        return int(labels.shape[0])
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError('Could not infer batch size from batch payload.')


def _slice_batch(batch: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    sliced: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] >= end:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _iter_micro_batches(batch: Dict[str, torch.Tensor], micro_batch_size: Optional[int]):
    batch_size = _batch_size_from_batch(batch)
    chunk_size = batch_size if micro_batch_size is None or micro_batch_size <= 0 else min(batch_size, micro_batch_size)
    for start in range(0, batch_size, chunk_size):
        end = min(batch_size, start + chunk_size)
        yield _slice_batch(batch, start, end), end - start


def _looks_like_modelnet_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and ((child / 'train').is_dir() or (child / 'test').is_dir()):
            return True
    return False


def resolve_modelnet_data_root(data_root: Optional[str] = None) -> Path:
    if data_root is not None:
        raw_candidates = [Path(data_root)]
    else:
        project_root = Path(__file__).resolve().parent
        raw_candidates = [
            project_root / 'ModelNet40',
            project_root / 'ModelNet40' / 'ModelNet40',
        ]

    candidates: List[Path] = []
    seen = set()
    for candidate in raw_candidates:
        expanded_candidates = [candidate]
        if candidate.name != 'ModelNet40':
            expanded_candidates.append(candidate / 'ModelNet40')
        for expanded_candidate in expanded_candidates:
            resolved = expanded_candidate.resolve(strict=False)
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

    for candidate in candidates:
        if _looks_like_modelnet_root(candidate):
            return candidate

    searched = '\n'.join(f'  - {candidate}' for candidate in candidates)
    raise FileNotFoundError(
        'Could not locate a valid ModelNet40 root. Checked:\n'
        f'{searched}'
    )


def create_default_modelnet_dataloaders(
    data_root: Optional[str] = None,
    num_points: int = 1024,
    batch_size: int = 4,
    num_workers: int = 2,
    train_data_augmentation: bool = False,
):
    from training.data_utils import ModelNetDataset, collate_fn

    resolved_root = resolve_modelnet_data_root(data_root)

    train_dataset = ModelNetDataset(
        data_dir=str(resolved_root),
        split="train",
        num_points=num_points,
        data_augmentation=train_data_augmentation,
    )
    val_dataset = ModelNetDataset(
        data_dir=str(resolved_root),
        split="test",
        num_points=num_points,
        data_augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, val_loader


def _dataset_num_classes(dataloader: DataLoader) -> int:
    dataset = getattr(dataloader, 'dataset', None)
    num_classes = getattr(dataset, 'num_classes', None)
    if num_classes is None:
        raise ValueError('Could not infer dataset num_classes from dataloader.dataset.')
    return int(num_classes)


def _dataset_class_names(dataloader: DataLoader) -> Optional[List[str]]:
    dataset = getattr(dataloader, 'dataset', None)
    class_names = getattr(dataset, 'class_names', None)
    if class_names is None:
        return None
    return [str(name) for name in class_names]


def _stabilize_fresh_classification_head(model: nn.Module) -> None:
    """
    Shrink fresh classification-head logits.

    The multivector backbone can produce relatively large pooled features at
    initialization. With the default Xavier init in came_net.py, the task head
    often starts with a strong single-class argmax bias, which then traps early
    training in class-collapse. A smaller fresh-head init keeps the initial
    logits close to uniform without changing the backbone or resume behavior.
    """
    if not hasattr(model, 'classification_head'):
        return

    head = getattr(model, 'classification_head')
    if not isinstance(head, nn.Sequential) or len(head) < 4:
        return

    first_linear = head[0]
    final_linear = head[3]
    if not isinstance(first_linear, nn.Linear) or not isinstance(final_linear, nn.Linear):
        return

    nn.init.xavier_uniform_(first_linear.weight, gain=0.1)
    nn.init.zeros_(first_linear.bias)
    nn.init.xavier_uniform_(final_linear.weight, gain=0.05)
    nn.init.zeros_(final_linear.bias)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    equiv_loss_weight: float = 0.1,
    equiv_loss_fn=None,
    equiv_warmup_steps: int = 0,
    start_step: int = 0,
    scaler: Optional[TorchGradScaler] = None,
    epoch: int = 0,
    num_epochs: int = 0,
    micro_batch_size: Optional[int] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Train for one epoch with optional mixed precision.

    Args:
        model: CAME-Net model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        equiv_loss_weight: Maximum weight for equivariance loss
        equiv_loss_fn: Function to compute equivariance loss
        equiv_warmup_steps: Number of steps to warm up equivariance weight
        start_step: Global step to start from
        scaler: GradScaler for mixed precision training (None to disable)
        epoch: Current epoch number
        num_epochs: Total number of epochs

    Returns:
        Tuple containing metrics dictionary and updated global step
    """
    model.train()
    use_amp = scaler is not None

    total_loss = 0.0
    total_task_loss = 0.0
    total_equiv_loss = 0.0
    correct = 0
    total = 0

    global_step = start_step

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                leave=False, ncols=120, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for batch in pbar:
        batch_size = _batch_size_from_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        batch_task_loss = 0.0
        batch_equiv_loss = 0.0

        try:
            for micro_batch, micro_size in _iter_micro_batches(batch, micro_batch_size):
                labels = micro_batch['labels'].to(device, non_blocking=True)
                micro_weight = float(micro_size) / float(batch_size)

                with autocast(device_type='cuda', enabled=use_amp):
                    logits = _model_forward_from_batch(model, micro_batch, device)
                    task_loss = criterion(logits, labels)

                    loss = task_loss
                    equiv_loss_value = None
                    if equiv_loss_fn is not None and equiv_loss_weight > 0:
                        if equiv_warmup_steps > 0:
                            warmup_progress = min(1.0, global_step / float(max(1, equiv_warmup_steps)))
                        else:
                            warmup_progress = 1.0
                        current_lambda = equiv_loss_weight * warmup_progress
                        if current_lambda > 0:
                            equiv_loss_value = _equivariance_loss_from_batch(equiv_loss_fn, model, micro_batch, device)
                            loss = task_loss + current_lambda * equiv_loss_value

                if not torch.isfinite(task_loss):
                    raise RuntimeError(
                        'Non-finite task loss detected during training. '
                        f'epoch={epoch + 1}, global_step={global_step}, '
                        f'logits_abs_max={float(logits.detach().abs().max()):.6g}'
                    )
                if not torch.isfinite(loss):
                    equiv_text = 'None' if equiv_loss_value is None else f'{float(equiv_loss_value.detach()):.6g}'
                    raise RuntimeError(
                        'Non-finite total loss detected during training. '
                        f'epoch={epoch + 1}, global_step={global_step}, '
                        f'task_loss={float(task_loss.detach()):.6g}, '
                        f'equiv_loss={equiv_text}, '
                        f'logits_abs_max={float(logits.detach().abs().max()):.6g}'
                    )

                scaled_loss = loss * micro_weight
                if use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                batch_loss += loss.item() * micro_weight
                batch_task_loss += task_loss.item() * micro_weight
                if equiv_loss_value is not None:
                    batch_equiv_loss += equiv_loss_value.item() * micro_weight

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        except RuntimeError as exc:
            if device.type == 'cuda' and _is_cuda_oom_error(exc):
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    'CUDA OOM during train forward/backward. '
                    f'Current outer batch size={batch_size}, micro_batch_size={micro_batch_size}. '
                    'Try a smaller --num-points (e.g. 256) or keep --micro-batch-size 1.'
                ) from exc
            raise

        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += batch_loss
        total_task_loss += batch_task_loss
        total_equiv_loss += batch_equiv_loss

        global_step += 1

        current_acc = 100.0 * correct / total
        current_loss = total_loss / (global_step - start_step)
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})

    num_batches = len(dataloader)

    metrics = {
        'loss': total_loss / num_batches,
        'task_loss': total_task_loss / num_batches,
        'equiv_loss': total_equiv_loss / num_batches,
        'accuracy': 100.0 * correct / total
    }

    return metrics, global_step

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    micro_batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Validate the model.

    Args:
        model: CAME-Net model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary containing average validation loss and accuracy
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Validating', leave=False, ncols=120,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    with torch.no_grad():
        for batch in pbar:
            batch_size = _batch_size_from_batch(batch)
            batch_loss = 0.0

            try:
                for micro_batch, micro_size in _iter_micro_batches(batch, micro_batch_size):
                    labels = micro_batch['labels'].to(device, non_blocking=True)
                    logits = _model_forward_from_batch(model, micro_batch, device)
                    loss = criterion(logits, labels)
                    batch_loss += loss.item() * (float(micro_size) / float(batch_size))

                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            except RuntimeError as exc:
                if device.type == 'cuda' and _is_cuda_oom_error(exc):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        'CUDA OOM during validation forward. '
                        f'Current outer batch size={batch_size}, micro_batch_size={micro_batch_size}. '
                        'Try a smaller --num-points (e.g. 256) or keep --micro-batch-size 1.'
                    ) from exc
                raise

            total_loss += batch_loss

            current_acc = 100.0 * correct / total
            pbar.set_postfix({'val_acc': f'{current_acc:.2f}%'})

    num_batches = len(dataloader)

    return {
        'val_loss': total_loss / num_batches,
        'val_accuracy': 100.0 * correct / total
    }


def _build_scheduler(
    optimizer: optim.Optimizer,
    num_epochs: int,
    learning_rate: float,
) -> optim.lr_scheduler._LRScheduler:
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs),
        eta_min=learning_rate / 100,
    )


def _empty_history() -> Dict[str, list]:
    return {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
    }


def _merge_history(saved_history: Optional[Dict]) -> Dict[str, list]:
    history = _empty_history()
    if not isinstance(saved_history, dict):
        return history
    for key in history:
        if key in saved_history and isinstance(saved_history[key], list):
            history[key] = list(saved_history[key])
    return history


def _extract_checkpoint_payload(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and 'model_state_dict' in checkpoint_obj:
        return checkpoint_obj, checkpoint_obj['model_state_dict']
    return None, checkpoint_obj


def _infer_model_config_from_state_dict(model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, object]:
    inferred: Dict[str, object] = {}

    layer_indices = set()
    for key in model_state_dict:
        if key.startswith('came_layers.'):
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
    if layer_indices:
        inferred['num_layers'] = max(layer_indices) + 1

    num_heads_key = 'came_layers.0.attention.grade_query_projs.0.weight'
    if num_heads_key in model_state_dict:
        inferred['num_heads'] = int(model_state_dict[num_heads_key].shape[0])

    hidden_dim_key = 'classification_head.0.weight'
    if hidden_dim_key in model_state_dict:
        inferred['hidden_dim'] = int(model_state_dict[hidden_dim_key].shape[0] // 2)

    num_classes_key = 'classification_head.2.weight'
    if num_classes_key in model_state_dict:
        inferred['num_classes'] = int(model_state_dict[num_classes_key].shape[0])

    return inferred


def _resolve_model_config_from_checkpoint(filepath: str, device: torch.device) -> Optional[Dict[str, object]]:
    checkpoint_obj = _torch_load_trusted_checkpoint(filepath, device)
    checkpoint_payload, model_state_dict = _extract_checkpoint_payload(checkpoint_obj)
    if not isinstance(model_state_dict, dict):
        return None

    resolved_config: Dict[str, object] = {}
    if checkpoint_payload is not None and isinstance(checkpoint_payload.get('model_config'), dict):
        resolved_config.update(checkpoint_payload['model_config'])
    resolved_config.update(
        {
            key: value
            for key, value in _infer_model_config_from_state_dict(model_state_dict).items()
            if key not in resolved_config
        }
    )

    if not resolved_config:
        return None

    resolved_config.setdefault('num_classes', 40)
    resolved_config.setdefault('point_feature_dim', 0)
    resolved_config.setdefault('num_layers', 4)
    resolved_config.setdefault('num_heads', 8)
    resolved_config.setdefault('hidden_dim', 64)
    resolved_config.setdefault('dropout', 0.1)
    resolved_config.setdefault('multimodal', False)
    return resolved_config


def _load_training_state(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    scaler: Optional[TorchGradScaler],
    filepath: str,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint_obj = _torch_load_trusted_checkpoint(filepath, device)
    checkpoint_payload, model_state_dict = _extract_checkpoint_payload(checkpoint_obj)
    model.load_state_dict(model_state_dict)

    if checkpoint_payload is None:
        print(f"Loaded model weights from {filepath} without optimizer/scheduler state.")
        return {
            'epoch': -1,
            'accuracy': 0.0,
            'best_val_acc': 0.0,
            'global_step': 0,
            'history': _empty_history(),
        }

    if optimizer is not None and checkpoint_payload.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint_payload['optimizer_state_dict'])

    if scheduler is not None and checkpoint_payload.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint_payload['scheduler_state_dict'])

    if scaler is not None and checkpoint_payload.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint_payload['scaler_state_dict'])

    epoch = int(checkpoint_payload.get('epoch', -1))
    accuracy = float(checkpoint_payload.get('accuracy', 0.0))
    best_val_acc = float(checkpoint_payload.get('best_val_acc', accuracy))
    global_step = int(checkpoint_payload.get('global_step', 0))
    history = _merge_history(checkpoint_payload.get('history'))

    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {epoch + 1} with accuracy {accuracy:.2f}%")

    return {
        'epoch': epoch,
        'accuracy': accuracy,
        'best_val_acc': best_val_acc,
        'global_step': global_step,
        'history': history,
    }


def train_came_net(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device('cpu'),
    equiv_loss_weight: float = 0.1,
    equiv_loss_fn=None,
    equiv_warmup_steps: int = 0,
    checkpoint_dir: str = './checkpoints',
    checkpoint_interval: int = 10,
    print_interval: int = 5,
    model_config: Optional[Dict] = None,
    resume_from: Optional[str] = None,
    micro_batch_size: Optional[int] = None,
    amp_enabled: bool = False,
) -> Dict[str, list]:
    """
    Train CAME-Net model.

    Args:
        model: CAME-Net model
        train_dataloader: Training data loader
        val_dataloader: Optional validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to train on
        equiv_loss_weight: Maximum weight for equivariance loss
        equiv_loss_fn: Function to compute equivariance loss
        equiv_warmup_steps: Steps to linearly warm up equivariance weight
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Interval for saving checkpoints
        print_interval: Interval for printing progress

    Returns:
        Dictionary containing training history
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = _build_scheduler(optimizer, num_epochs, learning_rate)

    os.makedirs(checkpoint_dir, exist_ok=True)

    history = _empty_history()

    best_val_acc = 0.0
    global_step = 0
    start_epoch = 0

    # Initialize GradScaler for mixed precision training on CUDA
    scaler = _create_grad_scaler(device) if amp_enabled else None
    if scaler is not None:
        print('Mixed Precision Training (AMP) enabled')
    else:
        print('Mixed Precision Training (AMP) disabled')

    if resume_from:
        resume_state = _load_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            filepath=resume_from,
            device=device,
        )
        history = resume_state['history']
        best_val_acc = float(resume_state['best_val_acc'])
        global_step = int(resume_state['global_step'])
        start_epoch = int(resume_state['epoch']) + 1

    print('Starting training...')
    print(f'Device: {device}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Number of epochs: {num_epochs}')
    print(f'Learning rate: {learning_rate}')
    print(f'Equivariance loss weight: {equiv_loss_weight}')
    print(f'Equivariance warmup steps: {equiv_warmup_steps}')
    print(f'Checkpoint directory: {Path(checkpoint_dir).resolve()}')
    print(f'Resume checkpoint: {resume_from if resume_from else "None"}')
    print(f'Micro-batch size: {micro_batch_size if micro_batch_size else "full batch"}')
    print(f'AMP enabled: {amp_enabled}')
    print('-' * 60)

    if start_epoch >= num_epochs:
        print(f'Checkpoint already reached epoch {start_epoch}. Nothing to do because num_epochs={num_epochs}.')
        return history

    history_path = Path(checkpoint_dir) / 'history.json'
    train_state_path = Path(checkpoint_dir) / 'train_state.json'
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pth')
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pth')
    training_config = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'equiv_loss_weight': equiv_loss_weight,
        'equiv_warmup_steps': equiv_warmup_steps,
        'checkpoint_interval': checkpoint_interval,
        'print_interval': print_interval,
        'micro_batch_size': micro_batch_size,
    }

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        train_metrics, global_step = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device,
            equiv_loss_weight,
            equiv_loss_fn,
            equiv_warmup_steps,
            global_step,
            scaler=scaler,
            epoch=epoch,
            num_epochs=num_epochs,
            micro_batch_size=micro_batch_size,
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start_time

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['lr'].append(scheduler.get_last_lr()[0])

        if not math.isfinite(train_metrics['loss']):
            raise RuntimeError(
                f"Training loss became non-finite at epoch {epoch + 1}. "
                "Stopping before checkpoint overwrite."
            )

        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"| Epoch {epoch+1:3d}/{num_epochs} | Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e} |")
        tqdm.write(f"|{'-'*68}|")
        tqdm.write(f"| Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}% |")

        if val_dataloader is not None:
            val_metrics = validate(
                model,
                val_dataloader,
                criterion,
                device,
                micro_batch_size=micro_batch_size,
            )

            tqdm.write(f"| Val Loss:   {val_metrics['val_loss']:.4f} | Val Acc:   {val_metrics['val_accuracy']:.2f}% |")

            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_accuracy'])

            if not math.isfinite(val_metrics['val_loss']):
                raise RuntimeError(
                    f"Validation loss became non-finite at epoch {epoch + 1}. "
                    "Stopping before checkpoint overwrite."
                )

            if val_metrics['val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_accuracy']
                tqdm.write(f"| {'*** New Best Model! ***':^64} |")
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_metrics['val_accuracy'],
                    os.path.join(checkpoint_dir, 'best_model.pth'),
                    model_config=model_config,
                    history=history,
                    best_val_acc=best_val_acc,
                    global_step=global_step,
                    scaler=scaler,
                    training_config=training_config,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )

        tqdm.write(f"{'='*70}\n")

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_acc if val_dataloader is not None else train_metrics['accuracy'],
            latest_checkpoint_path,
            model_config=model_config,
            history=history,
            best_val_acc=best_val_acc,
            global_step=global_step,
            scaler=scaler,
            training_config=training_config,
            train_metrics=train_metrics,
            val_metrics=val_metrics if val_dataloader is not None else None,
        )

        _write_json(history_path, history)
        _write_json(
            train_state_path,
            {
                'epoch': epoch + 1,
                'best_val_acc': best_val_acc,
                'global_step': global_step,
                'latest_checkpoint': latest_checkpoint_path,
                'best_checkpoint': os.path.join(checkpoint_dir, 'best_model.pth'),
                'final_checkpoint': final_checkpoint_path,
                'history_file': str(history_path),
            },
        )

        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                train_metrics['accuracy'],
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                model_config=model_config,
                history=history,
                best_val_acc=best_val_acc,
                global_step=global_step,
                scaler=scaler,
                training_config=training_config,
                train_metrics=train_metrics,
                val_metrics=val_metrics if val_dataloader is not None else None,
            )

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        num_epochs - 1,
        best_val_acc if val_dataloader is not None else history['train_acc'][-1],
        final_checkpoint_path,
        model_config=model_config,
        history=history,
        best_val_acc=best_val_acc,
        global_step=global_step,
        scaler=scaler,
        training_config=training_config,
    )

    print('-' * 60)
    print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')

    return history

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    accuracy: float,
    filepath: str,
    model_config: Optional[Dict] = None,
    history: Optional[Dict[str, list]] = None,
    best_val_acc: Optional[float] = None,
    global_step: Optional[int] = None,
    scaler: Optional[TorchGradScaler] = None,
    training_config: Optional[Dict] = None,
    train_metrics: Optional[Dict[str, float]] = None,
    val_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Save model checkpoint with complete configuration for transfer.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        accuracy: Current accuracy
        filepath: Path to save checkpoint
        model_config: Model configuration dictionary for reconstruction
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy,
        'model_config': model_config,
        'pytorch_version': torch.__version__,
        'history': history,
        'best_val_acc': best_val_acc,
        'global_step': global_step,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'training_config': training_config,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_filepath = f"{filepath}.tmp"
    torch.save(checkpoint, tmp_filepath)
    os.replace(tmp_filepath, filepath)
    print(f"Checkpoint saved to {filepath}")


def export_model_torchscript(
    model: nn.Module,
    filepath: str,
    example_input: Optional[torch.Tensor] = None,
) -> None:
    """
    Export model to TorchScript format for cross-platform deployment.

    TorchScript models can be loaded in Python/C++/Java without source code.

    Args:
        model: Trained model to export
        filepath: Path to save the TorchScript model (.pt or .pth)
        example_input: Example input for tracing (optional, if not provided uses scripting)
    """
    model.eval()

    try:
        if example_input is not None:
            # Use tracing for better optimization
            scripted_model = torch.jit.trace(model, example_input)
        else:
            # Use scripting when tracing fails
            scripted_model = torch.jit.script(model)

        scripted_model.save(filepath)
        print(f"TorchScript model exported to {filepath}")
        print(f"  Model can be loaded with: torch.jit.load('{filepath}')")
    except Exception as e:
        print(f"Warning: TorchScript export failed: {e}")
        print("  Trying alternative export method...")
        # Fallback to state dict only
        torch.save(model.state_dict(), filepath.replace('.pt', '_state_dict.pt'))
        print(f"  State dict saved to {filepath.replace('.pt', '_state_dict.pt')}")


def export_model_onnx(
    model: nn.Module,
    filepath: str,
    example_input: torch.Tensor,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> None:
    """
    Export model to ONNX format for cross-framework deployment.

    ONNX models can be used with TensorRT, OpenVINO, ONNX Runtime, etc.

    Args:
        model: Trained model to export
        filepath: Path to save the ONNX model (.onnx)
        example_input: Example input tensor for tracing
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes configuration for variable batch size
    """
    model.eval()

    if input_names is None:
        input_names = ['point_coords']
    if output_names is None:
        output_names = ['logits']
    if dynamic_axes is None:
        dynamic_axes = {
            'point_coords': {0: 'batch_size', 1: 'num_points'},
            'logits': {0: 'batch_size'},
        }

    try:
        torch.onnx.export(
            model,
            example_input,
            filepath,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
        )
        print(f"ONNX model exported to {filepath}")
        print(f"  Input names: {input_names}")
        print(f"  Output names: {output_names}")
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
        print("  This is common for complex geometric operations.")
        print("  Use TorchScript export instead for this model.")


def save_model_for_transfer(
    model: nn.Module,
    save_dir: str,
    model_config: Dict,
    epoch: Optional[int] = None,
    accuracy: Optional[float] = None,
    example_input: Optional[torch.Tensor] = None,
) -> Dict[str, str]:
    """
    Save model in multiple formats for easy transfer and deployment.

    Creates:
        1. checkpoint.pth - Full training checkpoint (resume training)
        2. model_state_dict.pth - Model weights only (load into new model)
        3. model_config.json - Model architecture configuration
        4. model_torchscript.pt - TorchScript format (cross-platform)
        5. model.onnx - ONNX format (if example_input provided)

    Args:
        model: Trained model
        save_dir: Directory to save all model files
        model_config: Model configuration dictionary
        epoch: Training epoch (optional)
        accuracy: Model accuracy (optional)
        example_input: Example input for ONNX export (optional)

    Returns:
        Dictionary mapping format names to file paths
    """
    import json

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Save model config as JSON (human readable)
    config_path = save_path / "model_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    saved_files['config'] = str(config_path)
    print(f"Model config saved to {config_path}")

    # 2. Save state dict only (smallest file, for weight loading)
    state_dict_path = save_path / "model_state_dict.pth"
    torch.save(model.state_dict(), state_dict_path)
    saved_files['state_dict'] = str(state_dict_path)
    print(f"Model state dict saved to {state_dict_path}")

    # 3. Save complete checkpoint (for resuming training)
    checkpoint_path = save_path / "checkpoint.pth"
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'epoch': epoch,
        'accuracy': accuracy,
        'pytorch_version': torch.__version__,
    }
    torch.save(checkpoint, checkpoint_path)
    saved_files['checkpoint'] = str(checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    # 4. Export to TorchScript
    try:
        torchscript_path = save_path / "model_torchscript.pt"
        export_model_torchscript(model, str(torchscript_path), example_input)
        saved_files['torchscript'] = str(torchscript_path)
    except Exception as e:
        print(f"TorchScript export skipped: {e}")

    # 5. Export to ONNX (if example input provided)
    if example_input is not None:
        try:
            onnx_path = save_path / "model.onnx"
            export_model_onnx(model, str(onnx_path), example_input)
            saved_files['onnx'] = str(onnx_path)
        except Exception as e:
            print(f"ONNX export skipped: {e}")

    # Save a README with usage instructions
    readme_path = save_path / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("CAME-Net Model Export\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model Accuracy: {accuracy:.2f}%\n" if accuracy else "")
        f.write(f"Training Epoch: {epoch}\n" if epoch else "")
        f.write(f"PyTorch Version: {torch.__version__}\n\n")
        f.write("Files:\n")
        for name, path in saved_files.items():
            f.write(f"  {name}: {path}\n")
        f.write("\nUsage:\n")
        f.write("  1. Load for inference (TorchScript):\n")
        f.write("     model = torch.jit.load('model_torchscript.pt')\n")
        f.write("     output = model(input)\n\n")
        f.write("  2. Load weights into new model:\n")
        f.write("     model = CAMENet(**config)\n")
        f.write("     model.load_state_dict(torch.load('model_state_dict.pth'))\n\n")
        f.write("  3. Resume training:\n")
        f.write("     checkpoint = torch.load('checkpoint.pth')\n")
        f.write("     model.load_state_dict(checkpoint['model_state_dict'])\n")
    saved_files['readme'] = str(readme_path)

    print(f"\nAll model files saved to {save_dir}")
    return saved_files


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    filepath: str,
    device: torch.device
) -> Tuple[int, float]:
    """
    Load model checkpoint.

    Args:
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        filepath: Path to checkpoint file
        device: Device to load on

    Returns:
        Tuple of (epoch, accuracy)
    """
    checkpoint_obj = _torch_load_trusted_checkpoint(filepath, device)
    checkpoint_payload, model_state_dict = _extract_checkpoint_payload(checkpoint_obj)

    model.load_state_dict(model_state_dict)

    if checkpoint_payload is None:
        print(f"Model weights loaded from {filepath}")
        return -1, 0.0

    if optimizer is not None and checkpoint_payload.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint_payload['optimizer_state_dict'])

    if scheduler is not None and checkpoint_payload.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint_payload['scheduler_state_dict'])

    epoch = int(checkpoint_payload.get('epoch', -1))
    accuracy = float(checkpoint_payload.get('accuracy', 0.0))

    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {epoch+1} with accuracy {accuracy:.2f}%")

    return epoch, accuracy


def evaluate_modelnet(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device,
    micro_batch_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on ModelNet test set.

    Args:
        model: CAME-Net model
        test_dataloader: Test data loader
        device: Device to evaluate on

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()

    correct = 0
    total = 0

    num_classes = _dataset_num_classes(test_dataloader)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for batch in test_dataloader:
            batch_size = _batch_size_from_batch(batch)
            try:
                for micro_batch, _ in _iter_micro_batches(batch, micro_batch_size):
                    labels = micro_batch['labels'].to(device, non_blocking=True)
                    logits = _model_forward_from_batch(model, micro_batch, device)

                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    for i in range(len(labels)):
                        label = labels[i].item()
                        class_total[label] += 1
                        if predicted[i] == label:
                            class_correct[label] += 1
            except RuntimeError as exc:
                if device.type == 'cuda' and _is_cuda_oom_error(exc):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        'CUDA OOM during evaluation forward. '
                        f'Current outer batch size={batch_size}, micro_batch_size={micro_batch_size}. '
                        'Try a smaller --num-points (e.g. 256) or keep --micro-batch-size 1.'
                    ) from exc
                raise

    accuracy = 100.0 * correct / total

    class_accuracies = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies.append(100.0 * class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    mean_class_accuracy = sum(class_accuracies) / len(class_accuracies)

    return {
        'overall_accuracy': accuracy,
        'mean_class_accuracy': mean_class_accuracy,
        'per_class_accuracy': class_accuracies
    }


def _resolve_device(device_arg: str) -> torch.device:
    requested = (device_arg or 'cuda').strip().lower()
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install PyTorch with CUDA support.")
    return torch.device(requested)


def _resolve_resume_checkpoint(resume_arg: str, checkpoint_dir: str) -> Optional[str]:
    normalized = (resume_arg or 'auto').strip()
    lowered = normalized.lower()
    if lowered in {'off', 'none', 'false', '0'}:
        return None

    checkpoint_path = Path(checkpoint_dir)
    if lowered == 'auto':
        direct_candidates = [
            checkpoint_path / 'latest_model.pth',
            checkpoint_path / 'final_model.pth',
            checkpoint_path / 'best_model.pth',
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                return str(candidate)

        periodic_candidates = sorted(
            checkpoint_path.glob('checkpoint_epoch_*.pth'),
            key=lambda path: int(path.stem.split('_')[-1]),
            reverse=True,
        )
        for candidate in periodic_candidates:
            if candidate.exists():
                return str(candidate)
        return None

    explicit_path = Path(normalized)
    if explicit_path.is_dir():
        return _resolve_resume_checkpoint('auto', str(explicit_path))
    if not explicit_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {explicit_path}")
    return str(explicit_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Long-run ModelNet40 training entrypoint for CAME-Net.',
    )
    parser.add_argument('--data-root', type=str, default=None, help='ModelNet40 root. Supports either .../ModelNet40 or .../ModelNet40/ModelNet40.')
    parser.add_argument('--device', type=str, default='cuda', help='Training device, e.g. cuda, cuda:0, cpu, or auto.')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/modelnet40_3080_long_run', help='Directory for latest/best/periodic checkpoints.')
    parser.add_argument('--resume', type=str, default='auto', help='Resume path, checkpoint directory, auto, or off.')
    parser.add_argument('--epochs', type=int, default=1000, help='Total epoch budget. When resuming, this is the final target epoch count.')
    parser.add_argument('--batch-size', type=int, default=4, help='CPU-side batch size. Training will be split into smaller micro-batches on GPU.')
    parser.add_argument('--micro-batch-size', type=int, default=1, help='Maximum per-forward GPU batch size. Keep at 1 or 2 for 10 GB GPUs.')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader worker count.')
    parser.add_argument('--num-points', type=int, default=512, help='Surface-sampled points per object. Use 256 if 10 GB still OOMs.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='AdamW learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW weight decay.')
    parser.add_argument('--equiv-loss-weight', type=float, default=0.1, help='Soft equivariance regularization weight.')
    parser.add_argument('--equiv-warmup-steps', type=int, default=10000, help='Warmup steps for equivariance regularization.')
    parser.add_argument('--checkpoint-interval', type=int, default=25, help='Save checkpoint_epoch_N every N epochs in addition to latest/best.')
    parser.add_argument('--print-interval', type=int, default=1, help='Reserved for trainer log cadence.')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of CAME layers for fresh training. Resume checkpoints override this.')
    parser.add_argument('--num-heads', type=int, default=4, help='Attention heads per layer for fresh training. Resume checkpoints override this.')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Point branch hidden dimension.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--disable-equiv-loss', action='store_true', help='Disable equivariance loss for ablations or faster training.')
    parser.add_argument('--enable-amp', action='store_true', help='Enable AMP mixed precision. Disabled by default because the geometric pipeline is numerically fragile in fp16.')
    parser.add_argument('--enable-train-augmentation', action='store_true', help='Enable stochastic surface resampling plus random rigid augmentation on the training split. Disabled by default because it easily causes single-class collapse in the current fp32/no-equiv regime.')
    return parser


def main() -> None:
    from method.came_net import CAMENet, count_parameters
    from method.equiv_loss import equivariance_loss_efficient

    args = _build_arg_parser().parse_args()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    device = _resolve_device(args.device)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

    resolved_data_root = resolve_modelnet_data_root(args.data_root)
    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    os.makedirs(checkpoint_dir, exist_ok=True)
    resume_checkpoint = _resolve_resume_checkpoint(args.resume, checkpoint_dir)
    resume_model_config = _resolve_model_config_from_checkpoint(resume_checkpoint, torch.device('cpu')) if resume_checkpoint else None

    print(f'Using device: {device}')
    if device.type == 'cuda':
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f'GPU: {torch.cuda.get_device_name(device_index)}')
        print(f'CUDA version: {torch.version.cuda}')

    train_loader, val_loader = create_default_modelnet_dataloaders(
        data_root=str(resolved_data_root),
        num_points=args.num_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_data_augmentation=args.enable_train_augmentation,
    )
    dataset_num_classes = _dataset_num_classes(train_loader)
    val_num_classes = _dataset_num_classes(val_loader)
    if dataset_num_classes != val_num_classes:
        raise RuntimeError(
            f'Train/validation dataset class count mismatch: '
            f'{dataset_num_classes} vs {val_num_classes}.'
        )
    dataset_class_names = _dataset_class_names(train_loader)

    print(f'PyTorch version: {torch.__version__}')
    print(f'ModelNet40 root: {resolved_data_root}')
    print(f'Dataset classes: {dataset_num_classes}')
    if dataset_class_names is not None:
        print(f'Class names: {dataset_class_names}')
    print(f'Checkpoint directory: {checkpoint_dir}')
    print(f'Resume source: {resume_checkpoint if resume_checkpoint else "None"}')
    print(f'Points / batch / micro-batch: {args.num_points} / {args.batch_size} / {args.micro_batch_size}')
    print(f'AMP requested: {args.enable_amp}')
    print(f'Train augmentation enabled: {args.enable_train_augmentation}')

    requested_model_config = {
        'num_classes': dataset_num_classes,
        'point_feature_dim': 0,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'multimodal': False,
    }
    if resume_model_config is not None:
        if int(resume_model_config.get('num_classes', dataset_num_classes)) != dataset_num_classes:
            raise RuntimeError(
                f"Resume checkpoint expects num_classes={resume_model_config.get('num_classes')}, "
                f"but the current dataset root exposes num_classes={dataset_num_classes}. "
                "Use a fresh checkpoint directory or pass --resume off."
            )
        overridden_fields = [
            f"{field}: {requested_model_config[field]} -> {resume_model_config[field]}"
            for field in ('num_layers', 'num_heads', 'hidden_dim', 'dropout')
            if field in resume_model_config and requested_model_config[field] != resume_model_config[field]
        ]
        if overridden_fields:
            print('Resume checkpoint architecture overrides CLI/default architecture:')
            for item in overridden_fields:
                print(f'  {item}')
        requested_model_config.update(resume_model_config)

    model = CAMENet(
        num_classes=int(requested_model_config['num_classes']),
        point_feature_dim=int(requested_model_config['point_feature_dim']),
        num_layers=int(requested_model_config['num_layers']),
        num_heads=int(requested_model_config['num_heads']),
        hidden_dim=int(requested_model_config['hidden_dim']),
        dropout=float(requested_model_config['dropout']),
        multimodal=bool(requested_model_config['multimodal']),
    ).to(device)
    if resume_checkpoint is None:
        _stabilize_fresh_classification_head(model)
        print('Applied stabilized fresh classification-head initialization.')

    model_config = {
        'num_classes': int(requested_model_config['num_classes']),
        'point_feature_dim': int(requested_model_config['point_feature_dim']),
        'num_layers': int(requested_model_config['num_layers']),
        'num_heads': int(requested_model_config['num_heads']),
        'hidden_dim': int(requested_model_config['hidden_dim']),
        'multivector_dim': 16,
        'dropout': float(requested_model_config['dropout']),
        'multimodal': bool(requested_model_config['multimodal']),
    }

    print(f"Model parameters: {count_parameters(model):,}")

    _write_json(
        Path(checkpoint_dir) / 'run_config.json',
        {
            'data_root': str(resolved_data_root),
            'device': str(device),
            'resume_checkpoint': resume_checkpoint,
            **vars(args),
        },
    )

    history = train_came_net(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        equiv_loss_weight=0.0 if args.disable_equiv_loss else args.equiv_loss_weight,
        equiv_loss_fn=None if args.disable_equiv_loss else equivariance_loss_efficient,
        equiv_warmup_steps=args.equiv_warmup_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        print_interval=args.print_interval,
        model_config=model_config,
        resume_from=resume_checkpoint,
        micro_batch_size=args.micro_batch_size,
        amp_enabled=args.enable_amp,
    )

    best_checkpoint = Path(checkpoint_dir) / 'best_model.pth'
    latest_checkpoint = Path(checkpoint_dir) / 'latest_model.pth'
    final_checkpoint = Path(checkpoint_dir) / 'final_model.pth'
    evaluation_source = best_checkpoint if best_checkpoint.exists() else latest_checkpoint if latest_checkpoint.exists() else final_checkpoint

    if evaluation_source.exists():
        load_checkpoint(model, optimizer=None, scheduler=None, filepath=str(evaluation_source), device=device)
        eval_metrics = evaluate_modelnet(model, val_loader, device, micro_batch_size=args.micro_batch_size)
        _write_json(Path(checkpoint_dir) / 'eval_metrics.json', eval_metrics)
        print('\nEvaluation on ModelNet40 test split:')
        print(f"  Overall accuracy: {eval_metrics['overall_accuracy']:.2f}%")
        print(f"  Mean class accuracy: {eval_metrics['mean_class_accuracy']:.2f}%")

    print('\nTraining history summary:')
    if history['train_acc']:
        print(f"  Final train accuracy: {history['train_acc'][-1]:.2f}%")
    if history['val_acc']:
        print(f"  Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    print(f'  Latest checkpoint: {latest_checkpoint}')
    print(f'  Best checkpoint:   {best_checkpoint}')
    print(f'  Final checkpoint:  {final_checkpoint}')


if __name__ == '__main__':
    main()

