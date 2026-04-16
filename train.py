"""
train.py - Training Script for CAME-Net

This module provides the training and evaluation functions for the CAME-Net model,
including the training loop, validation, and checkpointing utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from typing import Optional, Dict, Tuple
import os


def _model_forward_from_batch(model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    point_coords = batch.get('point_coords')
    point_features = batch.get('point_features')
    image_patches = batch.get('image_patches')
    text_tokens = batch.get('text_tokens')

    return model(
        point_coords=point_coords.to(device) if point_coords is not None else None,
        point_features=point_features.to(device) if point_features is not None else None,
        image_patches=image_patches.to(device) if image_patches is not None else None,
        text_tokens=text_tokens.to(device) if text_tokens is not None else None,
    )


def _equivariance_loss_from_batch(equiv_loss_fn, model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device):
    if equiv_loss_fn is None:
        return None

    point_coords = batch['point_coords'].to(device)
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
            labels.to(device) if labels is not None else None,
            point_features=point_features.to(device) if point_features is not None else None,
            image_patches=image_patches.to(device) if image_patches is not None else None,
            text_tokens=text_tokens.to(device) if text_tokens is not None else None,
        )
    finally:
        if was_training:
            model.train()


def create_default_modelnet_dataloaders(
    data_root: Optional[str] = None,
    num_points: int = 1024,
    batch_size: int = 8,
):
    from data_utils import ModelNetDataset, collate_fn

    if data_root is None:
        data_root = Path(__file__).resolve().parent / "ModelNet40" / "ModelNet40"

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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    equiv_loss_weight: float = 0.1,
    equiv_loss_fn=None,
    equiv_warmup_steps: int = 0,
    start_step: int = 0
) -> Tuple[Dict[str, float], int]:
    """
    Train for one epoch.

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

    Returns:
        Tuple containing metrics dictionary and updated global step
    """
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_equiv_loss = 0.0
    correct = 0
    total = 0

    global_step = start_step

    for batch in dataloader:
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        logits = _model_forward_from_batch(model, batch, device)

        task_loss = criterion(logits, labels)

        loss = task_loss
        if equiv_loss_fn is not None and equiv_loss_weight > 0:
            if equiv_warmup_steps > 0:
                warmup_progress = min(1.0, global_step / float(max(1, equiv_warmup_steps)))
            else:
                warmup_progress = 1.0
            current_lambda = equiv_loss_weight * warmup_progress
            if current_lambda > 0:
                equiv_loss = _equivariance_loss_from_batch(equiv_loss_fn, model, batch, device)
                loss = task_loss + current_lambda * equiv_loss
                total_equiv_loss += equiv_loss.item()

        loss.backward()

        gradients_exist = any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in model.parameters() if p.requires_grad
        )
        if not gradients_exist:
            print('Warning: No gradients computed')

        optimizer.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()

        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        global_step += 1

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
    device: torch.device
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

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device)

            logits = _model_forward_from_batch(model, batch, device)

            loss = criterion(logits, labels)

            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    num_batches = len(dataloader)

    return {
        'val_loss': total_loss / num_batches,
        'val_accuracy': 100.0 * correct / total
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
    print_interval: int = 5
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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate / 100
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    global_step = 0

    print('Starting training...')
    print(f'Device: {device}')
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Number of epochs: {num_epochs}')
    print(f'Learning rate: {learning_rate}')
    print(f'Equivariance loss weight: {equiv_loss_weight}')
    print(f'Equivariance warmup steps: {equiv_warmup_steps}')
    print('-' * 60)

    for epoch in range(num_epochs):
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
            global_step
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start_time

        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] ({epoch_time:.1f}s) "
                  f"Train Loss: {train_metrics['loss']:.4f} "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['lr'].append(scheduler.get_last_lr()[0])

        if val_dataloader is not None:
            val_metrics = validate(
                model,
                val_dataloader,
                criterion,
                device
            )

            if (epoch + 1) % print_interval == 0 or epoch == 0:
                print(f"                    "
                      f"Val Loss: {val_metrics['val_loss']:.4f} "
                      f"Val Acc: {val_metrics['val_accuracy']:.2f}%")

            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_accuracy'])

            if val_metrics['val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_accuracy']
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_metrics['val_accuracy'],
                    os.path.join(checkpoint_dir, 'best_model.pth')
                )

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                train_metrics['accuracy'],
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
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
    filepath: str
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch: Current epoch
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy': accuracy
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


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
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']

    print(f"Checkpoint loaded from {filepath}")
    print(f"Resuming from epoch {epoch+1} with accuracy {accuracy:.2f}%")

    return epoch, accuracy


def evaluate_modelnet(
    model: nn.Module,
    test_dataloader: DataLoader,
    device: torch.device
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

    class_correct = [0] * 40
    class_total = [0] * 40

    with torch.no_grad():
        for batch in test_dataloader:
            labels = batch['labels'].to(device)

            logits = _model_forward_from_batch(model, batch, device)

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    accuracy = 100.0 * correct / total

    class_accuracies = []
    for i in range(40):
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


if __name__ == '__main__':
    from came_net import CAMENet, count_parameters
    from equiv_loss import equivariance_loss_efficient

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = CAMENet(
        num_classes=40,
        point_feature_dim=0,
        num_layers=4,
        num_heads=8
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    train_loader, val_loader = create_default_modelnet_dataloaders(
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
        print_interval=2
    )

    print("\nTraining history:")
    print(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
    if len(history['val_acc']) > 0:
        print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")

