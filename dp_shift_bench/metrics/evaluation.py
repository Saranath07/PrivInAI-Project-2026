"""
dp_shift_bench/metrics/evaluation.py

Evaluation metrics: accuracy, per-class accuracy, loss.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: str = "cuda") -> float:
    """Standard top-1 accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            # Handle Opacus GradSampleModule wrapper transparently
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_per_class_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: str = "cuda",
) -> dict:
    """Per-class top-1 accuracy.

    Returns:
        {"class_0": 0.95, "class_1": 0.87, ...}
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for i in range(num_classes):
                mask = target == i
                class_correct[i] += ((pred == target) & mask).sum().item()
                class_total[i] += mask.sum().item()

    return {
        f"class_{i}": (class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0)
        for i in range(num_classes)
    }


def evaluate_loss(
    model: nn.Module, dataloader: DataLoader, device: str = "cuda"
) -> float:
    """Average cross-entropy loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0
