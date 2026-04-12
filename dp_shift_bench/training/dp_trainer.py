"""
dp_shift_bench/training/dp_trainer.py

DP-SGD finetuning using Opacus.
Supports epsilon in (0, ∞) and the non-private baseline (epsilon = inf).
"""

from typing import Optional

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from dp_shift_bench.metrics.evaluation import evaluate_accuracy


# ---------------------------------------------------------------------------
# Non-private baseline
# ---------------------------------------------------------------------------

def _train_standard(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    verbose: bool,
) -> dict:
    """Plain SGD training — used when epsilon = inf."""
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        epoch_loss = total_loss / max(len(train_loader), 1)
        train_acc = correct / max(total, 1)
        test_acc = evaluate_accuracy(model, test_loader, device)

        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if verbose:
            print(
                f"  [Non-private] Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | Train: {train_acc:.4f} | Test: {test_acc:.4f}"
            )

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "final_test_acc": test_accs[-1] if test_accs else 0.0,
        "best_test_acc": max(test_accs) if test_accs else 0.0,
        "epsilon_actual": float("inf"),
        "epochs_run": epochs,
    }


# ---------------------------------------------------------------------------
# DP-SGD finetuning
# ---------------------------------------------------------------------------

def finetune_dp(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epsilon: float,
    delta: float = 1e-5,
    epochs: int = 15,
    lr: float = 0.001,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Finetune a model with DP-SGD (Opacus) or without DP if epsilon = inf.

    Returns:
        {
            "train_losses"  : List[float],
            "train_accs"    : List[float],
            "test_accs"     : List[float],
            "final_test_acc": float,
            "best_test_acc" : float,
            "epsilon_actual": float,
            "epochs_run"    : int,
        }
    """
    torch.manual_seed(seed)
    model = model.to(device)

    # ---- non-private baseline ----
    if epsilon == float("inf"):
        return _train_standard(
            model, train_loader, test_loader, epochs, lr, device, verbose
        )

    # ---- validate + auto-fix model for Opacus ----
    model = ModuleValidator.fix(model)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        raise ValueError(f"Model not compatible with Opacus after fix: {errors}")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    train_losses, train_accs, test_accs = [], [], []
    eps_spent = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in private_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        epoch_loss = total_loss / max(len(private_loader), 1)
        train_acc = correct / max(total, 1)
        test_acc = evaluate_accuracy(model, test_loader, device)
        eps_spent = privacy_engine.get_epsilon(delta=delta)

        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if verbose:
            print(
                f"  [DP ε≤{epsilon}] Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | Train: {train_acc:.4f} | "
                f"Test: {test_acc:.4f} | ε_spent: {eps_spent:.4f}"
            )

    # Unwrap Opacus GradSampleModule before returning
    final_model = model._module if hasattr(model, "_module") else model

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "final_test_acc": test_accs[-1] if test_accs else 0.0,
        "best_test_acc": max(test_accs) if test_accs else 0.0,
        "epsilon_actual": float(eps_spent),
        "epochs_run": epochs,
        "model": final_model,
    }
