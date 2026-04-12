"""
dp_shift_bench/training/standard_trainer.py

Non-private pretraining loops: supervised (cross-entropy) and autoencoder (MSE).
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dp_shift_bench.models.image_models import ConvAutoencoder


def pretrain_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Standard supervised pretraining with SGD + cosine-annealing LR.

    Returns:
        {
            "train_losses": List[float],
            "train_accs": List[float],
            "final_train_acc": float,
            "model_path": str or None,
        }
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses: list = []
    train_accs: list = []

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

        scheduler.step()

        epoch_loss = total_loss / max(len(train_loader), 1)
        epoch_acc = correct / max(total, 1)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        if verbose:
            print(
                f"  [Pretrain] Epoch {epoch + 1:3d}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "final_train_acc": train_accs[-1] if train_accs else 0.0,
        "model_path": save_path,
    }


def pretrain_autoencoder(
    model: ConvAutoencoder,
    train_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.001,
    device: str = "cuda",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Autoencoder pretraining with MSE reconstruction loss.

    Returns:
        {
            "train_losses": List[float],
            "model_path": str or None,
        }
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses: list = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, _ in train_loader:          # ignore labels
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(len(train_loader), 1)
        train_losses.append(epoch_loss)

        if verbose:
            print(
                f"  [AE Pretrain] Epoch {epoch + 1:3d}/{epochs} | "
                f"Recon Loss: {epoch_loss:.6f}"
            )

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(model.state_dict(), save_path)

    return {
        "train_losses": train_losses,
        "model_path": save_path,
    }
