"""
tests/test_training.py

Tests for pretraining and DP-SGD finetuning.
Run with debug=True to keep datasets tiny.
"""

import pytest
import torch

from dp_shift_bench.datasets.image_shifts import get_image_experiment_data
from dp_shift_bench.models.image_models import ConvAutoencoder, SmallCNN
from dp_shift_bench.training.dp_trainer import finetune_dp
from dp_shift_bench.training.standard_trainer import pretrain_autoencoder, pretrain_supervised


def test_dp_finetune_runs():
    """DP finetuning should complete without errors on a tiny dataset."""
    model = SmallCNN(num_classes=10)
    data = get_image_experiment_data("IMG-00", debug=True)
    result = finetune_dp(
        model,
        data["finetune_train_loader"],
        data["finetune_test_loader"],
        epsilon=8.0,
        epochs=2,
        device="cpu",
    )
    assert "final_test_acc" in result
    assert 0.0 <= result["final_test_acc"] <= 1.0
    assert result["epochs_run"] == 2


def test_non_private_baseline():
    """epsilon=inf should run without Opacus and return epsilon_actual=inf."""
    model = SmallCNN(num_classes=10)
    data = get_image_experiment_data("IMG-00", debug=True)
    result = finetune_dp(
        model,
        data["finetune_train_loader"],
        data["finetune_test_loader"],
        epsilon=float("inf"),
        epochs=2,
        device="cpu",
    )
    assert result["epsilon_actual"] == float("inf")
    assert "train_losses" in result


def test_dp_finetune_small_epsilon():
    """DP training with small epsilon should still run (noise will be high)."""
    model = SmallCNN(num_classes=10)
    data = get_image_experiment_data("IMG-00", debug=True)
    result = finetune_dp(
        model,
        data["finetune_train_loader"],
        data["finetune_test_loader"],
        epsilon=0.5,
        epochs=1,
        device="cpu",
    )
    assert result["epsilon_actual"] <= 1.0  # should be ≤ target


def test_pretrain_supervised_runs():
    """Supervised pretraining should run and return correct keys."""
    model = SmallCNN(num_classes=10)
    data = get_image_experiment_data("IMG-01", debug=True)
    result = pretrain_supervised(
        model, data["pretrain_loader"], epochs=2, device="cpu", verbose=False
    )
    assert "train_losses" in result
    assert len(result["train_losses"]) == 2
    assert 0.0 <= result["final_train_acc"] <= 1.0


def test_pretrain_autoencoder_runs():
    """Autoencoder pretraining should run without errors."""
    ae = ConvAutoencoder()
    data = get_image_experiment_data("IMG-04", debug=True)
    result = pretrain_autoencoder(
        ae, data["pretrain_loader"], epochs=2, device="cpu", verbose=False
    )
    assert "train_losses" in result
    assert len(result["train_losses"]) == 2


def test_finetune_result_structure():
    """Result dict should have all required keys."""
    model = SmallCNN(num_classes=10)
    data = get_image_experiment_data("IMG-00", debug=True)
    result = finetune_dp(
        model,
        data["finetune_train_loader"],
        data["finetune_test_loader"],
        epsilon=float("inf"),
        epochs=1,
        device="cpu",
    )
    required = {
        "train_losses", "train_accs", "test_accs",
        "final_test_acc", "best_test_acc", "epsilon_actual", "epochs_run",
    }
    assert required.issubset(set(result.keys()))
