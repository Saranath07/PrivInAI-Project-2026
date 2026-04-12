"""
tests/test_datasets.py

Tests for dataset loaders (use debug=True to avoid full dataset downloads).
"""

import pytest
import torch

from dp_shift_bench.datasets.image_shifts import get_image_experiment_data


def test_svhn_mnist_shapes():
    """All finetune images should be 3×32×32 and in [0, 1]."""
    data = get_image_experiment_data("IMG-01", debug=True)
    batch, labels = next(iter(data["finetune_train_loader"]))
    assert batch.shape[1:] == (3, 32, 32), f"Unexpected shape: {batch.shape}"
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


def test_pretrain_shapes():
    """Pretrain images should also be 3×32×32 and in [0, 1]."""
    data = get_image_experiment_data("IMG-01", debug=True)
    batch, _ = next(iter(data["pretrain_loader"]))
    assert batch.shape[1:] == (3, 32, 32)
    assert batch.min() >= 0.0
    assert batch.max() <= 1.0


def test_class_subset_filtering():
    """IMG-02 finetune should only have classes 0–4."""
    data = get_image_experiment_data("IMG-02", debug=True)
    for batch, labels in data["finetune_train_loader"]:
        assert labels.max().item() <= 4
        break
    assert data["num_finetune_classes"] == 5


def test_class_subset_mismatch():
    """IMG-03: pretrain has 5 classes (5-9 SVHN), finetune has 5 classes (0-4 MNIST)."""
    data = get_image_experiment_data("IMG-03", debug=True)
    assert data["num_pretrain_classes"] == 5
    assert data["num_finetune_classes"] == 5


def test_img00_no_pretrain():
    """IMG-00 should have no pretrain loader."""
    data = get_image_experiment_data("IMG-00", debug=True)
    assert data["pretrain_loader"] is None
    assert data["num_pretrain_classes"] == 0


def test_data_scarcity_subset_loads():
    """IMG-05/06/07 pretrain loaders should exist."""
    for exp_id in ("IMG-05", "IMG-06", "IMG-07"):
        data = get_image_experiment_data(exp_id, debug=True)
        assert data["pretrain_loader"] is not None


def test_debug_mode_small():
    """Debug mode should produce small datasets (≤ 600 train samples)."""
    data = get_image_experiment_data("IMG-01", debug=True)
    total = sum(len(b) for b, _ in data["finetune_train_loader"])
    assert total <= 600


def test_all_experiments_load():
    """All 11 experiments should load without error in debug mode."""
    for exp_id in [f"IMG-{i:02d}" for i in range(11)]:
        data = get_image_experiment_data(exp_id, debug=True)
        assert "finetune_train_loader" in data
        assert "finetune_test_loader" in data


def test_return_dict_keys():
    """Return dict should have all required keys."""
    required = {
        "pretrain_loader",
        "finetune_train_loader",
        "finetune_test_loader",
        "num_pretrain_classes",
        "num_finetune_classes",
        "experiment_id",
        "pretrain_dataset",
        "finetune_train_dataset",
    }
    data = get_image_experiment_data("IMG-01", debug=True)
    assert required.issubset(set(data.keys()))
