"""
dp_shift_bench/datasets/image_shifts.py

Dataset loaders for the image track of DP-ShiftBench.
All datasets are normalized to: 32x32, 3-channel RGB, [0, 1] range.
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Canonical transforms — all outputs are 3 x 32 x 32, values in [0, 1]
# ---------------------------------------------------------------------------

# SVHN is already 32×32 RGB — just normalise to [0,1]
svhn_transform = transforms.Compose([
    transforms.ToTensor(),
])

# MNIST is 28×28 grayscale — resize and replicate to 3 channels
mnist_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# CIFAR-10 is already 32×32 RGB
cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
])

# FashionMNIST is 28×28 grayscale — same treatment as MNIST
fashionmnist_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


# ---------------------------------------------------------------------------
# ClassSubset: filter to specific classes and remap labels to 0..len-1
# ---------------------------------------------------------------------------

class ClassSubset(Dataset):
    """Wraps a dataset, filtering to specific classes and remapping labels."""

    def __init__(self, dataset: Dataset, classes: List[int]):
        self.classes = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Handle SVHN (.labels numpy array) vs MNIST/FashionMNIST (.targets tensor/list)
        if hasattr(dataset, "labels"):
            targets = np.array(dataset.labels)
        elif hasattr(dataset, "targets"):
            targets = np.array(dataset.targets)
        else:
            # Fall back to iterating (slow but safe for Subset wrappers)
            targets = np.array([dataset[i][1] for i in range(len(dataset))])

        self.indices = [i for i, t in enumerate(targets) if int(t) in self.classes]
        self.dataset = dataset

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, self.class_to_idx[int(label)]

    def __len__(self):
        return len(self.indices)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_class_subset(dataset: Dataset, classes: List[int]) -> ClassSubset:
    """Return a ClassSubset filtered to *classes* with remapped labels."""
    return ClassSubset(dataset, classes)


def get_random_subset(dataset: Dataset, fraction: float, seed: int = 42) -> Subset:
    """Randomly sample *fraction* of *dataset* (reproducible)."""
    rng = np.random.RandomState(seed)
    n = len(dataset)
    n_subset = max(1, int(n * fraction))
    indices = rng.choice(n, n_subset, replace=False).tolist()
    return Subset(dataset, indices)


class AugmentedDataset(Dataset):
    """Wraps a dataset and applies extra augmentation on top of ToTensor."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.aug = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.GaussianBlur(kernel_size=3),
        ])

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.aug(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


def get_augmented_dataset(dataset: Dataset) -> AugmentedDataset:
    """Wrap *dataset* with ColorJitter + GaussianBlur augmentation."""
    return AugmentedDataset(dataset)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def get_image_experiment_data(
    experiment_id: str,
    data_root: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
    seed: int = 42,
    debug: bool = False,
) -> dict:
    """
    Returns a dict with DataLoaders and metadata for one image experiment.

    Keys:
        pretrain_loader        : DataLoader or None  (None for IMG-00)
        finetune_train_loader  : DataLoader
        finetune_test_loader   : DataLoader
        num_pretrain_classes   : int
        num_finetune_classes   : int
        experiment_id          : str
        pretrain_dataset       : Dataset or None
        finetune_train_dataset : Dataset
    """
    os.makedirs(data_root, exist_ok=True)
    pin_memory = torch.cuda.is_available()

    # ---- always load MNIST for finetuning ----
    mnist_train_full = datasets.MNIST(
        data_root, train=True, download=True, transform=mnist_transform
    )
    mnist_test_full = datasets.MNIST(
        data_root, train=False, download=True, transform=mnist_transform
    )

    pretrain_dataset = None
    num_pretrain_classes = 0
    num_finetune_classes = 10
    finetune_train: Dataset = mnist_train_full
    finetune_test: Dataset = mnist_test_full

    # ---- experiment-specific setup ----
    if experiment_id == "IMG-00":
        # No pretraining — random init
        pass

    elif experiment_id == "IMG-01":
        svhn = datasets.SVHN(
            data_root, split="train", download=True, transform=svhn_transform
        )
        pretrain_dataset = svhn
        num_pretrain_classes = 10

    elif experiment_id == "IMG-02":
        svhn = datasets.SVHN(
            data_root, split="train", download=True, transform=svhn_transform
        )
        pretrain_dataset = ClassSubset(svhn, [0, 1, 2, 3, 4])
        finetune_train = ClassSubset(mnist_train_full, [0, 1, 2, 3, 4])
        finetune_test = ClassSubset(mnist_test_full, [0, 1, 2, 3, 4])
        num_pretrain_classes = 5
        num_finetune_classes = 5

    elif experiment_id == "IMG-03":
        # Pretrain on SVHN 5-9, finetune on MNIST 0-4 (mismatched labels)
        svhn = datasets.SVHN(
            data_root, split="train", download=True, transform=svhn_transform
        )
        pretrain_dataset = ClassSubset(svhn, [5, 6, 7, 8, 9])
        finetune_train = ClassSubset(mnist_train_full, [0, 1, 2, 3, 4])
        finetune_test = ClassSubset(mnist_test_full, [0, 1, 2, 3, 4])
        num_pretrain_classes = 5
        num_finetune_classes = 5

    elif experiment_id == "IMG-04":
        # Autoencoder pretrain — labels unused
        svhn = datasets.SVHN(
            data_root, split="train", download=True, transform=svhn_transform
        )
        pretrain_dataset = svhn
        num_pretrain_classes = 10

    elif experiment_id == "IMG-05":
        svhn = datasets.SVHN(
            data_root, split="train", download=True, transform=svhn_transform
        )
        pretrain_dataset = get_random_subset(svhn, 0.10, seed)
        num_pretrain_classes = 10

    elif experiment_id == "IMG-06":
        svhn = datasets.SVHN(
            data_root, split="train", download=True, transform=svhn_transform
        )
        pretrain_dataset = get_random_subset(svhn, 0.25, seed)
        num_pretrain_classes = 10

    elif experiment_id == "IMG-07":
        svhn = datasets.SVHN(
            data_root, split="train", download=True, transform=svhn_transform
        )
        pretrain_dataset = get_random_subset(svhn, 0.50, seed)
        num_pretrain_classes = 10

    elif experiment_id == "IMG-08":
        cifar10 = datasets.CIFAR10(
            data_root, train=True, download=True, transform=cifar10_transform
        )
        pretrain_dataset = cifar10
        num_pretrain_classes = 10

    elif experiment_id == "IMG-09":
        svhn = datasets.SVHN(
            data_root, split="train", download=True, transform=svhn_transform
        )
        pretrain_dataset = get_augmented_dataset(svhn)
        num_pretrain_classes = 10

    elif experiment_id == "IMG-10":
        fmnist = datasets.FashionMNIST(
            data_root, train=True, download=True, transform=fashionmnist_transform
        )
        pretrain_dataset = fmnist
        num_pretrain_classes = 10

    else:
        raise ValueError(f"Unknown experiment_id: {experiment_id}")

    # ---- debug subsets (first N samples) ----
    if debug:
        if pretrain_dataset is not None:
            n = min(500, len(pretrain_dataset))
            pretrain_dataset = Subset(pretrain_dataset, list(range(n)))
        n_ft = min(500, len(finetune_train))
        finetune_train = Subset(finetune_train, list(range(n_ft)))
        n_te = min(200, len(finetune_test))
        finetune_test = Subset(finetune_test, list(range(n_te)))

    # ---- DataLoaders ----
    pretrain_loader = None
    if pretrain_dataset is not None:
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    finetune_train_loader = DataLoader(
        finetune_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    finetune_test_loader = DataLoader(
        finetune_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "pretrain_loader": pretrain_loader,
        "finetune_train_loader": finetune_train_loader,
        "finetune_test_loader": finetune_test_loader,
        "num_pretrain_classes": num_pretrain_classes,
        "num_finetune_classes": num_finetune_classes,
        "experiment_id": experiment_id,
        "pretrain_dataset": pretrain_dataset,
        "finetune_train_dataset": finetune_train,
    }
