"""
dp_shift_bench/training/utils.py

Training utilities: seed setting, device detection, result I/O.
"""

import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across torch, numpy, random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def get_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_results(results: dict, path: str) -> None:
    """Save experiment results as JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(path: str) -> dict:
    """Load experiment results from JSON."""
    with open(path, "r") as f:
        return json.load(f)
