"""
experiments/run_linear_probing.py

Linear probing: measures how discriminative pretrained features are for the
downstream MNIST task WITHOUT any DP training.

For each pretrained checkpoint (IMG-01 to IMG-10), this script:
  1. Loads the pretrained SmallCNN feature extractor (frozen)
  2. Trains a single linear layer on MNIST (no DP, small LR, 10 epochs)
  3. Reports test accuracy

The linear probe accuracy directly measures whether pretrained features are
class-discriminative for the private task — our hypothesis for why standard
divergence metrics fail (Finding 13 in FINDINGS.md).

Expected result: linear probe accuracy should rank experiments the same way
as DP finetuning accuracy, confirming that discriminative feature quality is
the true predictor.

Usage (GPU machine):
    source ~/venv/bin/activate
    cd /path/to/PrivInAI-Project-2026
    python experiments/run_linear_probing.py

Requirements: torch, torchvision (same env as main experiments)
Output: results/image/linear_probe_results.json
        results/image/linear_probe_results.csv
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dp_shift_bench.models.image_models import SmallCNN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINTS_ROOT = "./checkpoints/image"
DATA_ROOT = "./data"
OUTPUT_PATH = "./results/image/linear_probe_results.json"

EXPERIMENT_LABELS = {
    "IMG-01": "SVHN full",
    "IMG-02": "SVHN {0-4}→MNIST {0-4}",
    "IMG-03": "SVHN {5-9}→MNIST {0-4}",
    "IMG-04": "SVHN AE",
    "IMG-05": "SVHN 10%",
    "IMG-06": "SVHN 25%",
    "IMG-07": "SVHN 50%",
    "IMG-08": "CIFAR-10",
    "IMG-09": "SVHN+Aug",
    "IMG-10": "FashionMNIST",
}

# For experiments with 5 pretrain classes (IMG-02, IMG-03), model has 5 output classes
PRETRAIN_CLASSES = {
    "IMG-01": 10, "IMG-02": 5, "IMG-03": 5, "IMG-04": 10,
    "IMG-05": 10, "IMG-06": 10, "IMG-07": 10, "IMG-08": 10,
    "IMG-09": 10, "IMG-10": 10,
}

# For experiments with filtered MNIST (IMG-02, IMG-03 use digits 0-4 only)
FINETUNE_CLASSES = {
    "IMG-02": [0, 1, 2, 3, 4],
    "IMG-03": [0, 1, 2, 3, 4],
}

# Probe training config
PROBE_EPOCHS = 10
PROBE_LR = 0.01
BATCH_SIZE = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

mnist_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def get_mnist_loaders(classes=None):
    train_ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=mnist_transform)
    test_ds = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=mnist_transform)

    if classes is not None:
        # Filter to specified classes and remap labels 0..len(classes)-1
        class_set = set(classes)
        label_map = {c: i for i, c in enumerate(sorted(classes))}

        def _filter(ds):
            targets = np.array(ds.targets)
            indices = [i for i, t in enumerate(targets) if t in class_set]
            ds.data = ds.data[indices]
            ds.targets = [label_map[targets[i]] for i in indices]
            return ds

        train_ds = _filter(train_ds)
        test_ds = _filter(test_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader, len(set(train_ds.targets if classes else range(10)))


def extract_features(model, loader, device):
    """Extract 256-dim features from SmallCNN's penultimate layer."""
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats = model.get_features(x)
            all_feats.append(feats.cpu())
            all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)


def train_linear_probe(train_feats, train_labels, test_feats, test_labels, n_classes, device):
    """Train a single linear layer on top of frozen features."""
    feat_dim = train_feats.shape[1]
    probe = nn.Linear(feat_dim, n_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=PROBE_LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Move to device
    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    test_feats = test_feats.to(device)
    test_labels = test_labels.to(device)

    # Mini-batch training
    n = len(train_feats)
    idx = torch.randperm(n)
    batch_size = 512

    for epoch in range(PROBE_EPOCHS):
        probe.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = train_feats[perm[start:end]]
            yb = train_labels[perm[start:end]]
            optimizer.zero_grad()
            loss = criterion(probe(xb), yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        preds = probe(test_feats).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item()
    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_linear_probing():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {}

    for exp_id, label in EXPERIMENT_LABELS.items():
        ckpt_dir = os.path.join(CHECKPOINTS_ROOT, exp_id)

        if exp_id == "IMG-04":
            ckpt_path = os.path.join(ckpt_dir, "autoencoder.pt")
        else:
            ckpt_path = os.path.join(ckpt_dir, "pretrained.pt")

        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {exp_id}: checkpoint not found at {ckpt_path}")
            results[exp_id] = {"error": "checkpoint not found", "label": label}
            continue

        print(f"\n{exp_id} ({label})")

        # Load model
        n_pretrain_classes = PRETRAIN_CLASSES[exp_id]
        model = SmallCNN(num_classes=n_pretrain_classes).to(device)

        if exp_id == "IMG-04":
            # Load autoencoder encoder weights into SmallCNN.features
            ae_state = torch.load(ckpt_path, map_location=device)
            encoder_keys = {k: v for k, v in ae_state.items() if k.startswith("encoder.")}
            # Remap encoder.X → features.X
            remapped = {k.replace("encoder.", "features."): v for k, v in encoder_keys.items()}
            missing, unexpected = model.features.load_state_dict(
                {k.replace("features.", ""): v for k, v in remapped.items()},
                strict=False
            )
            print(f"  Autoencoder encoder loaded. Missing: {missing}, Unexpected: {unexpected}")
        else:
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)

        model.eval()

        # Get MNIST loaders (filtered for IMG-02/03)
        classes = FINETUNE_CLASSES.get(exp_id, None)
        train_loader, test_loader, n_finetune_classes = get_mnist_loaders(classes)

        print(f"  Extracting features (n_finetune_classes={n_finetune_classes})…")
        train_feats, train_labels = extract_features(model, train_loader, device)
        test_feats, test_labels = extract_features(model, test_loader, device)

        print(f"  Training linear probe ({PROBE_EPOCHS} epochs)…")
        acc = train_linear_probe(
            train_feats, train_labels,
            test_feats, test_labels,
            n_finetune_classes, device,
        )
        print(f"  Linear probe accuracy: {acc:.4f} ({acc*100:.2f}%)")

        results[exp_id] = {
            "label": label,
            "linear_probe_acc": acc,
            "n_finetune_classes": n_finetune_classes,
        }

    # Save JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}")

    # Print summary table
    print("\n=== LINEAR PROBE SUMMARY ===")
    print(f"{'Exp':8s}  {'Label':30s}  {'LP Acc':>8s}  {'DP Acc @ ε=0.5':>14s}")
    dp_accs = {
        "IMG-01": 0.9597, "IMG-02": 0.9846, "IMG-03": 0.9684, "IMG-04": 0.8771,
        "IMG-05": 0.9534, "IMG-06": 0.9529, "IMG-07": 0.9552, "IMG-08": 0.9419,
        "IMG-09": 0.9585, "IMG-10": 0.9355,
    }
    for exp_id in EXPERIMENT_LABELS:
        if exp_id in results and "linear_probe_acc" in results[exp_id]:
            lp = results[exp_id]["linear_probe_acc"]
            dp = dp_accs.get(exp_id, float("nan"))
            print(f"{exp_id:8s}  {EXPERIMENT_LABELS[exp_id]:30s}  {lp:8.4f}  {dp:14.4f}")

    # Also save CSV for easy import
    import csv
    csv_path = OUTPUT_PATH.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_id", "label", "linear_probe_acc", "dp_acc_eps05"])
        for exp_id in EXPERIMENT_LABELS:
            if exp_id in results and "linear_probe_acc" in results[exp_id]:
                writer.writerow([
                    exp_id,
                    results[exp_id]["label"],
                    results[exp_id]["linear_probe_acc"],
                    dp_accs.get(exp_id, ""),
                ])
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    run_linear_probing()
