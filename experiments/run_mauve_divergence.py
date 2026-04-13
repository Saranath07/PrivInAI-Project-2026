"""
experiments/run_mauve_divergence.py

Compute MAUVE scores between pretrain and finetune feature distributions.

MAUVE (Pillutla et al., JMLR 2023) measures the area under the divergence
frontier between two distributions, using quantization of feature vectors.
It was explicitly suggested in the course project (Section 1.6, ref [11]).

For images, we compute MAUVE on the 256-dim CNN features extracted from
each experiment's pretrained SmallCNN feature extractor.

Usage (GPU machine):
    pip install mauve-text  # or: pip install mauve
    source ~/venv/bin/activate
    cd /path/to/PrivInAI-Project-2026
    python experiments/run_mauve_divergence.py

Output:
    results/image/divergences/mauve_scores.json
    results/image/divergences/mauve_scores.csv

Note: MAUVE was designed for text (using GPT-2 features), but the authors
show it works for any feature space. We use it on CNN features directly.
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dp_shift_bench.models.image_models import SmallCNN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINTS_ROOT = "./checkpoints/image"
DATA_ROOT = "./data"
OUTPUT_JSON = "./results/image/divergences/mauve_scores.json"
OUTPUT_CSV = "./results/image/divergences/mauve_scores.csv"

MAX_SAMPLES = 5000  # MAUVE is O(N log N), 5k per side is fast enough

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

PRETRAIN_CLASSES = {
    "IMG-01": 10, "IMG-02": 5, "IMG-03": 5, "IMG-04": 10,
    "IMG-05": 10, "IMG-06": 10, "IMG-07": 10, "IMG-08": 10,
    "IMG-09": 10, "IMG-10": 10,
}

# SVHN transforms
svhn_transform = transforms.Compose([transforms.ToTensor()])

# MNIST transforms (resize to 32×32, 3-channel)
mnist_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# CIFAR transforms
cifar_transform = transforms.Compose([transforms.ToTensor()])

# FashionMNIST transforms
fashion_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pretrain_loader(exp_id):
    """Return DataLoader for the pretrain dataset of this experiment."""
    batch_size = 256
    num_workers = 2

    if exp_id in ("IMG-01", "IMG-04", "IMG-05", "IMG-06", "IMG-07", "IMG-09"):
        ds = datasets.SVHN(DATA_ROOT, split="train", download=True, transform=svhn_transform)
        if exp_id == "IMG-05":
            n = int(0.10 * len(ds))
            idx = np.random.RandomState(42).choice(len(ds), n, replace=False)
            from torch.utils.data import Subset
            ds = Subset(ds, idx)
        elif exp_id == "IMG-06":
            n = int(0.25 * len(ds))
            idx = np.random.RandomState(42).choice(len(ds), n, replace=False)
            from torch.utils.data import Subset
            ds = Subset(ds, idx)
        elif exp_id == "IMG-07":
            n = int(0.50 * len(ds))
            idx = np.random.RandomState(42).choice(len(ds), n, replace=False)
            from torch.utils.data import Subset
            ds = Subset(ds, idx)

    elif exp_id == "IMG-02":
        ds = datasets.SVHN(DATA_ROOT, split="train", download=True, transform=svhn_transform)
        labels = np.array(ds.labels)
        idx = [i for i, l in enumerate(labels) if l in {0, 1, 2, 3, 4}]
        from torch.utils.data import Subset
        ds = Subset(ds, idx)

    elif exp_id == "IMG-03":
        ds = datasets.SVHN(DATA_ROOT, split="train", download=True, transform=svhn_transform)
        labels = np.array(ds.labels)
        idx = [i for i, l in enumerate(labels) if l in {5, 6, 7, 8, 9}]
        from torch.utils.data import Subset
        ds = Subset(ds, idx)

    elif exp_id == "IMG-08":
        ds = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=cifar_transform)

    elif exp_id == "IMG-10":
        ds = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=fashion_transform)

    else:
        raise ValueError(f"Unknown exp_id: {exp_id}")

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_finetune_loader(exp_id):
    """Return DataLoader for MNIST (finetune) dataset, filtered if needed."""
    if exp_id in ("IMG-02", "IMG-03"):
        classes = {0, 1, 2, 3, 4}
        ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=mnist_transform)
        targets = np.array(ds.targets)
        idx = [i for i, t in enumerate(targets) if t in classes]
        ds.data = ds.data[idx]
        ds.targets = [ds.targets[i] for i in idx]
    else:
        ds = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=mnist_transform)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)


def extract_features(model, loader, device, max_samples=MAX_SAMPLES):
    model.eval()
    feats = []
    n = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            f = model.get_features(x).cpu().numpy()
            feats.append(f)
            n += len(f)
            if n >= max_samples:
                break
    feats = np.concatenate(feats, axis=0)[:max_samples]
    return feats


# ---------------------------------------------------------------------------
# Direct MAUVE implementation (bypasses mauve library's broken PCA on NumPy 2.x)
# ---------------------------------------------------------------------------

def _compute_mauve_direct(
    p_feats: np.ndarray,
    q_feats: np.ndarray,
    num_buckets: int = 100,
    discretization_size: int = 25,
    mauve_scaling_factor: float = 5.0,
    seed: int = 42,
) -> float:
    """
    Compute MAUVE score from pre-processed feature vectors.

    Follows Pillutla et al. (JMLR 2023):
      1. K-means quantize combined features → discrete distributions P, Q
      2. For λ in [0,1], form mixture M_λ = λP + (1-λ)Q
      3. Plot divergence frontier: (KL(P||M_λ), KL(Q||M_λ))
      4. Area under the frontier curve (clipped to [0, c]²) = MAUVE

    Returns value in (0, 1] — higher = more similar distributions.
    """
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.RandomState(seed)

    # K-means on combined features
    combined = np.concatenate([p_feats, q_feats], axis=0).astype(np.float64)
    km = MiniBatchKMeans(
        n_clusters=num_buckets,
        random_state=seed,
        n_init=5,
        max_iter=300,
        batch_size=min(1000, len(combined)),
    )
    km.fit(combined)

    # Get cluster assignments and build histograms
    p_labels = km.predict(p_feats.astype(np.float64))
    q_labels = km.predict(q_feats.astype(np.float64))

    eps = 1e-8  # smoothing to avoid log(0)
    p_hist = np.bincount(p_labels, minlength=num_buckets).astype(np.float64) + eps
    q_hist = np.bincount(q_labels, minlength=num_buckets).astype(np.float64) + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    # Divergence frontier: sweep λ
    lambdas = np.linspace(1e-6, 1 - 1e-6, discretization_size)
    kl_p_list, kl_q_list = [], []

    for lam in lambdas:
        m = lam * p_hist + (1 - lam) * q_hist
        m = np.maximum(m, eps)
        kl_p = np.sum(p_hist * np.log(p_hist / m))   # KL(P || M)
        kl_q = np.sum(q_hist * np.log(q_hist / m))   # KL(Q || M)
        kl_p_list.append(kl_p)
        kl_q_list.append(kl_q)

    kl_p_arr = np.array(kl_p_list)
    kl_q_arr = np.array(kl_q_list)

    # Clip to [0, c] × [0, c] where c = mauve_scaling_factor * log(num_buckets)
    c = mauve_scaling_factor * np.log(num_buckets)
    kl_p_clipped = np.minimum(kl_p_arr, c)
    kl_q_clipped = np.minimum(kl_q_arr, c)

    # Area under the frontier curve (trapezoidal) normalised to [0,1]
    # Sort by kl_q to get a proper curve
    order = np.argsort(kl_q_clipped)
    x = kl_q_clipped[order]
    y = kl_p_clipped[order]
    area = np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x)

    # Normalise: divide by c² (maximum possible area = square [0,c]²)
    mauve_score = float(np.exp(-area / (c * c)))
    return max(0.0, min(1.0, mauve_score))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mauve():
    try:
        import mauve
    except ImportError:
        print("ERROR: mauve not installed. Run: pip install mauve-text")
        print("(Note: the package name is 'mauve-text' on PyPI but imports as 'mauve')")
        sys.exit(1)

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
            ae_state = torch.load(ckpt_path, map_location=device)
            encoder_state = {
                k.replace("encoder.", ""): v
                for k, v in ae_state.items() if k.startswith("encoder.")
            }
            model.features.load_state_dict(encoder_state, strict=False)
        else:
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        model.eval()

        # Extract features
        pretrain_loader = get_pretrain_loader(exp_id)
        finetune_loader = get_finetune_loader(exp_id)

        print(f"  Extracting pretrain features…")
        feats_p = extract_features(model, pretrain_loader, device)
        print(f"  Extracting finetune features…")
        feats_f = extract_features(model, finetune_loader, device)

        print(f"  pretrain shape: {feats_p.shape}, finetune shape: {feats_f.shape}")

        # Pre-process features before passing to MAUVE.
        # Root cause of overflow: dead ReLU neurons → zero-variance dimensions →
        # StandardScaler divides by zero → NaN → PCA overflow.
        # Fix: drop zero-variance dims, center-only scale, PCA to 32 dims, L2-norm.
        from sklearn.decomposition import PCA

        feats_p = feats_p.astype(np.float64)
        feats_f = feats_f.astype(np.float64)

        combined = np.concatenate([feats_p, feats_f], axis=0)

        # Drop dimensions with near-zero variance (dead ReLU neurons)
        var = combined.var(axis=0)
        active_dims = var > 1e-8
        combined = combined[:, active_dims]
        print(f"  Active dims: {active_dims.sum()} / {len(active_dims)}")

        # Center only (no scaling — avoids div-by-zero on low-variance dims)
        mean = combined.mean(axis=0)
        combined = combined - mean

        # PCA to 32 dims
        n_components = min(32, combined.shape[1], combined.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        combined_pca = pca.fit_transform(combined)

        feats_p = combined_pca[:len(feats_p)]
        feats_f = combined_pca[len(feats_p):]

        # L2-normalize
        feats_p = (feats_p / (np.linalg.norm(feats_p, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        feats_f = (feats_f / (np.linalg.norm(feats_f, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

        # Sanity check
        if not (np.isfinite(feats_p).all() and np.isfinite(feats_f).all()):
            raise ValueError("NaN/Inf in features after preprocessing")

        print(f"  Reduced to {feats_p.shape[1]}-dim PCA features")

        # num_buckets: k-means needs >= 39*k combined points → use 100
        n_combined = len(feats_p) + len(feats_f)
        num_buckets = min(100, n_combined // 40)

        # Compute MAUVE via direct implementation (bypasses mauve library's
        # broken PCA, which fails under NumPy 2.x).
        # Algorithm: k-means quantize → divergence frontier → area under curve.
        print(f"  Computing MAUVE (num_buckets={num_buckets})…")
        try:
            mauve_score = _compute_mauve_direct(feats_p, feats_f, num_buckets, seed=42)
            print(f"  MAUVE = {mauve_score:.4f}")
            results[exp_id] = {
                "label": label,
                "mauve": mauve_score,
            }
        except Exception as e:
            print(f"  MAUVE computation failed: {e}")
            import traceback; traceback.print_exc()
            results[exp_id] = {"error": str(e), "label": label}

    # Save JSON
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUTPUT_JSON}")

    # Print summary
    print("\n=== MAUVE SUMMARY ===")
    dp_accs = {
        "IMG-01": 0.9597, "IMG-02": 0.9846, "IMG-03": 0.9684, "IMG-04": 0.8771,
        "IMG-05": 0.9534, "IMG-06": 0.9529, "IMG-07": 0.9552, "IMG-08": 0.9419,
        "IMG-09": 0.9585, "IMG-10": 0.9355,
    }
    print(f"{'Exp':8s}  {'Label':30s}  {'MAUVE':>8s}  {'DP Acc @ ε=0.5':>14s}")
    for exp_id in EXPERIMENT_LABELS:
        if exp_id in results and "mauve" in results[exp_id]:
            m = results[exp_id]["mauve"]
            dp = dp_accs.get(exp_id, float("nan"))
            print(f"{exp_id:8s}  {EXPERIMENT_LABELS[exp_id]:30s}  {m:8.4f}  {dp:14.4f}")

    # Save CSV
    import csv
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment_id", "label", "mauve", "dp_acc_eps05"])
        for exp_id in EXPERIMENT_LABELS:
            if exp_id in results and "mauve" in results[exp_id]:
                writer.writerow([
                    exp_id,
                    results[exp_id]["label"],
                    results[exp_id]["mauve"],
                    dp_accs.get(exp_id, ""),
                ])
    print(f"Saved → {OUTPUT_CSV}")

    # Quick Spearman vs DP accuracy
    from scipy import stats
    paired = [
        (results[e]["mauve"], dp_accs[e])
        for e in EXPERIMENT_LABELS
        if e in results and "mauve" in results[e] and e in dp_accs
    ]
    if len(paired) >= 3:
        mauve_vals, dp_vals = zip(*paired)
        rho, p = stats.spearmanr(mauve_vals, dp_vals)
        print(f"\nSpearman ρ (MAUVE vs DP acc at ε=0.5): {rho:+.3f}  (p={p:.4f})")
        print("(Compare: Wasserstein ρ = -0.729, all others ≈ 0)")


if __name__ == "__main__":
    run_mauve()
