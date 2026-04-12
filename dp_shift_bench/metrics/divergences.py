"""
dp_shift_bench/metrics/divergences.py

Distribution divergence metrics between datasets.
For images: operates on CNN features extracted from the datasets.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    max_samples: int = 10000,
) -> np.ndarray:
    """
    Extract features from model's penultimate layer for all samples.
    Uses model.get_features(x) if available, otherwise model(x).

    Returns: ndarray of shape (N, feature_dim)
    """
    model.eval()
    feats_list = []
    total = 0

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            if hasattr(model, "get_features"):
                feats = model.get_features(imgs)
            else:
                feats = model(imgs)
            feats_list.append(feats.cpu().numpy())
            total += len(feats)
            if total >= max_samples:
                break

    return np.concatenate(feats_list, axis=0)[:max_samples]


def _get_inception_model(device: str):
    """Load pretrained InceptionV3 with fc replaced by Identity."""
    from torchvision.models import inception_v3

    model = inception_v3(weights="DEFAULT", transform_input=False)
    model.aux_logits = False
    model.AuxLogits = None
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


def _extract_inception_features(
    dataloader: DataLoader, device: str, max_samples: int = 10000
) -> np.ndarray:
    """Extract 2048-dim InceptionV3 features, resizing images to 299×299."""
    model = _get_inception_model(device)
    feats_list = []
    total = 0

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = F.interpolate(
                imgs, size=(299, 299), mode="bilinear", align_corners=False
            )
            imgs = imgs.to(device)
            feats = model(imgs)
            feats_list.append(feats.cpu().numpy())
            total += len(feats)
            if total >= max_samples:
                break

    return np.concatenate(feats_list, axis=0)[:max_samples]


# ---------------------------------------------------------------------------
# 1. FID (Fréchet Inception Distance)  — P0
# ---------------------------------------------------------------------------

def compute_fid(
    dataset1: DataLoader,
    dataset2: DataLoader,
    device: str = "cuda",
) -> float:
    """
    FID between two image datasets using InceptionV3 features (Option B).

    FID = ||μ₁−μ₂||² + Tr(Σ₁ + Σ₂ − 2·√(Σ₁Σ₂))
    """
    feats1 = _extract_inception_features(dataset1, device)
    feats2 = _extract_inception_features(dataset2, device)

    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

    diff = mu1 - mu2

    # Numerically stable matrix square root
    eps = 1e-6
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        if np.max(np.abs(covmean.imag)) > 1e-2:
            # Fallback: add small diagonal offset
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return max(0.0, fid)


# ---------------------------------------------------------------------------
# 2. MMD (Maximum Mean Discrepancy)  — P1
# ---------------------------------------------------------------------------

def compute_mmd(
    features1: np.ndarray,
    features2: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> float:
    """
    MMD² between two feature sets using RBF kernel + median heuristic.

    Subsamples to max 5 000 points from each set.
    """
    MAX = 5000
    rng = np.random.default_rng(0)

    if len(features1) > MAX:
        features1 = features1[rng.choice(len(features1), MAX, replace=False)]
    if len(features2) > MAX:
        features2 = features2[rng.choice(len(features2), MAX, replace=False)]

    x = torch.tensor(features1, dtype=torch.float32)
    y = torch.tensor(features2, dtype=torch.float32)

    if gamma is None:
        combined = torch.cat([x, y], dim=0)
        dists = torch.cdist(combined, combined)
        median_dist = torch.median(dists[dists > 0]).item()
        gamma = 1.0 / (2 * median_dist ** 2 + 1e-10)

    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma * torch.cdist(a, b) ** 2)

    kxx = rbf(x, x).mean()
    kyy = rbf(y, y).mean()
    kxy = rbf(x, y).mean()

    mmd2 = float(kxx + kyy - 2 * kxy)
    return max(0.0, mmd2)


# ---------------------------------------------------------------------------
# 3. Proxy A-distance  — P1
# ---------------------------------------------------------------------------

def compute_proxy_a_distance(
    features1: np.ndarray,
    features2: np.ndarray,
    max_samples: int = 5000,
) -> float:
    """
    Train a linear domain classifier and compute:
        PAD = 2·(1 − 2·error)

    PAD ∈ [0, 2]: 0 = identical, 2 = perfectly separable.
    """
    rng = np.random.default_rng(42)

    if len(features1) > max_samples:
        features1 = features1[rng.choice(len(features1), max_samples, replace=False)]
    if len(features2) > max_samples:
        features2 = features2[rng.choice(len(features2), max_samples, replace=False)]

    X = np.concatenate([features1, features2], axis=0)
    y = np.array([0] * len(features1) + [1] * len(features2))

    shuffle_idx = rng.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]

    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs")
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    error = 1.0 - scores.mean()
    pad = 2.0 * (1.0 - 2.0 * error)
    return float(np.clip(pad, 0.0, 2.0))


# ---------------------------------------------------------------------------
# 4. KL Divergence (k-NN estimator, Wang et al. 2009)  — P0
# ---------------------------------------------------------------------------

def compute_kl_divergence_estimated(
    features1: np.ndarray,
    features2: np.ndarray,
    k: int = 5,
) -> float:
    """
    Estimate KL(P ‖ Q) from samples X ~ P and Y ~ Q using the k-NN estimator.

    KL ≈ (d/n)·Σ log(ν_k(xᵢ)/ρ_k(xᵢ)) + log(m/(n−1))
    where ρ_k = k-th NN distance in X (excl. self), ν_k = k-th NN dist in Y.
    """
    MAX = 5000
    rng = np.random.default_rng(0)

    if len(features1) > MAX:
        features1 = features1[rng.choice(len(features1), MAX, replace=False)]
    if len(features2) > MAX:
        features2 = features2[rng.choice(len(features2), MAX, replace=False)]

    n, d = features1.shape
    m = features2.shape[0]

    if n < k + 2 or m < k + 1:
        return 0.0

    nn_x = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(features1)
    dists_x, _ = nn_x.kneighbors(features1)
    rho = np.maximum(dists_x[:, k], 1e-10)   # skip self (index 0)

    nn_y = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(features2)
    dists_y, _ = nn_y.kneighbors(features1)
    nu = np.maximum(dists_y[:, k - 1], 1e-10)

    kl = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / max(n - 1, 1))
    return float(max(0.0, kl))


# ---------------------------------------------------------------------------
# 5. Jensen–Shannon Divergence  — P0
# ---------------------------------------------------------------------------

def compute_jsd(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    JSD = 0.5·KL(P‖M) + 0.5·KL(Q‖M),  M = 0.5·(P+Q).

    Uses the k-NN KL estimator on a mixture.  Result ∈ [0, log 2].
    """
    MAX = 2500
    rng = np.random.default_rng(0)

    if len(features1) > MAX:
        features1 = features1[rng.choice(len(features1), MAX, replace=False)]
    if len(features2) > MAX:
        features2 = features2[rng.choice(len(features2), MAX, replace=False)]

    mixture = np.concatenate([features1, features2], axis=0)

    kl_pm = compute_kl_divergence_estimated(features1, mixture)
    kl_qm = compute_kl_divergence_estimated(features2, mixture)

    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    return float(np.clip(jsd, 0.0, np.log(2)))


# ---------------------------------------------------------------------------
# 6. Total Variation  — P0
# ---------------------------------------------------------------------------

def compute_total_variation(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    TV distance: per-dimension histogram, averaged across dimensions.

    TV = 0.5 · Σ|P(x) − Q(x)|  per bin,  then mean over dims.
    """
    N_BINS = 50
    MAX = 5000
    rng = np.random.default_rng(0)

    if len(features1) > MAX:
        features1 = features1[rng.choice(len(features1), MAX, replace=False)]
    if len(features2) > MAX:
        features2 = features2[rng.choice(len(features2), MAX, replace=False)]

    d = features1.shape[1]
    tv_per_dim = []

    for dim in range(d):
        x1 = features1[:, dim]
        x2 = features2[:, dim]

        lo = min(x1.min(), x2.min())
        hi = max(x1.max(), x2.max())
        if hi <= lo:
            tv_per_dim.append(0.0)
            continue

        bins = np.linspace(lo, hi, N_BINS + 1)
        h1, _ = np.histogram(x1, bins=bins)
        h2, _ = np.histogram(x2, bins=bins)

        h1 = h1 / (h1.sum() + 1e-10)
        h2 = h2 / (h2.sum() + 1e-10)

        tv_per_dim.append(0.5 * float(np.sum(np.abs(h1 - h2))))

    return float(np.mean(tv_per_dim))


# ---------------------------------------------------------------------------
# 7. Wasserstein-1 (POT library)  — P1
# ---------------------------------------------------------------------------

def compute_wasserstein(
    features1: np.ndarray,
    features2: np.ndarray,
    max_samples: int = 2000,
) -> float:
    """
    Wasserstein-1 distance via POT earth-mover's distance (O(n³) — subsample aggressively).
    """
    import ot  # Python Optimal Transport

    rng = np.random.default_rng(0)

    if len(features1) > max_samples:
        features1 = features1[rng.choice(len(features1), max_samples, replace=False)]
    if len(features2) > max_samples:
        features2 = features2[rng.choice(len(features2), max_samples, replace=False)]

    n1, n2 = len(features1), len(features2)
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2

    M = ot.dist(
        features1.astype(np.float64), features2.astype(np.float64), metric="euclidean"
    )
    W = ot.emd2(a, b, M)
    return float(W)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_all_divergences(
    pretrain_loader: DataLoader,
    finetune_loader: DataLoader,
    feature_model: nn.Module,
    device: str = "cuda",
    priority: str = "P0",
) -> dict:
    """
    Compute all divergence metrics between pretrain and finetune datasets.

    priority: "P0" → FID, KL, JSD, TV
              "P1" → P0 + MMD, PAD, Wasserstein
              "all" → same as P1

    Returns:
        {
            "fid": float,
            "kl_estimated": float,
            "jsd": float,
            "total_variation": float,
            "mmd_rbf": float,           # P1
            "proxy_a_distance": float,  # P1
            "wasserstein": float,       # P1
        }
    """
    print("  Extracting pretrain features…")
    feats_pre = extract_features(feature_model, pretrain_loader, device)
    print("  Extracting finetune features…")
    feats_ft = extract_features(feature_model, finetune_loader, device)

    results: dict = {}

    # ---- P0 metrics ----
    print("  Computing FID…")
    results["fid"] = compute_fid(pretrain_loader, finetune_loader, device)

    print("  Computing KL (estimated)…")
    results["kl_estimated"] = compute_kl_divergence_estimated(feats_pre, feats_ft)

    print("  Computing JSD…")
    results["jsd"] = compute_jsd(feats_pre, feats_ft)

    print("  Computing Total Variation…")
    results["total_variation"] = compute_total_variation(feats_pre, feats_ft)

    # ---- P1 metrics ----
    if priority in ("P1", "all"):
        print("  Computing MMD…")
        results["mmd_rbf"] = compute_mmd(feats_pre, feats_ft)

        print("  Computing Proxy A-Distance…")
        results["proxy_a_distance"] = compute_proxy_a_distance(feats_pre, feats_ft)

        print("  Computing Wasserstein…")
        results["wasserstein"] = compute_wasserstein(feats_pre, feats_ft)

    return results
