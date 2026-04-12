"""
dp_shift_bench/runners/run_divergences.py

Compute divergence metrics for all pretrain/finetune pairs.
Run AFTER pretraining checkpoints are available.
"""

import argparse
import os
from typing import List, Optional

import torch

from dp_shift_bench.datasets.image_shifts import get_image_experiment_data
from dp_shift_bench.metrics.divergences import compute_all_divergences
from dp_shift_bench.models.image_models import SmallCNN
from dp_shift_bench.training.utils import get_device, save_results

# Experiments that have a pretrain dataset (IMG-00 has none)
EXPERIMENTS_WITH_PRETRAIN = [
    "IMG-01", "IMG-02", "IMG-03", "IMG-04",
    "IMG-05", "IMG-06", "IMG-07", "IMG-08",
    "IMG-09", "IMG-10",
]

PRIORITY_FILTER = {
    "P0": ["IMG-01", "IMG-02", "IMG-03"],
    "P1": ["IMG-01", "IMG-02", "IMG-03", "IMG-04",
           "IMG-05", "IMG-06", "IMG-07", "IMG-08"],
    "all": EXPERIMENTS_WITH_PRETRAIN,
}


def compute_image_divergences(
    experiments: Optional[List[str]] = None,
    priority: str = "P0",
    data_root: str = "./data",
    checkpoints_root: str = "./checkpoints/image",
    results_root: str = "./results/image",
    device: Optional[str] = None,
    debug: bool = False,
) -> dict:
    """
    For each experiment (except IMG-00):
    1. Load pretrain + finetune datasets.
    2. Load the pretrained SmallCNN as feature extractor.
       Falls back to IMG-01's checkpoint as a shared extractor if unavailable.
    3. Compute all divergence metrics.
    4. Save to results_root/divergences/{experiment_id}.json

    Returns mapping experiment_id → divergence dict.
    """
    if device is None:
        device = get_device()

    if experiments is None:
        experiments = PRIORITY_FILTER.get(priority, EXPERIMENTS_WITH_PRETRAIN)

    all_divergences: dict = {}

    for exp_id in experiments:
        print(f"\n[Divergences] {exp_id}")

        data = get_image_experiment_data(
            exp_id, data_root=data_root, debug=debug, batch_size=256
        )

        if data["pretrain_loader"] is None:
            print(f"  Skipping {exp_id}: no pretrain data.")
            continue

        # ---- load feature extractor ----
        # For IMG-04 (autoencoder), use IMG-01's supervised checkpoint as extractor
        if exp_id == "IMG-04":
            ckpt_path = os.path.join(checkpoints_root, "IMG-01", "pretrained.pt")
        else:
            ckpt_path = os.path.join(checkpoints_root, exp_id, "pretrained.pt")

        feature_model = SmallCNN(num_classes=data["num_pretrain_classes"])
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            feature_model.load_state_dict(state, strict=False)
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: checkpoint not found ({ckpt_path}); using random features.")

        feature_model = feature_model.to(device)
        feature_model.eval()

        # ---- compute divergences ----
        div_metrics = compute_all_divergences(
            pretrain_loader=data["pretrain_loader"],
            finetune_loader=data["finetune_train_loader"],
            feature_model=feature_model,
            device=device,
            priority=priority,
        )
        div_metrics["experiment_id"] = exp_id
        all_divergences[exp_id] = div_metrics

        # ---- save ----
        div_dir = os.path.join(results_root, "divergences")
        os.makedirs(div_dir, exist_ok=True)
        out_path = os.path.join(div_dir, f"{exp_id}.json")
        save_results(div_metrics, out_path)
        print(f"  → Saved: {out_path}")
        for k, v in div_metrics.items():
            if k != "experiment_id":
                print(f"     {k}: {v:.6f}")

    return all_divergences


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute divergence metrics for image track")
    parser.add_argument("--priority", type=str, default="P0", choices=["P0", "P1", "all"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--checkpoints-root", type=str, default="./checkpoints/image")
    parser.add_argument("--results-root", type=str, default="./results/image")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--experiments", nargs="+", default=None)
    args = parser.parse_args()

    compute_image_divergences(
        experiments=args.experiments,
        priority=args.priority,
        data_root=args.data_root,
        checkpoints_root=args.checkpoints_root,
        results_root=args.results_root,
        debug=args.debug,
    )
