"""
dp_shift_bench/runners/run_single.py

Single-experiment runner: pretrain (optional) → DP-finetune → evaluate → save.
"""

import argparse
import os
from typing import Optional

import torch

from dp_shift_bench.datasets.image_shifts import get_image_experiment_data
from dp_shift_bench.metrics.evaluation import evaluate_accuracy
from dp_shift_bench.models.image_models import ConvAutoencoder, SmallCNN, create_model
from dp_shift_bench.training.dp_trainer import finetune_dp
from dp_shift_bench.training.standard_trainer import (
    pretrain_autoencoder,
    pretrain_supervised,
)
from dp_shift_bench.training.utils import get_device, save_results, set_seed


def run_single_experiment(
    experiment_id: str,
    epsilon: float,
    seed: int = 42,
    arch: str = "smallcnn",
    pretrain_epochs: int = 20,
    finetune_epochs: int = 15,
    max_grad_norm: float = 1.0,
    batch_size: int = 256,
    data_root: str = "./data",
    results_root: str = "./results/image",
    checkpoints_root: str = "./checkpoints/image",
    debug: bool = False,
    device: Optional[str] = None,
) -> dict:
    """
    Full pipeline for one (experiment_id, epsilon, seed) combination.

    Steps:
        1. Set seed
        2. Load data
        3. Pretrain if needed (checkpoint caching)
        4. Create finetuning model with pretrained weights
        5. DP-SGD finetuning
        6. Evaluate
        7. Save result JSON

    Returns the results dict.
    """
    set_seed(seed)
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"  {experiment_id}  |  ε={epsilon}  |  seed={seed}  |  device={device}")
    print(f"{'='*60}")

    # ---- 1. Load data ----
    data = get_image_experiment_data(
        experiment_id=experiment_id,
        data_root=data_root,
        batch_size=batch_size,
        seed=seed,
        debug=debug,
    )

    # ---- 2. Pretraining (if needed) ----
    pretrain_path = None
    if data["pretrain_loader"] is not None:
        pretrain_dir = os.path.join(checkpoints_root, experiment_id)
        os.makedirs(pretrain_dir, exist_ok=True)

        if experiment_id == "IMG-04":
            pretrain_path = os.path.join(pretrain_dir, "autoencoder.pt")
            if not os.path.exists(pretrain_path):
                print("  Autoencoder pretraining…")
                ae_model = ConvAutoencoder().to(device)
                ae_epochs = 30 if not debug else 2
                pretrain_autoencoder(
                    ae_model,
                    data["pretrain_loader"],
                    epochs=ae_epochs,
                    device=device,
                    save_path=pretrain_path,
                )
            else:
                print(f"  Autoencoder checkpoint found: {pretrain_path}")
        else:
            pretrain_path = os.path.join(pretrain_dir, "pretrained.pt")
            if not os.path.exists(pretrain_path):
                print("  Supervised pretraining…")
                pretrain_model = SmallCNN(
                    num_classes=data["num_pretrain_classes"]
                ).to(device)
                pt_epochs = pretrain_epochs if not debug else 2
                pretrain_supervised(
                    pretrain_model,
                    data["pretrain_loader"],
                    epochs=pt_epochs,
                    device=device,
                    save_path=pretrain_path,
                )
            else:
                print(f"  Pretrain checkpoint found: {pretrain_path}")

    # ---- 3. Create finetuning model ----
    if experiment_id == "IMG-04":
        ft_model = create_model(
            arch=arch,
            num_classes=data["num_finetune_classes"],
            autoencoder_pretrained_path=pretrain_path,
        )
    elif pretrain_path is not None:
        ft_model = create_model(
            arch=arch,
            num_classes=data["num_finetune_classes"],
            pretrained_path=pretrain_path,
            pretrain_num_classes=data["num_pretrain_classes"],
        )
    else:
        ft_model = create_model(
            arch=arch,
            num_classes=data["num_finetune_classes"],
        )
    ft_model = ft_model.to(device)

    # ---- 4. DP-SGD finetuning ----
    ft_epochs = finetune_epochs if not debug else 2
    ft_results = finetune_dp(
        model=ft_model,
        train_loader=data["finetune_train_loader"],
        test_loader=data["finetune_test_loader"],
        epsilon=epsilon,
        epochs=ft_epochs,
        max_grad_norm=max_grad_norm,
        device=device,
        seed=seed,
    )

    # ---- 5. Save results ----
    eps_str = "eps_inf" if epsilon == float("inf") else f"eps_{epsilon}"
    result = {
        "experiment_id": experiment_id,
        "epsilon": epsilon if epsilon != float("inf") else "inf",
        "seed": seed,
        "arch": arch,
        "debug": debug,
        "final_test_acc": ft_results["final_test_acc"],
        "best_test_acc": ft_results["best_test_acc"],
        "epsilon_actual": ft_results["epsilon_actual"],
        "train_losses": ft_results["train_losses"],
        "test_accs": ft_results["test_accs"],
        "epochs_run": ft_results["epochs_run"],
    }

    results_dir = os.path.join(results_root, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"{eps_str}_seed_{seed}.json")
    save_results(result, result_path)
    print(f"  → Saved: {result_path}")
    print(f"  → Final test acc: {result['final_test_acc']:.4f}  |  ε_actual: {result['epsilon_actual']}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single DP-ShiftBench image experiment")
    parser.add_argument("--experiment", type=str, required=True, help="e.g. IMG-01")
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        help="Target epsilon. Pass -1 for non-private (inf).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", type=str, default="smallcnn", choices=["smallcnn", "resnet18"])
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--results-root", type=str, default="./results/image")
    parser.add_argument("--checkpoints-root", type=str, default="./checkpoints/image")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    eps = float("inf") if args.epsilon < 0 else args.epsilon

    run_single_experiment(
        experiment_id=args.experiment,
        epsilon=eps,
        seed=args.seed,
        arch=args.arch,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        data_root=args.data_root,
        results_root=args.results_root,
        checkpoints_root=args.checkpoints_root,
        debug=args.debug,
    )
