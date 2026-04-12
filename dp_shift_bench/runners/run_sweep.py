"""
dp_shift_bench/runners/run_sweep.py

Full image experiment grid sweep: experiments × epsilons × seeds.
Skips already-completed runs (resume-friendly).
"""

import argparse
import os
from typing import List, Optional

from dp_shift_bench.runners.run_single import run_single_experiment
from dp_shift_bench.training.utils import get_device

# ---------------------------------------------------------------------------
# Experiment catalogue
# ---------------------------------------------------------------------------

EXPERIMENT_PRIORITIES = {
    "IMG-00": "P0",
    "IMG-01": "P0",
    "IMG-02": "P0",
    "IMG-03": "P0",
    "IMG-04": "P1",
    "IMG-05": "P1",
    "IMG-06": "P1",
    "IMG-07": "P1",
    "IMG-08": "P1",
    "IMG-09": "P2",
    "IMG-10": "P2",
}

PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}
ALL_EXPERIMENTS = list(EXPERIMENT_PRIORITIES.keys())

DEFAULT_EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]
DEFAULT_SEEDS = [42, 123, 456]


def _filter_by_priority(experiments: list, priority: str) -> list:
    if priority == "all":
        return experiments
    max_p = PRIORITY_ORDER[priority]
    return [e for e in experiments if PRIORITY_ORDER[EXPERIMENT_PRIORITIES[e]] <= max_p]


def _result_exists(exp_id: str, eps: float, seed: int, results_root: str) -> bool:
    eps_str = "eps_inf" if eps == float("inf") else f"eps_{eps}"
    path = os.path.join(results_root, exp_id, f"{eps_str}_seed_{seed}.json")
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_image_sweep(
    experiments: Optional[List[str]] = None,
    epsilons: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    priority: str = "P0",
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
) -> None:
    """
    Outer loop over all (experiment, epsilon, seed) combinations.

    Skips runs that already have a saved result file.
    """
    if experiments is None:
        experiments = _filter_by_priority(ALL_EXPERIMENTS, priority)
    if epsilons is None:
        epsilons = DEFAULT_EPSILONS
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if device is None:
        device = get_device()

    total = len(experiments) * len(epsilons) * len(seeds)
    done = 0

    print(f"\nDP-ShiftBench Image Sweep")
    print(f"  Experiments : {experiments}")
    print(f"  Epsilons    : {epsilons}")
    print(f"  Seeds       : {seeds}")
    print(f"  Total runs  : {total}")
    print(f"  Device      : {device}")
    print(f"  Debug       : {debug}\n")

    for exp_id in experiments:
        for eps in epsilons:
            for seed in seeds:
                done += 1
                tag = f"{exp_id} | ε={eps} | seed={seed} | [{done}/{total}]"

                if _result_exists(exp_id, eps, seed, results_root):
                    print(f"  [SKIP] {tag} (result exists)")
                    continue

                print(f"\n  [RUN ] {tag}")
                try:
                    run_single_experiment(
                        experiment_id=exp_id,
                        epsilon=eps,
                        seed=seed,
                        arch=arch,
                        pretrain_epochs=pretrain_epochs,
                        finetune_epochs=finetune_epochs,
                        max_grad_norm=max_grad_norm,
                        batch_size=batch_size,
                        data_root=data_root,
                        results_root=results_root,
                        checkpoints_root=checkpoints_root,
                        debug=debug,
                        device=device,
                    )
                except Exception as exc:
                    print(f"  [ERROR] {tag}: {exc}")

    print(f"\nSweep complete — {done} runs attempted.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DP-ShiftBench image sweep")
    parser.add_argument(
        "--priority",
        type=str,
        default="P0",
        choices=["P0", "P1", "P2", "all"],
        help="Run experiments up to this priority tier.",
    )
    parser.add_argument("--arch", type=str, default="smallcnn")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--results-root", type=str, default="./results/image")
    parser.add_argument("--checkpoints-root", type=str, default="./checkpoints/image")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Explicit list of experiments to run (overrides --priority).",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=None,
        help="Explicit epsilon list (use -1 for inf).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None
    )
    args = parser.parse_args()

    epsilons = None
    if args.epsilons is not None:
        epsilons = [float("inf") if e < 0 else e for e in args.epsilons]

    run_image_sweep(
        experiments=args.experiments,
        epsilons=epsilons,
        seeds=args.seeds,
        priority=args.priority,
        arch=args.arch,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        data_root=args.data_root,
        results_root=args.results_root,
        checkpoints_root=args.checkpoints_root,
        debug=args.debug,
    )
