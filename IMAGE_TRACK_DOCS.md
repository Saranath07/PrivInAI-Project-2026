# DP-ShiftBench — Image Track: Complete Documentation

> Everything implemented for the image track: architecture, every file, every function, and full results.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Project Structure](#2-project-structure)
3. [Experiments Overview](#3-experiments-overview)
4. [File-by-File Reference](#4-file-by-file-reference)
   - [Datasets](#41-dp_shift_benchdatasetsimage_shiftspy)
   - [Models](#42-dp_shift_benchmodelsimage_modelspy)
   - [Standard Trainer](#43-dp_shift_benchtrainingstandard_trainerpy)
   - [DP Trainer](#44-dp_shift_benchtrainingdp_trainerpy)
   - [Training Utils](#45-dp_shift_benchtrainingutilspy)
   - [Divergences](#46-dp_shift_benchmetricsdivergencespy)
   - [Evaluation](#47-dp_shift_benchmetricsevaluationpy)
   - [run\_single](#48-dp_shift_benchrunnersrun_singlepy)
   - [run\_sweep](#49-dp_shift_benchrunnersrun_sweeppy)
   - [run\_divergences](#410-dp_shift_benchrunnersrun_divergencespy)
   - [image\_analysis](#411-experimentsimage_analysispy)
   - [Configs](#412-configs)
   - [Tests](#413-tests)
5. [How to Run](#5-how-to-run)
6. [Full Results](#6-full-results)
   - [Accuracy Table](#61-accuracy-table-mean--std)
   - [Privacy Cost](#62-privacy-cost)
   - [Full Ranking](#63-full-ranking-at-05)
   - [Divergence Metrics](#64-divergence-metrics)
7. [Key Findings](#7-key-findings)
8. [Bugs Fixed During Implementation](#8-bugs-fixed-during-implementation)

---

## 1. What This Project Does

**DP-ShiftBench** studies how *distribution shift between pretrain and finetune data* affects the privacy-utility tradeoff in differentially private deep learning (DP-SGD).

The core question: if you pretrain a CNN on dataset A and then finetune it privately on dataset B, does the similarity between A and B change how much accuracy you lose from DP noise?

**Answer from our experiments:** Yes, dramatically. A well-matched pretrained feature extractor reduces the accuracy penalty from DP noise by up to 5×. But surprisingly, standard distribution metrics (FID, JSD) do not predict *how* useful a pretrained model will be — class-level semantic alignment is what matters, and it's invisible to those metrics.

The pipeline for each experiment:

```
Pretrain dataset ──► [pretrain: supervised or autoencoder] ──► checkpoint
                                                                    │
Finetune dataset ──────────────────────────────────────────────────►│
                                                                    ▼
                                              [DP-SGD finetune with Opacus]
                                                                    │
                                                                    ▼
                                                        test accuracy + ε_actual
```

---

## 2. Project Structure

```
PrivInAI-Project-2026/
├── dp_shift_bench/
│   ├── configs/
│   │   ├── default.yaml          # global hyperparameters
│   │   └── image_experiments.yaml # per-experiment dataset config
│   ├── datasets/
│   │   └── image_shifts.py       # all 11 dataset loaders
│   ├── metrics/
│   │   ├── divergences.py        # FID, KL, JSD, TV, MMD, PAD, Wasserstein
│   │   └── evaluation.py         # accuracy evaluation helper
│   ├── models/
│   │   └── image_models.py       # SmallCNN, ResNet-18-GN, ConvAutoencoder
│   ├── runners/
│   │   ├── run_single.py         # run one experiment (one ε, one seed)
│   │   ├── run_sweep.py          # grid sweep over all ε × seeds
│   │   └── run_divergences.py    # compute divergences post-sweep
│   └── training/
│       ├── dp_trainer.py         # Opacus DP-SGD finetuning
│       ├── standard_trainer.py   # standard pretrain (supervised + AE)
│       └── utils.py              # seed setting, LR scheduler helpers
├── experiments/
│   └── image_analysis.py         # figures, tables, stats
├── tests/
│   ├── test_datasets.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_divergences.py
├── results/
│   ├── figures/                  # 6 PNG figures
│   ├── image/
│   │   ├── IMG-00/ … IMG-10/     # JSON result files
│   │   ├── divergences/          # divergence JSON files
│   │   └── tables/               # LaTeX + CSV tables
├── checkpoints/
│   └── image/
│       ├── IMG-01/pretrained.pt  # supervised SVHN full
│       ├── IMG-02/pretrained.pt  # supervised SVHN {0-4}
│       ├── IMG-03/pretrained.pt  # supervised SVHN {5-9}
│       ├── IMG-04/autoencoder.pt # autoencoder on SVHN full
│       ├── IMG-05 … IMG-10/      # supervised pretrain checkpoints
└── FINDINGS.md                   # results analysis
```

---

## 3. Experiments Overview

| ID | Pretrain Data | Pretrain Method | Finetune Data | Purpose | Priority |
|---|---|---|---|---|---|
| IMG-00 | — | None | MNIST full | Baseline (no pretrain) | P0 |
| IMG-01 | SVHN full | Supervised | MNIST full | Domain shift baseline | P0 |
| IMG-02 | SVHN {0-4} | Supervised | MNIST {0-4} | Matched classes | P0 |
| IMG-03 | SVHN {5-9} | Supervised | MNIST {0-4} | Mismatched classes | P0 |
| IMG-04 | SVHN full | Autoencoder | MNIST full | Unsupervised pretrain | P1 |
| IMG-05 | SVHN 10% | Supervised | MNIST full | Data scarcity | P1 |
| IMG-06 | SVHN 25% | Supervised | MNIST full | Data scarcity | P1 |
| IMG-07 | SVHN 50% | Supervised | MNIST full | Data scarcity | P1 |
| IMG-08 | CIFAR-10 full | Supervised | MNIST full | Larger domain gap | P1 |
| IMG-09 | SVHN + augmentation | Supervised | MNIST full | Style shift | P2 |
| IMG-10 | FashionMNIST full | Supervised | MNIST full | Format-similar, different semantics | P2 |

**Epsilon grid:** {0.5, 1.0, 2.0, 4.0, 8.0, ∞} — δ=1e-5 for all finite ε  
**Seeds:** 42, 123, 456 → 3 runs per (experiment, ε) → **198 total runs**  
**Model:** SmallCNN (~500K params, GroupNorm) for all experiments  
**Hardware:** NVIDIA A100

---

## 4. File-by-File Reference

---

### 4.1 `dp_shift_bench/datasets/image_shifts.py`

Provides `DataLoader` pairs (pretrain loader, finetune loader) for each experiment.

**Transforms defined at module level:**

| Name | Input | Output | Notes |
|---|---|---|---|
| `svhn_transform` | 32×32 RGB | 3×32×32, [0,1] | Just ToTensor |
| `mnist_transform` | 28×28 grayscale | 3×32×32, [0,1] | Resize + Grayscale(3ch) |
| `cifar10_transform` | 32×32 RGB | 3×32×32, [0,1] | Just ToTensor |
| `fashionmnist_transform` | 28×28 grayscale | 3×32×32, [0,1] | Same as MNIST |
| `svhn_aug_transform` | 32×32 RGB | 3×32×32, [0,1] | + RandomCrop, ColorJitter, flip |

**Classes:**

#### `ClassSubset(Dataset)`
Wraps any dataset, keeps only specified classes, and remaps their labels to 0…N-1.

```python
ClassSubset(dataset, classes=[0,1,2,3,4])
# keeps only samples with label in classes
# remaps: 0→0, 1→1, 2→2, 3→3, 4→4
```

| Method | What it does |
|---|---|
| `__init__(dataset, classes)` | Filters indices; builds `class_to_idx` remap dict |
| `__len__()` | Returns number of filtered samples |
| `__getitem__(idx)` | Returns `(image, remapped_label)` |

#### `AugmentedDataset(Dataset)`
Wraps a dataset and applies a new transform at getitem time, replacing the original.

```python
AugmentedDataset(svhn_base_dataset, svhn_aug_transform)
```

| Method | What it does |
|---|---|
| `__init__(dataset, transform)` | Stores dataset reference and new transform |
| `__len__()` | Delegates to wrapped dataset |
| `__getitem__(idx)` | Gets raw PIL image, applies new transform |

**Main function:**

#### `get_image_experiment_data(experiment_id, batch_size, debug, data_root)`
Single entry point for all 11 experiments. Returns:

```python
{
    "pretrain_loader": DataLoader | None,   # None for IMG-00
    "finetune_loader": DataLoader,
    "test_loader": DataLoader,
    "num_pretrain_classes": int | None,
    "num_finetune_classes": int,
    "experiment_id": str,
}
```

Per-experiment logic:

| Experiment | Pretrain source | Finetune source | Special logic |
|---|---|---|---|
| IMG-00 | None | MNIST full train | No pretrain |
| IMG-01 | SVHN full train | MNIST full train | — |
| IMG-02 | SVHN labels {0-4} | MNIST labels {0-4} | ClassSubset on both |
| IMG-03 | SVHN labels {5-9} | MNIST labels {0-4} | ClassSubset; mismatched |
| IMG-04 | SVHN full train | MNIST full train | Same as IMG-01; AE trains on this |
| IMG-05 | SVHN 10% random | MNIST full train | `np.random.choice(73257, int(0.1*73257))` |
| IMG-06 | SVHN 25% random | MNIST full train | Same, 25% |
| IMG-07 | SVHN 50% random | MNIST full train | Same, 50% |
| IMG-08 | CIFAR-10 full train | MNIST full train | Different domain |
| IMG-09 | SVHN full + aug | MNIST full train | AugmentedDataset wrapping SVHN |
| IMG-10 | FashionMNIST full | MNIST full train | Grayscale→3ch |

**Debug mode:** when `debug=True`, all datasets are capped at 500 samples (fast smoke test).

---

### 4.2 `dp_shift_bench/models/image_models.py`

All models use **GroupNorm** instead of BatchNorm. BatchNorm is incompatible with Opacus because it computes statistics over the batch, leaking information across samples.

#### `SmallCNN(nn.Module)` — Primary model (~500K params)

Input: 3×32×32. Output: logits of shape (N, num_classes).

```
features:
  Conv2d(3→32, 3×3) → GroupNorm(8,32) → ReLU → MaxPool2d(2)    [32×16×16]
  Conv2d(32→64, 3×3) → GroupNorm(8,64) → ReLU → MaxPool2d(2)   [64×8×8]
  Conv2d(64→128, 3×3) → GroupNorm(8,128) → ReLU → MaxPool2d(2) [128×4×4]
  → flatten → 2048-dim

classifier:
  Linear(2048→256) → ReLU → Dropout(0.2)   [256-dim features]

fc:
  Linear(256→num_classes)
```

| Method | What it does |
|---|---|
| `__init__(num_classes)` | Builds features, classifier, fc layers |
| `forward(x)` | Full forward pass → logits |
| `get_features(x)` | Returns 256-dim vector before fc (used for divergence extraction) |

**Critical detail:** All `ReLU` are non-inplace (`nn.ReLU()`, not `nn.ReLU(inplace=True)`). Inplace ops break Opacus backward hooks.

#### `replace_bn_with_gn(module)` — utility function
Recursively replaces every `BatchNorm2d` in a module tree with `GroupNorm`. Finds the largest divisor ≤ 8 of the channel count to use as `num_groups`.

#### `resnet18_groupnorm(num_classes)` — Architecture 2
ResNet-18 adapted for 32×32 inputs with all BN replaced by GN.

- `conv1`: kernel 3×3, stride 1, padding 1 (original is 7×7 stride 2)
- `maxpool`: replaced with `nn.Identity()` (no downsampling at the start)
- All `BatchNorm2d` replaced by `GroupNorm` via `replace_bn_with_gn`

#### `ConvAutoencoder(nn.Module)` — Architecture 3 (for IMG-04)

Encoder mirrors `SmallCNN.features` exactly so weights transfer cleanly.

```
encoder:
  Conv2d(3→32) → GN → ReLU → MaxPool2d(2)
  Conv2d(32→64) → GN → ReLU → MaxPool2d(2)
  Conv2d(64→128) → GN → ReLU → MaxPool2d(2)   → 128×4×4 bottleneck

decoder:
  ConvTranspose2d(128→64, 4, stride=2) → GN → ReLU   → 64×8×8
  ConvTranspose2d(64→32, 4, stride=2) → GN → ReLU    → 32×16×16
  ConvTranspose2d(32→3, 4, stride=2) → Sigmoid        → 3×32×32
```

| Method | What it does |
|---|---|
| `forward(x)` | Encodes then decodes → reconstructed image in [0,1] |
| `get_encoder_state_dict()` | Returns state dict loadable directly into `SmallCNN.features` |

#### `create_model(arch, num_classes, pretrained_path, pretrain_num_classes, autoencoder_pretrained_path)`
Factory function. Handles 4 cases:

1. **No pretrain** (`pretrained_path=None`, `autoencoder_pretrained_path=None`): returns randomly initialised model.
2. **Full supervised pretrain** (`pretrained_path` set, classes match): loads weights directly.
3. **Supervised pretrain with head mismatch** (`pretrain_num_classes != num_classes`): drops `fc.*` keys, loads feature extractor only. FC is randomly initialised.
4. **Autoencoder pretrain** (`autoencoder_pretrained_path` set): strips `encoder.` prefix from AE weights and loads into `SmallCNN.features`.

---

### 4.3 `dp_shift_bench/training/standard_trainer.py`

Non-private training for the pretrain phase.

#### `pretrain_supervised(model, dataloader, epochs, lr, device, experiment_id)`
Standard SGD training with cosine annealing LR schedule.

- Optimizer: `SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)`
- Scheduler: `CosineAnnealingLR(T_max=epochs)`
- Loss: `CrossEntropyLoss`
- Prints per-epoch: loss, train accuracy, LR
- Returns the trained model

#### `pretrain_autoencoder(model, dataloader, epochs, lr, device)`
Trains a `ConvAutoencoder` with pixel reconstruction loss.

- Optimizer: `Adam(lr=0.001)`
- Loss: `MSELoss` (pixel-level reconstruction)
- Prints per-epoch: reconstruction loss
- Returns the trained autoencoder

---

### 4.4 `dp_shift_bench/training/dp_trainer.py`

Differentially private finetuning using [Opacus](https://opacus.ai/).

#### `finetune_dp(model, train_loader, test_loader, epsilon, delta, epochs, lr, max_grad_norm, device)`

Top-level entry point. Routes to either `_train_standard` (ε=∞) or `_train_private` (finite ε).

**For ε=∞ (non-private):**
Calls `_train_standard` — plain SGD+cosine LR, no clipping, no noise.

**For finite ε:**
Calls `_train_private` using Opacus:

1. Wraps model with `GradSampleModule` (Opacus per-sample gradient computation)
2. Wraps optimizer with `PrivacyEngine.make_private_with_epsilon()`:
   - Sets noise multiplier automatically to hit target ε at `delta=1e-5` over `epochs` epochs
   - Uses RDP (Rényi Differential Privacy) accounting
3. Trains with gradient clipping (`max_grad_norm=1.0`) + calibrated Gaussian noise
4. After training, unwraps model via `model._module` to get the plain `nn.Module` back

Returns:
```python
{
    "train_losses": [float, ...],   # per-epoch
    "test_accs": [float, ...],      # per-epoch
    "epsilon_actual": float,        # ε spent (from RDP accountant)
    "epochs_run": int,
}
```

**Why GroupNorm is required:** Opacus computes per-sample gradients by adding a batch dimension hook. BatchNorm's statistics over the batch make per-sample gradients ill-defined (the gradient of sample i depends on samples j≠i). GroupNorm operates per-sample, so this problem disappears.

---

### 4.5 `dp_shift_bench/training/utils.py`

Small utilities used across training scripts.

#### `set_seed(seed)`
Sets `random`, `numpy`, and `torch` seeds + `torch.backends.cudnn.deterministic = True`.

#### `get_lr(optimizer)`
Returns current LR from the first param group of an optimizer.

---

### 4.6 `dp_shift_bench/metrics/divergences.py`

Seven distribution divergence metrics computed between pretrain and finetune feature sets.

#### Feature extraction helpers

##### `extract_features(model, dataloader, device, max_samples)`
Runs the model in eval mode, calling `model.get_features(x)` if available (returns the 256-dim penultimate layer), otherwise `model(x)`. Returns `ndarray` of shape `(N, feature_dim)`.

##### `_get_inception_model(device)`
Loads pretrained InceptionV3, disables auxiliary logits, replaces fc with `nn.Identity()` to get 2048-dim features.

**Bug fixed:** Modern torchvision rejects `inception_v3(aux_logits=False)` with pretrained weights. Fix: instantiate without that argument, then set `model.aux_logits = False; model.AuxLogits = None` after construction.

##### `_extract_inception_features(dataloader, device, max_samples)`
Extracts 2048-dim InceptionV3 features, resizing inputs to 299×299 with bilinear interpolation.

#### Metric functions

##### `compute_fid(dataset1, dataset2, device)` → float
Fréchet Inception Distance using InceptionV3 features.

```
FID = ||μ₁−μ₂||² + Tr(Σ₁ + Σ₂ − 2·√(Σ₁Σ₂))
```

Uses `scipy.linalg.sqrtm` for the matrix square root. Falls back to diagonal offset if the result is complex (numerical instability for ill-conditioned covariances).

##### `compute_mmd(features1, features2, kernel, gamma)` → float
Maximum Mean Discrepancy with RBF kernel + median heuristic for bandwidth.

```
MMD² = E[k(x,x')] + E[k(y,y')] − 2·E[k(x,y)]
```

Subsamples to 5000 points. Gamma = 1 / (2 · median_pairwise_dist²).

##### `compute_proxy_a_distance(features1, features2, max_samples)` → float
Train a logistic regression classifier to separate the two domains. PAD = 2·(1 − 2·error).

- PAD = 0: identical distributions (classifier at chance)
- PAD = 2: perfectly separable (classifier perfect)

Uses 5-fold CV. Subsamples to 5000 points.

##### `compute_kl_divergence_estimated(features1, features2, k)` → float
k-NN KL divergence estimator (Wang et al. 2009):

```
KL(P‖Q) ≈ (d/n)·Σᵢ log(νₖ(xᵢ)/ρₖ(xᵢ)) + log(m/(n−1))
```

where ρₖ = k-th NN distance in P (excluding self), νₖ = k-th NN distance in Q.  
Subsamples to 5000 points. Returns `max(0, kl)` to avoid negative estimates.

##### `compute_jsd(features1, features2)` → float
Jensen-Shannon Divergence ∈ [0, log 2].

**Two bugs fixed vs naive implementation:**

1. **Self-inclusion bug:** Original code concatenated `features1` into the mixture, then queried the mixture using `features1`. Each point found itself at distance 0, making `log(ν/ρ)` systematically negative → JSD always 0. **Fix:** split each feature set in half; one half builds the mixture, the other half is the query (independent samples).

2. **High-dimensionality bug:** Even with independent samples, the k-NN estimator is unreliable at d=256 (n/d ratio only ~5×). **Fix:** apply PCA to reduce to 32 dimensions before the k-NN step (n/d ratio ~39×, well within the reliable regime).

##### `compute_total_variation(features1, features2)` → float
Per-dimension histogram TV distance, averaged across dimensions.

For each dimension: build 50-bin histogram, compute 0.5 · Σ|h₁−h₂|. Return mean over all d dimensions.

##### `compute_wasserstein(features1, features2, max_samples)` → float
Wasserstein-1 (Earth Mover's Distance) via the POT library (`ot.emd2`). Aggressively subsamples to 2000 points (O(n³) solver).

#### Master function

##### `compute_all_divergences(pretrain_loader, finetune_loader, feature_model, device, priority)` → dict
Runs all metrics in sequence. `priority="P0"` runs FID+KL+JSD+TV; `priority="P1"` or `"all"` adds MMD+PAD+Wasserstein.

---

### 4.7 `dp_shift_bench/metrics/evaluation.py`

#### `evaluate(model, dataloader, device)` → float
Runs model in eval mode, returns fraction of correct predictions across the full dataloader.

---

### 4.8 `dp_shift_bench/runners/run_single.py`

Full pipeline for a single (experiment_id, epsilon, seed) triple.

#### `run_single_experiment(experiment_id, epsilon, seed, arch, device, debug)`

Steps:
1. `set_seed(seed)`
2. Load dataset pair via `get_image_experiment_data`
3. **Pretrain phase** (skipped for IMG-00 and ε=∞ if checkpoint exists):
   - IMG-04: `pretrain_autoencoder` → saves `checkpoints/image/IMG-04/autoencoder.pt`
   - All others: `pretrain_supervised` → saves `checkpoints/image/{exp_id}/pretrained.pt`
   - Checkpoint is reused across seeds (pretrain is deterministic per experiment, not per seed)
4. Create model via `create_model` with appropriate pretrained weights
5. `finetune_dp` → get accuracy + privacy metrics
6. Save JSON result to `results/image/{exp_id}/eps_{epsilon}_seed_{seed}.json`

Result JSON schema:
```json
{
  "experiment_id": "IMG-01",
  "epsilon": 1.0,
  "seed": 42,
  "arch": "smallcnn",
  "debug": false,
  "final_test_acc": 0.9609,
  "best_test_acc": 0.9609,
  "epsilon_actual": 0.9987,
  "train_losses": [...],
  "test_accs": [...],
  "epochs_run": 15
}
```

---

### 4.9 `dp_shift_bench/runners/run_sweep.py`

Grid sweep over all (experiment, ε, seed) combinations.

#### `_result_exists(experiment_id, epsilon, seed)` → bool
Checks if the result JSON file already exists. Enables **resume** — interrupted sweeps restart where they left off without re-running completed experiments.

#### `run_sweep(priority, arch, device, debug)`
Builds the full grid:
- `priority="P0"`: IMG-00 to IMG-03
- `priority="P1"`: all P0 + IMG-04 to IMG-08 (162 total runs)
- `priority="P2"` or `"all"`: all P1 + IMG-09 to IMG-10 (198 total runs)

For each `(experiment_id, epsilon, seed)` triple:
- Prints `[SKIP]` if result exists, `[RUN]` otherwise
- Calls `run_single_experiment` for new runs

**CLI usage:**
```bash
python -m dp_shift_bench.runners.run_sweep --priority P2 --device cuda
```

---

### 4.10 `dp_shift_bench/runners/run_divergences.py`

Post-sweep divergence computation.

#### `compute_image_divergences(priority, device)`

For each experiment (IMG-01 to IMG-08 for P1, all for `--priority all`):

1. Load the experiment's own SmallCNN checkpoint as the feature extractor
   - Special case IMG-04 (autoencoder): falls back to IMG-01 checkpoint since the autoencoder has no supervised classification head
2. Get pretrain and finetune dataloaders
3. Call `compute_all_divergences(priority="P1")` → all 7 metrics
4. Save to `results/image/divergences/{exp_id}.json`

**CLI usage:**
```bash
python -m dp_shift_bench.runners.run_divergences --priority all
```

---

### 4.11 `experiments/image_analysis.py`

Loads all result and divergence JSONs, generates 6 figures and 2 tables.

#### Data loading

##### `load_results()` → `pd.DataFrame`
Walks `results/image/IMG-*/` directories, loads all JSON files, returns a DataFrame with columns: `experiment_id`, `description`, `epsilon`, `seed`, `final_test_acc`.

##### `load_divergences()` → `pd.DataFrame`
Loads `results/image/divergences/*.json` into a DataFrame with one row per experiment.

#### Figure generation (all saved to `results/figures/`)

| Function | Output file | What it shows |
|---|---|---|
| `fig1_acc_vs_epsilon(df)` | `fig1_acc_vs_epsilon.png` | Line plot: test accuracy vs ε for all 11 experiments (mean ± std) |
| `fig2_pretrain_gain(df)` | `fig2_pretrain_gain.png` | Bar chart: accuracy gain over no-pretrain baseline at ε=0.5 |
| `fig3_div_vs_acc(df, div_df)` | `fig3_div_vs_acc.png` | Scatter: JSD vs accuracy@ε=0.5 for each experiment |
| `fig4_corr_heatmap(df, div_df)` | `fig4_corr_heatmap.png` | Pearson correlation heatmap between all divergence metrics and accuracy |
| `fig5_phase_diagram(df)` | `fig5_phase_diagram.png` | Heatmap: experiment × ε, cell = accuracy |
| `fig6_hypothesis_tests(df)` | `fig6_hypotheses.png` | Two-way ANOVA results + pairwise experiment comparisons |

#### Table generation (saved to `results/image/tables/`)

##### `generate_tables(df, div_df)`
- `results_table.{tex,csv}`: accuracy mean±std for every (experiment, ε) combination
- `divergence_table.{tex,csv}`: all 7 divergence metrics for each experiment

#### Stats

##### `summary_stats(df)`
Prints mean and std of accuracy grouped by epsilon (across all experiments and seeds).

---

### 4.12 Configs

#### `dp_shift_bench/configs/default.yaml`

```yaml
pretrain:
  epochs: 20
  lr: 0.01
  batch_size: 256

autoencoder:
  epochs: 30
  lr: 0.001

finetune:
  epochs: 15
  lr: 0.001
  batch_size: 256
  delta: 1.0e-5
  max_grad_norm: 1.0

epsilons: [0.5, 1.0, 2.0, 4.0, 8.0, "inf"]
seeds: [42, 123, 456]
```

#### `dp_shift_bench/configs/image_experiments.yaml`

Maps each experiment ID to its dataset configuration:

```yaml
experiments:
  IMG-00:
    pretrain_data: null
    pretrain_method: null
    finetune_data: "mnist_full"
  IMG-01:
    pretrain_data: "svhn_full"
    pretrain_method: "supervised"
    finetune_data: "mnist_full"
  IMG-02:
    pretrain_data: "svhn_04"
    pretrain_method: "supervised"
    finetune_data: "mnist_04"
  # ... etc
```

---

### 4.13 Tests

Four test files with 30+ unit tests covering the full stack.

#### `tests/test_datasets.py`
- Checks all 11 `get_image_experiment_data` calls return non-None loaders
- Verifies batch shapes: `(B, 3, 32, 32)` images, integer labels
- Verifies label range: [0, num_classes-1] for ClassSubset experiments
- Verifies IMG-05/06/07 have correct proportional sizes (10/25/50% of SVHN)
- Smoke-tests debug mode (500 samples cap)

#### `tests/test_models.py`
- Tests SmallCNN forward pass shape `(B, num_classes)`
- Tests `get_features` returns `(B, 256)`
- Tests ResNet-18-GN forward pass
- Tests ConvAutoencoder reconstruction shape `(B, 3, 32, 32)`
- **Tests Opacus compatibility:** wraps SmallCNN with `GradSampleModule` and verifies no error (catches inplace-ReLU issues)
- Tests `create_model` with pretrained_path and head mismatch

#### `tests/test_training.py`
- Tests `pretrain_supervised` runs 2 epochs without error on a tiny dataset
- Tests `pretrain_autoencoder` runs 2 epochs
- Tests `finetune_dp` at ε=2.0 converges (accuracy > 50%) on MNIST subset
- Tests `finetune_dp` at ε=∞ (non-private path)

#### `tests/test_divergences.py`
- Tests each metric function on random feature arrays
- Verifies FID ≥ 0, MMD ≥ 0, PAD ∈ [0, 2], JSD ∈ [0, log 2]
- Tests `compute_all_divergences` with priority P0 and P1
- Tests `extract_features` returns correct shape

---

## 5. How to Run

```bash
# 1. P0 sweep (IMG-00 to IMG-03): 72 runs
python -m dp_shift_bench.runners.run_sweep --priority P0

# 2. P1 sweep (adds IMG-04 to IMG-08): 162 total runs (skips P0 if done)
python -m dp_shift_bench.runners.run_sweep --priority P1

# 3. P2 sweep (adds IMG-09, IMG-10): 198 total runs (skips P0+P1 if done)
python -m dp_shift_bench.runners.run_sweep --priority P2

# 4. Divergences (all experiments)
python -m dp_shift_bench.runners.run_divergences --priority all

# 5. Analysis figures and tables
python experiments/image_analysis.py

# Debug mode (fast, 500 samples, 2 epochs)
python -m dp_shift_bench.runners.run_sweep --priority P0 --debug
```

---

## 6. Full Results

### 6.1 Accuracy Table (mean ± std across 3 seeds)

| Experiment | Description | ε=0.5 | ε=1.0 | ε=2.0 | ε=4.0 | ε=8.0 | ε=∞ |
|---|---|---|---|---|---|---|---|
| IMG-00 | No pretrain | 0.923 ±0.004 | 0.924 ±0.004 | 0.924 ±0.004 | 0.923 ±0.003 | 0.923 ±0.004 | 0.990 ±0.001 |
| IMG-01 | SVHN full | 0.960 ±0.000 | 0.961 ±0.000 | 0.961 ±0.000 | 0.961 ±0.001 | 0.961 ±0.001 | 0.991 ±0.001 |
| IMG-02 | SVHN {0-4} matched | 0.985 ±0.001 | 0.985 ±0.001 | 0.984 ±0.000 | 0.984 ±0.000 | 0.984 ±0.000 | 0.998 ±0.000 |
| IMG-03 | SVHN {5-9} mismatched | 0.968 ±0.000 | 0.968 ±0.000 | 0.968 ±0.001 | 0.968 ±0.001 | 0.968 ±0.001 | 0.997 ±0.000 |
| IMG-04 | SVHN autoencoder | 0.877 ±0.006 | 0.877 ±0.003 | 0.877 ±0.001 | 0.877 ±0.002 | 0.877 ±0.002 | 0.988 ±0.000 |
| IMG-05 | SVHN 10% | 0.953 ±0.000 | 0.953 ±0.001 | 0.954 ±0.001 | 0.954 ±0.000 | 0.953 ±0.000 | 0.990 ±0.000 |
| IMG-06 | SVHN 25% | 0.953 ±0.001 | 0.953 ±0.000 | 0.953 ±0.000 | 0.953 ±0.000 | 0.953 ±0.000 | 0.989 ±0.001 |
| IMG-07 | SVHN 50% | 0.955 ±0.000 | 0.956 ±0.000 | 0.956 ±0.000 | 0.956 ±0.000 | 0.956 ±0.000 | 0.990 ±0.000 |
| IMG-08 | CIFAR-10 | 0.942 ±0.000 | 0.943 ±0.000 | 0.943 ±0.000 | 0.943 ±0.000 | 0.943 ±0.000 | 0.991 ±0.000 |
| IMG-09 | SVHN + augmentation | 0.958 ±0.001 | 0.959 ±0.000 | 0.959 ±0.000 | 0.960 ±0.000 | 0.960 ±0.000 | 0.990 ±0.000 |
| IMG-10 | FashionMNIST | 0.936 ±0.000 | 0.936 ±0.000 | 0.936 ±0.000 | 0.936 ±0.000 | 0.936 ±0.000 | 0.990 ±0.001 |

### 6.2 Privacy Cost

Privacy cost = accuracy drop from ε=∞ to ε=0.5. Lower is better.

| Experiment | ε=∞ acc | ε=0.5 acc | Privacy Cost |
|---|---|---|---|
| IMG-02 | 0.998 | 0.985 | **−1.33%** ← best |
| IMG-03 | 0.997 | 0.968 | −2.83% |
| IMG-01 | 0.991 | 0.960 | −3.16% |
| IMG-09 | 0.990 | 0.958 | −3.19% |
| IMG-07 | 0.990 | 0.955 | −3.52% |
| IMG-05 | 0.990 | 0.953 | −3.62% |
| IMG-06 | 0.989 | 0.953 | −3.64% |
| IMG-08 | 0.991 | 0.942 | −4.87% |
| IMG-10 | 0.990 | 0.936 | −5.45% |
| IMG-00 | 0.990 | 0.923 | −6.66% |
| IMG-04 | 0.988 | 0.877 | **−11.05%** ← worst |

### 6.3 Full Ranking at ε=0.5

| Rank | Experiment | Description | ε=0.5 acc |
|---|---|---|---|
| 1 | IMG-02 | SVHN {0-4} → MNIST {0-4} matched | **98.46%** |
| 2 | IMG-03 | SVHN {5-9} → MNIST {0-4} mismatched | 96.84% |
| 3 | IMG-01 | SVHN full → MNIST full | 95.97% |
| 4 | IMG-09 | SVHN + augmentation → MNIST | 95.85% |
| 5 | IMG-07 | SVHN 50% → MNIST | 95.52% |
| 6 | IMG-05 | SVHN 10% → MNIST | 95.34% |
| 7 | IMG-06 | SVHN 25% → MNIST | 95.29% |
| 8 | IMG-08 | CIFAR-10 → MNIST | 94.19% |
| 9 | IMG-10 | FashionMNIST → MNIST | 93.55% |
| 10 | IMG-00 | No pretrain (baseline) | 92.33% |
| 11 | IMG-04 | SVHN autoencoder → MNIST | **87.71%** |

### 6.4 Divergence Metrics

Computed between pretrain and finetune feature distributions (SmallCNN 256-dim → PCA 32-dim for JSD/KL).

| Experiment | FID | KL | JSD | TV | MMD | PAD | W1 |
|---|---|---|---|---|---|---|---|
| IMG-01 SVHN full | 196.7 | 52.7 | 0.263 | 0.139 | 0.061 | 1.974 | 10.96 |
| IMG-07 SVHN 50% | 194.7 | 51.1 | 0.377 | 0.139 | 0.059 | 1.972 | 10.51 |
| IMG-06 SVHN 25% | 194.0 | 56.6 | 0.430 | 0.143 | 0.063 | 1.977 | 10.13 |
| IMG-04 AE | 194.6 | 53.4 | 0.490 | 0.139 | 0.060 | 1.974 | 11.03 |
| IMG-05 SVHN 10% | 196.1 | 67.5 | 0.490 | 0.178 | 0.103 | 1.990 | 10.97 |
| IMG-09 SVHN+aug | 191.4 | 58.7 | 0.493 | 0.143 | 0.062 | 1.982 | 11.12 |
| IMG-03 mismatched | 190.1 | 89.2 | 0.504 | 0.181 | 0.138 | 1.984 | 9.72 |
| IMG-02 matched | 204.8 | 63.8 | 0.523 | 0.166 | 0.079 | 1.982 | 9.02 |
| IMG-08 CIFAR-10 | 254.8 | 65.8 | 0.623 | 0.257 | 0.169 | 1.996 | 12.75 |
| IMG-10 FashionMNIST | 239.4 | 185.9 | **0.641** | **0.289** | 0.139 | 1.991 | 13.41 |

---

## 7. Key Findings

### Finding 1 — Supervised pretraining consistently reduces privacy cost
All supervised pretrained models (IMG-01 through IMG-10 except IMG-04) suffer a smaller accuracy drop under DP than the no-pretrain baseline. The baseline loses **6.66%** from ε=∞ to ε=0.5; IMG-02 loses only **1.33%** — a **5× reduction**.

### Finding 2 — Matched classes (IMG-02) is the best strategy
IMG-02 (SVHN digits 0-4 → MNIST digits 0-4) achieves **98.46%** at ε=0.5, the best of all 11 experiments. Class-level alignment matters more than just domain proximity.

### Finding 3 — Accuracy is flat across ε for pretrained models
For all supervised pretrained models, the difference between ε=0.5 and ε=8.0 is < 0.2%. The epsilon dimension only separates pretrained from non-pretrained, not different epsilon values within an experiment. Once you have good pretrained features, tightening privacy from ε=8 to ε=0.5 costs almost nothing.

### Finding 4 — Mismatched classes still help (IMG-03 > IMG-00)
SVHN {5-9} → MNIST {0-4}: the pretrain classes are completely different from finetune classes, yet still +4.5% gain over baseline. Low-level visual features (edges, strokes) transfer even without semantic label overlap.

### Finding 5 — Very low variance across seeds (std ≤ 0.004)
DP noise does not introduce training instability at these epsilon values. Results are highly reproducible.

### Finding 6 — Autoencoder pretrain (IMG-04) is worse than no pretrain
IMG-04 achieves only 87.71% at ε=0.5 vs 92.33% baseline. Privacy cost is 11.05% — nearly double the no-pretrain baseline. Reconstruction-optimized features don't help DP-SGD; discriminative features are required.

### Finding 7 — Pretrain data volume plateaus above 10%
SVHN 10% (95.34%) vs 50% (95.52%): only 0.18% difference. Even ~7,300 images is sufficient.

### Finding 8 — Domain gap hierarchy is clear
`matched classes > mismatched SVHN > full SVHN ≈ SVHN subsets > CIFAR-10 > FashionMNIST > no pretrain >> autoencoder`

### Finding 9 — Augmentation during pretrain is negligible
IMG-09 (SVHN + aug, 95.85%) vs IMG-01 (SVHN clean, 95.97%): only 0.12% difference.

### Finding 10 — FashionMNIST (format-similar, semantically different) barely helps
+1.22% over baseline at ε=0.5. Sharing the 28×28 grayscale format is insufficient; semantic relevance is needed.

### Finding 11 — Divergence metrics do NOT predict DP-finetuning accuracy
IMG-02 (best accuracy, 98.46%) has *higher* JSD (0.523) than IMG-01 (95.97%, JSD=0.263). IMG-04 (worst among pretrained, 87.71%) and IMG-05 (95.34%) have identical JSD (0.490). Standard distribution metrics are blind to what matters: the discriminativeness of pretrained features. PAD is near 2.0 (maximum) for all experiments — all pretrain/finetune pairs are always nearly perfectly linearly separable, making PAD useless for ranking.

### Finding 12 — FashionMNIST has anomalously high KL (185.9 vs 50-90 for SVHN)
The FashionMNIST feature distribution is far more spread in the pretrained feature space, suggesting the SmallCNN features from SVHN pretraining have poor coverage of the clothing-image manifold.

---

## 8. Bugs Fixed During Implementation

### Bug 1: Inplace ReLU + Opacus conflict

**Error:**
```
RuntimeError: Output 0 of BackwardHookFunctionBackward is a view and is being modified inplace
```

**Cause:** `nn.ReLU(inplace=True)` modifies activations that Opacus backward hooks are monitoring.

**Fix:** Changed all 8 occurrences of `nn.ReLU(inplace=True)` → `nn.ReLU()` in `image_models.py`.

---

### Bug 2: InceptionV3 `aux_logits=False` rejection

**Error:**
```
ValueError: The parameter 'aux_logits' expected value True but got False instead.
```

**Cause:** Modern torchvision (`>=0.13`) enforces `aux_logits=True` when loading pretrained InceptionV3 weights.

**Fix:** Remove `aux_logits=False` from the constructor call. Set `model.aux_logits = False; model.AuxLogits = None` after construction.

---

### Bug 3: JSD always zero

**Cause (part 1 — self-inclusion):** The mixture `M = concat(features1, features2)` was passed as the reference to the k-NN KL estimator while `features1` was the query. Every query point found itself in `M` at distance 0, making `log(ν/ρ)` systematically negative. After `max(0, kl)` clipping → always 0.

**Cause (part 2 — high dimensionality):** Even with independent query/mixture sets, at d=256 and n≈1250 the k-NN density estimator is unreliable (n/d ratio only ~5×).

**Fix:** (1) Split each feature set in half — one half for query, other half for mixture. (2) Apply PCA to 32 dimensions before the k-NN step (n/d ratio → ~39×).

---

*Last updated: 2026-04-12. All 198 runs complete. All outputs pushed to [Saranath07/PrivInAI-Project-2026](https://github.com/Saranath07/PrivInAI-Project-2026).*
