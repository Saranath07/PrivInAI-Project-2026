# Image Track — Full Implementation Plan

> **Purpose**: Hand this to Sonnet (or any coding agent) to implement the entire image track of DP-ShiftBench end-to-end.
> **Repo root**: `/Users/saranathp/PrivInAI-Project-2026/`
> **Current state**: Skeleton only — all `__init__.py` files are empty, no implementation code exists.
> **GPU note**: Design everything to be **debuggable on CPU** (small subset flags, reduced epochs). The real runs will happen on GPU later.

---

## Table of Contents

1. [Overview & Goal](#1-overview--goal)
2. [Step 1: Dataset Loaders (`image_shifts.py`)](#step-1-dataset-loaders)
3. [Step 2: Model Definitions (`image_models.py`)](#step-2-model-definitions)
4. [Step 3: Standard Pretraining Loop (`standard_trainer.py`)](#step-3-standard-pretraining-loop)
5. [Step 4: DP-SGD Finetuning Loop (`dp_trainer.py`)](#step-4-dp-sgd-finetuning-loop)
6. [Step 5: Training Utilities (`training/utils.py`)](#step-5-training-utilities)
7. [Step 6: Divergence Metrics (`divergences.py`)](#step-6-divergence-metrics)
8. [Step 7: Evaluation Metrics (`evaluation.py`)](#step-7-evaluation-metrics)
9. [Step 8: Config Files](#step-8-config-files)
10. [Step 9: Experiment Runners](#step-9-experiment-runners)
11. [Step 10: Run All Image Experiments](#step-10-run-all-image-experiments)
12. [Step 11: Analysis & Plotting](#step-11-analysis--plotting)
13. [Step 12: Tests](#step-12-tests)
14. [Appendix: File Map](#appendix-file-map)

---

## 1. Overview & Goal

We study how distribution shift between public pretrain data (SVHN variants) and private finetune data (MNIST) affects the DP-SGD privacy-utility tradeoff.

**The image track runs 11 experiments (IMG-00 through IMG-10):**

| ID | Pretrain Data | Pretrain Method | Finetune Data (DP) | What It Tests | Priority |
|---|---|---|---|---|---|
| `IMG-00` | None (random init) | — | MNIST full | Baseline: no pretrain | P0 |
| `IMG-01` | SVHN full | Supervised | MNIST full | Full domain gap | P0 |
| `IMG-02` | SVHN {0-4} | Supervised | MNIST {0-4} | Matched classes (pure domain gap) | P0 |
| `IMG-03` | SVHN {5-9} | Supervised | MNIST {0-4} | Mismatched classes (domain + label shift) | P0 |
| `IMG-04` | SVHN full | Autoencoder | MNIST full | Unsupervised pretrain (task gap) | P1 |
| `IMG-05` | SVHN 10% | Supervised | MNIST full | Data scarcity | P1 |
| `IMG-06` | SVHN 25% | Supervised | MNIST full | Data scarcity | P1 |
| `IMG-07` | SVHN 50% | Supervised | MNIST full | Data scarcity | P1 |
| `IMG-08` | CIFAR-10 | Supervised | MNIST full | Larger domain gap | P1 |
| `IMG-09` | SVHN + augmentation | Supervised | MNIST full | Style shift | P2 |
| `IMG-10` | FashionMNIST | Supervised | MNIST full | Same format, different semantics | P2 |

**For each experiment, DP finetuning is run at:**
- epsilon in {0.5, 1.0, 2.0, 4.0, 8.0, inf}
- delta = 1e-5
- 3 random seeds per (experiment, epsilon) pair

**After experiments, compute divergence metrics + full analysis.**

---

## Step 1: Dataset Loaders

**File**: `dp_shift_bench/datasets/image_shifts.py`

### What to build

A module that provides PyTorch `DataLoader`s for every (pretrain, finetune) pair in the image track. All images must be normalized to a **common format**: 32x32, 3-channel, values in [0, 1].

### Detailed specification

```python
"""
dp_shift_bench/datasets/image_shifts.py

Dataset loaders for the image track of DP-ShiftBench.
All datasets are normalized to: 32x32, 3-channel RGB, [0, 1] range.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional, List
import numpy as np
```

**Core transforms to define:**

```python
# SVHN is already 32x32 RGB — just normalize to [0,1]
svhn_transform = transforms.Compose([
    transforms.ToTensor(),  # scales to [0,1] automatically
])

# MNIST is 28x28 grayscale — resize to 32x32 and replicate to 3 channels
mnist_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),  # replicate to 3 channels
    transforms.ToTensor(),
])

# CIFAR-10 is already 32x32 RGB
cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
])

# FashionMNIST is 28x28 grayscale — same as MNIST
fashionmnist_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
```

**Functions to implement:**

#### 1. `get_class_subset(dataset, classes: List[int]) -> Subset`
- Filter a dataset to only include samples whose label is in `classes`
- Remap labels to 0..len(classes)-1 (important! The model output dim depends on this)
- Return a wrapper `Dataset` that remaps the labels, not just a raw `Subset`

```python
class ClassSubset(Dataset):
    """Wraps a dataset, filtering to specific classes and remapping labels."""
    def __init__(self, dataset, classes: List[int]):
        self.classes = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        # Find indices where target is in classes
        # NOTE: SVHN uses .labels, MNIST uses .targets — handle both
        if hasattr(dataset, 'labels'):
            targets = np.array(dataset.labels)
        elif hasattr(dataset, 'targets'):
            targets = np.array(dataset.targets)
        else:
            raise ValueError("Dataset has neither .labels nor .targets")
        self.indices = [i for i, t in enumerate(targets) if t in self.classes]
        self.dataset = dataset

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, self.class_to_idx[label]

    def __len__(self):
        return len(self.indices)
```

#### 2. `get_random_subset(dataset, fraction: float, seed: int = 42) -> Subset`
- Randomly sample `fraction` of the dataset
- Use the seed for reproducibility

#### 3. `get_augmented_dataset(dataset) -> Dataset`
- Wrap SVHN with additional augmentation: `ColorJitter(brightness=0.3, contrast=0.3)`, `GaussianBlur(kernel_size=3)`
- Applied ON TOP of the standard svhn_transform

#### 4. Main loader function — `get_image_experiment_data(...)`

```python
def get_image_experiment_data(
    experiment_id: str,  # "IMG-00", "IMG-01", ..., "IMG-10"
    data_root: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
    seed: int = 42,
    debug: bool = False,  # if True, use tiny subsets for CPU testing
) -> dict:
    """
    Returns a dict with keys:
    {
        "pretrain_loader": DataLoader or None,  # None for IMG-00
        "finetune_train_loader": DataLoader,
        "finetune_test_loader": DataLoader,
        "num_pretrain_classes": int,
        "num_finetune_classes": int,
        "experiment_id": str,
        "pretrain_dataset": Dataset or None,  # raw dataset (for divergence computation)
        "finetune_train_dataset": Dataset,      # raw dataset (for divergence computation)
    }
    """
```

**Experiment-specific logic:**

| Experiment | Pretrain | Finetune train | Finetune test | num_pretrain_classes | num_finetune_classes |
|---|---|---|---|---|---|
| IMG-00 | None | MNIST train full | MNIST test full | 0 | 10 |
| IMG-01 | SVHN train full | MNIST train full | MNIST test full | 10 | 10 |
| IMG-02 | SVHN train classes {0,1,2,3,4} | MNIST train classes {0,1,2,3,4} | MNIST test classes {0,1,2,3,4} | 5 | 5 |
| IMG-03 | SVHN train classes {5,6,7,8,9} | MNIST train classes {0,1,2,3,4} | MNIST test classes {0,1,2,3,4} | 5 | 5 |
| IMG-04 | SVHN train full (autoencoder — labels not used) | MNIST train full | MNIST test full | 10 | 10 |
| IMG-05 | SVHN train 10% random subset | MNIST train full | MNIST test full | 10 | 10 |
| IMG-06 | SVHN train 25% random subset | MNIST train full | MNIST test full | 10 | 10 |
| IMG-07 | SVHN train 50% random subset | MNIST train full | MNIST test full | 10 | 10 |
| IMG-08 | CIFAR-10 train full | MNIST train full | MNIST test full | 10 | 10 |
| IMG-09 | SVHN train full + augmentation | MNIST train full | MNIST test full | 10 | 10 |
| IMG-10 | FashionMNIST train full | MNIST train full | MNIST test full | 10 | 10 |

**Debug mode:** When `debug=True`, take only the first 500 samples from each dataset. This allows CPU testing.

**IMPORTANT pitfalls to handle:**
- SVHN's `split` parameter is `"train"` / `"test"` / `"extra"` — use `"train"` for pretraining
- SVHN stores labels as `.labels` (numpy array), MNIST stores as `.targets` (tensor) — handle both in `ClassSubset`
- SVHN `download=True` may take a while the first time — that's fine
- For IMG-03, the pretrain classes are {5,6,7,8,9} but the finetune classes are {0,1,2,3,4}. The pretrain model has 5 output classes, the finetune model has 5 output classes, but they are DIFFERENT classes. The pretrained weights for the classifier head will NOT transfer — only the feature extractor transfers. Make sure this is clear in the returned dict.

---

## Step 2: Model Definitions

**File**: `dp_shift_bench/models/image_models.py`

### What to build

Two model architectures, both compatible with Opacus DP-SGD (i.e., **NO BatchNorm** — use GroupNorm instead).

```python
"""
dp_shift_bench/models/image_models.py

Image models for DP-ShiftBench. All models use GroupNorm (not BatchNorm)
to be compatible with Opacus DP-SGD.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
```

### Architecture 1: SmallCNN (~500K params) — PRIMARY MODEL

```
Input: 3x32x32
Conv2d(3, 32, 3, padding=1) -> GroupNorm(8, 32) -> ReLU -> MaxPool2d(2)   -> 32x16x16
Conv2d(32, 64, 3, padding=1) -> GroupNorm(8, 64) -> ReLU -> MaxPool2d(2)  -> 64x8x8
Conv2d(64, 128, 3, padding=1) -> GroupNorm(8, 128) -> ReLU -> MaxPool2d(2) -> 128x4x4
Flatten -> 128*4*4 = 2048
Linear(2048, 256) -> ReLU -> Dropout(0.2)
Linear(256, num_classes)
```

Implement as:

```python
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        ...
        # Store num_classes as attribute — needed for head replacement during finetuning
        self.num_classes = num_classes
        self.features = nn.Sequential(...)  # conv layers
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.fc(x)

    def get_features(self, x):
        """Extract features before the final classifier — used for divergence metrics."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x  # 256-dim feature vector
```

### Architecture 2: ResNet-18 with GroupNorm

```python
def resnet18_groupnorm(num_classes: int = 10) -> nn.Module:
    """ResNet-18 with all BatchNorm replaced by GroupNorm(num_groups=8)."""
    model = resnet18(weights=None)
    # Replace all BatchNorm layers with GroupNorm
    # Walk the model recursively and swap nn.BatchNorm2d -> nn.GroupNorm
    # GroupNorm groups = min(8, num_channels) as a safe default
    # Replace final fc: model.fc = nn.Linear(512, num_classes)
    # Also modify first conv: model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool (not needed for 32x32 inputs): model.maxpool = nn.Identity()
    return model
```

**Important**: Write a helper `replace_bn_with_gn(model)` that recursively replaces all `nn.BatchNorm2d` with `nn.GroupNorm`. This is a well-known pattern for Opacus compatibility.

### Architecture 3: Autoencoder (for IMG-04)

```python
class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for unsupervised pretraining on SVHN.
    The encoder portion has the SAME architecture as SmallCNN's feature extractor.
    This way, after pretraining, we can transfer encoder weights to SmallCNN.
    """
    def __init__(self):
        self.encoder = nn.Sequential(
            # Same as SmallCNN.features
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(), nn.MaxPool2d(2),
        )  # Output: 128x4x4
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # -> 64x8x8
            nn.GroupNorm(8, 64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # -> 32x16x16
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # -> 3x32x32
            nn.Sigmoid(),  # output in [0,1] since inputs are [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_encoder_state_dict(self):
        """Returns state dict compatible with SmallCNN.features"""
        return self.encoder.state_dict()
```

### Helper: `create_model(...)`

```python
def create_model(
    arch: str = "smallcnn",          # "smallcnn" or "resnet18"
    num_classes: int = 10,
    pretrained_path: Optional[str] = None,  # path to pretrained weights
    pretrain_num_classes: Optional[int] = None,  # num classes in pretrained model (for head mismatch)
    autoencoder_pretrained_path: Optional[str] = None,  # path to autoencoder weights (IMG-04)
) -> nn.Module:
    """
    Factory function. Handles:
    1. Creating the model
    2. Loading pretrained weights if provided
    3. Replacing the classification head if pretrain_num_classes != num_classes
    4. Loading autoencoder encoder weights into the feature extractor (IMG-04)
    """
```

**Head replacement logic** (critical for IMG-03 and any class mismatch):
- Load the full pretrained state dict
- If `pretrain_num_classes != num_classes`, replace `model.fc` with a new `nn.Linear(..., num_classes)` (random init)
- The feature extractor weights are kept, only the head is replaced

---

## Step 3: Standard Pretraining Loop

**File**: `dp_shift_bench/training/standard_trainer.py`

### What to build

A clean, non-private training loop for pretraining on SVHN/CIFAR/FashionMNIST. Also handles autoencoder pretraining.

```python
def pretrain_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 20,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    save_path: Optional[str] = None,  # where to save the final checkpoint
    verbose: bool = True,
) -> dict:
    """
    Standard supervised pretraining with SGD + cosine annealing LR.

    Returns:
        {
            "train_losses": List[float],   # per-epoch
            "train_accs": List[float],      # per-epoch
            "final_train_acc": float,
            "model_path": str,              # path to saved checkpoint
        }
    """
    # Use CrossEntropyLoss
    # Optimizer: SGD with momentum
    # Scheduler: CosineAnnealingLR(optimizer, T_max=epochs)
    # Save checkpoint at the end: torch.save(model.state_dict(), save_path)
```

```python
def pretrain_autoencoder(
    model: ConvAutoencoder,
    train_loader: DataLoader,
    epochs: int = 30,
    lr: float = 0.001,
    device: str = "cuda",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Autoencoder pretraining with MSE reconstruction loss.

    Returns:
        {
            "train_losses": List[float],
            "model_path": str,
        }
    """
    # Use MSELoss
    # Optimizer: Adam(lr=0.001)
    # Ignore labels — only use images: for batch in loader: imgs, _ = batch
    # Save full autoencoder state dict at the end
```

---

## Step 4: DP-SGD Finetuning Loop

**File**: `dp_shift_bench/training/dp_trainer.py`

### What to build

The core DP-SGD finetuning loop using Opacus. This is the most critical file.

```python
"""
dp_shift_bench/training/dp_trainer.py

DP-SGD finetuning using Opacus.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from typing import Optional
```

```python
def finetune_dp(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epsilon: float,             # target epsilon. Use float('inf') for non-private baseline
    delta: float = 1e-5,
    epochs: int = 15,
    lr: float = 0.001,
    max_grad_norm: float = 1.0,  # clipping norm C
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Finetune a model with DP-SGD (Opacus) or without DP if epsilon=inf.

    Returns:
        {
            "train_losses": List[float],
            "train_accs": List[float],
            "test_accs": List[float],       # per-epoch
            "final_test_acc": float,
            "epsilon_actual": float,         # actual epsilon spent (from Opacus accountant)
            "best_test_acc": float,
            "epochs_run": int,
        }
    """
```

**CRITICAL implementation details for Opacus:**

1. **Validate model first:**
```python
model = ModuleValidator.fix(model)  # auto-fixes common issues
errors = ModuleValidator.validate(model, strict=False)
if errors:
    raise ValueError(f"Model not compatible with Opacus: {errors}")
```

2. **Handle epsilon = inf (non-private) separately:**
```python
if epsilon == float('inf'):
    # Train normally without Opacus — just standard SGD, no noise, no clipping
    # This is the non-private baseline
    return _train_standard(model, train_loader, test_loader, epochs, lr, device, verbose)
```

3. **Opacus setup for private training:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=epochs,
    target_epsilon=epsilon,
    target_delta=delta,
    max_grad_norm=max_grad_norm,
)
```

4. **Training loop:**
```python
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

    # Evaluate on test set each epoch
    test_acc = evaluate(model, test_loader, device)

    # Get actual epsilon spent so far
    if privacy_engine is not None:
        eps = privacy_engine.get_epsilon(delta=delta)
```

5. **IMPORTANT Opacus gotcha**: Opacus wraps the model in `GradSampleModule`. When saving/returning the model, use `model._module` to get the original unwrapped model. Also, Opacus modifies the DataLoader — the batch size may change slightly due to Poisson sampling.

6. **Batch size note**: Opacus works best with `batch_size=256`. The DataLoader passed in should already have this batch size. Opacus will wrap it with Poisson sampling internally.

---

## Step 5: Training Utilities

**File**: `dp_shift_bench/training/utils.py`

```python
def set_seed(seed: int):
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

def save_results(results: dict, path: str):
    """Save experiment results as JSON."""
    import json
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_results(path: str) -> dict:
    """Load experiment results from JSON."""
    import json
    with open(path, 'r') as f:
        return json.load(f)
```

---

## Step 6: Divergence Metrics

**File**: `dp_shift_bench/metrics/divergences.py`

### What to build

Compute distribution divergence between pretrain and finetune datasets. For images, we work in **feature space** (not raw pixel space).

```python
"""
dp_shift_bench/metrics/divergences.py

Distribution divergence metrics between datasets.
For images: operates on CNN features extracted from the datasets.
"""

import torch
import numpy as np
from scipy import stats
from typing import Optional, Tuple
from torch.utils.data import DataLoader
```

### Feature extraction helper

```python
def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    max_samples: int = 10000,
) -> np.ndarray:
    """
    Extract features from a model's penultimate layer for all samples in the dataloader.
    Uses model.get_features(x) if available, otherwise hooks into the penultimate layer.

    Returns: np.ndarray of shape (N, feature_dim)
    """
```

### Metrics to implement

#### 1. FID (Frechet Inception Distance) — P0

```python
def compute_fid(
    dataset1: DataLoader,
    dataset2: DataLoader,
    device: str = "cuda",
) -> float:
    """
    Compute FID between two image datasets.
    Use the `clean-fid` library:

        from cleanfid import fid

    BUT clean-fid expects directory paths of images, not DataLoaders.
    Two options:
    Option A (recommended): Save a temp directory of images from each DataLoader,
              then call fid.compute_fid(dir1, dir2)
    Option B: Compute FID manually:
              - Extract InceptionV3 features for both datasets
              - Compute mean and covariance for each
              - FID = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*(C1*C2)^(1/2))
              Use scipy.linalg.sqrtm for the matrix square root.

    Go with Option B for more control and to avoid filesystem overhead.
    Use torchvision's inception_v3(pretrained=True) and extract from the
    penultimate layer (before final fc). The feature dim is 2048.
    """
```

#### 2. MMD (Maximum Mean Discrepancy) — P1

```python
def compute_mmd(
    features1: np.ndarray,
    features2: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,  # if None, use median heuristic
) -> float:
    """
    Compute MMD^2 between two sets of features.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    where k is the RBF kernel: k(x,y) = exp(-gamma * ||x-y||^2)

    If gamma is None, use the median heuristic:
        gamma = 1 / (2 * median(||x_i - x_j||^2))

    Subsample to max 5000 points from each set to keep computation tractable.
    """
```

#### 3. Proxy A-distance (H-divergence) — P1

```python
def compute_proxy_a_distance(
    features1: np.ndarray,  # pretrain features
    features2: np.ndarray,  # finetune features
    max_samples: int = 5000,
) -> float:
    """
    Train a linear domain classifier to distinguish pretrain from finetune features.
    PAD = 2 * (1 - 2 * classification_error)

    Steps:
    1. Label features1 as 0, features2 as 1
    2. Combine and shuffle
    3. Train sklearn LogisticRegression (or linear SVM) with 5-fold CV
    4. error = 1 - mean_cv_accuracy
    5. PAD = 2 * (1 - 2 * error)

    PAD ranges from 0 (identical distributions) to 2 (perfectly distinguishable).
    """
```

#### 4. KL Divergence (estimated) — P0

```python
def compute_kl_divergence_estimated(
    features1: np.ndarray,
    features2: np.ndarray,
    k: int = 5,
) -> float:
    """
    KL divergence estimated via k-nearest-neighbor density estimation.
    Uses the estimator from:
        Perez-Cruz (2008), "Kullback-Leibler Divergence Estimation of Continuous Distributions"

    Or simpler: use sklearn's KernelDensity to fit density on each,
    then compute KL = mean(log p1(x) - log p2(x)) for x ~ p1

    This is an ESTIMATE, not exact (exact KL is only for Markov chains).
    """
```

#### 5. Jensen-Shannon Divergence — P0

```python
def compute_jsd(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)

    For continuous features, discretize into histogram bins or use
    the KNN-based estimate. Alternatively:
    - Fit KDE to both
    - Sample from the mixture
    - Compute the two KL terms

    Or use scipy.spatial.distance.jensenshannon on histogrammed features
    (bin each dimension independently, then flatten).
    """
```

#### 6. Total Variation — P0

```python
def compute_total_variation(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    TV = 0.5 * sum |P(x) - Q(x)|

    For continuous features, discretize with histograms.
    Use per-dimension histograms and average TV across dimensions.
    """
```

#### 7. Wasserstein Distance — P1

```python
def compute_wasserstein(
    features1: np.ndarray,
    features2: np.ndarray,
    max_samples: int = 2000,
) -> float:
    """
    Compute Wasserstein-1 distance using the POT library.

    import ot
    M = ot.dist(features1[:max_samples], features2[:max_samples])
    a = np.ones(len(features1_sub)) / len(features1_sub)
    b = np.ones(len(features2_sub)) / len(features2_sub)
    return ot.emd2(a, b, M)

    Subsample aggressively — Wasserstein is O(n^3).
    """
```

### Master function

```python
def compute_all_divergences(
    pretrain_loader: DataLoader,
    finetune_loader: DataLoader,
    feature_model: nn.Module,  # model to extract features (e.g. pretrained SmallCNN)
    device: str = "cuda",
    priority: str = "P0",  # "P0" = only P0 metrics, "P1" = P0+P1, "all" = everything
) -> dict:
    """
    Compute all divergence metrics between pretrain and finetune datasets.

    Returns:
        {
            "fid": float,
            "kl_estimated": float,
            "jsd": float,
            "total_variation": float,
            "mmd_rbf": float,           # P1
            "proxy_a_distance": float,   # P1
            "wasserstein": float,        # P1
        }
    """
    # 1. Extract features from both datasets using feature_model
    features_pretrain = extract_features(feature_model, pretrain_loader, device)
    features_finetune = extract_features(feature_model, finetune_loader, device)

    results = {}

    # P0 metrics
    results["fid"] = compute_fid(pretrain_loader, finetune_loader, device)
    results["kl_estimated"] = compute_kl_divergence_estimated(features_pretrain, features_finetune)
    results["jsd"] = compute_jsd(features_pretrain, features_finetune)
    results["total_variation"] = compute_total_variation(features_pretrain, features_finetune)

    if priority in ("P1", "all"):
        results["mmd_rbf"] = compute_mmd(features_pretrain, features_finetune)
        results["proxy_a_distance"] = compute_proxy_a_distance(features_pretrain, features_finetune)
        results["wasserstein"] = compute_wasserstein(features_pretrain, features_finetune)

    return results
```

---

## Step 7: Evaluation Metrics

**File**: `dp_shift_bench/metrics/evaluation.py`

```python
def evaluate_accuracy(model, dataloader, device="cuda") -> float:
    """Standard top-1 accuracy on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

def evaluate_per_class_accuracy(model, dataloader, num_classes, device="cuda") -> dict:
    """Per-class accuracy breakdown."""
    # Returns {"class_0": 0.95, "class_1": 0.87, ...}

def evaluate_loss(model, dataloader, device="cuda") -> float:
    """Average cross-entropy loss."""
```

---

## Step 8: Config Files

### `dp_shift_bench/configs/default.yaml`

```yaml
# Default configuration for image track experiments

# Model
model:
  arch: "smallcnn"    # "smallcnn" or "resnet18"

# Pretraining
pretrain:
  epochs: 20
  lr: 0.01
  momentum: 0.9
  weight_decay: 1e-4
  batch_size: 256

# Autoencoder pretraining (IMG-04)
autoencoder:
  epochs: 30
  lr: 0.001
  batch_size: 256

# DP Finetuning
finetune:
  epochs: 15
  lr: 0.001
  max_grad_norm: 1.0
  batch_size: 256
  delta: 1e-5

# Epsilon grid
epsilons: [0.5, 1.0, 2.0, 4.0, 8.0, .inf]

# Reproducibility
seeds: [42, 123, 456]

# Data
data_root: "./data"
results_root: "./results/image"
checkpoints_root: "./checkpoints/image"

# Debug mode (for CPU testing)
debug: false

# Divergence metrics
divergences:
  priority: "P1"     # "P0", "P1", or "all"
```

### `dp_shift_bench/configs/image_experiments.yaml`

```yaml
# All image experiments

experiments:
  IMG-00:
    pretrain_data: null
    pretrain_method: null
    finetune_data: "mnist_full"
    description: "Baseline: no pretraining"
    priority: "P0"

  IMG-01:
    pretrain_data: "svhn_full"
    pretrain_method: "supervised"
    finetune_data: "mnist_full"
    description: "Full domain gap"
    priority: "P0"

  IMG-02:
    pretrain_data: "svhn_0to4"
    pretrain_method: "supervised"
    finetune_data: "mnist_0to4"
    description: "Matched class subset"
    priority: "P0"

  IMG-03:
    pretrain_data: "svhn_5to9"
    pretrain_method: "supervised"
    finetune_data: "mnist_0to4"
    description: "Mismatched class subset"
    priority: "P0"

  IMG-04:
    pretrain_data: "svhn_full"
    pretrain_method: "autoencoder"
    finetune_data: "mnist_full"
    description: "Unsupervised pretrain (task gap)"
    priority: "P1"

  IMG-05:
    pretrain_data: "svhn_10pct"
    pretrain_method: "supervised"
    finetune_data: "mnist_full"
    description: "Data scarcity 10%"
    priority: "P1"

  IMG-06:
    pretrain_data: "svhn_25pct"
    pretrain_method: "supervised"
    finetune_data: "mnist_full"
    description: "Data scarcity 25%"
    priority: "P1"

  IMG-07:
    pretrain_data: "svhn_50pct"
    pretrain_method: "supervised"
    finetune_data: "mnist_full"
    description: "Data scarcity 50%"
    priority: "P1"

  IMG-08:
    pretrain_data: "cifar10_full"
    pretrain_method: "supervised"
    finetune_data: "mnist_full"
    description: "Larger domain gap (CIFAR-10)"
    priority: "P1"

  IMG-09:
    pretrain_data: "svhn_augmented"
    pretrain_method: "supervised"
    finetune_data: "mnist_full"
    description: "Controlled style shift"
    priority: "P2"

  IMG-10:
    pretrain_data: "fashionmnist_full"
    pretrain_method: "supervised"
    finetune_data: "mnist_full"
    description: "Same format, different semantics"
    priority: "P2"
```

---

## Step 9: Experiment Runners

### `dp_shift_bench/runners/run_single.py`

The single-experiment runner. This is the core orchestrator.

```python
"""
Run a single image experiment: pretrain (optional) -> DP-finetune -> evaluate -> save.
"""

import argparse
import os
import json
import torch
from dp_shift_bench.datasets.image_shifts import get_image_experiment_data
from dp_shift_bench.models.image_models import create_model, SmallCNN, ConvAutoencoder
from dp_shift_bench.training.standard_trainer import pretrain_supervised, pretrain_autoencoder
from dp_shift_bench.training.dp_trainer import finetune_dp
from dp_shift_bench.metrics.evaluation import evaluate_accuracy
from dp_shift_bench.training.utils import set_seed, get_device, save_results


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
    2. Load data for this experiment
    3. If experiment needs pretraining:
       a. Check if pretrained checkpoint already exists (skip if so)
       b. Pretrain on SVHN/CIFAR/FashionMNIST
       c. Save pretrained checkpoint
    4. Create finetuning model (load pretrained weights if available)
    5. Finetune with DP-SGD at given epsilon
    6. Evaluate on test set
    7. Save all results to JSON

    Returns the results dict.
    """

    set_seed(seed)
    if device is None:
        device = get_device()

    # --- 1. Load data ---
    data = get_image_experiment_data(
        experiment_id=experiment_id,
        data_root=data_root,
        batch_size=batch_size,
        seed=seed,
        debug=debug,
    )

    # --- 2. Pretraining (if needed) ---
    pretrain_path = None
    if data["pretrain_loader"] is not None:
        pretrain_dir = os.path.join(checkpoints_root, experiment_id)
        os.makedirs(pretrain_dir, exist_ok=True)

        if experiment_id == "IMG-04":
            # Autoencoder pretraining
            pretrain_path = os.path.join(pretrain_dir, "autoencoder.pt")
            if not os.path.exists(pretrain_path):
                ae_model = ConvAutoencoder().to(device)
                pretrain_autoencoder(
                    ae_model, data["pretrain_loader"],
                    epochs=pretrain_epochs, device=device,
                    save_path=pretrain_path,
                )
        else:
            # Supervised pretraining
            pretrain_path = os.path.join(pretrain_dir, "pretrained.pt")
            if not os.path.exists(pretrain_path):
                pretrain_model = SmallCNN(
                    num_classes=data["num_pretrain_classes"]
                ).to(device)
                pretrain_supervised(
                    pretrain_model, data["pretrain_loader"],
                    epochs=pretrain_epochs, device=device,
                    save_path=pretrain_path,
                )

    # --- 3. Create finetuning model ---
    if experiment_id == "IMG-04":
        model = create_model(
            arch=arch,
            num_classes=data["num_finetune_classes"],
            autoencoder_pretrained_path=pretrain_path,
        )
    elif pretrain_path is not None:
        model = create_model(
            arch=arch,
            num_classes=data["num_finetune_classes"],
            pretrained_path=pretrain_path,
            pretrain_num_classes=data["num_pretrain_classes"],
        )
    else:
        model = create_model(
            arch=arch,
            num_classes=data["num_finetune_classes"],
        )
    model = model.to(device)

    # --- 4. DP Finetuning ---
    ft_results = finetune_dp(
        model=model,
        train_loader=data["finetune_train_loader"],
        test_loader=data["finetune_test_loader"],
        epsilon=epsilon,
        epochs=finetune_epochs,
        max_grad_norm=max_grad_norm,
        device=device,
        seed=seed,
    )

    # --- 5. Save results ---
    result = {
        "experiment_id": experiment_id,
        "epsilon": epsilon,
        "seed": seed,
        "arch": arch,
        "debug": debug,
        "final_test_acc": ft_results["final_test_acc"],
        "best_test_acc": ft_results["best_test_acc"],
        "epsilon_actual": ft_results["epsilon_actual"],
        "train_losses": ft_results["train_losses"],
        "test_accs": ft_results["test_accs"],
    }

    results_dir = os.path.join(results_root, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    eps_str = f"eps_{epsilon}" if epsilon != float('inf') else "eps_inf"
    result_path = os.path.join(results_dir, f"{eps_str}_seed_{seed}.json")
    save_results(result, result_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="IMG-00, IMG-01, ...")
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", type=str, default="smallcnn")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--results-root", type=str, default="./results/image")
    args = parser.parse_args()

    eps = float('inf') if args.epsilon < 0 else args.epsilon
    run_single_experiment(
        experiment_id=args.experiment,
        epsilon=eps,
        seed=args.seed,
        arch=args.arch,
        debug=args.debug,
        data_root=args.data_root,
        results_root=args.results_root,
    )
```

### `dp_shift_bench/runners/run_sweep.py`

```python
"""
Run the full image experiment grid:
  experiments x epsilons x seeds
"""

def run_image_sweep(
    experiments: List[str] = None,   # None = all experiments
    epsilons: List[float] = None,    # None = default grid
    seeds: List[int] = None,         # None = [42, 123, 456]
    priority: str = "P0",            # only run experiments at this priority or higher
    **kwargs,
) -> None:
    """
    Outer loop: for each experiment, for each epsilon, for each seed:
        run_single_experiment(...)

    Filter experiments by priority if specified.
    Skip experiments that already have saved results (resume-friendly).
    Print progress: "Running IMG-01 | eps=1.0 | seed=42 | [3/126]"
    """
```

**Usage:**
```bash
# Run all P0 experiments in debug mode (CPU)
python -m dp_shift_bench.runners.run_sweep --priority P0 --debug

# Run everything on GPU
python -m dp_shift_bench.runners.run_sweep --priority all

# Run a single experiment
python -m dp_shift_bench.runners.run_single --experiment IMG-01 --epsilon 1.0 --seed 42
```

### `dp_shift_bench/runners/run_divergences.py`

```python
"""
Compute divergence metrics for all pretrain/finetune pairs.
Run this AFTER pretraining is complete (needs the pretrained feature models).
"""

def compute_image_divergences(
    experiments: List[str] = None,
    priority: str = "P0",
    data_root: str = "./data",
    checkpoints_root: str = "./checkpoints/image",
    results_root: str = "./results/image",
    device: str = "cuda",
    debug: bool = False,
) -> dict:
    """
    For each experiment (except IMG-00 which has no pretrain):
    1. Load the pretrain and finetune datasets
    2. Load a feature extractor (the pretrained model for that experiment,
       or a freshly trained SmallCNN on SVHN as a shared feature extractor)
    3. Compute all divergence metrics
    4. Save to results_root/divergences/{experiment_id}.json

    Returns: dict mapping experiment_id -> divergence dict
    """
```

---

## Step 10: Run All Image Experiments

This is the execution order. Each step depends on the previous.

### Phase 1: Pretraining (run once, reuse across epsilons/seeds)

```bash
# This happens automatically inside run_single — checkpoints are cached.
# But you can also pretrain explicitly:
python -m dp_shift_bench.runners.run_single --experiment IMG-01 --epsilon inf --seed 42
# The pretrained checkpoint is saved and reused for all epsilon values.
```

### Phase 2: DP Finetuning grid

```bash
# Debug run on CPU first
python -m dp_shift_bench.runners.run_sweep --priority P0 --debug

# Full P0 run on GPU
python -m dp_shift_bench.runners.run_sweep --priority P0

# Then P1
python -m dp_shift_bench.runners.run_sweep --priority P1

# Then P2
python -m dp_shift_bench.runners.run_sweep --priority P2
```

### Phase 3: Divergence computation

```bash
python -m dp_shift_bench.runners.run_divergences --priority P1
```

### Phase 4: Analysis (see next step)

---

## Step 11: Analysis & Plotting

**File**: `experiments/image_analysis.py`

This is the analysis script that produces all figures and tables for the paper.

### 11A. Load and aggregate results

```python
def load_all_image_results(results_root="./results/image") -> pd.DataFrame:
    """
    Walk results_root, load all JSON result files, return a DataFrame with columns:
    [experiment_id, epsilon, seed, final_test_acc, best_test_acc, epsilon_actual]

    Compute mean and std across seeds for each (experiment_id, epsilon) pair.
    """
```

### 11B. Core figures to generate

#### Figure 1: Accuracy vs Epsilon for all experiments (the overview figure)

```python
def plot_accuracy_vs_epsilon(df: pd.DataFrame, save_path: str):
    """
    X-axis: epsilon (log scale)
    Y-axis: test accuracy
    One line per experiment, with error bars (std across 3 seeds).
    Color-code by experiment type.

    This shows at a glance how each shift condition affects DP finetuning.
    """
```

#### Figure 2: Pretrain gain vs epsilon (key insight figure)

```python
def plot_pretrain_gain(df: pd.DataFrame, save_path: str):
    """
    For each experiment, compute:
        gain = acc(pretrained, eps) - acc(IMG-00, eps)  # gain over no-pretrain baseline

    X-axis: epsilon
    Y-axis: gain
    One line per experiment.

    Key question: at which epsilon does pretraining help most? Does it ever hurt?
    """
```

#### Figure 3: Divergence vs Accuracy scatter plots

```python
def plot_divergence_vs_accuracy(
    results_df: pd.DataFrame,
    divergences_df: pd.DataFrame,
    save_path: str,
):
    """
    For each divergence metric, create a scatter plot:
    X-axis: divergence value
    Y-axis: DP finetuning accuracy
    Color: epsilon value
    One subplot per metric (2x3 grid for 6 metrics).

    Compute Spearman correlation for each (metric, epsilon) pair.
    Annotate each subplot with the correlation coefficient.
    """
```

#### Figure 4: Correlation heatmap

```python
def plot_correlation_heatmap(
    results_df: pd.DataFrame,
    divergences_df: pd.DataFrame,
    save_path: str,
):
    """
    Heatmap: rows = divergence metrics, columns = epsilon values
    Cell value = Spearman correlation between that metric and DP accuracy
                 across all experiments at that epsilon.

    This answers: "Which divergence metric best predicts DP accuracy, and does
    the answer depend on the privacy budget?"
    """
```

#### Figure 5: Phase diagram / heatmap (hero figure)

```python
def plot_phase_diagram(df: pd.DataFrame, save_path: str):
    """
    2D heatmap:
    X-axis: shift level (ordered by divergence: IMG-02 < IMG-01 < IMG-03 < IMG-08, etc.)
    Y-axis: epsilon
    Cell color: test accuracy (or pretrain gain)

    This is the "hero figure" — shows the Goldilocks zone where pretraining helps.
    """
```

#### Figure 6: Hypothesis testing (Contribution 3)

```python
def fit_and_compare_hypotheses(
    results_df: pd.DataFrame,
    divergences_df: pd.DataFrame,
    save_path: str,
):
    """
    Fit three models to the data:

    H_additive:      Error = C1/eps + C2*d + C3
    H_multiplicative: Error = C1*d/eps + C2
    H_threshold:      Error = C1/eps + C2*d  if d < d_thresh
                      Error = C1/eps + C3    if d >= d_thresh

    Where Error = 1 - accuracy, eps = epsilon, d = divergence (use best metric from Fig 4).

    Use scipy.optimize.curve_fit for H_additive and H_multiplicative.
    For H_threshold, sweep d_thresh values and pick the one with best fit.

    Report:
    - Fitted parameters for each model
    - R^2, AIC, BIC for each model
    - Plot: actual vs predicted error for each model (3 subplots)

    Also run a two-way ANOVA:
    - Factor 1: epsilon (categorical)
    - Factor 2: shift level (categorical)
    - Test: is the interaction term significant?
    Use statsmodels: ols('accuracy ~ C(epsilon) * C(shift_level)', data=df).fit()
    then sm.stats.anova_lm(model, typ=2)
    """
```

### 11C. Summary tables

```python
def generate_results_table(df: pd.DataFrame, save_path: str):
    """
    LaTeX table:
    Rows: experiments (IMG-00 to IMG-10)
    Columns: epsilon values
    Cells: mean accuracy +/- std (across seeds)

    Output as both CSV and LaTeX (\begin{tabular}...).
    """

def generate_divergence_table(divergences: dict, save_path: str):
    """
    LaTeX table:
    Rows: experiments
    Columns: divergence metrics (FID, KL, JSD, TV, MMD, PAD, Wasserstein)
    """
```

### 11D. Master analysis runner

```python
def run_full_analysis(
    results_root: str = "./results/image",
    figures_root: str = "./results/figures",
):
    """
    1. Load all results into DataFrame
    2. Load all divergences
    3. Generate all 6+ figures
    4. Generate all tables
    5. Print summary statistics to console
    """

if __name__ == "__main__":
    run_full_analysis()
```

**Usage:**
```bash
python experiments/image_analysis.py
# All figures saved to results/figures/
```

---

## Step 12: Tests

### `tests/test_datasets.py`

```python
"""Test dataset loaders produce correct shapes, labels, and splits."""

def test_svhn_mnist_shapes():
    """All images should be 3x32x32 and in [0,1]."""
    data = get_image_experiment_data("IMG-01", debug=True)
    batch, labels = next(iter(data["finetune_train_loader"]))
    assert batch.shape[1:] == (3, 32, 32)
    assert batch.min() >= 0 and batch.max() <= 1

def test_class_subset_filtering():
    """IMG-02 should only have classes 0-4."""
    data = get_image_experiment_data("IMG-02", debug=True)
    for batch, labels in data["finetune_train_loader"]:
        assert labels.max() <= 4
        break
    assert data["num_finetune_classes"] == 5

def test_class_subset_mismatch():
    """IMG-03: pretrain has classes 5-9, finetune has 0-4."""
    data = get_image_experiment_data("IMG-03", debug=True)
    assert data["num_pretrain_classes"] == 5
    assert data["num_finetune_classes"] == 5

def test_img00_no_pretrain():
    """IMG-00 should have no pretrain loader."""
    data = get_image_experiment_data("IMG-00", debug=True)
    assert data["pretrain_loader"] is None

def test_data_scarcity_subset_sizes():
    """IMG-05/06/07 should have roughly 10/25/50% of SVHN."""
    for exp_id, expected_frac in [("IMG-05", 0.10), ("IMG-06", 0.25), ("IMG-07", 0.50)]:
        data = get_image_experiment_data(exp_id, debug=True)
        # In debug mode, these will be subsets of 500, so just check it loads
        assert data["pretrain_loader"] is not None

def test_debug_mode_small():
    """Debug mode should produce small datasets."""
    data = get_image_experiment_data("IMG-01", debug=True)
    total = sum(len(b) for b, _ in data["finetune_train_loader"])
    assert total <= 600  # ~500 samples
```

### `tests/test_models.py`

```python
def test_smallcnn_forward():
    model = SmallCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)

def test_smallcnn_features():
    model = SmallCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    features = model.get_features(x)
    assert features.shape == (4, 256)

def test_resnet18_groupnorm():
    model = resnet18_groupnorm(num_classes=5)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 5)
    # Verify no BatchNorm layers
    for module in model.modules():
        assert not isinstance(module, nn.BatchNorm2d)

def test_autoencoder_reconstruction():
    ae = ConvAutoencoder()
    x = torch.randn(4, 3, 32, 32)
    recon = ae(x)
    assert recon.shape == (4, 3, 32, 32)

def test_autoencoder_encoder_transfer():
    """Autoencoder encoder weights should be loadable into SmallCNN.features."""
    ae = ConvAutoencoder()
    cnn = SmallCNN(num_classes=10)
    cnn.features.load_state_dict(ae.get_encoder_state_dict())

def test_opacus_compatibility():
    """Models should pass Opacus validation."""
    from opacus.validators import ModuleValidator
    model = SmallCNN(num_classes=10)
    errors = ModuleValidator.validate(model, strict=False)
    assert len(errors) == 0
```

### `tests/test_training.py`

```python
def test_dp_finetune_runs():
    """DP finetuning should run without errors on a tiny dataset."""
    model = SmallCNN(num_classes=10)
    # Create tiny dummy DataLoader
    data = get_image_experiment_data("IMG-00", debug=True)
    result = finetune_dp(
        model, data["finetune_train_loader"], data["finetune_test_loader"],
        epsilon=8.0, epochs=2, device="cpu",
    )
    assert "final_test_acc" in result
    assert 0 <= result["final_test_acc"] <= 1

def test_non_private_baseline():
    """epsilon=inf should run without Opacus."""
    model = SmallCNN(num_classes=10)
    data = get_image_experiment_data("IMG-00", debug=True)
    result = finetune_dp(
        model, data["finetune_train_loader"], data["finetune_test_loader"],
        epsilon=float('inf'), epochs=2, device="cpu",
    )
    assert result["epsilon_actual"] == float('inf')
```

### `tests/test_divergences.py`

```python
def test_compute_mmd_identical():
    """MMD between identical distributions should be ~0."""
    x = np.random.randn(500, 64)
    mmd = compute_mmd(x, x)
    assert mmd < 0.01

def test_compute_mmd_different():
    """MMD between different distributions should be > 0."""
    x = np.random.randn(500, 64)
    y = np.random.randn(500, 64) + 2  # shifted
    mmd = compute_mmd(x, y)
    assert mmd > 0.1

def test_proxy_a_distance_range():
    """PAD should be in [0, 2]."""
    x = np.random.randn(500, 64)
    y = np.random.randn(500, 64) + 1
    pad = compute_proxy_a_distance(x, y)
    assert 0 <= pad <= 2
```

---

## Appendix: File Map

Here is every file to create/modify, in implementation order:

```
dp_shift_bench/
    __init__.py                          # Add version string
    datasets/
        __init__.py                      # Export get_image_experiment_data
        image_shifts.py                  # STEP 1 — dataset loaders
    models/
        __init__.py                      # Export SmallCNN, create_model, etc.
        image_models.py                  # STEP 2 — model definitions
    training/
        __init__.py                      # Exports
        standard_trainer.py              # STEP 3 — pretraining loops
        dp_trainer.py                    # STEP 4 — DP-SGD finetuning
        utils.py                         # STEP 5 — seed, device, save/load
    metrics/
        __init__.py                      # Exports
        divergences.py                   # STEP 6 — all divergence metrics
        evaluation.py                    # STEP 7 — accuracy, per-class, loss
    configs/
        default.yaml                     # STEP 8a — default hyperparams
        image_experiments.yaml           # STEP 8b — experiment definitions
    runners/
        __init__.py                      # Exports
        run_single.py                    # STEP 9a — single experiment runner
        run_sweep.py                     # STEP 9b — full grid sweep
        run_divergences.py               # STEP 9c — divergence computation

experiments/
    image_analysis.py                    # STEP 11 — all analysis & plotting

tests/
    test_datasets.py                     # STEP 12a
    test_models.py                       # STEP 12b
    test_training.py                     # STEP 12c
    test_divergences.py                  # STEP 12d
```

**Total: 18 Python files + 2 YAML configs to create.**

### Implementation order (dependencies):

```
STEP 5 (utils.py)           — no dependencies
STEP 1 (image_shifts.py)    — no dependencies
STEP 2 (image_models.py)    — no dependencies
STEP 7 (evaluation.py)      — needs models
STEP 3 (standard_trainer.py)— needs models, utils
STEP 4 (dp_trainer.py)      — needs models, utils, evaluation
STEP 6 (divergences.py)     — needs models (feature extraction)
STEP 8 (configs)             — no dependencies
STEP 9 (runners)             — needs everything above
STEP 12 (tests)              — needs everything above
STEP 11 (analysis)           — needs results from running experiments
```

### Quick validation commands:

```bash
# After Steps 1-5: verify data loading and model creation
python -c "from dp_shift_bench.datasets.image_shifts import get_image_experiment_data; d = get_image_experiment_data('IMG-01', debug=True); print('OK', len(d['finetune_train_dataset']))"

# After Steps 1-9: run one debug experiment
python -m dp_shift_bench.runners.run_single --experiment IMG-00 --epsilon 8.0 --debug

# Run all tests
pytest tests/ -v

# Full P0 debug sweep
python -m dp_shift_bench.runners.run_sweep --priority P0 --debug
```
