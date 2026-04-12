"""
dp_shift_bench/models/image_models.py

Image models for DP-ShiftBench.
All models use GroupNorm (not BatchNorm) to be compatible with Opacus DP-SGD.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# Architecture 1: SmallCNN  (~500 K params) — PRIMARY MODEL
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """
    3-layer CNN with GroupNorm.  Input: 3×32×32.  Output: (N, num_classes).

    Feature extractor: 3 conv blocks → 128×4×4 = 2048-dim
    Classifier head  : Linear(2048, 256) → ReLU → Dropout → Linear(256, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Block 1: 3×32×32 → 32×16×16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 32×16×16 → 64×8×8
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3: 64×8×8 → 128×4×4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.fc(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-dim feature vector (before final linear)."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Architecture 2: ResNet-18 with GroupNorm
# ---------------------------------------------------------------------------

def replace_bn_with_gn(module: nn.Module) -> nn.Module:
    """Recursively replace all BatchNorm2d layers with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # Find largest divisor of num_channels that is ≤ 8
            num_groups = 8
            while num_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            setattr(module, name, nn.GroupNorm(num_groups, num_channels))
        else:
            replace_bn_with_gn(child)
    return module


def resnet18_groupnorm(num_classes: int = 10) -> nn.Module:
    """ResNet-18 with all BatchNorm replaced by GroupNorm, adapted for 32×32 inputs."""
    model = resnet18(weights=None)

    # Adapt for 32×32: smaller first conv, no strided maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace BatchNorm
    replace_bn_with_gn(model)

    # Replace final classifier
    model.fc = nn.Linear(512, num_classes)

    return model


# ---------------------------------------------------------------------------
# Architecture 3: Convolutional Autoencoder (for IMG-04)
# ---------------------------------------------------------------------------

class ConvAutoencoder(nn.Module):
    """
    Conv autoencoder for unsupervised pretraining on SVHN.
    Encoder mirrors SmallCNN.features so weights transfer cleanly.
    """

    def __init__(self):
        super().__init__()

        # Encoder — same topology as SmallCNN.features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),                         # 32×16×16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),                         # 64×8×8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(2),                         # 128×4×4
        )

        # Decoder — mirror of encoder with transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 64×8×8
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 32×16×16
            nn.GroupNorm(8, 32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 3×32×32
            nn.Sigmoid(),                                           # output in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def get_encoder_state_dict(self) -> dict:
        """Return state dict that is directly loadable into SmallCNN.features."""
        return self.encoder.state_dict()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(
    arch: str = "smallcnn",
    num_classes: int = 10,
    pretrained_path: Optional[str] = None,
    pretrain_num_classes: Optional[int] = None,
    autoencoder_pretrained_path: Optional[str] = None,
) -> nn.Module:
    """
    Create an image model, optionally loading pretrained weights.

    Handles:
    1. SmallCNN or ResNet-18 creation
    2. Loading supervised-pretrained weights (full or partial)
    3. Replacing the head when pretrain_num_classes != num_classes
    4. Loading autoencoder encoder weights into SmallCNN.features
    """
    # ---- create base model ----
    if arch == "smallcnn":
        model = SmallCNN(num_classes=num_classes)
    elif arch == "resnet18":
        model = resnet18_groupnorm(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose 'smallcnn' or 'resnet18'.")

    # ---- IMG-04: load autoencoder encoder into feature extractor ----
    if autoencoder_pretrained_path is not None:
        ae_state = torch.load(autoencoder_pretrained_path, map_location="cpu")
        # Strip 'encoder.' prefix to get keys compatible with SmallCNN.features
        encoder_state = {
            k[len("encoder."):]: v
            for k, v in ae_state.items()
            if k.startswith("encoder.")
        }
        if arch == "smallcnn":
            model.features.load_state_dict(encoder_state, strict=True)
        else:
            raise NotImplementedError(
                "Autoencoder weight transfer only supported for 'smallcnn'."
            )
        return model

    # ---- supervised pretrained weights ----
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location="cpu")

        if pretrain_num_classes is not None and pretrain_num_classes != num_classes:
            # Head mismatch: drop fc weights, keep feature extractor
            state_no_head = {k: v for k, v in state.items() if not k.startswith("fc.")}
            missing, unexpected = model.load_state_dict(state_no_head, strict=False)
            # fc remains randomly initialised
        else:
            model.load_state_dict(state, strict=False)

    return model
