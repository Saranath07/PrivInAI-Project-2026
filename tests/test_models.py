"""
tests/test_models.py

Tests for model definitions and the create_model factory.
"""

import pytest
import torch
import torch.nn as nn

from dp_shift_bench.models.image_models import (
    ConvAutoencoder,
    SmallCNN,
    create_model,
    replace_bn_with_gn,
    resnet18_groupnorm,
)


def test_smallcnn_forward():
    model = SmallCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_smallcnn_features():
    model = SmallCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    feats = model.get_features(x)
    assert feats.shape == (4, 256)


def test_smallcnn_num_classes_5():
    model = SmallCNN(num_classes=5)
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 5)


def test_resnet18_groupnorm_forward():
    model = resnet18_groupnorm(num_classes=5)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 5)


def test_resnet18_no_batchnorm():
    """ResNet-18 should have zero BatchNorm2d layers after replacement."""
    model = resnet18_groupnorm(num_classes=10)
    for module in model.modules():
        assert not isinstance(module, nn.BatchNorm2d), f"Found BatchNorm2d: {module}"


def test_autoencoder_reconstruction():
    ae = ConvAutoencoder()
    x = torch.randn(4, 3, 32, 32)
    recon = ae(x)
    assert recon.shape == (4, 3, 32, 32)
    # Output should be in [0, 1] (Sigmoid)
    assert recon.min() >= 0.0
    assert recon.max() <= 1.0


def test_autoencoder_encoder_transfer():
    """Autoencoder encoder state_dict should load cleanly into SmallCNN.features."""
    ae = ConvAutoencoder()
    cnn = SmallCNN(num_classes=10)
    cnn.features.load_state_dict(ae.get_encoder_state_dict(), strict=True)


def test_opacus_compatibility():
    """SmallCNN should pass Opacus ModuleValidator."""
    from opacus.validators import ModuleValidator

    model = SmallCNN(num_classes=10)
    errors = ModuleValidator.validate(model, strict=False)
    assert len(errors) == 0, f"Opacus validation errors: {errors}"


def test_create_model_smallcnn():
    model = create_model(arch="smallcnn", num_classes=10)
    assert isinstance(model, SmallCNN)
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 10)


def test_create_model_resnet18():
    model = create_model(arch="resnet18", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    assert model(x).shape == (2, 10)


def test_create_model_unknown_arch():
    with pytest.raises(ValueError):
        create_model(arch="vgg16")


def test_replace_bn_with_gn():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
    )
    replace_bn_with_gn(model)
    for mod in model.modules():
        assert not isinstance(mod, nn.BatchNorm2d)
