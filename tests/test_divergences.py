"""
tests/test_divergences.py

Tests for divergence metrics (CPU-only, no GPU required).
"""

import numpy as np
import pytest

from dp_shift_bench.metrics.divergences import (
    compute_jsd,
    compute_kl_divergence_estimated,
    compute_mmd,
    compute_proxy_a_distance,
    compute_total_variation,
)


def _make_identical(n=500, d=64):
    x = np.random.randn(n, d)
    return x, x.copy()


def _make_shifted(n=500, d=64, shift=2.0):
    x = np.random.randn(n, d)
    y = np.random.randn(n, d) + shift
    return x, y


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

def test_compute_mmd_identical():
    """MMD between identical distributions should be ~0."""
    x, y = _make_identical()
    mmd = compute_mmd(x, y)
    assert mmd < 0.05, f"MMD too large for identical distributions: {mmd}"


def test_compute_mmd_different():
    """MMD between distributions with large shift should be > 0."""
    x, y = _make_shifted(shift=3.0)
    mmd = compute_mmd(x, y)
    assert mmd > 0.05, f"MMD too small for shifted distributions: {mmd}"


def test_compute_mmd_nonnegative():
    x, y = _make_shifted()
    assert compute_mmd(x, y) >= 0.0


# ---------------------------------------------------------------------------
# Proxy A-distance
# ---------------------------------------------------------------------------

def test_proxy_a_distance_range():
    """PAD should be in [0, 2]."""
    x, y = _make_shifted()
    pad = compute_proxy_a_distance(x, y)
    assert 0.0 <= pad <= 2.0, f"PAD out of range: {pad}"


def test_proxy_a_distance_identical_low():
    """Identical distributions should have low PAD (~0)."""
    x, y = _make_identical()
    pad = compute_proxy_a_distance(x, y)
    assert pad < 1.0, f"PAD too high for identical distributions: {pad}"


def test_proxy_a_distance_separated_high():
    """Well-separated distributions should have PAD closer to 2."""
    x, y = _make_shifted(shift=5.0)
    pad = compute_proxy_a_distance(x, y)
    assert pad > 0.5, f"PAD too low for well-separated distributions: {pad}"


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def test_kl_nonnegative():
    x, y = _make_shifted()
    kl = compute_kl_divergence_estimated(x, y)
    assert kl >= 0.0


def test_kl_identical_near_zero():
    x, _ = _make_identical()
    kl = compute_kl_divergence_estimated(x, x)
    assert kl < 1.0, f"KL too large for same distribution: {kl}"


def test_kl_shifted_positive():
    x, y = _make_shifted(shift=2.0)
    kl = compute_kl_divergence_estimated(x, y)
    assert kl > 0.0


# ---------------------------------------------------------------------------
# JSD
# ---------------------------------------------------------------------------

def test_jsd_range():
    """JSD should be in [0, log(2)]."""
    x, y = _make_shifted()
    jsd = compute_jsd(x, y)
    assert 0.0 <= jsd <= np.log(2) + 1e-6, f"JSD out of range: {jsd}"


def test_jsd_identical_near_zero():
    x, _ = _make_identical()
    jsd = compute_jsd(x, x)
    assert jsd < 0.5


# ---------------------------------------------------------------------------
# Total Variation
# ---------------------------------------------------------------------------

def test_tv_range():
    """TV should be in [0, 1]."""
    x, y = _make_shifted()
    tv = compute_total_variation(x, y)
    assert 0.0 <= tv <= 1.0, f"TV out of range: {tv}"


def test_tv_identical_near_zero():
    x, _ = _make_identical()
    tv = compute_total_variation(x, x)
    assert tv < 0.2, f"TV too large for identical distributions: {tv}"


def test_tv_different_positive():
    x, y = _make_shifted(shift=3.0)
    tv = compute_total_variation(x, y)
    assert tv > 0.01
