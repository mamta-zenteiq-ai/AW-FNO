"""Tests for loss functions and metrics."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awfno.losses import LpLoss, H1Loss, CombinedLoss
from awfno.metrics import (
    relative_l2, relative_l1, mse, mae,
    max_pointwise_error, spectral_l2,
    compute_metrics, MetricTracker,
)


class TestLpLoss:
    def test_zero_on_identical(self):
        x = torch.randn(4, 1, 16, 16)
        loss = LpLoss()(x, x)
        assert loss.item() < 1e-6

    def test_positive(self):
        x = torch.randn(4, 1, 16, 16)
        y = torch.randn(4, 1, 16, 16)
        assert LpLoss()(x, y).item() > 0

    def test_reduction_sum(self):
        x = torch.randn(4, 1, 16, 16)
        y = torch.randn(4, 1, 16, 16)
        loss_mean = LpLoss(reduction="mean")(x, y)
        loss_sum = LpLoss(reduction="sum")(x, y)
        assert abs(loss_sum.item() - loss_mean.item() * 4) < 1e-4

    def test_backward(self):
        x = torch.randn(4, 1, 16, 16, requires_grad=True)
        y = torch.randn(4, 1, 16, 16)
        LpLoss()(x, y).backward()
        assert x.grad is not None

    def test_rel_alias(self):
        loss = LpLoss()
        x = torch.randn(4, 16)
        y = torch.randn(4, 16)
        assert abs(loss(x, y).item() - loss.rel(x, y).item()) < 1e-6


class TestH1Loss:
    def test_zero_on_identical(self):
        x = torch.randn(4, 1, 16, 16)
        assert H1Loss()(x, x).item() < 1e-6

    def test_positive(self):
        x = torch.randn(4, 1, 16, 16)
        y = torch.randn(4, 1, 16, 16)
        assert H1Loss()(x, y).item() > 0

    def test_works_1d(self):
        x = torch.randn(4, 1, 64)
        y = torch.randn(4, 1, 64)
        assert H1Loss()(x, y).item() > 0

    def test_backward(self):
        x = torch.randn(4, 1, 16, 16, requires_grad=True)
        y = torch.randn(4, 1, 16, 16)
        H1Loss()(x, y).backward()
        assert x.grad is not None


class TestCombinedLoss:
    def test_forward(self):
        x = torch.randn(4, 1, 16, 16)
        y = torch.randn(4, 1, 16, 16)
        val = CombinedLoss()(x, y)
        assert val.item() > 0

    def test_zero_on_identical(self):
        x = torch.randn(4, 1, 16, 16)
        val = CombinedLoss()(x, x)
        assert val.item() < 1e-5


class TestMetrics:
    def test_relative_l2_identical(self):
        x = torch.randn(4, 1, 16, 16)
        assert relative_l2(x, x).mean().item() < 1e-6

    def test_mse_identical(self):
        x = torch.randn(4, 1, 16, 16)
        assert mse(x, x).mean().item() < 1e-10

    def test_spectral_l2_shape(self):
        x = torch.randn(4, 1, 16, 16)
        y = torch.randn(4, 1, 16, 16)
        result = spectral_l2(x, y)
        assert result.shape == (4,)

    def test_compute_metrics_keys(self):
        x = torch.randn(4, 1, 16, 16)
        y = torch.randn(4, 1, 16, 16)
        m = compute_metrics(x, y)
        for key in ("rel_l2", "mse", "mae", "max_err"):
            assert key in m, f"Missing key {key}"

    def test_metric_tracker(self):
        tracker = MetricTracker()
        for _ in range(3):
            tracker.update({"rel_l2": 0.1, "mse": 0.01}, n=10)
        mean = tracker.mean()
        assert abs(mean["rel_l2"] - 0.1) < 1e-6

    def test_tracker_reset(self):
        tracker = MetricTracker()
        tracker.update({"x": 1.0}, n=1)
        tracker.reset()
        assert tracker._sums == {}
