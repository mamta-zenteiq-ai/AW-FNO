"""
Unit tests for all neural operator models.

Tests:
  - Correct output shapes for all model variants
  - Deterministic outputs with fixed seed
  - Gradient flow (backward pass)
  - Parameter count sanity
  - build_model registry
  - Gate ablation patch
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awfno.models import (
    AWFNO1d, AWFNO2d,
    AWFNOv2_1d, AWFNOv2_2d,
    FNO,
    WNO1d, WNO2d,
    build_model,
    count_parameters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_1d(device):
    return torch.randn(2, 1, 64, device=device)


@pytest.fixture
def batch_2d(device):
    return torch.randn(2, 1, 32, 32, device=device)


# ---------------------------------------------------------------------------
# AWFNO2d (paper primary)
# ---------------------------------------------------------------------------

class TestAWFNO2d:
    def _make(self, **kwargs) -> AWFNO2d:
        defaults = dict(
            in_channels=1, out_channels=1,
            n_modes=(8, 8), size=(32, 32),
            hidden_channels=8, n_layers=2,
            wno_level=2, wno_wavelet="db4",
        )
        defaults.update(kwargs)
        return AWFNO2d(**defaults)

    def test_output_shape(self, batch_2d):
        model = self._make()
        out = model(batch_2d)
        assert out.shape == (2, 1, 32, 32), f"Got {out.shape}"

    def test_output_shape_multichannel(self):
        model = AWFNO2d(
            in_channels=3, out_channels=2,
            n_modes=(8, 8), size=(16, 16),
            hidden_channels=8, n_layers=2, wno_level=1,
        )
        x = torch.randn(2, 3, 16, 16)
        out = model(x)
        assert out.shape == (2, 2, 16, 16)

    def test_gradient_flows(self, batch_2d):
        model = self._make()
        out = model(batch_2d)
        loss = out.mean()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_deterministic_with_seed(self, batch_2d):
        torch.manual_seed(0)
        m1 = self._make()
        torch.manual_seed(0)
        m2 = self._make()
        x = batch_2d.clone()
        with torch.no_grad():
            out1, out2 = m1(x), m2(x)
        assert torch.allclose(out1, out2)

    def test_parameter_count_positive(self):
        model = self._make()
        n = count_parameters(model)
        assert n > 0

    def test_padding(self, batch_2d):
        model = self._make(padding=2)
        out = model(batch_2d)
        assert out.shape == (2, 1, 32, 32)

    def test_no_norm(self, batch_2d):
        model = self._make(norm=None)
        out = model(batch_2d)
        assert out.shape == (2, 1, 32, 32)


# ---------------------------------------------------------------------------
# AWFNO1d
# ---------------------------------------------------------------------------

class TestAWFNO1d:
    def _make(self, **kwargs) -> AWFNO1d:
        defaults = dict(
            in_channels=1, out_channels=1,
            n_modes=(16,), size=(64,),
            hidden_channels=8, n_layers=2,
            wno_level=2, wno_wavelet="db4",
        )
        defaults.update(kwargs)
        return AWFNO1d(**defaults)

    def test_output_shape(self, batch_1d):
        model = self._make()
        out = model(batch_1d)
        assert out.shape == (2, 1, 64)

    def test_backward(self, batch_1d):
        model = self._make()
        out = model(batch_1d)
        out.mean().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# AWFNOv2_2d
# ---------------------------------------------------------------------------

class TestAWFNOv2_2d:
    def _make(self, **kwargs):
        defaults = dict(
            in_channels=1, out_channels=1,
            n_modes=(8, 8), size=(32, 32),
            hidden_channels=8, n_fno_layers=2, n_wno_layers=2,
            wno_level=2, wno_wavelet="db4",
        )
        defaults.update(kwargs)
        return AWFNOv2_2d(**defaults)

    def test_output_shape(self, batch_2d):
        model = self._make()
        assert model(batch_2d).shape == (2, 1, 32, 32)

    def test_backward(self, batch_2d):
        model = self._make()
        model(batch_2d).mean().backward()


# ---------------------------------------------------------------------------
# FNO
# ---------------------------------------------------------------------------

class TestFNO:
    def _make(self, n_modes=(8, 8), **kwargs):
        return FNO(
            n_modes=n_modes,
            in_channels=kwargs.get("in_channels", 1),
            out_channels=kwargs.get("out_channels", 1),
            hidden_channels=kwargs.get("hidden_channels", 8),
            n_layers=kwargs.get("n_layers", 2),
        )

    def test_output_shape_2d(self, batch_2d):
        model = self._make()
        assert model(batch_2d).shape == (2, 1, 32, 32)

    def test_output_shape_1d(self, batch_1d):
        model = self._make(n_modes=(16,))
        assert model(batch_1d).shape == (2, 1, 64)

    def test_backward_2d(self, batch_2d):
        model = self._make()
        model(batch_2d).mean().backward()


# ---------------------------------------------------------------------------
# WNO
# ---------------------------------------------------------------------------

class TestWNO2d:
    def _make(self, **kwargs):
        defaults = dict(in_channels=1, out_channels=1, width=8, size=(32, 32), level=2, n_layers=2)
        defaults.update(kwargs)
        return WNO2d(**defaults)

    def test_output_shape(self, batch_2d):
        model = self._make()
        assert model(batch_2d).shape == (2, 1, 32, 32)

    def test_backward(self, batch_2d):
        model = self._make()
        model(batch_2d).mean().backward()


class TestWNO1d:
    def test_output_shape(self, batch_1d):
        model = WNO1d(in_channels=1, out_channels=1, width=8, size=(64,), level=2, n_layers=2)
        assert model(batch_1d).shape == (2, 1, 64)


# ---------------------------------------------------------------------------
# build_model registry
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_awfno(self):
        m = build_model("awfno", in_channels=1, out_channels=1, n_modes=(8, 8),
                        size=(32, 32), hidden_channels=8, n_layers=2, wno_level=2)
        assert isinstance(m, AWFNO2d)

    def test_awfno_v2(self):
        m = build_model("awfno_v2", in_channels=1, out_channels=1, n_modes=(8, 8),
                        size=(32, 32), hidden_channels=8, n_fno_layers=2, n_wno_layers=2, wno_level=2)
        assert isinstance(m, AWFNOv2_2d)

    def test_fno(self):
        m = build_model("fno", n_modes=(8, 8), in_channels=1, out_channels=1,
                        hidden_channels=8, n_layers=2)
        assert isinstance(m, FNO)

    def test_wno_2d(self):
        m = build_model("wno", in_channels=1, out_channels=1, n_modes=(8, 8),
                        size=(32, 32), hidden_channels=8, n_layers=2)
        assert isinstance(m, WNO2d)

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("unknown_model", n_modes=(8,), size=(32,))

    def test_count_parameters(self):
        m = build_model("fno", n_modes=(8,), in_channels=1, out_channels=1,
                        hidden_channels=8, n_layers=2)
        assert count_parameters(m) > 0


# ---------------------------------------------------------------------------
# Gate ablation patch
# ---------------------------------------------------------------------------

def test_fixed_gate_ablation():
    """AW-FNO with fixed gate should still produce correct output shape."""
    from experiments.train import _patch_fixed_gate
    model = AWFNO2d(
        in_channels=1, out_channels=1, n_modes=(8, 8), size=(32, 32),
        hidden_channels=8, n_layers=2, wno_level=2,
    )
    _patch_fixed_gate(model)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)
