"""
Tests for dataset loading without actual data files.

These tests use synthetic in-memory data to validate the Dataset classes,
normalisation, and DataLoader integration — no disk access required.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awfno.data.ns2d import NavierStokes2DDataset
from awfno.data.burgers1d import Burgers1DDataset
from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer


# ---------------------------------------------------------------------------
# Helpers: write synthetic .pt files to a temp dir
# ---------------------------------------------------------------------------

def _write_ns_split_files(tmp_dir: Path, n_train=20, n_test=5, h=32):
    """Write synthetic ns_train_64.pt / ns_test_64.pt."""
    torch.save(
        {"x": torch.randn(n_train, 1, h, h), "y": torch.randn(n_train, 1, h, h)},
        tmp_dir / "ns_train_64.pt",
    )
    torch.save(
        {"x": torch.randn(n_test, 1, h, h), "y": torch.randn(n_test, 1, h, h)},
        tmp_dir / "ns_test_64.pt",
    )


# ---------------------------------------------------------------------------
# NavierStokes2DDataset
# ---------------------------------------------------------------------------

class TestNavierStokes2DDataset:
    def test_loads_from_split_files(self, tmp_path):
        _write_ns_split_files(tmp_path, n_train=16, n_test=4)
        ds = NavierStokes2DDataset(tmp_path, split="train", n_train=16, n_test=4)
        assert len(ds) == 16
        x, y = ds[0]
        assert x.shape == (1, 32, 32)
        assert y.shape == (1, 32, 32)

    def test_test_split(self, tmp_path):
        _write_ns_split_files(tmp_path, n_train=16, n_test=4)
        train_ds = NavierStokes2DDataset(tmp_path, split="train", n_train=16, n_test=4)
        test_ds = NavierStokes2DDataset(
            tmp_path, split="test", n_train=16, n_test=4,
            x_normalizer=train_ds.x_normalizer,
            y_normalizer=train_ds.y_normalizer,
        )
        assert len(test_ds) == 4

    def test_normalizer_fitted_on_train(self, tmp_path):
        _write_ns_split_files(tmp_path, n_train=16, n_test=4)
        ds = NavierStokes2DDataset(tmp_path, split="train", n_train=16, n_test=4)
        assert isinstance(ds.x_normalizer, UnitGaussianNormalizer)

    def test_normalizer_shared_with_test(self, tmp_path):
        _write_ns_split_files(tmp_path, n_train=16, n_test=4)
        train_ds = NavierStokes2DDataset(tmp_path, split="train", n_train=16, n_test=4)
        test_ds = NavierStokes2DDataset(
            tmp_path, split="test", n_train=16, n_test=4,
            x_normalizer=train_ds.x_normalizer,
            y_normalizer=train_ds.y_normalizer,
        )
        # Same normaliser object
        assert test_ds.x_normalizer is train_ds.x_normalizer

    def test_dataloader_integration(self, tmp_path):
        _write_ns_split_files(tmp_path, n_train=16, n_test=4)
        from torch.utils.data import DataLoader
        ds = NavierStokes2DDataset(tmp_path, split="train", n_train=16, n_test=4)
        loader = DataLoader(ds, batch_size=4)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape == (4, 1, 32, 32)
        assert y_batch.shape == (4, 1, 32, 32)

    def test_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NavierStokes2DDataset(tmp_path / "nonexistent", split="train")

    def test_respects_n_train_limit(self, tmp_path):
        _write_ns_split_files(tmp_path, n_train=20, n_test=5)
        ds = NavierStokes2DDataset(tmp_path, split="train", n_train=10, n_test=5)
        assert len(ds) == 10

    def test_input_output_shape_properties(self, tmp_path):
        _write_ns_split_files(tmp_path, n_train=8, n_test=2, h=16)
        ds = NavierStokes2DDataset(tmp_path, split="train", n_train=8, n_test=2)
        assert ds.input_shape == (1, 16, 16)
        assert ds.output_shape == (1, 16, 16)


# ---------------------------------------------------------------------------
# UnitGaussianNormalizer
# ---------------------------------------------------------------------------

class TestUnitGaussianNormalizer:
    def test_encode_decode_roundtrip(self):
        x = torch.randn(100, 1, 8, 8)
        norm = UnitGaussianNormalizer(x)
        x_enc = norm.encode(x)
        x_dec = norm.decode(x_enc)
        assert torch.allclose(x, x_dec, atol=1e-5)

    def test_encoded_approximately_standard_normal(self):
        x = torch.randn(1000, 1, 4, 4) * 5 + 3
        norm = UnitGaussianNormalizer(x)
        x_enc = norm.encode(x)
        # Mean should be ≈ 0, std ≈ 1 across batch+spatial
        flat = x_enc.view(-1)
        assert abs(flat.mean().item()) < 0.05
        assert abs(flat.std().item() - 1.0) < 0.05

    def test_to_device(self):
        x = torch.randn(10, 1, 4, 4)
        norm = UnitGaussianNormalizer(x)
        norm.to("cpu")  # Should not raise
