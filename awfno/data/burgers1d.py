"""
1D Burgers equation dataset for operator learning.

Task: initial condition u(x,0) → solution u(x,1).
Input shape:  (B, 1, N)   — single channel, N spatial points
Output shape: (B, 1, N)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import scipy.io
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class Burgers1DDataset(Dataset):
    """
    PyTorch Dataset for the 1D Burgers equation (FNO benchmark).

    Args:
        data_path: Directory containing the .mat dataset file.
        split: ``"train"`` or ``"test"``.
        n_train: Number of training samples.
        n_test: Number of test samples.
        resolution: Spatial resolution to use (sub-samples from 8192).
        seed: Random seed for reproducible split.
        x_normalizer: Pre-fitted input normalizer.
        y_normalizer: Pre-fitted output normalizer.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        n_train: int = 1000,
        n_test: int = 200,
        resolution: int = 1024,
        seed: int = 42,
        x_normalizer=None,
        y_normalizer=None,
    ) -> None:
        super().__init__()
        self.split = split
        data_path = Path(data_path)

        x, y = self._load(data_path, split, n_train, n_test, resolution, seed)

        from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer

        if x_normalizer is None and split == "train":
            self.x_normalizer = UnitGaussianNormalizer(x)
        else:
            self.x_normalizer = x_normalizer

        if y_normalizer is None and split == "train":
            self.y_normalizer = UnitGaussianNormalizer(y)
        else:
            self.y_normalizer = y_normalizer

        if self.x_normalizer is not None:
            x = self.x_normalizer.encode(x)

        self.x: Tensor = x.float()
        self.y: Tensor = y.float()

    def _load(
        self,
        data_path: Path,
        split: str,
        n_train: int,
        n_test: int,
        resolution: int,
        seed: int,
    ) -> Tuple[Tensor, Tensor]:
        candidates = sorted(data_path.glob("*.mat"))
        if not candidates:
            raise FileNotFoundError(
                f"No .mat files found in {data_path}.\n"
                "Run: python datasets/download_fno_data.py --dataset burgers1d"
            )
        mat = scipy.io.loadmat(str(candidates[0]))

        # FNO format: variables 'a' (IC), 'u' (solution)
        a = torch.from_numpy(mat.get("a", mat.get("input"))).float()   # (N, 8192)
        u = torch.from_numpy(mat.get("u", mat.get("output"))).float()  # (N, 8192)

        # Sub-sample to target resolution
        step = a.shape[-1] // resolution
        a = a[:, ::step]   # (N, resolution)
        u = u[:, ::step]

        # Add channel dim: (N, 1, resolution)
        a = a.unsqueeze(1)
        u = u.unsqueeze(1)

        # Deterministic split
        n = a.shape[0]
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)

        if split == "train":
            return a[idx[:n_train]], u[idx[:n_train]]
        else:
            return a[idx[n_train : n_train + n_test]], u[idx[n_train : n_train + n_test]]

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]


def load_burgers1d(
    data_path: str | Path,
    batch_size: int = 20,
    n_train: int = 1000,
    n_test: int = 200,
    resolution: int = 1024,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, object, object]:
    """Convenience function: returns train_loader, test_loader, x_normalizer, y_normalizer."""
    train_ds = Burgers1DDataset(
        data_path=data_path,
        split="train",
        n_train=n_train,
        n_test=n_test,
        resolution=resolution,
        seed=seed,
    )
    test_ds = Burgers1DDataset(
        data_path=data_path,
        split="test",
        n_train=n_train,
        n_test=n_test,
        resolution=resolution,
        seed=seed,
        x_normalizer=train_ds.x_normalizer,
        y_normalizer=train_ds.y_normalizer,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader, train_ds.x_normalizer, train_ds.y_normalizer
