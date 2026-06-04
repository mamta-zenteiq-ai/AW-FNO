"""
Navier-Stokes 2D dataset for operator learning.

Task: given T_in timesteps of vorticity ω, predict T_out future timesteps.
Input shape:  (B, T_in, H, W)  — channels = time steps
Output shape: (B, T_out, H, W)

The standard FNO benchmark uses:
  T_in = 10, T_out = 10, H = W = 64, Re = 1000, N_train = 1000, N_test = 200
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split


class NavierStokes2DDataset(Dataset):
    """
    PyTorch Dataset for the 2D Navier-Stokes vorticity field.

    Supports two loading modes:
    - Pre-split .pt files: ``ns_train_64.pt`` / ``ns_test_64.pt``
    - Full .pt file: ``ns_V1e-3_N1000_T50.pt`` (split via ``split`` argument)

    Args:
        data_path: Directory containing the dataset files.
        split: One of ``"train"`` or ``"test"``.
        T_in: Number of input timesteps.
        T_out: Number of output timesteps to predict.
        n_train: Maximum training samples (None = use all).
        n_test: Maximum test samples (None = use all).
        seed: Random seed for reproducible splits.
        x_normalizer: Pre-fitted normalizer for inputs. Fitted on train set if None
                      and ``split="train"``.
        y_normalizer: Pre-fitted normalizer for outputs.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        T_in: int = 10,
        T_out: int = 10,
        n_train: Optional[int] = 1000,
        n_test: Optional[int] = 200,
        seed: int = 42,
        x_normalizer=None,
        y_normalizer=None,
    ) -> None:
        super().__init__()
        self.split = split
        self.T_in = T_in
        self.T_out = T_out
        data_path = Path(data_path)

        x, y = self._load(data_path, split, n_train, n_test, seed)

        # Normalise
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
        n_train: Optional[int],
        n_test: Optional[int],
        seed: int,
    ) -> Tuple[Tensor, Tensor]:
        """Load tensors from disk, handling both pre-split and single-file layouts."""
        train_file = data_path / "ns_train_64.pt"
        test_file = data_path / "ns_test_64.pt"

        if train_file.exists() and test_file.exists():
            fname = train_file if split == "train" else test_file
            d = torch.load(fname, weights_only=True)
            x = d["x"].float()
            y = d["y"].float()
        else:
            # Look for the full FNO-format file
            candidates = sorted(data_path.glob("ns_V*.pt")) + sorted(data_path.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    f"No NS dataset files found in {data_path}.\n"
                    "Run: python datasets/download_fno_data.py --dataset ns2d"
                )
            full_path = candidates[0]
            d = torch.load(full_path, weights_only=True)
            x, y = self._parse_fno_format(d)
            # Deterministic split
            n = x.shape[0]
            n_tr = n_train or int(0.83 * n)
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(n, generator=g)
            if split == "train":
                x, y = x[idx[:n_tr]], y[idx[:n_tr]]
            else:
                x, y = x[idx[n_tr:]], y[idx[n_tr:]]

        # Trim to requested counts
        limit = n_train if split == "train" else n_test
        if limit is not None:
            x, y = x[:limit], y[:limit]

        # Ensure shape (N, 1, H, W) — single-channel temporal snapshots
        # Original FNO format may be (N, H, W) or (N, H, W, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if y.ndim == 3:
            y = y.unsqueeze(1)

        return x, y

    @staticmethod
    def _parse_fno_format(d: dict) -> Tuple[Tensor, Tensor]:
        """Handle the original FNO .pt tensor format."""
        if "x" in d and "y" in d:
            return d["x"].float(), d["y"].float()
        # FNO original: single tensor u of shape (N, T, H, W) or (N, H, W, T)
        u = d.get("u", None)
        if u is None:
            raise KeyError(f"Cannot parse dataset dict with keys: {list(d.keys())}")
        if u.shape[-1] not in (u.shape[1], u.shape[2]):
            # last dim is T
            u = u.permute(0, 3, 1, 2)  # (N, T, H, W)
        T_in, T_out = 10, 10
        x = u[:, :T_in]
        y = u[:, T_in : T_in + T_out]
        return x, y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return tuple(self.x.shape[1:])

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return tuple(self.y.shape[1:])


def load_ns2d(
    data_path: str | Path,
    batch_size: int = 20,
    T_in: int = 10,
    T_out: int = 10,
    n_train: int = 1000,
    n_test: int = 200,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, "UnitGaussianNormalizer", "UnitGaussianNormalizer"]:
    """
    Convenience function: load NS2D train and test DataLoaders.

    Returns:
        train_loader, test_loader, x_normalizer, y_normalizer
    """
    train_ds = NavierStokes2DDataset(
        data_path=data_path,
        split="train",
        T_in=T_in,
        T_out=T_out,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    test_ds = NavierStokes2DDataset(
        data_path=data_path,
        split="test",
        T_in=T_in,
        T_out=T_out,
        n_train=n_train,
        n_test=n_test,
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
