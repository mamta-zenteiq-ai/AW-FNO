"""
PDEBench 1D Burgers equation dataset for operator learning.

Source: PDEBench (Takamoto et al., NeurIPS 2022)
File:   1D_Burgers_Sols_Nu{viscosity}.hdf5
DOI:    10.18419/darus-2986

Each HDF5 file contains a single tensor of shape ``(N_samples, T, X)`` where
the spatial axis has 1024 points and there are typically 201 timesteps.
Random initial conditions evolve under
    u_t + u u_x = nu * u_xx
on a periodic domain.  At low viscosity (Nu = 1e-3 or 1e-4) the solution
develops a sharp travelling shock — the regime where FNO's spectral
truncation produces Gibbs oscillations and a wavelet basis has the edge.

Task formulations supported here:

  * "initial_to_final" (default): predict u(x, T) from u(x, 0).
      Input  shape: (B, 1, X)
      Output shape: (B, 1, X)
      Matches the standard PDEBench Burgers benchmark protocol.

  * "next_step":  predict u(x, t+dt) from u(x, t), with t sampled uniformly
                  from [0, T-dt).  Each sample yields T-1 (input, output)
                  pairs, multiplying the effective dataset size by ~T.
                  Useful for testing per-timestep gate behaviour at the
                  shock front (since the shock location varies with t).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


_VALID_TASKS = ("initial_to_final", "next_step")


class PDEBenchBurgers1DDataset(Dataset):
    """
    PDEBench Burgers 1D as a PyTorch Dataset.

    Args:
        data_path: Directory or path containing the HDF5 file.
        split: ``"train"`` or ``"test"``.  Deterministic split via seed.
        task: ``"initial_to_final"`` or ``"next_step"`` (see module docstring).
        n_train: Cap on training samples (None = all).
        n_test: Cap on test samples (None = all).
        train_frac: Fraction of available samples used for training (rest for test).
        viscosity_tag: Filename viscosity tag, e.g. ``"0.001"``.  Used only when
                       ``data_path`` is a directory; the loader will then look
                       for ``1D_Burgers_Sols_Nu{viscosity_tag}.hdf5``.
        seed: Random seed for reproducible split.
        x_normalizer: Optional pre-fitted input normalizer.
        y_normalizer: Optional pre-fitted output normalizer.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        task: str = "initial_to_final",
        n_train: Optional[int] = 9000,
        n_test: Optional[int] = 1000,
        train_frac: float = 0.9,
        viscosity_tag: str = "0.001",
        seed: int = 42,
        x_normalizer=None,
        y_normalizer=None,
    ) -> None:
        super().__init__()
        if task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {_VALID_TASKS}, got {task!r}")
        assert split in ("train", "test")

        self.split = split
        self.task = task
        data_path = Path(data_path)
        if data_path.is_dir():
            data_path = data_path / f"1D_Burgers_Sols_Nu{viscosity_tag}.hdf5"
        if not data_path.exists():
            raise FileNotFoundError(f"PDEBench Burgers file not found: {data_path}")

        u_full = self._load_full(data_path)              # (N, T, X) float32

        # Deterministic train/test split on the sample dimension
        N = u_full.shape[0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(N)
        n_tr = int(train_frac * N)
        idx = perm[:n_tr] if split == "train" else perm[n_tr:]

        # Apply user-imposed caps
        cap = n_train if split == "train" else n_test
        if cap is not None:
            idx = idx[:cap]
        u = u_full[idx]                                   # (N_split, T, X)

        x, y = self._make_pairs(u, task)                  # (M, 1, X) each

        # Normalise — convert to tensor first
        from awfno.utils.unit_gaussian_normalization import UnitGaussianNormalizer
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

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

        self.x: Tensor = x
        self.y: Tensor = y

    # ------------------------------------------------------------------
    # HDF5 loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_full(path: Path) -> np.ndarray:
        """
        Return the full solution tensor as ``(N, T, X)`` float32.

        PDEBench Burgers files come in two known layouts:
          (1) single 'tensor' key  → shape (N, T, X)
          (2) per-variable keys 'Vx', 't-coordinate', etc. (Sod-like) where
              each file is one simulation → shape (T, X); we promote to (1, T, X).
        """
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            if "tensor" in keys:
                arr = np.asarray(f["tensor"], dtype=np.float32)
                if arr.ndim == 2:                        # (T, X) → (1, T, X)
                    arr = arr[None]
                return arr
            # Fallback: single Vx-style simulation per file
            if "Vx" in keys:
                arr = np.asarray(f["Vx"], dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[None]
                return arr
            raise KeyError(
                f"Unrecognised PDEBench Burgers layout. Keys present: {keys}"
            )

    # ------------------------------------------------------------------
    # Task formulation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pairs(u: np.ndarray, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """Build (input, target) arrays from the (N, T, X) tensor."""
        if task == "initial_to_final":
            x = u[:, 0, :][:, None, :]                   # (N, 1, X)
            y = u[:, -1, :][:, None, :]
            return x, y
        # next_step: stack (u_t, u_{t+1}) pairs over time and samples
        # Resulting size: N * (T-1)
        u_t = u[:, :-1, :]                                # (N, T-1, X)
        u_tp1 = u[:, 1:, :]
        N, Tm1, X = u_t.shape
        x = u_t.reshape(N * Tm1, 1, X)
        y = u_tp1.reshape(N * Tm1, 1, X)
        return x, y

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

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


def load_pdebench_burgers(
    data_path: str | Path,
    batch_size: int = 64,
    task: str = "initial_to_final",
    n_train: Optional[int] = 9000,
    n_test: Optional[int] = 1000,
    train_frac: float = 0.9,
    viscosity_tag: str = "0.001",
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, "UnitGaussianNormalizer", "UnitGaussianNormalizer"]:
    """Convenience: build train + test DataLoaders for PDEBench Burgers 1D."""
    train_ds = PDEBenchBurgers1DDataset(
        data_path=data_path,
        split="train",
        task=task,
        n_train=n_train,
        n_test=n_test,
        train_frac=train_frac,
        viscosity_tag=viscosity_tag,
        seed=seed,
    )
    test_ds = PDEBenchBurgers1DDataset(
        data_path=data_path,
        split="test",
        task=task,
        n_train=n_train,
        n_test=n_test,
        train_frac=train_frac,
        viscosity_tag=viscosity_tag,
        seed=seed,
        x_normalizer=train_ds.x_normalizer,
        y_normalizer=train_ds.y_normalizer,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader, train_ds.x_normalizer, train_ds.y_normalizer
