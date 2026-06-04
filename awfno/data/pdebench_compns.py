"""
PDEBench 1D compressible Navier-Stokes (Sod-style Riemann problems).

Source: PDEBench `1D_CFD_Sod*.hdf5` files (already on disk under
        /media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d/).

Each file is a *single* Riemann simulation discretised as
    (T_i, X) tensors for `density`, `Vx`, and `pressure`,
where T_i varies per file (12, 16, 36, 41, 41, 12, 201) and X = 1024
(spatial grid with periodic boundaries).

Why this dataset is useful for AW-FNO testing
---------------------------------------------
A 1D Riemann problem generates THREE distinct wave features in a single
solution:
    1. Shock wave         — sharp jump in ρ, p, Vx
    2. Contact discontinuity — jump in ρ only (Vx and p continuous)
    3. Rarefaction fan    — smooth expansion (continuous derivatives)

This is a *richer* gate-routing target than the single-shock Burgers
benchmark: the gate must distinguish between three different feature
types and route appropriately.  Strong shock (Sod3/4) vs near-vacuum
(Sod2) provides additional out-of-distribution stress test.

Task formulation
----------------
Default: next-step prediction.  Each file gives (T_i − 1) (u_t, u_{t+1})
pairs; total ~352 pairs across the 7 files.  Small dataset — sufficient
for proof-of-concept and gate visualisation, not for SOTA training.

Variable selection
------------------
``variable``: which field(s) to use as the model's I/O channels.
  * ``"density"`` (default): predict ρ(x, t+dt) from ρ(x, t).  1 channel.
  * ``"vx"``, ``"pressure"``: single-variable variants.
  * ``"all"``: 3-channel input/output (ρ, Vx, p stacked).

The shock structure is clearest in density (largest jumps), so single-
variable density is the recommended default for gate-routing experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


_VAR_TO_KEY = {
    "density": "density",
    "vx": "Vx",
    "pressure": "pressure",
}


def _list_sod_files(data_path: Path) -> List[Path]:
    """Return all 1D_CFD_Sod*.hdf5 files in `data_path`, sorted by name."""
    files = sorted(data_path.glob("1D_CFD_Sod*.hdf5"))
    if not files:
        raise FileNotFoundError(
            f"No 1D_CFD_Sod*.hdf5 files found in {data_path}.  Expected PDEBench "
            "compressible-NS Sod data."
        )
    return files


def _load_one(file: Path, variable: str) -> np.ndarray:
    """Load one Sod HDF5; return (T_i, X) array for the chosen variable(s)."""
    with h5py.File(file, "r") as f:
        if variable == "all":
            return np.stack([
                np.asarray(f["density"], dtype=np.float32),
                np.asarray(f["Vx"], dtype=np.float32),
                np.asarray(f["pressure"], dtype=np.float32),
            ], axis=-1)  # (T, X, 3)
        key = _VAR_TO_KEY[variable]
        return np.asarray(f[key], dtype=np.float32)  # (T, X)


class PDEBenchSod1DDataset(Dataset):
    """
    PDEBench 1D Sod (compressible-NS Riemann) as a PyTorch Dataset.

    Args:
        data_path: Directory containing 1D_CFD_Sod*.hdf5 files.
        split: ``"train"`` or ``"test"``.  Split is by *file* (whole
               simulations held out), not by time step.  By default the
               last file (Sod6, longest evolution) is held out for test.
        variable: ``"density"`` (default), ``"vx"``, ``"pressure"``, or
                  ``"all"``.
        test_file_indices: Which file indices go to the test split.
                           Default ``[6]`` (Sod6, the long-time run).
        x_normalizer / y_normalizer: Optional pre-fitted normalisers.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        variable: str = "density",
        test_file_indices: Optional[List[int]] = None,
        x_normalizer=None,
        y_normalizer=None,
    ) -> None:
        super().__init__()
        assert split in ("train", "test")
        if variable not in (*_VAR_TO_KEY, "all"):
            raise ValueError(
                f"variable must be one of {list(_VAR_TO_KEY) + ['all']}, got {variable!r}"
            )

        data_path = Path(data_path)
        files = _list_sod_files(data_path)
        test_indices = set(test_file_indices) if test_file_indices is not None else {6}
        selected = [
            f for i, f in enumerate(files)
            if (i in test_indices) == (split == "test")
        ]
        if not selected:
            raise RuntimeError(f"Split '{split}' has no files; check test_file_indices.")

        # Build (input, target) pairs as (u_t, u_{t+1}) across all timesteps in all files
        x_list, y_list = [], []
        for f in selected:
            arr = _load_one(f, variable)         # (T, X) or (T, X, 3)
            if arr.shape[0] < 2:
                continue
            x_list.append(arr[:-1])
            y_list.append(arr[1:])
        x_np = np.concatenate(x_list, axis=0)    # (N, X) or (N, X, 3)
        y_np = np.concatenate(y_list, axis=0)

        # Reshape to (N, C, X)
        if x_np.ndim == 2:
            x_np = x_np[:, None, :]              # (N, 1, X)
            y_np = y_np[:, None, :]
        else:
            x_np = x_np.transpose(0, 2, 1)        # (N, 3, X)
            y_np = y_np.transpose(0, 2, 1)

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()

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

        self.x = x
        self.y = y
        self.variable = variable

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


# ---------------------------------------------------------------------------
# Multi-sample PDEBench 1D CFD (Riemann / shock-tube) loader
# ---------------------------------------------------------------------------
#
# Unlike the single-simulation Sod files above, the bulk PDEBench training
# files (e.g. ``1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5``) store MANY
# random Riemann ICs in one HDF5 with per-variable arrays of shape (N, T, X):
#   density, pressure, Vx  : float32 (N_samples, T, X)
#   t-coordinate, x-coordinate : grids
#
# Each random IC is a two-state shock tube, so the evolved field contains a
# shock + contact discontinuity + rarefaction fan -- the multi-wave gate
# target. Default task: predict the developed field u(x, T) from the IC
# u(x, 0) ("initial_to_final"), analogous to the Burgers loader.

def _cfd_var_array(f, variable: str) -> np.ndarray:
    if variable == "all":
        return np.stack([
            np.asarray(f["density"], dtype=np.float32),
            np.asarray(f["Vx"], dtype=np.float32),
            np.asarray(f["pressure"], dtype=np.float32),
        ], axis=-1)                                   # (N, T, X, 3)
    return np.asarray(f[_VAR_TO_KEY[variable]], dtype=np.float32)  # (N, T, X)


class PDEBenchCFD1DDataset(Dataset):
    """PDEBench multi-sample 1D compressible-NS (Riemann/shock-tube) dataset.

    Args:
        data_path: Directory containing the bulk ``1D_CFD_*Train.hdf5`` file.
        split: ``"train"`` or ``"test"`` (split by sample index).
        variable: ``"density"`` (default), ``"vx"``, ``"pressure"``, ``"all"``.
        task: ``"initial_to_final"`` (default) or ``"next_step"``.
        n_train, n_test, train_frac: sample-count controls.
        file_glob: which bulk file(s) to read (first match used).
        seed: split RNG seed.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        variable: str = "density",
        task: str = "initial_to_final",
        n_train: int = 9000,
        n_test: int = 1000,
        train_frac: float = 0.9,
        file_glob: str = "1D_CFD_Shock*Train.hdf5",
        seed: int = 42,
        x_normalizer=None,
        y_normalizer=None,
    ) -> None:
        super().__init__()
        assert split in ("train", "test")
        assert task in ("initial_to_final", "next_step")
        if variable not in (*_VAR_TO_KEY, "all"):
            raise ValueError(
                f"variable must be one of {list(_VAR_TO_KEY) + ['all']}, got {variable!r}"
            )

        data_path = Path(data_path)
        files = sorted(data_path.glob(file_glob))
        if not files:
            raise FileNotFoundError(
                f"No file matching {file_glob!r} in {data_path}. Expected the "
                "bulk PDEBench 1D CFD (Riemann/shock) training file."
            )

        with h5py.File(files[0], "r") as f:
            arr = _cfd_var_array(f, variable)         # (N, T, X[, 3])

        # Build (input, target) pairs.
        if task == "initial_to_final":
            x_all = arr[:, 0]                          # (N, X[, 3])
            y_all = arr[:, -1]
        else:  # next_step: flatten (N, T-1) consecutive pairs
            x_all = arr[:, :-1].reshape(-1, *arr.shape[2:])
            y_all = arr[:, 1:].reshape(-1, *arr.shape[2:])

        # Deterministic split by sample index.
        n = x_all.shape[0]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_tr = min(int(train_frac * n), n_train) if n_train else int(train_frac * n)
        idx = perm[:n_tr] if split == "train" else perm[n_tr:n_tr + (n_test or n)]

        x_np = x_all[idx]
        y_np = y_all[idx]

        # Reshape to (N, C, X).
        if x_np.ndim == 2:                            # (N, X)
            x_np = x_np[:, None, :]
            y_np = y_np[:, None, :]
        else:                                          # (N, X, 3)
            x_np = x_np.transpose(0, 2, 1)
            y_np = y_np.transpose(0, 2, 1)

        x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
        y = torch.from_numpy(np.ascontiguousarray(y_np)).float()

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

        self.x = x
        self.y = y
        self.variable = variable

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


def load_pdebench_cfd1d(
    data_path: str | Path,
    batch_size: int = 32,
    variable: str = "density",
    task: str = "initial_to_final",
    n_train: int = 9000,
    n_test: int = 1000,
    train_frac: float = 0.9,
    file_glob: str = "1D_CFD_Shock*Train.hdf5",
    seed: int = 42,
    num_workers: int = 0,
):
    """Convenience: train + test DataLoaders for the bulk PDEBench 1D CFD file."""
    train_ds = PDEBenchCFD1DDataset(
        data_path=data_path, split="train", variable=variable, task=task,
        n_train=n_train, n_test=n_test, train_frac=train_frac,
        file_glob=file_glob, seed=seed,
    )
    test_ds = PDEBenchCFD1DDataset(
        data_path=data_path, split="test", variable=variable, task=task,
        n_train=n_train, n_test=n_test, train_frac=train_frac,
        file_glob=file_glob, seed=seed,
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


def load_pdebench_sod(
    data_path: str | Path,
    batch_size: int = 16,
    variable: str = "density",
    test_file_indices: Optional[List[int]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, "UnitGaussianNormalizer", "UnitGaussianNormalizer"]:
    """Convenience: build train + test DataLoaders for PDEBench Sod."""
    train_ds = PDEBenchSod1DDataset(
        data_path=data_path,
        split="train",
        variable=variable,
        test_file_indices=test_file_indices,
    )
    test_ds = PDEBenchSod1DDataset(
        data_path=data_path,
        split="test",
        variable=variable,
        test_file_indices=test_file_indices,
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
