"""
Forced 2D Navier-Stokes — *spatial super-resolution* dataset.

Source files (already on disk):
    nsforcing_train_128.pt   : {"x": (N, 128, 128), "y": (N, 128, 128)}
    nsforcing_test_128.pt    : same layout

Task definition
---------------
This loader reframes the forced-NS data as a 4× spatial super-resolution
benchmark.  For each snapshot ``u``:

    u_lr  = avg_pool(u, factor)              # coarse:  128/factor × 128/factor
    x     = bicubic_upsample(u_lr, 128)      # back to 128×128 (LR aliasing)
    y     = u                                 # high-resolution target

The model therefore always operates at the target resolution (128×128).
The LR information bottleneck is encoded by the avg-pool + bicubic-upsample
preprocessing.  This is the standard "SR as dealiasing" setup used in the
turbulence-SR literature (Fukami et al. 2019; Liu et al. 2020) and avoids
the WNO fixed-input-size limitation, since both x and y share the 128×128
grid that the model is initialised on.

By default, both ``x`` and ``y`` use the snapshot ``y`` from the underlying
file (i.e., the future-time vorticity field).  The input snapshot ``x`` is
ignored in the pure-SR formulation; this keeps the task strictly spatial
and isolates the super-resolution capability from temporal prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class NSForcingSRDataset(Dataset):
    """
    Forced NS 2D super-resolution dataset (128×128 target, 4× by default).

    Args:
        data_path: Directory containing nsforcing_{train,test}_128.pt.
        split: ``"train"`` or ``"test"``.
        downsample_factor: LR/HR ratio.  Common values: 2, 4, 8.
        n_train, n_test: Sample caps; ``None`` uses all available samples.
        snapshot: Which field to use from the source file: ``"y"`` (default,
                  future snapshot) or ``"x"`` (initial snapshot).
        seed: Random seed (kept for parity with other dataset loaders;
              the file already encodes a fixed split).
        x_normalizer / y_normalizer: Pre-fitted UnitGaussianNormalizers.
                                     Fitted on the training split if not
                                     provided and ``split == "train"``.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        downsample_factor: int = 4,
        n_train: Optional[int] = 8000,
        n_test: Optional[int] = 2000,
        snapshot: str = "y",
        seed: int = 42,
        x_normalizer=None,
        y_normalizer=None,
    ) -> None:
        super().__init__()
        assert split in ("train", "test")
        assert snapshot in ("x", "y")

        data_path = Path(data_path)
        fname = data_path / f"nsforcing_{split}_128.pt"
        if not fname.exists():
            raise FileNotFoundError(f"NS-forcing file not found: {fname}")

        d = torch.load(fname, weights_only=True)
        u = d[snapshot].float()                  # (N, 128, 128)
        if u.ndim == 3:
            u = u.unsqueeze(1)                   # (N, 1, 128, 128)

        limit = n_train if split == "train" else n_test
        if limit is not None:
            u = u[:limit]

        # Build the LR input by avg-pool then bicubic up-sample back to 128².
        H = u.shape[-1]
        if H % downsample_factor != 0:
            raise ValueError(
                f"Spatial size {H} not divisible by downsample_factor "
                f"{downsample_factor}"
            )
        u_lr_small = F.avg_pool2d(u, downsample_factor)           # (N, 1, H/f, W/f)
        x = F.interpolate(
            u_lr_small, size=(H, H), mode="bicubic", align_corners=False
        )                                                          # (N, 1, 128, 128)
        y = u                                                      # HR target

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
        self.downsample_factor = downsample_factor

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


def load_nsforcing_sr(
    data_path: str | Path,
    batch_size: int = 16,
    downsample_factor: int = 4,
    n_train: Optional[int] = 8000,
    n_test: Optional[int] = 2000,
    snapshot: str = "y",
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, "UnitGaussianNormalizer", "UnitGaussianNormalizer"]:
    """Convenience function: train + test DataLoaders for NS-forcing SR."""
    train_ds = NSForcingSRDataset(
        data_path=data_path,
        split="train",
        downsample_factor=downsample_factor,
        n_train=n_train,
        n_test=n_test,
        snapshot=snapshot,
        seed=seed,
    )
    test_ds = NSForcingSRDataset(
        data_path=data_path,
        split="test",
        downsample_factor=downsample_factor,
        n_train=n_train,
        n_test=n_test,
        snapshot=snapshot,
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
