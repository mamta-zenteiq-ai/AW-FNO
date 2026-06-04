"""
JHTDB forced isotropic turbulence (``isotropic4096``) — spatial
super-resolution dataset.

Source files (already on disk)
------------------------------
``velocity_cutout_x{a}_{b}.npy`` cutouts, each of shape
``(z=1024, y=1024, x=4, V=3)`` float32 — a 3-component velocity field on a
1024x1024 (z, y) plane at 4 streamwise (x) stations. Roughly 23 chunks cover
x = 1..92, i.e. ~92 planes of 1024^2 3-component velocity. (The companion
``.nc`` files hold the same data plus coordinates; we read the faster ``.npy``.)

Task definition
---------------
Mirrors :mod:`awfno.data.nsforcing_sr` so JHTDB is a *drop-in second SR
benchmark* on the identical 128^2 / 4x pipeline (same models, same metrics):

  * each 1024^2 (z, y) plane of a single velocity component is tiled into
    non-overlapping ``patch_size`` x ``patch_size`` patches (default 128);
  * each patch ``u`` is the high-resolution (HR) target;
  * the low-resolution (LR) input is ``bicubic_upsample(avg_pool(u, f), 128)``.

The three velocity components are treated as independent scalar fields
(``in_channels = 1``), which both triples the sample count and lets the
existing single-channel model configs be reused unchanged.

Why isotropic turbulence here: at high Taylor-scale Reynolds number the
velocity-gradient field is strongly intermittent (flatness >> 3) and energy
concentrates on thin vortex filaments embedded in a smooth background -- the
textbook target for adaptive Fourier/wavelet routing. The gate signal is
filament/enstrophy based (high |grad u|), *not* wall-distance; see
``scripts/analyze_gate_jhtdb.py``.

Train/test are split *by chunk* (held-out streamwise stations) to avoid
patch-level leakage between splits.
"""

from __future__ import annotations

import re
from glob import glob
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def _streamwise_key(path: str) -> int:
    """Sort cutouts by their starting streamwise index (x{a}_{b})."""
    m = re.search(r"_x(\d+)_(\d+)", Path(path).stem)
    return int(m.group(1)) if m else 0


def _tile_plane(plane: np.ndarray, patch: int) -> np.ndarray:
    """Split an (H, W) plane into non-overlapping (patch, patch) tiles.

    Returns an array of shape ``(n_tiles, patch, patch)``. Any remainder along
    an edge (H % patch != 0) is dropped.
    """
    H, W = plane.shape
    nh, nw = H // patch, W // patch
    plane = plane[: nh * patch, : nw * patch]
    tiles = plane.reshape(nh, patch, nw, patch).transpose(0, 2, 1, 3)
    return tiles.reshape(nh * nw, patch, patch)


class JHTDBIsoSRDataset(Dataset):
    """JHTDB isotropic turbulence super-resolution dataset (128^2, 4x default).

    Args:
        data_path: Directory containing ``velocity_cutout_x*_*.npy``.
        split: ``"train"`` or ``"test"`` (split by chunk).
        downsample_factor: LR/HR ratio (default 4 -> 32^2 LR).
        patch_size: HR patch edge length (default 128, matches the NS-forcing
            SR model regime).
        n_train, n_test: Sample caps after tiling/shuffling; ``None`` = all.
        test_chunk_frac: Fraction of chunk files held out as the test split.
        components: Which velocity components to use (default all three).
        seed: Shuffling seed (deterministic).
        x_normalizer / y_normalizer: Pre-fitted UnitGaussianNormalizers; fitted
            on the training split when not provided.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        downsample_factor: int = 4,
        patch_size: int = 128,
        n_train: Optional[int] = 8000,
        n_test: Optional[int] = 2000,
        test_chunk_frac: float = 0.2,
        components: Sequence[int] = (0, 1, 2),
        seed: int = 42,
        x_normalizer=None,
        y_normalizer=None,
    ) -> None:
        super().__init__()
        assert split in ("train", "test")

        data_path = Path(data_path)
        files = sorted(glob(str(data_path / "velocity_cutout_x*.npy")),
                       key=_streamwise_key)
        if not files:
            raise FileNotFoundError(
                f"No velocity_cutout_x*.npy found in {data_path}"
            )

        n_test_chunks = max(1, round(len(files) * test_chunk_frac))
        train_files = files[:-n_test_chunks]
        test_files = files[-n_test_chunks:]
        use_files = train_files if split == "train" else test_files

        patches = []
        for f in use_files:
            arr = np.load(f)                       # (z, y, x, V)
            if arr.ndim != 4:
                raise ValueError(f"Unexpected cutout shape {arr.shape} in {f}")
            n_x, n_v = arr.shape[2], arr.shape[3]
            for xi in range(n_x):
                for c in components:
                    if c >= n_v:
                        continue
                    patches.append(_tile_plane(arr[:, :, xi, c], patch_size))
        u = np.concatenate(patches, axis=0)        # (M, patch, patch)
        u = torch.from_numpy(np.ascontiguousarray(u)).float().unsqueeze(1)

        # Deterministic shuffle, then cap.
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(u.shape[0], generator=g)
        u = u[perm]
        limit = n_train if split == "train" else n_test
        if limit is not None:
            u = u[:limit]

        # Build LR input: avg-pool then bicubic up-sample back to patch_size.
        H = u.shape[-1]
        if H % downsample_factor != 0:
            raise ValueError(
                f"patch_size {H} not divisible by downsample_factor "
                f"{downsample_factor}"
            )
        u_lr_small = F.avg_pool2d(u, downsample_factor)
        x = F.interpolate(u_lr_small, size=(H, H), mode="bicubic",
                          align_corners=False)
        y = u

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


def load_jhtdb_iso(
    data_path: str | Path,
    batch_size: int = 16,
    downsample_factor: int = 4,
    patch_size: int = 128,
    n_train: Optional[int] = 8000,
    n_test: Optional[int] = 2000,
    test_chunk_frac: float = 0.2,
    components: Sequence[int] = (0, 1, 2),
    num_workers: int = 0,
    seed: int = 42,
):
    """Convenience function: train + test DataLoaders for JHTDB isotropic SR."""
    train_ds = JHTDBIsoSRDataset(
        data_path=data_path, split="train",
        downsample_factor=downsample_factor, patch_size=patch_size,
        n_train=n_train, n_test=n_test, test_chunk_frac=test_chunk_frac,
        components=components, seed=seed,
    )
    test_ds = JHTDBIsoSRDataset(
        data_path=data_path, split="test",
        downsample_factor=downsample_factor, patch_size=patch_size,
        n_train=n_train, n_test=n_test, test_chunk_frac=test_chunk_frac,
        components=components, seed=seed,
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
