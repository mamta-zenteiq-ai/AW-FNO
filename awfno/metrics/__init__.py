"""
Evaluation metrics for operator learning.

All functions operate on batched tensors with shape (B, *spatial).
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict

# Lazy import: H1Loss lives in awfno.losses; importing at module level is fine
# since losses.py has no reverse dependency on metrics.py.
from awfno.losses import H1Loss

_h1_metric = H1Loss(reduction="mean")


def relative_l2(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Per-sample relative L2 error: ‖pred − target‖₂ / ‖target‖₂.

    Returns a 1-D tensor of shape (B,).
    """
    b = pred.shape[0]
    diff = (pred - target).reshape(b, -1).norm(dim=1)
    norm = target.reshape(b, -1).norm(dim=1).clamp_min(eps)
    return diff / norm


def relative_l1(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Per-sample relative L1 error."""
    b = pred.shape[0]
    diff = (pred - target).reshape(b, -1).abs().mean(dim=1)
    norm = target.reshape(b, -1).abs().mean(dim=1).clamp_min(eps)
    return diff / norm


def mse(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error, averaged over spatial dims, per sample."""
    b = pred.shape[0]
    return (pred - target).reshape(b, -1).pow(2).mean(dim=1)


def mae(pred: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error per sample."""
    b = pred.shape[0]
    return (pred - target).reshape(b, -1).abs().mean(dim=1)


def max_pointwise_error(pred: Tensor, target: Tensor) -> Tensor:
    """Maximum pointwise absolute error per sample (captures Gibbs spikes)."""
    b = pred.shape[0]
    return (pred - target).reshape(b, -1).abs().max(dim=1).values


def spectral_l2(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Relative L2 error in the Fourier domain (2D only).

    Captures frequency-resolved accuracy; particularly sensitive to
    high-frequency errors (Gibbs oscillations) missed by spatial L2.
    """
    if pred.ndim < 3:
        raise ValueError("spectral_l2 requires at least 3D tensors (B, H, W)")
    b = pred.shape[0]
    pred_f = torch.fft.rfft2(pred.reshape(b, -1, *pred.shape[-2:]).float())
    tgt_f = torch.fft.rfft2(target.reshape(b, -1, *target.shape[-2:]).float())
    diff = (pred_f - tgt_f).abs().reshape(b, -1).norm(dim=1)
    norm = tgt_f.abs().reshape(b, -1).norm(dim=1).clamp_min(eps)
    return diff / norm


def enstrophy(u: Tensor) -> Tensor:
    """
    Scalar enstrophy per sample: 0.5 * mean(omega^2).

    For 2D Navier-Stokes vorticity fields, enstrophy quantifies the
    rotational kinetic energy. Models that miss small-scale vortex
    structures will systematically underestimate enstrophy.
    """
    b = u.shape[0]
    return 0.5 * u.reshape(b, -1).pow(2).mean(dim=1)


def enstrophy_error(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Relative enstrophy error per sample: |E_pred - E_true| / E_true.
    """
    ens_pred = enstrophy(pred)
    ens_true = enstrophy(target).clamp_min(eps)
    return (ens_pred - ens_true).abs() / ens_true


def high_freq_spectral_error(
    pred: Tensor, target: Tensor, k_cutoff_frac: float = 0.5, eps: float = 1e-8
) -> Tensor:
    """
    Relative L2 error in the *high-frequency* band of the 2D Fourier spectrum.

    Specifically targets the spectral region beyond ``k_cutoff_frac * k_max``,
    where Gibbs oscillations from spectral-truncation models (e.g., vanilla
    FNO) tend to introduce spurious energy.  This is the metric that most
    directly quantifies the AW-FNO claim of reduced Gibbs artifacts.

    Args:
        pred, target: Tensors of shape (B, C, H, W) or (B, H, W).
        k_cutoff_frac: Fraction of the Nyquist wavenumber above which to
                       compute the error.  Default 0.5 = upper half of the
                       spectrum.
    """
    if pred.ndim < 3:
        raise ValueError("high_freq_spectral_error requires >= 3D tensors")
    b = pred.shape[0]
    p = pred.reshape(b, -1, *pred.shape[-2:]).float()
    t = target.reshape(b, -1, *target.shape[-2:]).float()
    pf = torch.fft.rfft2(p)
    tf = torch.fft.rfft2(t)
    H, W_half = pf.shape[-2], pf.shape[-1]
    # Build a mask for high-frequency modes
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=pf.device).abs()
    kx = torch.fft.rfftfreq(2 * (W_half - 1), d=1.0 / (2 * (W_half - 1)), device=pf.device).abs()
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    k_mag = (KX**2 + KY**2).sqrt()
    k_max = k_mag.max()
    mask = (k_mag > k_cutoff_frac * k_max).float()
    diff = ((pf - tf).abs() * mask).reshape(b, -1).norm(dim=1)
    norm = (tf.abs() * mask).reshape(b, -1).norm(dim=1).clamp_min(eps)
    return diff / norm


def _h1_for_compute(pred: Tensor, target: Tensor) -> float:
    """
    Wrap H1Loss to tolerate both 3D (B, H, W) and 4D (B, C, H, W) inputs.

    H1Loss treats 3D as a 1D field (B, C, L); for 2D fields without an
    explicit channel dim we add one before calling it.
    """
    p, t = pred, target
    if p.ndim == 3:
        p = p.unsqueeze(1)
        t = t.unsqueeze(1)
    return _h1_metric(p, t).item()


def compute_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    """
    Compute all standard metrics for a batch.

    Returns:
        Dict with keys: rel_l2, rel_l1, rel_h1, mse, mae, max_err,
        spectral_rel_l2 (2D only), high_freq_rel_l2 (2D only),
        enstrophy_err (2D only).
        All values are Python floats (batch mean).
    """
    with torch.no_grad():
        pred = pred.float()
        target = target.float()
        result = {
            "rel_l2": relative_l2(pred, target).mean().item(),
            "rel_l1": relative_l1(pred, target).mean().item(),
            "rel_h1": _h1_for_compute(pred, target),
            "mse": mse(pred, target).mean().item(),
            "mae": mae(pred, target).mean().item(),
            "max_err": max_pointwise_error(pred, target).mean().item(),
        }
        if pred.ndim >= 3 and pred.shape[-1] > 1 and pred.shape[-2] > 1:
            # 2D-field metrics: spectral L2, high-freq spectral L2, enstrophy
            result["spectral_rel_l2"] = spectral_l2(pred, target).mean().item()
            result["high_freq_rel_l2"] = high_freq_spectral_error(
                pred, target, k_cutoff_frac=0.5
            ).mean().item()
            result["enstrophy_err"] = enstrophy_error(pred, target).mean().item()
    return result


class MetricTracker:
    """
    Accumulates per-batch metrics and computes running means.

    Usage::

        tracker = MetricTracker()
        for batch_pred, batch_target in loader:
            metrics = compute_metrics(batch_pred, batch_target)
            tracker.update(metrics, n=batch_pred.shape[0])
        summary = tracker.mean()
    """

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + v * n
            self._counts[k] = self._counts.get(k, 0) + n

    def mean(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()
