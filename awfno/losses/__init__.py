"""
Loss functions for operator learning.

  LpLoss       — relative Lp norm (standard FNO benchmark loss)
  H1Loss       — H1 Sobolev seminorm (penalises gradient errors)
  CombinedLoss — weighted sum of the above
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LpLoss(nn.Module):
    """
    Relative Lp loss.

    L(u, û) = ‖u − û‖_p / ‖u‖_p

    This is the standard metric and training loss used in the FNO paper
    (Li et al., 2021).  Default p=2 gives the relative L2 error.

    Args:
        p: Norm order (default 2).
        reduction: ``"mean"`` or ``"sum"`` over the batch.
        eps: Small constant to avoid division by zero.
    """

    def __init__(self, p: int = 2, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        b = pred.shape[0]
        diff = torch.norm(pred.reshape(b, -1) - target.reshape(b, -1), p=self.p, dim=1)
        norm = torch.norm(target.reshape(b, -1), p=self.p, dim=1).clamp_min(self.eps)
        loss = diff / norm
        return loss.mean() if self.reduction == "mean" else loss.sum()

    # Convenience: allow calling as object (legacy compat with awfno.utils.losses)
    def rel(self, pred: Tensor, target: Tensor) -> Tensor:
        return self(pred, target)

    def abs(self, pred: Tensor, target: Tensor) -> Tensor:
        b = pred.shape[0]
        h = 1.0 / (pred.shape[1] - 1.0)
        norms = h ** (1.0 / self.p) * torch.norm(
            pred.reshape(b, -1) - target.reshape(b, -1), p=self.p, dim=1
        )
        return norms.mean() if self.reduction == "mean" else norms.sum()


class H1Loss(nn.Module):
    """
    Relative H1 Sobolev seminorm loss.

    L(u, û) = ‖∇(u − û)‖₂ / ‖∇u‖₂

    Penalises gradient errors in addition to pointwise errors.  Useful for
    capturing discontinuities and shear layers more accurately than L2 alone.

    Args:
        reduction: ``"mean"`` or ``"sum"`` over the batch.
        eps: Small constant to avoid division by zero.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def _gradient_norm(self, u: Tensor) -> Tensor:
        """Compute ‖∇u‖₂ using finite differences for 2D fields (B, C, H, W)."""
        if u.ndim == 4:
            dx = u[..., 1:, :] - u[..., :-1, :]
            dy = u[..., :, 1:] - u[..., :, :-1]
            b = u.shape[0]
            gx = dx.reshape(b, -1).norm(dim=1)
            gy = dy.reshape(b, -1).norm(dim=1)
            return (gx**2 + gy**2).sqrt()
        elif u.ndim == 3:
            dx = u[..., 1:] - u[..., :-1]
            b = u.shape[0]
            return dx.reshape(b, -1).norm(dim=1)
        raise ValueError(f"H1Loss expects 3D or 4D tensors, got {u.ndim}D")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff_norm = self._gradient_norm(pred - target)
        tgt_norm = self._gradient_norm(target).clamp_min(self.eps)
        loss = diff_norm / tgt_norm
        return loss.mean() if self.reduction == "mean" else loss.sum()


class CombinedLoss(nn.Module):
    """
    Weighted sum: λ_l2 * LpLoss + λ_h1 * H1Loss.

    Args:
        lambda_l2: Weight for the relative L2 term.
        lambda_h1: Weight for the H1 seminorm term.
    """

    def __init__(self, lambda_l2: float = 1.0, lambda_h1: float = 0.1) -> None:
        super().__init__()
        self.l2 = LpLoss(p=2)
        self.h1 = H1Loss()
        self.lambda_l2 = lambda_l2
        self.lambda_h1 = lambda_h1

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.lambda_l2 * self.l2(pred, target) + self.lambda_h1 * self.h1(pred, target)
