"""
Unified trainer for all neural operators (AW-FNO, FNO, WNO).

Features
--------
- Consistent train / eval loop for any nn.Module.
- Automatic mixed precision (AMP) via torch.cuda.amp.
- Gradient clipping to prevent instability on long runs.
- Best-model checkpoint saving based on test relative L2.
- Periodic checkpoints every ``save_every`` epochs.
- CSV logging of all metrics per epoch.
- Optional WandB logging.
- Deterministic reproducibility flags.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from awfno.metrics import compute_metrics, MetricTracker
from awfno.losses import LpLoss
from awfno.utils.logging import get_logger, CSVLogger


# log(2) ≈ 0.693 is the entropy of a uniform Bernoulli distribution.
# Gate entropy near this value means the gate is "collapsed" (α ≈ 0.5 everywhere)
# and provides no adaptive routing.  Decisive gates have entropy << 0.693.
_UNIFORM_BERNOULLI_ENTROPY = math.log(2.0)


class OperatorTrainer:
    """
    Generic trainer for PDE operator-learning models.

    Args:
        model: The neural operator (any nn.Module).
        optimizer: Torch optimiser (e.g. Adam).
        scheduler: LR scheduler (optional).
        criterion: Training loss callable.  Default: LpLoss (relative L2).
        y_normalizer: Normaliser to decode model outputs before metric
                      computation.  Required when the model is trained on
                      normalised targets but evaluated on physical targets.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
        output_dir: Directory for checkpoints, logs, and plots.
        amp: Enable automatic mixed precision.
        grad_clip: Gradient clipping max-norm (None = no clipping).
        log_every: Print frequency in epochs.
        save_every: Checkpoint frequency in epochs.
        use_wandb: Initialise WandB logging.
        experiment_name: Identifier used in logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        criterion: Optional[Callable] = None,
        y_normalizer=None,
        device: str = "auto",
        output_dir: str = "results/run",
        amp: bool = False,
        grad_clip: Optional[float] = 1.0,
        log_every: int = 50,
        save_every: int = 100,
        use_wandb: bool = False,
        experiment_name: str = "experiment",
        lambda_ent: float = 0.0,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion or LpLoss(p=2)
        self.y_normalizer = y_normalizer
        if self.y_normalizer is not None:
            self.y_normalizer.to(self.device)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.amp = amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.save_every = save_every
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name

        self.logger = get_logger(experiment_name)
        self.csv_logger = CSVLogger(self.output_dir / "metrics.csv")

        self._best_rel_l2 = float("inf")
        self._history: Dict[str, list] = {
            "train_loss": [], "train_rel_l2": [],
            "test_rel_l2": [], "test_mse": [],
            "lr": [], "gate_entropy": [],
        }
        # Detected once, on the first eval — None means "not an AW-FNO".
        self._has_gate: Optional[bool] = None
        # Warn the user once if the gate appears to be collapsed.
        self._gate_collapse_warned = False

        # Entropy penalty on the gate: λ_ent * mean(H(α)) added to the loss.
        # Pushes the optimizer to develop *decisive* (low-entropy) routing
        # instead of getting stuck at the α=0.5 saddle point.  Disabled by
        # default (λ=0); enabled by setting `lambda_ent` in the experiment YAML.
        self.lambda_ent: float = float(lambda_ent)
        # Per-step storage populated by forward hooks; cleared each iter.
        self._gate_alphas: List[Tensor] = []
        if self.lambda_ent > 0.0:
            self._register_training_gate_hooks()
            self.logger = get_logger(experiment_name)  # re-bind to ensure exists
            self.logger.info(
                f"Gate entropy penalty ENABLED: λ_ent = {self.lambda_ent}"
            )

    # ------------------------------------------------------------------
    # Core training / eval
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        tracker = MetricTracker()

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            # Reset alpha buffer; hooks (registered in __init__ when λ>0)
            # repopulate it during the forward pass below.
            self._gate_alphas.clear()

            with torch.autocast("cuda", enabled=self.amp):
                pred = self.model(x)
                # Decode normalised output for metric tracking
                pred_dec = self.y_normalizer.decode(pred) if self.y_normalizer else pred
                loss = self.criterion(pred_dec, y)
                # Optional gate-entropy penalty (rewards decisive routing)
                if self.lambda_ent > 0.0:
                    ent_pen = self._compute_gate_entropy_penalty()
                    if ent_pen is not None:
                        loss = loss + self.lambda_ent * ent_pen

            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * x.shape[0]
            tracker.update(compute_metrics(pred_dec.detach(), y), n=x.shape[0])

        n = len(loader.dataset)
        result = tracker.mean()
        result["loss"] = total_loss / n
        return result

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            pred = self.model(x)
            pred_dec = self.y_normalizer.decode(pred) if self.y_normalizer else pred
            tracker.update(compute_metrics(pred_dec, y), n=x.shape[0])

        return tracker.mean()

    # ------------------------------------------------------------------
    # Gate entropy — training penalty (active gradient)
    # ------------------------------------------------------------------

    def _register_training_gate_hooks(self) -> None:
        """
        Attach forward hooks that collect gate alphas during training forward.

        The hooks append to ``self._gate_alphas`` (which is cleared at the
        start of each training step).  Used only when ``lambda_ent > 0``.
        """
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            return
        for block in blocks:
            gfm = getattr(block, "gfm", None)
            gate = getattr(gfm, "gate", None) if gfm is not None else None
            if gate is None:
                continue

            def hook(_module, _inputs, output):
                # Only accumulate during training to avoid memory growth in eval.
                if self.model.training:
                    self._gate_alphas.append(output)

            gate.register_forward_hook(hook)

    def _compute_gate_entropy_penalty(self) -> Optional[Tensor]:
        """
        Mean binary entropy across all blocks' gates for the current batch.

        Returns ``None`` if no gates were captured (e.g. fixed-fusion
        ablation, FNO/WNO baselines), so the caller can skip the addition
        without a no-op tensor in the graph.
        """
        if not self._gate_alphas:
            return None
        eps = 1e-7
        ents = []
        for a in self._gate_alphas:
            a_c = a.float().clamp(eps, 1.0 - eps)
            h = -(a_c * a_c.log() + (1.0 - a_c) * (1.0 - a_c).log())
            ents.append(h.mean())
        return torch.stack(ents).mean()

    # ------------------------------------------------------------------
    # Gate entropy — AW-FNO diagnostic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_gate_entropy(self, x_batch: Tensor) -> float:
        """
        Compute mean per-pixel binary entropy of the AW-FNO gate(s).

            H(α) = -α log α - (1-α) log(1-α)

        Averaged over all spatial positions, channels, batch, and blocks.

        Interpretation
        --------------
        - H(α) ≈ 0      : decisive gate (α ≈ 0 or α ≈ 1 everywhere)
        - H(α) ≈ 0.693  : collapsed gate (α ≈ 0.5 everywhere — uniform mix)

        A trained AW-FNO whose gate is providing real adaptive routing
        should show entropy noticeably below ~0.65 after a few hundred
        epochs.  Returns ``float('nan')`` if the model has no gate.
        """
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            self._has_gate = False
            return float("nan")

        entropies: List[float] = []
        hooks = []
        eps = 1e-7

        def make_hook():
            def hook(_module, _inputs, output):
                a = output.detach().float().clamp(eps, 1.0 - eps)
                h = -(a * a.log() + (1.0 - a) * (1.0 - a).log())
                entropies.append(h.mean().item())
            return hook

        for block in blocks:
            gfm = getattr(block, "gfm", None)
            gate = getattr(gfm, "gate", None) if gfm is not None else None
            # `gate` is the Sequential(Conv, Sigmoid) inside AdaptiveGatedFusion.
            # Fixed-fusion ablations replace `gfm` entirely and won't have it.
            if gate is None:
                continue
            hooks.append(gate.register_forward_hook(make_hook()))

        if not hooks:
            self._has_gate = False
            return float("nan")

        self._has_gate = True
        try:
            self.model(x_batch.to(self.device))
        finally:
            for h in hooks:
                h.remove()

        if not entropies:
            return float("nan")
        return float(sum(entropies) / len(entropies))

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
    ) -> Dict[str, list]:
        """
        Run the full training loop.

        Args:
            train_loader: Training DataLoader.
            test_loader: Evaluation DataLoader.
            epochs: Total number of epochs.

        Returns:
            History dict with loss/metric lists.
        """
        self.logger.info(
            f"Starting training: {self.experiment_name} | "
            f"device={self.device} | amp={self.amp} | epochs={epochs}"
        )
        t0 = time.time()

        # Probe one batch for gate-entropy hook setup
        gate_probe_batch, _ = next(iter(test_loader))

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(train_loader)
            test_metrics = self._eval_epoch(test_loader)

            # Gate-entropy diagnostic (AW-FNO only — NaN otherwise)
            gate_entropy = self._compute_gate_entropy(gate_probe_batch)

            # One-time warning if the gate looks collapsed late in training
            if (
                self._has_gate
                and not self._gate_collapse_warned
                and epoch >= max(50, self.log_every)
                and not math.isnan(gate_entropy)
                and gate_entropy > 0.65
            ):
                self.logger.warning(
                    f"[GATE] Entropy={gate_entropy:.4f} at epoch {epoch} — "
                    f"gate appears collapsed (uniform ≈ {_UNIFORM_BERNOULLI_ENTROPY:.3f}). "
                    "Consider entropy regularization or a richer gate."
                )
                self._gate_collapse_warned = True

            lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self._history["train_loss"].append(train_metrics["loss"])
            self._history["train_rel_l2"].append(train_metrics["rel_l2"])
            self._history["test_rel_l2"].append(test_metrics["rel_l2"])
            self._history["test_mse"].append(test_metrics["mse"])
            self._history["lr"].append(lr)
            self._history["gate_entropy"].append(gate_entropy)

            # Save best model
            if test_metrics["rel_l2"] < self._best_rel_l2:
                self._best_rel_l2 = test_metrics["rel_l2"]
                self._save_checkpoint("best.pt", epoch, test_metrics)

            # Periodic checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, test_metrics)

            # CSV log every epoch
            self.csv_logger.log(
                epoch=epoch,
                lr=lr,
                train_loss=train_metrics["loss"],
                train_rel_l2=train_metrics["rel_l2"],
                test_rel_l2=test_metrics["rel_l2"],
                test_mse=test_metrics["mse"],
                test_mae=test_metrics.get("mae", float("nan")),
                test_rel_h1=test_metrics.get("rel_h1", float("nan")),
                test_enstrophy_err=test_metrics.get("enstrophy_err", float("nan")),
                test_high_freq_rel_l2=test_metrics.get("high_freq_rel_l2", float("nan")),
                gate_entropy=gate_entropy,
            )

            # Console log at intervals
            if epoch % self.log_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                gate_str = (
                    f" | gate_H={gate_entropy:.4f}"
                    if self._has_gate and not math.isnan(gate_entropy)
                    else ""
                )
                self.logger.info(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"train_rl2={train_metrics['rel_l2']:.6f} | "
                    f"test_rl2={test_metrics['rel_l2']:.6f} "
                    f"test_mse={test_metrics['mse']:.6f}"
                    f"{gate_str} | "
                    f"lr={lr:.2e} | elapsed={elapsed:.0f}s"
                )

            # Optional WandB
            if self.use_wandb:
                self._wandb_log(epoch, train_metrics, test_metrics, lr, gate_entropy)

        total_time = time.time() - t0
        self.logger.info(
            f"Training complete in {total_time:.1f}s | "
            f"best test Rel L2 = {self._best_rel_l2:.6f}"
        )
        self.csv_logger.close()
        return self._history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self, filename: str, epoch: int, metrics: Dict[str, float]
    ) -> None:
        path = self.output_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "experiment_name": self.experiment_name,
            },
            path,
        )

    def load_best(self) -> None:
        """Load the best checkpoint saved during training."""
        path = self.output_dir / "best.pt"
        if not path.exists():
            raise FileNotFoundError(f"No best checkpoint at {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.logger.info(
            f"Loaded best checkpoint (epoch {ckpt['epoch']}, "
            f"test Rel L2 = {ckpt['metrics'].get('rel_l2', '?'):.6f})"
        )

    def _wandb_log(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        lr: float,
        gate_entropy: float = float("nan"),
    ) -> None:
        try:
            import wandb
            payload = {
                "epoch": epoch,
                "lr": lr,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"test/{k}": v for k, v in test_metrics.items()},
            }
            if not math.isnan(gate_entropy):
                payload["diagnostics/gate_entropy"] = gate_entropy
            wandb.log(payload)
        except ImportError:
            pass
