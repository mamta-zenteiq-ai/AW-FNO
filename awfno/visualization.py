"""
Publication-quality visualisation utilities for AW-FNO.

All functions save to disk at 300 dpi with tight bounding boxes, suitable
for direct inclusion in a LaTeX paper.

Functions
---------
plot_field_comparison    — GT / prediction / error maps for ≥1 models
plot_gate_maps           — Learned α(x,y) maps from AW-FNO blocks
plot_convergence_curves  — Training loss curves for multiple runs
plot_spectral_energy     — Radial power spectral density comparison
plot_benchmark_table     — Text table rendered as a matplotlib figure
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared style defaults
# ---------------------------------------------------------------------------

STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style() -> None:
    plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 1. Field comparison (GT vs predictions)
# ---------------------------------------------------------------------------

def plot_field_comparison(
    ground_truth: Tensor,
    predictions: Dict[str, Tensor],
    sample_idx: int = 0,
    channel: int = 0,
    cmap: str = "seismic",
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ground truth next to one or more model predictions + error maps.

    Layout (1 GT + N models × 2 rows):
      Row 0: GT | pred_1 | pred_2 | ...
      Row 1: -- | err_1  | err_2  | ...

    Args:
        ground_truth: Tensor of shape (B, C, H, W) or (B, H, W).
        predictions: Dict mapping model name → prediction tensor (same shape).
        sample_idx: Which sample in the batch to visualise.
        channel: Which channel to visualise (for multi-channel outputs).
        cmap: Colormap for field values.
        save_path: If provided, save the figure here.
        title: Figure super-title.

    Returns:
        matplotlib Figure object.
    """
    _apply_style()

    def _extract(t: Tensor) -> np.ndarray:
        t = t.detach().cpu().float()
        if t.ndim == 4:
            return t[sample_idx, channel].numpy()
        elif t.ndim == 3:
            return t[sample_idx].numpy()
        return t.numpy()

    gt = _extract(ground_truth)
    preds = {k: _extract(v) for k, v in predictions.items()}
    n_models = len(preds)

    n_cols = 1 + n_models
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.2))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    vmin, vmax = gt.min(), gt.max()
    err_max = max((np.abs(p - gt).max() for p in preds.values()), default=1.0)

    # Ground truth
    im_gt = axes[0, 0].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    axes[0, 0].set_title("Ground Truth", fontweight="bold")
    axes[1, 0].axis("off")

    for j, (name, pred) in enumerate(preds.items(), start=1):
        err = np.abs(pred - gt)
        rel = np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-8)

        axes[0, j].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        axes[0, j].set_title(f"{name}\nRel L2={rel:.4f}", fontweight="bold")

        axes[1, j].imshow(err, cmap="hot", vmin=0, vmax=err_max, aspect="equal")
        axes[1, j].set_title(f"|Error|  max={err.max():.4f}")

    # Row labels
    axes[0, 0].set_ylabel("Prediction", fontsize=10)
    axes[1, 1 if n_models > 0 else 0].set_ylabel("|Error|", fontsize=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colourbar (field values)
    cbar_ax = fig.add_axes([0.92, 0.55, 0.012, 0.35])
    fig.colorbar(im_gt, cax=cbar_ax)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Gate α maps
# ---------------------------------------------------------------------------

def plot_gate_maps(
    model: "torch.nn.Module",
    x_input: Tensor,
    y_normalizer=None,
    n_samples: int = 3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualise the learned gate α(x,y) for each AW-FNO block.

    α ≈ 1 → FNO dominates (smooth regions)
    α ≈ 0 → WNO dominates (sharp gradients, vortex cores)

    Args:
        model: Trained AWFNO2d model.
        x_input: Input tensor (B, C, H, W).
        y_normalizer: Optional normalizer; used to decode inputs for display.
        n_samples: Number of samples from the batch to visualise.
        save_path: Where to save.

    Returns:
        matplotlib Figure.
    """
    _apply_style()
    model.eval()
    device = next(model.parameters()).device

    n_samples = min(n_samples, x_input.shape[0])
    x = x_input[:n_samples].to(device)

    # Collect gate activations via forward hooks
    gate_maps: Dict[int, List[Tensor]] = {}  # block_idx → list of alpha tensors

    hooks = []
    for i, block in enumerate(model.blocks):
        if hasattr(block, "gfm"):
            def make_hook(idx):
                def hook(module, inputs, output):
                    with torch.no_grad():
                        # Re-compute alpha from the saved inputs
                        v_fno, v_wno = inputs[0], inputs[1]
                        cat_v = torch.cat([v_fno, v_wno], dim=1)
                        alpha = module.gate(cat_v)  # (B, C, H, W)
                        gate_maps.setdefault(idx, []).append(
                            alpha.mean(dim=1).detach().cpu()  # average over channels
                        )
                return hook
            hooks.append(block.gfm.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    if not gate_maps:
        raise RuntimeError("No gate maps captured. Is this an AWFNO2d model?")

    n_blocks = len(gate_maps)
    fig, axes = plt.subplots(
        n_samples, n_blocks + 1,
        figsize=(3.0 * (n_blocks + 1), 2.5 * n_samples),
    )
    if n_samples == 1:
        axes = axes[None, :]
    if n_blocks == 0:
        axes = axes[:, None]

    for s in range(n_samples):
        # Show input field
        inp = x[s, 0].detach().cpu().numpy()
        axes[s, 0].imshow(inp, cmap="seismic", aspect="equal")
        axes[s, 0].set_title("Input ω" if s == 0 else "")
        axes[s, 0].axis("off")

        for b in sorted(gate_maps.keys()):
            alpha = gate_maps[b][0][s].numpy()  # (H, W)
            im = axes[s, b + 1].imshow(alpha, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
            if s == 0:
                axes[s, b + 1].set_title(f"Block {b+1}: α\n(blue=FNO, red=WNO)")
            axes[s, b + 1].axis("off")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="α (FNO weight)")

    fig.suptitle(
        "Adaptive Gate Maps  (α ≈ 1 → FNO,  α ≈ 0 → WNO)",
        fontsize=12, fontweight="bold",
    )
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Convergence curves
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = "test_rel_l2",
    save_path: Optional[str] = None,
    log_scale: bool = True,
) -> plt.Figure:
    """
    Plot training convergence curves for multiple models.

    Args:
        histories: {model_name: {metric_name: [values_per_epoch]}}
        metric: Which metric key to plot.
        save_path: Output path.
        log_scale: Use log y-axis.
    """
    _apply_style()
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
    styles = ["-", "--", "-.", ":", "-"]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for (name, history), color, ls in zip(histories.items(), colors, styles):
        vals = history.get(metric, [])
        if not vals:
            continue
        epochs = range(1, len(vals) + 1)
        ax.plot(epochs, vals, label=name, color=color, linestyle=ls, linewidth=1.8)

    ax.set_xlabel("Epoch")
    y_label = metric.replace("_", " ").title()
    ax.set_ylabel(y_label)
    ax.set_title(f"Training Convergence — {y_label}")
    if log_scale and all(v > 0 for h in histories.values() for v in h.get(metric, [])):
        ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")

    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Radial power spectral density
# ---------------------------------------------------------------------------

def plot_spectral_energy(
    fields: Dict[str, Tensor],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot radially averaged power spectral density (PSD) for 2D fields.

    This reveals whether a model under- or over-represents energy at
    particular spatial frequencies (e.g., FNO's spectral bias or Gibbs
    oscillations from WNO).

    Args:
        fields: {label: tensor of shape (H, W) or (B, C, H, W)}
        save_path: Output path.
    """
    _apply_style()

    def _radial_psd(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u2d = u.reshape(-1, u.shape[-2], u.shape[-1]).mean(0)
        F = np.fft.fft2(u2d)
        Fsq = (np.abs(np.fft.fftshift(F)) ** 2)
        H, W = Fsq.shape
        cx, cy = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
        psd = np.bincount(r.ravel(), Fsq.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
        k = np.arange(len(psd))
        return k, psd

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["black", "#2196F3", "#F44336", "#4CAF50"]
    styles = ["-", "--", "-.", ":"]

    for (label, field), color, ls in zip(fields.items(), colors, styles):
        if isinstance(field, torch.Tensor):
            arr = field.detach().cpu().float().numpy()
        else:
            arr = np.array(field)
        k, psd = _radial_psd(arr)
        k_max = min(len(k), arr.shape[-1] // 2 + 1)
        ax.semilogy(k[1:k_max], psd[1:k_max], label=label, color=color, linestyle=ls, lw=1.8)

    ax.set_xlabel("Wavenumber |k|")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("Radially Averaged Power Spectrum")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Benchmark table as figure
# ---------------------------------------------------------------------------

def render_table_figure(
    rows: List[Dict],
    columns: List[str],
    column_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Benchmark Results",
) -> plt.Figure:
    """
    Render a results table as a publication-quality matplotlib figure.

    Args:
        rows: List of dicts with result data.
        columns: Which keys to include as columns.
        column_labels: Display labels (defaults to column names).
        save_path: Output path.
        title: Table title.
    """
    _apply_style()
    col_labels = column_labels or columns
    cell_data = [[str(r.get(c, "")) for c in columns] for r in rows]
    row_labels = [str(r.get("model", r.get("ablation", i))) for i, r in enumerate(rows)]

    fig, ax = plt.subplots(figsize=(len(columns) * 1.8, len(rows) * 0.5 + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Bold the header row
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#E3F2FD")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    _save(fig, save_path)
    return fig
