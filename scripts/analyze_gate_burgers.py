#!/usr/bin/env python3
"""
Post-training gate analysis for PDEBench Burgers.

For a trained AW-FNO checkpoint, extracts per-block gate maps α(x) and
compares them to the spatial gradient magnitude |∂u/∂x| of the ground-truth
target.  This is the paper's key interpretability evidence.

Outputs:
  outputs/figures/burgers_gate_analysis.png
      4-panel figure for each of N selected test samples showing:
        (1) input u(x, 0)
        (2) target u(x, T) + predicted u(x, T)
        (3) |∂u/∂x| of target — shock locations
        (4) gate α(x) from the deepest block, overlaid on |∂u/∂x|

  outputs/tables/burgers_gate_correlation.csv
      Pearson correlation ρ((1-α), |∂u/∂x|) per block, per test sample.

  Console summary statistics — mean correlation across the test set.

Usage::
    python scripts/analyze_gate_burgers.py \
        --checkpoint /media/HDD/.../awfno_pdebench_burgers/best.pt \
        --config configs/experiment/train_awfno_pdebench_burgers.yaml \
        --data_path /media/HDD/.../PDEBench/burgers \
        --n_samples 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.train import load_config, build_from_config, load_dataset
from awfno.utils.seed import set_seed


def _extract_gate_maps(model, x_batch, device):
    """
    Forward `x_batch` through `model`, capturing gate outputs from every
    AdaptiveGatedFusion module via forward hooks.

    Returns: dict {block_idx: alpha tensor (B, C, X) on CPU}
    """
    gate_maps: Dict[int, torch.Tensor] = {}
    hooks = []

    blocks = getattr(model, "blocks", [])
    for i, block in enumerate(blocks):
        gfm = getattr(block, "gfm", None)
        gate = getattr(gfm, "gate", None) if gfm is not None else None
        if gate is None:
            continue

        def make_hook(idx):
            def hook(_m, _inp, out):
                gate_maps[idx] = out.detach().cpu()
            return hook
        hooks.append(gate.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = model(x_batch.to(device))

    for h in hooks:
        h.remove()

    if not gate_maps:
        raise RuntimeError(
            "No gate maps captured. Either the model is not AW-FNO, or "
            "the gate has been patched out (fixed/additive ablation)."
        )
    return gate_maps


def _grad_magnitude_1d(u: torch.Tensor) -> torch.Tensor:
    """|∂u/∂x| via centred finite difference on the last axis (periodic)."""
    # Centred differences with periodic BC
    u_p = torch.roll(u, -1, dims=-1)
    u_m = torch.roll(u, +1, dims=-1)
    du = 0.5 * (u_p - u_m)
    return du.abs()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_path", default=None)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--output_dir", default=str(_ROOT / "outputs"))
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = load_config(args.config, {"data_path": args.data_path})
    set_seed(cfg.get("seed", 42))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )

    # Build and load model
    model = build_from_config(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device).eval()

    # Data
    _, test_loader, _, y_norm = load_dataset(cfg)
    if y_norm is not None:
        y_norm.to(device)

    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch[: args.n_samples].to(device)
    y_batch = y_batch[: args.n_samples].to(device)

    # Forward + gate extraction
    gate_maps = _extract_gate_maps(model, x_batch, device)
    with torch.no_grad():
        pred = model(x_batch)
        if y_norm is not None:
            pred = y_norm.decode(pred)

    # |∂u/∂x| of target
    grad_target = _grad_magnitude_1d(y_batch.squeeze(1)).cpu()  # (B, X)

    # Compute per-sample, per-block correlation between (1 - α) and |∇u|
    out_dir = Path(args.output_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    block_ids = sorted(gate_maps.keys())
    last_block = block_ids[-1]
    correlations: List[Dict] = []
    for s in range(args.n_samples):
        for b in block_ids:
            alpha_bs = gate_maps[b][s].mean(dim=0).numpy()      # (X,) average over channels
            g = grad_target[s].numpy()
            # Pearson correlation between (1 - α) and |∇u|
            inv_alpha = 1.0 - alpha_bs
            if inv_alpha.std() < 1e-8 or g.std() < 1e-8:
                rho = float("nan")
            else:
                rho = float(np.corrcoef(inv_alpha, g)[0, 1])
            correlations.append({
                "sample": s,
                "block": b,
                "mean_alpha": float(alpha_bs.mean()),
                "std_alpha": float(alpha_bs.std()),
                "corr_inv_alpha_grad": rho,
            })

    # Write CSV
    csv_path = out_dir / "tables" / "burgers_gate_correlation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=correlations[0].keys())
        w.writeheader()
        w.writerows(correlations)
    print(f"Per-sample correlations written: {csv_path}")

    # Summary across all samples and the deepest block
    deepest = [r for r in correlations if r["block"] == last_block]
    rhos = [r["corr_inv_alpha_grad"] for r in deepest if not np.isnan(r["corr_inv_alpha_grad"])]
    if rhos:
        print(
            f"\nMean ρ((1-α), |∂u/∂x|) on deepest block ({last_block}): "
            f"{np.mean(rhos):+.3f}  (n={len(rhos)})"
        )
        print(
            "INTERPRETATION:  ρ > 0.3 → gate clearly routes to WNO at shock;\n"
            "                  ρ near 0  → gate is uninformative;\n"
            "                  ρ < -0.3 → unexpected: gate prefers FNO at shocks."
        )

    # Plot one figure per sample
    x_axis = np.linspace(0.0, 1.0, y_batch.shape[-1])
    for s in range(args.n_samples):
        fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        u_in = x_batch[s, 0].cpu().numpy()
        u_target = y_batch[s, 0].cpu().numpy()
        u_pred = pred[s, 0].cpu().numpy()
        g = grad_target[s].numpy()

        axes[0].plot(x_axis, u_in, "k-", lw=1.5)
        axes[0].set_ylabel("u(x, 0)")
        axes[0].set_title(f"Sample {s} — Initial condition")

        axes[1].plot(x_axis, u_target, "k-", lw=1.5, label="ground truth")
        axes[1].plot(x_axis, u_pred, "r--", lw=1.2, label="prediction")
        axes[1].set_ylabel("u(x, T)")
        axes[1].legend(frameon=False)
        rel = float(np.linalg.norm(u_pred - u_target) / max(np.linalg.norm(u_target), 1e-8))
        axes[1].set_title(f"Final state — rel L2 = {rel:.4f}")

        axes[2].plot(x_axis, g, "b-", lw=1.5)
        axes[2].set_ylabel("|∂u/∂x|")
        axes[2].set_title("Gradient magnitude — shock locations")

        # Gate from the deepest block, averaged over channels
        alpha = gate_maps[last_block][s].mean(dim=0).numpy()
        axes[3].plot(x_axis, alpha, "g-", lw=1.5, label="α (FNO weight)")
        axes[3].axhline(0.5, color="gray", linestyle=":", lw=0.8)
        ax3b = axes[3].twinx()
        ax3b.plot(x_axis, g / max(g.max(), 1e-8), "b-", lw=0.8, alpha=0.4,
                  label="|∂u/∂x| (normalised)")
        axes[3].set_ylabel("α(x)")
        ax3b.set_ylabel("|∂u/∂x| / max", color="b")
        axes[3].set_xlabel("x")
        axes[3].set_title(
            f"Gate (block {last_block}) — mean α={alpha.mean():.3f}, "
            f"ρ((1-α), |∇u|)={[r['corr_inv_alpha_grad'] for r in deepest if r['sample']==s][0]:+.3f}"
        )
        axes[3].legend(loc="upper left", frameon=False)
        ax3b.legend(loc="upper right", frameon=False)

        plt.tight_layout()
        fig_path = out_dir / "figures" / f"burgers_gate_sample{s}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
