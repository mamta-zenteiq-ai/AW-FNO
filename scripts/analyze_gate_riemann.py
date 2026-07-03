#!/usr/bin/env python3
"""
Post-training gate analysis for the PDEBench 1D Riemann (shock-tube) dataset.

Multi-wave extension of scripts/analyze_gate_burgers.py. A Riemann solution
contains three feature types in one field:
    * shock wave            — sharp jump in rho (large |d rho/dx|)
    * contact discontinuity — jump in rho (large |d rho/dx|)
    * rarefaction fan       — smooth expansion (small |d rho/dx|)

For a trained rich-gate AW-FNO it extracts per-block gate maps alpha(x) and:
  (1) reports Pearson rho((1-alpha), |d rho/dx|) per block/sample (as Burgers);
  (2) bands the domain by gradient magnitude into "discontinuity"
      (shock+contact, high |d rho/dx|) vs "smooth" (rarefaction/constant) and
      reports mean alpha in each band -- the multi-wave routing evidence
      (expect alpha LOWER, i.e. more WNO, in the discontinuity band).

Outputs:
  outputs/figures/riemann_gate_sample{0..n}.png
  outputs/tables/riemann_gate_correlation.csv
  console summary (mean rho + band means across the test set)

Usage::
    python scripts/analyze_gate_riemann.py \
        --checkpoint /media/HDD/.../awfno_riemann_richgate/best.pt \
        --config configs/experiment/train_awfno_riemann_richgate.yaml \
        --data_path /media/HDD/.../PDEBench/cfd \
        --n_samples 4
"""

from __future__ import annotations

import argparse
import csv
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
    """Capture alpha from every AdaptiveGatedFusion gate via forward hooks."""
    gate_maps: Dict[int, torch.Tensor] = {}
    hooks = []
    for i, block in enumerate(getattr(model, "blocks", [])):
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
            "No gate maps captured -- model is not rich-gate AW-FNO (gate may "
            "be patched out by a fixed/additive ablation)."
        )
    return gate_maps


def _grad_magnitude_1d(u: torch.Tensor) -> torch.Tensor:
    """|d u/dx| via centred finite difference (periodic) on the last axis."""
    du = 0.5 * (torch.roll(u, -1, dims=-1) - torch.roll(u, +1, dims=-1))
    return du.abs()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--data_path", default=None)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--disc_percentile", type=float, default=90.0,
                   help="|d rho/dx| percentile above which a point is a "
                        "discontinuity (shock/contact).")
    p.add_argument("--output_dir", default=str(_ROOT / "outputs"))
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    cfg = load_config(args.config, {"data_path": args.data_path})
    set_seed(cfg.get("seed", 42))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )

    model = build_from_config(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device).eval()

    _, test_loader, _, y_norm = load_dataset(cfg)
    if y_norm is not None:
        y_norm.to(device)

    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch[: args.n_samples].to(device)
    y_batch = y_batch[: args.n_samples].to(device)

    gate_maps = _extract_gate_maps(model, x_batch, device)
    with torch.no_grad():
        pred = model(x_batch)
        if y_norm is not None:
            pred = y_norm.decode(pred)

    grad_target = _grad_magnitude_1d(y_batch.squeeze(1)).cpu()  # (B, X)

    out_dir = Path(args.output_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    block_ids = sorted(gate_maps.keys())
    last_block = block_ids[-1]
    rows: List[Dict] = []
    for s in range(args.n_samples):
        g = grad_target[s].numpy()
        thr = np.percentile(g, args.disc_percentile)
        disc = g >= thr            # shock + contact band
        smooth = ~disc             # rarefaction / constant band
        for b in block_ids:
            alpha = gate_maps[b][s].mean(dim=0).numpy()      # (X,) mean over channels
            inv = 1.0 - alpha
            rho = (float("nan") if inv.std() < 1e-8 or g.std() < 1e-8
                   else float(np.corrcoef(inv, g)[0, 1]))
            rows.append({
                "sample": s,
                "block": b,
                "mean_alpha": float(alpha.mean()),
                "alpha_disc": float(alpha[disc].mean()) if disc.any() else float("nan"),
                "alpha_smooth": float(alpha[smooth].mean()) if smooth.any() else float("nan"),
                "corr_inv_alpha_grad": rho,
            })

    csv_path = out_dir / "tables" / "riemann_gate_correlation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Per-sample correlations written: {csv_path}")

    deep = [r for r in rows if r["block"] == last_block]
    rhos = [r["corr_inv_alpha_grad"] for r in deep if not np.isnan(r["corr_inv_alpha_grad"])]
    a_disc = [r["alpha_disc"] for r in deep if not np.isnan(r["alpha_disc"])]
    a_smooth = [r["alpha_smooth"] for r in deep if not np.isnan(r["alpha_smooth"])]
    if rhos:
        print(f"\nDeepest block {last_block} over {len(rhos)} samples:")
        print(f"  mean rho((1-alpha), |d rho/dx|) = {np.mean(rhos):+.3f}")
        print(f"  mean alpha @ discontinuity band = {np.mean(a_disc):.3f}  (lower => more WNO)")
        print(f"  mean alpha @ smooth band        = {np.mean(a_smooth):.3f}")
        print("  EXPECT alpha_disc < alpha_smooth (WNO at shock/contact, FNO in rarefaction).")

    x_axis = np.linspace(0.0, 1.0, y_batch.shape[-1])
    for s in range(args.n_samples):
        fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        u_in = x_batch[s, 0].cpu().numpy()
        u_t = y_batch[s, 0].cpu().numpy()
        u_p = pred[s, 0].cpu().numpy()
        g = grad_target[s].numpy()

        axes[0].plot(x_axis, u_in, "k-", lw=1.5)
        axes[0].set_ylabel(r"$\rho(x,0)$"); axes[0].set_title(f"Sample {s} — initial condition")

        axes[1].plot(x_axis, u_t, "k-", lw=1.5, label="ground truth")
        axes[1].plot(x_axis, u_p, "r--", lw=1.2, label="prediction")
        rel = float(np.linalg.norm(u_p - u_t) / max(np.linalg.norm(u_t), 1e-8))
        axes[1].set_ylabel(r"$\rho(x,T)$"); axes[1].legend(frameon=False)
        axes[1].set_title(f"Final state — rel L2 = {rel:.4f}")

        axes[2].plot(x_axis, g, "b-", lw=1.5)
        axes[2].set_ylabel(r"$|\partial\rho/\partial x|$")
        axes[2].set_title("Gradient magnitude — shock + contact locations")

        alpha = gate_maps[last_block][s].mean(dim=0).numpy()
        axes[3].plot(x_axis, alpha, "g-", lw=1.5, label=r"$\alpha$ (FNO weight)")
        axes[3].axhline(0.5, color="gray", ls=":", lw=0.8)
        ax3b = axes[3].twinx()
        ax3b.plot(x_axis, g / max(g.max(), 1e-8), "b-", lw=0.8, alpha=0.4,
                  label=r"$|\partial\rho/\partial x|$ (norm)")
        axes[3].set_ylabel(r"$\alpha(x)$"); ax3b.set_ylabel("grad / max", color="b")
        axes[3].set_xlabel("x")
        rr = [r['corr_inv_alpha_grad'] for r in deep if r['sample'] == s][0]
        axes[3].set_title(f"Gate (block {last_block}) — mean α={alpha.mean():.3f}, ρ={rr:+.3f}")
        axes[3].legend(loc="upper left", frameon=False)
        ax3b.legend(loc="upper right", frameon=False)

        plt.tight_layout()
        fig_path = out_dir / "figures" / f"riemann_gate_sample{s}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
