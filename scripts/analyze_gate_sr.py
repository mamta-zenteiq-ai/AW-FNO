#!/usr/bin/env python3
"""
Post-training gate analysis for the NS-forcing 128² super-resolution task (A2).

2D analogue of scripts/analyze_gate_burgers.py.  For a trained rich-gate
AW-FNO checkpoint, extracts per-block gate maps α(x, y) and compares them to
the vorticity-gradient magnitude |∇ω| of the ground-truth HR target.  This is
the paper's interpretability evidence for the SR task (Figure 4 / §5.3):

  - If ρ(1−α, |∇ω|) > 0.1 : the gate routes WNO at vortex filaments / sharp
    gradients, exactly as §3.3 of the paper claims.
  - If ρ ≈ 0              : routing is shock-specific (Burgers) and does not
    transfer to homogeneous-ish turbulence — discuss as regime dependence.

Outputs:
  outputs/figures/sr_gate_sample{s}.png
      4-panel figure per sample: LR input | target+prediction |
      |∇ω| of target | gate α(x,y) of the deepest block (high-|∇ω| contour
      overlaid).
  outputs/tables/sr_gate_correlation.csv
      Pearson ρ((1-α), |∇ω|) per block, per test sample, + mean/std α.
  Console summary — mean correlation on the deepest block.

Usage::
    python scripts/analyze_gate_sr.py \
        --checkpoint /media/HDD/.../awfno_nsforcing_sr_richgate/best.pt \
        --config configs/experiment/train_awfno_nsforcing_richgate.yaml \
        --data_path /media/HDD/mamta_backup/datasets/fno/navier_stokes \
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


def extract_gate_maps(model, x_batch, device) -> Dict[int, torch.Tensor]:
    """Forward x_batch, capturing each block's gate output α via forward hooks.

    Returns {block_idx: alpha tensor (B, C, H, W) on CPU}.
    """
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
            "No gate maps captured. Either the model is not AW-FNO, or the "
            "gate has been patched out (fixed/additive ablation)."
        )
    return gate_maps


def grad_magnitude_2d(u: torch.Tensor) -> torch.Tensor:
    """|∇u| = sqrt((∂u/∂x)² + (∂u/∂y)²), centred differences, periodic BC.

    u: (B, H, W) -> (B, H, W).
    """
    uy = 0.5 * (torch.roll(u, -1, dims=-2) - torch.roll(u, +1, dims=-2))
    ux = 0.5 * (torch.roll(u, -1, dims=-1) - torch.roll(u, +1, dims=-1))
    return torch.sqrt(ux ** 2 + uy ** 2)


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

    gate_maps = extract_gate_maps(model, x_batch, device)
    with torch.no_grad():
        pred = model(x_batch)
        if y_norm is not None:
            pred = y_norm.decode(pred)

    grad_target = grad_magnitude_2d(y_batch.squeeze(1)).cpu()  # (B, H, W)

    out_dir = Path(args.output_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    block_ids = sorted(gate_maps.keys())
    last_block = block_ids[-1]
    rows: List[Dict] = []
    for s in range(args.n_samples):
        g = grad_target[s].numpy().ravel()
        for b in block_ids:
            alpha = gate_maps[b][s].mean(dim=0).numpy()       # (H, W), channel-mean
            inv_alpha = (1.0 - alpha).ravel()
            if inv_alpha.std() < 1e-8 or g.std() < 1e-8:
                rho = float("nan")
            else:
                rho = float(np.corrcoef(inv_alpha, g)[0, 1])
            rows.append({
                "sample": s, "block": b,
                "mean_alpha": float(alpha.mean()),
                "std_alpha": float(alpha.std()),
                "corr_inv_alpha_grad": rho,
            })

    csv_path = out_dir / "tables" / "sr_gate_correlation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Per-sample correlations written: {csv_path}")

    deepest = [r for r in rows if r["block"] == last_block]
    rhos = [r["corr_inv_alpha_grad"] for r in deepest if not np.isnan(r["corr_inv_alpha_grad"])]
    if rhos:
        print(f"\nMean ρ((1-α), |∇ω|) on deepest block ({last_block}): "
              f"{np.mean(rhos):+.3f}  (n={len(rhos)})")
        print("INTERPRETATION:  ρ > 0.1 → gate routes WNO at sharp-gradient regions;\n"
              "                  ρ near 0 → routing does not transfer to this regime.")

    # Per-sample figures
    for s in range(args.n_samples):
        u_in = x_batch[s, 0].cpu().numpy()
        u_tgt = y_batch[s, 0].cpu().numpy()
        u_pred = pred[s, 0].cpu().numpy()
        g = grad_target[s].numpy()
        alpha = gate_maps[last_block][s].mean(dim=0).numpy()
        rel = float(np.linalg.norm(u_pred - u_tgt) / max(np.linalg.norm(u_tgt), 1e-8))
        rho_s = [r["corr_inv_alpha_grad"] for r in deepest if r["sample"] == s][0]

        fig, ax = plt.subplots(2, 2, figsize=(10, 9))
        im0 = ax[0, 0].imshow(u_in, cmap="RdBu_r"); ax[0, 0].set_title("LR input (bicubic-upsampled)")
        fig.colorbar(im0, ax=ax[0, 0], fraction=0.046)
        im1 = ax[0, 1].imshow(u_pred, cmap="RdBu_r"); ax[0, 1].set_title(f"Prediction (rel L2={rel:.4f})")
        fig.colorbar(im1, ax=ax[0, 1], fraction=0.046)
        im2 = ax[1, 0].imshow(g, cmap="magma"); ax[1, 0].set_title("|∇ω| of target — sharp features")
        fig.colorbar(im2, ax=ax[1, 0], fraction=0.046)
        im3 = ax[1, 1].imshow(alpha, cmap="viridis", vmin=0, vmax=1)
        # Overlay contour of the strongest gradients to show alignment with low-α
        thr = np.quantile(g, 0.9)
        ax[1, 1].contour(g, levels=[thr], colors="white", linewidths=0.8, alpha=0.7)
        ax[1, 1].set_title(f"Gate α (block {last_block}), mean={alpha.mean():.3f}\n"
                           f"ρ((1-α),|∇ω|)={rho_s:+.3f}  (white=top-10% |∇ω|)")
        fig.colorbar(im3, ax=ax[1, 1], fraction=0.046)
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"SR gate analysis — sample {s}", fontsize=13)
        plt.tight_layout()
        fig_path = out_dir / "figures" / f"sr_gate_sample{s}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
