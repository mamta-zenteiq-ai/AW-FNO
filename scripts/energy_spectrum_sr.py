#!/usr/bin/env python3
"""
Energy-spectrum analysis for the NS-forcing 128² super-resolution task (A3).

Computes the radially averaged power spectral density E(|k|) for the
ground-truth HR field, the bicubic baseline (the LR-upsampled model input),
and each trained model's prediction.  Reveals:

  - whether AW-FNO matches the turbulent inertial-range slope better than FNO,
  - whether AW-FNO puts less spurious energy in the high-|k| tail than FNO
    (the Gibbs-reduction claim, paper C10).

It auto-discovers whichever runs have finished under --results_root, so it can
be run on a partial queue and re-run as more models complete.

Outputs:
  outputs/figures/sr_energy_spectrum.png   — E(k) vs |k|, all models + k^{-5/3}
  outputs/tables/sr_high_freq_energy.csv    — fraction of energy in |k|>k_max/2

Usage::
    python scripts/energy_spectrum_sr.py \
        --data_path /media/HDD/mamta_backup/datasets/fno/navier_stokes \
        --results_root /media/HDD/mamta_backup/aw_fno_results \
        --n_samples 64
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.train import load_config, build_from_config, load_dataset
from awfno.utils.seed import set_seed

# run-dir name -> experiment config used to build/load that model
RUN_CONFIGS = {
    "fno_nsforcing_sr":            "configs/experiment/train_fno_nsforcing.yaml",
    "wno_nsforcing_sr":            "configs/experiment/train_wno_nsforcing.yaml",
    "awfno_nsforcing_sr_no_gate":  "configs/experiment/ablation_no_gate_nsforcing.yaml",
    "awfno_nsforcing_sr":          "configs/experiment/train_awfno_nsforcing.yaml",
    "fno_fat_nsforcing_sr":        "configs/experiment/train_fno_fat_nsforcing.yaml",
    "awfno_nsforcing_sr_richgate": "configs/experiment/train_awfno_nsforcing_richgate.yaml",
}
# Friendly legend labels
LABELS = {
    "fno_nsforcing_sr": "FNO", "wno_nsforcing_sr": "WNO",
    "awfno_nsforcing_sr_no_gate": "AW-FNO (fixed α)",
    "awfno_nsforcing_sr": "AW-FNO (1×1 gate)",
    "fno_fat_nsforcing_sr": "FNO-fat",
    "awfno_nsforcing_sr_richgate": "AW-FNO (rich gate)",
}


def radial_psd(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Radially averaged PSD of a 2D field.

    For a batch (B, H, W) the *power* |FFT|² is averaged over samples (NOT the
    fields — averaging fields first would let independent turbulent snapshots
    cancel and wash out the high-k energy).
    """
    arr = field.reshape(-1, field.shape[-2], field.shape[-1])
    F = np.fft.fftshift(np.fft.fft2(arr, axes=(-2, -1)), axes=(-2, -1))
    Fsq = (np.abs(F) ** 2).mean(0)            # mean power spectrum over samples
    H, W = Fsq.shape
    cx, cy = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    psd = np.bincount(r.ravel(), Fsq.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return np.arange(len(psd)), psd


def high_freq_fraction(psd: np.ndarray, k_cut: int) -> float:
    total = psd[1:].sum()
    return float(psd[k_cut:].sum() / max(total, 1e-12))


def predict(cfg, ckpt_path, x, y_norm, device) -> torch.Tensor:
    model = build_from_config(cfg, ablation_fixed_gate=cfg.get("ablation_fixed_gate", False))
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(state)
    model = model.to(device).eval()
    with torch.no_grad():
        pred = model(x.to(device))
        if y_norm is not None:
            pred = y_norm.decode(pred)
    return pred.cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="/media/HDD/mamta_backup/datasets/fno/navier_stokes")
    p.add_argument("--results_root", default="/media/HDD/mamta_backup/aw_fno_results")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--output_dir", default=str(_ROOT / "outputs"))
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )

    # Use any config just to load the dataset once (they share the SR dataset).
    base_cfg = load_config(_ROOT / "configs/experiment/train_fno_nsforcing.yaml",
                           {"data_path": args.data_path})
    set_seed(base_cfg.get("seed", 42))
    _, test_loader, x_norm, y_norm = load_dataset(base_cfg)
    if y_norm is not None:
        y_norm.to(device)
    if x_norm is not None:
        x_norm.to(device)

    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch[: args.n_samples]
    y_batch = y_batch[: args.n_samples]

    # Ground truth + bicubic baseline in physical space.
    gt = y_norm.decode(y_batch.to(device)).cpu() if y_norm is not None else y_batch
    bicubic = x_norm.decode(x_batch.to(device)).cpu() if x_norm is not None else x_batch

    spectra: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    spectra["Ground truth"] = radial_psd(gt.squeeze(1).numpy())
    spectra["Bicubic"] = radial_psd(bicubic.squeeze(1).numpy())

    results_root = Path(args.results_root)
    for run, cfg_rel in RUN_CONFIGS.items():
        ckpt = results_root / run / "best.pt"
        if not ckpt.exists():
            print(f"  skip {run} (no checkpoint yet)")
            continue
        cfg = load_config(_ROOT / cfg_rel, {"data_path": args.data_path})
        try:
            pred = predict(cfg, ckpt, x_batch, y_norm, device)
            spectra[LABELS.get(run, run)] = radial_psd(pred.squeeze(1).numpy())
            print(f"  loaded {run} -> {LABELS.get(run, run)}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {run}: {type(e).__name__}: {e}")

    out_dir = Path(args.output_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    res = 128
    k_max = res // 2
    k_cut = k_max // 2  # high-frequency band: |k| > k_max/2

    # CSV of high-frequency energy fraction (Gibbs metric)
    csv_path = out_dir / "tables" / "sr_high_freq_energy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "high_freq_energy_fraction"])
        for label, (k, psd) in spectra.items():
            w.writerow([label, f"{high_freq_fraction(psd, k_cut):.6e}"])
    print(f"High-freq energy fractions written: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (k, psd) in spectra.items():
        kk = min(len(k), k_max)
        lw = 2.4 if "rich" in label.lower() else 1.6
        z = 5 if "rich" in label.lower() else 2
        ax.loglog(k[1:kk], psd[1:kk], label=label, lw=lw, zorder=z)
    # Kolmogorov k^{-5/3} reference, anchored to the ground-truth low-k energy.
    gt_k, gt_psd = spectra["Ground truth"]
    anchor = gt_psd[2]
    kref = np.arange(2, k_max)
    ax.loglog(kref, anchor * (kref / 2.0) ** (-5.0 / 3.0), "k:", lw=1.0,
              label=r"$k^{-5/3}$ reference")
    ax.set_xlabel("Wavenumber |k|")
    ax.set_ylabel("Power spectral density E(|k|)")
    ax.set_title("NS-forcing 128² SR — radial energy spectrum")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig_path = out_dir / "figures" / "sr_energy_spectrum.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Spectrum figure saved: {fig_path}")


if __name__ == "__main__":
    main()
