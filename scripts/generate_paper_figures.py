#!/usr/bin/env python3
"""
Generate all paper-ready figures and LaTeX-compatible tables.

Produces:
  outputs/figures/fig_field_comparison.pdf   — GT / FNO / WNO / AW-FNO fields + errors
  outputs/figures/fig_gate_maps.pdf          — Learned α maps per block
  outputs/figures/fig_convergence.pdf        — Training curves
  outputs/figures/fig_spectral_psd.pdf       — Radial power spectral density
  outputs/tables/table_main_results.tex      — LaTeX table for main paper
  outputs/tables/table_ablation.tex          — LaTeX ablation table

Usage:
    python scripts/generate_paper_figures.py \\
        --data_path /media/HDD/mamta_backup/datasets/fno/navier_stokes \\
        --device cuda
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.train import load_config, build_from_config, load_dataset
from awfno.utils.seed import set_seed
from awfno.visualization import (
    plot_field_comparison,
    plot_gate_maps,
    plot_convergence_curves,
    plot_spectral_energy,
    render_table_figure,
)
from awfno.utils.logging import get_logger

logger = get_logger("paper-figures")

OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
TAB_DIR = OUTPUTS / "tables"

CHECKPOINT_MAP = {
    "AW-FNO": "results/awfno_ns/best.pt",
    "FNO":    "results/fno_ns/best.pt",
    "WNO":    "results/wno_ns/best.pt",
    "AW-FNO\n(no gate)": "results/awfno_ns_no_gate/best.pt",
}

CONFIG_MAP = {
    "AW-FNO": "configs/experiment/train_awfno_ns.yaml",
    "FNO":    "configs/experiment/train_fno_ns.yaml",
    "WNO":    "configs/experiment/train_wno_ns.yaml",
    "AW-FNO\n(no gate)": "configs/experiment/ablation_no_gate.yaml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(name, device, data_path):
    cfg_path = ROOT / CONFIG_MAP[name]
    ckpt_path = ROOT / CHECKPOINT_MAP[name]
    if not cfg_path.exists():
        return None, None, None, None
    cfg = load_config(str(cfg_path), {"data_path": data_path})
    model = build_from_config(cfg)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device).eval()
    return model, cfg, ckpt_path, ckpt_path.exists()


def _load_eval_metrics(ckpt_path: Path) -> dict:
    m = ckpt_path.parent / "eval_metrics.json"
    if m.exists():
        with open(m) as f:
            return json.load(f)
    return {}


def _load_history(ckpt_path: Path) -> dict:
    csv_path = ckpt_path.parent / "metrics.csv"
    if not csv_path.exists():
        return {}
    data: dict = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


# ---------------------------------------------------------------------------
# Figure A: Field comparison
# ---------------------------------------------------------------------------

def fig_field_comparison(data_path: str, device: torch.device) -> None:
    logger.info("Generating Fig A: field comparison ...")
    cfg = load_config(str(ROOT / "configs/experiment/train_awfno_ns.yaml"),
                      {"data_path": data_path})
    _, test_loader, _, y_norm = load_dataset(cfg)
    if y_norm:
        y_norm.to(device)

    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch[:1].to(device)
    y_batch = y_batch[:1].to(device)

    predictions = {}
    for name in ["FNO", "WNO", "AW-FNO"]:
        model, _, _, loaded = _load_model(name, device, data_path)
        if model is None:
            continue
        with torch.no_grad():
            pred = model(x_batch)
            if y_norm:
                pred = y_norm.decode(pred)
        predictions[name] = pred.cpu()
        del model

    if not predictions:
        logger.warning("No trained models found — skipping field comparison figure.")
        return

    plot_field_comparison(
        ground_truth=y_batch.cpu(),
        predictions=predictions,
        save_path=str(FIG_DIR / "fig_field_comparison.png"),
        title="Vorticity Field — Navier–Stokes 2D",
    )
    logger.info(f"  Saved: {FIG_DIR}/fig_field_comparison.png")


# ---------------------------------------------------------------------------
# Figure B: Gate maps
# ---------------------------------------------------------------------------

def fig_gate_maps(data_path: str, device: torch.device) -> None:
    logger.info("Generating Fig B: gate α maps ...")
    model, cfg, ckpt_path, loaded = _load_model("AW-FNO", device, data_path)
    if model is None or not loaded:
        logger.warning("AW-FNO checkpoint not found — skipping gate map figure.")
        return

    _, test_loader, _, y_norm = load_dataset(cfg)
    x_batch, _ = next(iter(test_loader))
    x_batch = x_batch[:3].to(device)

    try:
        plot_gate_maps(
            model=model,
            x_input=x_batch,
            n_samples=3,
            save_path=str(FIG_DIR / "fig_gate_maps.png"),
        )
        logger.info(f"  Saved: {FIG_DIR}/fig_gate_maps.png")
    except RuntimeError as e:
        logger.warning(f"  Gate map generation failed: {e}")


# ---------------------------------------------------------------------------
# Figure C: Convergence curves
# ---------------------------------------------------------------------------

def fig_convergence(data_path: str) -> None:
    logger.info("Generating Fig C: convergence curves ...")
    histories = {}
    for name, ckpt_str in CHECKPOINT_MAP.items():
        ckpt_path = ROOT / ckpt_str
        h = _load_history(ckpt_path)
        if h:
            histories[name.replace("\n", " ")] = h

    if not histories:
        logger.warning("No metrics.csv files found — skipping convergence figure.")
        return

    plot_convergence_curves(
        histories,
        metric="test_rel_l2",
        save_path=str(FIG_DIR / "fig_convergence.png"),
    )
    logger.info(f"  Saved: {FIG_DIR}/fig_convergence.png")


# ---------------------------------------------------------------------------
# Figure D: Spectral PSD
# ---------------------------------------------------------------------------

def fig_spectral_psd(data_path: str, device: torch.device) -> None:
    logger.info("Generating Fig D: spectral energy comparison ...")
    cfg = load_config(str(ROOT / "configs/experiment/train_awfno_ns.yaml"),
                      {"data_path": data_path})
    _, test_loader, _, y_norm = load_dataset(cfg)
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch[:1].to(device)
    y_batch = y_batch[:1].to(device)

    fields = {"Ground Truth": y_batch.cpu().squeeze()}
    for name in ["FNO", "WNO", "AW-FNO"]:
        model, _, _, loaded = _load_model(name, device, data_path)
        if model is None:
            continue
        with torch.no_grad():
            pred = model(x_batch)
            if y_norm:
                pred = y_norm.decode(pred)
        fields[name] = pred.cpu().squeeze()
        del model

    if len(fields) < 2:
        logger.warning("Not enough predictions for PSD figure.")
        return

    plot_spectral_energy(fields, save_path=str(FIG_DIR / "fig_spectral_psd.png"))
    logger.info(f"  Saved: {FIG_DIR}/fig_spectral_psd.png")


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def _format_float(v, fmt=".4f") -> str:
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def latex_main_table(data_path: str, device: torch.device) -> None:
    logger.info("Generating LaTeX main results table ...")
    rows = []
    for name in ["FNO", "WNO", "AW-FNO\n(no gate)", "AW-FNO"]:
        ckpt_path = ROOT / CHECKPOINT_MAP[name]
        m = _load_eval_metrics(ckpt_path)
        rows.append({
            "model": name.replace("\n", " "),
            "rel_l2": _format_float(m.get("rel_l2", "---")),
            "mse": _format_float(m.get("mse", "---"), ".2e"),
            "mae": _format_float(m.get("mae", "---")),
        })

    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out = TAB_DIR / "table_main_results.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by scripts/generate_paper_figures.py\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Quantitative comparison on 2D Navier-Stokes (Re=1000, 64×64). "
                "Lower is better for all metrics.}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\hline\n")
        f.write("Model & Rel~$L_2$ $\\downarrow$ & MSE $\\downarrow$ & MAE $\\downarrow$ \\\\\n")
        f.write("\\hline\n")
        for r in rows:
            star = " \\textbf{(ours)}" if r["model"] == "AW-FNO" else ""
            f.write(f"{r['model']}{star} & {r['rel_l2']} & {r['mse']} & {r['mae']} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    logger.info(f"  Saved: {out}")


def latex_ablation_table() -> None:
    csv_path = TAB_DIR / "ablation.csv"
    if not csv_path.exists():
        logger.warning("No ablation.csv found — run experiments/ablation.py first.")
        return

    rows = []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    out = TAB_DIR / "table_ablation.tex"
    with open(out, "w") as f:
        f.write("% Auto-generated by scripts/generate_paper_figures.py\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation study — 2D Navier-Stokes. "
                "All variants use identical hyper-parameters.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{lrr}\n")
        f.write("\\hline\n")
        f.write("Variant & Rel~$L_2$ $\\downarrow$ & MSE $\\downarrow$ \\\\\n")
        f.write("\\hline\n")
        for r in rows:
            f.write(f"{r['ablation']} & {r.get('rel_l2','---')} & {r.get('mse','---')} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    logger.info(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate paper figures and tables")
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip_figures", action="store_true")
    p.add_argument("--skip_tables", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    data_path = (
        args.data_path
        or __import__("os").environ.get("DATA_PATH")
        or "data/ns2d"
    )

    if not args.skip_figures:
        try:
            fig_field_comparison(data_path, device)
        except Exception as e:
            logger.warning(f"Field comparison failed: {e}")
        try:
            fig_gate_maps(data_path, device)
        except Exception as e:
            logger.warning(f"Gate map failed: {e}")
        try:
            fig_convergence(data_path)
        except Exception as e:
            logger.warning(f"Convergence fig failed: {e}")
        try:
            fig_spectral_psd(data_path, device)
        except Exception as e:
            logger.warning(f"Spectral PSD failed: {e}")

    if not args.skip_tables:
        try:
            latex_main_table(data_path, device)
        except Exception as e:
            logger.warning(f"Main table failed: {e}")
        try:
            latex_ablation_table()
        except Exception as e:
            logger.warning(f"Ablation table failed: {e}")

    logger.info(f"\nAll outputs written to:\n  {FIG_DIR}\n  {TAB_DIR}")


if __name__ == "__main__":
    main()
