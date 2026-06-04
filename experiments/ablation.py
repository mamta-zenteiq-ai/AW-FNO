#!/usr/bin/env python3
"""
Ablation study driver.

Runs all ablation variants and consolidates results into a single table.

Ablations implemented:
  1. no_gate   — Fixed 0.5/0.5 mix (proves GFM contributes)
  2. fno_only  — Disable WNO branch (α = 1 everywhere)
  3. wno_only  — Disable FNO branch (α = 0 everywhere)
  4. wavelet   — Swap db6 → db4 (wavelet sensitivity)

Usage:
    # Run all ablations from scratch
    python experiments/ablation.py --data_path /path/to/data --run_training

    # Evaluate existing ablation checkpoints only
    python experiments/ablation.py --data_path /path/to/data
"""

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from awfno.utils.logging import get_logger

logger = get_logger("ablation")

# Registry of ablation experiments.  Order matters: the table is printed in
# this order, and the additive / fixed-0.5 ablations only make sense when read
# against the full AW-FNO row.
ABLATIONS = [
    {
        "name": "FNO only",
        "config": "configs/experiment/train_fno_ns.yaml",
        "extra_args": [],
    },
    {
        "name": "WNO only",
        "config": "configs/experiment/train_wno_ns.yaml",
        "extra_args": [],
    },
    {
        "name": "AW-FNO additive (v_fno+v_wno)",
        "config": "configs/experiment/ablation_additive.yaml",
        "extra_args": [],
    },
    {
        "name": "AW-FNO fixed gate (α=0.5)",
        "config": "configs/experiment/ablation_no_gate.yaml",
        "extra_args": [],
    },
    {
        "name": "AW-FNO (adaptive gate)",
        "config": "configs/experiment/train_awfno_ns.yaml",
        "extra_args": [],
    },
]


def run_training(ablation: dict, data_path: str) -> None:
    cmd = [
        sys.executable, str(_ROOT / "experiments" / "train.py"),
        "--config", ablation["config"],
        "--data_path", data_path,
        *ablation.get("extra_args", []),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def collect_results(data_path: str) -> None:
    """Evaluate all ablation checkpoints and write consolidated table."""
    import csv
    import json

    rows = []
    for abl in ABLATIONS:
        cfg_path = _ROOT / abl["config"]
        if not cfg_path.exists():
            continue

        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        out_dir = _ROOT / cfg.get("output_dir", f"results/{cfg.get('experiment_name', 'run')}")
        ckpt = out_dir / "best.pt"
        metrics_file = out_dir / "eval_metrics.json"

        cmd = [
            sys.executable, str(_ROOT / "experiments" / "evaluate.py"),
            "--checkpoint", str(ckpt),
            "--config", str(cfg_path),
            "--data_path", data_path,
        ]
        if ckpt.exists():
            subprocess.run(cmd, check=False)

        m: dict = {}
        if metrics_file.exists():
            with open(metrics_file) as f:
                m = json.load(f)

        rows.append({
            "ablation": abl["name"],
            "rel_l2": m.get("rel_l2", "N/A"),
            "mse": m.get("mse", "N/A"),
            "checkpoint": str(ckpt) if ckpt.exists() else "missing",
        })

    # Print table
    print("\nAblation Study Results")
    print("=" * 60)
    print(f"{'Variant':<25} {'Rel L2':>10} {'MSE':>12}")
    print("-" * 60)
    for r in rows:
        print(f"{r['ablation']:<25} {str(r['rel_l2']):>10} {str(r['mse']):>12}")
    print("=" * 60)

    # Save CSV
    out_path = _ROOT / "outputs" / "tables" / "ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Ablation table saved to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--run_training", action="store_true",
                   help="Train all ablation variants from scratch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_training:
        for abl in ABLATIONS:
            logger.info(f"\n{'='*50}\nAblation: {abl['name']}\n{'='*50}")
            run_training(abl, args.data_path)
    collect_results(args.data_path)
