#!/usr/bin/env python3
"""
Benchmarking script: compare all trained models on the same test set.

Produces:
  - Console table (AW-FNO / FNO / WNO side-by-side)
  - outputs/tables/benchmark_ns2d.csv  (for LaTeX import)
  - outputs/figures/field_comparison.png
  - outputs/figures/convergence_curves.png

Usage:
    python experiments/benchmark.py \\
        --data_path /media/HDD/mamta_backup/datasets/fno/navier_stokes \\
        --dataset ns2d

    # If all results are in the default results/ structure this is sufficient;
    # add --checkpoint overrides to point to non-default paths.
    python experiments/benchmark.py --dataset ns2d \\
        --awfno_ckpt results/awfno_ns/best.pt \\
        --fno_ckpt   results/fno_ns/best.pt \\
        --wno_ckpt   results/wno_ns/best.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from experiments.train import load_config, build_from_config, load_dataset
from awfno.models import count_parameters
from awfno.utils.seed import set_seed
from awfno.metrics import compute_metrics, MetricTracker
from awfno.utils.logging import get_logger

logger = get_logger("benchmark")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIGS = {
    "awfno": "configs/experiment/train_awfno_ns.yaml",
    "fno":   "configs/experiment/train_fno_ns.yaml",
    "wno":   "configs/experiment/train_wno_ns.yaml",
    "awfno_no_gate": "configs/experiment/ablation_no_gate.yaml",
}

_CHECKPOINTS = {
    "awfno":         "results/awfno_ns/best.pt",
    "fno":           "results/fno_ns/best.pt",
    "wno":           "results/wno_ns/best.pt",
    "awfno_no_gate": "results/awfno_ns_no_gate/best.pt",
}


def load_model(name: str, config_path: str, ckpt_path: str, device: torch.device):
    cfg = load_config(config_path, {})
    model = build_from_config(cfg)
    if name == "awfno_no_gate":
        from experiments.train import _patch_fixed_gate
        _patch_fixed_gate(model)

    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
        logger.info(f"  ✓ Loaded {name} from {ckpt_path}")
    else:
        logger.warning(f"  ✗ Checkpoint not found: {ckpt_path} (using random weights)")

    return model.to(device).eval(), cfg


@torch.no_grad()
def eval_model(model, loader, y_norm, device) -> Dict[str, float]:
    tracker = MetricTracker()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        if y_norm:
            pred = y_norm.decode(pred)
        tracker.update(compute_metrics(pred, y), n=x.shape[0])
    return tracker.mean()


def load_csv_history(csv_path: str) -> Dict[str, List[float]]:
    """Read a metrics.csv produced by the trainer into column lists."""
    data: Dict[str, List] = {}
    if not Path(csv_path).exists():
        return data
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


# ---------------------------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------------------------

def print_table(rows: List[dict]) -> None:
    keys = ["model", "rel_l2", "mse", "mae", "params_m", "ckpt_exists"]
    header = f"{'Model':<18} {'Rel L2':>10} {'MSE':>12} {'MAE':>10} {'Params (M)':>12} {'Ckpt':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<18} {r.get('rel_l2', float('nan')):>10.6f} "
            f"{r.get('mse', float('nan')):>12.2e} {r.get('mae', float('nan')):>10.6f} "
            f"{r.get('params_m', 0):>12.2f} {'✓' if r.get('ckpt_exists') else '✗':>6}"
        )
    print("=" * len(header) + "\n")


def save_table_csv(rows: List[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Table saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ns2d")
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--awfno_ckpt", default=None)
    p.add_argument("--fno_ckpt", default=None)
    p.add_argument("--wno_ckpt", default=None)
    p.add_argument("--no_gate_ckpt", default=None)
    p.add_argument("--save_figures", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Resolve checkpoint overrides
    ckpts = {
        "awfno":         args.awfno_ckpt or _CHECKPOINTS["awfno"],
        "fno":           args.fno_ckpt   or _CHECKPOINTS["fno"],
        "wno":           args.wno_ckpt   or _CHECKPOINTS["wno"],
        "awfno_no_gate": args.no_gate_ckpt or _CHECKPOINTS["awfno_no_gate"],
    }

    rows = []
    all_histories = {}
    shared_test_loader = None
    y_norm = None

    for name, config_path in _CONFIGS.items():
        if not Path(config_path).exists():
            logger.warning(f"Config not found: {config_path}, skipping {name}")
            continue

        ckpt_path = ckpts[name]
        logger.info(f"\nLoading {name} ...")

        cfg = load_config(config_path, {"data_path": args.data_path})

        if shared_test_loader is None:
            _, shared_test_loader, _, y_norm = load_dataset(cfg)
            if y_norm:
                y_norm.to(device)

        model, cfg = load_model(name, config_path, ckpt_path, device)
        n_params = count_parameters(model)

        metrics = eval_model(model, shared_test_loader, y_norm, device)

        rows.append({
            "model": name,
            **{k: round(v, 8) for k, v in metrics.items()},
            "params_m": round(n_params / 1e6, 3),
            "ckpt_exists": Path(ckpt_path).exists(),
        })

        # Load training history for convergence plot
        csv_path = str(Path(ckpt_path).parent / "metrics.csv")
        all_histories[name] = load_csv_history(csv_path)

        del model

    print_table(rows)
    save_table_csv(rows, "outputs/tables/benchmark_ns2d.csv")

    if args.save_figures:
        try:
            from awfno.visualization import plot_convergence_curves
            plot_convergence_curves(
                all_histories,
                save_path="outputs/figures/convergence_curves.png",
            )
        except Exception as e:
            logger.warning(f"Convergence plot failed: {e}")

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
