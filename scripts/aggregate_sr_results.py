#!/usr/bin/env python3
"""
Aggregate the NS-forcing 128² super-resolution results into the paper's
main comparison table (Stage A1 / Table 1).

2D-SR analogue of scripts/aggregate_phase1_results.py. Reads `metrics.csv`
from each completed SR training under the results root and emits:
  - outputs/tables/sr_main_table.csv  (machine-readable)
  - outputs/tables/sr_main_table.tex  (LaTeX body, ready to \input)
  - stdout pretty-print

Each row is one model; metrics are taken from the epoch with the best
(lowest) test_rel_l2 — i.e. what best.pt corresponds to. Runs that have not
finished yet are reported with their current best so the table can be
inspected mid-queue; rerun once the queue is complete for final numbers.

The bicubic "no-model" baseline is rel_l2(x, y): since the dataset input x is
already the bicubic-upsampled LR field, this is computed directly from the
test tensors when --data_path is given.

Usage::
    python scripts/aggregate_sr_results.py \
        --data_path /media/HDD/mamta_backup/datasets/fno/navier_stokes
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
HDD = Path("/media/HDD/mamta_backup/aw_fno_results")

# Per-dataset table definition: ordered runs, the config used to compute the
# bicubic baseline from test tensors, and the output filename prefix.
DATASETS: Dict[str, Dict] = {
    "nsforcing": {
        "title": "NS-forcing 128² Super-Resolution",
        "bicubic_config": "configs/experiment/train_fno_nsforcing.yaml",
        "prefix": "sr_main_table",
        "runs": [
            {"name": "Bicubic (no model)",         "dir": None},
            {"name": "FNO",                         "dir": "fno_nsforcing_sr"},
            {"name": "FNO-fat (hc=128)",            "dir": "fno_fat_nsforcing_sr"},
            {"name": "WNO",                         "dir": "wno_nsforcing_sr"},
            {"name": "AW-FNO (fixed α=0.5)",   "dir": "awfno_nsforcing_sr_no_gate"},
            {"name": "AW-FNO (1×1 gate)",      "dir": "awfno_nsforcing_sr"},
            {"name": "AW-FNO (rich gate, k=5, 2L)", "dir": "awfno_nsforcing_sr_richgate"},
        ],
    },
    "jhtdb": {
        "title": "JHTDB isotropic-turbulence 128² Super-Resolution",
        "bicubic_config": "configs/experiment/train_fno_jhtdb.yaml",
        "prefix": "jhtdb_main_table",
        "runs": [
            {"name": "Bicubic (no model)",         "dir": None},
            {"name": "FNO",                         "dir": "fno_jhtdb_sr"},
            {"name": "WNO",                         "dir": "wno_jhtdb_sr"},
            {"name": "AW-FNO (fixed α=0.5)",   "dir": "awfno_jhtdb_sr_no_gate"},
            {"name": "AW-FNO (rich gate, k=5, 2L)", "dir": "awfno_jhtdb_sr_richgate"},
        ],
    },
}

# Column key in metrics.csv, display label, format spec.
METRICS = [
    ("test_rel_l2",           "Rel L2 ↓",    "{:.4e}"),
    ("test_rel_h1",           "Rel H1 ↓",    "{:.4e}"),
    ("test_high_freq_rel_l2", "High-f L2 ↓", "{:.2f}"),
    ("test_enstrophy_err",    "Enstrophy ↓", "{:.4e}"),
    ("gate_entropy",          "Gate H",      "{:.4f}"),
]


def _read_best_row(metrics_csv: Path) -> Optional[Dict]:
    if not metrics_csv.exists():
        return None
    best = None
    n = 0
    with open(metrics_csv) as f:
        for row in csv.DictReader(f):
            n += 1
            try:
                v = float(row["test_rel_l2"])
            except (KeyError, ValueError):
                continue
            if best is None or v < float(best["test_rel_l2"]):
                best = row
    if best is not None:
        best["_n_epochs"] = str(n)
    return best


def _bicubic_baseline(data_path: Optional[str], bicubic_config: str) -> Optional[Dict]:
    """rel_l2(x, y) on the test set — x is already the bicubic LR upsample."""
    if data_path is None:
        return None
    import sys
    sys.path.insert(0, str(ROOT))
    try:
        import torch
        from experiments.train import load_config, load_dataset
    except Exception as e:
        print(f"  [bicubic baseline skipped: {e}]")
        return None
    cfg = load_config(str(ROOT / bicubic_config), {"data_path": data_path})
    _, test_loader, _, y_norm = load_dataset(cfg)
    num, den = 0.0, 0.0
    with torch.no_grad():
        for x, y in test_loader:
            if y_norm is not None:
                yd = y_norm.decode(y)
                xd = y_norm.decode(x)  # input shares target normalisation
            else:
                yd, xd = y, x
            num += torch.linalg.vector_norm((xd - yd).flatten(1), dim=1).sum().item()
            den += torch.linalg.vector_norm(yd.flatten(1), dim=1).sum().item()
    return {"test_rel_l2": str(num / max(den, 1e-12)), "_n_epochs": "0"}


def _fmt(val, spec: str) -> str:
    if val in ("", "nan", "NaN", None):
        return "—"
    try:
        return spec.format(float(val))
    except (ValueError, TypeError):
        return str(val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=None,
                    help="If given, compute the bicubic baseline from test tensors.")
    ap.add_argument("--dataset", default="nsforcing", choices=list(DATASETS),
                    help="Which SR dataset table to aggregate.")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    RUNS = spec["runs"]
    prefix = spec["prefix"]

    out_dir = ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for run in RUNS:
        if run["dir"] is None:
            best = _bicubic_baseline(args.data_path, spec["bicubic_config"])
        else:
            best = _read_best_row(HDD / run["dir"] / "metrics.csv")
        if best is None:
            rows.append({"name": run["name"], "missing": True})
            continue
        row = {"name": run["name"],
               "epoch": best.get("epoch", "0"),
               "n_epochs": best.get("_n_epochs", "")}
        for key, _, _ in METRICS:
            row[key] = best.get(key, "")
        rows.append(row)

    # Pretty-print
    namew = max(len(r["name"]) for r in rows) + 2
    print(f"\n  {spec['title']} — Main Table (best-epoch metrics)")
    print("  " + "-" * (namew + 14 + 14 * len(METRICS)))
    header = f"  {'Model':<{namew}}{'Epoch':>10}"
    for _, label, _ in METRICS:
        header += f"{label:>14}"
    print(header)
    for r in rows:
        if r.get("missing"):
            print(f"  {r['name']:<{namew}}{'[pending]':>10}")
            continue
        ep = f"{r['epoch']}/{r['n_epochs']}" if r["n_epochs"] not in ("", "0") else r["epoch"]
        line = f"  {r['name']:<{namew}}{ep:>10}"
        for key, _, spec in METRICS:
            line += f"{_fmt(r.get(key), spec):>14}"
        print(line)
    print()

    # CSV
    csv_out = out_dir / f"{prefix}.csv"
    with open(csv_out, "w", newline="") as f:
        fieldnames = ["name", "epoch", "n_epochs"] + [k for k, _, _ in METRICS]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if r.get("missing"):
                w.writerow({"name": r["name"]})
            else:
                w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"CSV  table written: {csv_out}")

    # LaTeX body
    tex_out = out_dir / f"{prefix}.tex"
    with open(tex_out, "w") as f:
        f.write("% Auto-generated by scripts/aggregate_sr_results.py — do not edit by hand.\n")
        f.write("\\begin{tabular}{l" + "c" * (len(METRICS) + 1) + "}\n\\toprule\n")
        f.write("Model & Params & " + " & ".join(l for _, l, _ in METRICS) + " \\\\\n\\midrule\n")
        for r in rows:
            if r.get("missing"):
                continue
            name = r["name"].replace("α", "$\\alpha$").replace("×", "$\\times$")
            cells = [name, "—"] + [_fmt(r.get(k), spec) for k, _, spec in METRICS]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX table written: {tex_out}\n")


if __name__ == "__main__":
    main()
