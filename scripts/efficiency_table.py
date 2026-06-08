#!/usr/bin/env python3
"""
Efficiency table (v2-plan ablation B5) for the NS-forcing 128² SR model family.

Reports, per model: parameter count and measured forward-pass latency
(ms/sample, batch-amortised) at 128² on the current device. We measure
wall-clock latency rather than fvcore FLOPs because the dominant operators
here (FFT in the Fourier branch, DWT in the wavelet branch) are not counted by
fvcore, which would make a FLOP table misleading; latency captures their real
cost. Addresses the paper's computational-overhead claim.

Outputs:
  outputs/tables/sr_efficiency.csv
  stdout pretty-print

Usage::
    python scripts/efficiency_table.py            # auto device (GPU if free)
    python scripts/efficiency_table.py --device cpu
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.train import load_config, build_from_config  # noqa: E402

# display name -> (experiment config, fixed-gate ablation flag)
MODELS = [
    ("FNO",                "configs/experiment/train_fno_nsforcing.yaml",            False),
    ("FNO-fat (c=128)",    "configs/experiment/train_fno_fat_nsforcing.yaml",        False),
    ("WNO",                "configs/experiment/train_wno_nsforcing.yaml",            False),
    ("AW-FNO (fixed α)",   "configs/experiment/ablation_no_gate_nsforcing.yaml",     True),
    ("AW-FNO (1×1 gate)",  "configs/experiment/train_awfno_nsforcing.yaml",          False),
    ("AW-FNO (rich gate)", "configs/experiment/train_awfno_nsforcing_richgate.yaml", False),
]


def bench(model, x, device, iters=30, warmup=5):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return dt  # seconds per forward (whole batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu"))

    x = torch.randn(args.batch, 1, 128, 128, device=device)
    rows = []
    for name, cfg_path, fixed in MODELS:
        cfg = load_config(str(ROOT / cfg_path), {})
        model = build_from_config(cfg, ablation_fixed_gate=fixed).to(device)
        params = sum(p.numel() for p in model.parameters())
        dt = bench(model, x, device)
        ms_per_sample = 1e3 * dt / args.batch
        rows.append({"model": name, "params": params,
                     "ms_per_sample": round(ms_per_sample, 3)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = ROOT / "outputs" / "tables" / "sr_efficiency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "params", "ms_per_sample"])
        w.writeheader(); w.writerows(rows)

    print(f"\n  Efficiency on {device} (batch={args.batch}, 128²)")
    print("  " + "-" * 52)
    print(f"  {'Model':<22}{'Params':>14}{'ms/sample':>14}")
    for r in rows:
        print(f"  {r['model']:<22}{r['params']:>14,}{r['ms_per_sample']:>14.3f}")
    print(f"\n  CSV: {out}\n")


if __name__ == "__main__":
    main()
