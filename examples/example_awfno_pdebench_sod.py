#!/usr/bin/env python3
"""
Example: AW-FNO with rich gate on PDEBench 1D Sod (compressible NS).

Demonstrates the second dataset in the AW-FNO research plan after Phase 2.5
established that the rich-gate (Mitigation C) is the winning configuration.
The Sod Riemann problem is a richer multi-wave target than Burgers — each
sample contains a shock wave, a contact discontinuity, AND a rarefaction
fan.  We expect the rich gate to learn to route WNO at the shock and
contact discontinuity while routing FNO in the rarefaction region.

What this script does
---------------------
  1. Loads PDEBench Sod files (already on disk; no download needed).
  2. Builds AW-FNO with the rich gate (k=5, 2 layers) — same architecture
     as the Burgers winner.
  3. Trains for a small number of epochs (default 50) — proof-of-concept,
     not a paper-quality result, because the Sod dataset on disk is small
     (~352 next-step pairs across 7 simulations).
  4. Reports rel_l2 + gate_entropy and saves the best checkpoint.

Run
---
    python examples/example_awfno_pdebench_sod.py

Override key settings via CLI:
    python examples/example_awfno_pdebench_sod.py \\
        --epochs 200 --variable density --output_dir /tmp/sod_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the project importable from the examples/ subdirectory
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from awfno.data.pdebench_compns import load_pdebench_sod
from awfno.models import build_model, count_parameters
from awfno.training.operator_trainer import OperatorTrainer
from awfno.losses import LpLoss
from awfno.utils.seed import set_seed
from awfno.utils.logging import get_logger
from experiments.train import _patch_rich_gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("---", 1)[0])
    p.add_argument(
        "--data_path",
        default="/media/HDD/mamta_backup/datasets/PDEBench/comp_ns/1d",
        help="Directory containing 1D_CFD_Sod*.hdf5 files.",
    )
    p.add_argument(
        "--variable",
        default="density",
        choices=["density", "vx", "pressure", "all"],
        help="Which field to predict.  Density has the cleanest shock structure.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_modes", type=int, default=16)
    p.add_argument("--wno_level", type=int, default=3)
    p.add_argument("--gate_kernel", type=int, default=5)
    p.add_argument("--lambda_ent", type=float, default=0.01)
    p.add_argument(
        "--output_dir",
        default="/media/HDD/mamta_backup/aw_fno_results/example_sod",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("example_sod")
    set_seed(args.seed)

    # ----- Data -----
    logger.info(f"Loading PDEBench Sod files from {args.data_path} ...")
    train_loader, test_loader, _, y_norm = load_pdebench_sod(
        data_path=args.data_path,
        batch_size=args.batch_size,
        variable=args.variable,
    )
    in_channels = train_loader.dataset.input_shape[0]
    spatial = train_loader.dataset.input_shape[-1]
    logger.info(
        f"  variable={args.variable}, in_channels={in_channels}, "
        f"resolution={spatial}, train={len(train_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    # ----- Model -----
    model = build_model(
        "awfno",
        in_channels=in_channels,
        out_channels=in_channels,
        n_modes=(args.n_modes,),
        size=(spatial,),
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        wno_level=args.wno_level,
        wno_wavelet="db6",
        positional_embedding="grid",
        padding=0,
        dropout=0.0,
        norm="layer_norm",
    )
    # Apply Mitigation C — rich gate (kernel=5, 2 layers).  This is the
    # winning configuration from Phase 2.5 Burgers experiments.
    _patch_rich_gate(model, kernel_size=args.gate_kernel, hidden_layers=2)
    logger.info(f"Model built: AW-FNO + rich gate (k={args.gate_kernel}, 2 layers), "
                f"params={count_parameters(model):,}")

    # ----- Optimiser + scheduler -----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # ----- Trainer -----
    trainer = OperatorTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=LpLoss(p=2),
        y_normalizer=y_norm,
        device=args.device,
        output_dir=args.output_dir,
        amp=False,
        grad_clip=1.0,
        log_every=10,
        save_every=25,
        experiment_name="example_sod",
        lambda_ent=args.lambda_ent,    # entropy penalty (Mitigation A)
    )

    history = trainer.fit(train_loader, test_loader, epochs=args.epochs)

    # ----- Summary -----
    best_l2 = trainer._best_rel_l2
    final_H = history["gate_entropy"][-1] if history["gate_entropy"] else float("nan")
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Example complete")
    logger.info(f"    best test rel_l2: {best_l2:.4f}")
    logger.info(f"    final gate_H:     {final_H:.4f}  (uniform = 0.6931)")
    logger.info(f"    output_dir:       {args.output_dir}")
    logger.info("")
    logger.info("  Next: run scripts/analyze_gate_burgers.py with --task=next_step")
    logger.info("        and config pointing at this checkpoint to see gate")
    logger.info("        routing at the shock + contact + rarefaction features.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
