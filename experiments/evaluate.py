#!/usr/bin/env python3
"""
Evaluate a trained model on the test set and report all metrics.

Usage:
    python experiments/evaluate.py \\
        --checkpoint results/awfno_ns/best.pt \\
        --config configs/experiment/train_awfno_ns.yaml \\
        --data_path /media/HDD/mamta_backup/datasets/fno/navier_stokes

Output:
    - Metrics table printed to stdout
    - metrics.json saved alongside the checkpoint
    - Field comparison figure saved to the checkpoint directory
"""

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from experiments.train import load_config, build_from_config, load_dataset
from awfno.utils.seed import set_seed
from awfno.metrics import compute_metrics, MetricTracker
from awfno.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained neural operator")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--config", required=True, help="Experiment YAML config used for training")
    p.add_argument("--data_path", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--n_vis", type=int, default=4, help="Number of samples to visualise")
    p.add_argument("--save_figures", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, y_normalizer, device) -> dict:
    model.eval()
    tracker = MetricTracker()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        if y_normalizer is not None:
            pred = y_normalizer.decode(pred)
        tracker.update(compute_metrics(pred, y), n=x.shape[0])
    return tracker.mean()


def main() -> None:
    args = parse_args()
    logger = get_logger("evaluate")

    cfg = load_config(args.config, {"data_path": args.data_path})
    set_seed(cfg.get("seed", 42))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    model = build_from_config(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Data
    _, test_loader, x_norm, y_norm = load_dataset(cfg)
    if y_norm is not None:
        y_norm.to(device)

    # Evaluate
    logger.info(f"Evaluating on {len(test_loader.dataset)} test samples ...")
    metrics = evaluate(model, test_loader, y_norm, device)

    # Report
    print("\n" + "=" * 50)
    print(f"  Model:     {cfg['model_cfg'].get('name', '?')}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"  {k:<20s}: {v:.6f}")
    print("=" * 50 + "\n")

    # Save
    ckpt_dir = Path(args.checkpoint).parent
    out_path = ckpt_dir / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump({**metrics, "checkpoint": str(args.checkpoint)}, f, indent=2)
    logger.info(f"Metrics saved to {out_path}")

    # Optional visualisation
    if args.save_figures:
        try:
            from awfno.visualization import plot_field_comparison
            x_batch, y_batch = next(iter(test_loader))
            x_batch = x_batch[:args.n_vis].to(device)
            y_batch = y_batch[:args.n_vis].to(device)
            with torch.no_grad():
                pred_batch = model(x_batch)
                if y_norm:
                    pred_batch = y_norm.decode(pred_batch)
            model_name = cfg["model_cfg"].get("name", "model")
            fig_path = ckpt_dir / "field_comparison.png"
            plot_field_comparison(
                ground_truth=y_batch.cpu(),
                predictions={model_name: pred_batch.cpu()},
                title=f"{model_name} — Field Comparison",
                save_path=str(fig_path),
            )
            logger.info(f"Figure saved to {fig_path}")
        except Exception as e:
            logger.warning(f"Visualisation failed: {e}")


if __name__ == "__main__":
    main()
