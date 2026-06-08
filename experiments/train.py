#!/usr/bin/env python3
"""
Unified training entry-point for all AW-FNO experiments.

Usage examples:
    # Train AW-FNO on Navier-Stokes (default config)
    python experiments/train.py --config configs/experiment/train_awfno_ns.yaml

    # Override individual settings via CLI
    python experiments/train.py --config configs/experiment/train_fno_ns.yaml \\
        --epochs 500 --lr 1e-3 --output_dir results/fno_ns_run2

    # Ablation: fixed-gate AW-FNO
    python experiments/train.py --config configs/experiment/ablation_no_gate.yaml

    # Use local data path (no reliance on hardcoded paths)
    python experiments/train.py --config configs/experiment/train_awfno_ns.yaml \\
        --data_path /media/HDD/mamta_backup/datasets/fno/navier_stokes
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

# Make project root importable regardless of cwd
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from awfno.models import build_model, count_parameters
from awfno.utils.seed import set_seed
from awfno.data.ns2d import load_ns2d
from awfno.data.burgers1d import load_burgers1d
from awfno.data.nsforcing_sr import load_nsforcing_sr
from awfno.data.pdebench_burgers import load_pdebench_burgers
from awfno.data.pdebench_compns import load_pdebench_sod, load_pdebench_cfd1d
from awfno.data.jhtdb_iso import load_jhtdb_iso
from awfno.training.operator_trainer import OperatorTrainer
from awfno.losses import LpLoss
from awfno.utils.logging import get_logger


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict, overrides: dict) -> dict:
    result = {**base}
    result.update({k: v for k, v in overrides.items() if v is not None})
    return result


def load_config(exp_yaml: str, cli_overrides: dict) -> dict:
    """Load experiment YAML, resolve model + dataset sub-configs, apply CLI overrides."""
    exp = _load_yaml(exp_yaml)

    # Load model sub-config
    model_name = exp.get("model", "awfno_ns")
    model_yaml = _ROOT / "configs" / "model" / f"{model_name}.yaml"
    model_cfg = _load_yaml(str(model_yaml)) if model_yaml.exists() else {}

    # Load dataset sub-config
    ds_name = exp.get("dataset", "ns2d")
    ds_yaml = _ROOT / "configs" / "dataset" / f"{ds_name}.yaml"
    ds_cfg = _load_yaml(str(ds_yaml)) if ds_yaml.exists() else {}

    cfg = {**exp, "model_cfg": model_cfg, "dataset_cfg": ds_cfg}
    cfg = _merge(cfg, cli_overrides)
    return cfg


# ---------------------------------------------------------------------------
# Dataset loader dispatcher
# ---------------------------------------------------------------------------

def load_dataset(cfg: dict):
    ds_cfg = cfg.get("dataset_cfg", {})
    ds_name = ds_cfg.get("name", cfg.get("dataset", "ns2d"))

    # Allow CLI / env override of data path
    data_path = (
        cfg.get("data_path")
        or os.environ.get("DATA_PATH")
        or ds_cfg.get("data_path", f"data/{ds_name}")
    )

    if ds_name == "ns2d":
        return load_ns2d(
            data_path=data_path,
            batch_size=ds_cfg.get("batch_size", 20),
            T_in=ds_cfg.get("T_in", 10),
            T_out=ds_cfg.get("T_out", 10),
            n_train=ds_cfg.get("n_train", 1000),
            n_test=ds_cfg.get("n_test", 200),
            seed=ds_cfg.get("seed", 42),
        )
    elif ds_name in ("burgers1d", "burgers"):
        return load_burgers1d(
            data_path=data_path,
            batch_size=ds_cfg.get("batch_size", 20),
            n_train=ds_cfg.get("n_train", 1000),
            n_test=ds_cfg.get("n_test", 200),
            resolution=ds_cfg.get("resolution", 1024),
            seed=ds_cfg.get("seed", 42),
        )
    elif ds_name in ("nsforcing_sr", "nsforcing"):
        return load_nsforcing_sr(
            data_path=data_path,
            batch_size=ds_cfg.get("batch_size", 16),
            downsample_factor=ds_cfg.get("downsample_factor", 4),
            n_train=ds_cfg.get("n_train", 8000),
            n_test=ds_cfg.get("n_test", 2000),
            snapshot=ds_cfg.get("snapshot", "y"),
            seed=ds_cfg.get("seed", 42),
        )
    elif ds_name in ("pdebench_burgers", "pdebench_burgers1d"):
        return load_pdebench_burgers(
            data_path=data_path,
            batch_size=ds_cfg.get("batch_size", 64),
            task=ds_cfg.get("task", "initial_to_final"),
            n_train=ds_cfg.get("n_train", 9000),
            n_test=ds_cfg.get("n_test", 1000),
            train_frac=ds_cfg.get("train_frac", 0.9),
            viscosity_tag=ds_cfg.get("viscosity_tag", "0.001"),
            seed=ds_cfg.get("seed", 42),
        )
    elif ds_name in ("pdebench_sod", "pdebench_compns", "sod1d"):
        return load_pdebench_sod(
            data_path=data_path,
            batch_size=ds_cfg.get("batch_size", 16),
            variable=ds_cfg.get("variable", "density"),
            test_file_indices=ds_cfg.get("test_file_indices", None),
        )
    elif ds_name in ("pdebench_riemann", "pdebench_cfd1d", "riemann1d"):
        return load_pdebench_cfd1d(
            data_path=data_path,
            batch_size=ds_cfg.get("batch_size", 32),
            variable=ds_cfg.get("variable", "density"),
            task=ds_cfg.get("task", "initial_to_final"),
            n_train=ds_cfg.get("n_train", 9000),
            n_test=ds_cfg.get("n_test", 1000),
            train_frac=ds_cfg.get("train_frac", 0.9),
            file_glob=ds_cfg.get("file_glob", "1D_CFD_Shock*Train.hdf5"),
            seed=ds_cfg.get("seed", 42),
        )
    elif ds_name in ("jhtdb_iso", "jhtdb_iso_sr", "jhtdb"):
        return load_jhtdb_iso(
            data_path=data_path,
            batch_size=ds_cfg.get("batch_size", 16),
            downsample_factor=ds_cfg.get("downsample_factor", 4),
            patch_size=ds_cfg.get("patch_size", 128),
            n_train=ds_cfg.get("n_train", 8000),
            n_test=ds_cfg.get("n_test", 2000),
            test_chunk_frac=ds_cfg.get("test_chunk_frac", 0.2),
            components=ds_cfg.get("components", (0, 1, 2)),
            seed=ds_cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")


# ---------------------------------------------------------------------------
# Model builder from config
# ---------------------------------------------------------------------------

def build_from_config(cfg: dict, ablation_fixed_gate: bool = False) -> torch.nn.Module:
    model_cfg = cfg.get("model_cfg", {})
    name = model_cfg.get("name", "awfno")

    # Translate YAML keys to model constructor kwargs.  `n_modes` is required
    # only for Fourier-based models (FNO / AWFNO); WNO uses wavelet `level`
    # instead and is initialised from a different kwarg set.
    kwargs = {
        "in_channels": model_cfg.get("in_channels", 1),
        "out_channels": model_cfg.get("out_channels", 1),
        "hidden_channels": model_cfg.get("hidden_channels", 32),
        "n_layers": model_cfg.get("n_layers", 4),
        "positional_embedding": model_cfg.get("positional_embedding", "grid"),
        "padding": model_cfg.get("padding", 0),
        "dropout": model_cfg.get("dropout", 0.0),
    }
    if "n_modes" in model_cfg:
        kwargs["n_modes"] = tuple(model_cfg["n_modes"])

    if name in ("awfno", "awfno_v2"):
        kwargs["size"] = tuple(model_cfg["size"])
        kwargs["wno_level"] = model_cfg.get("wno_level", 3)
        kwargs["wno_wavelet"] = model_cfg.get("wno_wavelet", "db6")
        kwargs["norm"] = model_cfg.get("norm", "layer_norm")
        kwargs["lifting_channel_ratio"] = model_cfg.get("lifting_channel_ratio", 2)
        kwargs["projection_channel_ratio"] = model_cfg.get("projection_channel_ratio", 2)

    elif name == "wno":
        kwargs["size"] = tuple(model_cfg["size"])
        kwargs["width"] = model_cfg.get("width", model_cfg.get("hidden_channels", 32))
        kwargs["level"] = model_cfg.get("level", model_cfg.get("wno_level", 2))
        kwargs["wavelet"] = model_cfg.get("wavelet", model_cfg.get("wno_wavelet", "db4"))
        # WNO doesn't accept these; build_model filters its own kwargs but
        # we drop them here too for safety.
        for k in ("hidden_channels", "positional_embedding", "dropout"):
            kwargs.pop(k, None)

    elif name == "fno":
        kwargs["use_channel_mlp"] = model_cfg.get("use_channel_mlp", True)
        kwargs["norm"] = model_cfg.get("norm", None)
        # FNO doesn't accept `padding` / `dropout` — drop them.
        kwargs.pop("padding", None)
        kwargs.pop("dropout", None)
        # Optional FNO-specific kwargs (only forwarded if set in the config)
        for k in (
            "channel_mlp_expansion", "channel_mlp_dropout", "channel_mlp_skip",
            "fno_skip", "domain_padding", "fno_block_precision", "stabilizer",
        ):
            if k in model_cfg:
                kwargs[k] = model_cfg[k]

    model = build_model(name=name, **kwargs)

    # Ablation: replace adaptive gate with fixed-weight average
    if ablation_fixed_gate and name == "awfno":
        _patch_fixed_gate(model)

    # Ablation: replace adaptive gate with plain additive fusion (v_fno + v_wno)
    if cfg.get("ablation_additive_fusion", False) and name == "awfno":
        _patch_additive_fusion(model)

    # Mitigation C: richer gate (multi-layer Conv with spatial receptive field).
    # Used to test whether the 1×1 gate's failure to learn spatial routing
    # is a capacity issue (no spatial context) or fundamental.
    rich_cfg = cfg.get("rich_gate", None)
    if rich_cfg and name == "awfno":
        kernel_size = int(rich_cfg.get("kernel_size", 5))
        hidden_layers = int(rich_cfg.get("hidden_layers", 2))
        _patch_rich_gate(model, kernel_size=kernel_size, hidden_layers=hidden_layers)

    # Spatial gate: a single-channel α(x,y) shared across all feature channels.
    # The per-channel rich gate can satisfy the entropy penalty by committing
    # along the CHANNEL axis (a near-static FNO/WNO split, spatially flat); a
    # single output channel removes that escape, so the only way to be decisive
    # is to vary α spatially — i.e. genuine spatial routing.
    spatial_cfg = cfg.get("spatial_gate", None)
    if spatial_cfg and name == "awfno":
        kernel_size = int(spatial_cfg.get("kernel_size", 5))
        hidden_layers = int(spatial_cfg.get("hidden_layers", 2))
        _patch_spatial_gate(model, kernel_size=kernel_size, hidden_layers=hidden_layers)

    return model


def _patch_fixed_gate(model: torch.nn.Module) -> None:
    """Replace AdaptiveGatedFusion with a fixed equal-weight average (no-gate ablation)."""
    import torch.nn as nn

    class _FixedFusion(nn.Module):
        def forward(self, v_fno, v_wno):
            return 0.5 * v_fno + 0.5 * v_wno

    for block in model.blocks:
        if hasattr(block, "gfm"):
            block.gfm = _FixedFusion()

    print("  [Ablation] Replaced AdaptiveGatedFusion with fixed 0.5/0.5 mix.")


def _patch_additive_fusion(model: torch.nn.Module) -> None:
    """
    Replace AdaptiveGatedFusion with unconstrained additive fusion: v_fno + v_wno.

    Unlike the fixed-0.5 variant (which constrains the two branches to sum
    to 1), this lets both branches contribute freely.  Together with the
    fixed-0.5 ablation, this isolates the value of:
      (i)   constraining the branch outputs to a convex combination, and
      (ii)  making the combination spatially adaptive.
    """
    import torch.nn as nn

    class _AdditiveFusion(nn.Module):
        def forward(self, v_fno, v_wno):
            return v_fno + v_wno

    for block in model.blocks:
        if hasattr(block, "gfm"):
            block.gfm = _AdditiveFusion()

    print("  [Ablation] Replaced AdaptiveGatedFusion with additive fusion (v_fno + v_wno).")


def _patch_rich_gate(
    model: torch.nn.Module,
    kernel_size: int = 5,
    hidden_layers: int = 2,
) -> None:
    """
    Replace the 1×1 gate inside AdaptiveGatedFusion with a multi-layer
    convolutional gate that has spatial receptive field.

    Architecture (1D, analogous for 2D):
        Conv_k(2C → C) → GELU → Conv_k(C → C) → Sigmoid          (hidden_layers=2)
        Conv_k(2C → C) → Sigmoid                                  (hidden_layers=1)

    Effective receptive field for hidden_layers=2 with kernel=5:
        per layer: ±2 pixels; stacked: ±4 pixels (covers 9-pixel window).
        Shock width in PDEBench Burgers ν=1e-3 is ~5–10 pixels, so this
        receptive field is sufficient to detect shock signatures.

    The new conv layers are initialised with the same small-random scheme as
    the original 1×1 gate (std=0.2 for symmetry-breaking).
    """
    import torch.nn as nn

    n_patched = 0
    for block in model.blocks:
        if not hasattr(block, "gfm"):
            continue
        gfm = block.gfm
        existing_gate = getattr(gfm, "gate", None)
        if existing_gate is None:
            continue

        # The existing first layer is a Conv1d or Conv2d we can introspect.
        first = existing_gate[0]
        if isinstance(first, nn.Conv1d):
            ConvCls = nn.Conv1d
        elif isinstance(first, nn.Conv2d):
            ConvCls = nn.Conv2d
        else:
            continue

        c_out = first.out_channels      # = channels (per-channel α)
        c_in = first.in_channels        # = 2 * channels (concat of FNO, WNO feats)
        pad = kernel_size // 2

        if hidden_layers == 1:
            new_conv = ConvCls(c_in, c_out, kernel_size=kernel_size, padding=pad)
            nn.init.normal_(new_conv.weight, mean=0.0, std=0.2)
            nn.init.constant_(new_conv.bias, 0)
            new_gate = nn.Sequential(new_conv, nn.Sigmoid())
        else:
            l1 = ConvCls(c_in, c_out, kernel_size=kernel_size, padding=pad)
            l2 = ConvCls(c_out, c_out, kernel_size=kernel_size, padding=pad)
            nn.init.normal_(l1.weight, mean=0.0, std=0.2)
            nn.init.constant_(l1.bias, 0)
            nn.init.normal_(l2.weight, mean=0.0, std=0.2)
            nn.init.constant_(l2.bias, 0)
            new_gate = nn.Sequential(l1, nn.GELU(), l2, nn.Sigmoid())

        gfm.gate = new_gate
        n_patched += 1

    print(
        f"  [Mitigation C] Replaced 1×1 gate with kernel={kernel_size}, "
        f"hidden_layers={hidden_layers} Conv gate in {n_patched} blocks."
    )


def _patch_spatial_gate(
    model: torch.nn.Module,
    kernel_size: int = 5,
    hidden_layers: int = 2,
) -> None:
    """
    Replace the gate with a multi-layer convolutional gate whose FINAL layer
    outputs a SINGLE channel, i.e. α ∈ R^{B×1×H×W} broadcast over all feature
    channels in the fusion (α·v_fno + (1−α)·v_wno).

    Motivation: the per-channel rich gate satisfies the entropy penalty by
    committing along the channel axis (a near-static FNO/WNO subspace split
    that is spatially flat, so ρ((1−α),|∇ω|)≈0 on homogeneous turbulence).
    Collapsing the gate to one channel removes that degree of freedom — the
    only way for the gate to become decisive is to vary α(x,y) spatially, which
    forces genuine spatial routing. This is the channel-shared spatial gate of
    the original architecture (α ∈ R^{B×H×W×1}), but with the rich gate's
    spatial receptive field.

    Architecture (2D; 1D analogous):
        Conv_k(2C → C) → GELU → Conv_k(C → 1) → Sigmoid          (hidden_layers=2)
        Conv_k(2C → 1) → Sigmoid                                  (hidden_layers=1)
    """
    import torch.nn as nn

    n_patched = 0
    for block in model.blocks:
        if not hasattr(block, "gfm"):
            continue
        gfm = block.gfm
        existing_gate = getattr(gfm, "gate", None)
        if existing_gate is None:
            continue

        first = existing_gate[0]
        if isinstance(first, nn.Conv1d):
            ConvCls = nn.Conv1d
        elif isinstance(first, nn.Conv2d):
            ConvCls = nn.Conv2d
        else:
            continue

        c_in = first.in_channels        # = 2 * channels
        c_mid = first.out_channels      # = channels
        pad = kernel_size // 2

        if hidden_layers == 1:
            new_conv = ConvCls(c_in, 1, kernel_size=kernel_size, padding=pad)
            nn.init.normal_(new_conv.weight, mean=0.0, std=0.2)
            nn.init.constant_(new_conv.bias, 0)
            new_gate = nn.Sequential(new_conv, nn.Sigmoid())
        else:
            l1 = ConvCls(c_in, c_mid, kernel_size=kernel_size, padding=pad)
            l2 = ConvCls(c_mid, 1, kernel_size=kernel_size, padding=pad)
            nn.init.normal_(l1.weight, mean=0.0, std=0.2)
            nn.init.constant_(l1.bias, 0)
            nn.init.normal_(l2.weight, mean=0.0, std=0.2)
            nn.init.constant_(l2.bias, 0)
            new_gate = nn.Sequential(l1, nn.GELU(), l2, nn.Sigmoid())

        gfm.gate = new_gate
        n_patched += 1

    print(
        f"  [Spatial gate] Replaced gate with single-channel kernel={kernel_size}, "
        f"hidden_layers={hidden_layers} Conv gate in {n_patched} blocks "
        f"(α shared across channels → forces spatial routing)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AW-FNO / FNO / WNO")
    p.add_argument("--config", required=True, help="Path to experiment YAML config")
    p.add_argument("--data_path", default=None, help="Override data directory path")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, dest="learning_rate")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, choices=["cuda", "cpu", "auto"])
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    cfg = load_config(args.config, overrides)

    logger = get_logger("train")
    logger.info(f"Config: {args.config}")

    # Seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # Data
    logger.info("Loading dataset ...")
    train_loader, test_loader, x_norm, y_norm = load_dataset(cfg)
    logger.info(f"  train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")

    # Model
    ablation = cfg.get("ablation_fixed_gate", False)
    model = build_from_config(cfg, ablation_fixed_gate=ablation)
    n_params = count_parameters(model)
    logger.info(f"Model: {cfg['model_cfg'].get('name', '?')} | params={n_params:,}")

    # Optimiser & scheduler
    lr = cfg.get("learning_rate", 1e-3)
    wd = cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    sched_step = cfg.get("scheduler_step_size", 100)
    sched_gamma = cfg.get("scheduler_gamma", 0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=sched_step, gamma=sched_gamma
    )

    output_dir = cfg.get("output_dir", f"results/{cfg.get('experiment_name', 'run')}")

    trainer = OperatorTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=LpLoss(p=2),
        y_normalizer=y_norm,
        device=cfg.get("device", "auto"),
        output_dir=output_dir,
        amp=cfg.get("amp", False),
        grad_clip=cfg.get("grad_clip", 1.0),
        log_every=cfg.get("log_every", 50),
        save_every=cfg.get("save_every", 100),
        use_wandb=cfg.get("use_wandb", False) or args.use_wandb,
        experiment_name=cfg.get("experiment_name", "experiment"),
        lambda_ent=cfg.get("lambda_ent", 0.0),
    )

    epochs = cfg.get("epochs", 500)
    history = trainer.fit(train_loader, test_loader, epochs=epochs)

    # Save final checkpoint
    trainer._save_checkpoint("final.pt", epochs, {"experiment_name": cfg.get("experiment_name")})
    logger.info(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
