#!/usr/bin/env bash
# =============================================================================
# reproduce_all.sh — Full end-to-end reproduction script for AW-FNO paper
#
# This script reproduces all results in the paper:
#   AW-FNO: An Adaptive approach for Fluid flow Super-resolution via
#   Gated Wavelet-Fourier Learning  (ICCFD13, 2026)
#
# Requirements:
#   - CUDA GPU with ≥ 8 GB VRAM (or set DEVICE=cpu for CPU-only)
#   - Python environment with dependencies installed (see requirements.txt)
#   - Dataset files accessible at DATA_PATH (see below)
#
# Usage:
#   # With data on local HDD:
#   DATA_PATH=/media/HDD/mamta_backup/datasets/fno/navier_stokes bash scripts/reproduce_all.sh
#
#   # After running download script:
#   bash scripts/reproduce_all.sh
#
#   # CPU-only (slow):
#   DEVICE=cpu bash scripts/reproduce_all.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
DATA_PATH="${DATA_PATH:-data/ns2d}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-500}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-50}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================================"
echo " AW-FNO Reproduction Script"
echo "============================================================"
echo " Project root : $PROJECT_ROOT"
echo " Data path    : $DATA_PATH"
echo " Device       : $DEVICE"
echo " Epochs       : $EPOCHS"
echo " Seed         : $SEED"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Step 0: Verify environment
# ---------------------------------------------------------------------------
echo "[0/6] Checking Python environment ..."
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
python -c "import pytorch_wavelets; print('  pytorch_wavelets OK')"
python -c "import tensorly; print('  tensorly OK')"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Download data (skip if already present)
# ---------------------------------------------------------------------------
echo "[1/6] Checking / downloading datasets ..."
python datasets/download_fno_data.py --check --dataset ns2d --root "$PROJECT_ROOT" || true
echo ""

# ---------------------------------------------------------------------------
# Step 2: Train baselines (FNO, WNO)
# ---------------------------------------------------------------------------
echo "[2/6] Training FNO baseline ..."
python experiments/train.py \
    --config configs/experiment/train_fno_ns.yaml \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --log_every "$LOG_EVERY"

echo ""
echo "[2/6] Training WNO baseline ..."
python experiments/train.py \
    --config configs/experiment/train_wno_ns.yaml \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --log_every "$LOG_EVERY"

echo ""

# ---------------------------------------------------------------------------
# Step 3: Train AW-FNO (proposed model)
# ---------------------------------------------------------------------------
echo "[3/6] Training AW-FNO (proposed) ..."
python experiments/train.py \
    --config configs/experiment/train_awfno_ns.yaml \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --log_every "$LOG_EVERY"

echo ""

# ---------------------------------------------------------------------------
# Step 4: Ablation — no-gate variant
# ---------------------------------------------------------------------------
echo "[4/6] Training AW-FNO ablation (no gate) ..."
python experiments/train.py \
    --config configs/experiment/ablation_no_gate.yaml \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --log_every "$LOG_EVERY"

echo ""

# ---------------------------------------------------------------------------
# Step 5: Benchmark — compare all models
# ---------------------------------------------------------------------------
echo "[5/6] Running benchmark comparison ..."
python experiments/benchmark.py \
    --dataset ns2d \
    --data_path "$DATA_PATH" \
    --device "$DEVICE" \
    --save_figures

echo ""

# ---------------------------------------------------------------------------
# Step 6: Generate paper figures and tables
# ---------------------------------------------------------------------------
echo "[6/6] Generating paper figures and LaTeX tables ..."
python scripts/generate_paper_figures.py \
    --data_path "$DATA_PATH" \
    --device "$DEVICE"

echo ""
echo "============================================================"
echo " Reproduction complete!"
echo " Results:   results/"
echo " Figures:   outputs/figures/"
echo " Tables:    outputs/tables/"
echo "============================================================"
