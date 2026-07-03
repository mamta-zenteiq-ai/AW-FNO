#!/usr/bin/env bash
# Master queue for the remaining v2-plan experiments, run in one sitting on a
# free GPU. Order = cheap/high-value first, longest last, so an early GPU
# reclaim still banks the quick wins:
#
#   1. efficiency table (B5)         ~1 min   — params + fwd latency, GPU numbers
#   2. spatial-gate experiment       ~10 h    — single-channel α: force SPATIAL
#                                               routing (Burgers control + NS SR)
#   3. Riemann 1D shock-tube (St.3)  ~4 h     — multi-wave gate-routing dataset
#   4. Burgers kernel sweep (B3)     ~3 h     — k=3,7,9 gate receptive-field sweep
#   5. JHTDB isotropic SR (St.4)     ~24 h    — high-Re turbulence SR benchmark
#
# Usage:  GPU=0 bash scripts/run_remaining_queue.sh
#   (normally launched via scripts/run_remaining_when_free.sh)
set -u

PROJECT=/home/mamta/Projects/AW-FNO
PYTHON=/home/mamta/miniconda3/envs/neuraloperator-official/bin/python
LOG=$PROJECT/logs
cd "$PROJECT"
export PYTHONPATH="$PROJECT"
GPU=${GPU:-0}
mkdir -p "$LOG"

echo "########################################################"
echo "[$(date '+%F %T')] REMAINING-EXPERIMENTS MASTER QUEUE on GPU $GPU"
echo "########################################################"

echo "[$(date +%T)] (1/5) efficiency table ..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/efficiency_table.py --device auto \
    > "$LOG/efficiency_table.log" 2>&1 \
    && echo "[$(date +%T)] efficiency table DONE" \
    || echo "[$(date +%T)] efficiency table FAILED — see $LOG/efficiency_table.log"

echo "[$(date +%T)] (2/5) spatial-gate experiment ..."
GPU=$GPU bash scripts/run_spatial_gate_queue.sh

echo "[$(date +%T)] (3/5) Riemann queue ..."
GPU=$GPU bash scripts/run_riemann_queue.sh

echo "[$(date +%T)] (4/5) Burgers kernel sweep ..."
GPU=$GPU bash scripts/run_kernel_sweep_queue.sh

echo "[$(date +%T)] (5/5) JHTDB queue ..."
GPU=$GPU bash scripts/run_jhtdb_queue.sh

echo "[$(date '+%F %T')] ALL REMAINING EXPERIMENTS COMPLETE."
