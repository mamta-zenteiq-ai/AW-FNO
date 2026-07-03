#!/usr/bin/env bash
# Gate kernel-size sweep on PDEBench Burgers Nu=0.001 — v2-plan ablation B3.
#
# Trains AW-FNO rich gate at k=3, 7, 9 (k=5 is the already-completed baseline
# at awfno_pdebench_burgers_richgate). Tests whether a wider gate receptive
# field strengthens the spatial shock-routing correlation ρ((1-α),|∂u/∂x|).
# 1D Burgers => ~1h per run.
#
# Usage:  GPU=0 bash scripts/run_kernel_sweep_queue.sh
set -u

PROJECT=/home/mamta/Projects/AW-FNO
PYTHON=/home/mamta/miniconda3/envs/neuraloperator-official/bin/python
OUT=/media/HDD/mamta_backup/aw_fno_results
LOG=$PROJECT/logs

cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONPATH="$PROJECT"
mkdir -p "$LOG"

run_one() {
    local name=$1 cfg=$2
    echo "=================================================="
    echo "[$(date +%T)] START: $name"
    echo "  config:  $cfg"
    echo "=================================================="
    $PYTHON experiments/train.py --config "$cfg" --output_dir "$OUT/$name" \
        > "$LOG/$name.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && echo "[$(date +%T)] DONE : $name" \
        || echo "[$(date +%T)] FAIL : $name (exit $rc) — see $LOG/$name.log"
    echo
}

run_one awfno_pdebench_burgers_richgate_k3 configs/experiment/train_awfno_pdebench_burgers_richgate_k3.yaml
run_one awfno_pdebench_burgers_richgate_k7 configs/experiment/train_awfno_pdebench_burgers_richgate_k7.yaml
run_one awfno_pdebench_burgers_richgate_k9 configs/experiment/train_awfno_pdebench_burgers_richgate_k9.yaml

echo "[$(date +%T)] Kernel-sweep queue complete."
