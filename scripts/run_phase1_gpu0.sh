#!/usr/bin/env bash
# Phase 1 training queue for GPU 0.
# Runs the four trainings sequentially with outputs on HDD to avoid the
# root-disk space crisis.  Each training logs to logs/<exp>.log; failures in
# one run do not block the next (we want a complete table even with partial
# failures).
set -u

PROJECT=/home/mamta/Projects/AW-FNO
PYTHON=/home/mamta/miniconda3/envs/neuraloperator-official/bin/python
DATA=/media/HDD/mamta_backup/datasets/fno/navier_stokes
OUT=/media/HDD/mamta_backup/aw_fno_results
LOG=$PROJECT/logs

cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PROJECT"

run_one() {
    local name=$1 cfg=$2
    echo "=================================================="
    echo "[$(date +%T)] START: $name"
    echo "  config:  $cfg"
    echo "  output:  $OUT/$name"
    echo "  log:     $LOG/$name.log"
    echo "=================================================="
    $PYTHON experiments/train.py \
        --config "$cfg" \
        --data_path "$DATA" \
        --output_dir "$OUT/$name" \
        > "$LOG/$name.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date +%T)] DONE : $name"
    else
        echo "[$(date +%T)] FAIL : $name (exit $rc) — see $LOG/$name.log"
    fi
    echo
}

# Order: most important first.  AW-FNO is the primary result; the ablations
# are derivative of it.  WNO last because it's a baseline, not the main story.
run_one awfno_ns          configs/experiment/train_awfno_ns.yaml
run_one awfno_ns_no_gate  configs/experiment/ablation_no_gate.yaml
run_one awfno_ns_additive configs/experiment/ablation_additive.yaml
run_one wno_ns            configs/experiment/train_wno_ns.yaml

echo "[$(date +%T)] Phase 1 GPU 0 queue complete."
