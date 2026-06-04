#!/usr/bin/env bash
# Phase 1 follow-up queue (GPU 0):
#   1. AW-FNO with mitigated gate (random init + entropy penalty λ=0.01)
#   2. WNO retry (after fixing the build_from_config `n_modes` bug)
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

run_one awfno_ns_fixed configs/experiment/train_awfno_ns_fixed.yaml
run_one wno_ns         configs/experiment/train_wno_ns.yaml

echo "[$(date +%T)] Phase 1 retry queue complete."
