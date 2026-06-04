#!/usr/bin/env bash
# NS-forcing 128² super-resolution training queue — the paper's PRIMARY task.
#
# This is Stage A1 of docs/paper_completion_plan.md. Trains the full SR model
# family sequentially on a single GPU so it can run hands-free overnight.
#
# The bicubic "no-model" baseline is NOT trained here: in this dataset the
# model input x is already the bicubic-upsampled LR field, so the bicubic
# baseline is just rel_l2(x, y) and is computed at evaluation time (see
# experiments/evaluate.py / the aggregation script), not via training.
#
# Usage:
#   bash scripts/run_sr_queue.sh                # uses GPU 0
#   GPU=1 bash scripts/run_sr_queue.sh          # pick a different GPU
set -u

PROJECT=/home/mamta/Projects/AW-FNO
PYTHON=/home/mamta/miniconda3/envs/neuraloperator-official/bin/python
# nsforcing_{train,test}_128.pt live here (loader joins data_path + train_file)
DATA=/media/HDD/mamta_backup/datasets/fno/navier_stokes
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

# Order: cheap baselines first to surface bugs early; rich gate (the primary
# contribution) last.
run_one fno_nsforcing_sr            configs/experiment/train_fno_nsforcing.yaml
run_one wno_nsforcing_sr            configs/experiment/train_wno_nsforcing.yaml
run_one awfno_nsforcing_sr_no_gate  configs/experiment/ablation_no_gate_nsforcing.yaml
run_one awfno_nsforcing_sr          configs/experiment/train_awfno_nsforcing.yaml
run_one fno_fat_nsforcing_sr        configs/experiment/train_fno_fat_nsforcing.yaml
run_one awfno_nsforcing_sr_richgate configs/experiment/train_awfno_nsforcing_richgate.yaml

echo "[$(date +%T)] NS-forcing SR queue complete."
