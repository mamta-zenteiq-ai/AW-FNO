#!/usr/bin/env bash
# PDEBench 1D Riemann (shock-tube) training queue — v2-plan Stage 3.
#
# The multi-wave gate-routing dataset: each sample develops a shock + contact
# discontinuity + rarefaction fan, the richest spatial-routing target. Trains
# the focused 4-run set (FNO, WNO, fixed-alpha control, rich-gate headline)
# sequentially on a single GPU. 1D => fast (~1h/run).
#
# Usage:
#   bash scripts/run_riemann_queue.sh            # uses GPU 0
#   GPU=1 bash scripts/run_riemann_queue.sh      # pick a different GPU
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
    echo "  output:  $OUT/$name"
    echo "  log:     $LOG/$name.log"
    echo "=================================================="
    $PYTHON experiments/train.py \
        --config "$cfg" \
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

# Order: cheap baselines first to surface bugs early; rich gate (the multi-wave
# headline) last.
run_one fno_riemann            configs/experiment/train_fno_riemann.yaml
run_one wno_riemann            configs/experiment/train_wno_riemann.yaml
run_one awfno_riemann_no_gate  configs/experiment/ablation_no_gate_riemann.yaml
run_one awfno_riemann_richgate configs/experiment/train_awfno_riemann_richgate.yaml

echo "[$(date +%T)] PDEBench Riemann queue complete."
