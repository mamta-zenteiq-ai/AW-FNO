#!/usr/bin/env bash
# JHTDB isotropic-turbulence 128² super-resolution queue — v2-plan Stage 4.
#
# Second SR benchmark on the identical 128²/4× pipeline as NS-forcing, on
# high-Re forced isotropic turbulence (vortex-filament routing target). Trains
# the focused 4-run set (FNO, WNO, fixed-alpha control, rich-gate headline).
# 2D => ~9h per AW-FNO run, so this queue is long (~24h).
#
# Usage:
#   bash scripts/run_jhtdb_queue.sh            # uses GPU 0
#   GPU=1 bash scripts/run_jhtdb_queue.sh
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

run_one fno_jhtdb            configs/experiment/train_fno_jhtdb.yaml
run_one wno_jhtdb            configs/experiment/train_wno_jhtdb.yaml
run_one awfno_jhtdb_no_gate  configs/experiment/ablation_no_gate_jhtdb.yaml
run_one awfno_jhtdb_richgate configs/experiment/train_awfno_jhtdb_richgate.yaml

echo "[$(date +%T)] JHTDB SR queue complete."
