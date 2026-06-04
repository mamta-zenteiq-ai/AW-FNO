#!/usr/bin/env bash
# PDEBench Burgers Nu=0.001 training queue — gate-validation experiment.
#
# Hypothesis: on shock-dominated data, the gate should clearly identify the
# shock front (high |∂u/∂x|) and route to WNO there, beating fixed α=0.5.
set -u

PROJECT=/home/mamta/Projects/AW-FNO
PYTHON=/home/mamta/miniconda3/envs/neuraloperator-official/bin/python
DATA=/media/HDD/mamta_backup/datasets/PDEBench/burgers
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

# Order: cheapest baselines first to surface bugs early; gate variants last.
run_one fno_pdebench_burgers              configs/experiment/train_fno_pdebench_burgers.yaml
run_one wno_pdebench_burgers              configs/experiment/train_wno_pdebench_burgers.yaml
run_one awfno_pdebench_burgers_no_gate    configs/experiment/ablation_no_gate_pdebench_burgers.yaml
run_one awfno_pdebench_burgers_additive   configs/experiment/ablation_additive_pdebench_burgers.yaml
run_one awfno_pdebench_burgers            configs/experiment/train_awfno_pdebench_burgers.yaml

echo "[$(date +%T)] PDEBench Burgers queue complete."
