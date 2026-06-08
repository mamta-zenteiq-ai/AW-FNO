#!/usr/bin/env bash
# Spatial-gate experiment: force genuine SPATIAL routing.
#
# The per-channel rich gate satisfied the entropy penalty by committing along
# the CHANNEL axis (a near-static FNO/WNO split, spatially flat, ρ≈0 on NS
# turbulence). This queue trains the single-channel spatial gate (α(x,y) shared
# across channels) so the gate can only be decisive SPATIALLY:
#
#   1. Burgers ν=1e-3 (1D, ~1h)  — shock control: spatial routing SHOULD emerge
#                                   (ρ((1-α),|∂u/∂x|) > 0 at shock fronts).
#   2. NS-forcing 128² SR (~9h)  — the key test: does forcing a spatial gate
#                                   make it route on homogeneous turbulence, and
#                                   at what accuracy vs the rich gate (0.0090)?
#
# Usage:  GPU=0 bash scripts/run_spatial_gate_queue.sh
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

run_one awfno_pdebench_burgers_spatialgate configs/experiment/train_awfno_pdebench_burgers_spatialgate.yaml
run_one awfno_nsforcing_sr_spatialgate     configs/experiment/train_awfno_nsforcing_spatialgate.yaml

echo "[$(date +%T)] Spatial-gate queue complete."
