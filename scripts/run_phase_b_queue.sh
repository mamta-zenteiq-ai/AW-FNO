#!/usr/bin/env bash
# Phase-B training queue for the strong-paper datasets, run AFTER the SR queue
# (scripts/run_sr_queue.sh) frees GPU 0. Sequential, single-GPU, hands-free.
#
#   Stage 3  1D CFD shock-tube (Sod/Riemann, multi-wave) : FNO, WNO, AW-FNO rich, fixed-alpha
#   Stage 4  JHTDB isotropic SR (2D)                       : FNO, WNO, AW-FNO rich, fixed-alpha
#
# The 1D CFD shock-tube experiment is the genuine-shock case ("Burgers with
# real shocks") and supersedes the dropped Burgers Nu=0.01 viscosity sweep
# (PDEBench has no Nu=1e-4; the shock-tube is a stronger shock story anyway).
#
# Order: cheap 1D runs first (~1h each) to surface bugs, then the 2D JHTDB runs.
#
# Guards:
#   * refuses to start if a trainer (experiments/train.py) is still running
#     (i.e. the SR queue has not finished) -- prevents GPU collision;
#   * skips a dataset whose download is incomplete.
#
# Usage:
#   bash scripts/run_phase_b_queue.sh            # GPU 0
#   GPU=1 bash scripts/run_phase_b_queue.sh
set -u

PROJECT=/home/mamta/Projects/AW-FNO
PYTHON=/home/mamta/miniconda3/envs/neuraloperator-official/bin/python
OUT=/media/HDD/mamta_backup/aw_fno_results
LOG=$PROJECT/logs
CFD=/media/HDD/mamta_backup/datasets/PDEBench/cfd
BURG=/media/HDD/mamta_backup/datasets/PDEBench/burgers

cd "$PROJECT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONPATH="$PROJECT"
mkdir -p "$LOG"

# --- guard: do not collide with a running trainer (SR queue still going) ---
if pgrep -f "experiments/train.py" >/dev/null 2>&1; then
    echo "ABORT: a trainer (experiments/train.py) is still running."
    echo "       The SR queue must finish first. Re-run this when GPU 0 is free."
    exit 1
fi

file_ready () {  # file_ready <path> <min_bytes>
    local f=$1 min=$2
    [ -f "$f" ] && [ "$(stat -c%s "$f" 2>/dev/null || echo 0)" -ge "$min" ]
}

run_one() {
    local name=$1 cfg=$2 data=$3
    echo "=================================================="
    echo "[$(date +%T)] START: $name"
    echo "  config: $cfg"
    echo "  data:   $data"
    echo "  output: $OUT/$name"
    echo "=================================================="
    $PYTHON experiments/train.py \
        --config "$cfg" \
        --data_path "$data" \
        --output_dir "$OUT/$name" \
        > "$LOG/$name.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] && echo "[$(date +%T)] DONE : $name" \
                  || echo "[$(date +%T)] FAIL : $name (exit $rc) — see $LOG/$name.log"
    echo
}

# ---- Stage 3: Riemann (needs the ~20GB shock-tube file) ----
if file_ready "$CFD/1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5" 19000000000; then
    run_one fno_riemann            configs/experiment/train_fno_riemann.yaml            "$CFD"
    run_one wno_riemann            configs/experiment/train_wno_riemann.yaml            "$CFD"
    run_one awfno_riemann_no_gate  configs/experiment/ablation_no_gate_riemann.yaml     "$CFD"
    run_one awfno_riemann_richgate configs/experiment/train_awfno_riemann_richgate.yaml "$CFD"
else
    echo "SKIP Stage 3 (Riemann): 1D_CFD_Shock file not fully downloaded yet."
fi

# ---- Stage 4: JHTDB isotropic SR (data already on disk) ----
JHTDB=/media/HDD/anjali/gazania_transolver/jhtdb_datasets
run_one fno_jhtdb_sr            configs/experiment/train_fno_jhtdb.yaml            "$JHTDB"
run_one wno_jhtdb_sr            configs/experiment/train_wno_jhtdb.yaml            "$JHTDB"
run_one awfno_jhtdb_sr_no_gate  configs/experiment/ablation_no_gate_jhtdb.yaml     "$JHTDB"
run_one awfno_jhtdb_sr_richgate configs/experiment/train_awfno_jhtdb_richgate.yaml "$JHTDB"

echo "[$(date +%T)] Phase-B queue complete."
