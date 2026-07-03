#!/usr/bin/env bash
# Download the extra PDEBench files for the strong-paper datasets, sequentially
# (resume-capable). DARUS dataset doi:10.18419/darus-2986; files are fetched by
# numeric datafile id via the dataverse access API (redirects to S3).
#
#   Stage 3 (Riemann / multi-wave): 1D_CFD_Shock ... trans_Train.hdf5  (~20 GB)
#   Stage 2 (Burgers sharpness sweep contrast point): Nu0.01.hdf5      (~8 GB)
#
# NOTE: PDEBench has NO Burgers Nu=1e-4 (lowest is Nu=0.001, already on disk),
# so the planned "sharper shock" is replaced by a sharp-vs-smooth sweep
# (Nu=0.001 sharp [have] vs Nu=0.01 smoother [this download]).
set -u

CFD=/media/HDD/mamta_backup/datasets/PDEBench/cfd
BURG=/media/HDD/mamta_backup/datasets/PDEBench/burgers
LOG=/home/mamta/Projects/AW-FNO/logs
mkdir -p "$CFD" "$BURG" "$LOG"

API=https://darus.uni-stuttgart.de/api/access/datafile

dl () {  # dl <id> <dest>
    local id=$1 dest=$2
    echo "[$(date +%T)] downloading id=$id -> $dest"
    wget -c --tries=20 --retry-connrefused --waitretry=10 \
         "$API/$id" -O "$dest"
    echo "[$(date +%T)] done id=$id rc=$? ($(du -h "$dest" 2>/dev/null | cut -f1))"
}

dl 133156 "$CFD/1D_CFD_Shock_Eta1.e-8_Zeta1.e-8_trans_Train.hdf5"
dl 281363 "$BURG/1D_Burgers_Sols_Nu0.01.hdf5"

echo "[$(date +%T)] all extra PDEBench downloads complete."
