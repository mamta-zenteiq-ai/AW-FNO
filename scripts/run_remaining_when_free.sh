#!/usr/bin/env bash
# Guard launcher for the remaining-experiments master queue. Polls the GPUs and
# launches scripts/run_remaining_queue.sh only once a card is GENUINELY idle, so
# it never contends with another user's running job on this shared box.
#
# "Free" = BOTH hold for N_CONSECUTIVE consecutive polls:
#   * utilization <= UTIL_MAX %        (no other job actively computing)
#   * free memory >= FREE_MIN_MIB MiB  (genuine headroom)
#
# Launch detached:
#   nohup setsid bash scripts/run_remaining_when_free.sh \
#       > logs/remaining_queue_guard.log 2>&1 < /dev/null &
#
# To cancel before it fires:  pkill -f run_remaining_when_free
#
# Tunables (env overrides): FREE_MIN_MIB[10000] UTIL_MAX[10] N_CONSECUTIVE[3]
#                           POLL_SECS[120] MAX_WAIT_HRS[72]
set -u

PROJECT=/home/mamta/Projects/AW-FNO
FREE_MIN_MIB=${FREE_MIN_MIB:-10000}
UTIL_MAX=${UTIL_MAX:-10}
N_CONSECUTIVE=${N_CONSECUTIVE:-3}
POLL_SECS=${POLL_SECS:-120}
MAX_WAIT_HRS=${MAX_WAIT_HRS:-72}

cd "$PROJECT"
deadline=$(( $(date +%s) + MAX_WAIT_HRS * 3600 ))

echo "[$(date '+%F %T')] remaining-experiments guard started."
echo "  launch when a GPU has >= ${FREE_MIN_MIB} MiB free AND <= ${UTIL_MAX}% util"
echo "  for ${N_CONSECUTIVE} consecutive polls (every ${POLL_SECS}s; give up after ${MAX_WAIT_HRS}h)."

declare -A streak
while :; do
    ready_gpu=""
    while IFS=',' read -r idx util free; do
        idx="${idx// /}"; util="${util// /}"; free="${free// /}"
        [ -z "${idx:-}" ] && continue
        if [ "${util:-100}" -le "$UTIL_MAX" ] && [ "${free:-0}" -ge "$FREE_MIN_MIB" ]; then
            streak[$idx]=$(( ${streak[$idx]:-0} + 1 ))
        else
            streak[$idx]=0
        fi
        if [ -z "$ready_gpu" ] && [ "${streak[$idx]}" -ge "$N_CONSECUTIVE" ]; then
            ready_gpu="$idx"
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.free --format=csv,noheader,nounits 2>/dev/null)

    if [ -n "$ready_gpu" ]; then
        read -r util free < <(nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits -i "$ready_gpu")
        echo "[$(date '+%F %T')] GPU $ready_gpu ready (${util}% util, ${free} MiB free). Launching master queue."
        GPU="$ready_gpu" bash scripts/run_remaining_queue.sh
        rc=$?
        echo "[$(date '+%F %T')] master queue finished (exit $rc)."
        exit $rc
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[$(date '+%F %T')] Gave up after ${MAX_WAIT_HRS}h — no GPU freed up. Queue NOT launched."
        exit 2
    fi
    sleep "$POLL_SECS"
done
