#!/usr/bin/env bash
# Guard launcher: wait until a GPU has genuinely FREED UP, then run the Riemann
# queue. Same logic as run_sr_queue_when_free.sh but launches the 1D Riemann
# queue and uses a smaller VRAM threshold (1D models are light).
#
# "Freed up" = BOTH hold for N_CONSECUTIVE consecutive polls:
#   * utilization <= UTIL_MAX %        (no other job actively computing)
#   * free memory >= FREE_MIN_MIB MiB  (genuine headroom)
# This is the polite default on a shared box: it never contends with another
# user's running job, it only starts once a card is actually idle.
#
# Launch detached:
#   nohup setsid bash scripts/run_riemann_queue_when_free.sh \
#       > logs/riemann_queue_guard.log 2>&1 < /dev/null &
#
# Tunables (env overrides):
#   FREE_MIN_MIB  min free VRAM (MiB) to consider a GPU free   [default 10000]
#   UTIL_MAX      max GPU utilization (%) to consider it idle  [default 10]
#   N_CONSECUTIVE consecutive passing polls before launch      [default 3]
#   POLL_SECS     seconds between polls                        [default 120]
#   MAX_WAIT_HRS  give up after this many hours                [default 48]
set -u

PROJECT=/home/mamta/Projects/AW-FNO
FREE_MIN_MIB=${FREE_MIN_MIB:-10000}
UTIL_MAX=${UTIL_MAX:-10}
N_CONSECUTIVE=${N_CONSECUTIVE:-3}
POLL_SECS=${POLL_SECS:-120}
MAX_WAIT_HRS=${MAX_WAIT_HRS:-48}

cd "$PROJECT"
deadline=$(( $(date +%s) + MAX_WAIT_HRS * 3600 ))

echo "[$(date '+%F %T')] Riemann queue guard started."
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
        echo "[$(date '+%F %T')] GPU $ready_gpu ready (${util}% util, ${free} MiB free). Launching Riemann queue."
        GPU="$ready_gpu" bash scripts/run_riemann_queue.sh
        rc=$?
        echo "[$(date '+%F %T')] Riemann queue finished (exit $rc)."
        exit $rc
    fi

    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[$(date '+%F %T')] Gave up after ${MAX_WAIT_HRS}h — no GPU freed up. Riemann queue NOT launched."
        exit 2
    fi
    sleep "$POLL_SECS"
done
