#!/usr/bin/env bash
# Drain a model's silver backlog by repeatedly running go_forward.sh until
# "nothing to do". Temporarily comments out the model's own cron line
# (avoids two same-model runs sharing .work/goforward-<model>) and restores
# it on exit.
#
# Usage (on the mini):
#   nohup ./drain_backlog.sh jma_msm 12 >> ~/om-logs/drain_jma_msm.log 2>&1 &
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?usage: drain_backlog.sh <model> [window_days]}"
WINDOW="${2:-12}"
MAX_ROUNDS=30
export PATH="/Users/e3center/.om-venv/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/sbin:/usr/sbin"
export ZARR_PY="${ZARR_PY:-/Users/e3center/.om-venv/bin/python}"

log() { echo "[$(date -u +%FT%TZ)] [drain $MODEL] $*"; }

disable_cron() {
  crontab -l | sed "/^[^#].*go_forward.sh $MODEL /s/^/#DRAIN-$MODEL /" | crontab -
  log "cron line for $MODEL disabled"
}
restore_cron() {
  crontab -l | sed "s/^#DRAIN-$MODEL //" | crontab -
  log "cron line for $MODEL restored"
}
trap restore_cron EXIT

disable_cron
cd "$DIR"
for round in $(seq 1 "$MAX_ROUNDS"); do
  log "round $round (MAX_RUNS=12, --backfill $WINDOW)"
  OUT=$(MAX_RUNS=12 ./go_forward.sh "$MODEL" --backfill "$WINDOW" 2>&1) || {
    echo "$OUT" | tail -5
    log "round $round failed; retrying in 120s"
    sleep 120
    continue
  }
  echo "$OUT" | grep -E "runs to process|bronze ok|written|DONE" | tail -4
  if echo "$OUT" | grep -q "nothing to do"; then
    log "backlog clear after $((round - 1)) round(s)"
    exit 0
  fi
done
log "reached MAX_ROUNDS=$MAX_ROUNDS without clearing backlog; check the log"
exit 1
