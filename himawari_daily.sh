#!/usr/bin/env bash
#
# Daily Himawari SWR update: collect any COMPLETE past UTC day in a rolling 5-day window
# that is missing from HF (jimtseng/apac-himawari-swr). Idempotent — skips days already
# uploaded; the himawari_fetch_day.py guard refuses today (in-progress). The 5-day window
# self-heals any missed day after downtime. Cron-installed on the Mac mini.
#
# Credentials: reads JAXA P-Tree FTP user/pw from ~/.himawari_ftp.env (chmod 600), which
# must export HIMAWARI_FTP_USER / HIMAWARI_FTP_PW. Never hard-code creds in the repo.
set -euo pipefail
cd "$(dirname "$0")"

[ -f "$HOME/.himawari_ftp.env" ] && . "$HOME/.himawari_ftp.env"
: "${HIMAWARI_FTP_USER:?set HIMAWARI_FTP_USER in ~/.himawari_ftp.env}"
: "${HIMAWARI_FTP_PW:?set HIMAWARI_FTP_PW in ~/.himawari_ftp.env}"
export HIMAWARI_FTP_USER HIMAWARI_FTP_PW
export PATH="$HOME/.om-venv/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"
export HF_HUB_DISABLE_PROGRESS_BARS=1

# macOS BSD date (Mac mini). Window = [today-5, today-1] UTC (yesterday is always complete).
SINCE="$(date -u -v-5d +%Y-%m-%d)"
UNTIL="$(date -u -v-1d +%Y-%m-%d)"
echo "[$(date -u)] himawari daily: window ${SINCE}..${UNTIL}"
python3 himawari_backfill.py --since "${SINCE}" --until "${UNTIL}" --workers 2

# ---------------------------------------------------------------------------
# Optional: region-write each completed day into the Himawari silver cube.
# Enabled only when HIMAWARI_CUBE=1 (add to ~/.himawari_ftp.env or crontab).
# Failures in the cube step are logged but do NOT break the zip pipeline.
# ---------------------------------------------------------------------------
if [ "${HIMAWARI_CUBE:-0}" = "1" ]; then
  echo "[$(date -u)] HIMAWARI_CUBE=1 — writing window to silver cube"
  # Re-iterate the same [SINCE..UNTIL] window (macOS BSD date).
  # We count forward from SINCE for N days where N = (UNTIL - SINCE) in days + 1.
  SINCE_EPOCH=$(date -u -j -f "%Y-%m-%d" "${SINCE}" +%s)
  UNTIL_EPOCH=$(date -u -j -f "%Y-%m-%d" "${UNTIL}" +%s)
  NDAYS=$(( (UNTIL_EPOCH - SINCE_EPOCH) / 86400 + 1 ))
  for i in $(seq 0 $((NDAYS - 1))); do
    CUBE_DAY=$(date -u -j -v+"${i}d" -f "%Y-%m-%d" "${SINCE}" +%Y-%m-%d)
    echo "[$(date -u)] cube: writing ${CUBE_DAY} ..."
    if python3 himawari_day_to_cube.py --day "${CUBE_DAY}" --from-hf; then
      echo "[$(date -u)] cube: ${CUBE_DAY} OK"
    else
      echo "[$(date -u)] cube: ${CUBE_DAY} FAILED (zip pipeline unaffected)" >&2
    fi
  done
  echo "[$(date -u)] cube window done"
fi
