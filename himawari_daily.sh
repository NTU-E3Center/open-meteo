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
