#!/usr/bin/env bash
#
# One-shot deployment for the APAC solar pipeline on a Mac mini.
# Usage:  ./deploy_mac_mini.sh [HF_TOKEN]
#   - Run from the cloned repo root (branch apac-solar-pipeline).
#   - Pass a Hugging Face WRITE token as $1 unless `hf auth login` was already done.
# What it does: checks prerequisites, installs python deps + hf CLI, builds the
# patched Docker image, logs in to HF, runs one smoke-test export, installs cron.
#
set -euo pipefail
cd "$(dirname "$0")"

echo "=== 1/6 Prerequisites ==="
if ! command -v docker >/dev/null; then
  echo "ERROR: docker not found. Install Docker Desktop (or OrbStack) first, then re-run."
  exit 1
fi
docker info >/dev/null 2>&1 || { echo "ERROR: docker daemon not running. Start Docker Desktop, then re-run."; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found."; exit 1; }
echo "docker + python3 OK"

echo "=== 2/6 Python deps + HF CLI (venv) ==="
# Homebrew pythons are PEP-668 "externally managed" — use a dedicated venv.
# Prefer 3.12 (mature wheel coverage) over the newest brew python.
VENV="${HOME}/.om-venv"
PYBIN="$(command -v python3.12 || command -v python3)"
"${PYBIN}" -m venv "${VENV}"
# xarray+zarr+numcodecs are needed by parquet_to_zarr_cube.py (process_run.sh converts
# each run to a per-run zarr cube before upload). zarr>=3 for the v3 BloscCodec path.
"${VENV}/bin/pip" install -q --upgrade pandas pyarrow "huggingface_hub[cli]" \
  xarray "zarr>=3" numcodecs
export PATH="${VENV}/bin:${PATH}"
command -v hf >/dev/null || { echo "ERROR: 'hf' CLI not found in venv."; exit 1; }
echo "venv: ${VENV} ($(python3 --version))"

echo "=== 3/6 Hugging Face auth ==="
if hf auth whoami >/dev/null 2>&1; then
  echo "already logged in as: $(hf auth whoami 2>/dev/null | head -1)"
elif [ -n "${1:-}" ]; then
  hf auth login --token "$1"
else
  echo "ERROR: not logged in and no token given. Re-run: ./deploy_mac_mini.sh hf_xxx"
  exit 1
fi

echo "=== 4/6 Build patched Docker image (~30-50 min first time) ==="
docker build --pull -f Dockerfile.local -t open-meteo:bbox-fix .

echo "=== 5/6 Smoke test: planner dry-run (lists what the sweep would recover) ==="
DRY_RUN=1 MODELS=jma_msm LOOKBACK_DAYS=1 ./sweep_data_run.sh

echo "=== 6/6 Install cron schedule ==="
REPO_DIR="$(pwd)"
LOG_DIR="${HOME}/om-logs"
mkdir -p "${LOG_DIR}"
CRON_MARK="# apac-solar-pipeline"
# The WHOLE pipeline is one daily reconciliation sweep over the immutable data_run archive.
# `sweep_data_run.sh` lists the runs that exist in data_run within the look-back window,
# diffs them against HF, and exports+uploads (as per-run parquet) only the missing ones —
# idempotent and order-independent. Because data_run is init-addressable and immutable
# (~3-month retention), there is NO capture-window race: a run can be fetched hours or
# days late and is guaranteed clean. Daily is enough (data is for model training); the
# 7-day look-back absorbs publication lag and tolerates up to a week of downtime.
# Runs once daily at 06:00 UTC. Overlap is safe: a still-running sweep holds the LOCKDIR.
# For a one-time historical backfill: SINCE=YYYY-MM-DD ./sweep_data_run.sh
# NOTE: cron env-var lines do NOT support trailing comments (would corrupt PATH) — so the
# PATH line carries no marker; dedup matches it by the venv path instead.
( crontab -l 2>/dev/null | grep -v "${CRON_MARK}" | grep -v "^PATH=.*om-venv" || true
  echo "PATH=${VENV}/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin"
  echo "0 6 * * * cd ${REPO_DIR} && MODELS=\"jma_msm dwd_icon\" ./sweep_data_run.sh >> ${LOG_DIR}/sweep.log 2>&1 ${CRON_MARK}"
) | crontab -
echo "cron installed (daily data_run reconciliation sweep at 06:00 UTC — jma_msm + dwd_icon):"
crontab -l | grep "${CRON_MARK}"

echo
echo "=== Deployment complete ==="
echo "Logs:   ${LOG_DIR}/sweep.log"
echo "Backfill history once: SINCE=YYYY-MM-DD ./sweep_data_run.sh"
echo "HF is the canonical archive (per-run ocean zarr.zip under data_zarr/model=<m>/.../<run>.zarr.zip)"
