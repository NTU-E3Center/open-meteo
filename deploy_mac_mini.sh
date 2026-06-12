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
"${VENV}/bin/pip" install -q --upgrade pandas pyarrow "huggingface_hub[cli]"
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

echo "=== 5/6 Smoke test: single-model small run (jma_msm, ~3-5 min) ==="
./run_solar_regions.sh jma_msm

echo "=== 6/6 Install cron schedule ==="
REPO_DIR="$(pwd)"
LOG_DIR="${HOME}/om-logs"
mkdir -p "${LOG_DIR}"
CRON_MARK="# apac-solar-pipeline"
# Target times are defined in UTC (model publication schedule); cron interprets times in
# the SYSTEM timezone, so convert UTC hours -> local hours at install time.
FULL_HOURS=$(python3 -c "
import datetime
off = round(datetime.datetime.now().astimezone().utcoffset().total_seconds()/3600)
print(','.join(str((h+off)%24) for h in (4,10,16,22)))")
JMA_HOURS=$(python3 -c "
import datetime
off = round(datetime.datetime.now().astimezone().utcoffset().total_seconds()/3600)
print(','.join(str((h+off)%24) for h in (1,7,13,19)))")
( crontab -l 2>/dev/null | grep -v "${CRON_MARK}" || true
  echo "PATH=${VENV}/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin ${CRON_MARK}"
  echo "30 ${FULL_HOURS} * * * cd ${REPO_DIR} && ./run_solar_regions.sh >> ${LOG_DIR}/full.log 2>&1 ${CRON_MARK}"
  echo "30 ${JMA_HOURS} * * * cd ${REPO_DIR} && ./run_solar_regions.sh jma_msm >> ${LOG_DIR}/jma.log 2>&1 ${CRON_MARK}"
) | crontab -
echo "cron installed (UTC 04:30/10:30/16:30/22:30 full + 01:30/07:30/13:30/19:30 jma, converted to local tz):"
crontab -l | grep "${CRON_MARK}"

echo
echo "=== Deployment complete ==="
echo "Logs:   ${LOG_DIR}/full.log, ${LOG_DIR}/jma.log"
echo "Local retention: LOCAL_RETENTION_DAYS=2 (HF is the canonical archive)"
