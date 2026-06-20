#!/usr/bin/env bash
#
# Daily reconciliation sweep over open-meteo's immutable per-run archive (data_run).
# For each model: list the runs that exist in data_run within the window, diff against
# what is already on Hugging Face, and for every missing run export it by init-time
# (`--run`, full native horizon), convert to a per-run zarr store, and upload.
#
# This SUPERSEDES the old rolling-DB heal (run_solar_regions.sh): data_run is immutable
# and init-addressable (~3-month retention), so there is no capture-window race — a run
# can be fetched hours or days late and is guaranteed clean (never overwritten by a later
# run). The sweep is idempotent and order-independent; re-running is a no-op for runs
# already on HF, and it auto-catches-up after any downtime within the retention window.
#
# Modes:
#   (default daily)  sweep the last LOOKBACK_DAYS days          (LOOKBACK_DAYS=7)
#   (backfill)       SINCE=YYYY-MM-DD ./sweep_data_run.sh       (window = SINCE..today)
#   restrict models: MODELS="jma_msm dwd_icon" ./sweep_data_run.sh
#
# Requires: docker image open-meteo:bbox-fix (with the --run patch), and the venv must
# have pandas pyarrow xarray zarr numcodecs huggingface_hub[cli].
set -euo pipefail
cd "$(dirname "$0")"

# --- config ---------------------------------------------------------------
IMAGE="open-meteo:bbox-fix"
REMOTE_DATA="https://openmeteo.s3.amazonaws.com/data/"
S3_BASE="https://openmeteo.s3.amazonaws.com"
CACHE_VOLUME="open-meteo-cache"
PARALLEL="${PARALLEL:-1}"               # concurrent runs; measured ~2.4x throughput at 3
# parallel exports share the colima VM RAM, so shrink per-export cache to fit (3 x 2GB = 6GB)
CACHE_SIZE="${CACHE_SIZE:-$([ "${PARALLEL}" -gt 1 ] && echo 2GB || echo 6GB)}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "$0")" && pwd)/out}"
HF_DATASET_REPO="${HF_DATASET_REPO-JimTseng/apac-nwp-forecast-archive}"
PYBIN="${PYBIN:-python3}"

REGION_LAT="m44,46"
REGION_LON="92,154"
REGION_NAME="apac"
# Ocean INCLUDED: no --ignore_sea. (For jma_msm this is a no-op — its sea cells have
# elevation 0, not NaN, so they were always kept; for dwd_icon it really keeps the sea.)
IGNORE_SEA=""
EXPORT_VARS="shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high"
CONCURRENT=8
# export config so process_run.sh (spawned per run by xargs) inherits it
export IMAGE REMOTE_DATA CACHE_VOLUME CACHE_SIZE OUT_DIR HF_DATASET_REPO PYBIN
export REGION_LAT REGION_LON REGION_NAME IGNORE_SEA EXPORT_VARS CONCURRENT

MODELS="${MODELS:-jma_msm dwd_icon ncep_gfs013 ecmwf_ifs025}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-7}"
SINCE="${SINCE:-}"

mkdir -p "${OUT_DIR}"
docker volume create "${CACHE_VOLUME}" >/dev/null

# Single-flight lock (atomic mkdir; a still-running sweep makes the next tick skip).
LOCKDIR="${OUT_DIR}/.pipeline.lock"
if ! mkdir "${LOCKDIR}" 2>/dev/null; then
  echo "[$(date -u)] SKIP: another pipeline run is in progress (${LOCKDIR} exists)"
  exit 0
fi
trap 'code=$?; rmdir "'"${LOCKDIR}"'" 2>/dev/null; [ $code -ne 0 ] && echo "[$(date -u)] FATAL: sweep aborted (exit ${code})"' EXIT

echo "[$(date -u)] === data_run reconciliation sweep ==="
echo "  models=[${MODELS}]  $( [ -n "${SINCE}" ] && echo "since=${SINCE}" || echo "lookback=${LOOKBACK_DAYS}d" )  repo=${HF_DATASET_REPO}"

# --- 1) Planner: which runs in data_run (window) are missing from HF? -------
# Emits TSV: DOMAIN <TAB> RUN_ISO <TAB> START_DATE <TAB> END_DATE <TAB> RUN_STAMP <TAB> HF_ZARR_PATH
PLAN="$("${PYBIN}" - "${S3_BASE}" "${HF_DATASET_REPO}" "${SINCE}" "${LOOKBACK_DAYS}" ${MODELS} <<'PY'
import sys, os, urllib.request, urllib.parse, datetime, json
import xml.etree.ElementTree as ET
from huggingface_hub import HfApi

S3_BASE, REPO, SINCE, LOOKBACK = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
MODELS = sys.argv[5:]
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

def s3_list(prefix, delimiter=None):
    keys, prefixes, token = [], [], None
    while True:
        p = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter: p["delimiter"] = delimiter
        if token: p["continuation-token"] = token
        url = f"{S3_BASE}/?" + urllib.parse.urlencode(p)
        root = ET.fromstring(urllib.request.urlopen(url, timeout=30).read())
        keys += [c.find(NS+"Key").text for c in root.findall(NS+"Contents")]
        prefixes += [x.find(NS+"Prefix").text for x in root.findall(NS+"CommonPrefixes")]
        nxt = root.find(NS+"NextContinuationToken")
        if nxt is None: break
        token = nxt.text
    return keys, prefixes

today = datetime.datetime.now(datetime.UTC).date()
if SINCE:
    start = datetime.date.fromisoformat(SINCE)
else:
    start = today - datetime.timedelta(days=LOOKBACK)
# UNTIL bounds the END of the window (default = today). Set it to split work across machines
# without overlap, e.g. mini does SINCE..UNTIL (old half), laptop does the new half.
until = os.environ.get("UNTIL")
end_day = datetime.date.fromisoformat(until) if until else today
# ORDER=newest (default): land fresh data first. ORDER=oldest: grind history forward.
days = [start + datetime.timedelta(days=i) for i in range((end_day - start).days + 1)]
if os.environ.get("ORDER", "newest") != "oldest":
    days = days[::-1]

# Generous per-model export window (>= each model max horizon, in days). The export only
# downloads what the run actually contains; a wider window just fills the tail with NaN,
# which parquet_to_zarr drops. So we skip a per-run meta fetch (slow over hundreds of runs)
# and let the converter trim — the S3 download (the bottleneck) is unaffected.
# gfs/ifs capped at 7 days: their 15-16 day tails have ~no skill for solar forecasting
# and were the main backfill/storage cost. jma (3 d) and icon (7.5 d) keep full horizon.
HORIZON = {"jma_msm": 4, "dwd_icon": 8, "ncep_gfs013": 7, "ecmwf_ifs025": 7}

# Existing runs on HF: a run is present iff its <stamp>.zarr.zip exists (the archive
# format is now the per-run ocean zarr cube under data_zarr/, not parquet under data/).
api = HfApi()
have = set()
for f in api.list_repo_files(REPO, repo_type="dataset"):
    if f.endswith(".zarr.zip"):
        stamp = f.rsplit("/", 1)[-1].removesuffix(".zarr.zip")
        model = f.split("model=")[1].split("/")[0] if "model=" in f else "?"
        have.add((model, stamp))

# date-major: newest day first, all models within each day -> recent all-model coverage
# lands first, then grinds back through history (matches the newest-first freshness goal).
counts = {m: 0 for m in MODELS}
for d in days:
    for model in MODELS:
        base = f"data_run/{model}/{d.year:04d}/{d.month:02d}/{d.day:02d}/"
        try:
            _, runprefixes = s3_list(base, delimiter="/")
        except Exception:
            continue
        for rp in runprefixes:                      # .../HHMMZ/
            hhmm = rp.rstrip("/").rsplit("/", 1)[-1] # e.g. 1200Z
            hh = hhmm[:2]
            stamp = f"{d.year:04d}{d.month:02d}{d.day:02d}T{hh}Z"
            if (model, stamp) in have:
                continue
            run_iso = f"{d.year:04d}-{d.month:02d}-{d.day:02d}T{hh}:00"
            start_date = f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
            end_date = (d + datetime.timedelta(days=HORIZON.get(model, 16))).isoformat()
            hf_path = f"data/model={model}/year={d.year:04d}/month={d.month:02d}/day={d.day:02d}/{stamp}.zarr.zip"
            print("\t".join([model, run_iso, start_date, end_date, stamp, hf_path]))
            counts[model] += 1
for m in MODELS:
    print(f"[plan] {m}: {counts[m]} missing run(s)", file=sys.stderr)
PY
)"

if [ -z "${PLAN}" ]; then
  echo "[$(date -u)] archive already complete for the window — nothing to do"
  exit 0
fi
N=$(printf '%s\n' "${PLAN}" | grep -c . || true)
echo "[$(date -u)] ${N} run(s) missing — recovering from data_run"

if [ -n "${DRY_RUN:-}" ]; then
  echo "[$(date -u)] DRY_RUN: would recover the following (DOMAIN / RUN / horizon / hf_path):"
  printf '%s\n' "${PLAN}" | awk -F'\t' '{printf "  %-14s %s  %s..%s  -> %s\n",$1,$5,$3,$4,$6}'
  exit 0
fi

# --- 2) Recover each missing run: export --run (full horizon) -> zarr -> upload.
# PARALLEL runs at a time via xargs -P (macOS bash 3.2 has no `wait -n`). Each plan line's
# whitespace-separated fields become process_run.sh's positional args $1..$6.
printf '%s\n' "${PLAN}" | xargs -P "${PARALLEL}" -L 1 ./process_run.sh

echo "[$(date -u)] sweep done"
