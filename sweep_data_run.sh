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
CACHE_SIZE="6GB"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "$0")" && pwd)/out}"
HF_DATASET_REPO="${HF_DATASET_REPO-JimTseng/apac-nwp-forecast-archive}"
PYBIN="${PYBIN:-python3}"

REGION_LAT="m44,46"
REGION_LON="92,154"
REGION_NAME="apac"
IGNORE_SEA="--ignore_sea"
EXPORT_VARS="shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high"
CONCURRENT=8

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
import sys, urllib.request, urllib.parse, datetime, json
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
days = [start + datetime.timedelta(days=i) for i in range((today - start).days + 1)]

# Existing zarr runs on HF: a run is present iff <stamp>.zarr/zarr.json exists.
api = HfApi()
have = set()
for f in api.list_repo_files(REPO, repo_type="dataset"):
    if f.endswith(".zarr/zarr.json"):
        stamp = f.rsplit("/", 1)[0].rsplit("/", 1)[-1].removesuffix(".zarr")
        model = f.split("model=")[1].split("/")[0] if "model=" in f else "?"
        have.add((model, stamp))

for model in MODELS:
    for d in days:
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
            # fetch the run meta for the exact horizon end (full horizon, no NaN tail guesswork)
            try:
                meta = json.load(urllib.request.urlopen(f"{S3_BASE}/{rp}meta.json", timeout=30))
                vt = meta["valid_times"]
            except Exception:
                continue
            run_iso = f"{d.year:04d}-{d.month:02d}-{d.day:02d}T{hh}:00"
            start_date = f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
            end_date = vt[-1][:10]
            hf_path = f"data/model={model}/year={d.year:04d}/month={d.month:02d}/{stamp}.zarr"
            print("\t".join([model, run_iso, start_date, end_date, stamp, hf_path]))
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

# --- 2) For each missing run: export --run (full horizon) -> zarr -> upload -
printf '%s\n' "${PLAN}" | while IFS=$'\t' read -r DOMAIN RUN_ISO START_DATE END_DATE RUN_STAMP HF_PATH; do
  [ -z "${DOMAIN}" ] && continue
  OUT_FILE="${REGION_NAME}_${DOMAIN}_${RUN_STAMP}.parquet"
  echo "[$(date -u)] --- ${DOMAIN} run ${RUN_STAMP} (${RUN_ISO}) horizon ${START_DATE}..${END_DATE}"

  EXPORT_OK=false
  for attempt in 1 2; do
    if docker run --rm \
      -v "${CACHE_VOLUME}":/app/data -v "${OUT_DIR}":/out \
      -e REMOTE_DATA_DIRECTORY="${REMOTE_DATA}" -e CACHE_SIZE="${CACHE_SIZE}" \
      --entrypoint /app/openmeteo-api "${IMAGE}" \
      export "${DOMAIN}" "${EXPORT_VARS}" \
      --run "${RUN_ISO}" \
      --start_date "${START_DATE}" --end_date "${END_DATE}" \
      --latitude-bounds "${REGION_LAT}" --longitude-bounds "${REGION_LON}" \
      ${IGNORE_SEA} --concurrent "${CONCURRENT}" \
      --format parquet -o "/out/${OUT_FILE}"; then
      EXPORT_OK=true; break
    fi
    echo "[$(date -u)]     WARN: ${DOMAIN} ${RUN_STAMP} export attempt ${attempt} failed"
    [ "${attempt}" -lt 2 ] && sleep 60
  done
  if [ "${EXPORT_OK}" != "true" ]; then
    echo "[$(date -u)]     ERROR: ${DOMAIN} ${RUN_STAMP} export failed twice — skipping (next sweep retries)"
    rm -f "${OUT_DIR}/${OUT_FILE}"; continue
  fi

  # parquet -> per-run zarr (int-cast, provenance, drop out-of-run NaN rows)
  ZARR_TMP="$(mktemp -d)/${RUN_STAMP}.zarr"
  if ! RUN_STAMP="${RUN_STAMP}" SCRAPED_AT="$(date -u +%Y%m%dT%H%M%SZ)" \
       "${PYBIN}" parquet_to_zarr.py "${OUT_DIR}/${OUT_FILE}" "${ZARR_TMP}"; then
    echo "[$(date -u)]     ERROR: ${DOMAIN} ${RUN_STAMP} zarr conversion failed — skipping"
    rm -rf "$(dirname "${ZARR_TMP}")" "${OUT_DIR}/${OUT_FILE}"; continue
  fi

  echo "[$(date -u)]     uploading ${DOMAIN} ${RUN_STAMP} -> hf://datasets/${HF_DATASET_REPO}/${HF_PATH}"
  if hf upload "${HF_DATASET_REPO}" "${ZARR_TMP}" "${HF_PATH}" --repo-type dataset --quiet; then
    echo "[$(date -u)]     OK ${DOMAIN} ${RUN_STAMP}"
  else
    echo "[$(date -u)]     WARN: HF upload failed for ${DOMAIN} ${RUN_STAMP} (next sweep retries)"
  fi
  rm -rf "$(dirname "${ZARR_TMP}")" "${OUT_DIR}/${OUT_FILE}"
done

echo "[$(date -u)] sweep done"
