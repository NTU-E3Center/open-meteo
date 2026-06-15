#!/usr/bin/env bash
#
# Open-Meteo remote-mode pipeline: export solar forecast grids for multiple regions
# (Japan/Taiwan, Southeast Asia, Australia) directly from Open-Meteo S3 open-data.
#
# No sync, no local database: `REMOTE_DATA_DIRECTORY` makes `export` read .om files
# from S3 via HTTP range requests, fetching only the bytes covering each bounding box.
# Run every 6 hours (cron). Each run writes one timestamped Parquet per region;
# keep them to accumulate an init_time x lead_time forecast archive per region.
#
# REQUIRES the locally built image `open-meteo:bbox-fix` (patched ExportCommand that
# accepts 'm' as minus prefix in --latitude-bounds/--longitude-bounds, e.g. "m44,m10"
# = -44..-10 — the stock CLI parser rejects values starting with "-").
# Build with: docker build -t open-meteo:bbox-fix .
#
# Background and validation history: see RADIATION_INTERPOLATION.md and the git log.
#
set -euo pipefail

# Optional: pass a model name to run ONLY that model (e.g. `./run_solar_regions.sh jma_msm`
# for the extra 3-hourly JMA refreshes). No argument = run all models.
# `--heal`: self-heal mode — only export models whose CURRENT run is missing from HF
# (retries failed/missed scrapes while the run is still in the rolling DB; near no-op
# when the archive is complete). Intended to run hourly from cron.
ONLY_MODEL="${1:-}"
HEAL=""
if [ "${ONLY_MODEL}" = "--heal" ]; then
  HEAL=1
  ONLY_MODEL=""
fi

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
IMAGE="open-meteo:bbox-fix"
REMOTE_DATA="https://openmeteo.s3.amazonaws.com/data/"
CACHE_VOLUME="open-meteo-cache"
CACHE_SIZE="6GB"                       # multi-model transfers ~2GB+/cycle; keep headroom
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "$0")" && pwd)/out}"   # default: ./out next to this script

# Local retention: HF is the canonical archive; local files are only a debugging buffer.
# Files older than this many days are deleted at the end of each run (0 = delete all
# already-uploaded local files immediately; set to a huge number to keep everything).
LOCAL_RETENTION_DAYS="${LOCAL_RETENTION_DAYS:-2}"

# Models: "domain forecastDays" — horizons differ per model.
#   dwd_icon             : DWD ICON global, 11 km, 6-hourly runs, 7.5-day horizon. direct/diffuse NATIVE.
#   ncep_gfs013          : NOAA GFS, 13 km, 6-hourly runs, up to 16 days. diffuse native, direct = GHI - diffuse.
#   jma_msm              : JMA MSM, 5 km, 3-hourly runs, ~3 days. Grid only covers 22.4-47.6N 120-150E
#                          (Japan; southern Taiwan cut off). direct/diffuse via empirical separation model.
#   ecmwf_ifs025         : ECMWF IFS, 25 km, 6-hourly runs, 15 days. DB is 3-HOURLY (exports 3 h rows).
#                          Highest large-scale skill; radiation split via separation model.
# DROPPED ecmwf_aifs025_single (2026-06-13): fully redundant with dynamical.org's
# "ECMWF AIFS Single Forecast" Zarr archive (same 0.25°, 6-hourly, GHI, archived since
# 2024-04 — deeper than ours). Our copy added only format convenience at a permanent
# storage cost; pull AIFS from dynamical.org if ever needed. JMA/ICON-APAC have NO
# equivalent there and stay. See git log / project memory.
# ORDER MATTERS in --heal mode: list the shortest-interval model FIRST. jma_msm is
# 3-hourly (its rolling run rotates every 3 h); the others are 6-hourly. Processing is
# serial, so a slow 6-hourly recovery (icon/gfs/ifs exports take 30-40 min) running
# before jma would push jma's missing-run check past a whole JMA rotation, silently
# skipping the run that was current at cycle start. Putting jma first guarantees its
# check + export happen at the very start of every cycle, before any slow recovery.
# (Post-mortem: 2026-06-14 12Z JMA was lost exactly this way — gfs's 37 min recovery
# delayed jma's check from 18:05 to 18:44, by which point 12Z had rotated to 15Z.)
MODELS=(
  "jma_msm               3"
  "dwd_icon              7"
  "ncep_gfs013           7"
  "ecmwf_ifs025         15"
)

# Regions: "name latBounds lonBounds" — 'm' prefix = minus (e.g. m44 = -44)
# Single full rectangle covering JP/TW + SEA + Australia (incl. China interior),
# so merged data forms a complete grid with no seams or duplicate overlap.
REGIONS=(
  "apac  m44,46  92,154"
)

# Land-only: drops ocean grid cells (ICON land-fraction >= 50% counts as land;
# small islands like 蘭嶼/宮古島 fall below that and are excluded — use
# --ignore_sea_search_radius N instead to keep near-land sea points if needed).
IGNORE_SEA="--ignore_sea"

# Derived vars are computed on the fly (shortwave = direct + diffuse; DNI from
# direct + solar geometry; wind_speed from u/v; surface_pressure from pressure_msl).
EXPORT_VARS="shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high"

CONCURRENT=8

# Hugging Face upload: set HF_DATASET_REPO="" to disable. Requires `hf auth login` once
# and python3 with pandas+pyarrow (for the zstd re-compression before upload).
# Files land under data/model=<domain>/ for Hive-style partitioning.
HF_DATASET_REPO="${HF_DATASET_REPO-JimTseng/apac-nwp-forecast-archive}"

# ---------------------------------------------------------------------------
# Derived values (dates in UTC; macOS BSD date — Linux: date -u -d "+N days")
# ---------------------------------------------------------------------------
START_DATE="$(date -u +%Y-%m-%d)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${OUT_DIR}"
docker volume create "${CACHE_VOLUME}" >/dev/null

# Leave a trace if set -e aborts the script (otherwise failures are silent in cron logs)
trap 'code=$?; [ $code -ne 0 ] && echo "[$(date -u)] FATAL: pipeline aborted (exit ${code})"' EXIT

# Prevent overlapping runs (a slow cold-cache cycle can exceed the cron interval).
# mkdir is atomic; macOS has no flock(1). A skipped refresh beats two racing pipelines.
LOCKDIR="${OUT_DIR}/.pipeline.lock"
if ! mkdir "${LOCKDIR}" 2>/dev/null; then
  echo "[$(date -u)] SKIP: another pipeline run is in progress (${LOCKDIR} exists)"
  trap - EXIT
  exit 0
fi
trap 'code=$?; rmdir "'"${LOCKDIR}"'" 2>/dev/null; [ $code -ne 0 ] && echo "[$(date -u)] FATAL: pipeline aborted (exit ${code})"' EXIT

echo "[$(date -u)] === Open-Meteo multi-model solar pipeline (remote mode) ==="
echo "  models=${#MODELS[@]}  regions=${#REGIONS[@]}  start=${START_DATE}  land_only=${IGNORE_SEA:-no}"

for model in "${MODELS[@]}"; do
  read -r DOMAIN DAYS <<< "${model}"
  if [ -n "${ONLY_MODEL}" ] && [ "${DOMAIN}" != "${ONLY_MODEL}" ]; then
    continue
  fi
  END_DATE="$(date -u -v+${DAYS}d +%Y-%m-%d)"
  # Authoritative run label from the domain's meta.json, fetched BEFORE the export.
  fetch_run_stamp() {
    python3 -c "
import urllib.request, json, datetime
m = json.load(urllib.request.urlopen('https://openmeteo.s3.amazonaws.com/data/${DOMAIN}/static/meta.json', timeout=30))
print(datetime.datetime.fromtimestamp(m['last_run_initialisation_time'], datetime.UTC).strftime('%Y%m%dT%HZ'))
" 2>/dev/null
  }
  if [ -n "${HEAL}" ]; then
    # Self-heal: only act when the model's CURRENT run is absent from HF.
    RUN_STAMP=$(fetch_run_stamp) || { echo "[$(date -u)] heal: ${DOMAIN} meta unavailable — skipping"; continue; }
    HF_PATH="data/model=${DOMAIN}/year=${RUN_STAMP:0:4}/month=${RUN_STAMP:4:2}/${RUN_STAMP}.parquet"
    if python3 -c "
from huggingface_hub import HfApi
import sys
sys.exit(0 if HfApi().file_exists('${HF_DATASET_REPO}', '${HF_PATH}', repo_type='dataset') else 1)
" 2>/dev/null; then
      continue  # current run already archived
    fi
    echo "[$(date -u)] heal: ${DOMAIN} run ${RUN_STAMP} missing from HF — recovering"
  else
    RUN_STAMP=$(fetch_run_stamp) || RUN_STAMP="${STAMP}"
  fi
  for region in "${REGIONS[@]}"; do
    read -r NAME LAT LON <<< "${region}"
    OUT_FILE="${NAME}_${DOMAIN}_${STAMP}.parquet"
    echo "[$(date -u)] --- ${DOMAIN} run ${RUN_STAMP} / ${NAME}: ${START_DATE}..${END_DATE}, lat ${LAT}, lon ${LON} -> ${OUT_FILE}"
    # One model's failure must not kill the remaining models (transient S3 stream
    # breaks crash the exporter with an uncaught HTTPParserError) — retry once,
    # then skip this model and continue.
    EXPORT_OK=false
    for attempt in 1 2; do
      if docker run --rm \
        -v "${CACHE_VOLUME}":/app/data \
        -v "${OUT_DIR}":/out \
        -e REMOTE_DATA_DIRECTORY="${REMOTE_DATA}" \
        -e CACHE_SIZE="${CACHE_SIZE}" \
        --entrypoint /app/openmeteo-api "${IMAGE}" \
        export "${DOMAIN}" "${EXPORT_VARS}" \
        --start_date "${START_DATE}" --end_date "${END_DATE}" \
        --latitude-bounds "${LAT}" --longitude-bounds "${LON}" \
        ${IGNORE_SEA} \
        --concurrent "${CONCURRENT}" \
        --format parquet -o "/out/${OUT_FILE}"; then
        EXPORT_OK=true; break
      fi
      echo "[$(date -u)]     WARN: ${DOMAIN} export attempt ${attempt} failed"
      [ "${attempt}" -lt 2 ] && sleep 60
    done
    if [ "${EXPORT_OK}" != "true" ]; then
      echo "[$(date -u)]     ERROR: ${DOMAIN} export failed twice — skipping this model this cycle"
      continue
    fi
    ls -lh "${OUT_DIR}/${OUT_FILE}"

    # Upload to Hugging Face if configured and logged in.
    # Re-compress first: round to physical precision + zstd (~20% smaller, information-lossless
    # since source values are quantized in the .om database anyway).
    if [ -n "${HF_DATASET_REPO}" ] && hf auth whoami >/dev/null 2>&1; then
      # Guard against a new run landing MID-export: if the run label changed between the
      # start and end of this export, the file mixes two runs — warn (still uploaded).
      RUN_AFTER=$(fetch_run_stamp) || RUN_AFTER="${RUN_STAMP}"
      if [ "${RUN_AFTER}" != "${RUN_STAMP}" ]; then
        echo "[$(date -u)]     WARN: run rotated mid-export (${RUN_STAMP} -> ${RUN_AFTER}); discarding to avoid a mislabelled mixed-run file — next cycle will capture ${RUN_AFTER} cleanly"
        rm -f "${OUT_DIR}/${OUT_FILE}"
        continue
      fi
      TMP_FILE="$(mktemp -d)/${OUT_FILE}"
      RUN_STAMP="${RUN_STAMP}" SCRAPED_AT="${STAMP}" python3 - "$OUT_DIR/$OUT_FILE" "$TMP_FILE" <<'PYEOF'
import sys, os, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
# Drop the "already-past" rows: export starts at 00:00 of START_DATE, so a run
# initialised later (e.g. 21Z) carries up to 21 h of pre-run hours that are NOT this
# run's forecast and are stored redundantly by every later run that day. Keep only
# time >= run_init so each file is a clean lead>=0 forecast (~10-15% smaller, lossless
# — the dropped hours belong to earlier runs' files).
run_init = pd.to_datetime(os.environ['RUN_STAMP'], format='%Y%m%dT%HZ', utc=True).tz_localize(None)
df = df[pd.to_datetime(df['time']) >= run_init]
meta = {'location_id','latitude','longitude','elevation','time'}
for c in df.columns:
    if c in meta:
        continue
    df[c] = df[c].round(2)
    # Integer-cast columns whose SOURCE quantization step is 1 (values identical,
    # integers compress ~10% better than float32 carrying conversion noise).
    # Variables with sub-unit steps (temperature 0.05; wind/pressure/precip 0.1) stay float32.
    if 'cloud_cover' in c or 'relative_humidity' in c:
        df[c] = df[c].clip(lower=0, upper=255).round().astype('UInt8')
    elif 'radiation' in c or 'irradiance' in c:
        df[c] = df[c].clip(lower=0).round().astype('UInt16')
df['location_id'] = df['location_id'].astype('int32')
# Self-describing provenance columns (constant -> RLE-compresses to almost nothing):
df['run_init'] = pd.Timestamp(pd.to_datetime(os.environ['RUN_STAMP'], format='%Y%m%dT%HZ', utc=True)).tz_localize(None)
df['scraped_at'] = pd.Timestamp(pd.to_datetime(os.environ['SCRAPED_AT'], format='%Y%m%dT%H%M%SZ', utc=True)).tz_localize(None)
df.to_parquet(dst, compression='zstd', compression_level=12)
PYEOF
      # HF filename = model run init time. Re-scrapes of the same run overwrite (idempotent).
      echo "[$(date -u)]     uploading ${DOMAIN} run ${RUN_STAMP} to hf://datasets/${HF_DATASET_REPO}"
      # Hive-style layout: data/model=<domain>/year=YYYY/month=MM/<runStamp>.parquet
      hf upload "${HF_DATASET_REPO}" "${TMP_FILE}" \
        "data/model=${DOMAIN}/year=${RUN_STAMP:0:4}/month=${RUN_STAMP:4:2}/${RUN_STAMP}.parquet" \
        --repo-type dataset --quiet \
        || echo "[$(date -u)]     WARN: HF upload failed for ${OUT_FILE} (kept locally)"
      rm -f "${TMP_FILE}"
    fi
  done
done

# Clean up old local parquet files (HF holds the canonical archive)
if [ -n "${LOCAL_RETENTION_DAYS}" ]; then
  DELETED=$(find "${OUT_DIR}" -name '*.parquet' -mtime +"${LOCAL_RETENTION_DAYS}" -print -delete | wc -l | tr -d ' ')
  [ "${DELETED}" != "0" ] && echo "[$(date -u)] Local cleanup: removed ${DELETED} parquet file(s) older than ${LOCAL_RETENTION_DAYS} days"
fi

echo "[$(date -u)] Done: exported $( [ -n "${ONLY_MODEL}" ] && echo "1 model (${ONLY_MODEL})" || echo "${#MODELS[@]} models" ) x ${#REGIONS[@]} regions"
