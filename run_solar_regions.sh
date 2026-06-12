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
# See HANDOFF_japan_taiwan_solar.md for background and validation history.
#
set -euo pipefail

# Optional: pass a model name to run ONLY that model (e.g. `./run_solar_regions.sh jma_msm`
# for the extra 3-hourly JMA refreshes). No argument = run all models.
ONLY_MODEL="${1:-}"

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
IMAGE="open-meteo:bbox-fix"
REMOTE_DATA="https://openmeteo.s3.amazonaws.com/data/"
CACHE_VOLUME="open-meteo-cache"
CACHE_SIZE="6GB"                       # multi-model transfers ~2GB+/cycle; keep headroom
OUT_DIR="${HOME}/Desktop/projects/open-meteo/out"

# Models: "domain forecastDays" — horizons differ per model.
#   dwd_icon             : DWD ICON global, 11 km, 6-hourly runs, 7.5-day horizon. direct/diffuse NATIVE.
#   ncep_gfs013          : NOAA GFS, 13 km, 6-hourly runs, up to 16 days. diffuse native, direct = GHI - diffuse.
#   jma_msm              : JMA MSM, 5 km, 3-hourly runs, ~3 days. Grid only covers 22.4-47.6N 120-150E
#                          (Japan; southern Taiwan cut off). direct/diffuse via empirical separation model.
#   ecmwf_ifs025         : ECMWF IFS, 25 km, 6-hourly runs, 15 days. DB is 3-HOURLY (exports 3 h rows).
#                          Highest large-scale skill; radiation split via separation model.
#   ecmwf_aifs025_single : ECMWF AIFS (AI model), 25 km, 15 days, DB 6-HOURLY (exports 6 h rows).
#                          Diversity member for medium-range trends; smooth fields, coarse time.
MODELS=(
  "dwd_icon              7"
  "ncep_gfs013           7"
  "jma_msm               3"
  "ecmwf_ifs025         15"
  "ecmwf_aifs025_single 15"
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
  RUN_STAMP=$(fetch_run_stamp) || RUN_STAMP="${STAMP}"
  for region in "${REGIONS[@]}"; do
    read -r NAME LAT LON <<< "${region}"
    OUT_FILE="${NAME}_${DOMAIN}_${STAMP}.parquet"
    echo "[$(date -u)] --- ${DOMAIN} run ${RUN_STAMP} / ${NAME}: ${START_DATE}..${END_DATE}, lat ${LAT}, lon ${LON} -> ${OUT_FILE}"
    docker run --rm \
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
      --format parquet -o "/out/${OUT_FILE}"
    ls -lh "${OUT_DIR}/${OUT_FILE}"

    # Upload to Hugging Face if configured and logged in.
    # Re-compress first: round to physical precision + zstd (~20% smaller, information-lossless
    # since source values are quantized in the .om database anyway).
    if [ -n "${HF_DATASET_REPO}" ] && hf auth whoami >/dev/null 2>&1; then
      # Guard against a new run landing MID-export: if the run label changed between the
      # start and end of this export, the file mixes two runs — warn (still uploaded).
      RUN_AFTER=$(fetch_run_stamp) || RUN_AFTER="${RUN_STAMP}"
      if [ "${RUN_AFTER}" != "${RUN_STAMP}" ]; then
        echo "[$(date -u)]     WARN: run changed during export (${RUN_STAMP} -> ${RUN_AFTER}); file mixes both runs, labelled ${RUN_STAMP}"
      fi
      TMP_FILE="$(mktemp -d)/${OUT_FILE}"
      RUN_STAMP="${RUN_STAMP}" SCRAPED_AT="${STAMP}" python3 - "$OUT_DIR/$OUT_FILE" "$TMP_FILE" <<'PYEOF'
import sys, os, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
meta = {'location_id','latitude','longitude','elevation','time'}
for c in df.columns:
    if c not in meta:
        df[c] = df[c].round(2)
# Self-describing provenance columns (constant -> RLE-compresses to almost nothing):
df['run_init'] = pd.Timestamp(pd.to_datetime(os.environ['RUN_STAMP'], format='%Y%m%dT%HZ', utc=True)).tz_localize(None)
df['scraped_at'] = pd.Timestamp(pd.to_datetime(os.environ['SCRAPED_AT'], format='%Y%m%dT%H%M%SZ', utc=True)).tz_localize(None)
df.to_parquet(dst, compression='zstd')
PYEOF
      # HF filename = model run init time. Re-scrapes of the same run overwrite (idempotent).
      echo "[$(date -u)]     uploading run ${RUN_STAMP} to hf://datasets/${HF_DATASET_REPO}"
      # Hive-style layout: data/model=<domain>/year=YYYY/month=MM/<runStamp>.parquet
      hf upload "${HF_DATASET_REPO}" "${TMP_FILE}" \
        "data/model=${DOMAIN}/year=${RUN_STAMP:0:4}/month=${RUN_STAMP:4:2}/${RUN_STAMP}.parquet" \
        --repo-type dataset --quiet \
        || echo "[$(date -u)]     WARN: HF upload failed for ${OUT_FILE} (kept locally)"
      rm -f "${TMP_FILE}"
    fi
  done
done

echo "[$(date -u)] Done: ${#MODELS[@]} models x ${#REGIONS[@]} regions exported"
