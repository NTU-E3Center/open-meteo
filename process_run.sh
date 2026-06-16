#!/usr/bin/env bash
#
# Process ONE run: export it from data_run by init-time (--run, full horizon), convert to
# a per-run zarr store, upload to HF. Idempotent. Invoked (optionally in parallel via
# `xargs -P`) by sweep_data_run.sh, one call per missing run.
#
# Args (tab/space separated, from the planner):
#   $1 DOMAIN  $2 RUN_ISO  $3 START_DATE  $4 END_DATE  $5 RUN_STAMP  $6 HF_PATH
# Config comes from the environment (exported by sweep_data_run.sh) with safe defaults.
set -euo pipefail
cd "$(dirname "$0")"

DOMAIN="$1"; RUN_ISO="$2"; START_DATE="$3"; END_DATE="$4"; RUN_STAMP="$5"; HF_PATH="$6"

IMAGE="${IMAGE:-open-meteo:bbox-fix}"
REMOTE_DATA="${REMOTE_DATA:-https://openmeteo.s3.amazonaws.com/data/}"
CACHE_VOLUME="${CACHE_VOLUME:-open-meteo-cache}"
CACHE_SIZE="${CACHE_SIZE:-2GB}"
OUT_DIR="${OUT_DIR:-$(pwd)/out}"
HF_DATASET_REPO="${HF_DATASET_REPO:-JimTseng/apac-nwp-forecast-archive}"
PYBIN="${PYBIN:-python3}"
REGION_LAT="${REGION_LAT:-m44,46}"
REGION_LON="${REGION_LON:-92,154}"
REGION_NAME="${REGION_NAME:-apac}"
IGNORE_SEA="${IGNORE_SEA:---ignore_sea}"
EXPORT_VARS="${EXPORT_VARS:-shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high}"
CONCURRENT="${CONCURRENT:-8}"

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
  rm -f "${OUT_DIR}/${OUT_FILE}"; exit 0
fi

ZARR_TMP="$(mktemp -d)/${RUN_STAMP}.zarr"
if ! RUN_STAMP="${RUN_STAMP}" SCRAPED_AT="$(date -u +%Y%m%dT%H%M%SZ)" \
     "${PYBIN}" parquet_to_zarr.py "${OUT_DIR}/${OUT_FILE}" "${ZARR_TMP}"; then
  echo "[$(date -u)]     ERROR: ${DOMAIN} ${RUN_STAMP} zarr conversion failed — skipping"
  rm -rf "$(dirname "${ZARR_TMP}")" "${OUT_DIR}/${OUT_FILE}"; exit 0
fi

echo "[$(date -u)]     uploading ${DOMAIN} ${RUN_STAMP} -> ${HF_PATH}"
if hf upload "${HF_DATASET_REPO}" "${ZARR_TMP}" "${HF_PATH}" --repo-type dataset --quiet; then
  echo "[$(date -u)]     OK ${DOMAIN} ${RUN_STAMP}"
else
  echo "[$(date -u)]     WARN: HF upload failed for ${DOMAIN} ${RUN_STAMP} (next sweep retries)"
fi
rm -rf "$(dirname "${ZARR_TMP}")" "${OUT_DIR}/${OUT_FILE}"
