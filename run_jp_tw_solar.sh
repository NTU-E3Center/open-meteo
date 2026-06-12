#!/usr/bin/env bash
#
# Open-Meteo remote-mode pipeline: export Japan/Taiwan grid solar forecast
# directly from the Open-Meteo S3 open-data — no local database, no sync.
#
# `REMOTE_DATA_DIRECTORY` makes `export` read .om files from S3 via HTTP range
# requests, fetching only the bytes covering the bounding box (verified
# 2026-06-12: full JP/TW land box, 9,468 pts x 168 h, in ~19 s, 0% NaN).
# Designed to run every 6 hours (cron). Each run writes a timestamped Parquet;
# keep these files to accumulate your own init_time x lead_time forecast archive.
#
# See HANDOFF_japan_taiwan_solar.md for background and validation history.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
IMAGE="ghcr.io/open-meteo/open-meteo"
REMOTE_DATA="https://openmeteo.s3.amazonaws.com/data/"
CACHE_VOLUME="open-meteo-cache"        # small docker volume for the block cache (not a full DB)
CACHE_SIZE="2GB"                       # block cache size; cache.bin is preallocated at this size
DOMAIN="dwd_icon"                      # global ICON: 0.125°, 6-hourly, covers JP+TW, 7.5-day horizon
OUT_DIR="${HOME}/Desktop/projects/open-meteo/out"   # host dir for parquet output
FORECAST_DAYS=7                        # how many days of forecast to export

# Japan + Taiwan bounding box (lat 20-46N, lon 120-150E)
LAT_BOUNDS="20,46"
LON_BOUNDS="120,150"

# Land-only? For solar you usually want land points; this drops ocean cells and
# shrinks the output massively. Set to "" to keep ALL grid points (incl. sea).
IGNORE_SEA="--ignore_sea"

# Variables to EXPORT. Derived vars are computed on the fly from raw components
# fetched remotely (shortwave = direct + diffuse; DNI from direct + solar geometry;
# wind_speed from u/v; surface_pressure from pressure_msl).
EXPORT_VARS="shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high"

CONCURRENT=8                           # export threads

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------
# Dates in UTC. macOS (BSD date) syntax; for Linux use: date -u -d "+${FORECAST_DAYS} days"
START_DATE="$(date -u +%Y-%m-%d)"
END_DATE="$(date -u -v+${FORECAST_DAYS}d +%Y-%m-%d)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_FILE="jp_tw_solar_${STAMP}.parquet"

mkdir -p "${OUT_DIR}"
docker volume create "${CACHE_VOLUME}" >/dev/null

echo "[$(date -u)] === Open-Meteo JP/TW solar pipeline (remote mode) ==="
echo "  domain=${DOMAIN}  dates=${START_DATE}..${END_DATE}  bbox=lat[${LAT_BOUNDS}] lon[${LON_BOUNDS}]  land_only=${IGNORE_SEA:-no}"

# ---------------------------------------------------------------------------
# Export: read the bbox directly from S3 (HTTP range requests) and write Parquet.
# Notes:
#  - bbox filtering ONLY works with --format parquet (netcdf dumps the whole globe)
#  - values are raw grid-cell values (no per-coordinate elevation downscaling);
#    the parquet includes an `elevation` column (model grid elevation)
#  - times are UTC
# ---------------------------------------------------------------------------
docker run --rm \
  -v "${CACHE_VOLUME}":/app/data \
  -v "${OUT_DIR}":/out \
  -e REMOTE_DATA_DIRECTORY="${REMOTE_DATA}" \
  -e CACHE_SIZE="${CACHE_SIZE}" \
  --entrypoint /app/openmeteo-api "${IMAGE}" \
  export "${DOMAIN}" "${EXPORT_VARS}" \
  --start_date "${START_DATE}" --end_date "${END_DATE}" \
  --latitude-bounds "${LAT_BOUNDS}" --longitude-bounds "${LON_BOUNDS}" \
  ${IGNORE_SEA} \
  --concurrent "${CONCURRENT}" \
  --format parquet -o "/out/${OUT_FILE}"

echo "[$(date -u)] Done: ${OUT_DIR}/${OUT_FILE}"
ls -lh "${OUT_DIR}/${OUT_FILE}"
