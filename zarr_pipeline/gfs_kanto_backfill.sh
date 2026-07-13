#!/usr/bin/env bash
# One-off: export NCEP GFS 0.13 shortwave for the kanto bbox from Open-Meteo
# S3 data_run archive (retained ~2 weeks) into local parquets, for the
# ICON+GFS blend A/B. Light exports (1 var, small bbox, 40h window).
#
# Usage: ZARR_PY=... ./gfs_kanto_backfill.sh 2026-06-29 2026-07-12 /path/out
set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OM="$(dirname "$DIR")"
PY="${ZARR_PY:-/Users/e3center/.om-venv/bin/python}"
IMAGE="${IMAGE:-open-meteo:bbox-fix}"
REMOTE_DATA="${REMOTE_DATA:-https://openmeteo.s3.amazonaws.com/data/}"
MODEL="ncep_gfs013"
START_DAY="${1:?start day YYYY-MM-DD}"
END_DAY="${2:?end day YYYY-MM-DD}"
OUT_DIR="${3:?output dir}"
mkdir -p "$OUT_DIR"

day="$START_DAY"
while [ "$(date -u -j -f %Y-%m-%d "$day" +%s)" -le "$(date -u -j -f %Y-%m-%d "$END_DAY" +%s)" ]; do
  for hh in 00 06 12 18; do
    STAMP="$(date -u -j -f %Y-%m-%d "$day" +%Y%m%d)T${hh}Z"
    RUN_ISO="${day}T${hh}:00:00"
    FINAL="$OUT_DIR/${STAMP}.parquet"
    if [ -s "$FINAL" ]; then
      echo "[skip] $STAMP exists"
      continue
    fi
    END_DATE=$(date -u -j -v+2d -f %Y-%m-%d "$day" +%Y-%m-%d)
    RAW="$OUT_DIR/${STAMP}_raw.parquet"
    echo "[$(date -u +%H:%M:%S)] exporting $STAMP"
    if docker run --rm -v open-meteo-cache:/app/data -v "$OUT_DIR":/out \
        -e REMOTE_DATA_DIRECTORY="$REMOTE_DATA" --entrypoint /app/openmeteo-api "$IMAGE" \
        export "$MODEL" shortwave_radiation --run "$RUN_ISO" \
        --start_date "$day" --end_date "$END_DATE" \
        --latitude-bounds 33.75,37.85 --longitude-bounds 137.45,141.75 \
        --concurrent 2 --format parquet -o "/out/${STAMP}_raw.parquet" >/dev/null 2>&1; then
      if RUN_STAMP="$STAMP" SCRAPED_AT="$(date -u +%Y%m%dT%H%M%SZ)" \
          "$PY" "$OM/postprocess.py" "$RAW" "$FINAL" >/dev/null 2>&1; then
        echo "  $STAMP ok"
      else
        echo "  $STAMP postprocess FAILED"
      fi
    else
      echo "  $STAMP export FAILED (run may be expired)"
    fi
    rm -f "$RAW"
  done
  day=$(date -u -j -v+1d -f %Y-%m-%d "$day" +%Y-%m-%d)
done
echo "GFS backfill DONE"
