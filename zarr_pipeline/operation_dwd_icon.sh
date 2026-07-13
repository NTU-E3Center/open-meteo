#!/usr/bin/env bash
# Direct, high-frequency DWD ICON operation ingestion. This is intentionally separate
# from go_forward.sh: it neither uploads Bronze nor updates the Silver archive.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OM="$(dirname "$DIR")"
PY="${ZARR_PY:-$DIR/.venv/bin/python}"
IMAGE="${IMAGE:-open-meteo:bbox-fix}"
REMOTE_DATA="${REMOTE_DATA:-https://openmeteo.s3.amazonaws.com/data/}"
LAT_BOUNDS="${LAT_BOUNDS:-20,46}"
LON_BOUNDS="${LON_BOUNDS:-119,146}"
OUT_STORE="${OUT_STORE:-/nas/solar-operation/raw/nwp/dwd_icon_latest.zarr}"
RUN_ISO=""
DRY_RUN=0
VARS="${VARS:-shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,temperature_2m,wind_speed_10m}"

usage() {
  echo "usage: operation_dwd_icon.sh [--run ISO] [--out PATH] [--dry-run]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run) RUN_ISO="${2:?--run needs an ISO timestamp}"; shift 2 ;;
    --out) OUT_STORE="${2:?--out needs a directory Zarr path}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [ -z "$RUN_ISO" ]; then
  RUN_ISO=$(python3 - "$REMOTE_DATA" <<'PY'
import datetime as dt
import json
import sys
import urllib.request

meta = json.load(urllib.request.urlopen(sys.argv[1] + "dwd_icon/static/meta.json", timeout=30))
print(dt.datetime.fromtimestamp(meta["last_run_initialisation_time"], dt.UTC).strftime("%Y-%m-%dT%H:00:00"))
PY
)
fi
RUN_ISO=$(python3 - "$RUN_ISO" <<'PY'
import datetime as dt
import sys

value = dt.datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00"))
if value.tzinfo is None:
    value = value.replace(tzinfo=dt.UTC)  # bare timestamps are UTC, never local
print(value.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:00:00"))
PY
)
STAMP=$(python3 - "$RUN_ISO" <<'PY'
import datetime as dt
import sys

value = dt.datetime.fromisoformat(sys.argv[1])
print(value.strftime("%Y%m%dT%HZ"))
PY
)
START=$(python3 - "$RUN_ISO" <<'PY'
import datetime as dt
import sys

value = dt.datetime.fromisoformat(sys.argv[1])
print(value.strftime("%Y-%m-%d"))
PY
)
END=$(python3 - "$RUN_ISO" <<'PY'
import datetime as dt
import sys

value = dt.datetime.fromisoformat(sys.argv[1])
print((value + dt.timedelta(hours=48)).strftime("%Y-%m-%d"))
PY
)

# Skip when the resolved run is already in the cache: lets the cron run hourly
# (minimizing fetch delay) while the heavy docker export still only happens
# when DWD actually publishes a new run (4x/day). FORCE=1 overrides.
RUN_MARKER="${OUT_STORE%.zarr}.run"
if [ "${FORCE:-0}" != "1" ] && [ -f "$RUN_MARKER" ] && [ "$(cat "$RUN_MARKER" 2>/dev/null)" = "$STAMP" ]; then
  echo "status=ok run=$RUN_ISO already cached (marker $RUN_MARKER); skipping export"
  exit 0
fi

echo "model=dwd_icon"
echo "run=$RUN_ISO"
echo "bounds=$LAT_BOUNDS,$LON_BOUNDS"
echo "output=$OUT_STORE"
echo "docker export dwd_icon $VARS --run $RUN_ISO --start_date $START --end_date $END --latitude-bounds $LAT_BOUNDS --longitude-bounds $LON_BOUNDS --format parquet"
echo "$PY $OM/postprocess.py <raw.parquet> <${STAMP}.parquet>"
echo "$PY $DIR/convert_to_zarr.py --source parquet --data-dir <work> --out $OUT_STORE --overwrite --model dwd_icon"

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi
if [ ! -x "$PY" ]; then
  echo "missing ZARR_PY interpreter: $PY" >&2
  exit 1
fi

# Work dir must live under $HOME: colima only shares $HOME into the Docker VM,
# so a /tmp path mounted as /out is not writable from inside the container.
WORK="${WORK_ROOT:-$DIR/.work}/operation-dwd-icon.$$"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
RAW="$WORK/${STAMP}_raw.parquet"
PARQUET="$WORK/${STAMP}.parquet"

docker run --rm -v open-meteo-cache:/app/data -v "$WORK":/out \
  -e REMOTE_DATA_DIRECTORY="$REMOTE_DATA" --entrypoint /app/openmeteo-api "$IMAGE" \
  export dwd_icon "$VARS" --run "$RUN_ISO" --start_date "$START" --end_date "$END" \
  --latitude-bounds "$LAT_BOUNDS" --longitude-bounds "$LON_BOUNDS" --concurrent 8 \
  --format parquet -o "/out/${STAMP}_raw.parquet"

RUN_STAMP="$STAMP" SCRAPED_AT="$(date -u +%Y%m%dT%H%M%SZ)" \
  "$PY" "$OM/postprocess.py" "$RAW" "$PARQUET"
"$PY" "$DIR/convert_to_zarr.py" --source parquet --data-dir "$WORK" \
  --out "$OUT_STORE" --overwrite --model dwd_icon
echo "$STAMP" > "$RUN_MARKER"
echo "status=ok output_store=$OUT_STORE run=$RUN_ISO"
