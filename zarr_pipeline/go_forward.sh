#!/usr/bin/env bash
# Go-forward pipeline -- keeps Bronze + the SINGLE Silver cube up to date with Open-Meteo runs.
# The Silver cube (<model>_silver.zarr on HF) has its run_init axis pre-extended to 2028
# (scripts/extend_hf_cube.sh), so new runs just region-write into their empty slot.
#
# Two modes:
#   ./go_forward.sh jma_msm                 # LATEST run only
#   ./go_forward.sh jma_msm --backfill 3    # every MISSING run in the last 3 days (self-healing)
#   ./go_forward.sh dwd_icon 2026-06-30T06  # one specific run
#
# --backfill N is the recommended daily cron: it reads the cube's slot_filled, finds every run
# in the last N days that ISN'T in the cube yet (cron miss, crash, transient failure), and fills
# only those -- so a once-a-day run can't permanently drop data (S3 keeps ~3 months).
#
# Per run: export S3 -> postprocess -> parquet -> upload to Bronze -> region-write into the cube
# skeleton (downloaded once, a few MB). All new shard files are uploaded in ONE commit at the end.
# The existing 28.8 GB / 176 GB of data on HF is never downloaded nor re-uploaded.
set -euo pipefail

MODEL="${1:?usage: go_forward.sh <jma_msm|dwd_icon> [RUN_ISO | --backfill N]}"
ARG2="${2:-}"; ARG3="${3:-}"
# Self-locating: this script lives in open-meteo/zarr_pipeline/. OM = the open-meteo root
# (holds postprocess.py + the Docker build). One venv with everything (see requirements.txt).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # .../open-meteo/zarr_pipeline
OM="$(dirname "$DIR")"                                     # .../open-meteo
PY="${ZARR_PY:-$DIR/.venv/bin/python}"                    # xarray+zarr+pandas+huggingface_hub>=1.19
IMAGE="${IMAGE:-open-meteo:bbox-fix}"
REMOTE_DATA="https://openmeteo.s3.amazonaws.com/data/"
REGION_LAT="${REGION_LAT:-m44,46}"; REGION_LON="${REGION_LON:-92,154}"
REPO_BRONZE="jimtseng/apac-nwp-forecast-raw"
REPO_SILVER="jimtseng/apac-nwp-forecast"
CUBE_NAME="${MODEL}_silver.zarr"
WORK="$DIR/.work/goforward"; rm -rf "$WORK"; mkdir -p "$WORK"; SKEL="$WORK/skel"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [ "$MODEL" = "jma_msm" ]; then
  VARS="shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high"
elif [ "$MODEL" = "dwd_icon" ]; then
  VARS="shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,snow_depth,snowfall_water_equivalent"
else echo "unknown model $MODEL"; exit 1; fi

# ---- 0. download cube skeleton once (metadata + coords; a few MB, no data chunks) ----
log "$MODEL -> bronze + $CUBE_NAME | downloading skeleton..."
"$PY" - "$REPO_SILVER" "$CUBE_NAME" "$SKEL" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo, cube, dest = sys.argv[1], sys.argv[2], sys.argv[3]
coords = "run_init slot_filled lead latitude longitude elevation location_id".split()
patterns = [f"{cube}/zarr.json", f"{cube}/*/zarr.json"] + [f"{cube}/{c}/c/**" for c in coords]
snapshot_download(repo, repo_type="dataset", allow_patterns=patterns, local_dir=dest)
PY
LOCAL_CUBE="$SKEL/$CUBE_NAME"

# ---- 1. decide which run STAMPS to process --------------------------------
STAMPS=""
if [ "$ARG2" = "--backfill" ]; then
  N="${ARG3:?--backfill needs N days}"
  # missing stamps = run_init slots in the last N days with slot_filled==0 and run_init <= now
  STAMPS=$("$PY" - "$LOCAL_CUBE" "$N" <<'PY'
import sys, numpy as np, xarray as xr, pandas as pd, warnings; warnings.filterwarnings("ignore")
ds = xr.open_zarr(sys.argv[1], consolidated=True)
ri = ds["run_init"].values.astype("datetime64[h]")
filled = ds["slot_filled"].values.astype(int)
now = np.datetime64(pd.Timestamp.utcnow().tz_localize(None), "h")
cutoff = now - np.timedelta64(int(sys.argv[2]) * 24, "h")
for t, f in zip(ri, filled):
    if f == 0 and cutoff <= t <= now:
        print(pd.Timestamp(t).strftime("%Y%m%dT%HZ"))
PY
)
elif [ -n "$ARG2" ]; then
  STAMPS="$(echo "$ARG2" | sed 's/[-:]//g;s/T\([0-9]\{2\}\).*/T\1Z/')"
else
  RS=$("$PY" -c "import urllib.request,json,datetime;m=json.load(urllib.request.urlopen('${REMOTE_DATA}${MODEL}/static/meta.json',timeout=30));print(datetime.datetime.fromtimestamp(m['last_run_initialisation_time'],datetime.UTC).strftime('%Y%m%dT%HZ'))")
  STAMPS="$RS"
fi

if [ -z "$STAMPS" ]; then log "nothing to do (no missing runs)"; rm -rf "$WORK"; exit 0; fi
log "runs to process: $(echo "$STAMPS" | wc -w | tr -d ' ') -> $(echo $STAMPS | tr '\n' ' ')"

# ---- 2. per stamp: export -> postprocess -> bronze -> region-write append --
MARK="$WORK/.mark"; : > "$MARK"; sleep 1
DD="$WORK/dd"; mkdir -p "$DD"
wrote_any=0
for STAMP in $STAMPS; do
  RUN_ISO="${STAMP:0:4}-${STAMP:4:2}-${STAMP:6:2}T${STAMP:9:2}:00"
  START="${STAMP:0:4}-${STAMP:4:2}-${STAMP:6:2}"; END="$(date -u -j -v+5d -f '%Y-%m-%d' "$START" '+%Y-%m-%d')"
  RAW="$WORK/${STAMP}_raw.parquet"; PARQUET="$DD/${STAMP}.parquet"
  if ! docker run --rm -v open-meteo-cache:/app/data -v "$WORK":/out \
      -e REMOTE_DATA_DIRECTORY="$REMOTE_DATA" --entrypoint /app/openmeteo-api "$IMAGE" \
      export "$MODEL" "$VARS" --run "$RUN_ISO" --start_date "$START" --end_date "$END" \
      --latitude-bounds "$REGION_LAT" --longitude-bounds "$REGION_LON" --concurrent 8 \
      --format parquet -o "/out/${STAMP}_raw.parquet" >/dev/null 2>&1; then
    log "  $STAMP: export failed (S3 may not have it) -> skip"; rm -f "$RAW"; continue
  fi
  RUN_STAMP="$STAMP" SCRAPED_AT="$(date -u +%Y%m%dT%H%M%SZ)" "$PY" "$OM/postprocess.py" "$RAW" "$PARQUET" >/dev/null 2>&1 || { log "  $STAMP: postprocess failed -> skip"; rm -f "$RAW" "$PARQUET"; continue; }
  rm -f "$RAW"
  HFPATH="data/model=${MODEL}/year=${STAMP:0:4}/month=${STAMP:4:2}/${STAMP}.parquet"
  "$PY" -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='$PARQUET', path_in_repo='$HFPATH', repo_id='$REPO_BRONZE', repo_type='dataset')" >/dev/null 2>&1
  log "  $STAMP: bronze ok"
done

# region-write ALL fetched parquets into the cube skeleton in one pass (skips already-filled)
if ls "$DD"/*.parquet >/dev/null 2>&1; then
  OUT=$("$PY" "$DIR/convert_to_zarr.py" --source parquet --append --data-dir "$DD" --out "$LOCAL_CUBE" 2>&1) || { echo "$OUT"; exit 1; }
  echo "$OUT" | grep -E "written|already-filled|off-grid" | tail -1
  echo "$OUT" | grep -q "+0 written" || wrote_any=1
fi

# ---- 3. upload only new/changed files (the runs' shards + slot_filled), one commit ----
if [ "$wrote_any" = "1" ]; then
  "$PY" - "$REPO_SILVER" "$CUBE_NAME" "$LOCAL_CUBE" "$MARK" <<'PY'
import sys, os
from huggingface_hub import HfApi, CommitOperationAdd
repo, cube, local, mark = sys.argv[1:5]
cutoff = os.path.getmtime(mark)
ops = []
for dp, _, files in os.walk(local):
    for f in files:
        full = os.path.join(dp, f)
        if os.path.getmtime(full) <= cutoff: continue
        rel = os.path.relpath(full, os.path.dirname(local))
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=full))
print(f"  uploading {len(ops)} new/changed files to silver")
if ops:
    HfApi().create_commit(repo, repo_type="dataset", operations=ops,
                          commit_message=f"go-forward: update {cube}")
PY
else
  log "  nothing new written -> no silver upload"
fi
rm -rf "$WORK"
log "DONE"
