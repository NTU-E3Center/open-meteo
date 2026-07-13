#!/usr/bin/env bash
# Backfill <model>_silver.zarr from EXISTING Bronze parquets on Hugging Face.
# No Docker / Open-Meteo export involved: for every empty silver slot
# (slot_filled==0) that has a Bronze parquet, download the parquet and
# region-write it into the cube, then upload changed files in one commit.
#
# Usage:
#   ZARR_PY=/path/to/python ./backfill_silver_from_bronze.sh jma_msm [--days 14] [--list-only]
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ZARR_PY:-$DIR/.venv/bin/python}"
MODEL="${1:?usage: backfill_silver_from_bronze.sh <model> [--days N] [--list-only]}"
DAYS=14
LIST_ONLY=0
shift
while [ "$#" -gt 0 ]; do
  case "$1" in
    --days) DAYS="${2:?--days needs N}"; shift 2 ;;
    --list-only) LIST_ONLY=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

REPO_BRONZE="jimtseng/apac-nwp-forecast-raw"
REPO_SILVER="jimtseng/apac-nwp-forecast"
CUBE_NAME="${MODEL}_silver.zarr"
WORK="$DIR/.work/backfill-$MODEL"; mkdir -p "$WORK"; SKEL="$WORK/skel"
rm -rf "$WORK/dd" "$WORK/.mark"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ---- 1. download cube skeleton (metadata + coords, no data chunks) --------
# Reused across invocations (list-only then real run); delete $WORK to force
# a fresh skeleton if silver was updated elsewhere since.
if [ -f "$SKEL/$CUBE_NAME/zarr.json" ]; then
  log "$MODEL: reusing existing skeleton at $SKEL/$CUBE_NAME"
else
log "$MODEL: downloading $CUBE_NAME skeleton..."
"$PY" - "$REPO_SILVER" "$CUBE_NAME" "$SKEL" <<'PY'
import os
import shutil
import sys
import time
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

repo, cube, dest = sys.argv[1], sys.argv[2], sys.argv[3]
coords = "run_init slot_filled lead latitude longitude elevation location_id".split()
api = HfApi()

files = [f"{cube}/zarr.json"]
for item in api.list_repo_tree(repo_id=repo, repo_type="dataset", path_in_repo=cube, recursive=False):
    path = getattr(item, "path", "")
    if not path or path == f"{cube}/zarr.json":
        continue
    if path.startswith(f"{cube}/"):
        files.append(f"{path}/zarr.json")

for coord in coords:
    for item in api.list_repo_tree(repo_id=repo, repo_type="dataset", path_in_repo=f"{cube}/{coord}/c", recursive=True):
        path = getattr(item, "path", "")
        if path and isinstance(item, RepoFile):
            files.append(path)

seen = set()
for path in files:
    if path in seen:
        continue
    seen.add(path)
    for attempt in range(1, 4):
        try:
            cached = hf_hub_download(repo_id=repo, repo_type="dataset", filename=path)
            target = os.path.join(dest, path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(cached, target)
            break
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"download failed for {path} on attempt {attempt}: {exc}; retrying", file=sys.stderr)
            time.sleep(2 * attempt)
print(f"downloaded skeleton files: {len(seen)}")
PY
fi
LOCAL_CUBE="$SKEL/$CUBE_NAME"

# ---- 2. stamps = empty silver slots that HAVE a bronze parquet ------------
STAMPS=$("$PY" - "$LOCAL_CUBE" "$DAYS" "$REPO_BRONZE" "$MODEL" <<'PY'
import sys, warnings
import numpy as np, pandas as pd, xarray as xr
from huggingface_hub import HfApi
warnings.filterwarnings("ignore")

cube, days, repo, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
ds = xr.open_zarr(cube, consolidated=True)
ri = ds["run_init"].values.astype("datetime64[h]")
filled = ds["slot_filled"].values.astype(int)
now = np.datetime64(pd.Timestamp.utcnow().floor("h").tz_localize(None), "h")
cutoff = now - np.timedelta64(days * 24, "h")
empty = {pd.Timestamp(t).strftime("%Y%m%dT%HZ") for t, f in zip(ri, filled) if f == 0 and cutoff <= t <= now}

api = HfApi()
bronze = set()
for item in api.list_repo_tree(repo_id=repo, repo_type="dataset", path_in_repo=f"data/model={model}", recursive=True):
    path = getattr(item, "path", "")
    if path.endswith(".parquet"):
        bronze.add(path.rsplit("/", 1)[-1].replace(".parquet", ""))

for stamp in sorted(empty & bronze, reverse=True):
    print(stamp)
PY
)

if [ -z "$STAMPS" ]; then log "nothing to backfill (no empty slots with bronze parquets)"; exit 0; fi
log "backfillable runs: $(echo "$STAMPS" | wc -w | tr -d ' ') -> $(echo $STAMPS | tr '\n' ' ')"
if [ "$LIST_ONLY" = "1" ]; then log "LIST_ONLY; exiting (skeleton kept for the real run)"; exit 0; fi

# ---- 3. download bronze parquets ------------------------------------------
MARK="$WORK/.mark"; : > "$MARK"; sleep 1
DD="$WORK/dd"; mkdir -p "$DD"
for STAMP in $STAMPS; do
  HFPATH="data/model=${MODEL}/year=${STAMP:0:4}/month=${STAMP:4:2}/${STAMP}.parquet"
  log "  fetching bronze $STAMP"
  "$PY" - "$REPO_BRONZE" "$HFPATH" "$DD/${STAMP}.parquet" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
cached = hf_hub_download(repo_id=sys.argv[1], repo_type="dataset", filename=sys.argv[2])
shutil.copy2(cached, sys.argv[3])
PY
done

# ---- 4. region-write into the cube, upload changed files ------------------
OUT=$("$PY" "$DIR/convert_to_zarr.py" --source parquet --append --data-dir "$DD" --out "$LOCAL_CUBE" 2>&1) || { echo "$OUT"; exit 1; }
echo "$OUT" | grep -E "written|already-filled|off-grid" | tail -1
if echo "$OUT" | grep -q "+0 written"; then
  log "nothing newly written -> no silver upload"
  rm -rf "$WORK/dd"
  exit 0
fi

"$PY" - "$REPO_SILVER" "$CUBE_NAME" "$LOCAL_CUBE" "$MARK" <<'PY'
import os, sys
from huggingface_hub import HfApi, CommitOperationAdd
repo, cube, local, mark = sys.argv[1:5]
cutoff = os.path.getmtime(mark)
ops = []
for dp, _, files in os.walk(local):
    for f in files:
        full = os.path.join(dp, f)
        if os.path.getmtime(full) <= cutoff:
            continue
        rel = os.path.relpath(full, os.path.dirname(local))
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=full))
print(f"  uploading {len(ops)} new/changed files to silver")
if ops:
    HfApi().create_commit(repo, repo_type="dataset", operations=ops,
                          commit_message=f"backfill-from-bronze: update {cube}")
PY
rm -rf "$WORK/dd"
log "DONE (skeleton kept in $SKEL; delete $WORK when finished backfilling)"
