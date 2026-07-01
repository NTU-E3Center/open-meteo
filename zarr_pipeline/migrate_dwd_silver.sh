#!/usr/bin/env bash
# Migrate the dwd_icon .zarr.zip archive (Hugging Face) -> Mode A Silver cube.
#
# ~216 GB / 404 files. Works in BATCHES (parallel download -> append -> delete) to stay
# disk-safe and fast; resumable via done.txt (skips finished runs, no re-download).
#
# Variable heterogeneity: 2026-03-19..21 lack snow (13 vars); 03-22+ have 15. We SEED-INIT
# from a 15-var run so the cube is a SUPERSET; 13-var runs append as a subset (snow stays
# the missing-value sentinel -- unrecoverable anyway). The source `model` attr is
# mislabelled "jma_msm" -> overridden to dwd_icon. After build: reshard + upload (hub 1.19.0).
set -euo pipefail

REPO="jimtseng/apac-nwp-forecast-zip"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZARR_PY="${ZARR_PY:-$DIR/.venv/bin/python}"
WORK="${WORK:-$DIR/.work/migrate_dwd}"
ZIPS="$WORK/zips"
SILVER="${SILVER:-$DIR/.work/zarr_data/dwd_icon_silver.zarr}"
DONE="$WORK/done.txt"
BATCH="${BATCH:-8}"                 # files downloaded (in parallel) then appended per round
DL_JOBS="${DL_JOBS:-8}"            # concurrent downloads (HF caps per-connection -> parallel helps)
GRID_START="${GRID_START:-2026-03-19}"; GRID_END="${GRID_END:-2026-07-01}"
export BASE="https://huggingface.co/datasets/$REPO/resolve/main"
export ZIPS
mkdir -p "$ZIPS" "$(dirname "$SILVER")"; touch "$DONE"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# download one "path size" line (skip if already present at the right size); used by xargs
dl_one() {
  set -- $1
  local path="$1" size="$2" stamp out
  stamp="$(basename "$path" .zarr.zip)"; out="$ZIPS/$stamp.zarr.zip"
  [ -f "$out" ] && [ "$(stat -f%z "$out" 2>/dev/null)" = "$size" ] && return 0
  curl -sL --retry 8 --retry-delay 5 --retry-all-errors --connect-timeout 30 \
    "$BASE/$path" -o "$out"
}
export -f dl_one

log "listing dwd_icon files from HF..."
curl -s "https://huggingface.co/api/datasets/$REPO/tree/main?recursive=true" \
  | "$ZARR_PY" -c "import sys,json;[print(x['path'],x['size']) for x in json.load(sys.stdin) if x['type']=='file' and 'model=dwd_icon' in x['path'] and x['path'].endswith('.zarr.zip')]" \
  | sort > "$WORK/manifest.txt"
N=$(wc -l < "$WORK/manifest.txt" | tr -d ' ')
log "$N dwd files | already done: $(wc -l < "$DONE" | tr -d ' ')"

# --- seed-init: 15-var superset cube from the latest run (skipped if cube exists) ----
if [ ! -d "$SILVER" ]; then
  SEED_PATH=$(tail -1 "$WORK/manifest.txt" | awk '{print $1}')
  SEED_STAMP=$(basename "$SEED_PATH" .zarr.zip)
  SEED_DIR="$WORK/seed"; rm -rf "$SEED_DIR"; mkdir -p "$SEED_DIR"
  log "seed-init from latest run $SEED_STAMP (full-variable schema)"
  curl -sL --retry 8 --retry-all-errors "$BASE/$SEED_PATH" -o "$SEED_DIR/$SEED_STAMP.zarr.zip"
  "$ZARR_PY" "$DIR/convert_to_zarr.py" --source zarrzip --init \
    --data-dir "$SEED_DIR" --start "$GRID_START" --end "$GRID_END" --step 6 --lead-max 180 \
    --out "$SILVER" --target-mb 8 --model dwd_icon
  echo "$SEED_STAMP" >> "$DONE"; rm -rf "$SEED_DIR"
  log "  seed-init done"
fi

flush() {
  ls "$ZIPS"/*.zarr.zip >/dev/null 2>&1 || return 0
  "$ZARR_PY" "$DIR/convert_to_zarr.py" --source zarrzip --append \
    --data-dir "$ZIPS" --out "$SILVER"
  for z in "$ZIPS"/*.zarr.zip; do basename "$z" .zarr.zip >> "$DONE"; done
  rm -f "$ZIPS"/*.zarr.zip
  log "  flushed; $(wc -l < "$DONE" | tr -d ' ')/$N runs ($(du -sh "$SILVER" 2>/dev/null | cut -f1))"
}

# pending = manifest lines whose stamp isn't done yet
pending=()
while read -r path size; do
  grep -qx "$(basename "$path" .zarr.zip)" "$DONE" && continue
  pending+=("$path $size")
done < "$WORK/manifest.txt"
total=${#pending[@]}
log "to do: $total runs | BATCH=$BATCH DL_JOBS=$DL_JOBS"

idx=0
while [ "$idx" -lt "$total" ]; do
  batch=("${pending[@]:$idx:$BATCH}")
  printf '%s\n' "${batch[@]}" | xargs -P "$DL_JOBS" -I {} bash -c 'dl_one "$1"' _ {}
  log "downloaded batch ($((idx+${#batch[@]}))/$total) -> append"
  flush
  idx=$((idx + BATCH))
done

log "DONE build: $(wc -l < "$DONE" | tr -d ' ')/$N runs -> $SILVER ($(du -sh "$SILVER" | cut -f1))"
log "NEXT: reshard + upload"
