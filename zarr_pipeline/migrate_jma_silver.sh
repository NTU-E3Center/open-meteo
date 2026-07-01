#!/usr/bin/env bash
# Migrate the jma_msm .zarr.zip archive (Hugging Face) -> Mode A Silver cube.
#
# Resumable: downloads skip files already present at the right size; the Silver build
# uses --init on first run (pre-allocates the run_init grid + appends) and --append on
# re-runs (idempotent via slot_filled), so re-running after an interruption continues.
#
#   ./migrate_jma_silver.sh          # full migration (background-friendly)
set -euo pipefail

REPO="jimtseng/apac-nwp-forecast-zip"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZARR_PY="${ZARR_PY:-$DIR/.venv/bin/python}"
WORK="${WORK:-$DIR/.work/migrate_jma}"
ZIPS="${ZIPS:-$WORK/zips}"
SILVER="${SILVER:-$DIR/.work/zarr_data/jma_msm_silver.zarr}"
GRID_START="${GRID_START:-2026-05-12}"; GRID_END="${GRID_END:-2026-07-01}"   # run_init grid
mkdir -p "$ZIPS" "$(dirname "$SILVER")"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# 1) list jma_msm .zarr.zip (path + size) from HF
log "listing jma_msm files from HF..."
curl -s "https://huggingface.co/api/datasets/$REPO/tree/main?recursive=true" \
  | "$ZARR_PY" -c "import sys,json;[print(x['path'],x['size']) for x in json.load(sys.stdin) if x['type']=='file' and 'model=jma_msm' in x['path'] and x['path'].endswith('.zarr.zip')]" \
  | sort > "$WORK/manifest.txt"
N=$(wc -l < "$WORK/manifest.txt" | tr -d ' ')
log "$N files to migrate"

# 2) download (resumable: skip if local size already matches)
i=0; dl=0
while read -r path size; do
  i=$((i+1))
  out="$ZIPS/$(basename "$path")"
  if [ -f "$out" ] && [ "$(stat -f%z "$out" 2>/dev/null)" = "$size" ]; then continue; fi
  curl -sL "https://huggingface.co/datasets/$REPO/resolve/main/$path" -o "$out"
  dl=$((dl+1)); [ $((dl % 25)) -eq 0 ] && log "  downloaded $i/$N ($(du -sh "$ZIPS" | cut -f1))"
done < "$WORK/manifest.txt"
log "download complete: $(ls "$ZIPS"/*.zarr.zip | wc -l | tr -d ' ') files, $(du -sh "$ZIPS" | cut -f1)"

# 3) build Silver: --init on first run (grid + append), --append to resume
if [ ! -d "$SILVER" ]; then
  log "building Silver (init grid $GRID_START..$GRID_END, step 3h, lead-max 78) -> $SILVER"
  "$ZARR_PY" "$DIR/convert_to_zarr.py" --source zarrzip --init \
    --data-dir "$ZIPS" --start "$GRID_START" --end "$GRID_END" --step 3 --lead-max 78 \
    --out "$SILVER" --target-mb 8
else
  log "Silver exists -> appending any missing runs (idempotent)"
  "$ZARR_PY" "$DIR/convert_to_zarr.py" --source zarrzip --append \
    --data-dir "$ZIPS" --out "$SILVER"
fi

# 4) verify
log "verifying..."
"$ZARR_PY" - "$SILVER" <<'PYEOF'
import sys, numpy as np, xarray as xr
ds = xr.open_zarr(sys.argv[1], consolidated=True)
filled = int(ds["slot_filled"].sum()) if "slot_filled" in ds else -1
print("  dims:", dict(ds.sizes))
print(f"  filled run_init slots: {filled} / {ds.sizes['run_init']}")
v = next(x for x in ds.data_vars if x != "slot_filled" and ds[x].dtype.kind == "f")
pt = ds[v].sel(latitude=25.0, longitude=121.5, method="nearest").load()
print(f"  sample {v} @(25,121.5): {int(np.isfinite(pt.values).sum())}/{pt.size} finite")
PYEOF
log "DONE. Silver at $SILVER  (downloaded zips kept in $ZIPS; safe to delete -- they are on HF)"
