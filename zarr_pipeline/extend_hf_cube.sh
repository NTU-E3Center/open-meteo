#!/usr/bin/env bash
# Extend a Silver cube's run_init axis on Hugging Face -- METADATA ONLY (data chunks untouched).
#
# Downloads just the cube's SKELETON (all zarr.json + the small coordinate/slot_filled arrays,
# a few MB -- NOT the GBs of data-var chunks), extends the run_init axis with
# scripts/extend_run_init.py, then uploads back ONLY the skeleton (changed zarr.json +
# run_init/slot_filled). The data-var chunk folders are never downloaded nor uploaded, so the
# large data on HF stays exactly as-is.
#
#   ./extend_hf_cube.sh jma_msm_silver.zarr 2028-01-01 6
set -euo pipefail

CUBE_NAME="${1:?usage: extend_hf_cube.sh <cube.zarr name> <end YYYY-MM-DD> <step hours>}"
END="${2:?need end date}"
STEP="${3:?need step hours}"
REPO="jimtseng/apac-nwp-forecast"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # .../open-meteo/zarr_pipeline
PY="${ZARR_PY:-$DIR/.venv/bin/python}"                    # single venv (xarray + huggingface_hub>=1.19)
WORK="$DIR/.work/extend"; SKEL="$WORK/skel"
rm -rf "$WORK"; mkdir -p "$SKEL"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

COORDS="run_init slot_filled lead latitude longitude elevation location_id"

# 1) download skeleton only (all zarr.json everywhere + coord/slot_filled chunk data)
log "downloading skeleton of $CUBE_NAME (metadata + coords only)..."
"$PY" - "$REPO" "$CUBE_NAME" "$SKEL" "$COORDS" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
repo, cube, dest, coords = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
patterns = [f"{cube}/zarr.json", f"{cube}/*/zarr.json"] + [f"{cube}/{c}/c/**" for c in coords]
p = snapshot_download(repo, repo_type="dataset", allow_patterns=patterns, local_dir=dest)
print("  downloaded skeleton to", os.path.join(p, cube))
PY
LOCAL_CUBE="$SKEL/$CUBE_NAME"
echo "  skeleton size: $(du -sh "$LOCAL_CUBE" | cut -f1) (data chunks NOT included)"

# 2) extend the axis on the local skeleton (data-var resize is metadata-only; coords are present)
log "extending run_init axis to $END (step ${STEP}h)..."
"$PY" "$DIR/extend_run_init.py" "$LOCAL_CUBE" --end "$END" --step "$STEP"

# 3) upload back ONLY the skeleton (zarr.json + run_init/slot_filled). The skeleton contains NO
#    data-var chunks, so the large data on HF is untouched. path_in_repo keeps the cube prefix.
log "uploading changed metadata back to $REPO ..."
"$PY" - "$REPO" "$CUBE_NAME" "$LOCAL_CUBE" <<'PY'
import sys, os
from huggingface_hub import HfApi, CommitOperationAdd
repo, cube, local = sys.argv[1], sys.argv[2], sys.argv[3]
ops = []
for dp, _, files in os.walk(local):
    for f in files:
        full = os.path.join(dp, f)
        rel = os.path.relpath(full, os.path.dirname(local))  # keeps "<cube>/..." prefix
        ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=full))
print(f"  uploading {len(ops)} skeleton files (no data chunks) in one commit")
HfApi().create_commit(repo, repo_type="dataset", operations=ops,
                      commit_message=f"extend {cube} run_init axis (metadata only)")
print("  committed")
PY

log "DONE. Verify with: open the cube from HF and check run_init length + old data intact."
