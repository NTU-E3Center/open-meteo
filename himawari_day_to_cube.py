#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write one UTC day of Himawari SWR into the silver cube on HuggingFace.

Two modes:
    himawari_day_to_cube.py --day YYYY-MM-DD --from-hf
        Download the day's zip from HF (jimtseng/apac-himawari-swr,
        data/year=YYYY/month=MM/YYYYMMDD.zarr.zip) via hf_hub_download,
        then open and region-write into the silver cube.

    himawari_day_to_cube.py --day YYYY-MM-DD --from-zip /path/to/YYYYMMDD.zarr.zip
        Open a local zip using the same _open_zip + _normalize_times logic.

Go-forward idiom (mirrors go_forward.sh):
    1. Download cube SKELETON ONLY (metadata + coords, no data chunks).
    2. Check idempotency: if day_filled on the local skeleton is already True, exit 0.
    3. Region-write the day into the local skeleton copy.
    4. Upload ONLY the new/changed chunk files in ONE HF commit.

Testing hook: --store-path <path> overrides the HF skeleton download and upload.
When --store-path is given the script writes directly into that local store and
skips all HF operations, making the core local write path testable without network.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

# Ensure zarr_pipeline is importable when the script lives in open-meteo/ root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "zarr_pipeline"))

from himawari_cube import day_filled, write_day  # noqa: E402
from himawari_transcode import _normalize_times, _open_zip  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO = "jimtseng/apac-himawari-swr"
HF_REPO_TYPE = "dataset"
HF_SILVER_REPO = "jimtseng/apac-himawari-swr"
HF_SILVER_REPO_TYPE = "dataset"
CUBE_NAME = "himawari_swr_silver.zarr"

# Coordinate variable names needed for skeleton download
_COORD_VARS = ["time", "latitude", "longitude", "slot_filled"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day_hf_path(day: str) -> str:
    """Return the HF repo path for the given day's zarr.zip.

    Parameters
    ----------
    day: ISO date string, e.g. "2026-03-23"
    """
    ymd = day.replace("-", "")          # YYYYMMDD
    yyyy, mm = ymd[:4], ymd[4:6]
    return f"data/year={yyyy}/month={mm}/{ymd}.zarr.zip"


def _download_day_zip(day: str) -> str:
    """Download a day zip from HF. Returns the local cache path."""
    from huggingface_hub import hf_hub_download
    hf_path = _day_hf_path(day)
    return hf_hub_download(
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        filename=hf_path,
    )


def _download_cube_skeleton(skel_dir: str) -> str:
    """Download the silver cube skeleton (metadata + coords) to skel_dir.

    Mirrors the go_forward.sh skeleton-only download pattern using
    snapshot_download with allow_patterns restricted to .json files and
    coordinate chunk files.  Returns the path to the local cube.
    """
    from huggingface_hub import snapshot_download

    # coord chunk patterns: c/** covers all chunk files under the coord arrays
    patterns = (
        [f"{CUBE_NAME}/zarr.json", f"{CUBE_NAME}/*/zarr.json"]
        + [f"{CUBE_NAME}/{v}/c/**" for v in _COORD_VARS]
    )
    snapshot_download(
        HF_SILVER_REPO,
        repo_type=HF_SILVER_REPO_TYPE,
        allow_patterns=patterns,
        local_dir=skel_dir,
    )
    return os.path.join(skel_dir, CUBE_NAME)


def _upload_changed_files(local_cube: str, mark_mtime: float) -> None:
    """Upload all files in local_cube newer than mark_mtime in a single HF commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    ops = []
    for dirpath, _, files in os.walk(local_cube):
        for fname in files:
            full = os.path.join(dirpath, fname)
            if os.path.getmtime(full) <= mark_mtime:
                continue
            # path in repo is relative to the parent of local_cube
            rel = os.path.relpath(full, os.path.dirname(local_cube))
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=full))

    if not ops:
        print("  no new/changed files — nothing to upload", flush=True)
        return

    print(f"  uploading {len(ops)} new/changed file(s) to silver cube …", flush=True)
    HfApi().create_commit(
        HF_SILVER_REPO,
        repo_type=HF_SILVER_REPO_TYPE,
        operations=ops,
        commit_message=f"himawari-day-to-cube: add {os.path.basename(local_cube)} day",
    )
    print("  upload done", flush=True)


# ---------------------------------------------------------------------------
# Core write logic
# ---------------------------------------------------------------------------

def _write_day_to_store(store_path: str, day: str, local_zip: str) -> None:
    """Open *local_zip*, normalize timestamps, and region-write into *store_path*.

    Raises on any error so the caller can propagate a hard exit.
    """
    tmpdir: str | None = None
    try:
        day_ds, tmpdir = _open_zip(local_zip)
        # _open_zip already calls _normalize_times internally; no double call needed.
        write_day(store_path, day_ds)
        day_ds.close()
    finally:
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Region-write one UTC day of Himawari SWR into the silver cube."
    )
    ap.add_argument("--day", required=True, metavar="YYYY-MM-DD",
                    help="UTC day to write (ISO format).")

    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--from-hf", action="store_true",
                           help="Fetch the day zip from HuggingFace (jimtseng/apac-himawari-swr).")
    src_group.add_argument("--from-zip", metavar="PATH",
                           help="Use a local day zarr.zip instead of fetching from HF.")

    ap.add_argument(
        "--store-path", metavar="PATH", default=None,
        help=(
            "Override: write into this local zarr store instead of downloading the HF "
            "skeleton and uploading changes.  Intended for local testing only — skips "
            "all HF operations."
        ),
    )
    args = ap.parse_args()

    day = args.day

    # ------------------------------------------------------------------
    # 1. Resolve the day zip (local path)
    # ------------------------------------------------------------------
    if args.from_zip:
        if not os.path.exists(args.from_zip):
            print(f"ERROR: --from-zip path not found: {args.from_zip}", flush=True)
            sys.exit(1)
        local_zip = args.from_zip
        print(f"[{day}] using local zip: {local_zip}", flush=True)
    else:
        # --from-hf
        print(f"[{day}] downloading day zip from HF …", flush=True)
        try:
            local_zip = _download_day_zip(day)
        except Exception as exc:
            print(f"ERROR: failed to download day zip for {day}: {exc}", flush=True)
            sys.exit(1)
        print(f"[{day}] zip at: {local_zip}", flush=True)

    # ------------------------------------------------------------------
    # 2a. Testing mode: write into a user-supplied local store, no HF.
    # ------------------------------------------------------------------
    if args.store_path:
        store_path = args.store_path
        print(f"[{day}] --store-path mode (no HF); store: {store_path}", flush=True)
        if not os.path.exists(store_path):
            print(f"ERROR: --store-path does not exist: {store_path}", flush=True)
            sys.exit(1)
        if day_filled(store_path, day):
            print(f"[{day}] already filled in local store — nothing to do.", flush=True)
            sys.exit(0)
        try:
            _write_day_to_store(store_path, day, local_zip)
        except Exception as exc:
            print(f"ERROR: write_day failed for {day}: {exc}", flush=True)
            sys.exit(1)
        print(f"[{day}] write_day OK (local store)", flush=True)
        return

    # ------------------------------------------------------------------
    # 2b. Production mode: skeleton download → idempotency check →
    #     region-write → upload changed chunks.
    # ------------------------------------------------------------------
    work_dir = tempfile.mkdtemp(prefix="hima_cube_")
    try:
        skel_dir = os.path.join(work_dir, "skel")
        os.makedirs(skel_dir)

        # ---- 2. download skeleton ----
        print(f"[{day}] downloading cube skeleton …", flush=True)
        try:
            store_path = _download_cube_skeleton(skel_dir)
        except Exception as exc:
            print(f"ERROR: skeleton download failed: {exc}", flush=True)
            sys.exit(1)
        print(f"[{day}] skeleton at: {store_path}", flush=True)

        # ---- 3. idempotency check ----
        if day_filled(store_path, day):
            print(f"[{day}] already filled in remote cube — nothing to do.", flush=True)
            sys.exit(0)

        # ---- 4. mark mtime BEFORE writing so we can find changed files ----
        mark_file = os.path.join(work_dir, ".mark")
        open(mark_file, "w").close()
        import time as _time; _time.sleep(1)   # ensure mtime gap
        mark_mtime = os.path.getmtime(mark_file)

        # ---- 5. region-write ----
        print(f"[{day}] writing day into skeleton …", flush=True)
        try:
            _write_day_to_store(store_path, day, local_zip)
        except Exception as exc:
            print(f"ERROR: write_day failed for {day}: {exc}", flush=True)
            sys.exit(1)
        print(f"[{day}] write_day OK", flush=True)

        # ---- 6. upload changed files ----
        try:
            _upload_changed_files(store_path, mark_mtime)
        except Exception as exc:
            print(f"ERROR: upload failed for {day}: {exc}", flush=True)
            sys.exit(1)

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"[{day}] DONE", flush=True)


if __name__ == "__main__":
    main()
