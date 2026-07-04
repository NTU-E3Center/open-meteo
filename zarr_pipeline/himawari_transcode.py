#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transcode HF daily zarr.zips → Himawari silver cube with bit-exact verify gate.

CLI:
    himawari_transcode.py STORE [--verify N] [--limit K]

Source:  HF dataset jimtseng/apac-himawari-swr
         data/year=YYYY/month=MM/YYYYMMDD.zarr.zip  (ZIP_STORED zarr v3 groups)
Sink:    STORE — preallocated zarr v3 silver cube via himawari_cube.create_store /
         write_day / day_filled.

Behaviour
---------
- If STORE is absent, create it using lat/lon from the first downloaded day.
- Enumerate all data/**.zarr.zip, sort by day; for each day not day_filled:
  download via hf_hub_download (cached), open, write_day.
- Per-day progress: "[i/total] YYYYMMDD written" (flush=True).
- A single day failure → print "[i/total] YYYYMMDD FAILED: <err>", continue.
- --verify N: after transcode, sample N random filled days (seeded RNG), assert
  bit-exact over all 24h SWR; print "VERIFY YYYYMMDD PASS/FAIL"; exit nonzero
  if any mismatch.
- --limit K: process only the first K days, written+skipped+failed (smoke-test escape hatch).
- Summary line: "days written=W / skipped=S / failed=F / verified=V"

Resume: rerun is safe — day_filled skip gate makes it idempotent.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
from huggingface_hub import HfApi, hf_hub_download

# Ensure the package root is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from himawari_cube import create_store, day_filled, write_day  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO = "jimtseng/apac-himawari-swr"
HF_REPO_TYPE = "dataset"

# Regex: data/year=YYYY/month=MM/YYYYMMDD.zarr.zip
_PATH_RE = re.compile(r"data/year=\d{4}/month=\d{2}/(\d{8})\.zarr\.zip$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_days() -> list[tuple[str, str]]:
    """Return [(day_str, hf_path), ...] sorted by day for all zarr.zip files in HF repo."""
    api = HfApi()
    files = api.list_repo_files(HF_REPO, repo_type=HF_REPO_TYPE)
    pairs: list[tuple[str, str]] = []
    for f in files:
        m = _PATH_RE.search(f)
        if m:
            day_str = m.group(1)           # YYYYMMDD
            day_iso = f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:]}"
            pairs.append((day_iso, f))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _open_zip(local_zip: str) -> tuple[xr.Dataset, str]:
    """Extract zarr.zip to a tempdir and open as xr.Dataset.

    The zip may contain the zarr group either at the root or under a
    ``YYYYMMDD.zarr/`` subdirectory — we handle both layouts.

    Returns (dataset, tmpdir); caller is responsible for closing the dataset
    AND deleting tmpdir (shutil.rmtree) when done.
    """
    tmpdir = tempfile.mkdtemp(prefix="hima_zip_")
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(tmpdir)

    # Detect whether the zarr group root is at tmpdir or one level down.
    zarr_root = tmpdir
    entries = os.listdir(tmpdir)
    if len(entries) == 1:
        candidate = os.path.join(tmpdir, entries[0])
        if os.path.isdir(candidate):
            # Check for zarr group marker (zarr.json for v3 or .zgroup for v2)
            if os.path.exists(os.path.join(candidate, "zarr.json")) or \
               os.path.exists(os.path.join(candidate, ".zgroup")):
                zarr_root = candidate

    ds = xr.open_zarr(zarr_root, consolidated=True)
    return _normalize_times(ds), tmpdir


def _normalize_times(ds: xr.Dataset) -> xr.Dataset:
    """Snap off-grid scan timestamps to the hour (observed source anomaly: a
    handful of days carry one ``HH:10`` stamp from a delayed Himawari scan,
    e.g. 2026-03-23T20:10). Applied ONLY when safe and unambiguous:
    every adjustment < 30 min AND the floored axis is exactly the source's
    n unique consecutive hourly steps. Otherwise the dataset is returned
    unchanged and write_day's validator rejects it loudly. Each adjustment
    is disclosed on stdout."""
    t = pd.DatetimeIndex(ds.time.values)
    if ((t.minute == 0) & (t.second == 0)).all():
        return ds
    floored = t.floor("h")
    off = t != floored
    ok = ((t - floored) < pd.Timedelta(minutes=30)).all() \
        and floored.is_unique \
        and (np.diff(floored.values).astype("timedelta64[h]").astype(int) == 1).all()
    if not ok:
        return ds
    for a, b in zip(t[off], floored[off]):
        print(f"  NORMALIZED off-grid scan stamp: {a} -> {b}", flush=True)
    return ds.assign_coords(time=floored.values)


def _download_day(hf_path: str) -> str:
    """Download a day zip via hf_hub_download (uses default HF cache). Returns local path."""
    return hf_hub_download(
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        filename=hf_path,
    )


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def transcode(store_path: str, verify_n: int = 0, limit: int | None = None) -> int:
    """Run the full transcode + optional verify. Returns exit code (0 or 1)."""
    days = _list_days()
    if not days:
        print("No zarr.zip files found in HF repo — nothing to do.", flush=True)
        return 0

    total = len(days)
    written = skipped = failed = 0
    # Apply --limit: restrict to first K days in sorted order (written+skipped+failed).
    # This makes reruns with the same --limit idempotent (filled days count toward K).
    if limit is not None:
        days = days[:limit]

    store_created = False

    for i, (day_iso, hf_path) in enumerate(days, 1):
        # Check fill status (requires store to exist)
        if os.path.exists(store_path) and day_filled(store_path, day_iso):
            skipped += 1
            print(f"[{i}/{total}] {day_iso} skipped (already filled)", flush=True)
            continue

        # Download the zip
        tmpdir: str | None = None
        try:
            local_zip = _download_day(hf_path)
            day_ds, tmpdir = _open_zip(local_zip)

            # Bootstrap store from the first real day's coordinates
            if not os.path.exists(store_path):
                lat = day_ds["latitude"].values.astype("float32")
                lon = day_ds["longitude"].values.astype("float32")
                print(
                    f"Creating store {store_path!r} "
                    f"(grid {lat.size}x{lon.size})",
                    flush=True,
                )
                create_store(store_path, lat=lat, lon=lon)
                store_created = True

            write_day(store_path, day_ds)
            day_ds.close()

            written += 1
            print(f"[{i}/{total}] {day_iso} written", flush=True)

        except Exception as exc:
            failed += 1
            print(f"[{i}/{total}] {day_iso} FAILED: {exc}", flush=True)
        finally:
            if tmpdir and os.path.isdir(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Verify gate
    # ------------------------------------------------------------------
    verified = 0
    verify_failed = 0

    if verify_n > 0 and os.path.exists(store_path):
        # Collect all filled days (from the days list we enumerated)
        filled_days: list[tuple[str, str]] = []
        for day_iso, hf_path in days:
            if day_filled(store_path, day_iso):
                filled_days.append((day_iso, hf_path))

        if not filled_days:
            print("VERIFY: no filled days found — skipping.", flush=True)
        else:
            sample_n = min(verify_n, len(filled_days))
            rng = np.random.default_rng(42)
            indices = rng.choice(len(filled_days), size=sample_n, replace=False)
            indices = sorted(indices)

            for idx in indices:
                day_iso, hf_path = filled_days[idx]
                tmpdir = None
                try:
                    local_zip = _download_day(hf_path)
                    day_ds, tmpdir = _open_zip(local_zip)

                    # Read source SWR
                    src_swr = day_ds["SWR"].values.astype("float32")
                    day_ds.close()

                    # Read cube SWR for this day
                    t0_ns = np.datetime64(f"{day_iso}T00", "ns")
                    t23_ns = np.datetime64(f"{day_iso}T23", "ns")
                    cube_ds = xr.open_zarr(store_path, consolidated=True)
                    try:
                        cube_swr = cube_ds["SWR"].sel(
                            time=slice(t0_ns, t23_ns)
                        ).values.astype("float32")
                    finally:
                        cube_ds.close()

                    np.testing.assert_array_equal(cube_swr, src_swr)
                    verified += 1
                    print(f"VERIFY {day_iso} PASS", flush=True)

                except AssertionError as ae:
                    verify_failed += 1
                    print(f"VERIFY {day_iso} FAIL: {ae}", flush=True)
                except Exception as exc:
                    verify_failed += 1
                    print(f"VERIFY {day_iso} FAIL: {exc}", flush=True)
                finally:
                    if tmpdir and os.path.isdir(tmpdir):
                        shutil.rmtree(tmpdir, ignore_errors=True)

    print(
        f"days written={written} / skipped={skipped} / failed={failed} "
        f"/ verified={verified}",
        flush=True,
    )

    if failed > 0 or verify_failed > 0:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transcode HF Himawari daily zarr.zips into the silver cube."
    )
    ap.add_argument("store", help="Path to the silver cube zarr store (created if absent)")
    ap.add_argument(
        "--verify", type=int, default=0, metavar="N",
        help="After transcode, sample N random filled days and verify bit-exact SWR.",
    )
    ap.add_argument(
        "--limit", type=int, default=None, metavar="K",
        help="Process only the first K days, written+skipped+failed (for smoke tests).",
    )
    args = ap.parse_args()
    sys.exit(transcode(args.store, verify_n=args.verify, limit=args.limit))


if __name__ == "__main__":
    main()
