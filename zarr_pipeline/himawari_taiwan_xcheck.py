#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-check Himawari silver cube against solar repo's monthly Taiwan label caches.

CLI:
    himawari_taiwan_xcheck.py STORE JAXA_CACHE_DIR

For each month in 202603, 202604, 202605:
  - Open JAXA_CACHE_DIR/swr_taiwan_adv_YYYYMM.nc (xr.open_dataarray)
  - Open STORE (xr.open_zarr) and crop to monthly nc's lat/lon bounds
  - Match grids: try exact sel(); on KeyError/float mismatch, use method="nearest"
    and PRINT disclosure of max coordinate offset
  - Intersect valid_time (nc) ∩ filled hours (cube, via slot_filled)
  - Report per month: n_hours, n_px, Pearson r, max|Δ|, %bit-exact (as float32)

Output
------
Per-month line: "YYYYMM: n_hours=… n_px=… r=… max|d|=… bitexact=…%"
If month file absent or zero overlap: print clear message, continue.
Exit 0 unless a month errored unexpectedly.

Differences are REPORTED not patched — grid variants and H8/H9 eras can differ
between sources.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def open_nc_month(jaxa_cache: str, month: str) -> xr.DataArray | None:
    """Open swr_taiwan_adv_YYYYMM.nc if it exists. Returns None if file absent."""
    path = os.path.join(jaxa_cache, f"swr_taiwan_adv_{month}.nc")
    if not os.path.exists(path):
        return None
    try:
        return xr.open_dataarray(path)
    except Exception as exc:
        print(f"{month}: ERROR opening {path}: {exc}", flush=True)
        return None


def get_filled_hours(store_path: str) -> set:
    """Return set of datetime64[ns] for all hours where slot_filled==1."""
    ds = xr.open_zarr(store_path, consolidated=True)
    try:
        slot_filled = ds["slot_filled"].values
        times = ds["time"].values.astype("datetime64[ns]")
        filled_times = times[slot_filled == 1]
        return set(filled_times)
    finally:
        ds.close()


def try_exact_sel(cube: xr.DataArray, nc: xr.DataArray) -> tuple[xr.DataArray, str]:
    """Try exact coordinate selection. On failure, fall back to nearest.

    Returns (selected_cube, disclosure_str).
    disclosure_str is empty if exact, else a disclosure line about nearest.
    """
    nc_lat = nc.latitude.values
    nc_lon = nc.longitude.values

    try:
        # Try exact selection
        cropped = cube.sel(latitude=nc_lat, longitude=nc_lon)
        return cropped, ""
    except (KeyError, IndexError):
        # Fall back to nearest
        cropped = cube.sel(latitude=nc_lat, longitude=nc_lon, method="nearest")

        # True nearest-neighbour offset: what the cropped selection actually used
        # (searchsorted alone returns the upper-bound cell, over-reporting by up
        # to one grid step when the value sits at/just below a grid point)
        lat_offsets = np.abs(cropped.latitude.values - np.asarray(nc_lat))
        lon_offsets = np.abs(cropped.longitude.values - np.asarray(nc_lon))
        max_offset = max(lat_offsets.max(), lon_offsets.max())

        disclosure = (
            f"  [grid mismatch: using nearest-neighbor "
            f"(max offset {max_offset:.6f}°)]"
        )
        return cropped, disclosure


def xcheck_month(store_path: str, nc: xr.DataArray, month: str,
                 shift_hours: int = 0) -> str:
    """Compare nc (label) against cube for one month.

    shift_hours: subtract this from the nc valid_time before matching cube time.
    The label caches stamp the observation window END (+1h SWR_TIME_OFFSET in
    solar-ghi-nwp jaxa.py); the cube stores the raw JAXA file hour (window START).
    shift_hours=1 therefore compares like-for-like.

    Returns a report line (or error message).
    """
    if shift_hours:
        nc = nc.assign_coords(
            valid_time=nc.valid_time.values - np.timedelta64(shift_hours, "h"))
    # Open cube
    cube_ds = xr.open_zarr(store_path, consolidated=True)
    try:
        cube = cube_ds["SWR"]

        # Exact or nearest selection
        cropped, disclosure = try_exact_sel(cube, nc)

        # Get filled hours from cube
        filled_hours = get_filled_hours(store_path)

        # Intersect valid_time (nc) with filled hours
        nc_times = set(nc.valid_time.values.astype("datetime64[ns]"))
        overlap_times = sorted(nc_times & filled_hours)

        if not overlap_times:
            msg = f"{month}: no time overlap between nc and filled cube hours"
            if disclosure:
                msg += "\n" + disclosure
            return msg

        # Extract matching data
        nc_subset = nc.sel(valid_time=overlap_times)
        cropped_subset = cropped.sel(time=overlap_times)

        # Flatten for correlation
        nc_flat = nc_subset.values.astype("float32").flatten()
        cb_flat = cropped_subset.values.astype("float32").flatten()

        # Filter finite pairs
        finite_mask = np.isfinite(nc_flat) & np.isfinite(cb_flat)
        nc_fin = nc_flat[finite_mask]
        cb_fin = cb_flat[finite_mask]

        if len(nc_fin) == 0:
            msg = f"{month}: no finite data pairs"
            if disclosure:
                msg += "\n" + disclosure
            return msg

        n_hours = len(overlap_times)
        n_px = len(nc_fin)
        r = float(np.corrcoef(nc_fin, cb_fin)[0, 1])
        max_delta = float(np.max(np.abs(nc_fin - cb_fin)))
        bitexact_count = np.sum(nc_fin == cb_fin)
        bitexact_pct = 100.0 * bitexact_count / len(nc_fin)

        msg = (
            f"{month}: n_hours={n_hours} n_px={n_px} "
            f"r={r:.6f} max|d|={max_delta:.4f} bitexact={bitexact_pct:.1f}%"
        )
        if disclosure:
            msg += "\n" + disclosure
        return msg

    finally:
        cube_ds.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-check Himawari silver cube against Taiwan label caches."
    )
    ap.add_argument(
        "store",
        help="Path to the silver cube zarr store (var SWR, dims time/latitude/longitude)",
    )
    ap.add_argument(
        "jaxa_cache_dir",
        help="Directory containing swr_taiwan_adv_YYYYMM.nc files",
    )
    ap.add_argument(
        "--shift-hours", type=int, default=0,
        help="Subtract N hours from nc valid_time before matching (label caches "
             "stamp window END; cube stores raw file hour = window START; use 1).",
    )
    args = ap.parse_args()

    store_path = args.store
    jaxa_cache = args.jaxa_cache_dir

    # Verify store and cache dir exist
    if not os.path.exists(store_path):
        print(f"ERROR: store not found: {store_path}", flush=True)
        sys.exit(1)
    if not os.path.isdir(jaxa_cache):
        print(f"ERROR: jaxa_cache_dir not found: {jaxa_cache}", flush=True)
        sys.exit(1)

    months = ["202603", "202604", "202605"]
    exit_code = 0

    for month in months:
        nc = open_nc_month(jaxa_cache, month)
        if nc is None:
            print(f"{month}: file not found in {jaxa_cache}", flush=True)
            continue

        try:
            report = xcheck_month(store_path, nc, month,
                                  shift_hours=args.shift_hours)
            print(report, flush=True)
        except Exception as exc:
            print(f"{month}: UNEXPECTED ERROR: {exc}", flush=True)
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
