#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Himawari SWR silver-cube store creation and day-level region writer.

Provides three public functions:

    create_store(path, *, lat, lon, time_start, time_end, chunk_hours) -> None
        Create a preallocated zarr v3 store for Himawari SWR observations.
        Dims: (time, latitude, longitude).  Hourly time axis, NaN-filled SWR
        float32, int8 slot_filled initialised to 0.  Refuses if path exists.

    write_day(path, day_ds: xr.Dataset) -> int
        Region-write exactly 24 consecutive hourly SWR frames for one UTC day.
        Validates grid identity and 00Z alignment; sets that day's slot_filled
        chunk to 1 AFTER the SWR write.  Returns 24.

    day_filled(path_or_ds, day) -> bool
        True if all 24 slot_filled entries for ``day`` equal 1.

Design notes
------------
* Blosc zstd (clevel=5, shuffle) on SWR matches the source day-cube codec from
  himawari_fetch_day.py; uncompressed SWR is treated as a defect.
* slot_filled is chunked at chunk_hours (default 24) so each day is one zarr
  chunk — path-skip-safe for upload pipelines.  This invariant must never be
  resharded away silently.
* write_day is idempotent: re-writing the same day overwrites both SWR and
  slot_filled, keeping the cube consistent.
* Production spatial chunking: for the full 1801×1241 grid use lat_chunk=601,
  lon_chunk=640 so the Taiwan advisory region (23–26.8N, 119.5–123.5E) fits
  entirely within one tile.  For tests / tiny grids the full spatial extent is
  used as a single chunk (no tile needed).
"""
from __future__ import annotations

import os
from typing import Union

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_store(
    path: str,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    time_start: str = "2015-07-07",
    time_end: str = "2031-01-01",
    chunk_hours: int = 24,
    lat_chunk: int | None = None,
    lon_chunk: int | None = None,
) -> None:
    """Create the Himawari SWR silver cube at *path* (zarr v3, preallocated).

    Parameters
    ----------
    path:        Destination directory path for the zarr store.
    lat, lon:    Coordinate arrays (float32).  Shape determines the spatial grid.
    time_start:  First hour of the time axis (inclusive), ISO date string.
    time_end:    First hour PAST the time axis (exclusive), ISO date string.
    chunk_hours: Time chunk size (must divide into 24-h day boundary); default 24.
    lat_chunk:   Spatial chunk size along latitude.  Defaults to full lat extent
                 (sensible for tiny grids; use 601 for the 1801-row production grid).
    lon_chunk:   Spatial chunk size along longitude.  Defaults to full lon extent.

    Raises
    ------
    FileExistsError  if *path* already exists on disk.
    ValueError       if *path* already exists (alternate form so callers can catch either).
    """
    if os.path.exists(path):
        raise FileExistsError(f"store already exists: {path!r}")

    lat = np.asarray(lat, dtype="float32")
    lon = np.asarray(lon, dtype="float32")
    ny, nx = lat.size, lon.size

    # Spatial chunking: use full grid if no tile requested (small/test grids),
    # otherwise use caller-supplied tiles.
    t_lat = lat_chunk if lat_chunk is not None else ny
    t_lon = lon_chunk if lon_chunk is not None else nx

    # Build hourly time axis
    t0 = np.datetime64(time_start, "h")
    t1 = np.datetime64(time_end, "h")
    times = np.arange(t0, t1, np.timedelta64(1, "h")).astype("datetime64[ns]")
    n_hours = times.size

    # Codec: Blosc zstd matching himawari_fetch_day.py
    comp = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")

    # Preallocate dask arrays (compute=False means nothing is written until .to_zarr)
    swr_dask = da.full(
        (n_hours, ny, nx), fill_value=np.nan, dtype="float32",
        chunks=(chunk_hours, t_lat, t_lon),
    )
    sf_dask = da.zeros(n_hours, dtype="int8", chunks=(chunk_hours,))

    ds = xr.Dataset(
        {
            "SWR": xr.DataArray(
                swr_dask,
                dims=("time", "latitude", "longitude"),
                attrs={"long_name": "Surface solar irradiance (GHI)", "units": "W m-2"},
            ),
            "slot_filled": xr.DataArray(
                sf_dask,
                dims=("time",),
                attrs={"long_name": "1 if this hour's SWR slot has been written, else 0"},
            ),
        },
        coords={
            "time":      ("time",      times),
            "latitude":  ("latitude",  lat),
            "longitude": ("longitude", lon),
        },
        attrs={
            "title":   "Himawari SWR silver cube (preallocated, region-written)",
            "source":  "JAXA Himawari Monitor L3 PAR/021 SWR",
            "layout":  "(time, latitude, longitude); hourly UTC observations",
            "grid":    f"{ny}x{nx}",
        },
    )

    encoding = {
        "SWR": {
            "chunks":     (chunk_hours, t_lat, t_lon),
            "compressors": comp,
            "fill_value": float("nan"),
        },
        "slot_filled": {
            "chunks":     (chunk_hours,),
            "fill_value": 0,
        },
    }

    ds.to_zarr(
        path,
        mode="w-",           # refuse if exists (zarr-level guard too)
        compute=False,       # skeleton only; data allocated on first write
        consolidated=True,
        encoding=encoding,
        zarr_format=3,
    )


def write_day(path: str, day_ds: xr.Dataset) -> int:
    """Region-write one UTC day (24 hourly frames) of SWR into the silver cube.

    Parameters
    ----------
    path:    Path to an existing silver-cube store (created by create_store).
    day_ds:  xr.Dataset with dims (time, latitude, longitude), var SWR float32,
             time coord containing exactly 24 consecutive hourly timestamps
             starting at 00:00 UTC of a calendar day.

    Returns
    -------
    24  (number of frames written).

    Raises
    ------
    ValueError  for any of:
      - Dataset does not contain exactly 24 time steps.
      - First timestamp is not at 00:00 UTC (hour-aligned day boundary).
      - Grid (latitude / longitude) does not match the store exactly.
      - Day falls entirely outside the store's time axis.
    """
    times = day_ds["time"].values.astype("datetime64[ns]")

    # --- validate 24 hours ---
    if times.size != 24:
        raise ValueError(
            f"write_day requires exactly 24 time steps; got {times.size}"
        )

    # --- validate alignment: first hour must be T00:00Z ---
    t0 = pd.Timestamp(times[0])
    if t0.hour != 0 or t0.minute != 0 or t0.second != 0:
        raise ValueError(
            f"Day dataset must start at 00:00 UTC (hour-aligned); "
            f"got first timestamp {t0.isoformat()}"
        )

    # --- validate consecutive hourly steps ---
    diffs = np.diff(times).astype("timedelta64[h]").astype(int)
    if not (diffs == 1).all():
        raise ValueError(
            f"time steps are not consecutive hourly; diffs={diffs}"
        )

    # --- open store to read coords ---
    store_ds = xr.open_zarr(path, consolidated=True)
    try:
        store_lat = store_ds["latitude"].values.astype("float32")
        store_lon = store_ds["longitude"].values.astype("float32")
        store_times = store_ds["time"].values.astype("datetime64[ns]")
    finally:
        store_ds.close()

    # --- validate grid identity ---
    day_lat = day_ds["latitude"].values.astype("float32")
    day_lon = day_ds["longitude"].values.astype("float32")
    if not np.array_equal(day_lat, store_lat):
        raise ValueError(
            f"latitude mismatch: day grid has {day_lat.size} values "
            f"[{day_lat[0]:.4f}..{day_lat[-1]:.4f}], store has "
            f"[{store_lat[0]:.4f}..{store_lat[-1]:.4f}]"
        )
    if not np.array_equal(day_lon, store_lon):
        raise ValueError(
            f"longitude mismatch: day grid has {day_lon.size} values "
            f"[{day_lon[0]:.4f}..{day_lon[-1]:.4f}], store has "
            f"[{store_lon[0]:.4f}..{store_lon[-1]:.4f}]"
        )

    # --- find time index ---
    day_t0_ns = times[0]
    hits = np.where(store_times == day_t0_ns)[0]
    if hits.size == 0:
        raise ValueError(
            f"Day {t0.date()} (T00Z = {day_t0_ns}) not found on the store's "
            f"time axis [{store_times[0]}..{store_times[-1]}]"
        )
    t_idx = int(hits[0])
    ny, nx = store_lat.size, store_lon.size

    # --- region-write SWR first, then set slot_filled ---
    swr_data = np.asarray(day_ds["SWR"].values, dtype="float32")
    swr_region_ds = xr.Dataset(
        {"SWR": (("time", "latitude", "longitude"), swr_data)},
        coords={
            "time":      times,
            "latitude":  store_lat,
            "longitude": store_lon,
        },
    )
    swr_region_ds.to_zarr(
        path,
        region={
            "time":      slice(t_idx, t_idx + 24),
            "latitude":  slice(0, ny),
            "longitude": slice(0, nx),
        },
    )

    # --- set slot_filled = 1 for this day's 24 slots AFTER SWR write ---
    sf_ds = xr.Dataset(
        {"slot_filled": (("time",), np.ones(24, dtype="int8"))},
        coords={"time": times},
    )
    sf_ds.to_zarr(
        path,
        region={"time": slice(t_idx, t_idx + 24)},
    )

    return 24


def day_filled(path_or_ds: Union[str, xr.Dataset], day: str) -> bool:
    """Return True if all 24 slot_filled entries for *day* equal 1.

    Parameters
    ----------
    path_or_ds:  Path to a silver-cube store OR an already-open xr.Dataset.
    day:         ISO date string, e.g. ``"2015-07-08"``.

    Returns
    -------
    bool
    """
    t0_ns = np.datetime64(f"{day}T00", "ns")
    t23_ns = np.datetime64(f"{day}T23", "ns")

    if isinstance(path_or_ds, xr.Dataset):
        ds = path_or_ds
        sf = ds["slot_filled"].sel(time=slice(t0_ns, t23_ns)).values
    else:
        ds = xr.open_zarr(path_or_ds, consolidated=True)
        try:
            sf = ds["slot_filled"].sel(time=slice(t0_ns, t23_ns)).values
        finally:
            ds.close()

    return bool(sf.size == 24 and (sf == 1).all())
