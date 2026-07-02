#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert per-run NWP forecast files into a single append-friendly Zarr cube.

One tool, two input formats and two output layouts.

Input  (--source):
  * parquet  : bronze layer. One .parquet per run = forecasts for many point
               locations at hourly lead times. The points form a regular lat/lon
               grid, so we pivot each run to (lead, latitude, longitude).
  * zarrzip  : the per-run .zarr.zip files from the Hugging Face sample archive,
               already stored as (run_init=1, lead, lat, lon).
  * auto     : pick by what is in the data dir (.parquet -> parquet, else zarrzip).

Output (--mode):
  * A  (default) : run_init-indexed cube  (run_init, lead, latitude, longitude).
                   Each run = ONE run_init slice -> region writes are append-friendly
                   and order-independent (handles operational delays). This is the
                   "silver" serving cube the project standardised on.
  * B            : valid_time-indexed cube (valid_time, lead, latitude, longitude),
                   DERIVED from an existing mode-A store. Same-timestep forecasts sit
                   contiguously -> "all forecasts valid at T" is ~1 chunk. This is the
                   "gold"/query layer; rebuilt in batch, not appended. If the mode-A
                   store is missing it is built first.

valid_time = run_init + lead is never a stored axis in mode A; "all forecasts valid
at time T" is the anti-diagonal slice (run_init + lead == T). See read_examples.py.

Usage:
    python convert_to_zarr.py                         # auto source, mode A (one-shot)
    python convert_to_zarr.py --source parquet        # bronze parquet -> mode A
    python convert_to_zarr.py --mode B                # derive serving cube (gold)
    python convert_to_zarr.py --demo                  # build A, then print an overlap

Operational silver layer (pre-allocate once, then append on a schedule):
    # once: create an empty run_init grid covering the archive window, then load what's there
    python convert_to_zarr.py --source parquet --init \\
        --start 2026-01-01 --end 2027-01-01 --step 3 --lead-max 78
    # every run thereafter (cron): point at new bronze files; idempotent, order-independent
    python convert_to_zarr.py --source parquet --append --data-dir /path/to/new/runs
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
from zarr.storage import ZipStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "zarr_data")

# Static per-cell coords (depend on lat/lon only) -- stored ONCE, not per run_init.
STATIC_COORDS = ("location_id", "elevation")

# Mode A chunking: each run is its own run_init chunk (size 1) so a region write never
# rewrites a neighbour; the full lead axis stays in one chunk (a point/series read = 1 chunk);
# the lat/lon grid is tiled into near-square spatial tiles sized to a target byte budget
# (see spatial_tile) -- grid-aware instead of a hard-coded tile that fits no real grid.
DEFAULT_TARGET_MB = 8.0
# Mode B: a single-valid_time query should touch ~1 chunk along valid_time.
CHUNK_VT = 6


def spatial_tile(ny, nx, n_lead, target_mb=DEFAULT_TARGET_MB, dtype_bytes=4):
    """Pick a near-square (tlat, tlon) so one (run_init=1, all-lead, tlat, tlon) chunk of the
    largest dtype (float32 -> 4 bytes) stays <= target_mb uncompressed. Lead is never split;
    sizing on float32 keeps the same-shaped uint8/uint16 vars comfortably smaller. Returns the
    full grid when it already fits the budget. Tiling on cell COUNT makes it grid-agnostic, so
    jma_msm (473x481) and dwd_icon (721x497) both land on sensibly sized chunks from one knob."""
    tcells = max(1, int(target_mb * 1e6 / (n_lead * dtype_bytes)))
    if ny * nx <= tcells:
        return ny, nx
    k = -(-(ny * nx) // tcells)                      # ceil: spatial tiles needed
    n_lat = max(1, round((k * ny / nx) ** 0.5))      # split each axis ~proportional to the grid
    n_lon = max(1, -(-k // n_lat))
    while (-(-ny // n_lat)) * (-(-nx // n_lon)) > tcells:   # nudge until the tile fits the budget
        if -(-ny // n_lat) >= -(-nx // n_lon):
            n_lat += 1
        else:
            n_lon += 1
    return -(-ny // n_lat), -(-nx // n_lon)

# Columns in a parquet run that describe the point/run, not a forecast variable.
PARQUET_META_COLS = {
    "location_id", "latitude", "longitude", "elevation",
    "time", "run_init", "scraped_at",
}

_ZIP_RE = re.compile(r"(\d{8}T\d{2})Z\.zarr\.zip$")
_PARQUET_RE = re.compile(r"(\d{8}T\d{2})Z\.parquet$")
_RISH_RE = re.compile(r"Z__C_RJTD_(\d{14})_MSM_GPV_Rjp_Lsurf_FH00-15_grib2\.bin$")
_RISH_SEGMENTS = ("00-15", "16-33", "34-39", "40-51", "52-78")

# RISH GRIB (cfgrib names / JMA local params) -> cube variable names. All cube data vars
# are float32 (verified on jma_msm_silver.zarr 2026-07-02); units converted here.
# Total cloud cover arrives as an 'unknown' JMA local parameter; it is identified
# positionally as the unknown var in the instantaneous cloud sub-dataset and renamed.
_RISH_MAP = {
    "avg_sdswrf": "shortwave_radiation_wattPerSquareMetre",  # W/m2 time-mean (hour-ending)
    "t": "temperature_2m_celsius",                            # K -> degC
    "r": "relative_humidity_2m_percentage",
    "lcc": "cloud_cover_low_percentage",
    "mcc": "cloud_cover_mid_percentage",
    "hcc": "cloud_cover_high_percentage",
    "sp": "surface_pressure_hectopascal",                     # Pa -> hPa
}


def fill_for(dtype):
    """Missing-value sentinel for a dtype: NaN for float, iinfo.max for integer."""
    dt = np.dtype(dtype)
    return np.array(np.nan, dt) if dt.kind == "f" else np.array(np.iinfo(dt).max, dt)


# --------------------------------------------------------------------------- #
# Source discovery + per-run readers. Each reader returns a normalised one-run
# dataset with dims (lead, latitude, longitude), an integer `lead` coord, the
# lat/lon coords, and any static coords -- ready to region-write into the cube.
# --------------------------------------------------------------------------- #
def detect_source(data_dir):
    if glob.glob(os.path.join(data_dir, "*.parquet")):
        return "parquet"
    if glob.glob(os.path.join(data_dir, "*.zarr.zip")):
        return "zarrzip"
    if glob.glob(os.path.join(data_dir, "Z__C_RJTD_*_FH00-15_grib2.bin")):
        return "rish"
    raise SystemExit(f"no *.parquet or *.zarr.zip files in {data_dir}")


def discover_runs(data_dir, source):
    """Return [(run_init: datetime64, path)] sorted by run_init."""
    pat, rx = {"parquet": ("*.parquet", _PARQUET_RE),
               "zarrzip": ("*.zarr.zip", _ZIP_RE),
               "rish": ("Z__C_RJTD_*_FH00-15_grib2.bin", _RISH_RE)}[source]
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, pat))):
        m = rx.search(os.path.basename(f))
        if not m:
            continue
        g = m.group(1)
        if len(g) == 14:                                      # rish: YYYYMMDDHHMMSS
            ts = np.datetime64(f"{g[:4]}-{g[4:6]}-{g[6:8]}T{g[8:10]}:{g[10:12]}:00")
        else:                                                 # YYYYMMDDTHH
            ts = np.datetime64(f"{g[:4]}-{g[4:6]}-{g[6:8]}T{g[9:11]}:00:00")
        out.append((ts, f))
    out.sort(key=lambda x: x[0])
    if not out:
        raise SystemExit(f"no {source} run files matched in {data_dir}")
    return out


def read_run_zarrzip(path):
    """One .zarr.zip run -> (lead, lat, lon) dataset. Already gridded; just reshape."""
    ds = xr.open_zarr(ZipStore(path, mode="r"), consolidated=False)
    if "run_init" in ds.dims:
        ds = ds.isel(run_init=0)
    ds = ds.drop_vars([c for c in ("run_init",) if c in ds.coords], errors="ignore")
    return ds.load()


def read_run_parquet(path):
    """One .parquet run (point rows) -> (lead, lat, lon) dataset on the lat/lon grid."""
    df = pd.read_parquet(path)
    data_vars = [c for c in df.columns if c not in PARQUET_META_COLS]
    run_init = pd.Timestamp(df["run_init"].iloc[0])

    df = df.sort_values(["location_id", "time"], kind="stable").reset_index(drop=True)
    loc_ids, first_idx, counts = np.unique(
        df["location_id"].to_numpy(), return_index=True, return_counts=True
    )
    n_loc, n_lead = loc_ids.size, int(counts[0])
    if not (counts == n_lead).all():
        raise ValueError(f"{os.path.basename(path)}: uneven step counts {set(counts)}")

    times = df["time"].to_numpy().reshape(n_loc, n_lead)
    if not np.array_equal(times, np.broadcast_to(times[0], times.shape)):
        raise ValueError(f"{os.path.basename(path)}: locations differ in time vector")
    leads = ((times[0] - np.datetime64(run_init)) / np.timedelta64(1, "h")).astype("int32")

    lat_pt = df["latitude"].to_numpy()[first_idx].astype("float32")
    lon_pt = df["longitude"].to_numpy()[first_idx].astype("float32")
    elev_pt = df["elevation"].to_numpy()[first_idx].astype("float32")
    lat, lon = np.unique(lat_pt), np.unique(lon_pt)
    ny, nx = lat.size, lon.size
    if ny * nx != n_loc:
        raise ValueError(f"{os.path.basename(path)}: points are not a full {ny}x{nx} grid")
    if not (np.allclose(lat_pt.reshape(ny, nx), lat[:, None])
            and np.allclose(lon_pt.reshape(ny, nx), lon[None, :])):
        raise ValueError(f"{os.path.basename(path)}: location_id is not row-major lat/lon")

    def to_grid(col):  # (n_loc, n_lead) -> (n_lead, ny, nx)
        return col.reshape(ny, nx, n_lead).transpose(2, 0, 1)

    # Keep each column's NATIVE dtype (uint8/uint16/float32) instead of inflating everything
    # to float32 -- matches the zarrzip path and keeps the cube ~native size (float32 would
    # ~quadruple the uint8 fields like cloud cover / humidity). Open-Meteo's archive parquet
    # carries cloud/humidity as uint8 and radiation as uint16; pandas may surface these as
    # numpy ("uint8") or nullable ("UInt8") dtypes -- .numpy_dtype normalises both. Fall back
    # to float32 only if a column actually carries NA (can't live in an integer array).
    grids = {}
    for v in data_vars:
        s = df[v]
        npdt = np.dtype(getattr(s.dtype, "numpy_dtype", s.dtype))
        if npdt.kind in "iu" and s.isna().any():
            npdt = np.dtype("float32")
        grids[v] = to_grid(np.asarray(s.to_numpy(dtype=npdt)).reshape(n_loc, n_lead))
    ds = xr.Dataset(
        {v: (("lead", "latitude", "longitude"), grids[v]) for v in data_vars},
        coords={
            "lead": ("lead", leads),
            "latitude": ("latitude", lat),
            "longitude": ("longitude", lon),
            "elevation": (("latitude", "longitude"), elev_pt.reshape(ny, nx)),
            "location_id": (("latitude", "longitude"), loc_ids.reshape(ny, nx)),
        },
    )
    for v in data_vars:                       # self-describing units from the suffix
        if "_" in v:
            ds[v].attrs["units"] = v.rsplit("_", 1)[-1]
    return ds


def rish_segment_paths(anchor: str) -> list[str]:
    """Anchor = the FH00-15 file; siblings derived by segment substitution."""
    return [anchor.replace("_FH00-15_", f"_FH{seg}_") for seg in _RISH_SEGMENTS]


def _open_rish_segment(path):
    """Seam for tests: cfgrib sub-datasets of one RISH segment file."""
    import cfgrib
    return cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})


def read_run_rish(anchor: str) -> xr.Dataset:
    """RISH MSM surface GRIB2 (5 segments, missing extensions tolerated) -> reader-contract
    dataset: (lead, latitude, longitude), lead 1..hi contiguous (per-var NaN where absent),
    float32 cube-named vars, lat ascending cropped to <= 46.0 (silver grid north bound)."""
    per_var: dict[str, list[xr.DataArray]] = {}
    for seg_path in rish_segment_paths(anchor):
        if not os.path.exists(seg_path):
            continue
        for sub in _open_rish_segment(seg_path):
            step = np.asarray(sub["step"].values)
            if np.issubdtype(step.dtype, np.timedelta64):     # real cfgrib gives timedelta
                step = (step / np.timedelta64(1, "h")).astype(int)
            names = set(sub.data_vars)
            rename = {}
            for src, dst in _RISH_MAP.items():
                if src in names:
                    rename[src] = dst
            unknowns = [v for v in names if v not in _RISH_MAP
                        and v not in ("u10", "v10") and "unknown" in v.lower()]
            if unknowns and {"lcc", "mcc", "hcc"} <= names:   # total cloud rides with l/m/h
                rename[unknowns[0]] = "cloud_cover_percentage"
            keep = dict(rename)
            d = sub.rename(keep)[list(keep.values())] if keep else None
            if {"u10", "v10"} <= names:
                ws = np.hypot(sub["u10"].values, sub["v10"].values).astype("float32")
                wda = xr.DataArray(ws, dims=sub["u10"].dims, coords=sub["u10"].coords)
                d = d.assign(wind_speed_10m_metrePerSecond=wda) if d is not None else \
                    xr.Dataset({"wind_speed_10m_metrePerSecond": wda})
            if d is None:
                continue
            for v in d.data_vars:
                da = d[v]
                for i, s in enumerate(step):
                    if s < 1:
                        continue                              # cube lead starts at 1
                    frame = da.isel(step=i).assign_coords(lead=int(s))
                    # Drop cfgrib auxiliary scalar coords that conflict across sub-datasets
                    # (heightAboveGround, valid_time, time, meanSea, surface, step, etc.)
                    _keep = {"lead", "latitude", "longitude"}
                    frame = frame.drop_vars(
                        [c for c in frame.coords if c not in _keep], errors="ignore")
                    per_var.setdefault(v, []).append(frame.astype("float64"))
    if not per_var:
        raise ValueError(f"{os.path.basename(anchor)}: no decodable RISH fields")
    hi = max(int(fr["lead"]) for frames in per_var.values() for fr in frames)
    lead = np.arange(1, hi + 1, dtype="int32")
    out = {}
    for v, frames in per_var.items():
        stacked = xr.concat(sorted(frames, key=lambda a: int(a["lead"])), dim="lead")
        out[v] = stacked.reindex(lead=lead)                   # per-var gaps -> NaN
    ds = xr.Dataset(out).sortby("latitude")
    lat = ds.latitude.values
    if not (abs(float(lat[0]) - 22.4) < 1e-3 and lat.size >= 473
            and abs(float(lat[472]) - 46.0) < 1e-3):
        raise ValueError(f"unexpected RISH latitude grid: {lat[0]}..{lat[-1]} n={lat.size}")
    ds = ds.isel(latitude=slice(0, 473))
    if ds.sizes["longitude"] != 481:
        raise ValueError(f"unexpected RISH longitude size: {ds.sizes['longitude']} (expected 481)")
    if "temperature_2m_celsius" in ds:
        ds["temperature_2m_celsius"] = (ds["temperature_2m_celsius"] - 273.15).astype("float32")
    if "surface_pressure_hectopascal" in ds:
        ds["surface_pressure_hectopascal"] = (ds["surface_pressure_hectopascal"] / 100.0).astype("float32")
    # Cast ALL remaining float64 vars to float32 to match the silver store's dtype contract.
    # temperature and pressure already cast above; wind speed already float32 from np.hypot.
    ds = ds.assign({v: ds[v].astype("float32")
                    for v in ds.data_vars
                    if ds[v].dtype != np.float32})
    for v in ds.data_vars:
        if "_" in v:
            ds[v].attrs["units"] = v.rsplit("_", 1)[-1]
    return ds.load()


READERS = {"parquet": read_run_parquet, "zarrzip": read_run_zarrzip, "rish": read_run_rish}


# --------------------------------------------------------------------------- #
# Mode A : run_init-indexed cube, written one run per run_init slice (region write)
# --------------------------------------------------------------------------- #
def build_template(sample, run_inits, lead_values, with_marker=False,
                   target_mb=DEFAULT_TARGET_MB, model=None):
    """Empty (run_init, lead, lat, lon) cube filled with per-var sentinels.

    sample      : one already-read run (gives variables, dtypes, grid, static coords).
    run_inits   : the run_init axis to allocate (datetime64). For --init this is the
                  full pre-allocated grid; for a one-shot build it is just the runs found.
    with_marker : add a `slot_filled` (run_init,) int8 var (fill 0) so --append can tell
                  which slots already hold data and stay idempotent.
    target_mb   : per-chunk byte budget that sets the spatial tile (see spatial_tile).
    """
    import dask.array as da

    run_inits = np.asarray(run_inits, dtype="datetime64[ns]")
    n_run, n_lead = run_inits.size, lead_values.size
    lat, lon = sample["latitude"], sample["longitude"]
    ny, nx = lat.size, lon.size
    tlat, tlon = spatial_tile(ny, nx, n_lead, target_mb)
    print(f"  chunk (run_init=1, lead={n_lead}, lat={tlat}, lon={tlon}) "
          f"-> {-(-ny // tlat)}x{-(-nx // tlon)} spatial tiles "
          f"(~{n_lead * tlat * tlon * 4 / 1e6:.1f} MB/chunk f32, target {target_mb} MB)")

    data_vars, encoding = {}, {}
    for name, var in sample.data_vars.items():
        fv = fill_for(var.dtype)
        arr = da.full((n_run, n_lead, ny, nx), fv, dtype=var.dtype,
                      chunks=(1, n_lead, tlat, tlon))
        data_vars[name] = xr.DataArray(
            arr, dims=("run_init", "lead", "latitude", "longitude"),
            attrs=dict(var.attrs))
        # fill_value = zarr-level sentinel for unwritten (beyond-horizon) cells;
        # _FillValue = CF attr so xarray masks the sentinel to NaN on read. Both needed:
        # without fill_value, unwritten int cells default to 0, indistinguishable from a
        # real 0 -> beyond-horizon reads would look like valid data.
        encoding[name] = {"fill_value": fv, "_FillValue": fv}

    if with_marker:
        # chunk run_init=1 so independent slots can be marked without rewriting a neighbour
        data_vars["slot_filled"] = xr.DataArray(
            da.zeros(n_run, dtype="int8", chunks=1), dims=("run_init",),
            attrs={"long_name": "1 if this run_init slot has been written, else 0"})
        encoding["slot_filled"] = {"fill_value": 0}

    ds = xr.Dataset(
        data_vars,
        coords={
            "run_init": run_inits,
            "lead": lead_values.astype("int32"),
            "latitude": lat.values,
            "longitude": lon.values,
        },
        attrs={
            "title": "Combined NWP forecast cube (per-run, region-written)",
            "model": str(model if model else sample.attrs.get("model", "")),
            "grid": str(sample.attrs.get("grid", f"{ny}x{nx}")),
            "layout": "(run_init, lead, latitude, longitude); valid_time = run_init + lead",
            "lead_units": "hours since run_init",
        },
    )
    for c in STATIC_COORDS:
        if c in sample.coords:
            ds = ds.assign_coords({c: (("latitude", "longitude"), sample[c].values)})
    ds["lead"].attrs["units"] = "hours"
    return ds, encoding


def _scan_lead_range(runs, reader):
    """Global (min, max) lead across runs -- they may have different horizons."""
    lo, hi = None, None
    for _, f in runs:
        d = reader(f)
        lv = d["lead"].values.astype(int)
        lo = lv.min() if lo is None else min(lo, lv.min())
        hi = lv.max() if hi is None else max(hi, lv.max())
        d.close()
    return int(lo), int(hi)


def _write_run(out, run, j, lead_min, ny, nx, mark=False):
    """Region-write one already-read run into run_init slot j. Sole writer of that chunk."""
    lv = run["lead"].values.astype(int)
    run = run.drop_vars(
        [c for c in ("lead", "latitude", "longitude", *STATIC_COORDS)
         if c in run.coords], errors="ignore")
    run = run.expand_dims(run_init=1)
    # Collapse lead into one dask block so one dask chunk == one zarr chunk (source files
    # may be chunked in lead sub-blocks); safe_chunks=False then permits a partial tail.
    run = run.chunk({"lead": -1, "latitude": -1, "longitude": -1})
    if mark:
        run["slot_filled"] = ("run_init", np.array([1], dtype="int8"))
    start = int(lv.min() - lead_min)
    run.to_zarr(out, region={
        "run_init": slice(j, j + 1),
        "lead": slice(start, start + lv.size),
        "latitude": slice(0, ny),
        "longitude": slice(0, nx),
    }, safe_chunks=False)
    run.close()
    return lv


def build_mode_a(data_dir, source, out, overwrite, target_mb=DEFAULT_TARGET_MB, model=None):
    """One-shot build: size the run_init axis to exactly the runs found, then fill."""
    runs = discover_runs(data_dir, source)
    reader = READERS[source]
    lo, hi = _scan_lead_range(runs, reader)
    lead_values = np.arange(lo, hi + 1, dtype="int32")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[mode A] {len(runs)} {source} runs | lead {lo}..{hi} | -> {out}")

    sample = reader(runs[0][1])
    run_inits = np.array([ts for ts, _ in runs], dtype="datetime64[ns]")
    ds, encoding = build_template(sample, run_inits, lead_values, target_mb=target_mb,
                                  model=model)
    sample.close()
    ds.to_zarr(out, mode=("w" if overwrite else "w-"), compute=False,
               consolidated=True, encoding=encoding)
    print("  wrote template skeleton")

    ny, nx = ds.sizes["latitude"], ds.sizes["longitude"]
    for i, (ts, f) in enumerate(runs):
        lv = _write_run(out, reader(f), i, lo, ny, nx)
        print(f"  [{i + 1}/{len(runs)}] {str(ts)[:13]}Z  lead {lv.min()}..{lv.max()}")
    print("done:", out)
    return out


def init_cube(data_dir, source, out, start, end, step_h, lead_max, overwrite,
              target_mb=DEFAULT_TARGET_MB, model=None):
    """Create an EMPTY pre-allocated mode-A cube: a regular run_init grid of empty slots.

    Schema (variables, dtypes, grid, static coords) is read from one sample run in the
    data dir. Empty slots cost nothing on disk (unwritten chunks read back as the
    sentinel); --append then region-writes each run into its matching slot.
    """
    runs = discover_runs(data_dir, source)
    reader = READERS[source]
    if lead_max is None:                       # default: the longest horizon available
        _, lead_max = _scan_lead_range(runs, reader)
    lead_values = np.arange(1, lead_max + 1, dtype="int32")
    run_inits = np.arange(np.datetime64(start), np.datetime64(end),
                          np.timedelta64(step_h, "h")).astype("datetime64[ns]")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[init] {run_inits.size} run_init slots | {start}..{end} every {step_h}h "
          f"| lead 1..{lead_max} | -> {out}")

    sample = reader(runs[0][1])
    ds, encoding = build_template(sample, run_inits, lead_values, with_marker=True,
                                  target_mb=target_mb, model=model)
    sample.close()
    ds.to_zarr(out, mode=("w" if overwrite else "w-"), compute=False,
               consolidated=True, encoding=encoding)
    print(f"  wrote empty skeleton ({run_inits.size} slots, all empty)")


def append_runs(data_dir, source, out, force=False):
    """Idempotently region-write new runs into a pre-allocated cube (from --init).

    Each run goes to the slot whose run_init matches it. Already-filled slots are skipped
    (unless --force) and runs whose run_init is not on the grid are skipped with a warning,
    so this is safe to re-run and order-independent -- exactly what an operational cron
    appending the silver layer needs.
    """
    if not os.path.isdir(out):
        raise SystemExit(f"{out} does not exist -- run with --init first")
    store = xr.open_zarr(out, consolidated=True)
    if "slot_filled" not in store:
        store.close()
        raise SystemExit(f"{out} was not created with --init (no slot_filled marker); "
                         "use --init for an appendable cube")
    reader = READERS[source]
    axis = store.run_init.values.astype("datetime64[h]")
    lead_min = int(store.lead.values.min())
    ny, nx = store.sizes["latitude"], store.sizes["longitude"]
    filled = store["slot_filled"].values.copy()      # 0/1 per slot
    store.close()

    runs = discover_runs(data_dir, source)
    written = skipped = offgrid = 0
    for ts, f in runs:
        ri = np.datetime64(ts, "h")
        hit = np.where(axis == ri)[0]
        if hit.size == 0:
            print(f"  off-grid {str(ts)[:13]}Z (no matching slot) -> skip")
            offgrid += 1
            continue
        j = int(hit[0])
        if filled[j] and not force:
            print(f"  [{j}] {str(ts)[:13]}Z already filled -> skip")
            skipped += 1
            continue
        lv = _write_run(out, reader(f), j, lead_min, ny, nx, mark=True)
        filled[j] = 1
        print(f"  [{j}] {str(ts)[:13]}Z written  lead {lv.min()}..{lv.max()}")
        written += 1
    print(f"done: +{written} written, {skipped} already-filled, {offgrid} off-grid "
          f"| {int(filled.sum())}/{filled.size} slots populated")
    return out


# --------------------------------------------------------------------------- #
# Mode B : valid_time-indexed serving cube, DERIVED from a mode-A store.
# Native dtypes are preserved (uint8/uint16/float32) with the same integer fill
# sentinel as A, so B is ~the same size as A (empty (valid_time, lead) cells
# compress to almost nothing; using float32 here instead would inflate it ~1.3x).
# --------------------------------------------------------------------------- #
def derive_mode_b(src, out, target_mb=DEFAULT_TARGET_MB):
    # mask_and_scale=False -> raw native dtypes with their fill sentinel intact,
    # so we can rewrite B in the SAME native dtype.
    run = xr.open_zarr(src, consolidated=True, mask_and_scale=False)
    run_init = run.run_init.values.astype("datetime64[h]")
    leads = run.lead.values.astype(int)
    ny, nx = run.sizes["latitude"], run.sizes["longitude"]
    # B's chunk holds CHUNK_VT valid_times x all-lead; tile space to the same byte budget.
    tlat, tlon = spatial_tile(ny, nx, CHUNK_VT * leads.size, target_mb)

    vmin = run_init.min() + np.timedelta64(int(leads.min()), "h")
    vmax = run_init.max() + np.timedelta64(int(leads.max()), "h")
    full_vt = np.arange(vmin, vmax + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    n_vt = full_vt.size
    print(f"[mode B] valid_time {n_vt}h {str(vmin)}..{str(vmax)} | lead {leads.size} "
          f"| grid {ny}x{nx} | -> {out}")

    base = full_vt[0]
    vt_idx = {i: (((ri + leads.astype("timedelta64[h]")) - base)
                  // np.timedelta64(1, "h")).astype(int)
              for i, ri in enumerate(run_init)}
    lead_pos = np.arange(leads.size)

    coords = {
        "valid_time": full_vt,
        "lead": run.lead.values,
        "latitude": run.latitude.values,
        "longitude": run.longitude.values,
    }
    for c in STATIC_COORDS:
        if c in run.coords:
            coords[c] = (("latitude", "longitude"), run[c].values)

    first = True
    for vname in run.data_vars:
        if vname == "slot_filled":      # append bookkeeping, not a forecast field
            continue
        dt = run[vname].dtype
        fv = fill_for(dt)
        src_arr = run[vname].values                  # native, beyond-horizon == fv
        arr = np.full((n_vt, leads.size, ny, nx), fv, dtype=dt)
        for i in range(run_init.size):
            arr[vt_idx[i], lead_pos, :, :] = src_arr[i]   # scatter this run's diagonal
        # drop CF keys surfaced as attrs by mask_and_scale=False (they collide with encoding=)
        attrs = {k: v for k, v in run[vname].attrs.items()
                 if k not in ("_FillValue", "fill_value", "scale_factor",
                              "add_offset", "missing_value")}
        da_ = xr.DataArray(arr, dims=("valid_time", "lead", "latitude", "longitude"),
                           attrs=attrs)
        ds = xr.Dataset({vname: da_}, coords=coords if first else None)
        enc = {vname: {"chunks": (CHUNK_VT, leads.size, tlat, tlon),
                       "fill_value": fv, "_FillValue": fv}}
        ds.to_zarr(out, mode=("w" if first else "a"), consolidated=True, encoding=enc)
        filled = (float((arr != fv).mean() * 100) if dt.kind != "f"
                  else float(np.isfinite(arr).mean() * 100))
        print(f"  {vname:42s} {str(dt):8s} written (filled {filled:4.1f}%)")
        first = False
        del src_arr, arr, da_, ds
        gc.collect()
    print("done:", out)
    return out


# --------------------------------------------------------------------------- #
def demo_overlap(store, var=None, lat=25.0, lon=121.5):
    """Print every forecast valid at the most-overlapped timestep for one point."""
    ds = xr.open_zarr(store, consolidated=True)
    if var is None:   # first float var (NaN-masked on read) makes the cleanest demo
        var = next((v for v in ds.data_vars if ds[v].dtype.kind == "f"),
                   list(ds.data_vars)[0])
    pt = ds[var].sel(latitude=lat, longitude=lon, method="nearest")
    ri, lead = pt["run_init"].values, pt["lead"].values
    vt = ri[:, None] + lead[None, :].astype("timedelta64[h]")
    vals = pt.values
    uniq, counts = np.unique(vt, return_counts=True)
    target = uniq[np.argmax(counts)]
    fc = [(ri[r], lead[c], vals[r, c]) for r, c in zip(*np.where(vt == target))]
    fc = sorted((x for x in fc if np.isfinite(x[2])), key=lambda x: x[1])
    print(f"\n{var} @ ({float(pt.latitude):.3f},{float(pt.longitude):.3f})  "
          f"valid_time={str(target)[:16]} -> {len(fc)} forecasts")
    for run_t, lh, v in fc:
        print(f"  run {str(run_t)[:13]}Z  lead +{int(lh):>3}h  =  {v:.2f}")
    ds.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--source", choices=("auto", "parquet", "zarrzip", "rish"), default="auto")
    ap.add_argument("--mode", choices=("A", "B"), default="A",
                    help="A = run_init cube (default); B = derived valid_time cube")
    ap.add_argument("--out", default=None, help="output store path")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--demo", action="store_true",
                    help="print an overlap example from the mode-A cube after building")
    # operational silver layer: pre-allocate a run_init grid (--init) then append into it
    ap.add_argument("--init", action="store_true",
                    help="create an empty pre-allocated mode-A cube (needs --start/--end), "
                         "then append whatever runs are in the data dir")
    ap.add_argument("--append", action="store_true",
                    help="region-write new runs into a pre-allocated cube (idempotent)")
    ap.add_argument("--start", help="run_init grid start, ISO e.g. 2026-01-01 (for --init)")
    ap.add_argument("--end", help="run_init grid end, exclusive (for --init)")
    ap.add_argument("--step", type=int, default=3,
                    help="run_init spacing in hours for --init (jma_msm=3, dwd_icon=6)")
    ap.add_argument("--lead-max", type=int, default=None, dest="lead_max",
                    help="max lead hours for --init (jma_msm=78, dwd_icon=180; "
                         "default: scan the data dir)")
    ap.add_argument("--force", action="store_true",
                    help="with --append, overwrite slots that are already filled")
    ap.add_argument("--target-mb", type=float, default=DEFAULT_TARGET_MB, dest="target_mb",
                    help=f"per-chunk byte budget (uncompressed, float32) that sets the spatial "
                         f"tile; grid-aware, so jma_msm and dwd_icon each get right-sized chunks "
                         f"from one knob (default {DEFAULT_TARGET_MB})")
    ap.add_argument("--model", default=None,
                    help="override the cube's `model` attr (e.g. dwd_icon), instead of trusting "
                         "the source's -- use when the source attr is mislabelled")
    a = ap.parse_args()

    out_a = os.path.join(OUT_DIR, "forecast_combined.zarr")
    out_b = os.path.join(OUT_DIR, "forecast_validtime.zarr")

    if a.mode == "A":
        source = detect_source(a.data_dir) if a.source == "auto" else a.source
        out = a.out or out_a
        if a.init:
            if not (a.start and a.end):
                ap.error("--init requires --start and --end")
            init_cube(a.data_dir, source, out, a.start, a.end, a.step,
                      a.lead_max, a.overwrite, target_mb=a.target_mb, model=a.model)
            append_runs(a.data_dir, source, out, a.force)
        elif a.append:
            append_runs(a.data_dir, source, out, a.force)
        else:
            build_mode_a(a.data_dir, source, out, a.overwrite, target_mb=a.target_mb,
                         model=a.model)
        if a.demo:
            demo_overlap(out)
    else:  # mode B
        if not os.path.isdir(out_a):
            print(f"mode-A store {out_a} missing -> building it first")
            source = detect_source(a.data_dir) if a.source == "auto" else a.source
            build_mode_a(a.data_dir, source, out_a, a.overwrite, target_mb=a.target_mb)
        derive_mode_b(out_a, a.out or out_b, target_mb=a.target_mb)


if __name__ == "__main__":
    main()
