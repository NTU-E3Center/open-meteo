#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backfill historical RISH JMA-MSM runs into a silver cube (spec: solar-ghi-nwp
docs/superpowers/specs/2026-07-02-rish-silver-backfill-design.md).

Per pending run (slot empty on the cube axis): download the 5 Lsurf segments into a temp
dir (sequential, throttled, pinned-CA), append via convert_to_zarr's rish reader (slot
lookup + idempotency live there), delete the GRIB files. Resume = rerun; slot_filled
decides what is pending. Peak temp disk ~350 MB.

Overlap gate (run BEFORE any bulk backfill; spec 3.2): --validate-overlap N converts N
already-filled 00/12Z runs from RISH WITHOUT writing and compares fields against the cube
(open-meteo-sourced). Tolerances: shortwave rel-RMSE < 5%, temperature MAE < 0.5 degC,
cloud/RH/wind Pearson r > 0.85. A timestamp-convention bug (the result-Q class of error)
fails here loudly instead of poisoning training.

    python rish_backfill.py <cube.zarr> --validate-overlap 5
    python rish_backfill.py <cube.zarr> --start 2025-07-01 --end 2026-05-12
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_zarr import append_runs, read_run_rish, rish_segment_paths  # noqa: E402

BASE = "https://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
CA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rish_ca.pem")
GATE_VARS = ["shortwave_radiation_wattPerSquareMetre", "temperature_2m_celsius",
             "relative_humidity_2m_percentage", "wind_speed_10m_metrePerSecond",
             "cloud_cover_percentage"]


def rish_urls(ts) -> list[str]:
    t = pd.Timestamp(ts)
    anchor = (f"{BASE}/{t:%Y/%m/%d}/Z__C_RJTD_{t:%Y%m%d%H%M%S}"
              f"_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin")
    return rish_segment_paths(anchor)


def pending_runs(cube: str, start: str, end: str, cycles=(0, 12)) -> list[np.datetime64]:
    """Empty (slot_filled==0) run_init slots in [start, end) whose hour is in cycles."""
    ds = xr.open_zarr(cube, consolidated=True)
    ri = pd.DatetimeIndex(ds.run_init.values)
    filled = ds["slot_filled"].values > 0
    ds.close()
    m = ((ri >= pd.Timestamp(start)) & (ri < pd.Timestamp(end))
         & np.isin(ri.hour, list(cycles)) & ~filled)
    return list(ri[m].values.astype("datetime64[h]"))


def download_run(ts, dest_dir: str, sleep: float, timeout: int = 300) -> tuple[int, int]:
    """Fetch this run's segments into dest_dir. Missing extension segments (404) are fine
    (39h-era). Returns (fetched, missing); raises only on the anchor being unavailable."""
    fetched = missing = 0
    for url in rish_urls(ts):
        name = url.rsplit("/", 1)[1]
        out = os.path.join(dest_dir, name)
        r = requests.get(url, verify=CA, timeout=timeout, stream=True)
        if r.status_code == 404:
            if "_FH00-15_" in name:
                raise FileNotFoundError(f"anchor missing on RISH: {name}")
            missing += 1
            continue
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        if open(out, "rb").read(4) != b"GRIB":
            raise IOError(f"not GRIB2: {name}")
        fetched += 1
        time.sleep(sleep)
    return fetched, missing


def backfill(cube: str, start: str, end: str, cycles=(0, 12), sleep: float = 5.0,
             limit: int | None = None, dry_run: bool = False) -> int:
    todo = pending_runs(cube, start, end, cycles)
    if limit:
        todo = todo[:limit]
    print(f"backfill: {len(todo)} pending runs ({start}..{end}, cycles={cycles})")
    if dry_run or not todo:
        return 0
    failed = 0
    for k, ts in enumerate(todo, 1):
        d = tempfile.mkdtemp(prefix="rish_")
        try:
            fetched, missing = download_run(ts, d, sleep)
            append_runs(d, "rish", cube)
            print(f"[{k}/{len(todo)}] {str(ts)[:13]}Z ok ({fetched} segs, {missing} n/a)")
        except Exception as e:
            failed += 1
            print(f"[{k}/{len(todo)}] {str(ts)[:13]}Z FAILED: {type(e).__name__} {str(e)[:80]}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print(f"done: {len(todo) - failed} ok, {failed} failed (rerun to retry failures)")
    return 0 if failed == 0 else 2


def validate_overlap(cube: str, n: int = 5, sleep: float = 5.0) -> int:
    """Compare RISH-decoded fields vs already-filled (open-meteo-sourced) cube slots."""
    ds = xr.open_zarr(cube, consolidated=True)
    ri = pd.DatetimeIndex(ds.run_init.values)
    cand = ri[(ds["slot_filled"].values > 0) & np.isin(ri.hour, [0, 12])]
    picks = cand[:: max(1, len(cand) // n)][:n]
    print(f"overlap gate: {len(picks)} runs {list(picks.strftime('%m-%d %HZ'))}")
    ok = True
    for ts in picks:
        d = tempfile.mkdtemp(prefix="rishval_")
        try:
            download_run(np.datetime64(ts, "h"), d, sleep)
            rish = read_run_rish(os.path.join(d, os.path.basename(rish_urls(ts)[0])))
            cube_run = ds.sel(run_init=ts)
            for v in GATE_VARS:
                if v not in rish or v not in cube_run:
                    print(f"  {ts:%m-%d %HZ} {v}: MISSING -> FAIL"); ok = False
                    continue
                a = rish[v].interp(latitude=cube_run.latitude,
                                   longitude=cube_run.longitude) \
                    .reindex(lead=cube_run.lead).values.ravel()
                b = cube_run[v].values.ravel()
                m = np.isfinite(a) & np.isfinite(b)
                if v.startswith("shortwave"):
                    day = m & (b > 50)
                    rel = float(np.sqrt(np.mean((a[day] - b[day]) ** 2)) / np.mean(b[day]))
                    good, msg = rel < 0.05, f"relRMSE={rel:.3f} (<0.05)"
                elif v.startswith("temperature"):
                    mae = float(np.mean(np.abs(a[m] - b[m])))
                    good, msg = mae < 0.5, f"MAE={mae:.2f}degC (<0.5)"
                else:
                    r = float(np.corrcoef(a[m], b[m])[0, 1])
                    good, msg = r > 0.85, f"r={r:.3f} (>0.85)"
                print(f"  {ts:%m-%d %HZ} {v:44s} {msg} {'OK' if good else 'FAIL'}")
                ok = ok and good
        finally:
            shutil.rmtree(d, ignore_errors=True)
    ds.close()
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--cycles", default="0,12")
    ap.add_argument("--sleep", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-overlap", type=int, default=None, dest="gate")
    a = ap.parse_args()
    if a.gate:
        sys.exit(validate_overlap(a.cube, a.gate, a.sleep))
    if not (a.start and a.end):
        ap.error("--start/--end required (or --validate-overlap N)")
    cycles = tuple(int(c) for c in a.cycles.split(","))
    sys.exit(backfill(a.cube, a.start, a.end, cycles, a.sleep, a.limit, a.dry_run))


if __name__ == "__main__":
    main()
