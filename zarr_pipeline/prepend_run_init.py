#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extend a Mode A cube's run_init axis to an EARLIER start (backward extension).

Counterpart of extend_run_init.py (which only grows the end, metadata-only). Growing the
START shifts every existing run_init index by a constant offset, so this is NOT metadata-
only -- but because the cube chunks run_init at size 1, the shift is pure chunk-DIRECTORY
renames (<var>/c/<j> -> <var>/c/<j+offset>), no data re-encoding. Steps:
  1. resize every run_init-dimensioned array to the new length (grows the END internally)
  2. rename chunk dirs j -> j+offset for data vars and slot_filled (processed
     highest-first so no rename clobbers a source that hasn't moved yet)
  3. rewrite the run_init coordinate array with the new axis
  4. re-consolidate metadata
Verification (bit-exact old slots at shifted positions) is the caller's job -- see
MIGRATION_RUNBOOK.md; tests cover it on a mini cube.

    python prepend_run_init.py <cube.zarr> --start 2017-12-01 [--dry-run]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
import zarr


def prepend(cube_path: str, new_start: str, dry_run: bool = False) -> dict:
    ds = xr.open_zarr(cube_path, consolidated=True)
    ri = ds["run_init"].values.astype("datetime64[h]")
    names = [v for v in ds.data_vars]                       # incl. slot_filled
    ds.close()
    n_old = ri.size
    if n_old < 2:
        raise SystemExit("run_init has only 1 slot; cannot infer step")
    step = int((ri[1] - ri[0]) / np.timedelta64(1, "h"))
    diffs = np.unique(np.diff(ri).astype("timedelta64[h]").astype(int))
    if not (diffs == step).all():
        raise SystemExit(f"existing run_init not a regular {step}h grid: steps={diffs}")
    start = np.datetime64(new_start, "h")
    span_h = int((ri[0] - start) / np.timedelta64(1, "h"))
    if span_h <= 0:
        raise SystemExit(f"--start {new_start} is not earlier than axis start {ri[0]}")
    if span_h % step != 0:
        raise SystemExit(f"--start {new_start} is off the {step}h grid of {ri[0]}")
    offset = span_h // step
    new_axis = np.arange(start, ri[-1] + np.timedelta64(step, "h"),
                         np.timedelta64(step, "h"))
    n_new = new_axis.size
    assert n_new == n_old + offset and np.array_equal(new_axis[offset:], ri)
    if offset <= n_old:
        # rename j->j+offset could collide when offset <= max existing index; process
        # highest-first to stay safe either way
        order_note = "descending-order renames (offset within old range)"
    else:
        order_note = "collision-free (offset beyond old range)"

    info = {"offset": int(offset), "n_old": int(n_old), "n_new": int(n_new), "renamed": 0}
    print(f"prepend run_init: {n_old} -> {n_new} slots (+{offset} before start), "
          f"step {step}h, new axis {new_axis[0]} .. {new_axis[-1]} | {order_note}")
    if dry_run:
        print("  (dry-run, no changes)")
        return info

    g = zarr.open_group(cube_path, mode="r+", use_consolidated=False)
    renamed = 0
    for name in names:
        arr = g[name]
        shape = list(arr.shape)
        shape[0] = n_new
        arr.resize(shape)                                   # grows the END; chunks untouched
        cdir = os.path.join(cube_path, name, "c")
        if os.path.isdir(cdir):
            idxs = sorted((int(d) for d in os.listdir(cdir) if d.isdigit()), reverse=True)
            for j in idxs:
                os.rename(os.path.join(cdir, str(j)), os.path.join(cdir, str(j + offset)))
                renamed += 1

    # rewrite the coordinate array with the full new axis
    # encode using the stored CF units ("<interval> since <ref>") to stay consistent
    rc = g["run_init"]
    enc_units = rc.attrs.get("units", "")
    m = re.match(r"\s*(\w+)\s+since\s+(.+)", enc_units)
    if m:
        interval, ref_str = m.group(1).lower(), m.group(2).strip()
        np_unit = {"days": "D", "hours": "h", "minutes": "m", "seconds": "s"}.get(interval)
        if np_unit is None:
            raise SystemExit(f"unsupported run_init interval {interval!r}")
        ref = np.datetime64(pd.Timestamp(ref_str))
        one = np.timedelta64(1, np_unit)
        full_ns = new_axis.astype("datetime64[ns]")
        encoded = ((full_ns - ref) / one)
        int_dtype = np.dtype(rc.dtype).kind in "iu"
        encoded_store = np.round(encoded).astype(rc.dtype) if int_dtype else encoded.astype(rc.dtype)
        # round-trip check
        decoded = ref + (encoded_store.astype("int64") * one if int_dtype else encoded_store * one)
        if not np.array_equal(decoded.astype("datetime64[h]"), new_axis.astype("datetime64[h]")):
            raise SystemExit("run_init re-encode round-trip mismatch -> aborting (no coord written)")
        rc.resize((n_new,))
        rc[:] = encoded_store
    else:
        # fallback: store as int64 nanoseconds since epoch matching ra.dtype
        rc.resize((n_new,))
        rc[:] = new_axis.astype(rc.dtype)

    zarr.consolidate_metadata(cube_path)
    info["renamed"] = renamed
    print(f"  renamed {renamed} chunk dirs; coordinate + metadata rewritten")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube")
    ap.add_argument("--start", required=True, help="new axis start, ISO, on the step grid")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    prepend(a.cube, a.start, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
