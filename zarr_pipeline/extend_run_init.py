#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extend a Mode A cube's run_init axis to a later end date -- METADATA ONLY.

A Mode A cube pre-allocates its run_init axis at build time; once filled to the end you can no
longer --append new runs (no slots past the end). This grows the axis (adds empty future slots)
WITHOUT touching the data chunks: it resizes each array's run_init dimension, rewrites the small
run_init coordinate + slot_filled arrays, and re-consolidates. Existing shard/chunk files keep
their positions; new slots simply have no files yet (read back as the fill sentinel).

The new slots must continue the EXISTING run_init cadence exactly (same step). Empty slots cost
~0 on disk. Idempotent-ish: if the axis already reaches --end, it's a no-op.

    python extend_run_init.py <cube.zarr> --end 2028-01-01
    python extend_run_init.py <cube.zarr> --end 2028-01-01 --step 6   # step in hours (default: infer)
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cube")
    ap.add_argument("--end", required=True, help="new exclusive end, ISO e.g. 2028-01-01")
    ap.add_argument("--step", type=int, default=None, help="run_init step in hours (default: infer)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ds = xr.open_zarr(a.cube, consolidated=True)
    ri = ds["run_init"].values.astype("datetime64[h]")
    n_old = ri.size
    step = a.step or int((ri[1] - ri[0]) / np.timedelta64(1, "h"))
    # sanity: existing axis must be a regular grid of this step
    diffs = np.unique(np.diff(ri).astype("timedelta64[h]").astype(int))
    if not (diffs == step).all():
        raise SystemExit(f"existing run_init not a regular {step}h grid: steps={diffs}")

    new_axis = np.arange(ri[0], np.datetime64(a.end, "h"), np.timedelta64(step, "h"))
    n_new = new_axis.size
    if n_new <= n_old:
        print(f"axis already reaches {a.end} ({n_old} slots) -> nothing to do")
        return
    if not np.array_equal(new_axis[:n_old], ri):
        raise SystemExit("new axis does not extend the existing one (prefix mismatch)")

    data_vars = [v for v in ds.data_vars if v != "slot_filled"]
    print(f"extend run_init: {n_old} -> {n_new} slots (+{n_new - n_old}), step {step}h, "
          f"{ri[0]} .. {new_axis[-1]}")
    print(f"  data vars to resize: {len(data_vars)} | data chunks: UNTOUCHED")
    if a.dry_run:
        print("  (dry-run, no changes)")
        return

    # open WITHOUT consolidated metadata so resize() writes per-array zarr.json and the later
    # consolidate_metadata() re-scans them (opening consolidated would shadow the resizes).
    g = zarr.open_group(a.cube, mode="r+", use_consolidated=False)

    # 1) resize each forecast var's run_init dim (metadata only; no data written)
    for v in data_vars:
        arr = g[v]
        if arr.shape[0] != n_new:
            arr.resize((n_new,) + arr.shape[1:])

    # 2) rewrite run_init coordinate array. Encode the WHOLE axis manually from the existing CF
    #    units ("<interval> since <ref>") with numpy -- no cftime/xarray dependency -- so old and
    #    new values stay consistent. Verify by decoding back before trusting it.
    import re
    rc = g["run_init"]
    enc_units = rc.attrs.get("units", "")
    m = re.match(r"\s*(\w+)\s+since\s+(.+)", enc_units)
    if not m:
        raise SystemExit(f"cannot parse run_init units {enc_units!r}; aborting (axis untouched in coord)")
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
    # round-trip check: decode encoded ints back to datetime and compare to the intended axis
    decoded = ref + (encoded_store.astype("int64") * one if int_dtype else encoded_store * one)
    if not np.array_equal(decoded.astype("datetime64[h]"), new_axis.astype("datetime64[h]")):
        raise SystemExit("run_init re-encode round-trip mismatch -> aborting (no coord written)")
    rc.resize((n_new,))
    rc[:] = encoded_store

    # 3) extend slot_filled (new slots = 0 = unfilled)
    if "slot_filled" in g:
        sf = g["slot_filled"]
        old = sf[:]
        sf.resize((n_new,))
        sf[n_old:] = 0
        sf[:n_old] = old

    # 4) re-consolidate
    zarr.consolidate_metadata(a.cube)
    print("  done (metadata extended; re-consolidated)")


if __name__ == "__main__":
    main()
