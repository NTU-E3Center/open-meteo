#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Disk-efficient re-shard: process ONE variable at a time and (unless --keep-source) delete
that variable from the SOURCE cube right after it is resharded into the destination. Peak
extra disk ~= one variable instead of a full second copy of the cube -- for cubes too big to
hold twice (e.g. dwd_icon ~177 GB on a disk with ~317 GB free).

DESTRUCTIVE to the source by default (source variables are removed as they are resharded).
Resumable: variables already present in the destination are skipped, and a leftover source
variable is removed if its destination copy exists. If the source's .zarr.zip are still on
HF, a failed run can always be rebuilt by re-migrating.

    python reshard_silver_lowdisk.py <src.zarr> <dst.zarr> [--keep-source]
"""
import argparse
import os
import shutil

import numpy as np
import xarray as xr
import zarr

DROP = {"_FillValue", "fill_value", "scale_factor", "add_offset", "missing_value"}


def fill_for(dt):
    dt = np.dtype(dt)
    return np.array(np.nan, dt) if dt.kind == "f" else np.array(np.iinfo(dt).max, dt)


def ceil(x, y):
    return -(-x // y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep-source", action="store_true", help="do NOT delete source vars")
    ap.add_argument("--shard-runs", type=int, default=1,
                    help="run_init per shard (1 = per-run, append-friendly; >1 = coarse, far "
                         "fewer files for HF upload but append rewrites a whole shard)")
    a = ap.parse_args()

    src = xr.open_zarr(a.src, consolidated=True, mask_and_scale=False)
    root_attrs = dict(src.attrs)                       # keep model/grid/layout for the dst
    nlead, ny, nx = src.sizes["lead"], src.sizes["latitude"], src.sizes["longitude"]
    ref = next(src[v] for v in src.data_vars if v != "slot_filled")
    ck = ref.encoding.get("chunks") or ref.chunks
    ilat, ilon = int(ck[-2]), int(ck[-1])
    chunk = (1, nlead, ilat, ilon)
    shard = (a.shard_runs, nlead, ceil(ny, ilat) * ilat, ceil(nx, ilon) * ilon)
    data_vars = [v for v in src.data_vars if v != "slot_filled"]
    print(f"reshard(low-disk) {a.src} -> {a.dst}\n  inner {chunk} | shard {shard} | "
          f"{len(data_vars)} vars | keep_source={a.keep_source}", flush=True)

    # destination skeleton: coords + slot_filled (created once; preserved on resume)
    if not os.path.isdir(a.dst):
        src.drop_vars(data_vars).to_zarr(a.dst, mode="w", consolidated=True)
    dst_have = set(xr.open_zarr(a.dst, consolidated=True).data_vars)

    for i, v in enumerate(data_vars, 1):
        if v not in dst_have:
            fv = fill_for(src[v].dtype)
            one = xr.Dataset({v: src[v].variable})
            one[v].attrs = {k: val for k, val in src[v].attrs.items() if k not in DROP}
            one = one.chunk({"run_init": a.shard_runs, "lead": -1, "latitude": -1, "longitude": -1})
            enc = {v: {"chunks": chunk, "shards": shard, "fill_value": fv, "_FillValue": fv}}
            one.to_zarr(a.dst, mode="a", encoding=enc, consolidated=True)
            print(f"[{i}/{len(data_vars)}] {v} resharded", flush=True)
        # free source space once the dest copy exists
        srcdir = os.path.join(a.src, v)
        if not a.keep_source and os.path.isdir(srcdir):
            shutil.rmtree(srcdir)
            print(f"[{i}/{len(data_vars)}] {v} source freed", flush=True)

    # restore the root attrs (model/grid/...) that per-var appends drop, then re-consolidate
    grp = zarr.open_group(a.dst)
    grp.attrs.update(root_attrs)
    zarr.consolidate_metadata(a.dst)
    print("done", flush=True)


if __name__ == "__main__":
    main()
