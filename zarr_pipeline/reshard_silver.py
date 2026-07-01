#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-shard a Mode A Silver cube for sane file counts on Hugging Face / object storage.

Keeps the SAME data and the SAME inner read chunks (point-query friendly), but bundles
every run's chunks into ONE shard (shard = run_init=1 x all-lead x whole-grid). That gives
~1 file per (variable, run) instead of one per chunk -- ~9x fewer files here -- while still
supporting partial/lazy reads (the shard has an internal index) and append (shards are
run_init-aligned, so writing a run never touches a neighbour).

    python reshard_silver.py <src.zarr> <dst.zarr> [--inner-lat 158 --inner-lon 161]
"""
import argparse
import numpy as np
import xarray as xr


def fill_for(dt):
    dt = np.dtype(dt)
    return np.array(np.nan, dt) if dt.kind == "f" else np.array(np.iinfo(dt).max, dt)


def ceil(x, y):
    return -(-x // y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--inner-lat", type=int, default=None,
                    help="inner chunk lat (default: read from the source cube's chunks)")
    ap.add_argument("--inner-lon", type=int, default=None,
                    help="inner chunk lon (default: read from the source cube's chunks)")
    a = ap.parse_args()

    # raw native dtype + sentinel intact (don't decode ints to float)
    src = xr.open_zarr(a.src, consolidated=True, mask_and_scale=False)
    nlead, ny, nx = src.sizes["lead"], src.sizes["latitude"], src.sizes["longitude"]
    # default the inner chunk to the source cube's own chunking (avoids hard-coding a grid)
    if a.inner_lat is None or a.inner_lon is None:
        ref = next(src[v] for v in src.data_vars if v != "slot_filled")
        ck = ref.encoding.get("chunks") or ref.chunks
        a.inner_lat = a.inner_lat or int(ck[-2])
        a.inner_lon = a.inner_lon or int(ck[-1])
    chunk = (1, nlead, a.inner_lat, a.inner_lon)
    shard = (1, nlead, ceil(ny, a.inner_lat) * a.inner_lat, ceil(nx, a.inner_lon) * a.inner_lon)

    # dask block = one shard (full grid per run) so each task writes a COMPLETE shard
    src = src.chunk({"run_init": 1, "lead": -1, "latitude": -1, "longitude": -1})
    drop = {"_FillValue", "fill_value", "scale_factor", "add_offset", "missing_value"}
    enc = {}
    for v in src.data_vars:
        if v == "slot_filled":          # tiny bookkeeping var, leave unsharded
            continue
        src[v].attrs = {k: val for k, val in src[v].attrs.items() if k not in drop}
        fv = fill_for(src[v].dtype)
        enc[v] = {"chunks": chunk, "shards": shard, "fill_value": fv, "_FillValue": fv}

    print(f"reshard {a.src} -> {a.dst}\n  inner chunk {chunk} | shard {shard} | "
          f"{len(enc)} sharded vars", flush=True)
    src.to_zarr(a.dst, mode="w", encoding=enc, consolidated=True, safe_chunks=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
