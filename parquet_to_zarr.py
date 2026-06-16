#!/usr/bin/env python3
"""Convert one run's exported parquet (from `export --run`) into a per-run zarr store.

Usage: RUN_STAMP=20260615T12Z SCRAPED_AT=... parquet_to_zarr.py <in.parquet> <out.zarr>

- Drops rows outside the run (export pads the requested [start,end] window with NaN for
  times the run does not cover; reading a single run via --run leaves those as NaN).
- Same value encoding as the parquet pipeline (information-lossless): round to physical
  precision, integer-cast unit-step columns (radiation->UInt16, cloud/humidity->UInt8).
- Reshapes to an xarray Dataset (time x location) and writes zarr v3 with Blosc zstd+shuffle.
- run_init / scraped_at are stored as dataset attributes (constant per run).
"""
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import zarr

src, dst = sys.argv[1], sys.argv[2]
run_stamp = os.environ["RUN_STAMP"]                     # 20260615T12Z
run_init = pd.to_datetime(run_stamp, format="%Y%m%dT%HZ", utc=True).tz_localize(None)
scraped_at = pd.to_datetime(os.environ["SCRAPED_AT"], format="%Y%m%dT%H%M%SZ", utc=True).tz_localize(None)

df = pd.read_parquet(src)
META = {"location_id", "latitude", "longitude", "elevation", "time"}
val_cols = [c for c in df.columns if c not in META and c not in ("run_init", "scraped_at")]

# Keep only this run's forecast. The export pads the requested [start,end] window with
# NaN for times beyond the run's horizon; flux/accumulated vars (radiation, precip) and
# instantaneous vars pad at slightly different boundaries, producing partial-NaN rows.
# Drop any TIME that contains a NaN (not individual rows) so the (time x location) grid
# stays a full product and the integer-cast below never sees NA.
df = df[pd.to_datetime(df["time"]) >= run_init]
bad_times = df.loc[df[val_cols].isna().any(axis=1), "time"].unique()
if len(bad_times):
    df = df[~df["time"].isin(bad_times)]
if df.empty:
    sys.exit(f"no in-run rows for {run_stamp}")

# Value encoding (mirror the parquet pipeline, value-identical).
for c in val_cols:
    df[c] = df[c].round(2)
    if "cloud_cover" in c or "relative_humidity" in c:
        df[c] = df[c].clip(lower=0, upper=255).round().astype("UInt8")
    elif "radiation" in c or "irradiance" in c:
        df[c] = df[c].clip(lower=0).round().astype("UInt16")
df["location_id"] = df["location_id"].astype("int32")

# Reshape (location, time) -> (time, location). Gridded run => full product.
df = df.sort_values(["location_id", "time"])
locs = df["location_id"].drop_duplicates().to_numpy()
times = np.sort(df["time"].unique())
nL, nT = len(locs), len(times)
if len(df) != nL * nT:
    # ragged (shouldn't happen for a gridded run) -> pivot fallback
    idx = pd.MultiIndex.from_product([locs, times], names=["location_id", "time"])
    df = df.set_index(["location_id", "time"]).reindex(idx).reset_index()

first = df.drop_duplicates("location_id").set_index("location_id").loc[locs]
ds = xr.Dataset(
    coords=dict(
        time=("time", pd.to_datetime(times)),
        location=("location", locs.astype("int32")),
        latitude=("location", first["latitude"].to_numpy().astype("float32")),
        longitude=("location", first["longitude"].to_numpy().astype("float32")),
        elevation=("location", first["elevation"].to_numpy().astype("float32")),
    ),
    attrs=dict(run_init=str(run_init), scraped_at=str(scraped_at), model_run=run_stamp),
)
for c in val_cols:
    # pandas nullable ints -> plain numpy (no NaN remain after dropna on a full grid)
    np_dtype = {"UInt8": "uint8", "UInt16": "uint16"}.get(str(df[c].dtype), "float32")
    arr = df[c].astype(np_dtype).to_numpy()
    ds[c] = (("time", "location"), arr.reshape(nL, nT).T)

comp = zarr.codecs.BloscCodec(cname="zstd", clevel=9, shuffle=zarr.codecs.BloscShuffle.shuffle)
enc = {v: {"compressors": (comp,), "chunks": (nT, nL)} for v in val_cols}
# consolidated metadata = one metadata read when opening over HTTP from HF (much faster
# for training-time access than fetching every array's metadata separately).
ds.to_zarr(dst, mode="w", encoding=enc, zarr_format=3, consolidated=True)
print(f"wrote {dst}: {nT} times x {nL} locations x {len(val_cols)} vars")
