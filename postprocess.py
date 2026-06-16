#!/usr/bin/env python3
"""Clean one run's exported parquet (from `export --run`) into the archive parquet.

Usage: RUN_STAMP=20260615T12Z SCRAPED_AT=... postprocess.py <in.parquet> <out.parquet>

- Drops rows outside the run (the export pads the requested [start,end] window with NaN
  for times the run does not cover; flux vars and instantaneous vars pad one step apart,
  so drop any whole TIME that contains a NaN rather than individual rows).
- Same value encoding as before (information-lossless): round to physical precision,
  integer-cast unit-step columns (radiation->UInt16, cloud/humidity->UInt8, id->int32).
- Adds run_init / scraped_at provenance columns (constant per run; RLE-compresses to ~0).
- Writes zstd parquet. One file per run.
"""
import os, sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
run_stamp = os.environ["RUN_STAMP"]                     # 20260615T12Z
run_init = pd.to_datetime(run_stamp, format="%Y%m%dT%HZ", utc=True).tz_localize(None)
scraped_at = pd.to_datetime(os.environ["SCRAPED_AT"], format="%Y%m%dT%H%M%SZ", utc=True).tz_localize(None)

df = pd.read_parquet(src)
META = {"location_id", "latitude", "longitude", "elevation", "time"}
val_cols = [c for c in df.columns if c not in META and c not in ("run_init", "scraped_at")]

# Keep only this run's forecast: time >= run_init, and drop any whole padding-time (NaN).
df = df[pd.to_datetime(df["time"]) >= run_init]
bad_times = df.loc[df[val_cols].isna().any(axis=1), "time"].unique()
if len(bad_times):
    df = df[~df["time"].isin(bad_times)]
if df.empty:
    sys.exit(f"no in-run rows for {run_stamp}")

# Value encoding (information-lossless; source values are already quantized in the .om DB).
for c in val_cols:
    df[c] = df[c].round(2)
    if "cloud_cover" in c or "relative_humidity" in c:
        df[c] = df[c].clip(lower=0, upper=255).round().astype("UInt8")
    elif "radiation" in c or "irradiance" in c:
        df[c] = df[c].clip(lower=0).round().astype("UInt16")
df["location_id"] = df["location_id"].astype("int32")

# Provenance (constant per file)
df["run_init"] = pd.Timestamp(run_init)
df["scraped_at"] = pd.Timestamp(scraped_at)

df.to_parquet(dst, compression="zstd", compression_level=12)
print(f"wrote {dst}: {len(df)} rows ({df['time'].nunique()} times)")
