#!/usr/bin/env python3
"""Convert ONE run's archive parquet -> a per-run 4D zarr cube (run_init, lead, lat, lon).

The archive parquet (from postprocess.py) is a FULL regular grid: every (lat,lon) is
present at every forecast time, with per-variable dtypes (UInt16/UInt8/float32). This
reshapes the long table into the map-first cube used for DL training:

    dims   : run_init(1), lead(h), latitude, longitude
    coords : run_init, lead, latitude, longitude, + 2D elevation/location_id (static)
    vars   : each weather variable as its own array, keeping its native dtype
    chunks : map-first (1 run, LEAD_CHUNK leads, full lat, full lon) -> one spatial map
             per chunk, the read pattern DL training wants.

Per-run stores stack into the single 4D cube at read time:
    xr.open_mfdataset("…/*.zarr", engine="zarr", concat_dim="run_init",
                      combine="nested", join="outer")   # join pads short-horizon runs

Usage: parquet_to_zarr_cube.py <in.parquet> <out.zarr> [LEAD_CHUNK=12]
"""
import sys
import numpy as np
import pandas as pd
import xarray as xr
from zarr.codecs import BloscCodec

src, dst = sys.argv[1], sys.argv[2]
LEAD_CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 12

META = {"location_id", "latitude", "longitude", "elevation", "time"}
PROV = {"run_init", "scraped_at"}

df = pd.read_parquet(src)
val_cols = [c for c in df.columns if c not in META and c not in PROV]
run_init = pd.Timestamp(df["run_init"].iloc[0])

# Regular full grid -> sort to a canonical (time, lat, lon) order and reshape.
df = df.sort_values(["time", "latitude", "longitude"], kind="stable")
times = np.sort(df["time"].unique())
lats = np.sort(df["latitude"].unique())
lons = np.sort(df["longitude"].unique())
nt, nlat, nlon = len(times), len(lats), len(lons)
assert len(df) == nt * nlat * nlon, (
    f"not a full grid: {len(df)} rows != {nt}*{nlat}*{nlon}={nt*nlat*nlon}")

lead = ((times - np.datetime64(run_init)) / np.timedelta64(1, "h")).astype("int32")

def reshape(col, dtype):
    return df[col].to_numpy(dtype=dtype).reshape(nt, nlat, nlon)

# static (lat,lon) fields come from the first time block
elevation = reshape("elevation", "float32")[0]
location_id = reshape("location_id", "int32")[0]

NP = {"UInt8": "uint8", "UInt16": "uint16", "Float32": "float32", "float32": "float32"}
data_vars = {}
for c in val_cols:
    npdt = NP[str(df[c].dtype)]
    arr = reshape(c, npdt)[np.newaxis]                       # (1, lead, lat, lon)
    data_vars[c] = (("run_init", "lead", "latitude", "longitude"), arr)

ds = xr.Dataset(
    data_vars,
    coords={
        "run_init": ("run_init", np.array([run_init], dtype="datetime64[ns]")),
        "lead": ("lead", lead),
        "latitude": ("latitude", lats.astype("float32")),
        "longitude": ("longitude", lons.astype("float32")),
        "elevation": (("latitude", "longitude"), elevation),
        "location_id": (("latitude", "longitude"), location_id),
    },
)
ds["lead"].attrs["units"] = "h"
ds["lead"].attrs["long_name"] = "forecast lead time in hours after run_init"
ds.attrs.update(model=df.get("model", pd.Series(["jma_msm"])).iloc[0]
                if "model" in df else "jma_msm",
                run_init=str(run_init), grid=f"{nlat}x{nlon}")

# map-first chunks + blosc zstd(+shuffle); bit-shuffle helps the uint maps a lot.
compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
lc = min(LEAD_CHUNK, nt)
enc = {}
for c in val_cols:
    enc[c] = {"chunks": (1, lc, nlat, nlon), "compressors": (compressor,)}
for c in ("elevation", "location_id"):
    enc[c] = {"chunks": (nlat, nlon), "compressors": (compressor,)}

ds.to_zarr(dst, mode="w", encoding=enc, consolidated=True, zarr_format=3)
print(f"wrote {dst}: dims run_init=1 lead={nt} lat={nlat} lon={nlon}  "
      f"({len(val_cols)} vars, lead_chunk={lc})")
