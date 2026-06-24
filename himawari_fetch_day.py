#!/usr/bin/env python3
"""Fetch one day of JAXA Himawari L3 SWR (GHI ground truth) -> APAC observation zarr cube.

JAXA P-Tree PAR/021 full-disk hourly product (H09_*_1H_RFL021_FLDK.02801_02401.nc,
0.05deg, lat -60..60 / lon 70..210). Per UTC day: download 24 hourly full-disk files,
subset to the APAC bbox, keep SWR(GHI) + TAOT_02/TAAE (aerosol), stack along time, and
write one <YYYYMMDD>.zarr.zip with dims (time, latitude, longitude) — an OBSERVATION cube
(no forecast lead), the ground-truth counterpart to the jma/icon forecast archive.

Usage: himawari_fetch_day.py YYYYMMDD <out_dir>
"""
import sys, os, ftplib, tempfile, zipfile, datetime
import numpy as np, pandas as pd, xarray as xr
from zarr.codecs import BloscCodec

HOST="ftp.ptree.jaxa.jp"
USER=os.environ.get("HIMAWARI_FTP_USER"); PW=os.environ.get("HIMAWARI_FTP_PW")
if not USER or not PW:
    sys.exit("set HIMAWARI_FTP_USER and HIMAWARI_FTP_PW env vars (JAXA P-Tree credentials)")
LAT0,LAT1, LON0,LON1 = -44.0, 46.0, 92.0, 154.0          # APAC bbox
# SWR (GHI) only. The bundled aerosol bands (TAOT_02/TAAE) are retrieved for ~1% of pixels
# (clear-sky only) yet cost ~2.3x SWR's storage — useless as a gridded field and the
# aerosol effect is already embedded in JAXA's SWR retrieval. PAR/UVA/UVB are redundant
# with SWR. So the ground-truth cube keeps just SWR.
KEEP = ["SWR"]
REMOTE="/pub/himawari/L3/PAR/021"

def main():
    day, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    d=datetime.datetime.strptime(day,"%Y%m%d")
    # Never build TODAY (UTC) — it is still accumulating hours; a partial cube would be
    # uploaded and then skipped forever by the idempotent check. Only past UTC days.
    if d.date() >= datetime.datetime.now(datetime.UTC).date():
        sys.exit(f"[{day}] is today/future (UTC) — still in progress, skip until complete")
    rdir=f"{REMOTE}/{d:%Y%m}/{d:%d}"
    f=ftplib.FTP(HOST,timeout=120); f.login(USER,PW); f.voidcmd("TYPE I"); f.cwd(rdir)
    files=sorted(x for x in f.nlst() if x.endswith("FLDK.02801_02401.nc") and "_1H_RFL021_" in x)
    print(f"[{day}] {len(files)} hourly full-disk files in {rdir}", flush=True)
    work=tempfile.mkdtemp(prefix="hima_")
    slabs=[]; times=[]
    for fn in files:
        hhmm=fn.split("_")[2]                                # e.g. 0300
        t=pd.Timestamp(f"{day}T{hhmm[:2]}:{hhmm[2:]}")
        lp=os.path.join(work,fn)
        with open(lp,"wb") as o: f.retrbinary(f"RETR {fn}", o.write)
        ds=xr.open_dataset(lp)
        latm=(ds.latitude>=LAT0)&(ds.latitude<=LAT1)
        lonm=(ds.longitude>=LON0)&(ds.longitude<=LON1)
        sub=ds[KEEP].isel(latitude=np.where(latm)[0], longitude=np.where(lonm)[0]).load()
        ds.close(); os.remove(lp)
        slabs.append(sub); times.append(t)
    f.quit()
    cube=xr.concat(slabs, dim=pd.Index(times,name="time"))
    cube=cube.sortby("latitude").sortby("longitude")
    nlat,nlon=cube.sizes["latitude"],cube.sizes["longitude"]
    comp=BloscCodec(cname="zstd",clevel=5,shuffle="shuffle")
    enc={v:{"chunks":(1,nlat,nlon),"compressors":(comp,)} for v in KEEP}
    cube.attrs.update(source="JAXA Himawari Monitor (P-Tree) L3 PAR/021 SWR",
                      product="H09 RFL021 full-disk 0.05deg", kind="satellite_observation")
    zdir=os.path.join(work,f"{day}.zarr")
    cube.to_zarr(zdir, mode="w", encoding=enc, consolidated=True, zarr_format=3)
    zp=os.path.join(out_dir,f"{day}.zarr.zip")
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_STORED) as zf:
        for r,_,fs in os.walk(zdir):
            for x in fs: zf.write(os.path.join(r,x), os.path.relpath(os.path.join(r,x),zdir))
    import shutil; shutil.rmtree(work, ignore_errors=True)
    print(f"[{day}] wrote {zp}  dims time={cube.sizes['time']} lat={nlat} lon={nlon}  "
          f"vars={KEEP}  size={os.path.getsize(zp)/1e6:.1f} MB", flush=True)

if __name__=="__main__": main()
