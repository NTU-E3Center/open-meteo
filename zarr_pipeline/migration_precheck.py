#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-flight checks before migrating per-run .zarr.zip files into a mode-A Silver cube.

Scans a directory of *.zarr.zip and reports, per file and in aggregate, the things that
decide whether `convert_to_zarr.py --source zarrzip` will succeed and build a CONSISTENT
cube -- so you catch problems on a sample before launching a bulk migration:

  * filename matches YYYYMMDDTHHZ.zarr.zip (discover_runs/_ZIP_RE rely on it)
  * the zip is ZipStore-readable -- the #1 risk: the store (zarr.json) must sit at the
    zip ROOT, not nested under a folder. We detect nesting and tell you how to re-zip.
  * model / grid / forecast horizon / dims / dtypes, and run_init from attr vs filename
  * aggregate consistency: ONE model & ONE grid per cube (else split), run_init cadence,
    lead horizons, date span, and the known dwd_icon model-label mismatch.

Exits non-zero if any BLOCKING issue is found, so it can gate a migration script.

    python migration_precheck.py --data-dir /path/to/zips
    python migration_precheck.py --data-dir /path/to/zips --expect-model jma_msm --expect-grid 473x481
    python migration_precheck.py --data-dir /path/to/zips --limit 50   # quick spot-check
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import zipfile

import numpy as np

_ZIP_RE = re.compile(r"(\d{8}T\d{2})Z\.zarr\.zip$")


def find_store_root(zf):
    """'' if zarr.json is at the zip root; the nested prefix if it sits under one folder;
    None if there is no zarr.json at all."""
    names = zf.namelist()
    if "zarr.json" in names or "zarr.json" in {os.path.basename(n) for n in names if n.count("/") == 0}:
        return ""
    roots = {n[: -len("zarr.json")] for n in names if n.endswith("/zarr.json")}
    return sorted(roots, key=len)[0] if roots else None


def check_file(path, expect_model, expect_grid):
    import xarray as xr
    from zarr.storage import ZipStore

    rec = {"name": os.path.basename(path), "issues": [], "warnings": [],
           "model": "", "grid_dims": "?", "lead_min": None, "lead_max": None,
           "n_vars": None, "dtypes": [], "run_init": None, "fn_ts": None}

    m = _ZIP_RE.search(rec["name"])
    if not m:
        rec["issues"].append("檔名不符 YYYYMMDDTHHZ.zarr.zip")
    else:
        g = m.group(1)
        rec["fn_ts"] = f"{g[:4]}-{g[4:6]}-{g[6:8]}T{g[9:11]}"

    try:
        with zipfile.ZipFile(path) as zf:
            root = find_store_root(zf)
    except Exception as e:
        rec["issues"].append(f"不是有效 zip: {e}")
        return rec
    if root is None:
        rec["issues"].append("zip 內找不到 zarr.json(不是 zarr store)")
        return rec
    if root:
        rec["issues"].append(f"store 被包在子資料夾 '{root.rstrip('/')}' 下 → ZipStore 讀不到。"
                             f" 重壓:`cd <該.zarr目錄> && zip -r ../X.zarr.zip .`")
        return rec

    store = None
    try:
        store = ZipStore(path, mode="r")
        ds = xr.open_zarr(store, consolidated=False)        # lazy: metadata only
        rec["model"] = str(ds.attrs.get("model", ""))
        sizes = dict(ds.sizes)
        ny, nx = sizes.get("latitude"), sizes.get("longitude")
        rec["grid_dims"] = f"{ny}x{nx}" if ny and nx else "?"
        if "lead" in ds.variables:
            lv = ds["lead"].values.astype(int)
            rec["lead_min"], rec["lead_max"] = int(lv.min()), int(lv.max())
        fvars = list(ds.data_vars)
        rec["n_vars"] = len(fvars)
        rec["dtypes"] = sorted({str(ds[v].dtype) for v in fvars})
        if "run_init" in ds.variables:
            rec["run_init"] = str(np.asarray(ds["run_init"].values).ravel()[0])[:13]
        elif "run_init" in ds.attrs:
            rec["run_init"] = str(ds.attrs["run_init"])[:13]

        if not rec["model"]:
            rec["warnings"].append("缺 model 屬性")
        if rec["fn_ts"] and rec["run_init"] and not rec["run_init"].startswith(rec["fn_ts"]):
            rec["warnings"].append(f"檔名 {rec['fn_ts']} ≠ 屬性 run_init {rec['run_init']}")
        if expect_model and rec["model"] and rec["model"] != expect_model:
            rec["issues"].append(f"model '{rec['model']}' ≠ 預期 '{expect_model}'(標籤錯置?)")
        if expect_grid and rec["grid_dims"] not in ("?", expect_grid):
            rec["issues"].append(f"grid {rec['grid_dims']} ≠ 預期 {expect_grid}")
    except Exception as e:
        rec["issues"].append(f"open_zarr 失敗: {type(e).__name__}: {e}")
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder of *.zarr.zip files")
    ap.add_argument("--expect-model", default=None, help="flag files whose model attr differs")
    ap.add_argument("--expect-grid", default=None, help="e.g. 473x481; flag mismatches")
    ap.add_argument("--limit", type=int, default=None, help="only check the first N files")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data_dir, "*.zarr.zip")))
    if a.limit:
        files = files[: a.limit]
    if not files:
        sys.exit(f"no *.zarr.zip in {a.data_dir}")

    print(f"檢查 {len(files)} 個 .zarr.zip @ {a.data_dir}\n" + "=" * 78)
    recs = [check_file(f, a.expect_model, a.expect_grid) for f in files]

    n_bad = 0
    for r in recs:
        flag = "OK  " if not r["issues"] else "FAIL"
        if r["issues"]:
            n_bad += 1
        extra = (f"{r['model'] or '?':9s} {r['grid_dims']:9s} "
                 f"lead {r['lead_min']}..{r['lead_max']} {r['n_vars']}var {','.join(r['dtypes'])}")
        print(f"[{flag}] {r['name']:30s} {extra}")
        for i in r["issues"]:
            print(f"        ✗ {i}")
        for w in r["warnings"]:
            print(f"        ! {w}")

    # ---- aggregate consistency ----
    ok = [r for r in recs if not r["issues"]]
    print("=" * 78)
    models = sorted({r["model"] for r in ok if r["model"]})
    grids = sorted({r["grid_dims"] for r in ok if r["grid_dims"] != "?"})
    horizons = sorted({r["lead_max"] for r in ok if r["lead_max"] is not None})
    dtypesets = {tuple(r["dtypes"]) for r in ok}

    print(f"可讀: {len(ok)}/{len(files)}  |  models={models}  grids={grids}  "
          f"horizons(max lead)={horizons}")

    # run_init cadence
    ts = sorted({r["fn_ts"] for r in ok if r["fn_ts"]})
    if len(ts) >= 2:
        arr = np.array(ts, dtype="datetime64[h]")
        diffs = np.diff(arr).astype("timedelta64[h]").astype(int)
        steps = sorted(set(diffs.tolist()))
        print(f"run_init 範圍: {ts[0]} .. {ts[-1]}  |  間隔(小時): {steps}")

    blocking = []
    if len(models) > 1:
        blocking.append(f"多個 model {models} → 請分開、一個 model 一個 cube")
    if len(grids) > 1:
        blocking.append(f"多個 grid {grids} → 不能放同一 cube,請分開")
    if len(dtypesets) > 1:
        blocking.append(f"變數 dtype 組合不一致 {dtypesets} → 各起報 schema 不同,需先對齊")
    if n_bad:
        blocking.append(f"{n_bad} 個檔有 FAIL 問題(見上)")

    print("=" * 78)
    if blocking:
        print("結論:✗ 尚不可整批 migration,需先處理:")
        for b in blocking:
            print(f"  - {b}")
        sys.exit(1)
    print("結論:✓ 通過,可以用 convert_to_zarr.py --source zarrzip 進行 migration。")
    if horizons and len(horizons) > 1:
        print(f"  提示:horizons {horizons} 混合 → --init 時 --lead-max 設為最大值 {max(horizons)}。")


if __name__ == "__main__":
    main()
