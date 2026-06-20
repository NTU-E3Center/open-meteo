#!/usr/bin/env python3
"""Re-collect the dwd_icon archive WITH OCEAN from open-meteo S3 and publish it as
per-run 4D zarr cubes (<stamp>.zarr.zip), superseding the land-only parquet on HF.

Why this is NOT batch_jma_to_zarr.py: JMA's HF parquet was already a full grid (its
--ignore_sea was a no-op), so JMA converted in place. dwd_icon's HF parquet is LAND-ONLY
(its --ignore_sea really dropped ~66% sea cells), so to get ocean values we must RE-EXPORT
each run from S3 without --ignore_sea. The export source is data_run on S3 (immutable,
init-addressable, ~3-month retention) — so only runs still inside that window can be
recovered; older ones stay land-only.

Per run:  docker export (no --ignore_sea, full horizon)  ->  postprocess.py  ->
          parquet_to_zarr_cube.py (lead_chunk=12, full-grid map-first)  ->  zip  ->
          batched CommitOperationAdd.   Then cutover: delete the superseded land-only
          parquet in batched CommitOperationDelete.

Idempotent/resumable: re-lists HF each run and skips any run whose .zarr.zip already
exists; a run's parquet is deleted only after its zip is confirmed, so no run is ever
left with neither. The same batched-commit + 429-aware backoff as the JMA batch avoids
HF's commit rate limit.

Usage: batch_icon_recollect.py [--workers 3] [--lead-chunk 12] [--add-batch 8]
                               [--del-batch 50] [--cache-size 2GB] [--since YYYY-MM-DD]
                               [--until YYYY-MM-DD] [--limit N] [--keep-parquet] [--dry-run]
"""
import argparse, datetime, os, shutil, subprocess, sys, tempfile, threading, time, urllib.parse, urllib.request, zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import (HfApi, CommitOperationAdd, CommitOperationDelete)

REPO = "JimTseng/apac-nwp-forecast-archive"
MODEL = "dwd_icon"
HERE = os.path.dirname(os.path.abspath(__file__))
S3_BASE = "https://openmeteo.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
IMAGE = "open-meteo:bbox-fix"
REMOTE_DATA = "https://openmeteo.s3.amazonaws.com/data/"
CACHE_VOLUME = "open-meteo-cache"
REGION_LAT, REGION_LON = "m44,46", "92,154"
HORIZON_DAYS = 8                                        # dwd_icon native 7.5-day horizon
# 15 vars: the 13 solar vars + snow_depth/snowfall (icon-only; ~free in storage, but the
# paper's 3rd-most-important bias factor). snow is requested for icon because its S3 source
# has it; jma_msm has no snow on S3 so the daily cron must NOT request it for jma.
VARS = ("shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance,"
        "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation,"
        "cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
        "snow_depth,snowfall_water_equivalent")

api = HfApi()
_lock = threading.Lock()

def log(m):
    with _lock:
        print(m, flush=True)

def s3_list(prefix, delim="/"):
    keys, pre, tok = [], [], None
    while True:
        p = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delim: p["delimiter"] = delim
        if tok: p["continuation-token"] = tok
        root = ET.fromstring(urllib.request.urlopen(
            f"{S3_BASE}/?" + urllib.parse.urlencode(p), timeout=30).read())
        keys += [c.find(NS+"Key").text for c in root.findall(NS+"Contents")]
        pre += [x.find(NS+"Prefix").text for x in root.findall(NS+"CommonPrefixes")]
        nxt = root.find(NS+"NextContinuationToken")
        if nxt is None: break
        tok = nxt.text
    return keys, pre

def s3_runs(since, until):
    """List every dwd_icon run available on S3 in [since,until] as (stamp, run_iso, start, end)."""
    runs = []
    _, years = s3_list(f"data_run/{MODEL}/")
    for y in years:
        _, months = s3_list(y)
        for m in months:
            _, days = s3_list(m)
            for d in days:
                day = d.rstrip("/").split(f"data_run/{MODEL}/")[1]      # YYYY/MM/DD
                dt = datetime.date.fromisoformat(day.replace("/", "-"))
                if (since and dt < since) or (until and dt > until):
                    continue
                _, rps = s3_list(d)
                for rp in rps:
                    hh = rp.rstrip("/").rsplit("/", 1)[-1][:2]          # HHMMZ -> HH
                    stamp = f"{dt:%Y%m%d}T{hh}Z"
                    run_iso = f"{dt:%Y-%m-%d}T{hh}:00"
                    start = f"{dt:%Y-%m-%d}"
                    end = (dt + datetime.timedelta(days=HORIZON_DAYS)).isoformat()
                    runs.append((stamp, run_iso, start, end))
    return sorted(set(runs))

def zip_store(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, src_dir))

def retry(fn, what, tries=8):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if i == tries:
                raise
            resp = getattr(e, "response", None)
            is_429 = getattr(resp, "status_code", None) == 429
            if is_429:
                ra = (getattr(resp, "headers", None) or {}).get("Retry-After")
                wait = int(ra) if (ra and str(ra).isdigit()) else min(180, 15 * i)
            else:
                wait = 5 * i
            log(f"      {what} attempt {i} failed ({type(e).__name__}"
                f"{' 429' if is_429 else ''}); waiting {wait}s")
            time.sleep(wait)

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def hf_state():
    files = api.list_repo_files(REPO, repo_type="dataset")
    zip_stamps = {f.rsplit("/", 1)[-1].removesuffix(".zarr.zip")
                  for f in files if f.endswith(".zarr.zip") and f"model={MODEL}/" in f}
    pq = {f.rsplit("/", 1)[-1].removesuffix(".parquet"): f
          for f in files if f.endswith(".parquet") and f"model={MODEL}/" in f}
    return zip_stamps, pq

def export_convert_zip(stamp, run_iso, start, end, workdir, lead_chunk, cache_size):
    raw = os.path.join(workdir, f"{stamp}_raw.parquet")
    cmd = ["docker", "run", "--rm",
           "-v", f"{CACHE_VOLUME}:/app/data", "-v", f"{workdir}:/out",
           "-e", f"REMOTE_DATA_DIRECTORY={REMOTE_DATA}", "-e", f"CACHE_SIZE={cache_size}",
           "--entrypoint", "/app/openmeteo-api", IMAGE,
           "export", MODEL, VARS, "--run", run_iso,
           "--start_date", start, "--end_date", end,
           "--latitude-bounds", REGION_LAT, "--longitude-bounds", REGION_LON,
           # NOTE: NO --ignore_sea -> ocean cells are kept (the whole point).
           "--concurrent", "8", "--format", "parquet", "-o", f"/out/{stamp}_raw.parquet"]
    # Retry the export: it is network-bound (S3) and an occasional docker/S3 blip must
    # NOT permanently skip a run — the oldest runs expire from S3 first, so a deferred
    # retry could lose them. 3 attempts with backoff; capture stderr tail on failure.
    for attempt in range(1, 4):
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if r.returncode == 0 and os.path.exists(raw):
            break
        tail = (r.stderr or b"").decode(errors="replace").strip().splitlines()[-3:]
        log(f"      {stamp} export attempt {attempt} failed (rc={r.returncode}): "
            f"{' | '.join(tail)[:240]}")
        if os.path.exists(raw):
            os.remove(raw)
        if attempt == 3:
            raise RuntimeError(f"{stamp} export failed after 3 attempts")
        time.sleep(30 * attempt)
    clean = os.path.join(workdir, f"{stamp}.parquet")
    env = dict(os.environ, RUN_STAMP=stamp,
               SCRAPED_AT=datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ"))
    subprocess.run([sys.executable, os.path.join(HERE, "postprocess.py"), raw, clean],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    os.remove(raw)
    zdir = os.path.join(workdir, f"{stamp}.zarr")
    zpath = os.path.join(workdir, f"{stamp}.zarr.zip")
    subprocess.run([sys.executable, os.path.join(HERE, "parquet_to_zarr_cube.py"),
                    clean, zdir, str(lead_chunk)], check=True, stderr=subprocess.DEVNULL)
    os.remove(clean)
    zip_store(zdir, zpath)
    shutil.rmtree(zdir, ignore_errors=True)
    return zpath

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--lead-chunk", type=int, default=12)
    ap.add_argument("--add-batch", type=int, default=8)
    ap.add_argument("--del-batch", type=int, default=50)
    ap.add_argument("--cache-size", default="2GB")
    ap.add_argument("--since", default="")
    ap.add_argument("--until", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-parquet", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-collect ALL runs, overwriting existing .zarr.zip "
                         "(needed when the variable set changed, e.g. adding snow)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    since = datetime.date.fromisoformat(a.since) if a.since else None
    until = datetime.date.fromisoformat(a.until) if a.until else None

    avail = s3_runs(since, until)
    zip_have, pq_have = hf_state()
    todo = list(avail) if a.force else [r for r in avail if r[0] not in zip_have]
    if a.limit:
        todo = todo[-a.limit:]                          # newest-first slice for testing
    log(f"[icon] s3_available={len(avail)}  have_zip={len(zip_have)}  "
        f"to_recollect={len(todo)}  workers={a.workers}  lead_chunk={a.lead_chunk}")
    if a.dry_run:
        for s, ri, st, en in todo[:10]:
            log(f"  would export {s}  run={ri}  {st}..{en}")
        log(f"  ... ({len(todo)} total)")
        return

    # ---- Phase 1: export+convert+upload each missing run INDIVIDUALLY ----------
    # Per-run single-file upload (api.upload_file), NOT a batched create_commit. The
    # multi-file batched commit can wedge SILENTLY with no timeout (observed: a 3-zip
    # ~1.5 GB commit hung 75 min at 0% CPU, no network) — and the retry wrapper can't
    # see a hang that throws no exception. Single-file uploads are stable; at ~20 min/run
    # build pace, per-run commits land minutes apart so there is no 429 pressure. Each
    # run uploads the instant it is built, so a crash loses at most one in-flight run.
    done = [0]
    def build(item):
        stamp, run_iso, start, end = item
        wd = tempfile.mkdtemp(prefix="iconzarr_")
        try:
            t0 = time.time()
            zpath = export_convert_zip(stamp, run_iso, start, end, wd,
                                       a.lead_chunk, a.cache_size)
            sub = f"model={MODEL}/year={stamp[:4]}/month={stamp[4:6]}/day={stamp[6:8]}"
            zip_repo = f"data/{sub}/{stamp}.zarr.zip"
            mb = os.path.getsize(zpath) / 1e6
            retry(lambda: api.upload_file(
                path_or_fileobj=zpath, path_in_repo=zip_repo,
                repo_id=REPO, repo_type="dataset",
                commit_message=f"Add {MODEL} {stamp} zarr cube"),
                f"{stamp} upload")
            with _lock:
                done[0] += 1; n = done[0]
            log(f"[icon] OK {stamp}  ({time.time()-t0:.0f}s build, {mb:.0f} MB)  ({n}/{len(todo)})")
            return stamp
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for _ in ex.map(lambda it: _safe(build, it), todo):
            pass
    converted = done[0]
    log(f"[icon] phase 1 done: uploaded {converted}/{len(todo)}")

    # ---- Phase 2: cutover — delete land-only parquets that now have a zip ------
    if a.keep_parquet:
        log("[icon] keep-parquet set; skipping delete"); return
    zip_now, pq_now = hf_state()
    to_delete = [path for s, path in sorted(pq_now.items()) if s in zip_now]
    log(f"[icon] cutover: {len(to_delete)} land-only parquet to delete (have ocean zip)")
    deleted = 0
    for di, group in enumerate(chunks(to_delete, a.del_batch)):
        delops = [CommitOperationDelete(path_in_repo=p) for p in group]
        retry(lambda: api.create_commit(
            repo_id=REPO, repo_type="dataset", operations=delops,
            commit_message=f"Cutover: delete {len(delops)} superseded land-only dwd_icon parquet (batch {di+1})"),
            f"commit del-batch {di+1}")
        deleted += len(delops)
        log(f"[icon] del-commit {di+1}: -{len(delops)} parquet  (total {deleted}/{len(to_delete)})")
    log(f"[icon] DONE: recollected={converted} parquet_deleted={deleted}")

def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception as e:
        log(f"[FAIL] {arg[0]}: {type(e).__name__}: {str(e)[:160]}")
        return None

if __name__ == "__main__":
    main()
