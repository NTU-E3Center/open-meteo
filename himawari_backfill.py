#!/usr/bin/env python3
"""Backfill JAXA Himawari SWR (GHI ground truth) -> per-day APAC zarr.zip on HF.

Iterates UTC days in [--since,--until], and for each day not already on HF: runs
himawari_fetch_day.py (download 24 hourly full-disk files -> APAC SWR cube) and uploads
the single <YYYYMMDD>.zarr.zip. Per-day single-file upload (no batched commit -> no silent
hang, the lesson from the icon re-collect). Idempotent/resumable: skips days already on HF.

Usage: himawari_backfill.py --since 2026-03-19 --until 2026-06-21 [--workers 2] [--force]
"""
import argparse, os, subprocess, sys, tempfile, threading, time, datetime, shutil
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi

REPO="jimtseng/apac-himawari-swr"
HERE=os.path.dirname(os.path.abspath(__file__))
api=HfApi(); _lock=threading.Lock()
def log(m):
    with _lock: print(m, flush=True)

def hf_days():
    out=set()
    for f in api.list_repo_files(REPO, repo_type="dataset"):
        if f.endswith(".zarr.zip"):
            out.add(f.rsplit("/",1)[-1].removesuffix(".zarr.zip"))   # YYYYMMDD
    return out

def retry(fn, what, tries=6):
    for i in range(1,tries+1):
        try: return fn()
        except Exception as e:
            if i==tries: raise
            resp=getattr(e,"response",None); is429=getattr(resp,"status_code",None)==429
            wait=min(180,15*i) if is429 else 5*i
            log(f"      {what} attempt {i} failed ({type(e).__name__}); wait {wait}s"); time.sleep(wait)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--since", required=True); ap.add_argument("--until", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    a=ap.parse_args()
    s=datetime.date.fromisoformat(a.since); u=datetime.date.fromisoformat(a.until)
    days=[(s+datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range((u-s).days+1)]
    have=hf_days()
    todo=days if a.force else [d for d in days if d not in have]
    log(f"[hima] range {a.since}..{a.until}  days={len(days)}  have={len(have)}  todo={len(todo)}  workers={a.workers}")
    done=[0]
    def one(day):
        wd=tempfile.mkdtemp(prefix="himabf_")
        try:
            t0=time.time()
            r=subprocess.run([sys.executable, os.path.join(HERE,"himawari_fetch_day.py"), day, wd],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            zp=os.path.join(wd, f"{day}.zarr.zip")
            if r.returncode!=0 or not os.path.exists(zp):
                tail=(r.stdout or b"").decode(errors="replace").strip().splitlines()[-2:]
                log(f"[FAIL] {day}: {' | '.join(tail)[:200]}"); return None
            repo_path=f"data/year={day[:4]}/month={day[4:6]}/{day}.zarr.zip"
            mb=os.path.getsize(zp)/1e6
            retry(lambda: api.upload_file(path_or_fileobj=zp, path_in_repo=repo_path,
                  repo_id=REPO, repo_type="dataset",
                  commit_message=f"Add Himawari SWR {day}"), f"{day} upload")
            with _lock: done[0]+=1; n=done[0]
            log(f"[hima] OK {day}  ({time.time()-t0:.0f}s, {mb:.0f} MB)  ({n}/{len(todo)})")
            return day
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, todo))
    log(f"[hima] DONE: uploaded {done[0]}/{len(todo)}")

if __name__=="__main__": main()
