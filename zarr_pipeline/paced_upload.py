#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paced batch upload of a local folder to a HF dataset, staying under free-tier rate limits
(1000 API requests / 5 min, 128 commits / hour). Uploads files in small batches with ONE
commit per batch and a sleep between batches; on 429 it backs off honouring the Retry-After /
"about 1 hour" hint instead of hammering (which is what makes upload-large-folder thrash).
Resumable: files already in the repo are skipped, so it just fills in what's missing.

    python paced_upload.py <repo_id> <local_root> [--batch 64] [--sleep 45]

local_root's contents map to the repo root, e.g. local_root/dwd_icon_silver.zarr/... ->
repo path dwd_icon_silver.zarr/...
"""
import argparse
import os
import re
import time

from huggingface_hub import CommitOperationAdd, HfApi


def backoff_seconds(msg, tries):
    low = msg.lower()
    m = re.search(r"retry after (\d+)\s*second", low)
    if m:
        return int(m.group(1)) + 5
    if "hour" in low:                         # commit-per-hour limit -> wait out the hour
        return 3700
    m = re.search(r"retry after (\d+)", low)
    if m:
        return int(m.group(1)) + 5
    return min(600, 60 * tries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("root")
    ap.add_argument("--batch", type=int, default=64, help="files per commit")
    ap.add_argument("--sleep", type=float, default=45, help="seconds between batches")
    ap.add_argument("--repo-type", default="dataset")
    a = ap.parse_args()
    api = HfApi()

    local = []
    for dp, dirs, files in os.walk(a.root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]   # skip .cache/ etc.
        for f in files:
            if f.startswith("."):
                continue
            full = os.path.join(dp, f)
            local.append((os.path.relpath(full, a.root), full))
    local.sort()
    have = set(api.list_repo_files(a.repo, repo_type=a.repo_type))
    todo = [(rel, full) for rel, full in local if rel not in have]
    print(f"local {len(local)} | on HF {len(have)} | to upload {len(todo)} "
          f"| batch {a.batch} sleep {a.sleep}s", flush=True)
    if not todo:
        print("UPLOAD DONE (nothing to upload)", flush=True)
        return

    def commit_batch(batch, label):
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=full) for rel, full in batch]
        tries = 0
        while True:
            tries += 1
            try:
                api.create_commit(a.repo, repo_type=a.repo_type, operations=ops,
                                  commit_message=f"add {len(batch)} files ({label})")
                return
            except Exception as e:
                msg = str(e)
                if "429" in msg or "rate limit" in msg.lower() or "too many" in msg.lower():
                    w = backoff_seconds(msg, tries)
                    print(f"  429 -> sleep {w}s (try {tries})", flush=True)
                    time.sleep(w)
                elif tries >= 6:
                    raise
                else:
                    time.sleep(10 * tries)

    done = 0
    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        commit_batch(batch, f"{i + len(batch)}/{len(todo)}")
        done += len(batch)
        print(f"[{done}/{len(todo)}] committed ({time.strftime('%H:%M:%S')})", flush=True)
        if i + a.batch < len(todo):
            time.sleep(a.sleep)
    print("UPLOAD DONE", flush=True)


if __name__ == "__main__":
    main()
