# Himawari Silver Cube — Migration Runbook

## Overview

The Himawari SWR silver cube (`himawari_swr_silver.zarr`) lives on HuggingFace at
`jimtseng/apac-himawari-swr`.  The daily ingestion pipeline (`himawari_daily.sh`) runs
on the Mac mini via cron and already writes per-day zarr.zips.  With `HIMAWARI_CUBE=1`
it additionally region-writes each completed day into the silver cube.

---

## Time convention

The cube's time axis represents **window START** (= the raw JAXA file hour).
The solar-ghi-nwp label caches use **window END = cube time + 1 h** (`SWR_TIME_OFFSET`).
Always confirm which convention a consumer expects before joining; a silent 1-hour
offset between the cube and label cache reduces model correlation from r=1.0 to r=0.84
(observed in the xcheck that uncovered this issue).

---

## Mac Mini Deployment Steps

> **Machine:** all steps in this section run on the **Mac mini** unless noted otherwise.
> Use `python3` (the `~/.om-venv` on PATH per `himawari_daily.sh`).

### Preflight (Mac mini)

Before enabling `HIMAWARI_CUBE`, verify that all required Python dependencies are
present in `~/.om-venv`.  `dask` is required by `himawari_cube.py` and may be absent
from the mini's venv:

```bash
python3 -c "import dask, xarray, zarr, huggingface_hub; print('deps OK')"
```

If this fails, install the missing package(s) into `~/.om-venv` first:

```bash
pip install dask   # or whichever package is missing
```

### 1. Pull the latest code

```bash
cd /path/to/open-meteo   # wherever the repo is checked out on the Mac mini
git pull
```

### 2. Enable dual-write (zip + cube)

Add `HIMAWARI_CUBE=1` to the environment that the cron job sees.  Two options:

**Option A — add to `~/.himawari_ftp.env`** (recommended; already sourced by the script):

```bash
echo 'export HIMAWARI_CUBE=1' >> ~/.himawari_ftp.env
```

**Option B — add to the crontab environment line:**

```
HIMAWARI_CUBE=1
```
(Place it above the crontab entry that calls `himawari_daily.sh`.)

### 3. Verify 3 consecutive days of dual-write

> **Machine:** Mac mini.  Use `python3`.

After the cron has run at least 3 times (one run per day):

**Check `slot_filled` on the remote cube for a specific day:**

```python
# Run with: python3
import sys
sys.path.insert(0, "/path/to/open-meteo/zarr_pipeline")

from huggingface_hub import snapshot_download
import os, tempfile
from himawari_cube import day_filled

# Download cube skeleton (coords + slot_filled chunks only — fast)
skel = tempfile.mkdtemp(prefix="hima_verify_")
cube_name = "himawari_swr_silver.zarr"
snapshot_download(
    "jimtseng/apac-himawari-swr",
    repo_type="dataset",
    allow_patterns=[
        f"{cube_name}/zarr.json",
        f"{cube_name}/*/zarr.json",
        f"{cube_name}/slot_filled/c/**",
        f"{cube_name}/time/c/**",
    ],
    local_dir=skel,
)
store = os.path.join(skel, cube_name)

for day in ["YYYY-MM-DD", "YYYY-MM-DD", "YYYY-MM-DD"]:  # fill in actual dates
    status = "FILLED" if day_filled(store, day) else "MISSING"
    print(f"{day}: {status}")
```

**Quick one-liner to check the three most-recent days:**

```bash
DAY=$(date -u -v-1d +%Y-%m-%d)
for d in $DAY $(date -u -v-2d +%Y-%m-%d) $(date -u -v-3d +%Y-%m-%d); do
  echo -n "$d: "
  python3 - "$d" <<'PY'
import sys
sys.path.insert(0, "/path/to/open-meteo/zarr_pipeline")
from huggingface_hub import snapshot_download
import os, tempfile
from himawari_cube import day_filled
skel = tempfile.mkdtemp()
cube = "himawari_swr_silver.zarr"
snapshot_download("jimtseng/apac-himawari-swr", repo_type="dataset",
    allow_patterns=[f"{cube}/zarr.json", f"{cube}/*/zarr.json",
                    f"{cube}/slot_filled/c/**", f"{cube}/time/c/**"],
    local_dir=skel)
print("FILLED" if day_filled(os.path.join(skel, cube), sys.argv[1]) else "MISSING")
PY
done
```

> **Cache note:** `--from-hf` (and `snapshot_download` above) re-downloads each day zip
> (~52 MB) into `~/.cache/huggingface`.  Prune periodically with
> `huggingface-cli delete-cache`, or switch to `--from-zip` with retained local zips to
> avoid repeat downloads.

**Confirm the cube was updated on HF (check commit history):**

```bash
# Using huggingface_hub CLI
pip install huggingface_hub
python3 -c "
from huggingface_hub import HfApi
commits = HfApi().list_repo_commits('jimtseng/apac-himawari-swr', repo_type='dataset')
for c in list(commits)[:5]:
    print(c.created_at, c.title[:80])
"
```

---

## P4 Checklist — Retire the Daily Zips (USER-GATED)

> **STOP.** Each step below requires explicit user confirmation before execution.
> Do NOT proceed to the next step until the stated precondition is verified.

### P4.1 — Stop new zip writes

> **Machine:** Mac mini.

**PRECONDITION:** Confirm that at least 3 consecutive days have `slot_filled == True`
in the remote cube (run the verification block above).  All three must show `FILLED`.

**Action:** Comment out the `himawari_daily.sh` crontab entry (the ONLY clean way to
stop zip writes — do NOT just remove the FTP credentials: the `:?` guard would then
make every cron run exit with an error and spam the cron log).

```bash
# Option: comment out the crontab entry for himawari_daily.sh
crontab -e   # comment out the relevant line
```

**Verify:** Wait one cron cycle (24 h) and confirm no new `.zarr.zip` files appear
on HF at `jimtseng/apac-himawari-swr/data/`.

---

### P4.2 — Delete HF daily zips from `jimtseng/apac-himawari-swr/data/`

> **Machine:** laptop or Mac mini.

**PRECONDITION:** P4.1 complete AND confirmed. No new zips have been added for ≥ 24 h.

**MANDATORY PRECONDITION — full-coverage check (data-loss guard):**

Before deleting any zips, verify that EVERY zip day is `day_filled` in the remote cube.
A "permanent-rejection" scenario exists: if a source day contains fewer than 24 hours of
data, `write_day` may reject it every time (the day is short at source but the zip is the
only surviving copy).  Deleting its zip without confirming `day_filled` would permanently
destroy that data.

Run the following check and require **ZERO MISSING** days before proceeding:

```python
# Run with: python3 (Mac mini) or the laptop venv python
# Machine: whichever has a HuggingFace token configured.
import sys
sys.path.insert(0, "/path/to/open-meteo/zarr_pipeline")

from huggingface_hub import snapshot_download
import os, tempfile
from himawari_cube import day_filled
from himawari_transcode import _list_days

# 1. Download cube skeleton (slot_filled + time coords only — fast)
skel = tempfile.mkdtemp(prefix="hima_p42_check_")
cube_name = "himawari_swr_silver.zarr"
snapshot_download(
    "jimtseng/apac-himawari-swr",
    repo_type="dataset",
    allow_patterns=[
        f"{cube_name}/zarr.json",
        f"{cube_name}/*/zarr.json",
        f"{cube_name}/slot_filled/c/**",
        f"{cube_name}/time/c/**",
    ],
    local_dir=skel,
)
store = os.path.join(skel, cube_name)

# 2. Enumerate every day present as a zarr.zip in the HF repo
days = [day_iso for day_iso, _hf_path in _list_days()]
print(f"Total zip days found: {len(days)}")

# 3. Assert day_filled for each; print MISSING for any gap
missing = []
for day in days:
    if not day_filled(store, day):
        print(f"MISSING: {day}")
        missing.append(day)

if missing:
    print(f"\nFAIL: {len(missing)} day(s) not filled in the cube.")
    print("Investigate each MISSING day before proceeding:")
    print("  - Short source day (<24h) rejected by write_day? -> decide: keep-zip-forever or accept gap.")
    print("  - Do NOT delete zips until all days are resolved.")
else:
    print(f"\nOK: all {len(days)} days are day_filled in the cube. Safe to proceed with deletion.")
```

**Require zero MISSING days.** If any day is MISSING, investigate before continuing:
- Short source day (<24 h of data) permanently rejected by `write_day` → decide whether
  to keep that zip forever (skip it from deletion) or accept the gap in the cube.
- Only after every day is resolved (filled or explicitly accepted as a gap) may you
  proceed to the deletion below.

**Action (user must run explicitly):**

```python
from huggingface_hub import HfApi
api = HfApi()
repo = "jimtseng/apac-himawari-swr"

# List all daily zip paths
files = [f for f in api.list_repo_files(repo, repo_type="dataset")
         if f.startswith("data/") and f.endswith(".zarr.zip")]
print(f"Files to delete: {len(files)}")
# Review the list before proceeding!
for f in files:
    print(f)

# ONLY run the deletion after reviewing the list above:
# from huggingface_hub import CommitOperationDelete
# ops = [CommitOperationDelete(path_in_repo=f) for f in files]
# api.create_commit(repo, repo_type="dataset", operations=ops,
#                   commit_message="P4: remove daily zarr.zips (superseded by silver cube)")
```

**Verify:** `api.list_repo_files(repo, repo_type="dataset")` returns no entries
starting with `data/` and ending with `.zarr.zip`.

---

### P4.3 — Delete `apac-nwp-forecast/himawari_swr/` directory

**PRECONDITION:** P4.2 complete AND confirmed.

> **Note:** This step removes the `himawari_swr` subtree from the
> `jimtseng/apac-nwp-forecast` dataset repo (if such a path exists and is redundant
> with the silver cube).  Confirm with the user which exact paths to delete before
> running.

**Action (user must run explicitly):**

```python
from huggingface_hub import HfApi, CommitOperationDelete
api = HfApi()
repo = "jimtseng/apac-nwp-forecast"

files = [f for f in api.list_repo_files(repo, repo_type="dataset")
         if f.startswith("himawari_swr/")]
print(f"Files to delete: {len(files)}")
for f in files:
    print(f)

# ONLY run after reviewing:
# ops = [CommitOperationDelete(path_in_repo=f) for f in files]
# api.create_commit(repo, repo_type="dataset", operations=ops,
#                   commit_message="P4: remove himawari_swr/ (migrated to silver cube)")
```

**Verify:** `api.list_repo_files("jimtseng/apac-nwp-forecast", repo_type="dataset")`
returns no entries starting with `himawari_swr/`.

---

## Quick Reference

| Task | Command |
|------|---------|
| Enable cube dual-write | `echo 'export HIMAWARI_CUBE=1' >> ~/.himawari_ftp.env` |
| Manual one-day write | `python3 himawari_day_to_cube.py --day YYYY-MM-DD --from-hf` |
| Manual one-day write (local zip) | `python3 himawari_day_to_cube.py --day YYYY-MM-DD --from-zip /path/to/YYYYMMDD.zarr.zip` |
| Check fill status | See Python snippet in §3 above |
| Check HF commit log | See `HfApi().list_repo_commits(...)` snippet above |
| Prune HF download cache | `huggingface-cli delete-cache` |
