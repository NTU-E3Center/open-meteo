#!/usr/bin/env python3
"""Daily completeness check for the HF forecast archive.

Lists hf://datasets/JimTseng/apac-nwp-forecast-archive and prints, for each of
the last N days, which model runs are present / missing versus the expected
schedule (ICON/GFS/IFS/AIFS: 00,06,12,18Z; JMA MSM: every 3 h).

Usage: python3 check_archive.py [days]   (default: 3)
"""
import sys
import datetime
from collections import defaultdict

from huggingface_hub import HfApi

REPO = "JimTseng/apac-nwp-forecast-archive"
EXPECTED = {
    "dwd_icon":             [0, 6, 12, 18],
    "ncep_gfs013":          [0, 6, 12, 18],
    "jma_msm":              [0, 3, 6, 9, 12, 15, 18, 21],
    "ecmwf_ifs025":         [0, 6, 12, 18],
    "ecmwf_aifs025_single": [0, 6, 12, 18],
}
# Publication lag (hours) per model: a run isn't "missing" until init + lag has passed.
LAG_H = {"dwd_icon": 5, "ncep_gfs013": 7, "jma_msm": 5,
         "ecmwf_ifs025": 9, "ecmwf_aifs025_single": 7}

def main(days: int) -> None:
    api = HfApi()
    have = defaultdict(set)  # (model, date) -> {hour, ...}
    for path in api.list_repo_files(REPO, repo_type="dataset"):
        if not path.endswith(".parquet") or "model=" not in path:
            continue
        model = path.split("model=")[1].split("/")[0]
        stamp = path.rsplit("/", 1)[-1].removesuffix(".parquet")  # 20260612T00Z
        d, h = stamp[:8], int(stamp[9:11])
        have[(model, d)].add(h)

    now = datetime.datetime.now(datetime.UTC)
    today = now.date()
    total_missing = 0
    for offset in range(days - 1, -1, -1):
        day = today - datetime.timedelta(days=offset)
        d = day.strftime("%Y%m%d")
        print(f"=== {day} (UTC) ===")
        for model, hours in EXPECTED.items():
            got = have.get((model, d), set())
            due = [h for h in hours
                   if datetime.datetime(day.year, day.month, day.day, h,
                                        tzinfo=datetime.UTC)
                   + datetime.timedelta(hours=LAG_H[model]) <= now]
            missing = [h for h in due if h not in got]
            not_due = [h for h in hours if h not in due]
            total_missing += len(missing)
            status = "OK " if not missing else "GAP"
            parts = f"{len([h for h in due if h in got])}/{len(due)} due"
            if missing:
                parts += "  missing: " + ",".join(f"{h:02d}Z" for h in missing)
            if not_due:
                parts += "  (not yet published: " + ",".join(f"{h:02d}Z" for h in not_due) + ")"
            print(f"  [{status}] {model:22} {parts}")
    print(f"\n{'✅ archive complete' if total_missing == 0 else f'⚠️  {total_missing} run(s) missing'}")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
