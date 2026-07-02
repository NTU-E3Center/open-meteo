# zarr_pipeline/tests/test_rish_backfill.py
import numpy as np
import pandas as pd
import xarray as xr
from rish_backfill import pending_runs, rish_urls


def test_rish_urls_five_segments_correct_layout():
    urls = rish_urls(np.datetime64("2025-08-01T12:00"))
    assert len(urls) == 5
    assert urls[0].endswith(
        "/2025/08/01/Z__C_RJTD_20250801120000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin")
    assert "database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/" in urls[0]


def test_pending_runs_skips_filled_and_offgrid(mini_cube):
    path, axis = mini_cube                       # slots 0 and 2 filled (00Z and 06Z)
    todo = pending_runs(path, "2026-05-12", "2026-05-13", cycles=(0, 3, 6))
    todo_ts = [pd.Timestamp(t) for t in todo]
    assert pd.Timestamp("2026-05-12 00:00") not in todo_ts    # filled -> skipped
    assert pd.Timestamp("2026-05-12 06:00") not in todo_ts    # filled -> skipped
    assert pd.Timestamp("2026-05-12 03:00") in todo_ts        # empty -> pending
    assert all(t.hour in (0, 3, 6) for t in todo_ts)          # cycle filter applied
