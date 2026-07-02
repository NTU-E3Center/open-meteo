# zarr_pipeline/tests/test_prepend.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from prepend_run_init import prepend
from convert_to_zarr import build_template, _write_run

VAR = "shortwave_radiation_wattPerSquareMetre"


def test_prepend_shifts_data_bitexact_and_extends_axis(mini_cube):
    path, old_axis = mini_cube
    before = xr.open_zarr(path, consolidated=True)
    v0 = before[VAR].sel(run_init=old_axis[0]).values.copy()
    v2 = before[VAR].sel(run_init=old_axis[2]).values.copy()
    before.close()

    info = prepend(path, "2026-05-11")          # 8 new 3h slots before old start
    assert info["offset"] == 8
    assert info["n_new"] == info["n_old"] + 8

    after = xr.open_zarr(path, consolidated=True)
    ri = pd.DatetimeIndex(after.run_init.values)
    assert ri[0] == pd.Timestamp("2026-05-11 00:00")
    assert ri[info["offset"]] == pd.Timestamp(old_axis[0])
    np.testing.assert_array_equal(after[VAR].sel(run_init=old_axis[0]).values, v0)
    np.testing.assert_array_equal(after[VAR].sel(run_init=old_axis[2]).values, v2)
    sf = after["slot_filled"].values
    assert list(np.where(sf == 1)[0]) == [info["offset"], info["offset"] + 2]
    # new historical slots are empty (sentinel -> NaN on read)
    assert np.isnan(after[VAR].isel(run_init=0).values).all()
    after.close()


def test_prepend_dry_run_changes_nothing(mini_cube):
    path, old_axis = mini_cube
    info = prepend(path, "2026-05-11", dry_run=True)
    assert info["offset"] == 8
    ds = xr.open_zarr(path, consolidated=True)
    assert pd.Timestamp(ds.run_init.values[0]) == pd.Timestamp(old_axis[0])
    ds.close()


def test_prepend_rejects_offgrid_start(mini_cube):
    path, _ = mini_cube
    import pytest
    with pytest.raises(SystemExit):
        prepend(path, "2026-05-11T01:30")        # not on the 3h grid


def test_prepend_rejects_single_slot_axis(tmp_path):
    out = str(tmp_path / "one.zarr")
    axis = np.array([np.datetime64("2026-05-12T00")], dtype="datetime64[ns]")
    # Create minimal sample dataset with same shape as _sample_run()
    sample = xr.Dataset(
        {"shortwave_radiation_wattPerSquareMetre":
            (("lead", "latitude", "longitude"),
             np.full((4, 3, 3), 100.0, dtype="float32"))},
        coords={"lead": np.arange(1, 5, dtype="int32"),
                "latitude": np.linspace(24.0, 25.0, 3).astype("float32"),
                "longitude": np.linspace(121.0, 122.0, 3).astype("float32")})
    ds, enc = build_template(sample, axis, np.arange(1, 5, dtype="int32"),
                             with_marker=True)
    ds.to_zarr(out, mode="w", compute=False, consolidated=True, encoding=enc)
    with pytest.raises(SystemExit):
        prepend(out, "2026-05-11")
