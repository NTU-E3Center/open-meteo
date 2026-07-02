# zarr_pipeline/tests/test_prepend.py
import numpy as np
import pandas as pd
import xarray as xr
from prepend_run_init import prepend

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
