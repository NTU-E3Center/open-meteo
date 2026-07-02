# zarr_pipeline/tests/conftest.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from convert_to_zarr import build_template, _write_run


def _sample_run(n_lead=4, ny=3, nx=3, val=100.0):
    """One-run dataset in the reader-output contract: (lead, lat, lon), int lead coord."""
    return xr.Dataset(
        {"shortwave_radiation_wattPerSquareMetre":
            (("lead", "latitude", "longitude"),
             np.full((n_lead, ny, nx), val, dtype="float32"))},
        coords={"lead": np.arange(1, n_lead + 1, dtype="int32"),
                "latitude": np.linspace(24.0, 25.0, ny).astype("float32"),
                "longitude": np.linspace(121.0, 122.0, nx).astype("float32")})


@pytest.fixture
def mini_cube(tmp_path):
    """Pre-allocated cube: axis 2026-05-12..+5 slots step 3h; slots 0 and 2 filled
    (values 100 and 300). Returns (path, axis)."""
    out = str(tmp_path / "mini.zarr")
    axis = np.arange(np.datetime64("2026-05-12T00"), np.datetime64("2026-05-12T15"),
                     np.timedelta64(3, "h")).astype("datetime64[ns]")
    sample = _sample_run()
    ds, enc = build_template(sample, axis, np.arange(1, 5, dtype="int32"), with_marker=True)
    ds.to_zarr(out, mode="w", compute=False, consolidated=True, encoding=enc)
    _write_run(out, _sample_run(val=100.0), 0, 1, 3, 3, mark=True)
    _write_run(out, _sample_run(val=300.0), 2, 1, 3, 3, mark=True)
    return out, axis
