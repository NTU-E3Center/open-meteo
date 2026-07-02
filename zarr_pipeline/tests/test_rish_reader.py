import numpy as np
import pandas as pd
import pytest
import xarray as xr
import convert_to_zarr as cz


def test_rish_segment_paths_from_anchor():
    p = "/x/Z__C_RJTD_20250801000000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin"
    segs = cz.rish_segment_paths(p)
    assert [s.split("_FH")[1] for s in segs] == \
        ["00-15_grib2.bin", "16-33_grib2.bin", "34-39_grib2.bin",
         "40-51_grib2.bin", "52-78_grib2.bin"]


def test_rish_regex_and_discover(tmp_path):
    f = tmp_path / "Z__C_RJTD_20250801120000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin"
    f.write_bytes(b"GRIB")
    (tmp_path / "Z__C_RJTD_20250801120000_MSM_GPV_Rjp_Lsurf_FH16-33_grib2.bin").write_bytes(b"GRIB")
    runs = cz.discover_runs(str(tmp_path), "rish")
    assert len(runs) == 1                                    # only the FH00-15 anchor counts
    ts, path = runs[0]
    assert ts == np.datetime64("2025-08-01T12:00:00")


def _fake_segment(path):
    """cfgrib-shaped stand-in: returns list of sub-datasets for one segment."""
    fh = path.split("_FH")[1].split("_")[0]                  # e.g. "00-15"
    lo, hi = (int(x) for x in fh.split("-"))
    inst_steps = np.arange(lo, hi + 1)                       # instantaneous incl. FH0
    avg_steps = np.arange(max(lo, 1), hi + 1)                # time-mean starts at FH1
    lat = np.arange(47.6, 22.39, -0.05).round(4)             # RISH native: descending, 505
    lon = np.arange(120.0, 150.01, 0.0625).round(4)
    def cube(steps, val):
        return (("step", "latitude", "longitude"),
                np.full((len(steps), len(lat), len(lon)), val, dtype="float32"))
    d_inst = xr.Dataset(
        {"t": cube(inst_steps, 300.0), "r": cube(inst_steps, 80.0),
         "u10": cube(inst_steps, 3.0), "v10": cube(inst_steps, 4.0),
         "lcc": cube(inst_steps, 10.0), "mcc": cube(inst_steps, 20.0),
         "hcc": cube(inst_steps, 30.0), "tcc_unknown": cube(inst_steps, 40.0),
         "sp": cube(inst_steps, 101325.0)},
        coords={"step": inst_steps, "latitude": lat, "longitude": lon})
    d_avg = xr.Dataset(
        {"avg_sdswrf": cube(avg_steps, 500.0)},
        coords={"step": avg_steps, "latitude": lat, "longitude": lon})
    return [d_inst, d_avg]


def test_read_run_rish_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(cz, "_open_rish_segment", _fake_segment)
    anchor = tmp_path / "Z__C_RJTD_20250801000000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin"
    for fh in ("00-15", "16-33", "34-39", "40-51", "52-78"):
        (tmp_path / f"Z__C_RJTD_20250801000000_MSM_GPV_Rjp_Lsurf_FH{fh}_grib2.bin").write_bytes(b"GRIB")
    ds = cz.read_run_rish(str(anchor))
    assert list(ds["lead"].values) == list(range(1, 79))     # contiguous 1..78, FH0 dropped
    assert ds.sizes["latitude"] == 473                        # positional crop: 473 rows
    assert ds.sizes["longitude"] == 481                       # full longitude extent
    assert abs(float(ds.latitude.max()) - 46.0) < 1e-3       # northmost row ≈ 46.0
    assert ds.latitude.values[0] < ds.latitude.values[-1]    # ascending
    assert ds["temperature_2m_celsius"].dtype == np.float32
    np.testing.assert_allclose(
        ds["temperature_2m_celsius"].isel(lead=0).values, 300.0 - 273.15, atol=1e-4)
    np.testing.assert_allclose(
        ds["wind_speed_10m_metrePerSecond"].isel(lead=0).values, 5.0, atol=1e-4)  # 3-4-5
    np.testing.assert_allclose(
        ds["surface_pressure_hectopascal"].isel(lead=0).values, 1013.25, atol=1e-2)
    assert float(ds["shortwave_radiation_wattPerSquareMetre"].isel(lead=0).mean()) == pytest.approx(500.0, rel=1e-5)
    assert all(ds[v].dtype == np.float32 for v in ds.data_vars)
    expected = {"shortwave_radiation_wattPerSquareMetre", "temperature_2m_celsius",
                "relative_humidity_2m_percentage", "wind_speed_10m_metrePerSecond",
                "cloud_cover_low_percentage", "cloud_cover_mid_percentage",
                "cloud_cover_high_percentage", "cloud_cover_percentage",
                "surface_pressure_hectopascal"}
    assert expected <= set(ds.data_vars)


def test_read_run_rish_tolerates_missing_extension_segments(monkeypatch, tmp_path):
    monkeypatch.setattr(cz, "_open_rish_segment", _fake_segment)
    anchor = tmp_path / "Z__C_RJTD_20250801000000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin"
    for fh in ("00-15", "16-33", "34-39"):                    # 39h-era run
        (tmp_path / f"Z__C_RJTD_20250801000000_MSM_GPV_Rjp_Lsurf_FH{fh}_grib2.bin").write_bytes(b"GRIB")
    ds = cz.read_run_rish(str(anchor))
    assert int(ds["lead"].max()) == 39
