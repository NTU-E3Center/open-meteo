# zarr_pipeline/tests/test_himawari_cube.py
"""Tests for himawari_cube.py — TDD: written before implementation (RED phase).

Synthetic tiny grid (10 lat × 12 lon) so tests run quickly. We verify:
- create_store: dims, coords, dtypes, chunk shapes, codec (zstd Blosc), fill values
- write_day:  bit-exact float32 read-back, slot_filled set to 1
- write_day rejections: wrong grid, ≠24 hours, non-aligned day
- idempotent re-write (same-day overwrite stays bit-exact)
- day_filled: true/false paths
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from himawari_cube import create_store, day_filled, write_day

# --------------------------------------------------------------------------- #
# Tiny synthetic grid / helpers
# --------------------------------------------------------------------------- #
NY, NX = 10, 12
LAT = np.linspace(-44.0, 46.0, NY, dtype="float32")
LON = np.linspace(92.0, 154.0, NX, dtype="float32")

TIME_START = "2015-07-07"
TIME_END   = "2015-07-12"   # 5-day window keeps the store tiny

DAY = "2015-07-08"          # a day within the window


def _make_day_ds(day: str, lat=LAT, lon=LON, seed: int = 42) -> xr.Dataset:
    """Synthetic 24-hour observation dataset matching the source layout."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(f"{day}T00", periods=24, freq="h")
    data = rng.random((24, len(lat), len(lon)), dtype=None).astype("float32") * 1000.0
    ds = xr.Dataset(
        {"SWR": (("time", "latitude", "longitude"), data)},
        coords={
            "time":      ("time",      times.to_numpy(dtype="datetime64[ns]")),
            "latitude":  ("latitude",  lat.copy()),
            "longitude": ("longitude", lon.copy()),
        },
    )
    return ds


# --------------------------------------------------------------------------- #
# create_store
# --------------------------------------------------------------------------- #
class TestCreateStore:
    def test_store_created_on_disk(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        assert os.path.isdir(p)

    def test_dims_and_sizes(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        ds = xr.open_zarr(p, consolidated=True)
        try:
            assert "time"      in ds.dims
            assert "latitude"  in ds.dims
            assert "longitude" in ds.dims
            n_hours = int(
                (np.datetime64(TIME_END) - np.datetime64(TIME_START))
                / np.timedelta64(1, "h")
            )
            assert ds.sizes["time"]      == n_hours
            assert ds.sizes["latitude"]  == NY
            assert ds.sizes["longitude"] == NX
        finally:
            ds.close()

    def test_coords_exact(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        ds = xr.open_zarr(p, consolidated=True)
        try:
            np.testing.assert_array_equal(ds["latitude"].values, LAT)
            np.testing.assert_array_equal(ds["longitude"].values, LON)
        finally:
            ds.close()

    def test_time_axis_hourly(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        ds = xr.open_zarr(p, consolidated=True)
        try:
            t = ds["time"].values
            assert t[0]  == np.datetime64(f"{TIME_START}T00", "ns")
            diffs = np.diff(t).astype("timedelta64[h]").astype(int)
            assert (diffs == 1).all(), "time axis is not strictly hourly"
        finally:
            ds.close()

    def test_swr_dtype_float32(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        ds = xr.open_zarr(p, consolidated=True)
        try:
            assert ds["SWR"].dtype == np.float32
        finally:
            ds.close()

    def test_slot_filled_dtype_int8(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        ds = xr.open_zarr(p, consolidated=True)
        try:
            assert ds["slot_filled"].dtype == np.int8
        finally:
            ds.close()

    def test_swr_fill_is_nan(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        ds = xr.open_zarr(p, consolidated=True)
        try:
            assert np.isnan(ds["SWR"].values).all(), "SWR initial fill should be NaN"
        finally:
            ds.close()

    def test_slot_filled_initial_zeros(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        ds = xr.open_zarr(p, consolidated=True)
        try:
            assert (ds["slot_filled"].values == 0).all()
        finally:
            ds.close()

    def test_slot_filled_chunk_equals_chunk_hours(self, tmp_path):
        """slot_filled chunk along time must == chunk_hours (load-bearing invariant)."""
        p = str(tmp_path / "silver.zarr")
        chunk_hours = 24
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END,
                     chunk_hours=chunk_hours)
        g = zarr.open_group(p, mode="r")
        sf_chunks = g["slot_filled"].chunks
        assert sf_chunks[0] == chunk_hours, (
            f"slot_filled chunk[0]={sf_chunks[0]} != chunk_hours={chunk_hours}"
        )

    def test_swr_chunk_time_axis_equals_chunk_hours(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        chunk_hours = 24
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END,
                     chunk_hours=chunk_hours)
        g = zarr.open_group(p, mode="r")
        swr_chunks = g["SWR"].chunks
        assert swr_chunks[0] == chunk_hours

    def test_swr_has_blosc_compressor(self, tmp_path):
        """SWR must be compressed (Blosc/zstd); uncompressed cube is a defect."""
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        g = zarr.open_group(p, mode="r")
        codecs = g["SWR"].metadata.codecs
        codec_names = [type(c).__name__ for c in codecs]
        assert any("Blosc" in n for n in codec_names), (
            f"SWR compressor not Blosc; got: {codec_names}"
        )

    def test_idempotent_refuses_existing_path(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        with pytest.raises((FileExistsError, ValueError)):
            create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)


# --------------------------------------------------------------------------- #
# write_day
# --------------------------------------------------------------------------- #
class TestWriteDay:
    def _store(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        return p

    def test_returns_24(self, tmp_path):
        p = self._store(tmp_path)
        ds = _make_day_ds(DAY)
        result = write_day(p, ds)
        assert result == 24

    def test_swr_bit_exact_readback(self, tmp_path):
        """float32 round-trip through codec must be bit-exact (lossless compression)."""
        p = self._store(tmp_path)
        day_ds = _make_day_ds(DAY)
        written_data = day_ds["SWR"].values.copy()
        write_day(p, day_ds)

        store = xr.open_zarr(p, consolidated=True)
        try:
            # select the 24 hours for DAY
            t0 = np.datetime64(f"{DAY}T00", "ns")
            t23 = np.datetime64(f"{DAY}T23", "ns")
            read_back = store["SWR"].sel(time=slice(t0, t23)).values
            # Strict float32 equality — NOT np.testing.assert_allclose
            np.testing.assert_array_equal(read_back, written_data)
        finally:
            store.close()

    def test_slot_filled_set_to_1_after_write(self, tmp_path):
        p = self._store(tmp_path)
        write_day(p, _make_day_ds(DAY))
        store = xr.open_zarr(p, consolidated=True)
        try:
            t0 = np.datetime64(f"{DAY}T00", "ns")
            t23 = np.datetime64(f"{DAY}T23", "ns")
            sf = store["slot_filled"].sel(time=slice(t0, t23)).values
            assert (sf == 1).all(), f"slot_filled for {DAY} not all 1; got {sf}"
        finally:
            store.close()

    def test_other_days_stay_unfilled(self, tmp_path):
        p = self._store(tmp_path)
        write_day(p, _make_day_ds(DAY))
        store = xr.open_zarr(p, consolidated=True)
        try:
            other_day = "2015-07-09"
            t0 = np.datetime64(f"{other_day}T00", "ns")
            t23 = np.datetime64(f"{other_day}T23", "ns")
            sf = store["slot_filled"].sel(time=slice(t0, t23)).values
            assert (sf == 0).all()
        finally:
            store.close()

    # --- rejection paths ---

    def test_rejects_wrong_lat(self, tmp_path):
        p = self._store(tmp_path)
        bad_lat = LAT + 1.0
        day_ds = _make_day_ds(DAY, lat=bad_lat.astype("float32"))
        with pytest.raises(ValueError, match="[Ll]at"):
            write_day(p, day_ds)

    def test_rejects_wrong_lon(self, tmp_path):
        p = self._store(tmp_path)
        bad_lon = LON + 1.0
        day_ds = _make_day_ds(DAY, lon=bad_lon.astype("float32"))
        with pytest.raises(ValueError, match="[Ll]on"):
            write_day(p, day_ds)

    def test_rejects_not_24_hours(self, tmp_path):
        p = self._store(tmp_path)
        times = pd.date_range(f"{DAY}T00", periods=23, freq="h")
        data = np.zeros((23, NY, NX), dtype="float32")
        ds = xr.Dataset(
            {"SWR": (("time", "latitude", "longitude"), data)},
            coords={
                "time":      times.to_numpy(dtype="datetime64[ns]"),
                "latitude":  LAT,
                "longitude": LON,
            },
        )
        with pytest.raises(ValueError, match="24"):
            write_day(p, ds)

    def test_rejects_non_aligned_day(self, tmp_path):
        """Day starting at 01Z instead of 00Z should be rejected."""
        p = self._store(tmp_path)
        times = pd.date_range(f"{DAY}T01", periods=24, freq="h")
        data = np.zeros((24, NY, NX), dtype="float32")
        ds = xr.Dataset(
            {"SWR": (("time", "latitude", "longitude"), data)},
            coords={
                "time":      times.to_numpy(dtype="datetime64[ns]"),
                "latitude":  LAT,
                "longitude": LON,
            },
        )
        with pytest.raises(ValueError, match="[Aa]lign|00:00|[Hh]our"):
            write_day(p, ds)

    # --- idempotent re-write ---

    def test_idempotent_overwrite_bit_exact(self, tmp_path):
        """Writing the same day twice keeps the second write's data bit-exactly."""
        p = self._store(tmp_path)
        ds1 = _make_day_ds(DAY, seed=1)
        ds2 = _make_day_ds(DAY, seed=2)
        write_day(p, ds1)
        write_day(p, ds2)    # overwrite

        t0 = np.datetime64(f"{DAY}T00", "ns")
        t23 = np.datetime64(f"{DAY}T23", "ns")
        store = xr.open_zarr(p, consolidated=True)
        try:
            read_back = store["SWR"].sel(time=slice(t0, t23)).values
            np.testing.assert_array_equal(read_back, ds2["SWR"].values)
        finally:
            store.close()


# --------------------------------------------------------------------------- #
# day_filled
# --------------------------------------------------------------------------- #
class TestDayFilled:
    def _store(self, tmp_path):
        p = str(tmp_path / "silver.zarr")
        create_store(p, lat=LAT, lon=LON, time_start=TIME_START, time_end=TIME_END)
        return p

    def test_false_before_write(self, tmp_path):
        p = self._store(tmp_path)
        assert day_filled(p, DAY) is False

    def test_true_after_write(self, tmp_path):
        p = self._store(tmp_path)
        write_day(p, _make_day_ds(DAY))
        assert day_filled(p, DAY) is True

    def test_accepts_dataset_object(self, tmp_path):
        """day_filled should accept an already-open xr.Dataset."""
        p = self._store(tmp_path)
        write_day(p, _make_day_ds(DAY))
        ds = xr.open_zarr(p, consolidated=True)
        try:
            assert day_filled(ds, DAY) is True
        finally:
            ds.close()

    def test_partial_fill_is_false(self, tmp_path):
        """If only some slots in the day are 1 (shouldn't happen normally, but be safe)."""
        p = self._store(tmp_path)
        # Manually set only 12 of 24 slots in DAY via zarr raw write
        g = zarr.open_group(p, mode="r+")
        ds_tmp = xr.open_zarr(p, consolidated=True)
        t_axis = ds_tmp["time"].values
        ds_tmp.close()
        day_start = np.datetime64(f"{DAY}T00", "ns")
        idx = np.where(t_axis == day_start)[0][0]
        sf = g["slot_filled"]
        arr = sf[:]
        arr[idx:idx+12] = 1
        sf[:] = arr
        assert day_filled(p, DAY) is False
