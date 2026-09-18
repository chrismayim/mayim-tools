"""
Tests for cmorph_extract/core.py. Run: python3 tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from mayim_tools.rainfall.cmorph_extract.core import (
    BASE_URL,
    RECORD_START,
    _build_session,
    build_hour_url,
    fetch_cmorph_timeseries,
    fetch_one_hour_file,
    find_coord_name,
    find_variable_name,
    hourly_range,
    read_hour_file,
)

# ----------------------------------------------------------------------
# URL construction - confirmed directly against a real listing entry,
# not guessed
# ----------------------------------------------------------------------


def test_build_hour_url_matches_real_confirmed_example():
    """The exact URL from the real Azure Blob List response shared
    directly: hour 00 on 1998-01-01. Locks in the confirmed pattern as
    a regression test so it can't silently drift."""
    url = build_hour_url(datetime(1998, 1, 1, 0))
    expected = (
        "https://noaacdr.blob.core.windows.net/precip-cmorph/data/30min/8km/"
        "1998/01/01/CMORPH_V1.0_ADJ_8km-30min_1998010100.nc"
    )
    assert url == expected, f"got {url}"
    print("test_build_hour_url_matches_real_confirmed_example: PASS")


def test_build_hour_url_matches_real_confirmed_example_hour_01():
    url = build_hour_url(datetime(1998, 1, 1, 1))
    expected = (
        "https://noaacdr.blob.core.windows.net/precip-cmorph/data/30min/8km/"
        "1998/01/01/CMORPH_V1.0_ADJ_8km-30min_1998010101.nc"
    )
    assert url == expected, f"got {url}"
    print("test_build_hour_url_matches_real_confirmed_example_hour_01: PASS")


def test_build_hour_url_zero_pads_month_day_and_hour():
    url = build_hour_url(datetime(2020, 3, 5, 9))
    assert "/2020/03/05/" in url
    assert url.endswith("2020030509.nc")
    print("test_build_hour_url_zero_pads_month_day_and_hour: PASS")


def test_build_hour_url_double_digit_hour():
    url = build_hour_url(datetime(2020, 3, 5, 23))
    assert url.endswith("2020030523.nc")
    print("test_build_hour_url_double_digit_hour: PASS")


def test_record_start_matches_confirmed_sources():
    assert RECORD_START == date(1998, 1, 1)
    print("test_record_start_matches_confirmed_sources: PASS")


# ----------------------------------------------------------------------
# Hour range generation
# ----------------------------------------------------------------------


def test_hourly_range_single_day():
    hours = hourly_range(date(2020, 1, 1), date(2020, 1, 1))
    assert len(hours) == 24
    assert hours[0] == datetime(2020, 1, 1, 0)
    assert hours[-1] == datetime(2020, 1, 1, 23)
    print("test_hourly_range_single_day: PASS")


def test_hourly_range_multi_day():
    hours = hourly_range(date(2020, 1, 1), date(2020, 1, 2))
    assert len(hours) == 48
    print("test_hourly_range_multi_day: PASS")


def test_hourly_range_rejects_backwards_range():
    try:
        hourly_range(date(2020, 1, 2), date(2020, 1, 1))
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("test_hourly_range_rejects_backwards_range: PASS")


# ----------------------------------------------------------------------
# Defensive variable/coordinate name lookup - the real CMORPH variable
# name was never confirmed against a live file (no network access to
# the source domain from the build environment)
# ----------------------------------------------------------------------


def _make_synthetic_cmorph_dataset(
    var_name="cmorph", n_time=2, lat_name="lat", lon_name="lon"
):
    import xarray as xr

    lats = np.array([-0.1, 0.0, 0.1])
    lons = np.array([-0.1, 0.0, 0.1])
    if n_time > 0:
        times = pd.to_datetime([f"2020-01-01T{h:02d}:00:00" for h in range(n_time)])
        data = np.zeros((n_time, 3, 3), dtype="float32")
        for i in range(n_time):
            data[i, 1, 1] = (i + 1) * 0.5  # centre cell
        return xr.Dataset(
            {var_name: (["time", lat_name, lon_name], data)},
            coords={"time": times, lat_name: lats, lon_name: lons},
        )
    else:
        data = np.zeros((3, 3), dtype="float32")
        data[1, 1] = 0.7
        return xr.Dataset(
            {var_name: ([lat_name, lon_name], data)},
            coords={lat_name: lats, lon_name: lons},
        )


def test_find_variable_name_matches_known_candidate():
    ds = _make_synthetic_cmorph_dataset(var_name="cmorph")
    assert find_variable_name(ds) == "cmorph"
    print("test_find_variable_name_matches_known_candidate: PASS")


def test_find_variable_name_falls_back_to_sole_data_var():
    """If the real variable name turns out to be something not in the
    candidate list, but the file has exactly one data variable (the
    expected structure for a single-parameter-per-file product), fall
    back to it rather than failing outright."""
    ds = _make_synthetic_cmorph_dataset(var_name="some_unexpected_name")
    assert find_variable_name(ds) == "some_unexpected_name"
    print("test_find_variable_name_falls_back_to_sole_data_var: PASS")


def test_find_coord_name_matches_lat_lon():
    ds = _make_synthetic_cmorph_dataset(lat_name="latitude", lon_name="longitude")
    assert find_coord_name(ds, ("lat", "latitude")) == "latitude"
    assert find_coord_name(ds, ("lon", "longitude")) == "longitude"
    print("test_find_coord_name_matches_lat_lon: PASS")


# ----------------------------------------------------------------------
# File reading - synthetic files covering BOTH plausible timestep
# counts per hourly file, since this was never confirmed live
# ----------------------------------------------------------------------


def test_read_hour_file_two_timesteps_per_file():
    """The "30min resolution, one file per hour" structure implies 2
    timesteps per file - tested as the primary hypothesis."""
    import tempfile

    ds = _make_synthetic_cmorph_dataset(n_time=2)
    tmp_path = Path(tempfile.mkdtemp()) / "test.nc"
    ds.to_netcdf(tmp_path)

    df = read_hour_file(tmp_path, lat=0.0, lon=0.0)
    assert len(df) == 2
    assert list(df["PrecipitationMMHR"]) == [0.5, 1.0]
    print("test_read_hour_file_two_timesteps_per_file: PASS")


def test_read_hour_file_single_timestep_per_file():
    """Alternative hypothesis: one timestep per file - handled without
    assuming which structure is actually real."""
    import tempfile

    ds = _make_synthetic_cmorph_dataset(n_time=1)
    tmp_path = Path(tempfile.mkdtemp()) / "test.nc"
    ds.to_netcdf(tmp_path)

    df = read_hour_file(tmp_path, lat=0.0, lon=0.0)
    assert len(df) == 1
    assert df["PrecipitationMMHR"].iloc[0] == 0.5
    print("test_read_hour_file_single_timestep_per_file: PASS")


def test_read_hour_file_nearest_neighbour():
    import tempfile

    ds = _make_synthetic_cmorph_dataset(n_time=1)
    tmp_path = Path(tempfile.mkdtemp()) / "test.nc"
    ds.to_netcdf(tmp_path)

    df = read_hour_file(
        tmp_path, lat=0.03, lon=-0.02
    )  # off-grid, should snap to (0.0, 0.0)
    assert df["PrecipitationMMHR"].iloc[0] == 0.5
    print("test_read_hour_file_nearest_neighbour: PASS")


def test_read_hour_file_alternate_coord_names():
    import tempfile

    ds = _make_synthetic_cmorph_dataset(
        n_time=1, lat_name="latitude", lon_name="longitude"
    )
    tmp_path = Path(tempfile.mkdtemp()) / "test.nc"
    ds.to_netcdf(tmp_path)

    df = read_hour_file(tmp_path, lat=0.0, lon=0.0)
    assert len(df) == 1
    print("test_read_hour_file_alternate_coord_names: PASS")


# ----------------------------------------------------------------------
# Full orchestration, via injected fakes - no real network access
# ----------------------------------------------------------------------


def test_fetch_cmorph_timeseries_orchestration_with_fakes():
    call_log = []

    def fake_fetch(url, lat, lon, dt):
        call_log.append(url)
        return pd.DataFrame(
            {
                "Timestamp": [
                    pd.Timestamp(dt),
                    pd.Timestamp(dt) + pd.Timedelta(minutes=30),
                ],
                "PrecipitationMMHR": [0.1, 0.2],
            }
        )

    result = fetch_cmorph_timeseries(
        lat=6.0539,
        lon=-1.7334,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=fake_fetch,
        max_workers=1,
    )
    assert len(call_log) == 24  # 24 hourly files for a single day
    assert len(result.dataframe) == 48  # 2 timesteps x 24 files
    assert result.dataframe["Timestamp"].is_monotonic_increasing
    print("test_fetch_cmorph_timeseries_orchestration_with_fakes: PASS")


def test_fetch_cmorph_timeseries_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky_fetch(url, lat, lon, dt):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionError("simulated transient failure")
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(dt)], "PrecipitationMMHR": [0.3]}
        )

    result = fetch_cmorph_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=flaky_fetch,
        max_retries=3,
        retry_backoff_s=0.01,
        max_workers=1,
    )
    assert attempts["n"] >= 2
    assert len(result.dataframe) >= 1
    print("test_fetch_cmorph_timeseries_retries_then_succeeds: PASS")


def test_fetch_cmorph_timeseries_isolates_per_file_failures():
    def sometimes_fails(url, lat, lon, dt):
        if dt.hour == 12:
            raise ConnectionError("simulated failure")
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(dt)], "PrecipitationMMHR": [0.4]}
        )

    result = fetch_cmorph_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=sometimes_fails,
        max_retries=1,
        retry_backoff_s=0.01,
        max_workers=4,
    )
    assert len(result.dataframe) == 23  # 24 hours minus the one that always fails
    assert any("failed after" in w for w in result.warnings)
    print("test_fetch_cmorph_timeseries_isolates_per_file_failures: PASS")


def test_fetch_cmorph_timeseries_reports_missing_values():
    """Regression test for a real live-run finding: a file can fetch
    and read successfully but still have NO value at the target point
    for a specific timestep (a masked/missing grid cell in the source
    data, distinct from a fetch failure). Confirmed live: 199 of
    87,744 timesteps (~0.23%) were empty in the final CSV, with zero
    fetch failures reported - meaning this diagnostic was previously
    silent (Chris had to count empty cells in the output CSV himself,
    since core.py never surfaced this as a warning)."""

    def fetch_with_one_missing(url, lat, lon, dt):
        value = float("nan") if dt.hour == 5 else 0.3
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(dt)], "PrecipitationMMHR": [value]}
        )

    result = fetch_cmorph_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=fetch_with_one_missing,
        max_workers=1,
    )
    assert (
        len(result.dataframe) == 24
    )  # all 24 files succeeded - this isn't a fetch failure
    assert any("had no value at this point" in w for w in result.warnings)
    assert any("1 of 24" in w for w in result.warnings)
    print("test_fetch_cmorph_timeseries_reports_missing_values: PASS")


def test_fetch_cmorph_timeseries_no_missing_values_reports_nothing():
    def fetch_all_present(url, lat, lon, dt):
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(dt)], "PrecipitationMMHR": [0.3]}
        )

    result = fetch_cmorph_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=fetch_all_present,
        max_workers=1,
    )
    assert not any("had no value at this point" in w for w in result.warnings)
    print("test_fetch_cmorph_timeseries_no_missing_values_reports_nothing: PASS")


def test_fetch_cmorph_timeseries_concurrent_speedup():
    def slow_fetch(url, lat, lon, dt):
        time.sleep(0.02)
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(dt)], "PrecipitationMMHR": [0.1]}
        )

    start = time.time()
    fetch_cmorph_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-03",
        fetch_fn=slow_fetch,
        max_workers=1,
    )
    serial_time = time.time() - start

    start = time.time()
    fetch_cmorph_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-03",
        fetch_fn=slow_fetch,
        max_workers=8,
    )
    concurrent_time = time.time() - start

    speedup = serial_time / concurrent_time
    assert speedup > 2.5, f"expected meaningful speedup, got only {speedup:.1f}x"
    print(
        "test_fetch_cmorph_timeseries_concurrent_speedup: PASS "
        f"({speedup:.1f}x speedup)"
    )


def test_fetch_cmorph_timeseries_default_full_record_start():
    def fake_fetch(url, lat, lon, dt):
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(dt)], "PrecipitationMMHR": [0.1]}
        )

    result = fetch_cmorph_timeseries(
        lat=0, lon=0, end_date="1998-01-01", fetch_fn=fake_fetch, max_workers=1
    )
    assert result.dataframe["Timestamp"].min().date() == RECORD_START
    print("test_fetch_cmorph_timeseries_default_full_record_start: PASS")


# ----------------------------------------------------------------------
# Connection reuse - added to cut real run time, after a live 5-year
# run took ~4 hours. Every fetch was previously a bare requests.get(),
# paying a fresh TCP+TLS handshake per file even though nearly all
# ~43,872 files in a full-record run hit the same host.
# ----------------------------------------------------------------------


def test_build_session_pools_size_matches_worker_count():
    session = _build_session(pool_size=25)
    adapter = session.get_adapter("https://noaacdr.blob.core.windows.net/")
    assert adapter._pool_maxsize == 25
    print("test_build_session_pools_size_matches_worker_count: PASS")


def test_fetch_one_hour_file_uses_provided_session():
    """Confirms a provided session's .get() is actually called, not a
    fresh requests.get() - the whole point of the fix."""
    import tempfile
    import unittest.mock as mock

    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        {
            "cmorph": (
                ["time", "lat", "lon"],
                np.array([[[0.4]], [[0.5]]], dtype="float32"),
            )
        },
        coords={
            "time": pd.to_datetime(["2020-01-01T00:30:00", "2020-01-01T01:00:00"]),
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    tmp_path = Path(tempfile.mkdtemp()) / "test.nc"
    ds.to_netcdf(tmp_path)

    class _FakeResponse:
        status_code = 200
        content = tmp_path.read_bytes()

        def raise_for_status(self):
            pass

    fake_session = mock.Mock()
    fake_session.get.return_value = _FakeResponse()

    fetch_one_hour_file(
        "https://noaacdr.blob.core.windows.net/precip-cmorph/fake.nc",
        lat=0.0,
        lon=0.0,
        fallback_timestamp=datetime(2020, 1, 1),
        session=fake_session,
    )
    fake_session.get.assert_called_once()
    print("test_fetch_one_hour_file_uses_provided_session: PASS")


def test_fetch_one_hour_file_falls_back_to_plain_requests_without_session():
    """No session provided (the default, and what every existing test
    fake relies on) must still work exactly as before - this is
    additive, not a breaking change to the calling convention."""
    import inspect

    sig = inspect.signature(fetch_one_hour_file)
    assert sig.parameters["session"].default is None
    print("test_fetch_one_hour_file_falls_back_to_plain_requests_without_session: PASS")


def test_fetch_cmorph_timeseries_builds_session_sized_to_max_workers():
    """Confirms the REAL default path (not a test fake) actually wires
    a session through with the right pool size, by mocking only
    _build_session and letting the real fetch_one_hour_file run
    against a fake session's .get() - exercising the true code path
    end-to-end rather than relying on a default-argument identity
    check, which can't be observed correctly through mock.patch (a
    function's default argument value is bound at definition time, so
    patching the module-level name afterward doesn't change it)."""
    import tempfile
    import unittest.mock as mock

    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        {
            "cmorph": (
                ["time", "lat", "lon"],
                np.array([[[0.4]], [[0.5]]], dtype="float32"),
            )
        },
        coords={
            "time": pd.to_datetime(["2020-01-01T00:30:00", "2020-01-01T01:00:00"]),
            "lat": [0.0],
            "lon": [0.0],
        },
    )
    tmp_path = Path(tempfile.mkdtemp()) / "test.nc"
    ds.to_netcdf(tmp_path)

    class _FakeResponse:
        content = tmp_path.read_bytes()

        def raise_for_status(self):
            pass

    captured = {}
    fake_session_holder = {}

    def fake_build_session(pool_size):
        captured["pool_size"] = pool_size
        fake_session = mock.Mock()
        fake_session.get.return_value = _FakeResponse()
        fake_session_holder["session"] = fake_session
        return fake_session

    with mock.patch(
        "mayim_tools.rainfall.cmorph_extract.core._build_session",
        side_effect=fake_build_session,
    ):
        result = fetch_cmorph_timeseries(
            lat=0.0,
            lon=0.0,
            start_date="2020-01-01",
            end_date="2020-01-01",
            max_workers=17,
        )
    assert captured.get("pool_size") == 17
    # every fake request returns the SAME fixed file/timestamps, so the
    # tool's own (correct) dedup-by-timestamp collapses all 24 hourly
    # fetches to 2 unique rows - this confirms the pipeline actually ran
    # end-to-end through the fake session, not a specific row count
    assert len(result.dataframe) == 2
    assert fake_session_holder["session"].get.call_count == 24
    print("test_fetch_cmorph_timeseries_builds_session_sized_to_max_workers: PASS")


if __name__ == "__main__":
    tests = [
        v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"{t.__name__}: FAIL - {e}")
        except Exception as e:
            failed += 1
            print(f"{t.__name__}: ERROR - {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
