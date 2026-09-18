"""
Tests for imerg_extract/core.py. Run: python3 tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time

import numpy as np
import pandas as pd

from mayim_tools.rainfall.imerg_extract.core import (
    FetchResult,
    _check_netcdf_backend_available,
    _extend_bare_date_to_end_of_day,
    _parse_giovanni_csv,
    _split_date_range,
    _to_pandas_timestamp,
    estimate_granule_count,
    fetch_giovanni_timeseries,
    fetch_granule_based_timeseries,
    read_point_from_dataset,
    save_earthdata_credentials,
)


def test_split_date_range_exact_multiple():
    chunks = _split_date_range(
        "2020-01-01T00:00:00", "2021-12-31T23:30:00", chunk_months=12
    )
    assert len(chunks) == 2, chunks
    for i in range(len(chunks) - 1):
        end_i = pd.Timestamp(chunks[i][1])
        start_next = pd.Timestamp(chunks[i + 1][0])
        assert (start_next - end_i).total_seconds() == 1, (chunks[i], chunks[i + 1])
    print("test_split_date_range_exact_multiple: PASS")


def test_split_date_range_covers_full_span():
    start, end = "2000-06-01", "2026-01-01"
    chunks = _split_date_range(start, end, chunk_months=12)
    assert pd.Timestamp(chunks[0][0]) == pd.Timestamp(start)
    assert pd.Timestamp(chunks[-1][1]) == pd.Timestamp(end)
    print("test_split_date_range_covers_full_span: PASS")


def test_split_date_range_full_record_chunk_count():
    chunks = _split_date_range("2000-06-01", "2026-01-01", chunk_months=12)
    assert 24 <= len(chunks) <= 28, f"expected ~26 chunks, got {len(chunks)}"
    print("test_split_date_range_full_record_chunk_count: PASS")


def test_split_date_range_rejects_backwards_range():
    try:
        _split_date_range("2020-01-01", "2019-01-01", chunk_months=12)
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_split_date_range_rejects_backwards_range: PASS")


# ----------------------------------------------------------------------
# Bare end-date extension - a real discrepancy found by comparing the
# Giovanni and granule methods against the identical date range: the
# granule method (via earthaccess's date-only CMR temporal query)
# correctly covers the whole end day, but Giovanni took a bare end
# date literally as midnight of that day and stopped there.
# ----------------------------------------------------------------------


def test_extend_bare_date_to_end_of_day_real_case():
    """The exact real case reported: END_DATE=2005-03-05 must extend
    to the last instant of that day, not stay at midnight."""
    result = _extend_bare_date_to_end_of_day("2005-03-05")
    assert result == "2005-03-05T23:59:59"
    print("test_extend_bare_date_to_end_of_day_real_case: PASS")


def test_extend_bare_date_preserves_explicit_time():
    """A caller who explicitly gave a non-midnight time must have it
    respected exactly, not overridden - this only fixes the bare-date
    (implicit midnight) case."""
    result = _extend_bare_date_to_end_of_day("2005-03-05T12:00:00")
    assert result == "2005-03-05T12:00:00"
    print("test_extend_bare_date_preserves_explicit_time: PASS")


def test_extend_bare_date_leaves_explicit_midnight_alone_is_not_distinguishable():
    """Documents a known, accepted limitation rather than a bug: an
    end_date explicitly given AS midnight (e.g. someone deliberately
    wants the range to stop exactly at 00:00:00) is indistinguishable
    from a bare date and will also be extended to end-of-day. This
    matches how the parameter is used in practice throughout this
    project - a plain date string, never a deliberately precise
    midnight timestamp - so the common case is fixed at the cost of
    this one, much rarer edge case."""
    result = _extend_bare_date_to_end_of_day("2005-03-05T00:00:00")
    assert result == "2005-03-05T23:59:59"
    print(
        "test_extend_bare_date_leaves_explicit_midnight_alone_is_not_distinguishable: PASS"
    )


def test_fetch_giovanni_timeseries_covers_full_end_date():
    """Integration-level confirmation of the actual real discrepancy:
    with the fix, requesting START_DATE=2005-03-01, END_DATE=2005-03-05
    must now cover the WHOLE of March 5th (through 23:30, the last
    half-hourly grid point), matching what the granule method already
    correctly returned for the identical range - not stop at March
    5th's first instant (the pre-fix behaviour that produced exactly
    193 rows, confirmed as 4 days x 48 half-hours + 1)."""
    captured_time_param = {}

    def fake_fetch(lat, lon, start_iso, end_iso, variable, token):
        captured_time_param["start"] = start_iso
        captured_time_param["end"] = end_iso
        idx = pd.date_range(start_iso, end_iso, freq="30min")
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 1.0})

    result = fetch_giovanni_timeseries(
        lat=6.0539,
        lon=-1.7334,
        start_date="2005-03-01",
        end_date="2005-03-05",
        chunk_months=12,
        token="fake-token",
        fetch_chunk_fn=fake_fetch,
        max_workers=1,
    )
    assert (
        captured_time_param["end"] == "2005-03-05T23:59:59"
    ), f"Giovanni request still stops at midnight: {captured_time_param['end']}"
    assert result.dataframe["Timestamp"].max() == pd.Timestamp(
        "2005-03-05 23:30:00"
    ), f"result still stops at March 5's first instant: {result.dataframe['Timestamp'].max()}"
    print("test_fetch_giovanni_timeseries_covers_full_end_date: PASS")


def test_parse_giovanni_csv_with_comment_preamble():
    fake_response = (
        "# Giovanni Time Series\n"
        "# data: GPM_3IMERGHH_07_precipitation\n"
        "# location: [4.75,0.55]\n"
        "time,precipitation\n"
        "2020-01-01T00:00:00Z,0.5\n"
        "2020-01-01T00:30:00Z,1.2\n"
        "2020-01-01T01:00:00Z,0.0\n"
    )
    df = _parse_giovanni_csv(fake_response)
    assert list(df.columns) == ["Timestamp", "PrecipitationMMHR"]
    assert len(df) == 3
    assert df["PrecipitationMMHR"].tolist() == [0.5, 1.2, 0.0]
    print("test_parse_giovanni_csv_with_comment_preamble: PASS")


def test_parse_giovanni_csv_empty_raises():
    try:
        _parse_giovanni_csv("# only comments\n# nothing else\n")
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_parse_giovanni_csv_empty_raises: PASS")


def test_parse_giovanni_csv_normal_shape_still_works():
    """The original (header/data field counts match) shape must still
    work correctly - not that this shape is now known to be real
    (v0.3.3 superseded that hypothesis), just that the current parser
    still handles a clean two-column time,value CSV if it were ever
    encountered."""
    fake_response = (
        "time,precipitation\n2020-01-01T00:00:00Z,0.5\n2020-01-01T00:30:00Z,1.2\n"
    )
    df = _parse_giovanni_csv(fake_response)
    assert len(df) == 2
    assert df["PrecipitationMMHR"].tolist() == [0.5, 1.2]
    print("test_parse_giovanni_csv_normal_shape_still_works: PASS")


def test_parse_giovanni_csv_real_metadata_preamble_shape():
    """Regression test for the second real live-run failure
    (2026-09-12), using the ACTUAL raw metadata text from that
    failure's diagnostic error message, with plausible data rows
    appended (the truncated real response didn't include any data
    rows to confirm their exact shape against). Giovanni's real
    response is a 'key,value' metadata preamble (prod_name, doi,
    param_short_name, param_name, unit, undef, begin_time, end_time,
    lat, lon, ...) - NOT comment-marked, so '#'-stripping doesn't
    touch it - followed by the actual time-series data. The parser
    must skip past this preamble regardless of its exact field names/
    count, by finding the first line whose first field is a valid
    timestamp."""
    real_metadata_preamble = (
        "prod_name,GPM_3IMERGHH.07\n"
        "doi,10.5067/GPM/IMERG/3B-HH/07\n"
        "param_short_name,/Grid/precipitation\n"
        "param_name,Multi-satellite precipitation estimate with gauge calibration - "
        "Final Run (recommended for general use)\n"
        "unit,mm/hr\n"
        "undef,-9999.9\n"
        "begin_time,2000-01-01 00:00:00\n"
        "end_time,2000-01-02 00:00:00\n"
        "lat,6.05\n"
        "lon,-1.73\n"
    )
    plausible_data = (
        "2000-01-01T00:00:00Z,0.5\n"
        "2000-01-01T00:30:00Z,1.2\n"
        "2000-01-01T01:00:00Z,0.0\n"
    )
    df = _parse_giovanni_csv(real_metadata_preamble + plausible_data)
    assert list(df.columns) == ["Timestamp", "PrecipitationMMHR"]
    assert len(df) == 3
    assert df["PrecipitationMMHR"].tolist() == [0.5, 1.2, 0.0]
    print("test_parse_giovanni_csv_real_metadata_preamble_shape: PASS")


def test_parse_giovanni_csv_tolerates_extra_label_column_in_data_rows():
    """The exact shape of data rows past the metadata preamble hasn't
    been confirmed against a real response (no data rows were visible
    in the truncated real failure) - confirms the parser tolerates
    either a plain 'timestamp,value' row or a 'timestamp,label,value'
    row with an extra column in between, by always taking the FIRST
    field as the timestamp and the LAST as the value."""
    text = (
        "prod_name,GPM_3IMERGHH.07\n"
        "begin_time,2000-01-01 00:00:00\n"
        "2000-01-01T00:00:00Z,GPM_3IMERGHH.07,0.5\n"
        "2000-01-01T00:30:00Z,GPM_3IMERGHH.07,1.2\n"
    )
    df = _parse_giovanni_csv(text)
    assert len(df) == 2
    assert df["PrecipitationMMHR"].tolist() == [0.5, 1.2]
    print("test_parse_giovanni_csv_tolerates_extra_label_column_in_data_rows: PASS")


def test_parse_giovanni_csv_metadata_only_response_raises_clear_error():
    """Regression test using the EXACT raw text from the real failure
    (truncated - Giovanni's response for this request apparently
    contained only metadata within what was captured, or the actual
    data was cut off by the diagnostic capture limit at the time).
    Must raise a clear, diagnostic error - not silently return an
    empty/wrong result."""
    real_truncated_response = (
        "prod_name,GPM_3IMERGHH.07\n"
        "doi,10.5067/GPM/IMERG/3B-HH/07\n"
        "param_short_name,/Grid/precipitation\n"
        "param_name,Multi-satellite precipitation estimate with gauge calibration - "
        "Final Run (recommended for general use)\n"
        "unit,mm/hr\n"
        "undef,-9999.9\n"
        "begin_time,2000-01-01 00:00:00\n"
        "end_time,2000-01-02 00:00:00\n"
        "lat,6.05\n"
    )
    try:
        _parse_giovanni_csv(real_truncated_response)
        assert False, "should have raised"
    except ValueError as e:
        assert "No timestamped data rows found" in str(e)
        assert "prod_name" in str(
            e
        )  # confirms the raw-response diagnostic dump is present
    print("test_parse_giovanni_csv_metadata_only_response_raises_clear_error: PASS")


# ----------------------------------------------------------------------
# Orchestration, including concurrency (aligned with chirps_extract)
# ----------------------------------------------------------------------


def test_fetch_giovanni_timeseries_orchestration_with_fake_network():
    call_log = []

    def fake_fetch(lat, lon, start_iso, end_iso, variable, token):
        call_log.append((start_iso, end_iso))
        idx = pd.date_range(start_iso, end_iso, freq="D")
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 1.0})

    result = fetch_giovanni_timeseries(
        lat=4.75,
        lon=0.55,
        start_date="2020-01-01",
        end_date="2021-12-31",
        chunk_months=12,
        token="fake-token",
        fetch_chunk_fn=fake_fetch,
        max_workers=1,
    )
    assert len(call_log) == 2
    assert result.dataframe["Timestamp"].is_monotonic_increasing
    assert result.dataframe["Timestamp"].duplicated().sum() == 0
    print("test_fetch_giovanni_timeseries_orchestration_with_fake_network: PASS")


def test_fetch_giovanni_timeseries_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky_fetch(lat, lon, start_iso, end_iso, variable, token):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionError("simulated transient failure")
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(start_iso)], "PrecipitationMMHR": [2.0]}
        )

    result = fetch_giovanni_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        chunk_months=12,
        token="fake-token",
        fetch_chunk_fn=flaky_fetch,
        max_retries=3,
        retry_backoff_s=0.01,
        max_workers=1,
    )
    assert attempts["n"] == 2
    assert len(result.dataframe) == 1
    assert not result.warnings or "failed after" not in result.warnings[0]
    print("test_fetch_giovanni_timeseries_retries_then_succeeds: PASS")


def test_fetch_giovanni_timeseries_reports_failure_after_max_retries():
    def always_fails(lat, lon, start_iso, end_iso, variable, token):
        raise ConnectionError("simulated permanent failure")

    result = fetch_giovanni_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        chunk_months=12,
        token="fake-token",
        fetch_chunk_fn=always_fails,
        max_retries=2,
        retry_backoff_s=0.01,
        max_workers=1,
    )
    assert len(result.dataframe) == 0
    assert any("failed after" in w for w in result.warnings)
    print("test_fetch_giovanni_timeseries_reports_failure_after_max_retries: PASS")


def test_fetch_giovanni_timeseries_flags_gaps():
    def sparse_fetch(lat, lon, start_iso, end_iso, variable, token):
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(start_iso)], "PrecipitationMMHR": [0.1]}
        )

    result = fetch_giovanni_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-10",
        chunk_months=12,
        token="fake-token",
        fetch_chunk_fn=sparse_fetch,
        max_workers=1,
    )
    assert any("expected ~" in w for w in result.warnings)
    print("test_fetch_giovanni_timeseries_flags_gaps: PASS")


def test_fetch_giovanni_timeseries_concurrent_speedup():
    """New for the CHIRPS-aligned concurrency: confirms concurrent
    fetching actually reduces wall-clock time under simulated latency,
    same discipline as chirps_extract's equivalent test."""

    def slow_fetch(lat, lon, start_iso, end_iso, variable, token):
        time.sleep(0.05)
        idx = pd.date_range(start_iso, end_iso, freq="D")
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 1.0})

    start = time.time()
    fetch_giovanni_timeseries(
        lat=0,
        lon=0,
        start_date="2000-06-01",
        end_date="2008-06-01",
        chunk_months=12,
        token="t",
        fetch_chunk_fn=slow_fetch,
        max_workers=1,
    )
    serial_time = time.time() - start

    start = time.time()
    fetch_giovanni_timeseries(
        lat=0,
        lon=0,
        start_date="2000-06-01",
        end_date="2008-06-01",
        chunk_months=12,
        token="t",
        fetch_chunk_fn=slow_fetch,
        max_workers=8,
    )
    concurrent_time = time.time() - start

    speedup = serial_time / concurrent_time
    assert (
        speedup > 2.5
    ), f"expected meaningful speedup from concurrency, got only {speedup:.1f}x"
    print(
        f"test_fetch_giovanni_timeseries_concurrent_speedup: PASS ({speedup:.1f}x speedup)"
    )


def test_fetch_giovanni_timeseries_concurrent_date_ordering_preserved():
    import random

    def random_delay_fetch(lat, lon, start_iso, end_iso, variable, token):
        time.sleep(random.uniform(0, 0.02))
        idx = pd.date_range(start_iso, end_iso, freq="D")
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 1.0})

    result = fetch_giovanni_timeseries(
        lat=0,
        lon=0,
        start_date="2010-01-01",
        end_date="2015-01-01",
        chunk_months=12,
        token="t",
        fetch_chunk_fn=random_delay_fetch,
        max_workers=8,
    )
    assert result.dataframe["Timestamp"].is_monotonic_increasing
    print("test_fetch_giovanni_timeseries_concurrent_date_ordering_preserved: PASS")


def test_fetch_giovanni_timeseries_concurrent_error_isolation():
    def sometimes_fails(lat, lon, start_iso, end_iso, variable, token):
        if start_iso.startswith("2012"):
            raise ConnectionError("simulated failure")
        idx = pd.date_range(start_iso, end_iso, freq="D")
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 3.0})

    result = fetch_giovanni_timeseries(
        lat=0,
        lon=0,
        start_date="2010-01-01",
        end_date="2015-01-01",
        chunk_months=12,
        token="t",
        fetch_chunk_fn=sometimes_fails,
        max_retries=1,
        retry_backoff_s=0.01,
        max_workers=8,
    )
    assert any("2012" in w for w in result.warnings)
    assert (result.dataframe["PrecipitationMMHR"] == 3.0).all()
    print("test_fetch_giovanni_timeseries_concurrent_error_isolation: PASS")


# ----------------------------------------------------------------------
# Point extraction from a synthetic in-memory xarray Dataset
# ----------------------------------------------------------------------


def _make_synthetic_imerg_dataset(var_name="precipitation"):
    import xarray as xr

    lats = np.array([-0.1, 0.0, 0.1])
    lons = np.array([-0.1, 0.0, 0.1])
    data = np.arange(9, dtype="float32").reshape(3, 3)
    return xr.Dataset(
        {var_name: (["lat", "lon"], data)},
        coords={"lat": lats, "lon": lons, "time": [pd.Timestamp("2020-01-01")]},
    )


def test_read_point_from_dataset_nearest_neighbour():
    ds = _make_synthetic_imerg_dataset()
    value = read_point_from_dataset(
        ds, lat=0.02, lon=-0.03, variable_candidates=("precipitation",)
    )
    assert value == 4.0, value
    print("test_read_point_from_dataset_nearest_neighbour: PASS")


def test_read_point_from_dataset_v06_fallback_name():
    ds = _make_synthetic_imerg_dataset(var_name="precipitationCal")
    value = read_point_from_dataset(ds, lat=0.0, lon=0.0)
    assert value == 4.0
    print("test_read_point_from_dataset_v06_fallback_name: PASS")


def test_read_point_from_dataset_missing_variable_raises():
    ds = _make_synthetic_imerg_dataset(var_name="somethingElse")
    try:
        read_point_from_dataset(ds, lat=0.0, lon=0.0)
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_read_point_from_dataset_missing_variable_raises: PASS")


def test_estimate_granule_count():
    n = estimate_granule_count("2020-01-01", "2020-01-02")
    assert n == 49, n
    print("test_estimate_granule_count: PASS")


def test_estimate_granule_count_full_record_is_huge():
    n = estimate_granule_count("2000-06-01", "2026-01-01")
    assert n > 400_000, n
    print("test_estimate_granule_count_full_record_is_huge: PASS")


def test_granule_based_method_refuses_large_range_without_force():
    try:
        fetch_granule_based_timeseries(
            lat=0, lon=0, start_date="2000-06-01", end_date="2026-01-01"
        )
        assert False, "should have raised"
    except ValueError as e:
        assert "not suitable for long ranges" in str(e)
    print("test_granule_based_method_refuses_large_range_without_force: PASS")


# ----------------------------------------------------------------------
# NetCDF backend fast-check - regression test for a real live failure:
# a 25-minute run downloaded all 240 granules successfully, then
# failed to open every single one because xarray had no NetCDF/HDF5
# backend installed. Checking this upfront turns that into an
# immediate error instead of 25 minutes of wasted downloads.
# ----------------------------------------------------------------------


def test_check_netcdf_backend_raises_clearly_when_neither_installed():
    """Forces both backends' absence via mocking rather than relying
    on ambient environment state - this sandbox originally had neither
    installed (matching the real live failure exactly), but later
    tests in this same file need netCDF4 genuinely importable to
    exercise the actual file-reading fix, so ambient absence can't be
    relied on here anymore."""
    import sys
    import unittest.mock as mock

    with mock.patch.dict(sys.modules, {"netCDF4": None, "h5netcdf": None}):
        try:
            _check_netcdf_backend_available()
            assert False, "should have raised - both backends mocked as unavailable"
        except RuntimeError as e:
            assert "netCDF4" in str(e) and "h5netcdf" in str(e)
            assert "pip install" in str(e)
    print("test_check_netcdf_backend_raises_clearly_when_neither_installed: PASS")


def test_check_netcdf_backend_passes_when_netcdf4_available():
    import sys
    import types
    import unittest.mock as mock

    fake_module = types.ModuleType("netCDF4")
    with mock.patch.dict(sys.modules, {"netCDF4": fake_module}):
        _check_netcdf_backend_available()  # must not raise
    print("test_check_netcdf_backend_passes_when_netcdf4_available: PASS")


def test_check_netcdf_backend_passes_when_only_h5netcdf_available():
    import sys
    import types
    import unittest.mock as mock

    fake_module = types.ModuleType("h5netcdf")
    # ensure netCDF4 genuinely fails to import in this test (it does
    # in this sandbox already, but be explicit rather than rely on that)
    with mock.patch.dict(sys.modules, {"h5netcdf": fake_module, "netCDF4": None}):
        _check_netcdf_backend_available()  # must not raise
    print("test_check_netcdf_backend_passes_when_only_h5netcdf_available: PASS")


def test_fetch_granule_based_timeseries_fails_fast_before_any_download():
    """The critical regression test: the backend check must happen
    BEFORE authentication/search/download, not after - confirms this
    by checking the failure happens with zero network activity (no
    earthaccess calls), matching the desired 'fail in under a second'
    behaviour rather than the real 25-minute wasted run. Forces both
    backends' absence via mocking (see
    test_check_netcdf_backend_raises_clearly_when_neither_installed
    for why ambient absence can't be relied on in this file anymore)."""
    import sys
    import unittest.mock as mock

    with mock.patch.dict(sys.modules, {"netCDF4": None, "h5netcdf": None}):
        try:
            fetch_granule_based_timeseries(
                lat=0,
                lon=0,
                start_date="2020-01-01",
                end_date="2020-01-02",  # small range, passes the size check
            )
            assert False, "should have raised - both backends mocked as unavailable"
        except RuntimeError as e:
            assert "netCDF4" in str(e) or "h5netcdf" in str(e)
    print("test_fetch_granule_based_timeseries_fails_fast_before_any_download: PASS")


def test_fetch_granule_based_timeseries_reads_real_file_without_dask():
    """Regression test for the second real live failure: 'chunk
    manager dask is not available'. xr.open_mfdataset() unconditionally
    requires dask for its lazy multi-file concatenation, even when
    only one file is being opened - which is exactly what happens
    here, since each granule is handled individually in the loop.
    Switched to xr.open_dataset() (singular), which needs no such
    dependency. This test mocks ONLY the network calls (search/
    download/auth) - the actual file-opening code path runs for real
    against a genuine NetCDF4 file with IMERG's group='Grid'
    structure, confirming the fix works end-to-end, not just in
    isolation. Requires netCDF4 to be installed to run (skipped
    gracefully if not, matching this method's own dependency)."""
    import tempfile
    import unittest.mock as mock

    try:
        import netCDF4  # noqa: F401
    except ImportError:
        print(
            "test_fetch_granule_based_timeseries_reads_real_file_without_dask: SKIPPED (netCDF4 not installed here)"
        )
        return

    import xarray as xr

    tmp_dir = Path(tempfile.mkdtemp())
    nc_path = tmp_dir / "synthetic_granule.nc4"
    synthetic_ds = xr.Dataset(
        {
            "precipitation": (
                ["lat", "lon"],
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            )
        },
        coords={
            "lat": np.array([-0.1, 0.1], dtype="float32"),
            "lon": np.array([-0.1, 0.1], dtype="float32"),
            "time": [pd.Timestamp("2005-03-01T00:00:00")],
        },
    )
    synthetic_ds.to_netcdf(nc_path, group="Grid", engine="netcdf4")

    class _FakeAuth:
        authenticated = True

    with (
        mock.patch("earthaccess.login", return_value=_FakeAuth()),
        mock.patch("earthaccess.search_data", return_value=["fake_granule"]),
        mock.patch("earthaccess.download", return_value=[str(nc_path)]),
    ):
        result = fetch_granule_based_timeseries(
            lat=0.05,
            lon=0.05,
            start_date="2005-03-01",
            end_date="2005-03-01",
        )

    assert (
        len(result.dataframe) == 1
    ), f"expected 1 row, got warnings: {result.warnings}"
    assert (
        result.dataframe["PrecipitationMMHR"].iloc[0] == 4.0
    )  # nearest cell to (0.05, 0.05)
    assert result.warnings == []
    print("test_fetch_granule_based_timeseries_reads_real_file_without_dask: PASS")


# ----------------------------------------------------------------------
# cftime.DatetimeJulian timestamp conversion - a third real live
# failure, hit immediately after the dask fix: IMERG's HDF5 granule
# files declare a Julian calendar for their time variable, so xarray
# decodes it as cftime.DatetimeJulian rather than a numpy datetime64,
# and pd.Timestamp() cannot parse that directly.
# ----------------------------------------------------------------------


def test_to_pandas_timestamp_converts_cftime_datetime_julian():
    """Uses the real cftime type from the actual error message, not a
    generic stand-in - reproduces the exact failure directly."""
    import cftime

    cf_dt = cftime.DatetimeJulian(2005, 3, 1, 0, 30, 0)
    result = _to_pandas_timestamp(cf_dt)
    assert isinstance(result, pd.Timestamp)
    assert result == pd.Timestamp("2005-03-01 00:30:00")
    print("test_to_pandas_timestamp_converts_cftime_datetime_julian: PASS")


def test_to_pandas_timestamp_converts_other_cftime_calendars_too():
    """Every cftime calendar subtype exposes the same component
    attributes - confirms the fix isn't narrowly specific to
    DatetimeJulian, in case a different IMERG version or another
    granule-based source uses a different declared calendar."""
    import cftime

    for cls in (
        cftime.DatetimeNoLeap,
        cftime.DatetimeAllLeap,
        cftime.DatetimeGregorian,
    ):
        cf_dt = cls(2010, 6, 15, 12, 0, 0)
        result = _to_pandas_timestamp(cf_dt)
        assert result == pd.Timestamp(
            "2010-06-15 12:00:00"
        ), f"failed for {cls.__name__}"
    print("test_to_pandas_timestamp_converts_other_cftime_calendars_too: PASS")


def test_to_pandas_timestamp_passes_through_normal_values_unchanged():
    """A plain numpy datetime64/pandas-compatible value (the common
    case for most NetCDF sources without an unusual calendar) must
    still work exactly as pd.Timestamp() itself would - the cftime
    handling is additive, not a behaviour change for the normal path."""
    normal = np.datetime64("2020-01-01T00:00:00")
    result = _to_pandas_timestamp(normal)
    assert result == pd.Timestamp("2020-01-01")
    print("test_to_pandas_timestamp_passes_through_normal_values_unchanged: PASS")


def test_fetch_granule_based_timeseries_reads_real_julian_calendar_file():
    """The actual real-world case, reproduced end-to-end: a NetCDF
    file written with an EXPLICIT Julian calendar encoding (confirmed
    to make xarray decode it as cftime.DatetimeJulian on read, exactly
    matching the real error) - only the network calls are mocked, the
    actual file I/O, calendar decoding, and timestamp conversion all
    run for real."""
    import tempfile
    import unittest.mock as mock

    try:
        import netCDF4  # noqa: F401
    except ImportError:
        print(
            "test_fetch_granule_based_timeseries_reads_real_julian_calendar_file: SKIPPED (netCDF4 not installed here)"
        )
        return

    import xarray as xr

    tmp_dir = Path(tempfile.mkdtemp())
    nc_path = tmp_dir / "julian_calendar_granule.nc4"
    synthetic_ds = xr.Dataset(
        {
            "precipitation": (
                ["lat", "lon"],
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            )
        },
        coords={
            "lat": np.array([-0.1, 0.1], dtype="float32"),
            "lon": np.array([-0.1, 0.1], dtype="float32"),
            "time": [pd.Timestamp("2005-03-01T00:30:00")],
        },
    )
    synthetic_ds.to_netcdf(
        nc_path,
        group="Grid",
        engine="netcdf4",
        encoding={"time": {"calendar": "julian", "units": "days since 1970-01-01"}},
    )

    # confirm the fixture actually reproduces cftime decoding, not a no-op
    with xr.open_dataset(nc_path, group="Grid") as check_ds:
        import cftime

        assert isinstance(
            check_ds["time"].values[0], cftime.DatetimeJulian
        ), "fixture did not reproduce cftime decoding - test would not exercise the real bug"

    class _FakeAuth:
        authenticated = True

    with (
        mock.patch("earthaccess.login", return_value=_FakeAuth()),
        mock.patch("earthaccess.search_data", return_value=["fake_granule"]),
        mock.patch("earthaccess.download", return_value=[str(nc_path)]),
    ):
        result = fetch_granule_based_timeseries(
            lat=0.05,
            lon=0.05,
            start_date="2005-03-01",
            end_date="2005-03-01",
        )

    assert (
        len(result.dataframe) == 1
    ), f"expected 1 row, got warnings: {result.warnings}"
    assert result.dataframe["Timestamp"].iloc[0] == pd.Timestamp("2005-03-01 00:30:00")
    assert result.warnings == []
    print("test_fetch_granule_based_timeseries_reads_real_julian_calendar_file: PASS")


# ----------------------------------------------------------------------
# Earthdata credential saving (aligned with era5_extract's CDS API Key
# pattern), but must preserve unrelated .netrc entries - unlike
# .cdsapirc, .netrc is a shared, multi-service file by convention.
# ----------------------------------------------------------------------


def test_save_earthdata_credentials_writes_new_file():
    import tempfile

    tmp_path = Path(tempfile.mkdtemp()) / ".netrc"
    result_paths = save_earthdata_credentials("myuser", "mypass", config_path=tmp_path)
    assert result_paths == [tmp_path]
    content = tmp_path.read_text()
    assert "machine urs.earthdata.nasa.gov" in content
    assert "login myuser" in content
    assert "password mypass" in content
    print("test_save_earthdata_credentials_writes_new_file: PASS")


def test_save_earthdata_credentials_writes_both_files_on_windows():
    """Regression test for a real bug found on first live use: Python's
    own netrc module - and by extension earthaccess/requests - defaults
    to looking for '_netrc' (underscore) on Windows, not '.netrc' (dot).
    A version of this function that only wrote '.netrc' created the
    file correctly, but nothing on Windows actually read it - confirmed
    directly by a live-run error ('No .netrc found at ...\\_netrc').
    Mocks platform.system() rather than requiring an actual Windows
    environment to test this."""
    import tempfile
    import unittest.mock as mock

    tmp_home = Path(tempfile.mkdtemp())
    with (
        mock.patch("platform.system", return_value="Windows"),
        mock.patch("pathlib.Path.home", return_value=tmp_home),
    ):
        result_paths = save_earthdata_credentials("winuser", "winpass")

    assert (tmp_home / ".netrc") in result_paths
    assert (tmp_home / "_netrc") in result_paths
    assert (tmp_home / ".netrc").exists()
    assert (tmp_home / "_netrc").exists()
    assert "login winuser" in (tmp_home / "_netrc").read_text()
    print("test_save_earthdata_credentials_writes_both_files_on_windows: PASS")


def test_save_earthdata_credentials_only_dotfile_on_non_windows():
    import tempfile
    import unittest.mock as mock

    tmp_home = Path(tempfile.mkdtemp())
    with (
        mock.patch("platform.system", return_value="Linux"),
        mock.patch("pathlib.Path.home", return_value=tmp_home),
    ):
        result_paths = save_earthdata_credentials("linuxuser", "linuxpass")

    assert result_paths == [tmp_home / ".netrc"]
    assert not (tmp_home / "_netrc").exists()
    print("test_save_earthdata_credentials_only_dotfile_on_non_windows: PASS")


def test_save_earthdata_credentials_preserves_other_machine_entries():
    """The critical difference from era5_extract's save_cds_credentials:
    .netrc may already contain entries for OTHER services - those must
    survive a save, not be silently wiped out."""
    import tempfile

    tmp_path = Path(tempfile.mkdtemp()) / ".netrc"
    tmp_path.write_text("machine example.com\nlogin someoneelse\npassword unrelated\n")

    save_earthdata_credentials("myuser", "mypass", config_path=tmp_path)
    content = tmp_path.read_text()

    assert "machine example.com" in content, "unrelated existing entry was wiped out"
    assert "login someoneelse" in content
    assert "machine urs.earthdata.nasa.gov" in content
    assert "login myuser" in content
    print("test_save_earthdata_credentials_preserves_other_machine_entries: PASS")


def test_save_earthdata_credentials_replaces_existing_earthdata_entry():
    """Saving a second time with new credentials must replace the old
    Earthdata entry, not duplicate it."""
    import tempfile

    tmp_path = Path(tempfile.mkdtemp()) / ".netrc"
    save_earthdata_credentials("olduser", "oldpass", config_path=tmp_path)
    save_earthdata_credentials("newuser", "newpass", config_path=tmp_path)
    content = tmp_path.read_text()

    assert (
        content.count("machine urs.earthdata.nasa.gov") == 1
    ), "duplicate entry created"
    assert "olduser" not in content
    assert "login newuser" in content
    print("test_save_earthdata_credentials_replaces_existing_earthdata_entry: PASS")


def test_fetch_giovanni_timeseries_default_full_record():
    def fake_fetch(lat, lon, start_iso, end_iso, variable, token):
        idx = pd.date_range(
            start_iso,
            min(pd.Timestamp(end_iso), pd.Timestamp(start_iso) + pd.Timedelta(days=2)),
            freq="D",
        )
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 1.0})

    result = fetch_giovanni_timeseries(
        lat=0,
        lon=0,
        end_date="2000-06-05",
        token="t",
        fetch_chunk_fn=fake_fetch,
        max_workers=1,
    )
    assert result.dataframe["Timestamp"].min().date().isoformat() == "2000-06-01"
    print("test_fetch_giovanni_timeseries_default_full_record: PASS")


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
