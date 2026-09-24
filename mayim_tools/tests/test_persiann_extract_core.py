"""
Tests for persiann_extract/core.py. Run: python3 tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gzip
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

from mayim_tools.rainfall.persiann_extract.core import (
    CCS_GRID_COLS,
    CCS_GRID_LEFT_LON,
    CCS_GRID_ROWS,
    CCS_GRID_TOP_LAT,
    RECORD_START,
    FetchResult,
    _build_session,
    _to_signed_lon,
    build_ccs_url,
    build_year_list_url,
    fetch_one_ccs_file,
    fetch_one_daily_file,
    fetch_persiann_3hourly,
    fetch_persiann_daily,
    fetch_year_listing,
    parse_azure_blob_list,
    read_ccs_file,
    read_ccs_file_with_center,
    read_persiann_daily_file,
    read_persiann_daily_file_with_center,
    three_hourly_range,
    to_0_360,
)

# ----------------------------------------------------------------------
# Longitude conversion - both products confirmed to use 0-360
# ----------------------------------------------------------------------


def test_to_0_360_negative_longitude():
    assert abs(to_0_360(-1.7334) - 358.2666) < 1e-6
    print("test_to_0_360_negative_longitude: PASS")


def test_to_0_360_positive_longitude_unchanged():
    assert abs(to_0_360(100.0) - 100.0) < 1e-9
    print("test_to_0_360_positive_longitude_unchanged: PASS")


def test_to_0_360_zero():
    assert to_0_360(0.0) == 0.0
    print("test_to_0_360_zero: PASS")


def test_record_start_confirmed():
    assert RECORD_START == date(1983, 1, 1)
    print("test_record_start_confirmed: PASS")


def test_to_signed_lon_converts_past_180():
    assert abs(_to_signed_lon(358.2666) - (-1.7334)) < 1e-6
    print("test_to_signed_lon_converts_past_180: PASS")


def test_to_signed_lon_leaves_normal_range_unchanged():
    assert _to_signed_lon(100.0) == 100.0
    assert _to_signed_lon(0.0) == 0.0
    print("test_to_signed_lon_leaves_normal_range_unchanged: PASS")


# ----------------------------------------------------------------------
# Daily PERSIANN-CDR / direct Azure NODD file access
# ----------------------------------------------------------------------


def test_build_year_list_url():
    url = build_year_list_url(1983)
    assert (
        url
        == "https://noaacdr.blob.core.windows.net/precip-persiann?restype=container&comp=list&prefix=data/1983/"
    )
    print("test_build_year_list_url: PASS")


AZURE_LIST_XML_REAL_EXAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<EnumerationResults ContainerName="https://noaacdr.blob.core.windows.net/precip-persiann">
<Blobs>
<Blob>
<Name>data/1983/PERSIANN-CDR_v01r01_19830101_c20140523.nc</Name>
<Url>https://noaacdr.blob.core.windows.net/precip-persiann/data/1983/PERSIANN-CDR_v01r01_19830101_c20140523.nc</Url>
<Properties>
<Last-Modified>Wed, 01 Sep 2021 16:32:04 GMT</Last-Modified>
<Content-Length>1058743</Content-Length>
<BlobType>BlockBlob</BlobType>
</Properties>
</Blob>
<Blob>
<Name>data/1983/PERSIANN-CDR_v01r01_19830102_c20140523.nc</Name>
<Url>https://noaacdr.blob.core.windows.net/precip-persiann/data/1983/PERSIANN-CDR_v01r01_19830102_c20140523.nc</Url>
<Properties>
<Last-Modified>Wed, 01 Sep 2021 16:32:26 GMT</Last-Modified>
<Content-Length>1120881</Content-Length>
<BlobType>BlockBlob</BlobType>
</Properties>
</Blob>
</Blobs>
</EnumerationResults>"""


def test_parse_azure_blob_list_real_example():
    """Parsed against the exact, real Azure Blob List response shared
    directly - not a synthetic guess at the XML structure."""
    date_to_url, warnings = parse_azure_blob_list(AZURE_LIST_XML_REAL_EXAMPLE)
    assert warnings == []
    assert date_to_url["19830101"] == (
        "https://noaacdr.blob.core.windows.net/precip-persiann/data/1983/"
        "PERSIANN-CDR_v01r01_19830101_c20140523.nc"
    )
    assert date_to_url["19830102"] == (
        "https://noaacdr.blob.core.windows.net/precip-persiann/data/1983/"
        "PERSIANN-CDR_v01r01_19830102_c20140523.nc"
    )
    assert len(date_to_url) == 2
    print("test_parse_azure_blob_list_real_example: PASS")


def test_parse_azure_blob_list_ignores_non_matching_names():
    xml = """<?xml version="1.0"?>
<EnumerationResults><Blobs>
<Blob><Name>data/1983/some_other_file.txt</Name><Url>https://x/y</Url></Blob>
<Blob><Name>data/1983/PERSIANN-CDR_v01r01_19830103_c20140523.nc</Name>
<Url>https://noaacdr.blob.core.windows.net/precip-persiann/data/1983/PERSIANN-CDR_v01r01_19830103_c20140523.nc</Url></Blob>
</Blobs></EnumerationResults>"""
    date_to_url, warnings = parse_azure_blob_list(xml)
    assert list(date_to_url.keys()) == ["19830103"]
    print("test_parse_azure_blob_list_ignores_non_matching_names: PASS")


def test_parse_azure_blob_list_warns_on_next_marker():
    """A NextMarker means Azure truncated the response - pagination
    isn't implemented, so this must be surfaced as a warning, not
    silently return an incomplete result."""
    xml = """<?xml version="1.0"?>
<EnumerationResults><Blobs>
<Blob><Name>data/1983/PERSIANN-CDR_v01r01_19830101_c20140523.nc</Name>
<Url>https://x/y.nc</Url></Blob>
</Blobs><NextMarker>abc123</NextMarker></EnumerationResults>"""
    date_to_url, warnings = parse_azure_blob_list(xml)
    assert any("paginated" in w for w in warnings)
    print("test_parse_azure_blob_list_warns_on_next_marker: PASS")


def test_parse_azure_blob_list_no_next_marker_no_warning():
    date_to_url, warnings = parse_azure_blob_list(AZURE_LIST_XML_REAL_EXAMPLE)
    assert warnings == []
    print("test_parse_azure_blob_list_no_next_marker_no_warning: PASS")


def test_fetch_persiann_daily_orchestration_with_fakes():
    listing_calls = []
    fetch_calls = []

    def fake_listing(year, **kwargs):
        listing_calls.append(year)
        return ({f"{year}0101": f"https://fake/{year}0101.nc"}, [])

    def fake_fetch(url, lat, lon, d):
        fetch_calls.append(url)
        return pd.DataFrame({"Date": [pd.Timestamp(d)], "PrecipitationMM": [1.0]})

    result = fetch_persiann_daily(
        lat=6.0539,
        lon=-1.7334,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=fake_fetch,
        listing_fn=fake_listing,
        max_workers=1,
    )
    assert listing_calls == [2020]  # exactly one listing call for the one year needed
    assert len(fetch_calls) == 1
    assert len(result.dataframe) == 1
    print("test_fetch_persiann_daily_orchestration_with_fakes: PASS")


def test_fetch_persiann_daily_lists_each_year_exactly_once():
    """A multi-year range must trigger exactly one listing call PER
    YEAR, not one per day - the whole point of listing years instead
    of guessing per-day filenames."""
    listing_calls = []

    def fake_listing(year, **kwargs):
        listing_calls.append(year)
        # only Jan 1st has a file, for simplicity - fine, since this test
        # only cares about how many times the year gets listed
        return ({f"{year}0101": f"https://fake/{year}0101.nc"}, [])

    def fake_fetch(url, lat, lon, d):
        return pd.DataFrame({"Date": [pd.Timestamp(d)], "PrecipitationMM": [1.0]})

    fetch_persiann_daily(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2022-12-31",
        fetch_fn=fake_fetch,
        listing_fn=fake_listing,
        max_workers=1,
    )
    assert listing_calls == [2020, 2021, 2022]
    print("test_fetch_persiann_daily_lists_each_year_exactly_once: PASS")


def test_fetch_persiann_daily_retries_then_succeeds():
    attempts = {"n": 0}

    def fake_listing(year, **kwargs):
        return ({"20200101": "https://fake/20200101.nc"}, [])

    def flaky_fetch(url, lat, lon, d):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionError("simulated transient failure")
        return pd.DataFrame({"Date": [pd.Timestamp(d)], "PrecipitationMM": [0.2]})

    result = fetch_persiann_daily(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=flaky_fetch,
        listing_fn=fake_listing,
        max_retries=3,
        retry_backoff_s=0.01,
        max_workers=1,
    )
    assert attempts["n"] == 2
    assert len(result.dataframe) == 1
    print("test_fetch_persiann_daily_retries_then_succeeds: PASS")


def test_fetch_persiann_daily_reports_missing_values():
    def fake_listing(year, **kwargs):
        return ({"20200101": "https://fake/1.nc", "20200102": "https://fake/2.nc"}, [])

    def fake_fetch(url, lat, lon, d):
        value = 1.0 if d.day == 1 else float("nan")
        return pd.DataFrame({"Date": [pd.Timestamp(d)], "PrecipitationMM": [value]})

    result = fetch_persiann_daily(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-02",
        fetch_fn=fake_fetch,
        listing_fn=fake_listing,
        max_workers=1,
    )
    assert any("had no value" in w for w in result.warnings)
    print("test_fetch_persiann_daily_reports_missing_values: PASS")


def test_fetch_persiann_daily_reports_dates_missing_from_listing():
    """A requested date with no matching file in the year's listing
    (a genuine archive gap, or a listing failure) must be reported
    explicitly, not silently skipped."""

    def fake_listing(year, **kwargs):
        return ({"20200101": "https://fake/1.nc"}, [])  # 20200102 deliberately absent

    def fake_fetch(url, lat, lon, d):
        return pd.DataFrame({"Date": [pd.Timestamp(d)], "PrecipitationMM": [1.0]})

    result = fetch_persiann_daily(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-02",
        fetch_fn=fake_fetch,
        listing_fn=fake_listing,
        max_workers=1,
    )
    assert any("no matching file" in w for w in result.warnings)
    assert len(result.dataframe) == 1  # only the one date that WAS found
    print("test_fetch_persiann_daily_reports_dates_missing_from_listing: PASS")


def test_fetch_persiann_daily_failure_warning_includes_attempted_url():
    def fake_listing(year, **kwargs):
        return ({"20200101": "https://fake/persiann/20200101.nc"}, [])

    def always_fails(url, lat, lon, d):
        raise TimeoutError("simulated read timeout")

    result = fetch_persiann_daily(
        lat=6.0539,
        lon=-1.7334,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=always_fails,
        listing_fn=fake_listing,
        max_retries=1,
        retry_backoff_s=0.01,
        max_workers=1,
    )
    assert any("20200101.nc" in w for w in result.warnings)
    print("test_fetch_persiann_daily_failure_warning_includes_attempted_url: PASS")


def test_fetch_persiann_daily_reports_matched_coordinate():
    """The fake fetch_fn stands in for fetch_one_daily_file, so it must
    attach the same .attrs a real read_persiann_daily_file_with_center()
    call would."""

    def fake_listing(year, **kwargs):
        return ({"20200101": "https://fake/20200101.nc"}, [])

    def fake_fetch(url, lat, lon, d):
        df = pd.DataFrame({"Date": [pd.Timestamp(d)], "PrecipitationMM": [1.0]})
        df.attrs["matched_lat"] = 6.0
        df.attrs["matched_lon"] = -1.75
        return df

    result = fetch_persiann_daily(
        lat=6.0539,
        lon=-1.7334,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=fake_fetch,
        listing_fn=fake_listing,
        max_workers=1,
    )
    assert isinstance(result, FetchResult)
    assert any("matched to the nearest grid cell" in w for w in result.warnings)
    assert result.requested_lat == 6.0539
    assert result.requested_lon == -1.7334
    assert result.matched_lat == 6.0
    assert result.matched_lon == -1.75
    assert result.distance_km is not None and result.distance_km > 0
    print("test_fetch_persiann_daily_reports_matched_coordinate: PASS")


def test_fetch_persiann_daily_without_matched_attrs_reports_nothing():
    """Backward compatibility: a fake fetch_fn returning a plain
    DataFrame with no .attrs (every OTHER test in this file) must leave
    matched_lat/matched_lon as None and add no grid-match warning."""

    def fake_listing(year, **kwargs):
        return ({"20200101": "https://fake/20200101.nc"}, [])

    def fake_fetch(url, lat, lon, d):
        return pd.DataFrame({"Date": [pd.Timestamp(d)], "PrecipitationMM": [1.0]})

    result = fetch_persiann_daily(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=fake_fetch,
        listing_fn=fake_listing,
        max_workers=1,
    )
    assert result.matched_lat is None
    assert not any("matched to the nearest grid cell" in w for w in result.warnings)
    print("test_fetch_persiann_daily_without_matched_attrs_reports_nothing: PASS")


def test_fetch_persiann_daily_concurrent_speedup():
    def fake_listing(year, **kwargs):
        return (
            {
                f"2020{m:02d}{d:02d}": f"https://fake/{m}{d}.nc"
                for m in range(1, 3)
                for d in range(1, 11)
            },
            [],
        )

    def slow_fetch(url, lat, lon, d):
        time.sleep(0.02)
        return pd.DataFrame({"Date": [pd.Timestamp(d)], "PrecipitationMM": [0.1]})

    start = time.time()
    fetch_persiann_daily(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-10",
        fetch_fn=slow_fetch,
        listing_fn=fake_listing,
        max_workers=1,
    )
    serial_time = time.time() - start

    start = time.time()
    fetch_persiann_daily(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-10",
        fetch_fn=slow_fetch,
        listing_fn=fake_listing,
        max_workers=8,
    )
    concurrent_time = time.time() - start

    speedup = serial_time / concurrent_time
    assert speedup > 2.5, f"expected meaningful speedup, got only {speedup:.1f}x"
    print(
        f"test_fetch_persiann_daily_concurrent_speedup: PASS ({speedup:.1f}x speedup)"
    )


def _make_synthetic_daily_netcdf(
    value_at_point, lat=6.0, lon_360=358.25, var_name="precipitation"
):
    """Builds a real, small NetCDF file matching the plausible daily
    PERSIANN-CDR structure (single day, lat/lon grid, 0-360 longitude)
    - a genuine file for xarray to open, not a mock of the reading
    logic. NOT a confirmed real PERSIANN-CDR file structure (never
    directly opened one - see core.py's docstring) - this tests that
    the reading logic works correctly against a plausible structure,
    which is the most that's verifiable without a live download."""
    import tempfile

    import xarray as xr

    lats = np.array([5.75, 6.0, 6.25])
    lons = np.array([358.0, 358.25, 358.5])
    data = np.zeros((1, 3, 3), dtype="float32")
    lat_idx = list(lats).index(lat)
    lon_idx = list(lons).index(lon_360)
    data[0, lat_idx, lon_idx] = value_at_point

    ds = xr.Dataset(
        {var_name: (["time", "lat", "lon"], data)},
        coords={"time": [np.datetime64("2020-01-01")], "lat": lats, "lon": lons},
    )
    tmp_path = Path(tempfile.mkdtemp()) / "test.nc"
    ds.to_netcdf(tmp_path)
    return tmp_path.read_bytes()


def test_read_persiann_daily_file_correct_value():
    raw = _make_synthetic_daily_netcdf(4.2, lat=6.0, lon_360=358.25)
    value = read_persiann_daily_file(
        raw, lat=6.0, lon=-1.75
    )  # -1.75 -> 358.25 after 0-360 conversion
    assert abs(value - 4.2) < 1e-4
    print("test_read_persiann_daily_file_correct_value: PASS")


def test_read_persiann_daily_file_falls_back_to_sole_data_var():
    """If the real per-file variable name turns out not to be
    'precipitation' (never directly confirmed - see core.py's
    docstring), falling back to the file's sole data variable keeps
    this working rather than failing outright."""
    raw = _make_synthetic_daily_netcdf(
        3.3, lat=6.0, lon_360=358.25, var_name="some_other_name"
    )
    value = read_persiann_daily_file(raw, lat=6.0, lon=-1.75)
    assert abs(value - 3.3) < 1e-4
    print("test_read_persiann_daily_file_falls_back_to_sole_data_var: PASS")


def test_read_persiann_daily_file_with_center_returns_matched_coordinate():
    """New: matched-grid-cell reporting, mirroring the treatment already
    applied to era5_extract/chirps/cmorph_extract - confirms the
    matched coordinate is the grid cell actually read (lat=6.0,
    lon_360=358.25 -> signed lon -1.75), not just the requested point
    echoed back."""
    raw = _make_synthetic_daily_netcdf(4.2, lat=6.0, lon_360=358.25)
    value, matched_lat, matched_lon = read_persiann_daily_file_with_center(
        raw, lat=6.03, lon=-1.78
    )  # off-grid, should snap to (6.0, 358.25 -> -1.75)
    assert abs(value - 4.2) < 1e-4
    assert abs(matched_lat - 6.0) < 1e-6
    assert abs(matched_lon - (-1.75)) < 1e-6
    print("test_read_persiann_daily_file_with_center_returns_matched_coordinate: PASS")


def test_fetch_one_daily_file_uses_provided_session():
    import unittest.mock as mock

    raw = _make_synthetic_daily_netcdf(2.1, lat=6.0, lon_360=358.25)

    class _FakeResponse:
        content = raw

        def raise_for_status(self):
            pass

    fake_session = mock.Mock()
    fake_session.get.return_value = _FakeResponse()

    df = fetch_one_daily_file(
        "https://fake/url.nc",
        lat=6.0,
        lon=-1.75,
        fallback_date=date(2020, 1, 1),
        session=fake_session,
    )
    fake_session.get.assert_called_once()
    assert abs(df["PrecipitationMM"].iloc[0] - 2.1) < 1e-3
    print("test_fetch_one_daily_file_uses_provided_session: PASS")


# ----------------------------------------------------------------------
# 3-Hourly PERSIANN-CCS-CDR / direct gzip-binary files
# ----------------------------------------------------------------------


def test_build_ccs_url_matches_real_confirmed_example():
    """The exact URL from a real directory listing entry: 3-hourly,
    2000-01-01, hour 00."""
    url = build_ccs_url(datetime(2000, 1, 1, 0))
    expected = (
        "https://persiann.eng.uci.edu/CHRSdata/PCCSCDR/3hrly/PCCSCDR3h00010100.bin.gz"
    )
    assert url == expected, f"got {url}"
    print("test_build_ccs_url_matches_real_confirmed_example: PASS")


def test_build_ccs_url_pre_2000_two_digit_year():
    """Confirms the 2-digit year convention for the pre-2000 portion
    of the record (1983-1999) - the record start is 1983, so year 83
    must map correctly, not wrap or collide with a 20xx date."""
    url = build_ccs_url(datetime(1983, 1, 1, 0))
    assert url.endswith("PCCSCDR3h83010100.bin.gz")
    print("test_build_ccs_url_pre_2000_two_digit_year: PASS")


def test_build_ccs_url_no_year_subdirectory():
    """Confirmed directly: unlike CMORPH, all files sit in ONE FLAT
    directory, not year/month subdirectories."""
    url = build_ccs_url(datetime(2010, 6, 15, 9))
    assert (
        url
        == "https://persiann.eng.uci.edu/CHRSdata/PCCSCDR/3hrly/PCCSCDR3h10061509.bin.gz"
    )
    print("test_build_ccs_url_no_year_subdirectory: PASS")


def test_three_hourly_range_single_day():
    periods = three_hourly_range(date(2020, 1, 1), date(2020, 1, 1))
    assert len(periods) == 8  # 00, 03, 06, ..., 21
    assert periods[0] == datetime(2020, 1, 1, 0)
    assert periods[-1] == datetime(2020, 1, 1, 21)
    print("test_three_hourly_range_single_day: PASS")


def test_three_hourly_range_rejects_backwards_range():
    try:
        three_hourly_range(date(2020, 1, 2), date(2020, 1, 1))
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("test_three_hourly_range_rejects_backwards_range: PASS")


def _make_synthetic_ccs_file(value_at_target, target_row, target_col):
    """Builds a real gzip-compressed flat float32 array matching the
    documented CCS-CDR grid layout exactly, with a known value at a
    known row/col - a genuine synthetic file, not a mock of the
    reading logic."""
    grid = np.zeros((CCS_GRID_ROWS, CCS_GRID_COLS), dtype="<f4")
    grid[target_row, target_col] = value_at_target
    return gzip.compress(grid.tobytes())


def test_read_ccs_file_correct_grid_indexing():
    """Confirms the documented grid formula (first row at 59.98N,
    first col at 0.02E, 0.04 deg resolution) correctly locates a
    known point."""
    lat, lon = 6.0539, -1.7334  # Ghana site used throughout this project
    lon_360 = to_0_360(lon)
    expected_row = round((CCS_GRID_TOP_LAT - lat) / 0.04)
    expected_col = round((lon_360 - CCS_GRID_LEFT_LON) / 0.04)

    raw = _make_synthetic_ccs_file(3.7, expected_row, expected_col)
    value = read_ccs_file(raw, lat=lat, lon=lon)
    assert abs(value - 3.7) < 1e-4
    print("test_read_ccs_file_correct_grid_indexing: PASS")


def test_read_ccs_file_with_center_returns_matched_coordinate():
    lat, lon = 6.0539, -1.7334
    lon_360 = to_0_360(lon)
    expected_row = round((CCS_GRID_TOP_LAT - lat) / 0.04)
    expected_col = round((lon_360 - CCS_GRID_LEFT_LON) / 0.04)
    expected_matched_lat = CCS_GRID_TOP_LAT - expected_row * 0.04
    expected_matched_lon = _to_signed_lon(CCS_GRID_LEFT_LON + expected_col * 0.04)

    raw = _make_synthetic_ccs_file(3.7, expected_row, expected_col)
    value, matched_lat, matched_lon = read_ccs_file_with_center(raw, lat=lat, lon=lon)
    assert abs(value - 3.7) < 1e-4
    assert abs(matched_lat - expected_matched_lat) < 1e-6
    assert abs(matched_lon - expected_matched_lon) < 1e-6
    print("test_read_ccs_file_with_center_returns_matched_coordinate: PASS")


def test_read_ccs_file_wrong_size_raises_clear_error():
    bad_data = gzip.compress(b"too short")
    try:
        read_ccs_file(bad_data, lat=0, lon=0)
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "expected exactly" in str(e)
    print("test_read_ccs_file_wrong_size_raises_clear_error: PASS")


def test_fetch_one_ccs_file_uses_provided_session():
    import unittest.mock as mock

    raw = _make_synthetic_ccs_file(2.5, 1000, 4500)

    class _FakeResponse:
        content = raw

        def raise_for_status(self):
            pass

    fake_session = mock.Mock()
    fake_session.get.return_value = _FakeResponse()

    lat = CCS_GRID_TOP_LAT - 1000 * 0.04
    lon = CCS_GRID_LEFT_LON + 4500 * 0.04

    df = fetch_one_ccs_file(
        "https://fake/url.bin.gz",
        lat=lat,
        lon=lon,
        fallback_timestamp=datetime(2020, 1, 1),
        session=fake_session,
    )
    fake_session.get.assert_called_once()
    assert abs(df["PrecipitationMM"].iloc[0] - 2.5) < 1e-3
    print("test_fetch_one_ccs_file_uses_provided_session: PASS")


def test_build_session_pool_size():
    session = _build_session(pool_size=12)
    adapter = session.get_adapter("https://persiann.eng.uci.edu/")
    assert adapter._pool_maxsize == 12
    print("test_build_session_pool_size: PASS")


def test_fetch_persiann_3hourly_orchestration_with_fakes():
    call_log = []

    def fake_fetch(url, lat, lon, dt):
        call_log.append(url)
        return pd.DataFrame({"Timestamp": [pd.Timestamp(dt)], "PrecipitationMM": [0.3]})

    result = fetch_persiann_3hourly(
        lat=6.0539,
        lon=-1.7334,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=fake_fetch,
        max_workers=1,
    )
    assert len(call_log) == 8  # 8 three-hourly files for a single day
    assert len(result.dataframe) == 8
    print("test_fetch_persiann_3hourly_orchestration_with_fakes: PASS")


def test_fetch_persiann_3hourly_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky_fetch(url, lat, lon, dt):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionError("simulated transient failure")
        return pd.DataFrame({"Timestamp": [pd.Timestamp(dt)], "PrecipitationMM": [0.2]})

    result = fetch_persiann_3hourly(
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
    print("test_fetch_persiann_3hourly_retries_then_succeeds: PASS")


def test_fetch_persiann_3hourly_isolates_per_file_failures():
    def sometimes_fails(url, lat, lon, dt):
        if dt.hour == 12:
            raise ConnectionError("simulated failure")
        return pd.DataFrame({"Timestamp": [pd.Timestamp(dt)], "PrecipitationMM": [0.4]})

    result = fetch_persiann_3hourly(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=sometimes_fails,
        max_retries=1,
        retry_backoff_s=0.01,
        max_workers=4,
    )
    assert len(result.dataframe) == 7  # 8 periods minus the one that always fails
    assert any("failed after" in w for w in result.warnings)
    print("test_fetch_persiann_3hourly_isolates_per_file_failures: PASS")


def test_fetch_persiann_3hourly_reports_matched_coordinate():
    def fake_fetch(url, lat, lon, dt):
        df = pd.DataFrame({"Timestamp": [pd.Timestamp(dt)], "PrecipitationMM": [0.3]})
        df.attrs["matched_lat"] = 6.0
        df.attrs["matched_lon"] = -1.76
        return df

    result = fetch_persiann_3hourly(
        lat=6.0539,
        lon=-1.7334,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=fake_fetch,
        max_workers=1,
    )
    assert any("matched to the nearest grid cell" in w for w in result.warnings)
    assert result.matched_lat == 6.0
    assert result.matched_lon == -1.76
    assert result.distance_km is not None and result.distance_km > 0
    print("test_fetch_persiann_3hourly_reports_matched_coordinate: PASS")


def test_fetch_persiann_3hourly_reports_matched_coordinate_concurrent():
    """Same as above but through the ThreadPoolExecutor path
    (max_workers > 1) - confirms the matched-coordinate capture's
    separate lock keeps this race-free too."""

    def fake_fetch(url, lat, lon, dt):
        df = pd.DataFrame({"Timestamp": [pd.Timestamp(dt)], "PrecipitationMM": [0.3]})
        df.attrs["matched_lat"] = 6.0
        df.attrs["matched_lon"] = -1.76
        return df

    result = fetch_persiann_3hourly(
        lat=6.0539,
        lon=-1.7334,
        start_date="2020-01-01",
        end_date="2020-01-03",
        fetch_fn=fake_fetch,
        max_workers=8,
    )
    assert result.matched_lat == 6.0
    assert result.matched_lon == -1.76
    print("test_fetch_persiann_3hourly_reports_matched_coordinate_concurrent: PASS")


def test_fetch_persiann_3hourly_concurrent_speedup():
    def slow_fetch(url, lat, lon, dt):
        time.sleep(0.03)
        return pd.DataFrame({"Timestamp": [pd.Timestamp(dt)], "PrecipitationMM": [0.1]})

    start = time.time()
    fetch_persiann_3hourly(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-05",
        fetch_fn=slow_fetch,
        max_workers=1,
    )
    serial_time = time.time() - start

    start = time.time()
    fetch_persiann_3hourly(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-05",
        fetch_fn=slow_fetch,
        max_workers=8,
    )
    concurrent_time = time.time() - start

    speedup = serial_time / concurrent_time
    assert speedup > 2.5, f"expected meaningful speedup, got only {speedup:.1f}x"
    print(
        f"test_fetch_persiann_3hourly_concurrent_speedup: PASS ({speedup:.1f}x speedup)"
    )


def test_fetch_persiann_3hourly_reports_missing_values():
    def fetch_with_missing(url, lat, lon, dt):
        value = float("nan") if dt.hour == 6 else 0.3
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(dt)], "PrecipitationMM": [value]}
        )

    result = fetch_persiann_3hourly(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        fetch_fn=fetch_with_missing,
        max_workers=1,
    )
    assert any(
        "had a value matching one of the candidate" in w for w in result.warnings
    )
    print("test_fetch_persiann_3hourly_reports_missing_values: PASS")


def test_fetch_persiann_3hourly_default_full_record_start():
    def fake_fetch(url, lat, lon, dt):
        return pd.DataFrame({"Timestamp": [pd.Timestamp(dt)], "PrecipitationMM": [0.1]})

    result = fetch_persiann_3hourly(
        lat=0, lon=0, end_date="1983-01-02", fetch_fn=fake_fetch, max_workers=1
    )
    assert result.dataframe["Timestamp"].min().date() == RECORD_START
    print("test_fetch_persiann_3hourly_default_full_record_start: PASS")


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
