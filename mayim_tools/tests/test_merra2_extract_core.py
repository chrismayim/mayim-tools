"""
Tests for merra2_extract/core.py. Run: python3 tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time

import pandas as pd

from mayim_tools.rainfall.merra2_extract.core import (
    DEFAULT_GIOVANNI_VARIABLE,
    MM_PER_SECOND_TO_MM_PER_HOUR,
    RECORD_START,
    FetchResult,
    _extend_bare_date_to_end_of_day,
    _parse_giovanni_csv,
    _split_date_range,
    fetch_merra2_timeseries,
    save_earthdata_credentials,
)

# ----------------------------------------------------------------------
# Constants - confirmed directly from research, not assumed
# ----------------------------------------------------------------------


def test_default_variable_matches_confirmed_giovanni_id():
    """Confirmed directly from Giovanni's own live WMS GetCapabilities
    listing, not a secondhand source - locks this in as a regression
    test so it can't drift silently."""
    assert DEFAULT_GIOVANNI_VARIABLE == "M2T1NXFLX_5_12_4_PRECTOTCORR"
    print("test_default_variable_matches_confirmed_giovanni_id: PASS")


def test_record_start_matches_confirmed_giovanni_range():
    assert RECORD_START == "1980-01-01"
    print("test_record_start_matches_confirmed_giovanni_range: PASS")


def test_unit_conversion_factor_confirmed_via_nasa_forum():
    """mm/hour = precip * 3600, confirmed directly from a NASA
    Earthdata Forum answer, not derived/assumed."""
    assert MM_PER_SECOND_TO_MM_PER_HOUR == 3600.0
    print("test_unit_conversion_factor_confirmed_via_nasa_forum: PASS")


# ----------------------------------------------------------------------
# Date range handling - reused logic from imerg_extract, re-tested here
# since this is a separate, independent copy (each plugin self-contained)
# ----------------------------------------------------------------------


def test_split_date_range_exact_multiple():
    chunks = _split_date_range(
        "2020-01-01T00:00:00", "2021-12-31T23:30:00", chunk_months=12
    )
    assert len(chunks) == 2
    print("test_split_date_range_exact_multiple: PASS")


def test_split_date_range_full_record_chunk_count():
    chunks = _split_date_range(RECORD_START, "2026-01-01", chunk_months=12)
    assert 44 <= len(chunks) <= 48  # ~46 years, 1980-2026
    print("test_split_date_range_full_record_chunk_count: PASS")


def test_split_date_range_rejects_backwards_range():
    try:
        _split_date_range("2020-01-01", "2019-01-01", chunk_months=12)
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("test_split_date_range_rejects_backwards_range: PASS")


def test_extend_bare_date_to_end_of_day():
    result = _extend_bare_date_to_end_of_day("2005-03-05")
    assert result == "2005-03-05T23:59:59"
    print("test_extend_bare_date_to_end_of_day: PASS")


def test_extend_bare_date_preserves_explicit_time():
    result = _extend_bare_date_to_end_of_day("2005-03-05T12:00:00")
    assert result == "2005-03-05T12:00:00"
    print("test_extend_bare_date_preserves_explicit_time: PASS")


# ----------------------------------------------------------------------
# Giovanni CSV parsing - the metadata-preamble-tolerant approach
# carried over from imerg_extract, re-tested against MERRA-2-flavoured
# synthetic responses (not a live-confirmed real response - see
# dev/README.md)
# ----------------------------------------------------------------------


def test_parse_giovanni_csv_plausible_merra2_metadata_preamble():
    """A plausible MERRA-2 Giovanni response shape, modelled on
    IMERG's actual real preamble structure (key,value metadata lines,
    not comment-marked) but with MERRA-2-flavoured field values -
    NOT a confirmed real response, a reasonable synthetic test given
    the same service produced IMERG's actual preamble shape."""
    text = (
        "prod_name,M2T1NXFLX_5_12_4_PRECTOTCORR\n"
        "doi,10.5067/0JRLVL8YV2Y4\n"
        "param_short_name,PRECTOTCORR\n"
        "param_name,Total precipitation bias corrected\n"
        "unit,kg m-2 s-1\n"
        "begin_time,2020-01-01 00:30:00\n"
        "end_time,2020-01-01 23:30:00\n"
        "lat,6.05\n"
        "lon,-1.73\n"
        "2020-01-01T00:30:00Z,0.0001\n"
        "2020-01-01T01:30:00Z,0.0002\n"
        "2020-01-01T02:30:00Z,0.0\n"
    )
    df = _parse_giovanni_csv(text)
    assert list(df.columns) == ["Timestamp", "PrecipitationMMHR"]
    assert len(df) == 3
    assert abs(df["PrecipitationMMHR"].iloc[0] - 0.36) < 1e-9  # 0.0001 * 3600
    assert abs(df["PrecipitationMMHR"].iloc[1] - 0.72) < 1e-9  # 0.0002 * 3600
    print("test_parse_giovanni_csv_plausible_merra2_metadata_preamble: PASS")


def test_parse_giovanni_csv_preserves_half_hour_offset_timestamps():
    """MERRA-2's real timestamp convention (hour-center, e.g. 00:30,
    01:30) must be preserved exactly, not silently re-aligned to the
    top of the hour."""
    text = "begin_time,2020-01-01\n2020-01-01T00:30:00Z,0.0001\n"
    df = _parse_giovanni_csv(text)
    assert df["Timestamp"].iloc[0] == pd.Timestamp("2020-01-01T00:30:00Z")
    print("test_parse_giovanni_csv_preserves_half_hour_offset_timestamps: PASS")


def test_parse_giovanni_csv_metadata_only_raises_clear_error():
    text = "prod_name,M2T1NXFLX_5_12_4_PRECTOTCORR\nunit,kg m-2 s-1\n"
    try:
        _parse_giovanni_csv(text)
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "No timestamped data rows found" in str(e)
        assert "prod_name" in str(e)
    print("test_parse_giovanni_csv_metadata_only_raises_clear_error: PASS")


def test_parse_giovanni_csv_empty_raises():
    try:
        _parse_giovanni_csv("# only comments\n")
        raise AssertionError("should have raised")
    except ValueError:
        pass
    print("test_parse_giovanni_csv_empty_raises: PASS")


def test_parse_giovanni_csv_tolerates_extra_label_column():
    """Same defensive tolerance as imerg_extract: a data row with an
    extra column between timestamp and value (e.g. a repeated product
    label) is handled by taking the first field as timestamp and the
    LAST as value, not assuming exactly 2 fields."""
    text = (
        "2020-01-01T00:30:00Z,PRECTOTCORR,0.0001\n"
        "2020-01-01T01:30:00Z,PRECTOTCORR,0.0002\n"
    )
    df = _parse_giovanni_csv(text)
    assert len(df) == 2
    assert abs(df["PrecipitationMMHR"].iloc[0] - 0.36) < 1e-9
    print("test_parse_giovanni_csv_tolerates_extra_label_column: PASS")


# ----------------------------------------------------------------------
# Full orchestration, via injected fakes - no real network/credentials
# ----------------------------------------------------------------------


def test_fetch_merra2_timeseries_orchestration_with_fake_network():
    call_log = []

    def fake_fetch(lat, lon, start_iso, end_iso, variable, token):
        call_log.append((start_iso, end_iso, variable))
        idx = pd.date_range(start_iso, end_iso, freq="h")
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 0.5})

    result = fetch_merra2_timeseries(
        lat=6.0539,
        lon=-1.7334,
        start_date="2020-01-01",
        end_date="2021-12-31",
        chunk_months=12,
        token="fake-token",
        fetch_chunk_fn=fake_fetch,
        max_workers=1,
    )
    assert len(call_log) == 2
    assert all(c[2] == DEFAULT_GIOVANNI_VARIABLE for c in call_log)
    assert result.dataframe["Timestamp"].is_monotonic_increasing
    assert result.dataframe["Timestamp"].duplicated().sum() == 0
    print("test_fetch_merra2_timeseries_orchestration_with_fake_network: PASS")


def test_fetch_merra2_timeseries_covers_full_end_date():
    """The same bare-end-date fix applied proactively - confirms it
    works for this plugin's own orchestration, not just imerg_extract's."""
    captured = {}

    def fake_fetch(lat, lon, start_iso, end_iso, variable, token):
        captured["end"] = end_iso
        idx = pd.date_range(start_iso, end_iso, freq="h")
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 0.1})

    fetch_merra2_timeseries(
        lat=6.0539,
        lon=-1.7334,
        start_date="2005-03-01",
        end_date="2005-03-05",
        chunk_months=12,
        token="fake-token",
        fetch_chunk_fn=fake_fetch,
        max_workers=1,
    )
    assert captured["end"] == "2005-03-05T23:59:59"
    print("test_fetch_merra2_timeseries_covers_full_end_date: PASS")


def test_fetch_merra2_timeseries_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky_fetch(lat, lon, start_iso, end_iso, variable, token):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionError("simulated transient failure")
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(start_iso)], "PrecipitationMMHR": [0.2]}
        )

    result = fetch_merra2_timeseries(
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
    print("test_fetch_merra2_timeseries_retries_then_succeeds: PASS")


def test_fetch_merra2_timeseries_reports_failure_after_max_retries():
    def always_fails(lat, lon, start_iso, end_iso, variable, token):
        raise ConnectionError("simulated permanent failure")

    result = fetch_merra2_timeseries(
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
    print("test_fetch_merra2_timeseries_reports_failure_after_max_retries: PASS")


def test_fetch_merra2_timeseries_flags_gaps():
    def sparse_fetch(lat, lon, start_iso, end_iso, variable, token):
        return pd.DataFrame(
            {"Timestamp": [pd.Timestamp(start_iso)], "PrecipitationMMHR": [0.1]}
        )

    result = fetch_merra2_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-05",
        chunk_months=12,
        token="fake-token",
        fetch_chunk_fn=sparse_fetch,
        max_workers=1,
    )
    assert any("expected ~" in w for w in result.warnings)
    print("test_fetch_merra2_timeseries_flags_gaps: PASS")


def test_fetch_merra2_timeseries_concurrent_speedup():
    def slow_fetch(lat, lon, start_iso, end_iso, variable, token):
        time.sleep(0.05)
        idx = pd.date_range(start_iso, end_iso, freq="h")
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 0.3})

    start = time.time()
    fetch_merra2_timeseries(
        lat=0,
        lon=0,
        start_date="1980-01-01",
        end_date="1988-01-01",
        chunk_months=12,
        token="t",
        fetch_chunk_fn=slow_fetch,
        max_workers=1,
    )
    serial_time = time.time() - start

    start = time.time()
    fetch_merra2_timeseries(
        lat=0,
        lon=0,
        start_date="1980-01-01",
        end_date="1988-01-01",
        chunk_months=12,
        token="t",
        fetch_chunk_fn=slow_fetch,
        max_workers=8,
    )
    concurrent_time = time.time() - start

    speedup = serial_time / concurrent_time
    assert speedup > 2.5, f"expected meaningful speedup, got only {speedup:.1f}x"
    print(
        "test_fetch_merra2_timeseries_concurrent_speedup: PASS "
        f"({speedup:.1f}x speedup)"
    )


def test_fetch_merra2_timeseries_default_full_record():
    def fake_fetch(lat, lon, start_iso, end_iso, variable, token):
        idx = pd.date_range(
            start_iso,
            min(pd.Timestamp(end_iso), pd.Timestamp(start_iso) + pd.Timedelta(days=2)),
            freq="h",
        )
        return pd.DataFrame({"Timestamp": idx, "PrecipitationMMHR": 0.1})

    result = fetch_merra2_timeseries(
        lat=0,
        lon=0,
        end_date="1980-01-05",
        token="t",
        fetch_chunk_fn=fake_fetch,
        max_workers=1,
    )
    assert result.dataframe["Timestamp"].min().date().isoformat() == RECORD_START
    print("test_fetch_merra2_timeseries_default_full_record: PASS")


# ----------------------------------------------------------------------
# Earthdata credential saving - identical logic/tests to imerg_extract,
# re-verified here since this is a separate, self-contained copy
# ----------------------------------------------------------------------


def test_save_earthdata_credentials_preserves_other_machine_entries():
    import tempfile

    tmp_path = Path(tempfile.mkdtemp()) / ".netrc"
    tmp_path.write_text("machine example.com\nlogin someoneelse\npassword unrelated\n")

    save_earthdata_credentials("myuser", "mypass", config_path=tmp_path)
    content = tmp_path.read_text()

    assert "machine example.com" in content
    assert "machine urs.earthdata.nasa.gov" in content
    assert "login myuser" in content
    print("test_save_earthdata_credentials_preserves_other_machine_entries: PASS")


def test_save_earthdata_credentials_writes_both_files_on_windows():
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
    print("test_save_earthdata_credentials_writes_both_files_on_windows: PASS")


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
