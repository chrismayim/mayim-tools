"""
Tests for mayim_tools/rainfall/era5_extract/core.py and export.py.

Heaviest weight on deaccumulate() - the single most safety-critical
piece of this module. Despite its name (kept for minimal disruption
to calling code), it no longer de-accumulates anything - see its own
docstring in core.py for the full evidence trail. This was rebuilt
three times before the real root cause was found: this tool's CDS
request (queried by valid time, not forecast step) already returns
each hour's correct, final precipitation value directly - the
previous differencing logic was taking two independently-correct
values and subtracting one from the other, producing silently
incorrect results, not an error. Confirmed directly against a real
side-by-side comparison against the CDS website, not reasoned about
in the abstract.

Also covers: request construction for both products, year-chunking,
zip-response defensiveness, the full orchestration logic via injected
fake cds_retrieve_fn/read_fn (no real CDS credentials or network
access used anywhere in this file), GRIB point-extraction logic
against a synthetic in-memory xarray Dataset matching cfgrib's real
ERA5 output structure, CDS credential saving, the manual CDS result
download step, and CSV export.
"""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import mayim_tools.rainfall.era5_extract.core as era5_core_module
from mayim_tools.rainfall.era5_extract.core import (
    Era5Result,
    build_request,
    chunk_years,
    deaccumulate,
    download_result_manually,
    extract_grib_from_zip,
    extract_point_series,
    fetch_era5_point,
    is_zip_file,
    point_to_area,
    save_cds_credentials,
)
from mayim_tools.rainfall.era5_extract.export import (
    write_era5_anomalies,
    write_era5_csv,
)

# ----------------------------------------------------------------------
# Deaccumulation (now a pass-through, not a differencing step) - the
# critical piece
# ----------------------------------------------------------------------


def test_deaccumulate_matches_real_cds_website_reference():
    """The actual real-world regression test, and the direct evidence
    behind this rebuild: a genuine anomaly row shared directly
    (reference_time=1950-01-03 06:00, step=8, accumulated_mm=0.6466,
    precipitation_mm_raw=-1.193 under the OLD, now-removed
    differencing logic) implies, by that old arithmetic, a step-7
    accumulated_mm of 0.6466 - (-1.193) = 1.8396. The CDS website's
    own reported value for that same hour is 1.8396378 - matching to
    four decimal places. This test reconstructs that exact real data
    directly (all values below are the CDS website's own true hourly
    figures for 1950-01-01 through 1950-01-02) and confirms
    deaccumulate() now passes them through as precipitation_mm
    unchanged - no differencing, no distortion."""
    website_true_hourly_mm = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.02193451,
        0.002384186,
        0.009059906,
        0.000476837,
        0.04673004,
        0.03194809,
        0.027656555,
        0.005722046,
        0.04053116,
        0.25844574,
        0.000953674,
        0.002384186,
        0.00667572,
        0.007629395,
        0.008583069,
        0.017166138,
        0.014305115,
        0.020503998,
        0.009536743,
        0.008106232,
        0.003814697,
        0.001907349,
        0,
        0,
        0,
        0,
        0.05531311,
        0.046253204,
        0.012874603,
        0,
        0,
        0,
        0,
    ]
    ref = pd.Timestamp("1950-01-01 00:00")
    valid_times = pd.date_range(ref, periods=len(website_true_hourly_mm), freq="h")
    df = pd.DataFrame(
        {
            "reference_time": [ref] * len(website_true_hourly_mm),
            "step_hours": range(len(website_true_hourly_mm)),
            "valid_time": valid_times,
            "accumulated_mm": website_true_hourly_mm,
        }
    )

    result, anomaly_rows = deaccumulate(df)
    recovered = result.sort_values("valid_time")["precipitation_mm"].to_numpy()

    assert np.allclose(recovered, website_true_hourly_mm, atol=1e-9)
    assert len(anomaly_rows) == 0


def test_deaccumulate_reconstructs_the_actual_anomaly_that_caught_this_bug():
    """Directly reproduces the arithmetic that revealed the real bug:
    a genuine value of 1.8396 (the CDS website's true hourly figure)
    must now pass through as 1.8396 - NOT be differenced against a
    neighbouring hour to produce -1.193, as the old (removed) logic
    did."""
    ref = pd.Timestamp("1950-01-03 06:00")
    df = pd.DataFrame(
        {
            "reference_time": [ref, ref],
            "step_hours": [7, 8],
            "valid_time": [
                pd.Timestamp("1950-01-03 13:00"),
                pd.Timestamp("1950-01-03 14:00"),
            ],
            "accumulated_mm": [1.8396378, 0.6465912],
        }
    )

    result, anomaly_rows = deaccumulate(df)
    recovered = result.sort_values("valid_time")["precipitation_mm"].to_numpy()

    assert np.allclose(recovered, [1.8396378, 0.6465912], atol=1e-9)
    assert len(anomaly_rows) == 0


def test_deaccumulate_is_no_longer_order_or_grouping_dependent():
    """Since no differencing occurs, shuffling row order and using
    arbitrary/inconsistent reference_time values must not change any
    row's own precipitation_mm - a direct confirmation that
    reference_time is no longer used for grouping or differencing at
    all, just carried through as an audit column."""
    df = pd.DataFrame(
        {
            "reference_time": [
                pd.Timestamp("2020-01-01 18:00"),
                pd.Timestamp("2020-01-01 06:00"),
                pd.Timestamp("2099-12-31 00:00"),
            ],
            "step_hours": [99, -5, 0],
            "valid_time": [
                pd.Timestamp("2020-01-01 09:00"),
                pd.Timestamp("2020-01-01 10:00"),
                pd.Timestamp("2020-01-01 11:00"),
            ],
            "accumulated_mm": [0.5, 1.5, 0.2],
        }
    )

    result, _ = deaccumulate(df)

    for vt, expected in zip(
        [
            pd.Timestamp("2020-01-01 09:00"),
            pd.Timestamp("2020-01-01 10:00"),
            pd.Timestamp("2020-01-01 11:00"),
        ],
        [0.5, 1.5, 0.2],
        strict=True,
    ):
        row = result[result["valid_time"] == vt]
        assert row["precipitation_mm"].iloc[0] == expected


def test_deaccumulate_small_negative_clipped_to_zero():
    """A genuinely negative raw hourly value within GRIB-packing
    tolerance (e.g. -0.001mm where the true value is 0) should clip to
    zero, not propagate as nonsensical negative rainfall."""
    ref = pd.Timestamp("2020-01-01")
    df = pd.DataFrame(
        {
            "reference_time": [ref, ref],
            "step_hours": [1, 2],
            "valid_time": [ref + pd.Timedelta(hours=1), ref + pd.Timedelta(hours=2)],
            "accumulated_mm": [1.0, -0.001],
        }
    )

    result, anomaly_rows = deaccumulate(df)

    assert len(anomaly_rows) == 0
    assert result["precipitation_mm"].iloc[1] == 0.0


def test_deaccumulate_large_negative_not_silently_clipped():
    """A large negative raw hourly value is a real anomaly worth
    seeing (now expected to be rare, since it's no longer an artifact
    of differencing) - must be flagged, not hidden."""
    ref = pd.Timestamp("2020-01-01")
    df = pd.DataFrame(
        {
            "reference_time": [ref, ref],
            "step_hours": [1, 2],
            "valid_time": [ref + pd.Timedelta(hours=1), ref + pd.Timedelta(hours=2)],
            "accumulated_mm": [1.0, -2.0],
        }
    )

    result, anomaly_rows = deaccumulate(df)

    assert len(anomaly_rows) == 1
    assert result["precipitation_mm"].iloc[1] == -2.0


def test_deaccumulate_anomaly_rows_carry_full_diagnostic_context():
    """The returned anomaly_rows DataFrame must carry full context
    (reference_time, step_hours, valid_time, the raw value), not just
    a count, so the actual magnitude and timing can be inspected
    directly."""
    ref = pd.Timestamp("2020-01-01 06:00")
    df = pd.DataFrame(
        {
            "reference_time": [ref, ref],
            "step_hours": [1, 2],
            "valid_time": [ref + pd.Timedelta(hours=1), ref + pd.Timedelta(hours=2)],
            "accumulated_mm": [1.0, -2.0],
        }
    )

    result, anomaly_rows = deaccumulate(df)

    assert len(anomaly_rows) == 1
    row = anomaly_rows.iloc[0]
    assert row["reference_time"] == pd.Timestamp("2020-01-01 06:00")
    assert row["step_hours"] == 2
    assert row["valid_time"] == pd.Timestamp("2020-01-01 08:00")
    assert row["accumulated_mm"] == -2.0
    assert row["precipitation_mm_raw"] == -2.0


def test_deaccumulate_reports_missing_raw_values():
    """A genuinely missing (NaN) raw value must be counted (via
    df.attrs['n_missing_raw']) and affects ONLY its own row now - no
    propagation to a neighbouring hour, since there is no longer any
    differencing chain for it to propagate through."""
    ref = pd.Timestamp("2020-01-01")
    steps = [1, 2, 3]
    df = pd.DataFrame(
        {
            "reference_time": [ref] * 3,
            "step_hours": steps,
            "valid_time": [ref + pd.Timedelta(hours=h) for h in steps],
            "accumulated_mm": [1.0, float("nan"), 3.0],
        }
    )

    result, _ = deaccumulate(df)

    assert result.attrs["n_missing_raw"] == 1
    assert pd.isna(result["precipitation_mm"].iloc[1])
    assert not pd.isna(result["precipitation_mm"].iloc[2])
    assert result["precipitation_mm"].iloc[2] == 3.0


def test_deaccumulate_no_missing_raw_reports_zero():
    ref = pd.Timestamp("2020-01-01")
    df = pd.DataFrame(
        {
            "reference_time": [ref] * 2,
            "step_hours": [1, 2],
            "valid_time": [ref + pd.Timedelta(hours=1), ref + pd.Timedelta(hours=2)],
            "accumulated_mm": [1.0, 2.0],
        }
    )

    result, _ = deaccumulate(df)

    assert result.attrs["n_missing_raw"] == 0


def test_deaccumulate_carries_valid_time_through_unchanged():
    ref = pd.Timestamp("2020-01-01 00:00")
    vt = pd.Timestamp("2020-01-02 00:00")
    df = pd.DataFrame(
        {
            "reference_time": [ref],
            "step_hours": [24],
            "valid_time": [vt],
            "accumulated_mm": [5.0],
        }
    )

    result, _ = deaccumulate(df)

    assert result["valid_time"].iloc[0] == vt


# ----------------------------------------------------------------------
# Request construction
# ----------------------------------------------------------------------


def test_build_request_era5():
    dataset, request = build_request("era5", years=[2020], area=[1, -1, -1, 1])

    assert dataset == "reanalysis-era5-single-levels"
    assert request["product_type"] == ["reanalysis"]
    assert request["variable"] == ["total_precipitation"]
    assert request["year"] == ["2020"]
    assert len(request["month"]) == 12
    assert len(request["day"]) == 31
    assert len(request["time"]) == 24
    assert request["data_format"] == "grib"
    assert request["download_format"] == "unarchived"


def test_build_request_era5_land_no_product_type():
    """era5-land has only one product type (a plain rerun) - the
    request schema doesn't take a product_type key at all."""
    dataset, request = build_request("era5_land", years=[2020], area=[1, -1, -1, 1])

    assert dataset == "reanalysis-era5-land"
    assert "product_type" not in request


def test_build_request_multi_year():
    _, request = build_request("era5", years=[2019, 2020, 2021], area=[1, -1, -1, 1])

    assert request["year"] == ["2019", "2020", "2021"]


def test_build_request_unknown_product_raises():
    with pytest.raises(ValueError):
        build_request("bogus", years=[2020], area=[1, -1, -1, 1])


def test_point_to_area():
    area = point_to_area(lat=5.0, lon=10.0, buffer_deg=0.2)
    assert area == [5.2, 9.8, 4.8, 10.2]  # [N, W, S, E]


def test_chunk_years_single_year_chunks():
    chunks = chunk_years(date(2018, 1, 1), date(2020, 12, 31), years_per_chunk=1)
    assert chunks == [[2018], [2019], [2020]]


def test_chunk_years_multi_year_chunks():
    chunks = chunk_years(date(2018, 1, 1), date(2023, 12, 31), years_per_chunk=3)
    assert chunks == [[2018, 2019, 2020], [2021, 2022, 2023]]


def test_chunk_years_rejects_backwards_range():
    with pytest.raises(ValueError):
        chunk_years(date(2020, 1, 1), date(2018, 1, 1))


# ----------------------------------------------------------------------
# Zip-response defensiveness
# ----------------------------------------------------------------------


def test_zip_detection_and_extraction(tmp_path):
    grib_content = b"fake grib bytes for testing"
    zip_path = tmp_path / "response.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data.grib", grib_content)

    assert is_zip_file(zip_path)
    extracted_path = extract_grib_from_zip(zip_path, tmp_path)
    assert Path(extracted_path).read_bytes() == grib_content


def test_non_zip_file_not_detected_as_zip(tmp_path):
    plain_path = tmp_path / "response.grib"
    plain_path.write_bytes(b"not a zip")
    assert not is_zip_file(plain_path)


def test_zip_with_multiple_files_raises(tmp_path):
    zip_path = tmp_path / "response.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data1.grib", b"a")
        zf.writestr("data2.grib", b"b")

    with pytest.raises(ValueError):
        extract_grib_from_zip(zip_path, tmp_path)


# ----------------------------------------------------------------------
# Full orchestration, via injected fakes - no real CDS access
# ----------------------------------------------------------------------


def test_fetch_era5_point_orchestration_with_fakes(tmp_path):
    calls = []

    def fake_retrieve(dataset, request, target_path):
        calls.append((dataset, request["year"]))
        Path(target_path).write_bytes(b"fake grib")

    def fake_read(path, lat, lon):
        year = int(calls[-1][1][0])
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        rows = []
        for d in dates:
            for step in range(1, 25):
                rows.append(
                    {
                        "reference_time": d,
                        "step_hours": step,
                        "valid_time": d + pd.Timedelta(hours=step),
                        "accumulated_mm": 1.0,
                    }
                )
        return pd.DataFrame(rows)

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2019-06-01",
        end_date="2020-06-30",
        product="era5_land",
        years_per_chunk=1,
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tmp_path,
    )

    assert len(calls) == 2  # 2019 and 2020 requested separately
    assert result.dataframe["ValidTime"].min().date().isoformat() == "2019-06-01"
    assert result.dataframe["ValidTime"].max().date().isoformat() == "2020-06-30"
    assert (result.dataframe["PrecipitationMM"] == 1.0).all()
    assert result.dataframe["ValidTime"].is_monotonic_increasing


def test_fetch_era5_point_partial_chunk_failure_isolated(tmp_path):
    def flaky_retrieve(dataset, request, target_path):
        if request["year"] == ["2020"]:
            raise ConnectionError("simulated CDS failure")
        Path(target_path).write_bytes(b"fake grib")

    def fake_read(path, lat, lon):
        return pd.DataFrame(
            {
                "reference_time": [pd.Timestamp("2019-01-01")],
                "step_hours": [1],
                "valid_time": [pd.Timestamp("2019-01-01 01:00")],
                "accumulated_mm": [3.0],
            }
        )

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2019-01-01",
        end_date="2020-12-31",
        product="era5",
        years_per_chunk=1,
        cds_retrieve_fn=flaky_retrieve,
        read_fn=fake_read,
        work_dir=tmp_path,
    )

    assert len(result.dataframe) == 1
    assert any("2020" in w for w in result.warnings)


def test_fetch_era5_point_rejects_unknown_product():
    with pytest.raises(ValueError):
        fetch_era5_point(lat=0, lon=0, product="bogus")


def test_fetch_era5_point_default_full_record(tmp_path):
    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_bytes(b"fake")

    def fake_read(path, lat, lon):
        return pd.DataFrame(
            {
                "reference_time": [pd.Timestamp("1950-01-01")],
                "step_hours": [1],
                "valid_time": [pd.Timestamp("1950-01-01 01:00")],
                "accumulated_mm": [1.0],
            }
        )

    result = fetch_era5_point(
        lat=0,
        lon=0,
        end_date="1950-01-05",
        product="era5",
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tmp_path,
    )

    assert len(result.dataframe) >= 1


# ----------------------------------------------------------------------
# GRIB extraction logic - tested against a synthetic in-memory xarray
# Dataset matching cfgrib's real ERA5 output structure (time, step,
# latitude, longitude dims; step as timedelta64), not a real GRIB file.
# ----------------------------------------------------------------------


def make_synthetic_era5_dataset(var_name="tp"):
    xr = pytest.importorskip("xarray")

    lats = np.array([-0.1, 0.0, 0.1])
    lons = np.array([-0.1, 0.0, 0.1])
    times = pd.to_datetime(["2020-01-01", "2020-01-02"])
    steps = pd.to_timedelta([1, 2, 3], unit="h")

    # shape: (time, step, lat, lon) - accumulated precip in METRES
    # (ERA5's native unit - extract_point_series must convert to mm)
    data = np.zeros((2, 3, 3, 3), dtype="float64")
    data[0, :, 1, 1] = [0.001, 0.0015, 0.003]  # centre cell, day 1
    data[1, :, 1, 1] = [0.0005, 0.001, 0.002]  # day 2

    return xr.Dataset(
        {var_name: (["time", "step", "latitude", "longitude"], data)},
        coords={"time": times, "step": steps, "latitude": lats, "longitude": lons},
    )


def test_extract_point_series_converts_metres_to_mm():
    ds = make_synthetic_era5_dataset()
    df = extract_point_series(ds, lat=0.0, lon=0.0)

    day1 = df[df["reference_time"] == pd.Timestamp("2020-01-01")].sort_values(
        "step_hours"
    )
    assert np.allclose(day1["accumulated_mm"].to_numpy(), [1.0, 1.5, 3.0])


def test_extract_point_series_nearest_neighbour():
    ds = make_synthetic_era5_dataset()
    # query a point slightly off-grid - should snap to the centre cell (0,0)
    df = extract_point_series(ds, lat=0.02, lon=-0.03)

    assert len(df) == 6  # 2 days x 3 steps
    day1 = df[df["reference_time"] == pd.Timestamp("2020-01-01")].sort_values(
        "step_hours"
    )
    assert np.allclose(day1["accumulated_mm"].to_numpy(), [1.0, 1.5, 3.0])


def test_extract_point_series_variable_name_fallback():
    """Confirms the variable-name candidate list works if a dataset
    uses a different name than 'tp' (defensive against ERA5/cfgrib
    naming differences across versions, same principle as IMERG's
    precipitation/precipitationCal fallback)."""
    ds = make_synthetic_era5_dataset(var_name="precipitation")
    df = extract_point_series(ds, lat=0.0, lon=0.0)
    assert len(df) == 6


def test_extract_point_series_missing_variable_raises():
    ds = make_synthetic_era5_dataset(var_name="something_else")
    with pytest.raises(ValueError):
        extract_point_series(ds, lat=0.0, lon=0.0)


def test_extract_point_series_trusts_native_valid_time_over_recomputed():
    """The actual bug this rebuild fixes: a real downloaded GRIB
    file's own valid_time field must be used directly, not recomputed
    from time+step. Constructs a synthetic dataset whose OWN
    valid_time coordinate deliberately does NOT match naive time+step
    arithmetic, and confirms the mismatched, authoritative value is
    what comes through - not the recomputed one."""
    xr = pytest.importorskip("xarray")

    lats = np.array([-0.1, 0.0, 0.1])
    lons = np.array([-0.1, 0.0, 0.1])
    times = pd.to_datetime(["2020-01-01 06:00"])
    steps = pd.to_timedelta([1], unit="h")
    data = np.zeros((1, 1, 3, 3), dtype="float64")
    data[0, 0, 1, 1] = 0.001

    # deliberately WRONG relative to naive time+step (which would give
    # 07:00) - simulates a file whose own valid_time field must be
    # trusted over recomputing it
    deliberately_different_valid_time = pd.to_datetime(["2020-01-01 07:30"])
    ds = xr.Dataset(
        {"tp": (["time", "step", "latitude", "longitude"], data)},
        coords={
            "time": times,
            "step": steps,
            "latitude": lats,
            "longitude": lons,
            "valid_time": ("time", deliberately_different_valid_time),
        },
    )

    df = extract_point_series(ds, lat=0.0, lon=0.0)

    assert df["valid_time"].iloc[0] == deliberately_different_valid_time[0]


def test_extract_point_series_falls_back_when_valid_time_absent():
    """Defensive fallback: a dataset genuinely lacking a valid_time
    coordinate must still work, computing it from time+step as before
    - confirmed by every other extract_point_series test above, all of
    which use a synthetic dataset with no valid_time coordinate."""
    ds = make_synthetic_era5_dataset()
    df = extract_point_series(ds, lat=0.0, lon=0.0)

    row = df[
        (df["reference_time"] == pd.Timestamp("2020-01-01")) & (df["step_hours"] == 2)
    ].iloc[0]
    assert row["valid_time"] == pd.Timestamp("2020-01-01 02:00")


def test_extract_point_series_feeds_deaccumulate_end_to_end():
    """Full pipeline test: synthetic dataset -> extract -> deaccumulate
    -> confirms values pass through unchanged, exercising the actual
    join between these two functions, not just each in isolation."""
    ds = make_synthetic_era5_dataset()
    raw = extract_point_series(ds, lat=0.0, lon=0.0)
    result, anomaly_rows = deaccumulate(raw)

    assert len(anomaly_rows) == 0
    day1 = result[result["reference_time"] == pd.Timestamp("2020-01-01")].sort_values(
        "step_hours"
    )
    assert np.allclose(day1["precipitation_mm"].to_numpy(), [1.0, 1.5, 3.0])


# ----------------------------------------------------------------------
# CDS credential saving - "enter once via the QGIS parameter, remember
# for next time" behaviour
# ----------------------------------------------------------------------


def test_save_cds_credentials_writes_expected_format(tmp_path):
    tmp_config = tmp_path / ".cdsapirc"
    result_path = save_cds_credentials("my-secret-token-123", config_path=tmp_config)

    assert result_path == tmp_config
    content = tmp_config.read_text()
    assert (
        content
        == "url: https://cds.climate.copernicus.eu/api\nkey: my-secret-token-123\n"
    )


def test_save_cds_credentials_overwrites_cleanly(tmp_path):
    """A second save with a different key must fully replace the
    first, not append to it."""
    tmp_config = tmp_path / ".cdsapirc"
    save_cds_credentials("old-token", config_path=tmp_config)
    save_cds_credentials("new-token", config_path=tmp_config)

    content = tmp_config.read_text()
    assert "old-token" not in content
    assert "key: new-token" in content


def test_fetch_era5_point_saves_provided_key_before_running(tmp_path):
    """Regression-style test for the actual feature requested: an
    api_key passed to fetch_era5_point() must be persisted to the
    credentials file BEFORE the retrieve function runs, so even a
    fake cds_retrieve_fn (which never touches the real cdsapi client
    at all) still exercises the save-then-clear-key behaviour."""
    tmp_config = tmp_path / ".cdsapirc"

    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_bytes(b"fake")

    def fake_read(path, lat, lon):
        return pd.DataFrame(
            {
                "reference_time": [pd.Timestamp("2020-01-01")],
                "step_hours": [1],
                "valid_time": [pd.Timestamp("2020-01-01 01:00")],
                "accumulated_mm": [1.0],
            }
        )

    fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        product="era5",
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tmp_path,
        api_key="entered-via-qgis-parameter",
        credentials_path=tmp_config,
    )

    assert tmp_config.exists()
    assert "entered-via-qgis-parameter" in tmp_config.read_text()


def test_fetch_era5_point_save_credentials_false_skips_saving(tmp_path):
    """save_credentials=False must use the key for this run only,
    without writing it to disk at all."""
    tmp_config = tmp_path / ".cdsapirc"

    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_bytes(b"fake")

    def fake_read(path, lat, lon):
        return pd.DataFrame(
            {
                "reference_time": [pd.Timestamp("2020-01-01")],
                "step_hours": [1],
                "valid_time": [pd.Timestamp("2020-01-01 01:00")],
                "accumulated_mm": [1.0],
            }
        )

    fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        product="era5",
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tmp_path,
        api_key="one-time-key",
        save_credentials=False,
        credentials_path=tmp_config,
    )

    assert not tmp_config.exists()


# ----------------------------------------------------------------------
# Manual CDS result download - replaces dependence on cdsapi's own
# Result.download()/_download().
# ----------------------------------------------------------------------


class _FakeCdsResult:
    def __init__(self, location, content_length):
        self.location = location
        self.content_length = content_length


class _FakeStreamedResponse:
    """Mimics requests.Response enough for download_result_manually:
    raise_for_status() and iter_content()."""

    def __init__(self, chunks, status_ok=True):
        self._chunks = chunks
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            import requests

            raise requests.exceptions.HTTPError("simulated HTTP error")

    def iter_content(self, chunk_size=1024):
        yield from self._chunks


def test_download_result_manually_writes_correct_bytes(tmp_path):
    target_path = tmp_path / "result.grib"
    fake_bytes = b"fake grib content, twenty bytes"
    fake_result = _FakeCdsResult(
        location="https://example.com/fake.grib", content_length=len(fake_bytes)
    )
    fake_response = _FakeStreamedResponse(chunks=[fake_bytes[:10], fake_bytes[10:]])

    with mock.patch("requests.get", return_value=fake_response) as mock_get:
        download_result_manually(fake_result, target_path)

    mock_get.assert_called_once()
    assert mock_get.call_args[0][0] == "https://example.com/fake.grib"
    assert target_path.read_bytes() == fake_bytes


def test_download_result_manually_detects_size_mismatch(tmp_path):
    target_path = tmp_path / "result.grib"
    fake_result = _FakeCdsResult(
        location="https://example.com/fake.grib", content_length=100
    )
    fake_response = _FakeStreamedResponse(chunks=[b"only ten b"])  # 10 bytes, not 100

    with mock.patch("requests.get", return_value=fake_response):
        with pytest.raises(RuntimeError, match="10.*100|100.*10"):
            download_result_manually(fake_result, target_path)


def test_download_result_manually_propagates_http_errors(tmp_path):
    target_path = tmp_path / "result.grib"
    fake_result = _FakeCdsResult(
        location="https://example.com/fake.grib", content_length=10
    )
    fake_response = _FakeStreamedResponse(chunks=[], status_ok=False)

    with mock.patch("requests.get", return_value=fake_response):
        import requests

        with pytest.raises(requests.exceptions.HTTPError):
            download_result_manually(fake_result, target_path)


def test_download_result_manually_skips_size_check_when_length_unknown(tmp_path):
    """A result with no content_length (or 0/None) shouldn't spuriously
    fail the size check - only compare when a real expected size is
    actually known."""
    target_path = tmp_path / "result.grib"
    fake_result = _FakeCdsResult(
        location="https://example.com/fake.grib", content_length=None
    )
    fake_response = _FakeStreamedResponse(chunks=[b"whatever size this is"])

    with mock.patch("requests.get", return_value=fake_response):
        download_result_manually(fake_result, target_path)  # must not raise

    assert target_path.read_bytes() == b"whatever size this is"


# ----------------------------------------------------------------------
# CSV export - the actual crash fix (iterrows() -> itertuples()) and
# the anomalies audit export.
# ----------------------------------------------------------------------


def test_write_era5_csv_handles_missing_values_without_crashing(tmp_path):
    """Direct reproduction of the real crash: a row with a genuinely
    missing (NaN) PrecipitationMM value must be written as an empty
    cell, not crash - iterrows() silently coerced the NaN to NaT
    before switching to itertuples()."""
    df = pd.DataFrame(
        {
            "ValidTime": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
            ),
            "PrecipitationMM": [1.5, float("nan"), 2.5],
        }
    )
    result = Era5Result(dataframe=df, product="era5")
    out_path = tmp_path / "out.csv"

    n_rows, n_missing = write_era5_csv({"Site A": result}, out_path)

    assert n_rows == 3
    assert n_missing == 1
    lines = out_path.read_text().splitlines()
    assert lines[2].endswith(",")  # the NaN row's PrecipitationMM cell is empty


def test_write_era5_csv_no_missing_values_reports_zero(tmp_path):
    df = pd.DataFrame(
        {
            "ValidTime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
            "PrecipitationMM": [1.5, 2.5],
        }
    )
    result = Era5Result(dataframe=df, product="era5")
    out_path = tmp_path / "out.csv"

    n_rows, n_missing = write_era5_csv({"Site A": result}, out_path)

    assert n_rows == 2
    assert n_missing == 0


def test_write_era5_anomalies_writes_full_detail(tmp_path):
    anomalies = pd.DataFrame(
        {
            "reference_time": [pd.Timestamp("2020-01-01 06:00")],
            "step_hours": [2],
            "valid_time": [pd.Timestamp("2020-01-01 08:00")],
            "accumulated_mm": [2.0],
            "precipitation_mm_raw": [-8.0],
        }
    )
    result = Era5Result(dataframe=pd.DataFrame(), product="era5", anomalies=anomalies)
    out_path = tmp_path / "anomalies.csv"

    n_rows = write_era5_anomalies({"Site A": result}, out_path)

    assert n_rows == 1
    content = out_path.read_text()
    assert "Site A" in content
    assert "-8.0" in content


def test_write_era5_anomalies_empty_when_no_anomalies(tmp_path):
    result = Era5Result(
        dataframe=pd.DataFrame(),
        product="era5",
        anomalies=pd.DataFrame(
            columns=[
                "reference_time",
                "step_hours",
                "valid_time",
                "accumulated_mm",
                "precipitation_mm_raw",
            ]
        ),
    )
    out_path = tmp_path / "anomalies.csv"

    n_rows = write_era5_anomalies({"Site A": result}, out_path)

    assert n_rows == 0


def test_fetch_era5_point_aggregates_anomalies_and_missing_across_chunks(tmp_path):
    """Integration-level confirmation that both diagnostics survive
    the full multi-chunk orchestration, not just the isolated
    deaccumulate() call."""

    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_bytes(b"fake")

    call_count = {"n": 0}

    def fake_read(path, lat, lon):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return pd.DataFrame(
                {
                    "reference_time": [pd.Timestamp("2019-01-01")] * 2,
                    "step_hours": [1, 2],
                    "valid_time": [
                        pd.Timestamp("2019-01-01 01:00"),
                        pd.Timestamp("2019-01-01 02:00"),
                    ],
                    "accumulated_mm": [10.0, -2.0],
                }
            )
        return pd.DataFrame(
            {
                "reference_time": [pd.Timestamp("2020-01-01")] * 2,
                "step_hours": [1, 2],
                "valid_time": [
                    pd.Timestamp("2020-01-01 01:00"),
                    pd.Timestamp("2020-01-01 02:00"),
                ],
                "accumulated_mm": [1.0, float("nan")],
            }
        )

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2019-01-01",
        end_date="2020-12-31",
        product="era5",
        years_per_chunk=1,
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tmp_path,
    )

    assert len(result.anomalies) == 1
    assert any("missing" in w and "1 raw precipitation" in w for w in result.warnings)
    assert any("negative beyond" in w for w in result.warnings)


def test_fetch_era5_point_reports_duplicate_valid_times(tmp_path):
    """Confirms the duplicate-ValidTime warning fires with the correct
    count when chunks genuinely overlap."""

    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_bytes(b"fake")

    def fake_read(path, lat, lon):
        # every "chunk" returns the exact same single row - guarantees
        # a duplicate ValidTime when two chunks are combined
        return pd.DataFrame(
            {
                "reference_time": [pd.Timestamp("2020-01-01")],
                "step_hours": [1],
                "valid_time": [pd.Timestamp("2020-01-01 01:00")],
                "accumulated_mm": [0.5],
            }
        )

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2021-12-31",
        product="era5",
        years_per_chunk=1,
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tmp_path,
    )

    assert any("duplicate ValidTime" in w for w in result.warnings)


def test_fetch_era5_point_excludes_anomalies_outside_requested_range(tmp_path):
    """Regression test for a real, confirmed bug: a year-chunk request
    always pulls the WHOLE calendar year(s), even when the user's
    end_date only covers part of the final one. The main output was
    already correctly trimmed via a date mask, but anomaly_rows was
    collected from the chunk's raw de-accumulated data BEFORE that
    same trim was applied - so an anomaly occurring in the "extra"
    untrimmed portion was counted in the anomaly warning/audit CSV
    despite never appearing in the output CSV at all. This test
    constructs exactly that: a single chunk whose raw data spans
    beyond the requested end_date, with a genuine anomaly ONLY in that
    out-of-range portion, and confirms it is correctly excluded from
    both the anomalies list and the reported count."""

    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_bytes(b"fake")

    def fake_read(path, lat, lon):
        # step 1 (valid_time 2020-01-01 01:00) is WITHIN the requested
        # range and unremarkable; step 2 (valid_time 2020-01-02 01:00,
        # beyond the requested end_date of 2020-01-01) is a large
        # negative anomaly that must never surface anywhere in the
        # output.
        return pd.DataFrame(
            {
                "reference_time": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-01"),
                ],
                "step_hours": [1, 25],
                "valid_time": [
                    pd.Timestamp("2020-01-01 01:00"),
                    pd.Timestamp("2020-01-02 01:00"),
                ],
                "accumulated_mm": [1.0, -10.0],
            }
        )

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        product="era5",
        years_per_chunk=1,
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tmp_path,
    )

    assert len(result.anomalies) == 0
    assert not any("negative beyond" in w for w in result.warnings)
    assert len(result.dataframe) == 1


def test_fetch_era5_point_default_cds_retrieve_fn_passes_api_key(monkeypatch, tmp_path):
    """Regression test for a bug where the default cds_retrieve_fn (used when
    cds_retrieve_fn=None) referenced an undefined name instead of its own
    apikey parameter, which would have raised NameError on first live use
    (this default path is only exercised when no cds_retrieve_fn is injected,
    so it was not covered by the existing injected-fn tests). This executes
    that exact code path and asserts the api_key value actually reaches
    default_cds_retrieve.
    """
    captured = {}

    def fake_default_cds_retrieve(dataset, request, target_path, api_key=None):
        captured["called"] = True
        captured["api_key"] = api_key

    monkeypatch.setattr(
        era5_core_module, "default_cds_retrieve", fake_default_cds_retrieve
    )

    era5_core_module.fetch_era5_point(
        lat=1.0,
        lon=2.0,
        start_date="2020-01-01",
        end_date="2020-01-02",
        cds_retrieve_fn=None,
        api_key="my-test-key",
        save_credentials=False,
        work_dir=str(tmp_path),
    )

    assert captured.get("called") is True
    assert captured.get("api_key") == "my-test-key"
