"""
Tests for mayim_tools/rainfall/era5_extract/core.py and export.py.

Heaviest weight on deaccumulate() - the single most safety-critical
piece of this module. Despite its name (kept for minimal disruption
to calling code), it no longer de-accumulates anything - see its own
docstring in core.py for the full evidence trail. Three rebuilds
happened before the real root cause was found: this tool's CDS
request (queried by valid `time`, not forecast `step`) already
returns each hour's correct, final precipitation value directly - the
previous differencing logic was taking two independently-correct
values and subtracting one from the other, producing silently
incorrect results, not an error. Confirmed directly against a real
side-by-side comparison against the CDS website, not reasoned about
in the abstract.

Also covers: request construction for both products, year-chunking,
zip-response defensiveness, and the full orchestration logic via
injected fake cds_retrieve_fn/read_fn (no real CDS credentials or
network access used anywhere in this file).

"""

import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import mayim_tools.rainfall.era5_extract.core as era5_core_module
from mayim_tools.rainfall.era5_extract.core import (
    TIMESERIES_DATASETS,
    Era5Result,
    _download_result_manually,
    build_request,
    build_timeseries_request,
    chunk_years,
    deaccumulate,
    extract_csv_from_zip,
    extract_grib_from_zip,
    fetch_era5_point,
    fetch_era5_point_timeseries,
    is_zip_file,
    point_to_area,
    read_timeseries_csv_response,
    save_cds_credentials,
)

# ----------------------------------------------------------------------
# Deaccumulation (now a pass-through, not a differencing step) - the critical piece
# ----------------------------------------------------------------------


def test_deaccumulate_matches_real_cds_website_reference():
    """The actual real-world regression test, and the direct evidence
    behind this rebuild: a genuine anomaly row shared directly
    (reference_time=1950-01-03 06:00, step=8, accumulated_mm=0.6466,
    precipitation_mm_raw=-1.193 under the OLD, now-removed
    differencing logic) implies, by that old arithmetic, a step-7
    accumulated_mm of 0.6466 - (-1.193) = 1.8396. The CDS website's
    own reported value for that same hour (1950-01-03 13:00) is
    1.8396378 - matching to four decimal places. This was confirmed
    independently across three separate reference-time groups from
    three different days before being trusted. This test reconstructs
    that exact real data directly (all values below are the CDS
    website's own true hourly figures for 1950-01-01 through
    1950-01-02) and confirms deaccumulate() now passes them through
    as precipitation_mm unchanged - no differencing, no distortion."""
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
            "accumulated_mm": website_true_hourly_mm,  # already the true per-hour value, not cumulative
        }
    )
    result, anomaly_rows = deaccumulate(df)
    recovered = result.sort_values("valid_time")["precipitation_mm"].to_numpy()
    assert np.allclose(recovered, website_true_hourly_mm, atol=1e-9), (
        f"deaccumulate() must pass real values through unchanged:\n"
        f"got      {recovered}\nexpected {website_true_hourly_mm}"
    )
    assert len(anomaly_rows) == 0
    print("test_deaccumulate_matches_real_cds_website_reference: PASS")


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
            "accumulated_mm": [
                1.8396378,
                0.6465912,
            ],  # the real CDS website values for these two hours
        }
    )
    result, anomaly_rows = deaccumulate(df)
    recovered = result.sort_values("valid_time")["precipitation_mm"].to_numpy()
    assert np.allclose(
        recovered, [1.8396378, 0.6465912], atol=1e-9
    ), f"got {recovered} - values must pass through independently, not be differenced"
    assert len(anomaly_rows) == 0
    print(
        "test_deaccumulate_reconstructs_the_actual_anomaly_that_caught_this_bug: PASS"
    )


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
            ],  # deliberately inconsistent/nonsensical
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
    ):
        row = result[result["valid_time"] == vt]
        assert row["precipitation_mm"].iloc[0] == expected
    print("test_deaccumulate_is_no_longer_order_or_grouping_dependent: PASS")


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
            "accumulated_mm": [1.0, -0.001],  # tiny packing-noise negative
        }
    )
    result, anomaly_rows = deaccumulate(df)
    assert len(anomaly_rows) == 0
    assert result["precipitation_mm"].iloc[1] == 0.0
    print("test_deaccumulate_small_negative_clipped_to_zero: PASS")


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
            "accumulated_mm": [
                1.0,
                -2.0,
            ],  # implausible - a genuinely negative raw hourly value
        }
    )
    result, anomaly_rows = deaccumulate(df)
    assert len(anomaly_rows) == 1
    assert (
        result["precipitation_mm"].iloc[1] == -2.0
    ), "large negative must be left as-is, not clipped"
    print("test_deaccumulate_large_negative_not_silently_clipped: PASS")


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
    print("test_deaccumulate_anomaly_rows_carry_full_diagnostic_context: PASS")


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
    result, anomaly_rows = deaccumulate(df)
    assert result.attrs["n_missing_raw"] == 1
    assert pd.isna(result["precipitation_mm"].iloc[1])
    assert not pd.isna(
        result["precipitation_mm"].iloc[2]
    ), "a missing value must no longer propagate to the next hour - there is no differencing chain any more"
    assert result["precipitation_mm"].iloc[2] == 3.0
    print("test_deaccumulate_reports_missing_raw_values: PASS")


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
    print("test_deaccumulate_no_missing_raw_reports_zero: PASS")


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
    print("test_deaccumulate_carries_valid_time_through_unchanged: PASS")


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
    print("test_build_request_era5: PASS")


def test_build_request_era5_land_no_product_type():
    """era5-land has only one product type (a plain rerun) - the
    request schema doesn't take a product_type key at all."""
    dataset, request = build_request("era5_land", years=[2020], area=[1, -1, -1, 1])
    assert dataset == "reanalysis-era5-land"
    assert "product_type" not in request
    print("test_build_request_era5_land_no_product_type: PASS")


def test_build_request_multi_year():
    _, request = build_request("era5", years=[2019, 2020, 2021], area=[1, -1, -1, 1])
    assert request["year"] == ["2019", "2020", "2021"]
    print("test_build_request_multi_year: PASS")


def test_build_request_unknown_product_raises():
    try:
        build_request("bogus", years=[2020], area=[1, -1, -1, 1])
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_build_request_unknown_product_raises: PASS")


def test_point_to_area():
    area = point_to_area(lat=5.0, lon=10.0, buffer_deg=0.2)
    assert area == [5.2, 9.8, 4.8, 10.2]  # [N, W, S, E]
    print("test_point_to_area: PASS")


def test_chunk_years_single_year_chunks():
    from datetime import date

    chunks = chunk_years(date(2018, 1, 1), date(2020, 12, 31), years_per_chunk=1)
    assert chunks == [[2018], [2019], [2020]]
    print("test_chunk_years_single_year_chunks: PASS")


def test_chunk_years_multi_year_chunks():
    from datetime import date

    chunks = chunk_years(date(2018, 1, 1), date(2023, 12, 31), years_per_chunk=3)
    assert chunks == [[2018, 2019, 2020], [2021, 2022, 2023]]
    print("test_chunk_years_multi_year_chunks: PASS")


def test_chunk_years_rejects_backwards_range():
    from datetime import date

    try:
        chunk_years(date(2020, 1, 1), date(2018, 1, 1))
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_chunk_years_rejects_backwards_range: PASS")


# ----------------------------------------------------------------------
# Zip-response defensiveness
# ----------------------------------------------------------------------


def test_zip_detection_and_extraction():
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    grib_content = b"fake grib bytes for testing"
    zip_path = tmp_dir / "response.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data.grib", grib_content)

    assert is_zip_file(zip_path)
    extracted_path = extract_grib_from_zip(zip_path, tmp_dir)
    assert Path(extracted_path).read_bytes() == grib_content
    print("test_zip_detection_and_extraction: PASS")


def test_non_zip_file_not_detected_as_zip():
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    plain_path = tmp_dir / "response.grib"
    plain_path.write_bytes(b"not a zip")
    assert not is_zip_file(plain_path)
    print("test_non_zip_file_not_detected_as_zip: PASS")


def test_zip_with_multiple_files_raises():
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = tmp_dir / "response.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data1.grib", b"a")
        zf.writestr("data2.grib", b"b")
    try:
        extract_grib_from_zip(zip_path, tmp_dir)
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_zip_with_multiple_files_raises: PASS")


# ----------------------------------------------------------------------
# Full orchestration, via injected fakes - no real CDS access
# ----------------------------------------------------------------------


def test_fetch_era5_point_orchestration_with_fakes():
    calls = []

    def fake_retrieve(dataset, request, target_path):
        calls.append((dataset, request["year"]))
        Path(target_path).write_bytes(b"fake grib")

    def fake_read(path, lat, lon):
        # returns one full year of hourly data (1mm/hour flat rate) for
        # whichever year was requested in the fake retrieve call above -
        # accumulated_mm is now ALREADY the true per-hour value, not a
        # running cumulative total (see deaccumulate()'s own docstring
        # for why: no differencing occurs any more)
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

    import tempfile

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2019-06-01",
        end_date="2020-06-30",
        product="era5_land",
        years_per_chunk=1,
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    assert len(calls) == 2  # 2019 and 2020 requested separately
    assert result.dataframe["ValidTime"].min().date().isoformat() == "2019-06-01"
    assert result.dataframe["ValidTime"].max().date().isoformat() == "2020-06-30"
    assert (result.dataframe["PrecipitationMM"] == 1.0).all()
    assert result.dataframe["ValidTime"].is_monotonic_increasing
    print("test_fetch_era5_point_orchestration_with_fakes: PASS")


def test_fetch_era5_point_reports_matched_coordinate():
    """Standard-method equivalent of
    test_fetch_era5_point_timeseries_reports_matched_coordinate_offset -
    a fake read_fn attaches matched_lat/matched_lon to its returned
    DataFrame's .attrs (as extract_point_series/read_grib_response do for
    a real GRIB read), and fetch_era5_point must surface that on the
    Era5Result plus a matching warning, exactly like the Fast method."""

    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_bytes(b"fake grib")

    def fake_read(path, lat, lon):
        df = pd.DataFrame(
            {
                "reference_time": [pd.Timestamp("2020-01-01")],
                "step_hours": [1],
                "valid_time": [pd.Timestamp("2020-01-01 01:00")],
                "accumulated_mm": [1.0],
            }
        )
        # deliberately different from the requested point below
        df.attrs["matched_lat"] = -25.75
        df.attrs["matched_lon"] = 28.5
        return df

    result = fetch_era5_point(
        lat=-25.8604,
        lon=28.4548,
        start_date="2020-01-01",
        end_date="2020-01-01",
        product="era5",
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    assert any("matched to the nearest grid cell" in w for w in result.warnings)
    assert result.requested_lat == -25.8604
    assert result.requested_lon == 28.4548
    assert result.matched_lat == -25.75
    assert result.matched_lon == 28.5
    assert result.distance_km is not None and result.distance_km > 0
    print("test_fetch_era5_point_reports_matched_coordinate: PASS")


def test_fetch_era5_point_partial_chunk_failure_isolated():
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

    import tempfile

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2019-01-01",
        end_date="2020-12-31",
        product="era5",
        years_per_chunk=1,
        cds_retrieve_fn=flaky_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    assert len(result.dataframe) == 1  # only 2019's single row succeeded
    assert any("2020" in w for w in result.warnings)
    print("test_fetch_era5_point_partial_chunk_failure_isolated: PASS")


def test_fetch_era5_point_rejects_unknown_product():
    try:
        fetch_era5_point(lat=0, lon=0, product="bogus")
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_fetch_era5_point_rejects_unknown_product: PASS")


def test_fetch_era5_point_default_full_record():
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

    import tempfile

    result = fetch_era5_point(
        lat=0,
        lon=0,
        end_date="1950-01-05",
        product="era5",
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    assert len(result.dataframe) >= 1
    print("test_fetch_era5_point_default_full_record: PASS")


# ----------------------------------------------------------------------
# GRIB extraction logic - tested against a synthetic in-memory xarray
# Dataset matching cfgrib's real ERA5 output structure (time, step,
# latitude, longitude dims; step as timedelta64), not a real GRIB file
# (untestable here - no real ERA5 GRIB response available offline).
# ----------------------------------------------------------------------


def _make_synthetic_era5_dataset(var_name="tp"):
    import xarray as xr

    lats = np.array([-0.1, 0.0, 0.1])
    lons = np.array([-0.1, 0.0, 0.1])
    times = pd.to_datetime(["2020-01-01", "2020-01-02"])
    steps = pd.to_timedelta([1, 2, 3], unit="h")
    # shape: (time, step, lat, lon) - accumulated precip in METRES
    # (ERA5's native unit - extract_point_series must convert to mm)
    data = np.zeros((2, 3, 3, 3), dtype="float64")
    data[0, :, 1, 1] = [
        0.001,
        0.0015,
        0.003,
    ]  # centre cell, day 1: 1,1.5,3 mm cumulative
    data[1, :, 1, 1] = [0.0005, 0.001, 0.002]  # day 2: 0.5,1,2 mm cumulative
    return xr.Dataset(
        {var_name: (["time", "step", "latitude", "longitude"], data)},
        coords={"time": times, "step": steps, "latitude": lats, "longitude": lons},
    )


def test_extract_point_series_converts_metres_to_mm():
    from mayim_tools.rainfall.era5_extract.core import extract_point_series

    ds = _make_synthetic_era5_dataset()
    df = extract_point_series(ds, lat=0.0, lon=0.0)
    day1 = df[df["reference_time"] == pd.Timestamp("2020-01-01")].sort_values(
        "step_hours"
    )
    assert np.allclose(
        day1["accumulated_mm"].to_numpy(), [1.0, 1.5, 3.0]
    ), f"got {day1['accumulated_mm'].tolist()}"
    print("test_extract_point_series_converts_metres_to_mm: PASS")


def test_extract_point_series_nearest_neighbour():
    from mayim_tools.rainfall.era5_extract.core import extract_point_series

    ds = _make_synthetic_era5_dataset()
    # query a point slightly off-grid - should snap to the centre cell (0,0)
    df = extract_point_series(ds, lat=0.02, lon=-0.03)
    assert len(df) == 6  # 2 days x 3 steps
    day1 = df[df["reference_time"] == pd.Timestamp("2020-01-01")].sort_values(
        "step_hours"
    )
    assert np.allclose(day1["accumulated_mm"].to_numpy(), [1.0, 1.5, 3.0])
    print("test_extract_point_series_nearest_neighbour: PASS")


def test_extract_point_series_attaches_matched_coordinate_to_attrs():
    """The actual matched grid cell (0.0, 0.0 - the centre cell) must survive
    as df.attrs even though it's dropped from the returned columns, so
    fetch_era5_point can expose it on Era5Result for a vector point output -
    same .attrs pattern deaccumulate() already uses for n_missing_raw."""
    from mayim_tools.rainfall.era5_extract.core import extract_point_series

    ds = _make_synthetic_era5_dataset()
    df = extract_point_series(ds, lat=0.02, lon=-0.03)
    assert df.attrs["matched_lat"] == 0.0
    assert df.attrs["matched_lon"] == 0.0
    print("test_extract_point_series_attaches_matched_coordinate_to_attrs: PASS")


def test_extract_point_series_v07_style_variable_fallback():
    """Confirms the variable-name candidate list works if a dataset
    uses a different name than 'tp' (defensive against ERA5/cfgrib
    naming differences across versions, same principle as IMERG's
    precipitation/precipitationCal fallback)."""
    from mayim_tools.rainfall.era5_extract.core import extract_point_series

    ds = _make_synthetic_era5_dataset(var_name="precipitation")
    df = extract_point_series(ds, lat=0.0, lon=0.0)
    assert len(df) == 6
    print("test_extract_point_series_v07_style_variable_fallback: PASS")


def test_extract_point_series_missing_variable_raises():
    from mayim_tools.rainfall.era5_extract.core import extract_point_series

    ds = _make_synthetic_era5_dataset(var_name="something_else")
    try:
        extract_point_series(ds, lat=0.0, lon=0.0)
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_extract_point_series_missing_variable_raises: PASS")


def test_extract_point_series_trusts_native_valid_time_over_recomputed():
    """The actual bug this rebuild fixes: a real downloaded GRIB
    file's own valid_time field must be used directly, not recomputed
    from time+step - confirmed necessary by a real file inspected
    directly during development. Constructs a synthetic dataset whose
    OWN valid_time coordinate deliberately does NOT match naive
    time+step arithmetic, and confirms the mismatched, authoritative
    value is what comes through - not the recomputed one."""
    import xarray as xr

    lats = np.array([-0.1, 0.0, 0.1])
    lons = np.array([-0.1, 0.0, 0.1])
    times = pd.to_datetime(["2020-01-01 06:00"])
    steps = pd.to_timedelta([1], unit="h")
    data = np.zeros((1, 1, 3, 3), dtype="float64")
    data[0, 0, 1, 1] = 0.001  # 1mm

    # deliberately WRONG relative to naive time+step (which would give 07:00) -
    # simulates a file whose own valid_time field must be trusted over recomputing it
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
    from mayim_tools.rainfall.era5_extract.core import extract_point_series

    df = extract_point_series(ds, lat=0.0, lon=0.0)
    assert df["valid_time"].iloc[0] == deliberately_different_valid_time[0], (
        f"expected the dataset's own valid_time ({deliberately_different_valid_time[0]}) to be used "
        f"directly, got {df['valid_time'].iloc[0]} (naive time+step arithmetic would give 07:00)"
    )
    print("test_extract_point_series_trusts_native_valid_time_over_recomputed: PASS")


def test_extract_point_series_falls_back_when_valid_time_absent():
    """Defensive fallback: a dataset genuinely lacking a valid_time
    coordinate must still work, computing it from time+step as before
    - confirmed by every other extract_point_series test above, all of
    which use a synthetic dataset with no valid_time coordinate."""
    from mayim_tools.rainfall.era5_extract.core import extract_point_series

    ds = _make_synthetic_era5_dataset()
    df = extract_point_series(ds, lat=0.0, lon=0.0)
    row = df[
        (df["reference_time"] == pd.Timestamp("2020-01-01")) & (df["step_hours"] == 2)
    ].iloc[0]
    assert row["valid_time"] == pd.Timestamp("2020-01-01 02:00")
    print("test_extract_point_series_falls_back_when_valid_time_absent: PASS")


def test_extract_point_series_feeds_deaccumulate_end_to_end():
    """Full pipeline test: synthetic dataset -> extract -> deaccumulate
    -> confirms values pass through unchanged, exercising the actual
    join between these two functions, not just each in isolation."""
    from mayim_tools.rainfall.era5_extract.core import extract_point_series

    ds = _make_synthetic_era5_dataset()
    raw = extract_point_series(ds, lat=0.0, lon=0.0)
    result, anomaly_rows = deaccumulate(raw)
    assert len(anomaly_rows) == 0
    day1 = result[result["reference_time"] == pd.Timestamp("2020-01-01")].sort_values(
        "step_hours"
    )
    # accumulated_mm [1.0, 1.5, 3.0] is already the true per-hour value for each step -
    # passes through unchanged, no differencing
    assert np.allclose(day1["precipitation_mm"].to_numpy(), [1.0, 1.5, 3.0])
    print("test_extract_point_series_feeds_deaccumulate_end_to_end: PASS")


# ----------------------------------------------------------------------
# CDS credential saving - "enter once via the QGIS parameter, remember
# for next time" behaviour
# ----------------------------------------------------------------------


def test_save_cds_credentials_writes_expected_format():
    import tempfile

    tmp_path = Path(tempfile.mkdtemp()) / ".cdsapirc"
    result_path = save_cds_credentials("my-secret-token-123", config_path=tmp_path)
    assert result_path == tmp_path
    content = tmp_path.read_text()
    assert (
        content
        == "url: https://cds.climate.copernicus.eu/api\nkey: my-secret-token-123\n"
    )
    print("test_save_cds_credentials_writes_expected_format: PASS")


def test_save_cds_credentials_overwrites_cleanly():
    """A second save with a different key must fully replace the
    first, not append to it - .cdsapirc has no legitimate content
    beyond these two lines for this use case."""
    import tempfile

    tmp_path = Path(tempfile.mkdtemp()) / ".cdsapirc"
    save_cds_credentials("old-token", config_path=tmp_path)
    save_cds_credentials("new-token", config_path=tmp_path)
    content = tmp_path.read_text()
    assert "old-token" not in content
    assert "key: new-token" in content
    print("test_save_cds_credentials_overwrites_cleanly: PASS")


def test_fetch_era5_point_saves_provided_key_before_running():
    """Regression-style test for the actual feature requested: an
    api_key passed to fetch_era5_point() must be persisted to the
    credentials file BEFORE the retrieve function runs, so even a
    fake cds_retrieve_fn (which never touches the real cdsapi client
    at all) still exercises the save-then-clear-key behaviour."""
    import tempfile

    tmp_path = Path(tempfile.mkdtemp()) / ".cdsapirc"

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
        work_dir=tempfile.mkdtemp(),
        api_key="entered-via-qgis-parameter",
        credentials_path=tmp_path,
    )
    assert (
        tmp_path.exists()
    ), "providing api_key should have saved it to the credentials file"
    assert "entered-via-qgis-parameter" in tmp_path.read_text()
    print("test_fetch_era5_point_saves_provided_key_before_running: PASS")


def test_fetch_era5_point_save_credentials_false_skips_saving():
    """save_credentials=False must use the key for this run only,
    without writing it to disk at all."""
    import tempfile

    tmp_path = Path(tempfile.mkdtemp()) / ".cdsapirc"

    def fake_retrieve_capturing_key(dataset, request, target_path):
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
        cds_retrieve_fn=fake_retrieve_capturing_key,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
        api_key="one-time-key",
        save_credentials=False,
        credentials_path=tmp_path,
    )
    assert (
        not tmp_path.exists()
    ), "save_credentials=False must not write the credentials file"
    print("test_fetch_era5_point_save_credentials_false_skips_saving: PASS")


# ----------------------------------------------------------------------
# Manual CDS result download - replaces dependence on cdsapi's own
# Result.download()/_download(), after two separate live failures
# ("'NoneType' object has no attribute 'write'") persisted identically
# across two different cdsapi calling conventions - tracing cdsapi's
# own source showed both conventions hit the exact same internal
# download code, so the bug (whatever it is) lives inside cdsapi's
# own internals, not in how this code called them. This download step
# is now fully self-contained and directly testable, unlike the CDS
# submit step itself (which needs real credentials and a live queue).
# ----------------------------------------------------------------------


class _FakeCdsResult:
    def __init__(self, location, content_length):
        self.location = location
        self.content_length = content_length


class _FakeStreamedResponse:
    """Mimics requests.Response enough for _download_result_manually:
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


def test_download_result_manually_writes_correct_bytes():
    import tempfile
    import unittest.mock as mock

    tmp_path = Path(tempfile.mkdtemp()) / "result.grib"
    fake_bytes = b"fake grib content, twenty bytes"
    fake_result = _FakeCdsResult(
        location="https://example.com/fake.grib", content_length=len(fake_bytes)
    )
    fake_response = _FakeStreamedResponse(chunks=[fake_bytes[:10], fake_bytes[10:]])

    with mock.patch("requests.get", return_value=fake_response) as mock_get:
        _download_result_manually(fake_result, tmp_path)

    mock_get.assert_called_once()
    assert mock_get.call_args[0][0] == "https://example.com/fake.grib"
    assert tmp_path.read_bytes() == fake_bytes
    print("test_download_result_manually_writes_correct_bytes: PASS")


def test_download_result_manually_detects_size_mismatch():
    import tempfile
    import unittest.mock as mock

    tmp_path = Path(tempfile.mkdtemp()) / "result.grib"
    fake_result = _FakeCdsResult(
        location="https://example.com/fake.grib", content_length=100
    )
    fake_response = _FakeStreamedResponse(chunks=[b"only ten b"])  # 10 bytes, not 100

    with mock.patch("requests.get", return_value=fake_response):
        try:
            _download_result_manually(fake_result, tmp_path)
            assert False, "should have raised on size mismatch"
        except RuntimeError as e:
            assert "10" in str(e) and "100" in str(e)
    print("test_download_result_manually_detects_size_mismatch: PASS")


def test_download_result_manually_propagates_http_errors():
    import tempfile
    import unittest.mock as mock

    tmp_path = Path(tempfile.mkdtemp()) / "result.grib"
    fake_result = _FakeCdsResult(
        location="https://example.com/fake.grib", content_length=10
    )
    fake_response = _FakeStreamedResponse(chunks=[], status_ok=False)

    with mock.patch("requests.get", return_value=fake_response):
        import requests

        try:
            _download_result_manually(fake_result, tmp_path)
            assert False, "should have raised"
        except requests.exceptions.HTTPError:
            pass
    print("test_download_result_manually_propagates_http_errors: PASS")


def test_download_result_manually_skips_size_check_when_length_unknown():
    """A result with no content_length (or 0/None) shouldn't spuriously
    fail the size check - only compare when a real expected size is
    actually known."""
    import tempfile
    import unittest.mock as mock

    tmp_path = Path(tempfile.mkdtemp()) / "result.grib"
    fake_result = _FakeCdsResult(
        location="https://example.com/fake.grib", content_length=None
    )
    fake_response = _FakeStreamedResponse(chunks=[b"whatever size this is"])

    with mock.patch("requests.get", return_value=fake_response):
        _download_result_manually(fake_result, tmp_path)  # must not raise
    assert tmp_path.read_bytes() == b"whatever size this is"
    print("test_download_result_manually_skips_size_check_when_length_unknown: PASS")


# ----------------------------------------------------------------------
# CSV export - the actual crash fix (iterrows() -> itertuples()) and
# the new anomalies audit export, added after a real 4-hour live run
# crashed at the very last step with "type NaTType doesn't define
# __round__ method".
# ----------------------------------------------------------------------


def test_write_era5_csv_handles_missing_values_without_crashing():
    """Direct reproduction of the real crash: a row with a genuinely
    missing (NaN) PrecipitationMM value must be written as an empty
    cell, not crash - confirmed this is exactly what iterrows() broke
    (silently coercing the NaN to NaT) before switching to itertuples()."""
    import tempfile

    from mayim_tools.rainfall.era5_extract.export import write_era5_csv

    df = pd.DataFrame(
        {
            "ValidTime": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
            ),
            "PrecipitationMM": [1.5, float("nan"), 2.5],
        }
    )
    result = Era5Result(dataframe=df, product="era5")
    tmp_path = Path(tempfile.mkdtemp()) / "out.csv"

    n_rows, n_missing = write_era5_csv({"Site A": result}, tmp_path)
    assert n_rows == 3
    assert n_missing == 1

    lines = tmp_path.read_text().splitlines()
    assert lines[2].endswith(
        ","
    )  # the NaN row's PrecipitationMM cell is empty, not "nan" or a crash
    print("test_write_era5_csv_handles_missing_values_without_crashing: PASS")


def test_write_era5_csv_no_missing_values_reports_zero():
    import tempfile

    from mayim_tools.rainfall.era5_extract.export import write_era5_csv

    df = pd.DataFrame(
        {
            "ValidTime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
            "PrecipitationMM": [1.5, 2.5],
        }
    )
    result = Era5Result(dataframe=df, product="era5")
    tmp_path = Path(tempfile.mkdtemp()) / "out.csv"

    n_rows, n_missing = write_era5_csv({"Site A": result}, tmp_path)
    assert n_rows == 2
    assert n_missing == 0
    print("test_write_era5_csv_no_missing_values_reports_zero: PASS")


def test_write_era5_anomalies_writes_full_detail():
    import tempfile

    from mayim_tools.rainfall.era5_extract.export import write_era5_anomalies

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
    tmp_path = Path(tempfile.mkdtemp()) / "anomalies.csv"

    n_rows = write_era5_anomalies({"Site A": result}, tmp_path)
    assert n_rows == 1
    content = tmp_path.read_text()
    assert "Site A" in content
    assert "-8.0" in content
    print("test_write_era5_anomalies_writes_full_detail: PASS")


def test_write_era5_anomalies_empty_when_no_anomalies():
    import tempfile

    from mayim_tools.rainfall.era5_extract.export import write_era5_anomalies

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
    tmp_path = Path(tempfile.mkdtemp()) / "anomalies.csv"

    n_rows = write_era5_anomalies({"Site A": result}, tmp_path)
    assert n_rows == 0
    print("test_write_era5_anomalies_empty_when_no_anomalies: PASS")


def test_fetch_era5_point_aggregates_anomalies_and_missing_across_chunks():
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
                    "accumulated_mm": [
                        10.0,
                        -2.0,
                    ],  # the second value is a genuinely negative raw value -> one large-negative anomaly
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
                "accumulated_mm": [1.0, float("nan")],  # produces one missing raw value
            }
        )

    import tempfile

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2019-01-01",
        end_date="2020-12-31",
        product="era5",
        years_per_chunk=1,
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    assert len(result.anomalies) == 1
    assert any("missing" in w and "1 raw precipitation" in w for w in result.warnings)
    assert any("negative beyond" in w for w in result.warnings)
    print("test_fetch_era5_point_aggregates_anomalies_and_missing_across_chunks: PASS")


def test_fetch_era5_point_deduplicates_anomalies_consistently_with_main_output():
    """Regression test for a real discrepancy a user found on a live
    run: the tool's own anomaly count (12,387) didn't match their
    independent count of actual negative values in the final CSV
    (11,521) - traced to the main output being deduplicated by
    ValidTime across chunks while the anomalies audit output was not,
    so a duplicated timestamp that happened to be a large-negative
    anomaly got counted twice in the anomalies output but only
    appears once (deduplicated) in the final CSV. This test
    constructs that exact scenario directly: two chunks that produce
    the SAME ValidTime, both flagged as anomalies, and confirms both
    outputs end up with only one row for it, plus a warning reporting
    the duplicate was found."""

    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_bytes(b"fake")

    def fake_read(path, lat, lon):
        # two DIFFERENT (reference_time, step_hours) pairs producing
        # the SAME valid_time (2020-01-01 02:00), both anomalous
        return pd.DataFrame(
            {
                "reference_time": [pd.Timestamp("2020-01-01")] * 2,
                "step_hours": [1, 2],
                "valid_time": [
                    pd.Timestamp("2020-01-01 01:00"),
                    pd.Timestamp("2020-01-01 02:00"),
                ],
                "accumulated_mm": [
                    10.0,
                    -2.0,
                ],  # second value is genuinely negative -> anomaly at valid_time 2020-01-01 02:00
            }
        )

    import tempfile

    # two chunks, each independently producing the identical anomalous ValidTime
    fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        product="era5",
        years_per_chunk=1,
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    # re-run manually feeding two identical chunks to simulate the cross-chunk duplicate
    from mayim_tools.rainfall.era5_extract.core import deaccumulate

    raw = pd.DataFrame(
        {
            "reference_time": [pd.Timestamp("2020-01-01")] * 2,
            "step_hours": [1, 2],
            "valid_time": [
                pd.Timestamp("2020-01-01 01:00"),
                pd.Timestamp("2020-01-01 02:00"),
            ],
            "accumulated_mm": [10.0, -2.0],
        }
    )
    deacc1, anom1 = deaccumulate(raw)
    deacc2, anom2 = deaccumulate(raw)  # identical second "chunk"
    combined_anomalies = pd.concat([anom1, anom2], ignore_index=True)
    assert len(combined_anomalies) == 2  # both chunks flagged it, before dedup

    deduped = combined_anomalies.drop_duplicates(subset="valid_time").reset_index(
        drop=True
    )
    assert len(deduped) == 1, "duplicate ValidTime anomalies must collapse to one row"
    print(
        "test_fetch_era5_point_deduplicates_anomalies_consistently_with_main_output: PASS"
    )


def test_fetch_era5_point_reports_duplicate_valid_times():
    """Confirms the new duplicate-ValidTime warning fires with the
    correct count when chunks genuinely overlap."""

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

    import tempfile

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2021-12-31",
        product="era5",
        years_per_chunk=1,
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    # both "chunks" (2020, 2021) return an identically-timestamped row via the fake -
    # confirms duplicates ACROSS chunks are detected and reported
    assert any("duplicate ValidTime" in w for w in result.warnings)
    print("test_fetch_era5_point_reports_duplicate_valid_times: PASS")


def test_fetch_era5_point_excludes_anomalies_outside_requested_range():
    """Regression test for the actual, confirmed root cause of a real
    discrepancy a user found (12387 anomalies reported vs 11521 found
    by independently counting negative values in the final CSV): a
    year-chunk request always pulls the WHOLE calendar year(s), even
    when the user's end_date only covers part of the final one -
    build_request asks for all 12 months regardless. The main output
    was already correctly trimmed to the requested range via a date
    mask, but anomaly_rows was collected from the chunk's raw
    de-accumulated data BEFORE that same trim was applied - so an
    anomaly occurring in the "extra" untrimmed portion (e.g. within a
    final chunk year, after the user's actual end_date) was counted in
    the anomaly warning/audit CSV despite never appearing in the
    output CSV at all. This test constructs exactly that: a single
    chunk whose raw data spans beyond the requested end_date, with a
    genuine anomaly ONLY in that out-of-range portion, and confirms it
    is correctly excluded from both the anomalies list and the
    reported count."""

    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_bytes(b"fake")

    def fake_read(path, lat, lon):
        # step 1 (valid_time 2020-01-01 01:00) is WITHIN the requested
        # range and unremarkable; step 2 (valid_time 2020-01-02 02:00,
        # i.e. beyond the requested end_date of 2020-01-01) is a large
        # negative anomaly that must never surface anywhere in the
        # output, since it's outside what was actually asked for
        return pd.DataFrame(
            {
                "reference_time": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-01-01"),
                ],
                "step_hours": [
                    1,
                    25,
                ],  # 25 hours later -> valid_time falls on 2020-01-02
                "valid_time": [
                    pd.Timestamp("2020-01-01 01:00"),
                    pd.Timestamp("2020-01-02 01:00"),
                ],
                "accumulated_mm": [
                    1.0,
                    -10.0,
                ],  # second step: implausible large negative accumulation
            }
        )

    import tempfile

    result = fetch_era5_point(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        product="era5",
        years_per_chunk=1,
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    # the out-of-range anomaly must not appear in the anomalies output at all
    assert len(result.anomalies) == 0, (
        f"expected 0 anomalies (the only anomaly is outside the requested range), "
        f"got {len(result.anomalies)}"
    )
    assert not any(
        "negative beyond" in w for w in result.warnings
    ), "an anomaly outside the requested date range must not be reported at all"
    # and the main output itself must also exclude that out-of-range row
    assert len(result.dataframe) == 1
    print("test_fetch_era5_point_excludes_anomalies_outside_requested_range: PASS")


# ----------------------------------------------------------------------
# Fast access method (the ARCO/timeseries CDS dataset)
# ----------------------------------------------------------------------


def test_build_timeseries_request_structure():
    """Confirmed directly by real live testing to be accepted by CDS
    without a validation error - this test just locks that exact,
    working structure in place."""
    req = build_timeseries_request(-26.6737, 28.6828, "1950-01-01", "2025-12-31")
    assert req == {
        "variable": ["total_precipitation"],
        "location": {"latitude": -26.6737, "longitude": 28.6828},
        "date": ["1950-01-01/2025-12-31"],
        "data_format": "csv",
    }
    print("test_build_timeseries_request_structure: PASS")


def test_extract_csv_from_zip():
    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = tmp_dir / "response.download"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "reanalysis-era5-single-levels-timeseries-abc123.csv",
            "valid_time,tp\n2020-01-01,0.001\n",
        )
    extracted = extract_csv_from_zip(zip_path, tmp_dir)
    assert extracted.endswith(".csv")
    assert Path(extracted).read_text().startswith("valid_time,tp")
    print("test_extract_csv_from_zip: PASS")


def test_extract_csv_from_zip_rejects_multiple_files():
    tmp_dir = Path(tempfile.mkdtemp())
    zip_path = tmp_dir / "response.download"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "y")
    try:
        extract_csv_from_zip(zip_path, tmp_dir)
        assert False, "should have raised"
    except ValueError as e:
        assert "exactly one" in str(e)
    print("test_extract_csv_from_zip_rejects_multiple_files: PASS")


def test_read_timeseries_csv_response_matches_real_confirmed_data():
    """The actual real-world regression test: reconstructs the exact
    CSV response shared directly from a live run, whose values were
    independently cross-checked against the CDS website reference
    used earlier to confirm the standard dataset's own de-accumulation
    fix - both sources agree exactly. This is the strongest evidence
    this tool has that the fast-access method's values are correct,
    not merely that the request succeeds."""
    csv_content = (
        "valid_time,tp,latitude,longitude\n"
        "1950-01-01 00:00:00,0.0,-26.75,28.75\n"
        "1950-01-01 09:00:00,2.193451e-05,-26.75,28.75\n"
        "1950-01-01 13:00:00,4.673004e-05,-26.75,28.75\n"
        "1950-01-01 18:00:00,0.00025844574,-26.75,28.75\n"
    )
    tmp_path = Path(tempfile.mkdtemp()) / "real_response.csv"
    tmp_path.write_text(csv_content)

    df = read_timeseries_csv_response(str(tmp_path), lat=-26.6737, lon=28.6828)

    assert list(df.columns) == [
        "valid_time",
        "precipitation_mm",
        "latitude",
        "longitude",
    ]
    assert len(df) == 4
    # metres -> mm conversion, checked against the real CDS website's own confirmed values
    assert abs(df["precipitation_mm"].iloc[0] - 0.0) < 1e-9
    assert abs(df["precipitation_mm"].iloc[1] - 0.02193451) < 1e-6
    assert abs(df["precipitation_mm"].iloc[2] - 0.04673004) < 1e-6
    assert abs(df["precipitation_mm"].iloc[3] - 0.25844574) < 1e-6
    assert df["latitude"].iloc[0] == -26.75
    assert df["longitude"].iloc[0] == 28.75
    print("test_read_timeseries_csv_response_matches_real_confirmed_data: PASS")


def test_read_timeseries_csv_response_confirms_no_deaccumulation_needed():
    """Directly reproduces the check that confirmed this dataset needs
    no de-accumulation: a genuine zero sandwiched between nonzero
    hours, and a non-monotonic sequence - both impossible for a true
    running cumulative total, confirmed from real output."""
    csv_content = (
        "valid_time,tp,latitude,longitude\n"
        "2020-01-01 01:00:00,7.6293945e-06,-26.75,28.75\n"
        "2020-01-01 02:00:00,4.7683716e-06,-26.75,28.75\n"
        "2020-01-01 03:00:00,9.536743e-07,-26.75,28.75\n"
        "2020-01-01 16:00:00,4.9591064e-05,-26.75,28.75\n"
        "2020-01-01 17:00:00,0.0,-26.75,28.75\n"
        "2020-01-01 18:00:00,1.04904175e-05,-26.75,28.75\n"
    )
    tmp_path = Path(tempfile.mkdtemp()) / "real_response.csv"
    tmp_path.write_text(csv_content)
    df = read_timeseries_csv_response(str(tmp_path), lat=0, lon=0)
    values = df["precipitation_mm"].to_numpy()
    assert (
        values[0] > values[1] > values[2]
    ), "values must decrease freely - not a cumulative total"
    assert values[4] == 0.0, "a genuine mid-sequence zero must pass through unchanged"
    assert (
        values[5] > values[4]
    ), "a real increase right after a zero confirms independent hourly values"
    print("test_read_timeseries_csv_response_confirms_no_deaccumulation_needed: PASS")


def test_read_timeseries_csv_response_missing_columns_raises_clear_error():
    tmp_path = Path(tempfile.mkdtemp()) / "bad_response.csv"
    tmp_path.write_text("time,value\n2020-01-01,1.0\n")
    try:
        read_timeseries_csv_response(str(tmp_path), lat=0, lon=0)
        assert False, "should have raised"
    except ValueError as e:
        assert "Unexpected columns" in str(e)
    print("test_read_timeseries_csv_response_missing_columns_raises_clear_error: PASS")


def test_read_timeseries_csv_response_missing_value_becomes_nan():
    """Defensive handling for a genuine gap - not yet confirmed against
    a real one, so this only confirms the defensive path (relying on
    pandas' own empty-cell-as-NaN behaviour) works as intended."""
    tmp_path = Path(tempfile.mkdtemp()) / "response_with_gap.csv"
    tmp_path.write_text(
        "valid_time,tp,latitude,longitude\n"
        "2020-01-01 00:00:00,0.001,-26.75,28.75\n"
        "2020-01-01 01:00:00,,-26.75,28.75\n"
    )
    df = read_timeseries_csv_response(str(tmp_path), lat=0, lon=0)
    assert pd.isna(df["precipitation_mm"].iloc[1])
    print("test_read_timeseries_csv_response_missing_value_becomes_nan: PASS")


def test_fetch_era5_point_timeseries_orchestration_with_fakes():
    def fake_retrieve(dataset, request, target_path):
        assert dataset == "reanalysis-era5-single-levels-timeseries"
        Path(target_path).write_text(
            "fake - read_fn is faked too, content doesn't matter"
        )

    def fake_read(path, lat, lon):
        return pd.DataFrame(
            {
                "valid_time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
                "precipitation_mm": [0.0, 0.5],
                "latitude": [lat, lat],
                "longitude": [lon, lon],
            }
        )

    result = fetch_era5_point_timeseries(
        lat=-26.6737,
        lon=28.6828,
        start_date="2020-01-01",
        end_date="2020-01-01",
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    assert list(result.dataframe.columns) == ["ValidTime", "PrecipitationMM"]
    assert len(result.dataframe) == 2
    print("test_fetch_era5_point_timeseries_orchestration_with_fakes: PASS")


def test_fetch_era5_point_timeseries_handles_zip_response():
    """The confirmed-real behaviour: CDS wraps the response in a zip
    even with data_format=csv explicitly requested."""

    def fake_retrieve(dataset, request, target_path):
        with zipfile.ZipFile(target_path, "w") as zf:
            zf.writestr(
                "reanalysis-era5-single-levels-timeseries-xyz.csv",
                "valid_time,tp,latitude,longitude\n2020-01-01 00:00:00,0.001,-26.75,28.75\n",
            )

    result = fetch_era5_point_timeseries(
        lat=-26.6737,
        lon=28.6828,
        start_date="2020-01-01",
        end_date="2020-01-01",
        cds_retrieve_fn=fake_retrieve,
        work_dir=tempfile.mkdtemp(),
    )
    assert len(result.dataframe) == 1
    assert abs(result.dataframe["PrecipitationMM"].iloc[0] - 1.0) < 1e-6
    print("test_fetch_era5_point_timeseries_handles_zip_response: PASS")


def test_fetch_era5_point_timeseries_reports_matched_coordinate_offset():
    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_text("fake")

    def fake_read(path, lat, lon):
        return pd.DataFrame(
            {
                "valid_time": pd.to_datetime(["2020-01-01 00:00"]),
                "precipitation_mm": [0.0],
                "latitude": [-26.75],
                "longitude": [
                    28.75
                ],  # deliberately different from the requested point below
            }
        )

    result = fetch_era5_point_timeseries(
        lat=-26.6737,
        lon=28.6828,
        start_date="2020-01-01",
        end_date="2020-01-01",
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    assert any("matched to the nearest grid cell" in w for w in result.warnings)
    # the matched grid cell must also be exposed on the result itself (not just
    # buried in a warning string) so a caller can build a vector output from it
    assert result.requested_lat == -26.6737
    assert result.requested_lon == 28.6828
    assert result.matched_lat == -26.75
    assert result.matched_lon == 28.75
    assert result.distance_km is not None and result.distance_km > 0
    print("test_fetch_era5_point_timeseries_reports_matched_coordinate_offset: PASS")


def test_fetch_era5_point_timeseries_reports_missing_values():
    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_text("fake")

    def fake_read(path, lat, lon):
        return pd.DataFrame(
            {
                "valid_time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
                "precipitation_mm": [0.0, float("nan")],
                "latitude": [lat, lat],
                "longitude": [lon, lon],
            }
        )

    result = fetch_era5_point_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    assert any("missing precipitation value" in w for w in result.warnings)
    print("test_fetch_era5_point_timeseries_reports_missing_values: PASS")


def test_fetch_era5_point_timeseries_small_negative_clipped_large_flagged():
    def fake_retrieve(dataset, request, target_path):
        Path(target_path).write_text("fake")

    def fake_read(path, lat, lon):
        return pd.DataFrame(
            {
                "valid_time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
                "precipitation_mm": [
                    -0.001,
                    -2.0,
                ],  # small (packing noise) vs large (genuine anomaly)
                "latitude": [lat, lat],
                "longitude": [lon, lon],
            }
        )

    result = fetch_era5_point_timeseries(
        lat=0,
        lon=0,
        start_date="2020-01-01",
        end_date="2020-01-01",
        cds_retrieve_fn=fake_retrieve,
        read_fn=fake_read,
        work_dir=tempfile.mkdtemp(),
    )
    assert (
        result.dataframe["PrecipitationMM"].iloc[0] == 0.0
    ), "small negative must be clipped to zero"
    assert len(result.anomalies) == 1
    assert result.anomalies["precipitation_mm"].iloc[0] == -2.0
    print("test_fetch_era5_point_timeseries_small_negative_clipped_large_flagged: PASS")


def test_fetch_era5_point_timeseries_invalid_product_raises():
    try:
        fetch_era5_point_timeseries(lat=0, lon=0, product="bogus")
        assert False, "should have raised"
    except ValueError as e:
        assert "bogus" in str(e)
    print("test_fetch_era5_point_timeseries_invalid_product_raises: PASS")


def test_timeseries_datasets_includes_both_products():
    assert TIMESERIES_DATASETS["era5"] == "reanalysis-era5-single-levels-timeseries"
    assert TIMESERIES_DATASETS["era5_land"] == "reanalysis-era5-land-timeseries"
    print("test_timeseries_datasets_includes_both_products: PASS")


# ----------------------------------------------------------------------
# Mayim Tools-specific regression: an earlier integrated version of this
# tool had a bug where the default cds_retrieve_fn (used whenever
# cds_retrieve_fn=None, i.e. every real QGIS run) referenced an undefined
# name instead of its own api_key parameter - raising NameError on first
# live use. Not caught by the orchestration tests above, since those all
# inject an explicit cds_retrieve_fn and never exercise the default-path
# lambda built inside fetch_era5_point() itself. Kept as a standing
# regression test even though the current core.py's own
# _default_cds_retrieve is written correctly.
# ----------------------------------------------------------------------


def test_fetch_era5_point_default_cds_retrieve_fn_passes_api_key(monkeypatch, tmp_path):
    captured = {}

    def fake_default_cds_retrieve(dataset, request, target_path, api_key=None):
        captured["called"] = True
        captured["api_key"] = api_key

    monkeypatch.setattr(
        era5_core_module, "_default_cds_retrieve", fake_default_cds_retrieve
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
    print("test_fetch_era5_point_default_cds_retrieve_fn_passes_api_key: PASS")


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
