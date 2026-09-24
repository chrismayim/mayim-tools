"""
Tests for mayim_tools/rainfall/chirps/core.py.

Focused on the matched-grid-cell-coordinate behaviour added alongside
the ERA5 extractor's own vector point outputs (see
test_era5_extract_core.py) - CHIRPS reads from a fixed raster grid too
(rasterio's src.index() finds whichever pixel a point falls inside),
so the same "requested point -> matched grid cell, with a distance"
treatment applies here. No real network access is used anywhere in
this file: rasterio.open() is monkeypatched to open a small synthetic
local GeoTIFF instead of a real /vsicurl/ URL, and
fetch_chirps_timeseries's orchestration is tested via its injectable
read_fn, exactly like every other tool in this suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from mayim_tools.rainfall.chirps.core import (
    ChirpsResult,
    fetch_chirps_timeseries,
    read_point_value,
    read_point_value_with_center,
)

# ----------------------------------------------------------------------
# Synthetic raster fixture - a small local GeoTIFF standing in for a
# real CHIRPS COG, so read_point_value()/read_point_value_with_center()
# can be exercised without network access.
# ----------------------------------------------------------------------


@pytest.fixture()
def synthetic_raster(tmp_path, monkeypatch):
    """Writes a 4x4 pixel GeoTIFF (0.05 deg resolution, matching
    CHIRPS's native grid) and monkeypatches rasterio.open so any
    /vsicurl/ URL core.py builds is transparently redirected to this
    local file - the actual pixel geometry (row/col -> value, row/col
    -> centre coordinate) is real rasterio behaviour, not mocked."""
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(28.0, -25.0, 0.05, 0.05)  # west, north, xsize, ysize
    data = np.arange(16, dtype="float32").reshape(4, 4)
    path = tmp_path / "synthetic.tif"
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)

    real_open = rasterio.open

    def fake_open(vsi_url, *args, **kwargs):
        # core.py always passes a /vsicurl/-prefixed URL - redirect to
        # the real local file regardless of what URL was requested.
        return real_open(str(path), *args, **kwargs)

    monkeypatch.setattr(rasterio, "open", fake_open)
    return path


def test_read_point_value_reads_matched_pixel(synthetic_raster):
    # row 0, col 1 -> value 1.0 (see np.arange(16).reshape(4, 4))
    value = read_point_value("http://example.invalid/fake.tif", lat=-25.03, lon=28.07)
    assert value == 1.0
    print("test_read_point_value_reads_matched_pixel: PASS")


def test_read_point_value_with_center_returns_matched_coordinate(synthetic_raster):
    value, matched_lat, matched_lon = read_point_value_with_center(
        "http://example.invalid/fake.tif", lat=-25.03, lon=28.07
    )
    assert value == 1.0
    # pixel [row0, col1] spans lat [-25.05, -25.00], lon [28.05, 28.10]
    assert matched_lat == pytest.approx(-25.025)
    assert matched_lon == pytest.approx(28.075)
    print("test_read_point_value_with_center_returns_matched_coordinate: PASS")


def test_read_point_value_out_of_bounds_raises(synthetic_raster):
    with pytest.raises(ValueError):
        read_point_value("http://example.invalid/fake.tif", lat=50.0, lon=50.0)
    print("test_read_point_value_out_of_bounds_raises: PASS")


def test_read_point_value_nodata_returns_nan(tmp_path, monkeypatch):
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(28.0, -25.0, 0.05, 0.05)
    data = np.full((4, 4), -9999.0, dtype="float32")
    path = tmp_path / "all_nodata.tif"
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)

    real_open = rasterio.open
    monkeypatch.setattr(
        rasterio, "open", lambda url, *a, **k: real_open(str(path), *a, **k)
    )

    value = read_point_value("http://example.invalid/fake.tif", lat=-25.03, lon=28.07)
    assert np.isnan(value)
    print("test_read_point_value_nodata_returns_nan: PASS")


# ----------------------------------------------------------------------
# fetch_chirps_timeseries orchestration - matched-coordinate capture
# ----------------------------------------------------------------------


def test_fetch_chirps_timeseries_orchestration_with_fakes():
    def fake_read(url, lat, lon):
        return (1.5, -25.025, 28.075)

    result = fetch_chirps_timeseries(
        lat=-25.03,
        lon=28.07,
        start_date="2020-01-01",
        end_date="2020-02-28",
        product="monthly",
        read_fn=fake_read,
        max_workers=1,
    )
    assert len(result.dataframe) == 2  # Jan + Feb
    assert (result.dataframe["PrecipitationMM"] == 1.5).all()
    assert result.dataframe["Date"].is_monotonic_increasing
    print("test_fetch_chirps_timeseries_orchestration_with_fakes: PASS")


def test_fetch_chirps_timeseries_reports_matched_coordinate():
    def fake_read(url, lat, lon):
        # deliberately different from the requested point below
        return (0.0, -25.025, 28.075)

    result = fetch_chirps_timeseries(
        lat=-25.03,
        lon=28.07,
        start_date="2020-01-01",
        end_date="2020-01-31",
        product="monthly",
        read_fn=fake_read,
        max_workers=1,
    )
    assert isinstance(result, ChirpsResult)
    assert any("matched to the nearest grid cell" in w for w in result.warnings)
    assert result.requested_lat == -25.03
    assert result.requested_lon == 28.07
    assert result.matched_lat == -25.025
    assert result.matched_lon == 28.075
    assert result.distance_km is not None and result.distance_km > 0
    print("test_fetch_chirps_timeseries_reports_matched_coordinate: PASS")


def test_fetch_chirps_timeseries_reports_matched_coordinate_concurrent():
    """Same as above but through the ThreadPoolExecutor path
    (max_workers > 1) - the matched-coordinate capture uses a separate
    lock from the progress counter specifically so this concurrent path
    is race-free too, not just the serial fallback."""

    def fake_read(url, lat, lon):
        return (0.0, -25.025, 28.075)

    result = fetch_chirps_timeseries(
        lat=-25.03,
        lon=28.07,
        start_date="2020-01-01",
        end_date="2020-06-30",
        product="monthly",
        read_fn=fake_read,
        max_workers=4,
    )
    assert result.matched_lat == -25.025
    assert result.matched_lon == 28.075
    assert any("matched to the nearest grid cell" in w for w in result.warnings)
    print("test_fetch_chirps_timeseries_reports_matched_coordinate_concurrent: PASS")


def test_fetch_chirps_timeseries_partial_failure_still_captures_matched_coordinate():
    def flaky_read(url, lat, lon):
        if "2020.02" in url:
            raise ConnectionError("simulated 404")
        return (2.0, -25.025, 28.075)

    result = fetch_chirps_timeseries(
        lat=-25.03,
        lon=28.07,
        start_date="2020-01-01",
        end_date="2020-03-31",
        product="monthly",
        read_fn=flaky_read,
        max_workers=1,
    )
    assert len(result.dataframe) == 2  # Jan + Mar succeeded, Feb failed
    assert any("2020-02" in w for w in result.warnings)
    assert result.matched_lat == -25.025
    assert result.matched_lon == 28.075
    print("test_fetch_chirps_timeseries_partial_failure_still_captures_match: PASS")


def test_fetch_chirps_timeseries_rejects_unknown_product():
    with pytest.raises(ValueError):
        fetch_chirps_timeseries(lat=0, lon=0, product="bogus")
    print("test_fetch_chirps_timeseries_rejects_unknown_product: PASS")
