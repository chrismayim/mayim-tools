"""
Tests for mayim_tools/data/netcdf4_to_csv/core.py. Zero QGIS
dependency - run under plain pytest (see also the quality-gate
commands in the project notes: -p no:pytest_qgis is required because
of a globally-installed pytest_qgis plugin that otherwise crashes
collection).

Mirrors the structure of the source plugin's own test suite: variable
resolution, variable description, coordinate name resolution,
longitude convention resolution, flatten mode, point extraction mode,
and full end-to-end conversions - all against genuine synthetic
NetCDF files, not mocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mayim_tools.data.netcdf4_to_csv.core import (
    DEFAULT_MAX_FLATTEN_ROWS,
    convert_netcdf4_to_csv,
    describe_variables,
    extract_at_points,
    find_coord_name,
    flatten_to_dataframe,
    list_variables,
    resolve_longitude_for_file,
    resolve_variable,
)

xr = pytest.importorskip("xarray")


def make_dataset(
    var_name="precip",
    lat_name="lat",
    lon_name="lon",
    extra_var=None,
    lon_values=None,
):
    times = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    lats = np.array([5.0, 6.0, 7.0])
    lons = np.array(lon_values if lon_values is not None else [-2.0, -1.0, 0.0])
    data = np.arange(len(times) * len(lats) * len(lons), dtype="float64").reshape(
        len(times), len(lats), len(lons)
    )
    data_vars = {var_name: (["time", lat_name, lon_name], data)}
    if extra_var:
        data_vars[extra_var] = (["time", lat_name, lon_name], data * 2)
    ds = xr.Dataset(
        data_vars,
        coords={"time": times, lat_name: lats, lon_name: lons},
    )
    return ds


def write_temp_nc(ds, tmp_path) -> str:
    out_path = tmp_path / "test.nc"
    ds.to_netcdf(out_path)
    return str(out_path)


# ----------------------------------------------------------------------
# Variable resolution
# ----------------------------------------------------------------------


def test_list_variables_single(tmp_path):
    ds = make_dataset()
    path = write_temp_nc(ds, tmp_path)
    assert list_variables(path) == ["precip"]


def test_list_variables_multiple(tmp_path):
    ds = make_dataset(extra_var="temperature")
    path = write_temp_nc(ds, tmp_path)
    assert set(list_variables(path)) == {"precip", "temperature"}


def test_describe_variables_reports_dims_and_shape(tmp_path):
    ds = make_dataset()
    path = write_temp_nc(ds, tmp_path)
    info = describe_variables(path)
    assert len(info) == 1
    row = info[0]
    assert row["Variable"] == "precip"
    assert row["Dimensions"] == "time, lat, lon"
    assert row["Shape"] == "(3, 3, 3)"


def test_describe_variables_reads_units_and_long_name(tmp_path):
    ds = make_dataset()
    ds["precip"].attrs["units"] = "mm"
    ds["precip"].attrs["long_name"] = "Total precipitation"
    path = write_temp_nc(ds, tmp_path)
    info = describe_variables(path)
    assert info[0]["Units"] == "mm"
    assert info[0]["Description"] == "Total precipitation"


def test_describe_variables_falls_back_to_standard_name(tmp_path):
    ds = make_dataset()
    ds["precip"].attrs["standard_name"] = "precipitation_amount"
    # Deliberately no long_name - confirms the fallback, not just
    # long_name working.
    path = write_temp_nc(ds, tmp_path)
    info = describe_variables(path)
    assert info[0]["Description"] == "precipitation_amount"


def test_describe_variables_missing_attrs_are_empty_not_guessed(tmp_path):
    ds = make_dataset()  # no units/long_name/standard_name attrs set at all
    path = write_temp_nc(ds, tmp_path)
    info = describe_variables(path)
    assert info[0]["Units"] == ""
    assert info[0]["Description"] == ""


def test_describe_variables_multiple_variables_all_reported(tmp_path):
    ds = make_dataset(extra_var="temperature")
    path = write_temp_nc(ds, tmp_path)
    info = describe_variables(path)
    assert {row["Variable"] for row in info} == {"precip", "temperature"}


def test_resolve_variable_explicit_match():
    ds = make_dataset(extra_var="temperature")
    assert resolve_variable(ds, "temperature") == "temperature"


def test_resolve_variable_not_found_lists_options():
    ds = make_dataset(extra_var="temperature")
    with pytest.raises(ValueError, match="precip"):
        resolve_variable(ds, "nonexistent")


def test_resolve_variable_auto_detects_sole_variable():
    ds = make_dataset()
    assert resolve_variable(ds, None) == "precip"


def test_resolve_variable_ambiguous_without_selection_raises():
    ds = make_dataset(extra_var="temperature")
    with pytest.raises(ValueError, match="precip"):
        resolve_variable(ds, None)


# ----------------------------------------------------------------------
# Coordinate name resolution
# ----------------------------------------------------------------------


def test_find_coord_name_standard():
    ds = make_dataset(lat_name="lat", lon_name="lon")
    assert find_coord_name(ds, ("lat", "latitude")) == "lat"


def test_find_coord_name_alternate_convention():
    ds = make_dataset(lat_name="latitude", lon_name="longitude")
    assert find_coord_name(ds, ("lat", "latitude")) == "latitude"
    assert find_coord_name(ds, ("lon", "longitude")) == "longitude"


def test_find_coord_name_no_match_returns_none():
    ds = make_dataset()
    assert find_coord_name(ds, ("nonexistent_a", "nonexistent_b")) is None


# ----------------------------------------------------------------------
# Longitude convention resolution - a genuine problem specific to this
# general-purpose tool, since there's no known source to confirm the
# convention in advance.
# ----------------------------------------------------------------------


def test_resolve_longitude_already_in_range():
    ds = make_dataset(lon_values=[-2.0, -1.0, 0.0])
    result = resolve_longitude_for_file(ds, "lon", -1.5)
    assert result == -1.5


def test_resolve_longitude_falls_back_to_0_360():
    """File uses 0-360 convention; caller passes a standard -180/180
    value that doesn't fall in range - must be converted."""
    ds = make_dataset(lon_values=[357.0, 358.0, 359.0])
    result = resolve_longitude_for_file(ds, "lon", -1.7334)  # -> 358.2666
    assert abs(result - 358.2666) < 1e-4


def test_resolve_longitude_neither_convention_matches_returns_original():
    ds = make_dataset(lon_values=[10.0, 20.0, 30.0])
    result = resolve_longitude_for_file(ds, "lon", -170.0)
    assert result == -170.0  # falls back to the original value, not an error


# ----------------------------------------------------------------------
# Flatten mode
# ----------------------------------------------------------------------


def test_flatten_to_dataframe_correct_row_count_and_values():
    ds = make_dataset()
    df = flatten_to_dataframe(ds, "precip")
    assert len(df) == 3 * 3 * 3  # time x lat x lon
    assert set(df.columns) >= {"time", "lat", "lon", "precip"}
    row = df[
        (df["lat"] == 6.0)
        & (df["lon"] == -1.0)
        & (df["time"] == pd.Timestamp("2020-01-02"))
    ]
    assert len(row) == 1


def test_flatten_to_dataframe_size_safety_check():
    ds = make_dataset()
    with pytest.raises(ValueError, match="exceeds"):
        flatten_to_dataframe(ds, "precip", max_rows=5)  # 27 actual rows


def test_flatten_to_dataframe_drop_na():
    ds = make_dataset()
    ds["precip"].values[0, 0, 0] = np.nan
    df_kept = flatten_to_dataframe(ds, "precip", drop_na=False)
    df_dropped = flatten_to_dataframe(ds, "precip", drop_na=True)
    assert len(df_kept) == 27
    assert len(df_dropped) == 26


def test_default_max_flatten_rows_is_reasonable():
    assert 100_000 < DEFAULT_MAX_FLATTEN_ROWS < 50_000_000


# ----------------------------------------------------------------------
# Point extraction mode
# ----------------------------------------------------------------------


def test_extract_at_points_nearest_neighbour():
    ds = make_dataset()
    # nearest match: lat=6.0, lon=-1.0
    df = extract_at_points(ds, "precip", [("Site A", 6.1, -0.9)])
    assert len(df) == 3  # one row per timestep
    assert list(df["Site"].unique()) == ["Site A"]
    assert "lat" not in df.columns and "lon" not in df.columns

    flat = flatten_to_dataframe(ds, "precip")
    expected = (
        flat[(flat["lat"] == 6.0) & (flat["lon"] == -1.0)]
        .sort_values("time")["precip"]
        .values
    )
    actual = df.sort_values("time")["precip"].values
    assert np.array_equal(expected, actual)


def test_extract_at_points_multiple_sites():
    ds = make_dataset()
    df = extract_at_points(ds, "precip", [("Site A", 5.0, -2.0), ("Site B", 7.0, 0.0)])
    assert set(df["Site"].unique()) == {"Site A", "Site B"}
    assert len(df) == 6  # 2 sites x 3 timesteps


def test_extract_at_points_handles_0_360_convention_file():
    ds = make_dataset(lon_values=[357.0, 358.0, 359.0])
    df = extract_at_points(ds, "precip", [("Ghana site", 6.0, -1.7334)])
    assert len(df) == 3


def test_extract_at_points_missing_lat_lon_raises_clear_error():
    ds = xr.Dataset(
        {"precip": (["time"], [1.0, 2.0, 3.0])},
        coords={"time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])},
    )
    with pytest.raises(ValueError, match="(?i)latitude"):
        extract_at_points(ds, "precip", [("Site A", 6.0, -1.0)])


# ----------------------------------------------------------------------
# End-to-end, real file I/O
# ----------------------------------------------------------------------


def test_convert_netcdf4_to_csv_flatten_mode_end_to_end(tmp_path):
    ds = make_dataset()
    in_path = write_temp_nc(ds, tmp_path)
    out_path = str(tmp_path / "out.csv")
    info = convert_netcdf4_to_csv(in_path, out_path, mode="flatten")
    assert info["rows_written"] == 27
    assert info["variable"] == "precip"
    result_df = pd.read_csv(out_path)
    assert len(result_df) == 27


def test_convert_netcdf4_to_csv_points_mode_end_to_end(tmp_path):
    ds = make_dataset()
    in_path = write_temp_nc(ds, tmp_path)
    out_path = str(tmp_path / "out.csv")
    info = convert_netcdf4_to_csv(
        in_path, out_path, mode="points", points=[("Site A", 6.0, -1.0)]
    )
    assert info["rows_written"] == 3
    result_df = pd.read_csv(out_path)
    assert list(result_df["Site"].unique()) == ["Site A"]


def test_convert_netcdf4_to_csv_points_mode_requires_points(tmp_path):
    ds = make_dataset()
    in_path = write_temp_nc(ds, tmp_path)
    out_path = str(tmp_path / "out.csv")
    with pytest.raises(ValueError, match="(?i)points"):
        convert_netcdf4_to_csv(in_path, out_path, mode="points", points=None)


def test_convert_netcdf4_to_csv_invalid_mode_raises(tmp_path):
    ds = make_dataset()
    in_path = write_temp_nc(ds, tmp_path)
    out_path = str(tmp_path / "out.csv")
    with pytest.raises(ValueError, match="(?i)mode"):
        convert_netcdf4_to_csv(in_path, out_path, mode="nonsense")
