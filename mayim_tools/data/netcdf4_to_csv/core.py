"""
Core NetCDF4 to CSV conversion logic. Zero QGIS dependency - unit
tested standalone (see tests/test_netcdf4_to_csv_core.py).

General-purpose - unlike every point-extraction plugin in this suite
(CHIRPS, IMERG, MERRA2, CMORPH, PERSIANN), this isn't tied to a
specific, known data source, so it can't rely on a confirmed variable
name, coordinate naming convention, or longitude convention the way
those tools could. Everything here is written defensively for an
arbitrary, CF-ish NetCDF4 file rather than assuming a known structure.

Two modes:

- "flatten": every value in one variable, across ALL its dimensions,
  as a long-format table (one row per unique combination of dimension
  values - e.g. one row per time x lat x lon). The general-purpose
  conversion mirror of the existing GRIB to CSV tool, for NetCDF4.

- "points": nearest-neighbour extraction at one or more given points,
  matching the point/point-layer extraction pattern every other
  plugin in this suite uses.

Coordinate/variable names are resolved defensively (tries several
common conventions) since an arbitrary file's exact naming isn't known
in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAT_COORD_CANDIDATES = ("lat", "latitude", "y", "Latitude", "Y", "LAT", "LATITUDE")
LON_COORD_CANDIDATES = ("lon", "longitude", "x", "Longitude", "X", "LON", "LONGITUDE")

# A reasonable "still usable in a spreadsheet/pandas" ceiling.
DEFAULT_MAX_FLATTEN_ROWS = 5_000_000


def list_variables(path: str) -> list:
    """Lists the data variable names in a NetCDF4 file (coordinate
    variables like lat/lon/time are excluded by xarray's own
    data_vars/coords distinction). Used both for a QGIS parameter's
    own validation and to report available options when the user
    hasn't specified one and the file has more than one."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return list(ds.data_vars)


def describe_variables(path: str) -> list:
    """Returns per-variable metadata (name, dimensions, shape, units,
    a human-readable description) for every data variable in a
    NetCDF4 file - a bare variable name (often a short, cryptic
    short_name like 'tp' or 'var129') isn't enough on its own to tell
    a user which variable they actually want. Units and description
    are read from each variable's own attributes when present (the
    'units' attribute, and 'long_name' or 'standard_name' - whichever
    is present, long_name preferred since it's usually the more
    human-readable of the two) - genuinely absent for a non-CF-
    compliant file, in which case they're reported as empty rather
    than guessed at."""
    import xarray as xr

    rows = []
    with xr.open_dataset(path) as ds:
        for name in ds.data_vars:
            da = ds[name]
            attrs = da.attrs
            description = attrs.get("long_name") or attrs.get("standard_name") or ""
            rows.append(
                {
                    "Variable": name,
                    "Dimensions": ", ".join(da.dims),
                    "Shape": str(da.shape),
                    "Units": attrs.get("units", ""),
                    "Description": description,
                }
            )
    return rows


def resolve_variable(ds, variable):
    """If variable is given, confirms it exists (fails with the real
    list of options if not, rather than a bare KeyError). If not
    given, auto-selects it only when the file has exactly one data
    variable - otherwise raises rather than guessing which one the
    user wants."""
    data_vars = list(ds.data_vars)
    if variable:
        if variable not in data_vars:
            raise ValueError(
                f"Variable {variable!r} not found in this file. "
                f"Available variables: {data_vars}"
            )
        return variable
    if len(data_vars) == 1:
        return data_vars[0]
    raise ValueError(
        f"No variable specified, and this file has {len(data_vars)} data "
        f"variables - specify one explicitly. Available variables: {data_vars}"
    )


def find_coord_name(ds, candidates):
    """Defensive coordinate lookup - tries each candidate name in
    turn, returns None if none match. The caller decides whether a
    miss is fatal (missing lat/lon is fatal for point extraction; a
    missing time dimension is fine - plenty of real NetCDF files are
    static/single-timestep grids with no time axis at all)."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def resolve_longitude_for_file(ds, lon_name, lon):
    """An arbitrary NetCDF file might use either the -180/180 or the
    0-360 longitude convention, and unlike every other plugin in this
    suite, there's no specific known data source to confirm which one
    in advance. If the requested longitude doesn't fall within the
    file's own coordinate range, tries the 0-360-converted equivalent
    and uses that instead if IT falls in range - a defensive fallback
    for the single most common convention mismatch, not a full
    general solution (a file using some other, unusual convention
    would still need the caller to pass in an already-matching
    value)."""
    lon_coord = np.asarray(ds[lon_name].values)
    lon_min, lon_max = float(lon_coord.min()), float(lon_coord.max())
    if lon_min <= lon <= lon_max:
        return lon
    lon_360 = lon % 360.0
    if lon_min <= lon_360 <= lon_max:
        return lon_360
    return lon  # neither matches - let nearest-neighbour do its best


def flatten_to_dataframe(
    ds, variable, max_rows=DEFAULT_MAX_FLATTEN_ROWS, drop_na=False
):
    """Converts one variable's ENTIRE contents to a long-format
    DataFrame - one row per unique combination of its dimension
    values (e.g. time x lat x lon). Uses xarray's own .to_dataframe()
    (handles arbitrary dimension combinations correctly, not hand-
    rolled), then resets the index so every dimension becomes a plain
    column.

    Raises BEFORE attempting the conversion if the estimated row
    count exceeds max_rows - flattening a large multi-dimensional grid
    can produce an unusable number of rows, which would exhaust
    memory or hang rather than produce something usable - better to
    fail fast with a clear, actionable message.

    drop_na: if True, rows where the variable's value is NaN are
    dropped (useful for e.g. a land-only variable on a global grid,
    where most ocean cells would otherwise just be empty rows) -
    defaults to False, since a general-purpose conversion tool should
    faithfully reproduce what's in the file unless asked not to.
    """
    da = ds[variable]
    estimated_rows = int(np.prod(da.shape)) if da.shape else 1
    if estimated_rows > max_rows:
        raise ValueError(
            f"Flattening '{variable}' would produce approximately "
            f"{estimated_rows:,} rows (shape {da.shape} across dims "
            f"{da.dims}), which exceeds the safety limit of {max_rows:,}. "
            f"This is usually too large to be a usable CSV - consider "
            f"point extraction mode instead, or explicitly raise the row "
            f"limit if the full flattened output is genuinely what's "
            f"wanted."
        )
    df = da.to_dataframe().reset_index()
    if drop_na:
        df = df.dropna(subset=[variable])
    return df


def extract_at_points(ds, variable, points, lat_name=None, lon_name=None):
    """Nearest-neighbour extraction of one variable at each given
    point, matching the point-extraction pattern every other plugin
    in this suite uses. Returns a long-format DataFrame with a 'Site'
    column plus whatever other dimensions the variable has (typically
    time) and the variable's own values under its own name.

    points: list of (label, lat, lon) tuples.
    """
    lat_name = lat_name or find_coord_name(ds, LAT_COORD_CANDIDATES)
    lon_name = lon_name or find_coord_name(ds, LON_COORD_CANDIDATES)
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not identify latitude/longitude coordinates in this "
            f"file - tried {LAT_COORD_CANDIDATES} for latitude and "
            f"{LON_COORD_CANDIDATES} for longitude. Available "
            f"coordinates: {list(ds.coords)}"
        )

    da = ds[variable]
    frames = []
    for label, lat, lon in points:
        resolved_lon = resolve_longitude_for_file(ds, lon_name, lon)
        point_da = da.sel(**{lat_name: lat, lon_name: resolved_lon}, method="nearest")
        point_df = point_da.to_dataframe(name=variable).reset_index()
        # Drop the (now-constant, nearest-MATCHED) lat/lon columns - they
        # describe the grid cell that was matched, not the requested
        # point; every other extraction plugin in this suite identifies
        # sites by label, not by echoing the matched cell's coordinates.
        point_df = point_df.drop(
            columns=[c for c in (lat_name, lon_name) if c in point_df.columns]
        )
        point_df.insert(0, "Site", label)
        frames.append(point_df)
    return pd.concat(frames, ignore_index=True)


def convert_netcdf4_to_csv(
    input_path,
    output_path,
    variable=None,
    mode="flatten",
    points=None,
    max_rows=DEFAULT_MAX_FLATTEN_ROWS,
    drop_na=False,
):
    """End-to-end: open the file, resolve the variable, convert via
    the requested mode, write CSV. Returns a small info dict (rows
    written, variable used, mode) for the caller to report back -
    e.g. so a QGIS algorithm can log a clear summary."""
    import xarray as xr

    if mode not in ("flatten", "points"):
        raise ValueError(f"mode must be 'flatten' or 'points', got {mode!r}")
    if mode == "points" and not points:
        raise ValueError("mode='points' requires at least one point")

    with xr.open_dataset(input_path) as ds:
        resolved_variable = resolve_variable(ds, variable)
        if mode == "flatten":
            df = flatten_to_dataframe(
                ds, resolved_variable, max_rows=max_rows, drop_na=drop_na
            )
        else:
            df = extract_at_points(ds, resolved_variable, points)

    df.to_csv(output_path, index=False)
    return {"rows_written": len(df), "variable": resolved_variable, "mode": mode}
