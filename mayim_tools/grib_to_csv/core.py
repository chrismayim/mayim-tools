"""
Core GRIB -> CSV conversion logic.

BACKEND CHANGE (v0.2): switched from GDAL/rasterio to xarray + cfgrib
(eccodes). The original rasterio-based version iterated GDAL raster
bands one at a time, which is fine for small files but catastrophically
slow on ERA5-scale data: GDAL's built-in GRIB driver does NOT use
eccodes - it has its own general-purpose decoder that is dramatically
slower than eccodes at ECMWF's typical "complex packing with spatial
differencing" GRIB2 encoding. A multi-year hourly ERA5 file has tens
of thousands of individual GRIB messages, so that per-message decode
penalty compounds into roughly a 60x real-world slowdown (~1-2 hours
vs ~2 minutes, confirmed against a working reference implementation).

eccodes is ECMWF's own purpose-built library for exactly this data, so
this version uses it via cfgrib/xarray instead - the same approach
proven fast on real ERA5 files. Logic mirrors that proven reference
implementation closely (variable selection, metre->mm conversion,
NaN handling) rather than introducing untested cleverness.

Zero QGIS/Qt dependency - independently testable outside QGIS, same
principle as design_rainfall's core.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import xarray as xr

_METRE_UNITS = {"m", "meter", "metre", "meters", "metres"}


@dataclass
class VariableInfo:
    name: str
    long_name: str
    units: str
    dims: tuple[str, ...]


def read_variable_info(path: str | Path) -> list[VariableInfo]:
    """Inspects a GRIB file's variables without materialising the full
    grid - used to tell the user what's available before they commit to
    an export (e.g. surfaced in the Processing algorithm's log output,
    since QGIS has no built-in widget that can populate a dropdown from
    an arbitrary GRIB file's contents)."""
    infos = []
    with xr.open_dataset(str(path), engine="cfgrib") as ds:
        for name, da in ds.data_vars.items():
            infos.append(VariableInfo(
                name=str(name),
                long_name=str(da.attrs.get("long_name", "")),
                units=str(da.attrs.get("units", "")),
                dims=tuple(str(d) for d in da.dims),
            ))
    return infos


def grib_to_csv(
    input_path: str | Path,
    output_path: str | Path,
    variable: str | None = None,
    convert_metres_to_mm: bool = True,
    drop_na: bool = True,
    decimal_places: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> int:
    """Converts one variable from a GRIB file to CSV - one row per
    grid-cell/time (or grid-cell/step/level, whatever dimensions the
    variable actually has) combination. Returns the row count.

    variable: GRIB shortName to export (e.g. 'tp' for ERA5 total
    precipitation); None = first variable in the file.
    convert_metres_to_mm: ERA5 precipitation is stored in metres by
    convention; multiplies by 1000 and relabels units when the source
    units are metre-based. No-ops if units are already something else.
    drop_na: omits rows with a NaN value for the selected variable
    (matches the reference implementation's behaviour).
    decimal_places: None (default) preserves full float precision,
    matching the reference implementation exactly. Set explicitly to
    round if a smaller/tidier file is wanted.
    """
    input_path = str(input_path)
    if progress_callback:
        progress_callback(5)

    with xr.open_dataset(input_path, engine="cfgrib") as dataset:
        data_variables = list(dataset.data_vars)
        if not data_variables:
            raise ValueError(f"No data variables found in {input_path}")
        selected = variable or data_variables[0]
        if selected not in dataset.data_vars:
            raise ValueError(f"Variable {selected!r} not found. Available: {data_variables}")

        if progress_callback:
            progress_callback(20)

        subset = dataset[[selected]]
        units = str(subset[selected].attrs.get("units", ""))
        if convert_metres_to_mm and units.lower() in _METRE_UNITS:
            subset[selected] = subset[selected] * 1000.0
            subset[selected].attrs["units"] = "mm"

        if progress_callback:
            progress_callback(50)

        frame = subset.to_dataframe().reset_index()
        if drop_na:
            frame = frame.dropna(subset=[selected])

        if progress_callback:
            progress_callback(80)

        if decimal_places is not None:
            frame[selected] = frame[selected].round(decimal_places)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    if progress_callback:
        progress_callback(100)

    return len(frame)
