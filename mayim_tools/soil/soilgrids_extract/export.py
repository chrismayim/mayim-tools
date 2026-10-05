"""File writers for Extract: SoilGrids 2.0 (GeoTIFF, point CSV, metadata CSV)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .core import NODATA_OUT, TOOL_VERSION, TargetGrid

POINT_COLUMNS = [
    "Site",
    "Longitude",
    "Latitude",
    "Product",
    "Variable",
    "Description",
    "DepthTop_cm",
    "DepthBottom_cm",
    "Statistic",
    "Value",
    "Units",
    "Route",
]


def write_multiband_geotiff(
    path: str | Path,
    grid: TargetGrid,
    bands: list[tuple[str, np.ndarray]],
    units: str,
) -> None:
    """One Float32 GeoTIFF, one band per (description, array).

    Values are already in conventional units. NaN is written as the
    output nodata value (-9999); nodata is never written as zero.
    DEFLATE with the floating-point predictor keeps files small."""
    from osgeo import gdal

    gdal.UseExceptions()
    if not bands:
        raise ValueError("No bands to write.")
    rows, cols = bands[0][1].shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(path),
        cols,
        rows,
        len(bands),
        gdal.GDT_Float32,
        options=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
        ],
    )
    ds.SetGeoTransform(grid.geotransform)
    ds.SetProjection(grid.crs_wkt)
    ds.SetMetadataItem("SOURCE", "ISRIC SoilGrids (via mayim_tools)")
    ds.SetMetadataItem("TOOL_VERSION", TOOL_VERSION)
    for i, (description, arr) in enumerate(bands, start=1):
        if arr.shape != (rows, cols):
            raise ValueError(
                f"Band {i} ({description}) has shape {arr.shape}, "
                f"expected {(rows, cols)}."
            )
        out = np.where(np.isfinite(arr), arr, NODATA_OUT).astype(np.float32)
        band = ds.GetRasterBand(i)
        band.SetNoDataValue(NODATA_OUT)
        band.SetDescription(description)
        band.SetUnitType(units)
        band.WriteArray(out)
    ds.FlushCache()
    ds = None
    write_style_sidecar(path, bands[0][1])


# Single-band grey style for band 1. QGIS applies <name>.qml automatically
# when the raster is opened, so files the user opens by hand display as one
# band (not an RGB composite of the first three depths).
_QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.40.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="singlebandgray" grayBand="1" gradient="BlackToWhite"
     opacity="1" alphaBand="-1" nodataColor="">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>MinMax</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <contrastEnhancement>
        <minValue>{vmin}</minValue>
        <maxValue>{vmax}</maxValue>
        <algorithm>StretchToMinimumMaximum</algorithm>
      </contrastEnhancement>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def write_style_sidecar(path: str | Path, band1: np.ndarray) -> Path:
    """Write <raster>.qml (single-band grey, stretched to band 1's range)."""
    valid = np.asarray(band1, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    vmin, vmax = (float(valid.min()), float(valid.max())) if valid.size else (0, 1)
    if vmax <= vmin:
        vmax = vmin + 1
    qml = Path(path).with_suffix(".qml")
    qml.write_text(_QML.format(vmin=round(vmin, 6), vmax=round(vmax, 6)), "utf-8")
    return qml


def write_points_csv(rows: list[dict], path: str | Path) -> tuple[int, int]:
    """Long-format point CSV. Missing values are empty cells, never zero.
    Returns (n_rows, n_missing)."""
    n_missing = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(POINT_COLUMNS)
        for row in rows:
            value = row["Value"]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                n_missing += 1
                value = ""
            else:
                value = round(float(value), 4)
            w.writerow(
                [
                    row["Site"],
                    round(float(row["Longitude"]), 6),
                    round(float(row["Latitude"]), 6),
                    row["Product"],
                    row["Variable"],
                    row["Description"],
                    row["DepthTop_cm"],
                    row["DepthBottom_cm"],
                    row["Statistic"],
                    value,
                    row["Units"],
                    row["Route"],
                ]
            )
    return len(rows), n_missing


def write_metadata_csv(
    sections: list[tuple[str, list[str], list[list]]], path: str | Path
) -> None:
    """Sectioned CSV: '# <title>' line, header, rows, blank line."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for title, header, rows in sections:
            w.writerow([f"# {title}"])
            w.writerow(header)
            for row in rows:
                w.writerow(row)
            w.writerow([])


def metadata_path_for(output: str | Path, mode: str) -> Path:
    """Area mode: <folder>/soilgrids_metadata.csv.
    Point mode: <csv stem>_metadata.csv next to the point CSV."""
    output = Path(output)
    if mode == "area":
        return output / "soilgrids_metadata.csv"
    return output.with_name(output.stem + "_metadata.csv")
