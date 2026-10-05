"""File writers shared by the soil tools (GeoTIFF + QML style, point CSV,
sectioned metadata CSV)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .grid import TargetGrid

NODATA_OUT = -9999.0


def write_multiband_geotiff(
    path: str | Path,
    grid: TargetGrid,
    bands: list[tuple[str, np.ndarray]],
    units: str,
    source: str = "",
    tool_version: str = "",
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
    if source:
        ds.SetMetadataItem("SOURCE", source)
    if tool_version:
        ds.SetMetadataItem("TOOL_VERSION", tool_version)
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


_PALETTED_QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.40.0" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="paletted" band="{band}" opacity="1" alphaBand="-1"
     nodataColor="">
      <rasterTransparency/>
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def class_colour(code: int) -> str:
    """Stable, well-spread colour for a class code (golden-angle hues)."""
    import colorsys

    hue = (code * 0.618033988749895) % 1.0
    sat = 0.55 + 0.35 * ((code * 7) % 3) / 2
    val = 0.75 + 0.2 * ((code * 5) % 2)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def write_paletted_style(
    path: str | Path, classes: list[tuple[int, str]], band: int = 1
) -> Path:
    """Write <raster>.qml with a paletted (categorised) renderer: one entry
    per (value, label), stable colours by value."""
    from xml.sax.saxutils import quoteattr

    lines = [
        f'        <paletteEntry value="{value}" color="{class_colour(value)}" '
        f'alpha="255" label={quoteattr(label)}/>'
        for value, label in classes
    ]
    qml = Path(path).with_suffix(".qml")
    qml.write_text(_PALETTED_QML.format(band=band, entries="\n".join(lines)), "utf-8")
    return qml


def write_points_csv(
    rows: list[dict], path: str | Path, columns: list[str]
) -> tuple[int, int]:
    """Long-format point CSV with the given columns. Missing values are
    empty cells, never zero; Longitude/Latitude are rounded to 6 decimals.
    Returns (n_rows, n_missing)."""
    n_missing = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in rows:
            out = []
            for col in columns:
                value = row[col]
                if col == "Value":
                    if value is None or (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        n_missing += 1
                        value = ""
                    else:
                        value = round(float(value), 4)
                elif col in ("Longitude", "Latitude"):
                    value = round(float(value), 6)
                out.append(value)
            w.writerow(out)
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


def metadata_path_for(
    output: str | Path, mode: str, folder_name: str = "metadata.csv"
) -> Path:
    """Area mode: <folder>/<folder_name>.
    Point mode: <csv stem>_metadata.csv next to the point CSV."""
    output = Path(output)
    if mode == "area":
        return output / folder_name
    return output.with_name(output.stem + "_metadata.csv")
