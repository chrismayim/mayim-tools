"""Contour shapefile from the corrected DEM for the DEM Hydrological
Processing tool (GDAL / OGR only, no QGIS dependency).

The contours are traced at every multiple of the interval by GDAL's
contour generator (gdal.ContourGenerateEx), at the DEM's full
resolution and without smoothing. Lines stop at NoData. Every
MAJOR_FACTOR-th level is a major contour (report standard 3.2).

Fields: id, elev_m (contour elevation, m), type ('Minor' or 'Major').
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

MAJOR_FACTOR = 5
MAX_LEVELS = 2000  # more levels than this: the interval is too small
SHAPEFILE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qmd", ".qix")


def contour_levels(lo: float, hi: float, interval: float) -> int:
    """Number of contour levels (multiples of interval) between lo and hi."""
    if interval <= 0 or not (math.isfinite(lo) and math.isfinite(hi)):
        return 0
    return max(0, int(math.floor(hi / interval) - math.ceil(lo / interval)) + 1)


def is_major(elev: float, interval: float, factor: int = MAJOR_FACTOR) -> bool:
    """True when elev is a multiple of factor x interval (to rounding)."""
    q = elev / (factor * interval)
    return abs(q - round(q)) < 1e-6


def _remove_shapefile(path: str) -> None:
    base = os.path.splitext(path)[0]
    for ext in SHAPEFILE_PARTS:
        part = base + ext
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError as e:
                raise RuntimeError(
                    f"{Path(path).name} exists and could not be replaced - it is "
                    "probably open in QGIS. Remove it from the project and run "
                    "again."
                ) from e


def write_contours(
    path: str,
    surface: np.ndarray,
    geotransform,
    projection_wkt: str,
    interval: float,
    nodata: float | None = None,
    max_levels: int = MAX_LEVELS,
) -> int:
    """Writes the contours of ``surface`` (NaN or ``nodata`` = no data) at
    ``interval`` m to the shapefile ``path``, replacing an existing one.
    Returns the number of contour lines. Raises ValueError when the
    interval is not positive or gives more than ``max_levels`` levels."""
    from osgeo import gdal, ogr, osr

    if interval is None or interval <= 0:
        raise ValueError("the contour interval must be greater than 0 m")
    z = np.asarray(surface, dtype=np.float64).copy()
    if nodata is not None:
        z[z == nodata] = np.nan
    finite = np.isfinite(z)
    if finite.sum() < 4:
        raise ValueError("the corrected DEM has too few valid cells for contours")
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    n_levels = contour_levels(lo, hi, interval)
    if n_levels > max_levels:
        raise ValueError(
            f"a {interval:g} m interval gives {n_levels} contour levels over the "
            f"{hi - lo:.1f} m relief of the DEM (more than {max_levels}); use a "
            "larger contour interval"
        )
    fill = -1.0e30
    z[~finite] = fill

    rows, cols = z.shape
    src = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Float64)
    src.SetGeoTransform(tuple(geotransform))
    if projection_wkt:
        src.SetProjection(projection_wkt)
    band = src.GetRasterBand(1)
    band.SetNoDataValue(fill)
    band.WriteArray(z)

    _remove_shapefile(path)
    srs = None
    if projection_wkt:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection_wkt)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(path)
    if ds is None:
        raise RuntimeError(f"{Path(path).name} could not be created.")
    layer = ds.CreateLayer(Path(path).stem, srs, ogr.wkbLineString)
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    elev = ogr.FieldDefn("elev_m", ogr.OFTReal)
    elev.SetWidth(12)
    elev.SetPrecision(3)
    layer.CreateField(elev)
    kind = ogr.FieldDefn("type", ogr.OFTString)
    kind.SetWidth(5)
    layer.CreateField(kind)

    gdal.ContourGenerateEx(
        band,
        layer,
        options=[
            f"LEVEL_INTERVAL={interval!r}",
            "LEVEL_BASE=0",
            f"NODATA={fill!r}",
            "ID_FIELD=0",  # field indexes: id, elev_m
            "ELEV_FIELD=1",
        ],
    )
    layer.ResetReading()
    n = 0
    for feat in layer:
        level = feat.GetFieldAsDouble("elev_m")
        feat.SetField("type", "Major" if is_major(level, interval) else "Minor")
        layer.SetFeature(feat)
        n += 1
    layer.SyncToDisk()
    with open(os.path.splitext(path)[0] + ".cpg", "w", encoding="ascii") as f:
        f.write("UTF-8")
    ds = None
    src = None
    return n
