"""Contour shapefile from the corrected DEM for the DEM Hydrological
Processing tool (GDAL / OGR only, no QGIS dependency).

The contours are traced at every multiple of the interval by GDAL's
contour generator (gdal.ContourGenerateEx) at the DEM's full resolution,
then smoothed with Chaikin corner cutting (SMOOTH_ITERATIONS passes;
closed lines stay closed, open lines keep their end points) and thinned
of the near-collinear vertices the smoothing adds (Douglas-Peucker,
SIMPLIFY_FRACTION of a cell). Lines stop
at NoData. Every MAJOR_FACTOR-th level is a major contour (report
standard 3.2).

Fields: id, elev_m (contour elevation, m), type ('Minor' or 'Major').
"""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import numpy as np

from mayim_tools.hydrology._common.contours import chaikin

MAJOR_FACTOR = 5
SMOOTH_ITERATIONS = 2  # Chaikin passes; 0 = the lines as traced
SIMPLIFY_FRACTION = 0.05  # after smoothing: tolerance as a fraction of a cell
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


def _line_wkb(pts: np.ndarray) -> bytes:
    """Little-endian WKB LineString of an (n, 2) float array."""
    head = struct.pack("<BII", 1, 2, len(pts))
    return head + np.ascontiguousarray(pts, dtype="<f8").tobytes()


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
    smooth_iterations: int = SMOOTH_ITERATIONS,
) -> int:
    """Writes the contours of ``surface`` (NaN or ``nodata`` = no data) at
    ``interval`` m to the shapefile ``path``, replacing an existing one,
    smoothed with ``smooth_iterations`` Chaikin passes (0 = as traced).
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

    srs = None
    if projection_wkt:
        srs = osr.SpatialReference()
        srs.ImportFromWkt(projection_wkt)
    mem = ogr.GetDriverByName("Memory").CreateDataSource("contours")
    traced = mem.CreateLayer("contours", srs, ogr.wkbLineString)
    traced.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    traced.CreateField(ogr.FieldDefn("elev_m", ogr.OFTReal))
    gdal.ContourGenerateEx(
        band,
        traced,
        options=[
            f"LEVEL_INTERVAL={interval!r}",
            "LEVEL_BASE=0",
            f"NODATA={fill!r}",
            "ID_FIELD=0",  # field indexes: id, elev_m
            "ELEV_FIELD=1",
        ],
    )

    _remove_shapefile(path)
    ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(path)
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
    defn = layer.GetLayerDefn()
    tolerance = SIMPLIFY_FRACTION * min(abs(geotransform[1]), abs(geotransform[5]))
    n = 0
    traced.ResetReading()
    for src_feat in traced:
        geom = src_feat.GetGeometryRef()
        if geom is None or geom.GetPointCount() < 2:
            continue
        pts = np.array(geom.GetPoints(), dtype=np.float64)[:, :2]
        if smooth_iterations > 0 and len(pts) >= 3:
            pts = chaikin(pts, smooth_iterations)
        line = ogr.CreateGeometryFromWkb(_line_wkb(pts))
        if smooth_iterations > 0 and tolerance > 0:
            # drop the near-collinear vertices corner cutting adds
            simpler = line.Simplify(tolerance)
            if simpler is not None and simpler.GetPointCount() >= 2:
                line = simpler
        level = src_feat.GetFieldAsDouble("elev_m")
        feat = ogr.Feature(defn)
        feat.SetGeometry(line)
        feat.SetField("id", n)
        feat.SetField("elev_m", level)
        feat.SetField("type", "Major" if is_major(level, interval) else "Minor")
        layer.CreateFeature(feat)
        n += 1
    layer.SyncToDisk()
    mem = None
    with open(os.path.splitext(path)[0] + ".cpg", "w", encoding="ascii") as f:
        f.write("UTF-8")
    ds = None
    src = None
    return n
