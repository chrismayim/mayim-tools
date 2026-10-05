"""Thread-safe GDAL reading of (remote) rasters - the only module that
imports osgeo. Worker threads only download native pixels; every PROJ
operation runs under PROJ_LOCK (see the note above PROJ_LOCK)."""

from __future__ import annotations

import math
import threading
import uuid
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np

from .errors import SoilDataError
from .grid import TargetGrid

WINDOW_LIMIT_CELLS = 2000  # point mode: read one window if points fit in this


HTTP_OPTIONS = {
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "VSI_CACHE": "TRUE",
}


def _gdal():
    from osgeo import gdal

    gdal.UseExceptions()
    return gdal


def gdal_version() -> str:
    try:
        return _gdal().__version__
    except Exception:  # noqa: BLE001
        return "unavailable"


@contextmanager
def http_options(options: dict | None = None):
    """Set GDAL HTTP options for a whole run (set once, before any worker
    thread starts, because config options are process-wide), then restore."""
    gdal = _gdal()
    options = HTTP_OPTIONS if options is None else options
    old = {k: gdal.GetConfigOption(k) for k in options}
    for key, value in options.items():
        gdal.SetConfigOption(key, value)
    try:
        yield
    finally:
        for key, value in old.items():
            gdal.SetConfigOption(key, value)


def _crs_transform(src_wkt: str, dst_wkt: str):
    """Coordinate transformation between two WKT CRSs (x/y = lon/lat order).
    Callers hold PROJ_LOCK when used from worker threads."""
    from osgeo import osr

    osr.UseExceptions()
    src = osr.SpatialReference()
    src.ImportFromWkt(src_wkt)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromWkt(dst_wkt)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(src, dst)


def _wgs84_wkt() -> str:
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs.ExportToWkt()


def _lonlat_to(srs_wkt: str):
    return _crs_transform(_wgs84_wkt(), srs_wkt)


def _unlink(gdal, mem) -> None:
    """Remove a temporary in-memory mosaic. Never raises: a failed clean-up
    must not hide the error that is being reported."""
    if not mem:
        return
    try:
        gdal.Unlink(mem)
    except Exception:  # noqa: BLE001
        pass


def _open_source(gdal, source):
    """Open a single path, or mosaic a list of tile paths into a small
    in-memory VRT. Returns (dataset, vsimem_path or None)."""
    if isinstance(source, str):
        return gdal.Open(source), None
    mem = f"/vsimem/mayim_soil_{uuid.uuid4().hex}.vrt"
    ds = gdal.BuildVRT(mem, list(source))
    if ds is None:
        raise SoilDataError("Could not open the tiles.")
    return ds, mem


# Reprojection (PROJ) is not reliably thread-safe for the Interrupted
# Goode Homolosine projection: parallel gdal.Warp calls crashed
# intermittently in testing. So worker threads only DOWNLOAD native pixels
# (the slow, network-bound part); every coordinate transformation and the
# quick warp of the downloaded block run one at a time under this lock.
PROJ_LOCK = threading.Lock()


def _native_window(gdal, ds, grid: TargetGrid, pad: int = 2, strict: bool = False):
    """Pixel window of ``ds`` that covers ``grid`` (densified edges).
    With ``strict`` (tile mosaics) the raster must cover the whole grid."""
    tr = _crs_transform(grid.crs_wkt, ds.GetProjection())
    inv = gdal.InvGeoTransform(ds.GetGeoTransform())
    n = 16
    xs, ys = [], []
    for i in range(n + 1):
        for j in range(n + 1):
            gx = grid.xmin + (grid.xmax - grid.xmin) * i / n
            gy = grid.ymin + (grid.ymax - grid.ymin) * j / n
            try:
                x, y, _ = tr.TransformPoint(gx, gy)
            except Exception:  # noqa: BLE001 - point outside the projection
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            xs.append(inv[0] + inv[1] * x + inv[2] * y)
            ys.append(inv[3] + inv[4] * x + inv[5] * y)
    if not xs:
        if strict:
            raise SoilDataError("the tiles do not cover the area of interest")
        return None
    if strict and (
        min(xs) < 0
        or min(ys) < 0
        or max(xs) > ds.RasterXSize
        or max(ys) > ds.RasterYSize
    ):
        raise SoilDataError("the tiles do not cover the area of interest")
    x0 = max(0, int(math.floor(min(xs))) - pad)
    y0 = max(0, int(math.floor(min(ys))) - pad)
    x1 = min(ds.RasterXSize, int(math.floor(max(xs))) + 1 + pad)
    y1 = min(ds.RasterYSize, int(math.floor(max(ys))) + 1 + pad)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1 - x0, y1 - y0


def gdal_read_grid(source, grid: TargetGrid, with_scale: bool = False):
    """Warp a (remote) raster - one path or a list of tiles - onto ``grid``
    with nearest neighbour. Float64 with NaN for nodata / outside coverage.
    Raw stored values (no scale applied). With ``with_scale`` returns
    (array, scale, offset) from the band metadata (None if not set).

    Thread-safe: the native block is downloaded without the lock; the
    window calculation and the in-memory warp run under PROJ_LOCK."""
    gdal = _gdal()
    src, mem = _open_source(gdal, source)
    try:
        band = src.GetRasterBand(1)
        scale, offset = band.GetScale(), band.GetOffset()

        def result(arr):
            return (arr, scale, offset) if with_scale else arr

        with PROJ_LOCK:
            window = _native_window(gdal, src, grid, strict=mem is not None)
        out = np.full((grid.height, grid.width), np.nan)
        if window is None:
            return result(out)
        x0, y0, w, h = window
        nodata = band.GetNoDataValue()
        block = band.ReadAsArray(x0, y0, w, h).astype(np.float32)  # network I/O
        if nodata is not None:
            block[block == nodata] = np.nan
        gt = src.GetGeoTransform()
        block_gt = (
            gt[0] + x0 * gt[1] + y0 * gt[2],
            gt[1],
            gt[2],
            gt[3] + x0 * gt[4] + y0 * gt[5],
            gt[4],
            gt[5],
        )
        srs_wkt = src.GetProjection()
        with PROJ_LOCK:
            mem_src = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Float32)
            mem_src.SetGeoTransform(block_gt)
            mem_src.SetProjection(srs_wkt)
            mband = mem_src.GetRasterBand(1)
            mband.SetNoDataValue(float("nan"))
            mband.WriteArray(block)
            ds = gdal.Warp(
                "",
                mem_src,
                format="MEM",
                outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
                width=grid.width,
                height=grid.height,
                dstSRS=grid.crs_wkt,
                resampleAlg="near",
                outputType=gdal.GDT_Float32,
                srcNodata=float("nan"),
                dstNodata=float("nan"),
            )
            if ds is None:
                raise SoilDataError(f"GDAL could not warp {source}")
            out = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
            ds = None
            mem_src = None
        return result(out)
    finally:
        src = None
        _unlink(gdal, mem)


def gdal_sample_points(
    source, lonlats: Sequence[tuple[float, float]], with_scale: bool = False
):
    """Nearest-cell values of a (remote) raster - one path or a list of
    tiles - at lon/lat points. One window read when the points fit in
    WINDOW_LIMIT_CELLS, otherwise one cell per point. NaN for nodata or
    points outside the raster. Raw stored values; with ``with_scale``
    returns (values, scale, offset) from the band metadata."""
    gdal = _gdal()
    ds, mem = _open_source(gdal, source)
    try:
        inv = gdal.InvGeoTransform(ds.GetGeoTransform())
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        cols, rows = ds.RasterXSize, ds.RasterYSize
        pix = []
        with PROJ_LOCK:
            tr = _lonlat_to(ds.GetProjection())
            for lon, lat in lonlats:
                x, y, _ = tr.TransformPoint(lon, lat)
                px = int(math.floor(inv[0] + inv[1] * x + inv[2] * y))
                py = int(math.floor(inv[3] + inv[4] * x + inv[5] * y))
                pix.append((px, py))
        inside = [(px, py) for px, py in pix if 0 <= px < cols and 0 <= py < rows]
        if mem is not None and len(inside) < len(pix):
            raise SoilDataError("the tiles do not cover every point")
        window = None
        if inside:
            x0 = min(p[0] for p in inside)
            x1 = max(p[0] for p in inside)
            y0 = min(p[1] for p in inside)
            y1 = max(p[1] for p in inside)
            if (x1 - x0 + 1) <= WINDOW_LIMIT_CELLS and (
                y1 - y0 + 1
            ) <= WINDOW_LIMIT_CELLS:
                window = (x0, y0, band.ReadAsArray(x0, y0, x1 - x0 + 1, y1 - y0 + 1))
        values: list[float] = []
        for px, py in pix:
            if not (0 <= px < cols and 0 <= py < rows):
                values.append(math.nan)
                continue
            if window is not None:
                wx0, wy0, arr = window
                v = float(arr[py - wy0, px - wx0])
            else:
                v = float(band.ReadAsArray(px, py, 1, 1)[0, 0])
            if (nodata is not None and v == nodata) or math.isnan(v):
                v = math.nan
            values.append(v)
        if with_scale:
            return values, band.GetScale(), band.GetOffset()
        return values
    finally:
        ds = None
        _unlink(gdal, mem)
