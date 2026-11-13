"""
Import RAS result data - result rasters on the terrain grid (GDAL, NumPy).

The water surface between cells follows RAS Mapper's two options:

- "sloping": each facepoint takes the mean peak water level of the wet
  cells around it; inside a cell the surface is interpolated linearly on
  the fan of triangles (cell centre, facepoint i, facepoint i+1), so it
  slopes smoothly from cell to cell (RAS Mapper's "Sloping (cell corners)"
  rendering, simplified). As in RAS Mapper's "pretty" option (reduce
  shallow to horizontal), a cell that is only partly under water keeps its
  own level. Checked on project 2625038 against the RAS Mapper flood edge
  of the 2% AEP plan (sloping pretty): median distance 0.5 m, 90% within
  1.5 m (terrain cells 0.5 m); without the shallow rule 0.9 m and 3.7 m;
- "horizontal": one level per cell.

Depth = water level - terrain, where positive (above the threshold).
Velocity is the cell's peak speed; depth x velocity and the flood hazard
classes use the peak depth and the peak velocity, as a RAS Mapper
calculated layer on the "Max" profiles does.

Flood hazard classes H1-H6: Smith, Davey and Cox (2014), UNSW Water
Research Laboratory Technical Report 2014/07, as cited in Australian
Rainfall and Runoff (2019) Book 6 Chapter 7 (Table 6.7.4); the same
thresholds as Chris's RAS Mapper calculated layer 2pct_FHA.rasscript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import PlanResults

NODATA = -9999.0

HAZARD_CLASSES = {
    1: "H1 - generally safe for people, vehicles and buildings",
    2: "H2 - unsafe for small vehicles",
    3: "H3 - unsafe for vehicles, children and the elderly",
    4: "H4 - unsafe for people and all vehicles",
    5: "H5 - unsafe for people and vehicles; buildings need special design",
    6: "H6 - unsafe for people and vehicles; all buildings vulnerable",
}
HAZARD_COLOURS = {  # Chris's RAS_ARR_Flood_Hazard style
    1: "#96b9e1", 2: "#bee5e9", 3: "#bfdcb7",
    4: "#d9e8ad", 5: "#ffe7a6", 6: "#fabda6",
}  # fmt: skip


def hazard_class(depth, velocity) -> np.ndarray:
    """H1-H6 (0 where depth or velocity is missing), thresholds as in the
    rasscript: H6 d>4 or v>4 or dv>4; H5 d>2 or v>2 or dv>1; H4 d>1.2 or
    dv>0.6; H3 d>0.5; H2 d>0.3 or dv>0.3; else H1."""
    d = np.asarray(depth, float)
    v = np.asarray(velocity, float)
    dv = d * v
    out = np.ones(d.shape, np.uint8)
    out[(d > 0.3) | (dv > 0.3)] = 2
    out[d > 0.5] = 3
    out[(d > 1.2) | (dv > 0.6)] = 4
    out[(d > 2) | (v > 2) | (dv > 1)] = 5
    out[(d > 4) | (v > 4) | (dv > 4)] = 6
    out[~(np.isfinite(d) & np.isfinite(v))] = 0
    return out


@dataclass
class ResultGrids:
    gt: tuple
    projection: str
    depth: np.ndarray
    wse: np.ndarray
    velocity: np.ndarray
    dv: np.ndarray
    hazard: np.ndarray  # uint8, 0 = no data
    arrival_h: np.ndarray
    duration_h: np.ndarray
    terrain_path: str


def _mesh_bounds(plan: PlanResults):
    pts = np.vstack([a.facepoints for a in plan.areas])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def read_terrain(path: str, bounds, pad_cells: int = 2):
    """(array with NaN, geotransform, projection WKT) of the terrain over
    ``bounds`` (xmin, ymin, xmax, ymax)."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
        raise ValueError("Rotated terrain rasters are not supported.")
    xmin, ymin, xmax, ymax = bounds
    c0 = int(np.floor((xmin - gt[0]) / gt[1])) - pad_cells
    c1 = int(np.ceil((xmax - gt[0]) / gt[1])) + pad_cells
    r0 = int(np.floor((ymax - gt[3]) / gt[5])) - pad_cells
    r1 = int(np.ceil((ymin - gt[3]) / gt[5])) + pad_cells
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, ds.RasterXSize), min(r1, ds.RasterYSize)
    if c1 <= c0 or r1 <= r0:
        raise ValueError("The terrain does not cover the 2D flow areas.")
    band = ds.GetRasterBand(1)
    z = band.ReadAsArray(c0, r0, c1 - c0, r1 - r0).astype(float)
    nd = band.GetNoDataValue()
    if nd is not None:
        z[z == nd] = np.nan
    z[~np.isfinite(z)] = np.nan
    wgt = (gt[0] + c0 * gt[1], gt[1], 0.0, gt[3] + r0 * gt[5], 0.0, gt[5])
    return z, wgt, ds.GetProjection()


def rasterize_cells(polygons, ids, gt, shape) -> np.ndarray:
    """Cell index per pixel (-1 outside), pixel centres inside a cell."""
    from osgeo import gdal, ogr

    gdal.UseExceptions()
    drv = gdal.GetDriverByName("MEM")
    ds = drv.Create("", shape[1], shape[0], 1, gdal.GDT_Int32)
    ds.SetGeoTransform(gt)
    band = ds.GetRasterBand(1)
    band.Fill(-1)
    mem = ogr.GetDriverByName("Memory").CreateDataSource("cells")
    layer = mem.CreateLayer("cells", geom_type=ogr.wkbPolygon, srs=None)
    layer.CreateField(ogr.FieldDefn("cid", ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    for ring, cid in zip(polygons, ids, strict=True):
        if len(ring) < 4:
            continue
        r = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in ring:
            r.AddPoint_2D(float(x), float(y))
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(r)
        feat = ogr.Feature(defn)
        feat.SetField("cid", int(cid))
        feat.SetGeometry(poly)
        layer.CreateFeature(feat)
    gdal.PushErrorHandler("CPLQuietErrorHandler")  # layer and raster share a CRS
    try:
        gdal.RasterizeLayer(ds, [1], layer, options=["ATTRIBUTE=cid"])
    finally:
        gdal.PopErrorHandler()
    return band.ReadAsArray().astype(np.int64)


def _facepoint_levels(area, cell_level, wet) -> np.ndarray:
    out = np.full(len(area.facepoints), np.nan)
    for i, cells in enumerate(area.facepoint_cells):
        vals = [cell_level[c] for c in cells if wet[c]]
        if vals:
            out[i] = float(np.mean(vals))
    return out


def _sloping_levels(area, cell_ids_local, px, py, cell_level, fp_level):
    """Water level at pixel centres (px, py) of cells ``cell_ids_local``
    (one entry per pixel), on the fan of triangles of each cell."""
    out = np.full(len(px), np.nan)
    order = np.argsort(cell_ids_local, kind="stable")
    ids_sorted = cell_ids_local[order]
    starts = np.flatnonzero(np.r_[True, ids_sorted[1:] != ids_sorted[:-1]])
    ends = np.r_[starts[1:], len(ids_sorted)]
    for s, e in zip(starts, ends, strict=True):
        c = ids_sorted[s]
        sel = order[s:e]
        wc = cell_level[c]
        ring = area.cell_facepoints[c]
        if len(ring) < 3 or not np.isfinite(wc):
            out[sel] = wc
            continue
        cx, cy = area.centers[c]
        p = area.facepoints[ring]
        wp = fp_level[ring]
        wp = np.where(np.isfinite(wp), wp, wc)
        ang = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
        k = np.argsort(ang)
        ang, p, wp = ang[k], p[k], wp[k]
        dx, dy = px[sel] - cx, py[sel] - cy
        pa = np.arctan2(dy, dx)
        j = np.searchsorted(ang, pa) % len(ang)  # sector (j-1, j)
        i = (j - 1) % len(ang)
        v0x, v0y = p[i, 0] - cx, p[i, 1] - cy
        v1x, v1y = p[j, 0] - cx, p[j, 1] - cy
        det = v0x * v1y - v0y * v1x
        det = np.where(np.abs(det) < 1e-12, np.nan, det)
        a = (dx * v1y - dy * v1x) / det
        b = (v0x * dy - v0y * dx) / det
        a = np.clip(np.nan_to_num(a), 0, 1)
        b = np.clip(np.nan_to_num(b), 0, 1)
        scale = np.maximum(a + b, 1.0)  # outside the fan: onto its edge
        a, b = a / scale, b / scale
        out[sel] = wc + a * (wp[i] - wc) + b * (wp[j] - wc)
    return out


def build_grids(
    plan: PlanResults,
    terrain_path: str,
    method: str = "sloping",
    threshold: float = 0.001,
) -> ResultGrids:
    z, gt, projection = read_terrain(terrain_path, _mesh_bounds(plan))
    shape = z.shape
    rows, cols = np.indices(shape)
    px_all = gt[0] + (cols + 0.5) * gt[1]
    py_all = gt[3] + (rows + 0.5) * gt[5]
    wse = np.full(shape, np.nan)
    vel = np.full(shape, np.nan)
    arr = np.full(shape, np.nan)
    dur = np.full(shape, np.nan)
    offset = 0
    for area in plan.areas:
        n = len(area.centers)
        ids = np.arange(n) + offset
        cid = rasterize_cells(area.polygons, ids, gt, shape)
        inside = (cid >= offset) & (cid < offset + n)
        local = cid[inside] - offset
        wet = np.nan_to_num(area.max_depth) > threshold
        level = np.where(wet, area.max_wse, np.nan)
        vals = level[local]
        if method == "sloping":
            # RAS Mapper's "reduce shallow to horizontal": a cell that is
            # only partly under water keeps its own level; the surface
            # slopes only across cells the water covers completely
            zin = np.nan_to_num(z[inside], nan=-np.inf)
            zmax = np.full(n, -np.inf)
            np.maximum.at(zmax, local, zin)
            full = wet & (area.max_wse >= zmax)
            sel = full[local]
            if sel.any():
                fp = _facepoint_levels(area, area.max_wse, wet)
                vals[sel] = _sloping_levels(
                    area, local[sel], px_all[inside][sel], py_all[inside][sel],
                    level, fp,
                )  # fmt: skip
        wse[inside] = vals
        vel[inside] = area.max_velocity[local]
        arr[inside] = area.arrival_h[local]
        dur[inside] = area.duration_h[local]
        offset += n
    depth = wse - z
    ok = np.isfinite(depth) & (depth > threshold)
    depth = np.where(ok, depth, np.nan)
    wse = np.where(ok, wse, np.nan)
    vel = np.where(ok, vel, np.nan)
    dv = depth * vel
    return ResultGrids(
        gt=gt,
        projection=projection,
        depth=depth,
        wse=wse,
        velocity=vel,
        dv=dv,
        hazard=hazard_class(depth, vel) * ok,
        arrival_h=np.where(ok, arr, np.nan),
        duration_h=np.where(ok, dur, np.nan),
        terrain_path=terrain_path,
    )


def write_grid(path: str, array: np.ndarray, gt, projection, dtype="float32"):
    from osgeo import gdal

    gdal.UseExceptions()
    gtype = gdal.GDT_Byte if dtype == "uint8" else gdal.GDT_Float32
    nodata = 0 if dtype == "uint8" else NODATA
    data = (
        array.astype(dtype)
        if dtype == "uint8"
        else np.where(np.isfinite(array), array, NODATA).astype("float32")
    )
    ds = gdal.GetDriverByName("GTiff").Create(
        path, array.shape[1], array.shape[0], 1, gtype,
        options=["COMPRESS=LZW", "TILED=YES"],
    )  # fmt: skip
    ds.SetGeoTransform(gt)
    if projection:
        ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(data)
    band.FlushCache()
    ds = None


def terrain_sampler(path: str):
    """``f(x, y) -> terrain level`` (NaN off the raster or on no-data),
    nearest pixel."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    nd = band.GetNoDataValue()
    nx, ny = ds.RasterXSize, ds.RasterYSize

    def sample(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = np.full(len(x), np.nan)
        if not len(x):
            return out
        c = np.floor((x - gt[0]) / gt[1]).astype(np.int64)
        r = np.floor((y - gt[3]) / gt[5]).astype(np.int64)
        ok = (c >= 0) & (c < nx) & (r >= 0) & (r < ny)
        if not ok.any():
            return out
        c0, c1 = c[ok].min(), c[ok].max() + 1
        r0, r1 = r[ok].min(), r[ok].max() + 1
        z = band.ReadAsArray(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
        z = z.astype(float)
        if nd is not None:
            z[z == nd] = np.nan
        out[ok] = z[r[ok] - r0, c[ok] - c0]
        return out

    sample.cell_size = abs(gt[1])
    sample._ds = ds  # keep the dataset open
    return sample
