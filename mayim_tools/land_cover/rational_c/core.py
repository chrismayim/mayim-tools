"""
Core of "Run-off Coefficients (SANRAL Deterministic)" - no QGIS dependency (GDAL for the
raster I/O, NumPy for the arithmetic).

All inputs are brought onto one analysis grid (default 20 m) in the
catchments' projected CRS: the land cover and the soil rasters by nearest
neighbour, the DEM by averaging (bilinear when the DEM is coarser than the
analysis cell). Every cell gets a role, slope, permeability and vegetation
class and an urban C2 (cells.py); each catchment then gets the area shares
of the classes and the coefficients of the SANRAL Drainage Manual (2013)
calculation form (coefficients.py).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from mayim_tools.soil._common.grid import TargetGrid

from . import cells as cl
from . import coefficients as co
from . import lookup as lk
from . import tables as tb

TOOL_NAME = "Run-off Coefficients (SANRAL Deterministic)"
TOOL_VERSION = "0.1.0"
NODATA_F = -9999.0
NODATA_U8 = 255
DEFAULT_CELL_M = 20.0
MAX_CELLS = 40_000_000  # analysis grid cells (memory guard)
MAX_NATIVE_CELLS = 25_000_000  # per catchment, for the native-resolution slope
COVERAGE_WARN = 0.95
MINE_FLAG_SHARE = 0.01
FELLED_FLAG_SHARE = 0.30
METHODS = {
    "NEH630": "NEH 630 (USDA-NRCS 2009)",
    "SCSSA": "SCS-SA (Schulze et al. 1992)",
    "": "unknown (codes 1-4 = A-D assumed)",
}


class RationalCError(Exception):
    pass


# ----------------------------------------------------------------------
# Settings and results
# ----------------------------------------------------------------------


@dataclass
class Catchment:
    cid: str
    wkt: str  # in the analysis CRS
    map_mm: float | None = None  # per-catchment MAP (else the run's MAP)
    ft_override: str | None = None


@dataclass
class Override:
    """A site override polygon. Any of: a land-cover code, a vegetation
    class (0-3), a Table 3.7 urban row key (or 'lawn'), a fixed C2."""

    wkt: str
    lc_code: int | None = None
    cv: int | None = None
    urban: str | None = None
    c2: float | None = None
    note: str = ""


@dataclass
class RunSettings:
    catchments: list  # [Catchment]
    crs_wkt: str
    dem_path: str
    landcover_path: str
    hsg_path: str
    map_mm: float
    cell_m: float = DEFAULT_CELL_M
    lookup_key: str = "sanlc"  # or "custom"
    lookup_csv: str = ""
    position: str = "upper"  # urban C2 within the Table 3.7 range
    wetlands_impermeable: bool = False
    c3: float = 1.0
    ft_on_c2: bool = False
    use_probability: bool = True
    texture_path: str = ""
    dolomite: list = field(default_factory=list)  # [wkt]
    overrides: list = field(default_factory=list)  # [Override]
    native_slope: bool = True
    max_cells: int = MAX_CELLS
    rasters: dict = field(default_factory=dict)  # key -> path (see RASTERS)


@dataclass
class SoilInput:
    path: str
    band: int
    method: str
    prob_path: str = ""
    prob_bands: tuple = ()
    notes: list = field(default_factory=list)


@dataclass
class CatchmentResult:
    cid: str
    area_km2: float
    map_mm: float
    cells: int = 0
    cells_lc: int = 0
    role_cells: dict = field(default_factory=dict)  # role code -> cells
    class_cells: dict = field(default_factory=dict)  # lc code -> cells
    ineffective: dict = field(default_factory=dict)  # lc code -> km2
    unknown: dict = field(default_factory=dict)  # lc code -> cells
    a_eff_km2: float = 0.0
    a_ineff_km2: float = 0.0
    shares: co.Shares | None = None
    rural_cells: int = 0
    slope_known: float = 0.0  # fractions of the rural cells with a value
    perm_known: float = 0.0
    veg_known: float = 0.0
    forced_impermeable: float = 0.0  # share of rural cells (rock, wetlands)
    slope_all: tuple = (0.0, 0.0, 0.0, 0.0)  # all cells, analysis grid
    slope_native: tuple | None = None  # all cells, native DEM
    native_res: float = 0.0
    wetland_share: float = 0.0  # SANLC wetlands, share of rural
    urban_rows: dict = field(default_factory=dict)  # urban row key -> share of urban
    texture_agree: float | None = None
    texture_cells: int = 0
    d_fraction: float = 0.0
    coeff: co.CatchmentC | None = None
    flags: list = field(default_factory=list)


@dataclass
class RunResult:
    settings: RunSettings
    lookup: lk.Lookup
    soil: SoilInput
    grid: TargetGrid
    cell_km2: float
    dem_res: float = 0.0
    dem_resampling: str = ""
    catchments: list = field(default_factory=list)  # [CatchmentResult]
    files: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    unknown: dict = field(default_factory=dict)
    preview: dict = field(default_factory=dict)
    run_time_utc: str = ""
    seconds: float = 0.0


# raster outputs: key -> (band description, data type, value meaning)
RASTERS = {
    "role": ("Land-cover role", "u8", "1 rural, 2 urban, 3 water, 4 ineffective"),
    "slope_class": ("Slope class (Cs)", "u8", "1 vleis/pans, 2 flat, 3 hilly, 4 steep"),
    "perm_class": (
        "Permeability class (Cp)",
        "u8",
        "1 very permeable ... 7 impermeable (even = SCS-SA intermediate)",
    ),
    "veg_class": ("Vegetation class (Cv)", "u8", "1 thick bush ... 4 no vegetation"),
    "c1": ("C1 rural (design)", "f4", "C1 = Cs + Cp + Cv"),
    "c2": ("C2 urban (design)", "f4", "Table 3.7 urban row"),
    "c50": ("C 1:50 year", "f4", "C1 x dolomite x Ft(50), C2, C3"),
}


# ----------------------------------------------------------------------
# GDAL helpers
# ----------------------------------------------------------------------


def _gdal():
    from osgeo import gdal

    gdal.UseExceptions()
    return gdal


def _is_projected(wkt: str) -> tuple[bool, float]:
    from osgeo import osr

    osr.UseExceptions()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return bool(srs.IsProjected()), float(srs.GetLinearUnits() or 1.0)


def analysis_grid(catchments: list, crs_wkt: str, cell: float) -> TargetGrid:
    """Grid covering every catchment, snapped to multiples of the cell and
    padded by one cell."""
    from osgeo import ogr

    ogr.UseExceptions()
    xs0, ys0, xs1, ys1 = [], [], [], []
    for c in catchments:
        g = ogr.CreateGeometryFromWkt(c.wkt)
        x0, x1, y0, y1 = g.GetEnvelope()
        xs0.append(x0)
        xs1.append(x1)
        ys0.append(y0)
        ys1.append(y1)
    x0 = math.floor(min(xs0) / cell) * cell - cell
    y0 = math.floor(min(ys0) / cell) * cell - cell
    x1 = math.ceil(max(xs1) / cell) * cell + cell
    y1 = math.ceil(max(ys1) / cell) * cell + cell
    return TargetGrid(x0, y0, x1, y1, cell, crs_wkt)


def warp_to_grid(
    path: str, grid: TargetGrid, band: int = 1, alg: str = "near", pad: int = 0
) -> np.ndarray:
    """One band warped onto the grid (optionally padded by ``pad`` cells on
    every side); NaN where there is no data."""
    gdal = _gdal()
    src = gdal.Open(path)
    if src is None:
        raise RationalCError(f"Cannot open {path}")
    try:
        nd = src.GetRasterBand(band).GetNoDataValue()
        # a VRT reads lazily (the national SANLC is several GB uncompressed)
        one = gdal.Translate("", src, format="VRT", bandList=[band])
        r = grid.res
        ds = gdal.Warp(
            "",
            one,
            format="MEM",
            outputBounds=(
                grid.xmin - pad * r,
                grid.ymin - pad * r,
                grid.xmax + pad * r,
                grid.ymax + pad * r,
            ),
            width=grid.width + 2 * pad,
            height=grid.height + 2 * pad,
            dstSRS=grid.crs_wkt,
            resampleAlg=alg,
            outputType=gdal.GDT_Float64,
            srcNodata=nd,
            dstNodata=float("nan"),
        )
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        ds = None
        one = None
        if nd is not None:
            arr[arr == nd] = np.nan
        return arr
    finally:
        src = None


def raster_res(path: str, crs_wkt: str) -> float:
    """Cell size of a raster, in the units of ``crs_wkt`` when the raster
    is in another CRS (approximate, at the raster's centre)."""
    gdal = _gdal()
    from osgeo import osr

    ds = gdal.Open(path)
    if ds is None:
        raise RationalCError(f"Cannot open {path}")
    gt = ds.GetGeoTransform()
    res = abs(gt[1])
    src = osr.SpatialReference()
    src.ImportFromWkt(ds.GetProjection())
    if src.IsGeographic():
        lat = gt[3] + gt[5] * ds.RasterYSize / 2
        res = res * 111320.0 * math.cos(math.radians(lat))
    else:
        res *= src.GetLinearUnits() or 1.0
    ds = None
    _, unit = _is_projected(crs_wkt)
    return res / unit


def horn_slope_pct(z: np.ndarray, res: float) -> np.ndarray:
    """Horn (1981) slope (%) of the inner cells of ``z`` (padded by one
    cell on every side); NaN where any neighbour is missing."""
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * res)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * res)
    return 100.0 * np.sqrt(dzdx * dzdx + dzdy * dzdy)


def rasterise(wkts: list, grid: TargetGrid, values=None) -> np.ndarray:
    """Burn polygons into the grid: value i+1 (or ``values[i]``) for the
    i-th polygon, 0 elsewhere; later polygons overwrite earlier ones."""
    from osgeo import gdal, ogr, osr

    gdal.UseExceptions()
    ogr.UseExceptions()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(grid.crs_wkt)
    ds = ogr.GetDriverByName("Memory").CreateDataSource("z")
    layer = ds.CreateLayer("z", srs, ogr.wkbUnknown)
    layer.CreateField(ogr.FieldDefn("v", ogr.OFTReal))
    for i, wkt in enumerate(wkts):
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("v", float(values[i] if values is not None else i + 1))
        feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
        layer.CreateFeature(feat)
    rds = gdal.GetDriverByName("MEM").Create(
        "", grid.width, grid.height, 1, gdal.GDT_Float64
    )
    rds.SetGeoTransform(grid.geotransform)
    rds.SetProjection(grid.crs_wkt)
    gdal.RasterizeLayer(rds, [1], layer, options=["ATTRIBUTE=v"])
    out = rds.GetRasterBand(1).ReadAsArray()
    rds = None
    ds = None
    return out


def _descriptions(path: str) -> list[str]:
    gdal = _gdal()
    ds = gdal.Open(path)
    try:
        return [
            ds.GetRasterBand(i).GetDescription() for i in range(1, ds.RasterCount + 1)
        ]
    finally:
        ds = None


def inspect_soil(path: str, use_probability: bool = True) -> SoilInput:
    """Recommended-group band and method of a Hydrological Soil Groups (HSG) output,
    and the matching group-probability raster (4 bands, A-D, %)."""
    if not os.path.exists(path):
        raise RationalCError(f"Hydrologic soil group raster not found: {path}")
    descs = _descriptions(path)
    band = next((i for i, d in enumerate(descs, 1) if "recommended" in d.lower()), 1)
    text = (descs[band - 1] + " " + os.path.basename(path)).lower()
    method = "NEH630" if "neh630" in text else "SCSSA" if "scssa" in text else ""
    s = SoilInput(path, band, method)
    if not method:
        s.notes.append(
            "The soil-group raster is not a Hydrological Soil Groups (HSG) output; codes "
            "1-4 = A-D (11-13 = A/D, B/D, C/D; 21-23 = A/B, B/C, C/D) are assumed."
        )
    if not use_probability:
        return s
    base = os.path.basename(path)
    if "_recommended" in base:
        sib = os.path.join(
            os.path.dirname(path), base.replace("_recommended", "_probability")
        )
        if os.path.exists(sib):
            sd = [d.lower() for d in _descriptions(sib)]
            bands = []
            for g in "abcd":
                bands.append(
                    next(
                        (
                            i
                            for i, d in enumerate(sd, 1)
                            if d.split("(")[0].strip().endswith(f"_{g}")
                        ),
                        0,
                    )
                )
            if all(bands):
                s.prob_path, s.prob_bands = sib, tuple(bands)
            elif len(sd) == 4:
                s.prob_path, s.prob_bands = sib, (1, 2, 3, 4)
    if not s.prob_path:
        s.notes.append(
            "No matching hsg_<method>_probability.tif beside the soil-group raster: "
            "the C1 range covers the land cover only, not the soil."
        )
    return s


def polygon_area_km2(wkt: str, unit: float) -> float:
    from osgeo import ogr

    g = ogr.CreateGeometryFromWkt(wkt)
    return g.GetArea() * unit * unit / 1e6


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------


def _noop(*_a, **_k):
    return None


def load_lookup(s: RunSettings) -> lk.Lookup:
    if s.lookup_key == "custom":
        if not s.lookup_csv:
            raise RationalCError("Choose a lookup CSV for a custom lookup.")
        try:
            return lk.read_lookup_csv(s.lookup_csv)
        except lk.CLookupError as exc:
            raise RationalCError(str(exc)) from exc
    return lk.sanlc_lookup()


def _warn(r: RunResult, log_fn, msg: str) -> None:
    r.warnings.append(msg)
    log_fn(msg)


def _validate(s: RunSettings) -> float:
    if not s.catchments:
        raise RationalCError("The catchment layer has no polygons.")
    proj, unit = _is_projected(s.crs_wkt)
    if not proj:
        raise RationalCError(
            "The catchments are in a geographic CRS. Reproject them to a projected "
            "CRS in metres (e.g. the DEM's UTM or Lo zone) first."
        )
    if not s.map_mm or s.map_mm <= 0:
        if any(c.map_mm is None for c in s.catchments):
            raise RationalCError(
                "Give the MAP (mm) or a MAP field for every catchment."
            )
    if s.position not in tb.POSITIONS:
        raise RationalCError(f"Unknown urban position '{s.position}'.")
    if not 0.0 <= s.c3 <= 1.0:
        raise RationalCError("C3 must lie between 0 and 1.")
    for p, what in (
        (s.dem_path, "DEM"),
        (s.landcover_path, "land-cover raster"),
        (s.hsg_path, "soil-group raster"),
    ):
        if not p or not os.path.exists(p):
            raise RationalCError(f"The {what} was not found: {p}")
    return unit


def run(
    s: RunSettings,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = lambda: False,
) -> RunResult:
    t0 = datetime.now(UTC)
    unit = _validate(s)
    lookup = load_lookup(s)
    soil = inspect_soil(s.hsg_path, s.use_probability)
    cell = s.cell_m / unit
    grid = analysis_grid(s.catchments, s.crs_wkt, cell)
    if grid.width * grid.height > s.max_cells:
        raise RationalCError(
            f"The analysis grid would have {grid.width * grid.height} cells "
            f"({grid.width} x {grid.height}), above the limit of {s.max_cells}. "
            "Run fewer or smaller catchments at a time, or use a larger analysis cell."
        )
    r = RunResult(s, lookup, soil, grid, (s.cell_m**2) / 1e6)
    r.run_time_utc = t0.strftime("%Y-%m-%d %H:%M:%S UTC")
    for n in soil.notes:
        _warn(r, log_fn, n)

    progress_fn(0.05, "Land cover")
    lc = warp_to_grid(s.landcover_path, grid, 1, "near")
    if s.overrides:
        codes = [o for o in s.overrides if o.lc_code is not None]
        if codes:
            burn = rasterise([o.wkt for o in codes], grid, [o.lc_code for o in codes])
            lc = np.where(burn > 0, burn, lc)
    if cancel_fn():
        raise InterruptedError("Cancelled by user")

    progress_fn(0.15, "Slope")
    r.dem_res = raster_res(s.dem_path, s.crs_wkt) * unit
    alg = "average" if r.dem_res < s.cell_m * 0.999 else "bilinear"
    r.dem_resampling = alg
    if alg == "bilinear":
        _warn(
            r,
            log_fn,
            f"The DEM cell ({r.dem_res:.1f} m) is coarser than the analysis cell "
            f"({s.cell_m:g} m): it was interpolated (bilinear), and slopes are "
            "those of the coarser DEM.",
        )
    z = warp_to_grid(s.dem_path, grid, 1, alg, pad=1)
    slope = horn_slope_pct(z, cell * unit)
    del z

    progress_fn(0.3, "Soil")
    hsg = warp_to_grid(soil.path, grid, soil.band, "near")
    prob = None
    if soil.prob_path:
        prob = np.stack(
            [warp_to_grid(soil.prob_path, grid, b) for b in soil.prob_bands]
        )
    tex = warp_to_grid(s.texture_path, grid) if s.texture_path else None

    ov = None
    rest = [o for o in s.overrides if o.cv is not None or o.urban or o.c2 is not None]
    if rest:
        ov = {
            "cv": np.full((grid.height, grid.width), cl.NONE, dtype=np.int8),
            "urban": np.full((grid.height, grid.width), cl.NONE, dtype=np.int16),
            "c2": np.full((grid.height, grid.width), np.nan),
        }
        for o in rest:
            m = rasterise([o.wkt], grid) > 0
            if o.cv is not None:
                ov["cv"][m] = o.cv
            if o.urban:
                if o.urban not in cl.URBAN_INDEX:
                    raise RationalCError(f"Override urban row '{o.urban}' is unknown.")
                ov["urban"][m] = cl.URBAN_INDEX[o.urban]
            if o.c2 is not None:
                ov["c2"][m] = o.c2
                if not o.urban:
                    ov["urban"][m] = np.where(
                        ov["urban"][m] >= 0, ov["urban"][m], cl.URBAN_INDEX["houses"]
                    )

    progress_fn(0.4, "Classes")
    band0 = tb.map_band(s.map_mm) if s.map_mm and s.map_mm > 0 else 1
    opts = cl.CellOptions(band0, s.position, s.wetlands_impermeable)
    cells = cl.classify(lc, slope, hsg, lookup, opts, prob, tex, ov)
    del lc, hsg, prob, tex
    dolo = (
        rasterise(s.dolomite, grid, [1] * len(s.dolomite)) > 0
        if s.dolomite
        else np.zeros((grid.height, grid.width), bool)
    )

    c50 = np.full((grid.height, grid.width), np.nan)
    n = len(s.catchments)
    for i, c in enumerate(s.catchments):
        if cancel_fn():
            raise InterruptedError("Cancelled by user")
        progress_fn(0.45 + 0.4 * i / n, f"Catchment {c.cid} ({i + 1} of {n})")
        cr = summarise(c, cells, dolo, grid, s, unit, r)
        if s.native_slope:
            cr.slope_native, cr.native_res = native_slope_shares(c, s, unit)
        flags(cr, r)
        r.catchments.append(cr)
        paint_c50(c50, c, cr, cells, dolo, grid, s)
        for f in cr.flags:
            log_fn(f"{c.cid}: {f}")
    for cr in r.catchments:
        for k, v in cr.unknown.items():
            r.unknown[k] = r.unknown.get(k, 0) + v
    if r.unknown:
        txt = ", ".join(
            f"{k} ({v * r.cell_km2:.3f} km2)" for k, v in sorted(r.unknown.items())
        )
        _warn(r, log_fn, f"Land-cover codes not in the lookup (left out): {txt}.")

    progress_fn(0.88, "Rasters")
    write_rasters(r, cells, c50)
    r.preview = {
        "role": cells["role"],
        "scls": cells["scls"],
        "perm": cells["perm"],
        "veg": cells["veg"],
        "c1": cells["c1"],
        "c50": c50,
    }
    r.seconds = (datetime.now(UTC) - t0).total_seconds()
    return r


def summarise(
    c: Catchment, cells: dict, dolo, grid: TargetGrid, s: RunSettings, unit, r
) -> CatchmentResult:
    from osgeo import ogr

    g = ogr.CreateGeometryFromWkt(c.wkt)
    x0, x1, y0, y1 = g.GetEnvelope()
    res = grid.res
    c0 = max(0, int(math.floor((x0 - grid.xmin) / res)))
    c1 = min(grid.width, int(math.ceil((x1 - grid.xmin) / res)))
    r0 = max(0, int(math.floor((grid.ymax - y1) / res)))
    r1 = min(grid.height, int(math.ceil((grid.ymax - y0) / res)))
    sub = TargetGrid(
        grid.xmin + c0 * res,
        grid.ymax - r1 * res,
        grid.xmin + c1 * res,
        grid.ymax - r0 * res,
        res,
        grid.crs_wkt,
    )
    m = rasterise([c.wkt], sub) > 0
    win = (slice(r0, r1), slice(c0, c1))

    def w(key):
        return cells[key][win][m]

    map_mm = c.map_mm if c.map_mm and c.map_mm > 0 else s.map_mm
    cr = CatchmentResult(c.cid, polygon_area_km2(c.wkt, unit), map_mm)
    cr.cells = int(m.sum())
    if cr.cells == 0:
        cr.flags.append("smaller than one analysis cell: no coefficients")
        return cr
    role = w("role")
    lc = w("lc")
    cr.cells_lc = int((lc >= 0).sum())
    for k, v in zip(*np.unique(role, return_counts=True), strict=True):
        cr.role_cells[int(k)] = int(v)
    for k, v in zip(*np.unique(lc[lc >= 0], return_counts=True), strict=True):
        cr.class_cells[int(k)] = int(v)
    unk = w("unknown")
    for k, v in zip(*np.unique(lc[unk], return_counts=True), strict=True):
        cr.unknown[int(k)] = int(v)

    known = int(sum(v for k, v in cr.role_cells.items() if k > 0))
    eff = int(sum(cr.role_cells.get(k, 0) for k in (1, 2, 3)))
    scale = cr.area_km2 / known if known else 0.0
    cr.a_eff_km2 = eff * scale if known else cr.area_km2
    cr.a_ineff_km2 = cr.area_km2 - cr.a_eff_km2
    ineff = role == 4
    for k, v in zip(*np.unique(lc[ineff], return_counts=True), strict=True):
        cr.ineffective[int(k)] = int(v) * scale
    alpha = cr.role_cells.get(1, 0) / eff if eff else 0.0
    beta = cr.role_cells.get(2, 0) / eff if eff else 0.0
    gamma = cr.role_cells.get(3, 0) / eff if eff else 0.0

    scls = w("scls")
    cr.slope_all = tuple(cl.class_shares(scls))
    rural = role == 1
    cr.rural_cells = int(rural.sum())
    sh = None
    if cr.rural_cells:
        rs, rp, rv = scls[rural], w("perm")[rural], w("veg")[rural]
        cr.slope_known = float((rs >= 0).mean())
        cr.perm_known = float((rp >= 0).mean())
        cr.veg_known = float((rv >= 0).mean())
        cr.forced_impermeable = float(w("forced_impermeable")[rural].mean())
        sh = co.Shares(
            tuple(cl.class_shares(rs)),
            tuple(cl.perm_shares(rp)),
            tuple(cl.class_shares(rv)),
            tuple(cl.perm_shares(w("perm_lo")[rural])),
            tuple(cl.perm_shares(w("perm_hi")[rural])),
            tuple(cl.class_shares(w("veg_lo")[rural])),
            tuple(cl.class_shares(w("veg_hi")[rural])),
        )
        cr.shares = sh
        if r.lookup.key == "sanlc":
            cr.wetland_share = float(np.isin(lc[rural], lk.SANLC_WETLANDS).mean())
        cr.d_fraction = float(dolo[win][m][rural].mean()) if s.dolomite else 0.0
        tex = cells.get("tex")
        if tex is not None:
            t = tex[win][m][rural]
            both = (t >= 0) & (rp >= 0) & (rp % 2 == 0)
            cr.texture_cells = int(both.sum())
            if cr.texture_cells:
                cr.texture_agree = float((t[both] == rp[both] // 2).mean())
    urban = role == 2
    c2 = (0.0, 0.0, 0.0)
    if urban.any():
        c2 = tuple(float(np.nanmean(w(k)[urban])) for k in ("c2", "c2_lo", "c2_hi"))
        urb = w("urb")[urban]
        keys = (*tb.URBAN_KEYS, "lawn")
        for k, v in zip(*np.unique(urb, return_counts=True), strict=True):
            if k >= 0:
                cr.urban_rows[keys[int(k)]] = int(v) / int(urban.sum())
    cr.coeff = co.catchment_c(
        sh,
        alpha,
        beta,
        gamma,
        map_mm,
        c2,
        s.c3,
        cr.d_fraction,
        c.ft_override,
        s.ft_on_c2,
    )
    return cr


def native_slope_shares(c: Catchment, s: RunSettings, unit: float):
    """Slope-class shares of the whole catchment at the DEM's own cell
    size (None if the window would be too large)."""
    from osgeo import ogr

    res = raster_res(s.dem_path, s.crs_wkt)
    g = ogr.CreateGeometryFromWkt(c.wkt)
    x0, x1, y0, y1 = g.GetEnvelope()
    x0 = math.floor(x0 / res) * res
    y0 = math.floor(y0 / res) * res
    x1 = math.ceil(x1 / res) * res
    y1 = math.ceil(y1 / res) * res
    grid = TargetGrid(x0, y0, x1, y1, res, s.crs_wkt)
    if grid.width * grid.height > MAX_NATIVE_CELLS or grid.width < 3 or grid.height < 3:
        return None, res * unit
    z = warp_to_grid(s.dem_path, grid, 1, "bilinear", pad=1)
    slope = horn_slope_pct(z, res * unit)
    m = rasterise([c.wkt], grid) > 0
    return tuple(cl.class_shares(cl.slope_class(slope[m]))), res * unit


def flags(cr: CatchmentResult, r: RunResult) -> None:
    f = cr.flags
    k = cr.coeff
    if cr.cells == 0 or k is None:
        return
    if cr.cells_lc < COVERAGE_WARN * cr.cells:
        f.append(
            f"land cover covers only {100 * cr.cells_lc / cr.cells:.0f} % of the catchment"
        )
    if cr.rural_cells:
        for what, v in (
            ("slope", cr.slope_known),
            ("soil group", cr.perm_known),
            ("vegetation class", cr.veg_known),
        ):
            if v < COVERAGE_WARN:
                f.append(f"{what} known for only {100 * v:.0f} % of the rural area")
    if k.gamma > co.WATER_FLAG:
        f.append(
            f"open water is {100 * k.gamma:.0f} % of the effective area: check for dams "
            "on the main stem and consider flood routing (DM Chapter 7)"
        )
    if cr.ineffective:
        names = ", ".join(
            f"{_name(r.lookup, code)} {km2:.3f} km2"
            for code, km2 in sorted(cr.ineffective.items())
        )
        f.append(f"excluded from the effective area: {names}")
    if r.lookup.key == "sanlc":
        plant = sum(cr.class_cells.get(c, 0) for c in lk.SANLC_PLANTATION)
        felled = cr.class_cells.get(lk.SANLC_FELLED, 0)
        if plant and felled / plant > FELLED_FLAG_SHARE:
            f.append(
                f"clear-felled plantation is {100 * felled / plant:.0f} % of the "
                "plantation area (over 30 %): reconsider the return period (DM p. 3-19)"
            )
    for code, rule in r.lookup.rules.items():
        if rule.flag == lk.FLAG_MINE:
            n = cr.class_cells.get(code, 0)
            if n and n / cr.cells >= MINE_FLAG_SHARE:
                f.append(f"{rule.name} {100 * n / cr.cells:.1f} %: {rule.flag}")
    if k.ft_borderline:
        f.append(
            f"Ft score {k.ft_score:.2f} is borderline (0.4 to 0.6): C_T given for both "
            "Table 3.8 rows; choose with the override field if needed"
        )
    if k.c1_neighbour_band is not None:
        nb, c1n = k.c1_neighbour_band
        f.append(
            f"MAP {k.map_mm:g} mm is within 50 mm of a band edge: C1D would be "
            f"{c1n:.3f} in the {tb.MAP_BANDS[nb]} mm band"
        )
    if cr.d_fraction > 0:
        f.append(
            f"dolomite under {100 * cr.d_fraction:.0f} % of the rural area: sinkholes "
            "and closed depressions may make part of it ineffective"
        )
    if cr.texture_agree is not None and cr.texture_agree < 0.5:
        f.append(
            f"soil texture agrees with the soil-group permeability on only "
            f"{100 * cr.texture_agree:.0f} % of the rural area"
        )


def _name(lookup, code):
    rule = lookup.rule(code)
    return rule.name if rule else str(code)


def paint_c50(out, c: Catchment, cr: CatchmentResult, cells, dolo, grid, s) -> None:
    """Per-cell C at 1:50 years for viewing: rural C1 (x the dolomite factor
    on dolomite) x Ft(50) of the catchment's class, urban C2, water C3."""
    k = cr.coeff
    if k is None or cr.cells == 0:
        return
    m = rasterise([c.wkt], grid) > 0
    role = cells["role"]
    ft50 = tb.ft(k.ft_class, 50)
    band = k.band
    c1 = (
        cl.table_value(cells["scls"], tb.CS, band)
        + cl.cp_value(cells["perm"], band)
        + cl.table_value(cells["veg"], tb.CV, band)
    )
    c1 = np.where(dolo, c1 * k.d_factor, c1)
    c2 = cells["c2"] * (ft50 if s.ft_on_c2 else 1.0)
    val = np.where(role == 1, c1 * ft50, np.nan)
    val = np.where(role == 2, c2, val)
    val = np.where(role == 3, s.c3, val)
    out[m] = val[m]


def write_rasters(r: RunResult, cells: dict, c50) -> None:
    gdal = _gdal()
    g = r.grid
    # class arrays may be int8; widen first, as numpy 2 refuses to mix 255
    # into an int8 result (NEP 50) where numpy 1 silently upcast
    role = cells["role"].astype(np.int16)
    scls = cells["scls"].astype(np.int16)
    perm = cells["perm"].astype(np.int16)
    veg = cells["veg"].astype(np.int16)
    data = {
        "role": np.where(role > 0, role, NODATA_U8),
        "slope_class": np.where(scls >= 0, scls + 1, NODATA_U8),
        "perm_class": np.where((perm >= 0) & (role == 1), perm + 1, NODATA_U8),
        "veg_class": np.where(veg >= 0, veg + 1, NODATA_U8),
        "c1": cells["c1"],
        "c2": cells["c2"],
        "c50": c50,
    }
    for key, path in r.settings.rasters.items():
        if not path or key not in RASTERS:
            continue
        desc, kind, meaning = RASTERS[key]
        arr = data[key]
        u8 = kind == "u8"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ds = gdal.GetDriverByName("GTiff").Create(
            path,
            g.width,
            g.height,
            1,
            gdal.GDT_Byte if u8 else gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "TILED=YES"] + ([] if u8 else ["PREDICTOR=3"]),
        )
        ds.SetGeoTransform(g.geotransform)
        ds.SetProjection(g.crs_wkt)
        ds.SetMetadataItem("SOURCE", f"{TOOL_NAME} (mayim_tools) v{TOOL_VERSION}")
        ds.SetMetadataItem("VALUES", meaning)
        b = ds.GetRasterBand(1)
        b.SetDescription(desc)
        if u8:
            b.SetNoDataValue(NODATA_U8)
            b.WriteArray(arr.astype(np.uint8))
        else:
            b.SetNoDataValue(NODATA_F)
            b.WriteArray(np.where(np.isfinite(arr), arr, NODATA_F).astype(np.float32))
        ds.FlushCache()
        ds = None
        r.files.append(path)
