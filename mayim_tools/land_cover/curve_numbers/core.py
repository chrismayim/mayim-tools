"""
Core logic for "Curve numbers" - no QGIS dependency (GDAL for raster I/O).

Per cell of the land-cover raster:

* design CN = the lookup's design value for the land-cover class and the
  recommended hydrologic soil group (HSG);
* soil range = the CN at the HSG of the Ksat P5 and P95 (from the
  Hydrologic soil groups output hsg_<method>_by_ksat.tif, when present);
* condition range = the lookup's low and high values (e.g. veld in good
  and poor condition) at the recommended group;
* the two ranges are treated as independent 90 % ranges and combined by
  root-sum-square on each side: P5 = design - sqrt(dS^2 + dC^2), and
  likewise for P95. The soil share of the uncertainty is
  dS^2 / (dS^2 + dC^2) summed over both sides.

Antecedent moisture adjustments map every CN (design, P5, P95) through the
same monotonic function. The land-cover raster is processed in row blocks
(10 m rasters can be large); the HSG layers are warped onto each block with
nearest neighbour.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.grid import TargetGrid
from mayim_tools.soil.hydrologic_soil_groups.core import read_band_to_grid
from mayim_tools.soil.regional_parameterisation.core import (
    cell_area_km2,
    rasterise_zones,
)

from . import lookup as lk
from . import tables as tb

TOOL_NAME = "Curve numbers"
TOOL_VERSION = "0.2.0"
NODATA_OUT = -9999.0
OUT_FILE = "curve_numbers.tif"
BLOCK_ROWS = 1024
PREVIEW_MAX = 1200  # cells along the longer side of the report maps

# HSG codes of the Hydrologic soil groups tool -> column in tables.GROUPS
A, B, C, D = 1, 2, 3, 4
AD, BD, CD = 11, 12, 13
AB, BC, CD_SA = 21, 22, 23
CODE_COLUMN = {A: 0, AB: 1, B: 2, BC: 3, C: 4, CD_SA: 5, D: 6}
DUAL_DRAINED = {AD: 0, BD: 2, CD: 4}
CODE_LABEL = {
    A: "A",
    B: "B",
    C: "C",
    D: "D",
    AD: "A/D",
    BD: "B/D",
    CD: "C/D",
    AB: "A/B",
    BC: "B/C",
    CD_SA: "C/D",
}

AMC_OPTIONS = {
    "none": "None (average condition, CN-II)",
    "arc1": "ARC I, dry (NEH 630 Ch. 10; Hawkins et al. 1985 fit)",
    "arc3": "ARC III, wet (NEH 630 Ch. 10; Hawkins et al. 1985 fit)",
    "sobhani": "Wet, near-saturated (Sobhani 1976; SCS-SA Eq. 4)",
    "median": "SCS-SA Median Condition (Eq. 3) with a given change in storage",
}
CONF_LIMITS = (10.0, 20.0)  # CN range P95 - P5
CONF_LABELS = {1: "High", 2: "Medium", 3: "Low"}
CONF_TEXT = {
    1: "range under 10 CN points",
    2: "range 10 to 20 CN points",
    3: "range of 20 CN points or more",
}

BANDS = [
    ("cn_design", "CN"),
    ("cn_P05 (less runoff)", "CN"),
    ("cn_P95 (more runoff)", "CN"),
    ("cn_range_P95_minus_P05", "CN"),
    ("cn_soil_share_of_uncertainty", "%"),
]


# ----------------------------------------------------------------------
# Settings and results
# ----------------------------------------------------------------------


@dataclass
class RunSettings:
    landcover_path: str
    hsg_path: str
    out_dir: str
    lookup_key: str = "worldcover_scs_sa"  # or "custom"
    lookup_csv: str = ""
    use_ksat_range: bool = True
    dual: str = "undrained"  # NEH 630 dual groups: "undrained" (D) or "drained"
    amc: str = "none"
    delta_s_mm: float = 0.0
    design_depth_mm: float = 50.0
    ia_ratio: float | None = None  # None: 0.1 for SCS-SA, 0.2 for TR-55
    zones: list = field(default_factory=list)  # [(name, wkt in the land-cover CRS)]
    write_report: bool = True
    block_rows: int = BLOCK_ROWS


@dataclass
class HsgInput:
    path: str
    band: int
    method: str  # "NEH630", "SCSSA" or "" (unknown)
    by_ksat_path: str = ""
    p05_band: int = 0
    p95_band: int = 0
    grid: TargetGrid | None = None
    notes: list = field(default_factory=list)


@dataclass
class Acc:
    """Running sums for one class or zone."""

    n: int = 0
    design: float = 0.0
    p5: float = 0.0
    p95: float = 0.0
    width: float = 0.0
    soil: float = 0.0
    soil_n: int = 0
    q_design: float = 0.0
    q_p5: float = 0.0
    q_p95: float = 0.0
    classes: dict = field(default_factory=dict)  # code -> cells
    groups: dict = field(default_factory=dict)  # column -> cells
    conf: dict = field(default_factory=dict)  # class 1-3 -> cells


@dataclass
class RunResult:
    settings: RunSettings
    lookup: lk.Lookup
    hsg: HsgInput
    grid: TargetGrid
    ia_ratio: float
    cell_km2: float = 0.0
    files: list = field(default_factory=list)
    total: Acc = field(default_factory=Acc)
    by_class: dict = field(default_factory=dict)  # code -> Acc
    by_zone: dict = field(default_factory=dict)  # index -> Acc
    zone_names: list = field(default_factory=list)
    zone_rows: list = field(default_factory=list)
    class_rows: list = field(default_factory=list)
    unknown_classes: dict = field(default_factory=dict)  # code -> cells
    cells_lc_nodata: int = 0
    cells_no_soil: int = 0
    cells_total: int = 0
    histogram: np.ndarray | None = None
    preview: dict = field(default_factory=dict)
    preview_step: int = 1
    warnings: list = field(default_factory=list)
    report_path: str = ""
    run_time_utc: str = ""
    seconds: float = 0.0


# ----------------------------------------------------------------------
# Pure functions (unit-tested)
# ----------------------------------------------------------------------


def group_columns(codes: np.ndarray, dual: str = "undrained") -> np.ndarray:
    """HSG codes -> column index in tables.GROUPS (-1 where not a group)."""
    c = np.asarray(codes, dtype=float)
    out = np.full(c.shape, -1, dtype=np.int16)
    ok = np.isfinite(c)
    ci = np.where(ok, c, 0).astype(np.int64)
    for code, col in CODE_COLUMN.items():
        out[ok & (ci == code)] = col
    for code, col in DUAL_DRAINED.items():
        out[ok & (ci == code)] = col if dual == "drained" else CODE_COLUMN[D]
    return out


def amc_adjust(cn, mode: str, delta_s_mm: float = 0.0):
    """Antecedent-moisture adjustment of CN-II (array or scalar), 0-100."""
    cn = np.asarray(cn, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "none":
            out = cn.copy()
        elif mode == "arc1":
            out = cn / (2.281 - 0.01281 * cn)
        elif mode == "arc3":
            out = cn / (0.427 + 0.00573 * cn)
        elif mode == "sobhani":
            out = cn / (0.4036 + 0.0059 * cn)
        elif mode == "median":
            denom = 1100.0 / cn - delta_s_mm / 25.4
            out = np.where(denom > 11.0, 1100.0 / denom, 100.0)
            out = np.where(cn > 0, out, 0.0)
        else:
            raise SoilDataError(f"Unknown antecedent-moisture option '{mode}'.")
    return np.where(np.isfinite(cn), np.clip(out, 0.0, 100.0), np.nan)


def runoff_mm(p_mm: float, cn, ia_ratio: float):
    """SCS stormflow depth Q (mm) for rainfall P and curve number CN."""
    cn = np.asarray(cn, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(cn > 0, 25400.0 / cn - 254.0, np.inf)
        s = np.maximum(s, 0.0)
        ia = ia_ratio * s
        q = np.where(p_mm > ia, (p_mm - ia) ** 2 / (p_mm - ia + s), 0.0)
    q = np.where(cn >= 100, p_mm, q)
    return np.where(np.isfinite(cn), q, np.nan)


def equivalent_cn(p_mm: float, q_mm: float, ia_ratio: float) -> float:
    """The CN that gives stormflow q_mm from rainfall p_mm (bisection)."""
    if not (math.isfinite(q_mm) and p_mm > 0):
        return math.nan
    if q_mm <= 0:
        return 0.0
    if q_mm >= p_mm:
        return 100.0
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(runoff_mm(p_mm, mid, ia_ratio)) < q_mm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def conf_class(width: np.ndarray) -> np.ndarray:
    w = np.asarray(width, dtype=float)
    out = np.zeros(w.shape, dtype=np.int16)
    ok = np.isfinite(w)
    out[ok & (w < CONF_LIMITS[0])] = 1
    out[ok & (w >= CONF_LIMITS[0]) & (w < CONF_LIMITS[1])] = 2
    out[ok & (w >= CONF_LIMITS[1])] = 3
    return out


def compute_block(lc, rec, p05, p95, lookup: lk.Lookup, dual: str):
    """CN-II design, P5, P95 and soil share (%) for a block. ``p05`` and
    ``p95`` may be None (no soil range). Returns a dict of arrays plus the
    recommended-group column used (-1 where none)."""
    lc = np.asarray(lc)
    shape = lc.shape
    col_r = group_columns(rec, dual)
    col_a = group_columns(p05, dual) if p05 is not None else col_r
    col_b = group_columns(p95, dual) if p95 is not None else col_r
    col_a = np.where(col_a >= 0, col_a, col_r)
    col_b = np.where(col_b >= 0, col_b, col_r)
    design = np.full(shape, np.nan)
    p5 = np.full(shape, np.nan)
    p95_ = np.full(shape, np.nan)
    share = np.full(shape, np.nan)
    used = np.full(shape, -1, dtype=np.int16)
    for code, e in lookup.classes.items():
        sel = lc == code
        if not sel.any():
            continue
        if e.override:
            oc = tb.GROUPS.index(e.override)
            cr = np.where(sel, oc, -1)
            ca = cb = cr
        else:
            cr, ca, cb = col_r, col_a, col_b
        ok = sel & (cr >= 0)
        if not ok.any():
            continue
        r, a, b = cr[ok], ca[ok], cb[ok]
        vd, vl, vh = e.values["design"], e.values["low"], e.values["high"]
        d = vd[r]
        sa, sb = vd[a], vd[b]
        cl, ch = vl[r], vh[r]
        s_lo = np.fmin(d, np.fmin(sa, sb))
        s_hi = np.fmax(d, np.fmax(sa, sb))
        c_lo = np.fmin(d, np.fmin(cl, ch))
        c_hi = np.fmax(d, np.fmax(cl, ch))
        dls, dlc = d - s_lo, d - c_lo
        dhs, dhc = s_hi - d, c_hi - d
        design[ok] = d
        p5[ok] = np.clip(d - np.sqrt(dls**2 + dlc**2), 0, 100)
        p95_[ok] = np.clip(d + np.sqrt(dhs**2 + dhc**2), 0, 100)
        soil = dls**2 + dhs**2
        tot = soil + dlc**2 + dhc**2
        with np.errstate(invalid="ignore", divide="ignore"):
            share[ok] = np.where(tot > 0, 100.0 * soil / tot, np.nan)
        used[ok] = r
    return {"design": design, "p5": p5, "p95": p95_, "share": share, "col": used}


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------


def _gdal():
    from osgeo import gdal

    gdal.UseExceptions()
    return gdal


def raster_grid(path: str) -> TargetGrid:
    gdal = _gdal()
    ds = gdal.Open(path)
    if ds is None:
        raise SoilDataError(f"Cannot open {path}")
    try:
        gt = ds.GetGeoTransform()
        if gt[2] != 0 or gt[4] != 0:
            raise SoilDataError(f"Rotated rasters are not supported: {path}")
        if abs(abs(gt[5]) - gt[1]) > 1e-6 * gt[1]:
            raise SoilDataError(
                f"{os.path.basename(path)} has non-square cells; resample it to "
                "square cells first."
            )
        return TargetGrid(
            gt[0],
            gt[3] + gt[5] * ds.RasterYSize,
            gt[0] + gt[1] * ds.RasterXSize,
            gt[3],
            gt[1],
            ds.GetProjection(),
        )
    finally:
        ds = None


def _descriptions(path: str) -> list[str]:
    gdal = _gdal()
    ds = gdal.Open(path)
    try:
        return [
            ds.GetRasterBand(i).GetDescription() for i in range(1, ds.RasterCount + 1)
        ]
    finally:
        ds = None


def inspect_hsg(path: str, use_ksat_range: bool = True) -> HsgInput:
    """Recommended-group band, method and the matching _by_ksat file."""
    if not os.path.exists(path):
        raise SoilDataError(f"Hydrologic soil group raster not found: {path}")
    descs = _descriptions(path)
    band = next((i for i, d in enumerate(descs, 1) if "recommended" in d.lower()), 1)
    text = (descs[band - 1] + " " + os.path.basename(path)).lower()
    method = "NEH630" if "neh630" in text else "SCSSA" if "scssa" in text else ""
    h = HsgInput(path, band, method, grid=raster_grid(path))
    if not method:
        h.notes.append(
            "The soil-group raster is not a Hydrologic soil groups output; codes "
            "1-4 = A-D (11-13 = A/D, B/D, C/D; 21-23 = A/B, B/C, C/D) are expected."
        )
    base = os.path.basename(path)
    if use_ksat_range and "_recommended" in base:
        sib = os.path.join(
            os.path.dirname(path), base.replace("_recommended", "_by_ksat")
        )
        if os.path.exists(sib):
            sd = _descriptions(sib)
            p05 = next((i for i, d in enumerate(sd, 1) if "p05" in d.lower()), 0)
            p95 = next((i for i, d in enumerate(sd, 1) if "p95" in d.lower()), 0)
            if p05 and p95:
                h.by_ksat_path, h.p05_band, h.p95_band = sib, p05, p95
    if use_ksat_range and not h.by_ksat_path:
        h.notes.append(
            "No matching hsg_<method>_by_ksat.tif beside the soil-group raster: "
            "the CN range covers the land-cover condition only, not the soil."
        )
    return h


def load_lookup(s: RunSettings) -> lk.Lookup:
    if s.lookup_key == "custom":
        if not s.lookup_csv:
            raise SoilDataError("Choose a lookup CSV for a custom lookup.")
        return lk.read_lookup_csv(s.lookup_csv)
    return lk.default_lookup(s.lookup_key)


def default_ia_ratio(lookup: lk.Lookup) -> float:
    if lookup.table in tb.TABLES:
        return tb.TABLES[lookup.table][2]
    return 0.1


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------


def _noop(*_a, **_k):
    return None


def _add(acc: Acc, sel, out, cfg, col, lc, conf):
    n = int(sel.sum())
    if not n:
        return
    acc.n += n
    d, a, b = out["design"][sel], out["p5"][sel], out["p95"][sel]
    acc.design += float(d.sum())
    acc.p5 += float(a.sum())
    acc.p95 += float(b.sum())
    acc.width += float((b - a).sum())
    sh = out["share"][sel]
    fin = np.isfinite(sh)
    acc.soil += float(sh[fin].sum())
    acc.soil_n += int(fin.sum())
    p, lam = cfg
    acc.q_design += float(runoff_mm(p, d, lam).sum())
    acc.q_p5 += float(runoff_mm(p, a, lam).sum())
    acc.q_p95 += float(runoff_mm(p, b, lam).sum())
    for k, v in zip(*np.unique(lc[sel], return_counts=True), strict=True):
        acc.classes[int(k)] = acc.classes.get(int(k), 0) + int(v)
    for k, v in zip(*np.unique(col[sel], return_counts=True), strict=True):
        acc.groups[int(k)] = acc.groups.get(int(k), 0) + int(v)
    for k, v in zip(*np.unique(conf[sel], return_counts=True), strict=True):
        acc.conf[int(k)] = acc.conf.get(int(k), 0) + int(v)


def run(
    s: RunSettings,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = lambda: False,
) -> RunResult:
    t0 = datetime.now(UTC)
    gdal = _gdal()
    lookup = load_lookup(s)
    hsg = inspect_hsg(s.hsg_path, s.use_ksat_range)
    grid = raster_grid(s.landcover_path)
    lam = s.ia_ratio if s.ia_ratio is not None else default_ia_ratio(lookup)
    r = RunResult(s, lookup, hsg, grid, lam, cell_km2=cell_area_km2(grid))
    r.run_time_utc = t0.strftime("%Y-%m-%d %H:%M:%S UTC")
    r.zone_names = [z[0] for z in s.zones]
    r.by_zone = {i: Acc() for i in range(1, len(s.zones) + 1)}
    r.histogram = np.zeros(101, dtype=np.int64)
    for note in hsg.notes + lookup.notes:
        _warn(r, log_fn, note)
    _method_checks(r, log_fn)
    os.makedirs(s.out_dir, exist_ok=True)

    src = gdal.Open(s.landcover_path)
    lc_band = src.GetRasterBand(1)
    lc_nd = lc_band.GetNoDataValue()
    w, h = grid.width, grid.height
    out_path = os.path.join(s.out_dir, OUT_FILE)
    dst = gdal.GetDriverByName("GTiff").Create(
        out_path,
        w,
        h,
        len(BANDS),
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    dst.SetGeoTransform(grid.geotransform)
    dst.SetProjection(grid.crs_wkt)
    dst.SetMetadataItem("SOURCE", f"{TOOL_NAME} (mayim_tools) v{TOOL_VERSION}")
    dst.SetMetadataItem("LOOKUP", lookup.name)
    dst.SetMetadataItem("ANTECEDENT_MOISTURE", AMC_OPTIONS[s.amc])
    suffix = "" if s.amc == "none" else f"_{s.amc}"
    for i, (name, units) in enumerate(BANDS, 1):
        band = dst.GetRasterBand(i)
        band.SetNoDataValue(NODATA_OUT)
        desc = name if i > 3 or not suffix else name.replace("cn_", f"cn{suffix}_", 1)
        band.SetDescription(f"{desc} ({units})")
        band.SetUnitType(units)

    step = max(1, int(math.ceil(max(w, h) / PREVIEW_MAX)))
    r.preview_step = step
    prev = {k: [] for k in ("design", "width", "lc", "share")}
    cfg = (s.design_depth_mm, lam)
    try:
        nblocks = int(math.ceil(h / s.block_rows))
        for bi, r0 in enumerate(range(0, h, s.block_rows)):
            if cancel_fn():
                raise InterruptedError("Cancelled by user")
            bh = min(s.block_rows, h - r0)
            bgrid = TargetGrid(
                grid.xmin,
                grid.ymax - (r0 + bh) * grid.res,
                grid.xmax,
                grid.ymax - r0 * grid.res,
                grid.res,
                grid.crs_wkt,
            )
            lc = lc_band.ReadAsArray(0, r0, w, bh)
            lc_valid = np.ones(lc.shape, dtype=bool) if lc_nd is None else lc != lc_nd
            rec = read_band_to_grid(hsg.path, bgrid, hsg.band)
            p05 = p95 = None
            if hsg.by_ksat_path:
                p05 = read_band_to_grid(hsg.by_ksat_path, bgrid, hsg.p05_band)
                p95 = read_band_to_grid(hsg.by_ksat_path, bgrid, hsg.p95_band)
            out = compute_block(lc, rec, p05, p95, lookup, s.dual)
            for key in ("design", "p5", "p95"):
                out[key] = amc_adjust(out[key], s.amc, s.delta_s_mm)
            width = out["p95"] - out["p5"]
            conf = conf_class(width)
            valid = np.isfinite(out["design"])
            # bookkeeping of cells without a CN
            r.cells_total += int(lc.size)
            r.cells_lc_nodata += int((~lc_valid).sum())
            known = np.isin(lc, list(lookup.classes))
            unk = lc_valid & ~known
            if unk.any():
                for k, v in zip(*np.unique(lc[unk], return_counts=True), strict=True):
                    r.unknown_classes[int(k)] = r.unknown_classes.get(int(k), 0) + int(
                        v
                    )
            r.cells_no_soil += int((lc_valid & known & ~valid).sum())
            # statistics
            _add(r.total, valid, out, cfg, out["col"], lc, conf)
            for code in np.unique(lc[valid]):
                acc = r.by_class.setdefault(int(code), Acc())
                _add(acc, valid & (lc == code), out, cfg, out["col"], lc, conf)
            if s.zones:
                zr = rasterise_zones(s.zones, bgrid)
                for zi in np.unique(zr[valid & (zr > 0)]):
                    _add(
                        r.by_zone[int(zi)],
                        valid & (zr == zi),
                        out,
                        cfg,
                        out["col"],
                        lc,
                        conf,
                    )
            d = out["design"][valid]
            r.histogram += np.bincount(
                np.clip(np.rint(d), 0, 100).astype(np.int64), minlength=101
            )[:101]
            # write
            for i, arr in enumerate(
                (out["design"], out["p5"], out["p95"], width, out["share"]), 1
            ):
                dst.GetRasterBand(i).WriteArray(
                    np.where(np.isfinite(arr), arr, NODATA_OUT).astype(np.float32),
                    0,
                    r0,
                )
            rows = [i for i in range(bh) if (r0 + i) % step == 0]
            if rows:
                prev["design"].append(out["design"][rows, ::step])
                prev["width"].append(width[rows, ::step])
                prev["share"].append(out["share"][rows, ::step])
                prev["lc"].append(np.where(lc_valid, lc, 0)[rows, ::step])
            progress_fn(
                0.9 * (bi + 1) / nblocks, f"Curve numbers: block {bi + 1} of {nblocks}"
            )
    finally:
        dst.FlushCache()
        dst = None
        src = None
    r.files.append(out_path)
    r.preview = {k: np.vstack(v) if v else np.zeros((0, 0)) for k, v in prev.items()}
    _finish(r, log_fn)
    r.seconds = (datetime.now(UTC) - t0).total_seconds()
    progress_fn(0.92, "Tables")
    return r


def _warn(r: RunResult, log_fn, msg: str) -> None:
    r.warnings.append(msg)
    log_fn(msg)


def _method_checks(r: RunResult, log_fn) -> None:
    m, t = r.hsg.method, r.lookup.table
    if m == "NEH630" and t == "scs_sa":
        _warn(
            r,
            log_fn,
            "NEH 630 soil groups with the SCS-SA table: NEH 630 (2009) uses much "
            "higher Ksat limits, so its groups give more runoff than the SCS-SA "
            "groups the table was made for. Prefer the SCS-SA groups with this table.",
        )
    if m == "SCSSA" and t == "tr55":
        _warn(
            r,
            log_fn,
            "SCS-SA soil groups with the TR-55 table: the intermediate groups A/B, "
            "B/C and C/D are interpolated between the TR-55 columns.",
        )


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------


def _mean(total, n):
    return total / n if n else math.nan


def _dominant(counts: dict, label) -> tuple[str, float]:
    if not counts:
        return "-", math.nan
    tot = sum(counts.values())
    k, v = max(counts.items(), key=lambda kv: kv[1])
    return label(k), 100.0 * v / tot


def _class_label(r):
    def f(code):
        e = r.lookup.classes.get(code)
        return e.name if e else str(code)

    return f


def _group_label(col):
    return tb.GROUPS[col] if 0 <= col < len(tb.GROUPS) else "-"


def zone_row(r: RunResult, name: str, acc: Acc) -> dict:
    p, lam = r.settings.design_depth_mm, r.ia_ratio
    q_d, q_a, q_b = (_mean(x, acc.n) for x in (acc.q_design, acc.q_p5, acc.q_p95))
    width = _mean(acc.width, acc.n)
    soil = _mean(acc.soil, acc.soil_n)
    lc_name, lc_share = _dominant(acc.classes, _class_label(r))
    grp, grp_share = _dominant(acc.groups, _group_label)
    eq = [equivalent_cn(p, q, lam) for q in (q_d, q_a, q_b)]
    eq_width = eq[2] - eq[1]
    conf = int(conf_class(np.array([eq_width]))[0]) if math.isfinite(eq_width) else 0
    return {
        "Zone": name,
        "Area with CN (km2)": acc.n * r.cell_km2,
        "Mean CN design": _mean(acc.design, acc.n),
        "Mean CN P05": _mean(acc.p5, acc.n),
        "Mean CN P95": _mean(acc.p95, acc.n),
        f"Runoff-equivalent CN design (P = {p:g} mm)": eq[0],
        "Runoff-equivalent CN P05": eq[1],
        "Runoff-equivalent CN P95": eq[2],
        "Stormflow design (mm)": q_d,
        "Stormflow P05 (mm)": q_a,
        "Stormflow P95 (mm)": q_b,
        "Mean cell range P95 - P05 (CN)": width,
        "Confidence": CONF_LABELS.get(conf, "-"),
        "Soil share of uncertainty (%)": soil,
        "Main uncertainty source": (
            "-"
            if not math.isfinite(soil)
            else "soil group" if soil >= 50 else "land-cover condition"
        ),
        "Dominant land cover": lc_name,
        "Dominant land cover share (%)": lc_share,
        "Dominant soil group": grp,
        "Dominant soil group share (%)": grp_share,
    }


def _finish(r: RunResult, log_fn) -> None:
    rows = [zone_row(r, "Whole raster", r.total)]
    for i, name in enumerate(r.zone_names, 1):
        acc = r.by_zone[i]
        if acc.n == 0:
            _warn(r, log_fn, f"Zone '{name}' has no cells with a curve number.")
        rows.append(zone_row(r, name, acc))
    r.zone_rows = rows
    tot = r.total.n
    for code in sorted(r.by_class):
        acc = r.by_class[code]
        e = r.lookup.classes.get(code)
        r.class_rows.append(
            {
                "Code": code,
                "Class": e.name if e else str(code),
                "Area (km2)": acc.n * r.cell_km2,
                "Share (%)": 100.0 * acc.n / tot if tot else math.nan,
                "Mean CN design": _mean(acc.design, acc.n),
                "Mean CN P05": _mean(acc.p5, acc.n),
                "Mean CN P95": _mean(acc.p95, acc.n),
                "Soil share of uncertainty (%)": _mean(acc.soil, acc.soil_n),
                "Group override": e.override if e else "",
            }
        )
    if r.unknown_classes:
        txt = ", ".join(
            f"{c} ({n * r.cell_km2:.3f} km2)"
            for c, n in sorted(r.unknown_classes.items())
        )
        _warn(r, log_fn, f"Land-cover codes not in the lookup (no CN): {txt}.")
    if r.cells_no_soil:
        _warn(
            r,
            log_fn,
            f"{r.cells_no_soil * r.cell_km2:.3f} km2 of land cover has no soil group "
            "(outside the soil-group raster or nodata): no CN there.",
        )
    if tot == 0:
        raise SoilDataError(
            "No cell has both a land-cover class in the lookup and a soil group. "
            "Check that the two rasters overlap and that the lookup matches the "
            "land-cover codes."
        )
    built = sum(
        a.n
        for c, a in r.by_class.items()
        if c == 50 and r.settings.lookup_key.startswith("worldcover")
    )
    if built and built / tot >= 0.02:
        _warn(
            r,
            log_fn,
            f"Built-up land covers {100 * built / tot:.0f} % of the area. Its CN "
            "depends on the impervious share, which WorldCover does not give: "
            "check the built-up rows of the lookup against the actual development.",
        )


# ----------------------------------------------------------------------
# Outputs: CSVs and metadata
# ----------------------------------------------------------------------


def _fmt(v, nd=2):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not math.isfinite(v) else round(v, nd)
    return v


def write_tables(r: RunResult) -> dict:
    import csv

    paths = {}
    d = r.settings.out_dir
    path = os.path.join(d, "cn_zone_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(r.zone_rows[0]))
        for row in r.zone_rows:
            w.writerow([_fmt(v) for v in row.values()])
    paths["zones"] = path
    path = os.path.join(d, "cn_lookup_used.csv")
    lk.write_lookup_csv(r.lookup, path)
    paths["lookup"] = path
    tables = [r.lookup.table] if r.lookup.table in tb.TABLES else []
    if r.settings.lookup_key.startswith("sanlc"):
        tables.append("wrc2747")
    for t in tables:
        path = os.path.join(d, f"cn_table_{t}.csv")
        lk.write_reference_table(t, path)
        paths[f"table_{t}"] = path
    r.files.extend(paths.values())
    return paths


def build_metadata(r: RunResult, gdal_version_text: str = "") -> list:
    s = r.settings
    run_rows = [
        ["Tool", f"{TOOL_NAME} (mayim_tools) v{TOOL_VERSION}"],
        ["Run time", r.run_time_utc],
        ["Duration (s)", round(r.seconds, 1)],
        ["GDAL version", gdal_version_text],
        ["Land cover", s.landcover_path],
        ["Soil groups", s.hsg_path],
        [
            "Soil group method",
            {"NEH630": "NEH 630", "SCSSA": "SCS-SA"}.get(r.hsg.method, "not known"),
        ],
        ["Soil range from", r.hsg.by_ksat_path or "none (condition range only)"],
        [
            "NEH 630 dual groups",
            "undrained (D)" if s.dual == "undrained" else "drained (first letter)",
        ],
        ["Lookup", r.lookup.name],
        ["Lookup source", r.lookup.source],
        [
            "Antecedent moisture",
            AMC_OPTIONS[s.amc]
            + (f"; change in storage {s.delta_s_mm:g} mm" if s.amc == "median" else ""),
        ],
        ["Initial abstraction ratio (Ia / S)", r.ia_ratio],
        ["Design rainfall depth for zone runoff (mm)", s.design_depth_mm],
        [
            "Output grid",
            f"{r.grid.width} x {r.grid.height} cells of {r.grid.res:g} (land-cover grid)",
        ],
        ["Cell area (km2)", round(r.cell_km2, 8)],
        ["Output nodata", NODATA_OUT],
        [
            "Note - uncertainty",
            "P05 and P95 combine the soil range (CN at the soil group of the Ksat "
            "P5 and P95) and the land-cover condition range (lookup low and high) "
            "as independent 90 % ranges, by root-sum-square on each side.",
        ],
        [
            "Note - zone means",
            "Zone means of the cell P05 and P95 treat the cells' errors as fully "
            "related (cautious, wide). The runoff-equivalent CN gives the zone's "
            "mean stormflow at the design depth; a plain mean CN understates runoff "
            "from mixed catchments.",
        ],
        [
            "Note - confidence",
            "Zone confidence from the range of the runoff-equivalent CN (P95 - P05): "
            + "; ".join(f"{CONF_LABELS[k]}: {CONF_TEXT[k]}" for k in CONF_LABELS),
        ],
        ["Note - bands", "Select bands by description, never by band number."],
    ]
    look_rows = []
    for code in sorted(r.lookup.classes):
        e = r.lookup.classes[code]
        for case in lk.CASES:
            spec = e.rows.get(case, "")
            try:
                text = tb.describe(spec) if spec else ""
            except KeyError:
                text = spec
            look_rows.append(
                [
                    code,
                    e.name,
                    case,
                    text,
                    e.override,
                    e.note if case == "design" else "",
                ]
            )
    class_rows = [[_fmt(v, 3) for v in row.values()] for row in r.class_rows]
    zone_rows = [[_fmt(v) for v in row.values()] for row in r.zone_rows]
    sections = [
        ("Run", ["Item", "Value"], run_rows),
        (
            "Lookup",
            [
                "Code",
                "Class",
                "Case",
                "Curve-number table row",
                "Group override",
                "Note",
            ],
            look_rows,
        ),
        ("Classes", list(r.class_rows[0]) if r.class_rows else ["Code"], class_rows),
        ("Zones", list(r.zone_rows[0]), zone_rows),
        ("Warnings", ["Warning"], [[w] for w in r.warnings] or [["none"]]),
    ]
    return sections


def write_outputs(
    r: RunResult, gdal_version_text: str = "", log_fn: Callable = _noop
) -> None:
    """Tables, metadata CSV and (optionally) the Word report."""
    from mayim_tools.soil._common.export import write_metadata_csv

    write_tables(r)
    meta = os.path.join(r.settings.out_dir, "cn_metadata.csv")
    write_metadata_csv(build_metadata(r, gdal_version_text), meta)
    r.files.append(meta)
    if r.settings.write_report:
        try:
            from .report import write_report

            r.report_path = write_report(r)
            r.files.append(r.report_path)
        except ImportError as exc:
            _warn(
                r,
                log_fn,
                f"Word report not written (python-docx or matplotlib missing: {exc}).",
            )


CN_RAMP = [
    (30.0, "#1a9850", "30"),
    (55.0, "#91cf60", "55"),
    (70.0, "#fee08b", "70"),
    (80.0, "#fc8d59", "80"),
    (90.0, "#d73027", "90"),
    (100.0, "#7f0000", "100"),
]
