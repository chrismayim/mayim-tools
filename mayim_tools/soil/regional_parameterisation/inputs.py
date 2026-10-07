"""Reading the extraction outputs and harmonising them to the target depth
layers (stage 1). No QGIS; GDAL only in ``read_file`` / ``grid_of``.

Bands are selected by their DESCRIPTION, never by band number, so a run
with only some depths or statistics extracted cannot feed the wrong layer
into a calculation. Recognised descriptions:

    SoilGrids 2.0       clay_15-30cm_Q0.5 (%)          (intervals; mean, Q0.05,
                                                         Q0.5, Q0.95)
    SoilGrids 2017      clay_15cm_mean [SG2017] (%)    (depth points)
    OpenLandMap         clay_0-30cm_mean_30m_2020-2022 (%)
                        clay_0-30cm_p16_120m_2020-2022 (%)
    OpenLandMap water   field_capacity_33kPa_30cm_250m_1950-2017 (vol %)
    iSDAsoil            clay_0-20cm_mean_30m_isda (%)
                        clay_0-20cm_sd_30m_isda (%)   soc ... (ln(1+g/kg))

Each input variable becomes a per-cell distribution (``Dist``): a centre and
lower / upper standard deviations in a transform space (log for texture,
organic carbon; linear for bulk density and coarse fragments):

    SoilGrids 2.0   centre Q0.5; sigma_lo = (T(Q0.5) - T(Q0.05)) / 1.645,
                    sigma_hi = (T(Q0.95) - T(Q0.5)) / 1.645   (90 % interval)
    OpenLandMap     centre = 30 m mean; sigma = (T(P84) - T(P16)) / (2 x 0.994)
                    on both sides (68 % interval, 120 m)

    iSDAsoil        centre = 30 m mean; the published standard deviation
                    gives a 90 % interval mean -/+ 1.645 sd (for organic
                    carbon and stone content in ln(1 + x)), converted like
                    the SoilGrids quantiles

so the products' uncertainty is expressed on the same footing. iSDAsoil is
mapped for 0-20 and 20-50 cm only: its 20-50 cm value stands in for
50-60 cm (30-60 cm layer) and it takes no part in the 60-100 cm layer.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.grid import TargetGrid

PRODUCT_SG = "SoilGrids 2.0"
PRODUCT_OLM = "OpenLandMap-soildb"
PRODUCT_SG2017 = "SoilGrids 2017"
PRODUCT_ISDA = "iSDAsoil"
ISDA_EXTEND_CM = 10  # deepest iSDA interval may stand in this far below 50 cm

TARGET_LAYERS: tuple[tuple[str, int, int], ...] = (
    ("0-30cm", 0, 30),
    ("30-60cm", 30, 60),
    ("60-100cm", 60, 100),
)

# Inputs the methods use (others in the folders are ignored).
INPUT_VARIABLES = ("sand", "silt", "clay", "soc", "bdod", "cfvo", "phh2o", "cec")
OPTIONAL_VARIABLES = ("bdod", "cfvo", "phh2o", "cec")  # not every product has them
TEXTURE = ("sand", "silt", "clay")

# Transform space and plausible range per variable.
#   log: T(x) = ln(max(x, floor)); linear: T(x) = x
VARIABLE_SPACE: dict[str, tuple[str, float, float, float]] = {
    # code: (space, floor for log, minimum, maximum)
    "sand": ("log", 0.5, 0.0, 100.0),
    "silt": ("log", 0.5, 0.0, 100.0),
    "clay": ("log", 0.5, 0.0, 100.0),
    "soc": ("log", 0.1, 0.0, 600.0),
    "bdod": ("linear", 0.0, 0.3, 2.3),
    "cfvo": ("linear", 0.0, 0.0, 90.0),
    "phh2o": ("linear", 0.0, 3.0, 11.0),
    "cec": ("log", 0.5, 0.0, 300.0),
}

Z90 = 1.6448536  # standard-normal quantile for a central 90 % interval
Z68 = 0.9944579  # standard-normal quantile for a central 68 % interval

_SG_RE = re.compile(
    r"^(?P<var>[a-z0-9]+)_(?P<top>\d+)-(?P<bot>\d+)cm_"
    r"(?P<stat>mean|Q0\.05|Q0\.5|Q0\.95) \((?P<units>.*)\)$"
)
_SG17_RE = re.compile(
    r"^(?P<var>[a-z0-9]+)_(?P<pt>\d+)cm_mean \[SG2017\] \((?P<units>.*)\)$"
)
_OLM_RE = re.compile(
    r"^(?P<var>[a-z0-9]+)_(?P<top>\d+)-(?P<bot>\d+)cm_"
    r"(?P<stat>mean_30m|mean_120m|p16_120m|p84_120m)_(?P<period>\d{4}-\d{4}) "
    r"\((?P<units>.*)\)$"
)
_WATER_RE = re.compile(
    r"^(?P<name>field_capacity_33kPa|wilting_point_1500kPa)_(?P<pt>\d+)cm_250m_"
    r"(?P<period>[\d-]+) \((?P<units>.*)\)$"
)
_ISDA_RE = re.compile(
    r"^(?P<var>[a-z0-9]+)_(?P<top>\d+)-(?P<bot>\d+)cm_(?P<stat>mean|sd)_30m_isda "
    r"\((?P<units>.*)\)$"
)
WATER_CODES = {"field_capacity_33kPa": "fc", "wilting_point_1500kPa": "wp"}
SG_WATER_CODES = {"wv0033": "fc", "wv1500": "wp"}


# ----------------------------------------------------------------------
# Band catalogue
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BandRef:
    path: str
    index: int  # 1-based GDAL band index (found from the description)
    kind: str  # "sg", "sg2017", "olm", "water"
    var: str
    top: int | None  # interval top (cm), or the depth point
    bottom: int | None  # interval bottom (cm); None for a depth point
    stat: str  # mean / Q0.05 / Q0.5 / Q0.95 / mean_30m / p16_120m / ...
    period: str
    units: str


def parse_description(desc: str) -> dict | None:
    """Recognise a band description from the extraction tools."""
    desc = desc.strip()
    m = _SG_RE.match(desc)
    if m:
        return dict(
            kind="sg",
            var=m["var"],
            top=int(m["top"]),
            bottom=int(m["bot"]),
            stat=m["stat"],
            period="",
            units=m["units"],
        )
    m = _SG17_RE.match(desc)
    if m:
        return dict(
            kind="sg2017",
            var=m["var"],
            top=int(m["pt"]),
            bottom=None,
            stat="mean",
            period="",
            units=m["units"],
        )
    m = _OLM_RE.match(desc)
    if m:
        return dict(
            kind="olm",
            var=m["var"],
            top=int(m["top"]),
            bottom=int(m["bot"]),
            stat=m["stat"],
            period=m["period"],
            units=m["units"],
        )
    m = _ISDA_RE.match(desc)
    if m:
        return dict(
            kind="isda",
            var=m["var"],
            top=int(m["top"]),
            bottom=int(m["bot"]),
            stat=m["stat"],
            period="2001-2017",
            units=m["units"],
        )
    m = _WATER_RE.match(desc)
    if m:
        return dict(
            kind="water",
            var=WATER_CODES[m["name"]],
            top=int(m["pt"]),
            bottom=None,
            stat="mean",
            period=m["period"],
            units=m["units"],
        )
    return None


def band_descriptions(path: str) -> list[str]:
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    try:
        return [
            ds.GetRasterBand(i).GetDescription() for i in range(1, ds.RasterCount + 1)
        ]
    finally:
        ds = None


def scan_folder(folder: str, describe_fn=band_descriptions) -> list[BandRef]:
    """Every recognised band in the GeoTIFFs of an extraction folder."""
    if not folder or not os.path.isdir(folder):
        raise SoilDataError(f"Folder not found: {folder}")
    refs: list[BandRef] = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        path = os.path.join(folder, name)
        try:
            descs = describe_fn(path)
        except Exception:  # noqa: BLE001 - unreadable file: ignore it
            continue
        for i, desc in enumerate(descs, start=1):
            info = parse_description(desc or "")
            if info:
                refs.append(BandRef(path=path, index=i, **info))
    return refs


def detect_product(refs: list[BandRef]) -> str | None:
    kinds = {r.kind for r in refs}
    if "sg" in kinds:
        return PRODUCT_SG
    if "olm" in kinds:
        return PRODUCT_OLM
    if "isda" in kinds:
        return PRODUCT_ISDA
    return None


def read_run_metadata(folder: str) -> dict[str, str]:
    """'# Run' section of an extraction metadata CSV (Item -> Value)."""
    for name in (
        "soilgrids_metadata.csv",
        "openlandmap_metadata.csv",
        "isda_metadata.csv",
    ):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            break
    else:
        return {}
    out: dict[str, str] = {}
    section = None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].startswith("# "):
                section = row[0][2:].strip()
                continue
            if section == "Run" and len(row) >= 2 and row[0] != "Item":
                out[row[0]] = row[1]
    out["Metadata file"] = path
    return out


# ----------------------------------------------------------------------
# Grid and raster reading (GDAL)
# ----------------------------------------------------------------------


def grid_of(path: str) -> TargetGrid:
    """The (north-up) grid of a raster file."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    try:
        gt = ds.GetGeoTransform()
        if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
            raise SoilDataError(f"Rotated rasters are not supported: {path}")
        res = gt[1]
        xmin = gt[0]
        ymax = gt[3]
        xmax = xmin + ds.RasterXSize * res
        ymin = ymax + ds.RasterYSize * gt[5]
        return TargetGrid(xmin, ymin, xmax, ymax, res, ds.GetProjection())
    finally:
        ds = None


def read_file(path: str, grid: TargetGrid) -> np.ndarray:
    """All bands of a file warped onto ``grid`` (nearest neighbour); array
    (bands, rows, cols), NaN for nodata or outside the file."""
    from osgeo import gdal

    gdal.UseExceptions()
    src = gdal.Open(path)
    try:
        nodata = src.GetRasterBand(1).GetNoDataValue()
        ds = gdal.Warp(
            "",
            src,
            format="MEM",
            outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
            width=grid.width,
            height=grid.height,
            dstSRS=grid.crs_wkt,
            resampleAlg="near",
            outputType=gdal.GDT_Float32,
            srcNodata=nodata,
            dstNodata=float("nan"),
        )
        arr = ds.ReadAsArray().astype(np.float64)
        ds = None
    finally:
        src = None
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


class BandReader:
    """Reads each file once (all bands) and serves bands by reference."""

    def __init__(self, grid: TargetGrid, read_fn=read_file):
        self.grid = grid
        self.read_fn = read_fn
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, ref: BandRef) -> np.ndarray:
        if ref.path not in self._cache:
            self._cache[ref.path] = self.read_fn(ref.path, self.grid)
        return self._cache[ref.path][ref.index - 1]

    def clear(self):
        self._cache.clear()


# ----------------------------------------------------------------------
# Depth harmonisation
# ----------------------------------------------------------------------


def interval_weights(
    intervals: list[tuple[int, int]], top: int, bottom: int
) -> list[float] | None:
    """Thickness weights of the intervals overlapping [top, bottom];
    None if the intervals do not cover the whole target layer."""
    covered = 0.0
    weights = []
    for t, b in intervals:
        overlap = max(0.0, min(b, bottom) - max(t, top))
        weights.append(overlap / (bottom - top))
        covered += overlap
    if abs(covered - (bottom - top)) > 1e-6:
        return None
    return weights


def intervals_to_layer(
    values: dict[tuple[int, int], np.ndarray], top: int, bottom: int
) -> np.ndarray | None:
    """Thickness-weighted average of interval values over [top, bottom].
    None if the available intervals do not cover the layer."""
    keys = sorted(values)
    weights = interval_weights(keys, top, bottom)
    if weights is None:
        return None
    out = None
    for key, w in zip(keys, weights, strict=True):
        if w <= 0:
            continue
        term = w * values[key]
        out = term if out is None else out + term
    return out


def points_to_layer(
    values: dict[int, np.ndarray], top: int, bottom: int
) -> np.ndarray | None:
    """Layer average from depth-point values by the trapezoidal rule (the
    SoilGrids 2017 convention). Needs points at ``top`` and ``bottom``;
    points in between are used. None if a bounding point is missing."""
    if top not in values or bottom not in values:
        return None
    pts = sorted(p for p in values if top <= p <= bottom)
    total = None
    for z0, z1 in zip(pts[:-1], pts[1:], strict=True):
        term = (z1 - z0) * (values[z0] + values[z1]) / 2.0
        total = term if total is None else total + term
    return total / (bottom - top)


# ----------------------------------------------------------------------
# Distributions
# ----------------------------------------------------------------------


def to_space(var: str, x):
    space, floor, _, _ = VARIABLE_SPACE[var]
    x = np.asarray(x, dtype=np.float64)
    if space == "log":
        return np.log(np.maximum(x, floor))
    return x


def from_space(var: str, t):
    space, _, lo, hi = VARIABLE_SPACE[var]
    x = np.exp(t) if space == "log" else np.asarray(t, dtype=np.float64)
    return np.clip(x, lo, hi)


@dataclass
class Dist:
    """Per-cell distribution of one input on the target grid: centre value
    and lower / upper standard deviations in the variable's transform space
    (split normal). NaN where unknown."""

    centre: np.ndarray
    sig_lo: np.ndarray
    sig_hi: np.ndarray

    def copy(self) -> Dist:
        return Dist(self.centre.copy(), self.sig_lo.copy(), self.sig_hi.copy())


def dist_from_quantiles(var, q50, q05=None, q95=None) -> Dist:
    """SoilGrids: Q0.5 centre, 90 % interval -> split sigmas. Without
    quantiles the sigmas are zero (no input uncertainty)."""
    centre = np.asarray(q50, dtype=np.float64)
    if q05 is None or q95 is None:
        zero = np.where(np.isfinite(centre), 0.0, np.nan)
        return Dist(centre, zero, zero.copy())
    tc = to_space(var, centre)
    lo = np.maximum((tc - to_space(var, q05)) / Z90, 0.0)
    hi = np.maximum((to_space(var, q95) - tc) / Z90, 0.0)
    return Dist(centre, lo, hi)


def dist_from_p16_p84(var, mean, p16=None, p84=None) -> Dist:
    """OpenLandMap: 30 m mean centre, symmetric sigma from the 120 m 68 %
    interval width (P16-P84)."""
    centre = np.asarray(mean, dtype=np.float64)
    if p16 is None or p84 is None:
        zero = np.where(np.isfinite(centre), 0.0, np.nan)
        return Dist(centre, zero, zero.copy())
    sig = np.maximum((to_space(var, p84) - to_space(var, p16)) / (2.0 * Z68), 0.0)
    sig = np.where(np.isfinite(centre), sig, np.nan)
    return Dist(centre, sig, sig.copy())


# ----------------------------------------------------------------------
# Products
# ----------------------------------------------------------------------


@dataclass
class ProductInputs:
    name: str
    folder: str
    texture_system: str  # "USDA" or "ISO 11277"
    interval: str  # description of the input uncertainty interval
    layers: dict[str, dict[str, Dist]] = field(default_factory=dict)
    sg2017: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    water: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    water_source: str = ""
    run: dict[str, str] = field(default_factory=dict)
    periods: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _sg_inputs(refs, read, layers, notes) -> dict:
    """SoilGrids 2.0 intervals -> target layers, per variable."""
    out: dict[str, dict[str, Dist]] = {lab: {} for lab, _, _ in layers}
    by: dict[tuple[str, str], dict[tuple[int, int], BandRef]] = {}
    for r in refs:
        if r.kind == "sg":
            by.setdefault((r.var, r.stat), {})[(r.top, r.bottom)] = r
    for var in INPUT_VARIABLES:
        centre_stat = "Q0.5" if (var, "Q0.5") in by else "mean"
        if (var, centre_stat) not in by:
            notes.append(f"SoilGrids 2.0: {var} not in the folder.")
            continue
        for lab, top, bottom in layers:
            stats = {}
            for stat in (centre_stat, "Q0.05", "Q0.95"):
                bands = by.get((var, stat))
                if not bands:
                    continue
                vals = {k: read(ref) for k, ref in bands.items()}
                arr = intervals_to_layer(vals, top, bottom)
                if arr is not None:
                    stats[stat] = arr
            if centre_stat not in stats:
                notes.append(
                    f"SoilGrids 2.0: {var} depths do not cover {lab}; layer "
                    "skipped for this variable."
                )
                continue
            if "Q0.05" not in stats or "Q0.95" not in stats:
                notes.append(
                    f"SoilGrids 2.0: {var} {lab} has no Q0.05/Q0.95 - no input "
                    "uncertainty for this variable."
                )
            out[lab][var] = dist_from_quantiles(
                var, stats[centre_stat], stats.get("Q0.05"), stats.get("Q0.95")
            )
    return out


def _sg2017_inputs(refs, read, layers) -> dict:
    out: dict[str, dict[str, np.ndarray]] = {lab: {} for lab, _, _ in layers}
    by: dict[str, dict[int, BandRef]] = {}
    for r in refs:
        if r.kind == "sg2017":
            by.setdefault(r.var, {})[r.top] = r
    for var, pts in by.items():
        if var not in INPUT_VARIABLES:
            continue
        vals = {p: read(ref) for p, ref in pts.items()}
        for lab, top, bottom in layers:
            arr = points_to_layer(vals, top, bottom)
            if arr is not None:
                out[lab][var] = arr
    return out


def _olm_inputs(refs, read, layers, notes, periods_used) -> dict:
    """OpenLandMap intervals -> target layers (latest period per variable)."""
    out: dict[str, dict[str, Dist]] = {lab: {} for lab, _, _ in layers}
    by: dict[tuple[str, str, str], dict[tuple[int, int], BandRef]] = {}
    for r in refs:
        if r.kind == "olm":
            by.setdefault((r.var, r.stat, r.period), {})[(r.top, r.bottom)] = r
    for var in INPUT_VARIABLES:
        periods = sorted({p for (v, s, p) in by if v == var and s == "mean_30m"})
        centre_stat = "mean_30m"
        if not periods:
            periods = sorted({p for (v, s, p) in by if v == var and s == "mean_120m"})
            centre_stat = "mean_120m"
        if not periods:
            if var not in ("cfvo", "cec"):  # never published by OpenLandMap
                notes.append(f"OpenLandMap: {var} not in the folder.")
            continue
        period = periods[-1]
        periods_used[var] = period
        unc_periods = sorted(
            {p for (v, s, p) in by if v == var and s == "p16_120m"}
            & {p for (v, s, p) in by if v == var and s == "p84_120m"}
        )
        unc_period = period if period in unc_periods else (unc_periods or [None])[-1]
        for lab, top, bottom in layers:
            stats = {}
            for stat, per in (
                (centre_stat, period),
                ("p16_120m", unc_period),
                ("p84_120m", unc_period),
            ):
                bands = by.get((var, stat, per)) if per else None
                if not bands:
                    continue
                vals = {k: read(ref) for k, ref in bands.items()}
                arr = intervals_to_layer(vals, top, bottom)
                if arr is not None:
                    stats[stat] = arr
            if centre_stat not in stats:
                notes.append(
                    f"OpenLandMap: {var} depths do not cover {lab}; layer "
                    "skipped for this variable."
                )
                continue
            if "p16_120m" not in stats or "p84_120m" not in stats:
                notes.append(
                    f"OpenLandMap: {var} {lab} has no P16/P84 - no input "
                    "uncertainty for this variable."
                )
            out[lab][var] = dist_from_p16_p84(
                var, stats[centre_stat], stats.get("p16_120m"), stats.get("p84_120m")
            )
    return out


def dist_from_mean_sd(var, mean, sd=None, log1p_sd=False) -> Dist:
    """iSDAsoil: mean centre, 90 % interval mean -/+ 1.645 sd (sd of
    ln(1 + x) for the log-stored properties), then as SoilGrids."""
    centre = np.asarray(mean, dtype=np.float64)
    if sd is None:
        return dist_from_quantiles(var, centre)
    sd = np.asarray(sd, dtype=np.float64)
    if log1p_sd:
        t = np.log1p(np.maximum(centre, 0.0))
        q05 = np.expm1(t - Z90 * sd)
        q95 = np.expm1(t + Z90 * sd)
    else:
        q05 = centre - Z90 * sd
        q95 = centre + Z90 * sd
    lo = VARIABLE_SPACE[var][2]
    q05 = np.maximum(q05, lo)
    return dist_from_quantiles(var, centre, q05, q95)


def _isda_inputs(refs, read, layers, notes) -> dict:
    """iSDAsoil 0-20 / 20-50 cm -> target layers. The 20-50 cm value stands
    in for up to ISDA_EXTEND_CM below 50 cm; deeper layers are skipped."""
    out: dict[str, dict[str, Dist]] = {lab: {} for lab, _, _ in layers}
    by: dict[tuple[str, str], dict[tuple[int, int], BandRef]] = {}
    for r in refs:
        if r.kind == "isda":
            by.setdefault((r.var, r.stat), {})[(r.top, r.bottom)] = r
    extended = set()
    for var in INPUT_VARIABLES:
        if (var, "mean") not in by:
            if var != "cec":  # iSDA has effective CEC only
                notes.append(f"iSDAsoil: {var} not in the folder.")
            continue
        log1p_sd = any(
            r.units.startswith("ln(1+") for r in by.get((var, "sd"), {}).values()
        )
        for lab, top, bottom in layers:
            stats = {}
            for stat in ("mean", "sd"):
                bands = by.get((var, stat))
                if not bands:
                    continue
                vals = {k: read(ref) for k, ref in bands.items()}
                deepest = max(vals)
                if deepest[1] < bottom <= deepest[1] + ISDA_EXTEND_CM:
                    vals[(deepest[0], bottom)] = vals.pop(deepest)
                    extended.add(lab)
                arr = intervals_to_layer(vals, top, bottom)
                if arr is not None:
                    stats[stat] = arr
            if "mean" not in stats:
                continue
            if "sd" not in stats:
                notes.append(
                    f"iSDAsoil: {var} {lab} has no standard deviation - no input "
                    "uncertainty for this variable."
                )
            out[lab][var] = dist_from_mean_sd(
                var, stats["mean"], stats.get("sd"), log1p_sd
            )
    for lab in sorted(extended):
        notes.append(
            f"iSDAsoil {lab}: the 20-50 cm value stands in for the part of the "
            "layer below 50 cm."
        )
    skipped = [lab for lab, _, _ in layers if not out[lab]]
    if skipped:
        notes.append(
            f"iSDAsoil is mapped to 50 cm only: no iSDAsoil member for "
            f"{', '.join(skipped)}."
        )
    return out


def _water(refs, read, layers, kind_codes, product, out_notes) -> tuple[dict, str]:
    """Mapped water contents (vol %) -> m3/m3 per target layer."""
    out: dict[str, dict[str, np.ndarray]] = {lab: {} for lab, _, _ in layers}
    source = ""
    if product == PRODUCT_OLM:
        pts: dict[str, dict[int, BandRef]] = {}
        for r in refs:
            if r.kind == "water":
                pts.setdefault(r.var, {})[r.top] = r
        for code, bands in pts.items():
            vals = {p: read(ref) for p, ref in bands.items()}
            for lab, top, bottom in layers:
                arr = points_to_layer(vals, top, bottom)
                if arr is not None:
                    out[lab][code] = arr / 100.0
                    source = "OpenLandMap 250 m water content (Hengl & Gupta, 2019)"
    else:
        by: dict[str, dict[tuple[int, int], BandRef]] = {}
        for r in refs:
            if r.kind == "sg" and r.var in kind_codes:
                if r.stat in ("Q0.5", "mean"):
                    key = (r.top, r.bottom)
                    cur = by.setdefault(kind_codes[r.var], {})
                    if key not in cur or r.stat == "Q0.5":
                        cur[key] = r
        for code, bands in by.items():
            vals = {k: read(ref) for k, ref in bands.items()}
            for lab, top, bottom in layers:
                arr = intervals_to_layer(vals, top, bottom)
                if arr is not None:
                    out[lab][code] = arr / 100.0
                    source = "SoilGrids 2.0 water content (wv0033 / wv1500)"
    return out, source


def load_product(
    folder: str,
    reader: BandReader,
    layers=TARGET_LAYERS,
    expected: str | None = None,
    describe_fn=band_descriptions,
) -> ProductInputs:
    """Read and harmonise one extraction folder."""
    refs = scan_folder(folder, describe_fn)
    product = detect_product(refs)
    if product is None:
        raise SoilDataError(
            f"No SoilGrids, OpenLandMap or iSDAsoil layers were recognised in "
            f"{folder}. Point the tool at the output folder of 'Extract: "
            "SoilGrids 2.0', 'Extract: OpenLandMap Soils' or 'Extract: iSDAsoil "
            "(Africa)'."
        )
    if expected and product != expected:
        raise SoilDataError(
            f"The folder given for {expected} contains {product} layers: {folder}"
        )
    notes: list[str] = []
    if product == PRODUCT_SG:
        p = ProductInputs(
            product, folder, "USDA", "90 % (Q0.05-Q0.95, 250 m)", notes=notes
        )
        p.layers = _sg_inputs(refs, reader, layers, notes)
        p.sg2017 = _sg2017_inputs(refs, reader, layers)
        p.water, p.water_source = _water(
            refs, reader, layers, SG_WATER_CODES, product, notes
        )
    elif product == PRODUCT_ISDA:
        p = ProductInputs(
            product,
            folder,
            "USDA",
            "90 % (mean -/+ 1.645 sd, 30 m; 0-50 cm only)",
            notes=notes,
        )
        p.layers = _isda_inputs(refs, reader, layers, notes)
        p.periods = {"all": "2001-2017"}
    else:
        p = ProductInputs(
            product,
            folder,
            "ISO 11277",
            "68 % (P16-P84, 120 m) around the 30 m mean",
            notes=notes,
        )
        p.layers = _olm_inputs(refs, reader, layers, notes, p.periods)
        p.water, p.water_source = _water(refs, reader, layers, {}, product, notes)
    missing = [
        v for v in TEXTURE + ("soc",) if not any(v in d for d in p.layers.values())
    ]
    if missing:
        raise SoilDataError(
            f"{product}: required input(s) missing in {folder}: "
            f"{', '.join(missing)} (sand, silt, clay and soc are needed)."
        )
    p.run = read_run_metadata(folder)
    return p


def reference_grid(folders: list[str], describe_fn=band_descriptions) -> TargetGrid:
    """Grid of the finest clay raster among the folders (the default
    processing grid)."""
    best = None
    for folder in folders:
        for ref in scan_folder(folder, describe_fn):
            if ref.var == "clay" and ref.kind in ("sg", "olm", "isda"):
                if ref.kind in ("olm", "isda") and not ref.stat.startswith("mean"):
                    continue
                g = grid_of(ref.path)
                if best is None or g.res < best.res:
                    best = g
                break
    if best is None:
        raise SoilDataError(
            "No SoilGrids, OpenLandMap or iSDAsoil clay layer was recognised in "
            "the input folder(s). Point the tool at the output folder of 'Extract: "
            "SoilGrids 2.0', 'Extract: OpenLandMap Soils' or 'Extract: iSDAsoil "
            "(Africa)'."
        )
    return best
