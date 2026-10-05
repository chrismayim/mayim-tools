"""
Core logic for "Extract: SoilGrids 2.0" - no QGIS dependency.

GDAL is imported only inside the default reader/sampler functions at
the bottom of this module. Everything that touches the network is
injectable (``read_grid_fn``, ``sample_fn``, ``write_fn``), so the full
orchestration is unit-tested with fakes - no network, no ISRIC server,
no GDAL needed (see tests/test_soilgrids_extract_core.py).

DATA SOURCE
- ISRIC SoilGrids 2.0 (Poggio et al. 2021, SOIL 7:217-240), 250 m,
  six GlobalSoilMap depth intervals (0-5 ... 100-200 cm), CC-BY 4.0.
- Each layer is published as a WebDAV VRT that indexes ~13000 COG tiles:
      https://files.isric.org/soilgrids/latest/data/<var>/<var>_<depth>_<stat>.vrt
  Measured from South Africa (2026-10-03): opening one VRT costs ~17 s
  (5.9 MB of XML), reading the tiles themselves ~0.8 s. So the tool
  reads ONE reference VRT per run, finds the tiles that cover the area
  (every layer uses the same tiling), and then reads those tiles
  directly for every layer, several layers in parallel ("Tiles"
  route). If that fails for a layer, it falls back to that layer's
  full VRT ("VRT" route; slow but always correct).
  Native CRS is Interrupted Goode Homolosine, taken from the data.
- Not used: the REST API (paused by ISRIC) and the WCS (did not return
  GeoTIFFs in testing, 2026-10-03).
- Optional SoilGrids 2017 archive (Hengl et al. 2017; ODbL v1.0):
  depth to bedrock (Shangguan et al. 2017) and mean predictions of the
  same properties, which - unlike 2.0 - also cover urban and bare
  areas. Its depths are points (0, 5, 15, 30, 60, 100, 200 cm); the
  tool averages the two bounding points (trapezoidal rule) to give
  the same six intervals as 2.0, so both line up band for band.

UNITS
SoilGrids stores integers. Every value written by this tool has
already been divided by ISRIC's conversion factor (``VARIABLES``), so
outputs are in conventional units (%, g/kg, g/cm3, vol %, pH,
cmol(c)/kg). SOC is NOT converted to organic matter here: the SOC to
OM factor is a method choice made by the hydraulic-property tools.

UNCERTAINTY
Each variable can be extracted as the mean and the 5 %, 50 % and 95 %
prediction quantiles (SoilGrids' own 90 % prediction interval). The
derived relative 90 % interval width RU90 = (Q0.95 - Q0.05) / Q0.50 is
computed by this tool from those quantiles (ISRIC's own "uncertainty"
layer is not downloaded, which avoids its separate scaling rules).
Texture quantiles are MARGINAL: Q0.05 sand + Q0.05 clay is not a real
soil. Uncertainty propagation must sample and renormalise the
fractions, not combine quantiles directly.
"""

from __future__ import annotations

import math
import os
import re
import threading
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

TOOL_VERSION = "0.3.0"

BASE_URL = "https://files.isric.org/soilgrids/latest/data/"
SG2017_URL = "https://files.isric.org/soilgrids/former/2017-03-10/data/"
BASE_DIR = f"/vsicurl/{BASE_URL}"
SG2017_DIR = f"/vsicurl/{SG2017_URL}"
REF_LAYER = ("clay", "0-5cm", "mean")  # tile layout reference
DEFAULT_WORKERS = 8

CITATION = (
    "Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., "
    "Kempen, B., Ribeiro, E. and Rossiter, D. (2021). SoilGrids 2.0: "
    "producing soil information for the globe with quantified spatial "
    "uncertainty. SOIL, 7, 217-240. https://doi.org/10.5194/soil-7-217-2021"
)
CITATION_2017 = (
    "Shangguan, W., Hengl, T., Mendes de Jesus, J., Yuan, H. and Dai, Y. "
    "(2017). Mapping the global depth to bedrock for land surface "
    "modeling. Journal of Advances in Modeling Earth Systems, 9, 65-88. "
    "https://doi.org/10.1002/2016MS000686"
)
CITATION_SG2017 = (
    "Hengl, T. et al. (2017). SoilGrids250m: Global gridded soil "
    "information based on machine learning. PLoS ONE 12(2): e0169748. "
    "https://doi.org/10.1371/journal.pone.0169748"
)
LICENCE = "CC-BY 4.0 (ISRIC data policy: https://www.isric.org/about/data-policy)"
LICENCE_2017 = "Open Database License (ODbL) v1.0"

NODATA_OUT = -9999.0
BUFFER_CELLS = 2
WINDOW_LIMIT_CELLS = 2000  # point mode: read one window if points fit in this
TEXTURE_SUM_TOLERANCE = 2.0  # % - flag cells where sand+silt+clay is further off
DEFAULT_MAX_AREA_KM2 = 50000.0
EARTH_RADIUS_KM = 6371.0088


class SoilGridsError(Exception):
    """Raised for user-facing errors (bad inputs, area limit, failed reads)."""


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SoilVariable:
    code: str
    label: str
    mapped_units: str
    factor: float
    units: str
    default: bool
    purpose: str


VARIABLES: tuple[SoilVariable, ...] = (
    SoilVariable(
        "sand", "Sand (0.05-2 mm, USDA)", "g/kg", 10, "%", True, "texture; all PTFs"
    ),
    SoilVariable(
        "silt",
        "Silt (0.002-0.05 mm, USDA)",
        "g/kg",
        10,
        "%",
        True,
        "texture; some PTFs",
    ),
    SoilVariable(
        "clay", "Clay (< 0.002 mm)", "g/kg", 10, "%", True, "texture; all PTFs"
    ),
    SoilVariable(
        "soc",
        "Soil organic carbon content",
        "dg/kg",
        10,
        "g/kg",
        True,
        "organic matter (SOC is not converted to OM here)",
    ),
    SoilVariable(
        "bdod",
        "Bulk density of the fine earth",
        "cg/cm3",
        100,
        "g/cm3",
        True,
        "density factor; Rosetta/Wosten",
    ),
    SoilVariable(
        "cfvo",
        "Coarse fragments (> 2 mm), volumetric",
        "cm3/dm3 (vol per mille)",
        10,
        "vol %",
        True,
        "gravel correction; bulk-soil Ksat",
    ),
    SoilVariable("phh2o", "pH in water", "pH x 10", 10, "pH", True, "euptf2 / Toth"),
    SoilVariable(
        "cec",
        "Cation exchange capacity at pH 7",
        "mmol(c)/kg",
        10,
        "cmol(c)/kg",
        True,
        "euptf2 / Toth",
    ),
    SoilVariable(
        "wv0010",
        "Volumetric water content at 10 kPa",
        "0.1 vol %",
        10,
        "vol %",
        False,
        "independent check on PTF water retention",
    ),
    SoilVariable(
        "wv0033",
        "Volumetric water content at 33 kPa",
        "0.1 vol %",
        10,
        "vol %",
        False,
        "independent check on PTF field capacity",
    ),
    SoilVariable(
        "wv1500",
        "Volumetric water content at 1500 kPa",
        "0.1 vol %",
        10,
        "vol %",
        False,
        "independent check on PTF wilting point",
    ),
)
VARIABLE_BY_CODE = {v.code: v for v in VARIABLES}

BEDROCK_CODE = "bedrock2017"
BEDROCK_LABEL = "Depth to bedrock (SoilGrids 2017 archive)"


@dataclass(frozen=True)
class BedrockLayer:
    code: str
    label: str
    units: str


BEDROCK_LAYERS: tuple[BedrockLayer, ...] = (
    BedrockLayer("BDTICM", "Absolute depth to bedrock", "cm"),
    BedrockLayer("BDRICM", "Depth to R horizon, censored at 200 cm", "cm"),
    BedrockLayer("BDRLOG", "Probability of an R horizon within 200 cm", "%"),
)

DEPTHS: tuple[tuple[str, int, int], ...] = (
    ("0-5cm", 0, 5),
    ("5-15cm", 5, 15),
    ("15-30cm", 15, 30),
    ("30-60cm", 30, 60),
    ("60-100cm", 60, 100),
    ("100-200cm", 100, 200),
)
DEPTH_BY_LABEL = {d[0]: d for d in DEPTHS}

STATISTICS: tuple[str, ...] = ("mean", "Q0.05", "Q0.5", "Q0.95")
RU90 = "RU90"
RU90_LABEL = "Relative 90% interval width (Q0.95-Q0.05)/Q0.5"
QUANTILES_FOR_RU = ("Q0.05", "Q0.5", "Q0.95")
TEXTURE_CODES = ("sand", "silt", "clay")

PRODUCT_SG2 = "SoilGrids 2.0"
PRODUCT_SG2017 = "SoilGrids 2017"


@dataclass(frozen=True)
class Var2017:
    code: str  # 2017 file prefix
    divide_by: float  # mapped units -> the same output units as 2.0


# SoilGrids 2017 equivalents of the 2.0 variables (no water content).
VARS_2017: dict[str, Var2017] = {
    "sand": Var2017("SNDPPT", 1),  # %
    "silt": Var2017("SLTPPT", 1),  # %
    "clay": Var2017("CLYPPT", 1),  # %
    "soc": Var2017("ORCDRC", 1),  # g/kg
    "bdod": Var2017("BLDFIE", 1000),  # kg/m3 -> g/cm3
    "cfvo": Var2017("CRFVOL", 1),  # vol %
    "phh2o": Var2017("PHIHOX", 10),  # pH x 10 -> pH
    "cec": Var2017("CECSOL", 1),  # cmol(c)/kg
}
DEPTH_POINTS_2017: tuple[int, ...] = (0, 5, 15, 30, 60, 100, 200)  # sl1..sl7


def stat_tag(stat: str) -> str:
    """File-name tag for a statistic: Q0.5 -> Q0.50 (sorts and reads cleanly)."""
    return {"Q0.05": "Q0.05", "Q0.5": "Q0.50", "Q0.95": "Q0.95"}.get(stat, stat)


def layer_name(var: str, depth: str, stat: str) -> str:
    """ISRIC layer / coverage id, e.g. 'clay_0-5cm_Q0.5'."""
    return f"{var}_{depth}_{stat}"


def band_description(var: str, depth: str, stat: str, units: str) -> str:
    """Band description used to SELECT bands downstream (never band number)."""
    return f"{var}_{depth}_{stat} ({units})"


# ----------------------------------------------------------------------
# Selection planning
# ----------------------------------------------------------------------


@dataclass
class Selection:
    variables: list[str]
    depths: list[str]
    statistics: list[str]  # what is fetched (quantiles auto-added for RU90)
    written_statistics: list[str]  # what is written (user's choice)
    ru90: bool
    bedrock: bool
    sg2017: bool = False
    warnings: list[str] = field(default_factory=list)


def plan_selection(
    variables: Sequence[str],
    depths: Sequence[str],
    statistics: Sequence[str],
    ru90: bool = False,
    bedrock: bool = False,
    sg2017: bool = False,
) -> Selection:
    """Validate and order the user's selection.

    RU90 needs Q0.05, Q0.5 and Q0.95; missing ones are fetched (so RU90
    can be computed) but only written if the user selected them."""
    unknown = [v for v in variables if v not in VARIABLE_BY_CODE]
    if unknown:
        raise SoilGridsError(f"Unknown SoilGrids variable(s): {', '.join(unknown)}")
    bad_depths = [d for d in depths if d not in DEPTH_BY_LABEL]
    if bad_depths:
        raise SoilGridsError(f"Unknown depth interval(s): {', '.join(bad_depths)}")
    bad_stats = [s for s in statistics if s not in STATISTICS]
    if bad_stats:
        raise SoilGridsError(f"Unknown statistic(s): {', '.join(bad_stats)}")

    variables_ordered = [v.code for v in VARIABLES if v.code in variables]
    depths_ordered = [d[0] for d in DEPTHS if d[0] in depths]
    written = [s for s in STATISTICS if s in statistics]
    fetched = list(written)
    warnings: list[str] = []

    if variables_ordered and not depths_ordered:
        raise SoilGridsError("Select at least one depth interval.")
    if variables_ordered and not written and not ru90:
        raise SoilGridsError("Select at least one statistic.")
    if not variables_ordered and not bedrock:
        raise SoilGridsError("Select at least one variable.")

    if ru90:
        missing = [q for q in QUANTILES_FOR_RU if q not in fetched]
        if missing:
            warnings.append(
                "RU90 needs Q0.05, Q0.5 and Q0.95; also fetching "
                f"{', '.join(missing)} (used for RU90 only, not written)."
            )
            fetched = [s for s in STATISTICS if s in fetched or s in missing]

    if sg2017 and not any(v in VARS_2017 for v in variables_ordered):
        warnings.append(
            "SoilGrids 2017 was requested, but none of the selected variables "
            "exists in the 2017 archive (water content has no 2017 equivalent)."
        )

    return Selection(
        variables=variables_ordered,
        depths=depths_ordered,
        statistics=fetched,
        written_statistics=written,
        ru90=ru90,
        bedrock=bedrock,
        sg2017=sg2017,
        warnings=warnings,
    )


# ----------------------------------------------------------------------
# Area and grid
# ----------------------------------------------------------------------


def area_km2(bounds_ll: tuple[float, float, float, float]) -> float:
    """Area of a lon/lat box on a sphere (km2)."""
    lon0, lat0, lon1, lat1 = bounds_ll
    if lon1 <= lon0 or lat1 <= lat0:
        return 0.0
    dlon = math.radians(lon1 - lon0)
    band = math.sin(math.radians(lat1)) - math.sin(math.radians(lat0))
    return EARTH_RADIUS_KM**2 * dlon * band


def check_area(
    bounds_ll: tuple[float, float, float, float], max_area_km2: float
) -> float:
    """Return the area of interest in km2; raise if above the limit."""
    lon0, lat0, lon1, lat1 = bounds_ll
    if not (-180 <= lon0 < lon1 <= 180 and -90 <= lat0 < lat1 <= 90):
        raise SoilGridsError(
            "The area of interest has an invalid or empty extent "
            f"(lon {lon0:.4f} to {lon1:.4f}, lat {lat0:.4f} to {lat1:.4f})."
        )
    area = area_km2(bounds_ll)
    if max_area_km2 > 0 and area > max_area_km2:
        raise SoilGridsError(
            f"The area of interest is {area:.0f} km2 (bounding box), which is "
            f"above the maximum of {max_area_km2:.0f} km2. Reduce the area, or "
            "raise 'Maximum area per run (km2)' under Advanced parameters."
        )
    return area


@dataclass(frozen=True)
class TargetGrid:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    res: float
    crs_wkt: str

    @property
    def width(self) -> int:
        return int(round((self.xmax - self.xmin) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.ymax - self.ymin) / self.res))

    @property
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        return (self.xmin, self.res, 0.0, self.ymax, 0.0, -self.res)


def make_grid(
    bounds: tuple[float, float, float, float],
    res: float,
    crs_wkt: str,
    buffer_cells: int = BUFFER_CELLS,
) -> TargetGrid:
    """Output grid: AOI bounds buffered by ``buffer_cells`` and snapped
    outward to whole multiples of ``res`` (stable alignment between runs)."""
    if res <= 0:
        raise SoilGridsError("Output resolution must be greater than zero.")
    xmin, ymin, xmax, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise SoilGridsError("The area of interest has an empty extent.")
    pad = buffer_cells * res
    xmin = math.floor((xmin - pad) / res) * res
    ymin = math.floor((ymin - pad) / res) * res
    xmax = math.ceil((xmax + pad) / res) * res
    ymax = math.ceil((ymax + pad) / res) * res
    return TargetGrid(xmin, ymin, xmax, ymax, res, crs_wkt)


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------


def convert(raw: np.ndarray, factor: float) -> np.ndarray:
    """Mapped integers -> conventional units. NaN (nodata) stays NaN -
    nodata is never turned into zero."""
    return np.asarray(raw, dtype=np.float64) / float(factor)


def summarise(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64).ravel()
    valid = arr[np.isfinite(arr)]
    out = {"valid": int(valid.size), "nodata": int(arr.size - valid.size)}
    if valid.size:
        out.update(
            min=float(valid.min()),
            median=float(np.median(valid)),
            max=float(valid.max()),
        )
    else:
        out.update(min=math.nan, median=math.nan, max=math.nan)
    return out


def relative_width(q05: np.ndarray, q50: np.ndarray, q95: np.ndarray) -> np.ndarray:
    """RU90 = (Q0.95 - Q0.05) / Q0.5; NaN where Q0.5 <= 0 or any input is NaN."""
    q05 = np.asarray(q05, dtype=np.float64)
    q50 = np.asarray(q50, dtype=np.float64)
    q95 = np.asarray(q95, dtype=np.float64)
    out = np.full(np.broadcast(q05, q50, q95).shape, np.nan)
    ok = np.isfinite(q05) & np.isfinite(q50) & np.isfinite(q95) & (q50 > 0)
    out[ok] = (q95[ok] - q05[ok]) / q50[ok]
    return out


def texture_sum_check(
    sand: np.ndarray, silt: np.ndarray, clay: np.ndarray, tol: float = 2.0
) -> dict:
    total = np.asarray(sand, float) + np.asarray(silt, float) + np.asarray(clay, float)
    valid = total[np.isfinite(total)]
    if not valid.size:
        return {"checked": 0, "flagged": 0, "max_abs_dev": math.nan}
    dev = np.abs(valid - 100.0)
    return {
        "checked": int(valid.size),
        "flagged": int((dev > tol).sum()),
        "max_abs_dev": float(dev.max()),
    }


# ----------------------------------------------------------------------
# Reading with fallback
# ----------------------------------------------------------------------


def layer_vrt(var: str, depth: str, stat: str, base: str = BASE_DIR) -> str:
    """Full WebDAV VRT of one SoilGrids 2.0 layer (slow to open: ~17 s)."""
    return f"{base}{var}/{layer_name(var, depth, stat)}.vrt"


def bedrock_path(code: str, base: str = SG2017_DIR) -> str:
    return f"{base}{code}_M_250m_ll.tif"


def sg2017_path(var: str, point_index: int, base: str = SG2017_DIR) -> str:
    """2017 mean prediction at depth point sl<k> (k = 1..7)."""
    return f"{base}{VARS_2017[var].code}_M_sl{point_index + 1}_250m_ll.tif"


def sg2017_points_needed(depths: Sequence[str]) -> list[int]:
    """Depth-point indices (0..6) needed to average the selected intervals."""
    needed = set()
    for depth in depths:
        _, top, bottom = DEPTH_BY_LABEL[depth]
        needed.add(DEPTH_POINTS_2017.index(top))
        needed.add(DEPTH_POINTS_2017.index(bottom))
    return sorted(needed)


def sg2017_interval(points: dict[int, np.ndarray], depth: str) -> np.ndarray:
    """Interval mean from the two bounding depth points (trapezoidal rule:
    for a straight line between two points this is exact)."""
    _, top, bottom = DEPTH_BY_LABEL[depth]
    a = np.asarray(points[DEPTH_POINTS_2017.index(top)], dtype=np.float64)
    b = np.asarray(points[DEPTH_POINTS_2017.index(bottom)], dtype=np.float64)
    return (a + b) / 2.0


# ----------------------------------------------------------------------
# Tile index (read ONE VRT, reuse its tiles for every layer)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TileIndex:
    """Tiles (relative to a layer folder) that cover the area of interest.

    Every SoilGrids 2.0 layer uses the same tiling, so the tiles found in
    the reference VRT are re-pointed at any other layer's folder."""

    rel_paths: tuple[str, ...]
    ref_layer: str

    def paths_for(
        self, var: str, depth: str, stat: str, base: str = BASE_DIR
    ) -> list[str]:
        name = layer_name(var, depth, stat)
        return [f"{base}{var}/./{name}/{rel}" for rel in self.rel_paths]


_SOURCE_RE = re.compile(
    r"<SourceFilename[^>]*>([^<]+)</SourceFilename>.*?<DstRect([^>]*)/>", re.S
)
_ATTR_RE = re.compile(r'(\w+)="([-\d.eE+]+)"')


def tiles_in_window(
    vrt_xml: str, ref_name: str, window: tuple[int, int, int, int]
) -> tuple[str, ...]:
    """Parse VRT XML; return the tile paths (relative to the layer folder)
    whose DstRect intersects the pixel window (x0, y0, x1, y1), exclusive
    upper bounds. Raises SoilGridsError if the layout is not recognised."""
    x0, y0, x1, y1 = window
    marker = f"{ref_name}/"
    found: list[str] = []
    n_sources = 0
    for match in _SOURCE_RE.finditer(vrt_xml):
        n_sources += 1
        attrs = {k: float(v) for k, v in _ATTR_RE.findall(match.group(2))}
        try:
            dx, dy = attrs["xOff"], attrs["yOff"]
            dw, dh = attrs["xSize"], attrs["ySize"]
        except KeyError:
            continue
        if dx >= x1 or dx + dw <= x0 or dy >= y1 or dy + dh <= y0:
            continue
        filename = match.group(1).strip().replace("\\", "/")
        if marker not in filename:
            raise SoilGridsError(
                f"Unexpected tile path in the reference VRT: {filename}"
            )
        rel = filename.split(marker, 1)[1]
        if rel not in found:
            found.append(rel)
    if not n_sources:
        raise SoilGridsError("The reference VRT lists no tiles.")
    if not found:
        raise SoilGridsError("No SoilGrids tile covers the area of interest.")
    return tuple(found)


# ----------------------------------------------------------------------
# Reading with fallback, parallel jobs
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    route: str  # "Tiles" | "VRT" | "2017 archive"
    path: str | tuple[str, ...]


def layer_sources(
    index: TileIndex | None, var: str, depth: str, stat: str, base: str = BASE_DIR
) -> list[Source]:
    sources = []
    if index is not None:
        sources.append(Source("Tiles", tuple(index.paths_for(var, depth, stat, base))))
    sources.append(Source("VRT", layer_vrt(var, depth, stat, base)))
    return sources


def read_with_fallback(sources: Sequence[Source], reader: Callable, *args):
    """Try each source in turn. Returns (result, route, errors).
    Raises SoilGridsError listing every attempt if all fail."""
    errors: list[str] = []
    for i, src in enumerate(sources):
        try:
            out = reader(src.path, *args)
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            errors.append(f"{src.route}: {exc}")
            continue
        last = i == len(sources) - 1
        if src.route == "Tiles" and not last and _all_nan(out):
            # Tiles that exist but cover a different area give no data and
            # no error; confirm against the next route before accepting.
            errors.append(f"{src.route}: no data in the area")
            continue
        return out, src.route, errors
    raise SoilGridsError("All access routes failed - " + " | ".join(errors))


def _all_nan(values) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return arr.size > 0 and not np.isfinite(arr).any()


def run_jobs(
    keys: Sequence,
    fn: Callable,
    workers: int,
    cancel_fn: Callable,
    on_done: Callable,
) -> dict:
    """Run fn(key) for every key, ``workers`` at a time. Results by key.
    Cancellation is checked as each job finishes; pending jobs are dropped."""
    results: dict = {}
    if workers <= 1 or len(keys) <= 1:
        for key in keys:
            if cancel_fn():
                raise InterruptedError("Cancelled by user")
            results[key] = fn(key)
            on_done(key)
        return results
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(fn, key): key for key in keys}
    try:
        for fut in as_completed(futures):
            key = futures[fut]
            results[key] = fut.result()
            on_done(key)
            if cancel_fn():
                raise InterruptedError("Cancelled by user")
    finally:
        for fut in futures:
            fut.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    return results


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------


@dataclass
class LayerRecord:
    file: str
    band: int
    layer: str
    variable: str
    depth: str
    statistic: str
    units: str
    factor: float
    route: str
    stats: dict
    product: str = PRODUCT_SG2


@dataclass
class RunResult:
    mode: str
    files: list[str] = field(default_factory=list)
    layers: list[LayerRecord] = field(default_factory=list)
    uncertainty: list[dict] = field(default_factory=list)
    texture: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)  # point mode
    load_files: list[str] = field(default_factory=list)
    tiles: dict = field(default_factory=dict)  # variable -> tiles (0 = VRT)
    seconds: float = 0.0


def _noop(*_args, **_kwargs):
    return None


def _never_cancelled() -> bool:
    return False


class _Progress:
    def __init__(self, total: int, progress_fn: Callable):
        self.total = max(total, 1)
        self.done = 0
        self.progress_fn = progress_fn

    def step(self, message: str) -> None:
        self.done += 1
        self.progress_fn(min(self.done / self.total, 0.999), message)


def count_jobs(selection: Selection) -> int:
    n = len(selection.variables) * len(selection.statistics) * len(selection.depths)
    if selection.sg2017:
        n2017 = len(sg2017_points_needed(selection.depths))
        n += n2017 * sum(1 for v in selection.variables if v in VARS_2017)
    if selection.bedrock:
        n += len(BEDROCK_LAYERS)
    return n


def _options(options_ctx, default_used: bool):
    if options_ctx is not None:
        return options_ctx
    return http_options() if default_used else nullcontext()


def _build_indices(
    index_fn, variables, workers, cancel_fn, log_fn, result, bounds_ll, **kwargs
) -> dict:
    """One tile index per variable, read from that variable's own first
    layer (ISRIC's tile layouts differ between variables: bdod and phh2o did
    not match clay in the live test, 2026-10-05). Built in parallel; a
    variable whose index fails is read through full VRTs."""

    def job(var):
        try:
            return index_fn(bounds_ll, ref=(var, *REF_LAYER[1:]), **kwargs)
        except Exception as exc:  # noqa: BLE001 - fall back to full VRTs
            return exc

    got = run_jobs(list(variables), job, workers, cancel_fn, _noop)
    indices = {}
    for var in variables:
        index = got[var]
        if isinstance(index, Exception):
            msg = (
                f"{var}: could not build the tile index, so its layers are read "
                f"through the full VRT (much slower): {index}"
            )
            result.warnings.append(msg)
            log_fn(msg)
            indices[var] = None
            result.tiles[var] = 0
        else:
            indices[var] = index
            result.tiles[var] = len(index.rel_paths)
    return indices


def _check_empty(records, result, log_fn):
    """Warn about written layers that contain no valid value at all."""
    for rec in records:
        if rec.stats.get("valid") == 0:
            msg = (
                f"{rec.layer} ({rec.product}) has no valid values in the area "
                "of interest."
            )
            result.warnings.append(msg)
            log_fn(msg)


# ----------------------------------------------------------------------
# Area mode
# ----------------------------------------------------------------------


def output_file_name(var: str, stat: str) -> str:
    return f"soilgrids_{var}_{stat_tag(stat)}.tif"


def output_file_name_2017(var: str) -> str:
    return f"soilgrids2017_{var}_mean.tif"


BEDROCK_FILE_NAME = "soilgrids2017_depth_to_bedrock.tif"


def extract_area(
    selection: Selection,
    grid: TargetGrid,
    bounds_ll: tuple[float, float, float, float],
    out_dir: str,
    *,
    read_grid_fn: Callable | None = None,
    index_fn: Callable | None = None,
    write_fn: Callable | None = None,
    workers: int = DEFAULT_WORKERS,
    base: str = BASE_DIR,
    base_2017: str = SG2017_DIR,
    options_ctx=None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    """Fetch every selected layer onto ``grid`` and write one multi-band
    GeoTIFF per variable x statistic (one band per depth), plus RU90 files,
    optional SoilGrids 2017 mean files and the optional depth-to-bedrock
    file. ``write_fn(path, grid, bands, units)``; bands = [(description,
    array)] - see export.write_multiband_geotiff."""
    if write_fn is None:
        raise SoilGridsError("No raster writer supplied.")
    default_used = read_grid_fn is None or index_fn is None
    read = read_grid_fn or gdal_read_grid
    make_index = index_fn or gdal_tile_index
    result = RunResult(mode="area", warnings=list(selection.warnings))
    progress = _Progress(count_jobs(selection), progress_fn)
    t_start = datetime.now(UTC)

    with _options(options_ctx, default_used):
        indices: dict = {}
        if selection.variables:
            progress_fn(0.0, "Locating SoilGrids tiles (one index per variable)")
            indices = _build_indices(
                make_index,
                selection.variables,
                workers,
                cancel_fn,
                log_fn,
                result,
                bounds_ll,
                base=base,
            )
        texture_means: dict[str, dict[str, np.ndarray]] = {}

        for var in selection.variables:
            meta = VARIABLE_BY_CODE[var]
            keys = [(d, s) for s in selection.statistics for d in selection.depths]

            def job(key, var=var):
                depth, stat = key
                return read_with_fallback(
                    layer_sources(indices.get(var), var, depth, stat, base),
                    read,
                    grid,
                )

            def done(key, var=var):
                progress.step(f"Read {layer_name(var, key[0], key[1])}")

            got = run_jobs(keys, job, workers, cancel_fn, done)
            quantiles: dict[str, dict[str, np.ndarray]] = {}
            for stat in selection.statistics:
                bands: list[tuple[str, np.ndarray]] = []
                records: list[LayerRecord] = []
                path = os.path.join(out_dir, output_file_name(var, stat))
                for depth in selection.depths:
                    raw, route, errors = got[(depth, stat)]
                    name = layer_name(var, depth, stat)
                    for err in errors:
                        msg = f"{name}: primary route failed, used {route} ({err})"
                        result.warnings.append(msg)
                        log_fn(msg)
                    arr = convert(raw, meta.factor)
                    bands.append((band_description(var, depth, stat, meta.units), arr))
                    records.append(
                        LayerRecord(
                            file=os.path.basename(path),
                            band=len(bands),
                            layer=name,
                            variable=var,
                            depth=depth,
                            statistic=stat,
                            units=meta.units,
                            factor=meta.factor,
                            route=route,
                            stats=summarise(arr),
                        )
                    )
                    if stat in QUANTILES_FOR_RU and selection.ru90:
                        quantiles.setdefault(depth, {})[stat] = arr
                    if stat == "mean" and var in TEXTURE_CODES:
                        texture_means.setdefault(depth, {})[var] = arr
                if stat in selection.written_statistics:
                    write_fn(path, grid, bands, meta.units)
                    result.files.append(path)
                    result.layers.extend(records)
                    _check_empty(records, result, log_fn)
                    if stat == "Q0.5" or (
                        stat == "mean" and "Q0.5" not in selection.written_statistics
                    ):
                        result.load_files.append(path)
            del got

            if selection.ru90:
                _write_ru90(var, selection, quantiles, grid, out_dir, write_fn, result)

        for depth in selection.depths:
            means = texture_means.get(depth, {})
            if all(code in means for code in TEXTURE_CODES):
                result.texture.append(
                    {
                        "depth": depth,
                        **texture_sum_check(
                            means["sand"],
                            means["silt"],
                            means["clay"],
                            TEXTURE_SUM_TOLERANCE,
                        ),
                    }
                )

        if selection.sg2017:
            _area_2017(
                selection,
                grid,
                out_dir,
                read,
                write_fn,
                workers,
                base_2017,
                progress,
                log_fn,
                cancel_fn,
                result,
            )

        if selection.bedrock:
            _area_bedrock(
                grid,
                out_dir,
                read,
                write_fn,
                workers,
                base_2017,
                progress,
                cancel_fn,
                result,
            )

    result.seconds = (datetime.now(UTC) - t_start).total_seconds()
    progress_fn(1.0, "Done")
    return result


def _write_ru90(var, selection, quantiles, grid, out_dir, write_fn, result):
    ru_bands = []
    path = os.path.join(out_dir, output_file_name(var, RU90))
    for depth in selection.depths:
        q = quantiles[depth]
        ru = relative_width(q["Q0.05"], q["Q0.5"], q["Q0.95"])
        ru_bands.append((band_description(var, depth, RU90, "ratio"), ru))
        result.uncertainty.append({"variable": var, "depth": depth, **_ru_summary(ru)})
        result.layers.append(
            LayerRecord(
                file=os.path.basename(path),
                band=len(ru_bands),
                layer=f"{var}_{depth}_{RU90}",
                variable=var,
                depth=depth,
                statistic=RU90,
                units="ratio",
                factor=1.0,
                route="derived",
                stats=summarise(ru),
            )
        )
    write_fn(path, grid, ru_bands, "ratio")
    result.files.append(path)


def _area_2017(
    selection,
    grid,
    out_dir,
    read,
    write_fn,
    workers,
    base_2017,
    progress,
    log_fn,
    cancel_fn,
    result,
):
    point_idx = sg2017_points_needed(selection.depths)
    for var in [v for v in selection.variables if v in VARS_2017]:
        meta = VARIABLE_BY_CODE[var]
        v17 = VARS_2017[var]

        def job(k, var=var):
            return read(sg2017_path(var, k, base_2017), grid)

        def done(k, var=var):
            progress.step(f"Read {VARS_2017[var].code} sl{k + 1} (2017)")

        try:
            raw_points = run_jobs(point_idx, job, workers, cancel_fn, done)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - optional layer: warn, carry on
            msg = f"SoilGrids 2017 {v17.code} could not be read - skipped: {exc}"
            result.warnings.append(msg)
            log_fn(msg)
            continue
        points = {k: convert(raw, v17.divide_by) for k, raw in raw_points.items()}
        path = os.path.join(out_dir, output_file_name_2017(var))
        bands = []
        for depth in selection.depths:
            arr = sg2017_interval(points, depth)
            bands.append((f"{var}_{depth}_mean [SG2017] ({meta.units})", arr))
            result.layers.append(
                LayerRecord(
                    file=os.path.basename(path),
                    band=len(bands),
                    layer=f"{v17.code}_M (trapezoid {depth})",
                    variable=var,
                    depth=depth,
                    statistic="mean",
                    units=meta.units,
                    factor=v17.divide_by,
                    route="2017 archive",
                    stats=summarise(arr),
                    product=PRODUCT_SG2017,
                )
            )
        write_fn(path, grid, bands, meta.units)
        result.files.append(path)
        _check_empty(
            [r for r in result.layers if r.file == os.path.basename(path)],
            result,
            log_fn,
        )


def _area_bedrock(
    grid, out_dir, read, write_fn, workers, base_2017, progress, cancel_fn, result
):
    codes = [b.code for b in BEDROCK_LAYERS]

    def job(code):
        return read_with_fallback(
            [Source("2017 archive", bedrock_path(code, base_2017))], read, grid
        )

    got = run_jobs(
        codes,
        job,
        workers,
        cancel_fn,
        lambda code: progress.step(f"Read {code} (2017)"),
    )
    path = os.path.join(out_dir, BEDROCK_FILE_NAME)
    bands = []
    for layer in BEDROCK_LAYERS:
        raw, route, _ = got[layer.code]
        arr = np.asarray(raw, dtype=np.float64)
        bands.append((f"{layer.code} {layer.label} ({layer.units})", arr))
        result.layers.append(
            LayerRecord(
                file=os.path.basename(path),
                band=len(bands),
                layer=f"{layer.code}_M_250m_ll",
                variable=layer.code,
                depth="n/a",
                statistic="mean",
                units=layer.units,
                factor=1.0,
                route=route,
                stats=summarise(arr),
                product=PRODUCT_SG2017,
            )
        )
    write_fn(path, grid, bands, "cm / %")
    result.files.append(path)
    result.load_files.append(path)


def _ru_summary(ru: np.ndarray) -> dict:
    valid = ru[np.isfinite(ru)]
    if not valid.size:
        return {"cells": 0, "median_ru90": math.nan, "p90_ru90": math.nan}
    return {
        "cells": int(valid.size),
        "median_ru90": float(np.median(valid)),
        "p90_ru90": float(np.percentile(valid, 90)),
    }


# ----------------------------------------------------------------------
# Point mode
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    label: str
    lon: float
    lat: float


def extract_points(
    selection: Selection,
    sites: Sequence[Site],
    *,
    sample_fn: Callable | None = None,
    index_fn: Callable | None = None,
    workers: int = DEFAULT_WORKERS,
    base: str = BASE_DIR,
    base_2017: str = SG2017_DIR,
    options_ctx=None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    """Sample every selected layer at every site. Rows are long format:
    Site, Longitude, Latitude, Product, Variable, Description,
    DepthTop_cm, DepthBottom_cm, Statistic, Value, Units, Route."""
    if not sites:
        raise SoilGridsError("No points to sample.")
    default_used = sample_fn is None or index_fn is None
    sample = sample_fn or gdal_sample_points
    make_index = index_fn or gdal_tile_index
    result = RunResult(mode="points", warnings=list(selection.warnings))
    progress = _Progress(count_jobs(selection), progress_fn)
    lonlats = [(s.lon, s.lat) for s in sites]
    store: dict[tuple[str, str, str], list[float]] = {}
    t_start = datetime.now(UTC)

    def add_rows(product, var, label, depth, stat, values, units, route):
        top, bottom = ("", "")
        if depth in DEPTH_BY_LABEL:
            _, top, bottom = DEPTH_BY_LABEL[depth]
        for site, value in zip(sites, values, strict=True):
            result.rows.append(
                {
                    "Site": site.label,
                    "Longitude": site.lon,
                    "Latitude": site.lat,
                    "Product": product,
                    "Variable": var,
                    "Description": label,
                    "DepthTop_cm": top,
                    "DepthBottom_cm": bottom,
                    "Statistic": stat,
                    "Value": value,
                    "Units": units,
                    "Route": route,
                }
            )

    with _options(options_ctx, default_used):
        indices: dict = {}
        if selection.variables:
            progress_fn(0.0, "Locating SoilGrids tiles (one index per variable)")
            lons = [p[0] for p in lonlats]
            lats = [p[1] for p in lonlats]
            indices = _build_indices(
                make_index,
                selection.variables,
                workers,
                cancel_fn,
                log_fn,
                result,
                (min(lons), min(lats), max(lons), max(lats)),
                points=lonlats,
                base=base,
            )

        for var in selection.variables:
            meta = VARIABLE_BY_CODE[var]
            keys = [(d, s) for s in selection.statistics for d in selection.depths]

            def job(key, var=var):
                depth, stat = key
                return read_with_fallback(
                    layer_sources(indices.get(var), var, depth, stat, base),
                    sample,
                    lonlats,
                )

            def done(key, var=var):
                progress.step(f"Sampled {layer_name(var, key[0], key[1])}")

            got = run_jobs(keys, job, workers, cancel_fn, done)
            for stat in selection.statistics:
                for depth in selection.depths:
                    raw, route, errors = got[(depth, stat)]
                    name = layer_name(var, depth, stat)
                    for err in errors:
                        msg = f"{name}: primary route failed, used {route} ({err})"
                        result.warnings.append(msg)
                        log_fn(msg)
                    values = convert(np.asarray(raw, dtype=np.float64), meta.factor)
                    store[(var, depth, stat)] = list(values)
                    if stat in selection.written_statistics:
                        add_rows(
                            PRODUCT_SG2,
                            var,
                            meta.label,
                            depth,
                            stat,
                            values,
                            meta.units,
                            route,
                        )
                        result.layers.append(
                            LayerRecord(
                                file="",
                                band=0,
                                layer=name,
                                variable=var,
                                depth=depth,
                                statistic=stat,
                                units=meta.units,
                                factor=meta.factor,
                                route=route,
                                stats=summarise(values),
                            )
                        )
            if selection.ru90:
                for depth in selection.depths:
                    ru = relative_width(
                        store[(var, depth, "Q0.05")],
                        store[(var, depth, "Q0.5")],
                        store[(var, depth, "Q0.95")],
                    )
                    add_rows(
                        PRODUCT_SG2,
                        var,
                        RU90_LABEL,
                        depth,
                        RU90,
                        ru,
                        "ratio",
                        "derived",
                    )
                    result.uncertainty.append(
                        {"variable": var, "depth": depth, **_ru_summary(ru)}
                    )

        for depth in selection.depths:
            if all((code, depth, "mean") in store for code in TEXTURE_CODES):
                result.texture.append(
                    {
                        "depth": depth,
                        **texture_sum_check(
                            np.array(store[("sand", depth, "mean")]),
                            np.array(store[("silt", depth, "mean")]),
                            np.array(store[("clay", depth, "mean")]),
                            TEXTURE_SUM_TOLERANCE,
                        ),
                    }
                )

        if selection.sg2017:
            point_idx = sg2017_points_needed(selection.depths)
            for var in [v for v in selection.variables if v in VARS_2017]:
                meta = VARIABLE_BY_CODE[var]
                v17 = VARS_2017[var]

                def job17(k, var=var):
                    return sample(sg2017_path(var, k, base_2017), lonlats)

                def done17(k, var=var):
                    progress.step(f"Sampled {VARS_2017[var].code} sl{k + 1} (2017)")

                try:
                    raw_points = run_jobs(point_idx, job17, workers, cancel_fn, done17)
                except InterruptedError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    msg = (
                        f"SoilGrids 2017 {v17.code} could not be read - skipped: {exc}"
                    )
                    result.warnings.append(msg)
                    log_fn(msg)
                    continue
                points = {
                    k: convert(np.asarray(raw, dtype=np.float64), v17.divide_by)
                    for k, raw in raw_points.items()
                }
                for depth in selection.depths:
                    values = sg2017_interval(points, depth)
                    add_rows(
                        PRODUCT_SG2017,
                        var,
                        meta.label,
                        depth,
                        "mean",
                        values,
                        meta.units,
                        "2017 archive",
                    )
                    result.layers.append(
                        LayerRecord(
                            file="",
                            band=0,
                            layer=f"{v17.code}_M (trapezoid {depth})",
                            variable=var,
                            depth=depth,
                            statistic="mean",
                            units=meta.units,
                            factor=v17.divide_by,
                            route="2017 archive",
                            stats=summarise(values),
                            product=PRODUCT_SG2017,
                        )
                    )

        if selection.bedrock:

            def job_b(code):
                try:
                    return sample(bedrock_path(code, base_2017), lonlats)
                except Exception as exc:  # noqa: BLE001
                    raise SoilGridsError(
                        f"{code} (2017 archive) could not be read: {exc}"
                    ) from exc

            got = run_jobs(
                [b.code for b in BEDROCK_LAYERS],
                job_b,
                workers,
                cancel_fn,
                lambda code: progress.step(f"Sampled {code} (2017)"),
            )
            for layer in BEDROCK_LAYERS:
                values = np.asarray(got[layer.code], dtype=np.float64)
                add_rows(
                    PRODUCT_SG2017,
                    layer.code,
                    layer.label,
                    "n/a",
                    "mean",
                    values,
                    layer.units,
                    "2017 archive",
                )
                result.layers.append(
                    LayerRecord(
                        file="",
                        band=0,
                        layer=f"{layer.code}_M_250m_ll",
                        variable=layer.code,
                        depth="n/a",
                        statistic="mean",
                        units=layer.units,
                        factor=1.0,
                        route="2017 archive",
                        stats=summarise(values),
                        product=PRODUCT_SG2017,
                    )
                )

    result.seconds = (datetime.now(UTC) - t_start).total_seconds()
    progress_fn(1.0, "Done")
    return result


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


def build_metadata(
    result: RunResult,
    selection: Selection,
    settings: dict,
    gdal_version: str = "",
) -> list[tuple[str, list[str], list[list]]]:
    """Sectioned metadata: list of (section title, header, rows)."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    run_rows: list[list] = [
        ["Tool", f"Extract: SoilGrids 2.0 (mayim_tools) v{TOOL_VERSION}"],
        ["Run time", now],
        ["Duration (s)", round(result.seconds, 1)],
        ["Mode", result.mode],
        ["Product", "ISRIC SoilGrids 2.0, 250 m (path 'latest' - see access date)"],
        ["Access date", now[:10]],
        ["WebDAV base", BASE_URL],
        [
            "Access route",
            (
                "Tiles (one index per variable, from <variable>_0-5cm_mean): "
                + ", ".join(f"{v} {n}" for v, n in result.tiles.items())
                + "; fallback per layer: that layer's full VRT"
                if any(result.tiles.values())
                else "Full VRT per layer"
            ),
        ],
        ["GDAL version", gdal_version],
        ["Citation", CITATION],
        ["Licence", LICENCE],
    ]
    for key, value in settings.items():
        run_rows.append([key, value])
    run_rows += [
        ["Variables", ", ".join(selection.variables) or "(none)"],
        ["Depths", ", ".join(selection.depths)],
        ["Statistics written", ", ".join(selection.written_statistics)],
        ["RU90 (derived)", "yes" if selection.ru90 else "no"],
        ["SoilGrids 2017 means", "yes" if selection.sg2017 else "no"],
        ["Depth to bedrock (2017)", "yes" if selection.bedrock else "no"],
        ["Output nodata", NODATA_OUT],
        [
            "Note - units",
            "All values are in conventional units (ISRIC conversion factors "
            "applied; see Variables section). SOC is not converted to OM.",
        ],
        [
            "Note - quantiles",
            "Texture quantiles are marginal: Q0.05 sand + Q0.05 clay is not a "
            "soil. Sample and renormalise when propagating uncertainty.",
        ],
        [
            "Note - bands",
            "Select bands by their description (e.g. clay_30-60cm_Q0.5 (%)), "
            "never by band number.",
        ],
        [
            "Note - masked areas",
            "SoilGrids 2.0 has no predictions for urban, inland water, glacier "
            "and bare-surface areas (ESA CCI land cover 2015); those cells are "
            "nodata. The optional SoilGrids 2017 means cover urban and bare "
            "areas.",
        ],
    ]
    if selection.sg2017 or selection.bedrock:
        run_rows += [
            ["Archive (2017)", SG2017_URL],
            ["Citation (2017)", CITATION_SG2017],
            ["Licence (2017)", LICENCE_2017],
        ]
    if selection.sg2017:
        run_rows.append(
            [
                "Note - SoilGrids 2017",
                "Older product: mean predictions only (no quantiles). Its depths "
                "are points (0, 5, 15, 30, 60, 100, 200 cm); each interval is the "
                "average of its two bounding points (trapezoidal rule). Bands "
                "are tagged [SG2017]. Use to fill cells masked in 2.0, with a "
                "wider uncertainty than 2.0.",
            ]
        )
    if selection.bedrock:
        run_rows += [
            ["Citation (depth to bedrock)", CITATION_2017],
            [
                "Note - depth to bedrock",
                "SoilGrids 2017 product (not part of SoilGrids 2.0). Gives depth "
                "to bedrock only, not other restrictive layers (hardpans, "
                "plinthite, duplex B horizons).",
            ],
        ]
    if any(v in selection.variables for v in ("wv0010", "wv0033", "wv1500")):
        run_rows.append(
            [
                "Note - water content",
                "wv0010/wv0033/wv1500 conversion factor 10 (0.1 vol % -> vol %) "
                "per the soilDB reference implementation; confirm against "
                "ISRIC documentation before design use.",
            ]
        )

    var_rows = [
        [PRODUCT_SG2, v.code, v.label, v.mapped_units, v.factor, v.units, v.purpose]
        for v in VARIABLES
        if v.code in selection.variables
    ]
    if selection.sg2017:
        var_rows += [
            [
                PRODUCT_SG2017,
                f"{VARS_2017[v].code} -> {v}",
                VARIABLE_BY_CODE[v].label,
                "see Hengl et al. 2017",
                VARS_2017[v].divide_by,
                VARIABLE_BY_CODE[v].units,
                "fill for areas masked in 2.0",
            ]
            for v in selection.variables
            if v in VARS_2017
        ]
    if selection.bedrock:
        var_rows += [
            [
                PRODUCT_SG2017,
                b.code,
                b.label,
                b.units,
                1,
                b.units,
                "HSG restrictive depth",
            ]
            for b in BEDROCK_LAYERS
        ]

    layer_rows = [
        [
            rec.product,
            rec.file,
            rec.band,
            rec.layer,
            rec.variable,
            rec.depth,
            rec.statistic,
            rec.units,
            rec.route,
            rec.stats.get("valid"),
            rec.stats.get("nodata"),
            _r(rec.stats.get("min")),
            _r(rec.stats.get("median")),
            _r(rec.stats.get("max")),
        ]
        for rec in result.layers
    ]
    ru_rows = [
        [u["variable"], u["depth"], u["cells"], _r(u["median_ru90"]), _r(u["p90_ru90"])]
        for u in result.uncertainty
    ]
    tex_rows = [
        [t["depth"], t["checked"], t["flagged"], _r(t["max_abs_dev"])]
        for t in result.texture
    ]
    warn_rows = [[w] for w in result.warnings] or [["(none)"]]

    return [
        ("Run", ["Item", "Value"], run_rows),
        (
            "Variables",
            [
                "Product",
                "Code",
                "Description",
                "Mapped units",
                "Divided by",
                "Units",
                "Purpose",
            ],
            var_rows,
        ),
        (
            "Layers",
            [
                "Product",
                "File",
                "Band",
                "Layer",
                "Variable",
                "Depth",
                "Statistic",
                "Units",
                "Route",
                "Valid",
                "Nodata",
                "Min",
                "Median",
                "Max",
            ],
            layer_rows,
        ),
        (
            "Uncertainty (RU90 = (Q0.95-Q0.05)/Q0.5)",
            ["Variable", "Depth", "Cells/points", "Median RU90", "P90 RU90"],
            ru_rows,
        ),
        (
            f"Texture check (mean sand+silt+clay, flag if off 100 by > "
            f"{TEXTURE_SUM_TOLERANCE:g} %)",
            ["Depth", "Checked", "Flagged", "Max abs deviation (%)"],
            tex_rows,
        ),
        ("Warnings", ["Warning"], warn_rows),
    ]


def _r(value, nd: int = 4):
    if value is None:
        return ""
    try:
        if math.isnan(value):
            return ""
    except TypeError:
        return value
    return round(float(value), nd)


# ----------------------------------------------------------------------
# Default GDAL functions (the only part of this module that needs osgeo)
# ----------------------------------------------------------------------

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


def _index_windows(to_pixel, bounds_ll, points):
    """Fractional pixel windows (x0, y0, x1, y1) to look up tiles for:
    one per point, or one for the densified area bounding box."""
    windows = []
    if points:
        for lon, lat in points:
            px, py = to_pixel(lon, lat)
            windows.append((px, py, px, py))
        return windows
    lon0, lat0, lon1, lat1 = bounds_ll
    n = 20
    pix = [
        to_pixel(lon0 + (lon1 - lon0) * i / n, lat0 + (lat1 - lat0) * j / n)
        for i in range(n + 1)
        for j in range(n + 1)
    ]
    xs = [p[0] for p in pix]
    ys = [p[1] for p in pix]
    windows.append((min(xs), min(ys), max(xs), max(ys)))
    return windows


def gdal_tile_index(
    bounds_ll: tuple[float, float, float, float],
    points: Sequence[tuple[float, float]] | None = None,
    base: str = BASE_DIR,
    ref: tuple[str, str, str] = REF_LAYER,
    pad_cells: int = 3,
) -> TileIndex:
    """Open the reference VRT once and list the tiles covering the area
    (densified bounding box) or, for points, the cells around each point."""
    gdal = _gdal()
    ref_name = layer_name(*ref)
    ds = gdal.Open(layer_vrt(*ref, base=base))
    if ds is None:
        raise SoilGridsError("Could not open the reference VRT.")
    xml = ds.GetMetadata("xml:VRT")[0]
    srs_wkt = ds.GetProjection()
    inv = gdal.InvGeoTransform(ds.GetGeoTransform())
    cols, rows = ds.RasterXSize, ds.RasterYSize
    ds = None
    PROJ_LOCK.acquire()  # PROJ is not thread-safe for this projection
    try:
        tr = _lonlat_to(srs_wkt)

        def to_pixel(lon, lat):
            x, y, _ = tr.TransformPoint(lon, lat)
            return (
                inv[0] + inv[1] * x + inv[2] * y,
                inv[3] + inv[4] * x + inv[5] * y,
            )

        windows = _index_windows(to_pixel, bounds_ll, points)
    finally:
        PROJ_LOCK.release()
    rels: list[str] = []
    for wx0, wy0, wx1, wy1 in windows:
        window = (
            max(0, int(math.floor(wx0)) - pad_cells),
            max(0, int(math.floor(wy0)) - pad_cells),
            min(cols, int(math.floor(wx1)) + 1 + pad_cells),
            min(rows, int(math.floor(wy1)) + 1 + pad_cells),
        )
        if window[0] >= window[2] or window[1] >= window[3]:
            continue
        for rel in tiles_in_window(xml, ref_name, window):
            if rel not in rels:
                rels.append(rel)
    if not rels:
        raise SoilGridsError("No SoilGrids tile covers the area of interest.")
    return TileIndex(tuple(rels), ref_name)


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
    mem = f"/vsimem/soilgrids_{uuid.uuid4().hex}.vrt"
    ds = gdal.BuildVRT(mem, list(source))
    if ds is None:
        raise SoilGridsError("Could not open the SoilGrids tiles.")
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
            raise SoilGridsError("the tiles do not cover the area of interest")
        return None
    if strict and (
        min(xs) < 0
        or min(ys) < 0
        or max(xs) > ds.RasterXSize
        or max(ys) > ds.RasterYSize
    ):
        raise SoilGridsError("the tiles do not cover the area of interest")
    x0 = max(0, int(math.floor(min(xs))) - pad)
    y0 = max(0, int(math.floor(min(ys))) - pad)
    x1 = min(ds.RasterXSize, int(math.floor(max(xs))) + 1 + pad)
    y1 = min(ds.RasterYSize, int(math.floor(max(ys))) + 1 + pad)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1 - x0, y1 - y0


def gdal_read_grid(source, grid: TargetGrid) -> np.ndarray:
    """Warp a (remote) raster - one path or a list of tiles - onto ``grid``
    with nearest neighbour. Float64 with NaN for nodata / outside coverage.

    Thread-safe: the native block is downloaded without the lock; the
    window calculation and the in-memory warp run under PROJ_LOCK."""
    gdal = _gdal()
    src, mem = _open_source(gdal, source)
    try:
        with PROJ_LOCK:
            window = _native_window(gdal, src, grid, strict=mem is not None)
        out = np.full((grid.height, grid.width), np.nan)
        if window is None:
            return out
        x0, y0, w, h = window
        band = src.GetRasterBand(1)
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
                raise SoilGridsError(f"GDAL could not warp {source}")
            out = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
            ds = None
            mem_src = None
        return out
    finally:
        src = None
        _unlink(gdal, mem)


def gdal_sample_points(source, lonlats: Sequence[tuple[float, float]]) -> list[float]:
    """Nearest-cell values of a (remote) raster - one path or a list of
    tiles - at lon/lat points. One window read when the points fit in
    WINDOW_LIMIT_CELLS, otherwise one cell per point. NaN for nodata or
    points outside the raster."""
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
            raise SoilGridsError("the tiles do not cover every point")
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
        return values
    finally:
        ds = None
        _unlink(gdal, mem)
