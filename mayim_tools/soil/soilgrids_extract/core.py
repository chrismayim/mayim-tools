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
  tool stores them as published (one band per depth point); conversion
  to intervals is done by Regional soil parameterisation.

UNITS
SoilGrids stores integers. Every value written by this tool has
already been divided by ISRIC's conversion factor (``VARIABLES``), so
outputs are in conventional units (%, g/kg, g/cm3, vol %, pH,
cmol(c)/kg). SOC is NOT converted to organic matter here: the SOC to
OM factor is a method choice made by the hydraulic-property tools.

UNCERTAINTY
Each variable can be extracted as the mean and the 5 %, 50 % and 95 %
prediction quantiles (SoilGrids' own 90 % prediction interval). This tool
only acquires data: derived measures (e.g. relative interval widths) are
computed by Regional soil parameterisation, not here. ISRIC's own
"uncertainty" layer is not downloaded (it can be recomputed from the
quantiles). Texture quantiles are MARGINAL: Q0.05 sand + Q0.05 clay is not a real
soil. Uncertainty propagation must sample and renormalise the
fractions, not combine quantiles directly.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

# Generic building blocks live in soil/_common (shared with the other soil
# tools); they are re-exported here so existing callers and tests keep working.
from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.export import NODATA_OUT  # noqa: F401
from mayim_tools.soil._common.gdal_io import (  # noqa: F401
    HTTP_OPTIONS,
    PROJ_LOCK,
    WINDOW_LIMIT_CELLS,
    _crs_transform,
    _gdal,
    _lonlat_to,
    _native_window,
    _open_source,
    _unlink,
    _wgs84_wkt,
    gdal_read_grid,
    gdal_sample_points,
    gdal_version,
    http_options,
)
from mayim_tools.soil._common.grid import (  # noqa: F401
    BUFFER_CELLS,
    EARTH_RADIUS_KM,
    TargetGrid,
    area_km2,
    check_area,
    make_grid,
)
from mayim_tools.soil._common.jobs import (  # noqa: F401
    Source,
    _all_nan,
    _never_cancelled,
    _noop,
    _Progress,
    read_with_fallback,
    run_jobs,
)
from mayim_tools.soil._common.stats import (  # noqa: F401
    _r,
    convert,
    relative_width,
    summarise,
    texture_sum_check,
)

TOOL_VERSION = "0.4.0"

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

TEXTURE_SUM_TOLERANCE = 2.0  # % - flag cells where sand+silt+clay is further off
DEFAULT_MAX_AREA_KM2 = 50000.0


SoilGridsError = SoilDataError  # same class: callers catch either name


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
    statistics: list[str]
    written_statistics: list[str]
    bedrock: bool
    sg2017: bool = False
    warnings: list[str] = field(default_factory=list)


def plan_selection(
    variables: Sequence[str],
    depths: Sequence[str],
    statistics: Sequence[str],
    bedrock: bool = False,
    sg2017: bool = False,
) -> Selection:
    """Validate and order the user's selection."""
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
    if variables_ordered and not written:
        raise SoilGridsError("Select at least one statistic.")
    if not variables_ordered and not bedrock:
        raise SoilGridsError("Select at least one variable.")

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
        bedrock=bedrock,
        sg2017=sg2017,
        warnings=warnings,
    )


# ----------------------------------------------------------------------
# Area and grid
# ----------------------------------------------------------------------


def layer_vrt(var: str, depth: str, stat: str, base: str = BASE_DIR) -> str:
    """Full WebDAV VRT of one SoilGrids 2.0 layer (slow to open: ~17 s)."""
    return f"{base}{var}/{layer_name(var, depth, stat)}.vrt"


def bedrock_path(code: str, base: str = SG2017_DIR) -> str:
    return f"{base}{code}_M_250m_ll.tif"


def sg2017_path(var: str, point_index: int, base: str = SG2017_DIR) -> str:
    """2017 mean prediction at depth point sl<k> (k = 1..7)."""
    return f"{base}{VARS_2017[var].code}_M_sl{point_index + 1}_250m_ll.tif"


def point_label_2017(point_index: int) -> str:
    """'15cm' for sl3 - SoilGrids 2017 values are at depth POINTS."""
    return f"{DEPTH_POINTS_2017[point_index]}cm"


def sg2017_points_needed(depths: Sequence[str]) -> list[int]:
    """Depth-point indices (0..6) needed to average the selected intervals."""
    needed = set()
    for depth in depths:
        _, top, bottom = DEPTH_BY_LABEL[depth]
        needed.add(DEPTH_POINTS_2017.index(top))
        needed.add(DEPTH_POINTS_2017.index(bottom))
    return sorted(needed)


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


def layer_sources(
    index: TileIndex | None, var: str, depth: str, stat: str, base: str = BASE_DIR
) -> list[Source]:
    sources = []
    if index is not None:
        sources.append(Source("Tiles", tuple(index.paths_for(var, depth, stat, base))))
    sources.append(Source("VRT", layer_vrt(var, depth, stat, base)))
    return sources


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
    texture: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)  # point mode
    load_files: list[str] = field(default_factory=list)
    tiles: dict = field(default_factory=dict)  # variable -> tiles (0 = VRT)
    seconds: float = 0.0


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
    GeoTIFF per variable x statistic (one band per depth),
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
        for k in point_idx:
            arr = points[k]
            point = point_label_2017(k)
            bands.append((f"{var}_{point}_mean [SG2017] ({meta.units})", arr))
            result.layers.append(
                LayerRecord(
                    file=os.path.basename(path),
                    band=len(bands),
                    layer=f"{v17.code}_M_sl{k + 1} ({point})",
                    variable=var,
                    depth=point,
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
        elif depth.endswith("cm") and depth[:-2].isdigit():  # a depth point
            top = bottom = int(depth[:-2])
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
                for k in point_idx:
                    values = points[k]
                    point = point_label_2017(k)
                    add_rows(
                        PRODUCT_SG2017,
                        var,
                        meta.label,
                        point,
                        "mean",
                        values,
                        meta.units,
                        "2017 archive",
                    )
                    result.layers.append(
                        LayerRecord(
                            file="",
                            band=0,
                            layer=f"{v17.code}_M_sl{k + 1} ({point})",
                            variable=var,
                            depth=point,
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
    gdal_version_text: str = "",
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
        ["GDAL version", gdal_version_text],
        ["Citation", CITATION],
        ["Licence", LICENCE],
    ]
    for key, value in settings.items():
        run_rows.append([key, value])
    run_rows += [
        ["Variables", ", ".join(selection.variables) or "(none)"],
        ["Depths", ", ".join(selection.depths)],
        ["Statistics written", ", ".join(selection.written_statistics)],
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
                "Older product: mean predictions only (no quantiles). Stored at "
                "its native depth POINTS (0, 5, 15, 30, 60, 100, 200 cm; only "
                "the points bounding the selected intervals are read), one band "
                "per point, tagged [SG2017]. Conversion to intervals is done by "
                "Regional soil parameterisation, not here. Use to fill cells "
                "masked in 2.0, with a wider uncertainty than 2.0.",
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
            f"Texture check (mean sand+silt+clay, flag if off 100 by > "
            f"{TEXTURE_SUM_TOLERANCE:g} %)",
            ["Depth", "Checked", "Flagged", "Max abs deviation (%)"],
            tex_rows,
        ),
        ("Warnings", ["Warning"], warn_rows),
    ]


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
