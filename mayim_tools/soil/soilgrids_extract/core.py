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
- Primary route: the WebDAV VRTs over cloud-optimised GeoTIFF tiles,
  read with GDAL's /vsicurl/ so only the requested window is fetched:
      https://files.isric.org/soilgrids/latest/data/<var>/<var>_<depth>_<stat>.vrt
  Native CRS is Interrupted Goode Homolosine; the CRS is taken from
  the VRT itself, never hard-coded.
- Fallback route (per layer, automatic): ISRIC's WCS 2.0.1 at
  maps.isric.org, requested as a lon/lat subset served in EPSG:4326.
- The SoilGrids REST API is NOT used (paused by ISRIC; beta; 5 calls
  per minute).
- Optional: depth to bedrock from the SoilGrids 2017 archive
  (Shangguan et al. 2017), which SoilGrids 2.0 no longer provides.

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
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

TOOL_VERSION = "0.1.0"

BASE_URL = "https://files.isric.org/soilgrids/latest/data/"
SG2017_URL = "https://files.isric.org/soilgrids/former/2017-03-10/data/"
WCS_URL = "https://maps.isric.org/mapserv?map=/map/{var}.map"

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
LICENCE = "CC-BY 4.0 (ISRIC data policy: https://www.isric.org/about/data-policy)"

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


def stat_tag(stat: str) -> str:
    """File-name tag for a statistic: Q0.5 -> Q0.50 (sorts and reads cleanly)."""
    return {"Q0.05": "Q0.05", "Q0.5": "Q0.50", "Q0.95": "Q0.95"}.get(stat, stat)


def layer_name(var: str, depth: str, stat: str) -> str:
    """ISRIC layer / coverage id, e.g. 'clay_0-5cm_Q0.5'."""
    return f"{var}_{depth}_{stat}"


def band_description(var: str, depth: str, stat: str, units: str) -> str:
    """Band description used to SELECT bands downstream (never band number)."""
    return f"{var}_{depth}_{stat} ({units})"


def webdav_path(var: str, depth: str, stat: str) -> str:
    return f"/vsicurl/{BASE_URL}{var}/{layer_name(var, depth, stat)}.vrt"


def wcs_path(
    var: str, depth: str, stat: str, bounds_ll: tuple[float, float, float, float]
) -> str:
    """WCS 2.0.1 GetCoverage for a lon/lat subset, served in EPSG:4326."""
    lon0, lat0, lon1, lat1 = bounds_ll
    url = (
        WCS_URL.format(var=var) + "&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
        f"&COVERAGEID={layer_name(var, depth, stat)}"
        "&FORMAT=image/tiff"
        f"&SUBSET=long({lon0:.6f},{lon1:.6f})"
        f"&SUBSET=lat({lat0:.6f},{lat1:.6f})"
        "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
    )
    return f"/vsicurl/{url}"


def bedrock_path(code: str) -> str:
    return f"/vsicurl/{SG2017_URL}{code}_M_250m_ll.tif"


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
    warnings: list[str] = field(default_factory=list)


def plan_selection(
    variables: Sequence[str],
    depths: Sequence[str],
    statistics: Sequence[str],
    ru90: bool = False,
    bedrock: bool = False,
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

    return Selection(
        variables=variables_ordered,
        depths=depths_ordered,
        statistics=fetched,
        written_statistics=written,
        ru90=ru90,
        bedrock=bedrock,
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


def buffer_bounds_ll(
    bounds_ll: tuple[float, float, float, float], deg: float = 0.01
) -> tuple[float, float, float, float]:
    lon0, lat0, lon1, lat1 = bounds_ll
    return (
        max(-180.0, lon0 - deg),
        max(-90.0, lat0 - deg),
        min(180.0, lon1 + deg),
        min(90.0, lat1 + deg),
    )


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


@dataclass(frozen=True)
class Source:
    route: str  # "WebDAV" | "WCS" | "WebDAV (2017 archive)"
    path: str


def sources_for(
    var: str, depth: str, stat: str, bounds_ll: tuple[float, float, float, float]
) -> list[Source]:
    return [
        Source("WebDAV", webdav_path(var, depth, stat)),
        Source("WCS (EPSG:4326)", wcs_path(var, depth, stat, bounds_ll)),
    ]


def read_with_fallback(sources: Sequence[Source], reader: Callable, *args):
    """Try each source in turn. Returns (result, route, errors).
    Raises SoilGridsError listing every attempt if all fail."""
    errors: list[str] = []
    for src in sources:
        try:
            return reader(src.path, *args), src.route, errors
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            errors.append(f"{src.route}: {exc}")
    raise SoilGridsError("All access routes failed - " + " | ".join(errors))


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


def _noop(*_args, **_kwargs):
    return None


def _never_cancelled() -> bool:
    return False


# ----------------------------------------------------------------------
# Area mode
# ----------------------------------------------------------------------


def output_file_name(var: str, stat: str) -> str:
    return f"soilgrids_{var}_{stat_tag(stat)}.tif"


BEDROCK_FILE_NAME = "soilgrids2017_depth_to_bedrock.tif"


def extract_area(
    selection: Selection,
    grid: TargetGrid,
    bounds_ll: tuple[float, float, float, float],
    out_dir: str,
    *,
    read_grid_fn: Callable | None = None,
    write_fn: Callable | None = None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    """Fetch every selected layer onto ``grid`` and write one multi-band
    GeoTIFF per variable x statistic (one band per depth), plus RU90 files
    and the optional depth-to-bedrock file.

    ``write_fn(path, grid, bands, units)`` where bands is a list of
    (description, array) - see export.write_multiband_geotiff."""
    read_grid_fn = read_grid_fn or gdal_read_grid
    if write_fn is None:
        raise SoilGridsError("No raster writer supplied.")
    result = RunResult(mode="area", warnings=list(selection.warnings))
    wcs_bounds = buffer_bounds_ll(bounds_ll)

    n_steps = len(selection.variables) * len(selection.statistics) * len(
        selection.depths
    ) + (len(BEDROCK_LAYERS) if selection.bedrock else 0)
    done = 0
    texture_means: dict[str, dict[str, np.ndarray]] = {}

    for var in selection.variables:
        meta = VARIABLE_BY_CODE[var]
        quantiles: dict[str, dict[str, np.ndarray]] = {}
        for stat in selection.statistics:
            bands: list[tuple[str, np.ndarray]] = []
            pending: list[LayerRecord] = []
            path = os.path.join(out_dir, output_file_name(var, stat))
            for depth in selection.depths:
                if cancel_fn():
                    raise InterruptedError("Cancelled by user")
                name = layer_name(var, depth, stat)
                progress_fn(done / n_steps, f"Reading {name}")
                raw, route, errors = read_with_fallback(
                    sources_for(var, depth, stat, wcs_bounds), read_grid_fn, grid
                )
                for err in errors:
                    msg = f"{name}: primary route failed, used {route} ({err})"
                    result.warnings.append(msg)
                    log_fn(msg)
                arr = convert(raw, meta.factor)
                bands.append((band_description(var, depth, stat, meta.units), arr))
                pending.append(
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
                done += 1
            if stat in selection.written_statistics:
                write_fn(path, grid, bands, meta.units)
                result.files.append(path)
                result.layers.extend(pending)
                if stat == "Q0.5" or (
                    stat == "mean" and "Q0.5" not in selection.written_statistics
                ):
                    result.load_files.append(path)

        if selection.ru90:
            ru_bands = []
            path = os.path.join(out_dir, output_file_name(var, RU90))
            for depth in selection.depths:
                q = quantiles[depth]
                ru = relative_width(q["Q0.05"], q["Q0.5"], q["Q0.95"])
                ru_bands.append((band_description(var, depth, RU90, "ratio"), ru))
                result.uncertainty.append(
                    {
                        "variable": var,
                        "depth": depth,
                        **_ru_summary(ru),
                    }
                )
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

    if selection.bedrock:
        bands = []
        path = os.path.join(out_dir, BEDROCK_FILE_NAME)
        records = []
        for layer in BEDROCK_LAYERS:
            if cancel_fn():
                raise InterruptedError("Cancelled by user")
            progress_fn(done / n_steps, f"Reading {layer.code} (2017 archive)")
            raw, route, _ = read_with_fallback(
                [Source("WebDAV (2017 archive)", bedrock_path(layer.code))],
                read_grid_fn,
                grid,
            )
            arr = np.asarray(raw, dtype=np.float64)
            bands.append((f"{layer.code} {layer.label} ({layer.units})", arr))
            records.append(
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
                )
            )
            done += 1
        write_fn(path, grid, bands, "cm / %")
        result.files.append(path)
        result.layers.extend(records)
        result.load_files.append(path)

    progress_fn(1.0, "Done")
    return result


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


def point_groups(sites: Sequence[Site], span_deg: float = 0.5) -> list[list[int]]:
    """Group site indices for the WCS fallback: one request if all sites
    fit in ``span_deg``, otherwise one request per site."""
    if not sites:
        return []
    lons = [s.lon for s in sites]
    lats = [s.lat for s in sites]
    if max(lons) - min(lons) <= span_deg and max(lats) - min(lats) <= span_deg:
        return [list(range(len(sites)))]
    return [[i] for i in range(len(sites))]


def _sample_layer(
    var: str,
    depth: str,
    stat: str,
    sites: Sequence[Site],
    sample_fn: Callable,
) -> tuple[list[float], str, list[str]]:
    """Sample one layer at every site: WebDAV for all sites at once, then
    WCS per site group for anything WebDAV could not provide."""
    lonlats = [(s.lon, s.lat) for s in sites]
    errors: list[str] = []
    try:
        values = sample_fn(webdav_path(var, depth, stat), lonlats)
        return list(values), "WebDAV", errors
    except Exception as exc:  # noqa: BLE001
        errors.append(f"WebDAV: {exc}")
    values = [math.nan] * len(sites)
    for group in point_groups(sites):
        sub = [sites[i] for i in group]
        bounds = buffer_bounds_ll(
            (
                min(s.lon for s in sub),
                min(s.lat for s in sub),
                max(s.lon for s in sub),
                max(s.lat for s in sub),
            )
        )
        try:
            got = sample_fn(
                wcs_path(var, depth, stat, bounds), [(s.lon, s.lat) for s in sub]
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"WCS: {exc}")
            raise SoilGridsError(
                f"{layer_name(var, depth, stat)}: all access routes failed - "
                + " | ".join(errors)
            ) from exc
        for i, v in zip(group, got, strict=True):
            values[i] = v
    return values, "WCS (EPSG:4326)", errors


def extract_points(
    selection: Selection,
    sites: Sequence[Site],
    *,
    sample_fn: Callable | None = None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    """Sample every selected layer at every site. Rows are long format:
    Site, Longitude, Latitude, Variable, Description, DepthTop_cm,
    DepthBottom_cm, Statistic, Value, Units, Route."""
    sample_fn = sample_fn or gdal_sample_points
    if not sites:
        raise SoilGridsError("No points to sample.")
    result = RunResult(mode="points", warnings=list(selection.warnings))
    n_steps = len(selection.variables) * len(selection.statistics) * len(
        selection.depths
    ) + (len(BEDROCK_LAYERS) if selection.bedrock else 0)
    done = 0
    store: dict[tuple[str, str, str], list[float]] = {}

    def add_rows(var, label, depth, stat, values, units, route):
        top, bottom = ("", "")
        if depth in DEPTH_BY_LABEL:
            _, top, bottom = DEPTH_BY_LABEL[depth]
        for site, value in zip(sites, values, strict=True):
            result.rows.append(
                {
                    "Site": site.label,
                    "Longitude": site.lon,
                    "Latitude": site.lat,
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

    for var in selection.variables:
        meta = VARIABLE_BY_CODE[var]
        for stat in selection.statistics:
            for depth in selection.depths:
                if cancel_fn():
                    raise InterruptedError("Cancelled by user")
                name = layer_name(var, depth, stat)
                progress_fn(done / n_steps, f"Sampling {name}")
                raw, route, errors = _sample_layer(var, depth, stat, sites, sample_fn)
                for err in errors:
                    msg = f"{name}: primary route failed, used {route} ({err})"
                    result.warnings.append(msg)
                    log_fn(msg)
                values = convert(np.asarray(raw, dtype=np.float64), meta.factor)
                store[(var, depth, stat)] = list(values)
                if stat in selection.written_statistics:
                    add_rows(var, meta.label, depth, stat, values, meta.units, route)
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
                done += 1
        if selection.ru90:
            for depth in selection.depths:
                ru = relative_width(
                    store[(var, depth, "Q0.05")],
                    store[(var, depth, "Q0.5")],
                    store[(var, depth, "Q0.95")],
                )
                add_rows(var, RU90_LABEL, depth, RU90, ru, "ratio", "derived")
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

    if selection.bedrock:
        lonlats = [(s.lon, s.lat) for s in sites]
        for layer in BEDROCK_LAYERS:
            if cancel_fn():
                raise InterruptedError("Cancelled by user")
            progress_fn(done / n_steps, f"Sampling {layer.code} (2017 archive)")
            try:
                values = sample_fn(bedrock_path(layer.code), lonlats)
            except Exception as exc:  # noqa: BLE001
                raise SoilGridsError(
                    f"{layer.code} (2017 archive) could not be read: {exc}"
                ) from exc
            values = np.asarray(values, dtype=np.float64)
            add_rows(
                layer.code,
                layer.label,
                "n/a",
                "mean",
                values,
                layer.units,
                "WebDAV (2017 archive)",
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
                    route="WebDAV (2017 archive)",
                    stats=summarise(values),
                )
            )
            done += 1

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
        ["Mode", result.mode],
        ["Product", "ISRIC SoilGrids 2.0, 250 m (path 'latest' - see access date)"],
        ["Access date", now[:10]],
        ["WebDAV base", BASE_URL],
        ["WCS fallback", WCS_URL.format(var="<variable>")],
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
    ]
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
        [v.code, v.label, v.mapped_units, v.factor, v.units, v.purpose]
        for v in VARIABLES
        if v.code in selection.variables
    ]
    if selection.bedrock:
        var_rows += [
            [b.code, b.label, b.units, 1, b.units, "HSG restrictive depth"]
            for b in BEDROCK_LAYERS
        ]

    layer_rows = [
        [
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
        [
            u["variable"],
            u["depth"],
            u["cells"],
            _r(u["median_ru90"]),
            _r(u["p90_ru90"]),
        ]
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
            ["Code", "Description", "Mapped units", "Divided by", "Units", "Purpose"],
            var_rows,
        ),
        (
            "Layers",
            [
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
# Default GDAL readers (only part of this module that needs osgeo)
# ----------------------------------------------------------------------

HTTP_OPTIONS = {
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "3",
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "VSI_CACHE": "TRUE",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
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
    """Set GDAL HTTP options for the duration of a read, then restore."""
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


def gdal_read_grid(path: str, grid: TargetGrid) -> np.ndarray:
    """Warp a (remote) raster onto ``grid`` with nearest neighbour.
    Returns float64 with NaN for nodata and for cells outside coverage."""
    gdal = _gdal()
    with http_options():
        ds = gdal.Warp(
            "",
            path,
            format="MEM",
            outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
            width=grid.width,
            height=grid.height,
            dstSRS=grid.crs_wkt,
            resampleAlg="near",
            outputType=gdal.GDT_Float32,
            dstNodata=float("nan"),
            multithread=True,
        )
    if ds is None:
        raise SoilGridsError(f"GDAL could not read {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None and not math.isnan(nodata):
        arr[arr == nodata] = np.nan
    ds = None
    return arr


def gdal_sample_points(
    path: str, lonlats: Sequence[tuple[float, float]]
) -> list[float]:
    """Nearest-cell values of a (remote) raster at lon/lat points.
    Reads one window when the points fit in WINDOW_LIMIT_CELLS, otherwise
    one cell per point. NaN for nodata or points outside the raster."""
    gdal = _gdal()
    from osgeo import osr

    osr.UseExceptions()

    with http_options():
        ds = gdal.Open(path)
        if ds is None:
            raise SoilGridsError(f"GDAL could not open {path}")
        src = osr.SpatialReference()
        src.ImportFromEPSG(4326)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst = osr.SpatialReference()
        dst.ImportFromWkt(ds.GetProjection())
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        tr = osr.CoordinateTransformation(src, dst)
        gt = ds.GetGeoTransform()
        inv = gdal.InvGeoTransform(gt)
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        cols, rows = ds.RasterXSize, ds.RasterYSize

        pix = []
        for lon, lat in lonlats:
            x, y, _ = tr.TransformPoint(lon, lat)
            px = int(math.floor(inv[0] + inv[1] * x + inv[2] * y))
            py = int(math.floor(inv[3] + inv[4] * x + inv[5] * y))
            pix.append((px, py))
        inside = [(px, py) for px, py in pix if 0 <= px < cols and 0 <= py < rows]
        values: list[float] = []
        window = None
        if inside:
            x0 = min(p[0] for p in inside)
            x1 = max(p[0] for p in inside)
            y0 = min(p[1] for p in inside)
            y1 = max(p[1] for p in inside)
            if (x1 - x0 + 1) <= WINDOW_LIMIT_CELLS and (
                y1 - y0 + 1
            ) <= WINDOW_LIMIT_CELLS:
                window = (
                    x0,
                    y0,
                    band.ReadAsArray(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                )
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
        ds = None
    return values
