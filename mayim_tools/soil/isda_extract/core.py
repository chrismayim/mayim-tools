"""
Core logic for "Extract: iSDAsoil (Africa)" - no QGIS dependency.

Reads the iSDAsoil continent-wide cloud-optimised GeoTIFFs window by window
with QGIS's own GDAL (/vsicurl/, with an unsigned /vsis3/ fallback), several
layers in parallel, using the shared soil building blocks in
mayim_tools.soil._common. Network access is injectable, so the
orchestration is unit-tested with fakes. See catalogue.py for the data
facts (depths, band order, stored transforms).

This tool only acquires data: means are back-transformed to physical units,
standard deviations are written as published (for log-stored properties as
a standard deviation of ln(1 + x)). Uncertainty propagation, depth
harmonisation and all derived quantities belong to Regional soil
parameterisation.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.export import NODATA_OUT
from mayim_tools.soil._common.gdal_io import (
    HTTP_OPTIONS,
    gdal_read_grid,
    gdal_sample_points,
    http_options,
)
from mayim_tools.soil._common.grid import TargetGrid
from mayim_tools.soil._common.jobs import (
    Source,
    _never_cancelled,
    _noop,
    _Progress,
    read_with_fallback,
    run_jobs,
)
from mayim_tools.soil._common.stats import _r, summarise, texture_sum_check

from .catalogue import (
    CITATION,
    CITATION_2,
    HOST,
    LICENCE,
    PERIOD,
    PRODUCT,
    RESOLUTION_M,
    STATISTICS,
    TEXTURE_CODES,
    VARIABLE_BY_CODE,
    VARIABLES,
    band_description,
    band_from_descriptions,
    cog_url,
    default_band,
    depths_of,
    layer_label,
    output_file_name,
    s3_path,
    to_value,
    units_of,
)

TOOL_VERSION = "0.1.0"
DEFAULT_WORKERS = 8
DEFAULT_MAX_AREA_KM2 = 5000.0  # 30 m data
TEXTURE_SUM_TOLERANCE = 2.0
ISDA_OPTIONS = {**HTTP_OPTIONS, "AWS_NO_SIGN_REQUEST": "YES"}

SoilIsdaError = SoilDataError


# ----------------------------------------------------------------------
# Selection and jobs
# ----------------------------------------------------------------------


@dataclass
class Selection:
    variables: list[str]
    depths: list[str]
    statistics: list[str]
    warnings: list[str] = field(default_factory=list)


def plan_selection(variables, depths, statistics) -> Selection:
    if not variables:
        raise SoilDataError("Select at least one variable.")
    unknown = [v for v in variables if v not in VARIABLE_BY_CODE]
    if unknown:
        raise SoilDataError(f"Unknown variable(s): {', '.join(unknown)}")
    stats = [s for s, _ in STATISTICS if s in statistics]
    if not stats:
        raise SoilDataError("Select at least one statistic (mean and/or sd).")
    order = [v.code for v in VARIABLES]
    variables = sorted(set(variables), key=order.index)
    only_bedrock = variables == ["bedrock"]
    if not depths and not only_bedrock:
        raise SoilDataError("Select at least one depth interval.")
    sel = Selection(variables, list(depths), stats)
    if "ecec" in variables:
        sel.warnings.append(
            "ecec is the EFFECTIVE cation exchange capacity (at soil pH), not the "
            "CEC at pH 7 that some pedotransfer functions (Toth et al. 2015) use. "
            "Regional soil parameterisation does not use it."
        )
    if "sd" not in stats:
        sel.warnings.append(
            "Standard deviations not selected: Regional soil parameterisation will "
            "have no input uncertainty for iSDAsoil."
        )
    return sel


@dataclass(frozen=True)
class Job:
    var: str
    stat: str
    depth: str


def plan_jobs(selection: Selection) -> list[Job]:
    jobs = []
    for var in selection.variables:
        for depth, _, _ in depths_of(var):
            if var != "bedrock" and depth not in selection.depths:
                continue
            for stat in selection.statistics:
                jobs.append(Job(var, stat, depth))
    return jobs


def layer_sources(var: str) -> list[Source]:
    prop = VARIABLE_BY_CODE[var].prop
    return [
        Source("COG (https)", "/vsicurl/" + cog_url(prop)),
        Source("S3 (unsigned)", s3_path(prop)),
    ]


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


def resolve_bands(var, describe_fn) -> tuple[dict, str]:
    """{(stat, depth): band} for a property, from the file's band
    descriptions when they identify every band, otherwise the published
    order. Returns the mapping and a note on how it was found."""
    descs = None
    for src in layer_sources(var):
        try:
            descs = describe_fn(src.path)
            break
        except Exception:  # noqa: BLE001 - fall back to the published order
            continue
    keys = [(s, d[0]) for s, _ in STATISTICS for d in depths_of(var)]
    if descs:
        found = {k: band_from_descriptions(descs, *k) for k in keys}
        if all(found.values()) and len(set(found.values())) == len(found):
            return found, "band descriptions in the file"
    return (
        {k: default_band(var, *k) for k in keys},
        "published band order (mean bands first)",
    )


# ----------------------------------------------------------------------
# Results
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
    source_band: int
    route: str
    stats: dict


@dataclass
class RunResult:
    mode: str
    files: list[str] = field(default_factory=list)
    layers: list[LayerRecord] = field(default_factory=list)
    texture: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    band_notes: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    load_files: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _options(options_ctx, default_used: bool):
    if options_ctx is not None:
        return options_ctx
    return http_options(ISDA_OPTIONS) if default_used else nullcontext()


def _log(result: RunResult, log_fn: Callable, msg: str) -> None:
    result.warnings.append(msg)
    log_fn(msg)


def _read(jobs, reader, arg, bands, workers, cancel_fn, progress, verb):
    def job_fn(job):
        band = bands[job.var][(job.stat, job.depth)]

        def one(path, a):
            return reader(path, a, False, band)

        values, route, errors = read_with_fallback(layer_sources(job.var), one, arg)
        return values, route, errors, band

    def done(job):
        progress.step(f"{verb} {layer_label(job.var, job.depth, job.stat)}")

    return run_jobs(jobs, job_fn, workers, cancel_fn, done)


def _convert(job, got, result, log_fn):
    raw, route, errors, band = got[job]
    var = VARIABLE_BY_CODE[job.var]
    for err in errors:
        _log(
            result,
            log_fn,
            f"{layer_label(job.var, job.depth, job.stat)}: primary route failed, "
            f"used {route} ({err})",
        )
    return to_value(raw, var, job.stat), route, band


def _prepare(selection, describe_fn, result, log_fn):
    jobs = plan_jobs(selection)
    if not jobs:
        raise SoilDataError("Nothing to read.")
    bands = {}
    for var in selection.variables:
        bands[var], note = resolve_bands(var, describe_fn)
        result.band_notes[var] = note
        if note.startswith("published"):
            log_fn(f"{var}: bands taken in the {note}.")
    return jobs, bands


# ----------------------------------------------------------------------
# Area mode
# ----------------------------------------------------------------------


def extract_area(
    selection: Selection,
    grid: TargetGrid,
    out_dir: str,
    *,
    read_grid_fn: Callable | None = None,
    describe_fn: Callable | None = None,
    write_fn: Callable | None = None,
    workers: int = DEFAULT_WORKERS,
    options_ctx=None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    """One multi-band GeoTIFF per variable and statistic (one band per
    depth). ``write_fn(path, grid, bands, units)``."""
    if write_fn is None:
        raise SoilDataError("No raster writer supplied.")
    read = read_grid_fn or gdal_read_grid
    result = RunResult(mode="area", warnings=list(selection.warnings))
    t0 = datetime.now(UTC)
    means: dict[str, dict[str, np.ndarray]] = {}
    with _options(options_ctx, read_grid_fn is None):
        jobs, bands = _prepare(
            selection, describe_fn or band_descriptions, result, log_fn
        )
        progress = _Progress(len(jobs), progress_fn)
        for var in selection.variables:
            meta = VARIABLE_BY_CODE[var]
            vjobs = [j for j in jobs if j.var == var]
            got = _read(vjobs, read, grid, bands, workers, cancel_fn, progress, "Read")
            for stat in selection.statistics:
                sjobs = [j for j in vjobs if j.stat == stat]
                if not sjobs:
                    continue
                units = units_of(meta, stat)
                path = os.path.join(out_dir, output_file_name(var, stat))
                out_bands, records = [], []
                for job in sjobs:
                    values, route, src_band = _convert(job, got, result, log_fn)
                    if stat == "mean" and var in TEXTURE_CODES:
                        means.setdefault(job.depth, {})[var] = values
                    out_bands.append(
                        (band_description(var, job.depth, stat, units), values)
                    )
                    records.append(
                        LayerRecord(
                            file=os.path.basename(path),
                            band=len(out_bands),
                            layer=layer_label(var, job.depth, stat),
                            variable=var,
                            depth=job.depth,
                            statistic=stat,
                            units=units,
                            source_band=src_band,
                            route=route,
                            stats=summarise(values),
                        )
                    )
                write_fn(path, grid, out_bands, units)
                result.files.append(path)
                result.layers.extend(records)
                if stat == "mean" and var in ("clay", "bedrock"):
                    result.load_files.append(path)
                for rec in records:
                    if rec.stats.get("valid") == 0:
                        _log(
                            result,
                            log_fn,
                            f"{rec.layer} has no valid values in the area of "
                            "interest (outside Africa, water, or masked).",
                        )
            del got
    _texture_checks(result, means)
    result.seconds = (datetime.now(UTC) - t0).total_seconds()
    progress_fn(1.0, "Done")
    return result


def _texture_checks(result, means):
    for depth, m in sorted(means.items()):
        if all(c in m for c in TEXTURE_CODES):
            result.texture.append(
                {
                    "depth": depth,
                    **texture_sum_check(
                        m["sand"], m["silt"], m["clay"], TEXTURE_SUM_TOLERANCE
                    ),
                }
            )


# ----------------------------------------------------------------------
# Points mode
# ----------------------------------------------------------------------


@dataclass
class Site:
    label: str
    lon: float
    lat: float


POINT_COLUMNS = [
    "Site",
    "Longitude",
    "Latitude",
    "Product",
    "Variable",
    "Description",
    "DepthTop_cm",
    "DepthBottom_cm",
    "Period",
    "Statistic",
    "Resolution_m",
    "Value",
    "Units",
    "Route",
]


def extract_points(
    selection: Selection,
    sites: Sequence[Site],
    *,
    sample_fn: Callable | None = None,
    describe_fn: Callable | None = None,
    workers: int = DEFAULT_WORKERS,
    options_ctx=None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    if not sites:
        raise SoilDataError("No points to sample.")
    sample = sample_fn or gdal_sample_points
    result = RunResult(mode="points", warnings=list(selection.warnings))
    lonlats = [(s.lon, s.lat) for s in sites]
    t0 = datetime.now(UTC)
    means: dict[str, dict[str, np.ndarray]] = {}
    with _options(options_ctx, sample_fn is None):
        jobs, bands = _prepare(
            selection, describe_fn or band_descriptions, result, log_fn
        )
        progress = _Progress(len(jobs), progress_fn)
        for var in selection.variables:
            meta = VARIABLE_BY_CODE[var]
            vjobs = [j for j in jobs if j.var == var]
            got = _read(
                vjobs, sample, lonlats, bands, workers, cancel_fn, progress, "Sampled"
            )
            for job in vjobs:
                values, route, src_band = _convert(job, got, result, log_fn)
                if job.stat == "mean" and var in TEXTURE_CODES:
                    means.setdefault(job.depth, {})[var] = values
                top, bottom = job.depth[:-2].split("-")
                units = units_of(meta, job.stat)
                for site, v in zip(sites, values, strict=True):
                    result.rows.append(
                        {
                            "Site": site.label,
                            "Longitude": site.lon,
                            "Latitude": site.lat,
                            "Product": PRODUCT,
                            "Variable": var,
                            "Description": meta.label,
                            "DepthTop_cm": int(top),
                            "DepthBottom_cm": int(bottom),
                            "Period": PERIOD,
                            "Statistic": dict(STATISTICS)[job.stat],
                            "Resolution_m": RESOLUTION_M,
                            "Value": float(v),
                            "Units": units,
                            "Route": route,
                        }
                    )
                result.layers.append(
                    LayerRecord(
                        file="",
                        band=0,
                        layer=layer_label(var, job.depth, job.stat),
                        variable=var,
                        depth=job.depth,
                        statistic=job.stat,
                        units=units,
                        source_band=src_band,
                        route=route,
                        stats=summarise(np.asarray(values, dtype=float)),
                    )
                )
    _texture_checks(
        result, {d: {k: np.asarray(v) for k, v in m.items()} for d, m in means.items()}
    )
    result.seconds = (datetime.now(UTC) - t0).total_seconds()
    progress_fn(1.0, "Done")
    return result


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


def build_metadata(result, selection, settings, gdal_version_text=""):
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    run_rows = [
        ["Tool", f"Extract: iSDAsoil (Africa) (mayim_tools) v{TOOL_VERSION}"],
        ["Run time", now],
        ["Duration (s)", round(result.seconds, 1)],
        ["Mode", result.mode],
        ["Product", f"{PRODUCT} (30 m; mean and standard deviation; {PERIOD})"],
        ["Access date", now[:10]],
        ["Server", f"https://{HOST}/soil_data/ (unsigned S3 fallback)"],
        ["GDAL version", gdal_version_text],
        ["Citation", CITATION],
        ["Citation (product description)", CITATION_2],
        ["Licence", LICENCE],
    ]
    for key, value in settings.items():
        run_rows.append([key, value])
    run_rows += [
        ["Variables", ", ".join(selection.variables)],
        ["Depths", ", ".join(selection.depths)],
        ["Statistics", ", ".join(dict(STATISTICS)[s] for s in selection.statistics)],
        ["Output nodata", NODATA_OUT],
        [
            "Note - texture limits",
            "iSDAsoil texture follows USDA limits (silt 0.002-0.05 mm).",
        ],
        [
            "Note - depths",
            "iSDAsoil is mapped for 0-20 and 20-50 cm only (depth to bedrock "
            "0-200 cm; 200 means 200 cm or deeper). Regional soil "
            "parameterisation uses it for 0-30 and 30-60 cm, not 60-100 cm.",
        ],
        [
            "Note - back-transformation",
            "Means are converted to physical units: exp(x/10) - 1 for organic "
            "carbon, stone content and effective CEC (stored as 10 x ln(1 + x)); "
            "x/100 for bulk density; x/10 for pH; texture and depth to bedrock "
            "as stored.",
        ],
        [
            "Note - standard deviation",
            "The standard deviation is the spread of the ensemble's learners "
            "(bootstrapped cross-validation), not a calibrated prediction "
            "interval, and is likely narrower than the true error. For organic "
            "carbon, stone content and effective CEC it is a standard deviation "
            "of ln(1 + x) (units ln(1+...)).",
        ],
        [
            "Note - bands",
            "Select bands by their description (e.g. clay_20-50cm_mean_30m_isda "
            "(%)), never by band number.",
        ],
    ]
    var_rows = [
        [
            v.code,
            v.prop,
            v.label,
            v.units,
            v.transform,
            result.band_notes.get(v.code, ""),
            v.purpose,
        ]
        for v in VARIABLES
        if v.code in selection.variables
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
            rec.source_band,
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
    sections = [
        ("Run", ["Item", "Value"], run_rows),
        (
            "Variables",
            [
                "Code",
                "iSDA property",
                "Description",
                "Units",
                "Stored as",
                "Bands found from",
                "Purpose",
            ],
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
                "Source band",
                "Route",
                "Valid",
                "Nodata",
                "Min",
                "Median",
                "Max",
            ],
            layer_rows,
        ),
    ]
    if tex_rows:
        sections.append(
            (
                "Texture check (mean sand + silt + clay)",
                ["Depth", "Checked", "More than 2 % from 100 %", "Max deviation (%)"],
                tex_rows,
            )
        )
    sections.append(
        ("Warnings", ["Warning"], [[w] for w in result.warnings] or [["none"]])
    )
    return sections
