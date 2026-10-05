"""
Core logic for "Extract: OpenLandMap Soils" - no QGIS dependency.

Reads OpenLandMap-soildb (Hengl et al. 2026, ESSD 18:989) cloud-optimised
GeoTIFFs window-by-window with QGIS's own GDAL (/vsicurl/), several
layers in parallel, using the shared soil building blocks in
mayim_tools.soil._common (grid, parallel jobs, thread-safe reading,
writers). Everything that touches the network is injectable, so the
orchestration is unit-tested with fakes.

See catalogue.py for the data facts (30 m means, 120 m P16/P84, periods,
ISO 11277 texture limits, stored scale factors).

SCALE FACTORS
Stored values are integers scaled by a factor that is written both in the
official catalogue and in each file's metadata (GDAL band scale). The tool
applies the file's scale/offset when present; if the file has none, the
catalogue scale is applied. A disagreement between the two is warned.

UNCERTAINTY
P16 and P84 bound the 68 % prediction interval (about +/- one standard
deviation), published at 120 m only. This tool only acquires data: any
derived measure (relative interval widths, available water capacity,
depth harmonisation) is computed by Regional soil parameterisation. Like
any marginal quantiles, P16 sand with P16 clay is not a real soil: sample
and renormalise to propagate.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.export import NODATA_OUT
from mayim_tools.soil._common.gdal_io import (
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
from mayim_tools.soil._common.stats import (
    _r,
    summarise,
    texture_sum_check,
)

from .catalogue import (
    DEPTH_BY_LABEL,
    DEPTHS,
    LATEST_PERIOD,
    LEGACY_BY_CODE,
    LEGACY_CITATION,
    LEGACY_DOI,
    LEGACY_LICENCE,
    LEGACY_PERIOD,
    LEGACY_VARIABLES,
    PERIODS,
    PRODUCT,
    STAT_BY_CODE,
    STATISTICS,
    SUBGROUP_PERIOD,
    TEXTURE_CODES,
    VARIABLE_BY_CODE,
    VARIABLES,
    band_description,
    cog_url,
    great_group,
    layer_label,
    legacy_points_needed,
    legacy_urls,
    output_file_name,
    resolve_period,
    soil_order,
    subgroup_label,
    subgroup_url,
)
from .soil_types import SUBGROUP_NAMES

TOOL_VERSION = "0.4.0"
DEFAULT_WORKERS = 8
DEFAULT_MAX_AREA_KM2 = 5000.0  # 30 m data: 5000 km2 is ~5.6 million cells
TEXTURE_SUM_TOLERANCE = 2.0
SCALE_TOLERANCE = 1e-6

CITATION = (
    "Hengl, T., Consoli, D., Tian, X., Nauman, T. W., Nussbaum, M., Isik, M. S., "
    "Parente, L., Ho, Y.-F., Simoes, R., Gupta, S., Samuel-Rosa, A., Zborowski "
    "Horst, T., Safanelli, J. L. and Harris, N. (2026). OpenLandMap-soildb: "
    "global soil information at 30 m spatial resolution for 2000-2022+ based on "
    "spatiotemporal machine learning and harmonized legacy soil samples and "
    "observations. Earth System Science Data, 18, 989. "
    "https://doi.org/10.5194/essd-18-989-2026"
)
DATA_DOI = "https://doi.org/10.5281/zenodo.15470431"
CATALOGUE_URL = (
    "https://github.com/openlandmap/soildb/blob/main/tables/"
    "OpenLandMap_soildb_COGS.csv"
)
LICENCE = "CC-BY 4.0"

SoilOlmError = SoilDataError  # same class: callers catch either name


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


@dataclass
class Selection:
    variables: list[str]
    depths: list[str]
    periods: list[str]
    statistics: list[str]
    written_statistics: list[str]
    subgroups: bool = False
    water: bool = False
    warnings: list[str] = field(default_factory=list)


def plan_selection(
    variables: Sequence[str],
    depths: Sequence[str],
    periods: Sequence[str],
    statistics: Sequence[str],
    subgroups: bool = False,
    water: bool = False,
) -> Selection:
    unknown = [v for v in variables if v not in VARIABLE_BY_CODE]
    if unknown:
        raise SoilDataError(f"Unknown OpenLandMap variable(s): {', '.join(unknown)}")
    bad = [d for d in depths if d not in DEPTH_BY_LABEL]
    if bad:
        raise SoilDataError(f"Unknown depth interval(s): {', '.join(bad)}")
    bad = [p for p in periods if p not in PERIODS]
    if bad:
        raise SoilDataError(f"Unknown period(s): {', '.join(bad)}")
    bad = [s for s in statistics if s not in STAT_BY_CODE]
    if bad:
        raise SoilDataError(f"Unknown statistic(s): {', '.join(bad)}")
    if not variables and not subgroups and not water:
        raise SoilDataError(
            "Select at least one variable (or the USDA subgroup or water-content "
            "option)."
        )
    if water and not depths:
        raise SoilDataError("Select at least one depth interval.")
    if variables and not depths:
        raise SoilDataError("Select at least one depth interval.")
    if variables and not periods:
        raise SoilDataError("Select at least one period.")
    if variables and not statistics:
        raise SoilDataError("Select at least one statistic.")

    written = [s.code for s in STATISTICS if s.code in statistics]
    fetched = list(written)
    warnings: list[str] = []
    return Selection(
        variables=[v.code for v in VARIABLES if v.code in variables],
        depths=[d[0] for d in DEPTHS if d[0] in depths],
        periods=[p for p in PERIODS if p in periods],
        statistics=fetched,
        written_statistics=written,
        subgroups=subgroups,
        water=water,
        warnings=warnings,
    )


@dataclass(frozen=True)
class Job:
    var: str
    stat: str
    period: str  # period actually read
    depth: str


def plan_jobs(selection: Selection) -> tuple[list[Job], list[str]]:
    """Every layer to read, with periods resolved against what is
    published. Returns (jobs, notes) - notes explain substitutions/skips."""
    jobs: list[Job] = []
    notes: list[str] = []
    seen: set[Job] = set()
    for var in selection.variables:
        meta = VARIABLE_BY_CODE[var]
        for stat in selection.statistics:
            for period in selection.periods:
                actual = resolve_period(var, stat, period)
                if actual is None:
                    notes.append(
                        f"{var} {STAT_BY_CODE[stat].label} is not published for "
                        f"{period} (only {', '.join(meta.periods_30m)}); skipped. "
                        "Use the 120 m mean for other periods."
                    )
                    continue
                for depth in selection.depths:
                    job = Job(var, stat, actual, depth)
                    if job not in seen:
                        seen.add(job)
                        jobs.append(job)
        if meta.static and any(p != meta.periods_30m[0] for p in selection.periods):
            notes.append(
                f"{var} is published for {meta.periods_30m[0]} only and treated "
                "as static: that map is used for every selected period."
            )
    return jobs, notes


# ----------------------------------------------------------------------
# Scale factors
# ----------------------------------------------------------------------


def resolve_scale(
    file_scale: float | None, file_offset: float | None, catalogue_scale: float
) -> tuple[float, float, str]:
    """(scale, offset, source note). The file's own metadata wins when set."""
    has_scale = file_scale is not None and abs(file_scale - 1.0) > SCALE_TOLERANCE
    has_offset = file_offset is not None and abs(file_offset) > SCALE_TOLERANCE
    if has_scale or has_offset:
        scale = file_scale if file_scale is not None else 1.0
        offset = file_offset or 0.0
        note = "file metadata"
        if abs(scale - catalogue_scale) > SCALE_TOLERANCE:
            note = (
                f"file metadata (differs from the catalogue scale {catalogue_scale:g})"
            )
        return scale, offset, note
    if abs(catalogue_scale - 1.0) > SCALE_TOLERANCE:
        return catalogue_scale, 0.0, "catalogue (file has no scale metadata)"
    return 1.0, 0.0, "none needed"


def apply_scale(raw, scale: float, offset: float) -> np.ndarray:
    """Stored -> physical units; NaN (nodata) stays NaN."""
    return np.asarray(raw, dtype=np.float64) * scale + offset


# ----------------------------------------------------------------------
# Sources, results
# ----------------------------------------------------------------------


def layer_sources(job: Job) -> list[Source]:
    return [
        Source(
            "COG (https)",
            "/vsicurl/" + cog_url(job.var, job.stat, job.depth, job.period),
        ),
        Source(
            "COG (http)",
            "/vsicurl/"
            + cog_url(job.var, job.stat, job.depth, job.period, scheme="http"),
        ),
    ]


@dataclass
class LayerRecord:
    file: str
    band: int
    layer: str
    variable: str
    depth: str
    period: str
    statistic: str
    resolution_m: int
    units: str
    scale: float
    scale_source: str
    route: str
    stats: dict


@dataclass
class RunResult:
    mode: str
    files: list[str] = field(default_factory=list)
    layers: list[LayerRecord] = field(default_factory=list)
    texture: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    load_files: list[str] = field(default_factory=list)
    subgroup: dict = field(default_factory=dict)
    water: dict = field(default_factory=dict)
    seconds: float = 0.0


def _options(options_ctx, default_used: bool):
    if options_ctx is not None:
        return options_ctx
    return http_options() if default_used else nullcontext()


def _log(result: RunResult, log_fn: Callable, msg: str) -> None:
    result.warnings.append(msg)
    log_fn(msg)


def _read_layers(jobs, reader, arg, workers, cancel_fn, progress, verb):
    """Read every job in parallel; each value is (raw, scale, offset, route,
    errors) where scale/offset come from the file."""

    def job_fn(job):
        (raw, f_scale, f_offset), route, errors = read_with_fallback(
            layer_sources(job), reader, arg, True
        )
        return raw, f_scale, f_offset, route, errors

    def done(job):
        tag = STAT_BY_CODE[job.stat].tag
        progress.step(f"{verb} {layer_label(job.var, job.depth, tag, job.period)}")

    return run_jobs(jobs, job_fn, workers, cancel_fn, done)


def _to_values(job, got, result, log_fn):
    """Scale one read layer; returns (values, scale, scale_note, route)."""
    raw, f_scale, f_offset, route, errors = got[job]
    meta = VARIABLE_BY_CODE[job.var]
    name = layer_label(job.var, job.depth, STAT_BY_CODE[job.stat].tag, job.period)
    for err in errors:
        _log(result, log_fn, f"{name}: primary route failed, used {route} ({err})")
    scale, offset, note = resolve_scale(f_scale, f_offset, meta.scale)
    if note.startswith("file metadata (differs"):
        _log(result, log_fn, f"{name}: scale from the {note}.")
    return apply_scale(raw, scale, offset), scale, note, route


# ----------------------------------------------------------------------
# Area mode
# ----------------------------------------------------------------------


def extract_area(
    selection: Selection,
    grid: TargetGrid,
    out_dir: str,
    *,
    read_grid_fn: Callable | None = None,
    write_fn: Callable | None = None,
    workers: int = DEFAULT_WORKERS,
    options_ctx=None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    """One multi-band GeoTIFF per variable x statistic x period (one band
    per depth). ``write_fn(path, grid, bands, units)``."""
    if write_fn is None:
        raise SoilDataError("No raster writer supplied.")
    read = read_grid_fn or gdal_read_grid
    jobs, notes = plan_jobs(selection)
    result = RunResult(mode="area", warnings=list(selection.warnings), notes=notes)
    for note in notes:
        log_fn(note)
    if not jobs and not selection.subgroups and not selection.water:
        raise SoilDataError(
            "Nothing to read: none of the selected layers is published for the "
            "selected period(s)."
        )
    n_sub = len(SUBGROUP_NAMES) if selection.subgroups else 0
    n_water = water_job_count(selection)
    progress = _Progress(len(jobs) + n_sub + n_water, progress_fn)
    t_start = datetime.now(UTC)
    texture: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    latest = selection.periods[-1] if selection.periods else LATEST_PERIOD

    with _options(options_ctx, read_grid_fn is None):
        for var in selection.variables:
            meta = VARIABLE_BY_CODE[var]
            var_jobs = [j for j in jobs if j.var == var]
            if not var_jobs:
                continue
            got = _read_layers(
                var_jobs, read, grid, workers, cancel_fn, progress, "Read"
            )
            arrays: dict[tuple[str, str, str], np.ndarray] = {}
            periods_read = sorted({j.period for j in var_jobs}, key=list(PERIODS).index)
            for period in periods_read:
                for stat in selection.statistics:
                    st = STAT_BY_CODE[stat]
                    keys = [
                        j for j in var_jobs if j.stat == stat and j.period == period
                    ]
                    if not keys:
                        continue
                    path = os.path.join(out_dir, output_file_name(var, st.tag, period))
                    bands, records = [], []
                    for job in sorted(
                        keys, key=lambda j: list(DEPTH_BY_LABEL).index(j.depth)
                    ):
                        values, scale, note, route = _to_values(
                            job, got, result, log_fn
                        )
                        arrays[(stat, period, job.depth)] = values
                        bands.append(
                            (
                                band_description(
                                    var, job.depth, st.tag, period, meta.units
                                ),
                                values,
                            )
                        )
                        records.append(
                            LayerRecord(
                                file=os.path.basename(path),
                                band=len(bands),
                                layer=layer_label(var, job.depth, st.tag, period),
                                variable=var,
                                depth=job.depth,
                                period=period,
                                statistic=stat,
                                resolution_m=st.resolution_m,
                                units=meta.units,
                                scale=scale,
                                scale_source=note,
                                route=route,
                                stats=summarise(values),
                            )
                        )
                        if stat == "mean30" and var in TEXTURE_CODES:
                            texture.setdefault((period, job.depth), {})[var] = values
                    if stat in selection.written_statistics:
                        write_fn(path, grid, bands, meta.units)
                        result.files.append(path)
                        result.layers.extend(records)
                        _check_empty(records, result, log_fn)
                        main = (
                            "mean30"
                            if "mean30" in selection.written_statistics
                            else "mean120"
                        )
                        newest = latest if latest in periods_read else periods_read[-1]
                        if stat == main and period == newest:
                            result.load_files.append(path)
            del got

    if selection.water:
        with _options(options_ctx, read_grid_fn is None):
            water = _read_water(
                read, grid, selection, workers, cancel_fn, progress, result, log_fn
            )
        _write_water(water, grid, out_dir, write_fn, result, log_fn)

    if selection.subgroups:
        with _options(options_ctx, read_grid_fn is None):
            top = _scan_subgroups(
                read,
                grid,
                (grid.height, grid.width),
                workers,
                cancel_fn,
                progress,
                result,
                log_fn,
            )
        path = os.path.join(out_dir, SUBGROUP_FILE)
        write_fn(path, grid, _subgroup_bands(top), "code / %")
        result.files.append(path)
        result.load_files.append(path)
        result.subgroup = _subgroup_summary(top, path)
        _subgroup_records(result, top, os.path.basename(path))
        warning = subgroup_warning(result.subgroup)
        if warning:
            _log(result, log_fn, warning)

    for (period, depth), means in sorted(texture.items()):
        if all(c in means for c in TEXTURE_CODES):
            result.texture.append(
                {
                    "period": period,
                    "depth": depth,
                    **texture_sum_check(
                        means["sand"],
                        means["silt"],
                        means["clay"],
                        TEXTURE_SUM_TOLERANCE,
                    ),
                }
            )
    result.seconds = (datetime.now(UTC) - t_start).total_seconds()
    progress_fn(1.0, "Done")
    return result


def _check_empty(records, result, log_fn):
    for rec in records:
        if rec.stats.get("valid") == 0:
            _log(
                result,
                log_fn,
                f"{rec.layer} has no valid values in the area of interest "
                "(deserts and permanent ice are not mapped).",
            )


# ----------------------------------------------------------------------
# Legacy 250 m water content (33 / 1500 kPa), at native depth points
# ----------------------------------------------------------------------


def water_job_count(selection: Selection) -> int:
    if not selection.water:
        return 0
    return len(LEGACY_VARIABLES) * len(legacy_points_needed(selection.depths))


def water_sources(code: str, depth_cm: int) -> list[Source]:
    arco, zenodo = legacy_urls(code, depth_cm)
    return [
        Source("COG (s3.openlandmap.org)", "/vsicurl/" + arco),
        Source("Zenodo (slower)", "/vsicurl/" + zenodo),
    ]


def water_file_name(code: str) -> str:
    return f"olm_{LEGACY_BY_CODE[code].out_name}_250m_{LEGACY_PERIOD}.tif"


def water_layer_name(code: str, depth_cm: int) -> str:
    return f"{LEGACY_BY_CODE[code].out_name}_{depth_cm}cm_250m_{LEGACY_PERIOD}"


def _read_water(reader, arg, selection, workers, cancel_fn, progress, result, log_fn):
    """{code: {depth_cm: (values, route)}} at the native depth POINTS that
    bound the selected intervals - stored as published, no averaging."""
    points = legacy_points_needed(selection.depths)
    keys = [(v.code, d) for v in LEGACY_VARIABLES for d in points]

    def job(key):
        (raw, f_scale, f_offset), route, errors = read_with_fallback(
            water_sources(*key), reader, arg, True
        )
        return raw, f_scale, f_offset, route, errors

    def done(key):
        progress.step(f"Read {water_layer_name(*key)}")

    got = run_jobs(keys, job, workers, cancel_fn, done)
    data: dict = {v.code: {} for v in LEGACY_VARIABLES}
    for code, depth_cm in keys:
        raw, f_scale, f_offset, route, errors = got[(code, depth_cm)]
        for err in errors:
            _log(
                result,
                log_fn,
                f"{water_layer_name(code, depth_cm)}: primary route failed, used "
                f"{route} ({err})",
            )
        scale, offset, _ = resolve_scale(f_scale, f_offset, 1.0)
        data[code][depth_cm] = (apply_scale(raw, scale, offset), route)
    return data


def _write_water(water, grid, out_dir, write_fn, result, log_fn):
    for v in LEGACY_VARIABLES:
        path = os.path.join(out_dir, water_file_name(v.code))
        bands, records = [], []
        for depth_cm, (values, route) in sorted(water[v.code].items()):
            name = water_layer_name(v.code, depth_cm)
            bands.append((f"{name} (vol %)", values))
            records.append(
                LayerRecord(
                    file=os.path.basename(path),
                    band=len(bands),
                    layer=name,
                    variable=v.out_name,
                    depth=f"{depth_cm}cm",
                    period=LEGACY_PERIOD,
                    statistic="mean",
                    resolution_m=250,
                    units="vol %",
                    scale=1.0,
                    scale_source="file metadata or none",
                    route=route,
                    stats=summarise(values),
                )
            )
        write_fn(path, grid, bands, "vol %")
        result.files.append(path)
        result.load_files.append(path)
        result.layers.extend(records)
        _check_empty(records, result, log_fn)


# ----------------------------------------------------------------------
# USDA subgroups: streaming "most probable" over 818 probability layers
# ----------------------------------------------------------------------

SUBGROUP_FILE = f"olm_usda_subgroup_{SUBGROUP_PERIOD}.tif"
SUBGROUP_BATCH = 32  # layers held in memory at once (per batch)


def subgroup_sources(name: str) -> list[Source]:
    return [
        Source("COG (https)", "/vsicurl/" + subgroup_url(name)),
        Source("COG (http)", "/vsicurl/" + subgroup_url(name, scheme="http")),
    ]


def top2_update(state: dict, values: np.ndarray, code: int) -> None:
    """Keep the two highest probabilities (and their codes) per cell.
    NaN never wins; on a tie the earlier (lower) code is kept."""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    first = finite & (v > state["p1"])
    second = finite & ~first & (v > state["p2"])
    state["p2"][first] = state["p1"][first]
    state["code2"][first] = state["code1"][first]
    state["p1"][first] = v[first]
    state["code1"][first] = code
    state["p2"][second] = v[second]
    state["code2"][second] = code
    state["psum"][finite] += v[finite]
    state["n"][finite] += 1


def _scan_subgroups(reader, arg, shape, workers, cancel_fn, progress, result, log_fn):
    state = {
        "p1": np.full(shape, -1.0),
        "p2": np.full(shape, -1.0),
        "code1": np.zeros(shape, dtype=np.int32),
        "code2": np.zeros(shape, dtype=np.int32),
        "psum": np.zeros(shape),
        "n": np.zeros(shape, dtype=np.int32),
    }
    routes: dict[str, int] = {}

    def job(k):
        (raw, f_scale, f_offset), route, errors = read_with_fallback(
            subgroup_sources(SUBGROUP_NAMES[k]), reader, arg, True
        )
        return raw, f_scale, f_offset, route, errors

    def done(k):
        progress.step(f"Read subgroup {subgroup_label(SUBGROUP_NAMES[k])}")

    n_names = len(SUBGROUP_NAMES)
    batch = max(SUBGROUP_BATCH, workers * 4)
    for start in range(0, n_names, batch):
        keys = list(range(start, min(n_names, start + batch)))
        got = run_jobs(keys, job, workers, cancel_fn, done)
        for k in keys:  # in code order, so ties are deterministic
            raw, f_scale, f_offset, route, errors = got[k]
            for err in errors:
                _log(
                    result,
                    log_fn,
                    f"{SUBGROUP_NAMES[k]}: primary route failed, used {route} ({err})",
                )
            scale, offset, _ = resolve_scale(f_scale, f_offset, 1.0)
            top2_update(state, apply_scale(raw, scale, offset), k + 1)
            routes[route] = routes.get(route, 0) + 1
        del got

    mapped = state["n"] > 0
    for key in ("p1", "p2", "psum"):
        state[key] = np.where(mapped, state[key], np.nan)
    state["p1"][state["p1"] < 0] = np.nan
    state["p2"][state["p2"] < 0] = np.nan
    state["code1"][~mapped] = 0
    state["code2"][~np.isfinite(state["p2"])] = 0
    state["route"] = max(routes, key=routes.get) if routes else ""
    state["routes"] = routes
    return state


def _subgroup_bands(top: dict) -> list[tuple[str, np.ndarray]]:
    def code_array(codes):
        return np.where(codes > 0, codes.astype(np.float64), np.nan)

    period = SUBGROUP_PERIOD
    return [
        (f"usda_subgroup_code_rank1_{period} (code)", code_array(top["code1"])),
        (f"usda_subgroup_probability_rank1_{period} (%)", top["p1"]),
        (f"usda_subgroup_code_rank2_{period} (code)", code_array(top["code2"])),
        (f"usda_subgroup_probability_rank2_{period} (%)", top["p2"]),
        (f"usda_subgroup_probability_sum_{period} (%)", top["psum"]),
    ]


SUBGROUP_LOW_PROBABILITY = 20.0  # % - below this the map is indicative only


def subgroup_warning(summary: dict) -> str | None:
    """Warning text when the most probable subgroup is itself improbable."""
    med = summary.get("median_probability", math.nan)
    if not summary.get("mapped") or not math.isfinite(med):
        return None
    if med < SUBGROUP_LOW_PROBABILITY:
        return (
            f"USDA subgroups: the most probable subgroup has a median probability "
            f"of only {med:.0f}% (probabilities spread over {len(SUBGROUP_NAMES)} "
            "subgroups). The subgroup map is indicative only here - the top class "
            "barely beats the alternatives. Compare rank 1 and rank 2 "
            "probabilities before relying on it."
        )
    return None


def _subgroup_summary(top: dict, path: str) -> dict:
    codes = top["code1"].ravel()
    probs = top["p1"].ravel()
    mapped = codes > 0
    total = int(mapped.sum())
    table = []
    for code in sorted(set(codes[mapped].tolist())):
        sel = codes == code
        name = SUBGROUP_NAMES[code - 1]
        table.append(
            {
                "code": code,
                "name": name,
                "label": subgroup_label(name),
                "great_group": great_group(name),
                "order": soil_order(name),
                "cells": int(sel.sum()),
                "share_pct": 100.0 * sel.sum() / total if total else math.nan,
                "mean_probability": float(np.nanmean(probs[sel])),
            }
        )
    table.sort(key=lambda r: -r["cells"])
    psum = top["psum"][np.isfinite(top["psum"])]
    return {
        "file": path,
        "table": table,
        "mapped": total,
        "median_probability": float(np.nanmedian(probs[mapped])) if total else math.nan,
        "median_sum": float(np.median(psum)) if psum.size else math.nan,
        "routes": top["routes"],
    }


def _subgroup_records(result: RunResult, top: dict, file_name: str) -> None:
    for band, (desc, arr) in enumerate(_subgroup_bands(top), start=1):
        result.layers.append(
            LayerRecord(
                file=file_name,
                band=band,
                layer=desc.split(" (")[0],
                variable="usda_subgroup",
                depth="n/a",
                period=SUBGROUP_PERIOD,
                statistic=desc.split("_")[2],
                resolution_m=30,
                units=desc.split("(")[-1].rstrip(")"),
                scale=1.0,
                scale_source="file metadata or none",
                route=top["route"],
                stats=summarise(arr),
            )
        )


# ----------------------------------------------------------------------
# Point mode
# ----------------------------------------------------------------------


@dataclass(frozen=True)
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
    workers: int = DEFAULT_WORKERS,
    options_ctx=None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    """Sample every selected layer at every site (long-format rows)."""
    if not sites:
        raise SoilDataError("No points to sample.")
    sample = sample_fn or gdal_sample_points
    jobs, notes = plan_jobs(selection)
    result = RunResult(mode="points", warnings=list(selection.warnings), notes=notes)
    for note in notes:
        log_fn(note)
    if not jobs and not selection.subgroups and not selection.water:
        raise SoilDataError(
            "Nothing to read: none of the selected layers is published for the "
            "selected period(s)."
        )
    n_sub = len(SUBGROUP_NAMES) if selection.subgroups else 0
    n_water = water_job_count(selection)
    progress = _Progress(len(jobs) + n_sub + n_water, progress_fn)
    lonlats = [(s.lon, s.lat) for s in sites]
    t_start = datetime.now(UTC)
    store: dict[tuple[str, str, str, str], np.ndarray] = {}

    def add_rows(var, label, depth, period, stat, res, values, units, route):
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
                    "Product": PRODUCT,
                    "Variable": var,
                    "Description": label,
                    "DepthTop_cm": top,
                    "DepthBottom_cm": bottom,
                    "Period": period,
                    "Statistic": stat,
                    "Resolution_m": res,
                    "Value": value,
                    "Units": units,
                    "Route": route,
                }
            )

    with _options(options_ctx, sample_fn is None):
        for var in selection.variables:
            meta = VARIABLE_BY_CODE[var]
            var_jobs = [j for j in jobs if j.var == var]
            if not var_jobs:
                continue
            got = _read_layers(
                var_jobs, sample, lonlats, workers, cancel_fn, progress, "Sampled"
            )
            for job in var_jobs:
                values, scale, note, route = _to_values(job, got, result, log_fn)
                store[(var, job.stat, job.period, job.depth)] = values
                st = STAT_BY_CODE[job.stat]
                if job.stat in selection.written_statistics:
                    add_rows(
                        var,
                        meta.label,
                        job.depth,
                        job.period,
                        st.label,
                        st.resolution_m,
                        values,
                        meta.units,
                        route,
                    )
                    result.layers.append(
                        LayerRecord(
                            file="",
                            band=0,
                            layer=layer_label(var, job.depth, st.tag, job.period),
                            variable=var,
                            depth=job.depth,
                            period=job.period,
                            statistic=job.stat,
                            resolution_m=st.resolution_m,
                            units=meta.units,
                            scale=scale,
                            scale_source=note,
                            route=route,
                            stats=summarise(values),
                        )
                    )
    if selection.water:
        with _options(options_ctx, sample_fn is None):
            water = _read_water(
                sample,
                lonlats,
                selection,
                workers,
                cancel_fn,
                progress,
                result,
                log_fn,
            )
        for v in LEGACY_VARIABLES:
            for depth_cm, (values, route) in sorted(water[v.code].items()):
                add_rows(
                    v.out_name,
                    v.label,
                    f"{depth_cm}cm",
                    LEGACY_PERIOD,
                    "mean",
                    250,
                    values,
                    "vol %",
                    route,
                )
                result.layers.append(
                    LayerRecord(
                        file="",
                        band=0,
                        layer=water_layer_name(v.code, depth_cm),
                        variable=v.out_name,
                        depth=f"{depth_cm}cm",
                        period=LEGACY_PERIOD,
                        statistic="mean",
                        resolution_m=250,
                        units="vol %",
                        scale=1.0,
                        scale_source="file metadata or none",
                        route=route,
                        stats=summarise(values),
                    )
                )

    if selection.subgroups:
        with _options(options_ctx, sample_fn is None):
            top = _scan_subgroups(
                sample,
                lonlats,
                (len(sites),),
                workers,
                cancel_fn,
                progress,
                result,
                log_fn,
            )
        for rank, (codes, probs) in enumerate(
            ((top["code1"], top["p1"]), (top["code2"], top["p2"])), start=1
        ):
            label = "Most probable" if rank == 1 else "Second most probable"
            for i, site in enumerate(sites):
                code = int(codes[i])
                name = SUBGROUP_NAMES[code - 1] if code > 0 else ""
                result.rows.append(
                    {
                        "Site": site.label,
                        "Longitude": site.lon,
                        "Latitude": site.lat,
                        "Product": PRODUCT,
                        "Variable": "usda_subgroup",
                        "Description": (
                            f"{subgroup_label(name)} ({great_group(name)}, "
                            f"{soil_order(name)}; code {code})"
                            if name
                            else "not mapped"
                        ),
                        "DepthTop_cm": "",
                        "DepthBottom_cm": "",
                        "Period": SUBGROUP_PERIOD,
                        "Statistic": f"{label} subgroup - probability",
                        "Resolution_m": 30,
                        "Value": float(probs[i]),
                        "Units": "%",
                        "Route": top["route"],
                    }
                )
        result.subgroup = _subgroup_summary(top, "")
        warning = subgroup_warning(result.subgroup)
        if warning:
            _log(result, log_fn, warning)

    for period in selection.periods + [LATEST_PERIOD]:
        for depth in selection.depths:
            keys = [(c, "mean30", period, depth) for c in TEXTURE_CODES]
            if all(k in store for k in keys) and not any(
                t["period"] == period and t["depth"] == depth for t in result.texture
            ):
                result.texture.append(
                    {
                        "period": period,
                        "depth": depth,
                        **texture_sum_check(
                            *(store[k] for k in keys), TEXTURE_SUM_TOLERANCE
                        ),
                    }
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
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    run_rows: list[list] = [
        ["Tool", f"Extract: OpenLandMap Soils (mayim_tools) v{TOOL_VERSION}"],
        ["Run time", now],
        ["Duration (s)", round(result.seconds, 1)],
        ["Mode", result.mode],
        ["Product", f"{PRODUCT} (30 m means; 120 m mean, P16, P84)"],
        ["Access date", now[:10]],
        ["Server", "https://s3.opengeohub.org/global-soil/ (http fallback)"],
        ["Layer catalogue", CATALOGUE_URL],
        ["GDAL version", gdal_version_text],
        ["Citation", CITATION],
        ["Data DOI", DATA_DOI],
        ["Licence", LICENCE],
    ]
    for key, value in settings.items():
        run_rows.append([key, value])
    run_rows += [
        ["Variables", ", ".join(selection.variables)],
        ["Depths", ", ".join(selection.depths)],
        ["Periods requested", ", ".join(selection.periods)],
        [
            "Statistics written",
            ", ".join(STAT_BY_CODE[s].label for s in selection.written_statistics)
            or "(none)",
        ],
        ["Output nodata", NODATA_OUT],
        [
            "Note - texture limits",
            "OpenLandMap texture follows ISO 11277: silt 0.002-0.063 mm, sand "
            "0.063-2 mm. USDA uses 0.05 mm. Convert before using the USDA "
            "texture triangle or USDA-based pedotransfer functions.",
        ],
        [
            "Note - statistics",
            "The 30 m values are means (not medians). P16/P84 bound the 68 % "
            "prediction interval and exist at 120 m only; on a finer output "
            "grid they repeat in 120 m blocks. The 120 m mean is published so "
            "that P16, mean and P84 can be compared on the same support.",
        ],
        [
            "Note - quantiles",
            "Texture quantiles are marginal: P16 sand + P16 clay is not a soil. "
            "Sample and renormalise when propagating uncertainty.",
        ],
        [
            "Note - coverage",
            "Deserts and permanent ice are not mapped (nodata). Texture is "
            "mapped for 2020-2022 only and used for every requested period.",
        ],
        [
            "Note - units",
            "Values are converted to physical units with the scale stored in "
            "each file (catalogue scale as fallback; see the Layers section). "
            "SOC is not converted to OM.",
        ],
        [
            "Note - bands",
            "Select bands by their description (e.g. clay_30-60cm_mean_30m_2020-"
            "2022 (%)), never by band number.",
        ],
    ]
    var_rows = [
        [
            v.code,
            v.label,
            v.filename,
            v.scale,
            v.units,
            ", ".join(v.periods_30m),
            ", ".join(v.periods_120m),
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
            rec.period,
            rec.statistic,
            rec.resolution_m,
            rec.units,
            rec.scale,
            rec.scale_source,
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
        [t["period"], t["depth"], t["checked"], t["flagged"], _r(t["max_abs_dev"])]
        for t in result.texture
    ]
    if selection.water:
        run_rows += [
            ["Water content (250 m)", "33 kPa and 1500 kPa, 1950-2017"],
            ["Citation (water content)", LEGACY_CITATION],
            ["Data DOI (water content)", LEGACY_DOI],
            ["Licence (water content)", LEGACY_LICENCE + " (share-alike)"],
            [
                "Note - water content",
                "Older OpenLandMap layers (v0.1, 250 m) mapped from measured "
                "volumetric water contents - no pedotransfer functions except to "
                "fill missing bulk densities - so they are an independent check "
                "on PTF field capacity and wilting point. Stored as published at "
                "depth POINTS (0, 30, 60, 100 cm; those bounding the selected "
                "intervals), one band per point. Conversion to intervals and "
                "available water capacity are computed by Regional soil "
                "parameterisation, not here. Read from the cloud-optimised copy "
                "on s3.openlandmap.org where available, otherwise Zenodo (route "
                "per layer in the Layers section).",
            ],
        ]
    if selection.subgroups:
        run_rows += [
            ["USDA subgroups", f"most probable of {len(SUBGROUP_NAMES)} (2000-2022)"],
            [
                "Note - subgroups",
                "OpenLandMap gives the probability of each USDA soil-taxonomy "
                "subgroup. The tool keeps the two most probable subgroups and "
                "their probabilities per cell, plus the sum of all probabilities "
                "(should be close to 100 %). A low top probability means the "
                "class is uncertain; treat the map as indicative.",
            ],
        ]
    sub = result.subgroup
    sub_rows = [
        [
            r["code"],
            r["label"],
            r["great_group"],
            r["order"],
            r["cells"],
            _r(r["share_pct"], 2),
            _r(r["mean_probability"], 1),
        ]
        for r in sub.get("table", [])
    ]
    if sub:
        sub_rows.append(
            [
                "",
                "(all mapped cells)",
                "",
                "",
                sub["mapped"],
                100,
                _r(sub["median_probability"], 1),
            ]
        )
        sub_rows.append(
            [
                "",
                "Median sum of probabilities (%)",
                "",
                "",
                "",
                "",
                _r(sub["median_sum"], 1),
            ]
        )
    note_rows = [[n] for n in result.notes] or [["(none)"]]
    warn_rows = [[w] for w in result.warnings] or [["(none)"]]
    return [
        ("Run", ["Item", "Value"], run_rows),
        (
            "Variables",
            [
                "Code",
                "Description",
                "OpenLandMap name",
                "Catalogue scale",
                "Units",
                "Periods (30 m)",
                "Periods (120 m)",
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
                "Period",
                "Statistic",
                "Resolution (m)",
                "Units",
                "Scale applied",
                "Scale source",
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
            f"Texture check (30 m mean sand+silt+clay, flag if off 100 by > "
            f"{TEXTURE_SUM_TOLERANCE:g} %)",
            ["Period", "Depth", "Checked", "Flagged", "Max abs deviation (%)"],
            tex_rows,
        ),
        (
            "USDA subgroups (most probable)",
            [
                "Code",
                "Subgroup",
                "Great group",
                "Order",
                "Cells/points",
                "Share (%)",
                "Mean top probability (%)",
            ],
            sub_rows or [["(not requested)"]],
        ),
        ("Notes (periods)", ["Note"], note_rows),
        ("Warnings", ["Warning"], warn_rows),
    ]
