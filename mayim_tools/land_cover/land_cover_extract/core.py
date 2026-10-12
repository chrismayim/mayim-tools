"""
Core logic for "Extract: Land cover" - no QGIS dependency.

ESA WorldCover 10 m tiles are read straight from the public cloud-optimised
GeoTIFFs with QGIS's own GDAL (/vsicurl/, with an unsigned /vsis3/
fallback). Only the tiles that touch the area of interest are used; tiles
that do not exist (open ocean) are skipped rather than failing the run.
Area mode warps the tiles onto the output grid with nearest neighbour in
one GDAL call (class codes are never averaged); points mode samples the
nearest cell. Network access is injectable, so the orchestration is
unit-tested with fakes.

This tool only acquires data: class codes are written as published.
Curve numbers, imperviousness and every other derived quantity belong to
the downstream tools.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.gdal_io import (
    HTTP_OPTIONS,
    gdal_sample_points,
    http_options,
)
from mayim_tools.soil._common.grid import EARTH_RADIUS_KM, TargetGrid
from mayim_tools.soil._common.jobs import _never_cancelled, _noop

from .catalogue import (
    CLASS_BY_CODE,
    CLASSES,
    EDITION_BY_YEAR,
    HOST,
    KNOWN_TILE,
    LICENCE,
    NODATA,
    PRODUCT,
    PRODUCT_LONG,
    RESOLUTION_M,
    ROUTES,
    class_name,
    output_file_name,
    tile_of,
    tiles_for_bounds,
)

TOOL_VERSION = "0.1.0"
DEFAULT_WORKERS = 8
DEFAULT_MAX_AREA_KM2 = 5000.0  # 10 m data: 50 million cells, about 50 MB
WORLDCOVER_OPTIONS = {
    **HTTP_OPTIONS,
    "AWS_NO_SIGN_REQUEST": "YES",
    "AWS_REGION": "eu-central-1",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}

LandCoverError = SoilDataError


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


@dataclass
class Selection:
    years: list[int]
    warnings: list[str] = field(default_factory=list)


def plan_selection(years: Sequence[int]) -> Selection:
    if not years:
        raise SoilDataError("Select at least one year.")
    unknown = [y for y in years if y not in EDITION_BY_YEAR]
    if unknown:
        raise SoilDataError(f"No WorldCover edition for {unknown}.")
    years = sorted(set(years), reverse=True)
    sel = Selection(years)
    if len(years) > 1:
        sel.warnings.append(
            "WorldCover 2020 (v100) and 2021 (v200) were made with different "
            "algorithms, so part of any difference between them is a change of "
            "method, not a change on the ground. Do not use them as a "
            "change map."
        )
    return sel


# ----------------------------------------------------------------------
# GDAL defaults (replaced by fakes in the tests)
# ----------------------------------------------------------------------


def gdal_exists(path: str) -> bool:
    """True if the remote file exists (an HTTP HEAD, no PROJ work)."""
    from osgeo import gdal

    return gdal.VSIStatL(path) is not None


def gdal_warp_classes(paths: Sequence[str], grid: TargetGrid, cancel_fn) -> np.ndarray:
    """Mosaic and warp class tiles onto ``grid`` with nearest neighbour.
    UInt8, NODATA where no tile covers a cell. One GDAL call: GDAL reads
    only the blocks of each COG that the grid needs."""
    from osgeo import gdal

    gdal.UseExceptions()

    def callback(_fraction, _msg, _data):
        return 0 if cancel_fn() else 1

    ds = gdal.Warp(
        "",
        list(paths),
        format="MEM",
        outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
        width=grid.width,
        height=grid.height,
        dstSRS=grid.crs_wkt,
        resampleAlg="near",
        outputType=gdal.GDT_Byte,
        srcNodata=NODATA,
        dstNodata=NODATA,
        multithread=True,
        warpOptions=["NUM_THREADS=ALL_CPUS"],
        callback=callback,
    )
    if ds is None:
        if cancel_fn():
            raise InterruptedError("Cancelled by user")
        raise SoilDataError("GDAL could not warp the WorldCover tiles.")
    try:
        return ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    finally:
        ds = None


# ----------------------------------------------------------------------
# Tiles
# ----------------------------------------------------------------------


@dataclass
class TileSet:
    year: int
    route: str
    found: list[str]
    missing: list[str]


def _stat_all(paths, exists_fn, workers):
    def safe(p):
        try:
            return bool(exists_fn(p))
        except Exception:  # noqa: BLE001 - treated as not found
            return False

    if workers <= 1 or len(paths) <= 1:
        return [safe(p) for p in paths]
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool:
        return list(pool.map(safe, paths))


def find_tiles(year: int, tiles: Sequence[str], exists_fn, workers: int) -> TileSet:
    """Which of ``tiles`` exist, on the first route that finds any. If no
    route finds one, a known tile tells an all-ocean area (empty result)
    from a server that cannot be reached (error)."""
    for route, path_fn in ROUTES:
        ok = _stat_all([path_fn(year, t) for t in tiles], exists_fn, workers)
        if any(ok):
            found = [t for t, k in zip(tiles, ok, strict=True) if k]
            missing = [t for t, k in zip(tiles, ok, strict=True) if not k]
            return TileSet(year, route, found, missing)
    for route, path_fn in ROUTES:
        if _stat_all([path_fn(year, KNOWN_TILE)], exists_fn, 1)[0]:
            return TileSet(year, route, [], list(tiles))
    raise SoilDataError(
        f"Could not reach the ESA WorldCover server (https://{HOST}/ and the "
        "unsigned S3 route both failed). Check the internet connection or "
        "proxy settings in QGIS."
    )


def _paths(year, route, tiles):
    order = [r for r in ROUTES if r[0] == route] + [r for r in ROUTES if r[0] != route]
    return [(name, [fn(year, t) for t in tiles]) for name, fn in order]


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------


@dataclass
class YearRecord:
    year: int
    file: str
    route: str
    tiles: TileSet
    counts: dict = field(default_factory=dict)  # code -> cells
    areas: dict = field(default_factory=dict)  # code -> km2
    nodata_cells: int = 0
    total_cells: int = 0


@dataclass
class RunResult:
    mode: str
    files: list[str] = field(default_factory=list)
    years: list[YearRecord] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _options(options_ctx, default_used: bool):
    if options_ctx is not None:
        return options_ctx
    return http_options(WORLDCOVER_OPTIONS) if default_used else nullcontext()


def _log(result: RunResult, log_fn: Callable, msg: str) -> None:
    result.warnings.append(msg)
    log_fn(msg)


# ----------------------------------------------------------------------
# Class areas
# ----------------------------------------------------------------------


def row_cell_areas_km2(grid: TargetGrid, geographic: bool) -> np.ndarray:
    """Area of one cell in each row (km2). Projected CRS: res squared (map
    units taken as metres). Geographic CRS: spherical area by latitude."""
    if not geographic:
        return np.full(grid.height, (grid.res**2) / 1e6)
    tops = grid.ymax - grid.res * np.arange(grid.height)
    bottoms = tops - grid.res
    band = np.sin(np.radians(np.clip(tops, -90, 90))) - np.sin(
        np.radians(np.clip(bottoms, -90, 90))
    )
    return EARTH_RADIUS_KM**2 * math.radians(grid.res) * band


def class_areas(arr: np.ndarray, row_area: np.ndarray, chunk: int = 512):
    """(cells by code, km2 by code) for every code present, nodata included."""
    counts = np.zeros(256, dtype=np.int64)
    areas = np.zeros(256, dtype=np.float64)
    for r0 in range(0, arr.shape[0], chunk):
        block = arr[r0 : r0 + chunk]
        counts += np.bincount(block.ravel(), minlength=256)
        for i, row in enumerate(block):
            areas += np.bincount(row, minlength=256) * row_area[r0 + i]
    present = np.nonzero(counts)[0]
    return (
        {int(c): int(counts[c]) for c in present},
        {int(c): float(areas[c]) for c in present},
    )


# ----------------------------------------------------------------------
# Area mode
# ----------------------------------------------------------------------


def extract_area(
    selection: Selection,
    grid: TargetGrid,
    bounds_ll,
    out_dir: str,
    *,
    geographic: bool = False,
    exists_fn: Callable | None = None,
    warp_fn: Callable | None = None,
    write_fn: Callable | None = None,
    workers: int = DEFAULT_WORKERS,
    options_ctx=None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    """One class GeoTIFF per year. ``write_fn(path, grid, array, year)``."""

    if write_fn is None:
        raise SoilDataError("No raster writer supplied.")
    exists = exists_fn or gdal_exists
    warp = warp_fn or gdal_warp_classes
    result = RunResult(mode="area", warnings=list(selection.warnings))
    t0 = datetime.now(UTC)
    tiles = tiles_for_bounds(bounds_ll)
    row_area = row_cell_areas_km2(grid, geographic)
    n_steps = 2 * len(selection.years)
    step = 0
    with _options(options_ctx, warp_fn is None):
        for year in selection.years:
            if cancel_fn():
                raise InterruptedError("Cancelled by user")
            progress_fn(step / n_steps, f"Finding WorldCover {year} tiles")
            ts = find_tiles(year, tiles, exists, workers)
            step += 1
            if not ts.found:
                _log(
                    result,
                    log_fn,
                    f"WorldCover {year}: no tiles cover the area of interest "
                    "(open ocean); the raster is all nodata.",
                )
                arr = np.full((grid.height, grid.width), NODATA, dtype=np.uint8)
                route = ts.route
            else:
                progress_fn(
                    step / n_steps,
                    f"Reading WorldCover {year} ({len(ts.found)} tile(s): "
                    f"{', '.join(ts.found)})",
                )
                arr, route = _warp_with_fallback(year, ts, grid, warp, cancel_fn)
            step += 1
            path = os.path.join(out_dir, output_file_name(year))
            write_fn(path, grid, arr, year)
            counts, areas = class_areas(arr, row_area)
            rec = YearRecord(
                year=year,
                file=os.path.basename(path),
                route=route,
                tiles=ts,
                counts={c: n for c, n in counts.items() if c != NODATA},
                areas={c: a for c, a in areas.items() if c != NODATA},
                nodata_cells=counts.get(NODATA, 0),
                total_cells=int(arr.size),
            )
            unknown = [c for c in rec.counts if c not in CLASS_BY_CODE]
            if unknown:
                _log(
                    result,
                    log_fn,
                    f"WorldCover {year}: unexpected class codes {unknown}.",
                )
            if rec.nodata_cells and ts.found:
                share = 100.0 * rec.nodata_cells / rec.total_cells
                if share >= 0.05:
                    _log(
                        result,
                        log_fn,
                        f"WorldCover {year}: {share:.1f} % of the grid has no "
                        "class (outside the tiles, usually open sea).",
                    )
            result.years.append(rec)
            result.files.append(path)
            del arr
    result.seconds = (datetime.now(UTC) - t0).total_seconds()
    progress_fn(1.0, "Done")
    return result


def _warp_with_fallback(year, ts, grid, warp, cancel_fn):
    errors = []
    for route, paths in _paths(year, ts.route, ts.found):
        try:
            return warp(paths, grid, cancel_fn), route
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            if cancel_fn():
                raise InterruptedError("Cancelled by user") from exc
            errors.append(f"{route}: {exc}")
    raise SoilDataError(
        f"WorldCover {year}: all access routes failed - " + " | ".join(errors)
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
    "Year",
    "Version",
    "Tile",
    "ClassCode",
    "ClassName",
    "Resolution_m",
    "Route",
]


def extract_points(
    selection: Selection,
    sites: Sequence[Site],
    *,
    exists_fn: Callable | None = None,
    sample_fn: Callable | None = None,
    workers: int = DEFAULT_WORKERS,
    options_ctx=None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> RunResult:
    """Nearest-cell class at each site, for each year. Sites over open
    ocean (no tile) or nodata get an empty class, never zero."""
    if not sites:
        raise SoilDataError("No points to sample.")
    exists = exists_fn or gdal_exists
    sample = sample_fn or gdal_sample_points
    result = RunResult(mode="points", warnings=list(selection.warnings))
    t0 = datetime.now(UTC)
    by_tile: dict[str, list[int]] = {}
    for i, s in enumerate(sites):
        by_tile.setdefault(tile_of(s.lon, s.lat), []).append(i)
    tiles = sorted(by_tile)
    total = len(selection.years) * (len(tiles) + 1)
    done = 0
    with _options(options_ctx, sample_fn is None):
        for year in selection.years:
            ed = EDITION_BY_YEAR[year]
            ts = find_tiles(year, tiles, exists, workers)
            done += 1
            progress_fn(done / total, f"Found {len(ts.found)} {year} tile(s)")
            codes: list = [None] * len(sites)
            routes: list = [""] * len(sites)
            for tile in ts.found:
                if cancel_fn():
                    raise InterruptedError("Cancelled by user")
                idx = by_tile[tile]
                lonlats = [(sites[i].lon, sites[i].lat) for i in idx]
                values, route = _sample_with_fallback(
                    year, ts.route, tile, lonlats, sample
                )
                for i, v in zip(idx, values, strict=True):
                    ok = v is not None and math.isfinite(v) and int(v) != NODATA
                    codes[i] = int(v) if ok else None
                    routes[i] = route
                done += 1
                progress_fn(min(done / total, 0.999), f"Sampled {year} tile {tile}")
            done += len(ts.missing)
            for i, s in enumerate(sites):
                result.rows.append(
                    {
                        "Site": s.label,
                        "Longitude": s.lon,
                        "Latitude": s.lat,
                        "Product": PRODUCT,
                        "Year": year,
                        "Version": ed.version,
                        "Tile": tile_of(s.lon, s.lat),
                        "ClassCode": "" if codes[i] is None else codes[i],
                        "ClassName": class_name(codes[i]),
                        "Resolution_m": RESOLUTION_M,
                        "Route": routes[i],
                    }
                )
            counts: dict[int, int] = {}
            for c in codes:
                if c is not None:
                    counts[c] = counts.get(c, 0) + 1
            n_missing = sum(c is None for c in codes)
            result.years.append(
                YearRecord(
                    year=year,
                    file="",
                    route=ts.route,
                    tiles=ts,
                    counts=counts,
                    nodata_cells=n_missing,
                    total_cells=len(sites),
                )
            )
            if n_missing:
                _log(
                    result,
                    log_fn,
                    f"WorldCover {year}: {n_missing} of {len(sites)} site(s) have no "
                    "class (open sea or outside the tiles); written as empty cells.",
                )
    result.seconds = (datetime.now(UTC) - t0).total_seconds()
    progress_fn(1.0, "Done")
    return result


def _sample_with_fallback(year, route, tile, lonlats, sample):
    errors = []
    for name, paths in _paths(year, route, [tile]):
        try:
            return list(sample(paths[0], lonlats)), name
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            errors.append(f"{name}: {exc}")
    raise SoilDataError(
        f"WorldCover {year} tile {tile}: all access routes failed - "
        + " | ".join(errors)
    )


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


def _fmt(x, nd=3):
    return "" if x is None else round(float(x), nd)


def build_metadata(
    result: RunResult, selection: Selection, settings, gdal_version_text=""
):
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    run_rows = [
        ["Tool", f"Extract: Land cover (mayim_tools) v{TOOL_VERSION}"],
        ["Run time", now],
        ["Duration (s)", round(result.seconds, 1)],
        ["Mode", result.mode],
        ["Product", f"{PRODUCT_LONG} (11 classes)"],
        ["Access date", now[:10]],
        ["Server", f"https://{HOST}/ (unsigned S3 fallback)"],
        ["GDAL version", gdal_version_text],
        ["Licence", LICENCE],
    ]
    for key, value in settings.items():
        run_rows.append([key, value])
    run_rows += [
        ["Years", ", ".join(str(y) for y in selection.years)],
        ["Output nodata", f"{NODATA} (no tile or no class)"],
        [
            "Note - what the classes are",
            "WorldCover classes are land cover as seen by Sentinel-1 and "
            "Sentinel-2 over one year. They carry no information on hydrological "
            "condition (veld or crop condition, tillage, conservation practice) or "
            "on the impervious share of built-up land; those are set in the "
            "curve-number step.",
        ],
        [
            "Note - accuracy",
            "Accuracy is reported for the map as a whole. Shrubland, herbaceous "
            "wetland and moss and lichen are the least accurate classes; mixed "
            "shrub-grass veld can appear as either class.",
        ],
        [
            "Note - resampling",
            "Nearest neighbour only: class codes are never averaged. At an output "
            "cell size above 10 m the class of the nearest source cell is kept, "
            "so small patches can be lost or exaggerated.",
        ],
        [
            "Note - areas",
            "Class areas are counted over the whole output grid (the bounding box "
            "of the area of interest plus two cells), not inside a polygon. In a "
            "projected CRS the cell area is the square of the cell size.",
        ],
    ]
    edition_rows = [
        [
            e.year,
            e.version,
            e.algorithm,
            e.accuracy,
            e.citation,
            e.attribution,
        ]
        for e in (EDITION_BY_YEAR[y] for y in selection.years)
    ]
    class_rows = [[c.code, c.name, c.colour, c.note] for c in CLASSES]
    tile_rows = []
    for rec in result.years:
        for t in rec.tiles.found:
            tile_rows.append([rec.year, t, "found", rec.route])
        for t in rec.tiles.missing:
            tile_rows.append([rec.year, t, "no tile (open sea)", ""])
    sections = [
        ("Run", ["Item", "Value"], run_rows),
        (
            "Editions",
            ["Year", "Version", "Algorithm", "Accuracy", "Citation", "Attribution"],
            edition_rows,
        ),
        ("Classes", ["Code", "Class", "Colour", "Note"], class_rows),
        ("Tiles", ["Year", "Tile", "Status", "Route"], tile_rows or [["", "", "", ""]]),
    ]
    if result.mode == "area":
        area_rows = []
        for rec in result.years:
            total = sum(rec.areas.values())
            for c in sorted(rec.counts):
                area_rows.append(
                    [
                        rec.year,
                        rec.file,
                        c,
                        class_name(c),
                        rec.counts[c],
                        _fmt(rec.areas.get(c)),
                        _fmt(100.0 * rec.areas.get(c, 0) / total if total else None, 2),
                    ]
                )
            area_rows.append(
                [rec.year, rec.file, NODATA, "No class", rec.nodata_cells, "", ""]
            )
        sections.append(
            (
                "Class areas",
                [
                    "Year",
                    "File",
                    "Code",
                    "Class",
                    "Cells",
                    "Area (km2)",
                    "Share of classified area (%)",
                ],
                area_rows,
            )
        )
    else:
        count_rows = []
        for rec in result.years:
            for c in sorted(rec.counts):
                count_rows.append([rec.year, c, class_name(c), rec.counts[c]])
            count_rows.append([rec.year, "", "No class", rec.nodata_cells])
        sections.append(
            ("Class counts", ["Year", "Code", "Class", "Sites"], count_rows)
        )
    sections.append(
        ("Warnings", ["Warning"], [[w] for w in result.warnings] or [["none"]])
    )
    return sections
