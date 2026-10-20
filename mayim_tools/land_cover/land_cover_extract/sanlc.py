"""South African National Land Cover (SANLC) for "Extract: Land Cover" -
the class legend, editions and the extraction from a local GeoTIFF. No QGIS.

The SANLC is published by the Department of Forestry, Fisheries and the
Environment (DFFE) as national 20 m GeoTIFFs (Sentinel-2), downloaded from
the DFFE e-GIS site; there is no public cloud-optimised copy, so the tool
reads a local copy (e.g. SA_NLC_2022_ALBERS.tif). The 2018, 2020 and 2022
editions share the same 73 gazetted classes (codes checked against the 2022
attribute table, October 2026); value 0 is no data (outside South Africa).
Only the window around the area of interest is read (a VRT, not the whole
national file), and it is warped onto the output grid with nearest
neighbour, so class codes are never averaged.

Class names and the SALCC level 1 / 2 groups are those of the SANLC 2022
attribute table (two spelling slips corrected); colours are the published
SANLC 2018 legend colours.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.grid import TargetGrid
from mayim_tools.soil._common.jobs import _never_cancelled, _noop

PRODUCT = "SANLC"
PRODUCT_LONG = "South African National Land Cover (SANLC)"
RESOLUTION_M = 20
NODATA = 0
PUBLISHER = "Department of Forestry, Fisheries and the Environment (DFFE)"
DOWNLOAD = "https://egis.environment.gov.za"
LICENCE = "DFFE data - acknowledge the source (see the DFFE terms of use)"


@dataclass(frozen=True)
class Edition:
    year: int
    accuracy: str
    citation: str


EDITIONS = [
    Edition(
        2022,
        "84.22 % overall accuracy (DFFE accuracy assessment of the 73-class map)",
        "Department of Forestry, Fisheries and the Environment (2024). South "
        "African National Land-Cover 2022 (SANLC 2022), 20 m, 73 classes. "
        "Released 23 May 2024. Pretoria. https://egis.environment.gov.za",
    ),
    Edition(
        2020,
        "see the DFFE accuracy assessment report for the 2020 map",
        "Department of Forestry, Fisheries and the Environment. South African "
        "National Land-Cover 2020 (SANLC 2020), 20 m, 73 classes. Pretoria. "
        "https://egis.environment.gov.za",
    ),
    Edition(
        2018,
        "see the DEA accuracy assessment report for the 2018 map",
        "Department of Environmental Affairs. South African National Land-Cover "
        "2018 (SANLC 2018), 20 m, 73 classes. Pretoria. "
        "https://egis.environment.gov.za",
    ),
]
EDITION_BY_YEAR = {e.year: e for e in EDITIONS}
DEFAULT_YEAR = 2022


@dataclass(frozen=True)
class LandClass:
    code: int
    name: str
    colour: str
    salcc1: str
    salcc2: str


CLASSES = [
    LandClass(1, "Contiguous (indigenous) forest", "#0b3d0b", "Forested land", "Natural Wooded Land"),
    LandClass(2, "Contiguous low forest & thicket", "#065106", "Forested land", "Natural Wooded Land"),
    LandClass(3, "Dense forest & woodland", "#005f00", "Forested land", "Natural Wooded Land"),
    LandClass(4, "Open woodland", "#008500", "Forested land", "Natural Wooded Land"),
    LandClass(5, "Contiguous & dense plantation forest", "#f74006", "Forested land", "Planted Forest"),
    LandClass(6, "Open & sparse plantation forest", "#f9764d", "Forested land", "Planted Forest"),
    LandClass(7, "Temporary unplanted (clear-felled) plantation forest", "#f9906c", "Forested land", "Planted Forest"),
    LandClass(8, "Low shrubland (other)", "#b8abd1", "Shrubland", "Shrubs"),
    LandClass(9, "Low shrubland (fynbos)", "#8fab39", "Shrubland", "Karoo & Fynbos Shrubland"),
    LandClass(10, "Low shrubland (succulent karoo)", "#ac92c5", "Shrubland", "Karoo & Fynbos Shrubland"),
    LandClass(11, "Low shrubland (nama karoo)", "#ac9cda", "Shrubland", "Karoo & Fynbos Shrubland"),
    LandClass(12, "Sparsely wooded grassland", "#85d285", "Grassland", "Natural Grassland"),
    LandClass(13, "Natural grassland", "#d2b485", "Grassland", "Natural Grassland"),
    LandClass(14, "Natural rivers", "#00009f", "Waterbodies", "Natural Waterbodies"),
    LandClass(15, "Natural estuaries & lagoons", "#041fa7", "Waterbodies", "Natural Waterbodies"),
    LandClass(16, "Natural ocean & coastal", "#0639ab", "Waterbodies", "Natural Waterbodies"),
    LandClass(17, "Natural lakes", "#0d50ac", "Waterbodies", "Natural Waterbodies"),
    LandClass(18, "Natural pans (flooded @ observation times)", "#125fac", "Waterbodies", "Natural Waterbodies"),
    LandClass(19, "Artificial dams (including canals)", "#1373b4", "Waterbodies", "Artificial Waterbodies"),
    LandClass(20, "Artificial sewage ponds", "#1d81b6", "Waterbodies", "Artificial Waterbodies"),
    LandClass(21, "Artificial flooded mine pits", "#1f8eb8", "Waterbodies", "Artificial Waterbodies"),
    LandClass(22, "Herbaceous wetlands (currently mapped)", "#06dedc", "Wetlands", "Herbaceous Wetlands"),
    LandClass(23, "Herbaceous wetlands (previously mapped)", "#06e0d0", "Wetlands", "Herbaceous Wetlands"),
    LandClass(24, "Mangrove wetlands", "#9f1fec", "Wetlands", "Woody Wetlands"),
    LandClass(25, "Natural rock surfaces", "#ffffe0", "Barren land", "Consolidated"),
    LandClass(26, "Dry pans", "#dcdac5", "Barren land", "Consolidated"),
    LandClass(27, "Eroded lands", "#f9e0e0", "Barren land", "Unconsolidated"),
    LandClass(28, "Sand dunes (terrestrial)", "#f9f9c5", "Barren land", "Unconsolidated"),
    LandClass(29, "Coastal sand & dunes", "#f9f9a7", "Barren land", "Unconsolidated"),
    LandClass(30, "Bare riverbed material", "#cdd2e0", "Barren land", "Unconsolidated"),
    LandClass(31, "Other bare", "#f5f5c8", "Barren land", "Unconsolidated"),
    LandClass(32, "Cultivated commercial permanent orchards", "#a62c39", "Cultivated", "Permanent Crops"),
    LandClass(33, "Cultivated commercial permanent vines", "#b31f5c", "Cultivated", "Permanent Crops"),
    LandClass(34, "Cultivated commercial sugarcane pivot irrigated", "#db0000", "Cultivated", "Permanent Crops"),
    LandClass(35, "Cultivated commercial permanent pineapples", "#9f3978", "Cultivated", "Permanent Crops"),
    LandClass(36, "Cultivated commercial sugarcane non-pivot", "#ff0000", "Cultivated", "Permanent Crops"),
    LandClass(37, "Cultivated emerging farmer sugarcane non-pivot", "#f64d6c", "Cultivated", "Permanent Crops"),
    LandClass(38, "Commercial annual crops pivot irrigated", "#381a12", "Cultivated", "Temporary Crops"),
    LandClass(39, "Commercial annual crops non-pivot irrigated", "#521f1c", "Cultivated", "Temporary Crops"),
    LandClass(40, "Commercial annual crops rain-fed / dryland", "#85402c", "Cultivated", "Temporary Crops"),
    LandClass(41, "Subsistence / small-scale annual crops", "#c5735f", "Cultivated", "Temporary Crops"),
    LandClass(42, "Fallow land & old fields (trees)", "#c1436c", "Cultivated", "Fallow Lands & Old Fields"),
    LandClass(43, "Fallow land & old fields (bush)", "#c55e82", "Cultivated", "Fallow Lands & Old Fields"),
    LandClass(44, "Fallow land & old fields (grass)", "#d27592", "Cultivated", "Fallow Lands & Old Fields"),
    LandClass(45, "Fallow land & old fields (bare)", "#e0aab8", "Cultivated", "Fallow Lands & Old Fields"),
    LandClass(46, "Fallow land & old fields (low shrub)", "#db90a9", "Cultivated", "Fallow Lands & Old Fields"),
    LandClass(47, "Residential formal (tree)", "#ecdb0f", "Built-up", "Residential"),
    LandClass(48, "Residential formal (bush)", "#f6ec13", "Built-up", "Residential"),
    LandClass(49, "Residential formal (low veg / grass)", "#f9f81f", "Built-up", "Residential"),
    LandClass(50, "Residential formal (bare)", "#ffff29", "Built-up", "Residential"),
    LandClass(51, "Residential informal (tree)", "#ec82ec", "Built-up", "Residential"),
    LandClass(52, "Residential informal (bush)", "#f691e0", "Built-up", "Residential"),
    LandClass(53, "Residential informal (low veg / grass)", "#f99fcf", "Built-up", "Residential"),
    LandClass(54, "Residential informal (bare)", "#ffc5cf", "Built-up", "Residential"),
    LandClass(55, "Village scattered (bare & low veg/ grass combo)", "#ecc500", "Built-up", "Village"),
    LandClass(56, "Village dense (bare & low veg / grass combo)", "#ffd91f", "Built-up", "Village"),
    LandClass(57, "Smallholdings (tree)", "#ac7879", "Built-up", "Smallholdings"),
    LandClass(58, "Smallholdings (bush)", "#b89192", "Built-up", "Smallholdings"),
    LandClass(59, "Smallholdings (low veg / grass)", "#c49c9e", "Built-up", "Smallholdings"),
    LandClass(60, "Smallholdings (bare)", "#d2b8b8", "Built-up", "Smallholdings"),
    LandClass(61, "Urban recreational fields (tree)", "#bfff00", "Built-up", "Urban Vegetation"),
    LandClass(62, "Urban recreational fields (bush)", "#33ff33", "Built-up", "Urban Vegetation"),
    LandClass(63, "Urban recreational fields (grass)", "#66ff66", "Built-up", "Urban Vegetation"),
    LandClass(64, "Urban recreational fields (bare)", "#99ff99", "Built-up", "Urban Vegetation"),
    LandClass(65, "Commercial", "#c49f0d", "Built-up", "Commercial"),
    LandClass(66, "Industrial", "#8f8506", "Built-up", "Industrial"),
    LandClass(67, "Roads & rails (major linear)", "#f9dd03", "Built-up", "Transport"),
    LandClass(68, "Mines: surface infrastructure", "#e6e600", "Mines & quarries", "Surface Infrastructure"),
    LandClass(69, "Mines: extraction pits, quarries", "#b30606", "Mines & quarries", "Extraction Sites"),
    LandClass(70, "Mines: salt mines", "#c50606", "Mines & quarries", "Extraction Sites"),
    LandClass(71, "Mine: tailings and resource dumps", "#d21d1a", "Mines & quarries", "Waste & Resource Dumps"),
    LandClass(72, "Land-fills", "#f95479", "Mines & quarries", "Waste & Resource Dumps"),
    LandClass(73, "Fallow land & old fields (wetlands)", "#6ce7dc", "Cultivated", "Fallow Lands & Old Fields"),
]  # fmt: skip
CLASS_BY_CODE = {c.code: c for c in CLASSES}


def class_name(code) -> str:
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return ""
    c = CLASS_BY_CODE.get(int(code))
    return c.name if c else f"Unknown ({int(code)})"


def output_file_name(year: int) -> str:
    return f"sanlc_{year}_map.tif"


def band_description(year: int) -> str:
    return f"landcover_sanlc_{year}_20m (class)"


def year_in_name(path: str) -> int | None:
    """An edition year (2013/14 to 2030) found in the file name, if any."""
    m = re.findall(r"(?<!\d)(20[1-3]\d)(?!\d)", os.path.basename(path))
    return int(m[-1]) if m else None


def edition_warnings(path: str, year: int) -> list[str]:
    found = year_in_name(path)
    if found is not None and found != year:
        return [
            f"The file name suggests SANLC {found}, but SANLC {year} was chosen: "
            "the citation and metadata follow the choice; check it."
        ]
    if found is None:
        return [
            f"The file name does not show the SANLC year; SANLC {year} is "
            "assumed for the citation and metadata."
        ]
    return []


# ----------------------------------------------------------------------
# GDAL defaults (replaced by fakes in the tests)
# ----------------------------------------------------------------------


def gdal_check_raster(path: str) -> dict:
    """Basic facts of the local raster; raises if it cannot be read."""
    from osgeo import gdal

    gdal.UseExceptions()
    if not path or not os.path.exists(path):
        raise SoilDataError(f"SANLC raster not found: {path}")
    ds = gdal.Open(path)
    try:
        b = ds.GetRasterBand(1)
        gt = ds.GetGeoTransform()
        return {
            "type": gdal.GetDataTypeName(b.DataType),
            "nodata": b.GetNoDataValue(),
            "res": abs(gt[1]),
            "size": (ds.RasterXSize, ds.RasterYSize),
            "crs": ds.GetProjection(),
        }
    finally:
        ds = None


def gdal_warp_local(path: str, grid: TargetGrid, cancel_fn) -> np.ndarray:
    """The window of a local class raster warped onto ``grid`` (nearest
    neighbour, UInt8, NODATA outside). The national file is opened through
    a VRT, so only the blocks the grid needs are read."""
    from osgeo import gdal

    gdal.UseExceptions()

    def callback(_fraction, _msg, _data):
        return 0 if cancel_fn() else 1

    src = gdal.Open(path)
    vrt = gdal.Translate("", src, format="VRT", bandList=[1])
    ds = gdal.Warp(
        "",
        vrt,
        format="MEM",
        outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
        width=grid.width,
        height=grid.height,
        dstSRS=grid.crs_wkt,
        resampleAlg="near",
        outputType=gdal.GDT_Byte,
        srcNodata=NODATA,
        dstNodata=NODATA,
        callback=callback,
    )
    if ds is None:
        if cancel_fn():
            raise InterruptedError("Cancelled by user")
        raise SoilDataError("GDAL could not warp the SANLC raster.")
    try:
        return ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    finally:
        ds = None
        vrt = None
        src = None


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------


@dataclass
class SanlcRecord:
    year: int
    file: str
    source: str
    counts: dict = field(default_factory=dict)  # code -> cells (or sites)
    areas: dict = field(default_factory=dict)  # code -> km2
    nodata_cells: int = 0
    total_cells: int = 0


@dataclass
class SanlcResult:
    mode: str
    year: int
    source: str
    info: dict = field(default_factory=dict)
    files: list = field(default_factory=list)
    records: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    seconds: float = 0.0


def _log(result, log_fn, msg):
    result.warnings.append(msg)
    log_fn(msg)


def _check(result, info, log_fn):
    if info.get("type") not in (None, "Byte", "UInt8", "Int16", "UInt16"):
        _log(
            result,
            log_fn,
            f"The SANLC raster is stored as {info['type']}; class codes 1-73 "
            "are expected (is this the class map?).",
        )
    res = info.get("res")
    if res and 1.0 < res < 1000 and abs(res - RESOLUTION_M) > 1.0:
        _log(
            result,
            log_fn,
            f"The SANLC raster has {res:g}-unit cells, not the published 20 m "
            "(a resampled copy?).",
        )


def extract_area(
    year: int,
    source_path: str,
    grid: TargetGrid,
    out_dir: str,
    *,
    geographic: bool = False,
    check_fn: Callable | None = None,
    warp_fn: Callable | None = None,
    write_fn: Callable | None = None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> SanlcResult:
    """One class GeoTIFF of the SANLC window. ``write_fn(path, grid, arr,
    year)``."""
    from .core import class_areas, row_cell_areas_km2

    if year not in EDITION_BY_YEAR:
        raise SoilDataError(f"No SANLC edition {year}.")
    if write_fn is None:
        raise SoilDataError("No raster writer supplied.")
    t0 = datetime.now(UTC)
    result = SanlcResult("area", year, source_path)
    for w in edition_warnings(source_path, year):
        _log(result, log_fn, w)
    result.info = (check_fn or gdal_check_raster)(source_path)
    _check(result, result.info, log_fn)
    progress_fn(0.1, f"Reading SANLC {year} from {os.path.basename(source_path)}")
    arr = (warp_fn or gdal_warp_local)(source_path, grid, cancel_fn)
    if not np.any(arr != NODATA):
        raise SoilDataError(
            "The SANLC raster has no land-cover classes in the area of interest: "
            "the area lies outside South Africa or outside this raster (check the "
            "file and the area's CRS)."
        )
    progress_fn(0.8, "Writing")
    path = os.path.join(out_dir, output_file_name(year))
    write_fn(path, grid, arr, year)
    counts, areas = class_areas(arr, row_cell_areas_km2(grid, geographic))
    rec = SanlcRecord(
        year,
        os.path.basename(path),
        source_path,
        counts={c: n for c, n in counts.items() if c != NODATA},
        areas={c: a for c, a in areas.items() if c != NODATA},
        nodata_cells=counts.get(NODATA, 0),
        total_cells=int(arr.size),
    )
    unknown = sorted(c for c in rec.counts if c not in CLASS_BY_CODE)
    if unknown:
        _log(
            result,
            log_fn,
            f"SANLC {year}: codes {unknown} are not SANLC classes (1-73) - is "
            "this the 73-class map?",
        )
    if rec.nodata_cells:
        share = 100.0 * rec.nodata_cells / rec.total_cells
        if share >= 0.05:
            _log(
                result,
                log_fn,
                f"SANLC {year}: {share:.1f} % of the grid has no class (outside "
                "South Africa or the raster).",
            )
    result.records.append(rec)
    result.files.append(path)
    result.seconds = (datetime.now(UTC) - t0).total_seconds()
    progress_fn(1.0, "Done")
    return result


POINT_COLUMNS = [
    "Site",
    "Longitude",
    "Latitude",
    "Product",
    "Year",
    "ClassCode",
    "ClassName",
    "SALCC_1",
    "SALCC_2",
    "Resolution_m",
    "Source",
]


def extract_points(
    year: int,
    source_path: str,
    sites: Sequence,
    *,
    check_fn: Callable | None = None,
    sample_fn: Callable | None = None,
    progress_fn: Callable = _noop,
    log_fn: Callable = _noop,
    cancel_fn: Callable = _never_cancelled,
) -> SanlcResult:
    """Nearest-cell SANLC class at each site (sites outside the raster or on
    no data get an empty class, never zero)."""
    if year not in EDITION_BY_YEAR:
        raise SoilDataError(f"No SANLC edition {year}.")
    if not sites:
        raise SoilDataError("No points to sample.")
    if sample_fn is None:
        from mayim_tools.soil._common.gdal_io import gdal_sample_points as sample_fn
    t0 = datetime.now(UTC)
    result = SanlcResult("points", year, source_path)
    for w in edition_warnings(source_path, year):
        _log(result, log_fn, w)
    result.info = (check_fn or gdal_check_raster)(source_path)
    _check(result, result.info, log_fn)
    if cancel_fn():
        raise InterruptedError("Cancelled by user")
    values = list(sample_fn(source_path, [(s.lon, s.lat) for s in sites]))
    counts: dict[int, int] = {}
    n_missing = 0
    for s, v in zip(sites, values, strict=True):
        ok = v is not None and math.isfinite(v) and int(v) != NODATA
        code = int(v) if ok else None
        c = CLASS_BY_CODE.get(code) if code is not None else None
        if code is None:
            n_missing += 1
        else:
            counts[code] = counts.get(code, 0) + 1
        result.rows.append(
            {
                "Site": s.label,
                "Longitude": s.lon,
                "Latitude": s.lat,
                "Product": PRODUCT,
                "Year": year,
                "ClassCode": "" if code is None else code,
                "ClassName": class_name(code),
                "SALCC_1": c.salcc1 if c else "",
                "SALCC_2": c.salcc2 if c else "",
                "Resolution_m": RESOLUTION_M,
                "Source": os.path.basename(source_path),
            }
        )
    result.records.append(
        SanlcRecord(
            year,
            "",
            source_path,
            counts=counts,
            nodata_cells=n_missing,
            total_cells=len(sites),
        )
    )
    if n_missing:
        _log(
            result,
            log_fn,
            f"SANLC {year}: {n_missing} of {len(sites)} site(s) have no class "
            "(outside South Africa or the raster); written as empty cells.",
        )
    result.seconds = (datetime.now(UTC) - t0).total_seconds()
    progress_fn(1.0, "Done")
    return result


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


def _fmt(x, nd=3):
    return "" if x is None else round(float(x), nd)


def build_metadata(result: SanlcResult, settings: dict, gdal_version_text="", tool=""):
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    ed = EDITION_BY_YEAR[result.year]
    info = result.info or {}
    run_rows = [
        ["Tool", tool],
        ["Run time", now],
        ["Duration (s)", round(result.seconds, 1)],
        ["Mode", result.mode],
        ["Product", f"{PRODUCT_LONG} {result.year} (73 classes, 20 m)"],
        ["Publisher", PUBLISHER],
        ["Source file", result.source],
        ["Source raster", f"{info.get('type', '')}, cell {info.get('res', '')}"],
        ["Download", DOWNLOAD],
        ["Accuracy", ed.accuracy],
        ["Citation", ed.citation],
        ["Licence", LICENCE],
        ["GDAL version", gdal_version_text],
    ]
    for key, value in settings.items():
        run_rows.append([key, value])
    run_rows += [
        ["Output nodata", f"{NODATA} (outside South Africa or the raster)"],
        [
            "Note - editions",
            "SANLC 2018, 2020 and 2022 use the same 73 classes, but were mapped "
            "with different imagery and methods: part of a difference between "
            "editions is a change of method, so use them as a change map with "
            "care.",
        ],
        [
            "Note - accuracy",
            "Accuracy is for the map as a whole; the tree / bush / grass / bare "
            "sub-classes of urban land and small features under 20 m are the "
            "least certain.",
        ],
        [
            "Note - resampling",
            "Nearest neighbour only: class codes are never averaged. At an output "
            "cell other than 20 m (or in another CRS than the source) each cell "
            "takes the class of the nearest source cell.",
        ],
        [
            "Note - areas",
            "Class areas are counted over the whole output grid (the bounding box "
            "of the area of interest plus two cells), not inside a polygon.",
        ],
    ]
    class_rows = [[c.code, c.name, c.salcc1, c.salcc2, c.colour] for c in CLASSES]
    sections = [
        ("Run", ["Item", "Value"], run_rows),
        ("Classes", ["Code", "Class", "SALCC_1", "SALCC_2", "Colour"], class_rows),
    ]
    rec = result.records[0] if result.records else None
    if result.mode == "area" and rec is not None:
        total = sum(rec.areas.values())
        rows = []
        for c in sorted(rec.counts):
            k = CLASS_BY_CODE.get(c)
            rows.append(
                [
                    c,
                    class_name(c),
                    k.salcc1 if k else "",
                    rec.counts[c],
                    _fmt(rec.areas.get(c)),
                    _fmt(100.0 * rec.areas.get(c, 0) / total if total else None, 2),
                ]
            )
        rows.append([NODATA, "No class", "", rec.nodata_cells, "", ""])
        sections.append(
            (
                "Class areas",
                [
                    "Code",
                    "Class",
                    "SALCC_1",
                    "Cells",
                    "Area (km2)",
                    "Share of classified area (%)",
                ],
                rows,
            )
        )
        groups: dict[str, float] = {}
        for c, a in rec.areas.items():
            k = CLASS_BY_CODE.get(c)
            g = k.salcc1 if k else "Unknown"
            groups[g] = groups.get(g, 0.0) + a
        sections.append(
            (
                "SALCC level 1 areas",
                ["Group", "Area (km2)", "Share of classified area (%)"],
                [
                    [g, _fmt(a), _fmt(100.0 * a / total if total else None, 2)]
                    for g, a in sorted(groups.items(), key=lambda kv: -kv[1])
                ],
            )
        )
    elif rec is not None:
        rows = [[c, class_name(c), rec.counts[c]] for c in sorted(rec.counts)]
        rows.append(["", "No class", rec.nodata_cells])
        sections.append(("Class counts", ["Code", "Class", "Sites"], rows))
    sections.append(
        ("Warnings", ["Warning"], [[w] for w in result.warnings] or [["none"]])
    )
    return sections
