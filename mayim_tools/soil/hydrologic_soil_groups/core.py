"""Hydrologic soil groups from the Regional soil parameterisation outputs.
No QGIS (GDAL for reading and writing only).

Inputs: the Ksat P5 / P50 / P95 maps of Regional soil parameterisation
(rsp_ksat.tif, three layers), the depth to a water impermeable layer
(SoilGrids 2017 depth to bedrock, a raster, a constant or none) and the
depth to the high water table (a raster, a constant or none).

Methods:
- NEH630: USDA-NRCS NEH Part 630 Chapter 7 (2009), Table 7-1 (neh630.py);
- SCSSA: SCS-SA groups (Schulze, Schmidt & Smithers, 2004, Table 2.1)
  with the intermediate groups A/B, B/C, C/D (scs_sa.py).

Per cell and method: the probability of each group (probability.py), the
recommended group (that of the median Ksat; intermediate where straddling for
SCS-SA) with its confidence, and the groups at the Ksat P5 (more runoff),
P50 and P95 (less runoff).
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.export import write_metadata_csv
from mayim_tools.soil._common.grid import TargetGrid

from . import neh630, scs_sa
from . import probability as prob_mod

TOOL_NAME = "Hydrologic soil groups"
TOOL_VERSION = "0.1.0"

LAYERS = ("0-30cm", "30-60cm", "60-100cm")
TAGS = ("P05", "P50", "P95")
KSAT_FILE = "rsp_ksat.tif"
INPUTS_FILE = "rsp_inputs_P50.tif"
REGIONAL_META = "rsp_metadata.csv"
BEDROCK_FILE = "soilgrids2017_depth_to_bedrock.tif"
DEEP_CM = 1.0e6  # "no impermeable layer / no water table within reach"

METHODS = {
    "NEH630": "USDA-NRCS NEH 630 Chapter 7 (2009)",
    "SCSSA": "SCS-SA (Schulze, Schmidt & Smithers, 2004)",
}
METHOD_SHORT = {"NEH630": "neh630", "SCSSA": "scssa"}

BEDROCK_SOURCES = {
    "BDRICM": "SoilGrids 2017: depth to R horizon (BDRICM, censored at 200 cm)",
    "BDTICM": "SoilGrids 2017: absolute depth to bedrock (BDTICM)",
    "raster": "User raster (depth in m)",
    "constant": "Constant depth",
    "none": "None (no impermeable layer within 100 cm assumed)",
}
WATER_SOURCES = {
    "none": "None (water table deeper than 100 cm assumed)",
    "raster": "User raster (depth in m)",
    "constant": "Constant depth",
}

CONF_LIMITS = (80.0, 60.0)  # % : High >= 80, Medium 60-80, Low < 60
CONF_LABELS = {1: "High", 2: "Medium", 3: "Low"}
CONF_TEXT = {
    1: "the recommended group holds 80 % or more of the probability",
    2: "the recommended group holds 60-80 % of the probability",
    3: "the recommended group holds less than 60 % of the probability",
}

# NEH 630 typical textures (Chapter 7 group descriptions), surface layer
TEXTURE_RULES = (
    (neh630.A, "clay < 10 % and sand > 90 %"),
    (neh630.B, "clay 10-20 % and sand 50-90 %"),
    (neh630.C, "clay 20-40 % and sand < 50 %"),
    (neh630.D, "clay > 40 % and sand < 50 %"),
)

ALL_LABEL = {**neh630.GROUP_LABEL, **scs_sa.GROUP_LABEL}
ALL_COLOUR = {**neh630.GROUP_COLOUR, **scs_sa.GROUP_COLOUR}


def labels_for(method: str) -> dict:
    return neh630.GROUP_LABEL if method == "NEH630" else scs_sa.GROUP_LABEL


def colours_for(method: str) -> dict:
    return neh630.GROUP_COLOUR if method == "NEH630" else scs_sa.GROUP_COLOUR


# ----------------------------------------------------------------------
# Settings and results
# ----------------------------------------------------------------------


@dataclass
class RunSettings:
    regional_folder: str
    out_dir: str
    methods: tuple = ("NEH630", "SCSSA")
    bedrock_source: str = "BDRICM"
    sg_folder: str = ""
    bedrock_raster: str = ""
    bedrock_constant_m: float = 2.0
    water_source: str = "none"
    water_raster: str = ""
    water_constant_m: float = 2.0
    sa_adjust_shallow: bool = True
    sa_adjust_water: bool = True
    zones: list = field(default_factory=list)  # [(name, wkt)]
    write_report: bool = True


@dataclass
class MethodResult:
    code: str
    prob: np.ndarray  # (4, rows, cols) A, B, C, D (fraction)
    recommended: np.ndarray
    confidence: np.ndarray  # %
    conf_class: np.ndarray
    at: dict  # tag -> group codes
    steps: np.ndarray | None = None  # SCS-SA adjustment steps


@dataclass
class RunResult:
    settings: RunSettings
    grid: TargetGrid
    zone_raster: np.ndarray | None = None
    zone_names: list = field(default_factory=list)
    mask: np.ndarray | None = None
    ksat: dict = field(default_factory=dict)  # tag -> layer -> array
    impermeable_cm: np.ndarray | None = None
    water_cm: np.ndarray | None = None
    case: np.ndarray | None = None
    texture: np.ndarray | None = None  # NEH typical-texture group (0 none)
    methods: dict = field(default_factory=dict)
    zone_rows: list = field(default_factory=list)
    cell_area_km2: float = 0.0
    regional_meta: dict = field(default_factory=dict)
    input_notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    files: list = field(default_factory=list)
    report_path: str = ""
    run_time_utc: str = ""
    seconds: float = 0.0


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


def _gdal():
    from osgeo import gdal

    gdal.UseExceptions()
    return gdal


def grid_of(path: str) -> TargetGrid:
    gdal = _gdal()
    ds = gdal.Open(path)
    try:
        gt = ds.GetGeoTransform()
        if gt[2] != 0 or gt[4] != 0:
            raise SoilDataError(f"Rotated rasters are not supported: {path}")
        res = gt[1]
        return TargetGrid(
            gt[0],
            gt[3] + gt[5] * ds.RasterYSize,
            gt[0] + gt[1] * ds.RasterXSize,
            gt[3],
            res,
            ds.GetProjection(),
        )
    finally:
        ds = None


def read_bands(path: str) -> dict:
    """{description: array (float64, NaN for nodata)}."""
    gdal = _gdal()
    ds = gdal.Open(path)
    try:
        out = {}
        for i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(i)
            arr = band.ReadAsArray().astype(np.float64)
            nd = band.GetNoDataValue()
            if nd is not None:
                arr[arr == nd] = np.nan
            out[band.GetDescription()] = arr
        return out
    finally:
        ds = None


def read_band_to_grid(path: str, grid: TargetGrid, band: int = 1) -> np.ndarray:
    """One band of any raster, warped (nearest neighbour) onto ``grid``."""
    gdal = _gdal()
    src = gdal.Open(path)
    if src is None:
        raise SoilDataError(f"Cannot open {path}")
    try:
        one = gdal.Translate("", src, format="MEM", bandList=[band])
        nd = src.GetRasterBand(band).GetNoDataValue()
        ds = gdal.Warp(
            "",
            one,
            format="MEM",
            outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
            width=grid.width,
            height=grid.height,
            dstSRS=grid.crs_wkt,
            resampleAlg="near",
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


def read_regional(folder: str) -> tuple[TargetGrid, dict, dict, dict]:
    """Grid, Ksat quantiles {tag: {layer: arr}}, surface texture
    {'sand','clay'} (may be empty) and the regional run metadata."""
    if not folder or not os.path.isdir(folder):
        raise SoilDataError(f"Folder not found: {folder}")
    path = os.path.join(folder, KSAT_FILE)
    if not os.path.isfile(path):
        raise SoilDataError(
            f"{KSAT_FILE} not found in {folder}. Point the tool at the output "
            "folder of 'Regional soil parameterisation'."
        )
    grid = grid_of(path)
    bands = read_bands(path)
    ksat = {tag: {} for tag in TAGS}
    for desc, arr in bands.items():
        name = desc.split(" ")[0]  # ksat_0-30cm_P50
        parts = name.split("_")
        if len(parts) == 3 and parts[0] == "ksat" and parts[2] in TAGS:
            ksat[parts[2]][parts[1]] = arr
    missing = [f"{lab} {tag}" for tag in TAGS for lab in LAYERS if lab not in ksat[tag]]
    if missing:
        raise SoilDataError(
            f"{KSAT_FILE} lacks Ksat bands for: {', '.join(missing)}. Re-run "
            "Regional soil parameterisation (all three layers are needed)."
        )
    texture = {}
    ipath = os.path.join(folder, INPUTS_FILE)
    if os.path.isfile(ipath):
        ib = read_bands(ipath)
        for var in ("sand", "clay"):
            key = next((d for d in ib if d.startswith(f"{var}_0-30cm_P50")), None)
            if key:
                texture[var] = ib[key]
    return grid, ksat, texture, read_regional_metadata(folder)


def read_regional_metadata(folder: str) -> dict:
    """Run rows of rsp_metadata.csv (Tool, Products, Methods, ...)."""
    path = os.path.join(folder, REGIONAL_META)
    out = {}
    if not os.path.isfile(path):
        return out
    section = None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0].startswith("#"):
                section = row[0][1:].strip()
                continue
            if section == "Run" and len(row) >= 2 and row[0] != "Item":
                out[row[0]] = row[1]
    return out


def _depth_layer(
    source: str, folder: str, raster: str, constant_m: float, grid, what: str
) -> tuple[np.ndarray, str, list]:
    """Depth (cm) on the grid, a description and notes."""
    notes = []
    shape = (grid.height, grid.width)
    if source == "none":
        return np.full(shape, DEEP_CM), WATER_SOURCES.get(source, source), notes
    if source == "constant":
        return (
            np.full(shape, float(constant_m) * 100.0),
            f"Constant {constant_m:g} m",
            notes,
        )
    if source == "raster":
        if not raster or not os.path.isfile(raster):
            raise SoilDataError(f"{what} raster not found: {raster}")
        arr = read_band_to_grid(raster, grid) * 100.0
        return arr, f"User raster {os.path.basename(raster)} (m)", notes
    if source in ("BDRICM", "BDTICM"):
        path = os.path.join(folder or "", BEDROCK_FILE)
        if not folder or not os.path.isfile(path):
            raise SoilDataError(
                f"{BEDROCK_FILE} not found in the SoilGrids folder ({folder}). "
                "Run 'Extract: SoilGrids 2.0' with 'Depth to bedrock (SoilGrids "
                "2017 archive)' ticked, or choose another source for the depth "
                "to an impermeable layer."
            )
        gdal = _gdal()
        ds = gdal.Open(path)
        band = None
        for i in range(1, ds.RasterCount + 1):
            if ds.GetRasterBand(i).GetDescription().startswith(source):
                band = i
                break
        ds = None
        if band is None:
            raise SoilDataError(f"{BEDROCK_FILE} has no {source} band.")
        arr = read_band_to_grid(path, grid, band)
        if source == "BDRICM":
            notes.append(
                "BDRICM is censored at 200 cm: 200 means no R horizon within 200 cm."
            )
        return arr, BEDROCK_SOURCES[source], notes
    raise SoilDataError(f"Unknown depth source: {source}")


def rasterise_zones(zones, grid):
    from mayim_tools.soil.regional_parameterisation.core import (
        rasterise_zones as _rz,
    )

    return _rz(zones, grid)


def cell_area_km2(grid):
    from mayim_tools.soil.regional_parameterisation.core import cell_area_km2 as _ca

    return _ca(grid)


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------


def conf_class(conf_pct: np.ndarray) -> np.ndarray:
    c = np.asarray(conf_pct, dtype=float)
    out = np.zeros(c.shape, dtype=np.int16)
    ok = np.isfinite(c)
    out[ok & (c >= CONF_LIMITS[0])] = 1
    out[ok & (c < CONF_LIMITS[0]) & (c >= CONF_LIMITS[1])] = 2
    out[ok & (c < CONF_LIMITS[1])] = 3
    return out


def texture_group(sand, clay) -> np.ndarray:
    """NEH 630 typical-texture group of the surface layer (0 = none)."""
    s = np.asarray(sand, dtype=float)
    c = np.asarray(clay, dtype=float)
    out = np.zeros(s.shape, dtype=np.int16)
    with np.errstate(invalid="ignore"):
        out[(c < 10) & (s > 90)] = neh630.A
        out[(c >= 10) & (c <= 20) & (s >= 50) & (s <= 90)] = neh630.B
        out[(c > 20) & (c <= 40) & (s < 50)] = neh630.C
        out[(c > 40) & (s < 50)] = neh630.D
    return out


def _letter_probability(prob: np.ndarray, letter: np.ndarray) -> np.ndarray:
    idx = np.clip(np.asarray(letter) - 1, 0, 3)
    p = np.take_along_axis(prob, idx[None], axis=0)[0]
    return np.where(np.asarray(letter) > 0, p, np.nan)


def _neh(result: RunResult) -> MethodResult:
    """Recommended group = group of the median Ksat (Table 7-1 applied to the
    P50); confidence = the probability of that group."""
    case = result.case
    prob = prob_mod.probabilities(result.ksat, case)
    at = {
        tag: neh630.classify({lab: result.ksat[tag][lab] for lab in LAYERS}, case)
        for tag in TAGS
    }
    rec = at["P50"].copy()
    conf = _letter_probability(prob, neh630.drained_letter(rec)) * 100.0
    return MethodResult("NEH630", prob, rec, conf, conf_class(conf), at)


def _scssa(result: RunResult) -> MethodResult:
    s = result.settings
    prob = scs_sa.probabilities(result.ksat)
    q = scs_sa.profile_quantiles(result.ksat)
    at = {tag: scs_sa.group_at(q[tag]) for tag in TAGS}
    rec, conf = scs_sa.recommended(prob, at["P50"])
    rec, steps = scs_sa.adjust(
        rec,
        result.impermeable_cm,
        result.water_cm,
        s.sa_adjust_shallow,
        s.sa_adjust_water,
    )
    for tag in TAGS:
        at[tag], _ = scs_sa.adjust(
            at[tag],
            result.impermeable_cm,
            result.water_cm,
            s.sa_adjust_shallow,
            s.sa_adjust_water,
        )
    conf = conf * 100.0
    return MethodResult("SCSSA", prob, rec, conf, conf_class(conf), at, steps)


def _apply_mask(mr: MethodResult, mask: np.ndarray) -> None:
    out = ~mask
    mr.prob[:, out] = np.nan
    mr.recommended[out] = 0
    mr.confidence[out] = np.nan
    mr.conf_class[out] = 0
    for tag in TAGS:
        mr.at[tag][out] = 0
    if mr.steps is not None:
        mr.steps[out] = 0


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------


def _noop(*_a, **_k):
    return None


def run(
    settings: RunSettings,
    log_fn: Callable = _noop,
    progress_fn: Callable = _noop,
    write_fn: Callable | None = None,
) -> RunResult:
    t0 = time.time()
    if not settings.methods:
        raise SoilDataError("Select at least one method.")
    unknown = [m for m in settings.methods if m not in METHODS]
    if unknown:
        raise SoilDataError(f"Unknown method(s): {', '.join(unknown)}")
    os.makedirs(settings.out_dir, exist_ok=True)

    grid, ksat, texture, meta = read_regional(settings.regional_folder)
    result = RunResult(settings=settings, grid=grid, ksat=ksat, regional_meta=meta)
    result.run_time_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    result.cell_area_km2 = cell_area_km2(grid)
    log_fn(
        f"Regional soil parameterisation outputs: {grid.width} x {grid.height} "
        f"cells at {grid.res:g}" + (f" ({meta['Tool']})" if meta.get("Tool") else "")
    )
    progress_fn(0.1, "Depths")

    imp, imp_desc, notes = _depth_layer(
        settings.bedrock_source,
        settings.sg_folder,
        settings.bedrock_raster,
        settings.bedrock_constant_m,
        grid,
        "Impermeable-layer depth",
    )
    wt, wt_desc, notes2 = _depth_layer(
        settings.water_source,
        "",
        settings.water_raster,
        settings.water_constant_m,
        grid,
        "Water-table depth",
    )
    result.input_notes = notes + notes2
    result.input_notes.insert(0, f"Depth to impermeable layer: {imp_desc}")
    result.input_notes.insert(1, f"Depth to water table: {wt_desc}")

    valid = np.isfinite(ksat["P50"]["0-30cm"])
    if settings.zones:
        result.zone_raster = rasterise_zones(settings.zones, grid)
        result.zone_names = [z[0] for z in settings.zones]
        valid &= result.zone_raster > 0
        if not valid.any():
            raise SoilDataError(
                "The zones do not overlap the Regional soil parameterisation outputs."
            )
    else:
        result.zone_raster = valid.astype(np.int32)
        result.zone_names = ["Area of interest"]
    if not valid.any():
        raise SoilDataError(f"{KSAT_FILE} has no valid Ksat values.")
    result.mask = valid

    n_imp_missing = int((valid & ~np.isfinite(imp)).sum())
    n_wt_missing = int((valid & ~np.isfinite(wt)).sum())
    if n_imp_missing:
        result.warnings.append(
            f"Depth to an impermeable layer missing in {n_imp_missing} cells; "
            "treated as deeper than 100 cm."
        )
    if n_wt_missing:
        result.warnings.append(
            f"Depth to the water table missing in {n_wt_missing} cells; treated "
            "as deeper than 100 cm."
        )
    imp = np.where(np.isfinite(imp), imp, DEEP_CM)
    wt = np.where(np.isfinite(wt), wt, DEEP_CM)
    result.impermeable_cm = np.where(valid, imp, np.nan)
    result.water_cm = np.where(valid, wt, np.nan)
    result.case = np.where(valid, neh630.case_of(imp, wt), 0).astype(np.int16)

    if texture.get("sand") is not None and texture.get("clay") is not None:
        result.texture = np.where(
            valid, texture_group(texture["sand"], texture["clay"]), 0
        )
    else:
        result.input_notes.append(f"{INPUTS_FILE} not found: no texture check.")

    progress_fn(0.3, "Classifying")
    for m in settings.methods:
        mr = _neh(result) if m == "NEH630" else _scssa(result)
        _apply_mask(mr, valid)
        result.methods[m] = mr
        log_fn(f"{METHODS[m]}: classified.")

    progress_fn(0.6, "Summaries")
    _zone_summary(result)
    for r in result.zone_rows:
        log_fn(
            f"{r['Zone']} - {r['Method']}: dominant group {r['Dominant group']} "
            f"({r['Dominant group share (%)']:.0f} % of the zone), mean confidence "
            f"{r['Mean confidence (%)']:.0f} %"
        )
    progress_fn(0.7, "Writing")
    if write_fn is not None:
        _write_rasters(result, write_fn)
    _write_tables(result)
    if settings.write_report:
        try:
            from . import report

            result.report_path = report.write_report(result)
            result.files.append(result.report_path)
        except ImportError as exc:
            result.warnings.append(
                f"Word report not written ({exc}); install python-docx and "
                "matplotlib in QGIS's Python."
            )
        except Exception as exc:  # noqa: BLE001 - the report must not fail a run
            result.warnings.append(f"Word report failed: {exc}")
    result.seconds = time.time() - t0
    _write_metadata(result)
    progress_fn(1.0, "Done")
    return result


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------


def _share(sel_codes, code) -> float:
    n = sel_codes.size
    return 100.0 * float((sel_codes == code).sum()) / n if n else math.nan


def _mode(codes) -> int:
    codes = codes[codes > 0]
    if codes.size == 0:
        return 0
    vals, counts = np.unique(codes, return_counts=True)
    # ties -> more runoff (later in the runoff order)
    order = {c: i for i, c in enumerate(scs_sa.STEP_CODES)}
    order.update({neh630.AD: 7, neh630.BD: 8, neh630.CD: 9})
    best = max(zip(counts, [order.get(int(v), 0) for v in vals], vals, strict=True))
    return int(best[2])


def _zone_summary(result: RunResult) -> None:
    rows = []
    for z, zname in enumerate(result.zone_names, start=1):
        zm = result.mask & (result.zone_raster == z)
        n = int(zm.sum())
        if n == 0:
            continue
        for m, mr in result.methods.items():
            labels = labels_for(m)
            rec = mr.recommended[zm]
            dom = _mode(rec)
            row = {
                "Zone": zname,
                "Method": m,
                "Cells": n,
                "Area (km2)": n * result.cell_area_km2,
                "Dominant group": labels.get(dom, "-"),
                "Dominant group share (%)": _share(rec, dom),
                "Mean confidence (%)": float(np.nanmean(mr.confidence[zm])),
            }
            for code, lab in labels.items():
                row[f"Share {lab} (%)"] = _share(rec, code)
            for i, lab in enumerate("ABCD"):
                row[f"Mean P({lab}) (%)"] = 100.0 * float(np.nanmean(mr.prob[i][zm]))
            for k, lab in CONF_LABELS.items():
                row[f"Confidence {lab} (%)"] = _share(mr.conf_class[zm], k)
            row["Most common group at Ksat P05 (more runoff)"] = labels.get(
                _mode(mr.at["P05"][zm]), "-"
            )
            row["Most common group at Ksat P50"] = labels.get(
                _mode(mr.at["P50"][zm]), "-"
            )
            row["Most common group at Ksat P95 (less runoff)"] = labels.get(
                _mode(mr.at["P95"][zm]), "-"
            )
            with np.errstate(invalid="ignore"):
                row["Impermeable layer < 50 cm (%)"] = 100.0 * float(
                    (result.impermeable_cm[zm] < 50).mean()
                )
                row["Water table < 60 cm (%)"] = 100.0 * float(
                    (result.water_cm[zm] < 60).mean()
                )
            if mr.steps is not None:
                row["Cells adjusted down (%)"] = 100.0 * float(
                    (mr.steps[zm] > 0).mean()
                )
            if result.texture is not None and m == "NEH630":
                tex = result.texture[zm]
                ind = tex > 0
                row["Texture indicates a group (%)"] = 100.0 * float(ind.mean())
                if ind.any():
                    mine = neh630.drained_letter(rec[ind])
                    row["Texture agrees with recommended (%)"] = 100.0 * float(
                        (mine == tex[ind]).mean()
                    )
                else:
                    row["Texture agrees with recommended (%)"] = math.nan
            rows.append(row)
    result.zone_rows = rows


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------


def output_layers(result: RunResult) -> list[dict]:
    """Rasters written, for loading and the layer file:
    {path, name, main, classes (value, label, colour) or []}."""
    out = []
    folder = result.settings.out_dir
    for m in result.methods:
        short = METHOD_SHORT[m]
        cls = [(c, lab, colours_for(m)[c]) for c, lab in labels_for(m).items()]
        out.append(
            {
                "path": os.path.join(folder, f"hsg_{short}_recommended.tif"),
                "name": f"HSG {m} recommended",
                "main": True,
                "classes": cls,
            }
        )
        out.append(
            {
                "path": os.path.join(folder, f"hsg_{short}_by_ksat.tif"),
                "name": f"HSG {m} at Ksat P05 (more runoff)",
                "main": False,
                "classes": cls,
            }
        )
        out.append(
            {
                "path": os.path.join(folder, f"hsg_{short}_probability.tif"),
                "name": f"HSG {m} probabilities",
                "main": False,
                "classes": [],
            }
        )
    out.append(
        {
            "path": os.path.join(folder, "hsg_conditions.tif"),
            "name": "HSG conditions (depths, Ksat, case)",
            "main": False,
            "classes": [],
        }
    )
    return out


def _write_rasters(result: RunResult, write_fn) -> None:
    folder = result.settings.out_dir
    src = f"{TOOL_NAME} (mayim_tools) v{TOOL_VERSION}"
    g = result.grid

    def f32(a):
        a = np.asarray(a, dtype=np.float64)
        return np.where(result.mask, a, np.nan)

    def codes(a):
        return np.where(result.mask & (np.asarray(a) > 0), a, np.nan).astype(float)

    for m, mr in result.methods.items():
        short = METHOD_SHORT[m]
        path = os.path.join(folder, f"hsg_{short}_recommended.tif")
        write_fn(
            path,
            g,
            [
                (f"hsg_{short}_recommended (code)", codes(mr.recommended)),
                (f"hsg_{short}_confidence (%)", f32(mr.confidence)),
                (f"hsg_{short}_confidence_class (code)", codes(mr.conf_class)),
            ],
            "code / %",
            src,
            TOOL_VERSION,
        )
        result.files.append(path)
        path = os.path.join(folder, f"hsg_{short}_by_ksat.tif")
        write_fn(
            path,
            g,
            [
                (f"hsg_{short}_at_ksat_P05 (code)", codes(mr.at["P05"])),
                (f"hsg_{short}_at_ksat_P50 (code)", codes(mr.at["P50"])),
                (f"hsg_{short}_at_ksat_P95 (code)", codes(mr.at["P95"])),
            ],
            "code",
            src,
            TOOL_VERSION,
        )
        result.files.append(path)
        path = os.path.join(folder, f"hsg_{short}_probability.tif")
        write_fn(
            path,
            g,
            [
                (f"hsg_{short}_probability_{lab} (%)", f32(100.0 * mr.prob[i]))
                for i, lab in enumerate("ABCD")
            ],
            "%",
            src,
            TOOL_VERSION,
        )
        result.files.append(path)
    q = {
        rng: neh630.least_transmissive(result.ksat["P50"], rng)
        for rng in ("0-50cm", "0-100cm")
    }
    bands = [
        (
            "depth_to_impermeable_layer (cm)",
            f32(
                np.where(
                    result.impermeable_cm >= DEEP_CM, np.nan, result.impermeable_cm
                )
            ),
        ),
        (
            "depth_to_water_table (cm)",
            f32(np.where(result.water_cm >= DEEP_CM, np.nan, result.water_cm)),
        ),
        ("neh630_table_case (code)", codes(result.case)),
        ("ksat_least_transmissive_0-50cm_P50 (mm/h)", f32(q["0-50cm"])),
        ("ksat_least_transmissive_0-100cm_P50 (mm/h)", f32(q["0-100cm"])),
    ]
    if result.texture is not None:
        bands.append(("neh630_texture_group_0-30cm (code)", codes(result.texture)))
    path = os.path.join(folder, "hsg_conditions.tif")
    write_fn(path, g, bands, "cm / code / mm/h", src, TOOL_VERSION)
    result.files.append(path)


def _fmt(v, nd=2):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}"


def _write_tables(result: RunResult) -> None:
    path = os.path.join(result.settings.out_dir, "hsg_zone_summary.csv")
    rows = result.zone_rows
    header = []
    for r in rows:
        for k in r:
            if k not in header:
                header.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(r.get(h, "")) for h in header])
    result.files.append(path)


def _write_metadata(result: RunResult) -> None:
    s = result.settings
    g = result.grid
    run_rows = [
        ["Tool", f"{TOOL_NAME} (mayim_tools) v{TOOL_VERSION}"],
        ["Run time", result.run_time_utc],
        ["Duration (s)", round(result.seconds, 1)],
        ["Methods", "; ".join(METHODS[m] for m in s.methods)],
        ["Regional soil parameterisation folder", s.regional_folder],
        ["Grid", f"{g.width} x {g.height} cells at {g.res:g}"],
        ["Cell area (km2)", _fmt(result.cell_area_km2, 6)],
        ["Zones", ", ".join(result.zone_names)],
        ["SCS-SA adjustment: shallow phase", "yes" if s.sa_adjust_shallow else "no"],
        ["SCS-SA adjustment: water table", "yes" if s.sa_adjust_water else "no"],
        ["Output nodata", -9999.0],
        [
            "Group codes",
            "; ".join(f"{c} = {lab}" for c, lab in sorted(ALL_LABEL.items()))
            + " (11-13: NEH 630 dual groups, drained/undrained; 21-23: SCS-SA "
            "intermediate groups)",
        ],
        [
            "Confidence classes",
            "; ".join(f"{k} = {CONF_LABELS[k]}: {CONF_TEXT[k]}" for k in CONF_LABELS),
        ],
        [
            "NEH 630 table cases",
            "; ".join(f"{k} = {v[0]}" for k, v in neh630.CASES.items()),
        ],
        [
            "Note - bands",
            "Select bands by description, never by band number.",
        ],
    ]
    run_rows += [[f"Input - {i + 1}", n] for i, n in enumerate(result.input_notes)]
    run_rows += [[f"Regional run - {k}", v] for k, v in result.regional_meta.items()]
    sections = [("Run", ["Item", "Value"], run_rows)]
    if result.zone_rows:
        header = []
        for r in result.zone_rows:
            for k in r:
                if k not in header:
                    header.append(k)
        sections.append(
            (
                "Zones",
                header,
                [[_fmt(r.get(h, "")) for h in header] for r in result.zone_rows],
            )
        )
    sections.append(
        (
            "Outputs",
            ["File"],
            [[os.path.basename(p)] for p in result.files] + [["hsg_metadata.csv"]],
        )
    )
    sections.append(
        ("Warnings", ["Warning"], [[w] for w in result.warnings] or [["none"]])
    )
    path = os.path.join(s.out_dir, "hsg_metadata.csv")
    write_metadata_csv(sections, path)
    result.files.append(path)
