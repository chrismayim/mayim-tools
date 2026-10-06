"""Regional soil parameterisation - orchestration (no QGIS).

Stages (see the design note):
    1. harmonise  - inputs.py: read each extraction folder, select bands by
                    description, average to the target depth layers
    2. fill       - fill.py: other product -> SoilGrids 2017 -> neighbours
    3. parameterise and 4. uncertainty - uncertainty.py: Monte Carlo through
                    the method ensemble (ptf/)
then rasters, zone summary, checks, metadata (this module) and the Word
report (report.py).
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.export import (
    write_metadata_csv,
    write_multiband_geotiff,
)
from mayim_tools.soil._common.grid import TargetGrid, make_grid
from mayim_tools.soil._common.stats import _r

from . import fill as fill_mod
from . import inputs as inp
from .ptf import METHOD_BY_CODE, PARAMETERS, rawls_1983, saxton_rawls
from .texture import (
    CLASS_NAME,
    ISO_TO_USDA_SILT_FACTOR,
    TEXTURE_CLASSES,
    usda_class,
)
from .uncertainty import (
    KSAT_ROBUST_RATIO,
    STAT_TAGS,
    TEXTURE_ROBUST_SHARE,
    VARIANCE_PARTS,
    ProductCells,
    Settings,
    run_layer,
)

TOOL_NAME = "Regional soil parameterisation"
TOOL_VERSION = "0.5.0"
CHUNK_CELLS = 1024
MAX_CELLS_DEFAULT = 2_000_000  # ~1800 km2 at 30 m; about 6 GB of memory
SAMPLE_POINTS = 3000  # cells kept for scatter plots in the report

PARAM_FILE = "rsp_{code}.tif"
INPUTS_FILE = "rsp_inputs_P50.tif"
CLASS_FILE = "rsp_texture_class.tif"
QUALITY_FILE = "rsp_quality.tif"
DIFF_FILE = "rsp_product_difference.tif"
ZONES_CSV = "rsp_zone_summary.csv"
METADATA_CSV = "rsp_metadata.csv"
REPORT_FILE = "regional_soil_parameterisation_report.docx"
LAYER_FILE = "rsp_layers.qlr"

ROBUST_LABELS = {
    0: "Neither texture class nor Ksat class robust",
    1: "Texture class robust only",
    2: "Ksat class robust only",
    3: "Texture class and Ksat class robust",
}
INPUT_BANDS = (
    ("sand", "%"),
    ("silt", "%"),
    ("clay", "%"),
    ("om", "%"),
    ("bulk_density", "g/cm3"),
    ("gravel", "vol %"),
)
SHORT = {inp.PRODUCT_SG: "SG", inp.PRODUCT_OLM: "OLM"}


@dataclass
class RunSettings:
    out_dir: str
    sg_folder: str = ""
    olm_folder: str = ""
    resolution: float = 0.0  # 0 = the finest product grid
    layers: tuple = inp.TARGET_LAYERS
    fill_radius_m: float = 1000.0
    mc: Settings = field(default_factory=Settings)
    zones: list = field(default_factory=list)  # [(name, wkt in grid CRS)]
    zone_field: str = ""
    only_zones: bool = True  # process only cells inside the zones
    workers: int = 0  # 0 = automatic
    max_cells: int = MAX_CELLS_DEFAULT
    write_report: bool = True


@dataclass
class RunResult:
    settings: RunSettings
    grid: TargetGrid
    products: list = field(default_factory=list)  # ProductInputs
    sources: dict = field(default_factory=dict)  # product -> layer -> array
    zone_raster: np.ndarray | None = None
    zone_names: list = field(default_factory=list)
    cell_area_km2: float = 0.0
    stats: dict = field(default_factory=dict)  # param -> layer -> (3, r, c)
    inputs_p50: dict = field(default_factory=dict)  # var -> layer -> (r, c)
    texture_class: dict = field(default_factory=dict)
    texture_share: dict = field(default_factory=dict)
    robust: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    central: dict = field(default_factory=dict)  # product -> layer -> dict
    variance: dict = field(default_factory=dict)  # zone -> layer -> param -> parts
    zone_rows: list = field(default_factory=list)
    zone_texture: list = field(default_factory=list)
    reference_rows: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    check_samples: dict = field(default_factory=dict)
    comparison: list = field(default_factory=list)
    comparison_samples: dict = field(default_factory=dict)
    files: list = field(default_factory=list)  # (path, description)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    seconds: float = 0.0
    run_time_utc: str = ""
    report_path: str = ""


# ----------------------------------------------------------------------
# Grid, zones, cell area
# ----------------------------------------------------------------------


def processing_grid(folders: list[str], resolution: float, describe_fn=None):
    describe_fn = describe_fn or inp.band_descriptions
    ref = inp.reference_grid(folders, describe_fn)
    if resolution and resolution > 0 and abs(resolution - ref.res) > 1e-9:
        return make_grid(
            (ref.xmin, ref.ymin, ref.xmax, ref.ymax),
            resolution,
            ref.crs_wkt,
            buffer_cells=0,
        )
    return ref


def rasterise_zones(zones: list, grid: TargetGrid) -> np.ndarray:
    """Zone index raster (1..n in the order given; 0 outside every zone).
    Later zones overwrite earlier ones where they overlap."""
    from osgeo import gdal, ogr, osr

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(grid.crs_wkt)
    drv = ogr.GetDriverByName("Memory")
    ds = drv.CreateDataSource("zones")
    layer = ds.CreateLayer("zones", srs, ogr.wkbUnknown)
    layer.CreateField(ogr.FieldDefn("zid", ogr.OFTInteger))
    for i, (_, wkt) in enumerate(zones, start=1):
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetField("zid", i)
        feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
        layer.CreateFeature(feat)
    rds = gdal.GetDriverByName("MEM").Create(
        "", grid.width, grid.height, 1, gdal.GDT_Int32
    )
    rds.SetGeoTransform(grid.geotransform)
    rds.SetProjection(grid.crs_wkt)
    gdal.RasterizeLayer(rds, [1], layer, options=["ATTRIBUTE=zid"])
    out = rds.GetRasterBand(1).ReadAsArray().astype(np.int32)
    rds = None
    ds = None
    return out


def cell_area_km2(grid: TargetGrid) -> float:
    """Cell area; for a geographic CRS at the grid's centre latitude."""
    try:
        from osgeo import osr

        osr.UseExceptions()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(grid.crs_wkt)
        if srs.IsGeographic():
            lat = math.radians((grid.ymin + grid.ymax) / 2.0)
            km = 111.32 * grid.res
            return km * km * math.cos(lat)
        factor = srs.GetLinearUnits() or 1.0
    except Exception:  # noqa: BLE001
        factor = 1.0
    side_m = grid.res * factor
    return side_m * side_m / 1e6


# ----------------------------------------------------------------------
# Main run
# ----------------------------------------------------------------------


def _layer_cells(products, lab) -> np.ndarray:
    """Cells where at least one product has every required input."""
    mask = None
    for p in products:
        d = p.layers.get(lab, {})
        if not all(v in d for v in inp.TEXTURE + ("soc",)):
            continue
        ok = np.ones_like(d["clay"].centre, dtype=bool)
        for v in inp.TEXTURE + ("soc",):
            ok &= np.isfinite(d[v].centre)
        mask = ok if mask is None else (mask | ok)
    return mask


def _cells_for(p, lab, idx, shared) -> ProductCells | None:
    d = p.layers.get(lab, {})
    if not all(v in d for v in inp.TEXTURE + ("soc",)):
        return None
    dists = {}
    for v in inp.INPUT_VARIABLES:
        src = d.get(v) or shared.get(v)
        if src is None:
            continue
        dists[v] = inp.Dist(src.centre[idx], src.sig_lo[idx], src.sig_hi[idx])
    return ProductCells(p.name, p.texture_system, dists)


def run(
    settings: RunSettings,
    *,
    log_fn: Callable[[str], None] = lambda m: None,
    progress_fn: Callable[[float], None] = lambda f: None,
    cancel_fn: Callable[[], bool] = lambda: False,
    describe_fn=None,
    read_fn=None,
    write_fn=write_multiband_geotiff,
    zones_fn=rasterise_zones,
    report_fn=None,
) -> RunResult:
    t0 = time.time()
    describe_fn = describe_fn or inp.band_descriptions
    read_fn = read_fn or inp.read_file
    folders = [
        (inp.PRODUCT_SG, settings.sg_folder),
        (inp.PRODUCT_OLM, settings.olm_folder),
    ]
    folders = [(name, f) for name, f in folders if f]
    if not folders:
        raise SoilDataError(
            "Give the output folder of 'Extract: SoilGrids 2.0', of 'Extract: "
            "OpenLandMap Soils', or both."
        )
    unknown = [m for m in settings.mc.methods if m not in METHOD_BY_CODE]
    if unknown:
        raise SoilDataError(f"Unknown method(s): {', '.join(unknown)}")
    if settings.mc.draws < 10:
        raise SoilDataError("Use at least 10 Monte Carlo draws.")
    os.makedirs(settings.out_dir, exist_ok=True)

    grid = processing_grid([f for _, f in folders], settings.resolution, describe_fn)
    n_cells = grid.width * grid.height
    if settings.max_cells and n_cells > settings.max_cells:
        raise SoilDataError(
            f"The processing grid has {n_cells} cells ({grid.width} x "
            f"{grid.height} at {grid.res:g}), above the limit of "
            f"{settings.max_cells}. Choose a coarser resolution, or raise "
            "'Maximum cells' under Advanced parameters."
        )
    result = RunResult(settings=settings, grid=grid)
    result.run_time_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    result.cell_area_km2 = cell_area_km2(grid)
    log_fn(
        f"Processing grid: {grid.width} x {grid.height} cells at {grid.res:g} "
        "(CRS of the finest input)."
    )

    # Stage 1 - harmonise
    for name, folder in folders:
        reader = inp.BandReader(grid, read_fn)
        p = inp.load_product(folder, reader, settings.layers, name, describe_fn)
        reader.clear()
        result.products.append(p)
        log_fn(f"Read {p.name} from {folder}")
        for note in p.notes:
            log_fn(f"  {note}")
    progress_fn(0.1)

    # Stage 2 - fill (each product from the other; originals kept for that)
    radius_cells = settings.fill_radius_m / grid.res if grid.res else 0.0
    originals = [_snapshot(p) for p in result.products]
    for i, p in enumerate(result.products):
        others = [o for j, o in enumerate(originals) if j != i]
        n_before = len(p.notes)
        result.sources[p.name] = fill_mod.fill_product(
            p, others[0] if others else None, radius_cells, settings.layers
        )
        for note in p.notes[n_before:]:
            log_fn(f"  {note}")
    if settings.mc.gravel and not any(
        "cfvo" in d for p in result.products for d in p.layers.values()
    ):
        result.warnings.append(
            "Gravel correction requested but no coarse-fragment layer (cfvo) is "
            "available (SoilGrids only) - not applied."
        )
        settings.mc.gravel = False
    if settings.mc.density and not any(
        "bdod" in d for p in result.products for d in p.layers.values()
    ):
        result.warnings.append(
            "Density adjustment requested but no bulk-density layer is "
            "available - not applied."
        )
        settings.mc.density = False
    progress_fn(0.15)

    # Zones
    if settings.zones:
        result.zone_raster = zones_fn(settings.zones, grid)
        result.zone_names = [z[0] for z in settings.zones]
        if not (result.zone_raster > 0).any():
            raise SoilDataError("The zones do not overlap the soil layers.")
    else:
        result.zone_raster = np.ones((grid.height, grid.width), dtype=np.int32)
        result.zone_names = ["Area of interest"]

    # Stages 3-4 - Monte Carlo per layer and chunk
    _monte_carlo(result, settings, log_fn, progress_fn, cancel_fn)
    progress_fn(0.85)

    _zone_summary(result)
    _reference_check(result)
    _water_checks(result)
    if len(result.products) == 2:
        _product_comparison(result)
    _write_rasters(result, write_fn)
    _write_tables(result)
    progress_fn(0.9)
    result.seconds = time.time() - t0
    if settings.write_report:
        try:
            if report_fn is None:
                from .report import write_report as report_fn
            path = os.path.join(settings.out_dir, REPORT_FILE)
            report_fn(result, path)
            result.report_path = path
            result.files.append((path, "Word report"))
        except ImportError as exc:
            result.warnings.append(f"Word report not written ({exc}).")
    _write_metadata(result)
    progress_fn(1.0)
    result.seconds = time.time() - t0
    return result


def _snapshot(p):
    """Copy of a product's layer distributions (the fill source must be
    the other product's ORIGINAL values, not its filled values)."""
    q = inp.ProductInputs(p.name, p.folder, p.texture_system, p.interval)
    q.layers = {
        lab: {v: d.copy() for v, d in ds.items()} for lab, ds in p.layers.items()
    }
    q.sg2017 = p.sg2017
    return q


def _monte_carlo(result, settings, log_fn, progress_fn, cancel_fn):
    grid = result.grid
    shape = (grid.height, grid.width)
    zones = result.zone_raster
    mc = settings.mc
    workers = settings.workers or min(8, os.cpu_count() or 1)
    nz = len(result.zone_names)
    n_layers = len(settings.layers)
    for li, (lab, _, _) in enumerate(settings.layers):
        mask = _layer_cells(result.products, lab)
        if mask is None:
            result.warnings.append(f"No product has every input for {lab}.")
            continue
        if settings.only_zones and settings.zones:
            mask &= zones > 0
        idx_all = np.flatnonzero(mask.ravel())
        if idx_all.size == 0:
            result.warnings.append(f"{lab}: no cells with complete inputs.")
            continue
        # Shared inputs (e.g. SoilGrids coarse fragments for OpenLandMap)
        shared = {}
        for p in result.products:
            for v in ("bdod", "cfvo"):
                if v in p.layers.get(lab, {}) and v not in shared:
                    shared[v] = p.layers[lab][v]
        flat = {
            p.name: {
                v: inp.Dist(d.centre.ravel(), d.sig_lo.ravel(), d.sig_hi.ravel())
                for v, d in p.layers.get(lab, {}).items()
            }
            for p in result.products
        }
        flat_shared = {
            v: inp.Dist(d.centre.ravel(), d.sig_lo.ravel(), d.sig_hi.ravel())
            for v, d in shared.items()
        }
        flat_products = [
            inp.ProductInputs(p.name, p.folder, p.texture_system, p.interval)
            for p in result.products
        ]
        for fp in flat_products:
            fp.layers = {lab: flat[fp.name]}

        chunks = [
            idx_all[i : i + CHUNK_CELLS] for i in range(0, idx_all.size, CHUNK_CELLS)
        ]
        # One seed per (layer, chunk): results do not depend on thread count.
        seeds = np.random.SeedSequence([mc.seed, li]).spawn(len(chunks))

        f4 = np.float32  # result grids: float32 halves the memory
        stats = {prm.code: np.full((3,) + shape, np.nan, f4) for prm in PARAMETERS}
        inputs_p50 = {k: np.full(shape, np.nan, f4) for k, _ in INPUT_BANDS}
        tex = np.zeros(shape, dtype=np.int16)
        share = np.full(shape, np.nan, f4)
        robust = np.full(shape, -1, dtype=np.int16)
        flags = np.zeros(shape, dtype=np.int16)
        central = {
            p.name: {
                "sand": np.full(shape, np.nan),
                "clay": np.full(shape, np.nan),
                "silt": np.full(shape, np.nan),
                "ksat": np.full(shape, np.nan),
                "theta_fc": np.full(shape, np.nan),
                "psi_f": np.full(shape, np.nan),
            }
            for p in result.products
        }
        var_sum = {
            z: {
                prm.code: dict.fromkeys(VARIANCE_PARTS + ("total",), 0.0)
                for prm in PARAMETERS
            }
            for z in range(1, nz + 1)
        }

        def job(
            k,
            chunks=chunks,
            products=flat_products,
            lab=lab,
            shared=flat_shared,
            seeds=seeds,
        ):
            idx = chunks[k]
            cells = []
            for fp in products:
                pc = _cells_for(fp, lab, idx, shared)
                if pc is not None:
                    cells.append(pc)
            return run_layer(cells, mc, np.random.default_rng(seeds[k]))

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            next_k = 0
            while next_k < len(chunks) and len(futures) < workers * 2:
                futures[next_k] = pool.submit(job, next_k)
                next_k += 1
            while futures:
                k = min(futures)
                r = futures.pop(k).result()
                if cancel_fn():
                    for f in futures.values():
                        f.cancel()
                    raise SoilDataError("Cancelled.")
                if next_k < len(chunks):
                    futures[next_k] = pool.submit(job, next_k)
                    next_k += 1
                idx = chunks[k]
                rr, cc = np.unravel_index(idx, shape)
                for code, a in r.stats.items():
                    stats[code][:, rr, cc] = a
                for key, a in r.inputs_p50.items():
                    inputs_p50[key][rr, cc] = a
                tex[rr, cc] = r.texture_class
                share[rr, cc] = r.texture_share
                robust[rr, cc] = r.robust
                flags[rr, cc] = r.flags
                for pname, cdict in r.central.items():
                    ci = cdict["inputs"]
                    m0 = cdict["methods"][mc.methods[0]]
                    central[pname]["sand"][rr, cc] = ci["sand"]
                    central[pname]["silt"][rr, cc] = ci["silt"]
                    central[pname]["clay"][rr, cc] = ci["clay"]
                    central[pname]["ksat"][rr, cc] = m0["ksat"]
                    central[pname]["theta_fc"][rr, cc] = m0["theta_fc"]
                    central[pname]["psi_f"][rr, cc] = m0["psi_f"]
                zc = zones[rr, cc]
                for z in range(1, nz + 1):
                    sel = zc == z
                    if not sel.any():
                        continue
                    for code, parts in r.variance.items():
                        acc = var_sum[z][code]
                        for part, arr in parts.items():
                            vals = arr[sel]
                            acc[part] += float(np.nansum(vals))
                done += 1
                progress_fn(0.15 + 0.7 * (li + done / len(chunks)) / n_layers)
        for code in stats:
            result.stats.setdefault(code, {})[lab] = stats[code]
        for key in inputs_p50:
            result.inputs_p50.setdefault(key, {})[lab] = inputs_p50[key]
        result.texture_class[lab] = tex
        result.texture_share[lab] = share
        result.robust[lab] = robust
        result.flags[lab] = flags
        for pname, arrs in central.items():
            result.central.setdefault(pname, {})[lab] = arrs
        for z, per in var_sum.items():
            result.variance.setdefault(z, {})[lab] = per
        log_fn(f"{lab}: {idx_all.size} cells, {mc.draws} draws per product.")


# ----------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------


def _valid(a):
    return a[np.isfinite(a)]


def _zone_cells(result, z):
    return result.zone_raster == z


def _zone_summary(result: RunResult) -> None:
    rows = []
    tex_rows = []
    for z, zname in enumerate(result.zone_names, start=1):
        zmask = _zone_cells(result, z)
        for lab, _, _ in result.settings.layers:
            if lab not in result.texture_class:
                continue
            for prm in PARAMETERS:
                st = result.stats[prm.code][lab]
                p50 = st[1][zmask]
                ok = np.isfinite(p50)
                if not ok.any():
                    continue
                parts = result.variance.get(z, {}).get(lab, {}).get(prm.code, {})
                total = parts.get("total", 0.0)
                share = {
                    k: (parts.get(k, 0.0) / total if total > 0 else math.nan)
                    for k in VARIANCE_PARTS
                }
                v = p50[ok]
                rows.append(
                    {
                        "Zone": zname,
                        "Layer": lab,
                        "Parameter": prm.code,
                        "Units": prm.units,
                        "Cells": int(ok.sum()),
                        "Area (km2)": ok.sum() * result.cell_area_km2,
                        "Median of P50": float(np.median(v)),
                        "Mean of P50": float(
                            10 ** np.mean(np.log10(np.maximum(v, 1e-6)))
                            if prm.log
                            else np.mean(v)
                        ),
                        "P5 of P50 (spatial)": float(np.percentile(v, 5)),
                        "P95 of P50 (spatial)": float(np.percentile(v, 95)),
                        "Median cell P5": float(np.nanmedian(st[0][zmask][ok])),
                        "Median cell P95": float(np.nanmedian(st[2][zmask][ok])),
                        **{f"Share {k}": share[k] for k in VARIANCE_PARTS},
                    }
                )
            cls = result.texture_class[lab][zmask]
            cls = cls[cls > 0]
            if cls.size:
                counts = np.bincount(cls, minlength=13)
                dom = int(counts.argmax())
                rob = result.robust[lab][zmask]
                rob = rob[rob >= 0]
                tex_rows.append(
                    {
                        "Zone": zname,
                        "Layer": lab,
                        "Dominant class": CLASS_NAME[dom],
                        "Dominant class code": dom,
                        "Dominant share (%)": 100.0 * counts[dom] / cls.size,
                        "Classes present": int((counts[1:] > 0).sum()),
                        "Texture robust (%)": 100.0 * np.isin(rob, (1, 3)).mean(),
                        "Ksat robust (%)": 100.0 * np.isin(rob, (2, 3)).mean(),
                        "Both robust (%)": 100.0 * (rob == 3).mean(),
                        "Sand (%)": float(
                            np.nanmedian(result.inputs_p50["sand"][lab][zmask])
                        ),
                        "Silt (%)": float(
                            np.nanmedian(result.inputs_p50["silt"][lab][zmask])
                        ),
                        "Clay (%)": float(
                            np.nanmedian(result.inputs_p50["clay"][lab][zmask])
                        ),
                        "OM (%)": float(
                            np.nanmedian(result.inputs_p50["om"][lab][zmask])
                        ),
                        "class_counts": counts,
                    }
                )
    result.zone_rows = rows
    result.zone_texture = tex_rows


def _reference_check(result: RunResult) -> None:
    """Rawls, Brakensiek & Miller (1983) class values vs the tool's medians,
    per zone and layer (dominant class) and cell by cell (ratio)."""
    rows = []
    for t in result.zone_texture:
        z = result.zone_names.index(t["Zone"]) + 1
        lab = t["Layer"]
        zmask = _zone_cells(result, z)
        ref = rawls_1983.lookup(result.texture_class[lab][zmask])
        ksat = result.stats["ksat"][lab][1][zmask]
        psi = result.stats["psi_f"][lab][1][zmask]
        ths = result.stats["theta_s"][lab][1][zmask]
        ok = np.isfinite(ref["k"]) & np.isfinite(ksat)
        with np.errstate(invalid="ignore", divide="ignore"):
            k_ratio = np.median(ksat[ok] / ref["k"][ok]) if ok.any() else math.nan
            p_ratio = np.median(psi[ok] / ref["psi_f"][ok]) if ok.any() else math.nan
        dom = t["Dominant class code"]
        tab = rawls_1983.RAWLS_1983.get(dom)
        rows.append(
            {
                "Zone": t["Zone"],
                "Layer": lab,
                "Dominant class": t["Dominant class"],
                "Rawls porosity": tab[0] if tab else math.nan,
                "Rawls effective porosity": tab[1] if tab else math.nan,
                "Rawls psi_f (mm)": tab[2] * 10 if tab else math.nan,
                "Rawls K (mm/h)": tab[3] * 10 if tab else math.nan,
                "Tool theta_s": float(np.nanmedian(ths)),
                "Tool psi_f (mm)": float(np.nanmedian(psi)),
                "Tool Ksat (mm/h)": float(np.nanmedian(ksat)),
                "Median Ksat ratio (tool/Rawls)": float(k_ratio),
                "Median psi_f ratio (tool/Rawls)": float(p_ratio),
                "Cells compared": int(ok.sum()),
            }
        )
    result.reference_rows = rows


def _water_checks(result: RunResult) -> None:
    """PTF field capacity / wilting point vs mapped water contents."""
    rng = np.random.default_rng(0)
    for p in result.products:
        if not p.water_source:
            continue
        for lab, _, _ in result.settings.layers:
            mapped = p.water.get(lab, {})
            for code, prm in (("fc", "theta_fc"), ("wp", "theta_wp")):
                if code not in mapped or lab not in result.stats.get(prm, {}):
                    continue
                pred = result.stats[prm][lab][1]
                obs = mapped[code]
                ok = np.isfinite(pred) & np.isfinite(obs) & (result.zone_raster > 0)
                if ok.sum() < 3:
                    continue
                d = pred[ok] - obs[ok]
                r = (
                    float(np.corrcoef(pred[ok], obs[ok])[0, 1])
                    if d.size > 2
                    else math.nan
                )
                result.checks.append(
                    {
                        "Source": p.water_source,
                        "Layer": lab,
                        "Quantity": (
                            "Field capacity (33 kPa)"
                            if code == "fc"
                            else "Wilting point (1500 kPa)"
                        ),
                        "Cells": int(ok.sum()),
                        "Mapped median (m3/m3)": float(np.median(obs[ok])),
                        "PTF median (m3/m3)": float(np.median(pred[ok])),
                        "Bias PTF - mapped (m3/m3)": float(np.mean(d)),
                        "RMSE (m3/m3)": float(np.sqrt(np.mean(d * d))),
                        "Correlation r": r,
                    }
                )
                pick = np.flatnonzero(ok.ravel())
                if pick.size > SAMPLE_POINTS:
                    pick = rng.choice(pick, SAMPLE_POINTS, replace=False)
                result.check_samples[(p.water_source, lab, code)] = (
                    obs.ravel()[pick],
                    pred.ravel()[pick],
                )


def _product_comparison(result: RunResult) -> None:
    """Both products: central texture (USDA limits) and Saxton & Rawls Ksat,
    on cells where both products have their OWN values (not filled)."""
    sg, olm = (p.name for p in result.products)
    rng = np.random.default_rng(1)
    for lab, _, _ in result.settings.layers:
        if lab not in result.central.get(sg, {}) or lab not in result.central.get(
            olm, {}
        ):
            continue
        a, b = result.central[sg][lab], result.central[olm][lab]
        own = (
            (result.sources[sg].get(lab) == fill_mod.SOURCE_OWN)
            & (result.sources[olm].get(lab) == fill_mod.SOURCE_OWN)
            & (result.zone_raster > 0)
        )
        ok = own & np.isfinite(a["clay"]) & np.isfinite(b["clay"])
        if ok.sum() < 3:
            continue
        ca = usda_class(a["sand"][ok], a["silt"][ok], a["clay"][ok])
        cb = usda_class(b["sand"][ok], b["silt"][ok], b["clay"][ok])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = b["ksat"][ok] / a["ksat"][ok]
        result.comparison.append(
            {
                "Layer": lab,
                "Cells": int(ok.sum()),
                f"Clay {SHORT[sg]} (%)": float(np.median(a["clay"][ok])),
                f"Clay {SHORT[olm]} (%)": float(np.median(b["clay"][ok])),
                f"Sand {SHORT[sg]} (%)": float(np.median(a["sand"][ok])),
                f"Sand {SHORT[olm]} (%)": float(np.median(b["sand"][ok])),
                "Mean clay difference OLM - SG (%)": float(
                    np.mean(b["clay"][ok] - a["clay"][ok])
                ),
                "Mean sand difference OLM - SG (%)": float(
                    np.mean(b["sand"][ok] - a["sand"][ok])
                ),
                "Texture class agreement (%)": float(100.0 * np.mean(ca == cb)),
                "Ksat differs by more than x4 (%)": float(
                    100.0 * np.mean((ratio > 4) | (ratio < 0.25))
                ),
                "Median Ksat ratio OLM / SG": float(np.nanmedian(ratio)),
            }
        )
        pick = np.flatnonzero(ok.ravel())
        if pick.size > SAMPLE_POINTS:
            pick = rng.choice(pick, SAMPLE_POINTS, replace=False)
        result.comparison_samples[lab] = {
            "clay": (a["clay"].ravel()[pick], b["clay"].ravel()[pick]),
            "sand": (a["sand"].ravel()[pick], b["sand"].ravel()[pick]),
            "ksat": (a["ksat"].ravel()[pick], b["ksat"].ravel()[pick]),
        }


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------


def _layers_done(result):
    return [lab for lab, _, _ in result.settings.layers if lab in result.texture_class]


def band_name(code: str, lab: str, tag: str, units: str) -> str:
    return f"{code}_{lab}_{tag} ({units})"


def _write_rasters(result: RunResult, write_fn) -> None:
    out = result.settings.out_dir
    grid = result.grid
    labs = _layers_done(result)
    src = f"{TOOL_NAME} (mayim_tools) v{TOOL_VERSION}"
    if not labs:
        raise SoilDataError("No depth layer could be processed (see warnings).")
    order = (1, 0, 2)  # P50 first so band 1 is the 0-30 cm median
    for prm in PARAMETERS:
        bands = []
        for si in order:
            for lab in labs:
                bands.append(
                    (
                        band_name(prm.code, lab, STAT_TAGS[si], prm.units),
                        result.stats[prm.code][lab][si],
                    )
                )
        path = os.path.join(out, PARAM_FILE.format(code=prm.code))
        write_fn(path, grid, bands, prm.units, src, TOOL_VERSION)
        result.files.append((path, f"{prm.label}: P50, P05, P95 per layer"))

    bands = []
    for key, units in INPUT_BANDS:
        if key in ("bulk_density", "gravel") and all(
            not np.isfinite(result.inputs_p50[key][lab]).any() for lab in labs
        ):
            continue
        for lab in labs:
            bands.append(
                (band_name(key, lab, "P50", units), result.inputs_p50[key][lab])
            )
    path = os.path.join(out, INPUTS_FILE)
    write_fn(path, grid, bands, "", src, TOOL_VERSION)
    result.files.append((path, "Harmonised inputs (USDA texture, OM, ...): median"))

    bands = [
        (f"texture_class_{lab} (USDA code)", result.texture_class[lab].astype(float))
        for lab in labs
    ]
    for b in bands:
        b[1][b[1] == 0] = np.nan
    bands += [
        (f"texture_class_share_{lab} (fraction of draws)", result.texture_share[lab])
        for lab in labs
    ]
    path = os.path.join(out, CLASS_FILE)
    write_fn(path, grid, bands, "", src, TOOL_VERSION)
    result.files.append((path, "USDA texture class (mode of the draws) and its share"))

    bands = []
    for lab in labs:
        rob = result.robust[lab].astype(float)
        rob[rob < 0] = np.nan
        bands.append((f"robustness_{lab} (code)", rob))
    for lab in labs:
        fl = result.flags[lab].astype(float)
        fl[~np.isfinite(result.stats["ksat"][lab][1])] = np.nan
        bands.append((f"validity_flags_{lab} (bit mask)", fl))
    for p in result.products:
        for lab in labs:
            s = result.sources[p.name].get(lab)
            if s is None:
                continue
            s = s.astype(float)
            s[s == 0] = np.nan
            bands.append((f"source_{SHORT[p.name]}_{lab} (code)", s))
    path = os.path.join(out, QUALITY_FILE)
    write_fn(path, grid, bands, "", src, TOOL_VERSION)
    result.files.append((path, "Robustness, validity flags and input source codes"))

    if len(result.products) == 2:
        sg, olm = (p.name for p in result.products)
        bands = []
        for lab in labs:
            a, b = result.central[sg][lab], result.central[olm][lab]
            bands.append((f"clay_{lab}_OLM-SG (%)", b["clay"] - a["clay"]))
            bands.append((f"sand_{lab}_OLM-SG (%)", b["sand"] - a["sand"]))
            with np.errstate(invalid="ignore", divide="ignore"):
                bands.append(
                    (
                        f"ksat_{lab}_log10_OLM/SG (-)",
                        np.log10(b["ksat"]) - np.log10(a["ksat"]),
                    )
                )
        path = os.path.join(out, DIFF_FILE)
        write_fn(path, grid, bands, "", src, TOOL_VERSION)
        result.files.append((path, "Product difference (central soils, USDA limits)"))


def _fmt(v, nd=4):
    return _r(v, nd)


def _write_tables(result: RunResult) -> None:
    import csv

    path = os.path.join(result.settings.out_dir, ZONES_CSV)
    cols = [
        "Zone",
        "Layer",
        "Parameter",
        "Units",
        "Cells",
        "Area (km2)",
        "Median of P50",
        "Mean of P50",
        "P5 of P50 (spatial)",
        "P95 of P50 (spatial)",
        "Median cell P5",
        "Median cell P95",
    ] + [f"Share {k}" for k in VARIANCE_PARTS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in result.zone_rows:
            w.writerow(
                [_fmt(row[c]) if not isinstance(row[c], str) else row[c] for c in cols]
            )
    result.files.append((path, "Zone summary per layer and parameter"))


def _section(title, rows):
    if not rows:
        return None
    header = [k for k in rows[0] if k != "class_counts"]
    return (
        title,
        header,
        [
            [_fmt(r[h]) if not isinstance(r[h], str) else r[h] for h in header]
            for r in rows
        ],
    )


def _write_metadata(result: RunResult) -> None:
    s = result.settings
    mc = s.mc
    g = result.grid
    run_rows = [
        ["Tool", f"{TOOL_NAME} (mayim_tools) v{TOOL_VERSION}"],
        ["Run time", result.run_time_utc],
        ["Duration (s)", round(result.seconds, 1)],
        ["Products", ", ".join(p.name for p in result.products)],
        ["Grid", f"{g.width} x {g.height} cells at {g.res:g} (finest input CRS)"],
        ["Bounds", f"{g.xmin:.3f}, {g.ymin:.3f}, {g.xmax:.3f}, {g.ymax:.3f}"],
        ["Cell area (km2)", _fmt(result.cell_area_km2, 6)],
        ["Depth layers", ", ".join(lab for lab, _, _ in s.layers)],
        ["Methods", ", ".join(METHOD_BY_CODE[m].name for m in mc.methods)],
        ["Reference check", "Rawls, Brakensiek & Miller (1983) class values"],
        ["Monte Carlo draws per product", mc.draws],
        ["Random seed", mc.seed],
        ["OM factor (OM = SOC x factor)", mc.om_factor],
        ["Density adjustment (S&R Eq. 7-10)", "yes" if mc.density else "no"],
        ["Gravel correction (S&R Eq. 19-22)", "yes" if mc.gravel else "no"],
        ["Neighbour fill radius (m)", s.fill_radius_m],
        ["Zones", ", ".join(result.zone_names)],
        ["Only cells inside zones", "yes" if (s.only_zones and s.zones) else "no"],
        ["Output nodata", -9999.0],
        [
            "Note - bands",
            "Select bands by description (e.g. ksat_0-30cm_P50 (mm/h)), never by "
            "band number. Each parameter file holds P50, P05 and P95 per layer.",
        ],
        [
            "Note - texture",
            "Texture outputs use USDA limits (silt 2-50 um). OpenLandMap ISO 11277 "
            "texture (silt 2-63 um) is converted by log-linear interpolation: "
            f"silt_USDA = {ISO_TO_USDA_SILT_FACTOR:.4f} x silt_ISO.",
        ],
        [
            "Note - uncertainty",
            "P05/P95 come from the pooled Monte Carlo ensemble (methods x products "
            "x draws). Inputs are treated as independent; the spread is a lower "
            "bound (soil structure and macropores are not represented).",
        ],
        [
            "Robustness codes",
            "; ".join(f"{k} = {v}" for k, v in ROBUST_LABELS.items())
            + f" (texture: modal class in >= {TEXTURE_ROBUST_SHARE:.0%} of draws; "
            f"Ksat: P95/P5 <= {KSAT_ROBUST_RATIO:g})",
        ],
        [
            "Validity flag bits",
            "; ".join(f"{k} = {v}" for k, v in saxton_rawls.FLAG_LABELS.items()),
        ],
        [
            "Source codes",
            "; ".join(f"{k} = {v}" for k, v in fill_mod.SOURCE_LABELS.items() if k),
        ],
        [
            "Texture class codes",
            "; ".join(f"{c} = {n}" for c, n, _ in TEXTURE_CLASSES),
        ],
    ]
    sections = [("Run", ["Item", "Value"], run_rows)]
    prod_rows = []
    for p in result.products:
        r = p.run
        prod_rows.append(
            [
                p.name,
                p.folder,
                r.get("Tool", ""),
                r.get("Access date", r.get("Run time", "")),
                p.texture_system,
                p.interval,
                ", ".join(f"{k} {v}" for k, v in p.periods.items()),
                r.get("Citation", ""),
                r.get("Licence", ""),
            ]
        )
    sections.append(
        (
            "Inputs",
            [
                "Product",
                "Folder",
                "Extraction tool",
                "Access date",
                "Texture system",
                "Input uncertainty",
                "Periods used",
                "Citation",
                "Licence",
            ],
            prod_rows,
        )
    )
    sections.append(
        (
            "Methods",
            ["Code", "Method", "Family", "Citation"],
            [
                [m.code, m.name, m.family, m.citation]
                for m in (METHOD_BY_CODE[c] for c in mc.methods)
            ]
            + [
                [
                    "RBM1983",
                    "Rawls, Brakensiek & Miller (1983)",
                    "reference only",
                    rawls_1983.CITATION,
                ]
            ],
        )
    )
    sections.append(
        (
            "Parameters",
            ["Code", "Parameter", "Units", "File"],
            [
                [p.code, p.label, p.units, PARAM_FILE.format(code=p.code)]
                for p in PARAMETERS
            ],
        )
    )
    sections.append(
        (
            "Outputs",
            ["File", "Content"],
            [[os.path.basename(f), d] for f, d in result.files]
            + [[METADATA_CSV, "This file"]],
        )
    )
    for title, rows in (
        ("Zones - texture and robustness", result.zone_texture),
        ("Reference check - Rawls, Brakensiek & Miller (1983)", result.reference_rows),
        ("Check - mapped water contents", result.checks),
        ("Product comparison", result.comparison),
    ):
        sec = _section(title, rows)
        if sec:
            sections.append(sec)
    notes = [[n] for p in result.products for n in p.notes]
    if notes:
        sections.append(("Harmonisation and fill notes", ["Note"], notes))
    sections.append(
        ("Warnings", ["Warning"], [[w] for w in result.warnings] or [["none"]])
    )
    path = os.path.join(s.out_dir, METADATA_CSV)
    write_metadata_csv(sections, path)
    result.files.append((path, "Run metadata"))
