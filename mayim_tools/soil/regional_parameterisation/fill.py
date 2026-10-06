"""Gap filling (stage 2). No QGIS; GDAL only for the neighbour fill.

SoilGrids 2.0 has no predictions for built-up land, water, glaciers and
bare surfaces; OpenLandMap leaves deserts and permanent ice unmapped. Each
product's missing cells are filled, per variable and depth layer, in this
order (agreed design):

    1. the other product (texture converted between USDA and ISO limits)
    2. SoilGrids 2017 means (built-up land is mapped there); the
       uncertainty is the median uncertainty of the layer's own cells
    3. nearest valid neighbours within a search radius (inverse-distance
       fill of centre and uncertainty)

and the source of every cell is recorded (SOURCE_* codes, from clay).
"""

from __future__ import annotations

import numpy as np

from .inputs import INPUT_VARIABLES, Dist, ProductInputs
from .texture import ISO_TO_USDA_SILT_FACTOR

SOURCE_NONE = 0
SOURCE_OWN = 1
SOURCE_OTHER = 2
SOURCE_SG2017 = 3
SOURCE_NEIGHBOUR = 4
FILL_TEXT = {
    2: "the other product",
    3: "SoilGrids 2017",
    4: "neighbouring cells",
}
SOURCE_LABELS = {
    SOURCE_NONE: "No data",
    SOURCE_OWN: "Product itself",
    SOURCE_OTHER: "Other product",
    SOURCE_SG2017: "SoilGrids 2017",
    SOURCE_NEIGHBOUR: "Neighbour fill",
}


def _texture_between(
    dists: dict[str, Dist], var: str, to_system: str
) -> np.ndarray | None:
    """Centre value of ``var`` converted to ``to_system`` ("USDA" or
    "ISO 11277") using the silt factor k (texture.py); clay is unchanged."""
    k = ISO_TO_USDA_SILT_FACTOR
    if var == "clay":
        return dists["clay"].centre
    if "silt" not in dists or "sand" not in dists:
        return None
    silt = dists["silt"].centre
    sand = dists["sand"].centre
    if to_system == "USDA":  # ISO -> USDA
        return k * silt if var == "silt" else sand + (1.0 - k) * silt
    # USDA -> ISO
    return silt / k if var == "silt" else sand - (1.0 / k - 1.0) * silt


def _from_other(
    target: ProductInputs, other: ProductInputs, lab: str, var: str
) -> Dist | None:
    src = other.layers.get(lab, {})
    if var not in src:
        return None
    d = src[var]
    if var in ("sand", "silt") and target.texture_system != other.texture_system:
        centre = _texture_between(src, var, target.texture_system)
        if centre is None:
            return None
        return Dist(centre, d.sig_lo, d.sig_hi)
    return d


def _sg2017_values(
    target: ProductInputs, other: ProductInputs | None, lab: str, var: str
) -> np.ndarray | None:
    """SoilGrids 2017 layer means for ``var`` from either folder, with
    texture converted from USDA to ISO limits for an ISO product."""
    for p in (target, other):
        if p is None:
            continue
        vals = p.sg2017.get(lab, {})
        if var not in vals:
            continue
        if var in ("sand", "silt") and target.texture_system != "USDA":
            if "sand" not in vals or "silt" not in vals:
                return None
            k = ISO_TO_USDA_SILT_FACTOR
            silt = vals["silt"]
            return silt / k if var == "silt" else vals["sand"] - (1.0 / k - 1.0) * silt
        return vals[var]
    return None


def _typical(arr: np.ndarray) -> float:
    v = arr[np.isfinite(arr)]
    return float(np.median(v)) if v.size else 0.0


def neighbour_fill(arr: np.ndarray, max_cells: float) -> np.ndarray:
    """Fill NaN cells from valid cells within ``max_cells`` (GDAL
    FillNodata, inverse-distance weighting, no smoothing)."""
    if max_cells <= 0 or not np.isnan(arr).any() or np.isfinite(arr).sum() == 0:
        return arr
    from osgeo import gdal

    gdal.UseExceptions()
    rows, cols = arr.shape
    ds = gdal.GetDriverByName("MEM").Create("", cols, rows, 1, gdal.GDT_Float32)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32))
    gdal.FillNodata(band, None, float(max_cells), 0)
    out = band.ReadAsArray().astype(np.float64)
    out[out == -9999.0] = np.nan
    ds = None
    return np.where(np.isfinite(arr), arr, out)


def fill_product(
    target: ProductInputs,
    other: ProductInputs | None,
    radius_cells: float,
    layers,
    fill_fn=neighbour_fill,
) -> dict[str, np.ndarray]:
    """Fill ``target`` in place; returns the source raster per layer
    (SOURCE_* codes, judged from clay). Counts of filled cells are added
    to ``target.notes``."""
    sources: dict[str, np.ndarray] = {}
    for lab, _, _ in layers:
        dists = target.layers.get(lab, {})
        for var in INPUT_VARIABLES:
            if var not in dists:
                continue
            d = dists[var].copy()
            src = np.where(np.isfinite(d.centre), SOURCE_OWN, SOURCE_NONE)
            src = src.astype(np.int16)
            steps = []
            if other is not None:
                o = _from_other(target, other, lab, var)
                if o is not None:
                    steps.append((SOURCE_OTHER, o.centre, o.sig_lo, o.sig_hi))
            s17 = _sg2017_values(target, other, lab, var)
            if s17 is not None:
                lo = np.full(s17.shape, _typical(d.sig_lo))
                hi = np.full(s17.shape, _typical(d.sig_hi))
                steps.append((SOURCE_SG2017, s17, lo, hi))
            for code, c, lo, hi in steps:
                gap = ~np.isfinite(d.centre) & np.isfinite(c)
                if gap.any():
                    d.centre[gap] = c[gap]
                    d.sig_lo[gap] = lo[gap]
                    d.sig_hi[gap] = hi[gap]
                    src[gap] = code
            if radius_cells > 0 and np.isnan(d.centre).any():
                before = ~np.isfinite(d.centre)
                d = Dist(
                    fill_fn(d.centre, radius_cells),
                    fill_fn(d.sig_lo, radius_cells),
                    fill_fn(d.sig_hi, radius_cells),
                )
                src[before & np.isfinite(d.centre)] = SOURCE_NEIGHBOUR
            # Sigmas must exist wherever the centre does.
            has = np.isfinite(d.centre)
            d.sig_lo = np.where(has & ~np.isfinite(d.sig_lo), 0.0, d.sig_lo)
            d.sig_hi = np.where(has & ~np.isfinite(d.sig_hi), 0.0, d.sig_hi)
            dists[var] = d
            if var == "clay":
                sources[lab] = src
                for code in (SOURCE_OTHER, SOURCE_SG2017, SOURCE_NEIGHBOUR):
                    n = int((src == code).sum())
                    if n:
                        target.notes.append(
                            f"{target.name} {lab}: {n} cells filled from "
                            f"{FILL_TEXT[code]}."
                        )
    return sources
