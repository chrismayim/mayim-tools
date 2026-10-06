"""USDA-NRCS NEH Part 630 Chapter 7 (2009), Table 7-1: hydrologic soil
group from the depth to a water impermeable layer, the depth to the high
water table and the saturated hydraulic conductivity (Ksat) of the least
transmissive layer in a given depth range. No QGIS, no GDAL.

Table 7-1 (Ksat in µm/s; 1 µm/s = 3.6 mm/h):

    Impermeable  Water table   Ksat depth range  Thresholds        Groups
    < 50 cm      any           -                 -                 D
    50-100 cm    < 60 cm       0-60 cm           40 / 10 / 1       A/D B/D C/D D
    50-100 cm    >= 60 cm      0-50 cm           40 / 10 / 1       A B C D
    > 100 cm     < 60 cm       0-100 cm          10 / 4 / 0.4      A/D B/D C/D D
    > 100 cm     60-100 cm     0-50 cm           40 / 10 / 1       A B C D
    > 100 cm     > 100 cm      0-100 cm          10 / 4 / 0.4      A B C D

A group applies when Ksat is above its lower threshold (A: > 40, B: > 10
to <= 40, C: > 1 to <= 10, D: <= 1 for the shallow thresholds).
"""

from __future__ import annotations

import numpy as np

UM_S_TO_MM_H = 3.6

# Ksat thresholds (mm/h) between A|B, B|C, C|D
SHALLOW_MM_H = (40.0 * UM_S_TO_MM_H, 10.0 * UM_S_TO_MM_H, 1.0 * UM_S_TO_MM_H)
DEEP_MM_H = (10.0 * UM_S_TO_MM_H, 4.0 * UM_S_TO_MM_H, 0.4 * UM_S_TO_MM_H)

# Group codes (rasters and CSVs). Dual groups (water table < 60 cm): the
# first letter applies if drained, D if undrained.
A, B, C, D = 1, 2, 3, 4
AD, BD, CD = 11, 12, 13
GROUP_LABEL = {
    A: "A",
    B: "B",
    C: "C",
    D: "D",
    AD: "A/D",
    BD: "B/D",
    CD: "C/D",
}
GROUP_COLOUR = {
    A: "#1a9850",
    B: "#a6d96a",
    C: "#fdae61",
    D: "#d73027",
    AD: "#4575b4",
    BD: "#74add1",
    CD: "#abd9e9",
}
SINGLE = (A, B, C, D)
DUAL_OF = {A: AD, B: BD, C: CD, D: D}

# Layers of the Regional soil parameterisation outputs used for each
# depth range. The 30-60 cm layer stands in for 30-50 cm (0-50 cm range):
# its Ksat is the average over 30-60 cm.
DEPTH_RANGE_LAYERS = {
    "0-50cm": ("0-30cm", "30-60cm"),
    "0-60cm": ("0-30cm", "30-60cm"),
    "0-100cm": ("0-30cm", "30-60cm", "60-100cm"),
}

# Table 7-1 cases: code -> (description, depth range, thresholds, dual)
CASE_IMPERMEABLE_SHALLOW = 1
CASES = {
    1: ("Impermeable layer < 50 cm", None, None, False),
    2: (
        "Impermeable layer 50-100 cm, water table < 60 cm",
        "0-60cm",
        SHALLOW_MM_H,
        True,
    ),
    3: (
        "Impermeable layer 50-100 cm, water table >= 60 cm",
        "0-50cm",
        SHALLOW_MM_H,
        False,
    ),
    4: ("Impermeable layer > 100 cm, water table < 60 cm", "0-100cm", DEEP_MM_H, True),
    5: (
        "Impermeable layer > 100 cm, water table 60-100 cm",
        "0-50cm",
        SHALLOW_MM_H,
        False,
    ),
    6: (
        "Impermeable layer > 100 cm, water table > 100 cm",
        "0-100cm",
        DEEP_MM_H,
        False,
    ),
}


def case_of(impermeable_cm, water_table_cm) -> np.ndarray:
    """Table 7-1 case (1-6) per cell; 0 where a depth is unknown (NaN).
    Depths in cm below the surface; 50-100 cm includes both limits."""
    imp = np.asarray(impermeable_cm, dtype=float)
    wt = np.asarray(water_table_cm, dtype=float)
    imp, wt = np.broadcast_arrays(imp, wt)
    out = np.zeros(imp.shape, dtype=np.int16)
    known = np.isfinite(imp) & np.isfinite(wt)
    shallow = known & (imp < 50.0)
    mid = known & (imp >= 50.0) & (imp <= 100.0)
    deep = known & (imp > 100.0)
    out[shallow] = 1
    out[mid & (wt < 60.0)] = 2
    out[mid & (wt >= 60.0)] = 3
    out[deep & (wt < 60.0)] = 4
    out[deep & (wt >= 60.0) & (wt <= 100.0)] = 5
    out[deep & (wt > 100.0)] = 6
    return out


def single_group(ksat_mm_h, thresholds) -> np.ndarray:
    """A/B/C/D (1-4) from Ksat (mm/h) and the three thresholds; 0 for NaN."""
    k = np.asarray(ksat_mm_h, dtype=float)
    t_ab, t_bc, t_cd = thresholds
    out = np.zeros(k.shape, dtype=np.int16)
    ok = np.isfinite(k)
    out[ok & (k > t_ab)] = A
    out[ok & (k > t_bc) & (k <= t_ab)] = B
    out[ok & (k > t_cd) & (k <= t_bc)] = C
    out[ok & (k <= t_cd)] = D
    return out


def least_transmissive(ksat_by_layer: dict, depth_range: str) -> np.ndarray:
    """Minimum Ksat over the layers covering ``depth_range`` (NaN-aware;
    NaN only where every layer is NaN)."""
    layers = [ksat_by_layer[lab] for lab in DEPTH_RANGE_LAYERS[depth_range]]
    stack = np.stack([np.asarray(a, dtype=float) for a in layers])
    with np.errstate(invalid="ignore"):
        allnan = np.all(~np.isfinite(stack), axis=0)
        out = np.nanmin(np.where(np.isfinite(stack), stack, np.inf), axis=0)
    out[allnan] = np.nan
    return out


def classify(ksat_by_layer: dict, case: np.ndarray) -> np.ndarray:
    """Hydrologic soil group code per cell for one Ksat value per layer
    (e.g. all P50). 0 where Ksat or the depths are unknown."""
    case = np.asarray(case)
    out = np.zeros(case.shape, dtype=np.int16)
    out[case == CASE_IMPERMEABLE_SHALLOW] = D
    for code, (_, rng, thresholds, dual) in CASES.items():
        if rng is None:
            continue
        sel = case == code
        if not sel.any():
            continue
        k = least_transmissive(ksat_by_layer, rng)
        g = single_group(k, thresholds)
        if dual:
            g = np.select(
                [g == A, g == B, g == C, g == D], [AD, BD, CD, D], default=0
            ).astype(np.int16)
        out[sel] = g[sel]
    return out


def drained_letter(codes) -> np.ndarray:
    """Single letter (1-4) of a group code: A/D -> A, B/D -> B, C/D -> C."""
    c = np.asarray(codes)
    return np.select(
        [np.isin(c, (A, AD)), np.isin(c, (B, BD)), np.isin(c, (C, CD)), c == D],
        [A, B, C, D],
        default=0,
    ).astype(np.int16)


def runoff_rank(codes) -> np.ndarray:
    """Higher = more runoff (undrained dual groups rank as D)."""
    c = np.asarray(codes)
    return np.select(
        [c == A, c == B, c == C, np.isin(c, (D, AD, BD, CD))], [1, 2, 3, 4], default=0
    )
