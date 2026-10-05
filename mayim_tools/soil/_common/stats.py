"""Unit conversion and summary statistics (NaN = nodata throughout)."""

from __future__ import annotations

import math

import numpy as np


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
# Default GDAL functions (the only part of this module that needs osgeo)
# ----------------------------------------------------------------------
