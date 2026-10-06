"""Texture systems and the USDA texture classes (no QGIS, no GDAL).

Texture is in percent by weight of the fine earth (< 2 mm). Two particle-size
systems occur in the inputs:

- USDA: clay < 2 um, silt 2-50 um, sand 50 um - 2 mm (SoilGrids 2.0 and
  2017; the system of the US pedotransfer functions);
- ISO 11277: clay < 2 um, silt 2-63 um, sand 63 um - 2 mm (OpenLandMap-soildb).

ISO is converted to USDA by log-linear interpolation of the cumulative
particle-size distribution between 2 and 63 um (Nemes et al., 1999;
Minasny & McBratney, 2001):

    P(50) = clay + silt_ISO * ln(50/2) / ln(63/2)

so silt_USDA = k * silt_ISO with k = ln(25) / ln(31.5) = 0.9330, clay is
unchanged and sand_USDA = 100 - clay - silt_USDA.
"""

from __future__ import annotations

import math

import numpy as np

ISO_TO_USDA_SILT_FACTOR = math.log(50.0 / 2.0) / math.log(63.0 / 2.0)

# (code, name, abbreviation) - USDA Soil Survey Manual (2017) classes.
TEXTURE_CLASSES: tuple[tuple[int, str, str], ...] = (
    (1, "Sand", "S"),
    (2, "Loamy sand", "LS"),
    (3, "Sandy loam", "SL"),
    (4, "Loam", "L"),
    (5, "Silt loam", "SiL"),
    (6, "Silt", "Si"),
    (7, "Sandy clay loam", "SCL"),
    (8, "Clay loam", "CL"),
    (9, "Silty clay loam", "SiCL"),
    (10, "Sandy clay", "SC"),
    (11, "Silty clay", "SiC"),
    (12, "Clay", "C"),
)
CLASS_NAME = {code: name for code, name, _ in TEXTURE_CLASSES}
CLASS_ABBR = {code: abbr for code, _, abbr in TEXTURE_CLASSES}


def iso_to_usda(sand_iso, silt_iso, clay):
    """ISO 11277 (silt to 63 um) -> USDA (silt to 50 um); percent in, percent
    out. Arrays or scalars; NaN propagates."""
    clay = np.asarray(clay, dtype=np.float64)
    silt = np.asarray(silt_iso, dtype=np.float64) * ISO_TO_USDA_SILT_FACTOR
    total = np.asarray(sand_iso, dtype=np.float64) + np.asarray(silt_iso) + clay
    sand = total - clay - silt
    return sand, silt, clay


def normalise_texture(sand, silt, clay):
    """Scale the three fractions to sum to 100 % (closure). Cells whose sum
    is not positive become NaN."""
    sand = np.asarray(sand, dtype=np.float64)
    silt = np.asarray(silt, dtype=np.float64)
    clay = np.asarray(clay, dtype=np.float64)
    total = sand + silt + clay
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(total > 0, 100.0 / total, np.nan)
    return sand * f, silt * f, clay * f


def usda_class(sand, silt, clay) -> np.ndarray:
    """USDA texture class code (1-12, see TEXTURE_CLASSES); 0 where any
    input is NaN. Inputs are percentages summing to ~100 (they are closed
    first, so small rounding differences do not leave gaps)."""
    sand, silt, clay = normalise_texture(sand, silt, clay)
    out = np.zeros(np.broadcast(sand, silt, clay).shape, dtype=np.int16)
    ok = np.isfinite(sand) & np.isfinite(silt) & np.isfinite(clay)
    sa, si, cl = sand, silt, clay
    # Assigned from the finest classes to the coarsest; later rules only
    # fill cells that are still unassigned.
    rules = (
        (12, (cl >= 40) & (sa <= 45) & (si < 40)),
        (11, (cl >= 40) & (si >= 40)),
        (10, (cl >= 35) & (sa > 45)),
        (9, (cl >= 27) & (cl < 40) & (sa <= 20)),
        (8, (cl >= 27) & (cl < 40) & (sa > 20) & (sa <= 45)),
        (7, (cl >= 20) & (cl < 35) & (si < 28) & (sa > 45)),
        (6, (si >= 80) & (cl < 12)),
        (5, ((si >= 50) & (cl >= 12) & (cl < 27)) | ((si >= 50) & (cl < 12))),
        (4, (cl >= 7) & (cl < 27) & (si >= 28) & (si < 50) & (sa <= 52)),
        (1, (si + 1.5 * cl) < 15),
        (2, ((si + 1.5 * cl) >= 15) & ((si + 2 * cl) < 30)),
        (3, np.ones_like(ok)),
    )
    for code, mask in rules:
        sel = ok & (out == 0) & mask
        out[sel] = code
    return out


def ternary_xy(sand, clay):
    """USDA triangle coordinates: sand 100 % bottom-left (0, 0), silt
    100 % bottom-right (100, 0), clay 100 % at the top."""
    sand = np.asarray(sand, dtype=np.float64)
    clay = np.asarray(clay, dtype=np.float64)
    return 100.0 - sand - clay / 2.0, clay * math.sqrt(3.0) / 2.0
