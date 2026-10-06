"""SCS-SA hydrological soil groups (Schulze, Schmidt & Smithers, 2004,
Visual SCS-SA User Manual, Section 2.2.3 and Table 2.1). No QGIS, no GDAL.

Table 2.1 gives the permeability rate (saturated soil profile) of the four
basic groups: A > 7.6 mm/h, B 3.8-7.6, C 1.3-3.8 and D < 1.3 mm/h (with
typical final infiltration rates of about 25, 13, 6 and 3 mm/h under short
grass). For southern Africa the intermediate groups A/B, B/C and C/D are
also used; the manual assigns them per soil series and gives no numeric
limits. Implementation choices in this tool (agreed with the user and
stated in the report):

- permeability = Ksat of the least transmissive layer within 0-100 cm
  (the manual: "permeability rates are controlled by properties of the
  soil profile");
- the recommended group is the group of the median Ksat; an intermediate
  group is assigned instead where the Ksat range straddles the boundary
  between two adjacent groups (both at least 35 % likely);
- the field adjustments of Section 2.2.3(d) move a group one step down
  (e.g. B -> B/C): a shallow phase (impermeable layer < 50 cm) and, as a
  stand-in for bottomland position, a water table < 60 cm.
"""

from __future__ import annotations

import numpy as np

from . import neh630
from .probability import group_probabilities

PERMEABILITY_MM_H = (7.6, 3.8, 1.3)  # A|B, B|C, C|D
DEPTH_RANGE = "0-100cm"
STRADDLE_MIN = 0.35

AB, BC, CD_SA = 21, 22, 23
STEP_CODES = (neh630.A, AB, neh630.B, BC, neh630.C, CD_SA, neh630.D)
STEP_OF = {c: i for i, c in enumerate(STEP_CODES)}
GROUP_LABEL = {
    neh630.A: "A",
    AB: "A/B",
    neh630.B: "B",
    BC: "B/C",
    neh630.C: "C",
    CD_SA: "C/D",
    neh630.D: "D",
}
GROUP_COLOUR = {
    neh630.A: neh630.GROUP_COLOUR[neh630.A],
    AB: "#66bd63",
    neh630.B: neh630.GROUP_COLOUR[neh630.B],
    BC: "#fee08b",
    neh630.C: neh630.GROUP_COLOUR[neh630.C],
    CD_SA: "#f46d43",
    neh630.D: neh630.GROUP_COLOUR[neh630.D],
}
ADJUST_SHALLOW_CM = 50.0
ADJUST_WATER_CM = 60.0


def profile_quantiles(quantiles: dict) -> dict:
    """P05 / P50 / P95 of the least transmissive layer within 0-100 cm."""
    return {
        tag: neh630.least_transmissive(quantiles[tag], DEPTH_RANGE)
        for tag in ("P05", "P50", "P95")
    }


def probabilities(quantiles: dict) -> np.ndarray:
    """(4, ...) probabilities of the basic groups A, B, C, D."""
    q = profile_quantiles(quantiles)
    return group_probabilities(q["P05"], q["P50"], q["P95"], PERMEABILITY_MM_H)


def group_at(ksat_mm_h) -> np.ndarray:
    """Basic group (1-4) for a single Ksat value per cell."""
    return neh630.single_group(ksat_mm_h, PERMEABILITY_MM_H)


def recommended(
    prob: np.ndarray, median_group: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recommended group and its probability: the group of the median Ksat,
    or the intermediate group where the Ksat range straddles a boundary
    (both adjacent groups at least 35 % likely; probability = the two
    together)."""
    p = np.asarray(prob, dtype=float)
    g = np.asarray(median_group)
    valid = np.all(np.isfinite(p), axis=0) & (g > 0)
    filled = np.where(np.isfinite(p), p, 0.0)
    idx = np.clip(g - 1, 0, 3)
    code = g.astype(np.int16)
    conf = np.take_along_axis(filled, idx[None], axis=0)[0]
    for i, inter in enumerate((AB, BC, CD_SA)):
        both = (filled[i] >= STRADDLE_MIN) & (filled[i + 1] >= STRADDLE_MIN)
        code = np.where(both, inter, code)
        conf = np.where(both, filled[i] + filled[i + 1], conf)
    code = np.where(valid, code, 0).astype(np.int16)
    conf = np.where(valid, conf, np.nan)
    return code, conf


def adjust(codes, impermeable_cm, water_table_cm, shallow=True, water=True):
    """Field adjustments (Section 2.2.3(d)): one step down for each that
    applies, capped at D. Returns (codes, steps applied)."""
    c = np.asarray(codes)
    imp = np.asarray(impermeable_cm, dtype=float)
    wt = np.asarray(water_table_cm, dtype=float)
    steps = np.zeros(c.shape, dtype=np.int16)
    if shallow:
        with np.errstate(invalid="ignore"):
            steps += (np.isfinite(imp) & (imp < ADJUST_SHALLOW_CM)).astype(np.int16)
    if water:
        with np.errstate(invalid="ignore"):
            steps += (np.isfinite(wt) & (wt < ADJUST_WATER_CM)).astype(np.int16)
    out = np.zeros(c.shape, dtype=np.int16)
    for code, i in STEP_OF.items():
        sel = c == code
        new = np.minimum(i + steps[sel], len(STEP_CODES) - 1)
        out[sel] = np.asarray(STEP_CODES)[new]
    return out, np.where(c > 0, steps, 0)
