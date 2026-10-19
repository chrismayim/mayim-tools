"""
Tables of the SANRAL Drainage Manual, 6th edition (2013), section 3.5.1.2,
transcribed from the manual (pages 3-18 and 3-19; PDF pages 82-83) and
checked against Appendix 3C (Tables 3C.1-3C.4) and the Application Guide
(2013, Table 3.3 and worked example 3.1, pp. 3-18 to 3-20). No QGIS.
"""

from __future__ import annotations

SOURCE = "SANRAL (2013) Drainage Manual, 6th ed., Table 3.7 (p. 3-18)"

# MAP bands of Table 3.7: < 600, 600-900 (both limits included), > 900 mm
MAP_BANDS = ("< 600", "600 - 900", "> 900")


def map_band(map_mm: float) -> int:
    """Index of the MAP column: 0 below 600 mm, 1 from 600 to 900 mm
    (inclusive), 2 above 900 mm."""
    if map_mm is None or map_mm != map_mm or map_mm <= 0:
        raise ValueError("MAP must be a positive number of millimetres.")
    if map_mm < 600.0:
        return 0
    if map_mm <= 900.0:
        return 1
    return 2


# --- Rural C1 = Cs + Cp + Cv (Table 3.7) ---------------------------------
# classes are indexed 0-3, from the least to the most run-off
SLOPE_CLASSES = (
    "Vleis and pans (< 3 %)",
    "Flat areas (3 to 10 %)",
    "Hilly (10 to 30 %)",
    "Steep areas (> 30 %)",
)
SLOPE_LIMITS_PCT = (3.0, 10.0, 30.0)  # a value on a limit goes to the class above
CS = (
    (0.01, 0.03, 0.05),
    (0.06, 0.08, 0.11),
    (0.12, 0.16, 0.20),
    (0.22, 0.26, 0.30),
)

PERMEABILITY_CLASSES = ("Very permeable", "Permeable", "Semi-permeable", "Impermeable")
PERMEABILITY_TEXT = (
    "gravel, coarse sand",
    "sandy, sandy loam",
    "silt, loam, clayey sand",
    "clay, peat, rock",
)
CP = (
    (0.03, 0.04, 0.05),
    (0.06, 0.08, 0.10),
    (0.12, 0.16, 0.20),
    (0.21, 0.26, 0.30),
)

VEGETATION_CLASSES = (
    "Thick bush and plantation",
    "Light bush and farm lands",
    "Grasslands",
    "No vegetation",
)
CV = (
    (0.03, 0.04, 0.05),
    (0.07, 0.11, 0.15),
    (0.17, 0.21, 0.25),
    (0.26, 0.28, 0.30),
)

# --- Urban C2 (Table 3.7): key -> (label, low, high) ----------------------
URBAN_ROWS = {
    "lawn_sandy_flat": ("Lawns - sandy, flat (< 2 %)", 0.05, 0.10),
    "lawn_sandy_steep": ("Lawns - sandy, steep (> 7 %)", 0.15, 0.20),
    "lawn_heavy_flat": ("Lawns - heavy soil, flat (< 2 %)", 0.13, 0.17),
    "lawn_heavy_steep": ("Lawns - heavy soil, steep (> 7 %)", 0.25, 0.35),
    "houses": ("Residential - houses", 0.30, 0.50),
    "flats": ("Residential - flats", 0.50, 0.70),
    "light_industry": ("Industry - light industry", 0.50, 0.80),
    "heavy_industry": ("Industry - heavy industry", 0.60, 0.90),
    "city_centre": ("Business - city centre", 0.70, 0.95),
    "suburban": ("Business - suburban", 0.50, 0.70),
    "streets": ("Streets", 0.70, 0.95),
    "max_flood": ("Maximum flood", 1.00, 1.00),
}
URBAN_KEYS = tuple(URBAN_ROWS)
LAWN_FLAT_PCT, LAWN_STEEP_PCT = 2.0, 7.0
POSITIONS = ("upper", "middle", "lower")


def urban_value(key: str, position: str = "upper") -> float:
    _label, lo, hi = URBAN_ROWS[key]
    return {"upper": hi, "middle": 0.5 * (lo + hi), "lower": lo}[position]


# --- Dolomite reduction factors on C1, by slope class (p. 3-19) -----------
DOLOMITE_FACTORS = (0.10, 0.20, 0.35, 0.50)

# --- Table 3.8: adjustment factor Ft for C1 --------------------------------
RETURN_PERIODS = (2, 5, 10, 20, 50, 100)
FT = {
    "steep_impermeable": (0.75, 0.80, 0.85, 0.90, 0.95, 1.00),
    "flat_permeable": (0.50, 0.55, 0.60, 0.67, 0.83, 1.00),
}
FT_LABELS = {
    "steep_impermeable": "Steep and impermeable",
    "flat_permeable": "Flat and permeable",
}
FT_SCORE_LIMIT = 0.5
FT_BORDERLINE = (0.4, 0.6)


def ft(ft_class: str, period: int) -> float:
    return FT[ft_class][RETURN_PERIODS.index(period)]


def c1_from_shares(slope, perm, veg, band: int) -> float:
    """C1 = Cs + Cp + Cv, each the share-weighted table value (shares as
    fractions of the rural area, ordered as the class tuples)."""
    cs = sum(f * CS[i][band] for i, f in enumerate(slope))
    cp = sum(f * CP[i][band] for i, f in enumerate(perm))
    cv = sum(f * CV[i][band] for i, f in enumerate(veg))
    return cs + cp + cv


def dolomite_factor(slope_shares) -> float:
    """Weighted reduction factor sum(Dfactor x Cs%) of the manual's form."""
    return sum(f * DOLOMITE_FACTORS[i] for i, f in enumerate(slope_shares))


def c1_dolomite(c1: float, d_fraction: float, weighted_factor: float) -> float:
    """C1D = C1 (1 - D%) + C1 D% sum(Dfactor x Cs%) (Appendix 3A form)."""
    return c1 * (1.0 - d_fraction) + c1 * d_fraction * weighted_factor


def ft_score(slope_shares, perm_shares) -> float:
    """Share of the rural area that is hilly or steep, averaged with the
    share that is semi-permeable or impermeable (0 to 1)."""
    return 0.5 * (slope_shares[2] + slope_shares[3]) + 0.5 * (
        perm_shares[2] + perm_shares[3]
    )


def ft_class_from_score(score: float) -> str:
    return "steep_impermeable" if score >= FT_SCORE_LIMIT else "flat_permeable"


def c1_pmf(slope_shares, band: int) -> float:
    """PMF: C1 = Cs + Cp,max + Cv,max (p. 3-19)."""
    cs = sum(f * CS[i][band] for i, f in enumerate(slope_shares))
    return cs + max(r[band] for r in CP) + max(r[band] for r in CV)
