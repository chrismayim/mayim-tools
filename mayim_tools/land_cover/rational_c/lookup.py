"""
Land-cover lookup for the Rational run-off coefficients: each class of the
South African National Land Cover (SANLC 2018 / 2020 / 2022, same 73 codes)
gets a role (rural, urban, water or ineffective), a Table 3.7 vegetation
class (with a lower and upper class for the uncertainty) for rural classes,
or a Table 3.7 urban row for urban classes. Every run writes the lookup it
used as a CSV, which can be edited and read back (custom lookup). No QGIS.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from . import tables as tb

RURAL, URBAN, WATER, INEFFECTIVE = "rural", "urban", "water", "ineffective"
ROLES = (RURAL, URBAN, WATER, INEFFECTIVE)
ROLE_CODES = {RURAL: 1, URBAN: 2, WATER: 3, INEFFECTIVE: 4}
TB, LB, GR, NV = 0, 1, 2, 3  # vegetation classes of Table 3.7
LAWN = "lawn"  # urban lawns: sandy / heavy from the soil, flat / steep from the slope


@dataclass
class ClassRule:
    code: int
    name: str
    role: str
    cv: int | None = None  # design vegetation class (rural)
    cv_low: int | None = None  # less run-off
    cv_high: int | None = None  # more run-off
    urban: str | None = None  # Table 3.7 urban row key, or "lawn"
    flag: str = ""  # note raised when the class occurs in a catchment


@dataclass
class Lookup:
    key: str
    title: str
    rules: dict  # code -> ClassRule

    def rule(self, code: int) -> ClassRule | None:
        return self.rules.get(int(code))


FLAG_MINE = (
    "mine pit, tailings or landfill: may not drain to the outlet (ineffective area?)"
)
FLAG_FELLED = "clear-felled plantation"

# code, SANLC 2022 name, role, cv, cv_low, cv_high, urban row, flag
_SANLC = [
    (1, "Contiguous (indigenous) forest", RURAL, TB, TB, LB, None, ""),
    (2, "Contiguous low forest & thicket", RURAL, TB, TB, LB, None, ""),
    (3, "Dense forest & woodland", RURAL, TB, TB, LB, None, ""),
    (4, "Open woodland", RURAL, LB, TB, LB, None, ""),
    (5, "Contiguous & dense plantation forest", RURAL, TB, TB, LB, None, ""),
    (6, "Open & sparse plantation forest", RURAL, LB, TB, LB, None, ""),
    (7, "Temporary unplanted (clear-felled) plantation forest", RURAL, NV, GR, NV, None, FLAG_FELLED),
    (8, "Low shrubland (other)", RURAL, LB, LB, GR, None, ""),
    (9, "Low shrubland (fynbos)", RURAL, LB, LB, GR, None, ""),
    (10, "Low shrubland (succulent karoo)", RURAL, LB, LB, GR, None, ""),
    (11, "Low shrubland (nama karoo)", RURAL, LB, LB, GR, None, ""),
    (12, "Sparsely wooded grassland", RURAL, GR, LB, GR, None, ""),
    (13, "Natural grassland", RURAL, GR, GR, NV, None, ""),
    (14, "Natural rivers", WATER, None, None, None, None, ""),
    (15, "Natural estuaries & lagoons", WATER, None, None, None, None, ""),
    (16, "Natural ocean & coastal", INEFFECTIVE, None, None, None, None, "ocean inside a catchment"),
    (17, "Natural lakes", WATER, None, None, None, None, ""),
    (18, "Natural pans (flooded at observation)", INEFFECTIVE, None, None, None, None, "pan excluded"),
    (19, "Artificial dams (incl. canals)", WATER, None, None, None, None, ""),
    (20, "Artificial sewage ponds", INEFFECTIVE, None, None, None, None, ""),
    (21, "Artificial flooded mine pits", INEFFECTIVE, None, None, None, None, ""),
    (22, "Herbaceous wetlands (currently mapped)", RURAL, GR, LB, GR, None, ""),
    (23, "Herbaceous wetlands (previously mapped)", RURAL, GR, LB, GR, None, ""),
    (24, "Mangrove wetlands", RURAL, TB, TB, LB, None, ""),
    (25, "Natural rock surfaces", RURAL, NV, NV, NV, None, ""),
    (26, "Dry pans", INEFFECTIVE, None, None, None, None, "pan excluded"),
    (27, "Eroded lands", RURAL, NV, GR, NV, None, ""),
    (28, "Sand dunes (terrestrial)", RURAL, NV, GR, NV, None, ""),
    (29, "Coastal sand & dunes", RURAL, NV, GR, NV, None, ""),
    (30, "Bare riverbed material", RURAL, NV, NV, NV, None, ""),
    (31, "Other bare", RURAL, NV, GR, NV, None, ""),
    (32, "Cultivated commercial permanent orchards", RURAL, LB, LB, NV, None, ""),
    (33, "Cultivated commercial permanent vines", RURAL, LB, LB, NV, None, ""),
    (34, "Cultivated commercial sugarcane pivot irrigated", RURAL, LB, LB, NV, None, ""),
    (35, "Cultivated commercial permanent pineapples", RURAL, LB, LB, NV, None, ""),
    (36, "Cultivated commercial sugarcane non-pivot", RURAL, LB, LB, NV, None, ""),
    (37, "Cultivated emerging farmer sugarcane non-pivot", RURAL, LB, LB, NV, None, ""),
    (38, "Commercial annual crops pivot irrigated", RURAL, LB, LB, NV, None, ""),
    (39, "Commercial annual crops non-pivot irrigated", RURAL, LB, LB, NV, None, ""),
    (40, "Commercial annual crops rain-fed / dryland", RURAL, LB, LB, NV, None, ""),
    (41, "Subsistence / small-scale annual crops", RURAL, LB, LB, NV, None, ""),
    (42, "Fallow land & old fields (trees)", RURAL, LB, TB, LB, None, ""),
    (43, "Fallow land & old fields (bush)", RURAL, LB, LB, GR, None, ""),
    (44, "Fallow land & old fields (grass)", RURAL, GR, GR, NV, None, ""),
    (45, "Fallow land & old fields (bare)", RURAL, NV, GR, NV, None, ""),
    (46, "Fallow land & old fields (low shrub)", RURAL, LB, LB, GR, None, ""),
    (47, "Residential formal (tree)", URBAN, None, None, None, "houses", ""),
    (48, "Residential formal (bush)", URBAN, None, None, None, "houses", ""),
    (49, "Residential formal (low veg / grass)", URBAN, None, None, None, "houses", ""),
    (50, "Residential formal (bare)", URBAN, None, None, None, "houses", ""),
    (51, "Residential informal (tree)", URBAN, None, None, None, "houses", ""),
    (52, "Residential informal (bush)", URBAN, None, None, None, "houses", ""),
    (53, "Residential informal (low veg / grass)", URBAN, None, None, None, "houses", ""),
    (54, "Residential informal (bare)", URBAN, None, None, None, "houses", ""),
    (55, "Village scattered (bare & low veg / grass)", URBAN, None, None, None, "houses", ""),
    (56, "Village dense (bare & low veg / grass)", URBAN, None, None, None, "houses", ""),
    (57, "Smallholdings (tree)", RURAL, LB, TB, LB, None, ""),
    (58, "Smallholdings (bush)", RURAL, LB, LB, GR, None, ""),
    (59, "Smallholdings (low veg / grass)", RURAL, GR, GR, NV, None, ""),
    (60, "Smallholdings (bare)", RURAL, NV, GR, NV, None, ""),
    (61, "Urban recreational fields (tree)", URBAN, None, None, None, LAWN, ""),
    (62, "Urban recreational fields (bush)", URBAN, None, None, None, LAWN, ""),
    (63, "Urban recreational fields (grass)", URBAN, None, None, None, LAWN, ""),
    (64, "Urban recreational fields (bare)", URBAN, None, None, None, LAWN, ""),
    (65, "Commercial", URBAN, None, None, None, "suburban", ""),
    (66, "Industrial", URBAN, None, None, None, "light_industry", ""),
    (67, "Roads & rails (major linear)", URBAN, None, None, None, "streets", ""),
    (68, "Mines: surface infrastructure", URBAN, None, None, None, "heavy_industry", ""),
    (69, "Mines: extraction pits, quarries", RURAL, NV, NV, NV, None, FLAG_MINE),
    (70, "Mines: salt mines", RURAL, NV, NV, NV, None, ""),
    (71, "Mines: tailings and resource dumps", RURAL, NV, NV, NV, None, FLAG_MINE),
    (72, "Land-fills", RURAL, NV, NV, NV, None, FLAG_MINE),
    (73, "Fallow land & old fields (wetlands)", RURAL, GR, LB, GR, None, ""),
]  # fmt: skip

SANLC_ROCK = 25
SANLC_WETLANDS = (22, 23, 73)
SANLC_PLANTATION = (5, 6, 7)
SANLC_FELLED = 7


def sanlc_lookup() -> Lookup:
    rules = {
        c: ClassRule(c, n, role, cv, lo, hi, urb, flag)
        for c, n, role, cv, lo, hi, urb, flag in _SANLC
    }
    return Lookup("sanlc", "SANLC 2018 / 2020 / 2022 (73 classes)", rules)


CSV_COLUMNS = (
    "code",
    "name",
    "role",
    "cv_design",
    "cv_low",
    "cv_high",
    "urban_row",
    "flag",
)
VEG_KEYS = ("thick_bush", "light_bush", "grassland", "no_vegetation")


def _veg_key(i):
    return "" if i is None else VEG_KEYS[i]


def write_lookup_csv(path: str, lookup: Lookup) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in sorted(lookup.rules.values(), key=lambda r: r.code):
            w.writerow(
                [
                    r.code,
                    r.name,
                    r.role,
                    _veg_key(r.cv),
                    _veg_key(r.cv_low),
                    _veg_key(r.cv_high),
                    r.urban or "",
                    r.flag,
                ]
            )


class CLookupError(ValueError):
    pass


def read_lookup_csv(path: str) -> Lookup:
    """Read an edited lookup (the columns written by write_lookup_csv)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise CLookupError("The lookup CSV is empty.")
    missing = [c for c in ("code", "role") if c not in rows[0]]
    if missing:
        raise CLookupError(f"The lookup CSV lacks column(s): {', '.join(missing)}.")

    def veg(text, code, col):
        text = (text or "").strip().lower()
        if not text:
            return None
        if text not in VEG_KEYS:
            raise CLookupError(
                f"Lookup code {code}: {col} '{text}' is not one of {', '.join(VEG_KEYS)}."
            )
        return VEG_KEYS.index(text)

    rules = {}
    for row in rows:
        try:
            code = int(float(row["code"]))
        except (TypeError, ValueError):
            continue
        role = (row.get("role") or "").strip().lower()
        if role not in ROLES:
            raise CLookupError(
                f"Lookup code {code}: role '{role}' is not one of {', '.join(ROLES)}."
            )
        cv = veg(row.get("cv_design"), code, "cv_design")
        lo = veg(row.get("cv_low"), code, "cv_low")
        hi = veg(row.get("cv_high"), code, "cv_high")
        urban = (row.get("urban_row") or "").strip().lower() or None
        if role == RURAL and cv is None:
            raise CLookupError(f"Lookup code {code}: a rural class needs cv_design.")
        if role == URBAN and urban not in (*tb.URBAN_KEYS, LAWN):
            raise CLookupError(
                f"Lookup code {code}: urban_row '{urban}' is not a Table 3.7 row "
                f"({', '.join((*tb.URBAN_KEYS, LAWN))})."
            )
        rules[code] = ClassRule(
            code,
            (row.get("name") or "").strip() or str(code),
            role,
            cv,
            lo if lo is not None else cv,
            hi if hi is not None else cv,
            urban,
            (row.get("flag") or "").strip(),
        )
    return Lookup("custom", f"Custom lookup ({path})", rules)


# --- soil texture (USDA classes 1-12) -> permeability class (cross-check) ---
TEXTURE_TO_PERMEABILITY = {
    1: 0,  # sand -> very permeable
    2: 1,  # loamy sand
    3: 1,  # sandy loam -> permeable
    4: 2,  # loam
    5: 2,  # silt loam
    6: 2,  # silt
    7: 2,  # sandy clay loam ("clayey sand")
    8: 2,  # clay loam
    9: 2,  # silty clay loam -> semi-permeable
    10: 3,  # sandy clay
    11: 3,  # silty clay
    12: 3,  # clay -> impermeable
}
