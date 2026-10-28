"""Tables of the 'Catchments: 2. Time of Concentration' tool (no QGIS): Kerby
roughness r, NRCS shallow concentrated flow surfaces, Tc method codes and the land-cover
suggestions.

Sources
- Kerby r: SANRAL Drainage Manual (2013) Table 3.10; the two values marked
  TxDOT are from the TxDOT Hydraulic Design Manual (2019) Table 4-5, which
  uses the same Kerby equation.
- Shallow concentrated flow: USDA-NRCS (2010) NEH Part 630 Chapter 15,
  Table 15-3, V = k S^0.5 (ft/s, S ft/ft); k converted to m/s (x 0.3048).
"""

from __future__ import annotations

# Kerby roughness coefficient r: value -> description (source)
KERBY_R = {
    0.02: "Paved areas (DM Table 3.10)",
    0.1: "Clean compacted soil, no stones (DM Table 3.10)",
    0.2: "Poor grass, cultivated row crops (TxDOT Table 4-5)",
    0.3: "Sparse grass over fairly rough surface (DM Table 3.10)",
    0.4: "Medium grass cover (DM Table 3.10)",
    0.6: "Deciduous forest (TxDOT Table 4-5)",
    0.8: "Thick grass cover; dense forest with deep litter (DM Table 3.10)",
}
KERBY_MAX_LENGTH_M = 366.0  # 1200 ft, the longest length Kerby's data covered

FT_TO_M = 0.3048
# shallow concentrated flow surface: code -> (description, k in m/s)
SC_TYPES = {
    "paved": ("Pavement and small upland gullies", 20.3282 * FT_TO_M),
    "grassed": ("Grassed waterways", 16.1345 * FT_TO_M),
    "bare": ("Nearly bare and untilled; alluvial fans", 9.9653 * FT_TO_M),
    "cultivated": ("Cultivated straight row crops", 8.7622 * FT_TO_M),
    "pasture": ("Short-grass pasture", 6.9620 * FT_TO_M),
    "woodland": ("Minimum tillage, contour or strip crops; woodlands",
                 5.0320 * FT_TO_M),
    "forest": ("Forest with heavy ground litter; hay meadows", 2.5160 * FT_TO_M),
}  # fmt: skip

# Tc method chosen per catchment (tc_method) -> label
TC_METHODS = {
    "combined": "Combined: overland + shallow concentrated + channel",
    "usbr": "USBR / SANRAL DM Eq 3.12 (whole path, 10-85 slope)",
    "kirpich": "Kirpich (whole path, average slope)",
    "bransby_williams": "Bransby-Williams (whole path, equal-area slope)",
    "user": "User value (tc_user_mn)",
}
TC_FIELDS = {
    "combined": "tc_comb_mn",
    "usbr": "tc_usbr_mn",
    "kirpich": "tc_kirp_mn",
    "bransby_williams": "tc_bw_mn",
    "user": "tc_user_mn",
}

S_MIN = 0.005  # minimum slope used for overland and shallow flow (m/m)
S_MIN_CHANNEL = 0.0005  # minimum slope used for the channel (m/m)

# ------------------------------------------------- land-cover suggestions ---
# (Kerby r, shallow-flow type) per class; None = no suggestion (water, pans).
_SANLC_GROUPS = [
    ((1, 2, 3, 5, 24), (0.8, "forest")),  # forest, dense plantation, mangroves
    ((4, 6, 42, 57), (0.6, "woodland")),  # open woodland / plantation, trees
    ((8, 9, 10, 11, 43, 46, 58), (0.4, "pasture")),  # shrublands
    ((12, 13, 44, 59), (0.4, "pasture")),  # grasslands
    ((22, 23, 73), (0.8, "forest")),  # herbaceous wetlands (hay meadow)
    ((32, 33, 34, 35, 36, 37, 38, 39, 40, 41), (0.2, "cultivated")),
    ((7, 25, 27, 28, 29, 30, 31, 45, 60, 69, 70, 71, 72), (0.1, "bare")),
    ((47, 48, 49, 50, 51, 52, 53, 54, 55, 56), (0.3, "paved")),  # residential
    ((61, 62, 63, 64), (0.4, "pasture")),  # recreational fields
    ((65, 66, 67, 68), (0.02, "paved")),  # commercial, industrial, roads
]
SANLC_SUGGESTIONS = {c: v for codes, v in _SANLC_GROUPS for c in codes}

WORLDCOVER_SUGGESTIONS = {
    10: (0.6, "woodland"),  # tree cover
    20: (0.4, "pasture"),  # shrubland
    30: (0.4, "pasture"),  # grassland
    40: (0.2, "cultivated"),  # cropland
    50: (0.02, "paved"),  # built-up
    60: (0.1, "bare"),  # bare / sparse vegetation
    90: (0.8, "forest"),  # herbaceous wetland
    95: (0.8, "forest"),  # mangroves
    100: (0.4, "pasture"),  # moss and lichen
}

LAND_COVER_CLASSIFICATIONS = {
    "sanlc": ("South African National Land Cover (SANLC 2018 / 2020 / 2022)",
              SANLC_SUGGESTIONS),
    "worldcover": ("ESA WorldCover (2020 / 2021)", WORLDCOVER_SUGGESTIONS),
}  # fmt: skip

REFERENCES = [
    "Bransby-Williams, G. (1922). Flood discharge and the dimensions of "
    "spillways in India. The Engineer, 134, 321-322.",
    "Kerby, W.S. (1959). Time of concentration for overland flow. Civil "
    "Engineering, 29(3).",
    "Kirpich, Z.P. (1940). Time of concentration of small agricultural "
    "watersheds. Civil Engineering, 10(6), 362.",
    "SANRAL (2013). Drainage Manual, 6th edition, Chapter 3: Flood "
    "calculations. The South African National Roads Agency SOC Ltd, Pretoria.",
    "TxDOT (2019). Hydraulic Design Manual, Chapter 4, Section 11: Time of "
    "Concentration. Texas Department of Transportation, Austin.",
    "USBR (1973). Design of Small Dams, 2nd edition. United States Bureau of "
    "Reclamation, Washington DC.",
    "USDA-NRCS (2007). National Engineering Handbook, Part 630 Hydrology, "
    "Chapter 16: Hydrographs. United States Department of Agriculture.",
    "USDA-NRCS (2010). National Engineering Handbook, Part 630 Hydrology, "
    "Chapter 15: Time of Concentration. United States Department of "
    "Agriculture.",
]
