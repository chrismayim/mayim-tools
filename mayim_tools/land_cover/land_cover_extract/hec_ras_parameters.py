"""HEC-RAS land cover parameters per land cover class (no QGIS).

For each class of the extracted land cover: Manning's n (with a typical
range for calibration), percent impervious and the Spiral Intensity Source
Factor, as entered in a HEC-RAS land cover layer. Written in area mode as
``<product>_<year>_hecras_land_cover.csv`` with the class areas of the
extract, so it can be pasted into the RAS Mapper land cover table.

Basis (recommended starting values, to be calibrated):
- Manning's n for 2D flow of appreciable depth, mapped to the closest NLCD
  class of the HEC-RAS 2D User's Manual (6.6, "Creating Land Cover, Manning's
  n values and % Impervious Layers") and Chow (1959) floodplain ranges,
  adjusted for South African vegetation and settlement types. Shallow sheet
  flow (rain on grid) is rougher: use depth-varied n or raise vegetated
  classes. Built-up values stand in for buildings that are not in the
  terrain; with buildings in the terrain use about 0.03-0.05.
- Percent impervious only matters for precipitation and infiltration on the
  mesh: open water 100 (no infiltration loss), natural and cultivated land
  0 (the soil parameters carry their infiltration).
- Spiral Intensity Source Factor (0-1, HEC-RAS 2D User's Manual 7.0,
  "Secondary Flow"): 1 for river channels only, 0 elsewhere (HEC: its main
  use is to switch secondary flow off where it is not needed, which also
  speeds up the run). Narrow channels the land cover misses should be set
  to 1 with a separate channel polygon.

Recommended by Claude for Chris, 2026-11-16.
"""

from __future__ import annotations

import csv

# code: (Manning's n, typical n range, percent impervious, spiral factor)
SANLC = {
    1: (0.15, "0.10-0.20", 0, 0),
    2: (0.15, "0.10-0.20", 0, 0),
    3: (0.12, "0.10-0.16", 0, 0),
    4: (0.08, "0.06-0.10", 0, 0),
    5: (0.12, "0.10-0.16", 0, 0),
    6: (0.08, "0.06-0.10", 0, 0),
    7: (0.06, "0.04-0.08", 0, 0),
    8: (0.06, "0.05-0.08", 0, 0),
    9: (0.07, "0.05-0.10", 0, 0),
    10: (0.045, "0.035-0.06", 0, 0),
    11: (0.045, "0.035-0.06", 0, 0),
    12: (0.045, "0.035-0.06", 0, 0),
    13: (0.035, "0.030-0.050", 0, 0),
    14: (0.035, "0.030-0.050", 100, 1),
    15: (0.03, "0.025-0.035", 100, 1),
    16: (0.025, "0.020-0.030", 100, 0),
    17: (0.03, "0.025-0.035", 100, 0),
    18: (0.035, "0.030-0.045", 100, 0),
    19: (0.03, "0.025-0.035", 100, 0),
    20: (0.03, "0.025-0.035", 100, 0),
    21: (0.03, "0.025-0.035", 100, 0),
    22: (0.06, "0.05-0.085", 0, 0),
    23: (0.05, "0.04-0.07", 0, 0),
    24: (0.12, "0.08-0.15", 0, 0),
    25: (0.035, "0.030-0.045", 50, 0),
    26: (0.025, "0.020-0.030", 20, 0),
    27: (0.03, "0.025-0.035", 5, 0),
    28: (0.03, "0.025-0.035", 0, 0),
    29: (0.03, "0.025-0.035", 0, 0),
    30: (0.035, "0.030-0.045", 0, 1),
    31: (0.03, "0.025-0.035", 5, 0),
    32: (0.08, "0.06-0.12", 0, 0),
    33: (0.06, "0.05-0.08", 0, 0),
    34: (0.06, "0.05-0.08", 0, 0),
    35: (0.05, "0.04-0.06", 0, 0),
    36: (0.06, "0.05-0.08", 0, 0),
    37: (0.06, "0.05-0.08", 0, 0),
    38: (0.04, "0.030-0.050", 0, 0),
    39: (0.04, "0.030-0.050", 0, 0),
    40: (0.04, "0.030-0.050", 0, 0),
    41: (0.045, "0.035-0.060", 2, 0),
    42: (0.1, "0.08-0.15", 0, 0),
    43: (0.07, "0.05-0.10", 0, 0),
    44: (0.04, "0.030-0.050", 0, 0),
    45: (0.03, "0.025-0.035", 0, 0),
    46: (0.05, "0.04-0.06", 0, 0),
    47: (0.1, "0.08-0.15", 35, 0),
    48: (0.1, "0.08-0.15", 35, 0),
    49: (0.08, "0.06-0.12", 40, 0),
    50: (0.08, "0.06-0.12", 50, 0),
    51: (0.12, "0.10-0.15", 45, 0),
    52: (0.12, "0.10-0.15", 45, 0),
    53: (0.12, "0.10-0.15", 50, 0),
    54: (0.12, "0.10-0.15", 60, 0),
    55: (0.06, "0.05-0.08", 15, 0),
    56: (0.08, "0.06-0.10", 25, 0),
    57: (0.08, "0.06-0.10", 10, 0),
    58: (0.07, "0.05-0.09", 10, 0),
    59: (0.05, "0.04-0.06", 10, 0),
    60: (0.04, "0.030-0.050", 15, 0),
    61: (0.06, "0.05-0.08", 5, 0),
    62: (0.05, "0.04-0.07", 5, 0),
    63: (0.035, "0.030-0.045", 5, 0),
    64: (0.03, "0.025-0.035", 10, 0),
    65: (0.15, "0.12-0.20", 85, 0),
    66: (0.12, "0.10-0.16", 80, 0),
    67: (0.025, "0.016-0.035", 90, 0),
    68: (0.08, "0.05-0.12", 60, 0),
    69: (0.035, "0.030-0.045", 10, 0),
    70: (0.025, "0.020-0.030", 20, 0),
    71: (0.03, "0.025-0.035", 5, 0),
    72: (0.035, "0.030-0.045", 10, 0),
    73: (0.05, "0.04-0.07", 0, 0),
}

WORLDCOVER = {
    10: (0.12, "0.08-0.20", 0, 0),  # tree cover
    20: (0.07, "0.05-0.10", 0, 0),  # shrubland
    30: (0.035, "0.030-0.050", 0, 0),  # grassland
    40: (0.040, "0.030-0.050", 0, 0),  # cropland
    50: (0.10, "0.06-0.20", 60, 0),  # built-up (no density split)
    60: (0.030, "0.025-0.035", 5, 0),  # bare / sparse vegetation
    70: (0.025, "0.020-0.030", 100, 0),  # snow and ice
    80: (0.030, "0.025-0.050", 100, 0),  # permanent water (rivers and lakes)
    90: (0.06, "0.05-0.085", 0, 0),  # herbaceous wetland
    95: (0.12, "0.08-0.15", 0, 0),  # mangroves
    100: (0.035, "0.030-0.050", 0, 0),  # moss and lichen
}

NODATA_ROW = (0.035, "-", 0, 0)

NOTES = {
    "SANLC": "Spiral factor 1 for natural rivers, estuaries and bare riverbed "
    "material; set narrow channels to 1 with a separate polygon.",
    "ESA WorldCover": "WorldCover has one water class for rivers, dams and "
    "lakes, so its spiral factor is 0; set river channels to 1 with a "
    "separate channel polygon. Built-up has no density split: adjust "
    "percent impervious (about 20-90) and n to the area.",
}

COLUMN_KEY = [
    ("Value", "land cover class code (the raster value)"),
    ("Name", "class name"),
    ("ManningN", "recommended Manning's n for 2D flow of appreciable depth"),
    ("N_Range", "typical Manning's n range, for calibration or sensitivity"),
    ("PctImpervious", "recommended percent impervious (%)"),
    ("SpiralFactor", "HEC-RAS Spiral Intensity Source Factor (0-1)"),
    ("Area_km2", "area of the class in the extract (whole output grid)"),
    ("Area_pct", "share of the extract area (%)"),
]


def parameter_rows(product: str, names: dict, areas: dict) -> list[list]:
    """One row per class present in ``areas`` (code -> km2), plus NoData;
    ``product`` is "SANLC" or "ESA WorldCover"."""
    table = SANLC if product == "SANLC" else WORLDCOVER
    total = sum(areas.values())
    rows = [["0", "NoData", *NODATA_ROW, "", ""]]
    for code in sorted(c for c in areas if c in table or c in names):
        n, rng, imp, spiral = table.get(code, NODATA_ROW)
        area = areas.get(code, 0.0)
        rows.append(
            [
                str(code),
                names.get(code, ""),
                n,
                rng,
                imp,
                spiral,
                f"{area:.4f}",
                f"{100.0 * area / total:.2f}" if total else "",
            ]
        )
    return rows


def write_hecras_csv(
    path: str, product: str, names: dict, areas: dict, encoding="utf-8-sig"
):
    """The HEC-RAS land cover parameter CSV (house CSV format: BOM, column
    key, no thousands separators)."""
    with open(path, "w", newline="", encoding=encoding) as fh:
        w = csv.writer(fh)
        w.writerow(["# Column key"])
        for col, text in COLUMN_KEY:
            w.writerow([f"# {col}", text])
        w.writerow(["# Note", NOTES[product]])
        w.writerow([c for c, _ in COLUMN_KEY])
        w.writerows(parameter_rows(product, names, areas))
