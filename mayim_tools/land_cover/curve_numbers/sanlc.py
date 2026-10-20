"""SANLC 2018 / 2020 / 2022 (73 classes) to curve-number lookup.

Built from Chris's composite table (October 2026), which takes the WRC
Report 2747/1/20 values (Loots & Smithers 2020, Table 9.7, DEA 2013/14 land
uses) through a crosswalk from the 2018 / 2020 classes, and SCS-SA Table 5.1
rows for the classes the report does not cover. Design values are the
composite's, with these review changes (agreed 2026-10-12):

* crosswalk fixes: Residential Formal (47-50) -> DEA urban residential
  48-51 and Residential Informal (51-54) -> DEA urban informal 44-47 (the
  composite had them the wrong way round); Urban Recreational Fields
  (61-64) -> DEA 57-60 (were shifted by one); villages (55, 56) -> DEA 68;
* misprints: low shrubland A/B 39 -> 36; the WRC Table 9.7 misprints
  corrected in tables.WRC_ROWS;
* open water 100 and natural rock surfaces 98 (Chris's choice);
* herbaceous wetlands (22, 23): veld in good condition on group D, with
  irrigated pasture (the WRC's association for wetlands) as the low end
  and veld in poor condition as the high end.

Low (less runoff) and high (more runoff) ends: neighbouring condition rows
of Table 5.1 for natural cover; for the urban families, the neighbouring
vegetation sub-class of the same land use (SANLC's tree / bush / grass /
bare split is the least certain part of the map), and for calibrated
residential classes the uncalibrated Table 5.1 value (WRC Table 8.1) as the
high end, because the report calls Table 9.7 preliminary. Where the
composite's value looked low for the cover, the alternative is the high
end (annual crops, Karoo and other low shrubland, sugarcane, clear-felled
plantations).

Colours are the published SANLC 2018 legend colours. SANLC 2022 uses the
same 73 codes (checked against the 2022 attribute table, October 2026).
"""

from __future__ import annotations

from .tables import DefaultClass as C

SANLC_SOURCE = (
    "Composite of WRC Report 2747/1/20 (Loots & Smithers 2020, Table 9.7) and "
    "SCS-SA Table 5.1 (Schulze, Schmidt & Smithers 2004), crosswalked from the "
    "SANLC 2018 / 2020 legend to the DEA 2013/14 land-use types"
)

_RES_NOTE = (
    "WRC calibrated (Tshwane); high end = uncalibrated Table 5.1 (WRC Table 8.1)"
)
_WRC = "WRC 2747/1/20 Table 9.7"
_X = "derived from WRC 2747/1/20 by a similar class"

SANLC_2020 = [
    # Forest, woodland, plantations ------------------------------------
    C(
        1,
        "Contiguous (indigenous) forest",
        "scs_sa:forest_9",
        "wrc2747:5",
        "scs_sa:woods_2",
        note=f"DEA 4 as thicket (DEA 5); {_X}",
        colour="#0b3d0b",
    ),
    C(
        2,
        "Contiguous low forest & thicket",
        "scs_sa:forest_9",
        "wrc2747:5",
        "scs_sa:woods_2",
        note=f"DEA 5; {_WRC}",
        colour="#065106",
    ),
    C(
        3,
        "Dense forest & woodland (35-75 % cc)",
        "scs_sa:woods_2",
        "wrc2747:6",
        "scs_sa:veld_2",
        note=f"DEA 6; {_WRC}",
        colour="#005f00",
    ),
    C(
        4,
        "Open woodland (10-35 % cc)",
        "scs_sa:woods_2",
        "wrc2747:6",
        "scs_sa:veld_2",
        note=f"DEA 6; {_WRC}",
        colour="#008500",
    ),
    C(
        5,
        "Contiguous & dense planted forest",
        "scs_sa:forest_9",
        "wrc2747:32",
        "scs_sa:forest_7",
        note=f"DEA 32; {_WRC}",
        colour="#f74006",
    ),
    C(
        6,
        "Open & sparse planted forest",
        "scs_sa:forest_6",
        "wrc2747:33",
        "scs_sa:forest_1",
        note=f"DEA 33; {_WRC}",
        colour="#f9764d",
    ),
    C(
        7,
        "Temporary unplanted forest",
        "scs_sa:forest_6",
        "wrc2747:33",
        "scs_sa:forest_1",
        note=f"DEA 34 as young plantation; {_X}; high end = compact humus 25 mm (clear-felled)",
        colour="#f9906c",
    ),
    # Shrubland and grassland -------------------------------------------
    C(
        8,
        "Low shrubland (other regions)",
        "wrc2747:9",
        "wrc2747:9",
        "scs_sa:veld_1",
        note=f"DEA 9; {_X}; brush (winter rainfall) may be low outside the winter-rainfall region",
        colour="#b8abd1",
    ),
    C(
        9,
        "Low shrubland (fynbos)",
        "wrc2747:9",
        "wrc2747:9",
        "scs_sa:veld_2",
        note=f"DEA 8; {_X}",
        colour="#8fab39",
    ),
    C(
        10,
        "Low shrubland (Succulent Karoo)",
        "wrc2747:9",
        "wrc2747:9",
        "scs_sa:veld_1",
        note=f"DEA 9; {_X}; sparse Karoo cover may give more runoff",
        colour="#ac92c5",
    ),
    C(
        11,
        "Low shrubland (Nama Karoo)",
        "wrc2747:9",
        "wrc2747:9",
        "scs_sa:veld_1",
        note=f"DEA 9; {_X}; sparse Karoo cover may give more runoff",
        colour="#ac9cda",
    ),
    C(
        12,
        "Sparsely wooded grassland (5-10 % cc)",
        "scs_sa:veld_3",
        "wrc2747:7",
        "scs_sa:veld_1",
        note=f"DEA 7; {_WRC}",
        colour="#85d285",
    ),
    C(
        13,
        "Natural grassland",
        "scs_sa:veld_3",
        "wrc2747:7",
        "scs_sa:veld_1",
        note=f"DEA 7; {_WRC}",
        colour="#d2b485",
    ),
    # Water and wetlands -------------------------------------------------
    *[
        C(
            code,
            name,
            "const:100",
            "const:100",
            "const:100",
            note="Open water: rain on water becomes runoff",
            colour=col,
        )
        for code, name, col in (
            (14, "Natural rivers", "#00009f"),
            (15, "Natural estuaries & lagoons", "#041fa7"),
            (16, "Natural ocean, coastal", "#0639ab"),
            (17, "Natural lakes", "#0d50ac"),
            (18, "Natural pans (flooded at observation)", "#125fac"),
            (19, "Artificial dams (incl. canals)", "#1373b4"),
            (20, "Artificial sewage ponds", "#1d81b6"),
            (21, "Artificial flooded mine pits", "#1f8eb8"),
        )
    ],
    C(
        22,
        "Herbaceous wetlands (currently mapped)",
        "scs_sa:irrigated_pasture",
        "scs_sa:veld_3",
        "scs_sa:veld_1",
        "D",
        note="Saturated soils: group D; dense herbaceous cover (veld good); low end = irrigated "
        "pasture (WRC association for wetlands)",
        colour="#06dedc",
    ),
    C(
        23,
        "Herbaceous wetlands (previous mapped extent)",
        "scs_sa:irrigated_pasture",
        "scs_sa:veld_3",
        "scs_sa:veld_1",
        "D",
        note="As 22; check on site whether still wet (if drained, remove the D override)",
        colour="#06e0d0",
    ),
    C(
        24,
        "Mangrove wetlands",
        "wrc2747:5",
        "wrc2747:5",
        "scs_sa:woods_1",
        note=f"DEA 5 as in the composite; {_X}",
        colour="#9f1fec",
    ),
    # Bare --------------------------------------------------------------
    C(
        25,
        "Natural rock surfaces",
        "const:98",
        "const:98",
        "const:98",
        note="Bare rock: near impervious",
        colour="#ffffe0",
    ),
    *[
        C(
            code,
            name,
            "scs_sa:veld_1",
            "wrc2747:47",
            "scs_sa:urban_3",
            note=f"DEA {dea} with urban informal (bare) values; {_X}",
            colour=col,
        )
        for code, name, dea, col in (
            (26, "Dry pans", 41, "#dcdac5"),
            (27, "Eroded lands", 40, "#f9e0e0"),
            (30, "Bare riverbed material", 41, "#cdd2e0"),
            (31, "Other bare", 41, "#f5f5c8"),
        )
    ],
    C(
        28,
        "Sand dunes (terrestrial)",
        "scs_sa:forest_9",
        "wrc2747:32",
        "scs_sa:veld_2",
        note="Composite value (plantation mature); sand is group A in practice",
        colour="#f9f9c5",
    ),
    C(
        29,
        "Coastal sand dunes & beach sand",
        "scs_sa:forest_9",
        "wrc2747:32",
        "scs_sa:veld_2",
        note="Composite value (plantation mature); sand is group A in practice",
        colour="#f9f9a7",
    ),
    # Cultivation -------------------------------------------------------
    C(
        32,
        "Cultivated commercial permanent orchards",
        "scs_sa:forest_9",
        "wrc2747:32",
        "scs_sa:veld_2",
        note=f"DEA 16 with plantation values; {_X}",
        colour="#a62c39",
    ),
    C(
        33,
        "Cultivated commercial permanent vines",
        "scs_sa:forest_9",
        "wrc2747:32",
        "scs_sa:veld_2",
        note=f"DEA 19 with plantation values; {_X}",
        colour="#b31f5c",
    ),
    C(
        34,
        "Cultivated commercial sugarcane pivot irrigated",
        "scs_sa:forest_9",
        "wrc2747:32",
        "scs_sa:sugarcane_4",
        note=f"{_X}; high end = Table 5.1 sugarcane, partial cover",
        colour="#db0000",
    ),
    C(
        35,
        "Commercial permanent pineapples",
        "scs_sa:forest_9",
        "wrc2747:32",
        "scs_sa:veld_2",
        note=f"DEA 17 with plantation values; {_X}",
        colour="#9f3978",
    ),
    C(
        36,
        "Cultivated commercial sugarcane non-pivot",
        "scs_sa:forest_9",
        "wrc2747:32",
        "scs_sa:sugarcane_4",
        note=f"{_X}; high end = Table 5.1 sugarcane, partial cover",
        colour="#ff0000",
    ),
    C(
        37,
        "Cultivated emerging farmer sugarcane non-pivot",
        "scs_sa:forest_9",
        "wrc2747:32",
        "scs_sa:sugarcane_4",
        note=f"{_X}; high end = Table 5.1 sugarcane, partial cover",
        colour="#f64d6c",
    ),
    *[
        C(
            code,
            name,
            "scs_sa:veld_3",
            "wrc2747:7",
            "scs_sa:row_crops_1",
            note=f"Composite: grassland values; {_X}; high end = Table 5.1 row crops "
            "(straight row) because annual crops leave soil bare part of the year",
            colour=col,
        )
        for code, name, col in (
            (38, "Commercial annuals pivot irrigated", "#381a12"),
            (39, "Commercial annuals non-pivot irrigated", "#521f1c"),
            (40, "Commercial annual crops rain-fed / dryland", "#85402c"),
            (41, "Subsistence / small-scale annual crops", "#c5735f"),
        )
    ],
    C(
        42,
        "Fallow land & old fields (trees)",
        "scs_sa:woods_2",
        "wrc2747:6",
        "scs_sa:veld_2",
        note=f"DEA 6; {_X}",
        colour="#c1436c",
    ),
    C(
        43,
        "Fallow land & old fields (bush)",
        "scs_sa:woods_2",
        "wrc2747:6",
        "scs_sa:veld_2",
        note=f"DEA 6; {_X}",
        colour="#c55e82",
    ),
    C(
        44,
        "Fallow land & old fields (grass)",
        "scs_sa:veld_3",
        "wrc2747:7",
        "scs_sa:veld_1",
        note=f"DEA 7; {_X}",
        colour="#d27592",
    ),
    C(
        45,
        "Fallow land & old fields (bare)",
        "scs_sa:veld_1",
        "wrc2747:47",
        "scs_sa:urban_3",
        note=f"Urban informal (bare) values; {_X}",
        colour="#e0aab8",
    ),
    C(
        46,
        "Fallow land & old fields (low shrub)",
        "wrc2747:9",
        "wrc2747:9",
        "scs_sa:veld_1",
        note=f"DEA 9; {_X}",
        colour="#db90a9",
    ),
    # Residential --------------------------------------------------------
    C(
        47,
        "Residential formal (tree)",
        "wrc2747:48",
        "wrc2747:48",
        "scs_sa:urban_7",
        note=f"DEA 48; {_RES_NOTE}",
        colour="#ecdb0f",
    ),
    C(
        48,
        "Residential formal (bush)",
        "wrc2747:48",
        "wrc2747:49",
        "scs_sa:urban_6",
        note=f"DEA 49; {_RES_NOTE}",
        colour="#f6ec13",
    ),
    C(
        49,
        "Residential formal (low veg / grass)",
        "wrc2747:49",
        "wrc2747:50",
        "scs_sa:urban_5 + scs_sa:urban_6",
        note="DEA 50; WRC extrapolated; high end = uncalibrated (mean of 500 and 1000 m2 lots)",
        colour="#f9f81f",
    ),
    C(
        50,
        "Residential formal (bare)",
        "wrc2747:50",
        "wrc2747:51",
        "scs_sa:urban_5",
        note=f"DEA 51; {_RES_NOTE}",
        colour="#ffff29",
    ),
    C(
        51,
        "Residential informal (tree)",
        "wrc2747:44",
        "wrc2747:44",
        "wrc2747:45",
        note=f"DEA 44; {_WRC}",
        colour="#ec82ec",
    ),
    C(
        52,
        "Residential informal (bush)",
        "wrc2747:44",
        "wrc2747:45",
        "wrc2747:46",
        note=f"DEA 45; {_WRC}",
        colour="#f691e0",
    ),
    C(
        53,
        "Residential informal (low veg / grass)",
        "wrc2747:45",
        "wrc2747:46",
        "wrc2747:47",
        note=f"DEA 46; {_WRC}",
        colour="#f99fcf",
    ),
    C(
        54,
        "Residential informal (bare)",
        "wrc2747:46",
        "wrc2747:47",
        "scs_sa:urban_3",
        note=f"DEA 47; {_WRC}",
        colour="#ffc5cf",
    ),
    C(
        55,
        "Village scattered (bare only)",
        "wrc2747:67",
        "wrc2747:68",
        "wrc2747:68",
        note=f"DEA 68; {_WRC}; low end = village (low veg / grass)",
        colour="#ecc500",
    ),
    C(
        56,
        "Village dense (bare only)",
        "wrc2747:67",
        "wrc2747:68",
        "wrc2747:68",
        note=f"DEA 68; {_WRC}",
        colour="#ffd91f",
    ),
    C(
        57,
        "Smallholdings (tree)",
        "wrc2747:53",
        "wrc2747:53",
        "wrc2747:54",
        note=f"DEA 53; {_WRC}",
        colour="#ac7879",
    ),
    C(
        58,
        "Smallholdings (bush)",
        "wrc2747:53",
        "wrc2747:54",
        "wrc2747:55",
        note=f"DEA 54; {_WRC}",
        colour="#b89192",
    ),
    C(
        59,
        "Smallholdings (low veg / grass)",
        "wrc2747:54",
        "wrc2747:55",
        "wrc2747:56",
        note=f"DEA 55; {_WRC}",
        colour="#c49c9e",
    ),
    C(
        60,
        "Smallholdings (bare)",
        "wrc2747:55",
        "wrc2747:56",
        "scs_sa:fallow_1",
        note=f"DEA 56; {_WRC}",
        colour="#d2b8b8",
    ),
    C(
        61,
        "Urban recreational fields (tree)",
        "wrc2747:57",
        "wrc2747:57",
        "wrc2747:58",
        note=f"DEA 57; {_WRC}",
        colour="#bfff00",
    ),
    C(
        62,
        "Urban recreational fields (bush)",
        "wrc2747:57",
        "wrc2747:58",
        "wrc2747:60",
        note=f"DEA 58; {_WRC}",
        colour="#33ff33",
    ),
    C(
        63,
        "Urban recreational fields (grass)",
        "wrc2747:59",
        "wrc2747:59",
        "scs_sa:urban_2",
        note=f"DEA 59; {_WRC} (the report uses woods, low stormflow potential here); "
        "high end = open space, fair condition",
        colour="#66ff66",
    ),
    C(
        64,
        "Urban recreational fields (bare)",
        "scs_sa:urban_2",
        "wrc2747:60",
        "scs_sa:fallow_1",
        note=f"DEA 60; {_WRC}",
        colour="#99ff99",
    ),
    # Commercial, industrial, infrastructure, mines ------------------------
    C(
        65,
        "Commercial",
        "scs_sa:urban_4",
        "wrc2747:42",
        "const:98",
        note=f"DEA 42; {_WRC}",
        colour="#c49f0d",
    ),
    C(
        66,
        "Industrial",
        "scs_sa:urban_4",
        "wrc2747:43",
        "const:98",
        note=f"DEA 43; {_WRC}",
        colour="#8f8506",
    ),
    C(
        67,
        "Roads & rail (major linear)",
        "scs_sa:urban_12",
        "wrc2747:42",
        "const:98",
        note=f"Commercial values; {_X}; gravel road to paved as the range",
        colour="#f9dd03",
    ),
    C(
        68,
        "Mines: surface infrastructure",
        "scs_sa:urban_4",
        "wrc2747:35",
        "const:98",
        note=f"DEA 35; {_WRC}",
        colour="#e6e600",
    ),
    C(
        69,
        "Mines: extraction sites (open cast & quarries)",
        "scs_sa:urban_4",
        "wrc2747:36",
        "const:98",
        note=f"DEA 36; {_WRC}",
        colour="#b30606",
    ),
    C(
        70,
        "Mines: extraction sites (salt mines)",
        "scs_sa:veld_1",
        "wrc2747:47",
        "scs_sa:urban_3",
        note=f"Urban informal (bare) values; {_X}",
        colour="#c50606",
    ),
    C(
        71,
        "Mines: waste (tailings) & resource dumps",
        "scs_sa:veld_1",
        "wrc2747:47",
        "scs_sa:urban_3",
        note=f"Urban informal (bare) values; {_X}",
        colour="#d21d1a",
    ),
    C(
        72,
        "Land-fills",
        "scs_sa:veld_1",
        "wrc2747:47",
        "scs_sa:urban_3",
        note=f"Urban informal (bare) values; {_X}",
        colour="#f95479",
    ),
    C(
        73,
        "Fallow land & old fields (wetlands)",
        "scs_sa:veld_1",
        "wrc2747:47",
        "scs_sa:urban_3",
        note=f"Composite value (urban informal bare); {_X}",
        colour="#6ce7dc",
    ),
]

SANLC_2020.sort(key=lambda c: c.code)
assert [c.code for c in SANLC_2020] == list(range(1, 74))
