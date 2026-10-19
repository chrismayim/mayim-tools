"""
Word report for "Rational run-off coefficients", in the Mayim report
standard (title page, summary, method, inputs, parameters, results,
outputs, notes, references). python-docx and matplotlib are imported
lazily. Layout maps are rendered by QGIS through ``map_fn`` when given
(the QGIS wrapper supplies it); otherwise simple matplotlib maps are drawn.
Numbers have no thousands separators.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from mayim_tools._common.docx_report import Doc

from . import cells as cl
from . import lookup as lk
from . import tables as tb
from .core import METHODS, TOOL_NAME, TOOL_VERSION, RunResult

REFERENCES = [
    "SANRAL (2013). Drainage Manual, 6th edition. South African National Roads "
    "Agency SOC Ltd, Pretoria. Section 3.5.1.2 (Rational method run-off "
    "coefficient), Tables 3.7 and 3.8, Appendix 3A.",
    "SANRAL (2013). Drainage Manual Application Guide, 6th edition. South African "
    "National Roads Agency SOC Ltd, Pretoria. Example 3.1.",
    "Department of Forestry, Fisheries and the Environment (2024). South African "
    "National Land-Cover 2022 (SANLC 2022), 73 classes, 20 m. Pretoria. "
    "https://egis.environment.gov.za",
    "Horn, B.K.P. (1981). Hill shading and the reflectance map. Proceedings of the "
    "IEEE 69(1): 14-47.",
    "USDA-NRCS (2009). National Engineering Handbook, Part 630 Hydrology, Chapter 7: "
    "Hydrologic Soil Groups. United States Department of Agriculture, Natural "
    "Resources Conservation Service, Washington, D.C.",
    "Schulze, R.E., Schmidt, E.J. and Smithers, J.C. (1992). SCS-SA User Manual. "
    "ACRUcons Report 40, Department of Agricultural Engineering, University of "
    "Natal, Pietermaritzburg.",
]

ROLE_LABELS = {1: "Rural", 2: "Urban", 3: "Lakes and dams", 4: "Ineffective"}
MAPS = [
    ("role", "Land-cover role (rural, urban, lakes and dams, ineffective)"),
    ("slope_class", "Surface slope class (Cs, Table 3.7)"),
    ("perm_class", "Permeability class of the rural area (Cp, Table 3.7)"),
    ("veg_class", "Vegetation class of the rural area (Cv, Table 3.7)"),
    ("c1", "Rural run-off coefficient C1 = Cs + Cp + Cv per cell"),
    ("c50", "Run-off coefficient C for the 1:50-year flood per cell"),
]
FALLBACK_COLOURS = {
    "role": {1: "#00CC00", 2: "#FF0000", 3: "#0000FF", 4: "#7F7F7F"},
    "slope_class": {1: "#1a9850", 2: "#d9ef8b", 3: "#fdae61", 4: "#d73027"},
    "perm_class": {
        1: "#2166ac",
        2: "#67a9cf",
        3: "#d1e5f0",
        4: "#fddbc7",
        5: "#ef8a62",
        6: "#d6604d",
        7: "#b2182b",
    },
    "veg_class": {1: "#006400", 2: "#7fbf3f", 3: "#ffff00", 4: "#c8a064"},
}


def _f(v, nd=3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float) and not math.isfinite(v):
        return "-"
    return f"{v:.{nd}f}"


def _pct(v, nd=1) -> str:
    return "-" if v is None else _f(100.0 * v, nd)


def _ft_label(key):
    return tb.FT_LABELS.get(key, key)


def write_report(
    path: str,
    r: RunResult,
    context: dict,
    map_fn=None,
) -> list[str]:
    """Writes the report; returns notes (maps that could not be drawn)."""
    s = r.settings
    notes: list[str] = []
    d = Doc(house_style=True)
    cats = [c for c in r.catchments if c.coeff is not None]
    d.title_page(
        f"{TOOL_NAME}",
        "This report gives the run-off coefficients C of the Rational method for "
        "each catchment, worked out from the slope, soil and land cover with the "
        "SANRAL Drainage Manual (2013) Table 3.7, as input to the peak-flow "
        "calculations.",
        [
            ("Project number", context.get("project_number") or "-"),
            ("Project / site", context.get("project_name") or "-"),
            ("Prepared by", "Mayim Consulting"),
            ("Tool and version", f"{TOOL_NAME}, version {TOOL_VERSION}"),
            ("Software", context.get("software") or "-"),
            ("Report date", datetime.now().strftime("%Y-%m-%d")),
            ("Document status", "Draft"),
        ],
        subtitle=context.get("project_name") or None,
    )

    # ------------------------------------------------------------ summary
    d.heading("Summary", 1)
    d.para(
        f"Run-off coefficients were worked out for {len(r.catchments)} catchment(s) "
        f"on a {s.cell_m:g} m analysis grid. The rural coefficient C1 is the sum of "
        "the slope, permeability and vegetation components of Table 3.7, weighted "
        "by their area shares in the catchment's rural area, as on the Drainage "
        "Manual calculation form; the urban C2 is the area-weighted Table 3.7 "
        f"value ({s.position} end of each range); lakes and dams take C3 = "
        f"{s.c3:g}. The weighted C for each return period is C_T = alpha C1D Ft + "
        "beta C2 + gamma C3."
    )
    d.caption("Table", "Run-off coefficients per catchment")
    rows = []
    for c in r.catchments:
        k = c.coeff
        if k is None:
            rows.append([c.cid, _f(c.area_km2, 4)] + ["-"] * 9)
            continue
        rows.append(
            [
                c.cid,
                _f(c.area_km2, 4),
                _f(c.a_eff_km2, 4),
                _f(k.alpha, 2),
                _f(k.beta, 2),
                _f(k.gamma, 2),
                _f(k.c1d, 3) if k.alpha > 0 else "-",
                "steep/imp." if k.ft_class == "steep_impermeable" else "flat/perm.",
                _f(k.c2, 3) if k.beta > 0 else "-",
                _f(k.c_t[10], 3),
                _f(k.c_t[50], 3),
            ]
        )
    d.table(
        [
            "Catchment",
            "A (km²)",
            "A eff (km²)",
            "α",
            "β",
            "γ",
            "C1D",
            "Ft class",
            "C2",
            "C10",
            "C50",
        ],
        rows,
        [2.0, 1.6, 1.6, 1.1, 1.1, 1.1, 1.3, 1.9, 1.2, 1.3, 1.3],
    )
    n_flags = sum(len(c.flags) for c in r.catchments)
    if n_flags:
        d.para(
            f"{n_flags} flag(s) were raised; they are listed under Notes and "
            "limitations and should be checked before the coefficients are used."
        )

    # ------------------------------------------------------------ method
    d.heading("Tool and method", 1)
    for step in [
        f"The DEM is averaged to the {s.cell_m:g} m analysis cell (bilinear when "
        "the DEM is coarser) and the slope is worked out by the Horn (1981) method "
        "as a percentage grade. Slope classes: vleis and pans below 3 %, flat 3 to "
        "10 %, hilly 10 to 30 %, steep 30 % and over (a value on a limit goes to "
        "the class above).",
        "The land cover (SANLC 73 classes, nearest neighbour) gives each cell a "
        "role: rural, urban, lakes and dams, or ineffective. Ineffective cells "
        "(ocean, pans, sewage ponds, flooded mine pits) are left out of the "
        "effective area (Drainage Manual p. 3-29); alpha, beta and gamma are the "
        "rural, urban and water fractions of the effective area.",
        "Permeability comes from the hydrologic soil group (nearest neighbour): A "
        "very permeable, B permeable, C semi-permeable, D impermeable; the dual "
        "groups A/D, B/D and C/D are taken as D (undrained); the SCS-SA "
        "intermediate groups take the average of the neighbouring Cp values. "
        "Natural rock (SANLC 25) is taken as impermeable"
        + (
            ", and so are the wetlands (22, 23, 73)." if s.wetlands_impermeable else "."
        ),
        "Vegetation classes come from the land-cover lookup (thick bush and "
        "plantation, light bush and farm lands, grasslands, no vegetation), each "
        "with a lower and an upper class for the uncertainty.",
        "Per catchment, the area shares of the four classes of each component "
        "give Cs, Cp and Cv (Table 3.7, MAP band column) and C1 = Cs + Cp + Cv. "
        "Dolomite reduces C1: C1D = C1 (1 - D%) + C1 D% sum(Dfactor x Cs%), with "
        "the factors 0.10, 0.20, 0.35 and 0.50 for the four slope classes.",
        "The Ft class (Table 3.8) follows from a score: half the hilly-plus-steep "
        "share plus half the semi-permeable-plus-impermeable share of the rural "
        "area; 0.5 or more is steep and impermeable. Scores of 0.4 to 0.6 are "
        "flagged as borderline. An override field takes precedence.",
        f"Urban cells take the Table 3.7 row of their class, at the {s.position} "
        "end of its range; recreational fields are lawns (sandy for soil groups A "
        "and B, heavy soil otherwise; flat below 2 %, steep above 7 %, interpolated "
        "between). Ft is "
        + ("applied" if s.ft_on_c2 else "not applied")
        + " to C2 (Drainage Manual p. 3-20).",
        "C_T = alpha C1D Ft + beta C2 + gamma C3 for 2 to 100 years, and for the "
        "PMF with C1 = Cs + Cp,max + Cv,max, C2 = 1 and Ft = 1.",
        "Uncertainty: the lower (P05) and upper (P95) C1 combine the soil range "
        "(the 5 % and 95 % soil groups from the group probabilities) and the "
        "vegetation range (the lower and upper classes) by root-sum-square; the "
        "C2 range is the ends of the Table 3.7 ranges.",
    ]:
        d.numbered(step)

    # ------------------------------------------------------------ inputs
    d.heading("Inputs and settings", 1)
    d.caption("Table", "Inputs and settings")
    rows = [[k, str(v)] for k, v in context.get("inputs", [])]
    rows += [
        ["Analysis cell (m)", f"{s.cell_m:g}"],
        ["DEM cell (m) and resampling", f"{r.dem_res:.2f}, {r.dem_resampling}"],
        ["Soil-group method", METHODS.get(r.soil.method, r.soil.method)],
        [
            "Soil-group probabilities",
            r.soil.prob_path or "not used (C1 range from the land cover only)",
        ],
        ["Land-cover lookup", r.lookup.title],
        ["MAP (mm)", f"{s.map_mm:g}" if s.map_mm else "per catchment"],
        ["Urban C2 within the Table 3.7 range", s.position],
        ["Wetlands as impermeable", "yes" if s.wetlands_impermeable else "no"],
        ["C3 lakes and dams", f"{s.c3:g}"],
        ["Ft applied to C2", "yes" if s.ft_on_c2 else "no"],
        ["Dolomite polygons", str(len(s.dolomite)) if s.dolomite else "none"],
        ["Site override polygons", str(len(s.overrides)) if s.overrides else "none"],
        ["Run time", r.run_time_utc],
    ]
    d.table(["Item", "Value"], rows, [6.0, 10.5])

    # ------------------------------------------------------------ parameters
    d.heading("Parameters and calculations", 1)
    d.caption("Table", "Definitions")
    d.table(
        ["Parameter", "Value", "Definition / equation (source)"],
        [
            ["Cs, Cp, Cv", "Table 3.7", "Sum over the classes of share x table value (DM Table 3.7, p. 3-18)"],
            ["C1 rural", "-", "C1 = Cs + Cp + Cv (DM p. 3-19)"],
            ["C1D", "-", "C1 (1 - D%) + C1 D% sum(Dfactor x Cs%) (DM p. 3-19, App. 3A)"],
            ["Ft", "Table 3.8", "Return-period factor on C1 (DM Table 3.8, p. 3-19)"],
            ["C2 urban", "Table 3.7", "Area-weighted urban row value"],
            ["C3 lakes", f"{s.c3:g}", "Lakes and dams (DM p. 3-19: between 0 and 1)"],
            ["α, β, γ", "-", "Rural, urban and water fractions of the effective area"],
            ["C_T", "-", "αC1D·Ft + βC2 + γC3 (DM Appendix 3A form)"],
            ["C_PMF", "-", "α(Cs + Cp,max + Cv,max) + β + γC3 (DM p. 3-19)"],
        ],  # fmt: skip
        [3.4, 2.4, 10.7],
    )
    d.caption("Table", "Rural run-off coefficient components (DM Table 3.7)")
    rows, groups = [], []
    for name, classes, table in (
        ("Surface slope Cs", tb.SLOPE_CLASSES, tb.CS),
        ("Permeability Cp", tb.PERMEABILITY_CLASSES, tb.CP),
        ("Vegetation Cv", tb.VEGETATION_CLASSES, tb.CV),
    ):
        groups.append(len(rows))
        rows.append([name, "", "", ""])
        for c, vals in zip(classes, table, strict=True):
            rows.append([c] + [f"{v:.2f}" for v in vals])
    d.table(
        ["Class", "MAP < 600 mm", "MAP 600 - 900 mm", "MAP > 900 mm"],
        rows,
        [7.5, 3.0, 3.0, 3.0],
        group_rows=groups,
    )
    d.caption("Table", "Urban run-off coefficients (DM Table 3.7)")
    d.table(
        ["Use", "C2 low", "C2 high"],
        [[lab, f"{lo:.2f}", f"{hi:.2f}"] for lab, lo, hi in tb.URBAN_ROWS.values()],
        [9.5, 3.5, 3.5],
    )
    d.caption("Table", "Adjustment factor Ft for C1 (DM Table 3.8)")
    d.table(
        ["Catchment"] + [f"{t} years" for t in tb.RETURN_PERIODS],
        [[tb.FT_LABELS[k]] + [f"{v:.2f}" for v in vals] for k, vals in tb.FT.items()],
        [4.5] + [2.0] * 6,
    )

    # ------------------------------------------------------------ results
    d.heading("Results", 1)
    d.heading("Layout maps", 2)
    osm_used = []
    for key, caption in MAPS:
        _map_figure(d, r, key, caption, map_fn, notes, osm_used)

    d.heading("Class shares", 2)
    d.caption("Table", "Class shares of the rural area per catchment (%)")
    rows = []
    for c in cats:
        sh = c.shares
        if sh is None:
            continue
        rows.append(
            [c.cid]
            + [_pct(x, 0) for x in sh.slope]
            + [_pct(x, 0) for x in sh.perm]
            + [_pct(x, 0) for x in sh.veg]
        )
    d.table(
        ["Catchment", "S1", "S2", "S3", "S4", "P1", "P2", "P3", "P4", "V1", "V2", "V3", "V4"],
        rows,
        [2.1] + [1.2] * 12,
    )  # fmt: skip
    d.para(
        "S1-S4: vleis and pans, flat, hilly, steep; P1-P4: very permeable, "
        "permeable, semi-permeable, impermeable; V1-V4: thick bush and plantation, "
        "light bush and farm lands, grasslands, no vegetation."
    )
    d.caption("Table", "Rural components, dolomite and Ft class per catchment")
    rows = []
    for c in cats:
        k = c.coeff
        if k.alpha <= 0:
            continue
        rows.append(
            [
                c.cid,
                tb.MAP_BANDS[k.band],
                _f(k.cs),
                _f(k.cp),
                _f(k.cv),
                _f(k.c1),
                _pct(k.d_fraction, 0),
                _f(k.c1d),
                _f(k.ft_score, 2),
                ("steep/imp." if k.ft_class == "steep_impermeable" else "flat/perm.")
                + (" (override)" if k.ft_source == "override" else ""),
            ]
        )
    d.table(
        ["Catchment", "MAP band", "Cs", "Cp", "Cv", "C1", "D (%)", "C1D", "Ft score", "Ft class"],
        rows,
        [2.1, 2.2, 1.2, 1.2, 1.2, 1.3, 1.2, 1.3, 1.5, 3.3],
    )  # fmt: skip

    if any(c.coeff.beta > 0 for c in cats):
        d.caption(
            "Table", "Urban area per catchment: Table 3.7 rows (% of urban area) and C2"
        )
        keys = sorted({k for c in cats for k in c.urban_rows})
        labels = [
            _short_row(tb.URBAN_ROWS[k][0]) if k in tb.URBAN_ROWS else "Lawns"
            for k in keys
        ]
        rows = []
        for c in cats:
            if c.coeff.beta <= 0:
                continue
            rows.append(
                [c.cid]
                + [
                    _pct(c.urban_rows.get(k), 0) if k in c.urban_rows else "-"
                    for k in keys
                ]
                + [_f(c.coeff.c2), f"{_f(c.coeff.c2_p05, 2)} - {_f(c.coeff.c2_p95, 2)}"]
            )
        w = max(1.0, min(2.0, 10.0 / max(len(keys), 1)))
        d.table(
            ["Catchment"] + labels + ["C2", "Range"],
            rows,
            [2.1] + [w] * len(keys) + [1.3, 2.0],
        )

    d.heading("Run-off coefficient per return period", 2)
    d.caption("Table", "Weighted run-off coefficient C_T per catchment")
    rows = []
    for c in cats:
        k = c.coeff
        rows.append([c.cid] + [_f(k.c_t[t]) for t in (*tb.RETURN_PERIODS, "PMF")])
        if k.c_t_other_ft:
            rows.append(
                [f"{c.cid} (other Ft row)"]
                + [_f(k.c_t_other_ft[t]) for t in tb.RETURN_PERIODS]
                + ["-"]
            )
    d.table(
        ["Catchment"] + [f"{t} yr" for t in tb.RETURN_PERIODS] + ["PMF"],
        rows,
        [3.4] + [1.87] * 7,
    )

    d.heading("Uncertainty", 2)
    d.caption("Table", "Range of C1 and C2 per catchment")
    rows = []
    for c in cats:
        k = c.coeff
        main = "-"
        if k.alpha > 0 and math.isfinite(k.soil_share):
            main = "soil group" if k.soil_share >= 0.5 else "land cover"
        rows.append(
            [
                c.cid,
                _f(k.c1_p05) if k.alpha > 0 else "-",
                _f(k.c1) if k.alpha > 0 else "-",
                _f(k.c1_p95) if k.alpha > 0 else "-",
                main,
                _f(k.c2_p05) if k.beta > 0 else "-",
                _f(k.c2) if k.beta > 0 else "-",
                _f(k.c2_p95) if k.beta > 0 else "-",
            ]
        )
    d.table(
        ["Catchment", "C1 P05", "C1", "C1 P95", "Main source", "C2 low", "C2", "C2 high"],
        rows,
        [2.5, 1.8, 1.8, 1.8, 2.8, 1.9, 1.9, 2.0],
    )  # fmt: skip
    d.caption("Table", "Checks per catchment")
    rows = []
    for c in cats:
        nat = c.slope_native
        rows.append(
            [
                c.cid,
                _pct(c.slope_known, 0),
                _pct(c.perm_known, 0),
                _pct(c.forced_impermeable, 1),
                _pct(c.wetland_share, 1),
                _pct(c.shares.slope[0], 1) if c.shares else "-",
                _pct(nat[0], 1) if nat else "-",
                _pct(c.texture_agree, 0) if c.texture_agree is not None else "-",
            ]
        )
    d.table(
        [
            "Catchment",
            "Slope known (%)",
            "Soil known (%)",
            "Rock / wetland imp. (%)",
            "SANLC wetland (%)",
            "Vlei slope (%)",
            "Vlei slope native (%)",
            "Texture agrees (%)",
        ],
        rows,
        [2.4, 1.9, 1.9, 2.1, 2.0, 1.9, 2.2, 2.1],
    )
    d.para(
        "Slope and soil known: share of the rural area with a value. Vlei slope: "
        f"share of the rural area below 3 % on the {s.cell_m:g} m grid, and of the "
        "whole catchment at the DEM's own cell size; SANLC wetland is shown for "
        "comparison, since vleis and pans are classed by slope alone. Texture "
        "agrees: share of the rural area where the soil texture gives the same "
        "permeability class as the soil group (when a texture raster was given)."
    )

    if context.get("detail") or len(cats) <= 3:
        d.heading("Calculation forms", 2)
        for c in cats:
            _form(d, c)

    # ------------------------------------------------------------ outputs
    d.heading("Outputs", 1)
    d.caption("Table", "Files written")
    d.table(
        ["Output", "File"],
        [[a, b] for a, b in context.get("outputs", [])],
        [5.0, 11.5],
    )

    # ------------------------------------------------------------ notes
    d.heading("Notes and limitations", 1)
    for text in [
        "The Table 3.7 classes are broad and the coefficients are a guide. Check "
        "them against site knowledge, aerial imagery and, where available, flood "
        "records; the class shares can be adjusted with the override polygons.",
        "SANLC 2022 has an overall accuracy of about 84 %; individual classes "
        "(e.g. the tree / bush / grass / bare sub-classes of urban land) are less "
        "certain. The land cover and soil rasters are resampled by nearest "
        "neighbour; small features below the analysis cell may be missed.",
        "Hydrologic soil groups are mapped to the permeability classes by their "
        "infiltration rates; the Drainage Manual describes the classes by texture "
        "(p. 3-19), so the texture cross-check is given where a texture raster is "
        "available. SCS-SA groups use lower Ksat limits than NEH 630 and so "
        "tend to class soils as more permeable.",
        "Vleis and pans are classed by slope only (below 3 %).",
        "Dolomite: sinkholes and closed depressions may make part of a dolomitic "
        "catchment ineffective; check the effective area.",
        "Dams and lakes on the main stem attenuate floods; where they are a large "
        "part of the catchment, flood routing (DM Chapter 7) should be considered.",
        "The C1 and C50 rasters use the run's MAP band and are for viewing; where "
        "catchments overlap, the last one drawn shows.",
    ]:
        d.bullet(text)
    flags = [(c.cid, f) for c in r.catchments for f in c.flags]
    if flags:
        d.label("Flags")
        for cid, f in flags:
            d.bullet(f"{cid}: {f}")
    msgs = list(dict.fromkeys(r.warnings + context.get("warnings", []) + notes))
    if msgs:
        d.label("Processing messages")
        for m in msgs:
            d.bullet(m)
    if osm_used:
        from mayim_tools._common.report_maps import OSM_LICENCE_NOTE

        d.para(OSM_LICENCE_NOTE)

    d.heading("References", 1)
    for ref in REFERENCES:
        d.para(ref)
    d.doc.save(path)
    return notes


def _short_row(label: str) -> str:
    for pre in ("Residential - ", "Industry - ", "Business - "):
        label = label.replace(pre, "")
    return label[:1].upper() + label[1:]


def _form(d, c) -> None:
    """The Drainage Manual form for one catchment: share, table value and
    weighted value per class."""
    k = c.coeff
    sh = c.shares
    d.label(f"Catchment {c.cid}")
    if sh is None or k.alpha <= 0:
        d.para("No rural area.")
        return
    rows, groups = [], []
    for name, classes, table, vals, comp in (
        ("Surface slope Cs", tb.SLOPE_CLASSES, tb.CS, sh.slope, k.cs),
        ("Permeability Cp", tb.PERMEABILITY_CLASSES, tb.CP, sh.perm, k.cp),
        ("Vegetation Cv", tb.VEGETATION_CLASSES, tb.CV, sh.veg, k.cv),
    ):
        groups.append(len(rows))
        rows.append([f"{name} = {comp:.4f}", "", "", ""])
        for cls, row, f in zip(classes, table, vals, strict=True):
            v = row[k.band]
            rows.append([cls, _pct(f, 1), f"{v:.2f}", f"{f * v:.4f}"])
    d.caption(
        "Table",
        f"Catchment {c.cid}: rural run-off coefficient (MAP {k.map_mm:g} mm, band "
        f"{tb.MAP_BANDS[k.band]} mm)",
    )
    d.table(
        ["Class", "Share (%)", "Table value", "Weighted"],
        rows,
        [7.5, 3.0, 3.0, 3.0],
        groups,
    )
    d.para(
        f"C1 = {k.c1:.4f}; dolomite D = {100 * k.d_fraction:.1f} %, weighted factor "
        f"{k.d_factor:.3f}, C1D = {k.c1d:.4f}; Ft class {_ft_label(k.ft_class)} "
        f"(score {k.ft_score:.2f}{', override' if k.ft_source == 'override' else ''}); "
        f"α = {k.alpha:.3f}, β = {k.beta:.3f}, γ = {k.gamma:.3f}; "
        f"C2 = {k.c2:.3f}; C3 = {k.c3:g}."
    )


def _map_figure(d, r, key, caption, map_fn, notes, osm_used) -> None:
    from mayim_tools._common.report_maps import (
        PLAN_DPI,
        PLAN_PLOT_WIDTH_CM,
        figure_rendered_map,
    )
    from mayim_tools.soil.regional_parameterisation import figures

    if map_fn is not None:
        try:
            pm = map_fn(key, PLAN_PLOT_WIDTH_CM, PLAN_DPI)
        except Exception as exc:  # noqa: BLE001 - the report must still be written
            pm = None
            notes.append(
                f"Map '{caption}': QGIS render failed ({exc}); simple map drawn."
            )
        if pm is not None:
            try:
                stream = figure_rendered_map(pm, figures.chart_style())
                if pm.attribution:
                    osm_used.append(True)
                    caption += " (background: OpenStreetMap)"
                d.picture(stream, caption)
                return
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Map '{caption}' could not be composed ({exc}).")
    try:
        d.picture(_fallback_map(r, key), caption)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Map '{caption}' could not be drawn ({exc}).")


def _fallback_map(r, key):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    from mayim_tools.soil.regional_parameterisation import figures

    style = figures.chart_style()
    pv = {
        "role": r.preview["role"].astype(float),
        "slope_class": r.preview["scls"].astype(float) + 1,
        "perm_class": np.where(r.preview["role"] == 1, r.preview["perm"] + 1.0, 0),
        "veg_class": r.preview["veg"].astype(float) + 1,
        "c1": r.preview["c1"],
        "c50": r.preview["c50"],
    }[key]
    g = r.grid
    ext = (g.xmin, g.xmax, g.ymin, g.ymax)
    h = 16.5 * (g.ymax - g.ymin) / max(g.xmax - g.xmin, 1e-9) * 0.8 + 3.0
    with style.context():
        fig = style.figure(min(max(h, 8.0), 22.0))
        ax = fig.add_subplot()
        if key in FALLBACK_COLOURS:
            cols = FALLBACK_COLOURS[key]
            vals = sorted(cols)
            cmap = ListedColormap([cols[v] for v in vals])
            norm = BoundaryNorm([v - 0.5 for v in vals] + [vals[-1] + 0.5], cmap.N)
            arr = np.ma.masked_where(~np.isin(pv, vals), pv)
            ax.imshow(arr, extent=ext, cmap=cmap, norm=norm, interpolation="nearest")
            labels = _class_labels(key)
            present = sorted({int(v) for v in np.unique(arr.compressed())})
            handles = [
                Patch(facecolor=cols[v], edgecolor="black", label=labels.get(v, str(v)))
                for v in present
            ]
            if handles:
                ax.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.08),
                    ncol=3,
                    frameon=False,
                    prop=style.font_props(style="italic"),
                )
        else:
            arr = np.ma.masked_invalid(pv)
            im = ax.imshow(arr, extent=ext, cmap="RdYlGn_r", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, shrink=0.7, label="C")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        style.axes(ax, grid=False)
        return style.png(fig)


def _class_labels(key):
    if key == "role":
        return ROLE_LABELS
    if key == "slope_class":
        return {i + 1: c for i, c in enumerate(tb.SLOPE_CLASSES)}
    if key == "perm_class":
        return {k + 1: v for k, v in cl.PERM2_LABELS.items()}
    if key == "veg_class":
        return {i + 1: c for i, c in enumerate(tb.VEGETATION_CLASSES)}
    return {}


__all__ = ["write_report", "REFERENCES", "lk"]
