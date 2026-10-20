"""Word report for SCS Curve Numbers (python-docx and matplotlib, imported
lazily). Numbers without thousands separators; guideline-neutral wording."""

from __future__ import annotations

import math
import os

from mayim_tools._common.docx_report import Doc

from . import tables as tb
from .core import (
    AMC_OPTIONS,
    CONF_LABELS,
    CONF_TEXT,
    OUT_FILE,
    TOOL_NAME,
    TOOL_VERSION,
)

REPORT_FILE = "curve_numbers_report.docx"

REFERENCES = [
    "Grove, M., Harbor, J. and Engel, B. (1998). Composite vs. distributed curve "
    "numbers: effects on estimates of storm runoff depths. Journal of the American "
    "Water Resources Association 34(5): 1015-1023.",
    "Hawkins, R.H., Hjelmfelt, A.T. and Zevenbergen, A.W. (1985). Runoff probability, "
    "storm depth, and curve numbers. Journal of Irrigation and Drainage Engineering "
    "111(4): 330-340.",
    "Schulze, R.E., Schmidt, E.J. and Smithers, J.C. (2004). Visual SCS-SA User Manual, "
    "Version 1.0: PC-based SCS design flood estimates for small catchments in southern "
    "Africa. ACRUcons Report 52, School of Bioresources Engineering and Environmental "
    "Hydrology, University of KwaZulu-Natal, Pietermaritzburg.",
    "Sobhani, G. (1976). A review of selected watershed design methods for possible "
    "adaptation to Iranian conditions. Unpublished M.Sc. thesis, Utah State "
    "University, Logan, Utah.",
    "USDA-NRCS (2004). National Engineering Handbook, Part 630 Hydrology, Chapter 9: "
    "Hydrologic Soil-Cover Complexes, and Chapter 10: Estimation of Direct Runoff from "
    "Storm Rainfall. United States Department of Agriculture, Natural Resources "
    "Conservation Service, Washington, D.C.",
    "USDA-SCS (1986). Urban Hydrology for Small Watersheds. Technical Release 55, 2nd "
    "edition. United States Department of Agriculture, Soil Conservation Service, "
    "Washington, D.C.",
]
WRC_REF = (
    "Loots, I. and Smithers, J.C. (2020). Design Flood Estimation in Urban Areas in "
    "South Africa: Preliminary Results from Tshwane Case Studies. WRC Report No. "
    "2747/1/20, Water Research Commission, Pretoria."
)
SANLC_REF = (
    "Department of Forestry, Fisheries and the Environment. South African National "
    "Land-Cover (SANLC) 2018 and 2020 datasets. Pretoria. "
    "https://egis.environment.gov.za"
)
WORLDCOVER_REF = (
    "Zanaga, D., Van De Kerchove, R., Daems, D., et al. (2022). ESA WorldCover 10 m "
    "2021 v200. Zenodo. https://doi.org/10.5281/zenodo.7254221"
)


def _f(v, nd=1) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}"


def _eq_key(row):
    return next(k for k in row if k.startswith("Runoff-equivalent CN design"))


def plain_confidence_text() -> str:
    return (
        "A curve number is never known exactly. Two things make it uncertain "
        "here. The first is the soil: its hydrologic group comes from soil maps "
        "and their own uncertainty, so the tool also looks up the curve number "
        "for the group the soil would have if it drained more slowly or more "
        "quickly than the best estimate. The second is the land cover: a map "
        "class such as grassland covers veld in good, fair or poor condition, "
        "and the curve-number tables give a different value for each. The tool "
        "combines the two into a likely middle value (the design curve number) "
        "and a range from a low value it is unlikely to fall below (P05, less "
        "runoff) to a high value it is unlikely to exceed (P95, more runoff). For "
        "a whole catchment the curve numbers of all cells are turned into the "
        "single value that gives the same total runoff for the design rainfall. "
        "The confidence is judged from how wide that range is: High when it is "
        "under 10 curve-number points, Medium from 10 to 20, and Low above 20. "
        "The report also says which of the two causes contributes most, because "
        "that tells you what to check: a soil survey or infiltration tests when "
        "the soil dominates, a site visit to judge the veld, crop or "
        "development when the land cover does. A Low rating does not mean the "
        "value is wrong; it means the mapped information cannot pin it down."
    )


def _class_lookup_rows(result):
    rows = []
    for code in sorted(result.lookup.classes):
        e = result.lookup.classes[code]
        if code not in result.by_class:
            continue
        cells = []
        for case in ("low", "design", "high"):
            spec = e.rows.get(case, "")
            try:
                cells.append(tb.describe(spec) if spec else "-")
            except KeyError:
                cells.append(spec)
        rows.append(
            [f"{code} {e.name}", cells[1], cells[0], cells[2], e.override or "-"]
        )
    return rows


def _recommendations(result) -> list[str]:
    rows = result.zone_rows
    zones = rows[1:] if len(rows) > 1 else rows
    out = []
    soil_z = [r["Zone"] for r in zones if r["Main uncertainty source"] == "soil group"]
    cond_z = [
        r["Zone"]
        for r in zones
        if r["Main uncertainty source"] == "land-cover condition"
    ]
    low_z = [r["Zone"] for r in zones if r["Confidence"] == "Low"]
    if soil_z:
        out.append(
            "Soil group drives the uncertainty in " + ", ".join(soil_z) + ": confirm "
            "the groups with a few infiltration tests or a soil inspection, and "
            "re-run with the confirmed groups."
        )
    if cond_z:
        out.append(
            "Land-cover condition drives the uncertainty in " + ", ".join(cond_z) + ": "
            "judge the veld, crop and plantation condition on site (or from recent "
            "imagery) and set the design rows of the lookup to the observed condition."
        )
    if low_z:
        out.append(
            "Confidence is Low in " + ", ".join(low_z) + ": carry the P05 and P95 "
            "curve numbers through the flood calculation as a sensitivity test, and "
            "state the range with the result."
        )
    sanlc = result.settings.lookup_key.startswith("sanlc")
    if sanlc:
        urban = [c for c in result.by_class if 47 <= c <= 72]
        if urban:
            out.append(
                "Urban curve numbers come from WRC Report 2747/1/20, calibrated on "
                "two Tshwane catchments and called preliminary by its authors (they "
                "may overestimate large floods and underestimate small ones). Check "
                "that the development types match those catchments, and use the "
                "P95 values (uncalibrated Table 5.1 for formal residential) as a "
                "cautious case."
            )
        out.append(
            "SANLC splits urban and fallow classes by tree, bush, grass and bare "
            "cover; this split is the least reliable part of the map. Check the "
            "dominant classes on recent imagery."
        )
        flags = [
            c
            for c in (8, 10, 11, 38, 39, 40, 41, 34, 36, 37, 7)
            if c in result.by_class
        ]
        if flags:
            names = ", ".join(result.lookup.classes[c].name.lower() for c in flags)
            out.append(
                "Design values that may be low for the cover (" + names + "): the "
                "lookup keeps the composite values (derived from WRC 2747/1/20), "
                "with Table 5.1 alternatives (row crops, veld in poor condition, "
                "sugarcane, compact plantation floor) as the high end of the range."
            )
    if 50 in result.by_class and result.settings.lookup_key.startswith("worldcover"):
        out.append(
            "Built-up land: WorldCover marks buildings and roads but not their "
            "impervious share. Replace the built-up rows of the lookup with the "
            "development type actually present (lot sizes, commercial or industrial "
            "areas), or split the built-up area by hand."
        )
    if any(
        result.lookup.classes[c].override
        for c in result.by_class
        if c in result.lookup.classes
    ):
        out.append(
            "Wetland and mangrove cells are set to group D whatever the soil map "
            "shows (saturated soils). Check their extent on site; WorldCover's "
            "herbaceous wetland class is one of its least accurate."
        )
    if result.settings.amc == "none" and result.lookup.table in ("scs_sa", "custom"):
        out.append(
            "These are average-condition curve numbers (CN-II). The SCS-SA manual "
            "adjusts them for the catchment's soil moisture before design use "
            "(Median Condition Method for return periods above 20 years, Joint "
            "Association Method up to 20 years); re-run with the Median Condition "
            "option and the change in storage for the catchment, or apply the "
            "adjustment in the design-flood calculation."
        )
    out.append(
        "Use the runoff-equivalent curve number, not the plain area-weighted "
        "mean, for a lumped catchment: averaging curve numbers understates runoff "
        "where high and low values mix. It depends on the design rainfall depth; "
        "re-run with the depth of the storm being designed for."
    )
    return out


def write_report(result) -> str:
    from . import figures

    d = Doc()
    s = result.settings
    notes = []
    rows = result.zone_rows
    eq_key = _eq_key(rows[0])
    p = s.design_depth_mm
    table_name = tb.TABLES.get(result.lookup.table, ("custom lookup",))[0]
    method = {"NEH630": "NEH 630", "SCSSA": "SCS-SA"}.get(
        result.hsg.method, "user-supplied"
    )

    d.heading("Curve Numbers", 1)

    # 1. Summary ---------------------------------------------------------
    d.heading("1. Summary", 2)
    d.para(
        f"This report gives curve numbers for {_f(rows[0]['Area with CN (km2)'], 2)} "
        f"km2 from the land cover ({os.path.basename(s.landcover_path)}), the "
        f"{method} hydrologic soil groups and the {result.lookup.name} lookup. "
        f"Antecedent moisture: {AMC_OPTIONS[s.amc]}. Zone values are given as the "
        f"runoff-equivalent curve number for a design rainfall of {p:g} mm."
    )
    for r in rows:
        d.bullet(
            f"{r['Zone']}: runoff-equivalent CN {_f(r[eq_key])} (P05 "
            f"{_f(r['Runoff-equivalent CN P05'])}, P95 {_f(r['Runoff-equivalent CN P95'])}); "
            f"area-weighted mean {_f(r['Mean CN design'])}; confidence {r['Confidence']}; "
            f"main uncertainty from the {r['Main uncertainty source']}; mostly "
            f"{r['Dominant land cover'].lower()} ({_f(r['Dominant land cover share (%)'], 0)} %) "
            f"on group {r['Dominant soil group']} ({_f(r['Dominant soil group share (%)'], 0)} %)."
        )

    # 2. Purpose ---------------------------------------------------------
    d.heading("2. Purpose and scope", 2)
    d.para(
        "The SCS curve-number method turns rainfall into stormflow depth with a "
        "single index of the catchment's runoff potential, the curve number (CN), "
        "set by the hydrologic soil group, the land cover and its treatment or "
        "condition. This tool maps the CN cell by cell from a land-cover raster "
        "and a soil-group raster through a lookup of a published curve-number "
        "table, with the range the inputs allow, and summarises it per zone."
    )
    d.heading("How confidence is judged", 3)
    d.para(plain_confidence_text())

    # 3. Data ------------------------------------------------------------
    d.heading("3. Data", 2)
    inp = [
        ["Land cover", s.landcover_path],
        ["Soil groups", f"{s.hsg_path} ({method})"],
        [
            "Soil range",
            result.hsg.by_ksat_path or "none (land-cover condition range only)",
        ],
        ["Lookup", f"{result.lookup.name}; {result.lookup.source}"],
        ["Curve-number table", table_name],
        [
            "Output grid",
            f"{result.grid.width} x {result.grid.height} cells of {result.grid.res:g} (land-cover grid)",
        ],
    ]
    d.caption("Table", "Inputs")
    d.table(["Item", "Value"], inp, widths_cm=[4.0, 12.0])
    try:
        d.picture(figures.landcover_map(result, _class_colours(result)), "Land cover")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Land-cover map not drawn ({exc}).")

    # 4. Methods ---------------------------------------------------------
    d.heading("4. Methods", 2)
    d.heading("4.1 Curve-number table and lookup", 3)
    if result.lookup.table == "scs_sa":
        d.para(
            "Curve numbers come from Table 5.1 of the Visual SCS-SA User Manual "
            "(Schulze, Schmidt & Smithers, 2004), which gives initial curve numbers "
            "(CN-II) for the groups A, A/B, B, B/C, C, C/D and D. The manual notes "
            "that the values derive from US experience, that southern African covers "
            "may need interpolation between rows, and that values below about 50 "
            "call for caution."
        )
        if result.settings.lookup_key.startswith("sanlc"):
            d.para(
                "For the South African National Land Cover (SANLC 2018 / 2020 / 2022, 73 "
                "classes) the lookup is a composite. Urban, plantation and natural "
                "vegetation classes take the preliminary calibrated curve numbers "
                "of WRC Report 2747/1/20 (Loots & Smithers, 2020, Table 9.7), which "
                "were derived for the DEA 2013/14 land-use types in two Tshwane "
                "catchments; the 2018 / 2020 classes are crosswalked to those types. "
                "Other classes use the nearest Table 5.1 row. Four misprints in "
                "Table 9.7 are corrected (woods B/C 72, township C, village D). "
                "Open water is 100 and bare rock 98. Herbaceous wetlands are set to "
                "group D (saturated soils)."
            )
    elif result.lookup.table == "tr55":
        d.para(
            "Curve numbers come from TR-55 (USDA-SCS, 1986) Tables 2-2a to 2-2d, the "
            "same values as NEH 630 Chapter 9, for the groups A to D. Intermediate "
            "groups, where present, take the mean of the neighbouring columns."
        )
    else:
        d.para(f"Curve numbers come from the user's lookup ({result.lookup.source}).")
    d.para(
        "Each land-cover class is given three rows: a design row and a plausible "
        "range from a low row (less runoff, e.g. veld in good condition) to a high "
        "row (more runoff, e.g. veld in poor condition). The rows used are below; "
        "the full lookup with its values is written to cn_lookup_used.csv, which "
        "can be edited and used as a custom lookup."
    )
    d.caption("Table", "Lookup rows for the classes present")
    d.table(
        ["Class", "Design", "Low (less runoff)", "High (more runoff)", "Group"],
        _class_lookup_rows(result),
        widths_cm=[2.6, 4.3, 3.8, 3.8, 1.5],
    )
    d.heading("4.2 Soil groups", 3)
    d.para(
        "The design CN uses the recommended group of each soil cell. The soil "
        "raster is resampled to the land-cover grid by nearest neighbour, so each "
        "land-cover cell takes the group of the soil cell it falls in."
        + (
            " NEH 630 dual groups (A/D, B/D, C/D) are taken as "
            + (
                "undrained (group D)."
                if s.dual == "undrained"
                else "drained (first letter)."
            )
            if result.hsg.method in ("NEH630", "")
            else ""
        )
    )
    d.heading("4.3 Uncertainty", 3)
    d.para(
        "The soil range is the CN at the group of the Ksat P5 and P95 "
        "(hsg_*_by_ksat.tif). The condition range is the lookup's low and high "
        "rows at the recommended group. Both are read as 90 % ranges of "
        "independent errors and combined on each side by root-sum-square: "
        "P05 = design - sqrt(dS^2 + dC^2), where dS and dC are the distances from "
        "the design CN to the soil and condition limits; likewise for P95. The "
        "soil share of the uncertainty is dS^2 / (dS^2 + dC^2), summed over both "
        "sides. Zone means of the cell P05 and P95 assume fully related errors "
        "and are therefore wide."
    )
    d.heading("4.4 Antecedent moisture", 3)
    amc_text = {
        "none": "No adjustment: the curve numbers are CN-II (average condition).",
        "arc1": "ARC I (dry): CN-I = CN-II / (2.281 - 0.01281 CN-II), the fit of "
        "Hawkins et al. (1985) to NEH 630 Chapter 10 Table 10-1.",
        "arc3": "ARC III (wet): CN-III = CN-II / (0.427 + 0.00573 CN-II), the fit of "
        "Hawkins et al. (1985) to NEH 630 Chapter 10 Table 10-1.",
        "sobhani": "Near-saturated: CNw = CN-II / (0.4036 + 0.0059 CN-II) (Sobhani, "
        "1976; SCS-SA Eq. 4), meant for probable maximum flood estimates.",
        "median": "SCS-SA Median Condition Method (Eq. 3): CNf = 1100 / (1100 / CN-II "
        f"- dS / 25.4), with a change in storage dS = {s.delta_s_mm:g} mm.",
    }[s.amc]
    d.para(
        amc_text + " The same adjustment is applied to the design, P05 and P95 values."
    )
    d.heading("4.5 Zone values", 3)
    d.para(
        "Runoff is not linear in CN, so the mean CN of a mixed catchment gives "
        "less runoff than the sum of its parts (Grove, Harbor & Engel, 1998). For "
        f"each zone the stormflow Q of every cell is computed for P = {p:g} mm with "
        f"Ia = {result.ia_ratio:g} S, and the runoff-equivalent CN is the single "
        "value that gives the zone's mean Q. It is computed for the design, P05 "
        "and P95 curve numbers."
    )
    d.caption(
        "Table", "Confidence classes (range of the runoff-equivalent CN, P95 - P05)"
    )
    d.table(
        ["Class", "Range"],
        [[CONF_LABELS[k], CONF_TEXT[k]] for k in CONF_LABELS],
        widths_cm=[3.0, 13.0],
    )

    # 5. Results ---------------------------------------------------------
    d.heading("5. Results", 2)
    try:
        d.picture(
            figures.cn_maps(result), "Design curve number and its range (P95 - P05)"
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Curve-number maps not drawn ({exc}).")
    d.caption("Table", "Curve numbers per land-cover class (whole raster)")
    d.table(
        [
            "Class",
            "Area (km2)",
            "Share (%)",
            "Mean CN",
            "P05",
            "P95",
            "Soil share of uncertainty (%)",
        ],
        [
            [
                f"{c['Code']} {c['Class']}",
                _f(c["Area (km2)"], 3),
                _f(c["Share (%)"]),
                _f(c["Mean CN design"]),
                _f(c["Mean CN P05"]),
                _f(c["Mean CN P95"]),
                _f(c["Soil share of uncertainty (%)"], 0),
            ]
            for c in result.class_rows
        ],
    )
    try:
        d.picture(figures.histogram(result), "Area by design curve number")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Histogram not drawn ({exc}).")
    d.caption("Table", f"Zone curve numbers (runoff-equivalent for P = {p:g} mm)")
    d.table(
        [
            "Zone",
            "Area (km2)",
            "Mean CN",
            "Runoff-equivalent CN",
            "P05",
            "P95",
            "Q design (mm)",
            "Confidence",
        ],
        [
            [
                r["Zone"],
                _f(r["Area with CN (km2)"], 2),
                _f(r["Mean CN design"]),
                _f(r[eq_key]),
                _f(r["Runoff-equivalent CN P05"]),
                _f(r["Runoff-equivalent CN P95"]),
                _f(r["Stormflow design (mm)"]),
                r["Confidence"],
            ]
            for r in rows
        ],
        widths_cm=[2.8, 1.7, 1.6, 2.6, 1.4, 1.4, 1.9, 2.0],
    )
    try:
        d.picture(figures.zone_ranges(result), "Zone curve numbers and their ranges")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Zone range figure not drawn ({exc}).")
    try:
        d.picture(
            figures.uncertainty_split(result), "What drives the uncertainty per zone"
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Uncertainty figure not drawn ({exc}).")

    # 6. Design values ---------------------------------------------------
    d.heading("6. Design values, recommendations, limitations and risks", 2)
    d.heading("6.1 Candidate design values", 3)
    d.table(
        ["Zone", "Design CN", "Sensitivity range", "Confidence", "Main uncertainty"],
        [
            [
                r["Zone"],
                _f(r[eq_key]),
                f"{_f(r['Runoff-equivalent CN P05'])} - {_f(r['Runoff-equivalent CN P95'])}",
                r["Confidence"],
                r["Main uncertainty source"],
            ]
            for r in rows
        ],
        widths_cm=[3.6, 2.4, 3.6, 2.6, 3.8],
    )
    d.para(
        f"Design CN = runoff-equivalent CN for P = {p:g} mm; the sensitivity range "
        "is its P05 to P95. They are candidates for the designer's judgement, not "
        "prescribed values."
    )
    d.heading("6.2 Recommendations", 3)
    for text in _recommendations(result):
        d.bullet(text)
    d.heading("6.3 Limitations", 3)
    for text in (
        _accuracy_text(result),
        "The curve-number tables come from US experience (Table 5.1 adapts them for "
        "southern Africa); values below about 50 are uncertain.",
        "The soil and condition ranges are combined as independent 90 % ranges; "
        "the soil range is the change of group across the Ksat P5 to P95, not a "
        "full probability distribution.",
        "Soil groups are resampled from a coarser grid by nearest neighbour; "
        "boundaries between groups are blocky at the land-cover resolution.",
        "The land cover is one year's map; use the cover expected over the design "
        "life (development, plantation cycles, cropping).",
    ):
        d.bullet(text)
    d.heading("6.4 Risks", 3)
    for text in (
        "Underestimating runoff from built-up land where the development is denser "
        "than the lookup assumes.",
        "Underestimating runoff from a mixed catchment if the plain mean CN is used "
        "instead of the runoff-equivalent CN.",
        "Mixing tables and soil groups from different methods (NEH 630 groups with "
        "the SCS-SA table, or the reverse) shifts the CN systematically.",
        "Applying CN-II where the governing guideline expects an antecedent-moisture "
        "adjustment.",
    ):
        d.bullet(text)

    # 7. References -------------------------------------------------------
    d.heading("7. References", 2)
    refs = list(REFERENCES)
    if s.lookup_key.startswith("worldcover"):
        refs.append(WORLDCOVER_REF)
    if s.lookup_key.startswith("sanlc"):
        refs += [WRC_REF, SANLC_REF]
    for ref in sorted(refs):
        d.para(ref)

    # Annexes --------------------------------------------------------------
    d.heading("Annex A. Run settings", 2)
    d.table(
        ["Setting", "Value"],
        [
            ["Tool", f"{TOOL_NAME} (mayim_tools) v{TOOL_VERSION}"],
            ["Run time", result.run_time_utc],
            ["Lookup", result.lookup.name],
            ["Antecedent moisture", AMC_OPTIONS[s.amc]],
            ["Initial abstraction ratio", f"{result.ia_ratio:g}"],
            ["Design rainfall depth (mm)", f"{p:g}"],
            ["Zones", ", ".join(result.zone_names) or "none (whole raster)"],
        ],
        widths_cm=[5.0, 11.0],
    )
    d.heading("Annex B. Output files", 2)
    files = sorted(
        {os.path.basename(f) for f in result.files}
        | {OUT_FILE, "cn_metadata.csv", "cn_layers.qlr", REPORT_FILE}
    )
    d.table(["File"], [[f] for f in files], widths_cm=[16.0])
    if notes or result.warnings:
        d.heading("Annex C. Warnings", 2)
        for n in result.warnings + notes:
            d.bullet(n)
    path = os.path.join(s.out_dir, REPORT_FILE)
    d.doc.save(path)
    return path


def _class_colours(result) -> dict:
    from mayim_tools.soil._common.export import class_colour

    colours = {c: e.colour or class_colour(c) for c, e in result.lookup.classes.items()}
    if result.settings.lookup_key.startswith("worldcover"):
        from mayim_tools.land_cover.land_cover_extract.catalogue import CLASSES

        colours.update({c.code: c.colour for c in CLASSES})
    return colours


def _accuracy_text(result) -> str:
    key = result.settings.lookup_key
    if key.startswith("sanlc"):
        lead = (
            "SANLC 2020 scored 91.5 % overall accuracy in an independent comparison "
            "with field data, but less for some classes (grassland about 58 % "
            "producer's accuracy)"
        )
    elif key.startswith("worldcover"):
        lead = (
            "Land-cover maps are about 75-80 % accurate overall (WorldCover 2021: "
            "76.7 % globally) and much less for some classes"
        )
    else:
        lead = "The land-cover map has its own classification errors"
    return (
        lead + "; a class says nothing about treatment, condition or "
        "imperviousness, which the lookup assumes."
    )
