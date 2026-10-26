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


TITLE = "SCS Curve Numbers"


def write_report(result, path: str | None = None) -> str:
    """Writes the report (Mayim report standard) to ``path`` (default: the
    legacy name in out_dir) and returns the path."""
    import datetime as _dt

    from . import figures

    d = Doc(house_style=True)
    s = result.settings
    ctx = s.report_context or {}
    notes = []
    rows = result.zone_rows
    eq_key = _eq_key(rows[0])
    p = s.design_depth_mm
    table_name = tb.TABLES.get(result.lookup.table, ("custom lookup",))[0]
    method = {"NEH630": "NEH 630", "SCSSA": "SCS-SA"}.get(
        result.hsg.method, "user-supplied"
    )

    def picture(fn, caption):
        try:
            d.picture(fn(), caption)
        except Exception as exc:  # noqa: BLE001 - the report must not fail
            notes.append(f"Figure not drawn: {caption} ({exc}).")

    d.title_page(
        TITLE,
        "This report maps SCS curve numbers from the land cover and the "
        "hydrologic soil groups through a published curve-number table, with "
        "the range the inputs allow, and gives runoff-equivalent zone values "
        "for lumped design-flood models.",
        [
            ("Project number", ctx.get("project_number") or "-"),
            ("Project / site", ctx.get("project_name") or "-"),
            ("Prepared by", "Mayim Consulting"),
            ("Tool and version", f"{TOOL_NAME}, version {TOOL_VERSION}"),
            ("Software", ctx.get("software") or "-"),
            ("Report date", _dt.date.today().isoformat()),
            ("Document status", "Draft"),
        ],
        subtitle=ctx.get("project_name") or None,
    )

    # Summary --------------------------------------------------------------
    d.heading("Summary", 1)
    d.para(
        f"Curve numbers were mapped for {_f(rows[0]['Area with CN (km2)'], 2)} km² "
        f"from the land cover ({os.path.basename(s.landcover_path)}), the {method} "
        f"hydrologic soil groups and the {result.lookup.name} lookup. Antecedent "
        f"moisture: {AMC_OPTIONS[s.amc]}. Zone values are runoff-equivalent curve "
        f"numbers for a design rainfall of {p:g} mm."
    )
    d.caption("Table", f"Summary of zone curve numbers (P = {p:g} mm)")
    d.table(
        ["Zone", "Runoff-equivalent CN (P05 to P95)", "Mean CN", "Confidence",
         "Main uncertainty", "Dominant cover and group"],
        [
            [
                r["Zone"],
                f"{_f(r[eq_key])} ({_f(r['Runoff-equivalent CN P05'])} to "
                f"{_f(r['Runoff-equivalent CN P95'])})",
                _f(r["Mean CN design"]),
                r["Confidence"],
                r["Main uncertainty source"],
                f"{r['Dominant land cover']} "
                f"({_f(r['Dominant land cover share (%)'], 0)} %), group "
                f"{r['Dominant soil group']} "
                f"({_f(r['Dominant soil group share (%)'], 0)} %)",
            ]
            for r in rows
        ],
        widths_cm=[2.6, 3.3, 1.6, 2.0, 2.6, 4.4],
    )  # fmt: skip

    # Tool and method --------------------------------------------------------
    d.heading("Tool and method", 1)
    d.para(
        "The SCS curve-number method turns rainfall into stormflow depth with a "
        "single index of the catchment's runoff potential, the curve number (CN), "
        "set by the hydrologic soil group, the land cover and its treatment or "
        "condition. The tool works cell by cell on the land-cover grid:"
    )
    if result.lookup.table == "scs_sa":
        table_text = (
            "Curve-number table: Table 5.1 of the Visual SCS-SA User Manual "
            "(Schulze, Schmidt and Smithers 2004), initial curve numbers (CN-II) "
            "for the groups A, A/B, B, B/C, C, C/D and D. The values derive from US "
            "experience; southern African covers may need interpolation between "
            "rows, and values below about 50 call for caution."
        )
        if s.lookup_key.startswith("sanlc"):
            table_text += (
                " For the South African National Land Cover (SANLC 2018 / 2020 / "
                "2022, 73 classes) the lookup is a composite: urban, plantation and "
                "natural vegetation classes take the preliminary calibrated curve "
                "numbers of WRC Report 2747/1/20 (Loots and Smithers 2020, Table "
                "9.7), derived for the DEA 2013/14 land-use types in two Tshwane "
                "catchments and crosswalked to the 2018 / 2020 classes; other "
                "classes use the nearest Table 5.1 row. Four misprints in Table 9.7 "
                "are corrected (woods B/C 72, township C, village D). Open water is "
                "100 and bare rock 98; herbaceous wetlands are set to group D."
            )
    elif result.lookup.table == "tr55":
        table_text = (
            "Curve-number table: TR-55 (USDA-SCS 1986) Tables 2-2a to 2-2d, the "
            "same values as NEH 630 Chapter 9, for the groups A to D; intermediate "
            "groups take the mean of the neighbouring columns."
        )
    else:
        table_text = f"Curve-number table: the user's lookup ({result.lookup.source})."
    amc_text = {
        "none": "No adjustment: the curve numbers are CN-II (average condition).",
        "arc1": "ARC I (dry), the fit of Hawkins et al. (1985) to NEH 630 Chapter "
        "10 Table 10-1.",
        "arc3": "ARC III (wet), the fit of Hawkins et al. (1985) to NEH 630 Chapter "
        "10 Table 10-1.",
        "sobhani": "Near-saturated (Sobhani 1976; SCS-SA Eq. 4), meant for "
        "probable maximum flood estimates.",
        "median": "SCS-SA Median Condition Method (Eq. 3) with a change in storage "
        f"dS = {s.delta_s_mm:g} mm.",
    }[s.amc]
    for step in (
        table_text,
        "Lookup: each land-cover class has a design row and a plausible range from "
        "a low row (less runoff, e.g. veld in good condition) to a high row (more "
        "runoff, e.g. veld in poor condition). The full lookup with its values is "
        "written to the lookup CSV, which can be edited and used as a custom lookup.",
        "Soil groups: the design CN uses the recommended group of each soil cell, "
        "resampled to the land-cover grid by nearest neighbour."
        + (
            " NEH 630 dual groups (A/D, B/D, C/D) are taken as "
            + (
                "undrained (group D)."
                if s.dual == "undrained"
                else "drained (first letter)."
            )
            if result.hsg.method in ("NEH630", "")
            else ""
        ),
        "Uncertainty: the soil range is the CN at the group of the Ksat P5 and P95; "
        "the condition range is the lookup's low and high rows at the recommended "
        "group. Both are read as 90 % ranges of independent errors and combined on "
        "each side by root-sum-square.",
        "Antecedent moisture: " + amc_text + " The same adjustment is applied to "
        "the design, P05 and P95 values.",
        "Zone values: runoff is not linear in CN, so the mean CN of a mixed "
        "catchment gives less runoff than the sum of its parts (Grove, Harbor and "
        f"Engel 1998). The stormflow of every cell is computed for P = {p:g} mm and "
        "the runoff-equivalent CN is the single value that gives the zone's mean "
        "stormflow, for the design, P05 and P95 curve numbers.",
        "Confidence: " + plain_confidence_text(),
    ):
        d.numbered(step)

    # Inputs and settings ------------------------------------------------------
    d.heading("Inputs and settings", 1)
    d.caption("Table", "Inputs and settings")
    d.table(
        ["Item", "Value"],
        [
            ["Land cover", s.landcover_path],
            ["Soil groups", f"{s.hsg_path} ({method})"],
            ["Soil range", result.hsg.by_ksat_path or "none (land-cover condition range only)"],
            ["Lookup", f"{result.lookup.name}; {result.lookup.source}"],
            ["Curve-number table", table_name],
            ["Output grid", f"{result.grid.width} x {result.grid.height} cells of "
             f"{result.grid.res:g} (land-cover grid)"],
            ["Antecedent moisture", AMC_OPTIONS[s.amc]],
            ["Initial abstraction ratio Ia / S (-)", f"{result.ia_ratio:g}"],
            ["Design rainfall depth (mm)", f"{p:g}"],
            ["Zones", ", ".join(result.zone_names) or "none (whole raster)"],
            ["Run time", result.run_time_utc],
        ],
        widths_cm=[5.5, 11.0],
    )  # fmt: skip
    picture(lambda: figures.landcover_map(result, _class_colours(result)), "Land cover")

    # Parameters and calculations -----------------------------------------------
    d.heading("Parameters and calculations", 1)
    d.caption("Table", "Parameters and equations")
    d.table(
        ["Parameter", "Value / equation", "Definition (source)"],
        [
            ["Potential retention S (mm)", "25400 / CN - 254",
             "SCS curve-number method (USDA-NRCS 2004, NEH 630 Ch. 10)"],
            ["Initial abstraction Ia (mm)", f"λ S, λ = {result.ia_ratio:g}",
             "0.1 in SCS-SA (Schulze et al. 2004), 0.2 in TR-55"],
            ["Stormflow Q (mm)", "(P - Ia)² / (P - Ia + S) for P > Ia",
             "NEH 630 Ch. 10"],
            ["Design rainfall P (mm)", f"{p:g}", "Depth for the runoff-equivalent CN"],
            ["CN P05, P95 (-)", "CN ∓ sqrt(dS² + dC²)",
             "dS, dC: distances to the soil and condition limits (root-sum-square)"],
            ["Soil share of uncertainty (%)", "dS² / (dS² + dC²)",
             "Summed over both sides"],
            ["CN-I (dry) (-)", "CN / (2.281 - 0.01281 CN)", "Hawkins et al. (1985)"],
            ["CN-III (wet) (-)", "CN / (0.427 + 0.00573 CN)", "Hawkins et al. (1985)"],
            ["CN wet, Sobhani (-)", "CN / (0.4036 + 0.0059 CN)",
             "Sobhani (1976); SCS-SA Eq. 4"],
            ["CN Median Condition (-)", "1100 / (1100 / CN - dS / 25.4)",
             "SCS-SA Eq. 3"],
            ["Runoff-equivalent CN (-)", "CN with Q(CN) = mean Q of the zone",
             "Grove, Harbor and Engel (1998)"],
        ],
        widths_cm=[4.6, 4.4, 7.5],
    )  # fmt: skip
    d.caption(
        "Table", "Confidence classes (range of the runoff-equivalent CN, P95 - P05)"
    )
    d.table(
        ["Class", "Range"],
        [[CONF_LABELS[k], CONF_TEXT[k]] for k in CONF_LABELS],
        widths_cm=[3.0, 13.5],
    )
    d.caption("Table", "Lookup rows for the classes present")
    d.table(
        ["Class", "Design", "Low (less runoff)", "High (more runoff)", "Group"],
        _class_lookup_rows(result),
        widths_cm=[2.8, 4.4, 3.9, 3.9, 1.5],
    )

    # Results --------------------------------------------------------------------
    d.heading("Results", 1)
    picture(
        lambda: figures.cn_maps(result), "Design curve number and its range (P95 - P05)"
    )
    d.caption("Table", "Curve numbers per land-cover class (whole raster)")
    d.table(
        ["Class", "Area (km²)", "Share (%)", "Mean CN", "P05", "P95",
         "Soil share of uncertainty (%)"],
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
        widths_cm=[4.9, 1.9, 1.7, 1.7, 1.5, 1.5, 3.3],
    )  # fmt: skip
    picture(lambda: figures.histogram(result), "Area by design curve number")
    d.caption("Table", f"Zone curve numbers (runoff-equivalent for P = {p:g} mm)")
    d.table(
        ["Zone", "Area (km²)", "Mean CN", "Runoff-equivalent CN", "P05", "P95",
         "Q design (mm)", "Confidence"],
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
        widths_cm=[2.9, 1.8, 1.7, 2.6, 1.5, 1.5, 2.2, 2.3],
    )  # fmt: skip
    picture(lambda: figures.zone_ranges(result), "Zone curve numbers and their ranges")
    picture(
        lambda: figures.uncertainty_split(result),
        "What drives the uncertainty per zone",
    )
    d.caption("Table", "Candidate design values")
    d.table(
        ["Zone", "Design CN", "Sensitivity range", "Confidence", "Main uncertainty"],
        [
            [
                r["Zone"],
                _f(r[eq_key]),
                f"{_f(r['Runoff-equivalent CN P05'])} to "
                f"{_f(r['Runoff-equivalent CN P95'])}",
                r["Confidence"],
                r["Main uncertainty source"],
            ]
            for r in rows
        ],
        widths_cm=[3.6, 2.4, 3.6, 2.8, 4.1],
    )
    d.para(
        f"Design CN = runoff-equivalent CN for P = {p:g} mm; the sensitivity range "
        "is its P05 to P95. They are candidates for the designer's judgement, not "
        "prescribed values."
    )

    # Outputs ----------------------------------------------------------------------
    path = path or os.path.join(s.out_dir, REPORT_FILE)
    outputs = list(ctx.get("outputs") or [])
    if not outputs:
        from .core import OUT_LOOKUP, OUT_METADATA, OUT_RASTER, OUT_TABLES, OUT_ZONES

        labels = ctx.get("labels") or {}
        for key in (OUT_RASTER, OUT_ZONES, OUT_LOOKUP, OUT_TABLES):
            for f in result.written.get(key, []):
                outputs.append((labels.get(key, os.path.basename(f)), f))
        meta = s.path(OUT_METADATA)  # written after the report
        if meta:
            outputs.append((labels.get(OUT_METADATA, "Run metadata"), meta))
        outputs.append(("This report", path))
    d.heading("Outputs", 1)
    d.caption("Table", "Files written")
    d.table(
        ["Output", "File"],
        [[a, os.path.basename(b) if b else "-"] for a, b in outputs],
        widths_cm=[6.0, 10.5],
    )
    folders = sorted({os.path.dirname(b) for _, b in outputs if b})
    if folders:
        d.para("Folder(s): " + "; ".join(folders))

    # Notes and limitations ----------------------------------------------------------
    d.heading("Notes and limitations", 1)
    d.label("Recommendations")
    for text in _recommendations(result):
        d.bullet(text)
    d.label("Limitations")
    for text in (
        _accuracy_text(result),
        "The curve-number tables come from US experience (Table 5.1 adapts them for "
        "southern Africa); values below about 50 are uncertain.",
        "The soil and condition ranges are combined as independent 90 % ranges; the "
        "soil range is the change of group across the Ksat P5 to P95, not a full "
        "probability distribution. Zone means of the cell P05 and P95 assume fully "
        "related errors and are therefore wide.",
        "Soil groups are resampled from a coarser grid by nearest neighbour; "
        "boundaries between groups are blocky at the land-cover resolution.",
        "The land cover is one year's map; use the cover expected over the design "
        "life (development, plantation cycles, cropping).",
    ):
        d.bullet(text)
    d.label("Risks")
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
    msgs = list(dict.fromkeys(result.warnings + notes))
    if msgs:
        d.label("Processing messages")
        for n in msgs:
            d.bullet(n)

    # References -------------------------------------------------------------------------
    d.heading("References", 1)
    refs = list(REFERENCES)
    if s.lookup_key.startswith("worldcover"):
        refs.append(WORLDCOVER_REF)
    if s.lookup_key.startswith("sanlc"):
        refs += [WRC_REF, SANLC_REF]
    for ref in sorted(refs):
        d.para(ref)
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
