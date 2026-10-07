"""Word report for Hydrologic soil groups (python-docx and matplotlib,
imported lazily). Numbers without thousands separators."""

from __future__ import annotations

import math
import os

from mayim_tools._common.docx_report import Doc

from . import neh630, scs_sa
from .core import (
    CONF_LABELS,
    CONF_TEXT,
    METHODS,
    TEXTURE_RULES,
    TOOL_NAME,
    TOOL_VERSION,
    labels_for,
)

REPORT_TITLE = "Hydrologic Soil Groups"
REPORT_FILE = "hydrologic_soil_groups_report.docx"
METHOD_TITLE = {"NEH630": "NEH 630", "SCSSA": "SCS-SA"}

REFERENCES = [
    "Hengl, T., Miller, M.A.E., Krizan, J., et al. (2021). African soil properties "
    "and nutrients mapped at 30 m spatial resolution using two-scale ensemble "
    "machine learning. Scientific Reports 11: 6130.",
    "Miller, M.A.E., Shepherd, K.D., Kisitu, B. and Collinson, J. (2021). iSDAsoil: "
    "the first continent-scale soil property map at 30 m resolution provides a soil "
    "information revolution for Africa. PLOS Biology 19(11): e3001441.",
    "Bouwer, H. (1986). Intake rate: cylinder infiltrometer. In: Klute, A. (ed.), "
    "Methods of Soil Analysis, Part 1, 2nd edition. Agronomy Monograph 9, ASA-SSSA, "
    "Madison, 825-844.",
    "MacVicar, C.N. et al. (1977). Soil Classification: a Binomial System for South "
    "Africa. Department of Agricultural Technical Services, Pretoria.",
    "Ross, C.W., Prihodko, L., Anchang, J., Kumar, S., Ji, W. and Hanan, N.P. (2018). "
    "HYSOGs250m, global gridded hydrologic soil groups for curve-number-based runoff "
    "modeling. Scientific Data 5: 180091.",
    "Schulze, R.E. and Schütte, S. (2023). Mapping SCS hydrological soil groups over "
    "South Africa at terrain unit spatial resolution. Journal of the South African "
    "Institution of Civil Engineering 65(4): 2-9.",
    "Schulze, R.E., Schmidt, E.J. and Smithers, J.C. (2004). Visual SCS-SA User Manual, "
    "Version 1.0: PC-based SCS design flood estimates for small catchments in southern "
    "Africa. ACRUcons Report 52, School of Bioresources Engineering and Environmental "
    "Hydrology, University of KwaZulu-Natal, Pietermaritzburg.",
    "Soil Classification Working Group (1991). Soil Classification: a Taxonomic System "
    "for South Africa. Memoirs on the Agricultural Natural Resources of South Africa 15, "
    "Department of Agricultural Development, Pretoria.",
    "USDA-NRCS (2009). National Engineering Handbook, Part 630 Hydrology, Chapter 7: "
    "Hydrologic Soil Groups. 210-VI-NEH, United States Department of Agriculture, "
    "Natural Resources Conservation Service, Washington, D.C.",
    "USDA-SCS (1986). Urban Hydrology for Small Watersheds. Technical Release 55, "
    "2nd edition. United States Department of Agriculture, Soil Conservation Service, "
    "Washington, D.C.",
]


def _f(v, nd=1) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}"


def _plain_confidence_text() -> str:
    return (
        "Which group a soil belongs to depends mainly on how fast water can "
        "move through its least permeable layer (the saturated hydraulic "
        "conductivity, Ksat). That rate is not known exactly. The soil maps "
        "behind it are estimates, so Regional soil parameterisation gives a "
        "likely middle value and a range, from a low value it is unlikely to "
        "fall below to a high value it is unlikely to exceed. This tool works "
        "out how likely each group is, given that range. The recommended group "
        "is the one the middle value falls in, and its confidence is the "
        "chance that the soil really belongs to that group: High when it is "
        "80 % or more, Medium between 60 and 80 %, and Low below 60 %. The "
        "tool also shows the group at the low end of the range (a less "
        "permeable soil with more runoff) and at the high end (a more "
        "permeable soil with less runoff). Where these differ, the group is "
        "uncertain; for flood estimation the low-end group is the cautious "
        "choice. A Low rating does not mean the group is wrong. It means the "
        "available information cannot pin it down, and a few infiltration "
        "tests or a soil survey would settle it."
    )


def _zone_table(d, result, method):
    rows = [r for r in result.zone_rows if r["Method"] == method]
    labels = labels_for(method)
    order = [
        c for c in scs_sa.STEP_CODES + (neh630.AD, neh630.BD, neh630.CD) if c in labels
    ]
    present = [
        labels[c]
        for c in order
        if any((r.get(f"Share {labels[c]} (%)") or 0) > 0 for r in rows)
    ]
    header = (
        ["Zone", "Dominant group", "Mean confidence (%)"]
        + [f"{p} (%)" for p in present]
        + ["At Ksat P5", "At Ksat P95"]
    )
    body = []
    for r in rows:
        body.append(
            [
                r["Zone"],
                f"{r['Dominant group']} ({_f(r['Dominant group share (%)'], 0)} %)",
                _f(r["Mean confidence (%)"], 0),
            ]
            + [_f(r.get(f"Share {p} (%)"), 0) for p in present]
            + [
                r["Most common group at Ksat P05 (more runoff)"],
                r["Most common group at Ksat P95 (less runoff)"],
            ]
        )
    d.caption(
        "Table", f"{METHODS[method]}: recommended groups per zone (share of area)"
    )
    d.table(header, body)


def write_report(result) -> str:
    from . import figures

    d = Doc()
    s = result.settings
    meta = result.regional_meta
    notes = []
    area = float(result.mask.sum()) * result.cell_area_km2
    methods = list(result.methods)
    both = len(methods) == 2

    d.heading(REPORT_TITLE, 1)

    # 1. Summary ---------------------------------------------------------
    d.heading("1. Summary", 2)
    d.para(
        f"This report assigns hydrologic soil groups to {len(result.zone_names)} "
        f"zone{'s' if len(result.zone_names) != 1 else ''} covering {_f(area, 1)} "
        "km2, using the saturated hydraulic conductivity (Ksat) estimated by "
        "Regional soil parameterisation"
        + (f" ({meta.get('Tool')})" if meta.get("Tool") else "")
        + ", the depth to a water impermeable layer and the depth to the water "
        f"table. Method{'s' if both else ''}: "
        + "; ".join(METHODS[m] for m in methods)
        + "."
    )
    for r in result.zone_rows:
        d.bullet(
            f"{r['Zone']}, {METHOD_TITLE[r['Method']]}: dominant group "
            f"{r['Dominant group']} on {_f(r['Dominant group share (%)'], 0)} % of the "
            f"area, mean confidence {_f(r['Mean confidence (%)'], 0)} %; at the low "
            f"end of the Ksat range {r['Most common group at Ksat P05 (more runoff)']}, "
            f"at the high end {r['Most common group at Ksat P95 (less runoff)']}."
        )
    if both:
        d.para(
            "The two methods use very different Ksat limits (Section 4.4): NEH 630 "
            "(2009) places group A above 144 mm/h, while the SCS-SA groups follow "
            "the older SCS permeability rates with group A above 7.6 mm/h. The "
            "same soil can therefore fall one or two groups apart. Use the method "
            "that matches the curve-number tables of the governing guideline: "
            "SCS-SA with the Schulze, Schmidt & Smithers (2004) tables in South "
            "Africa, NEH 630 with the NRCS / TR-55 tables."
        )

    # 2. Purpose ---------------------------------------------------------
    d.heading("2. Purpose and scope", 2)
    d.para(
        "Curve-number runoff methods need a hydrologic soil group for every "
        "sub-catchment. Groups are normally read from soil surveys; where none "
        "exist, this tool derives them from mapped soil properties with their "
        "uncertainty. It assigns groups only; curve numbers are a separate step "
        "that combines the groups with land cover."
    )
    d.heading("How confidence is judged", 3)
    d.para(_plain_confidence_text())

    # 3. Data ------------------------------------------------------------
    d.heading("3. Data", 2)
    rows = [
        ["Ksat (P5, P50, P95; 0-30, 30-60, 60-100 cm)", s.regional_folder],
    ]
    for key in ("Tool", "Products", "Methods", "Monte Carlo draws per product"):
        if meta.get(key):
            rows.append([f"Regional run: {key.lower()}", meta[key]])
    for n in result.input_notes:
        item, _, value = n.partition(": ")
        rows.append([item, value] if value else ["Note", n])
    d.caption("Table", "Inputs")
    d.table(["Item", "Value"], rows, widths_cm=[5.5, 10.5])
    if s.bedrock_source in ("BDRICM", "BDTICM"):
        d.para(
            "Depth to bedrock from SoilGrids 2017 (Shangguan et al., 2017) has no "
            "uncertainty estimate and is used as mapped; it describes bedrock "
            "only, not hardpans, duripans or other restrictive layers."
        )
    if s.bedrock_source == "ISDA":
        d.para(
            "Depth to bedrock from iSDAsoil (Hengl et al., 2021; Miller et al., "
            "2021): 30 m mean and standard deviation, Africa only; 200 cm means "
            "200 cm or deeper, and exposed bedrock is masked in the product. "
            "The groups follow the mapped mean depth; the confidence also "
            "includes the chance that bedrock lies in another depth class "
            "(Section 4.3)."
        )
    if s.water_source == "none":
        d.para(
            "No reliable global map of the water table exists and none was given: "
            "the water table is assumed deeper than 100 cm, so no dual groups "
            "(NEH 630) or water-table adjustments (SCS-SA) appear."
        )

    # 4. Methods ---------------------------------------------------------
    d.heading("4. Methods", 2)
    d.heading("4.1 USDA-NRCS NEH 630 Chapter 7 (2009)", 3)
    d.para(
        "Table 7-1 of NEH 630 assigns the group from the depth to a water "
        "impermeable layer, the depth to the high water table and the Ksat of "
        "the least transmissive layer in a given depth range. A shallow water "
        "table gives a dual group (A/D, B/D, C/D): the first letter applies if "
        "the soil is drained, D if it is not. Ksat limits converted from µm/s "
        "(1 µm/s = 3.6 mm/h):"
    )
    trows = []
    for desc, rng, thr, dual in neh630.CASES.values():
        if rng is None:
            trows.append([desc, "-", "-", "D"])
            continue
        groups = "A/D, B/D, C/D, D" if dual else "A, B, C, D"
        trows.append(
            [desc, rng.replace("cm", " cm"), " / ".join(_f(t, 1) for t in thr), groups]
        )
    d.caption(
        "Table", "NEH 630 Table 7-1 as applied (Ksat limits A|B, B|C, C|D in mm/h)"
    )
    d.table(
        ["Case", "Ksat depth range", "Limits (mm/h)", "Groups"],
        trows,
        widths_cm=[7.0, 2.6, 3.4, 3.0],
    )
    d.para(
        "The Ksat layers of Regional soil parameterisation are 0-30, 30-60 and "
        "60-100 cm. The 0-50 and 0-60 cm ranges use the 0-30 and 30-60 cm "
        "layers (the 30-60 cm value stands in for 30-50 cm); 0-100 cm uses all "
        "three. A group applies when Ksat exceeds its lower limit."
    )
    d.heading("4.2 SCS-SA hydrological soil groups", 3)
    d.para(
        "Schulze, Schmidt & Smithers (2004, Table 2.1) describe the four basic "
        "groups by their permeability rate (saturated soil profile) and typical "
        "final infiltration rate under short grass, and add the intermediate "
        "groups A/B, B/C and C/D for southern Africa. The manual assigns groups "
        "per soil series (Tables 5.2 and 5.3); where the soil form and series "
        "are known, those assignments take precedence over this tool."
    )
    d.caption(
        "Table", "SCS-SA basic groups (Schulze, Schmidt & Smithers, 2004, Table 2.1)"
    )
    d.table(
        [
            "Group",
            "Stormflow potential",
            "Permeability (mm/h)",
            "Final infiltration (mm/h)",
        ],
        [
            ["A", "Low", "> 7.6", "about 25"],
            ["B", "Moderately low", "3.8-7.6", "about 13"],
            ["C", "Moderately high", "1.3-3.8", "about 6"],
            ["D", "High", "< 1.3", "about 3"],
        ],
        widths_cm=[2.0, 4.0, 4.5, 5.5],
    )
    d.para("Implementation choices in this tool:")
    d.bullet(
        "Permeability is the Ksat of the least transmissive layer within 0-100 cm "
        "(the manual: permeability is controlled by the properties of the profile)."
    )
    d.bullet(
        "The manual gives no numeric limits for the intermediate groups. An "
        f"intermediate group is assigned where the Ksat range straddles the boundary "
        f"between two adjacent groups: both at least {scs_sa.STRADDLE_MIN * 100:.0f} % "
        "likely. Its confidence is the probability of the two groups together."
    )
    adj = []
    if s.sa_adjust_shallow:
        adj.append("an impermeable layer shallower than 50 cm (shallow phase)")
    if s.sa_adjust_water:
        adj.append(
            "a water table shallower than 60 cm (standing in for a bottomland position)"
        )
    if adj:
        d.bullet(
            "Field adjustments of Section 2.2.3(d): the group moves one step down "
            "(e.g. B to B/C) for " + " and for ".join(adj) + ", up to group D. "
            "Surface sealing, topographic position and parent material cannot be "
            "mapped from these inputs; apply them in the field."
        )
    else:
        d.bullet("The field adjustments of Section 2.2.3(d) were not applied.")
    d.heading("4.3 Probabilities and the recommended group", 3)
    d.para(
        "Per cell, ln(Ksat) of each layer is described by a two-piece normal "
        "distribution that reproduces its P5, P50 and P95. The least transmissive "
        "layer takes the minimum of the layer quantiles, which assumes that the "
        "layers' uncertainties move together (they come from the same soil maps "
        "and methods). The probability of each group is the probability that Ksat "
        "falls between its limits. The recommended group is the group of the "
        "median Ksat, matching the candidate values of Regional soil "
        "parameterisation; its confidence is the probability of that group."
    )
    if result.p_shallow is not None:
        d.para(
            "With iSDAsoil bedrock depth, the depth is described by a normal "
            "distribution (mapped mean and standard deviation). The confidence is "
            "multiplied by the probability that bedrock lies in the same depth "
            "class as the mean - shallower than 50 cm, 50-100 cm or deeper than "
            "100 cm for NEH 630; shallower or deeper than 50 cm for the SCS-SA "
            "shallow-phase adjustment. Another class is counted as another group, "
            "which is slightly cautious. The probability of bedrock within 50 cm "
            "is mapped in hsg_conditions.tif."
        )
    d.caption("Table", "Confidence classes")
    d.table(
        ["Class", "Meaning"],
        [[CONF_LABELS[k], CONF_TEXT[k]] for k in CONF_LABELS],
        widths_cm=[3.0, 13.0],
    )
    d.heading("4.4 Why the methods differ", 3)
    d.para(
        "NEH 630 (2009) uses Ksat limits of 144, 36 and 3.6 mm/h for the shallow "
        "range and 36, 14.4 and 1.44 mm/h for deep soils. The SCS-SA groups follow "
        "the earlier SCS permeability rates (7.6, 3.8 and 1.3 mm/h, as in TR-55, "
        "USDA-SCS 1986). For the same soil NEH 630 therefore gives a group of "
        "higher runoff potential. Neither is wrong: each belongs with its own "
        "curve-number tables."
    )
    try:
        d.picture(
            figures.ksat_thresholds(result),
            "Least transmissive Ksat (cell medians) against the group limits of "
            "both methods",
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Ksat limit figure not drawn ({exc}).")

    # 5. Results ---------------------------------------------------------
    d.heading("5. Results", 2)
    try:
        d.picture(
            figures.group_maps(
                result,
                [
                    (result.methods[m].recommended, f"{METHOD_TITLE[m]}: recommended")
                    for m in methods
                ],
            ),
            "Recommended hydrologic soil group",
        )
        d.picture(
            figures.confidence_maps(result), "Confidence of the recommended group"
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Group maps not drawn ({exc}).")
    for m in methods:
        _zone_table(d, result, m)
    try:
        d.picture(
            figures.zone_shares(result), "Share of each recommended group per zone"
        )
        d.picture(
            figures.probability_bars(result),
            "Mean probability of the basic groups A-D per zone (for NEH 630 dual "
            "cells the drained letter)",
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Zone figures not drawn ({exc}).")
    for m in methods:
        try:
            d.picture(
                figures.by_ksat_maps(result, m),
                f"{METHODS[m]}: group at the Ksat P5 (more runoff), P50 and P95 "
                "(less runoff)",
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Ksat range maps not drawn ({exc}).")

    # 6. Texture check ---------------------------------------------------
    if result.texture is not None and "NEH630" in methods:
        d.heading("6. Texture cross-check (NEH 630)", 2)
        d.para(
            "NEH 630 describes the textures typical of each group. Where the "
            "surface layer (0-30 cm) matches one of them, the group it indicates "
            "is compared with the recommended NEH 630 group (drained letter). "
            "Many soils match none of the descriptions; the check is indicative."
        )
        d.table(
            ["Group", "Typical texture (NEH 630)"],
            [[neh630.GROUP_LABEL[c], t] for c, t in TEXTURE_RULES],
            widths_cm=[2.5, 13.5],
        )
        rows = [
            [
                r["Zone"],
                _f(r.get("Texture indicates a group (%)"), 0),
                _f(r.get("Texture agrees with recommended (%)"), 0),
            ]
            for r in result.zone_rows
            if r["Method"] == "NEH630"
        ]
        d.caption("Table", "Texture cross-check per zone")
        d.table(
            ["Zone", "Cells with a typical texture (%)", "Of these, same group (%)"],
            rows,
        )

    # 7. Use ------------------------------------------------------------
    d.heading("7. Using the groups for curve numbers", 2)
    d.bullet(
        "Use the share of each group per zone (hsg_zone_summary.csv) as area "
        "fractions for an area-weighted curve number, or the cell rasters with a "
        "land-cover map."
    )
    d.bullet(
        "Run the model with the groups at the low end of the Ksat range "
        "(hsg_*_by_ksat.tif, band 1) as a cautious case for flood estimation."
    )
    if "NEH630" in methods:
        d.bullet(
            "NEH 630 dual groups: use the second letter (D) unless the land is "
            "artificially drained."
        )
    if "SCSSA" in methods:
        d.bullet(
            "SCS-SA intermediate groups have their own columns in the SCS-SA "
            "curve-number tables (Schulze, Schmidt & Smithers, 2004, Table 5.1)."
        )
    d.bullet(
        "Check the groups in the field: the manuals recommend infiltration tests "
        "at several sites and a soil inspection, because computed runoff is very "
        "sensitive to the group."
    )

    # 8. Limitations ----------------------------------------------------
    d.heading("8. Limitations", 2)
    for text in (
        "Ksat comes from pedotransfer functions applied to global soil maps; it "
        "describes the soil matrix, not macropores, crusting or compaction, and "
        "its uncertainty is often wide (see the Regional soil parameterisation "
        "report).",
        "Bedrock depth is mapped coarsely and without uncertainty; restrictive "
        "layers other than bedrock (duripans, plinthite, dense clay B horizons) "
        "are not mapped and can lower the true group.",
        "The water table is not mapped globally; dual groups appear only where a "
        "water-table depth is given.",
        "The SCS-SA intermediate-group rule and the mapped field adjustments are "
        "this tool's interpretation of the manual, which assigns groups per soil "
        "series.",
        "Groups describe soils when thoroughly wet; antecedent conditions and "
        "land cover enter through the curve number, not the group.",
    ):
        d.bullet(text)

    # 9. References -------------------------------------------------------
    d.heading("9. References", 2)
    refs = REFERENCES + [
        "Shangguan, W., Hengl, T., Mendes de Jesus, J., Yuan, H. and Dai, Y. (2017). "
        "Mapping the global depth to bedrock for land surface modeling. Journal of "
        "Advances in Modeling Earth Systems 9: 65-88."
    ]
    for ref in sorted(refs):
        d.para(ref)

    # Annexes -------------------------------------------------------------
    d.heading("Annex A. Run settings", 2)
    g = result.grid
    d.table(
        ["Setting", "Value"],
        [
            ["Tool", f"{TOOL_NAME} (mayim_tools) v{TOOL_VERSION}"],
            ["Run time", result.run_time_utc],
            ["Methods", "; ".join(METHODS[m] for m in methods)],
            ["Grid", f"{g.width} x {g.height} cells at {g.res:g}"],
            ["Zones", ", ".join(result.zone_names)],
            ["SCS-SA adjustments", ", ".join(adj) if adj else "none"],
        ],
        widths_cm=[5.0, 11.0],
    )
    d.heading("Annex B. Output files", 2)
    files = sorted(
        {os.path.basename(p) for p in result.files} | {"hsg_metadata.csv", REPORT_FILE}
    )
    d.table(["File"], [[f] for f in files], widths_cm=[16.0])
    if notes or result.warnings:
        d.heading("Annex C. Warnings", 2)
        for n in result.warnings + notes:
            d.bullet(n)
    path = os.path.join(s.out_dir, REPORT_FILE)
    d.doc.save(path)
    return path
