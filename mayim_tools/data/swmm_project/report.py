"""
Word report for "Prepare SWMM project", in the Mayim report standard: title
page, summary, model schematic, inputs, methods, subcatchment, node and
conduit parameters, design storms, computed peak flows (when the storms were
computed), files, notes and references. python-docx and matplotlib are
imported lazily. Numbers have no thousands separators.
"""

from __future__ import annotations

import math
from datetime import datetime

from mayim_tools._common.docx_report import Doc

from ..hec_hms_project.core import ENSEMBLE_METHOD_NAMES
from ..hec_hms_project.report import _schematic, add_arf_sections, arf_references
from ..hec_hms_project.storms import ari_label, duration_label
from .core import TOOL_NAME, TOOL_VERSION, XS_TRANSECT, SwmmModel
from .export import critical_rows, outlets

REFERENCES = [
    "Rossman, L.A. (2015). Storm Water Management Model User's Manual, Version 5.1. "
    "EPA/600/R-14/413b, US Environmental Protection Agency, Cincinnati, Ohio.",
    "Rossman, L.A. and Huber, W.C. (2016). Storm Water Management Model Reference "
    "Manual, Volume I - Hydrology (Revised). EPA/600/R-15/162A, US Environmental "
    "Protection Agency, Cincinnati, Ohio.",
    "Rossman, L.A. (2017). Storm Water Management Model Reference Manual, Volume II "
    "- Hydraulics. EPA/600/R-17/111, US Environmental Protection Agency, Cincinnati, "
    "Ohio.",
    "USDA-NRCS (2004). National Engineering Handbook, Part 630 Hydrology, Chapter "
    "10: Estimation of Direct Runoff from Storm Rainfall. Washington, D.C.",
    "Guo, J.C.Y. and Urbonas, B. (2009). Conversion of natural watershed to "
    "kinematic wave cascading plane. Journal of Hydrologic Engineering 14(8): 839-846.",
    "Huff, F.A. (1967). Time distribution of rainfall in heavy storms. Water "
    "Resources Research 3(4): 1007-1019.",
    "Chow, V.T., Maidment, D.R. and Mays, L.W. (1988). Applied Hydrology. "
    "McGraw-Hill, New York (alternating block method).",
]


def _f(v, nd=2) -> str:
    if v is None:
        return "-"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(x):
        return "-"
    return f"{x:.{nd}f}"


def write_report(
    path: str, m: SwmmModel, context: dict, geometry=None, results=None
) -> list[str]:
    notes: list[str] = []
    s = m.settings
    r = m.hms
    nodes = m.node_map()
    d = Doc(house_style=True)
    d.title_page(
        "SWMM model set-up",
        "This report describes the EPA SWMM model prepared from the GIS outputs "
        "(sub-catchments, stream reaches and nodes), its subcatchment, node and "
        "conduit parameters, and the design storms.",
        [
            ("Project number", context.get("project_number") or "-"),
            ("Project / site", context.get("project_name") or "-"),
            ("SWMM input file", m.files.get("SWMM input file", "-")),
            ("Prepared by", "Mayim Consulting"),
            ("Tool and version", f"{TOOL_NAME}, version {TOOL_VERSION}"),
            ("Software", context.get("software") or "-"),
            ("Report date", datetime.now().strftime("%Y-%m-%d")),
            ("Document status", "Draft"),
        ],
        subtitle=context.get("project_name") or None,
    )

    area = sum(x.area_ha for x in m.subcatchments)
    nj = sum(1 for n in m.nodes if n.kind == "junction")
    d.heading("Summary", 1)
    d.para(
        f"An EPA SWMM 5 model was prepared for {len(m.subcatchments)} subcatchment(s) "
        f"with a total area of {area:.2f} ha, {nj} junction(s), "
        f"{len(m.conduits)} open-channel conduit(s) and {len(outlets(m))} outfall(s). "
        "Runoff is computed with the SWMM non-linear reservoir and Curve Number "
        f"infiltration, and the channels are routed with "
        f"{'dynamic' if s.routing == 'DYNWAVE' else 'kinematic'} wave routing. "
        f"{len(m.series)} design storm(s) are in the file as cumulative rainfall time "
        f"series on a {m.step_min:g} min step; the rain gage uses "
        f"'{m.series[0].name}' - switch the gage's time series to run another storm."
    )
    if results:
        crit = critical_rows(m, results)
        if crit:
            d.caption(
                "Table", "Largest computed peak per storm method and return period"
            )
            d.table(
                ["Method", "Return period", "Outfall", "Peak (m3/s)", "Critical storm"],
                [[x[0], x[1], x[3], _f(x[4], 3), x[5]] for x in crit],
                widths_cm=[3.4, 2.6, 3.0, 2.5, 5.0],
            )

    d.heading("Model schematic", 1)
    d.para(
        "Subcatchments drain to the junction at the stream node below them; each "
        "conduit carries the flow of a stream reach from the junction at its top to "
        "the node at its lower end; the node with no reach leaving it is an outfall."
    )
    if geometry is not None:
        try:
            from mayim_tools._common.report_figures import ChartStyle

            style = ChartStyle()
            with style.context():
                fig = _schematic(r, geometry, style)
                d.picture(
                    style.png(fig), "Subcatchments, conduits, junctions and outfalls"
                )
        except Exception as e:  # noqa: BLE001
            notes.append(f"Schematic not drawn: {e}")

    d.heading("Inputs", 1)
    d.caption("Table", "Input data")
    d.table(
        ["Input", "Source"],
        [[k, str(v)] for k, v in context.get("inputs", [])],
        widths_cm=[5.0, 11.5],
    )

    d.heading("Methods", 1)
    d.heading("Runoff: non-linear reservoir", 2)
    d.para(
        "Each subcatchment is a non-linear reservoir: runoff Q = W (1/n) (d - ds)^(5/3) "
        "S^(1/2) (SI units), with the characteristic width W, "
        "slope S, Manning n and depression storage ds. The width sets how fast a "
        "subcatchment drains and is the most sensitive parameter. "
        f"Widths come from '{s.width_source}' ('Catchments: 1. Delineation': w_flow_m = A / mean "
        "overland flow length; w_chan_m = 2 x waterway length; w_gu_m = Guo and "
        "Urbonas 2009), else sqrt(area)."
    )
    d.heading("Infiltration: Curve Number", 2)
    d.para(
        "SWMM's Curve Number infiltration uses the curve number of each subcatchment "
        "(from the SCS curve number raster) as an infiltration capacity based on the "
        "SCS runoff relation, applied to the event's cumulative rainfall, and "
        "recovers that capacity between storms "
        f"over the drying time ({s.dry_days:g} days). It is close to, but not the same "
        "as, the SCS event equation with an explicit initial abstraction."
    )
    d.heading("Channels and nodes", 2)
    xs = (
        "irregular sections cut across the middle of each reach from the DEM "
        f"({2 * s.transect_half_width_m:g} m wide, {s.transect_points} points; HEC-2 "
        "transects)"
        if s.xs_method == XS_TRANSECT
        else f"a trapezoid {s.bottom_width_m:g} m wide at the bottom, {s.channel_depth_m:g} "
        f"m deep, side slopes {s.side_slope:g}H:1V"
    )
    d.para(
        f"The conduits follow the stream reaches with Manning n = {s.channel_n:g} and "
        f"{xs}. Node inverts are the lowest DEM cell within {s.node_search_m:g} m of the "
        f"node, lowered where needed so every conduit falls at least {s.min_slope:g} "
        "m/m downstream; outfalls use normal depth."
    )
    d.heading("Design storms", 2)
    d.para(
        "Every design storm is a cumulative rainfall time series (mm) starting at 0, "
        "on a common step: frequency storms as the nested curve of their DDF depths "
        "with the peak at the chosen position, Alternating Block, Huff and ensemble "
        "storms as their hyetographs. The rain gage is CUMULATIVE, as Storm and "
        "Sanitary Analysis writes it."
    )
    add_arf_sections(
        d,
        [ser.storm for ser in m.series],
        m.hms.arf,
        [(ser.run_name, ser.storm) for ser in m.series],
        level=2,
        hms=False,
    )

    d.heading("Subcatchment parameters", 1)
    d.caption("Table", "Subcatchment parameters")
    d.table(
        [
            "Subcatchment",
            "Outlet",
            "Area (ha)",
            "Width (m)",
            "Slope (%)",
            "CN",
            "Imperv. (%)",
        ],
        [
            [
                x.name,
                x.outlet,
                _f(x.area_ha, 3),
                _f(x.width_m, 1),
                _f(x.slope_pc, 2),
                _f(x.cn, 1),
                _f(x.imperv_pct, 1),
            ]
            for x in m.subcatchments
        ],
    )
    d.para(
        f"All subcatchments: Manning n {s.n_imperv:g} (impervious) and {s.n_perv:g} "
        f"(pervious), depression storage {s.ds_imperv_mm:g} mm and {s.ds_perv_mm:g} mm, "
        f"{s.pct_zero:g} % of the impervious area without depression storage."
    )
    d.heading("Junctions, outfalls and conduits", 1)
    d.caption("Table", "Nodes")
    d.table(
        ["Node", "Type", "Invert (m)", "DEM (m)", "Max. depth (m)"],
        [
            [
                n.name,
                n.kind,
                _f(n.invert, 2),
                _f(n.dem_invert, 2),
                _f(n.max_depth, 2) + (" *" if "lowered" in n.flags else ""),
            ]
            for n in m.nodes
        ],
    )
    if any("lowered" in n.flags for n in m.nodes):
        d.para(
            "* Invert lowered below the DEM to keep the conduits falling downstream."
        )
    d.caption("Table", "Conduits")
    d.table(
        ["Conduit", "From", "To", "Length (m)", "Slope (m/m)", "Section"],
        [
            [
                c.name,
                c.from_node,
                c.to_node,
                _f(c.length_m, 1),
                _f(c.slope(nodes), 4),
                "DEM transect" if c.transect else "Trapezoid",
            ]
            for c in m.conduits
        ],
    )

    d.heading("Design storms", 1)
    d.caption("Table", "Design storms in the SWMM file")
    d.table(
        ["Time series", "Method", "Return period", "Duration", "Depth (mm)"],
        [
            [
                ser.name,
                ENSEMBLE_METHOD_NAMES.get(ser.storm.source, ser.storm.source),
                ari_label(ser.storm.ari),
                duration_label(ser.storm.duration_min),
                _f(ser.cumulative_mm[-1], 1),
            ]
            for ser in m.series
        ],
    )

    if results:
        d.heading("Computed peak flows", 1)
        d.para(
            "Each storm was computed in the EPA SWMM 5 engine. Peak total inflow "
            "(m3/s) at the outfalls and the continuity errors of each run:"
        )
        rows = []
        for res in results:
            for out in outlets(m):
                if out in res.peaks:
                    rows.append(
                        [
                            res.run,
                            out,
                            _f(res.peaks[out][0], 3),
                            res.peaks[out][1],
                            _f(res.runoff_error_pct, 2),
                            _f(res.routing_error_pct, 2),
                        ]
                    )
        if len(rows) <= 80:
            d.caption("Table", "Peak outfall inflow per storm")
            d.table(
                [
                    "Storm",
                    "Outfall",
                    "Peak (m3/s)",
                    "Time (d hh:mm)",
                    "Runoff cont. (%)",
                    "Routing cont. (%)",
                ],
                rows,
            )
        else:
            d.para(f"{len(rows)} results - see the CSV output.")

    d.heading("Files", 1)
    d.para(
        "Open the .inp file in EPA SWMM 5 or PCSWMM, or import it into Storm and "
        "Sanitary Analysis (File > Import > EPA SWMM input file). The rain gage "
        "RG1 uses the first design storm; select another time series for the gage "
        "to run another storm."
    )
    d.caption("Table", "Outputs")
    d.table(
        ["Output", "Path"],
        [[k, str(v)] for k, v in context.get("outputs", [])],
        widths_cm=[5.0, 11.5],
    )

    d.heading("Notes and limitations", 1)
    for txt in [
        "The parameters are first estimates from GIS data; calibrate or verify them "
        "against observed floods, the HEC-HMS model or regional methods.",
        "Subcatchment width is the most sensitive SWMM runoff parameter; check it "
        "against the time of concentration.",
        "Channel sections are approximate (trapezoid or one DEM transect per reach); "
        "refine them where water levels matter.",
        "Check the continuity errors of each run (below 5 % is normally acceptable).",
    ] + list(context.get("warnings") or [])[:20]:
        d.bullet(txt)
    d.heading("References", 1)
    for ref in REFERENCES + arf_references(m.hms.arf):
        d.para(ref)
    d.doc.save(path)
    return notes
