"""
Word report for "Prepare HEC-HMS project", in the Mayim report standard:
title page, summary, model schematic, inputs, methods, sub-basin and
reach parameters, design storms, computed peak flows (when the runs were
computed), project files, notes and references. python-docx and matplotlib
are imported lazily. Numbers have no thousands separators.
"""

from __future__ import annotations

import math
from datetime import datetime

from mayim_tools._common.docx_report import Doc

from . import methods as mth
from .core import TOOL_NAME, TOOL_VERSION, TRANSFORM_CLARK, ProjectResult
from .export import ensemble_rows, method_labels, peak_table
from .storms import (
    SOURCE_AB,
    SOURCE_DDF,
    SOURCE_ENSEMBLE,
    SOURCE_HUFF,
    SOURCE_LABELS,
    ari_label,
    duration_label,
    nested_cumulative,
)

REFERENCES = [
    "US Army Corps of Engineers, Hydrologic Engineering Center (2025). HEC-HMS "
    "Hydrologic Modeling System, version 4.14, User's Manual and Technical "
    "Reference Manual. Davis, California.",
    "USDA-NRCS (2004). National Engineering Handbook, Part 630 Hydrology, Chapter "
    "10: Estimation of Direct Runoff from Storm Rainfall; Chapter 16: Hydrographs. "
    "United States Department of Agriculture, Washington, D.C.",
    "Schulze, R.E., Schmidt, E.J. and Smithers, J.C. (1992). SCS-SA User Manual. "
    "ACRUcons Report 40, Department of Agricultural Engineering, University of "
    "Natal, Pietermaritzburg.",
    "Clark, C.O. (1945). Storage and the unit hydrograph. Transactions of the "
    "American Society of Civil Engineers 110: 1419-1446.",
    "Cunge, J.A. (1969). On the subject of a flood propagation computation method "
    "(Muskingum method). Journal of Hydraulic Research 7(2): 205-230.",
    "Huff, F.A. (1967). Time distribution of rainfall in heavy storms. Water "
    "Resources Research 3(4): 1007-1019.",
    "Chow, V.T., Maidment, D.R. and Mays, L.W. (1988). Applied Hydrology. "
    "McGraw-Hill, New York (alternating block method).",
    "SANRAL (2013). Drainage Manual, 6th edition. South African National Roads "
    "Agency SOC Ltd, Pretoria.",
]
ARF_REFERENCES = {
    "eq310": [
        "SANRAL (2013). Drainage Manual, 6th edition, Eq 3.10 / 3.19 (areal "
        "reduction factor after Alexander, from NERC 1975 Flood Studies Report).",
    ],
    "pietersen": [
        "Pietersen, J.P.J. et al. (2024). Estimation of areal reduction factors in "
        "South Africa, Parts 1 and 2. Journal of the South African Institution of "
        "Civil Engineering 66(4).",
    ],
    "feh": [
        "NERC (1975). Flood Studies Report, Vol II. Natural Environment Research "
        "Council, London.",
        "Keers, J.F. and Wescott, P. (1977). A computer-based model for design "
        "rainfall in the United Kingdom. Meteorological Office Scientific Paper 36.",
        "Institute of Hydrology (1999). Flood Estimation Handbook, Vol 2: Rainfall "
        "frequency estimation. Wallingford.",
    ],
}


def arf_references(arf) -> list[str]:
    if arf is None:
        return []
    refs = list(ARF_REFERENCES.get(arf.method, []))
    if arf.method == "pietersen":
        refs = ARF_REFERENCES["eq310"] + refs
    return refs


LOSS_COLUMNS = {
    mth.LOSS_SCS: [("CN", "cn", 1), ("Ia (mm)", "ia_mm", 2)],
    mth.LOSS_IC: [
        ("Initial loss (mm)", "initial_loss_mm", 2),
        ("Constant rate (mm/h)", "constant_rate_mmh", 2),
    ],
    mth.LOSS_DC: [
        ("Initial deficit (mm)", "initial_deficit_mm", 1),
        ("Max deficit (mm)", "max_deficit_mm", 1),
        ("Constant rate (mm/h)", "constant_rate_mmh", 2),
    ],
    mth.LOSS_GA: [
        ("Ks (mm/h)", "ga_ks_mmh", 2),
        ("Suction (mm)", "ga_psi_mm", 0),
        ("theta s", "ga_ts", 3),
        ("theta i", "ga_ti", 3),
    ],
}


def method_references(s) -> list[str]:
    refs = []
    if s.methods.loss == mth.LOSS_GA:
        refs += [
            "Green, W.H. and Ampt, G.A. (1911). Studies on soil physics, 1. The flow "
            "of air and water through soils. Journal of Agricultural Science 4: 1-24.",
            "Rawls, W.J., Brakensiek, D.L. and Miller, N. (1983). Green-Ampt "
            "infiltration parameters from soils data. Journal of Hydraulic "
            "Engineering 109(1): 62-70.",
        ]
    if s.transform == mth.TRANSFORM_SNYDER:
        refs.append(
            "Snyder, F.F. (1938). Synthetic unit-graphs. Transactions of the "
            "American Geophysical Union 19: 447-454."
        )
    return refs


def _source_rows(subs) -> list[list[str]]:
    """[parameter, source, number of sub-basins] from params['sources']."""
    count: dict[tuple[str, str], int] = {}
    for e in subs:
        for k, v in (e.params.get("sources") or {}).items():
            count[(k, v)] = count.get((k, v), 0) + 1
    return [[k, v, str(n)] for (k, v), n in sorted(count.items())]


def _loss_methods_text(d, s, m) -> None:
    ms = s.methods
    imp = (
        "Impervious areas are taken as included in the CN unless a percentage "
        "field was given."
        if ms.loss == mth.LOSS_SCS
        else "Rain on the impervious percentage (field, if given) runs off without "
        "loss."
    )
    if ms.loss == mth.LOSS_SCS:
        d.heading("Losses: SCS curve number", 2)
        d.para(
            "Direct runoff Q = (P - Ia)^2 / (P - Ia + S) for P > Ia, with the potential "
            f"maximum retention S = 25400/CN - 254 (mm) and Ia = {s.lambda_ia:g} S. "
            f"Each sub-basin's curve number is the {m['cn']} of the CN raster cells "
            "within it (or the CN field given). The antecedent moisture condition is "
            "that of the CN raster. " + imp
        )
        return
    ks = (
        "the saturated hydraulic conductivity Ksat of the Regional Soil "
        "Parameterisation rasters (0-30 cm P50, geometric mean over the sub-basin"
        + (f", x {ms.ksat_factor:g}" if ms.ksat_factor != 1 else "")
        + ")"
    )
    if ms.loss == mth.LOSS_IC:
        d.heading("Losses: Initial and Constant", 2)
        d.para(
            "Rainfall is lost entirely until the initial loss is satisfied, then at "
            "the constant rate (HEC-HMS Technical Reference Manual). Initial loss: "
            f"an il_mm field, else Ia = {s.lambda_ia:g} S from the curve number, "
            f"else {ms.initial_loss_mm:g} mm. Constant rate: a cl_mmh field, else "
            f"{ks}, else {ms.constant_rate_mmh:g} mm/h. " + imp
        )
    elif ms.loss == mth.LOSS_DC:
        d.heading("Losses: Deficit and Constant", 2)
        d.para(
            "A single soil layer with a moisture deficit: rainfall fills the "
            "deficit, then infiltrates at the constant rate (HEC-HMS Technical "
            "Reference Manual). Maximum deficit = (theta_s - theta_wp) x "
            f"{ms.soil_depth_mm:g} mm from the soil rasters (else S from the CN, "
            "or an md_mm field); initial deficit = (theta_s - theta_i) x the same "
            f"depth with the initial content at {mth.antecedent_label(ms)} (else "
            "Ia from the CN, or an id_mm field). Constant rate: a cl_mmh field, "
            f"else {ks}, else {ms.constant_rate_mmh:g} mm/h. Recovery factor 1 "
            "(no recovery is significant within a design storm). " + imp
        )
    else:
        d.heading("Losses: Green and Ampt", 2)
        d.para(
            "Green and Ampt infiltration with the hydraulic conductivity K, the "
            "wetting front suction psi_f, the saturated content theta_s and the "
            "initial content theta_i (HEC-HMS Technical Reference Manual; Green and "
            "Ampt 1911; Rawls et al. 1983). K is "
            f"{ks}; psi_f, theta_s and theta_i are the sub-basin means of the "
            f"same rasters, with theta_i at {mth.antecedent_label(ms)}. Fields "
            "ga_ks_mmh, ga_psi_mm, ga_ts and ga_ti override them. " + imp
        )
    d.para(
        "The table 'Where the estimated sub-basin parameters came from' lists the "
        "source of every value. Soil-derived values are first estimates from "
        "global soil data; check them against site data or calibrate."
    )


def add_arf_sections(
    d, storms: list, arf, named_storms: list, level: int = 2, hms: bool = True
) -> None:
    """Methods text, ARF table and burst check for the HEC-HMS and SWMM
    reports. ``named_storms`` = [(run name, Storm)]."""
    from .export import arf_table, burst_table

    d.heading("Areal reduction", level)
    if arf is None or arf.is_unity:
        d.para(
            "No areal reduction was applied: the storm depths are the DDF depths "
            + (
                "(the DDF was taken as already areal)."
                if arf is not None and arf.method == "none"
                else "(point rainfall)."
            )
        )
    else:
        d.para(
            "The DDF depths (Design Rainfall estimation, Precipitation data to DDF, "
            "or gridded rainfall adjusted to a point DDF) are point rainfall and "
            f"were reduced to catchment rainfall with: {arf.label()} "
            f"({arf.area_note or 'given area'}). "
            + (
                "The ARF depends on the duration, so each nested depth of a "
                "frequency storm and each cumulative depth of an alternating block "
                "storm (the k largest blocks) is reduced with the ARF of its own "
                "duration: the short, intense bursts are reduced more than the "
                "storm total, and the storms follow the areal DDF curve. Huff and "
                "ensemble storms carry one DDF depth (the storm duration) and are "
                "scaled with the ARF of that duration; their temporal pattern is "
                "unchanged."
                if arf.varies_with_duration
                else "The same factor scales every depth."
            )
            + (" HEC-HMS's own depth-area reduction is not used." if hms else ""),
        )
        if len(storms) and any(st.arf_values for st in storms):
            head, rows = arf_table(storms, arf)
            if rows:
                d.caption(
                    "Table", "Areal reduction factor by duration and return period"
                )
                d.table(head, [[r[0]] + [f"{v:.3f}" for v in r[1:]] for r in rows])
    head, rows = burst_table(named_storms)
    if rows:
        tmax = max(st.ari for _, st in named_storms if st.burst)
        keep = {name for name, st in named_storms if st.burst and st.ari == tmax}
        rows_t = [r for r in rows if r[0] in keep]
        d.heading("Burst check", level + 1)
        d.para(
            "For each hyetograph storm, the largest 15 min, 30 min, 1 h ... depth "
            "inside the storm is compared with the areal DDF depth for that "
            "duration (point DDF x ARF). Near 100 %, the temporal pattern is as "
            "intense as the DDF for this catchment; well below 100 %, the pattern "
            "is smoother than the DDF, which is typical of patterns from gridded "
            "rainfall (e.g. ERA5, hourly averages over about 30 km cells) applied "
            "to a small catchment, and the peak flows of those storms may then be "
            "low. Frequency storms follow the areal DDF by construction (100 %). "
            f"The table lists the {ari_label(tmax)} year storms; every storm is in "
            "the CSV output (section 'Burst check')."
        )
        d.caption(
            "Table",
            f"Burst check, {ari_label(tmax)} year storms (% of the areal DDF depth)",
        )
        d.table(
            head,
            [
                [r[0], r[1]] + ["-" if v is None else f"{v:.0f}" for v in r[2:]]
                for r in rows_t
            ],
        )


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


def _schematic(r: ProjectResult, geometry: dict, style):
    """Plan of the sub-basins, reaches and nodes (matplotlib)."""
    from matplotlib.patches import Polygon

    fig = style.figure(12.0)
    ax = fig.add_subplot(111)
    for e in r.network.of("Subbasin"):
        for ring in geometry.get(("Subbasin", e.name), []):
            ax.add_patch(
                Polygon(
                    ring, closed=True, facecolor="#F2F2F2", edgecolor="#7F7F7F", lw=0.5
                )
            )
        ax.text(e.x, e.y, e.name, ha="center", va="center", fontsize=6, color="#000000")
    for e in r.network.of("Reach"):
        for line in geometry.get(("Reach", e.name), []):
            xs, ys = zip(*line, strict=True)
            ax.plot(xs, ys, color="#0000FF", lw=1.0)
    for kind, marker, colour in (
        ("Junction", "o", "#FFC000"),
        ("Sink", "s", "#FF0000"),
    ):
        els = r.network.of(kind)
        if els:
            ax.scatter(
                [e.x for e in els],
                [e.y for e in els],
                marker=marker,
                s=14,
                color=colour,
                edgecolor="#000000",
                linewidths=0.4,
                zorder=5,
                label=kind,
            )
    ax.plot([], [], color="#0000FF", lw=1.0, label="Reach")
    ax.set_aspect("equal", adjustable="datalim")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    style.axes(ax, grid=False)
    style.legend(ax, loc="best")
    return fig


def _storm_figure(r: ProjectResult, style):
    """Cumulative depth of the largest return period's storms."""
    import numpy as np

    fig = style.figure(8.0)
    ax = fig.add_subplot(111)
    tmax = max(run.storm.ari for run in r.runs)
    shown = 0
    # one peak position per duration first (50 % where there is one)
    order = sorted(
        r.runs,
        key=lambda run: (
            abs((run.storm.peak_pct if run.storm.peak_pct is not None else 50) - 50),
        ),
    )
    for run in order:
        st = run.storm
        if st.ari != tmax or shown >= 7:
            continue
        if st.is_hyetograph:
            t = np.arange(0, len(st.increments) + 1) * st.interval_min / 60.0
            c = np.r_[0.0, np.cumsum(st.increments)]
        else:
            t, c = nested_cumulative(st.depths, st.duration_min, st.peak_pct)
            t = np.asarray(t) / 60.0
        ax.plot(t, c, lw=0.75, label=run.name)
        shown += 1
    ax.set_xlabel("Time from storm start (h)")
    ax.set_ylabel("Cumulative rainfall (mm)")
    style.axes(ax)
    style.legend(ax, loc="lower right")
    return fig


def _peak_list(r: ProjectResult, source: str) -> str:
    peaks = sorted({run.storm.peak_pct for run in r.runs if run.storm.source == source})
    return ", ".join(f"{p:g} %" for p in peaks if p is not None) or "-"


def write_report(
    path: str,
    r: ProjectResult,
    context: dict,
    geometry: dict | None = None,
    results: list | None = None,
) -> list[str]:
    """Writes the report; returns notes (figures that could not be drawn)."""
    notes: list[str] = []
    s = r.settings
    m = method_labels(r)
    d = Doc(house_style=True)
    subs = r.network.of("Subbasin")
    reaches = r.network.of("Reach")
    sinks = r.network.of("Sink")
    sources = sorted({run.storm.source for run in r.runs})
    d.title_page(
        "HEC-HMS model set-up",
        "This report describes the HEC-HMS hydrological model prepared from the "
        "GIS outputs (sub-catchments, stream reaches and nodes), the loss, "
        "transform and routing parameters, and the design storms.",
        [
            ("Project number", context.get("project_number") or "-"),
            ("Project / site", context.get("project_name") or "-"),
            ("HEC-HMS project", s.project_name),
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
    area = sum(e.params["area_km2"] for e in subs)
    d.para(
        f"A HEC-HMS {s.hms_version} project was prepared for "
        f"{len(subs)} sub-basin(s) with a total area of {area:.3f} km2, "
        f"{len(reaches)} reach(es), {len(r.network.of('Junction'))} junction(s) and "
        f"{len(sinks)} outlet(s). Losses use the {m['loss']} method, runoff is "
        f"transformed with the {m['transform']}, and the reaches are routed with "
        f"{m['routing']}. The model computes at a {r.dt_min:g} min time step and "
        f"runs {r.tail_h:g} h past the end of each storm. {len(r.runs)} design "
        f"storm(s) were set up as simulation runs from: "
        + "; ".join(SOURCE_LABELS[x] for x in sources)
        + "."
    )
    rows = [
        ["HEC-HMS project", s.project_name],
        ["Basin model", s.basin_name],
        ["Coordinate system", r.crs_name or "-"],
        ["Sub-basins / reaches / junctions / outlets",
         f"{len(subs)} / {len(reaches)} / {len(r.network.of('Junction'))} / {len(sinks)}"],
        ["Total area (km2)", f"{area:.3f}"],
        ["Loss", m["loss"] + f" ({m['cn']})"],
        ["Transform", m["transform"]],
        ["Routing", m["routing"]],
        ["Time step / simulation after the storm", f"{r.dt_min:g} min / {r.tail_h:g} h"],
        ["Areal reduction", r.arf.label() if r.arf is not None else "None"],
        ["Simulation runs", str(len(r.runs))],
        ["Ensemble analyses", str(len(r.ensembles)) if r.ensembles else "-"],
    ]  # fmt: skip
    d.caption("Table", "Model summary")
    d.table(["Item", "Value"], rows, widths_cm=[6.0, 10.5])

    if results:
        hdr, prow = peak_table(r, results)
        outlets = [e.name for e in sinks]
        d.para(
            "The runs were computed in HEC-HMS. The largest peak at each outlet and "
            "the storm that gives it:"
        )
        crit = []
        for row in prow:
            if row[0] not in outlets:
                continue
            vals = [(v, hdr[i + 1]) for i, v in enumerate(row[1:]) if v is not None]
            if vals:
                v, rn = max(vals)
                crit.append([row[0], _f(v, 3), rn])
        if crit:
            d.caption("Table", "Largest computed peak flow per outlet")
            d.table(["Outlet", "Peak (m3/s)", "Storm"], crit, widths_cm=[5.0, 3.5, 8.0])

    # ------------------------------------------------------------ schematic
    d.heading("Model schematic", 1)
    d.para(
        "Sub-basins drain to the junction at the stream node below them; each "
        "junction passes its flow to the reach that starts there; the node with "
        "no reach leaving it is the outlet (sink) of its system. Element names "
        f"follow the GIS names, with the prefixes '{s.reach_prefix}' (reaches), "
        f"'{s.junction_prefix}' (junctions) and '{s.sink_prefix}' (outlets) because "
        "HEC-HMS needs names that are unique across element types."
    )
    if geometry is not None:
        try:
            from mayim_tools._common.report_figures import ChartStyle

            style = ChartStyle()
            with style.context():
                fig = _schematic(r, geometry, style)
                d.picture(style.png(fig), "Sub-basins, reaches, junctions and outlets")
        except Exception as e:  # noqa: BLE001 - the report must still be written
            notes.append(f"Schematic not drawn: {e}")
    d.caption("Table", "Element connectivity")
    con = [[e.name, e.kind, e.downstream or "-"] for e in r.network.elements]
    d.table(["Element", "Type", "Drains to"], con, widths_cm=[5.5, 3.5, 7.5])

    # ------------------------------------------------------------ inputs
    d.heading("Inputs", 1)
    d.caption("Table", "Input data")
    d.table(
        ["Input", "Source"],
        [[k, str(v)] for k, v in context.get("inputs", [])],
        widths_cm=[5.0, 11.5],
    )

    # ------------------------------------------------------------ methods
    d.heading("Methods", 1)
    _loss_methods_text(d, s, m)
    d.heading("Transform", 2)
    if s.transform == mth.TRANSFORM_SNYDER:
        ms = s.methods
        d.para(
            "Snyder unit hydrograph (standard): the standard lag Tp and the peaking "
            "coefficient Cp of each sub-basin define the peak of the unit "
            "hydrograph, qp = 2.75 Cp / Tp (m3/s per km2 per cm; Chow et al. 1988). "
            + (
                "Tp = 0.75 Ct (L Lca)^0.3 h, with the longest flow path L and the "
                "centroid distance Lca (km) of Catchment Delineation and "
                f"Ct = {ms.snyder_ct:g}; "
                if ms.snyder_tp == mth.SNYDER_TP_CT
                else "Tp = 0.6 Tc (the SCS lag); "
            )
            + f"Cp = {ms.snyder_cp:g} unless a snyder_cp field is given. Ct and Cp "
            "are regional coefficients (Snyder's Appalachian Ct 1.8 - 2.2 and Cp "
            "0.56 - 0.69; Cp generally 0.4 - 0.8); calibrate them or take them "
            "from regional guidance (for South Africa HRU 1/72)."
        )
    elif s.transform == TRANSFORM_CLARK:
        d.para(
            "Clark unit hydrograph with the time of concentration Tc of each "
            f"sub-basin and a storage coefficient R from R/(Tc + R) = "
            f"{s.clark_ratio:g}."
        )
    else:
        d.para(
            "SCS dimensionless unit hydrograph (standard, peak rate factor 484) with "
            "the lag tL = 0.6 Tc (NEH 630 Chapter 15)."
        )
    d.para(
        f"Tc comes from the sub-catchment layer; values below {s.tc_floor_min:g} min "
        "were raised to that minimum. HEC-HMS recommends a time step of at most "
        "0.29 times the lag."
    )
    d.heading("Channel routing", 2)
    d.para(
        "Muskingum routing uses the travel time K of each reach (from the stream "
        "network's routing inputs, or length over a constant velocity) and X; the "
        "number of sub-reaches is chosen for the model time step so that "
        "1/(2(1 - X)) <= K/(n dt) <= 1/(2X). Reaches too short for that at the time "
        "step are routed with Lag routing (lag = K). Muskingum-Cunge uses a "
        "trapezoidal section with the reach length and slope, Manning n and an "
        "index celerity; Lag routing shifts the hydrograph by the travel time."
    )
    if s.routing == mth.ROUTE_LAGK:
        d.para(
            "Lag and K (constant lag and K): the hydrograph is shifted by the lag and "
            "attenuated by a linear reservoir with storage constant K. Unless lag "
            "and K fields are given, the reach travel time T is split into lag = "
            f"{s.methods.lagk_lag_share:g} T and K = "
            f"{1 - s.methods.lagk_lag_share:g} T - a first estimate to be "
            "replaced by calibrated or hydraulically derived values."
        )
    elif s.routing == mth.ROUTE_KW:
        d.para(
            "Kinematic wave: the reach is a trapezoidal channel with its length and "
            "slope, Manning n, bottom width and side slope, solved by finite "
            "differences on sub-reaches about one time step of travel long (dx = "
            "index celerity x time step). Kinematic wave routing translates the "
            "flood wave without attenuation; use Muskingum-Cunge where floodplain "
            "storage matters."
        )
    d.heading("Design storms", 2)
    if SOURCE_DDF in sources:
        d.para(
            "DDF frequency storms: the depths at the HEC-HMS durations (5 min to the "
            "storm duration) are taken from the DDF table, interpolated log-log "
            "between its durations and linearly in ln(T) between its return "
            "periods. HEC-HMS nests them into one storm (the alternating block "
            "principle) with the peak at each chosen position ("
            + _peak_list(r, SOURCE_DDF)
            + ").",
            bold_lead="Frequency storm.",
        )
    if SOURCE_AB in sources:
        d.para(
            "The incremental depths of the Alternating Block design hyetographs "
            "are used for each chosen peak position (with the areal reduction "
            "below, unless the Alternating Block tool already applied one) ("
            + _peak_list(r, SOURCE_AB)
            + ").",
            bold_lead="Alternating Block.",
        )
    if SOURCE_HUFF in sources:
        d.para(
            "The storm depth from the DDF table is distributed in time with the "
            "Huff mass curve (quartile by duration as in Huff (1967), or as "
            "chosen).",
            bold_lead="Huff curves.",
        )
    if SOURCE_ENSEMBLE in sources:
        d.para(
            "The storm depth from the DDF table is distributed with the ensemble "
            "pattern of the matching duration, AEP band (frequent, intermediate or "
            "rare, from the return period) and storm shape.",
            bold_lead="Ensembles.",
        )
    d.para(
        "Hyetograph storms are written as precipitation gages in a HEC-DSS file and "
        "applied uniformly over the sub-basins (Specified Average)."
    )
    add_arf_sections(
        d, r.storms, r.arf, [(run.name, run.storm) for run in r.runs], level=2
    )

    # ------------------------------------------------------------ parameters
    d.heading("Sub-basin parameters", 1)
    d.caption("Table", "Sub-basin parameters")
    loss = s.methods.loss
    loss_hdr = LOSS_COLUMNS[loss]
    hdr = ["Sub-basin", "Drains to", "Area (km2)"] + [h for h, _, _ in loss_hdr]
    hdr.append("Tc (min)")
    if s.transform == TRANSFORM_CLARK:
        hdr.append("Tc / R (h)")
    elif s.transform == mth.TRANSFORM_SNYDER:
        hdr.append("Tp (h) / Cp")
    else:
        hdr.append("Lag (min)")
    rows = []
    for e in subs:
        p = e.params
        if s.transform == TRANSFORM_CLARK:
            last = f"{_f(p.get('clark_tc_h'), 3)} / {_f(p.get('clark_r_h'), 3)}"
        elif s.transform == mth.TRANSFORM_SNYDER:
            last = f"{_f(p.get('snyder_tp_h'), 3)} / {_f(p.get('snyder_cp'), 2)}"
        else:
            last = _f(p.get("lag_min"), 1)
        rows.append(
            [e.name, e.downstream, _f(p["area_km2"], 4)]
            + [_f(p.get(k), nd) for _, k, nd in loss_hdr]
            + [_f(p["tc_min"], 1) + ("*" if "tc_floor" in e.flags else ""), last]
        )
    d.table(hdr, rows)
    srcs = _source_rows(subs)
    if srcs:
        d.caption("Table", "Where the estimated sub-basin parameters came from")
        d.table(["Parameter", "Source", "Sub-basins"], srcs, widths_cm=[5.0, 8.5, 3.0])
    if any("tc_floor" in e.flags for e in subs):
        d.para(f"* Tc raised to the minimum of {s.tc_floor_min:g} min.")
    if any(e.params.get("cn_mean") is not None for e in subs):
        d.caption("Table", "Curve number statistics of the raster cells")
        rows = [
            [
                e.name,
                str(e.params.get("n_cells", "-")),
                _f(e.params.get("cn_mean"), 1),
                _f(e.params.get("cn_runoff"), 1),
                _f(e.params.get("cn_p05"), 1),
                _f(e.params.get("cn_p95"), 1),
            ]
            for e in subs
        ]
        d.table(["Sub-basin", "Cells", "Mean CN", "Runoff-eq. CN", "P05", "P95"], rows)

    if reaches:
        d.heading("Reach parameters", 1)
        d.caption("Table", "Reach routing parameters")
        rows = []
        for e in reaches:
            p = e.params
            meth = p.get("method", "-")
            if meth == "muskingum":
                par = f"K {_f(p['k_h'], 3)} h, X {_f(p['x'], 2)}, {p['steps']} step(s)"
            elif meth in ("muskingum_cunge", mth.ROUTE_KW):
                par = (
                    f"S {_f(p['slope'], 4)}, n {_f(p['n'], 3)}, B {_f(p['width_m'], 1)} m, "
                    f"z {_f(p['side_slope'], 1)}"
                )
                if meth == mth.ROUTE_KW:
                    par += f", {p['subreaches']} sub-reach(es)"
            elif meth == mth.ROUTE_LAGK:
                par = f"lag {_f(p['lagk_lag_h'], 3)} h, K {_f(p['lagk_k_h'], 3)} h"
            else:
                par = f"lag {_f(p.get('lag_min'), 1)} min"
            rows.append(
                [
                    e.name,
                    e.downstream,
                    meth.replace("_", "-").capitalize(),
                    _f(p.get("length_m"), 0),
                    _f(p.get("travel_min"), 1),
                    par,
                ]
            )
        d.table(
            ["Reach", "Flows to", "Method", "Length (m)", "Travel (min)", "Parameters"],
            rows,
        )

    # ------------------------------------------------------------ storms
    d.heading("Design storms", 1)
    d.caption("Table", "Design storms and simulation runs")
    rows = []
    for run in r.runs:
        st = run.storm
        rows.append(
            [
                run.name,
                ari_label(st.ari),
                duration_label(st.duration_min),
                _f(st.total_mm, 1),
                _f(st.peak_intensity_mmh, 1),
                st.pattern,
            ]
        )
    d.table(
        ["Run", "Return period", "Duration", "Depth (mm)", "Peak (mm/h)", "Pattern"],
        rows,
    )
    try:
        from mayim_tools._common.report_figures import ChartStyle

        style = ChartStyle()
        with style.context():
            fig = _storm_figure(r, style)
            tmax = max(run.storm.ari for run in r.runs)
            d.picture(
                style.png(fig),
                f"Cumulative rainfall of the {ari_label(tmax)} year storms",
            )
    except Exception as e:  # noqa: BLE001
        notes.append(f"Storm figure not drawn: {e}")

    if r.ensembles:
        d.heading("Ensemble analyses", 2)
        d.para(
            "Each storm method and return period is an ensemble analysis in "
            "HEC-HMS, with all its durations and peak positions / patterns as "
            "members and the outflow of every element as an analysis point. "
            "HEC-HMS reports the statistics (minimum, mean, maximum) of the "
            "members' hydrographs; the largest peak is the critical storm of that "
            "method and return period. The runs share one control specification, "
            "as HEC-HMS ensemble members need the same time window."
        )
        d.caption("Table", "Ensemble analyses")
        rows = ensemble_rows(r, results)
        if results:
            d.table(
                ["Ensemble", "Members", "Outlet", "Max peak (m3/s)", "Critical run"],
                [[x[0], str(x[3]), x[4], _f(x[5], 3), x[6]] for x in rows],
                widths_cm=[3.8, 2.0, 3.2, 2.8, 4.7],
            )
        else:
            d.table(
                ["Ensemble", "Method", "Return period", "Members"],
                [[x[0], x[1], x[2], str(x[3])] for x in rows],
                widths_cm=[5.0, 4.5, 3.5, 3.5],
            )

    # ------------------------------------------------------------ results
    if results:
        d.heading("Computed peak flows", 1)
        d.para(
            "Peak outflow (m3/s) at the junctions and outlets for each simulation "
            "run, as computed by HEC-HMS."
        )
        hdr, prow = peak_table(r, results)
        runs = hdr[1:]
        chunk = 5
        if len(runs) > 40:
            d.para(
                f"{len(runs)} runs were computed; the peak flow of every run and "
                "element is in the CSV output (section 'Computed peak flows')."
            )
            runs = []
        for i in range(0, len(runs), chunk):
            sub = runs[i : i + chunk]
            d.caption("Table", f"Peak flows (m3/s), runs {i + 1} to {i + len(sub)}")
            d.table(
                ["Element"] + sub,
                [
                    [row[0]] + [_f(v, 3) for v in row[1 + i : 1 + i + len(sub)]]
                    for row in prow
                ],
            )

    # ------------------------------------------------------------ files
    d.heading("HEC-HMS project files", 1)
    d.para(
        f"The project is in {r.folder}. Open the .hms file in HEC-HMS "
        f"{s.hms_version} (File > "
        "Open): the basin model (with the terrain, the sub-basin and reach map "
        "layers and the curve number grid), the meteorologic models, control "
        "specification(s), simulation runs, precipitation gages and ensemble "
        "analyses load with it. run_all_simulations.cmd computes every run "
        "without opening the interface."
    )
    d.caption("Table", "Outputs")
    d.table(
        ["Output", "Path"],
        [[k, str(v)] for k, v in context.get("outputs", [])],
        widths_cm=[5.0, 11.5],
    )

    # ------------------------------------------------------------ notes
    d.heading("Notes and limitations", 1)
    for txt in [
        "The parameters are first estimates from GIS data; calibrate or verify "
        "them against observed floods or regional methods where possible.",
        (
            "The design storms are point rainfall (no areal reduction); for larger "
            "catchments apply an areal reduction."
            if r.arf is None or r.arf.is_unity
            else "One areal reduction factor area applies to the whole model, so "
            "peaks at internal junctions with a much smaller upstream area are "
            "slightly underestimated."
        ),
        "Baseflow, canopy and surface storage are not modelled (event model).",
        "Check the HEC-HMS run logs for warnings, in particular the time step "
        "against the shortest lag and the Muskingum stability.",
    ] + list(context.get("warnings") or [])[:20]:
        d.bullet(txt)

    d.heading("References", 1)
    for ref in REFERENCES + method_references(s) + arf_references(r.arf):
        d.para(ref)
    d.doc.save(path)
    return notes
