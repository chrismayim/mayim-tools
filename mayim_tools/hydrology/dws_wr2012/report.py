"""
Word report of Extract: DWS and WR2012 data, in the Mayim house style
(report and figure standard): title page, summary, tool and method,
inputs, parameters, results, outputs, notes and references. Zero QGIS
dependency - the layout map arrives as a PlanMap rendered by the wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mayim_tools._common.docx_report import Doc
from mayim_tools._common.report_figures import ChartStyle
from mayim_tools._common.report_maps import (
    OSM_LICENCE_NOTE,
    PlanMap,
    figure_rendered_map,
)

from .core import TOOL_VERSION, WR_FIELD_BY_NAME, SummaryRow, fmt, summary_text

REPORT_TITLE = "DWS and WR2012 data for water balance studies"
TOOL_NAME = "Extract: DWS and WR2012 data"

REFERENCES = [
    "Bailey AK and Pitman WV (2015). Water Resources of South Africa 2012 "
    "Study (WR2012). WRC Report No. K5/2143/1. Water Research Commission, "
    "Pretoria.",
    "Department of Water Affairs and Forestry (2006). Groundwater Resource "
    "Assessment II (GRA II). DWAF, Pretoria.",
    "Department of Water and Sanitation (2023). National Water Resource "
    "Strategy, third edition (NWRS III) - Water Management Areas. "
    "Government Gazette 49225, Notice 3855, 1 September 2023.",
    "Department of Water and Sanitation. Hydrological services - verified "
    "station lists per water management area (dwafapp2_wma), station list "
    "output of 20 September 2026. www.dws.gov.za/Hydrology.",
    "Midgley DC, Pitman WV and Middleton BJ (1994). Surface Water Resources "
    "of South Africa 1990 (WR90). WRC Reports 298/1/94 to 298/6/94. Water "
    "Research Commission, Pretoria.",
    "Weepener HL, van den Berg HM, Metz M and Hamandawana H (2012). The "
    "development of a hydrologically improved Digital Elevation Model and "
    "derived products for South Africa based on the SRTM DEM. WRC Report "
    "No. 1908/1/11. Water Research Commission, Pretoria.",
]


@dataclass
class ReportData:
    """Everything the report shows (filled by the QGIS wrapper)."""

    project_number: str = ""
    run_time: str = ""
    study_area: str = ""
    study_area_km2: float = 0.0
    buffer_m: float = 0.0
    bbox: tuple = ()
    wma_version: str = "2023"
    inputs: list = field(default_factory=list)  # [(item, value)]
    settings: list = field(default_factory=list)  # [(item, value)]
    layers: list = field(default_factory=list)  # [(layer, features, source)]
    wmas: list = field(default_factory=list)  # [(id, name)]
    drainage: list = field(default_factory=list)  # [(level, codes text)]
    wr_rows: list | None = None  # WR2012 quaternary rows; None = not run
    summary: list = field(default_factory=list)  # [SummaryRow]
    stations: list = field(default_factory=list)  # station rows
    station_counts: list = field(default_factory=list)  # [(type, n, open)]
    station_types: list = field(default_factory=list)  # labels
    station_buffer_km: float = 0.0
    open_only: bool = False
    outputs: list = field(default_factory=list)  # [(label, path)]
    warnings: list = field(default_factory=list)
    plan_map: PlanMap | None = None


def _value(s: SummaryRow) -> str:
    return summary_text(s) if s.value is not None else "-"


def _quaternary_chart(rows, style: ChartStyle):
    """Per-quaternary MAP and A-pan MAE (top) and MAR, recharge and baseflow
    (bottom), mm/a."""
    import numpy as np

    rows = [r for r in rows if r.get("MAP_MM") is not None][:40]
    if not rows:
        return None
    names = [r["QUAT"] for r in rows]
    x = np.arange(len(rows))
    height = 13.0 if len(rows) <= 15 else 15.0

    def vals(name):
        return np.array(
            [np.nan if r.get(name) is None else float(r[name]) for r in rows]
        )

    with style.context():
        fig = style.figure(height)
        top, bottom = fig.subplots(2, 1, sharex=True)
        for ax, series in (
            (top, [("MAP_MM", "MAP"), ("MAE_MM", "A-pan MAE")]),
            (
                bottom,
                [
                    ("MAR_MM", "MAR"),
                    ("RECH_MM", "Recharge (GRA II)"),
                    ("BASEF_MM", "Baseflow (GRA II)"),
                ],
            ),
        ):
            present = [(n, lab) for n, lab in series if np.isfinite(vals(n)).any()]
            if not present:
                ax.set_visible(False)
                continue
            w = 0.8 / len(present)
            for i, (n, lab) in enumerate(present):
                ax.bar(x + (i - (len(present) - 1) / 2) * w, vals(n), w, label=lab)
            ax.set_ylabel("mm/a")
            ax.set_xlim(-0.6, len(rows) - 0.4)
            style.axes(ax, minor=False)
            ax.tick_params(axis="x", which="minor", bottom=False)
            style.legend(ax, loc="upper left", ncols=len(present))
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        bottom.set_xticks(x)
        bottom.set_xticklabels(names, rotation=90 if len(rows) > 12 else 0)
        bottom.set_xlabel("Quaternary catchment")
        for label in bottom.get_xticklabels():
            label.set_fontproperties(style.font_props())
        return style.png(fig)


def write_report_docx(path: str, data: ReportData) -> list[str]:
    """Writes the report; returns messages for the user."""
    d = Doc(house_style=True)
    messages: list[str] = []
    try:
        style = ChartStyle()
        messages.extend(style.warnings)
    except ImportError:  # pragma: no cover - matplotlib ships with QGIS
        style = None

    d.title_page(
        REPORT_TITLE,
        purpose=(
            "Catchment, water-resource and monitoring-station data from the "
            "Department of Water and Sanitation (DWS) and the WR2012 study for "
            "the study area, as inputs to a water balance assessment."
        ),
        details=[
            ("Project number", data.project_number),
            ("Study area", data.study_area),
            ("Date", data.run_time),
            ("Tool", f"{TOOL_NAME} (Mayim Tools, version {TOOL_VERSION})"),
            ("Prepared by", "Mayim Consulting"),
        ],
    )

    wr_rows = data.wr_rows or []
    summary = {s.item: s for s in data.summary}

    # 1. Summary ---------------------------------------------------------------
    d.heading("1. Summary", 1)
    quats = ", ".join(r["QUAT"] for r in wr_rows) if wr_rows else "-"
    wma_text = "; ".join(f"{i} {n}".strip() for i, n in data.wmas) or "-"
    d.para(
        f"The study area ({fmt(data.study_area_km2, 2)} km²) lies in the "
        f"{data.wma_version} water management area(s) {wma_text}."
        + (
            f" It intersects the WR2012 quaternary catchments {quats}."
            if wr_rows
            else ""
        )
    )
    if wr_rows and data.summary:
        key = [
            ("Mean annual precipitation (MAP)", "MAP"),
            ("Naturalised mean annual runoff (MAR)", "naturalised MAR"),
            (
                "Naturalised MAR generated in the study area",
                "naturalised MAR generated in the study area",
            ),
            ("A-pan mean annual evaporation (MAE)", "A-pan MAE"),
            ("Groundwater recharge (GRA II)", "groundwater recharge"),
        ]
        parts = [
            f"{text} {_value(summary[k])} {summary[k].unit}"
            for k, text in key
            if k in summary and summary[k].value is not None
        ]
        if parts:
            d.para(
                "Area-weighted over the study area, the WR2012 and GRA II "
                "values are: " + "; ".join(parts) + "."
            )
    if data.station_counts:
        d.para(
            f"{len(data.stations)} DWS monitoring "
            f"station(s) of the selected types lie within "
            f"{fmt(data.station_buffer_km)} km of the study area: "
            + "; ".join(
                f"{t.lower()} {n} ({o} open)" for t, n, o in data.station_counts
            )
            + "."
        )
    elif data.station_types:
        d.para(
            f"No DWS monitoring stations of the selected types lie within "
            f"{fmt(data.station_buffer_km)} km of the study area."
        )

    # 2. Tool and method -------------------------------------------------------
    d.heading("2. Tool and method", 1)
    d.para(
        f"The {TOOL_NAME} tool (Mayim Tools, version {TOOL_VERSION}) collects "
        "the published DWS and WR2012 data for a study area in one run:"
    )
    d.bullet(
        "DWS spatial layers - water management areas, primary to quaternary "
        "drainage regions, quinary catchments, 1:50 000 rivers and dams and "
        "the Reserve status layers - are queried from the DWS ArcGIS REST "
        "services (gia.dws.gov.za, with the DWS ArcGIS Online copies as a "
        "fall-back) for the bounding box of the study area. Catchments and "
        "management areas that intersect the study area are kept whole; "
        "rivers, dams and sites are clipped to it."
    )
    d.bullet(
        "WR2012 quaternary data are read from a local copy of the WR2012 GIS "
        "coverages. Each quaternary that intersects the study area receives "
        "its mean annual precipitation, naturalised mean annual runoff, "
        "WRSM/Pitman calibration parameters (WR2012 set), ecological "
        "importance and sensitivity, present ecological state, TDS and "
        "sediment yield. The rainfall zone, evaporation zone and A-pan "
        "evaporation class are the ones covering the largest part of the "
        "quaternary, and the GRA II groundwater values are the mean of the "
        "1 km grid cells inside it."
    )
    d.bullet(
        "The water-balance summary weights each quaternary's values by the "
        "area of the quaternary inside the study area; volumes are depth "
        "times that area."
    )
    d.bullet(
        "DWS monitoring stations (locations only) come from the DWS verified "
        f"station lists per water management area (output of "
        f"{STATION_LIST_DATE_TEXT}), bundled with the tool. A station is "
        "'open' when its record runs to 2025 or later."
    )

    # 3. Inputs ----------------------------------------------------------------
    d.heading("3. Inputs", 1)
    d.caption("Table", "Inputs")
    d.table(["Input", "Value"], [list(r) for r in data.inputs], widths_cm=[5.5, 10.5])

    # 4. Parameters ------------------------------------------------------------
    d.heading("4. Parameters", 1)
    d.caption("Table", "Settings used")
    d.table(
        ["Setting", "Value"], [list(r) for r in data.settings], widths_cm=[5.5, 10.5]
    )
    if wr_rows:
        d.caption("Table", "WR2012 quaternary attributes")
        d.table(
            ["Field", "Description", "Unit", "Source"],
            [
                [f.name, f.label, f.unit or "-", f.source]
                for f in WR_FIELD_BY_NAME.values()
            ],
            widths_cm=[2.7, 6.0, 2.3, 5.5],
        )

    # 5. Results ---------------------------------------------------------------
    d.heading("5. Results", 1)
    sec = 0
    if data.plan_map is not None and style is not None:
        sec += 1
        d.heading(f"5.{sec} Location", 3)
        try:
            d.picture(
                figure_rendered_map(data.plan_map, style),
                "Study area with the DWS layers and monitoring stations",
            )
        except Exception as e:  # the map must never stop the report
            messages.append(f"Layout map not drawn: {e}")

    sec += 1
    d.heading(f"5.{sec} Water management areas and drainage regions", 3)
    if data.wmas:
        d.caption("Table", f"Water management areas ({data.wma_version})")
        d.table(["WMA", "Name"], [[str(i), n] for i, n in data.wmas], widths_cm=[3, 13])
    if data.drainage:
        d.caption("Table", "Drainage regions intersecting the study area")
        d.table(["Level", "Codes"], [list(r) for r in data.drainage], widths_cm=[4, 12])
    d.caption("Table", "DWS layers downloaded")
    d.table(
        ["Layer", "Features", "Source"],
        [[a, str(b), c] for a, b, c in data.layers],
        widths_cm=[4.5, 2, 9.5],
    )

    if data.wr_rows is not None:
        sec += 1
        d.heading(f"5.{sec} WR2012 quaternary data", 3)
        if not wr_rows:
            d.para("No WR2012 quaternary intersects the study area.")
        else:
            groups = [
                (
                    "Area and water balance",
                    [
                        "QUAT",
                        "AREA_KM2",
                        "AOI_KM2",
                        "MAP_MM",
                        "MAR_MM",
                        "MAR_MCM",
                        "RUNOFF_PC",
                        "MAE_MM",
                    ],
                ),
                (
                    "Zones and classes",
                    [
                        "QUAT",
                        "HYDROZ",
                        "CURVE",
                        "RAINZONE",
                        "EVAPZONE",
                        "MAE_CLASS",
                        "EISC",
                        "PES",
                    ],
                ),
                (
                    "WRSM/Pitman parameters (WR2012)",
                    ["QUAT", "POW", "ST", "FT", "ZMIN", "ZMAX", "GPOW", "HGSL", "HGGW"],
                ),
                (
                    "Groundwater (GRA II), water quality and sediment",
                    [
                        "QUAT",
                        "RECH_MM",
                        "RECH_DRY",
                        "BASEF_MM",
                        "EXPL_FACT",
                        "GEP_WET",
                        "GEP_DRY",
                        "TDS_P95",
                        "SED_YLD",
                    ],
                ),
            ]
            for title, names in groups:
                fields = [WR_FIELD_BY_NAME[n] for n in names]
                if not any(
                    r.get(f.name) is not None for r in wr_rows for f in fields[1:]
                ):
                    continue
                d.caption("Table", f"WR2012 quaternaries - {title}")
                d.table(
                    [f"{f.name}\n({f.unit})" if f.unit else f.name for f in fields],
                    [
                        [fmt(r.get(f.name), f.decimals) or "-" for f in fields]
                        for r in wr_rows
                    ],
                    widths_cm=[16.0 / len(fields)] * len(fields),
                )
            if style is not None:
                try:
                    png = _quaternary_chart(wr_rows, style)
                    if png is not None:
                        d.picture(
                            png, "WR2012 and GRA II depths per quaternary catchment"
                        )
                except Exception as e:  # pragma: no cover
                    messages.append(f"Quaternary chart not drawn: {e}")
        if data.summary:
            sec += 1
            d.heading(f"5.{sec} Water-balance summary for the study area", 3)
            d.caption("Table", "Area-weighted water-balance indicators")
            d.table(
                ["Item", "Value", "Unit", "Note"],
                [[s.item, _value(s), s.unit or "-", s.note] for s in data.summary],
                widths_cm=[5.5, 2.5, 2.5, 5.5],
            )

    if data.station_types:
        sec += 1
        d.heading(f"5.{sec} DWS monitoring stations", 3)
        if data.station_counts:
            d.caption("Table", "Stations by type")
            d.table(
                ["Type", "Stations", "Open"],
                [[t, str(n), str(o)] for t, n, o in data.station_counts],
                widths_cm=[8, 4, 4],
            )
            listed = data.stations[:80]
            d.caption(
                "Table",
                "Stations"
                + (
                    f" (first {len(listed)} of {len(data.stations)}; all in the CSV)"
                    if len(data.stations) > len(listed)
                    else ""
                ),
            )
            d.table(
                ["Station", "Name", "Type", "Quat.", "Record", "Open", "km"],
                [
                    [
                        r.get("STATION", ""),
                        r.get("NAME", ""),
                        r.get("CATEGORY", ""),
                        r.get("QUAT", ""),
                        f"{r.get('START', '')[:4]}-{r.get('END', '')[:4]}",
                        r.get("OPEN", ""),
                        fmt(r.get("DIST_KM"), 1),
                    ]
                    for r in listed
                ],
                widths_cm=[2.2, 4.6, 2.8, 1.5, 2.2, 1.2, 1.5],
            )
        else:
            d.para("No stations of the selected types were found.")

    # 6. Outputs ---------------------------------------------------------------
    if data.outputs:
        d.heading("6. Outputs", 1)
        d.caption("Table", "Files produced by the tool")
        d.table(
            ["Output", "Location"], [list(o) for o in data.outputs], widths_cm=[5, 11]
        )

    # 7. Notes -----------------------------------------------------------------
    d.heading("7. Notes", 1)
    for note in [
        "WR2012 values are regional, quaternary-scale estimates for the "
        "1920-2010 naturalised record. The pro-rata values for a study area "
        "smaller than a quaternary do not account for local variation in "
        "rainfall, soils, land use or abstractions and are not a substitute "
        "for a site-specific water balance or a WRSM/Pitman simulation.",
        "MAR is the incremental naturalised runoff generated in each "
        "quaternary; upstream inflows are not included.",
        "The A-pan evaporation is the midpoint of the WR2012 class covering "
        "the largest part of the quaternary; apply pan-to-open-water factors "
        "for reservoir evaporation.",
        "GRA II values are the means of the 1 km grids inside each "
        "quaternary. The recharge grid is in mm/a; the dry-recharge and "
        "baseflow grids are in m³/km²/a and are converted to mm/a; the "
        "exploitation potential stays in m³/km²/a. In some areas the GRA II "
        "dry recharge exceeds the average recharge - check against the GRA II "
        "documentation before use.",
        "Pitman ZMIN and ZMAX values of 999 are given as such in the WR2012 "
        "calibration table (a placeholder rather than a rate); see the "
        "WRSM/Pitman documentation before using them in a model.",
        "The DWS drainage regions (Weepener et al. 2012) and the WR2012 "
        "quaternaries are separate delineations and differ along their "
        "boundaries; the WR2012 values belong to the WR2012 quaternaries.",
        "The station layer gives locations only. Use the separate station "
        "data tool or the DWS hydrology website for the records.",
    ]:
        d.bullet(note)
    for w in data.warnings:
        d.bullet(w)

    # 8. References ------------------------------------------------------------
    d.heading("8. References", 1)
    for ref in REFERENCES:
        d.para(ref)
    if data.plan_map is not None and data.plan_map.attribution:
        d.para(OSM_LICENCE_NOTE)

    d.doc.save(path)
    return list(dict.fromkeys(messages))


STATION_LIST_DATE_TEXT = "20 September 2026"
