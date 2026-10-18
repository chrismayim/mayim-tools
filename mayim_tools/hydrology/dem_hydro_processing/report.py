"""Word (.docx) report for the DEM Hydrological Processing tool - zero QGIS
dependency. python-docx and matplotlib are imported lazily. Numbers are
written without thousands separators."""

from __future__ import annotations

import numpy as np

from mayim_tools._common.docx_report import (
    C_BLUE,
    C_ORANGE,
    Doc,
    new_figure,
    png,
    style_axes,
)

from .core.conditioning import ConditioningResult
from .export import STAT_LABELS, fmt

REPORT_TITLE = "DEM Hydrological Processing"
MAX_TABLE_ROWS = 25

REFERENCES = [
    "Barnes, R., Lehman, C. & Mulla, D. (2014). Priority-flood: an optimal "
    "depression-filling and watershed-labeling algorithm for digital "
    "elevation models. Computers & Geosciences, 62, 117-127.",
    "Barnes, R., Callaghan, K.L. & Wickert, A.D. (2020). Computing water "
    "flow through complex landscapes, part 2: finding hierarchies in "
    "depressions and morphological segmentations. Earth Surface Dynamics, "
    "8, 431-445.",
    "Garbrecht, J. & Martz, L.W. (1997). The assignment of drainage "
    "direction over flat surfaces in raster digital elevation models. "
    "Journal of Hydrology, 193, 204-213.",
    "Lindsay, J.B. & Creed, I.F. (2005). Removal of artifact depressions "
    "from digital elevation models: towards a minimum impact approach. "
    "Hydrological Processes, 19, 3113-3126.",
    "Lindsay, J.B. & Creed, I.F. (2006). Distinguishing actual and "
    "artefact depressions in digital elevation data. Computers & "
    "Geosciences, 32, 1192-1204.",
    "Lindsay, J.B. (2016). Efficient hybrid breaching-filling sink removal "
    "methods for flow path enforcement in digital elevation models. "
    "Hydrological Processes, 30, 846-857.",
]


def _t(rows, limit=MAX_TABLE_ROWS):
    return rows[:limit]


def _hist_figure(result: ConditioningResult):
    diff = (result.corrected - result.original).astype("float64")
    changed = diff[np.isfinite(diff) & (np.abs(diff) > 0)]
    if changed.size == 0:
        return None
    fig = new_figure(6.0)
    ax = fig.add_subplot(111)
    ax.hist(changed, bins=60, color=C_BLUE)
    ax.set_xlabel("Elevation change (m)")
    ax.set_ylabel("Cells")
    style_axes(ax)
    return png(fig)


def _map_figure(result: ConditioningResult):
    diff = (result.corrected - result.original).astype("float64")
    diff[~np.isfinite(diff)] = 0.0
    if not np.any(np.abs(diff) > 0):
        return None
    lim = float(np.percentile(np.abs(diff[np.abs(diff) > 0]), 99))
    lim = max(lim, 1e-6)
    fig = new_figure(9.0)
    ax = fig.add_subplot(111)
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    for s in result.sinks:
        ax.plot(s.col, s.row, "o", ms=3, mfc="none", mec=C_ORANGE, mew=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Elevation change (m)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    return png(fig)


def write_report_docx(path: str, result: ConditioningResult, context: dict) -> None:
    s = result.settings
    st = result.stats
    d = Doc()
    d.doc.add_heading(REPORT_TITLE, level=0)
    d.para(
        f"{context.get('tool', REPORT_TITLE)} {context.get('version', '')}, "
        f"run {context.get('run_time', '')}."
    )

    d.heading("1. Purpose and method", 1)
    d.para(
        "Raster flow routing needs every cell to drain to a lower neighbour. "
        "Real elevation models contain depressions, most of them artefacts of "
        "vertical error, resolution and surface features. This tool finds the "
        "depressions with a depression hierarchy (Barnes et al., 2020), "
        "resolves each by filling or by least-cost breaching, whichever "
        "changes the surface less (Lindsay & Creed, 2005; Lindsay, 2016), and "
        "then runs a drainage-enforcement pass so that no corrected cell is "
        "left without a downhill neighbour (Garbrecht & Martz, 1997; Barnes "
        "et al., 2014)."
    )
    if s.mode == "continuous":
        d.para(
            "Mode: continuous drainage. Every depression was resolved except "
            "sinks the user confirmed as genuine (pans, regional sinks, lakes)."
        )
    else:
        d.para(
            "Mode: Likely Natural depressions preserved. Depressions that "
            "persist under simulated vertical error were classified "
            "(Lindsay & Creed, 2006) and those labelled Likely Natural were "
            "left in the surface. This label only means the depression is "
            "deeper than the DEM noise; it cannot distinguish a pan from a "
            "dam or borrow pit without external data."
        )

    d.heading("2. Inputs and parameters", 1)
    d.caption("Table", "Inputs and parameters")
    d.table(
        ["Item", "Value"],
        [
            ["Input DEM", fmt(context.get("dem_path", ""))],
            ["CRS", fmt(context.get("crs", ""))],
            ["Cell size (m)", fmt(context.get("cell_size", ""))],
            ["Grid (columns x rows)", fmt(context.get("grid", ""))],
            ["DEM source type", fmt(context.get("dem_source", ""))],
            ["Vertical error std (m)", fmt(s.vertical_error_std)],
            ["Stream centreline", fmt(context.get("stream_layer", "no"))],
            ["Confirmed-sink layer", fmt(context.get("keep_layer", "no"))],
            ["Classification table", fmt(context.get("review_table", "no"))],
            ["Maximum breach length (cells)", fmt(s.max_breach_length)],
            ["Maximum breach depth (m)", fmt(s.max_breach_depth)],
        ],
        widths_cm=[7, 9],
    )

    d.heading("3. Results", 1)
    d.caption("Table", "Summary of results")
    d.table(
        ["Item", "Value"],
        [[label, fmt(st.get(key))] for key, label in STAT_LABELS],
        widths_cm=[10, 6],
    )
    for message in result.warnings:
        d.para(message, bold_lead="Warning:")
    hist = _hist_figure(result)
    if hist:
        d.picture(hist, "Distribution of elevation changes made to the DEM")
    chg = _map_figure(result)
    if chg:
        d.picture(
            chg,
            "Elevation change (corrected minus original). Circles mark "
            "remaining sink clusters.",
        )

    if result.keep_matches:
        d.heading("4. Confirmed sinks supplied", 1)
        d.para(
            "Each supplied feature and what it matched. Nothing is ignored " "silently."
        )
        d.caption("Table", "Confirmed sink features and their matches")
        d.table(
            ["Feature", "Geometry", "Result", "Depressions kept", "Detail"],
            [
                [
                    m.label,
                    m.kind,
                    m.status,
                    " ".join(str(i) for i in m.leaf_ids),
                    m.detail,
                ]
                for m in _t(result.keep_matches, 60)
            ],
            widths_cm=[3, 2, 4, 3, 4],
        )

    d.heading("5. Sinks remaining in the corrected DEM", 1)
    if result.sinks:
        d.caption("Table", "Remaining sink clusters")
        d.table(
            ["Id", "Kind", "Reason", "Cells", "Row", "Col", "Elevation (m)"],
            [
                [
                    k.sink_id,
                    k.kind,
                    k.reason,
                    k.cells,
                    k.row,
                    k.col,
                    fmt(k.elevation),
                ]
                for k in _t(result.sinks, 60)
            ],
            widths_cm=[1.2, 2, 5, 1.5, 1.5, 1.5, 2.3],
        )
    else:
        d.para("None. Every valid cell drains to a lower neighbour or the edge.")

    d.heading("6. Largest corrections", 1)
    units = sorted(result.units, key=lambda u: -u.volume_m3)
    d.caption(
        "Table", f"Largest {min(len(units), MAX_TABLE_ROWS)} corrections by volume"
    )
    d.table(
        ["Unit", "Method", "Area (m2)", "Max depth (m)", "Volume (m3)", "Class"],
        [
            [
                u.unit_id,
                u.method,
                fmt(u.area_m2),
                fmt(u.max_depth_m),
                fmt(u.volume_m3),
                u.classification,
            ]
            for u in _t(units)
        ],
        widths_cm=[1.8, 2, 3, 3, 3, 3.2],
    )
    d.para("The full list is in the CSV report, section 7.")

    d.heading("7. Limitations", 1)
    d.bullet(
        "A DEM is often a surface model: canopy and structures are included, "
        "so some depressions are not terrain features."
    )
    d.bullet(
        "Corrections are bounded by the vertical error of the source data; "
        "changes larger than that error merit a visual check."
    )
    d.bullet(
        "Classification from the DEM alone cannot separate natural pans from "
        "dams, borrow pits or quarries. Confirm genuine sinks with imagery or "
        "field knowledge and supply them as a confirmed-sink layer."
    )
    d.bullet(
        "Embankments or ridges with no depression behind them are not "
        "detected or removed."
    )

    d.heading("References", 1)
    for ref in REFERENCES:
        d.bullet(ref)
    d.doc.save(path)
