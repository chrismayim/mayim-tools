"""Word (.docx) report for the DEM Hydrological Processing tool, in the Mayim
report and figure standard: title page, summary, tool and method, inputs and
settings, parameters and calculations, results (layout map of the elevation
changes and remaining sinks, distribution of the changes and the tables),
outputs, notes and limitations, and references.

Zero QGIS dependency: the layout map is rendered by QGIS in the wrapper
(qgis_map.make_plan_map, a PlanMapFn); without it a simplified map of the
elevation changes is drawn with matplotlib. python-docx and matplotlib are
imported lazily. No thousands separators; missing values are shown as "-".
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .core.conditioning import MODE_CONTINUOUS, ConditioningResult

if TYPE_CHECKING:
    from mayim_tools._common.report_maps import PlanMap

REPORT_TITLE = "DEM Hydrological Processing"
MAX_TABLE_ROWS = 25

# (item label, plot width in cm, dpi) -> PlanMap or None, supplied by the QGIS
# wrapper (qgis_map.make_plan_map).
PlanMapFn = Callable[[str, float, float], "PlanMap | None"]

# Classes of the elevation change (corrected minus original, m) used by the
# layout map: (upper limit, colour, label). Unchanged cells are transparent.
# The QML style styles/DEM_elevation_change.qml uses the same classes.
CHANGE_CLASSES = [
    (-2.0, "#000080", "Lowered more than 2 m"),
    (-0.5, "#0000FF", "Lowered 0.5 to 2 m"),
    (-0.1, "#4F81FF", "Lowered 0.1 to 0.5 m"),
    (-1e-6, "#00FFFF", "Lowered less than 0.1 m"),
    (1e-6, None, "Unchanged"),
    (0.1, "#FFFF00", "Raised less than 0.1 m"),
    (0.5, "#FFC000", "Raised 0.1 to 0.5 m"),
    (2.0, "#FF0000", "Raised 0.5 to 2 m"),
    (math.inf, "#800000", "Raised more than 2 m"),
]
SINK_STYLE = {  # fallback map only; the QML style is styles/DEM_remaining_sinks.qml
    "unresolved": ("o", "#FF0000", "Remaining sink, not confirmed (investigate)"),
    "kept": ("^", "#0000FF", "Confirmed sink (kept)"),
}

REFERENCES = [
    "Barnes, R., Lehman, C. & Mulla, D. (2014). Priority-flood: an optimal "
    "depression-filling and watershed-labeling algorithm for digital "
    "elevation models. Computers & Geosciences, 62, 117–127.",
    "Barnes, R., Callaghan, K.L. & Wickert, A.D. (2020). Computing water "
    "flow through complex landscapes, part 2: finding hierarchies in "
    "depressions and morphological segmentations. Earth Surface Dynamics, "
    "8, 431–445.",
    "Garbrecht, J. & Martz, L.W. (1997). The assignment of drainage "
    "direction over flat surfaces in raster digital elevation models. "
    "Journal of Hydrology, 193, 204–213.",
    "Lindsay, J.B. & Creed, I.F. (2005). Removal of artifact depressions "
    "from digital elevation models: towards a minimum impact approach. "
    "Hydrological Processes, 19, 3113–3126.",
    "Lindsay, J.B. & Creed, I.F. (2006). Distinguishing actual and "
    "artefact depressions in digital elevation data. Computers & "
    "Geosciences, 32, 1192–1204.",
    "Lindsay, J.B. (2016). Efficient hybrid breaching-filling sink removal "
    "methods for flow path enforcement in digital elevation models. "
    "Hydrological Processes, 30, 846–857.",
]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def num(value, fmt: str = "{:.2f}") -> str:
    """A number without thousands separators; '-' when missing."""
    if value is None or isinstance(value, bool):
        return "-" if value is None else ("yes" if value else "no")
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return "-"
        return fmt.format(value)
    except (TypeError, ValueError):
        return text(value)


def text(value) -> str:
    if value is None:
        return "-"
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none") else "-"


def contour_text(minor_interval) -> str:
    """'1 m minor, 5 m major' or 'none'."""
    if not minor_interval:
        return "none"
    return f"{minor_interval:g} m minor, {5 * minor_interval:g} m major"


def _limit(value, unit: str) -> str:
    return "unlimited" if not value else f"{value:g} {unit}"


def change_class_label(value: float) -> str:
    for upper, _colour, label in CHANGE_CLASSES:
        if value <= upper:
            return label
    return CHANGE_CLASSES[-1][2]


def change_classes_present(diff: np.ndarray) -> list[str]:
    """Labels of the elevation change classes that occur (unchanged
    excluded), in class order - the layout map legend shows only these."""
    d = diff[np.isfinite(diff)]
    labels = []
    lower = -math.inf
    for upper, colour, label in CHANGE_CLASSES:
        if colour is not None and np.any((d > lower) & (d <= upper)):
            labels.append(label)
        lower = upper
    return labels


def elevation_change(result: ConditioningResult, nodata=None) -> np.ndarray:
    """Corrected minus original (m); NaN where the DEM has no data."""
    diff = (result.corrected - result.original).astype("float64")
    invalid = ~np.isfinite(result.original)
    if nodata is not None:
        invalid |= result.original == nodata
    diff[invalid] = np.nan
    return diff


def cell_xy(geotransform, row: float, col: float) -> tuple[float, float]:
    """Map coordinates of a cell centre (north-up geotransform)."""
    gt = geotransform
    return gt[0] + (col + 0.5) * gt[1], gt[3] + (row + 0.5) * gt[5]


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


def title_purpose(context: dict) -> str:
    return (
        "This report documents the hydrological correction of the elevation "
        "model: the depressions found, how each was filled or breached, and "
        "the sinks that remain, so that the corrected DEM can be used with "
        "confidence for flow direction, flow accumulation and catchment "
        "delineation."
    )


def title_details(context: dict) -> list[tuple[str, str]]:
    tool = context.get("tool", REPORT_TITLE)
    version = context.get("version")
    return [
        ("Project number", context.get("project_number") or "-"),
        ("Project / site", context.get("project_name") or "-"),
        ("Prepared by", context.get("prepared_by") or "Mayim Consulting"),
        ("Tool and version", f"{tool}, version {version}" if version else tool),
        ("Software", context.get("software") or "-"),
        ("Report date", context.get("report_date") or date.today().isoformat()),
        ("Document status", context.get("status") or "Draft"),
    ]


def summary_rows(st: dict) -> list[list[str]]:
    return [
        ["Valid DEM cells", num(st.get("valid_cells"), "{:d}")],
        ["Depressions found", num(st.get("depressions_found"), "{:d}")],
        [
            "Depressions resolved (filled / breached)",
            f"{num(st.get('resolution_units'), '{:d}')} "
            f"({num(st.get('units_filled'), '{:d}')} / "
            f"{num(st.get('units_breached'), '{:d}')})",
        ],
        ["Depressions kept", num(st.get("kept_depressions"), "{:d}")],
        [
            "Cells changed (% of valid cells)",
            f"{num(st.get('cells_changed'), '{:d}')} "
            f"({num(st.get('pct_cells_changed'), '{:.2f}')} %)",
        ],
        ["Largest raise (m)", num(st.get("max_raise_m"), "{:.3f}")],
        ["Largest lowering (m)", num(st.get("max_lowering_m"), "{:.3f}")],
        ["Volume added by raising (m³)", num(st.get("fill_volume_m3"), "{:.1f}")],
        ["Volume removed by lowering (m³)", num(st.get("cut_volume_m3"), "{:.1f}")],
        [
            "Sink clusters remaining (not confirmed)",
            f"{num(st.get('remaining_sink_clusters'), '{:d}')} "
            f"({num(st.get('unresolved_sink_clusters'), '{:d}')})",
        ],
    ]


RESULT_ROWS = [  # (stats key, label, format)
    ("valid_cells", "Valid DEM cells", "{:d}"),
    ("depressions_found", "Depressions found (leaves holding water)", "{:d}"),
    ("resolution_units", "Resolution units (merged where adjacent)", "{:d}"),
    ("units_filled", "Units filled", "{:d}"),
    ("units_breached", "Units breached", "{:d}"),
    ("kept_depressions", "Depressions kept", "{:d}"),
    (
        "pits_after_fill_breach",
        "Cells with no downhill neighbour after fill / breach",
        "{:d}",
    ),
    ("cells_raised_by_enforcement", "Cells raised by drainage enforcement", "{:d}"),
    ("remaining_sink_clusters", "Sink clusters remaining", "{:d}"),
    ("unresolved_sink_clusters", "Of which not confirmed as kept", "{:d}"),
    ("cells_changed", "Cells changed", "{:d}"),
    ("pct_cells_changed", "Cells changed (% of valid cells)", "{:.2f}"),
    ("max_raise_m", "Largest raise (m)", "{:.3f}"),
    ("max_lowering_m", "Largest lowering (m)", "{:.3f}"),
    (
        "mean_abs_change_changed_cells_m",
        "Mean absolute change of changed cells (m)",
        "{:.3f}",
    ),
    ("fill_volume_m3", "Volume added by raising (m³)", "{:.1f}"),
    ("cut_volume_m3", "Volume removed by lowering (m³)", "{:.1f}"),
]


def method_steps(result: ConditioningResult) -> list[str]:
    s = result.settings
    steps = [
        "Depressions were found with a depression hierarchy (Barnes et al., "
        "2020): every local minimum (a leaf) holds water up to its spill "
        "elevation, and adjacent leaves that fill up merge into larger "
        "meta-depressions.",
    ]
    if s.mode == MODE_CONTINUOUS:
        steps.append(
            "Continuous drainage: every depression was resolved except sinks "
            "the user confirmed as genuine (pans, regional sinks, lakes), "
            "supplied as polygons or points."
        )
    else:
        steps.append(
            "Preserve mode: depressions that persist under simulated DEM "
            "vertical error were classified Likely Natural (Lindsay & Creed, "
            "2006) and left in the surface; Likely Artificial depressions"
            + (" and Uncertain ones" if s.resolve_uncertain else "")
            + " were resolved."
        )
    if result.stats.get("classified"):
        steps.append(
            "Classification: the DEM was perturbed with spatially correlated "
            "random error of the source's vertical accuracy in a Monte Carlo "
            "simulation, and each depression's persistence across the "
            "realisations gave its confidence of being natural (Lindsay & "
            "Creed, 2006)."
        )
    steps += [
        "Each depression to be resolved was either filled to its spill "
        "elevation or breached along a least-cost path, whichever changed "
        "the DEM less in total absolute elevation (impact reduction approach; "
        "Lindsay & Creed, 2005; Lindsay, 2016).",
        "A drainage-enforcement pass (epsilon Priority-Flood; Barnes et al., "
        "2014) raised any cell left without a downhill neighbour by the "
        "smallest step that makes it drain, so flat areas also route flow "
        "(Garbrecht & Martz, 1997).",
        "Cells that still have no downhill neighbour were grouped into sink "
        "clusters and reported as confirmed (kept) or unresolved.",
    ]
    return steps


def inputs_rows(result: ConditioningResult, ctx: dict) -> list[list[str]]:
    s = result.settings
    classified = bool(result.stats.get("classified"))
    rows = [
        ["Project number", text(ctx.get("project_number"))],
        ["Project / site", text(ctx.get("project_name"))],
        ["Input DEM", text(ctx.get("dem_path"))],
        ["Coordinate reference system", text(ctx.get("crs"))],
        ["Cell size (m)", text(ctx.get("cell_size"))],
        ["Grid (columns x rows)", text(ctx.get("grid"))],
        ["DEM source", text(ctx.get("dem_source"))],
        [
            "Mode",
            (
                "Continuous drainage"
                if s.mode == MODE_CONTINUOUS
                else "Preserve Likely Natural depressions"
            ),
        ],
        ["Classification run", "yes" if classified else "no"],
        ["Stream centrelines", text(ctx.get("stream_layer") or "no")],
        ["Confirmed sinks layer", text(ctx.get("keep_layer") or "no")],
        ["Classification table reused", text(ctx.get("review_table") or "no")],
        ["Point search radius for confirmed sinks (cells)", f"{s.keep_search_cells}"],
        ["Maximum breach length", _limit(s.max_breach_length, "cells")],
        ["Maximum breach depth", _limit(s.max_breach_depth, "m")],
        ["Drainage enforcement", "yes" if s.enforce else "no"],
    ]
    if classified:
        rows += [
            ["Monte Carlo iterations", f"{s.n_iterations}"],
            ["Random seed", f"{s.seed}"],
        ]
    rows += [
        ["Study area on the layout map", text(ctx.get("study_area") or "no")],
        ["Contours on the layout map", contour_text(ctx.get("contour_interval"))],
        ["Run", text(ctx.get("run_time"))],
    ]
    return rows


def parameter_rows(result: ConditioningResult, ctx: dict) -> list[list[str]]:
    s = result.settings
    classified = bool(result.stats.get("classified"))
    cell_area = ctx.get("cell_area_m2")
    lo, hi = s.confidence_thresholds
    return [
        [
            "Cell area, Ac (m²)",
            num(cell_area, "{:.3f}"),
            "Area of one DEM cell; used for depression areas and volumes.",
        ],
        [
            "Spill elevation, zs (m)",
            "-",
            "Elevation at which a depression overflows into a neighbouring "
            "one (Barnes et al., 2020).",
        ],
        [
            "Depression depth, d (m)",
            "-",
            "d = zs - z, for each cell below the spill elevation.",
        ],
        [
            "Depression volume, V (m³)",
            "-",
            "V = Σ (zs - zi) x Ac over the depression's footprint cells.",
        ],
        [
            "Correction impact, I (m)",
            "-",
            "I = Σ |z'i - zi| over the changed cells; the method (fill or "
            "breach) with the smaller I is used (Lindsay & Creed, 2005).",
        ],
        [
            "Maximum breach length, Lmax (cells)",
            text(s.max_breach_length or "unlimited"),
            "Longest breach channel allowed; a longer breach falls back to "
            "filling (Lindsay, 2016).",
        ],
        [
            "Maximum breach depth, Dmax (m)",
            text(s.max_breach_depth or "unlimited"),
            "Deepest cut allowed along a breach channel (Lindsay, 2016).",
        ],
        [
            "Vertical error, σz (m)",
            num(s.vertical_error_std, "{:g}"),
            "Standard deviation of the DEM source's vertical error; scales "
            "the simulated error fields (Lindsay & Creed, 2006).",
        ],
        [
            "Monte Carlo iterations, n (-)",
            num(s.n_iterations, "{:d}") if classified else "-",
            "Number of perturbed DEM realisations (Lindsay & Creed, 2006).",
        ],
        [
            "Error correlation length, L (cells)",
            num(s.correlation_length_cells, "{:g}") if classified else "-",
            "Spatial correlation length of the simulated error fields.",
        ],
        [
            "Persistence probability, p (-)",
            "-",
            "Share of realisations in which a depression persists; the "
            "basis of the confidence of being natural (Lindsay & Creed, 2006).",
        ],
        [
            "Confidence thresholds, ca / cn (-)",
            f"{lo:g} / {hi:g}" if classified else "-",
            "Below ca a depression is Likely Artificial, above cn Likely "
            "Natural; in between Uncertain - Review.",
        ],
    ]


def sink_rows(result: ConditioningResult, gt) -> tuple[list[str], list, list]:
    if gt is not None:
        header = [
            "Id",
            "Kind",
            "Reason",
            "Cells",
            "Easting (m)",
            "Northing (m)",
            "Elevation (m)",
        ]
        widths = [1.0, 2.0, 5.5, 1.3, 2.3, 2.4, 2.0]
    else:
        header = ["Id", "Kind", "Reason", "Cells", "Row", "Column", "Elevation (m)"]
        widths = [1.0, 2.0, 6.5, 1.3, 1.8, 1.8, 2.1]
    rows = []
    for k in result.sinks[:60]:
        if gt is not None:
            x, y = cell_xy(gt, k.row, k.col)
            where = [num(x, "{:.1f}"), num(y, "{:.1f}")]
        else:
            where = [f"{k.row}", f"{k.col}"]
        rows.append(
            [f"{k.sink_id}", text(k.kind), text(k.reason), f"{k.cells}"]
            + where
            + [num(k.elevation, "{:.2f}")]
        )
    return header, rows, widths


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_change_histogram(diff: np.ndarray, style):
    """Distribution of the elevation changes (chart house style)."""
    from mayim_tools._common.report_figures import CHART_COLOURS, CHART_INK

    changed = diff[np.isfinite(diff) & (np.abs(diff) > 0)]
    if changed.size == 0:
        return None
    with style.context():
        fig = style.figure(7.0)
        ax = fig.add_subplot(1, 1, 1)
        raised = changed[changed > 0]
        lowered = changed[changed < 0]
        lo, hi = float(changed.min()), float(changed.max())
        if hi - lo < 1e-9:
            lo, hi = lo - 0.05, hi + 0.05
        bins = np.linspace(lo, hi, 41)
        if lowered.size:
            ax.hist(
                lowered,
                bins=bins,
                color=CHART_COLOURS[4],
                edgecolor=CHART_INK,
                linewidth=0.75,
                label="Lowered (breaching)",
            )
        if raised.size:
            ax.hist(
                raised,
                bins=bins,
                color=CHART_COLOURS[0],
                edgecolor=CHART_INK,
                linewidth=0.75,
                label="Raised (filling and drainage enforcement)",
            )
        ax.set_xlabel("Elevation change, corrected minus original (m)")
        ax.set_ylabel("Number of cells")
        ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        style.axes(ax)
        style.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)
        return style.png(fig)


def figure_change_simple(result: ConditioningResult, diff: np.ndarray, gt, style):
    """Fallback layout map (no QGIS): the elevation change classes and the
    remaining sinks in the house map frame, without a background."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from mayim_tools._common.report_figures import CHART_INK
    from mayim_tools._common.report_maps import north_arrow, scale_bar

    rows, cols = diff.shape
    if gt is None:
        gt = (0.0, 1.0, 0.0, float(rows), 0.0, -1.0)
    xmin, xmax = gt[0], gt[0] + cols * gt[1]
    ymax, ymin = gt[3], gt[3] + rows * gt[5]
    classes = [c for c in CHANGE_CLASSES if c[1] is not None]
    idx = np.full(diff.shape, np.nan)
    lower = -math.inf
    k = 0
    for upper, colour, _label in CHANGE_CLASSES:
        if colour is not None:
            idx[(diff > lower) & (diff <= upper)] = k
            k += 1
        lower = upper
    cmap = ListedColormap([c[1] for c in classes])
    present = set(change_classes_present(diff))
    handles = [
        Patch(facecolor=c[1], edgecolor=CHART_INK, linewidth=0.5, label=c[2])
        for c in classes
        if c[2] in present
    ]
    width_cm = 14.0
    height_cm = min(max(width_cm * (ymax - ymin) / max(xmax - xmin, 1e-9), 6.0), 18.0)
    with style.context():
        fig = style.figure(height_cm + 2.5)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(
            idx,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(classes) - 0.5,
            extent=(xmin, xmax, ymin, ymax),
            interpolation="nearest",
            origin="upper",
        )
        for kind, (marker, colour, label) in SINK_STYLE.items():
            pts = [cell_xy(gt, s.row, s.col) for s in result.sinks if s.kind == kind]
            if pts:
                ax.scatter(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker=marker,
                    s=22,
                    color=colour,
                    edgecolor=CHART_INK,
                    linewidth=0.5,
                    zorder=5,
                )
                handles.append(
                    Line2D(
                        [],
                        [],
                        marker=marker,
                        linestyle="none",
                        markerfacecolor=colour,
                        markeredgecolor=CHART_INK,
                        label=label,
                    )
                )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        scale_bar(ax, (xmin, ymin, xmax, ymax))
        north_arrow(ax)
        style.axes(ax, grid=False)
        if handles:
            style.legend(
                ax,
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncols=3,
            )
        return style.png(fig)


def figure_layout(result, diff, gt, style, plan_map, messages):
    """(PNG stream, used the OSM background): the QGIS-rendered layout map
    when ``plan_map`` is given, else the simplified figure."""
    from mayim_tools._common.report_maps import (
        PLAN_DPI,
        PLAN_PLOT_WIDTH_CM,
        figure_rendered_map,
    )

    if plan_map is not None:
        try:
            pm = plan_map(REPORT_TITLE, PLAN_PLOT_WIDTH_CM, PLAN_DPI)
        except Exception as e:  # the report must still be written
            pm = None
            messages.append(
                f"The QGIS layout map could not be rendered ({e}); a simplified "
                "map of the elevation changes was drawn instead."
            )
        if pm is not None:
            return figure_rendered_map(pm, style), bool(pm.attribution)
    return figure_change_simple(result, diff, gt, style), False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report_docx(
    path: str | Path,
    result: ConditioningResult,
    context: dict | None = None,
    plan_map: PlanMapFn | None = None,
    include_figures: bool = True,
) -> list[str]:
    """Writes the Word report and returns messages for the log (font
    substitutions, map problems).

    ``context`` describes the run: tool, version, software, project_number,
    project_name, status, report_date, run_time, dem_path, crs, cell_size,
    cell_area_m2, grid, dem_source, stream_layer, keep_layer, review_table,
    study_area, contour_interval, nodata, geotransform (north-up GDAL geotransform,
    for sink coordinates and the fallback map), outputs [(label, location)],
    warnings, and map_messages (filled by ``plan_map`` while rendering)."""
    from mayim_tools._common.docx_report import Doc
    from mayim_tools._common.report_maps import OSM_LICENCE_NOTE

    ctx = context or {}
    s = result.settings
    st = result.stats
    gt = ctx.get("geotransform")
    messages: list[str] = []
    style = None
    if include_figures:
        try:
            from mayim_tools._common.report_figures import ChartStyle

            style = ChartStyle()
            messages.extend(style.warnings)
        except ImportError:
            include_figures = False
    diff = elevation_change(result, ctx.get("nodata"))

    d = Doc(house_style=True)
    d.title_page(
        REPORT_TITLE,
        purpose=title_purpose(ctx),
        details=title_details(ctx),
        subtitle=ctx.get("project_name") or None,
    )

    # 1. Summary --------------------------------------------------------------
    d.heading("1. Summary", 2)
    mode_text = (
        "every depression was resolved except confirmed sinks"
        if s.mode == MODE_CONTINUOUS
        else "depressions classified Likely Natural were preserved"
    )
    d.para(
        f"The DEM was corrected for hydrological use with the {REPORT_TITLE} "
        f"tool (version {ctx.get('version', '-')}, Mayim Tools): {mode_text}. "
        f"Of {num(st.get('depressions_found'), '{:d}')} depressions found, "
        f"{num(st.get('units_filled'), '{:d}')} resolution units were filled "
        f"and {num(st.get('units_breached'), '{:d}')} breached, changing "
        f"{num(st.get('pct_cells_changed'), '{:.2f}')} % of the valid cells."
    )
    n_unres = st.get("unresolved_sink_clusters") or 0
    if n_unres:
        d.para(
            f"{n_unres} sink cluster(s) remain that were not confirmed as "
            "genuine and should be investigated (Section 5.3)."
        )
    else:
        d.para(
            "Every valid cell of the corrected DEM drains to a lower neighbour, "
            "the DEM edge or a confirmed sink."
        )
    d.caption("Table", "Summary of the corrections")
    d.table(["Item", "Value"], summary_rows(st), widths_cm=[10.5, 6.0])

    # 2. Tool and method ------------------------------------------------------
    d.heading("2. Tool and method", 2)
    d.para(
        "Raster flow routing needs every cell to drain to a lower neighbour. "
        "Real elevation models contain depressions, most of them artefacts of "
        "vertical error, resolution and surface features. The processing "
        "steps were:"
    )
    for step in method_steps(result):
        d.numbered(step)

    # 3. Inputs and settings --------------------------------------------------
    d.heading("3. Inputs and settings", 2)
    d.caption("Table", "Inputs and processing settings")
    d.table(["Item", "Value"], inputs_rows(result, ctx), widths_cm=[6.0, 10.5])

    # 4. Parameters and calculations -------------------------------------------
    d.heading("4. Parameters and calculations", 2)
    d.caption("Table", "Parameters, values and definitions")
    d.table(
        ["Parameter (symbol, unit)", "Value", "Definition / equation (source)"],
        parameter_rows(result, ctx),
        widths_cm=[6.2, 2.8, 7.5],
    )

    # 5. Results ----------------------------------------------------------------
    d.heading("5. Results", 2)
    d.heading("5.1 Corrections to the DEM", 3)
    d.caption("Table", "Results of the correction")
    d.table(
        ["Item", "Value"],
        [[label, num(st.get(key), fmt)] for key, label, fmt in RESULT_ROWS],
        widths_cm=[10.5, 6.0],
    )
    used_osm = False
    if include_figures and style is not None:
        stream, used_osm = figure_layout(result, diff, gt, style, plan_map, messages)
        d.picture(
            stream,
            "Layout of the elevation changes made to the DEM (corrected minus "
            "original) and the remaining sinks"
            + (" (background: OpenStreetMap)" if used_osm else ""),
        )
        hist = figure_change_histogram(diff, style)
        if hist is not None:
            d.picture(hist, "Distribution of the elevation changes made to the DEM")

    d.heading("5.2 Confirmed sinks supplied", 3)
    if result.keep_matches:
        d.para("Each supplied feature and what it matched; nothing is ignored.")
        d.caption("Table", "Confirmed sink features and their matches")
        d.table(
            ["Feature", "Geometry", "Result", "Depressions kept", "Detail"],
            [
                [
                    text(m.label),
                    text(m.kind),
                    text(m.status),
                    text(" ".join(str(i) for i in m.leaf_ids)),
                    text(m.detail),
                ]
                for m in result.keep_matches[:60]
            ],
            widths_cm=[3.0, 2.0, 3.5, 3.0, 5.0],
        )
    else:
        d.para("No confirmed sinks were supplied.")

    d.heading("5.3 Sinks remaining in the corrected DEM", 3)
    if result.sinks:
        header, rows, widths = sink_rows(result, gt)
        d.caption("Table", "Remaining sink clusters (lowest cell of each)")
        d.table(header, rows, widths_cm=widths)
        if len(result.sinks) > 60:
            d.para(
                f"The first 60 of {len(result.sinks)} clusters are listed; all "
                "are in the remaining sinks layer and the results CSV."
            )
    else:
        d.para("None. Every valid cell drains to a lower neighbour or the edge.")

    if result.kept:
        d.heading("5.4 Depressions left in the DEM", 3)
        d.caption("Table", "Depressions kept in the corrected DEM")
        d.table(
            [
                "Id",
                "Reason",
                "Area (m²)",
                "Max depth (m)",
                "Volume (m³)",
                "Class",
            ],
            [
                [
                    f"{k.leaf_id}",
                    text(k.reason),
                    num(k.area_m2, "{:.1f}"),
                    num(k.max_depth_m, "{:.2f}"),
                    num(k.volume_m3, "{:.1f}"),
                    text(k.classification),
                ]
                for k in result.kept[:MAX_TABLE_ROWS]
            ],
            widths_cm=[1.2, 5.0, 2.4, 2.3, 2.4, 3.2],
        )

    units = sorted(result.units, key=lambda u: -u.volume_m3)
    d.heading(f"5.{5 if result.kept else 4} Largest corrections", 3)
    if units:
        d.caption(
            "Table",
            f"Largest {min(len(units), MAX_TABLE_ROWS)} corrections by volume",
        )
        d.table(
            [
                "Unit",
                "Method",
                "Area (m²)",
                "Max depth (m)",
                "Volume (m³)",
                "Max change (m)",
                "Class",
            ],
            [
                [
                    f"{u.unit_id}",
                    text(u.method),
                    num(u.area_m2, "{:.1f}"),
                    num(u.max_depth_m, "{:.2f}"),
                    num(u.volume_m3, "{:.1f}"),
                    num(u.max_change_m, "{:.2f}"),
                    text(u.classification),
                ]
                for u in units[:MAX_TABLE_ROWS]
            ],
            widths_cm=[1.3, 1.7, 2.4, 2.3, 2.4, 2.4, 4.0],
        )
        d.para("All corrections are listed in the results CSV (section 7).")
    else:
        d.para("No depressions needed correcting.")

    # 6. Outputs ------------------------------------------------------------------
    section = 6
    outputs = ctx.get("outputs") or []
    if outputs:
        d.heading(f"{section}. Outputs", 2)
        d.caption("Table", "Outputs produced by the tool")
        d.table(
            ["Output", "Location"], [list(o) for o in outputs], widths_cm=[5.0, 11.5]
        )
        section += 1

    # 7. Notes and limitations -------------------------------------------------------
    d.heading(f"{section}. Notes and limitations", 2)
    notes = [
        "A DEM is often a surface model: canopy and structures are included, "
        "so some depressions are not terrain features.",
        "Corrections are bounded by the vertical error of the source data; "
        "changes larger than that error merit a visual check.",
        "Classification from the DEM alone cannot separate natural pans from "
        "dams, borrow pits or quarries. Confirm genuine sinks with imagery or "
        "field knowledge and supply them as a confirmed sinks layer.",
        "Embankments or ridges with no depression behind them are not "
        "detected or removed.",
    ]
    if used_osm:
        notes.append(OSM_LICENCE_NOTE)
    for note in notes:
        d.bullet(note)
    log = (
        list(result.warnings)
        + list(ctx.get("warnings") or [])
        + list(ctx.get("map_messages") or [])
        + messages
    )
    if log:
        d.para("Processing messages recorded during this run:")
        for message in dict.fromkeys(log):
            d.bullet(message)
    section += 1

    # 8. References ---------------------------------------------------------------------
    d.heading(f"{section}. References", 2)
    for ref in sorted(REFERENCES):
        d.para(ref)

    d.doc.save(str(path))
    return messages
