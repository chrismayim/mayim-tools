"""Word (.docx) report for the DEM depression stage-storage tool.

Zero QGIS dependency. Needs python-docx and matplotlib (both imported
lazily, so the rest of the tool works without them). Figures use
matplotlib's object API (no pyplot), which is safe on QGIS's Processing
worker thread. Document and chart helpers are shared with the Catchment
Delineation report (hydrology/_common/docx_report.py).
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence

import numpy as np

from mayim_tools.hydrology._common.docx_report import (
    C_AQUA,
    C_BLUE,
    C_INK,
    C_INK_2,
    C_ORANGE,
    Doc,
    format_value,
    new_figure,
    png,
    style_axes,
)

from .core import StorageResult

REPORT_TITLE = "DEM Depression Stage-Storage Assessment"
MAX_REPORT_ROWS = 30

# (quantity, symbol (unit), definition / equation)
DEFINITIONS = [
    ("Stage", "h (m)", "Water-surface elevation, in the DEM's vertical datum."),
    ("Floor", "z_0 (m)", "Lowest valid DEM cell inside the storage polygon."),
    ("Depth", "d (m)", "d = h - z_0, depth of water above the floor."),
    ("Wet cell", "-", "Cell inside the polygon with ground below h (and, in connected mode, joined to the floor by cells below h)."),
    ("Cell weight", "w (-)", "Fraction of the cell area inside the polygon (1 for interior cells)."),
    ("Water-surface area", "A(h) (m²)", "A(h) = Σ w_i a over wet cells, with a the cell area."),
    ("Storage volume", "V(h) (m³)", "V(h) = Σ w_i a (h - z_i) over wet cells (prism method, exact for the DEM)."),
    ("Incremental volume", "ΔV (m³)", "Volume added between consecutive stages in the table."),
    ("Spill level", "h_s (m)", "Lowest stage at which the pool reaches the polygon (or DEM) edge: the lowest of the highest points along every path from the floor to the edge."),
    ("Top stage", "h_t (m)", "Upper limit of the table: the spill level, or a specified maximum stage."),
    ("Mean depth", "d_m (m)", "d_m = V(h_t) / A(h_t)."),
    ("Inundated share", "(%)", "A(h_t) as a share of the polygon area on the DEM grid."),
    ("Volume-depth relation", "V = a d^b", "Least-squares fit on log-log axes; b ≈ 1 for vertical sides, ≈ 2 for a wedge or trough, ≈ 3 for a cone or bowl."),
    ("Area-depth relation", "A = c d^n", "Least-squares fit on log-log axes; n = b - 1 for an ideal shape."),
]  # fmt: skip

# (group heading, [(label, attribute key, format, scale)])
RESULT_GROUPS = [
    (
        "Levels",
        [
            ("Floor z_0 (m)", "z_floor", "{:.3f}", 1),
            ("Spill level h_s (m)", "z_spill", "{:.3f}", 1),
            ("Top stage h_t (m)", "z_top", "{:.3f}", 1),
            ("Top stage source", "top_src", "{}", None),
            ("Maximum depth d (m)", "max_depth", "{:.3f}", 1),
            ("Mean depth d_m (m)", "mean_dep", "{:.3f}", 1),
        ],
    ),
    (
        "At the top stage",
        [
            ("Water-surface area (m²)", "area_m2", "{:.0f}", 1),
            ("Water-surface area (ha)", "area_ha", "{:.4f}", 1),
            ("Storage volume (m³)", "vol_m3", "{:.0f}", 1),
            ("Storage volume (ML)", "vol_ML", "{:.2f}", 1),
            ("Polygon area on the grid (ha)", "poly_ha", "{:.4f}", 1),
            ("Inundated share of polygon (%)", "wet_pct", "{:.1f}", 1),
        ],
    ),
    (
        "Location and grid",
        [
            ("Floor E", "x_floor", "{:.1f}", 1),
            ("Floor N", "y_floor", "{:.1f}", 1),
            ("Spill crest E", "x_spill", "{:.1f}", 1),
            ("Spill crest N", "y_spill", "{:.1f}", 1),
            ("DEM cell size (m)", "cell_m", "{:g}", 1),
            ("Stages in the table", "n_stages", "{:d}", None),
            ("Cells counted", "mode", "{}", None),
        ],
    ),
]

REFERENCES = [
    "Barnes, R., Lehman, C. & Mulla, D. (2014). Priority-flood: An optimal "
    "depression-filling and watershed-labeling algorithm for digital "
    "elevation models. Computers & Geosciences, 62, 117-127.",
    "Wang, L. & Liu, H. (2006). An efficient method for identifying and "
    "filling surface depressions in digital elevation models for hydrologic "
    "analysis and modelling. International Journal of Geographical "
    "Information Science, 20(2), 193-213.",
    "US Army Corps of Engineers, Hydrologic Engineering Center. HEC-RAS "
    "Hydraulic Reference Manual (storage areas).",
    "US Army Corps of Engineers, Hydrologic Engineering Center. HEC-HMS "
    "Technical Reference Manual (reservoir routing).",
]

MODE_TEXT = {
    "connected": "connected to the lowest point",
    "all": "all cells below the stage (level pool)",
}


def shape_description(b: float | None) -> str:
    """Plain-language reading of the volume-depth exponent."""
    if b is None:
        return "not determined"
    if b < 1.5:
        return "steep-sided with a broad, flat floor (close to prismatic)"
    if b < 2.5:
        return "wedge- or trough-shaped"
    return "bowl- or cone-shaped, with most of its volume near the top"


def report_rows(table: Sequence[dict], max_rows: int = MAX_REPORT_ROWS):
    """Every k-th stage so the report table stays short; the first and last
    stages are always kept."""
    n = len(table)
    if n <= max_rows:
        return list(table)
    k = math.ceil((n - 1) / (max_rows - 1))
    rows = [table[i] for i in range(0, n - 1, k)]
    rows.append(table[-1])
    return rows


def describe_storage(r: StorageResult) -> str:
    a = r.attributes
    fit = r.detail.get("fit_volume")
    b = None if fit is None else fit["b"]
    parts = [
        f"Storage area {r.storage_id} has its floor at {a['z_floor']:.2f} m.",
    ]
    if a["top_src"] == "spill":
        parts.append(
            f"It fills to its spill level of {a['z_top']:.2f} m, where the pool "
            "first reaches the edge of the storage polygon"
            + (
                f" (spill crest near E {a['x_spill']:.0f}, N {a['y_spill']:.0f})."
                if a.get("x_spill") is not None
                else "."
            )
        )
    else:
        spill = (
            f" The natural spill level is {a['z_spill']:.2f} m."
            if a.get("z_spill") is not None
            else ""
        )
        parts.append(
            f"The table runs to a specified maximum stage of {a['z_top']:.2f} m.{spill}"
        )
    parts.append(
        f"At that level the water is up to {a['max_depth']:.2f} m deep, covers "
        f"{a['area_ha']:.2f} ha ({format_value(a.get('wet_pct'), '{:.0f}', 1)} % "
        f"of the polygon) and stores {a['vol_m3']:.0f} m³ ({a['vol_ML']:.1f} ML), "
        f"a mean depth of {format_value(a.get('mean_dep'), '{:.2f}', 1)} m."
    )
    if b is not None:
        parts.append(
            f"The volume-depth relation V = {fit['a']:.4g} d^{b:.2f} "
            f"(R² = {fit['r2']:.3f}) indicates a depression that is "
            f"{shape_description(b)}."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _arrays(r: StorageResult):
    t = r.table
    h = np.array([row["stage_m"] for row in t], dtype=float)
    d = np.array([row["depth_m"] for row in t], dtype=float)
    area = np.array([row["area_m2"] for row in t], dtype=float)
    vol = np.array([row["volume_m3"] for row in t], dtype=float)
    return h, d, area, vol


def _vol_scale(vol_max: float):
    if vol_max >= 1e6:
        return 1e6, "Storage volume (10⁶ m³)"
    if vol_max >= 1e4:
        return 1e3, "Storage volume (10³ m³)"
    return 1.0, "Storage volume (m³)"


def _area_scale(area_max: float):
    if area_max >= 1e5:
        return 1e4, "Water-surface area (ha)"
    return 1.0, "Water-surface area (m²)"


def figure_plan(r: StorageResult) -> io.BytesIO | None:
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if r.window_gt is None or not np.isfinite(r.depth).any():
        return None
    gt = r.window_gt
    rows, cols = r.depth.shape
    extent = [gt[0], gt[0] + cols * gt[1], gt[3] + rows * gt[5], gt[3]]
    fig = new_figure(12.0)
    ax = fig.add_subplot(1, 1, 1)
    depth = np.ma.masked_invalid(r.depth)
    im = ax.imshow(depth, extent=extent, cmap="Blues", vmin=0, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Depth at {r.top_stage:.2f} m (m)", fontsize=8, color=C_INK)
    cbar.ax.tick_params(labelsize=7, colors=C_INK_2)
    for poly in r.polygons:
        verts, codes = [], []
        for ring in poly:
            if len(ring) < 3:
                continue
            verts.extend(ring)
            codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ring) - 2))
            codes.append(Path.CLOSEPOLY)
        if verts:
            ax.add_patch(
                PathPatch(
                    Path(verts, codes), facecolor="none", edgecolor=C_INK, linewidth=0.8
                )
            )
    ax.plot([], [], color=C_INK, linewidth=0.8, label="Storage polygon")
    if r.contours:
        for _, _, lines in r.contours:
            for line in lines:
                lx, ly = zip(*line, strict=True)
                ax.plot(lx, ly, color=C_INK_2, linewidth=0.4, alpha=0.8)
        ax.plot([], [], color=C_INK_2, linewidth=0.4, label="Water-edge contours")
    ax.plot(
        *r.floor_xy,
        marker="v",
        markersize=7,
        color=C_INK,
        linestyle="none",
        label=f"Floor {r.floor_z:.2f} m",
    )
    if r.spill_xy is not None:
        ax.plot(
            *r.spill_xy,
            marker="X",
            markersize=8,
            color=C_ORANGE,
            linestyle="none",
            label=f"Spill crest {r.spill_z:.2f} m",
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    style_axes(ax)
    ax.grid(False)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(fontsize=7, frameon=True, framealpha=0.85, loc="best")
    return png(fig)


def figure_elevation_area_capacity(r: StorageResult) -> io.BytesIO:
    """Classic reservoir chart: stage on y, volume on the bottom x axis and
    area on a reversed top x axis."""
    h, _, area, vol = _arrays(r)
    vs, vlabel = _vol_scale(vol.max())
    as_, alabel = _area_scale(area.max())
    fig = new_figure(9.0)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(vol / vs, h, color=C_BLUE, linewidth=1.8, label="Storage volume")
    ax.set_xlabel(vlabel)
    ax.set_ylabel("Stage (m)")
    ax.set_xlim(left=0)
    style_axes(ax)
    top = ax.twiny()
    top.plot(area / as_, h, color=C_ORANGE, linewidth=1.8, label="Water-surface area")
    top.set_xlim(left=0)
    top.invert_xaxis()
    top.set_xlabel(alabel, fontsize=8, color=C_INK)
    top.tick_params(colors=C_INK_2, labelsize=7, width=0.6)
    for side in ("right", "left", "bottom"):
        top.spines[side].set_visible(False)
    top.spines["top"].set_color(C_INK_2)
    top.spines["top"].set_linewidth(0.6)
    handles = ax.get_legend_handles_labels()[0] + top.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower center")
    ax.set_ylim(h.min(), h.max())
    return png(fig)


def _xy_figure(x, y, xlabel, ylabel, color, fill=False, fit=None, fit_label=None):
    fig = new_figure(7.5)
    ax = fig.add_subplot(1, 1, 1)
    if fill:
        ax.fill_between(x, y, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, y, color=color, linewidth=1.8, label="DEM")
    if fit is not None:
        xs = np.linspace(max(x.min(), 1e-6), x.max(), 100)
        ax.plot(
            xs,
            fit(xs),
            color=C_INK_2,
            linewidth=1.0,
            linestyle="--",
            label=fit_label,
        )
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=min(0.0, float(x.min())))
    ax.set_ylim(bottom=0)
    style_axes(ax)
    return png(fig)


def figure_stage_storage(r: StorageResult) -> io.BytesIO:
    h, _, _, vol = _arrays(r)
    vs, vlabel = _vol_scale(vol.max())
    fig = new_figure(7.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(vol / vs, h, color=C_BLUE, linewidth=1.8)
    ax.set_xlabel(vlabel)
    ax.set_ylabel("Stage (m)")
    ax.set_xlim(left=0)
    ax.set_ylim(h.min(), h.max())
    style_axes(ax)
    return png(fig)


def figure_stage_area(r: StorageResult) -> io.BytesIO:
    h, _, area, _ = _arrays(r)
    as_, alabel = _area_scale(area.max())
    fig = new_figure(7.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(area / as_, h, color=C_ORANGE, linewidth=1.8)
    ax.set_xlabel(alabel)
    ax.set_ylabel("Stage (m)")
    ax.set_xlim(left=0)
    ax.set_ylim(h.min(), h.max())
    style_axes(ax)
    return png(fig)


def figure_depth_area(r: StorageResult) -> io.BytesIO:
    _, d, area, _ = _arrays(r)
    as_, alabel = _area_scale(area.max())
    fit = r.detail.get("fit_area")
    fn = label = None
    if fit is not None:

        def fn(x):
            return fit["a"] * x ** fit["b"] / as_

        label = f"A = {fit['a']:.4g} d^{fit['b']:.2f} (R² = {fit['r2']:.3f})"
    return _xy_figure(d, area / as_, "Depth d (m)", alabel, C_ORANGE, True, fn, label)


def figure_depth_volume(r: StorageResult) -> io.BytesIO:
    _, d, _, vol = _arrays(r)
    vs, vlabel = _vol_scale(vol.max())
    fit = r.detail.get("fit_volume")
    fn = label = None
    if fit is not None:

        def fn(x):
            return fit["a"] * x ** fit["b"] / vs

        label = f"V = {fit['a']:.4g} d^{fit['b']:.2f} (R² = {fit['r2']:.3f})"
    return _xy_figure(d, vol / vs, "Depth d (m)", vlabel, C_BLUE, True, fn, label)


def figure_depth_distribution(r: StorageResult) -> io.BytesIO | None:
    bins = r.detail.get("depth_hist") or []
    if not bins:
        return None
    fig = new_figure(7.0)
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(
        [b["from_m"] for b in bins],
        [b["pct"] for b in bins],
        width=[b["to_m"] - b["from_m"] for b in bins],
        align="edge",
        color=C_AQUA,
        edgecolor="white",
        linewidth=1.0,
    )
    ax.set_xlabel(f"Depth at {r.top_stage:.2f} m (m)")
    ax.set_ylabel("Share of water-surface area (%)")
    style_axes(ax)
    return png(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report_docx(
    path: str,
    results: Sequence[StorageResult],
    context: dict,
    include_figures: bool = True,
) -> None:
    """Writes the report. ``context`` describes the run (see the QGIS
    wrapper): tool/version/run time, inputs, settings, outputs and
    processing warnings."""
    d = Doc()
    ctx = context or {}
    if include_figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            include_figures = False
    dem = ctx.get("dem", {})
    mode = ctx.get("mode", "connected")
    n = len(results)

    d.heading(REPORT_TITLE, 1)

    # 1. Summary ---------------------------------------------------------------
    d.heading("1. Summary", 2)
    d.para(
        f"This report documents the stage-storage (elevation-area-volume) "
        f"relationship of {n} storage area{'s' if n != 1 else ''} read from a "
        f"digital elevation model (DEM) with a {dem.get('cell_x', 0):g} m x "
        f"{dem.get('cell_y', 0):g} m cell size, using the "
        f"{ctx.get('tool', 'DEM depression stage-storage')} tool (Mayim Tools, "
        f"version {ctx.get('version', '-')}). Cells were counted when "
        f"{MODE_TEXT.get(mode, mode)}. For each storage area it gives the "
        "levels, the stage-area-volume table, the depth distribution and the "
        "stage-storage, stage-area, depth-area and depth-volume curves."
    )
    d.caption("Table", "Summary of storage areas at the top stage")
    d.table(
        [
            "Storage",
            "Floor (m)",
            "Top (m)",
            "Max depth (m)",
            "Area (ha)",
            "Volume (m³)",
            "Volume (ML)",
        ],
        [
            [
                r.storage_id,
                format_value(r.attributes["z_floor"], "{:.2f}", 1),
                format_value(r.attributes["z_top"], "{:.2f}", 1)
                + (" (spill)" if r.attributes["top_src"] == "spill" else ""),
                format_value(r.attributes["max_depth"], "{:.2f}", 1),
                format_value(r.attributes["area_ha"], "{:.3f}", 1),
                format_value(r.attributes["vol_m3"], "{:.0f}", 1),
                format_value(r.attributes["vol_ML"], "{:.1f}", 1),
            ]
            for r in results
        ],
    )

    # 2. Tool and method -------------------------------------------------------
    d.heading("2. Tool and method", 2)
    d.para(
        "Each DEM cell is treated as a flat-topped prism, and the storage is "
        "summed cell by cell. The processing steps were:"
    )
    coverage = ctx.get("coverage", "")
    for step in [
        "The storage polygon was converted to cell weights: "
        + (
            "the fraction of each cell inside the polygon, from an 8 x 8 "
            "sub-cell sample, so cells cut by the polygon edge count in part."
            if "Fractional" in coverage
            else "1 for cells whose centre lies inside the polygon, 0 otherwise."
        ),
        "The floor (lowest valid cell inside the polygon) was found, and a "
        "priority flood (Barnes et al. 2014) from the floor gave every cell "
        "the water level at which it first joins the pool - the lowest of the "
        "highest points along all paths to it (eight-direction connectivity).",
        "The spill level was taken as the lowest level at which the pool "
        "reaches the polygon or DEM edge, and the spill crest as the cell that "
        "controls it. "
        + (
            "The table runs to this level."
            if all(r.top_source == "spill" for r in results)
            else "Where a maximum stage was specified, the table runs to that "
            "level instead; above the spill level the polygon edge then acts "
            "as a vertical wall."
        ),
        "At every stage h, the wet cells were "
        + (
            "those joined to the floor below h, so hollows behind a ridge fill "
            "only once the pool overtops the ridge."
            if mode == "connected"
            else "all cells with ground below h, connected to the floor or not "
            "(the HEC-RAS storage-area convention)."
        )
        + " The water-surface area is the sum of their weighted areas and the "
        "volume the sum of their weighted areas times the water depth above "
        "each cell.",
        f"Stages were tabulated from the floor at {ctx.get('interval', '-')} m "
        "intervals (on round values) and at the top stage. Area and volume are "
        "exact for the DEM at every stage; no average-end-area or conic "
        "approximation is used.",
        "Volume and area were fitted against depth as power laws on log-log "
        "axes to describe the shape of the depression.",
    ]:
        d.numbered(step)

    # 3. Inputs -----------------------------------------------------------------
    d.heading("3. Inputs and settings", 2)
    d.caption("Table", "Inputs and processing settings")
    d.table(
        ["Item", "Value"],
        [
            ["DEM", dem.get("path", "-")],
            ["Coordinate reference system", dem.get("crs", "-")],
            [
                "DEM cell size",
                f"{dem.get('cell_x', 0):g} m x {dem.get('cell_y', 0):g} m",
            ],
            ["Storage polygon layer", ctx.get("storage", {}).get("layer", "-")],
            ["Storage features", str(ctx.get("storage", {}).get("n_features", "-"))],
            ["Storage name field", ctx.get("storage", {}).get("id_field") or "-"],
            ["Stage interval", f"{ctx.get('interval', '-')} m"],
            ["Maximum stage", ctx.get("max_stage") or "Spill level"],
            ["Cells counted", MODE_TEXT.get(mode, mode)],
            ["Polygon edge cells", coverage or "-"],
            ["Run", ctx.get("run_time", "-")],
        ],
        widths_cm=[5.0, 11.0],
    )
    d.para(
        "The DEM must be the unconditioned (unfilled) surface and in a "
        "projected coordinate system in metres; filling or breaching removes "
        "the depressions this tool measures."
    )

    # 4. Definitions ------------------------------------------------------------
    d.heading("4. Definitions", 2)
    d.caption("Table", "Quantities and equations")
    d.table(
        ["Quantity", "Symbol (unit)", "Definition / equation"],
        [list(row) for row in DEFINITIONS],
        widths_cm=[3.6, 2.8, 9.6],
    )

    # 5. Results -----------------------------------------------------------------
    d.heading("5. Storage area results", 2)
    for i, r in enumerate(results, start=1):
        a = r.attributes
        d.heading(f"5.{i} Storage area {r.storage_id}", 3)
        d.para(describe_storage(r))

        rows, groups = [], []
        for title, items in RESULT_GROUPS:
            groups.append(len(rows))
            rows.append([title, ""])
            for label, key, fmt, scale in items:
                rows.append([label, format_value(a.get(key), fmt, scale)])
        for key, label in (("fit_volume", "Volume-depth"), ("fit_area", "Area-depth")):
            fit = r.detail.get(key)
            if key == "fit_volume":
                groups.append(len(rows))
                rows.append(["Shape", ""])
            sym = "V" if key == "fit_volume" else "A"
            rows.append(
                [
                    f"{label} relation",
                    (
                        "-"
                        if fit is None
                        else f"{sym} = {fit['a']:.4g} d^{fit['b']:.3f} "
                        f"(R² = {fit['r2']:.3f})"
                    ),
                ]
            )
        d.caption("Table", f"Key results, storage area {r.storage_id}")
        d.table(["Quantity", "Value"], rows, widths_cm=[9.0, 7.0], group_rows=groups)

        shown = report_rows(r.table)
        d.caption(
            "Table",
            f"Stage-area-volume table, storage area {r.storage_id}"
            + (
                f" ({len(shown)} of {len(r.table)} stages shown; the CSV holds "
                "all of them)"
                if len(shown) < len(r.table)
                else ""
            ),
        )
        d.table(
            [
                "Stage (m)",
                "Depth (m)",
                "Area (m²)",
                "Area (ha)",
                "Volume (m³)",
                "Volume (ML)",
            ],
            [
                [
                    f"{row['stage_m']:.2f}",
                    f"{row['depth_m']:.2f}",
                    f"{row['area_m2']:.0f}",
                    f"{row['area_ha']:.4f}",
                    f"{row['volume_m3']:.0f}",
                    f"{row['volume_ML']:.2f}",
                ]
                for row in shown
            ],
            widths_cm=[2.5, 2.3, 2.9, 2.5, 3.2, 2.6],
        )

        bins = r.detail.get("depth_hist") or []
        if bins:
            d.caption(
                "Table",
                f"Water-surface area by depth class at {r.top_stage:.2f} m, "
                f"storage area {r.storage_id}",
            )
            d.table(
                ["Depth class (m)", "Area (m²)", "Share (%)"],
                [
                    [
                        f"{b['from_m']:.2f} - {b['to_m']:.2f}",
                        f"{b['area_m2']:.0f}",
                        f"{b['pct']:.1f}",
                    ]
                    for b in bins
                ],
                widths_cm=[6.0, 5.0, 5.0],
            )

        if include_figures:
            sid = r.storage_id
            for maker, cap in (
                (
                    figure_plan,
                    f"depth at the top stage ({r.top_stage:.2f} m), water-edge "
                    "contours, storage polygon, floor and spill crest",
                ),
                (
                    figure_elevation_area_capacity,
                    "elevation-area-capacity curves (volume on the bottom axis, "
                    "area on the reversed top axis)",
                ),
                (figure_stage_storage, "stage-storage curve"),
                (figure_stage_area, "stage-area curve"),
                (figure_depth_area, "depth-area curve"),
                (figure_depth_volume, "depth-volume curve"),
                (figure_depth_distribution, "distribution of depth at the top stage"),
            ):
                stream = maker(r)
                if stream is not None:
                    d.picture(stream, f"Storage area {sid}: {cap}")

    # 6. Outputs ------------------------------------------------------------------
    outputs = ctx.get("outputs", [])
    if outputs:
        d.heading("6. Outputs", 2)
        d.caption("Table", "Files produced by the tool")
        d.table(["Output", "Location"], [list(o) for o in outputs], widths_cm=[5, 11])

    # 7. Notes ---------------------------------------------------------------------
    d.heading("7. Notes and limitations", 2)
    for note in [
        "Results are only as good as the DEM. Its resolution, vertical accuracy "
        "and vegetation or building artefacts control the floor, the spill "
        "level and the volume. A coarse DEM (e.g. a 30 m global DEM) smooths "
        "small depressions and embankment crests.",
        "Remote-sensing DEMs (LiDAR, photogrammetry, radar) record the water "
        "surface of ponds and reservoirs, not the bed. Storage below the water "
        "surface at the time of capture is not included; use a bathymetric "
        "survey for it.",
        "The spill level is only as reliable as the storage polygon: draw it "
        "beyond the crest or embankment all the way round. A polygon that stops "
        "on the side slope gives a spill level that is too low.",
        "Culverts, outlet pipes, spillway crests and low points narrower than a "
        "DEM cell are not represented; check the spill level against survey or "
        "design levels.",
        "Above the spill level (with a specified maximum stage), the polygon "
        "edge is treated as a vertical wall, as for a proposed embankment.",
    ]:
        d.bullet(note)
    warnings = ctx.get("warnings", [])
    if warnings:
        d.para("Processing messages recorded during this run:")
        for w in warnings:
            d.bullet(w)

    # 8. References ----------------------------------------------------------------
    d.heading("8. References", 2)
    for ref in REFERENCES:
        d.para(ref)

    d.doc.save(path)
