"""Word (.docx) report for the DEM Inundation Boundary tool, in the Mayim
report and figure standard (Doc(house_style=True), ChartStyle figures at
16.5 cm, map composed like the Catchment Delineation layout map).

Zero QGIS dependency. Needs python-docx and matplotlib (both imported
lazily, so the rest of the tool works without them). Figures use
matplotlib's object API (no pyplot), which is safe on QGIS's Processing
worker thread. The layout map is rendered by QGIS in the wrapper (a
PlanMapFn); without it a simplified plan is drawn with matplotlib.

No thousands separators anywhere in the report.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence

import numpy as np

from mayim_tools._common.docx_report import Doc
from mayim_tools._common.report_figures import CHART_COLOURS, CHART_INK, ChartStyle
from mayim_tools.hydrology.catchment_delineation.report import (
    OSM_ATTRIBUTION,
    PLAN_DPI,
    PLAN_PLOT_WIDTH_CM,
    PlanMap,
    _figure_plan_rendered,
    _north_arrow,
    _plan_height_cm,
    _scale_bar,
)

from .core import InundationResult

REPORT_TITLE = "DEM Inundation Boundary"

C_RED, C_AMBER, C_YELLOW, C_GREEN, C_BLUE, C_CYAN, C_MAGENTA = CHART_COLOURS

# (result, plot width in cm, dpi) -> PlanMap or None, supplied by the QGIS
# wrapper; without it the plan is drawn with matplotlib.
PlanMapFn = Callable[[InundationResult, float, float], "PlanMap | None"]

MODE_TEXT = {
    "all": "all ground below the level (static bathtub, separate hollows included)",
    "connected": "only ground connected to the seed point(s) below the level",
}

REFERENCES = [
    "Barnes, R., Lehman, C. & Mulla, D. (2014). Priority-flood: An optimal "
    "depression-filling and watershed-labeling algorithm for digital "
    "elevation models. Computers & Geosciences, 62, 117-127.",
    "Chaikin, G.M. (1974). An algorithm for high-speed curve generation. "
    "Computer Graphics and Image Processing, 3(4), 346-349.",
    "Lorensen, W.E. & Cline, H.E. (1987). Marching cubes: A high resolution "
    "3D surface construction algorithm. Computer Graphics (SIGGRAPH '87), "
    "21(4), 163-169.",
]


def _fmt(value, fmt: str) -> str:
    """Formats a number, or "-" when it is missing."""
    if value is None:
        return "-"
    try:
        if not np.isfinite(value):
            return "-"
    except TypeError:
        return "-"
    return fmt.format(value)


def depth_classes(depth: np.ndarray, cell_area: float, weights=None) -> list[dict]:
    """Area per depth class (round class widths, at most about 10 classes)."""
    d = depth[np.isfinite(depth)]
    if d.size == 0:
        return []
    w = np.ones_like(d) if weights is None else np.asarray(weights)[np.isfinite(depth)]
    dmax = float(d.max())
    step = next(
        (s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0) if dmax / s <= 10),
        20.0,
    )
    edges = np.arange(0.0, dmax + step, step)
    if edges[-1] < dmax:
        edges = np.append(edges, edges[-1] + step)
    total = float(w.sum())
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (d >= lo) & ((d < hi) | ((hi >= dmax) & (d <= hi)))
        out.append(
            {
                "from_m": float(lo),
                "to_m": float(hi),
                "area_m2": float(w[sel].sum()) * cell_area,
                "pct": 100.0 * float(w[sel].sum()) / total if total else 0.0,
            }
        )
    return out


def describe_result(result: InundationResult, geometry=None) -> str:
    """Plain-language summary of the extent."""
    if result.wet_cells == 0:
        return (
            f"At {result.level:.2f} m no ground inside the analysis area lies "
            "below the water."
        )
    n_bodies = len(result.polygons) if geometry is None else geometry[2]
    return (
        f"At a water level of {result.level:.2f} m the water covers "
        f"{result.area_m2 / 1e4:.4f} ha ({result.area_m2:.0f} m²) in {n_bodies} "
        f"water bod{'y' if n_bodies == 1 else 'ies'}, holds {result.volume_m3:.0f} m³ "
        f"and is up to {result.max_depth:.2f} m deep (mean "
        f"{result.mean_depth:.2f} m)."
    )


def parameter_rows(result: InundationResult, geometry=None) -> list[list[str]]:
    """Parameter (symbol, unit) | Value | Definition / equation (source)."""
    g = geometry
    return [
        ["Water level, h (m)", _fmt(result.level, "{:.2f}"),
         "Elevation of the flat water surface, in the DEM's vertical datum (input)."],
        ["Inundated area, A (m²)", _fmt(result.area_m2, "{:.0f}"),
         "A = Σ w_i a over wet cells; a = cell area, w_i = share of the cell "
         "inside the bounding polygon. A cell is wet when its inundation level "
         "h_i < h."],
        ["Inundated area, A (ha)", _fmt(result.area_m2 / 1e4, "{:.4f}"), "A / 10000."],
        ["Volume, V (m³)", _fmt(result.volume_m3, "{:.0f}"),
         "V = Σ w_i a (h - z_i) over wet cells (prism method, exact for the DEM)."],
        ["Maximum depth, d_max (m)", _fmt(result.max_depth if result.wet_cells else None, "{:.2f}"),
         "d_max = h - min(z_i) over wet cells."],
        ["Mean depth, d_m (m)", _fmt(result.mean_depth, "{:.2f}"), "d_m = V / A."],
        ["Wet cells, n (-)", str(result.wet_cells), "DEM cells with h_i < h."],
        ["Water bodies, N (-)", str(len(result.polygons) if g is None else g[2]),
         "Separate inundation polygons after clipping to the bounding polygon."],
        ["Polygon area, A_p (m²)", "-" if g is None else _fmt(g[0], "{:.0f}"),
         "Area of the smoothed inundation polygons, islands excluded; differs "
         "slightly from A, which counts whole cells."],
        ["Perimeter, P (m)", "-" if g is None else _fmt(g[1], "{:.0f}"),
         "Length of all polygon rings, including island edges and cut edges "
         "along the bounding polygon."],
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _ring_patch(rings, **kwargs):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    verts, codes = [], []
    for ring in rings:
        if len(ring) < 3:
            continue
        verts.extend(ring)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ring) - 2))
        codes.append(Path.CLOSEPOLY)
    if not verts:
        return None
    return PathPatch(Path(verts, codes), **kwargs)


def plan_extent(result: InundationResult, bounding=None, pad: float = 0.08):
    """(xmin, ymin, xmax, ymax) of the water (or of the bounding polygon /
    analysis window when nothing is wet), padded on every side."""
    pts = [p for ring in result.lines for p in ring]
    if not pts and bounding:
        pts = [p for poly in bounding for ring in poly for p in ring]
    if not pts:
        gt = result.window_gt
        rows, cols = result.fill.shape
        pts = [(gt[0], gt[3]), (gt[0] + cols * gt[1], gt[3] + rows * gt[5])]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    margin = pad * max(xmax - xmin, ymax - ymin, 1.0)
    return (xmin - margin, ymin - margin, xmax + margin, ymax + margin)


def _figure_plan_simple(
    result: InundationResult, style: ChartStyle, bounding=None, seeds_xy=()
) -> io.BytesIO:
    """Plan without QGIS: depth (blue shades), water's edge, bounding polygon
    and seeds, in the chart style with scale bar and north arrow."""
    from matplotlib.colors import LinearSegmentedColormap

    extent = plan_extent(result, bounding)
    xmin, ymin, xmax, ymax = extent
    height = _plan_height_cm(extent, 2.6)
    gt = result.window_gt
    rows, cols = result.fill.shape
    img_extent = (gt[0], gt[0] + cols * gt[1], gt[3] + rows * gt[5], gt[3])
    with style.context():
        fig = style.figure(height)
        ax = fig.add_subplot(1, 1, 1)
        depth = result.depth
        if np.isfinite(depth).any():
            cmap = LinearSegmentedColormap.from_list("depth", ["#CCE5FF", C_BLUE])
            im = ax.imshow(
                np.ma.masked_invalid(depth),
                extent=img_extent,
                cmap=cmap,
                vmin=0,
                interpolation="nearest",
                zorder=1,
            )
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("Depth (m)", fontproperties=style.font_props("bold"))
            cbar.ax.tick_params(colors=CHART_INK, width=0.75)
        for line in result.lines:
            lx, ly = zip(*line, strict=True)
            ax.plot(lx, ly, color=C_BLUE, linewidth=1.5, zorder=3)
        ax.plot([], [], color=C_BLUE, linewidth=1.5, label="Inundation boundary")
        if bounding:
            for poly in bounding:
                patch = _ring_patch(
                    poly,
                    facecolor="none",
                    edgecolor=CHART_INK,
                    linewidth=1.4,
                    linestyle="--",
                    zorder=4,
                )
                if patch is not None:
                    ax.add_patch(patch)
            ax.plot(
                [], [], color=CHART_INK, linewidth=1.4, linestyle="--",
                label="Bounding polygon",
            )  # fmt: skip
        if seeds_xy:
            sx, sy = zip(*seeds_xy, strict=True)
            ax.plot(
                sx, sy, marker="o", markersize=5, color=C_RED,
                markeredgecolor=CHART_INK, linestyle="none", label="Seed point",
                zorder=5,
            )  # fmt: skip
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        _scale_bar(ax, extent)
        _north_arrow(ax)
        style.axes(ax, grid=False)
        style.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3)
        return style.png(fig)


def figure_plan(
    result: InundationResult,
    style: ChartStyle,
    plan_map: PlanMapFn | None = None,
    bounding=None,
    seeds_xy=(),
    messages: list | None = None,
) -> tuple[io.BytesIO, PlanMap | None]:
    """The layout map: rendered by ``plan_map`` (QGIS, with the layer styles
    and OpenStreetMap) when given, otherwise drawn with matplotlib."""
    if plan_map is not None:
        try:
            pm = plan_map(result, PLAN_PLOT_WIDTH_CM, PLAN_DPI)
        except Exception as e:  # the report must still be written
            pm = None
            if messages is not None:
                messages.append(
                    f"The QGIS layout map could not be rendered ({e}); a "
                    "simplified plan was drawn instead."
                )
        if pm is not None:
            return _figure_plan_rendered(pm, style), pm
    return _figure_plan_simple(result, style, bounding, seeds_xy), None


def figure_depth_distribution(
    classes: list[dict], level: float, style: ChartStyle
) -> io.BytesIO | None:
    if not classes:
        return None
    with style.context():
        fig = style.figure(7.5)
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(
            [c["from_m"] for c in classes],
            [c["pct"] for c in classes],
            width=[c["to_m"] - c["from_m"] for c in classes],
            align="edge",
            color=C_BLUE,
            edgecolor=CHART_INK,
            linewidth=0.75,
            zorder=2,
        )
        ax.set_xlabel(f"Water depth at {level:.2f} m (m)")
        ax.set_ylabel("Share of inundated area (%)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        style.axes(ax)
        return style.png(fig)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def write_report_docx(
    path: str,
    result: InundationResult,
    context: dict,
    geometry=None,
    *,
    bounding=None,
    seeds_xy: Sequence = (),
    plan_map: PlanMapFn | None = None,
    include_figures: bool = True,
) -> list[str]:
    """Writes the report and returns messages for the log (font
    substitutions, map problems). ``context`` describes the run (tool,
    version, run time, DEM, settings, outputs, warnings - see the QGIS
    wrapper); ``geometry`` is (polygon area, perimeter, number of
    polygons) measured on the clipped output polygons."""
    d = Doc(house_style=True)
    ctx = context or {}
    messages: list[str] = []
    style = None
    if include_figures:
        try:
            style = ChartStyle()
            messages.extend(style.warnings)
        except ImportError:
            include_figures = False
    dem = ctx.get("dem", {})
    mode = result.mode
    cell_area = abs(result.window_gt[1] * result.window_gt[5])
    g = geometry

    d.heading(REPORT_TITLE, 1)

    # 1. Summary ------------------------------------------------------------
    d.heading("1. Summary", 2)
    d.para(
        "This annexure documents the level-pool (bathtub) inundation extent at "
        f"a water level of {result.level:.2f} m, read from a digital elevation "
        f"model (DEM) with a {dem.get('cell_x', 0):g} m x "
        f"{dem.get('cell_y', 0):g} m cell size, using the "
        f"{ctx.get('tool', 'DEM Inundation Boundary')} tool (Mayim Tools, "
        f"version {ctx.get('version', '-')}). Inundated ground was "
        f"{MODE_TEXT.get(mode, mode)}"
        + (
            " within the bounding polygon."
            if ctx.get("bounding")
            else " over the whole DEM (no bounding polygon)."
        )
    )
    d.para(describe_result(result, g))
    d.caption("Table", "Summary of the inundation")
    d.table(
        [
            "Water level (m)",
            "Area (ha)",
            "Volume (m³)",
            "Max depth (m)",
            "Mean depth (m)",
            "Water bodies",
        ],
        [
            [
                _fmt(result.level, "{:.2f}"),
                _fmt(result.area_m2 / 1e4, "{:.4f}"),
                _fmt(result.volume_m3, "{:.0f}"),
                _fmt(result.max_depth if result.wet_cells else None, "{:.2f}"),
                _fmt(result.mean_depth, "{:.2f}"),
                str(len(result.polygons) if g is None else g[2]),
            ]
        ],
        widths_cm=[2.9, 2.6, 2.9, 2.7, 2.7, 2.7],
    )

    # 2. Method -------------------------------------------------------------
    d.heading("2. Tool and method", 2)
    d.para(
        "The water surface is taken as flat at the given level: no flow, "
        "water-surface slope or hydraulic losses are modelled. The processing "
        "steps were:"
    )
    for step in [
        (
            "The bounding polygon was converted to cell weights - the fraction "
            "of each cell inside the polygon from an 8 x 8 sub-cell sample - "
            "and the DEM was read for the polygon's extent."
            if ctx.get("bounding")
            else "The whole DEM was used; every valid cell has a weight of 1."
        ),
        (
            "Each cell's inundation level is its own ground elevation, so every "
            "cell below the water level floods, including hollows that are "
            "not connected to each other."
            if mode == "all"
            else "A priority flood (Barnes et al., 2014) from each seed point "
            "gave every cell the lowest water level at which water from the "
            "seed reaches it - the lowest of the highest points along all "
            "eight-connected paths. A cell floods when this level is below the "
            "water level, so hollows behind higher ground stay dry until the "
            "water overtops it. Water starts at the seed cell, so a level at or "
            "below the seed's ground floods nothing."
        ),
        "The inundated area is the sum of the weighted areas of the wet cells "
        "and the volume the sum of their weighted areas times the water depth "
        "above each cell (prism method, exact for the DEM).",
        "The water's edge was traced between cell centres by marching squares "
        "(Lorensen & Cline, 1987), with the crossing interpolated linearly "
        "between the two cell elevations, and rounded with "
        f"{ctx.get('smoothing', 3)} pass(es) of Chaikin corner cutting "
        "(Chaikin, 1974). Closed rings smaller than "
        f"{ctx.get('min_cells', 3):g} cells were dropped as DEM noise.",
        "Polygons were built from the same rings - the outer ring is water, "
        "rings inside it are dry islands, rings inside those are water "
        "again - so the boundary line and polygons match exactly. Where the "
        "water reaches the bounding polygon, both were cut at its edge.",
    ]:
        d.numbered(step)

    # 3. Inputs ---------------------------------------------------------------
    d.heading("3. Inputs and settings", 2)
    d.caption("Table", "Inputs and processing settings")
    seeds = ctx.get("seeds") or {}
    seed_text = "Not used"
    if mode == "connected":
        seed_text = f"{seeds.get('layer') or 'None'} ({result.n_seeds} used"
        if result.seed_ground:
            seed_text += (
                f", ground {min(result.seed_ground):.2f} to "
                f"{max(result.seed_ground):.2f} m"
            )
        seed_text += ")"
    d.table(
        ["Item", "Value"],
        [
            ["DEM", dem.get("path", "-")],
            ["Coordinate reference system", dem.get("crs", "-")],
            [
                "DEM cell size",
                f"{dem.get('cell_x', 0):g} m x {dem.get('cell_y', 0):g} m",
            ],
            ["Water level", f"{result.level:.2f} m"],
            ["Bounding polygon", ctx.get("bounding") or "None (whole DEM)"],
            ["Inundated ground", MODE_TEXT.get(mode, mode)],
            ["Seed points", seed_text],
            ["Smoothing passes", str(ctx.get("smoothing", "-"))],
            ["Smallest ring kept", f"{ctx.get('min_cells', 0):g} cells"],
            ["Project number", ctx.get("project_number") or "-"],
            ["Run", ctx.get("run_time", "-")],
        ],
        widths_cm=[5.0, 11.5],
    )

    # 4. Parameters ----------------------------------------------------------
    d.heading("4. Parameters and calculations", 2)
    d.caption("Table", "Inundation parameters, values and definitions")
    d.table(
        ["Parameter (symbol, unit)", "Value", "Definition / equation (source)"],
        parameter_rows(result, g),
        widths_cm=[6.2, 2.8, 7.5],
    )

    # 5. Results --------------------------------------------------------------
    d.heading("5. Results", 2)
    d.para(describe_result(result, g))
    classes = []
    if result.wet_cells:
        classes = depth_classes(result.depth, cell_area, result.weights)
        d.caption("Table", f"Inundated area by depth class at {result.level:.2f} m")
        d.table(
            ["Depth class (m)", "Area (m²)", "Share (%)"],
            [
                [
                    f"{c['from_m']:.2f} - {c['to_m']:.2f}",
                    f"{c['area_m2']:.0f}",
                    f"{c['pct']:.1f}",
                ]
                for c in classes
            ],
            widths_cm=[6.0, 5.25, 5.25],
        )
    used_osm = False
    if include_figures and style is not None:
        stream, pm = figure_plan(result, style, plan_map, bounding, seeds_xy, messages)
        used_osm = pm is not None and bool(pm.attribution)
        d.picture(
            stream,
            f"Inundation at {result.level:.2f} m: water depth, inundation "
            "boundary"
            + (", bounding polygon" if bounding else "")
            + (", seed points" if seeds_xy else "")
            + (" (background: OpenStreetMap)" if used_osm else ""),
        )
        hist = figure_depth_distribution(classes, result.level, style)
        if hist is not None:
            d.picture(hist, f"Distribution of water depth at {result.level:.2f} m")

    # 6. Outputs --------------------------------------------------------------
    outputs = ctx.get("outputs", [])
    if outputs:
        d.heading("6. Outputs", 2)
        d.caption("Table", "Files produced by the tool")
        d.table(
            ["Output", "Location"], [list(o) for o in outputs], widths_cm=[5.0, 11.5]
        )

    # 7. Notes ----------------------------------------------------------------
    d.heading("7. Notes and limitations", 2)
    notes = [
        "This is a level-pool (bathtub) extent: the water surface is flat. It "
        "does not model flow, the water-surface slope along a river, hydraulic "
        "losses, timing or storage routing. For flood lines along a river use "
        "a hydraulic model; this extent is a screening estimate or a check on "
        "one.",
        "Results are only as good as the DEM. Its resolution, vertical accuracy "
        "and vegetation or building artefacts control the extent. Embankments, "
        "walls, culverts and channels narrower than a DEM cell are not "
        "represented, so water may pass through a thin embankment or be "
        "blocked where a culvert would connect two areas.",
        "Remote-sensing DEMs record the water surface of ponds, dams and "
        "rivers at the time of capture, not the bed; depths and volumes below "
        "that surface are not included.",
        "The smoothed boundary interpolates between cell centres; the depth and "
        "water-surface rasters follow whole DEM cells, so their edges are "
        "stepped and the cell-based area differs slightly from the polygon "
        "area.",
    ]
    if mode == "connected":
        notes.append(
            "Seed points should lie on the lowest ground of the water body "
            "(channel bed, dam basin). Water starts at the seed cell, so a seed "
            "on a bank floods nothing below the bank level."
        )
    if used_osm:
        notes.append(
            f"Map background: {OSM_ATTRIBUTION}. OpenStreetMap data is "
            "available under the Open Database Licence (ODbL), "
            "openstreetmap.org/copyright."
        )
    for note in notes:
        d.bullet(note)
    warnings = list(ctx.get("warnings", [])) + messages
    if warnings:
        d.para("Processing messages recorded during this run:")
        for w in dict.fromkeys(warnings):
            d.bullet(w)

    # 8. References -------------------------------------------------------------
    d.heading("8. References", 2)
    for ref in sorted(REFERENCES):
        d.para(ref)

    d.doc.save(path)
    return messages
