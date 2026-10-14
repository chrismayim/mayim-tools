"""Word (.docx) report for the DEM Inundation Boundary tool.

Zero QGIS dependency. Needs python-docx and matplotlib (both imported
lazily, so the rest of the tool works without them). Figures use
matplotlib's object API (no pyplot), which is safe on QGIS's Processing
worker thread. Document and chart helpers are shared with the other
Mayim Tools reports (mayim_tools/_common/docx_report.py).

No thousands separators anywhere in the report.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np

from mayim_tools._common.docx_report import (
    C_BLUE,
    C_INK,
    C_INK_2,
    C_ORANGE,
    Doc,
    new_figure,
    png,
    style_axes,
)

from .core import InundationResult

REPORT_TITLE = "DEM Inundation Boundary Assessment"

MODE_TEXT = {
    "all": "all ground below the level (static bathtub, separate hollows included)",
    "connected": "only ground connected to the seed point(s) below the level",
}

# (quantity, symbol (unit), definition / equation)
DEFINITIONS = [
    ("Water level", "h (m)", "Elevation of the water surface, in the DEM's vertical datum; the same everywhere (level pool)."),
    ("Inundation level", "h_i (m)", "Level at which a cell first floods: its ground elevation (all-ground option), or the lowest of the highest points along the best path from a seed (connected option)."),
    ("Wet cell", "-", "Cell inside the bounding area with h_i < h."),
    ("Cell weight", "w (-)", "Fraction of the cell inside the bounding polygon (1 for interior cells, and for every cell without a bounding polygon)."),
    ("Inundated area", "A(h) (m²)", "A(h) = Σ w_i a over wet cells, with a the cell area."),
    ("Volume", "V(h) (m³)", "V(h) = Σ w_i a (h - z_i) over wet cells (prism method, exact for the DEM)."),
    ("Depth", "d (m)", "d = h - z, water depth over a wet cell."),
    ("Mean depth", "d_m (m)", "d_m = V(h) / A(h)."),
    ("Polygon area", "(m²)", "Area of the smoothed inundation polygons, after clipping to the bounding polygon. Differs slightly from A(h), which counts whole cells."),
    ("Perimeter", "(m)", "Length of all polygon rings, including the edges of dry islands and cut edges along the bounding polygon."),
]  # fmt: skip

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
        area = float(w[sel].sum()) * cell_area
        out.append(
            {
                "from_m": float(lo),
                "to_m": float(hi),
                "area_m2": area,
                "pct": 100.0 * float(w[sel].sum()) / total if total else 0.0,
            }
        )
    return out


def describe_result(result: InundationResult, geometry=None) -> str:
    """Plain-language summary of the extent."""
    if result.wet_cells == 0:
        return (
            f"At {result.level:.3f} m no ground inside the analysis area lies "
            "below the water."
        )
    n_bodies = len(result.polygons) if geometry is None else geometry[2]
    return (
        f"At {result.level:.3f} m the water covers {result.area_m2 / 1e4:.3f} ha "
        f"({result.area_m2:.0f} m²) in {n_bodies} water "
        f"bod{'y' if n_bodies == 1 else 'ies'}, holds {result.volume_m3:.0f} m³ "
        f"({result.volume_m3 / 1e3:.2f} ML) and is up to {result.max_depth:.2f} m "
        f"deep (mean {result.mean_depth:.2f} m)."
    )


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


def figure_plan(
    result: InundationResult, bounding=None, seeds_xy=()
) -> io.BytesIO | None:
    """Depth with the inundation boundary, the bounding polygon and the seed
    points."""
    depth = result.depth
    if not np.isfinite(depth).any():
        return None
    gt = result.window_gt
    rows, cols = depth.shape
    extent = [gt[0], gt[0] + cols * gt[1], gt[3] + rows * gt[5], gt[3]]
    fig = new_figure(12.0)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        np.ma.masked_invalid(depth),
        extent=extent,
        cmap="Blues",
        vmin=0,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Depth at {result.level:.2f} m (m)", fontsize=8, color=C_INK)
    cbar.ax.tick_params(labelsize=7, colors=C_INK_2)

    for line in result.lines:
        lx, ly = zip(*line, strict=True)
        ax.plot(lx, ly, color=C_INK, linewidth=0.7)
    ax.plot(
        [],
        [],
        color=C_INK,
        linewidth=0.7,
        label=f"Inundation boundary {result.level:.2f} m",
    )
    if bounding:
        for poly in bounding:
            patch = _ring_patch(
                poly, facecolor="none", edgecolor=C_ORANGE, linewidth=0.9
            )
            if patch is not None:
                ax.add_patch(patch)
        ax.plot([], [], color=C_ORANGE, linewidth=0.9, label="Bounding polygon")
    if seeds_xy:
        sx, sy = zip(*seeds_xy, strict=True)
        ax.plot(
            sx,
            sy,
            marker="o",
            markersize=5,
            color=C_ORANGE,
            markeredgecolor=C_INK,
            linestyle="none",
            label="Seed point",
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    style_axes(ax)
    ax.grid(False)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(fontsize=6.5, frameon=True, framealpha=0.85, loc="best")
    return png(fig)


def figure_depth_distribution(classes: list[dict], level: float) -> io.BytesIO | None:
    if not classes:
        return None
    fig = new_figure(7.0)
    ax = fig.add_subplot(1, 1, 1)
    lo = [c["from_m"] for c in classes]
    width = [c["to_m"] - c["from_m"] for c in classes]
    ax.bar(
        lo,
        [c["pct"] for c in classes],
        width=width,
        align="edge",
        color=C_BLUE,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_xlabel(f"Depth at {level:.2f} m (m)")
    ax.set_ylabel("Share of inundated area (%)")
    ax.set_xlim(left=0)
    style_axes(ax)
    return png(fig)


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
    weights=None,
    include_figures: bool = True,
) -> None:
    """Writes the report. ``context`` describes the run (tool, version, run
    time, DEM, settings, outputs, warnings - see the QGIS wrapper);
    ``geometry`` is (polygon area, perimeter, number of polygons) measured
    on the clipped output polygons; ``weights`` are the cell
    weights inside the bounding polygon (for the depth classes)."""
    d = Doc()
    ctx = context or {}
    if include_figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            include_figures = False
    dem = ctx.get("dem", {})
    mode = result.mode
    cell_area = abs(result.window_gt[1] * result.window_gt[5])

    d.heading(REPORT_TITLE, 1)

    # 1. Summary ------------------------------------------------------------
    d.heading("1. Summary", 2)
    d.para(
        f"This report documents the level-pool (bathtub) inundation extent "
        f"at a water level of {result.level:.3f} m, read from a digital elevation model (DEM) with a "
        f"{dem.get('cell_x', 0):g} m x {dem.get('cell_y', 0):g} m cell size, "
        f"using the {ctx.get('tool', 'DEM Inundation Boundary')} tool (Mayim "
        f"Tools, version {ctx.get('version', '-')}). Inundated ground was "
        f"{MODE_TEXT.get(mode, mode)}"
        + (
            " within the bounding polygon."
            if ctx.get("bounding")
            else " over the whole DEM (no bounding polygon)."
        )
    )
    d.para(describe_result(result, geometry))

    # 2. Method -------------------------------------------------------------
    d.heading("2. Tool and method", 2)
    d.para(
        "The water surface is taken as flat: no flow, gradient "
        "or hydraulic losses are modelled. The processing steps were:"
    )
    steps = [
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
        "The inundated area is the sum of the weighted areas "
        "of the wet cells and the volume the sum of their weighted areas times "
        "the water depth above each cell (prism method, exact for the DEM).",
        "The water's edge was traced between cell centres by marching squares "
        "(Lorensen & Cline, 1987), with the crossing interpolated linearly "
        "between the two cell elevations, and rounded with "
        f"{ctx.get('smoothing', 3)} pass(es) of Chaikin corner cutting "
        "(Chaikin, 1974). Closed rings smaller than "
        f"{ctx.get('min_cells', 3):g} cells were dropped as DEM noise.",
        "Polygons were built from the same rings - the outer ring is water, "
        "rings inside it are dry islands, rings inside those are water "
        "again - so the boundary lines and polygons match exactly. Where the "
        "water reaches the bounding polygon, both were cut at its edge.",
    ]
    for step in steps:
        d.numbered(step)

    # 3. Inputs ---------------------------------------------------------------
    d.heading("3. Inputs and settings", 2)
    d.caption("Table", "Inputs and processing settings")
    seeds = ctx.get("seeds") or {}
    d.table(
        ["Item", "Value"],
        [
            ["DEM", dem.get("path", "-")],
            ["Coordinate reference system", dem.get("crs", "-")],
            [
                "DEM cell size",
                f"{dem.get('cell_x', 0):g} m x {dem.get('cell_y', 0):g} m",
            ],
            ["Water level", f"{result.level:.3f} m"],
            ["Bounding polygon", ctx.get("bounding") or "None (whole DEM)"],
            ["Inundated ground", MODE_TEXT.get(mode, mode)],
            [
                "Seed points",
                (
                    "Not used"
                    if mode == "all"
                    else f"{seeds.get('layer') or 'None'} ({result.n_seeds} "
                    "used"
                    + (
                        f", ground {min(result.seed_ground):.3f} to "
                        f"{max(result.seed_ground):.3f} m"
                        if result.seed_ground
                        else ""
                    )
                    + ")"
                ),
            ],
            ["Smoothing passes", str(ctx.get("smoothing", "-"))],
            ["Smallest ring kept", f"{ctx.get('min_cells', 0):g} cells"],
            ["Run", ctx.get("run_time", "-")],
        ],
        widths_cm=[5.0, 11.0],
    )

    # 4. Definitions ----------------------------------------------------------
    d.heading("4. Definitions", 2)
    d.caption("Table", "Quantities and equations")
    d.table(
        ["Quantity", "Symbol (unit)", "Definition / equation"],
        [list(row) for row in DEFINITIONS],
        widths_cm=[3.6, 2.8, 9.6],
    )

    # 5. Results --------------------------------------------------------------
    d.heading("5. Results", 2)
    g = geometry
    rows = [
        ["Water level (m)", f"{result.level:.3f}"],
        ["Inundated area on the DEM cells (m²)", f"{result.area_m2:.0f}"],
        ["Inundated area on the DEM cells (ha)", f"{result.area_m2 / 1e4:.4f}"],
        ["Volume (m³)", f"{result.volume_m3:.0f}"],
        ["Volume (ML)", f"{result.volume_m3 / 1e3:.2f}"],
        ["Maximum depth (m)", f"{result.max_depth:.3f}"],
        [
            "Mean depth (m)",
            "-" if result.mean_depth is None else f"{result.mean_depth:.3f}",
        ],
        ["Wet cells", str(result.wet_cells)],
        ["Water bodies", str(len(result.polygons) if g is None else g[2])],
    ]
    if g is not None:
        rows += [
            ["Polygon area (m²)", f"{g[0]:.0f}"],
            ["Perimeter (m)", f"{g[1]:.0f}"],
        ]
    d.caption("Table", f"Inundation at {result.level:.3f} m")
    d.table(["Quantity", "Value"], rows, widths_cm=[9.0, 7.0])

    classes = []
    if result.wet_cells:
        classes = depth_classes(
            result.depth,
            cell_area,
            weights if weights is not None else result.weights,
        )
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
            widths_cm=[6.0, 5.0, 5.0],
        )

    if include_figures:
        for stream, cap in (
            (
                figure_plan(result, bounding, seeds_xy),
                f"Depth and inundation boundary at {result.level:.2f} m",
            ),
            (
                figure_depth_distribution(classes, result.level),
                f"Distribution of depth at {result.level:.2f} m",
            ),
        ):
            if stream is not None:
                d.picture(stream, cap)

    # 6. Outputs --------------------------------------------------------------
    outputs = ctx.get("outputs", [])
    if outputs:
        d.heading("6. Outputs", 2)
        d.caption("Table", "Files produced by the tool")
        d.table(["Output", "Location"], [list(o) for o in outputs], widths_cm=[5, 11])

    # 7. Notes ----------------------------------------------------------------
    d.heading("7. Notes and limitations", 2)
    for note in [
        "This is a level-pool (bathtub) extent: the water surface is flat. It does not model flow, the water-surface slope along a "
        "river, hydraulic losses, timing or storage routing. For flood lines "
        "along a river use a hydraulic model; this extent is a screening "
        "estimate or a check on one.",
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
        "With the connected option, place seed points on the lowest ground of "
        "the water body (channel bed, dam basin). Water starts at the seed "
        "cell, so a seed on a bank floods nothing below the bank level.",
    ]:
        d.bullet(note)
    warnings = ctx.get("warnings", [])
    if warnings:
        d.para("Processing messages recorded during this run:")
        for w in warnings:
            d.bullet(w)

    # 8. References -------------------------------------------------------------
    d.heading("8. References", 2)
    for ref in sorted(REFERENCES):
        d.para(ref)

    d.doc.save(path)
