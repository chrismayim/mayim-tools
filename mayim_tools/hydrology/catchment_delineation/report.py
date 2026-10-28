"""Word (.docx) annexure report for the catchment delineation tool.

Zero QGIS dependency. Needs python-docx and matplotlib (both imported
lazily, so the rest of the tool works without them). Figures are drawn
with matplotlib's object API (no pyplot), which is safe on QGIS's
Processing worker thread.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mayim_tools._common.docx_report import Doc as _Doc
from mayim_tools._common.docx_report import format_value
from mayim_tools._common.report_figures import (
    CHART_COLOURS,
    CHART_INK,
    CHART_LINE_PT,
    ChartStyle,
)

from .core import CatchmentResult
from .description import ELONGATION_CLASSES, REFERENCES, classify, describe_catchment

# Content colours, taken from the house chart palette in its order. Yellow
# (third) is kept for areas and fills; thin lines on white use green instead.
C_RED, C_AMBER, C_YELLOW, C_GREEN, C_BLUE, C_CYAN, C_MAGENTA = CHART_COLOURS


@dataclass
class PlanMap:
    """A catchment layout map rendered elsewhere (by QGIS, with the layer
    styles): the map image, the extent it covers (xmin, ymin, xmax, ymax),
    legend entries [(label, symbol image)] and an attribution line for the
    background (None when there is no background)."""

    image: object  # PNG bytes or an RGB(A) array
    extent: tuple[float, float, float, float]
    legend: list = field(default_factory=list)
    attribution: str | None = None


# (catchment, plot width in cm, dpi) -> PlanMap or None. Supplied by the
# QGIS wrapper; without it the plan is drawn with matplotlib.
PlanMapFn = Callable[[CatchmentResult, float, float], "PlanMap | None"]
PLAN_PLOT_WIDTH_CM = 14.0  # map frame width inside the 16.5 cm figure
PLAN_DPI = 150  # map render resolution (symbol sizes stay true to the QML)
OSM_ATTRIBUTION = "© OpenStreetMap contributors"

REPORT_TITLE = "Catchment Delineation and Characterisation"

# (parameter, symbol and unit, definition / equation, source)
PARAMETER_DEFINITIONS = [
    ("Contributing area", "A (km²)", "Number of cells draining to the outlet x cell area.", "-"),
    ("Polygon area", "A_p (km²)", "Area of the catchment outline, including filled interior holes.", "-"),
    ("Perimeter", "P (km)", "Length of the catchment outline, following cell edges.", "-"),
    ("Elevations", "z (m)", "Minimum, maximum and mean DEM elevation of contributing cells; z_o at the outlet cell (for a pour line, its lowest cell).", "-"),
    ("Relief", "z_max - z_min (m)", "Difference between the highest and lowest points of the catchment.", "-"),
    ("Elevation gain", "H (m)", "H = z_max - z_o, height of the highest point above the outlet. Equal to the relief when the outlet is the lowest point.", "Schumm (1956)"),
    ("Longest flow path", "L (km)", "Longest D8 flow path to the outlet, from cell centre to outlet cell centre.", "O'Callaghan & Mark (1984)"),
    ("Centroid distance", "Lca (km)", "Distance along L from the outlet to the point on L nearest the catchment centroid.", "-"),
    ("Basin length", "Lb (km)", "Longest straight-line distance from the outlet to any point in the catchment.", "-"),
    ("Average slope", "S_avg (m/m)", "S_avg = (z_top - z_o) / L, along the longest flow path.", "-"),
    ("10-85 slope", "S_1085 (m/m)", "S_1085 = (z_85 - z_10) / (0.75 L), elevations at 10 % and 85 % of L from the outlet.", "-"),
    ("Equal-area slope", "S_e (m/m)", "Slope of the line through the outlet with the same area under it as the profile: S_e = 2 ∫(z - z_o) dx / L².", "-"),
    ("Mean catchment slope", "S_m (%)", "Mean of the cell slopes (3 x 3 finite-difference method) over the catchment.", "Horn (1981)"),
    ("Form factor", "R_f", "R_f = A_p / L².", "Horton (1932)"),
    ("Circularity ratio", "R_c", "R_c = 4π A_p / P².", "Miller (1953)"),
    ("Elongation ratio", "R_e", "R_e = 2 √(A_p / π) / L.", "Schumm (1956)"),
    ("Compactness coefficient", "K_c", "K_c = P / (2 √(π A_p)).", "Gravelius (1914)"),
    ("Hypsometric integral", "HI", "HI = (z_mean - z_min) / (z_max - z_min) (elevation-relief ratio).", "Strahler (1952); Pike & Wilson (1971)"),
    ("Relief ratio", "R_h", "R_h = H / Lb, with H = z_max - z_o.", "Schumm (1956)"),
    ("Melton ratio", "M", "M = H / √A (H in km).", "Melton (1965)"),
    ("Ruggedness number", "R_n", "R_n = H x D_d (H in km).", "Strahler (1958)"),
    ("Streams", "-", "Cells whose upstream area is at least the stream threshold.", "-"),
    ("Stream order", "u", "Strahler ordering of the stream cells; the catchment order is the highest.", "Strahler (1957)"),
    ("Bifurcation ratio", "R_b", "Mean of N_u / N_u+1 over consecutive orders (N_u = number of segments of order u).", "Strahler (1964)"),
    ("Drainage density", "D_d (km/km²)", "D_d = total stream length / A.", "Horton (1932)"),
    ("Stream frequency", "F_s (1/km²)", "F_s = number of stream segments / A.", "Horton (1932)"),
    ("Length of overland flow", "L_o (km)", "L_o = 1 / (2 D_d).", "Horton (1945)"),
    ("Constant of channel maintenance", "C (km²/km)", "C = 1 / D_d.", "Schumm (1956)"),
    ("Sinuosity", "-", "L / straight-line distance between its ends.", "-"),
    ("Channel start on L", "L_ch (m)", "Distance from the outlet along L to the most upstream cell of L whose upstream area reaches the stream threshold (the channel head); blank when L has no stream cell. Used by the 'Catchments: 2. Time of Concentration' tool.", "-"),
    ("Waterway length", "L_w (m)", "Length of the stream through a stream sub-catchment; the longest flow path L otherwise.", "-"),
    ("Mean overland flow length", "L_om (m)", "Mean D8 flow length from every cell to the nearest stream cell (to the outlet where no stream is reached).", "-"),
    ("SWMM width - flow length", "W_f (m)", "W_f = A / L_om, with L_om capped at the maximum overland flow length when one is set.", "Rossman & Huber (2016)"),
    ("SWMM width - channel", "W_c (m)", "W_c = 2 L_w for a sub-catchment drained by a central waterway; W_c = L_w for one bank of a stream.", "Rossman & Huber (2016)"),
    ("Shape factor", "X", "X = A / L_w², capped at 4 (Guo & Urbonas recommend 4 to 6; UDFCD uses 4).", "Guo & Urbonas (2009)"),
    ("Skew factor", "Z", "Z = A_m / A, A_m the larger of the areas on either side of the waterway (0.5 to 1; 1 for one bank).", "Guo & Urbonas (2009)"),
    ("SWMM width - Guo & Urbonas", "W_g (m)", "W_g = L_w (1.5 - Z) (2.286 X - 0.286 X²): equivalent kinematic-wave plane conserving the area and the fall along the waterway.", "Guo & Urbonas (2009)"),
    ("Width ratio", "W_g / W_f", "Agreement of the two estimates; far from 1 means a shape far from a rectangle, where calibration should decide.", "-"),
]  # fmt: skip

# General parameters: (group heading, [(label, attribute key, format, scale)])
RESULT_GROUPS = [
    (
        "Geometry",
        [
            ("Contributing area (ha)", "area_ha", "{:.2f}", 1),
            ("Filled interior holes (ha)", "hole_ha", "{:.2f}", 1),
            ("Polygon parts", "n_parts", "{:d}", None),
            ("Centroid E", "cx", "{:.1f}", 1),
            ("Centroid N", "cy", "{:.1f}", 1),
        ],
    ),
    (
        "Outlet",
        [
            ("Outlet type", "out_type", "{}", None),
            ("Outlet E", "x_out", "{:.1f}", 1),
            ("Outlet N", "y_out", "{:.1f}", 1),
            ("Snap distance (m)", "snap_m", "{:.1f}", 1),
            ("Outlet cells", "n_outcells", "{:d}", None),
            ("Drains into", "ds_ids", "{}", None),
        ],
    ),
    (
        "Drainage network",
        [
            ("Highest Strahler order", "strm_ord", "{:d}", None),
            ("Stream segments", "n_strm", "{:d}", None),
            ("Stream length (km)", "strm_km", "{:.3f}", 1),
            ("Drainage density D_d (km/km²)", "dd_kmkm2", "{:.3f}", 1),
            ("Stream frequency F_s (1/km²)", "strm_freq", "{:.3f}", 1),
            ("Bifurcation ratio R_b", "bif_r", "{:.2f}", 1),
            ("Length of overland flow L_o (km)", "lo_km", "{:.3f}", 1),
            ("Constant of channel maintenance C (km²/km)", "c_maint", "{:.3f}", 1),
            ("Channel start on L, from the outlet L_ch (m)", "ch_head_m", "{:.1f}", 1),
        ],
    ),
    (
        "SWMM width",
        [
            ("Waterway length L_w (m)", "chan_len_m", "{:.1f}", 1),
            ("Mean overland flow length L_om (m)", "lo_mean_m", "{:.1f}", 1),
            ("Width from flow length W_f = A / L_om (m)", "w_flow_m", "{:.1f}", 1),
            ("Width from channel W_c (m)", "w_chan_m", "{:.1f}", 1),
            ("Shape factor X", "gu_x", "{:.3f}", 1),
            ("Skew factor Z", "gu_z", "{:.3f}", 1),
            ("Width, Guo & Urbonas W_g (m)", "w_gu_m", "{:.1f}", 1),
            ("Width ratio W_g / W_f", "w_ratio", "{:.2f}", 1),
        ],
    ),
]

# Shape parameters table: (label, attribute key, format, scale, definition)
SHAPE_ROWS = [
    ("Contributing area A (km²)", "area_km2", "{:.4f}", 1, "Cells draining to the outlet x cell area"),
    ("Polygon area A_p (km²)", "poly_km2", "{:.4f}", 1, "Outline area, filled holes included"),
    ("Perimeter P (km)", "perim_km", "{:.3f}", 1, "Length of the cell-edge outline"),
    ("Longest flow path L (km)", "lfp_km", "{:.3f}", 1, "Longest D8 flow path to the outlet"),
    ("Basin length Lb (km)", "lb_km", "{:.3f}", 1, "Straight line, outlet to farthest point"),
    ("Centroid distance Lca (km)", "lca_km", "{:.3f}", 1, "Along L, outlet to the point nearest the centroid"),
    ("Sinuosity of L", "sinuosity", "{:.3f}", 1, "L / straight-line distance between its ends"),
    ("Form factor R_f", "form_f", "{:.3f}", 1, "A_p / L² (Horton 1932)"),
    ("Circularity ratio R_c", "circ_r", "{:.3f}", 1, "4π A_p / P² (Miller 1953)"),
    ("Elongation ratio R_e", "elong_r", "{:.3f}", 1, "2 √(A_p / π) / L (Schumm 1956)"),
    ("Compactness coefficient K_c", "gravel_kc", "{:.3f}", 1, "P / (2 √(π A_p)) (Gravelius 1914)"),
]  # fmt: skip

# Relief parameters table: (label, attribute key, format, scale, definition)
RELIEF_ROWS = [
    ("Outlet elevation z_o (m)", "z_outlet", "{:.2f}", 1, "DEM elevation at the outlet"),
    ("Minimum elevation z_min (m)", "z_min", "{:.2f}", 1, "Lowest contributing cell"),
    ("Maximum elevation z_max (m)", "z_max", "{:.2f}", 1, "Highest contributing cell"),
    ("Mean elevation z_mean (m)", "z_mean", "{:.2f}", 1, "Mean of the contributing cells"),
    ("Relief (m)", "relief_m", "{:.2f}", 1, "z_max - z_min"),
    ("Elevation gain H (m)", "elev_gain", "{:.2f}", 1, "z_max - z_o"),
    ("Hypsometric integral HI", "hyps_int", "{:.3f}", 1, "(z_mean - z_min) / (z_max - z_min)"),
    ("Relief ratio R_h", "relief_r", "{:.4f}", 1, "H / Lb (Schumm 1956)"),
    ("Melton ratio M", "melton_r", "{:.3f}", 1, "H / √A, H in km (Melton 1965)"),
    ("Ruggedness number R_n", "rugged_n", "{:.3f}", 1, "H x D_d, H in km (Strahler 1958)"),
    ("Average slope S_avg (%)", "s_avg", "{:.3f}", 100, "Fall along L / L"),
    ("10-85 slope S_1085 (%)", "s_1085", "{:.3f}", 100, "Between 10 % and 85 % of L"),
    ("Equal-area slope S_e (%)", "s_ea", "{:.3f}", 100, "Equal-area line along L"),
    ("Mean catchment slope S_m (%)", "s_mean_pc", "{:.2f}", 1, "Mean cell slope (Horn 1981)"),
]  # fmt: skip

TABLE_WIDTHS_CM = [6.2, 2.8, 7.5]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plan_extent(c: CatchmentResult, pad: float = 0.08) -> tuple:
    """(xmin, ymin, xmax, ymax) of the catchment and its flow path, padded
    on every side by ``pad`` of the larger dimension."""
    xs = [x for poly in c.polygons for x, _ in poly[0]] if c.polygons else []
    ys = [y for poly in c.polygons for _, y in poly[0]] if c.polygons else []
    for x, y in c.flow_path:
        xs.append(x)
        ys.append(y)
    if c.outlet_xy:
        xs.append(c.outlet_xy[0])
        ys.append(c.outlet_xy[1])
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    margin = pad * max(xmax - xmin, ymax - ymin, 1.0)
    return (xmin - margin, ymin - margin, xmax + margin, ymax + margin)


def _plan_height_cm(extent, extra_cm: float) -> float:
    xmin, ymin, xmax, ymax = extent
    plot_h = PLAN_PLOT_WIDTH_CM * (ymax - ymin) / max(xmax - xmin, 1e-9)
    return min(max(plot_h + extra_cm, 7.0), 23.0)


def _as_image(image):
    import matplotlib.image as mpimg

    if isinstance(image, (bytes, bytearray)):
        return mpimg.imread(io.BytesIO(bytes(image)), format="png")
    return np.asarray(image)


def nice_scale_length(map_width_m: float, fraction: float = 0.25) -> float:
    """Largest 1, 2 or 5 x 10^k metres not longer than ``fraction`` of the
    map width."""
    target = max(map_width_m * fraction, 1e-9)
    power = 10 ** np.floor(np.log10(target))
    for mult in (5, 2, 1):
        if mult * power <= target:
            return float(mult * power)
    return float(power)


def _scale_bar(ax, extent) -> None:
    from matplotlib.patches import Rectangle

    xmin, ymin, xmax, ymax = extent
    width, height = xmax - xmin, ymax - ymin
    length = nice_scale_length(width)
    x0 = xmin + 0.04 * width
    y0 = ymin + 0.05 * height
    bar_h = 0.012 * max(width, height)
    km = length >= 1000
    unit_len = 1000.0 if km else 1.0
    ax.add_patch(
        Rectangle(
            (x0 - 0.015 * width, y0 - 0.035 * height),
            length + 0.06 * width,
            bar_h + 0.075 * height,
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            zorder=20,
        )
    )
    for i in range(4):
        ax.add_patch(
            Rectangle(
                (x0 + i * length / 4, y0),
                length / 4,
                bar_h,
                facecolor=CHART_INK if i % 2 == 0 else "white",
                edgecolor=CHART_INK,
                linewidth=CHART_LINE_PT,
                zorder=21,
            )
        )
    for frac in (0.0, 0.5, 1.0):
        value = frac * length / unit_len
        text = f"{value:g}"
        if frac == 1.0:
            text += " km" if km else " m"
        ax.text(
            x0 + frac * length,
            y0 + bar_h + 0.008 * height,
            text,
            ha="center",
            va="bottom",
            zorder=22,
        )


def _north_arrow(ax) -> None:
    ax.annotate(
        "",
        xy=(0.95, 0.92),
        xytext=(0.95, 0.80),
        xycoords="axes fraction",
        arrowprops={
            "facecolor": CHART_INK,
            "edgecolor": "white",
            "linewidth": 0.5,
            "width": 3,
            "headwidth": 10,
            "headlength": 10,
        },
        zorder=22,
    )
    ax.text(
        0.95,
        0.935,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontweight="bold",
        zorder=22,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
    )


def _image_legend(fig, ax, style: ChartStyle, entries) -> None:
    """Legend below the map whose keys are the symbol images rendered by
    QGIS from the layer styles."""
    from matplotlib.image import BboxImage
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import Patch
    from matplotlib.transforms import Bbox, TransformedBbox

    class ImageKey(HandlerBase):
        def __init__(self, image):
            super().__init__()
            self.image = image

        def create_artists(
            self,
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        ):
            box = TransformedBbox(
                Bbox.from_bounds(-xdescent, -ydescent, width, height), trans
            )
            key = BboxImage(box, interpolation="bilinear")
            key.set_data(self.image)
            return [key]

    handles, labels, handler_map = [], [], {}
    for label, image in entries:
        handle = Patch()
        handles.append(handle)
        labels.append(label)
        handler_map[handle] = ImageKey(_as_image(image))
    if not handles:
        return
    kwargs = {
        "handler_map": handler_map,
        "prop": style.font_props(style="italic"),
        "ncols": 3,
        "frameon": False,
        "handlelength": 2.0,
        "handleheight": 1.0,
        "columnspacing": 1.5,
    }
    try:
        legend = fig.legend(handles, labels, loc="outside lower center", **kwargs)
    except (TypeError, ValueError):  # older matplotlib: below the axes
        legend = ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            **kwargs,
        )
    for text in legend.get_texts():
        text.set_color(CHART_INK)


def _figure_plan_rendered(pm: PlanMap, style: ChartStyle) -> io.BytesIO:
    rows = -(-len(pm.legend) // 3)
    height = _plan_height_cm(pm.extent, 1.8 + 0.55 * rows)
    xmin, ymin, xmax, ymax = pm.extent
    with style.context():
        fig = style.figure(height)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(
            _as_image(pm.image),
            extent=(xmin, xmax, ymin, ymax),
            origin="upper",
            interpolation="bilinear",
            zorder=0,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        _scale_bar(ax, pm.extent)
        _north_arrow(ax)
        if pm.attribution:
            ax.text(
                0.995,
                0.005,
                pm.attribution,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                zorder=22,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                    "pad": 1,
                },
            )
        style.axes(ax, grid=False)
        _image_legend(fig, ax, style, pm.legend)
        return style.png(fig)


def figure_plan(
    c: CatchmentResult,
    style: ChartStyle,
    plan_map: PlanMapFn | None = None,
    messages: list | None = None,
) -> tuple[io.BytesIO, PlanMap | None]:
    """The catchment layout map: rendered by ``plan_map`` (QGIS, with the
    layer styles) when given, otherwise drawn with matplotlib."""
    if plan_map is not None:
        try:
            pm = plan_map(c, PLAN_PLOT_WIDTH_CM, PLAN_DPI)
        except Exception as e:  # the report must still be written
            pm = None
            if messages is not None:
                messages.append(
                    f"Catchment {c.outlet_id}: the QGIS layout map could not be "
                    f"rendered ({e}); a simplified plan was drawn instead."
                )
        if pm is not None:
            return _figure_plan_rendered(pm, style), pm
    return _figure_plan_simple(c, style), None


def _figure_plan_simple(c: CatchmentResult, style: ChartStyle) -> io.BytesIO:
    from matplotlib.collections import LineCollection
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    extent = plan_extent(c)
    height = _plan_height_cm(extent, 1.8)
    with style.context():
        fig = style.figure(height)
        ax = fig.add_subplot(1, 1, 1)
        for poly in c.polygons:
            verts, codes = [], []
            for ring in poly:
                verts.extend(ring)
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ring) - 2))
                codes.append(Path.CLOSEPOLY)
            ax.add_patch(
                PathPatch(
                    Path(verts, codes),
                    facecolor=C_YELLOW,
                    alpha=0.25,
                    edgecolor="none",
                    zorder=1,
                )
            )
            ax.add_patch(
                PathPatch(
                    Path(verts, codes),
                    facecolor="none",
                    edgecolor=C_RED,
                    linewidth=1.5,
                    zorder=3,
                )
            )
        ax.plot([], [], color=C_RED, linewidth=1.5, label="Catchment boundary")
        streams = c.detail.get("streams")
        if streams is not None and len(streams):
            orders = streams[:, 4]
            max_order = max(orders.max(), 1)
            segs = streams[:, :4].reshape(-1, 2, 2)
            widths = 0.5 + 1.5 * (orders - 1) / max(max_order - 1, 1)
            ax.add_collection(
                LineCollection(
                    segs, colors=C_BLUE, linewidths=widths, capstyle="round", zorder=4
                )
            )
            ax.plot(
                [], [], color=C_BLUE, linewidth=1.2, label="Streams (width by order)"
            )
        if len(c.flow_path) > 1:
            fx, fy = zip(*c.flow_path, strict=True)
            ax.plot(
                fx,
                fy,
                color=C_AMBER,
                linewidth=1.8,
                label="Longest flow path",
                zorder=5,
            )
        if c.outlet_xy:
            ax.plot(
                *c.outlet_xy,
                marker="v",
                markersize=8,
                color=CHART_INK,
                linestyle="none",
                label="Outlet",
                zorder=6,
            )
        cen = c.detail.get("centroid")
        if cen:
            ax.plot(
                *cen,
                marker="P",
                markersize=8,
                color=C_MAGENTA,
                markeredgecolor=CHART_INK,
                markeredgewidth=0.75,
                linestyle="none",
                label="Centroid",
                zorder=6,
            )
        for x, y, name in c.detail.get("labels") or []:
            ax.text(
                x, y, name, ha="center", va="center", fontsize=7, color=CHART_INK,
                zorder=7,
            )  # fmt: skip
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        _scale_bar(ax, extent)
        _north_arrow(ax)
        style.axes(ax)
        legend = style.legend(
            ax,
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor=CHART_INK,
            framealpha=0.9,
        )
        legend.get_frame().set_linewidth(CHART_LINE_PT)
        legend.set_zorder(30)
        return style.png(fig)


def figure_profile(c: CatchmentResult, style: ChartStyle) -> io.BytesIO | None:
    x = np.asarray(c.detail.get("profile_x_m", []), dtype=float)
    z = np.asarray(c.detail.get("profile_z_m", []), dtype=float)
    if x.size < 2 or np.isnan(z).all():
        return None
    a = c.attributes
    with style.context():
        fig = style.figure(8.5)
        ax = fig.add_subplot(1, 1, 1)
        xk = x / 1000.0
        ax.plot(
            xk,
            z,
            color=CHART_INK,
            linewidth=1.0,
            linestyle="--",
            label="Natural ground (longest flow path)",
        )
        length = x[-1]
        if a.get("s_1085") is not None:
            x10, x85 = 0.10 * length, 0.85 * length
            z10 = float(np.interp(x10, x, z))
            z85 = float(np.interp(x85, x, z))
            ax.plot(
                [x10 / 1000, x85 / 1000],
                [z10, z85],
                color=C_AMBER,
                linewidth=1.5,
                linestyle="--",
                marker="o",
                markersize=4,
                label=f"10-85 slope {100 * a['s_1085']:.2f} %",
            )
        if a.get("s_ea") is not None:
            ax.plot(
                [0, length / 1000],
                [z[0], z[0] + a["s_ea"] * length],
                color=C_GREEN,
                linewidth=1.5,
                linestyle=":",
                label=f"Equal-area slope {100 * a['s_ea']:.2f} %",
            )
        ax.set_xlabel("Distance upstream from outlet (km)")
        ax.set_ylabel("Elevation (m)")
        ax.set_xlim(0, xk[-1])
        ax.ticklabel_format(useOffset=False, style="plain", axis="y")
        style.axes(ax)
        style.legend(ax, loc="upper left")
        return style.png(fig)


def figure_hypsometric(c: CatchmentResult, style: ChartStyle) -> io.BytesIO | None:
    curve = c.hypsometric_curve
    if not curve:
        return None
    rel_a = [p["rel_area"] for p in curve]
    rel_h = [p["rel_height"] for p in curve]
    with style.context():
        fig = style.figure(9.0)
        ax = fig.add_subplot(1, 1, 1)
        ax.fill_between(rel_a, rel_h, color=C_RED, alpha=0.12, linewidth=0)
        ax.plot(rel_a, rel_h, color=C_RED, linewidth=1.5, label="Hypsometric curve")
        ax.plot(
            [0, 1],
            [1, 0],
            color=CHART_INK,
            linewidth=CHART_LINE_PT,
            linestyle="--",
            label="Uniform slope (HI = 0.5)",
        )
        hi = c.attributes.get("hyps_int")
        if hi is not None:
            ax.text(
                0.97,
                0.93,
                f"HI = {hi:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Relative area above height, a/A")
        ax.set_ylabel("Relative height, h/H")
        style.axes(ax)
        style.legend(ax, loc="lower left")
        return style.png(fig)


def figure_histogram(c: CatchmentResult, style: ChartStyle) -> io.BytesIO | None:
    counts = np.asarray(c.detail.get("elev_hist_counts", []), dtype=float)
    edges = np.asarray(c.detail.get("elev_hist_edges", []), dtype=float)
    if counts.size == 0 or counts.sum() == 0:
        return None
    pct = 100.0 * counts / counts.sum()
    with style.context():
        fig = style.figure(8.0)
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(
            edges[:-1],
            pct,
            width=np.diff(edges),
            align="edge",
            color=C_RED,
            edgecolor=CHART_INK,
            linewidth=CHART_LINE_PT,
        )
        ax.set_xlabel("Elevation (m)")
        ax.set_ylabel("Share of catchment area (%)")
        ax.ticklabel_format(useOffset=False, style="plain", axis="x")
        style.axes(ax)
        return style.png(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


# value of each PARAMETER_DEFINITIONS row for a single catchment:
# name -> (attribute key, format, scale)
PARAMETER_VALUE_KEYS = {
    "Contributing area": ("area_km2", "{:.4f}", 1),
    "Polygon area": ("poly_km2", "{:.4f}", 1),
    "Perimeter": ("perim_km", "{:.3f}", 1),
    "Relief": ("relief_m", "{:.2f}", 1),
    "Elevation gain": ("elev_gain", "{:.2f}", 1),
    "Longest flow path": ("lfp_km", "{:.3f}", 1),
    "Centroid distance": ("lca_km", "{:.3f}", 1),
    "Basin length": ("lb_km", "{:.3f}", 1),
    "Average slope": ("s_avg", "{:.4f}", 1),
    "10-85 slope": ("s_1085", "{:.4f}", 1),
    "Equal-area slope": ("s_ea", "{:.4f}", 1),
    "Mean catchment slope": ("s_mean_pc", "{:.2f}", 1),
    "Form factor": ("form_f", "{:.3f}", 1),
    "Circularity ratio": ("circ_r", "{:.3f}", 1),
    "Elongation ratio": ("elong_r", "{:.3f}", 1),
    "Compactness coefficient": ("gravel_kc", "{:.3f}", 1),
    "Hypsometric integral": ("hyps_int", "{:.3f}", 1),
    "Relief ratio": ("relief_r", "{:.4f}", 1),
    "Melton ratio": ("melton_r", "{:.3f}", 1),
    "Ruggedness number": ("rugged_n", "{:.3f}", 1),
    "Stream order": ("strm_ord", "{:d}", None),
    "Bifurcation ratio": ("bif_r", "{:.2f}", 1),
    "Drainage density": ("dd_kmkm2", "{:.3f}", 1),
    "Stream frequency": ("strm_freq", "{:.3f}", 1),
    "Length of overland flow": ("lo_km", "{:.3f}", 1),
    "Constant of channel maintenance": ("c_maint", "{:.3f}", 1),
    "Sinuosity": ("sinuosity", "{:.3f}", 1),
    "Channel start on L": ("ch_head_m", "{:.1f}", 1),
    "Waterway length": ("chan_len_m", "{:.1f}", 1),
    "Mean overland flow length": ("lo_mean_m", "{:.1f}", 1),
    "SWMM width - flow length": ("w_flow_m", "{:.1f}", 1),
    "SWMM width - channel": ("w_chan_m", "{:.1f}", 1),
    "Shape factor": ("gu_x", "{:.3f}", 1),
    "Skew factor": ("gu_z", "{:.3f}", 1),
    "SWMM width - Guo & Urbonas": ("w_gu_m", "{:.1f}", 1),
    "Width ratio": ("w_ratio", "{:.2f}", 1),
}


def parameter_rows(catchments) -> list[list[str]]:
    """Parameter (symbol, unit) | Value | Definition / equation (source), with
    values when a single catchment is reported (otherwise '-'; the values per
    catchment are in Section 5)."""
    single = catchments[0].attributes if len(catchments) == 1 else None
    rows = []
    for name, symbol, definition, source in PARAMETER_DEFINITIONS:
        label = name if symbol in ("-", "") else f"{name}, {symbol}"
        value = "-"
        if single is not None and name in PARAMETER_VALUE_KEYS:
            key, fmt, scale = PARAMETER_VALUE_KEYS[name]
            value = format_value(single.get(key), fmt, scale)
        text = definition if source in ("-", "") else f"{definition} ({source})"
        rows.append([label, value, text])
    return rows


def title_details(ctx: dict, catchments) -> list[tuple[str, str]]:
    from datetime import date

    site = ctx.get("project_name")
    if not site and len(catchments) == 1:
        site = f"Catchment {catchments[0].outlet_id}"
    tool = ctx.get("tool", REPORT_TITLE)
    version = ctx.get("version")
    return [
        ("Project number", ctx.get("project_number") or "-"),
        ("Project / site", site or "-"),
        ("Prepared by", ctx.get("prepared_by") or "Mayim Consulting"),
        ("Tool and version", f"{tool}, version {version}" if version else tool),
        ("Software", ctx.get("software") or "-"),
        ("Report date", ctx.get("report_date") or date.today().isoformat()),
        ("Document status", ctx.get("status") or "Draft"),
    ]


def contour_text(minor_interval) -> str:
    if not minor_interval:
        return "none"
    return f"{minor_interval:g} m minor, {5 * minor_interval:g} m major"


def is_network_mode(catchments) -> bool:
    return any(c.attributes.get("sub_type") for c in catchments)


def network_overview(catchments) -> CatchmentResult:
    """A stand-in catchment covering every sub-catchment, for the overview
    layout map."""
    polygons = [poly for c in catchments for poly in c.polygons]
    streams = [
        c.detail["streams"]
        for c in catchments
        if c.detail.get("streams") is not None and len(c.detail["streams"])
    ]
    root = next(
        (c for c in catchments if not c.attributes.get("ds_stream")), catchments[0]
    )
    return CatchmentResult(
        outlet_id="all sub-catchments",
        kind="point",
        src_fid=-1,
        attributes={},
        polygons=polygons,
        flow_path=[],
        outlet_xy=root.outlet_xy,
        hypsometric_curve=[],
        detail={
            "streams": np.vstack(streams) if streams else None,
            "labels": [
                (c.attributes["cx"], c.attributes["cy"], c.outlet_id)
                for c in catchments
            ],
            "label_catchments": True,
            "overview": True,
        },
    )


def link_text(a: dict) -> str:
    """'drains to node A4 and on into stream A3' for a sub-catchment."""
    node = a.get("node") or "-"
    into = a.get("ds_stream")
    return f"node {node}" + (f", then stream {into}" if into else " (the outlet)")


def width_note(ctx: dict) -> str:
    cap = ctx.get("max_overland_m") or 0
    return (
        "SWMM width W cannot be measured, so three published estimates are "
        "given: from the mean overland flow length (W_f = A / L_om"
        + (f", L_om capped at {cap:g} m" if cap else "")
        + "), from the waterway (W_c = 2 L_w, or L_w for one bank of a "
        "stream) and by Guo & Urbonas (2009) (W_g). W_f and W_c treat the "
        "overland flow as running across to the waterway; W_g converts the "
        "whole sub-catchment into an equivalent plane that conserves its area "
        "and the fall along the waterway, and gives a much narrower, longer "
        "plane for elongated shapes. Where W_g / W_f is far from 1 the shape is "
        "far from a rectangle; choose the estimate that matches how runoff "
        "reaches the drainage system, and calibrate W where flow records exist "
        "(runoff peaks are sensitive to W)."
    )


def write_report_docx(
    path: str,
    catchments: Sequence[CatchmentResult],
    context: dict,
    include_figures: bool = True,
    plan_map: PlanMapFn | None = None,
) -> list[str]:
    """Writes the annexure report. ``context`` describes the run (see the
    QGIS wrapper): tool/version/run time, inputs, settings, outputs and
    processing warnings. ``plan_map`` renders each catchment's layout map
    in QGIS with the layer styles (OSM background, study area, contours);
    None draws a simplified plan with matplotlib. Returns
    messages for the user (e.g. a background that could not be added or a
    substituted font)."""
    d = _Doc(house_style=True)
    ctx = context or {}
    messages: list[str] = []
    style = None
    osm_used = False
    if include_figures:
        try:
            import matplotlib  # noqa: F401

            style = ChartStyle()
            messages.extend(style.warnings)
        except ImportError:
            include_figures = False
    dem = ctx.get("dem", {})
    mode = ctx.get("mode", "total")

    n_c = len(catchments)
    d.title_page(
        REPORT_TITLE,
        purpose=(
            "This report documents the delineation and characterisation of "
            f"{'the catchment' if n_c == 1 else f'{n_c} catchments'} from a "
            "digital elevation model: the catchment boundaries, their "
            "geometric, relief, shape and drainage-network parameters and "
            "times of concentration, as input to hydrological assessment and "
            "modelling."
        ),
        details=title_details(ctx, catchments),
        subtitle=ctx.get("project_name") or None,
    )

    # 1. Summary ---------------------------------------------------------------
    d.heading("1. Summary", 2)
    n = len(catchments)
    d.para(
        f"This annexure documents the delineation and characterisation of "
        f"{n} catchment{'s' if n != 1 else ''} from a digital elevation model "
        f"(DEM) with a {dem.get('cell_x', 0):g} m x {dem.get('cell_y', 0):g} m "
        f"cell size, using the {ctx.get('tool', 'Catchments: 1. Delineation')} tool "
        f"(Mayim Tools, version {ctx.get('version', '-')}) in "
        f"{'total' if mode == 'total' else 'incremental'} mode. For each "
        "catchment it records the inputs and method, the geometric, relief, "
        "shape, drainage-network and time-of-concentration parameters, a "
        "geomorphological description and general considerations for "
        "hydrological assessment and modelling."
    )
    network_mode = is_network_mode(catchments)
    if network_mode:
        n_streams = sum(1 for c in catchments if c.attributes["sub_type"] == "stream")
        n_heads = n - n_streams
        total = sum(c.attributes["area_km2"] for c in catchments)
        d.para(
            f"In incremental mode the catchment of the final outlet "
            f"({total:.3f} km²) was divided along the stream network into {n} "
            f"sub-catchments: {n_streams} draining to a stream"
            + (
                " (split along the stream centreline into left bank a and right "
                "bank b)"
                if ctx.get("split_banks")
                else ""
            )
            + f" and {n_heads} draining to a channel head. The main stem is A, "
            "numbered upstream from the outlet (A1, A2 ...); branches are named "
            "after the stream they flow into plus a letter (A3A1 ...). Each "
            "sub-catchment is named after its stream, and drains to the node "
            "and stream given in the table."
        )
        d.caption("Table", "Summary of the sub-catchments and their links")
        d.table(
            [
                "Sub-catchment",
                "Type",
                "Drains to",
                "Area (km²)",
                "L (km)",
                "S_1085 (%)",
                "L_ch (m)",
            ],
            [
                [
                    c.outlet_id,
                    c.attributes["sub_type"]
                    + (
                        f", {'left' if c.attributes['bank'] == 'a' else 'right'} bank"
                        if c.attributes.get("bank")
                        else ""
                    ),
                    link_text(c.attributes),
                    format_value(c.attributes["area_km2"], "{:.3f}", 1),
                    format_value(c.attributes["lfp_km"], "{:.3f}", 1),
                    format_value(c.attributes["s_1085"], "{:.2f}", 100),
                    format_value(c.attributes.get("ch_head_m"), "{:.0f}", 1),
                ]
                for c in catchments
            ],
            widths_cm=[2.4, 2.8, 4.0, 1.9, 1.6, 1.9, 2.4],
        )
    else:
        d.caption("Table", "Summary of delineated catchments")
    if not network_mode:
        d.table(
            [
                "Catchment",
                "Area (km²)",
                "L (km)",
                "S_1085 (%)",
                "S_m (%)",
                "Order",
                "L_ch (m)",
            ],
            [
                [
                    c.outlet_id,
                    format_value(c.attributes["area_km2"], "{:.3f}", 1),
                    format_value(c.attributes["lfp_km"], "{:.3f}", 1),
                    format_value(c.attributes["s_1085"], "{:.2f}", 100),
                    format_value(c.attributes["s_mean_pc"], "{:.1f}", 1),
                    format_value(c.attributes["strm_ord"], "{:d}", None),
                    format_value(c.attributes.get("ch_head_m"), "{:.0f}", 1),
                ]
                for c in catchments
            ],
            widths_cm=[2.8, 2.2, 2.0, 2.2, 2.0, 1.8, 3.5],
        )

    # 2. Tool and method -------------------------------------------------------
    d.heading("2. Tool and method", 2)
    d.para(
        "The catchments were delineated with a D8 (eight-direction) flow model "
        "(O'Callaghan & Mark 1984), in which every DEM cell drains to the "
        "steepest-descent neighbour of its eight neighbours. The processing "
        "steps were:"
    )
    for step in [
        "The D8 flow-direction (pointer) raster was decoded and checked; codes "
        "that are not valid D8 directions were treated as NoData and reported.",
        "Flow accumulation (number of upstream cells) was computed in "
        "topological order from the pointer.",
        "Pour points were moved to the cell with the highest flow accumulation "
        f"within {ctx.get('snap_radius_cells', '-')} cells, so that they sit "
        "on the drainage line. Pour lines were rasterised as a 4-connected "
        "chain of cells, so no D8 flow path can cross a line without being "
        "counted.",
        "Every cell was traced downstream to the first outlet cell it reaches. "
        + (
            "In total mode each catchment includes everything upstream of its "
            "outlet, so nested catchments overlap."
            if mode == "total"
            else "In incremental mode the outlet is the final outlet of the "
            "system. The stream network upstream of it was extracted (cells with "
            "at least the stream threshold area upstream, split into streams at "
            "every confluence and stream breakpoint), and every cell was assigned "
            "to the first stream it meets: each stream's sub-catchment drains to "
            "the node at its lower end, and the area above each channel head "
            "forms a channel head sub-catchment."
            + (
                " Each stream sub-catchment was then cut along its stream "
                "centreline (through the stream cell centres, extended at both "
                "ends to the boundary) into its left bank a and right bank b, "
                "looking downstream; cells on the line went with the left bank. "
                "Pieces smaller than "
                f"{ctx.get('split_min_cells', 10)} cells, or narrower on average "
                f"than {ctx.get('split_min_width', 2.0):g} cells (area / length "
                "along the stream), joined the neighbouring piece they share the "
                "longest boundary with, and a stream whose whole bank was below "
                "this tolerance was not split."
                if ctx.get("split_banks")
                else ""
            )
        ),
        "The outline was traced along cell edges. Interior holes (cells inside "
        "the outline that do not drain to the outlet) were filled and their "
        "area reported. Cells touching only at a corner form separate polygon "
        "parts, and the geometry was checked and repaired to a valid "
        "(multi)polygon.",
        "The DEM was clipped to the catchment outlines, and the parameters, "
        "longest flow path, stream network and figures in this annexure were "
        "derived from the DEM and flow network.",
    ] + (
        [
            "Stream network reaches longer than the maximum stream length "
            f"({ctx['max_stream_length_m']:g} m) were split at stream "
            "breakpoints, measured upstream from each reach's lower end, where "
            "the rest of the reach was longer than "
            f"{ctx.get('min_final_reach_pct', 10.0):g} % of the maximum."
        ]
        if ctx.get("max_stream_length_m")
        else []
    ):
        d.numbered(step)

    # 3. Inputs ---------------------------------------------------------------
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
            [
                "DEM dimensions",
                f"{dem.get('cols', '-')} columns x {dem.get('rows', '-')} rows",
            ],
            ["D8 pointer", ctx.get("pointer", {}).get("path", "-")],
            ["Pointer encoding", ctx.get("pointer", {}).get("encoding", "-")],
            ["Outlet layer", ctx.get("outlets", {}).get("layer", "-")],
            [
                "Outlet features",
                str(ctx.get("outlets", {}).get("n_features", "-")),
            ],
            ["Outlet name field", ctx.get("outlets", {}).get("id_field") or "-"],
            ["Pour point snap radius", f"{ctx.get('snap_radius_cells', '-')} cells"],
            ["Catchment mode", "Total" if mode == "total" else "Incremental"],
            *(
                [
                    [
                        "Minimum final reach length",
                        f"{ctx.get('min_final_reach_pct', 10.0):g} % of the maximum "
                        "stream length",
                    ],
                    [
                        "Split along the stream centreline",
                        (
                            (
                                f"yes (tolerance {ctx.get('split_min_cells', 10)} "
                                f"cells, mean width {ctx.get('split_min_width', 2.0):g} "
                                "cells)"
                            )
                            if ctx.get("split_banks")
                            else "no"
                        ),
                    ],
                ]
                if mode != "total"
                else []
            ),
            [
                "Stream threshold",
                f"{ctx.get('stream_threshold_km2', '-')} km² upstream area",
            ],
            ["Project number", ctx.get("project_number") or "-"],
            ["Project / site", ctx.get("project_name") or "-"],
            [
                "Maximum stream length",
                (
                    f"{ctx['max_stream_length_m']:g} m "
                    f"({ctx.get('n_breakpoints') if ctx.get('n_breakpoints') is not None else '-'}"
                    " stream breakpoints)"
                    if ctx.get("max_stream_length_m")
                    else "none"
                ),
            ],
            ["Study area on the layout map", ctx.get("study_area") or "no"],
            ["Contours on the layout map", contour_text(ctx.get("contour_interval"))],
            ["Run", ctx.get("run_time", "-")],
        ],
        widths_cm=[5.5, 11.0],
    )
    d.para(
        "The DEM is assumed to be hydrologically conditioned (pits filled or "
        "breached, flats resolved) and in a projected coordinate system in "
        "metres. All lengths, areas and slopes are in that system."
    )

    # 4. Parameters ------------------------------------------------------------
    d.heading("4. Parameters and calculations", 2)
    if n_c > 1:
        d.para("The values for each catchment are given in Section 5.")
    d.caption("Table", "Parameters, values and definitions")
    d.table(
        ["Parameter (symbol, unit)", "Value", "Definition / equation (source)"],
        parameter_rows(catchments),
        widths_cm=[6.2, 2.8, 7.5],
    )
    d.para(
        "Times of concentration are not calculated here: the Time of "
        "Concentration tool takes the catchments, the longest flow paths and "
        "the DEM, with the overland, shallow concentrated and channel flow "
        "parameters set per catchment. L_ch marks where the channel starts on "
        "L for that tool."
    )
    d.para(width_note(ctx))

    # 5. Results ---------------------------------------------------------------
    d.heading("5. Catchment results", 2)
    first = 1
    if network_mode and include_figures:
        d.heading("5.1 Overview of the sub-catchments", 3)
        overview = network_overview(catchments)
        plan, pm = figure_plan(overview, style, plan_map, messages)
        on_osm = bool(pm is not None and pm.attribution)
        osm_used = osm_used or on_osm
        d.picture(
            plan,
            "Layout of the sub-catchments, stream network and nodes"
            + (" (background: OpenStreetMap)" if on_osm else ""),
        )
        first = 2
    word = "Sub-catchment" if network_mode else "Catchment"
    detailed = catchments
    if network_mode and not ctx.get("detail_sections"):
        # compact: the key parameters of every sub-catchment in two tables
        d.heading(f"5.{first} Sub-catchment parameters", 3)
        d.para(
            "The key parameters of each sub-catchment are listed below; every "
            "parameter of every sub-catchment is in the catchment parameters "
            "CSV and the catchment layer. Slopes are along the longest flow "
            "path L, which for a stream sub-catchment ends at the node it "
            "drains to."
        )
        d.caption("Table", "Sub-catchment geometry and slopes")
        d.table(
            [
                "Sub-catchment",
                "A (km²)",
                "L (km)",
                "L_ca (km)",
                "z_out (m)",
                "z_max (m)",
                "S_avg (%)",
                "S_1085 (%)",
                "S_e (%)",
                "S_m (%)",
            ],
            [
                [
                    c.outlet_id,
                    format_value(c.attributes["area_km2"], "{:.4f}", 1),
                    format_value(c.attributes["lfp_km"], "{:.3f}", 1),
                    format_value(c.attributes["lca_km"], "{:.3f}", 1),
                    format_value(c.attributes["z_outlet"], "{:.1f}", 1),
                    format_value(c.attributes["z_max"], "{:.1f}", 1),
                    format_value(c.attributes["s_avg"], "{:.2f}", 100),
                    format_value(c.attributes["s_1085"], "{:.2f}", 100),
                    format_value(c.attributes["s_ea"], "{:.2f}", 100),
                    format_value(c.attributes["s_mean_pc"], "{:.1f}", 1),
                ]
                for c in catchments
            ],
            widths_cm=[2.3, 1.6, 1.4, 1.5, 1.6, 1.6, 1.6, 1.7, 1.5, 1.4],
        )
        d.caption("Table", "Sub-catchment channel start and centroids")
        d.table(
            [
                "Sub-catchment",
                "Channel start on L, L_ch (m)",
                "Centroid X",
                "Centroid Y",
            ],
            [
                [
                    c.outlet_id,
                    format_value(c.attributes.get("ch_head_m"), "{:.1f}", 1),
                    format_value(c.attributes["cx"], "{:.1f}", 1),
                    format_value(c.attributes["cy"], "{:.1f}", 1),
                ]
                for c in catchments
            ],
            widths_cm=[3.5, 5.0, 4.0, 4.0],
        )
        d.caption("Table", "Sub-catchment SWMM width estimates")
        d.table(
            [
                "Sub-catchment",
                "A (ha)",
                "L_w (m)",
                "L_om (m)",
                "W_f (m)",
                "W_c (m)",
                "Z",
                "W_g (m)",
                "W_g / W_f",
            ],
            [
                [
                    c.outlet_id,
                    format_value(c.attributes["area_ha"], "{:.2f}", 1),
                    format_value(c.attributes.get("chan_len_m"), "{:.1f}", 1),
                    format_value(c.attributes.get("lo_mean_m"), "{:.1f}", 1),
                    format_value(c.attributes.get("w_flow_m"), "{:.1f}", 1),
                    format_value(c.attributes.get("w_chan_m"), "{:.1f}", 1),
                    format_value(c.attributes.get("gu_z"), "{:.2f}", 1),
                    format_value(c.attributes.get("w_gu_m"), "{:.1f}", 1),
                    format_value(c.attributes.get("w_ratio"), "{:.2f}", 1),
                ]
                for c in catchments
            ],
            widths_cm=[2.3, 1.6, 1.6, 1.7, 1.7, 1.7, 1.2, 1.7, 1.6],
        )
        detailed = []
    for i, c in enumerate(detailed, start=first):
        a = c.attributes
        text = describe_catchment(a, c.detail)
        d.heading(f"5.{i} {word} {c.outlet_id}", 3)
        if network_mode:
            d.para(
                f"{word} {c.outlet_id} is a {a['sub_type']} sub-catchment"
                + (
                    f" ({'left' if a['bank'] == 'a' else 'right'} bank of stream "
                    f"{a['stream']})"
                    if a.get("bank")
                    else ""
                )
                + f" and drains to {link_text(a)}."
            )
        d.para(text["overview"])
        d.label("Geomorphological description")
        for paragraph in text["geomorphology"]:
            lead, _, rest = paragraph.partition(". ")
            d.para(rest, bold_lead=lead + ".")
        d.label("Hydrological assessment and modelling considerations")
        for point in text["modelling"]:
            d.bullet(point)

        rows, groups = [], []
        for title, items in RESULT_GROUPS:
            groups.append(len(rows))
            rows.append([title, ""])
            for label, key, fmt, scale in items:
                rows.append([label, format_value(a.get(key), fmt, scale)])
        d.caption("Table", f"General parameters of catchment {c.outlet_id}")
        d.table(["Parameter", "Value"], rows, widths_cm=[10.0, 6.5], group_rows=groups)

        shape_rows = [
            [label, format_value(a.get(key), fmt, scale), definition]
            for label, key, fmt, scale, definition in SHAPE_ROWS
        ]
        shape_class = classify(a.get("elong_r"), ELONGATION_CLASSES)
        if shape_class:
            shape_rows.append(
                ["Shape class", shape_class, "From R_e (Schumm 1956 classes)"]
            )
        d.caption("Table", f"Shape parameters of catchment {c.outlet_id}")
        d.table(
            ["Parameter", "Value", "Definition"], shape_rows, widths_cm=TABLE_WIDTHS_CM
        )

        relief_rows = [
            [label, format_value(a.get(key), fmt, scale), definition]
            for label, key, fmt, scale, definition in RELIEF_ROWS
        ]
        d.caption("Table", f"Relief parameters of catchment {c.outlet_id}")
        d.table(
            ["Parameter", "Value", "Definition"], relief_rows, widths_cm=TABLE_WIDTHS_CM
        )

        # incremental mode keeps each sub-catchment short: the slope, aspect
        # and stream-order distributions, hypsometric curve and elevation
        # histogram are left out (all values are in the CSV outputs)
        if not network_mode:
            slope_rows = [
                [s["label"], f"{s['pct']:.1f}"]
                for s in c.detail.get("slope_classes", [])
            ]
            if slope_rows:
                d.caption(
                    "Table",
                    f"Slope class distribution, catchment {c.outlet_id} "
                    "(FAO 2006 classes)",
                )
                d.table(["Slope class", "Area (%)"], slope_rows, widths_cm=[10.0, 6.0])
            aspect_rows = [
                [s["label"], f"{s['pct']:.1f}"] for s in c.detail.get("aspect", [])
            ]
            if aspect_rows:
                d.caption("Table", f"Aspect distribution, catchment {c.outlet_id}")
                d.table(["Aspect", "Area (%)"], aspect_rows, widths_cm=[10.0, 6.0])
            order_rows = [
                [str(o["order"]), str(o["n_segments"]), f"{o['length_km']:.3f}"]
                for o in c.detail.get("stream_orders", [])
            ]
            if order_rows:
                d.caption("Table", f"Stream network by Strahler order, {c.outlet_id}")
                d.table(
                    ["Order u", "Segments N_u", "Length (km)"],
                    order_rows,
                    widths_cm=[5.0, 5.5, 5.5],
                )

        if include_figures:
            plan, pm = figure_plan(c, style, plan_map, messages)
            on_osm = bool(pm is not None and pm.attribution)
            osm_used = osm_used or on_osm
            d.picture(
                plan,
                f"{word} {c.outlet_id}: layout of the {word.lower()}, stream "
                "network, longest flow path, outlet and centroid"
                + (" (background: OpenStreetMap)" if on_osm else ""),
            )
            makers = [(figure_profile, "longitudinal profile of the longest flow path")]
            if not network_mode:
                makers += [
                    (figure_hypsometric, "hypsometric curve"),
                    (figure_histogram, "elevation distribution"),
                ]
            for maker, cap in makers:
                stream = maker(c, style)
                if stream is not None:
                    d.picture(stream, f"{word} {c.outlet_id}: {cap}")

    # 6. Outputs ----------------------------------------------------------------
    outputs = ctx.get("outputs", [])
    if outputs:
        d.heading("6. Outputs", 2)
        d.caption("Table", "Files produced by the tool")
        d.table(["Output", "Location"], [list(o) for o in outputs], widths_cm=[5, 11])

    # 7. Notes -----------------------------------------------------------------
    d.heading("7. Notes and limitations", 2)
    for note in [
        "Results are only as good as the DEM. Its resolution, vertical "
        "accuracy, vegetation and building artefacts, and hydrological "
        "conditioning all control the flow directions and hence the boundary.",
        "A bare-earth DEM does not contain culverts, bridges, pipes or "
        "diversions. Where these exist, check the boundary against site "
        "information.",
        "D8 routing sends all flow from a cell to one neighbour and cannot "
        "represent flow splitting (e.g. on fans or flat ground).",
        "Stream network parameters depend on the chosen stream threshold and "
        "the DEM resolution.",
        "Time-of-concentration values are empirical estimates for guidance; "
        "the design method and parameters should follow the governing local "
        "guideline.",
    ] + (
        [
            "Background map data © OpenStreetMap contributors, available under "
            "the Open Database License (ODbL); see openstreetmap.org/copyright."
        ]
        if osm_used
        else []
    ):
        d.bullet(note)
    warnings = (
        list(ctx.get("warnings", []))
        + list(ctx.get("map_messages") or [])
        + list(messages)
    )
    if warnings:
        d.para("Processing messages recorded during this run:")
        for w in dict.fromkeys(warnings):
            d.bullet(w)

    # 8. References --------------------------------------------------------------
    d.heading("8. References", 2)
    for ref in REFERENCES:
        d.para(ref)

    d.doc.save(path)
    return list(dict.fromkeys(messages))  # unique, in order
