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
    ("Tc - Kirpich", "t_c (min)", "t_c = 0.0195 L^0.77 S_avg^-0.385 (L in m).", "Kirpich (1940)"),
    ("Tc - USBR", "t_c (min)", "t_c = 60 (0.87 L² / (1000 S_1085))^0.385 (L in km).", "USBR (1973)"),
    ("Tc - Bransby-Williams", "t_c (min)", "t_c = 58 L / (A^0.1 S_e^0.2) (L in km, A in km², S_e in m/km).", "Bransby-Williams (1922)"),
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
        ],
    ),
    (
        "Time of concentration",
        [
            ("Kirpich (min)", "tc_kirp_mn", "{:.1f}", 1),
            ("USBR (min)", "tc_usbr_mn", "{:.1f}", 1),
            ("Bransby-Williams (min)", "tc_bw_mn", "{:.1f}", 1),
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


def _tc_range(a):
    vals = [a.get(k) for k in ("tc_kirp_mn", "tc_usbr_mn", "tc_bw_mn")]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{min(vals):.1f} - {max(vals):.1f}"


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

    d.heading(REPORT_TITLE, 1)

    # 1. Summary ---------------------------------------------------------------
    d.heading("1. Summary", 2)
    n = len(catchments)
    d.para(
        f"This annexure documents the delineation and characterisation of "
        f"{n} catchment{'s' if n != 1 else ''} from a digital elevation model "
        f"(DEM) with a {dem.get('cell_x', 0):g} m x {dem.get('cell_y', 0):g} m "
        f"cell size, using the {ctx.get('tool', 'Catchment Delineation')} tool "
        f"(Mayim Tools, version {ctx.get('version', '-')}) in "
        f"{'total' if mode == 'total' else 'incremental'} mode. For each "
        "catchment it records the inputs and method, the geometric, relief, "
        "shape, drainage-network and time-of-concentration parameters, a "
        "geomorphological description and general considerations for "
        "hydrological assessment and modelling."
    )
    d.caption("Table", "Summary of delineated catchments")
    d.table(
        [
            "Catchment",
            "Area (km²)",
            "L (km)",
            "S_1085 (%)",
            "S_m (%)",
            "Order",
            "Tc range (min)",
        ],
        [
            [
                c.outlet_id,
                format_value(c.attributes["area_km2"], "{:.3f}", 1),
                format_value(c.attributes["lfp_km"], "{:.3f}", 1),
                format_value(c.attributes["s_1085"], "{:.2f}", 100),
                format_value(c.attributes["s_mean_pc"], "{:.1f}", 1),
                format_value(c.attributes["strm_ord"], "{:d}", None),
                _tc_range(c.attributes),
            ]
            for c in catchments
        ],
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
            else "In incremental mode each cell belongs only to the first outlet "
            "it meets, which gives non-overlapping sub-catchments linked to the "
            "outlet they drain into."
        ),
        "The outline was traced along cell edges. Interior holes (cells inside "
        "the outline that do not drain to the outlet) were filled and their "
        "area reported. Cells touching only at a corner form separate polygon "
        "parts, and the geometry was checked and repaired to a valid "
        "(multi)polygon.",
        "The DEM was clipped to the catchment outlines, and the parameters, "
        "longest flow path, stream network and figures in this annexure were "
        "derived from the DEM and flow network.",
    ]:
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
            [
                "Stream threshold",
                f"{ctx.get('stream_threshold_km2', '-')} km² upstream area",
            ],
            ["Run", ctx.get("run_time", "-")],
        ],
        widths_cm=[5.0, 11.0],
    )
    d.para(
        "The DEM is assumed to be hydrologically conditioned (pits filled or "
        "breached, flats resolved) and in a projected coordinate system in "
        "metres. All lengths, areas and slopes are in that system."
    )

    # 4. Parameters ------------------------------------------------------------
    d.heading("4. Parameters and calculations", 2)
    d.caption("Table", "Parameter definitions")
    d.table(
        ["Parameter", "Symbol (unit)", "Definition / equation", "Source"],
        [list(r) for r in PARAMETER_DEFINITIONS],
        widths_cm=[3.4, 2.6, 7.0, 3.0],
    )
    d.para(
        "Kirpich uses S_avg, USBR uses S_1085 and Bransby-Williams uses S_e, "
        "each with the longest flow path L. None of the three separates "
        "overland from channel flow, so they apply the whole path length as "
        "channel-type flow."
    )

    # 5. Results ---------------------------------------------------------------
    d.heading("5. Catchment results", 2)
    for i, c in enumerate(catchments, start=1):
        a = c.attributes
        text = describe_catchment(a, c.detail)
        d.heading(f"5.{i} Catchment {c.outlet_id}", 3)
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

        slope_rows = [
            [s["label"], f"{s['pct']:.1f}"] for s in c.detail.get("slope_classes", [])
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
                f"Catchment {c.outlet_id}: layout of the catchment, stream "
                "network, longest flow path, outlet and centroid"
                + (" (background: OpenStreetMap)" if on_osm else ""),
            )
            for maker, cap in (
                (figure_profile, "longitudinal profile of the longest flow path"),
                (figure_hypsometric, "hypsometric curve"),
                (figure_histogram, "elevation distribution"),
            ):
                stream = maker(c, style)
                if stream is not None:
                    d.picture(stream, f"Catchment {c.outlet_id}: {cap}")

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
    ]:
        d.bullet(note)
    warnings = ctx.get("warnings", [])
    if warnings:
        d.para("Processing messages recorded during this run:")
        for w in warnings:
            d.bullet(w)

    # 8. References --------------------------------------------------------------
    d.heading("8. References", 2)
    for ref in REFERENCES:
        d.para(ref)

    if osm_used:
        d.para(
            "Map background: © OpenStreetMap contributors, available under the "
            "Open Database Licence (openstreetmap.org/copyright)."
        )
    d.doc.save(path)
    return list(dict.fromkeys(messages))  # unique, in order
