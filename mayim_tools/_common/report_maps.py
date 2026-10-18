"""
Report layout maps in the Mayim house style (report and figure standard,
section 3) - the frame around a map rendered elsewhere (by QGIS, with the
layer QML styles): Easting / Northing axes, legend below the map in three
columns, scale bar, north arrow and the background attribution. Zero QGIS
dependency; matplotlib's object API only. The QGIS side (rendering, legend
keys, OpenStreetMap) is in qgis_maps.py.

Moved here from hydrology/catchment_delineation (report.py) for reuse.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .report_figures import CHART_INK, CHART_LINE_PT, ChartStyle


@dataclass
class PlanMap:
    """A layout map rendered elsewhere (by QGIS, with the layer styles): the
    map image, the extent it covers (xmin, ymin, xmax, ymax), legend entries
    [(label, symbol image)] and an attribution line for the background
    (None when there is no background)."""

    image: object  # PNG bytes or an RGB(A) array
    extent: tuple[float, float, float, float]
    legend: list = field(default_factory=list)
    attribution: str | None = None


# (item, plot width in cm, dpi) -> PlanMap or None, supplied by a QGIS wrapper.
PlanMapFn = Callable[[object, float, float], "PlanMap | None"]
PLAN_PLOT_WIDTH_CM = 14.0  # map frame width inside the 16.5 cm figure
PLAN_DPI = 150  # map render resolution (symbol sizes stay true to the QML)
OSM_ATTRIBUTION = "© OpenStreetMap contributors"
OSM_LICENCE_NOTE = (
    "Background map data © OpenStreetMap contributors, available under the "
    "Open Database License (ODbL); see openstreetmap.org/copyright."
)


def plan_height_cm(extent, extra_cm: float) -> float:
    xmin, ymin, xmax, ymax = extent
    plot_h = PLAN_PLOT_WIDTH_CM * (ymax - ymin) / max(xmax - xmin, 1e-9)
    return min(max(plot_h + extra_cm, 7.0), 23.0)


def as_image(image):
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


def scale_bar(ax, extent) -> None:
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


def north_arrow(ax) -> None:
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


def image_legend(fig, ax, style: ChartStyle, entries) -> None:
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
        handler_map[handle] = ImageKey(as_image(image))
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


def figure_rendered_map(pm: PlanMap, style: ChartStyle) -> io.BytesIO:
    """The map image in the house map frame: Easting / Northing axes, scale
    bar, north arrow, attribution and the legend below the map."""
    rows = -(-len(pm.legend) // 3)
    height = plan_height_cm(pm.extent, 1.8 + 0.55 * rows)
    xmin, ymin, xmax, ymax = pm.extent
    with style.context():
        fig = style.figure(height)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(
            as_image(pm.image),
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
        scale_bar(ax, pm.extent)
        north_arrow(ax)
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
        image_legend(fig, ax, style, pm.legend)
        return style.png(fig)
