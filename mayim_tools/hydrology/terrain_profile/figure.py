"""
Report-ready section figure for the Terrain Profile tool (no QGIS
dependency; matplotlib object API only).

Exactly 16.5 cm wide. The height follows from the vertical exaggeration:
the plot area is drawn to scale, so one metre up is ``ve`` times as long
on paper as one metre along. The figure height is kept between a minimum
and maximum: when the section is flat the elevation range is widened
(exaggeration kept); when it is steep the exaggeration is reduced to fit
and the value used is printed on the figure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mayim_tools._common.report_figures import (
    CHART_COLOURS,
    CHART_FONT,
    CHART_GRID_MAJOR,
    CHART_GRID_MINOR,
    CHART_GRID_MINOR_PT,
    CHART_INK,
    CHART_LINE_PT,
    CHART_SIZE,
    REPORT_FIG_WIDTH_CM,
    FigureFonts,
    new_report_figure,
)

from .core import EDGE_WATERLINE, SectionProfile, nice_step, point_at

DEFAULT_VE = 10.0
DEFAULT_MIN_HEIGHT_CM = 6.0
DEFAULT_MAX_HEIGHT_CM = 14.0

LEFT_CM = 1.6
RIGHT_CM = 0.4
TOP_CM = 1.0  # room for the property bracket and label
TOP_PLAIN_CM = 0.35  # without a property
TITLE_CM = 0.55  # 'Section A-A′' below the legend
AXIS_CM = 1.15  # tick labels and the axis title below the plot
LEGEND_ROW_CM = 0.55
NOTES_BASE_CM = 0.08  # gap below the last note line
BOTTOM_CM = AXIS_CM + LEGEND_ROW_CM + 0.55 + NOTES_BASE_CM
TEXT_WIDTH_CM = REPORT_FIG_WIDTH_CM - 0.2  # usable width for legend and notes

# Mayim report and figure standard, section 2: all figure text Arial Narrow
# 9 pt black (axis titles and the figure title bold, legend italic), black
# 0.75 pt border, major / minor gridlines, house colours, natural ground
# black 1 pt dashed, data lines 1.5 pt, fills at about 12 % opacity.
C_INK = CHART_INK
C_INK_2 = CHART_INK
C_GROUND = CHART_INK
C_GROUND_FILL = "#0000001F"  # black at 12 %
GROUND_LW = 1.0
GROUND_LS = (0, (4, 2))
FILL_ALPHA = 0.12
DATA_LW = 1.5
WATER_COLOURS = [CHART_COLOURS[i] for i in (4, 5, 3, 1, 6, 0)]  # blues first
C_PROPERTY = CHART_COLOURS[0]  # red
C_PLAN_LINE = CHART_COLOURS[4]  # blue (cyan is not legible on a map)
MAIN_SIZE = CHART_SIZE


class HouseFonts(FigureFonts):
    """Figure text of the report standard: Arial Narrow 9 pt throughout
    (Arial, then Liberation Sans Narrow / Liberation Sans when it is not
    installed - reported in ``warnings``)."""

    def __init__(self):
        super().__init__(main=CHART_FONT, minor=CHART_FONT)

    def main_kw(self, size: float = CHART_SIZE, **kw) -> dict:
        return super().main_kw(size, **kw)

    def minor_kw(self, size: float = CHART_SIZE, **kw) -> dict:
        return super().minor_kw(size, **kw)


PRIME = "′"


@dataclass
class FigureLayout:
    height_cm: float
    axes_h_cm: float
    ve: float
    ve_reduced: bool
    y0: float
    y1: float
    h_scale: float  # 1:h_scale horizontally at the printed width
    v_scale: float


def _data_range(section: SectionProfile):
    vals = [section.ground[np.isfinite(section.ground)]]
    for e in section.events:
        for w in e.intervals:
            vals.append(w.wse)
    z = np.concatenate(vals)
    return float(z.min()), float(z.max())


def layout(
    section: SectionProfile,
    ve: float = DEFAULT_VE,
    min_height_cm: float = DEFAULT_MIN_HEIGHT_CM,
    max_height_cm: float = DEFAULT_MAX_HEIGHT_CM,
    bottom_cm: float = BOTTOM_CM,
    top_cm: float = TOP_CM,
) -> FigureLayout:
    """Figure height, y-limits and the exaggeration actually used."""
    axes_w = REPORT_FIG_WIDTH_CM - LEFT_CM - RIGHT_CM
    hs = axes_w / section.length  # cm of paper per metre along
    zmin, zmax = _data_range(section)
    rng = max(zmax - zmin, 0.5)
    pad_lo, pad_hi = 0.08 * rng, 0.22 * rng  # room for the dimension labels
    span = rng + pad_lo + pad_hi
    margins = top_cm + bottom_cm
    min_axes = max(min_height_cm - margins, 1.0)
    max_axes = max(max_height_cm - margins, min_axes)
    need = span * ve * hs
    reduced = False
    if need < min_axes:
        axes_h = min_axes
        extra = axes_h / (ve * hs) - span
        y0 = zmin - pad_lo - 0.35 * extra
        y1 = y0 + axes_h / (ve * hs)
    elif need > max_axes:
        axes_h = max_axes
        ve = axes_h / (span * hs)
        reduced = True
        y0, y1 = zmin - pad_lo, zmax + pad_hi
    else:
        axes_h = need
        y0, y1 = zmin - pad_lo, zmax + pad_hi
    h_scale = 100.0 * section.length / axes_w
    return FigureLayout(
        height_cm=axes_h + margins,
        axes_h_cm=axes_h,
        ve=ve,
        ve_reduced=reduced,
        y0=y0,
        y1=y1,
        h_scale=h_scale,
        v_scale=h_scale / ve,
    )


def _nice_ticks(lo: float, hi: float, target: int):
    from matplotlib.ticker import MaxNLocator

    ticks = MaxNLocator(nbins=target, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    ticks = [t for t in ticks if lo - 1e-9 <= t <= hi + 1e-9]
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    decimals = max(0, -int(np.floor(np.log10(step) + 1e-9))) if step < 1 else 0
    if step < 1 and abs(step * 10**decimals - round(step * 10**decimals)) > 1e-6:
        decimals += 1
    return ticks, [f"{t:.{decimals}f}" for t in ticks]


def _runs(mask: np.ndarray):
    """(start, stop) index pairs of True runs."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(idx[::2], idx[1::2], strict=True))


def _fmt_m(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f} m"
    return text.replace("-", "−")


def section_title(section: SectionProfile) -> str:
    return f"Section {section.label}\u2013{section.label}{PRIME}"


def text_width_cm(text: str, family: str, size: float) -> float:
    """Rendered width of a single line of text, cm."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    prop = FontProperties(family=family, size=size)
    width_pt = TextToPath().get_text_width_height_descent(text, prop, ismath=False)[0]
    return width_pt / 72.0 * 2.54


def _legend_columns(labels, fonts: FigureFonts) -> int:
    """Most columns (up to 4) for which the legend fits the width."""
    em = MAIN_SIZE / 72.0 * 2.54
    widths = [text_width_cm(t, fonts.main, MAIN_SIZE) + 2.6 * em for t in labels]
    for ncol in range(min(4, len(labels)), 0, -1):
        cols = [max(widths[c::ncol]) for c in range(ncol)]
        if sum(cols) + 1.4 * em * (ncol - 1) <= TEXT_WIDTH_CM:
            return ncol
    return 1


def draw_section(
    section: SectionProfile,
    dimension: dict | None = None,
    ve: float = DEFAULT_VE,
    min_height_cm: float = DEFAULT_MIN_HEIGHT_CM,
    max_height_cm: float = DEFAULT_MAX_HEIGHT_CM,
    property_label: str = "Property",
    source_note: str = "",
    fonts: FigureFonts | None = None,
    plan: PlanView | None = None,
):
    """Returns (figure, layout). ``dimension`` is measure()['dimension'] for
    the event to dimension (normally the first one). With ``plan`` the plan
    view is drawn above the profile in the same figure."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    fonts = fonts or HouseFonts()

    # legend entries first: their number and length set the bottom margin
    handles = [
        (
            Patch(
                facecolor=C_GROUND_FILL,
                edgecolor=C_GROUND,
                linewidth=GROUND_LW,
                linestyle=GROUND_LS,
            ),
            "Natural ground",
        )
    ]
    for i, ev in enumerate(section.events):
        colour = WATER_COLOURS[i % len(WATER_COLOURS)]
        handles.append(
            (
                Patch(facecolor=colour + "1F", edgecolor=colour, linewidth=DATA_LW),
                f"{ev.label} water surface",
            )
        )
    if section.property_intervals:
        handles.append(
            (
                Line2D([0], [0], color=C_PROPERTY, lw=CHART_LINE_PT, ls=(0, (4, 2))),
                f"{property_label} boundary",
            )
        )
    ncol = _legend_columns([t for _, t in handles], fonts)
    legend_rows = -(-len(handles) // ncol)

    bottom_cm = AXIS_CM + legend_rows * LEGEND_ROW_CM + TITLE_CM + NOTES_BASE_CM
    top_cm = TOP_CM if section.property_intervals else TOP_PLAIN_CM
    lay = layout(section, ve, min_height_cm, max_height_cm, bottom_cm, top_cm)
    fig_w = REPORT_FIG_WIDTH_CM
    fig_h = lay.height_cm
    if plan is not None:
        fig_h += PLAN_GAP_CM + plan.height_cm + PLAN_TOP_CM
    fig = new_report_figure(fig_h)
    if plan is not None:
        _draw_plan(fig, fig_h, lay.height_cm + PLAN_GAP_CM, section, plan, fonts)
    axes_w = fig_w - LEFT_CM - RIGHT_CM
    ax = fig.add_axes(
        [LEFT_CM / fig_w, bottom_cm / fig_h, axes_w / fig_w, lay.axes_h_cm / fig_h]
    )
    length = section.length
    ax.set_xlim(0, length)
    ax.set_ylim(lay.y0, lay.y1)
    yr = lay.y1 - lay.y0
    cm_per_m_y = lay.axes_h_cm / yr
    cm_per_m_x = axes_w / length

    # frame and grid
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, which="minor", color=CHART_GRID_MINOR, linewidth=CHART_GRID_MINOR_PT)
    ax.grid(True, which="major", color=CHART_GRID_MAJOR, linewidth=CHART_LINE_PT)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(C_INK)
        ax.spines[side].set_linewidth(CHART_LINE_PT)
    ax.tick_params(which="both", colors=C_INK, width=CHART_LINE_PT)
    ax.tick_params(which="major", length=3)
    ax.tick_params(which="minor", length=1.5)
    xt, xl = _nice_ticks(0, length, 10)
    yt, yl = _nice_ticks(lay.y0, lay.y1, max(3, int(lay.axes_h_cm / 1.2)))
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, **fonts.minor_kw(color=C_INK_2))
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, **fonts.minor_kw(color=C_INK_2))
    ax.set_xlabel(
        "Chainage (m)", fontweight="bold", labelpad=3, **fonts.main_kw(color=C_INK)
    )
    ax.set_ylabel(
        "Elevation (m)", fontweight="bold", labelpad=4, **fonts.main_kw(color=C_INK)
    )

    ch, g = section.chainage, section.ground
    finite = np.isfinite(g)

    # property shading and boundary lines
    xa = blended_transform_factory(ax.transData, ax.transAxes)
    for a, b in section.property_intervals:
        ax.axvspan(a, b, color=C_PROPERTY, alpha=0.06, lw=0, zorder=0.5)
    for c in section.property_crossings:
        ax.axvline(c, color=C_PROPERTY, lw=CHART_LINE_PT, ls=(0, (4, 2)), zorder=1.5)

    # ground
    for s, e in _runs(finite):
        ax.fill_between(ch[s:e], g[s:e], lay.y0, color=C_GROUND_FILL, lw=0, zorder=1)
        ax.plot(ch[s:e], g[s:e], color=C_GROUND, lw=GROUND_LW, ls=GROUND_LS, zorder=4)

    # water (primary event drawn last, on top)
    events = list(enumerate(section.events))
    for i, ev in reversed(events):
        colour = WATER_COLOURS[i % len(WATER_COLOURS)]
        for w in ev.intervals:
            ax.fill_between(
                w.ch, w.ground, w.wse, color=colour, alpha=FILL_ALPHA, lw=0, zorder=2
            )
            ax.plot(w.ch, w.wse, color=colour, lw=DATA_LW, zorder=3)

    # property bracket above the plot
    for a, b in section.property_intervals:
        ax.plot(
            [a, b], [1.03, 1.03], color=C_PROPERTY, lw=0.9, transform=xa, clip_on=False
        )
        for c in (a, b):
            if 1e-6 < c < length - 1e-6:
                ax.plot(
                    [c, c],
                    [1.0, 1.03],
                    color=C_PROPERTY,
                    lw=0.9,
                    transform=xa,
                    clip_on=False,
                )
        mid = min(max(0.5 * (a + b) / length, 0.1), 0.9)
        ax.text(
            mid,
            1.045,
            property_label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color=C_PROPERTY,
            **fonts.main_kw(),
        )

    if dimension:
        h_text, v_text, v_cands = _draw_dimension(
            ax, section, dimension, fonts, yr, cm_per_m_x, cm_per_m_y
        )
        # keep the vertical value clear of the horizontal one
        _resolve_overlaps(fig, ax, [h_text], [(v_text, v_cands)], keep_if_stuck=True)

    # legend, then the title below it
    fig.legend(
        [h for h, _ in handles],
        [t for _, t in handles],
        loc="lower left",
        bbox_to_anchor=(0.1 / fig_w, (NOTES_BASE_CM + TITLE_CM) / fig_h),
        ncol=ncol,
        frameon=False,
        handlelength=1.6,
        handleheight=0.8,
        columnspacing=1.4,
        borderaxespad=0.0,
        borderpad=0.1,
        prop={"family": fonts.main, "size": CHART_SIZE, "style": "italic"},
    )
    fig.text(
        (LEFT_CM + 0.5 * axes_w) / fig_w,
        (NOTES_BASE_CM + 0.05) / fig_h,
        section_title(section),
        ha="center",
        va="bottom",
        color=C_INK,
        fontweight="bold",
        **fonts.main_kw(),
    )
    return fig, lay


# ----------------------------------------------------------------------
# Plan view (map image of the visible layers with the section on top)

PLAN_GAP_CM = 0.55  # between the plan and the profile's top labels
PLAN_TOP_CM = 0.15
PLAN_MIN_H_CM = 5.0
PLAN_MAX_H_CM = 9.0
PLAN_DPI = 300


@dataclass
class PlanView:
    extent: tuple  # (xmin, ymin, xmax, ymax), map units (m)
    height_cm: float
    image: np.ndarray | None = None  # rendered map, rows x cols x 3/4, uint8
    polygons: list | None = None  # property rings
    plan_lines: list | None = None  # [((x, y), (x, y), label), ...]

    @property
    def width_cm(self) -> float:
        return REPORT_FIG_WIDTH_CM - LEFT_CM - RIGHT_CM

    @property
    def pixels(self) -> tuple[int, int]:
        """(width, height) of the map image at 300 dpi."""
        w = int(round(self.width_cm / 2.54 * PLAN_DPI))
        h = int(round(self.height_cm / 2.54 * PLAN_DPI))
        return w, h

    @property
    def scale(self) -> float:
        """1:scale at the printed size."""
        return 100.0 * (self.extent[2] - self.extent[0]) / self.width_cm


def plan_view(
    section: SectionProfile,
    min_height_cm: float = PLAN_MIN_H_CM,
    max_height_cm: float = PLAN_MAX_H_CM,
    margin: float = 0.12,
) -> PlanView:
    """Map extent fitted to the section (plus a margin of 12 % of its length
    on every side), shaped to the plan's width and a height between the
    minimum and maximum."""
    width_cm = REPORT_FIG_WIDTH_CM - LEFT_CM - RIGHT_CM
    v = section.vertices
    pad = max(margin * section.length, 5.0)
    x0, x1 = v[:, 0].min() - pad, v[:, 0].max() + pad
    y0, y1 = v[:, 1].min() - pad, v[:, 1].max() + pad
    dx, dy = x1 - x0, y1 - y0
    h = min(max(width_cm * dy / dx, min_height_cm), max_height_cm)
    ratio = width_cm / h  # required dx / dy
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if dx / dy < ratio:
        dx = dy * ratio
    else:
        dy = dx / ratio
    return PlanView(
        extent=(cx - dx / 2, cy - dy / 2, cx + dx / 2, cy + dy / 2), height_cm=h
    )


def _halo(width=2.2):
    from matplotlib import patheffects

    return [patheffects.withStroke(linewidth=width, foreground="white")]


def _scale_bar_length(width_m: float) -> float:
    target = 0.25 * width_m
    mag = 10 ** np.floor(np.log10(target))
    for m in (5, 2, 1):
        if m * mag <= target:
            return float(m * mag)
    return float(mag)


def _draw_plan(fig, fig_h, bottom_cm, section, plan: PlanView, fonts):
    from matplotlib.patches import Polygon, Rectangle

    fig_w = REPORT_FIG_WIDTH_CM
    ax = fig.add_axes(
        [
            LEFT_CM / fig_w,
            bottom_cm / fig_h,
            plan.width_cm / fig_w,
            plan.height_cm / fig_h,
        ]
    )
    x0, y0, x1, y1 = plan.extent
    if plan.image is not None:
        ax.imshow(
            plan.image,
            extent=(x0, x1, y0, y1),
            origin="upper",
            interpolation="antialiased",
            zorder=0,
        )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C_INK)
        spine.set_linewidth(CHART_LINE_PT)
    m_per_cm = (x1 - x0) / plan.width_cm
    fixed = []  # labels that stay where they are
    plan_labels = []  # (text, candidate positions) - placed first
    tick_labels = []  # chainage labels - hidden when no clear spot

    # property outline
    for poly in plan.polygons or ():
        for ring in poly:
            ax.add_patch(
                Polygon(
                    np.asarray(ring),
                    closed=True,
                    fill=False,
                    edgecolor=C_PROPERTY,
                    lw=1.3,
                    zorder=3,
                    path_effects=_halo(2.6),
                )
            )

    # section line, chainage ticks and end labels
    v = section.vertices
    ax.plot(v[:, 0], v[:, 1], color=C_INK, lw=1.5, zorder=5, path_effects=_halo(3.2))
    step = nice_step(section.length, 5)
    tick = 0.12 * m_per_cm
    c = step
    while c < section.length - 0.25 * step:
        px, py = point_at(v, [c - 0.01, c + 0.01])
        dx, dy = px[1] - px[0], py[1] - py[0]
        n = np.hypot(dx, dy) or 1.0
        nx, ny = -dy / n, dx / n
        cx, cy = 0.5 * (px[0] + px[1]), 0.5 * (py[0] + py[1])
        ax.plot(
            [cx - nx * tick, cx + nx * tick],
            [cy - ny * tick, cy + ny * tick],
            color=C_INK,
            lw=1.0,
            zorder=6,
            path_effects=_halo(2.4),
        )
        t = ax.text(
            cx + nx * 2.4 * tick,
            cy + ny * 2.4 * tick,
            f"{c:g}",
            ha="center",
            va="center",
            color=C_INK,
            zorder=7,
            path_effects=_halo(),
            **fonts.minor_kw(),
        )
        tick_labels.append(
            (
                t,
                [
                    (cx + nx * 2.4 * tick, cy + ny * 2.4 * tick),
                    (cx - nx * 2.4 * tick, cy - ny * 2.4 * tick),
                    (cx + nx * 3.6 * tick, cy + ny * 3.6 * tick),
                    (cx - nx * 3.6 * tick, cy - ny * 3.6 * tick),
                ],
            )
        )
        c += step
    for end, text in ((0, section.label), (-1, section.label + PRIME)):
        other = 1 if end == 0 else -2
        dx, dy = v[end] - v[other]
        n = np.hypot(dx, dy) or 1.0
        off = 0.35 * m_per_cm
        t = ax.text(
            v[end][0] + dx / n * off,
            v[end][1] + dy / n * off,
            text,
            ha="center",
            va="center",
            color=C_INK,
            fontweight="bold",
            zorder=7,
            path_effects=_halo(2.6),
            **fonts.main_kw(),
        )
        fixed.append(t)

    # flood edges on the section
    for i, ev in enumerate(section.events):
        colour = WATER_COLOURS[i % len(WATER_COLOURS)]
        chs = ev.waterlines()
        if chs:
            ex, ey = point_at(v, chs)
            ax.plot(
                ex,
                ey,
                ls="none",
                marker="^",
                ms=5,
                mfc=colour,
                mec="white",
                mew=0.7,
                zorder=8,
            )

    # shortest plan distance property to flood
    for a, b, label in plan.plan_lines or ():
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=C_PLAN_LINE,
            lw=1.0,
            ls=(0, (3, 2)),
            zorder=6,
            path_effects=_halo(2.4),
        )
        t = ax.text(
            0.5 * (a[0] + b[0]),
            0.5 * (a[1] + b[1]),
            label,
            ha="center",
            va="center",
            color=C_PLAN_LINE,
            zorder=7,
            path_effects=_halo(),
            **fonts.minor_kw(),
        )
        ddx, ddy = b[0] - a[0], b[1] - a[1]
        dn = np.hypot(ddx, ddy) or 1.0
        qx, qy = -ddy / dn, ddx / dn
        cands = []
        for frac in (0.5, 0.3, 0.7, 0.15, 0.85):
            mx, my = a[0] + frac * ddx, a[1] + frac * ddy
            for k in (1, -1, 2, -2):
                off = k * 0.22 * m_per_cm
                cands.append((mx + qx * off, my + qy * off))
        plan_labels.append((t, cands))

    # scale bar (bottom left) and north arrow (top right)
    bar = _scale_bar_length(x1 - x0)
    bx, by = x0 + 0.35 * m_per_cm, y0 + 0.45 * m_per_cm
    bh = 0.1 * m_per_cm
    ax.add_patch(
        Rectangle(
            (bx - 0.15 * m_per_cm, by - 0.25 * m_per_cm),
            bar + 0.9 * m_per_cm,
            0.75 * m_per_cm,
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            zorder=8,
        )
    )
    for k in range(2):
        ax.add_patch(
            Rectangle(
                (bx + k * bar / 2, by),
                bar / 2,
                bh,
                facecolor=C_INK if k == 0 else "white",
                edgecolor=C_INK,
                lw=0.6,
                zorder=9,
            )
        )
    for xpos, text in (
        (bx, "0"),
        (bx + bar / 2, f"{bar / 2:g}"),
        (bx + bar, f"{bar:g} m"),
    ):
        t = ax.text(
            xpos,
            by + bh + 0.05 * m_per_cm,
            text,
            ha="center",
            va="bottom",
            color=C_INK,
            zorder=9,
            **fonts.minor_kw(),
        )
        fixed.append(t)
    nx_, ny_ = x1 - 0.55 * m_per_cm, y1 - 1.05 * m_per_cm
    s = 0.22 * m_per_cm
    ax.add_patch(
        Polygon(
            [
                (nx_ - s, ny_),
                (nx_, ny_ + 2.4 * s),
                (nx_ + s, ny_),
                (nx_, ny_ + 0.7 * s),
            ],
            closed=True,
            facecolor=C_INK,
            edgecolor="white",
            lw=0.6,
            zorder=9,
        )
    )
    t = ax.text(
        nx_,
        ny_ + 2.6 * s,
        "N",
        ha="center",
        va="bottom",
        color=C_INK,
        fontweight="bold",
        zorder=9,
        path_effects=_halo(),
        **fonts.main_kw(),
    )
    fixed.append(t)
    _resolve_overlaps(fig, ax, fixed, plan_labels, keep_if_stuck=True)
    _resolve_overlaps(fig, ax, fixed + [t for t, _ in plan_labels], tick_labels)
    return ax


def _resolve_overlaps(fig, ax, fixed, movable, keep_if_stuck=False):
    """Moves each movable label to the first candidate position where it
    stays inside the axes and clears every label placed so far. A label
    with no clear spot is hidden (or kept at its first position)."""
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    placed = [
        t.get_window_extent(renderer).expanded(1.08, 1.2)
        for t in fixed
        if t.get_visible()
    ]
    for text, candidates in movable:
        for x, y in candidates:
            text.set_position((x, y))
            bb = text.get_window_extent(renderer).expanded(1.08, 1.2)
            inside = (
                bb.x0 >= box.x0
                and bb.x1 <= box.x1
                and bb.y0 >= box.y0
                and bb.y1 <= box.y1
            )
            if inside and not any(bb.overlaps(o) for o in placed):
                placed.append(bb)
                break
        else:
            if keep_if_stuck:
                text.set_position(candidates[0])
                placed.append(text.get_window_extent(renderer).expanded(1.08, 1.2))
            else:
                text.set_visible(False)


def _dim_line(ax, p0, p1, long_enough, arrow):
    """Dimension line with arrowheads, or with short end ticks when it is
    too short on paper for arrowheads."""
    if long_enough:
        ax.annotate("", xy=p1, xytext=p0, arrowprops=arrow, zorder=6)
        return
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C_INK, lw=0.7, zorder=6)
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        ls="none",
        marker="_" if p0[0] == p1[0] else "|",
        ms=4,
        mew=0.7,
        color=C_INK,
        zorder=6,
    )


def _label_box():
    return {
        "boxstyle": "square,pad=0.12",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.85,
    }


def _draw_dimension(ax, section, dim, fonts, yr, cm_x, cm_y):
    arrow = {
        "arrowstyle": "<|-|>",
        "lw": 0.7,
        "color": C_INK,
        "shrinkA": 0,
        "shrinkB": 0,
        "mutation_scale": 6,
    }
    ext = {"color": C_INK_2, "lw": 0.5, "ls": (0, (2, 2)), "zorder": 5}
    off = 0.07 * yr
    if dim["kind"] == "dry":
        cb, gb = dim["boundary_ch"], dim["ground_at_boundary"]
        ce, w = dim["edge_ch"], dim["wse"]
        lo, hi = sorted((cb, ce))
        between = (section.chainage >= lo) & (section.chainage <= hi)
        gmax = np.nanmax(section.ground[between]) if between.any() else gb
        y = max(gb, w, gmax) + off
        approx = "≈ " if dim["edge_kind"] != EDGE_WATERLINE else ""
        ax.plot([cb, cb], [gb, y + 0.02 * yr], **ext)
        ax.plot([ce, ce], [w, y + 0.02 * yr], **ext)
        _dim_line(ax, (cb, y), (ce, y), abs(ce - cb) * cm_x >= 0.35, arrow)
        h_text = ax.text(
            0.5 * (cb + ce),
            y + 0.015 * yr,
            approx + _fmt_m(dim["horiz"], 1),
            ha="center",
            va="bottom",
            color=C_INK,
            zorder=7,
            bbox=_label_box(),
            **fonts.main_kw(),
        )
        # water level carried to the boundary, then the vertical dimension
        ax.plot(
            [ce, cb], [w, w], color=WATER_COLOURS[0], lw=0.8, ls=(0, (1, 1.5)), zorder=5
        )
        _dim_line(ax, (cb, w), (cb, gb), abs(gb - w) * cm_y >= 0.3, arrow)
        side = 1 if ce < cb else -1
        vx = cb + side * 0.02 * section.length
        vy = 0.5 * (gb + w)
        v_cands = [(vx, vy - k * 0.06 * yr) for k in range(0, 8)]
        v_text = ax.text(
            vx,
            vy,
            _fmt_m(dim["vert"], 2),
            ha="left" if side > 0 else "right",
            va="center",
            color=C_INK,
            zorder=7,
            bbox=_label_box(),
            **fonts.main_kw(),
        )
    else:
        a, b = dim["wet_from"], dim["wet_to"]
        w = dim["wse"]
        y = w + off
        for c in (a, b):
            ax.plot([c, c], [section.ground_at(c), y + 0.02 * yr], **ext)
        _dim_line(ax, (a, y), (b, y), (b - a) * cm_x >= 0.35, arrow)
        h_text = ax.text(
            0.5 * (a + b),
            y + 0.015 * yr,
            _fmt_m(b - a, 1) + " flooded",
            ha="center",
            va="bottom",
            color=C_INK,
            zorder=7,
            bbox=_label_box(),
            **fonts.main_kw(),
        )
        c, d = dim["depth_ch"], dim["depth"]
        gz = section.ground_at(c)
        _dim_line(ax, (c, gz), (c, w), d * cm_y >= 0.3, arrow)
        vx, vy = c + 0.02 * section.length, 0.5 * (gz + w)
        v_cands = [(vx, vy - k * 0.06 * yr) for k in range(0, 8)]
        v_text = ax.text(
            vx,
            vy,
            _fmt_m(d, 2) + " deep",
            ha="left",
            va="center",
            color=C_INK,
            zorder=7,
            bbox=_label_box(),
            **fonts.main_kw(),
        )
    return h_text, v_text, v_cands
