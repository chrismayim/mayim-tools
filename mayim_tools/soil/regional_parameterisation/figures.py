"""Publication-quality figures for the Regional soil parameterisation report.

No QGIS. matplotlib's object API only (no pyplot), which is safe on QGIS's
Processing worker thread. Every builder returns a PNG stream (300 dpi,
16 cm wide). Common conventions:

- one sans-serif type style, 7-8 pt, thin dark axes, light grid;
- panel letters (a), (b), ... on multi-panel figures;
- maps: cells outside the zones in light grey, zone outlines, a scale bar
  and north arrow (projected CRS), colour bars with units;
- perceptually uniform colour maps; log colour scales for Ksat and suctions.
"""

from __future__ import annotations

import io
import math
from contextlib import contextmanager

import numpy as np

from mayim_tools._common.docx_report import C_AQUA, C_BLUE, C_INK, C_INK_2, C_ORANGE

from .ptf import PARAMETER_BY_CODE, PARAMETERS, rawls_1983
from .texture import CLASS_ABBR, TEXTURE_CLASSES, ternary_xy, usda_class
from .uncertainty import VARIANCE_PARTS

WIDTH_CM = 16.0
DPI = 300
C_GRID = "#e6e6e6"
C_OUTSIDE = "#f0f0f0"
DIVERGING = {"RdBu_r", "RdBu", "PuOr", "BrBG", "coolwarm"}
METHOD_COLOURS = {"SR2006": C_BLUE, "TOTH2015": C_AQUA}
METHOD_SHORT = {"SR2006": "S&R", "TOTH2015": "Tóth"}
PRODUCT_SHORT = {"SoilGrids 2.0": "SG", "OpenLandMap-soildb": "OLM", "iSDAsoil": "iSDA"}
PRODUCT_MARKERS = {"SoilGrids 2.0": "o", "OpenLandMap-soildb": "s", "iSDAsoil": "^"}
PRODUCT_COLOURS = ("#2a78d6", "#eb6834", "#8e44ad")
SHORT_MARKER = {PRODUCT_SHORT[k]: v for k, v in PRODUCT_MARKERS.items()}
PART_COLOURS = {
    "input": C_BLUE,
    "product": C_ORANGE,
    "method": C_AQUA,
    "interaction": "#a0a0a0",
}
PART_LABELS = {
    "input": "Input data",
    "product": "Product",
    "method": "Method",
    "interaction": "Interaction",
}
SYMBOL = {
    "theta_s": "θs",
    "theta_fc": "θ33",
    "theta_wp": "θ1500",
    "paw": "PAW",
    "ksat": "Ksat",
    "psi_b": "ψb",
    "lambda": "λ",
    "psi_f": "ψf",
    "theta_r": "θr",
    "alpha": "α",
    "n_vg": "n",
}

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": C_INK_2,
    "axes.labelcolor": C_INK,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.color": C_INK_2,
    "ytick.color": C_INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.0,
}


@contextmanager
def _style():
    import matplotlib

    with matplotlib.rc_context(RC):
        yield


def _figure(height_cm: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(WIDTH_CM / 2.54, height_cm / 2.54), dpi=DPI)
    FigureCanvasAgg(fig)
    return fig


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


def _sig(v, n=2) -> str:
    if v is None or not np.isfinite(v) or v == 0:
        return "0" if v == 0 else "-"
    digits = max(0, n - 1 - int(math.floor(math.log10(abs(v)))))
    return f"{v:.{digits}f}"


def _panel(ax, i: int, title: str = "") -> None:
    label = f"({chr(97 + i)})"
    ax.set_title(f"{label} {title}".strip(), loc="left", fontsize=8, color=C_INK)


def _clean(ax, grid=True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, color=C_GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def _log_axis(axis, lo, hi) -> None:
    """Readable ticks on a log axis: 3-5 'nice' values, no minor labels."""
    from matplotlib.ticker import (
        FixedLocator,
        FuncFormatter,
        NullFormatter,
        NullLocator,
    )

    lo, hi = max(lo, 1e-6), max(hi, lo * 1.001)

    def candidates(mults):
        out = []
        e0, e1 = int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi)))
        for e in range(e0, e1 + 1):
            for m in mults:
                v = m * 10.0**e
                if lo <= v <= hi:
                    out.append(v)
        return out

    nice = candidates((1, 2, 5))
    if len(nice) < 3:
        nice = candidates((1, 1.5, 2, 3, 5, 7))
    if len(nice) < 3:
        nice = candidates((1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9))
    if len(nice) > 6:
        nice = [v for v in nice if str(v).lstrip("0.").startswith("1")] or nice
    if len(nice) < 2:
        nice = sorted({float(_sig(t, 2)) for t in np.geomspace(lo, hi, 3)})
    axis.set_major_locator(FixedLocator(nice))
    axis.set_major_formatter(FuncFormatter(lambda v, _: _sig(v, 2)))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


# ----------------------------------------------------------------------
# Maps
# ----------------------------------------------------------------------


def _extent(grid):
    return (grid.xmin, grid.xmax, grid.ymin, grid.ymax)


def _map_frame(ax, result, scalebar: bool) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(C_INK_2)
        side.set_linewidth(0.5)
    ax.set_aspect("equal")
    zones = result.zone_raster
    if zones is not None and (zones > 0).any() and not (zones > 0).all():
        ax.contour(
            (zones > 0).astype(float),
            levels=[0.5],
            colors=C_INK,
            linewidths=0.6,
            extent=_extent(result.grid),
            origin="upper",
        )
    if scalebar:
        _scale_bar(ax, result.grid)
        _north_arrow(ax)


def _scale_bar(ax, grid) -> None:
    """Scale bar for a projected CRS in metres (skipped for degrees)."""
    if grid.res < 1.0:  # geographic grid
        return
    width = grid.xmax - grid.xmin
    target = width / 4.0
    options = [50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000]
    length = max([v for v in options if v <= target] or [options[0]])
    x0 = grid.xmin + 0.05 * width
    y0 = grid.ymin + 0.06 * (grid.ymax - grid.ymin)
    h = 0.012 * (grid.ymax - grid.ymin)
    from matplotlib.patches import Rectangle

    span_y = grid.ymax - grid.ymin
    ax.add_patch(
        Rectangle(
            (x0 - 0.02 * width, y0 - 0.025 * span_y),
            length + 0.04 * width,
            0.09 * span_y,
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (x0, y0), length / 2, h, facecolor=C_INK, edgecolor=C_INK, lw=0.4, zorder=6
        )
    )
    ax.add_patch(
        Rectangle(
            (x0 + length / 2, y0),
            length / 2,
            h,
            facecolor="white",
            edgecolor=C_INK,
            lw=0.4,
            zorder=6,
        )
    )
    text = f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m"
    ax.text(
        x0 + length / 2,
        y0 + 2.2 * h,
        text,
        ha="center",
        va="bottom",
        fontsize=6,
        color=C_INK,
        zorder=7,
    )


def _north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.93, 0.93),
        xytext=(0.93, 0.80),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=7,
        color=C_INK,
        arrowprops={"arrowstyle": "-|>", "color": C_INK, "lw": 0.8},
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
        zorder=7,
    )


def _masked(result, arr):
    a = np.array(arr, dtype=float)
    if result.zone_raster is not None:
        a[result.zone_raster == 0] = np.nan
    return a


def continuous_maps(result, panels, height_cm=7.2):
    """panels: [(array, title, cmap, log, units)] - up to 3 side by side."""
    from matplotlib.colors import LogNorm, Normalize

    with _style():
        fig = _figure(height_cm)
        n = len(panels)
        for i, (arr, title, cmap, log, units) in enumerate(panels):
            ax = fig.add_subplot(1, n, i + 1)
            ax.set_facecolor(C_OUTSIDE)
            a = _masked(result, arr)
            v = a[np.isfinite(a)]
            if log:
                v = v[v > 0]
            _panel(ax, i, title)
            if v.size == 0:
                _map_frame(ax, result, i == 0)
                continue
            lo, hi = np.percentile(v, (2, 98))
            if cmap in DIVERGING:  # centre the colour scale on zero
                m = max(abs(lo), abs(hi)) or 1e-6
                lo, hi = -m, m
            if log:
                lo = max(lo, 1e-4)
                hi = max(hi, lo * 1.05)
                norm = LogNorm(vmin=lo, vmax=hi)
            else:
                hi = hi if hi > lo else lo + 1e-6
                norm = Normalize(vmin=lo, vmax=hi)
            im = ax.imshow(
                a,
                extent=_extent(result.grid),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            _map_frame(ax, result, i == 0)
            cb = fig.colorbar(
                im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, aspect=25
            )
            cb.outline.set_linewidth(0.4)
            if log:
                _log_axis(cb.ax.xaxis, lo, hi)
            cb.ax.tick_params(labelsize=6, length=2)
            cb.set_label(units, fontsize=7)
        fig.tight_layout(w_pad=1.0)
        return _png(fig)


def categorical_maps(result, panels, colours, labels, height_cm=7.2):
    """panels: [(int array, title)]; colours/labels: code -> colour/label."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    codes = sorted(colours)
    cmap = ListedColormap([colours[c] for c in codes])
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)
    with _style():
        fig = _figure(height_cm)
        n = len(panels)
        present = set()
        for i, (arr, title) in enumerate(panels):
            ax = fig.add_subplot(1, n, i + 1)
            ax.set_facecolor(C_OUTSIDE)
            a = _masked(result, np.where(np.isin(arr, codes), arr, np.nan))
            present |= {int(c) for c in np.unique(a[np.isfinite(a)])}
            ax.imshow(
                a,
                extent=_extent(result.grid),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            _panel(ax, i, title)
            _map_frame(ax, result, i == 0)
        handles = [
            Patch(facecolor=colours[c], edgecolor="none", label=labels[c])
            for c in codes
            if c in present
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(4, len(handles)),
                handlelength=1.2,
                columnspacing=1.2,
            )
        fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.0)
        return _png(fig)


# ----------------------------------------------------------------------
# Texture triangle
# ----------------------------------------------------------------------


def _class_colours():
    from mayim_tools.soil._common.export import class_colour

    return {c: class_colour(c) for c, _, _ in TEXTURE_CLASSES}


def texture_triangle(result, lab):
    """USDA texture triangle: class areas and boundaries from the tool's own
    classifier, 10 % grid, and the area's soils (each product's central soil
    and the ensemble median) as points."""
    from matplotlib.colors import ListedColormap

    with _style():
        fig = _figure(11.5)
        ax = fig.add_subplot(1, 1, 1)
        step = 0.25
        g = np.arange(0, 100 + step / 2, step)
        sa, cl = np.meshgrid(g, g)
        si = 100 - sa - cl
        inside = si >= -1e-9
        codes = np.where(inside, usda_class(sa, np.maximum(si, 0), cl), 0).astype(float)
        codes[~inside] = np.nan
        x, y = ternary_xy(sa, cl)
        cols = _class_colours()
        cmap = ListedColormap([cols[c] for c in range(1, 13)])
        ax.pcolormesh(
            x,
            y,
            codes,
            cmap=cmap,
            vmin=0.5,
            vmax=12.5,
            alpha=0.16,
            shading="nearest",
            rasterized=True,
        )
        ax.contour(
            x,
            y,
            codes,
            levels=np.arange(1.5, 12.5, 1.0),
            colors=C_INK_2,
            linewidths=0.45,
        )
        # 10 % grid lines for the three fractions
        for v in range(10, 100, 10):
            for a, b in (
                (ternary_xy(v, 0), ternary_xy(v, 100 - v)),  # constant sand
                (ternary_xy(100 - v, v), ternary_xy(0, v)),  # constant clay
                (ternary_xy(100 - v, 0), ternary_xy(0, 100 - v)),  # constant silt
            ):
                ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=0.6, zorder=1)
        for c in range(1, 13):
            sel = codes == c
            if sel.any():
                ax.text(
                    x[sel].mean(),
                    y[sel].mean(),
                    CLASS_ABBR[c],
                    fontsize=6.5,
                    ha="center",
                    va="center",
                    color=C_INK_2,
                    zorder=3,
                )
        tri = np.array([[0, 0], [100, 0], [50, 50 * math.sqrt(3)], [0, 0]])
        ax.plot(tri[:, 0], tri[:, 1], color=C_INK, linewidth=0.8, zorder=4)
        rng = np.random.default_rng(3)
        zone = result.zone_raster.ravel() > 0
        for p, colour in zip(result.products, PRODUCT_COLOURS, strict=False):
            c = result.central.get(p.name, {}).get(lab)
            if not c:
                continue
            ok = np.flatnonzero(np.isfinite(c["clay"].ravel()) & zone)
            if ok.size > 2000:
                ok = rng.choice(ok, 2000, replace=False)
            px, py = ternary_xy(c["sand"].ravel()[ok], c["clay"].ravel()[ok])
            ax.scatter(
                px,
                py,
                s=2.5,
                color=colour,
                alpha=0.4,
                linewidths=0,
                zorder=5,
                label=f"{p.name} (central soil)",
                rasterized=True,
            )
        med = result.inputs_p50
        ok = np.flatnonzero(np.isfinite(med["clay"][lab].ravel()) & zone)
        if ok.size > 2000:
            ok = rng.choice(ok, 2000, replace=False)
        px, py = ternary_xy(med["sand"][lab].ravel()[ok], med["clay"][lab].ravel()[ok])
        ax.scatter(
            px,
            py,
            s=2.5,
            color=C_INK,
            alpha=0.55,
            linewidths=0,
            zorder=6,
            label="Ensemble median",
            rasterized=True,
        )
        for v in (20, 40, 60, 80):
            xs, ys = ternary_xy(v, 0)
            ax.text(xs, ys - 3.2, f"{v}", fontsize=6, ha="center", color=C_INK_2)
            xc, yc = ternary_xy(100 - v, v)
            ax.text(
                xc - 1.8, yc, f"{v}", fontsize=6, ha="right", va="center", color=C_INK_2
            )
            xr, yr = ternary_xy(0, 100 - v)
            ax.text(
                xr + 1.8, yr, f"{v}", fontsize=6, ha="left", va="center", color=C_INK_2
            )
        ax.text(50, -9.5, "Sand (%)", fontsize=7.5, ha="center", color=C_INK)
        ax.text(15, 47, "Clay (%)", fontsize=7.5, rotation=60, ha="center", color=C_INK)
        ax.text(
            85, 47, "Silt (%)", fontsize=7.5, rotation=-60, ha="center", color=C_INK
        )
        ax.set_xlim(-8, 108)
        ax.set_ylim(-13, 92)
        ax.set_aspect("equal")
        ax.axis("off")
        leg = ax.legend(loc="upper right", markerscale=3.5, handletextpad=0.3)
        for h in getattr(leg, "legend_handles", None) or leg.legendHandles:
            h.set_alpha(1)
        fig.tight_layout()
        return _png(fig)


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------


def scatter_panels(panels, height_cm=6.4):
    """panels: [(x, y, xlabel, ylabel, log)] - density (hexbin) with a 1:1
    line and n, bias, RMSE and r in the corner."""
    with _style():
        fig = _figure(height_cm)
        n = len(panels)
        for i, (x, y, xl, yl, log) in enumerate(panels):
            ax = fig.add_subplot(1, n, i + 1)
            ok = np.isfinite(x) & np.isfinite(y)
            if log:
                ok &= (x > 0) & (y > 0)
            x, y = np.asarray(x)[ok], np.asarray(y)[ok]
            _panel(ax, i)
            if x.size:
                lo = min(np.percentile(x, 0.5), np.percentile(y, 0.5))
                hi = max(np.percentile(x, 99.5), np.percentile(y, 99.5))
                pad = (hi / lo) ** 0.05 if log and lo > 0 else 0.03 * (hi - lo)
                lo, hi = (
                    (lo / pad, hi * pad) if log and lo > 0 else (lo - pad, hi + pad)
                )
                ax.hexbin(
                    x,
                    y,
                    gridsize=35,
                    cmap="Blues",
                    mincnt=1,
                    bins="log",
                    xscale="log" if log else "linear",
                    yscale="log" if log else "linear",
                    extent=(
                        (math.log10(lo), math.log10(hi), math.log10(lo), math.log10(hi))
                        if log
                        else (lo, hi, lo, hi)
                    ),
                    linewidths=0,
                )
                ax.plot([lo, hi], [lo, hi], color=C_INK, lw=0.7, ls="--")
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                if log:
                    _log_axis(ax.xaxis, lo, hi)
                    _log_axis(ax.yaxis, lo, hi)
                    d = np.log10(y) - np.log10(x)
                    stats = f"n = {x.size}\nmedian ratio = {_sig(10 ** np.median(d))}"
                else:
                    d = y - x
                    r = (
                        np.corrcoef(x, y)[0, 1]
                        if x.size > 2 and np.std(x) > 0 and np.std(y) > 0
                        else math.nan
                    )
                    stats = (
                        f"n = {x.size}\nbias = {d.mean():+.3g}\n"
                        f"RMSE = {math.sqrt(float(np.mean(d * d))):.3g}\nr = {r:.2f}"
                    )
                ax.text(
                    0.04,
                    0.96,
                    stats,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=6.3,
                    color=C_INK,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.8,
                        "pad": 1.5,
                    },
                )
                ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            _clean(ax)
        fig.tight_layout(w_pad=1.2)
        return _png(fig)


def _zone_row(result, zone, lab, code):
    for r in result.zone_rows:
        if r["Zone"] == zone and r["Layer"] == lab and r["Parameter"] == code:
            return r
    return None


def profiles(result, zone):
    """Depth profiles: a band per layer from the median cell P5 to P95 and
    the median of the cell medians, drawn over the layer's depth span."""
    codes = [
        c
        for c in ("theta_s", "theta_fc", "theta_wp", "ksat", "psi_f")
        if c in result.stats
    ]
    layers = [
        (lab, t, b)
        for lab, t, b in result.settings.layers
        if lab in result.texture_class
    ]
    with _style():
        fig = _figure(6.8)
        for i, code in enumerate(codes):
            ax = fig.add_subplot(1, len(codes), i + 1)
            prm = PARAMETER_BY_CODE[code]
            los, his = [], []
            for lab, top, bot in layers:
                r = _zone_row(result, zone, lab, code)
                if r is None:
                    continue
                lo, med, hi = (
                    r["Median cell P5"],
                    r["Median of P50"],
                    r["Median cell P95"],
                )
                ax.fill_betweenx([top, bot], lo, hi, color=C_BLUE, alpha=0.18, lw=0)
                ax.plot([med, med], [top, bot], color=C_BLUE, lw=1.4)
                los.append(lo)
                his.append(hi)
            ax.set_ylim(max(b for _, _, b in layers), 0)
            ax.set_yticks(sorted({0} | {b for _, _, b in layers}))
            if prm.log and los:
                ax.set_xscale("log")
                _log_axis(ax.xaxis, min(los), max(his))
            ax.set_xlabel(f"{SYMBOL.get(code, code)} ({prm.units})")
            if i == 0:
                ax.set_ylabel("Depth (cm)")
            else:
                ax.tick_params(labelleft=False)
            _panel(ax, i)
            _clean(ax)
        fig.tight_layout(w_pad=0.6)
        return _png(fig)


def box_by_member(result, lab):
    """Distribution over the zone cells of each product x method central soil
    and of the pooled median; the Rawls et al. (1983) value of the dominant
    texture class as a reference line."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    zone = result.zone_raster > 0
    tex = result.texture_class.get(lab)
    dom = None
    if tex is not None:
        cls = tex[zone]
        cls = cls[cls > 0]
        if cls.size:
            dom = int(np.bincount(cls).argmax())
    ref = rawls_1983.RAWLS_1983.get(dom) if dom else None
    with _style():
        fig = _figure(7.4)
        codes = [c for c in ("ksat", "psi_f", "theta_fc") if c in result.stats]
        for i, code in enumerate(codes):
            ax = fig.add_subplot(1, len(codes), i + 1)
            data, names, colours = [], [], []
            for p in result.products:
                by = result.central.get(p.name, {}).get(lab, {}).get("by_method", {})
                for mcode, vals in by.items():
                    if code not in vals:
                        continue
                    v = vals[code][zone]
                    v = v[np.isfinite(v)]
                    if not v.size:
                        continue
                    data.append(v)
                    names.append(
                        f"{PRODUCT_SHORT.get(p.name, p.name)}\n{METHOD_SHORT.get(mcode, mcode)}"
                    )
                    colours.append(METHOD_COLOURS.get(mcode, C_BLUE))
            v = result.stats[code][lab][1][zone]
            data.append(v[np.isfinite(v)])
            names.append("Pooled\nP50")
            colours.append("#7f7f7f")
            data = [d if d.size else np.array([np.nan]) for d in data]
            bp = ax.boxplot(
                data,
                showfliers=False,
                widths=0.55,
                patch_artist=True,
                medianprops={"color": C_INK, "lw": 1.0},
                whiskerprops={"color": C_INK_2, "lw": 0.6},
                capprops={"color": C_INK_2, "lw": 0.6},
                boxprops={"lw": 0.6, "edgecolor": C_INK_2},
            )
            for patch, col in zip(bp["boxes"], colours, strict=True):
                patch.set_facecolor(col)
                patch.set_alpha(0.45)
            ax.set_xticks(range(1, len(names) + 1))
            ax.set_xticklabels(names, fontsize=6)
            prm = PARAMETER_BY_CODE[code]
            if ref and code in ("ksat", "psi_f"):
                val = ref[3] * 10 if code == "ksat" else ref[2] * 10
                ax.axhline(val, color=C_ORANGE, lw=0.9, ls="--")
            if prm.log:
                ax.set_yscale("log")
                allv = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in data])
                if allv.size:
                    lo, hi = np.percentile(allv, 1), np.percentile(allv, 99)
                    if ref and code in ("ksat", "psi_f"):
                        val = ref[3] * 10 if code == "ksat" else ref[2] * 10
                        lo, hi = min(lo, val), max(hi, val)
                    _log_axis(ax.yaxis, lo, hi)
            ax.set_ylabel(f"{SYMBOL.get(code, code)} ({prm.units})")
            _panel(ax, i)
            _clean(ax)
        handles = [
            Patch(facecolor=METHOD_COLOURS[m], alpha=0.45, label=METHOD_SHORT[m])
            for m in result.settings.mc.methods
            if m in METHOD_COLOURS
        ] + [Patch(facecolor="#7f7f7f", alpha=0.45, label="Pooled ensemble")]
        if ref:
            from .texture import CLASS_NAME

            handles.append(
                Line2D(
                    [],
                    [],
                    color=C_ORANGE,
                    ls="--",
                    lw=0.9,
                    label=f"Rawls et al. (1983), {CLASS_NAME[dom]}",
                )
            )
        fig.legend(handles=handles, loc="lower center", ncol=len(handles))
        fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
        return _png(fig)


def variance_bars(result, zone_index, lab):
    """Share of variance by source per parameter, with percentages."""
    per = result.variance.get(zone_index, {}).get(lab, {})
    codes = [
        p.code for p in PARAMETERS if p.code in per and per[p.code].get("total", 0) > 0
    ]
    with _style():
        fig = _figure(1.6 + 0.55 * max(len(codes), 1))
        ax = fig.add_subplot(1, 1, 1)
        left = np.zeros(len(codes))
        ypos = np.arange(len(codes))
        for part in VARIANCE_PARTS:
            vals = np.array([per[c].get(part, 0.0) / per[c]["total"] for c in codes])
            ax.barh(
                ypos,
                vals,
                left=left,
                height=0.62,
                color=PART_COLOURS[part],
                label=PART_LABELS[part],
                edgecolor="white",
                linewidth=0.5,
            )
            for y, (l0, v) in enumerate(zip(left, vals, strict=True)):
                if v >= 0.08:
                    ax.text(
                        l0 + v / 2,
                        y,
                        f"{100 * v:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        color="white",
                    )
            left += vals
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{SYMBOL.get(c, c)}" for c in codes])
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(
            __import__(
                "matplotlib.ticker", fromlist=["PercentFormatter"]
            ).PercentFormatter(1.0)
        )
        ax.set_xlabel("Share of variance (log10 scale for Ksat, α and suctions)")
        ax.invert_yaxis()
        ax.legend(
            ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0
        )
        _clean(ax, grid=False)
        fig.tight_layout()
        return _png(fig)


def design_forest(result, zone):
    """Candidate design values per layer: candidate (dot), sensitivity range
    (bar) and each product x method value (small markers)."""
    from matplotlib.lines import Line2D

    rows = [r for r in result.design_rows if r["Zone"] == zone]
    codes = [
        c
        for c in ("ksat", "psi_f", "theta_s", "theta_fc", "theta_wp")
        if any(r["Parameter"] == c for r in rows)
    ]
    layers = [
        lab for lab, _, _ in result.settings.layers if lab in result.texture_class
    ]
    conf_col = {"High": "#1baf7a", "Medium": "#f0ad4e", "Low": "#d9534f"}
    with _style():
        fig = _figure(6.6)
        for i, code in enumerate(codes):
            ax = fig.add_subplot(1, len(codes), i + 1)
            prm = PARAMETER_BY_CODE[code]
            vals_all = []
            for j, lab in enumerate(layers):
                r = next(
                    (x for x in rows if x["Parameter"] == code and x["Layer"] == lab),
                    None,
                )
                if r is None:
                    continue
                lo, c, hi = (
                    r["Sensitivity low (P5)"],
                    r["Candidate value"],
                    r["Sensitivity high (P95)"],
                )
                ax.plot([lo, hi], [j, j], color=C_INK_2, lw=1.0, solid_capstyle="butt")
                for name, v in (
                    (k2[7:], v2) for k2, v2 in r.items() if k2.startswith("Member ")
                ):
                    prod, meth = name.split(" ", 1)
                    marker = SHORT_MARKER.get(prod, "o")
                    ax.plot(
                        v,
                        j + 0.22,
                        marker=marker,
                        ms=3.2,
                        ls="none",
                        color=METHOD_COLOURS.get(meth, C_BLUE),
                        alpha=0.9,
                    )
                    vals_all.append(v)
                ax.plot(
                    c,
                    j,
                    marker="D",
                    ms=5,
                    color=conf_col[r["Confidence"]],
                    markeredgecolor=C_INK,
                    markeredgewidth=0.5,
                    ls="none",
                )
                vals_all += [lo, hi, c]
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers if i == 0 else [""] * len(layers))
            ax.set_ylim(len(layers) - 0.5, -0.6)
            if prm.log:
                ax.set_xscale("log")
                v = [x for x in vals_all if np.isfinite(x) and x > 0]
                if v:
                    _log_axis(ax.xaxis, min(v), max(v))
            ax.set_xlabel(f"{SYMBOL.get(code, code)} ({prm.units})")
            _panel(ax, i)
            _clean(ax)
        handles = (
            [
                Line2D(
                    [],
                    [],
                    marker="D",
                    ls="none",
                    color=col,
                    markeredgecolor=C_INK,
                    markeredgewidth=0.5,
                    ms=5,
                    label=f"Candidate - {lvl.lower()} confidence",
                )
                for lvl, col in conf_col.items()
            ]
            + [
                Line2D(
                    [], [], color=C_INK_2, lw=1.0, label="Sensitivity range (P5-P95)"
                ),
            ]
            + [
                Line2D(
                    [],
                    [],
                    marker=PRODUCT_MARKERS.get(p.name, "o"),
                    ls="none",
                    color="#7f7f7f",
                    ms=3.2,
                    label=f"{p.name.split('-')[0].replace(' 2.0', '')} member",
                )
                for p in result.products
            ]
            + [
                Line2D(
                    [],
                    [],
                    marker="o",
                    ls="none",
                    color=METHOD_COLOURS[m],
                    ms=3.2,
                    label=METHOD_SHORT[m],
                )
                for m in result.settings.mc.methods
                if m in METHOD_COLOURS
            ]
        )
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            handletextpad=0.3,
            columnspacing=1.0,
        )
        fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=0.8)
        return _png(fig)
