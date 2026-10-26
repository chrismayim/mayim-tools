"""Figures for the SCS Curve Numbers report in the Mayim chart house style
(the soil tools' map frame; 16.5 cm, Arial Narrow 9 pt, house colours,
italic legends without a frame). matplotlib object API only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mayim_tools.soil._common.grid import TargetGrid
from mayim_tools.soil.regional_parameterisation.figures import (
    C_AMBER,
    C_BLUE,
    C_INK,
    C_RED,
    _clean,
    _figure,
    _png,
    _style,
    categorical_maps,
    chart_style,
    continuous_maps,
)

OTHER_CODE = 9999


def preview_view(result):
    """A stand-in 'result' for the shared map helpers: the decimated preview
    grid and its zone raster."""
    g, step = result.grid, result.preview_step
    rows, cols = result.preview["design"].shape
    res = g.res * step
    pg = TargetGrid(
        g.xmin, g.ymax - rows * res, g.xmin + cols * res, g.ymax, res, g.crs_wkt
    )
    zr = None
    if result.settings.zones:
        try:
            from mayim_tools.soil.regional_parameterisation.core import rasterise_zones

            zr = rasterise_zones(result.settings.zones, pg)
        except Exception:  # noqa: BLE001 - maps without zone outlines
            zr = None
    return SimpleNamespace(grid=pg, zone_raster=zr)


def cn_maps(result):
    v = preview_view(result)
    return continuous_maps(
        v,
        [
            (result.preview["design"], "Design curve number", "RdYlGn_r", False, "CN"),
            (result.preview["width"], "Range P95 - P05", "magma_r", False, "CN points"),
        ],
    )


def landcover_map(result, colours):
    v = preview_view(result)
    lc = result.preview["lc"]
    codes, counts = np.unique(lc, return_counts=True)
    known = [
        (int(c), int(n))
        for c, n in zip(codes, counts, strict=True)
        if int(c) in result.lookup.classes
    ]
    if not known:
        raise ValueError("no mapped classes")
    total = sum(n for _, n in known)
    # Legend: the largest classes (at least 1 % of the area, at most 14); the
    # rest are drawn grey as "other classes".
    ranked = sorted(known, key=lambda cn: -cn[1])
    shown = {c for c, n in ranked[:14] if n / total >= 0.01} or {ranked[0][0]}
    other = OTHER_CODE
    arr = lc.astype(float)
    small = np.isin(lc, [c for c, _ in known if c not in shown])
    arr[small] = other
    cols = {c: colours.get(c, "#999999") for c in sorted(shown)}
    labels = {c: result.lookup.classes[c].name for c in cols}
    if small.any():
        cols[other] = "#c8c8c8"
        labels[other] = "Other classes (< 1 % each)"
    return categorical_maps(v, [(arr, "Land cover")], cols, labels)


def _legend(ax, handles, **kw):
    legend = ax.legend(
        handles=handles,
        prop=chart_style().font_props(style="italic"),
        frameon=False,
        **kw,
    )
    for text in legend.get_texts():
        text.set_color(C_INK)
    return legend


def histogram(result):
    from matplotlib.lines import Line2D

    with _style():
        fig = _figure(6.0)
        ax = fig.add_subplot(1, 1, 1)
        h = result.histogram.astype(float)
        area = h * result.cell_km2
        ax.bar(np.arange(101), area, width=1.0, color=C_BLUE, edgecolor=C_INK, lw=0.75)
        ax.set_xlim(20, 100.5)
        ax.set_xlabel("Design curve number (-)")
        ax.set_ylabel("Area (km²)")
        row = result.zone_rows[0]
        mean = row["Mean CN design"]
        eq = next(
            v for k, v in row.items() if k.startswith("Runoff-equivalent CN design")
        )
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.3)
        _clean(ax)
        handles = []
        for x, lab, ls, col in (
            (mean, "Area-weighted mean", "--", C_INK),
            (eq, "Runoff-equivalent", "-", C_RED),
        ):
            if np.isfinite(x):
                ax.axvline(x, color=col, lw=1.5 if col == C_RED else 0.75, ls=ls)
                handles.append(
                    Line2D(
                        [], [], color=col, lw=1.5 if col == C_RED else 0.75, ls=ls,
                        label=f"{lab} CN {x:.1f}",
                    )
                )  # fmt: skip
        if handles:
            _legend(ax, handles, loc="upper left")
        fig.tight_layout()
        return _png(fig)


def zone_ranges(result):
    """Per zone: mean CN P05-P95 (bar), mean design (dot), runoff-equivalent
    design (diamond) and its P05-P95 (line)."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    rows = result.zone_rows
    key_eq = next(k for k in rows[0] if k.startswith("Runoff-equivalent CN design"))
    with _style():
        fig = _figure(2.4 + 0.75 * len(rows))
        ax = fig.add_subplot(1, 1, 1)
        for i, r in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot(
                [r["Mean CN P05"], r["Mean CN P95"]],
                [y + 0.12] * 2,
                color=C_BLUE,
                lw=5,
                alpha=0.35,
                solid_capstyle="butt",
            )
            ax.plot(r["Mean CN design"], y + 0.12, "o", color=C_BLUE, ms=4)
            ax.plot(
                [r["Runoff-equivalent CN P05"], r["Runoff-equivalent CN P95"]],
                [y - 0.12] * 2,
                color=C_RED,
                lw=1.5,
            )
            ax.plot(r[key_eq], y - 0.12, "D", color=C_RED, ms=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["Zone"] for r in rows][::-1])
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlabel("Curve number (-)")
        _clean(ax)
        ax.yaxis.set_minor_locator(NullLocator())
        _legend(
            ax,
            [
                Line2D([], [], color=C_BLUE, marker="o", lw=5, alpha=0.6,
                       label="Area-weighted mean, P05 to P95"),
                Line2D([], [], color=C_RED, marker="D", lw=1.5,
                       label="Runoff-equivalent, P05 to P95"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
        )  # fmt: skip
        fig.tight_layout()
        return _png(fig)


def uncertainty_split(result):
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator

    rows = [
        r for r in result.zone_rows if np.isfinite(r["Soil share of uncertainty (%)"])
    ]
    if not rows:
        raise ValueError("no uncertainty")
    with _style():
        fig = _figure(2.4 + 0.6 * len(rows))
        ax = fig.add_subplot(1, 1, 1)
        names = [r["Zone"] for r in rows][::-1]
        soil = np.array([r["Soil share of uncertainty (%)"] for r in rows][::-1])
        ax.barh(names, soil, color=C_BLUE, edgecolor=C_INK, lw=0.75)
        ax.barh(names, 100 - soil, left=soil, color=C_AMBER, edgecolor=C_INK, lw=0.75)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of the CN uncertainty (%)")
        _clean(ax)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(False, axis="y", which="both")
        _legend(
            ax,
            [
                Patch(facecolor=C_BLUE, edgecolor=C_INK, lw=0.75, label="Soil group"),
                Patch(facecolor=C_AMBER, edgecolor=C_INK, lw=0.75,
                      label="Land-cover condition"),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
        )  # fmt: skip
        fig.tight_layout()
        return _png(fig)
