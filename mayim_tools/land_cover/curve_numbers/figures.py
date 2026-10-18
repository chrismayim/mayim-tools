"""Figures for the Curve numbers report, in the publication style of the
soil tools (same map frame, fonts and sizes). matplotlib object API only."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mayim_tools._common.docx_report import C_AQUA, C_BLUE, C_INK, C_INK_2
from mayim_tools.soil._common.grid import TargetGrid
from mayim_tools.soil.regional_parameterisation.figures import (
    _clean,
    _figure,
    _png,
    _style,
    categorical_maps,
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


def histogram(result):
    with _style():
        fig = _figure(5.5)
        ax = fig.add_subplot(1, 1, 1)
        h = result.histogram.astype(float)
        area = h * result.cell_km2
        ax.bar(np.arange(101), area, width=1.0, color=C_BLUE, edgecolor="none")
        _clean(ax)
        ax.set_xlim(20, 100.5)
        ax.set_xlabel("Design curve number")
        ax.set_ylabel("Area (km2)")
        row = result.zone_rows[0]
        mean = row["Mean CN design"]
        eq = next(
            v for k, v in row.items() if k.startswith("Runoff-equivalent CN design")
        )
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.25)
        for x, lab, ls, y in (
            (mean, "area-weighted mean", "--", 1.18),
            (eq, "runoff-equivalent", "-", 1.06),
        ):
            if np.isfinite(x):
                ax.axvline(x, color=C_INK, lw=0.8, ls=ls)
                ax.text(
                    x,
                    top * y,
                    f" {lab} {x:.1f} ",
                    fontsize=6,
                    va="center",
                    ha="right",
                    color=C_INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
                )
        fig.tight_layout()
        return _png(fig)


def zone_ranges(result):
    """Per zone: mean CN P05-P95 (bar), mean design (dot), runoff-equivalent
    design (diamond) and its P05-P95 (thin line)."""
    rows = result.zone_rows
    key_eq = next(k for k in rows[0] if k.startswith("Runoff-equivalent CN design"))
    with _style():
        fig = _figure(1.6 + 0.65 * len(rows))
        ax = fig.add_subplot(1, 1, 1)
        for i, r in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot(
                [r["Mean CN P05"], r["Mean CN P95"]],
                [y + 0.12] * 2,
                color=C_BLUE,
                lw=4,
                alpha=0.35,
                solid_capstyle="butt",
            )
            ax.plot(r["Mean CN design"], y + 0.12, "o", color=C_BLUE, ms=4)
            ax.plot(
                [r["Runoff-equivalent CN P05"], r["Runoff-equivalent CN P95"]],
                [y - 0.12] * 2,
                color=C_AQUA,
                lw=1.2,
            )
            ax.plot(r[key_eq], y - 0.12, "D", color=C_AQUA, ms=4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["Zone"] for r in rows][::-1])
        ax.set_ylim(-0.7, len(rows) - 0.3)
        _clean(ax)
        ax.set_xlabel("Curve number")
        from matplotlib.lines import Line2D

        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=C_BLUE,
                    marker="o",
                    lw=4,
                    alpha=0.6,
                    label="area-weighted mean, P05-P95",
                ),
                Line2D(
                    [],
                    [],
                    color=C_AQUA,
                    marker="D",
                    lw=1.2,
                    label="runoff-equivalent, P05-P95",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
        )
        fig.tight_layout()
        return _png(fig)


def uncertainty_split(result):
    rows = [
        r for r in result.zone_rows if np.isfinite(r["Soil share of uncertainty (%)"])
    ]
    if not rows:
        raise ValueError("no uncertainty")
    with _style():
        fig = _figure(1.4 + 0.5 * len(rows))
        ax = fig.add_subplot(1, 1, 1)
        names = [r["Zone"] for r in rows][::-1]
        soil = np.array([r["Soil share of uncertainty (%)"] for r in rows][::-1])
        ax.barh(names, soil, color=C_BLUE, label="soil group")
        ax.barh(
            names, 100 - soil, left=soil, color="#c9c7c2", label="land-cover condition"
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of the CN uncertainty (%)")
        _clean(ax, grid=False)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        for s in ("left",):
            ax.spines[s].set_color(C_INK_2)
        fig.tight_layout()
        return _png(fig)
