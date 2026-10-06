"""Figures for the Hydrologic soil groups report, in the publication style
of Regional soil parameterisation (same map frame, fonts and sizes).
matplotlib object API only (no pyplot)."""

from __future__ import annotations

import numpy as np

from mayim_tools._common.docx_report import C_INK
from mayim_tools.soil.regional_parameterisation.figures import (
    C_OUTSIDE,
    _clean,
    _extent,
    _figure,
    _log_axis,
    _map_frame,
    _masked,
    _panel,
    _png,
    _style,
    categorical_maps,
)

from . import neh630, scs_sa
from .core import colours_for, labels_for

MAP_LABEL = {
    **{c: lab for c, lab in scs_sa.GROUP_LABEL.items()},
    neh630.AD: "A/D (dual)",
    neh630.BD: "B/D (dual)",
    neh630.CD: "C/D (dual)",
    scs_sa.CD_SA: "C/D (SCS-SA)",
}
MAP_COLOUR = {**neh630.GROUP_COLOUR, **scs_sa.GROUP_COLOUR}
METHOD_TITLE = {"NEH630": "NEH 630", "SCSSA": "SCS-SA"}
ORDER = scs_sa.STEP_CODES + (neh630.AD, neh630.BD, neh630.CD)


def group_maps(result, panels):
    """panels: [(codes, title)]."""
    present = set()
    for arr, _ in panels:
        present |= {int(c) for c in np.unique(arr[result.mask]) if c > 0}
    colours = {c: MAP_COLOUR[c] for c in sorted(present) or [neh630.A]}
    labels = {c: MAP_LABEL[c] for c in colours}
    return categorical_maps(result, panels, colours, labels)


def confidence_maps(result):
    from matplotlib.colors import Normalize

    methods = list(result.methods)
    with _style():
        fig = _figure(7.2)
        for i, m in enumerate(methods):
            ax = fig.add_subplot(1, len(methods), i + 1)
            ax.set_facecolor(C_OUTSIDE)
            a = _masked(result, result.methods[m].confidence)
            im = ax.imshow(
                a,
                extent=_extent(result.grid),
                cmap="viridis",
                norm=Normalize(0, 100),
                interpolation="nearest",
            )
            _panel(ax, i, f"{METHOD_TITLE[m]}: confidence")
            _map_frame(ax, result, i == 0)
            cb = fig.colorbar(
                im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, aspect=25
            )
            cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(labelsize=6, length=2)
            cb.set_label("probability of the recommended group (%)", fontsize=7)
        fig.tight_layout(w_pad=1.0)
        return _png(fig)


def zone_shares(result):
    """Horizontal stacked bars: share of each recommended group per zone and
    method."""
    rows = result.zone_rows
    with _style():
        fig = _figure(2.4 + 0.55 * len(rows))
        ax = fig.add_subplot(1, 1, 1)
        names = [f"{r['Zone']} - {METHOD_TITLE[r['Method']]}" for r in rows]
        y = np.arange(len(rows))[::-1]
        used = {}
        for yi, r in zip(y, rows, strict=True):
            left = 0.0
            labels = labels_for(r["Method"])
            colours = colours_for(r["Method"])
            for code in ORDER:
                if code not in labels:
                    continue
                v = r.get(f"Share {labels[code]} (%)", 0.0) or 0.0
                if not np.isfinite(v) or v <= 0:
                    continue
                key = MAP_LABEL[code]
                h = ax.barh(
                    yi,
                    v,
                    left=left,
                    color=colours[code],
                    height=0.6,
                    edgecolor="white",
                    linewidth=0.5,
                )
                used.setdefault(key, h)
                if v >= 8:
                    ax.text(
                        left + v / 2,
                        yi,
                        labels[code],
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=C_INK,
                    )
                left += v
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of zone area (%)")
        _clean(ax)
        ax.grid(axis="y", visible=False)
        order = [MAP_LABEL[c] for c in ORDER]
        keys = [k for k in order if k in used]
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        if keys:
            fig.legend(
                [used[k] for k in keys],
                keys,
                loc="lower center",
                ncol=min(7, len(keys)),
                handlelength=1.0,
                columnspacing=1.0,
            )
        return _png(fig)


def ksat_thresholds(result):
    """Histogram of the least transmissive Ksat (P50) with the class limits
    of both methods marked: shows why the methods differ."""
    k50 = neh630.least_transmissive(result.ksat["P50"], "0-50cm")[result.mask]
    k100 = neh630.least_transmissive(result.ksat["P50"], "0-100cm")[result.mask]
    vals = np.concatenate([k50[np.isfinite(k50)], k100[np.isfinite(k100)]])
    vals = vals[vals > 0]
    with _style():
        fig = _figure(7.2)
        ax = fig.add_subplot(1, 1, 1)
        lo = min(0.5, vals.min() if vals.size else 0.5)
        hi = max(300.0, vals.max() if vals.size else 300.0)
        bins = np.geomspace(lo, hi, 60)
        ax.hist(
            k50[np.isfinite(k50) & (k50 > 0)],
            bins=bins,
            color="#7f7f7f",
            alpha=0.55,
            label="least transmissive 0-50 cm (P50)",
        )
        ax.hist(
            k100[np.isfinite(k100) & (k100 > 0)],
            bins=bins,
            histtype="step",
            color=C_INK,
            linewidth=0.8,
            label="least transmissive 0-100 cm (P50)",
        )
        ax.set_xscale("log")
        _log_axis(ax.xaxis, lo, hi)
        marks = [
            (neh630.SHALLOW_MM_H, "#2a78d6", "-", "NEH 630, 0-50 cm"),
            (neh630.DEEP_MM_H, "#2a78d6", "--", "NEH 630, 0-100 cm"),
            (scs_sa.PERMEABILITY_MM_H, "#eb6834", "-", "SCS-SA"),
        ]
        for thresholds, colour, ls, lab in marks:
            for j, t in enumerate(thresholds):
                ax.axvline(
                    t,
                    color=colour,
                    linestyle=ls,
                    linewidth=0.9,
                    label=lab if j == 0 else None,
                )
        ax.set_xlabel("Ksat (mm/h)")
        ax.set_ylabel("cells")
        _clean(ax)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5)
        return _png(fig)


def probability_bars(result):
    """Mean probability of A-D per zone and method (grouped bars)."""
    rows = result.zone_rows
    letters = "ABCD"
    cols = [neh630.GROUP_COLOUR[c] for c in neh630.SINGLE]
    with _style():
        fig = _figure(5.6)
        ax = fig.add_subplot(1, 1, 1)
        x = np.arange(len(rows))
        w = 0.2
        for i, lab in enumerate(letters):
            vals = [r.get(f"Mean P({lab}) (%)", np.nan) for r in rows]
            ax.bar(
                x + (i - 1.5) * w,
                vals,
                w,
                color=cols[i],
                label=lab,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['Zone']}\n{METHOD_TITLE[r['Method']]}" for r in rows], fontsize=6.5
        )
        ax.set_ylabel("mean probability (%)")
        ax.set_ylim(0, 100)
        _clean(ax)
        ax.legend(ncol=4, loc="upper right", title=None)
        fig.tight_layout()
        return _png(fig)


def by_ksat_maps(result, method):
    mr = result.methods[method]
    t = METHOD_TITLE[method]
    return group_maps(
        result,
        [
            (mr.at["P05"], f"{t}: Ksat P5"),
            (mr.at["P50"], f"{t}: Ksat P50"),
            (mr.at["P95"], f"{t}: Ksat P95"),
        ],
    )
