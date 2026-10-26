"""Charts for Precipitation data to DDF in the Mayim chart style: 16.5 cm
wide, Arial Narrow 9 pt (axis titles bold, legend italic), black border
and ticks, #BFBFBF / #F2F2F2 gridlines, house colours, 1.5 pt data lines,
12 % bands, legend without a frame below the plot. Built with the
matplotlib object API (thread-safe inside QGIS Processing).

The ``*_figure`` functions return (PNG stream, style warnings) and raise
ImportError without matplotlib; the ``write_*`` functions save a PNG and
return False when matplotlib is unavailable, so the CSV outputs never
depend on it.
"""

from __future__ import annotations

import numpy as np

from mayim_tools.rainfall._common.rfa.distributions import (
    GUMBEL_TAU3,
    GUMBEL_TAU4,
    gev_tau3_tau4_exact,
    glo_tau3_tau4_exact,
)
from mayim_tools.rainfall._common.rfa.timebase import duration_label

# House line colours (yellow is for fills only, so it is skipped).
LINE_COLOURS = ["#FF0000", "#FFC000", "#00CC00", "#0000FF", "#FF00FF", "#00FFFF"]
INK = "#000000"
BAND_ALPHA = 0.12
DATA_LW = 1.5
REF_LW = 0.75
DDF_HEIGHT_CM = 11.0
RATIO_HEIGHT_CM = 11.0
LMOM_HEIGHT_CM = 7.0
DURATION_TICKS = (5, 15, 30, 60, 120, 360, 720, 1440, 2880, 4320, 10080)
SHOW_RPS = (2, 5, 10, 20, 50, 100)


def _duration_axis(ax, durations, ticks=DURATION_TICKS, rotate=True):
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    ax.set_xscale("log")
    lo, hi = min(durations), max(durations)
    ticks = [t for t in ticks if lo <= t <= hi] or sorted(durations)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([duration_label(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(lo / 1.15, hi * 1.15)
    if rotate:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    ax.set_xlabel("Duration")


def _plain_y(ax):
    from matplotlib.ticker import ScalarFormatter

    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)


def _legend_below(fig, style, handles, ncol=4):
    legend = fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=ncol,
        frameon=False,
        prop=style.font_props(style="italic"),
    )
    for text in legend.get_texts():
        text.set_color(INK)


def _shown_rps(result):
    rps = list(result.return_periods)
    shown = [t for t in rps if any(abs(t - s) < 1e-6 for s in SHOW_RPS)]
    return shown or rps[: len(LINE_COLOURS)]


def ddf_figure(result):
    """Recommended (duration-consistent) DDF curves with their 5-95 %
    bootstrap bands, and the independent per-duration GEV depths as hollow
    circles."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from mayim_tools._common.report_figures import ChartStyle

    if not result.ddf_rows:
        raise ValueError("no recommended DDF to plot")
    style = ChartStyle()
    rps = _shown_rps(result)
    durs = sorted({r["duration_min"] for r in result.ddf_rows})
    look = {(r["duration_min"], r["return_period_yr"]): r for r in result.ddf_rows}
    comp = {
        (c["duration_min"], c["return_period_yr"]): c["independent_gev_mm"]
        for c in result.model_comparison
    }
    has_band = False
    with style.context():
        fig = style.figure(DDF_HEIGHT_CM)
        ax = fig.subplots()
        handles = []
        for t, col in zip(rps, LINE_COLOURS, strict=False):
            y = [look[(d, t)]["depth_mm"] for d in durs]
            lo = np.array([look[(d, t)]["lower_mm"] for d in durs], float)
            up = np.array([look[(d, t)]["upper_mm"] for d in durs], float)
            (line,) = ax.plot(durs, y, color=col, lw=DATA_LW, label=f"{t:g}-year")
            handles.append(line)
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(up)):
                ax.fill_between(durs, lo, up, color=col, alpha=BAND_ALPHA, lw=0)
                has_band = True
            ind = [comp.get((d, t), np.nan) for d in durs]
            ax.plot(
                durs, ind, ls="none", marker="o", ms=4, mfc="none", mec=col, mew=0.75
            )
        style.axes(ax)
        _duration_axis(ax, durs)
        ax.set_ylabel("Depth (mm)")
        _plain_y(ax)
        if has_band:
            handles.append(
                Patch(facecolor=INK, alpha=BAND_ALPHA, label="5-95 % bootstrap band")
            )
        if comp:
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="o",
                    ms=4,
                    mfc="none",
                    mec=INK,
                    mew=0.75,
                    label="GEV fitted to one duration",
                )
            )
        _legend_below(fig, style, handles, ncol=4)
        stream = style.png(fig)
    return stream, list(style.warnings)


def ratio_figure(result):
    """L-moment ratio diagram: the sample (tau3, tau4) of each duration and
    the theoretical GEV, GLO and Pearson III curves and the Gumbel point."""
    from matplotlib.lines import Line2D

    from mayim_tools._common.report_figures import ChartStyle

    if not result.recommendations:
        raise ValueError("no ratio-diagram points to plot")
    style = ChartStyle()
    with style.context():
        fig = style.figure(RATIO_HEIGHT_CM)
        ax = fig.subplots()
        k = np.linspace(-0.6, 0.9, 300)
        gev = np.array([gev_tau3_tau4_exact(float(x)) for x in k])
        ax.plot(gev[:, 0], gev[:, 1], color=LINE_COLOURS[0], lw=DATA_LW, label="GEV")
        glo = np.array(
            [glo_tau3_tau4_exact(float(x)) for x in np.linspace(-0.6, 0.6, 200)]
        )
        ax.plot(glo[:, 0], glo[:, 1], color=LINE_COLOURS[3], lw=DATA_LW, label="GLO")
        t = np.linspace(-0.2, 0.6, 200)
        # Pearson III polynomial approximation (Hosking and Wallis 1997)
        pe3 = 0.1224 + 0.30115 * t**2 + 0.95812 * t**4 - 0.57488 * t**6 + 0.19383 * t**8
        ax.plot(t, pe3, color=LINE_COLOURS[2], lw=DATA_LW, ls="--", label="Pearson III")
        ax.plot(
            GUMBEL_TAU3, GUMBEL_TAU4, ls="none", marker="^", ms=7, color=INK,
            label="Gumbel",
        )  # fmt: skip
        x = [r.tau3_sample for r in result.recommendations]
        y = [r.tau4_sample for r in result.recommendations]
        ax.scatter(
            x, y, s=22, facecolor=LINE_COLOURS[4], edgecolor=INK, lw=0.5, zorder=5
        )
        for r in result.recommendations:
            ax.annotate(
                r.duration_label,
                (r.tau3_sample, r.tau4_sample),
                xytext=(3, 2),
                textcoords="offset points",
            )
        ax.set_xlim(min(-0.1, min(x) - 0.05), max(0.5, max(x) + 0.05))
        ax.set_ylim(min(0.0, min(y) - 0.03), max(0.4, max(y) + 0.03))
        ax.set_xlabel("L-skewness τ3 (-)")
        ax.set_ylabel("L-kurtosis τ4 (-)")
        style.axes(ax)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=5,
                mfc=LINE_COLOURS[4],
                mec=INK,
                mew=0.5,
                label="Sample (one per duration)",
            )
        )
        _legend_below(fig, style, handles, ncol=5)
        stream = style.png(fig)
    return stream, list(style.warnings)


def lmoment_figure(result):
    """Sample L-moments per duration (points) against the DDF model's
    smooth curves (lines): mean, L-CV and L-skewness."""
    from matplotlib.lines import Line2D

    from mayim_tools._common.report_figures import ChartStyle

    m = result.ddf_model
    if m is None:
        raise ValueError("no DDF model to plot")
    style = ChartStyle()
    ds = np.array(sorted(m.per_duration), float)
    dd = np.geomspace(ds.min(), ds.max(), 200)
    l1 = [m.per_duration[d]["l1"] for d in sorted(m.per_duration)]
    cv = [
        m.per_duration[d]["l2"] / m.per_duration[d]["l1"]
        for d in sorted(m.per_duration)
    ]
    t3 = [m.per_duration[d]["t3"] for d in sorted(m.per_duration)]
    with style.context():
        fig = style.figure(LMOM_HEIGHT_CM)
        axs = fig.subplots(1, 3)
        for ax, ys, ylabel, model in (
            (axs[0], l1, "Mean l1 (mm)", m.l1(dd)),
            (axs[1], cv, "L-CV (-)", m.cv(dd)),
            (axs[2], t3, "L-skewness τ3 (-)", m.t3(dd)),
        ):
            ax.plot(dd, model, color=LINE_COLOURS[3], lw=DATA_LW)
            ax.plot(
                ds, ys, ls="none", marker="o", ms=4, mfc=LINE_COLOURS[0], mec=INK,
                mew=0.5,
            )  # fmt: skip
            style.axes(ax)
            _duration_axis(ax, ds, ticks=(5, 60, 1440, 10080))
            ax.set_ylabel(ylabel)
            _plain_y(ax)
        handles = [
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=4,
                mfc=LINE_COLOURS[0],
                mec=INK,
                mew=0.5,
                label="Sample (one per duration)",
            ),
            Line2D([], [], color=LINE_COLOURS[3], lw=DATA_LW, label="DDF model"),
        ]
        _legend_below(fig, style, handles, ncol=2)
        stream = style.png(fig)
    return stream, list(style.warnings)


def _write(fn, result, path) -> bool:
    try:
        stream, _ = fn(result)
    except ImportError:
        return False
    except ValueError:
        return False
    with open(path, "wb") as fh:
        fh.write(stream.getvalue())
    return True


def write_ddf_chart(result, path) -> bool:
    return _write(ddf_figure, result, path)


def write_ratio_diagram(result, path) -> bool:
    return _write(ratio_figure, result, path)


def write_lmoment_smoothing_chart(result, path) -> bool:
    return _write(lmoment_figure, result, path)
