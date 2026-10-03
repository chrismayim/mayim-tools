"""PNG charts for Precipitation Data to DDF (matplotlib, soft import).

Every function returns True when the chart was written and False when
matplotlib is unavailable, so the CSV outputs never depend on it.
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


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _duration_axis(ax, durations):
    ax.set_xscale("log")
    ticks = [d for d in (5, 15, 30, 60, 120, 360, 720, 1440, 2880, 4320, 10080)]
    ticks = [t for t in ticks if min(durations) <= t <= max(durations)] or list(
        durations
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels([duration_label(t) for t in ticks], rotation=45, ha="right")
    ax.minorticks_off()
    ax.set_xlabel("Duration")


def write_ddf_chart(result, path, show_rps=(2, 10, 50, 100, 200)) -> bool:
    """Recommended (duration-consistent) DDF curves with 5-95% bands,
    and the independent per-duration GEV quantiles as markers."""
    plt = _plt()
    if plt is None or not result.ddf_rows:
        return False
    rps = [t for t in result.return_periods if round(t, 3) in show_rps] or list(
        result.return_periods
    )
    durs = sorted({r["duration_min"] for r in result.ddf_rows})
    look = {(r["duration_min"], r["return_period_yr"]): r for r in result.ddf_rows}
    comp = {
        (c["duration_min"], c["return_period_yr"]): c["independent_gev_mm"]
        for c in result.model_comparison
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(rps)))
    for t, col in zip(rps, colours, strict=False):
        y = [look[(d, t)]["depth_mm"] for d in durs]
        lo = [look[(d, t)]["lower_mm"] for d in durs]
        up = [look[(d, t)]["upper_mm"] for d in durs]
        ax.plot(durs, y, color=col, lw=1.8, label=f"{t:g}-yr")
        if np.all(np.isfinite(lo)):
            ax.fill_between(durs, lo, up, color=col, alpha=0.15, lw=0)
        ind = [comp.get((d, t), np.nan) for d in durs]
        ax.plot(durs, ind, ls="none", marker="o", ms=3.5, mfc="none", color=col)
    _duration_axis(ax, durs)
    ax.set_ylabel("Depth (mm)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(
        "Recommended DDF (lines, 5-95% band) and independent GEV fits (circles)",
        fontsize=9,
    )
    ax.legend(title="Return period", fontsize=8, title_fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_ratio_diagram(result, path) -> bool:
    """L-moment ratio diagram: sample (tau3, tau4) per duration and the
    theoretical GEV, GLO and Pearson III curves and the Gumbel point."""
    plt = _plt()
    if plt is None or not result.recommendations:
        return False
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=150)
    k = np.linspace(-0.6, 0.9, 300)
    gev = np.array([gev_tau3_tau4_exact(float(x)) for x in k])
    ax.plot(gev[:, 0], gev[:, 1], label="GEV", color="C0")
    glo = np.array([glo_tau3_tau4_exact(float(x)) for x in np.linspace(-0.6, 0.6, 200)])
    ax.plot(glo[:, 0], glo[:, 1], label="GLO", color="C1")
    t = np.linspace(-0.2, 0.6, 200)
    # Pearson III polynomial approximation (Hosking & Wallis 1997, Appendix)
    pe3 = 0.1224 + 0.30115 * t**2 + 0.95812 * t**4 - 0.57488 * t**6 + 0.19383 * t**8
    ax.plot(t, pe3, label="Pearson III", color="C2", ls="--")
    ax.plot(GUMBEL_TAU3, GUMBEL_TAU4, "k^", ms=7, label="Gumbel")
    x = [r.tau3_sample for r in result.recommendations]
    y = [r.tau4_sample for r in result.recommendations]
    ax.scatter(x, y, s=18, color="crimson", zorder=5, label="Sample (per duration)")
    for r in result.recommendations:
        ax.annotate(
            r.duration_label,
            (r.tau3_sample, r.tau4_sample),
            fontsize=6,
            xytext=(3, 2),
            textcoords="offset points",
        )
    mt3 = float(np.nanmean(x)) if x else 0.2
    ax.set_xlim(min(-0.1, min(x) - 0.05), max(0.5, max(x) + 0.05))
    ax.set_ylim(min(0.0, min(y) - 0.03), max(0.4, max(y) + 0.03))
    ax.axvline(mt3, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("L-skewness tau3")
    ax.set_ylabel("L-kurtosis tau4")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("L-moment ratio diagram", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def write_lmoment_smoothing_chart(result, path) -> bool:
    """Sample L-moments per duration against the DDF model's smooth
    curves (mean, L-CV, L-skewness) - shows what the model smooths."""
    plt = _plt()
    m = result.ddf_model
    if plt is None or m is None:
        return False
    ds = np.array(sorted(m.per_duration))
    dd = np.geomspace(ds.min(), ds.max(), 200)
    l1 = [m.per_duration[d]["l1"] for d in ds]
    cv = [m.per_duration[d]["l2"] / m.per_duration[d]["l1"] for d in ds]
    t3 = [m.per_duration[d]["t3"] for d in ds]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150)
    for ax, ys, yl, model in (
        (axs[0], l1, "Mean of AMS l1 (mm)", m.l1(dd)),
        (axs[1], cv, "L-CV l2/l1", m.cv(dd)),
        (axs[2], t3, "L-skewness tau3", m.t3(dd)),
    ):
        ax.plot(ds, ys, "o", ms=4, color="crimson", label="Sample")
        ax.plot(dd, model, color="C0", label="Model")
        _duration_axis(ax, ds)
        ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    axs[0].set_yscale("log")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
