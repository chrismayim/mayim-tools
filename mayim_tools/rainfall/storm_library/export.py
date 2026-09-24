"""CSV / PNG export for the Storm Library & DDF Consistency Check."""

from __future__ import annotations

import csv
import math

import numpy as np


def _fmt(value, ndigits=4):
    if value is None:
        return ""
    if isinstance(value, bool | np.bool_):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float | np.floating):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(float(value), ndigits)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_rows(rows, path, columns=None):
    """Write a list of dicts. Missing/NaN values become empty cells -
    never zero."""
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in columns])
    return len(rows)


def write_consistency_csv(result, path):
    return _write_rows(result.consistency_rows, path)


def write_duration_summary_csv(result, path):
    return _write_rows(result.duration_summary, path)


def write_library_csv(result, path):
    return _write_rows(result.library_rows, path)


def write_events_csv(result, path):
    return _write_rows(result.event_rows, path)


def write_metadata_csv(result, path):
    rows = [{"key": k, "value": v} for k, v in result.metadata.items()]
    rows += [
        {"key": f"warning_{i + 1}", "value": w} for i, w in enumerate(result.warnings)
    ]
    return _write_rows(rows, path, columns=["key", "value"])


def write_flatness_png(result, path):
    """Flatness index F (bootstrap median, 5-95% band) against duration,
    one line per reference return period, with the tolerance band.
    Returns False (and writes nothing) if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    rows = result.consistency_rows
    tol = float(result.metadata.get("tolerance_pct", 10.0)) / 100.0
    anchor = float(result.metadata.get("anchor_duration_min", 1440.0))
    periods = sorted({r["return_period_yr"] for r in rows})
    # sequential single-hue ramp: frequent = light, rare = dark
    ramp = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.axhspan(
        1 - tol, 1 + tol, color="#e5e5e5", zorder=0, label=f"±{tol:.0%} tolerance"
    )
    ax.axhline(1.0, color="#555555", lw=0.8, zorder=1)
    ax.axvline(anchor / 60.0, color="#999999", lw=0.8, ls="--", zorder=1)
    for i, t in enumerate(periods):
        sub = sorted(
            (r for r in rows if r["return_period_yr"] == t),
            key=lambda r: r["duration_min"],
        )
        x = np.array([r["duration_min"] / 60.0 for r in sub])
        p50 = np.array([r["F_p50"] for r in sub], float)
        p05 = np.array([r["F_p05"] for r in sub], float)
        p95 = np.array([r["F_p95"] for r in sub], float)
        colour = ramp[min(i, len(ramp) - 1)] if len(periods) <= len(ramp) else None
        ax.fill_between(x, p05, p95, color=colour, alpha=0.12, lw=0)
        ax.plot(x, p50, marker="o", ms=3, lw=1.5, color=colour, label=f"1 in {t:g} yr")
    ax.set_xscale("log")
    ticks = sorted({r["duration_min"] / 60.0 for r in rows})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Duration (h)")
    ax.set_ylabel("Flatness index F = implied / reference")
    ax.set_title(
        "Storm structure vs reference DDF (anchor dashed)"
        + ("\n" + result.status_stamp if "INDICATIVE" in result.status_stamp else "")
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
