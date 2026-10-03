"""CSV / PNG export for Adjust Sub-daily Rainfall to DDF."""

from __future__ import annotations

import csv
import math

import numpy as np
import pandas as pd


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


def _original_times(result):
    """Timestamps back in the INPUT time base (the time-zone offset is
    only applied internally for year/month assignment)."""
    offset = float(result.metadata.get("timezone_offset_h", 0.0) or 0.0)
    return pd.DatetimeIndex(result.index) - pd.Timedelta(hours=offset)


def consistency_rows(result):
    """Before rows (with their full bootstrap) merged with the after
    rows (adjusted series), one row per duration x return period."""
    after = {(r["duration_min"], r["return_period_yr"]): r for r in result.after_rows}
    out = []
    for r in result.consistency_rows:
        row = dict(r)
        row = {
            (
                k
                if k in ("duration_min", "duration", "return_period_yr", "aep_pct")
                or k.startswith("reference")
                or k in ("n_years", "is_anchor", "status")
                else f"before_{k}"
            ): v
            for k, v in row.items()
        }
        a = after.get((r["duration_min"], r["return_period_yr"]))
        if a:
            row["role_in_adjustment"] = a["role"]
            row["after_implied_depth_mm"] = a["adjusted_implied_depth_mm"]
            row["after_F"] = a["F_after"]
            row["after_F_p05"] = a["F_after_p05"]
            row["after_F_p50"] = a["F_after_p50"]
            row["after_F_p95"] = a["F_after_p95"]
            row["after_decision"] = a["decision_after"]
        out.append(row)
    return out


def write_consistency_csv(result, path):
    return _write_rows(consistency_rows(result), path)


def write_adjusted_series_csv(result, path, site=None):
    """Adjusted series in the suite's extractor format first (Site,
    ValidTime, PrecipitationMM) so it can be fed straight into any other
    Mayim Tools rainfall tool, followed by audit columns. Missing values
    stay blank."""
    times = _original_times(result)
    adj = result.adjusted_values
    inp = result.input_values
    ev = result.step_event_id
    sc = result.step_scale
    pk = result.step_exponent
    site = site or ""
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Site",
                "ValidTime",
                "PrecipitationMM",
                "PrecipitationMM_input",
                "EventID",
                "Pass1Scale",
                "PeakConcentration",
            ]
        )
        for i in range(len(adj)):
            w.writerow(
                [
                    site,
                    times[i].strftime("%Y-%m-%d %H:%M:%S"),
                    _fmt(adj[i]),
                    _fmt(inp[i]),
                    int(ev[i]) if ev[i] >= 0 else "",
                    _fmt(sc[i]),
                    _fmt(pk[i]),
                ]
            )
            n += 1
    return n


def write_ensemble_csv(result, path):
    """Wide CSV: ValidTime + one adjusted-depth column per realisation."""
    if not result.ensemble:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ValidTime"])
        return 0
    times = _original_times(result)
    cols = [f"R{j + 1:03d}" for j in range(len(result.ensemble))]
    arr = np.vstack(result.ensemble).T
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ValidTime", *cols])
        for i in range(arr.shape[0]):
            w.writerow(
                [times[i].strftime("%Y-%m-%d %H:%M:%S")] + [_fmt(v) for v in arr[i]]
            )
    return arr.shape[0]


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


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

_RAMP = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]


def _panel(ax, rows, key50, key05, key95, title, tol, anchor):
    periods = sorted({r["return_period_yr"] for r in rows})
    ax.axhspan(1 - tol, 1 + tol, color="#e5e5e5", zorder=0)
    ax.axhline(1.0, color="#555555", lw=0.8, zorder=1)
    ax.axvline(anchor / 60.0, color="#999999", lw=0.8, ls="--", zorder=1)
    for i, t in enumerate(periods):
        sub = sorted(
            (r for r in rows if r["return_period_yr"] == t),
            key=lambda r: r["duration_min"],
        )
        x = np.array([r["duration_min"] / 60.0 for r in sub])
        colour = _RAMP[min(i, len(_RAMP) - 1)] if len(periods) <= len(_RAMP) else None
        p50 = np.array([r[key50] for r in sub], float)
        if key05:
            p05 = np.array([r[key05] for r in sub], float)
            p95 = np.array([r[key95] for r in sub], float)
            ax.fill_between(x, p05, p95, color=colour, alpha=0.10, lw=0)
        ax.plot(x, p50, marker="o", ms=3, lw=1.5, color=colour, label=f"1 in {t:g} yr")
    ticks = sorted({r["duration_min"] / 60.0 for r in rows})
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Duration (h)")
    ax.set_title(title, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def write_flatness_png(result, path):
    """F = implied / reference against duration: before adjustment (left)
    and after (right; validation durations marked). Returns False if
    matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    tol = float(result.metadata.get("tolerance_pct", 10.0)) / 100.0
    anchor = float(result.metadata.get("anchor_duration_min", 1440.0))
    has_after = bool(result.after_rows)
    fig, axes = plt.subplots(
        1,
        2 if has_after else 1,
        figsize=(10 if has_after else 6.5, 4.6),
        dpi=150,
        sharey=True,
        squeeze=False,
    )
    _panel(
        axes[0][0],
        result.consistency_rows,
        "F_p50",
        "F_p05",
        "F_p95",
        "Input series, scaled at the anchor only",
        tol,
        anchor,
    )
    axes[0][0].set_ylabel("F = implied / reference DDF")
    if has_after:
        ax = axes[0][1]
        _panel(
            ax,
            result.after_rows,
            "F_after",
            "F_after_p05",
            "F_after_p95",
            "Adjusted series (hollow = validation duration)",
            tol,
            anchor,
        )
        valid = {
            r["duration_min"] for r in result.after_rows if r["role"] == "validation"
        }
        for d in sorted(valid):
            ys = [r["F_after"] for r in result.after_rows if r["duration_min"] == d]
            ax.scatter(
                [d / 60.0] * len(ys),
                ys,
                s=40,
                facecolors="none",
                edgecolors="#333333",
                zorder=5,
                lw=0.8,
            )
    axes[0][-1].legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    stamp = result.status_stamp if "INDICATIVE" in result.status_stamp else ""
    fig.suptitle(
        "Rainfall structure vs reference DDF (anchor dashed, ±tolerance shaded)"
        + (f"\n{stamp}" if stamp else ""),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
