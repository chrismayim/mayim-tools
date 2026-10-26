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


# CSV outputs are UTF-8 with a byte-order mark (Mayim output standard), so
# Excel shows units correctly; the suite's CSV readers remove the mark.
CSV_ENCODING = "utf-8-sig"
KEY_PREFIX = "# "

CONSISTENCY_KEY = {
    "duration_min": "duration (min)",
    "duration": "duration label",
    "return_period_yr": "annual-maximum return period T (years); AEP = 1/T",
    "aep_pct": "annual exceedance probability (%)",
    "reference_depth_mm": "reference DDF depth (mm)",
    "reference_flag": "how the reference depth was found (table, interpolated, extrapolated)",
    "before_implied_depth_mm": "depth implied by the input series scaled at the anchor (mm)",
    "reference_lower_mm": "reference lower bound (mm), when the DDF gives one",
    "reference_upper_mm": "reference upper bound (mm), when the DDF gives one",
    "before_F": "ratio F = implied / reference depth before adjustment (-)",
    "before_F_p05": "5 % bootstrap value of F before adjustment (-)",
    "before_F_p50": "median bootstrap value of F before adjustment (-)",
    "before_F_p95": "95 % bootstrap value of F before adjustment (-)",
    "before_n_bootstrap_ok": "bootstrap replicates that could be fitted",
    "n_years": "complete years used",
    "before_decision": "verdict before adjustment (consistent, inconclusive, flatter, peakier)",
    "is_anchor": "TRUE for the anchor duration",
    "reference_tier": "reference tier (1 published gauge DDF, 2 other gauge DDF, 3 gridded only)",
    "status": "VALIDATED or INDICATIVE - NOT VALIDATED (tier 3)",
    "role_in_adjustment": "calibration (fitted rung), validation (not fitted) or anchor",
    "after_implied_depth_mm": "depth implied by the adjusted series (mm)",
    "after_F": "ratio F after adjustment (-)",
    "after_F_p05": "5 % bootstrap value of F after adjustment (-)",
    "after_F_p50": "median bootstrap value of F after adjustment (-)",
    "after_F_p95": "95 % bootstrap value of F after adjustment (-)",
    "after_decision": "verdict after adjustment",
}
EVENTS_KEY = {
    "event_id": "storm number",
    "start": "storm start (time-zone shifted time base)",
    "end": "storm end (time-zone shifted time base)",
    "event_depth_mm": "storm depth of the input series (mm)",
    "anchor_window_depth_mm": "largest anchor-duration depth inside the storm (mm)",
    "anchor_depth_corrected_mm": "anchor depth after the fixed-interval correction (mm)",
    "return_period_in_product_yr": "return period of that depth in the product's own record (years)",
    "scale_factor": "pass-1 scale factor applied to the whole storm (-)",
    "rescaled_anchor_depth_mm": "anchor depth after pass 1 (mm)",
    "peak_concentration_factor": "pass-2 peak concentration factor (-)",
    "range_flag": "in range, below or above the reference return periods",
    "touches_missing": "TRUE when the storm touches missing data (not reshaped)",
    "status": "VALIDATED or INDICATIVE - NOT VALIDATED (tier 3)",
}
LIBRARY_KEY = {
    "storm_id": "pattern number",
    "event_id": "storm number in the storm audit",
    "duration_min": "pattern duration (min)",
    "duration": "duration label",
    "start": "pattern start",
    "end": "pattern end",
    "month": "calendar month of the start",
    "raw_depth_mm": "depth of the input series over the pattern (mm)",
    "scale_factor": "pass-1 scale factor of the storm (-)",
    "rescaled_depth_mm": "depth of the adjusted series over the pattern (mm)",
    "return_period_yr": "return period of that depth in the reference DDF (years)",
    "aep_pct": "annual exceedance probability (%)",
    "aep_band": "AEP band (frequent, intermediate, rare)",
    "range_flag": "in range, below or above the reference return periods",
    "shape": "Early, Middle or Late peak",
    "timestep_min": "time step of the increments (min)",
    "n_increments": "number of increments",
    "embedded_burst_flag": "TRUE when a shorter burst inside is rarer than the pattern (ARR 2019)",
    "embedded_burst_detail": "the embedded burst found",
    "reference_tier": "reference tier",
    "status": "VALIDATED or INDICATIVE - NOT VALIDATED (tier 3)",
    "inc_001 ...": "share of the pattern depth in each time step (%), in order",
}


def _write_rows(rows, path, columns=None, key=None):
    """Write a list of dicts. Missing/NaN values become empty cells -
    never zero. ``key`` (column -> meaning) is written above the header as
    '# column: meaning' lines (read with pandas ``comment='#'``)."""
    if columns is None:
        columns = []
        for r in rows:
            for k in r:
                if k not in columns:
                    columns.append(k)
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        if key:
            f.write(f"{KEY_PREFIX}Column key\n")
            for col, meaning in key.items():
                if col in columns or col.endswith("..."):
                    f.write(f"{KEY_PREFIX}{col}: {meaning}\n")
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
    return _write_rows(consistency_rows(result), path, key=CONSISTENCY_KEY)


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
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
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
        with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
            csv.writer(f).writerow(["ValidTime"])
        return 0
    times = _original_times(result)
    cols = [f"R{j + 1:03d}" for j in range(len(result.ensemble))]
    arr = np.vstack(result.ensemble).T
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        w = csv.writer(f)
        w.writerow(["ValidTime", *cols])
        for i in range(arr.shape[0]):
            w.writerow(
                [times[i].strftime("%Y-%m-%d %H:%M:%S")] + [_fmt(v) for v in arr[i]]
            )
    return arr.shape[0]


def write_library_csv(result, path):
    return _write_rows(result.library_rows, path, key=LIBRARY_KEY)


def write_events_csv(result, path):
    return _write_rows(result.event_rows, path, key=EVENTS_KEY)


def write_metadata_csv(result, path):
    rows = [{"key": k, "value": v} for k, v in result.metadata.items()]
    rows += [
        {"key": f"warning_{i + 1}", "value": w} for i, w in enumerate(result.warnings)
    ]
    return _write_rows(rows, path, columns=["key", "value"])


# ---------------------------------------------------------------------------
# Chart (Mayim figure standard: 16.5 cm, Arial Narrow 9 pt, house colours)
# ---------------------------------------------------------------------------

# Line colours in the house order; yellow (the third house colour) is for
# fills only, so line series skip it.
_LINE_COLOURS = ["#FF0000", "#FFC000", "#00CC00", "#0000FF", "#00FFFF", "#FF00FF"]
_TOLERANCE_FILL = "#F2F2F2"
FIGURE_HEIGHT_CM = 9.5


def _panel(ax, style, rows, key50, key05, key95, title, tol, anchor):
    periods = sorted({r["return_period_yr"] for r in rows})
    ax.axhspan(1 - tol, 1 + tol, color=_TOLERANCE_FILL, zorder=0, lw=0)
    ax.axhline(1.0, color="#000000", lw=0.75, ls="--", zorder=1)
    ax.axvline(anchor / 60.0, color="#000000", lw=0.75, ls=":", zorder=1)
    for i, t in enumerate(periods):
        sub = sorted(
            (r for r in rows if r["return_period_yr"] == t),
            key=lambda r: r["duration_min"],
        )
        x = np.array([r["duration_min"] / 60.0 for r in sub])
        colour = _LINE_COLOURS[i % len(_LINE_COLOURS)]
        p50 = np.array([r[key50] for r in sub], float)
        if key05:
            p05 = np.array([r[key05] for r in sub], float)
            p95 = np.array([r[key95] for r in sub], float)
            ax.fill_between(x, p05, p95, color=colour, alpha=0.12, lw=0, zorder=2)
        ax.plot(
            x,
            p50,
            marker="o",
            ms=3,
            lw=1.5,
            color=colour,
            label=f"1 in {t:g} years",
            zorder=3,
        )
    ticks = sorted({r["duration_min"] / 60.0 for r in rows})
    ax.set_xscale("log")
    style.axes(ax)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(_null_locator())
    ax.set_xlabel("Duration (h)")
    ax.set_title(title)


def _null_locator():
    from matplotlib.ticker import NullLocator

    return NullLocator()


def flatness_figure(result):
    """(PNG bytes stream, style warnings) of the ratio F = implied /
    reference DDF depth against duration: before adjustment (left) and
    after (right, validation durations as hollow circles). Raises
    ImportError without matplotlib."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from mayim_tools._common.report_figures import ChartStyle

    style = ChartStyle()
    tol = float(result.metadata.get("tolerance_pct", 10.0)) / 100.0
    anchor = float(result.metadata.get("anchor_duration_min", 1440.0))
    has_after = bool(result.after_rows)
    with style.context():
        fig = style.figure(FIGURE_HEIGHT_CM)
        axes = fig.subplots(1, 2 if has_after else 1, sharey=True, squeeze=False)[0]
        _panel(
            axes[0],
            style,
            result.consistency_rows,
            "F_p50",
            "F_p05",
            "F_p95",
            "Before adjustment (input scaled at the anchor)",
            tol,
            anchor,
        )
        axes[0].set_ylabel("Ratio F = implied / reference depth (-)")
        if has_after:
            ax = axes[1]
            _panel(
                ax,
                style,
                result.after_rows,
                "F_after",
                "F_after_p05",
                "F_after_p95",
                "After adjustment",
                tol,
                anchor,
            )
            valid = {
                r["duration_min"]
                for r in result.after_rows
                if r["role"] == "validation"
            }
            for d in sorted(valid):
                ys = [r["F_after"] for r in result.after_rows if r["duration_min"] == d]
                ax.scatter(
                    [d / 60.0] * len(ys),
                    ys,
                    s=40,
                    facecolors="none",
                    edgecolors="#000000",
                    zorder=5,
                    lw=0.75,
                )
        handles, labels = axes[0].get_legend_handles_labels()
        handles += [
            Patch(facecolor="#000000", alpha=0.12, label="5-95 % bootstrap interval"),
            Patch(facecolor=_TOLERANCE_FILL, edgecolor="none", label="Tolerance"),
            Line2D([], [], color="#000000", lw=0.75, ls="--", label="F = 1"),
            Line2D([], [], color="#000000", lw=0.75, ls=":", label="Anchor duration"),
        ]
        if has_after:
            handles.append(
                Line2D(
                    [],
                    [],
                    ls="none",
                    marker="o",
                    mfc="none",
                    mec="#000000",
                    label="Validation duration",
                )
            )
        legend = fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=4,
            frameon=False,
            prop=style.font_props(style="italic"),
        )
        for text in legend.get_texts():
            text.set_color("#000000")
        stream = style.png(fig)
    return stream, list(style.warnings)


def write_flatness_png(result, path):
    """Writes the before/after chart (house style) to ``path``. Returns
    False if matplotlib is unavailable."""
    try:
        stream, _ = flatness_figure(result)
    except ImportError:
        return False
    with open(path, "wb") as f:
        f.write(stream.getvalue())
    return True
