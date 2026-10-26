"""Word (.docx) report for Precipitation data to DDF, in the Mayim report
standard: title page, summary, tool and method, inputs and settings,
parameters and calculations, results, outputs, notes and limitations, and
references (house text styles, 9 pt tables with #ADADCC headers, numbered
captions, 16.5 cm house-style figures).

Requires python-docx (soft import) so the analysis and CSV outputs never
depend on it. Numbers have no thousands separators.
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path

import numpy as np

from . import charts

TITLE = "Rainfall Depth-Duration-Frequency Analysis"
TOOL_NAME = "Precipitation data to DDF"
TEXT_WIDTH_CM = 16.5
RPS_PER_TABLE = 6
AMS_DURATIONS_PER_TABLE = 8

REFERENCES = [
    "Ball, J., Babister, M., Nathan, R., Weeks, W., Weinmann, E., Retallick, M. "
    "and Testoni, I. (eds) (2019). Australian Rainfall and Runoff: A Guide to "
    "Flood Estimation, Book 2 (Rainfall Estimation). Commonwealth of Australia "
    "(Geoscience Australia).",
    "Barlow, R.E., Bartholomew, D.J., Bremner, J.M. and Brunk, H.D. (1972). "
    "Statistical Inference under Order Restrictions. Wiley, London.",
    "Efron, B. and Tibshirani, R.J. (1993). An Introduction to the Bootstrap. "
    "Chapman and Hall, New York.",
    "Hosking, J.R.M. and Wallis, J.R. (1997). Regional Frequency Analysis: An "
    "Approach Based on L-Moments. Cambridge University Press.",
    "Koutsoyiannis, D., Kozonis, D. and Manetas, A. (1998). A mathematical "
    "framework for studying rainfall intensity-duration-frequency relationships. "
    "Journal of Hydrology 206(1-2): 118-135.",
    "Overeem, A., Buishand, A. and Holleman, I. (2008). Rainfall depth-duration-"
    "frequency curves and their uncertainties. Journal of Hydrology 348(1-2): "
    "124-134.",
    "Roksvåg, T., Lutz, J., Grinde, L., Dyrrdal, A.V. and Thorarinsdottir, T.L. "
    "(2021). Consistent intensity-duration-frequency curves by post-processing of "
    "estimated Bayesian posterior quantiles. Journal of Hydrology 603: 127000.",
    "Smithers, J.C. and Schulze, R.E. (2003). Design Rainfall and Flood Estimation "
    "in South Africa. WRC Report K5/1060, Water Research Commission, Pretoria.",
    "Vogel, R.M. and Fennessey, N.M. (1993). L moment diagrams should replace "
    "product moment diagrams. Water Resources Research 29(6): 1745-1752.",
    "Weiss, L.L. (1964). Ratio of true to fixed-interval maximum rainfall. Journal "
    "of the Hydraulics Division, ASCE 90(1): 77-82.",
]


def _f(v, nd=2):
    """Number with ``nd`` decimals and no thousands separator; '-' when
    missing."""
    if v is None or v == "":
        return "-"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(x) or math.isinf(x):
        return "-"
    return f"{x:.{nd}f}"


def _txt(v) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "-"
    return str(v)


def _list(v) -> str:
    """A list joined with ', ' (never read as a thousands separator)."""
    if v is None or v == "":
        return "-"
    if isinstance(v, list | tuple):
        return ", ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in v) or "-"
    return ", ".join(p.strip() for p in str(v).split(",") if p.strip()) or "-"


def _even(n_cols: int, first: float) -> list[float]:
    rest = (TEXT_WIDTH_CM - first) / max(n_cols - 1, 1)
    return [first] + [rest] * (n_cols - 1)


def _rp(t) -> str:
    """Return period label: 1.0101 -> '1.01', 100.0 -> '100'."""
    return f"{round(float(t), 2):g}"


def _chunks(seq, n):
    seq = list(seq)
    return [seq[i : i + n] for i in range(0, len(seq), n)] or [[]]


def write_docx(result, path, inputs=None, context=None, charts_in=None) -> list[str]:
    """Writes the report and returns a list of notes (figures that could not
    be added, chart font substitutions). ``context`` gives project_number,
    project_name, site, tool_version, software, outputs [(label, path)],
    settings [(item, value)] and warnings. ``charts_in`` may hold ready PNG
    streams {"ddf", "ratio", "lmoment"} so a chart is drawn only once."""
    from mayim_tools._common.docx_report import Doc

    inputs = inputs or {}
    context = context or {}
    charts_in = charts_in or {}
    notes: list[str] = []
    md = result.metadata
    m = result.ddf_model
    rps = list(result.return_periods)
    ds_list = result.duration_series
    site = context.get("site") or inputs.get("site")
    series = inputs.get("series") or "-"
    n_years = max((len(d.values_mm) for d in ds_list), default=0)
    yrs = sorted({r.year for d in ds_list for r in d.year_records})
    excl = sorted({r.year for d in ds_list for r in d.year_records if not r.included})

    def figure(key, fn, caption):
        stream = charts_in.get(key)
        if stream is None:
            try:
                stream, warns = fn(result)
                notes.extend(warns)
            except ImportError:
                notes.append(f"Figure not added (matplotlib not available): {caption}")
                return
            except ValueError:
                return
        stream.seek(0)
        d.picture(stream, caption)

    d = Doc(house_style=True)
    purpose = (
        "This report derives a depth-duration-frequency (DDF) table from a recorded "
        "precipitation series"
        + (f" for {site}" if site else "")
        + ". One duration-consistent model is fitted across all durations, so the "
        "recommended design depths always increase with both duration and return "
        "period, for use in stormwater and flood design."
    )
    d.title_page(
        TITLE,
        purpose,
        [
            ("Project number", context.get("project_number") or "-"),
            ("Project / site", context.get("project_name") or site or "-"),
            ("Prepared by", "Mayim Consulting"),
            (
                "Tool and version",
                f"{TOOL_NAME}, version {context.get('tool_version') or '-'}",
            ),
            ("Software", context.get("software") or "-"),
            ("Report date", _dt.date.today().isoformat()),
            ("Document status", "Draft"),
        ],
        subtitle=context.get("project_name") or None,
    )

    # ------------------------------------------------------------ summary
    d.heading("Summary", 1)
    d.para(
        f"The rainfall series{f' for {site}' if site else ''} "
        f"({md.get('record_start')} to {md.get('record_end')}, {n_years} complete "
        f"years at a {_f(md.get('native_interval_min'), 0)} min time step) was "
        "analysed for "
        f"{len(ds_list)} durations from {ds_list[0].duration_label if ds_list else '-'} "
        f"to {ds_list[-1].duration_label if ds_list else '-'}. "
        + (
            f"The recommended DDF uses a duration-consistent {m.distribution} model."
            if m is not None
            else "The duration-consistent model could not be fitted; the recommended "
            "DDF falls back to the per-duration best fits."
        )
    )
    look = {(r["duration_min"], r["return_period_yr"]): r for r in result.ddf_rows}
    durs = sorted({r["duration_min"] for r in result.ddf_rows})
    key_durs = [x for x in (60, 1440) if x in durs] or durs[:2]
    key_rps = [t for t in rps if round(t) in (2, 50, 100)] or rps[:3]
    summary_rows = [
        ["Record", f"{md.get('record_start')} to {md.get('record_end')}"],
        ["Complete years used", str(n_years)],
        ["Durations analysed", str(len(ds_list))],
        ["DDF model distribution", _txt(md.get("ddf_model_distribution"))],
        ["Bootstrap replicates (5-95 % bounds)", _txt(md.get("n_bootstrap"))],
    ]
    for dmin in key_durs:
        for t in key_rps:
            r = look.get((dmin, t))
            if r:
                bounds = (
                    f" ({_f(r['lower_mm'], 1)} to {_f(r['upper_mm'], 1)})"
                    if np.isfinite(r["lower_mm"])
                    else ""
                )
                summary_rows.append(
                    [
                        f"Depth, {r['duration']}, {_rp(t)}-year (mm)",
                        f"{_f(r['depth_mm'], 1)}{bounds}",
                    ]
                )
    d.caption("Table", "Summary of the DDF analysis")
    d.table(["Item", "Value"], summary_rows, [8.0, 8.5])

    # ------------------------------------------------------------ method
    d.heading("Tool and method", 1)
    steps = [
        "The series is checked and regularised to its native time step. Missing data "
        "stay missing (never zero). Years below the completeness threshold are left "
        "out of every duration, so all durations share the same years. A time-zone "
        "offset, when given, is added before the calendar-year split.",
        "Annual maximum series (AMS) are extracted for each calendar year with "
        "sliding windows at each duration. Return periods are annual-maximum values "
        "(AEP = 1/T).",
    ]
    if md.get("fixed_interval_correction") not in (None, "off"):
        steps.append(
            "Maxima from fixed-interval data under-record the true maxima, because a "
            "storm rarely aligns with the clock. Each window of n native steps is "
            "multiplied by the Weiss (1964) factor 1/(1 - 1/(8n)), e.g. 1.143 for "
            "n = 1, 1.021 for n = 6 and 1.005 for n = 24."
        )
    steps += [
        "Diagnostics: the GEV, Gumbel, GLO and LP3 distributions are fitted to each "
        "duration separately by L-moments (Hosking and Wallis 1997), and the "
        "L-moment ratio diagram (Vogel and Fennessey 1993) ranks them per duration. "
        "Fitting each duration on its own can give curves that cross, so these fits "
        "are for review only.",
        "Recommended DDF: one model is fitted to all durations together "
        "(Koutsoyiannis et al. 1998; Overeem et al. 2008). The annual maximum X(d) "
        "= l1(d) × Y(d), where the mean l1(d) increases smoothly with duration and "
        "Y(d) is a growth variable with mean 1 from one distribution family.",
        (
            "The L-CV and L-skewness of Y(d) vary smoothly with log-duration, fitted "
            "by weighted least squares across durations; this borrows strength "
            "across durations and damps sampling noise (cf. the L-moment smoothing "
            "of the ARR 2016 IFDs; Ball et al. 2019)."
            if str(md.get("ddf_model_growth", "")).startswith("smoothed")
            else "One growth curve is used for all durations (simple scaling)."
        ),
        "The model table is checked for depths increasing with duration and return "
        "period; any remaining crossings are removed by isotonic regression "
        "(Barlow et al. 1972; Roksvåg et al. 2021).",
    ]
    if int(md.get("n_bootstrap") or 0):
        steps.append(
            f"Uncertainty: 5-95 % bounds from {md.get('n_bootstrap')} bootstrap "
            "replicates in which whole years are resampled (the same years for every "
            "duration) and the full model is refitted (Efron and Tibshirani 1993). "
            "The bounds express sampling uncertainty only."
        )
    for s in steps:
        d.numbered(s)

    # ------------------------------------------------------------ inputs
    d.heading("Inputs and settings", 1)
    rows = [
        ["Rainfall series", _txt(series)],
        ["Site", _txt(site)],
        ["Record", f"{md.get('record_start')} to {md.get('record_end')}"],
        ["Native time step (min)", _f(md.get("native_interval_min"), 0)],
        ["Time-zone offset applied (h)", _f(md.get("timezone_offset_h"), 1)],
        ["Year definition", _txt(md.get("year_definition", "calendar year"))],
        ["Years in record / used", f"{len(yrs)} / {n_years}"],
        [
            "Minimum year completeness (%)",
            _f(100 * float(md.get("min_completeness", 0.9)), 0),
        ],
        ["Years left out", _list(excl) if excl else "none"],
        ["Durations analysed", _list([x.duration_label for x in ds_list])],
        ["Return periods (years)", _list([round(t, 3) for t in rps])],
        ["Fixed-interval correction", _txt(md.get("fixed_interval_correction"))],
        [
            "Distributions fitted per duration",
            _list(list(md.get("distributions_fitted", []))),
        ],
        ["GEV fitting method (per duration)", _txt(md.get("gev_method"))],
        ["DDF model distribution", _txt(md.get("ddf_model_distribution"))],
        ["DDF model growth curve", _txt(md.get("ddf_model_growth"))],
        ["Bootstrap replicates", _txt(md.get("n_bootstrap"))],
        ["Random seed", _txt(md.get("random_seed"))],
    ] + [[a, _txt(b)] for a, b in context.get("settings", [])]
    d.caption("Table", "Inputs and settings")
    d.table(["Item", "Value"], rows, [7.0, 9.5])

    # ------------------------------------------------------------ parameters
    d.heading("Parameters and calculations", 1)
    prow = [
        ["AEP (-)", "1/T", "Annual exceedance probability of return period T"],
        [
            "Weiss factor W (-)",
            "1/(1 - 1/(8n))",
            "True to fixed-interval maximum for an n-step window (Weiss 1964)",
        ],
        [
            "Mean l1(d) (mm)",
            "λ1 (d/60) ((d+θ)/(60+θ))^-η",
            "Mean annual maximum at duration d (min), increasing with d "
            "(Koutsoyiannis et al. 1998)",
        ],
    ]
    if m is not None:
        prow += [
            ["λ1 (mm)", _f(m.lambda1, 2), "Mean annual maximum at 60 min"],
            ["θ (min)", _f(m.theta, 1), "Duration shift of the mean curve"],
            ["η (-)", _f(m.eta, 3), "Duration exponent of the mean curve (0 to 1)"],
            ["L-CV(60) (-)", _f(m.cv60, 3), "L-CV of the growth curve at 60 min"],
            ["β (-)", _f(m.beta, 3), "L-CV(d) = L-CV(60) (d/60)^β"],
            ["τ3(60) (-)", _f(m.tau3, 3), "L-skewness of the growth curve at 60 min"],
            ["g (-)", _f(m.tau3_slope, 3), "τ3(d) = τ3(60) + g ln(d/60)"],
            ["Growth distribution", _txt(m.distribution),
             "Distribution of Y(d) (Hosking and Wallis 1997)"],
            ["RMSE ln(mean) (-)", _f(m.rmse_log_l1, 3), "Fit of l1(d) to the sample means"],
            ["RMSE ln(L-CV) (-)", _f(m.rmse_log_l2, 3), "Fit of L-CV(d) to the samples"],
            ["RMSE τ3 (-)", _f(m.rmse_tau3, 3), "Fit of τ3(d) to the samples"],
        ]  # fmt: skip
    d.caption("Table", "Parameters and definitions")
    d.table(
        ["Parameter", "Value", "Definition / equation (source)"], prow, [4.0, 4.0, 8.5]
    )
    if m is not None and m.notes:
        for note in m.notes:
            d.bullet(note)

    # ------------------------------------------------------------ results
    d.heading("Results", 1)
    d.heading("Recommended DDF", 2)
    if result.ddf_rows:
        has_b = any(np.isfinite(r["lower_mm"]) for r in result.ddf_rows)
        d.para(
            "Design depths (mm) from the duration-consistent model"
            + (", with the 5-95 % bootstrap bounds below each depth." if has_b else ".")
        )
        for part in _chunks(rps, RPS_PER_TABLE):
            header = ["Duration"] + [f"{_rp(t)}-year" for t in part]
            rows = []
            for dmin in durs:
                row = [look[(dmin, rps[0])]["duration"]]
                for t in part:
                    r = look[(dmin, t)]
                    cell = _f(r["depth_mm"], 1)
                    if has_b:
                        cell += f"\n({_f(r['lower_mm'], 1)} to {_f(r['upper_mm'], 1)})"
                    row.append(cell)
                rows.append(row)
            d.caption(
                "Table",
                f"Recommended design depths (mm), {_rp(part[0])} to {_rp(part[-1])} years",
            )
            d.table(header, rows, _even(len(header), 2.2))
        figure(
            "ddf",
            charts.ddf_figure,
            "Recommended DDF curves (lines) with 5-95 % bootstrap bands, and the "
            "depths of a GEV fitted to each duration on its own (circles)",
        )
    else:
        d.para(
            "The duration-consistent model could not be fitted (see the processing "
            "messages); the recommended DDF CSV falls back to the per-duration "
            "best-fit table."
        )

    if m is not None:
        d.heading("DDF model fit", 2)
        d.para(
            "The model smooths the sample L-moments of the durations. Points far from "
            "the lines show durations the smooth trend does not follow."
        )
        figure(
            "lmoment",
            charts.lmoment_figure,
            "Sample L-moments per duration (points) and the DDF model (lines)",
        )

    d.heading("Annual maximum series", 2)
    rows = []
    for ds in ds_list:
        v = np.asarray(ds.values_mm, float)
        if not len(v):
            continue
        imax = int(np.argmax(v))
        rows.append(
            [
                ds.duration_label,
                str(len(v)),
                _f(v.mean(), 1),
                _f(v.min(), 1),
                _f(v.max(), 1),
                str(ds.years[imax]),
                _f(ds.fixed_interval_factor, 3),
            ]
        )
    d.caption("Table", "Annual maximum series per duration")
    d.table(
        ["Duration", "Years", "Mean (mm)", "Minimum (mm)", "Maximum (mm)",
         "Year of maximum", "Weiss factor (-)"],
        rows,
        _even(7, 2.5),
    )  # fmt: skip

    d.heading("Per-duration fits and ratio diagram", 2)
    recs = {r.duration_label: r for r in result.recommendations}
    rows = []
    for ds in ds_list:
        fit = next(
            (x for x in result.fits if x.duration_label == ds.duration_label), None
        )
        if fit is None:
            continue
        lm = fit.l_moments
        rec = recs.get(ds.duration_label)
        rows.append(
            [
                ds.duration_label,
                _f(lm.get("l1"), 1),
                _f(lm["l2"] / lm["l1"] if lm.get("l1") else None, 3),
                _f(lm.get("t3"), 3),
                _f(lm.get("t4"), 3),
                rec.recommended_distribution if rec else "-",
                (
                    "; ".join(f"{n} {_f(dist, 3)}" for n, dist in rec.ranking[1:3])
                    if rec
                    else "-"
                ),
            ]
        )
    d.caption("Table", "Sample L-moments and ratio-diagram ranking per duration")
    d.table(
        ["Duration", "l1 (mm)", "L-CV (-)", "τ3 (-)", "τ4 (-)", "Closest",
         "Next (distance)"],
        rows,
        [2.2, 1.8, 1.8, 1.8, 1.8, 2.3, 4.8],
    )  # fmt: skip
    figure(
        "ratio",
        charts.ratio_figure,
        "L-moment ratio diagram: sample points per duration and the theoretical "
        "curves. Scatter about the curves is expected from sampling variability",
    )

    if result.model_comparison:
        d.heading("Model against independent GEV fits", 2)
        d.para(
            "Recommended (model) depth and the depth of a GEV fitted to that duration "
            "alone, with the difference. Moderate differences are expected because "
            "the model pools information across durations; a large difference at one "
            "duration can point to a data problem (e.g. one dominant storm)."
        )
        show = [t for t in rps if round(t) in (2, 10, 50, 100)] or rps[:4]
        cl = {
            (c["duration_min"], c["return_period_yr"]): c
            for c in result.model_comparison
        }
        cdurs = sorted({c["duration_min"] for c in result.model_comparison})
        header = ["Duration"] + [
            f"{_rp(t)}-year model / GEV (difference)" for t in show
        ]
        rows = []
        for dmin in cdurs:
            row = [cl[(dmin, show[0])]["duration"] if (dmin, show[0]) in cl else "-"]
            for t in show:
                c = cl.get((dmin, t))
                row.append(
                    f"{_f(c['model_mm'], 1)} / {_f(c['independent_gev_mm'], 1)} "
                    f"({_f(c['difference_pct'], 0)} %)"
                    if c
                    else "-"
                )
            rows.append(row)
        d.caption("Table", "Model depths (mm) against independent GEV fits (mm)")
        d.table(header, rows, _even(len(header), 2.2))

    d.heading("Annual maxima by year", 2)
    years = sorted({y for ds in ds_list for y in ds.years})
    maps = [dict(zip(ds.years, ds.values_mm, strict=True)) for ds in ds_list]
    for part in _chunks(range(len(ds_list)), AMS_DURATIONS_PER_TABLE):
        if not part:
            continue
        header = ["Year"] + [ds_list[j].duration_label for j in part]
        rows = [[str(y)] + [_f(maps[j].get(y), 1) for j in part] for y in years]
        d.caption(
            "Table",
            f"Annual maxima (mm), {ds_list[part[0]].duration_label} to "
            f"{ds_list[part[-1]].duration_label}",
        )
        d.table(header, rows, _even(len(header), 1.6))

    # ------------------------------------------------------------ outputs
    outputs = context.get("outputs") or []
    if outputs:
        d.heading("Outputs", 1)
        d.caption("Table", "Files written")
        d.table(
            ["Output", "File"],
            [[a, Path(b).name if b else "-"] for a, b in outputs],
            [6.0, 10.5],
        )
        folders = sorted({str(Path(b).parent) for _, b in outputs if b})
        if folders:
            d.para("Folder(s): " + "; ".join(folders))

    # ------------------------------------------------------------ notes
    d.heading("Notes and limitations", 1)
    for txt in (
        "Single-site frequency analysis. Records of 20 to 40 years give reliable "
        "estimates up to about two to three times the record length; rarer return "
        "periods are extrapolations and the bounds widen accordingly. Compare with "
        "published regional DDF values (e.g. Smithers and Schulze 2003; ARR 2019 IFDs).",
        "Gridded products (ERA5, IMERG, CMORPH ...) are grid-cell averages: their DDF "
        "is an areal, product-specific DDF, typically low at short durations and rare "
        "return periods compared with gauge DDFs. Use Adjust Sub-daily Rainfall to "
        "DDF to reconcile a gridded series with a gauge-based DDF.",
        "The bootstrap bounds express sampling uncertainty only, not uncertainty in "
        "the data or in the choice of model.",
    ):
        d.bullet(txt)
    msgs = list(
        dict.fromkeys(list(result.warnings) + list(context.get("warnings", [])) + notes)
    )
    if msgs:
        d.label("Processing messages")
        for msg in msgs:
            d.bullet(msg)

    d.heading("References", 1)
    for ref in REFERENCES:
        d.para(ref)
    d.doc.save(str(path))
    return notes
