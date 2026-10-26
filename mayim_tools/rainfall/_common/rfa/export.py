"""CSV table exporters for Precipitation data to DDF.

Mayim output standard: every CSV is UTF-8 with a byte-order mark (so Excel
opens it cleanly). The diagnostic CSVs, whose headings are abbreviated,
start with a column key as '# column: meaning' lines (read them with pandas
``comment='#'``). The recommended DDF CSV has no key: its headings are
written out in full and other tools read it as a DDF input.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .schemas import FrequencyAnalysisResult

CSV_ENCODING = "utf-8-sig"
KEY_PREFIX = "# "

PARAMETERS_KEY = {
    "Duration": "duration",
    "Distribution": "distribution fitted (GEV, GUMBEL, GLO or LP3)",
    "Method": "fitting method",
    "Recommended": "True for the distribution closest on the L-moment ratio diagram",
    "NYears": "number of years in the annual maximum series",
    "LocationXi": "location parameter xi (mm)",
    "ScaleAlpha": "scale parameter alpha (mm)",
    "ShapeKappa": "shape parameter kappa (-; Hosking sign convention)",
    "LP3_Log10Mean": "LP3 only: mean of log10 of the annual maxima",
    "LP3_Log10SD": "LP3 only: standard deviation of log10 of the annual maxima",
    "LP3_Log10Skew": "LP3 only: skewness of log10 of the annual maxima",
    "L1": "first sample L-moment, the mean (mm)",
    "L2": "second sample L-moment (mm)",
    "L3": "third sample L-moment (mm)",
    "T3_Lskewness": "sample L-skewness tau3 = L3/L2 (-)",
    "T4_Lkurtosis": "sample L-kurtosis tau4 (-)",
}
QUANTILES_KEY = {
    "Duration": "duration",
    "Distribution": "distribution fitted (GEV, GUMBEL, GLO or LP3)",
    "Recommended": "True for the distribution closest on the L-moment ratio diagram",
    "AEP_<p>_RT<T>yr ...": (
        "design depth (mm) at annual exceedance probability p, return period "
        "T = 1/p years, from that duration's own fit"
    ),
}
RECOMMENDATIONS_KEY = {
    "Duration": "duration",
    "Tau3_sample": "sample L-skewness tau3 (-)",
    "Tau4_sample": "sample L-kurtosis tau4 (-)",
    "Recommended": "distribution closest to the sample point on the ratio diagram",
    "Rank<n>_Distribution ...": "n-th closest distribution",
    "Rank<n>_Tau4Distance ...": (
        "its distance from the sample point on the ratio diagram (-)"
    ),
}
AMS_KEY = {
    "Duration": "duration",
    "Year": "calendar year (after any time-zone offset)",
    "AnnualMaxMM": (
        "annual maximum depth for that duration (mm; fixed-interval correction "
        "included when it is on)"
    ),
}
YEAR_COMPLETENESS_KEY = {
    "Year": "calendar year (after any time-zone offset)",
    "NExpectedReadings": "readings expected at the native time step",
    "NValidReadings": "valid (not missing) readings",
    "Completeness": "valid / expected readings (0-1)",
    "Included": "True if the year is used in the annual maximum series",
    "ExclusionReason": "why the year was left out",
}


def _open(path):
    return open(path, "w", newline="", encoding=CSV_ENCODING)


def _write_key(f, key: dict, columns) -> None:
    """'# Column key' then '# column: meaning' for the columns present (and
    for pattern entries ending in '...')."""
    columns = set(columns)
    f.write(f"{KEY_PREFIX}Column key\n")
    for col, meaning in key.items():
        if col in columns or col.endswith("..."):
            f.write(f"{KEY_PREFIX}{col}: {meaning}\n")


def write_parameters(result: FrequencyAnalysisResult, path: str | Path) -> None:
    """Stage 1.3: distribution parameters per duration per distribution
    (one row each). LP3 rows also carry the standard log-space
    mu/sigma/gamma reporting convention alongside the Pearson xi/beta/
    alpha reparameterization used in the shared Location/Scale/Shape
    columns, for direct comparability with GEV/Gumbel/GLO."""
    recommended_lookup = {
        r.duration_label: r.recommended_distribution for r in result.recommendations
    }

    with _open(path) as f:
        _write_key(f, PARAMETERS_KEY, PARAMETERS_KEY)
        w = csv.writer(f)
        w.writerow(
            [
                "Duration",
                "Distribution",
                "Method",
                "Recommended",
                "NYears",
                "LocationXi",
                "ScaleAlpha",
                "ShapeKappa",
                "LP3_Log10Mean",
                "LP3_Log10SD",
                "LP3_Log10Skew",
                "L1",
                "L2",
                "L3",
                "T3_Lskewness",
                "T4_Lkurtosis",
            ]
        )
        for fit in result.fits:
            lm = fit.l_moments
            is_recommended = (
                recommended_lookup.get(fit.duration_label) == fit.distribution
            )
            extra = fit.extra or {}
            w.writerow(
                [
                    fit.duration_label,
                    fit.distribution,
                    fit.method,
                    is_recommended,
                    fit.n_years,
                    round(fit.location_xi, 4),
                    round(fit.scale_alpha, 4),
                    round(fit.shape_kappa, 5) if fit.shape_kappa is not None else "",
                    round(extra["mu_log10"], 5) if "mu_log10" in extra else "",
                    round(extra["sigma_log10"], 5) if "sigma_log10" in extra else "",
                    round(extra["gamma_log10"], 5) if "gamma_log10" in extra else "",
                    round(lm["l1"], 4),
                    round(lm["l2"], 4),
                    round(lm["l3"], 4) if lm.get("l3") is not None else "",
                    round(lm["t3"], 5) if lm.get("t3") is not None else "",
                    round(lm["t4"], 5) if lm.get("t4") is not None else "",
                ]
            )


def write_quantiles(result: FrequencyAnalysisResult, path: str | Path) -> None:
    """Stage 1.4: quantile (design depth) table, wide format - one row
    per duration per distribution, one column per exceedance
    probability."""
    rows_keys = sorted({(q.duration_label, q.distribution) for q in result.quantiles})
    aeps = sorted({q.exceedance_probability for q in result.quantiles}, reverse=True)
    recommended_lookup = {
        r.duration_label: r.recommended_distribution for r in result.recommendations
    }

    lookup = {
        (q.duration_label, q.distribution, q.exceedance_probability): q.depth_mm
        for q in result.quantiles
    }

    with _open(path) as f:
        w = csv.writer(f)
        header = ["Duration", "Distribution", "Recommended"] + [
            f"AEP_{aep}_RT{round(1/aep,1)}yr" for aep in aeps
        ]
        _write_key(f, QUANTILES_KEY, header)
        w.writerow(header)
        for dur, dist in rows_keys:
            is_recommended = recommended_lookup.get(dur) == dist
            row = [dur, dist, is_recommended] + [
                round(lookup[(dur, dist, aep)], 3) if (dur, dist, aep) in lookup else ""
                for aep in aeps
            ]
            w.writerow(row)


def write_recommendations(result: FrequencyAnalysisResult, path: str | Path) -> None:
    """L-moment ratio diagram diagnostic: per duration, the sample's
    actual (tau3, tau4) point, each candidate distribution's distance
    from it (best fit first), and the recommended distribution."""
    with _open(path) as f:
        w = csv.writer(f)
        if not result.recommendations:
            w.writerow(["No recommendations computed - see warnings."])
            return
        max_candidates = max(len(r.ranking) for r in result.recommendations)
        header = ["Duration", "Tau3_sample", "Tau4_sample", "Recommended"]
        for i in range(1, max_candidates + 1):
            header += [f"Rank{i}_Distribution", f"Rank{i}_Tau4Distance"]
        _write_key(f, RECOMMENDATIONS_KEY, header)
        w.writerow(header)
        for r in result.recommendations:
            row = [
                r.duration_label,
                round(r.tau3_sample, 5),
                round(r.tau4_sample, 5),
                r.recommended_distribution,
            ]
            for dist, dist_val in r.ranking:
                row += [dist, round(dist_val, 5)]
            w.writerow(row)


def write_recommended_ddf(result: FrequencyAnalysisResult, path: str | Path) -> None:
    """Recommended (design) DDF table from the duration-consistent DDF
    model: one row per duration, depths that always increase with both
    duration and return period, plus 5-95% bootstrap bounds when the
    bootstrap was run.

    Column naming ('{RT}yr Depth (mm)', '{RT}yr Lower (mm)', '{RT}yr
    Upper (mm)') matches the Design Rainfall (South Africa) CSV layout,
    so the table feeds straight into DDF to Hyetographs, Design Storm
    Ensembles and Adjust Sub-daily Rainfall to DDF. Falls back to the
    per-duration best-fit table (with a note) if the model could not be
    fitted."""
    if not result.ddf_rows:
        write_per_duration_best_fit_ddf(result, path)
        return
    rps = list(result.return_periods)
    durs = sorted({r["duration_min"] for r in result.ddf_rows})
    look = {(r["duration_min"], r["return_period_yr"]): r for r in result.ddf_rows}
    has_bounds = any(np.isfinite(r["lower_mm"]) for r in result.ddf_rows)
    model = result.ddf_model
    label = f"{model.distribution} (duration-consistent)" if model else ""

    def fmt(v):
        return round(float(v), 3) if np.isfinite(v) else ""

    with _open(path) as f:
        w = csv.writer(f)
        header = ["Duration", "Distribution"]
        for t in rps:
            header.append(f"{round(t, 3):g}yr Depth (mm)")
            if has_bounds:
                header += [
                    f"{round(t, 3):g}yr Lower (mm)",
                    f"{round(t, 3):g}yr Upper (mm)",
                ]
        w.writerow(header)
        for d in durs:
            row = [look[(d, rps[0])]["duration"], label]
            for t in rps:
                r = look[(d, t)]
                row.append(fmt(r["depth_mm"]))
                if has_bounds:
                    row += [fmt(r["lower_mm"]), fmt(r["upper_mm"])]
            w.writerow(row)


def write_per_duration_best_fit_ddf(
    result: FrequencyAnalysisResult, path: str | Path
) -> None:
    """DIAGNOSTIC (not the design table): one row per duration using only
    that duration's own ratio-diagram-recommended distribution. Because
    each duration is fitted and chosen independently, curves from this
    table can cross between durations - use write_recommended_ddf()
    for design. Originally: one row per duration, using ONLY
    that duration's ratio-diagram-recommended distribution's quantiles
    - the single clean design table most downstream use actually
    wants, rather than all 4 distributions x all durations.

    Column naming ('{RT}yr Depth (mm)') matches the Design Rainfall
    plugin's own CSV output format exactly (and is what the SCS Design
    Storm plugin's parser expects), so this table can be fed straight
    into that plugin with zero reformatting - the natural downstream
    connection in this suite."""
    recommended_lookup = {
        r.duration_label: r.recommended_distribution for r in result.recommendations
    }
    duration_order = {
        ds.duration_label: ds.duration_minutes for ds in result.duration_series
    }

    durations = sorted(
        recommended_lookup.keys(), key=lambda d: duration_order.get(d, 0)
    )
    aeps = sorted({q.exceedance_probability for q in result.quantiles}, reverse=True)
    lookup = {
        (q.duration_label, q.distribution, q.exceedance_probability): q.depth_mm
        for q in result.quantiles
    }

    with _open(path) as f:
        w = csv.writer(f)
        if not durations:
            w.writerow(["No recommended distributions available - see warnings."])
            return
        header = ["Duration", "RecommendedDistribution"] + [
            f"{round(1/aep, 3):g}yr Depth (mm)" for aep in aeps
        ]
        w.writerow(header)
        for dur in durations:
            dist = recommended_lookup[dur]
            row = [dur, dist] + [
                round(lookup[(dur, dist, aep)], 3) if (dur, dist, aep) in lookup else ""
                for aep in aeps
            ]
            w.writerow(row)


def write_ams(result: FrequencyAnalysisResult, path: str | Path) -> None:
    """Stage 1.1: the annual maximum series itself, long format - one
    row per duration per year (useful for auditing/plotting)."""
    with _open(path) as f:
        _write_key(f, AMS_KEY, AMS_KEY)
        w = csv.writer(f)
        w.writerow(list(AMS_KEY))
        for ds in result.duration_series:
            for year, value in zip(ds.years, ds.values_mm, strict=False):
                w.writerow([ds.duration_label, year, round(value, 3)])


def write_year_completeness(result: FrequencyAnalysisResult, path: str | Path) -> None:
    """Transparency/audit table: every year considered, included or not."""
    if not result.duration_series:
        return
    year_records = result.duration_series[
        0
    ].year_records  # identical across all durations by construction
    with _open(path) as f:
        _write_key(f, YEAR_COMPLETENESS_KEY, YEAR_COMPLETENESS_KEY)
        w = csv.writer(f)
        w.writerow(
            [
                "Year",
                "NExpectedReadings",
                "NValidReadings",
                "Completeness",
                "Included",
                "ExclusionReason",
            ]
        )
        for yr in year_records:
            w.writerow(
                [
                    yr.year,
                    yr.n_expected,
                    yr.n_valid,
                    round(yr.completeness, 4),
                    yr.included,
                    yr.exclusion_reason or "",
                ]
            )


def write_metadata(result: FrequencyAnalysisResult, path: str | Path) -> None:
    with _open(path) as f:
        w = csv.writer(f)
        w.writerow(["Key", "Value"])
        for k, v in result.metadata.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["Warnings"])
        for warn in result.warnings:
            w.writerow([warn])
