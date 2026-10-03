"""CSV table exporters for Stage 1 results."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .schemas import FrequencyAnalysisResult


def write_parameters(result: FrequencyAnalysisResult, path: str | Path) -> None:
    """Stage 1.3: distribution parameters per duration per distribution
    (one row each). LP3 rows also carry the standard log-space
    mu/sigma/gamma reporting convention alongside the Pearson xi/beta/
    alpha reparameterization used in the shared Location/Scale/Shape
    columns, for direct comparability with GEV/Gumbel/GLO."""
    recommended_lookup = {
        r.duration_label: r.recommended_distribution for r in result.recommendations
    }

    with open(path, "w", newline="") as f:
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

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["Duration", "Distribution", "Recommended"] + [
            f"AEP_{aep}_RT{round(1/aep,1)}yr" for aep in aeps
        ]
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
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if not result.recommendations:
            w.writerow(["No recommendations computed - see warnings."])
            return
        max_candidates = max(len(r.ranking) for r in result.recommendations)
        header = ["Duration", "Tau3_sample", "Tau4_sample", "Recommended"]
        for i in range(1, max_candidates + 1):
            header += [f"Rank{i}_Distribution", f"Rank{i}_Tau4Distance"]
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

    with open(path, "w", newline="") as f:
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

    with open(path, "w", newline="") as f:
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
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Duration", "Year", "AnnualMaxMM"])
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
    with open(path, "w", newline="") as f:
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
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Key", "Value"])
        for k, v in result.metadata.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["Warnings"])
        for warn in result.warnings:
            w.writerow([warn])
