"""
Stage 1 orchestrator: CSV -> AMS per duration -> fit ALL candidate
distributions -> ratio-diagram recommendation -> quantile table for
every distribution. Stage 2 (DDF table construction) is a deliberately
separate future addition - see dev/README.md.
"""

from __future__ import annotations

import pandas as pd

from .ams import build_regular_grid, compute_year_completeness, extract_ams_for_duration
from .distributions import fit_distribution, quantile_for
from .ratio_diagram import compare_distributions
from .schemas import (
    DistributionFit,
    DurationRecommendation,
    FrequencyAnalysisResult,
    QuantileEstimate,
)
from .timebase import (
    STANDARD_DURATIONS_MIN,
    applicable_durations,
    detect_native_interval,
)
from .validation import parse_and_validate

DEFAULT_EXCEEDANCE_PROBABILITIES = (
    0.99,
    0.9,
    0.5,
    0.2,
    0.1,
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
)
DEFAULT_DISTRIBUTIONS = ("GEV", "Gumbel", "GLO", "LP3")
MIN_YEARS_WARNING_THRESHOLD = (
    10  # matches the spirit of similar sample-size warnings elsewhere in this suite
)


def run_frequency_analysis(
    df: pd.DataFrame,
    timestamp_col: str,
    depth_col: str,
    timestamp_format: str | None = None,
    distributions: tuple = DEFAULT_DISTRIBUTIONS,
    gev_method: str = "L-moments",
    min_completeness: float = 0.90,
    min_years_warning: int = MIN_YEARS_WARNING_THRESHOLD,
    exceedance_probabilities: tuple = DEFAULT_EXCEEDANCE_PROBABILITIES,
    requested_durations_min: tuple = STANDARD_DURATIONS_MIN,
) -> FrequencyAnalysisResult:
    """Fits every distribution in `distributions` to every applicable
    duration, runs the L-moment ratio diagram diagnostic per duration
    to recommend one, and computes quantiles for ALL of them (not just
    the recommended one) - nothing is hidden behind the recommendation."""
    warnings: list = []
    metadata: dict = {
        "timestamp_col": timestamp_col,
        "depth_col": depth_col,
        "distributions_fitted": list(distributions),
        "gev_method": gev_method,
        "min_completeness": min_completeness,
        "year_definition": "calendar year",
        "processing_timestamp": str(pd.Timestamp.now()),
    }

    parsed, val_diag = parse_and_validate(
        df, timestamp_col, depth_col, timestamp_format
    )
    warnings += val_diag["warnings"]
    metadata.update({k: v for k, v in val_diag.items() if k != "warnings"})

    if len(parsed) < 2:
        raise ValueError(
            "Fewer than 2 valid rows after parsing - cannot determine a time interval."
        )

    native_interval_min = detect_native_interval(parsed["timestamp"])
    metadata["native_interval_min"] = native_interval_min

    durations = applicable_durations(native_interval_min, requested_durations_min)
    skipped = sorted(set(requested_durations_min) - set(durations))
    if skipped:
        warnings.append(
            f"Skipped duration(s) finer than the data's native interval ({native_interval_min:g} min): "
            f"{skipped} minutes."
        )
    if not durations:
        raise ValueError(
            f"No requested duration is coarse enough for this data's native interval "
            f"({native_interval_min:g} min)."
        )

    grid = build_regular_grid(parsed, native_interval_min)
    year_records = compute_year_completeness(
        grid, native_interval_min, min_completeness
    )
    n_excluded_years = sum(1 for yr in year_records if not yr.included)
    if n_excluded_years:
        warnings.append(
            f"{n_excluded_years} of {len(year_records)} year(s) excluded from ALL durations' AMS "
            f"for completeness below {min_completeness:.0%} - see the year-completeness diagnostic output."
        )

    duration_series_list = []
    fits = []
    quantiles = []
    recommendations = []

    for dur_min in durations:
        ds = extract_ams_for_duration(grid, dur_min, native_interval_min, year_records)
        duration_series_list.append(ds)

        n_years = len(ds.values_mm)
        if n_years < 3:
            warnings.append(
                f"{ds.duration_label}: only {n_years} valid year(s) - cannot fit a distribution, skipped."
            )
            continue
        if n_years < min_years_warning:
            warnings.append(
                f"{ds.duration_label}: only {n_years} valid year(s), below the recommended minimum "
                f"({min_years_warning}) for a reliable fit - treat this duration's results with caution."
            )

        duration_fits = (
            {}
        )  # distribution name -> DistributionFit, for the ratio-diagram step below
        for dist_name in distributions:
            method = gev_method if dist_name.upper() == "GEV" else "L-moments"
            try:
                fit_dict = fit_distribution(
                    ds.values_mm, distribution=dist_name, method=method
                )
            except Exception as e:
                warnings.append(
                    f"{ds.duration_label} [{dist_name}]: distribution fit failed ({e}), skipped."
                )
                continue

            fit_warnings = fit_dict.get("warnings", [])
            for w in fit_warnings:
                warnings.append(f"{ds.duration_label} [{dist_name}]: {w}")

            fit = DistributionFit(
                duration_label=ds.duration_label,
                distribution=dist_name.upper(),
                method=method,
                location_xi=fit_dict["xi"],
                scale_alpha=fit_dict["alpha"],
                shape_kappa=fit_dict["kappa"],
                n_years=n_years,
                l_moments=fit_dict["l_moments"],
                extra=fit_dict.get("extra", {}),
                warnings=fit_warnings,
            )
            fits.append(fit)
            duration_fits[dist_name.upper()] = fit

            for aep in exceedance_probabilities:
                F = 1 - aep  # non-exceedance probability
                depth = quantile_for(
                    dist_name,
                    F,
                    fit.location_xi,
                    fit.scale_alpha,
                    fit.shape_kappa,
                    fit.extra,
                )
                quantiles.append(
                    QuantileEstimate(
                        duration_label=ds.duration_label,
                        distribution=dist_name.upper(),
                        exceedance_probability=aep,
                        return_period_years=round(1 / aep, 3),
                        depth_mm=depth,
                    )
                )

        # Ratio-diagram diagnostic: uses the sample's own tau3/tau4
        # directly (not any one distribution's fitted approximation of
        # them), so this works even if only some distributions in
        # `distributions` were requested/fitted successfully.
        lm = (
            duration_fits[next(iter(duration_fits))].l_moments
            if duration_fits
            else None
        )
        if lm is not None and lm.get("t3") is not None and lm.get("t4") is not None:
            kappa_gev = (
                duration_fits["GEV"].shape_kappa if "GEV" in duration_fits else None
            )
            rd = compare_distributions(ds.duration_label, lm["t3"], lm["t4"], kappa_gev)
            recommendations.append(
                DurationRecommendation(
                    duration_label=ds.duration_label,
                    tau3_sample=rd.tau3_sample,
                    tau4_sample=rd.tau4_sample,
                    recommended_distribution=rd.recommended,
                    ranking=[(e.distribution, e.distance) for e in rd.entries],
                )
            )

    diagnostics = {
        "native_interval_min": native_interval_min,
        "durations_requested": list(requested_durations_min),
        "durations_applicable": durations,
        "durations_skipped_too_fine": skipped,
        "n_years_total": len(year_records),
        "n_years_excluded": n_excluded_years,
    }

    return FrequencyAnalysisResult(
        duration_series=duration_series_list,
        fits=fits,
        quantiles=quantiles,
        recommendations=recommendations,
        diagnostics=diagnostics,
        warnings=warnings,
        metadata=metadata,
    )
