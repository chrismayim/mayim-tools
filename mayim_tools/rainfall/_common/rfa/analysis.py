"""
Orchestrator: CSV -> AMS per duration (optionally fixed-interval
corrected) -> fit ALL candidate distributions per duration (diagnostic)
-> ratio-diagram recommendation per duration (diagnostic) -> ONE
duration-consistent DDF model across all durations (the recommended DDF)
-> year-bootstrap 5-95% bounds on the recommended DDF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ams import build_regular_grid, compute_year_completeness, extract_ams_for_duration
from .ddf_model import enforce_consistency, fit_ddf_model
from .distributions import fit_distribution, fit_gev_lmoments, quantile_for
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
    duration_label,
    whole_multiple_durations,
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
    fixed_interval_correction: bool = False,
    timezone_offset_h: float = 0.0,
    model_distribution: str = "GEV",
    vary_cv: bool = True,
    n_bootstrap: int = 0,
    random_seed: int = 42,
    whole_multiples_only: bool = False,
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

    if timezone_offset_h:
        parsed = parsed.copy()
        parsed["timestamp"] = parsed["timestamp"] + pd.Timedelta(
            hours=timezone_offset_h
        )
    metadata["timezone_offset_h"] = timezone_offset_h
    metadata["fixed_interval_correction"] = (
        "Weiss (1964) 1/(1-1/(8n))" if fixed_interval_correction else "off"
    )

    if len(parsed) < 2:
        raise ValueError(
            "Fewer than 2 valid rows after parsing - cannot determine a time interval."
        )

    native_interval_min = detect_native_interval(parsed["timestamp"])
    metadata["native_interval_min"] = native_interval_min
    metadata["record_start"] = str(parsed["timestamp"].min())
    metadata["record_end"] = str(parsed["timestamp"].max())

    if whole_multiples_only:
        durations, skipped = whole_multiple_durations(
            native_interval_min, requested_durations_min
        )
    else:
        durations = applicable_durations(native_interval_min, requested_durations_min)
    skipped = sorted(set(requested_durations_min) - set(durations))
    not_multiple = [d for d in skipped if d >= native_interval_min]
    if not_multiple and whole_multiples_only:
        warnings.append(
            f"Skipped duration(s) that are not whole multiples of the data's native "
            f"interval ({native_interval_min:g} min): {not_multiple} minutes."
        )
    if skipped and not whole_multiples_only:
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
        if fixed_interval_correction:
            n_int = ds.n_native_intervals_per_window
            fac = 1.0 / (1.0 - 1.0 / (8.0 * n_int))
            ds.values_mm = tuple(v * fac for v in ds.values_mm)
            ds.fixed_interval_factor = fac
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

    # ---- duration-consistent DDF model (the recommended DDF) -------------
    return_periods = [1.0 / a for a in exceedance_probabilities]
    ams = {
        ds.duration_minutes: np.array(ds.values_mm)
        for ds in duration_series_list
        if len(ds.values_mm) >= 5
    }
    ddf_model = None
    ddf_rows = []
    comparison = []
    try:
        ddf_model = fit_ddf_model(ams, model_distribution, vary_cv=vary_cv)
    except ValueError as e:
        warnings.append(f"Duration-consistent DDF model not fitted: {e}")

    if ddf_model is not None:
        raw_table = ddf_model.table(ddf_model.durations, return_periods)
        table, iso_change = enforce_consistency(
            raw_table, ddf_model.durations, return_periods
        )
        if iso_change > 1e-6:
            ddf_model.notes.append(
                "Small crossings of the smoothed model were removed by isotonic "
                f"regression (largest change {100 * iso_change:.1f}%)."
            )
            if iso_change > 0.05:
                warnings.append(
                    "The DDF model needed an isotonic correction of "
                    f"{100 * iso_change:.1f}% to keep depths increasing with duration "
                    "and return period - check the longest durations / rarest AEPs."
                )
        # model vs independent per-duration GEV (diagnostic only)
        comparison = []
        for d in ddf_model.durations:
            lm = ddf_model.per_duration[d]
            try:
                gxi, gal, gka = fit_gev_lmoments(lm["l1"], lm["l2"], lm["t3"])
            except (ValueError, FloatingPointError, ZeroDivisionError):
                continue
            for k, t in enumerate(return_periods):
                ind = quantile_for("GEV", 1.0 - 1.0 / t, gxi, gal, gka)
                comparison.append(
                    {
                        "duration_min": d,
                        "duration": duration_label(d),
                        "return_period_yr": t,
                        "model_mm": table[d][k],
                        "independent_gev_mm": float(ind),
                        "difference_pct": (
                            100.0 * (table[d][k] - ind) / ind if ind > 0 else np.nan
                        ),
                    }
                )
        big = [
            c
            for c in comparison
            if c["return_period_yr"] <= 100 and abs(c["difference_pct"]) > 25
        ]
        if big:
            worst = max(big, key=lambda c: abs(c["difference_pct"]))
            warnings.append(
                f"The DDF model differs from the independent GEV fit by more than 25% "
                f"at {len(big)} duration/return-period combination(s) (worst: "
                f"{worst['duration']}, {worst['return_period_yr']:g}-yr, "
                f"{worst['difference_pct']:+.0f}%). Differences of this size are "
                "usually within the sampling uncertainty of a single-site record, "
                "but review the per-duration fits."
            )
        boot = {d: [] for d in ddf_model.durations}
        if n_bootstrap:
            by_year = {
                ds.duration_minutes: dict(zip(ds.years, ds.values_mm, strict=True))
                for ds in duration_series_list
                if ds.duration_minutes in ams
            }
            all_years = sorted({y for v in by_year.values() for y in v})
            rng = np.random.default_rng(random_seed)
            n_fail = 0
            for _ in range(int(n_bootstrap)):
                sample = rng.choice(all_years, size=len(all_years), replace=True)
                ams_b = {
                    d: np.array([m[y] for y in sample if y in m])
                    for d, m in by_year.items()
                }
                try:
                    mb = fit_ddf_model(
                        ams_b,
                        model_distribution,
                        vary_cv=vary_cv,
                        start=(ddf_model.theta, ddf_model.eta),
                    )
                    tb, _ = enforce_consistency(
                        mb.table(ddf_model.durations, return_periods),
                        ddf_model.durations,
                        return_periods,
                    )
                except (ValueError, FloatingPointError, ZeroDivisionError):
                    n_fail += 1
                    continue
                for d in ddf_model.durations:
                    boot[d].append(tb[d])
            if n_fail:
                warnings.append(
                    f"{n_fail} of {n_bootstrap} bootstrap replicates of the DDF model "
                    "failed and were skipped."
                )
        for d in ddf_model.durations:
            arr = np.array(boot[d]) if boot[d] else None
            for k, t in enumerate(return_periods):
                lo = up = np.nan
                if arr is not None and len(arr) >= 10:
                    lo, up = np.nanpercentile(arr[:, k], [5, 95])
                ddf_rows.append(
                    {
                        "duration_min": d,
                        "duration": duration_label(d),
                        "return_period_yr": t,
                        "aep": 1.0 / t,
                        "depth_mm": table[d][k],
                        "lower_mm": float(lo),
                        "upper_mm": float(up),
                    }
                )
        metadata.update(
            {
                "ddf_model_distribution": ddf_model.distribution,
                "ddf_model_theta_min": ddf_model.theta,
                "ddf_model_eta": ddf_model.eta,
                "ddf_model_lambda1_60min": ddf_model.lambda1,
                "ddf_model_cv60": ddf_model.cv60,
                "ddf_model_cv_beta": ddf_model.beta,
                "ddf_model_tau3_60min": ddf_model.tau3,
                "ddf_model_tau3_slope": ddf_model.tau3_slope,
                "ddf_model_growth": (
                    "smoothed L-CV and L-skewness" if vary_cv else "constant"
                ),
                "ddf_model_rmse_tau3": ddf_model.rmse_tau3,
                "ddf_model_rmse_log_mean": ddf_model.rmse_log_l1,
                "ddf_model_rmse_log_cv": ddf_model.rmse_log_l2,
                "ddf_model_notes": "; ".join(ddf_model.notes),
                "n_bootstrap": n_bootstrap,
                "random_seed": random_seed,
            }
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
        ddf_model=ddf_model,
        ddf_rows=ddf_rows,
        model_comparison=comparison,
        return_periods=return_periods,
    )
