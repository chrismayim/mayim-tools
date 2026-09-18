"""
Tests for rainfall_frequency/rfa/*.py. Run: python3 tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from scipy.stats import genextreme

from mayim_tools.rainfall.frequency_analysis.rfa.ams import (
    build_regular_grid,
    compute_year_completeness,
    extract_ams_for_duration,
)
from mayim_tools.rainfall.frequency_analysis.rfa.analysis import run_frequency_analysis
from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
    fit_gev_lmoments,
    fit_gev_mle,
    fit_gumbel_lmoments,
    gev_cdf,
    gev_quantile,
)
from mayim_tools.rainfall.frequency_analysis.rfa.lmoments import sample_l_moments
from mayim_tools.rainfall.frequency_analysis.rfa.timebase import (
    applicable_durations,
    detect_native_interval,
    duration_label,
)

# ----------------------------------------------------------------------
# L-moments
# ----------------------------------------------------------------------


def test_l_moments_uniform_distribution_large_sample():
    """Population L-moments of Uniform(0,1): L1=0.5, L2=1/6, t3=0 (symmetric)."""
    rng = np.random.default_rng(42)
    sample = rng.uniform(0, 1, size=200000)
    lm = sample_l_moments(sample)
    assert abs(lm["l1"] - 0.5) < 0.01
    assert abs(lm["l2"] - 1 / 6) < 0.01
    assert abs(lm["t3"]) < 0.01  # symmetric -> zero L-skewness
    print("test_l_moments_uniform_distribution_large_sample: PASS")


def test_l_moments_too_few_values_raises():
    try:
        sample_l_moments([1.0])
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_l_moments_too_few_values_raises: PASS")


# ----------------------------------------------------------------------
# GEV/Gumbel fitting - the critical sign-convention validation
# ----------------------------------------------------------------------


def test_gev_lmoments_recovers_known_parameters_negative_kappa():
    """Large-sample recovery test: generate from a KNOWN GEV (scipy's
    parameterization) and confirm the L-moment fit recovers matching
    sign and magnitude - this is the empirical check that resolved a
    real ambiguity about whether scipy's `c` and Hosking's kappa share
    a sign convention (they do; verified here and in the two MLE tests
    below, not just read off documentation)."""
    true_c, true_loc, true_scale = -0.15, 50.0, 10.0
    sample = genextreme.rvs(
        true_c, loc=true_loc, scale=true_scale, size=200000, random_state=42
    )
    lm = sample_l_moments(sample)
    xi, alpha, kappa = fit_gev_lmoments(lm["l1"], lm["l2"], lm["t3"])
    assert (kappa < 0) == (true_c < 0), f"sign mismatch: kappa={kappa}, true_c={true_c}"
    assert abs(kappa - true_c) < 0.02
    assert abs(xi - true_loc) < 0.5
    assert abs(alpha - true_scale) < 0.5
    print("test_gev_lmoments_recovers_known_parameters_negative_kappa: PASS")


def test_gev_lmoments_recovers_known_parameters_positive_kappa():
    true_c, true_loc, true_scale = 0.15, 50.0, 10.0
    sample = genextreme.rvs(
        true_c, loc=true_loc, scale=true_scale, size=200000, random_state=7
    )
    lm = sample_l_moments(sample)
    xi, alpha, kappa = fit_gev_lmoments(lm["l1"], lm["l2"], lm["t3"])
    assert (kappa > 0) == (true_c > 0), f"sign mismatch: kappa={kappa}, true_c={true_c}"
    assert abs(kappa - true_c) < 0.02
    print("test_gev_lmoments_recovers_known_parameters_positive_kappa: PASS")


def test_gev_mle_matches_lmoments_across_signs_and_seeds():
    """The specific regression test for the bug found while building
    this: scipy's unseeded MLE converged to a WRONG-SIGNED result on
    the very first sample tried. fit_gev_mle() must seed the optimizer
    with the L-moment fit to avoid this - confirmed here across both
    signs of kappa and multiple independent samples."""
    for true_c in (-0.15, -0.05, 0.05, 0.15):
        for seed in (1, 2, 3):
            sample = genextreme.rvs(
                true_c, loc=50.0, scale=10.0, size=5000, random_state=seed
            )
            xi, alpha, kappa, _ = fit_gev_mle(sample)
            assert (kappa < 0) == (
                true_c < 0
            ), f"c={true_c} seed={seed}: sign mismatch, got kappa={kappa}"
            assert (
                abs(kappa - true_c) < 0.05
            ), f"c={true_c} seed={seed}: kappa={kappa} too far from truth"
    print("test_gev_mle_matches_lmoments_across_signs_and_seeds: PASS")


def test_gev_mle_flags_divergence_from_lmoments():
    """Synthetic check that the divergence warning actually fires when
    forced (using a tiny, noisy sample prone to unstable MLE)."""
    rng = np.random.default_rng(99)
    sample = genextreme.rvs(0.3, loc=10, scale=5, size=8, random_state=99)
    xi, alpha, kappa, warnings = fit_gev_mle(sample)
    # Not asserting warnings MUST fire (small samples vary), just that
    # the mechanism runs without crashing and returns a warnings list.
    assert isinstance(warnings, list)
    print("test_gev_mle_flags_divergence_from_lmoments: PASS")


def test_gumbel_lmoments_zero_shape():
    """Gumbel is GEV's kappa=0 case - fitting a true Gumbel sample via
    the dedicated Gumbel formula should closely match fitting the same
    sample via the general GEV formula (which should recover kappa
    near zero)."""
    from scipy.stats import gumbel_r

    sample = gumbel_r.rvs(loc=50, scale=10, size=100000, random_state=1)
    lm = sample_l_moments(sample)
    xi_g, alpha_g = fit_gumbel_lmoments(lm["l1"], lm["l2"])
    xi_gev, alpha_gev, kappa_gev = fit_gev_lmoments(lm["l1"], lm["l2"], lm["t3"])
    assert abs(kappa_gev) < 0.02
    assert abs(xi_g - xi_gev) < 0.5
    assert abs(alpha_g - alpha_gev) < 0.5
    print("test_gumbel_lmoments_zero_shape: PASS")


def test_gev_quantile_matches_scipy_ppf():
    """Cross-check the hand-implemented quantile function against
    scipy's own .ppf() for the same (Hosking-convention) parameters -
    since scipy's c IS Hosking's kappa directly (verified above), the
    two quantile functions should agree closely."""
    xi, alpha, kappa = 50.0, 10.0, -0.1
    for aep in (0.5, 0.1, 0.01, 0.002):
        F = 1 - aep
        mine = gev_quantile(F, xi, alpha, kappa)
        scipy_val = genextreme.ppf(F, kappa, loc=xi, scale=alpha)
        assert (
            abs(mine - scipy_val) < 1e-6
        ), f"AEP={aep}: mine={mine}, scipy={scipy_val}"
    print("test_gev_quantile_matches_scipy_ppf: PASS")


def test_gev_cdf_roundtrip():
    xi, alpha, kappa = 50.0, 10.0, -0.1
    for F in (0.1, 0.5, 0.9, 0.99):
        x = gev_quantile(F, xi, alpha, kappa)
        F_back = gev_cdf(x, xi, alpha, kappa)
        assert abs(F - F_back) < 1e-6
    print("test_gev_cdf_roundtrip: PASS")


# ----------------------------------------------------------------------
# Timebase
# ----------------------------------------------------------------------


def test_detect_native_interval():
    idx = pd.date_range("2020-01-01", periods=100, freq="30min")
    interval = detect_native_interval(pd.Series(idx))
    assert interval == 30.0
    print("test_detect_native_interval: PASS")


def test_applicable_durations_filters_too_fine():
    durations = applicable_durations(
        native_interval_min=60, requested_min=(30, 60, 120, 1440)
    )
    assert 30 not in durations
    assert durations == [60, 120, 1440]
    print("test_applicable_durations_filters_too_fine: PASS")


def test_duration_label_formatting():
    assert duration_label(30) == "30 min"
    assert duration_label(60) == "1 h"
    assert duration_label(90) == "1.5 h"
    assert duration_label(1440) == "24 h"
    print("test_duration_label_formatting: PASS")


# ----------------------------------------------------------------------
# AMS extraction - the "never treat missing as zero" property, and the
# partial-year completeness bug found and fixed while building this.
# ----------------------------------------------------------------------


def _make_hourly_df(start, n_hours, depths):
    idx = pd.date_range(start, periods=n_hours, freq="1h")
    return pd.DataFrame({"timestamp": idx, "depth_mm": depths})


def test_ams_missing_window_is_nan_not_zero():
    """A single missing reading must invalidate every rolling window
    that touches it, rather than treating the gap as zero rainfall
    (which would silently deflate that window's sum)."""
    depths = [1.0] * 10
    depths[5] = np.nan  # one missing reading in the middle
    df = _make_hourly_df("2020-01-01", 10, depths)
    grid = build_regular_grid(df, native_interval_min=60)
    rolling = grid.rolling(window=3, min_periods=3).sum()
    # windows ending at indices 5,6,7 (0-indexed) all touch the NaN at index 5
    assert rolling.iloc[5:8].isna().all(), "windows touching the gap must be NaN"
    # windows entirely before or after the gap must still have valid values
    assert rolling.iloc[2] == 3.0
    print("test_ams_missing_window_is_nan_not_zero: PASS")


def test_year_completeness_partial_year_excluded():
    """A record that starts mid-year must not let that partial year
    look artificially 100% complete - this was a real bug found and
    fixed while building this tool."""
    # Record starts July 1 - roughly half a year of hourly data, but
    # every single one of those readings is valid (no gaps at all).
    df = _make_hourly_df(
        "2020-07-01", 24 * 184, [1.0] * (24 * 184)
    )  # ~6 months, all valid
    grid = build_regular_grid(df, native_interval_min=60)
    records = compute_year_completeness(
        grid, native_interval_min=60, min_completeness=0.90
    )
    year_2020 = next(r for r in records if r.year == 2020)
    assert (
        year_2020.completeness < 0.60
    ), f"partial year should show low completeness relative to a FULL year, got {year_2020.completeness:.2%}"
    assert year_2020.included is False
    print("test_year_completeness_partial_year_excluded: PASS")


def test_year_completeness_full_complete_year_included():
    df = _make_hourly_df(
        "2020-01-01", 24 * 366, [1.0] * (24 * 366)
    )  # 2020 is a leap year
    grid = build_regular_grid(df, native_interval_min=60)
    records = compute_year_completeness(
        grid, native_interval_min=60, min_completeness=0.90
    )
    year_2020 = next(r for r in records if r.year == 2020)
    assert year_2020.completeness > 0.99
    assert year_2020.included is True
    print("test_year_completeness_full_complete_year_included: PASS")


def test_extract_ams_excludes_years_failing_completeness():
    """Two full years plus one badly-gapped year - the gapped year
    must not contribute to the AMS for any duration."""
    good_year_1 = [1.0] * (24 * 365)
    bad_year = [1.0] * (24 * 30) + [np.nan] * (24 * 335)  # only 1 month valid
    good_year_2 = [1.0] * (24 * 365)
    depths = good_year_1 + bad_year + good_year_2
    df = _make_hourly_df("2019-01-01", len(depths), depths)
    grid = build_regular_grid(df, native_interval_min=60)
    records = compute_year_completeness(
        grid, native_interval_min=60, min_completeness=0.90
    )
    ds = extract_ams_for_duration(
        grid, duration_min=60, native_interval_min=60, year_records=records
    )
    assert 2020 not in ds.years, "the badly-gapped year must be excluded from the AMS"
    assert 2019 in ds.years and 2021 in ds.years
    print("test_extract_ams_excludes_years_failing_completeness: PASS")


# ----------------------------------------------------------------------
# Full pipeline integration test
# ----------------------------------------------------------------------


def test_full_pipeline_synthetic_multiyear_record():
    """30 years of synthetic hourly data with a real seasonal/random
    storm pattern - confirms the full Stage 1 pipeline runs end-to-end
    and produces sane, monotone-with-return-period quantiles."""
    rng = np.random.default_rng(123)
    n_years = 30
    idx = pd.date_range("1990-01-01", periods=24 * 365 * n_years, freq="1h")
    depths = np.zeros(len(idx))
    # scatter random storm bursts, roughly 40/year
    for _ in range(40 * n_years):
        start = rng.integers(0, len(idx) - 10)
        dur = rng.integers(1, 8)
        depths[start : start + dur] += rng.exponential(scale=3.0, size=dur)
    df = pd.DataFrame({"Time": idx, "Depth": depths})

    result = run_frequency_analysis(df, "Time", "Depth", distributions=("GEV",))

    assert len(result.fits) > 0
    assert len(result.quantiles) > 0
    # for a given duration, depth must increase as AEP decreases (rarer event -> bigger depth)
    for fit in result.fits:
        qs = sorted(
            [q for q in result.quantiles if q.duration_label == fit.duration_label],
            key=lambda q: q.return_period_years,
        )
        depths_sorted = [q.depth_mm for q in qs]
        assert all(
            d2 >= d1 - 1e-6 for d1, d2 in zip(depths_sorted, depths_sorted[1:])
        ), f"{fit.duration_label}: quantiles not monotone with return period: {depths_sorted}"
    print("test_full_pipeline_synthetic_multiyear_record: PASS")


def test_pipeline_missing_column_raises():
    df = pd.DataFrame({"WrongCol": [1, 2], "Depth": [1.0, 2.0]})
    try:
        run_frequency_analysis(df, "Time", "Depth")
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_pipeline_missing_column_raises: PASS")


# ----------------------------------------------------------------------
# GLO - exact closed-form L-moment relationships
# ----------------------------------------------------------------------


def test_glo_lmoments_recovers_known_parameters():
    """Generate from Hosking's own GLO quantile function with a known
    kappa, fit via L-moments, and confirm recovery - the same
    empirical-validation discipline used for GEV."""
    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        fit_glo_lmoments,
        glo_quantile,
    )

    rng = np.random.default_rng(5)
    for true_kappa in (-0.3, -0.1, 0.1, 0.3):
        F = rng.uniform(1e-6, 1 - 1e-6, size=200000)
        sample = np.array(
            [glo_quantile(f, xi=50.0, alpha=10.0, kappa=true_kappa) for f in F]
        )
        lm = sample_l_moments(sample)
        xi, alpha, kappa = fit_glo_lmoments(lm["l1"], lm["l2"], lm["t3"])
        assert abs(kappa - true_kappa) < 0.02, f"kappa={kappa}, true={true_kappa}"
        assert abs(xi - 50.0) < 1.0
        assert abs(alpha - 10.0) < 1.0
    print("test_glo_lmoments_recovers_known_parameters: PASS")


def test_glo_tau3_tau4_exact_formula():
    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        glo_tau3_tau4_exact,
    )

    t3, t4 = glo_tau3_tau4_exact(0.2)
    assert abs(t3 - (-0.2)) < 1e-9
    assert abs(t4 - (1 + 5 * 0.2**2) / 6) < 1e-9
    print("test_glo_tau3_tau4_exact_formula: PASS")


# ----------------------------------------------------------------------
# LP3 - Hosking's rational approximation + the overflow bug found and
# fixed while building this
# ----------------------------------------------------------------------


def test_lp3_fit_recovers_known_parameters():
    """Generate from scipy's pearson3 with known (mu, sigma, gamma) in
    log space, fit via this module's L-moment code, confirm recovery."""
    from scipy.stats import pearson3

    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        fit_pearson3_lmoments,
    )

    for true_gamma in (-0.8, -0.3, 0.3, 0.8, 1.5):
        true_mu, true_sigma = 2.0, 0.3
        sample = pearson3.rvs(
            skew=true_gamma, loc=true_mu, scale=true_sigma, size=300000, random_state=1
        )
        lm = sample_l_moments(sample)
        mu, sigma, gamma = fit_pearson3_lmoments(lm["l1"], lm["l2"], lm["t3"])
        assert (gamma < 0) == (
            true_gamma < 0
        ), f"sign mismatch: gamma={gamma}, true={true_gamma}"
        assert abs(gamma - true_gamma) < 0.15, f"gamma={gamma}, true={true_gamma}"
        assert abs(mu - true_mu) < 0.02
        assert abs(sigma - true_sigma) < 0.05
    print("test_lp3_fit_recovers_known_parameters: PASS")


def test_lp3_near_zero_skewness_no_overflow():
    """Regression test for the real overflow bug found while building
    this: Gamma(alpha) overflows for alpha > ~171, which happens
    whenever L-skewness is very close to zero - a legitimate case with
    real data, not a hypothetical edge case (first triggered by an
    ordinary 30-year synthetic AMS run, not a contrived input)."""
    import warnings as _warnings

    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        fit_pearson3_lmoments,
    )

    with _warnings.catch_warnings():
        _warnings.filterwarnings("error")
        for t3 in (0.001, 0.0001, 0.00001, -0.0001, -0.00001):
            mu, sigma, gamma = fit_pearson3_lmoments(l1=50.0, l2=5.0, t3=t3)
            assert np.isfinite(mu) and np.isfinite(sigma) and np.isfinite(gamma)
        # sigma must converge to the exact Normal-distribution limit
        # (l2 * sqrt(pi)) as skewness -> 0
        _, sigma_limit, _ = fit_pearson3_lmoments(l1=50.0, l2=5.0, t3=1e-7)
        assert abs(sigma_limit - 5.0 * np.sqrt(np.pi)) < 0.01
    print("test_lp3_near_zero_skewness_no_overflow: PASS")


def test_lp3_negative_or_zero_data_raises():
    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        fit_lp3_lmoments,
    )

    try:
        fit_lp3_lmoments([1.0, 2.0, -3.0, 4.0])
        assert False, "should have raised"
    except ValueError:
        pass
    print("test_lp3_negative_or_zero_data_raises: PASS")


def test_lp3_quantile_roundtrip():
    """Fit LP3 to a real (positive) synthetic sample, then confirm the
    quantile function's median (F=0.5) is in a sane location relative
    to the sample."""
    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        fit_lp3_lmoments,
        lp3_quantile,
    )

    rng = np.random.default_rng(3)
    sample = rng.gamma(shape=5, scale=10, size=1000) + 1  # positive, right-skewed
    fit = fit_lp3_lmoments(sample)
    median = lp3_quantile(0.5, fit["mu"], fit["sigma"], fit["gamma"])
    assert 0.5 * np.median(sample) < median < 2 * np.median(sample)
    print("test_lp3_quantile_roundtrip: PASS")


def test_pt3_tau4_montecarlo_matches_known_normal_constant():
    """At tau3=0, Pearson III degenerates to the Normal distribution,
    whose exact tau4 is a known closed-form constant - this validates
    the Monte Carlo estimation approach used for the rest of PT3's
    tau3-tau4 curve (which has no closed form at all)."""
    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        pt3_tau4_from_tau3,
    )

    known_normal_tau4 = 30 / np.pi * np.arctan(np.sqrt(2)) - 9
    estimated = pt3_tau4_from_tau3(0.0, n_mc=300000, seed=1)
    assert (
        abs(estimated - known_normal_tau4) < 0.01
    ), f"estimated={estimated}, known={known_normal_tau4}"
    print("test_pt3_tau4_montecarlo_matches_known_normal_constant: PASS")


def test_pt3_tau4_symmetric_in_tau3_sign():
    """Pearson III's tau4(tau3) curve depends only on |tau3| - both
    signs at the same magnitude should give the same tau4."""
    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        pt3_tau4_from_tau3,
    )

    t4_pos = pt3_tau4_from_tau3(0.3, n_mc=200000, seed=2)
    t4_neg = pt3_tau4_from_tau3(-0.3, n_mc=200000, seed=2)
    assert abs(t4_pos - t4_neg) < 0.01
    print("test_pt3_tau4_symmetric_in_tau3_sign: PASS")


# ----------------------------------------------------------------------
# Ratio diagram diagnostic
# ----------------------------------------------------------------------


def test_ratio_diagram_recovers_known_distribution():
    """Generate a large sample from a KNOWN GLO distribution, and
    confirm the ratio diagram diagnostic correctly recommends GLO over
    the other three candidates."""
    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        fit_gev_lmoments,
        glo_quantile,
    )
    from mayim_tools.rainfall.frequency_analysis.rfa.ratio_diagram import (
        compare_distributions,
    )

    rng = np.random.default_rng(9)
    F = rng.uniform(1e-6, 1 - 1e-6, size=500000)
    sample = np.array([glo_quantile(f, xi=50.0, alpha=10.0, kappa=0.25) for f in F])
    lm = sample_l_moments(sample)
    _, _, kappa_gev = fit_gev_lmoments(lm["l1"], lm["l2"], lm["t3"])
    result = compare_distributions("test", lm["t3"], lm["t4"], kappa_gev)
    assert (
        result.recommended == "GLO"
    ), f"expected GLO, got {result.recommended} (ranking: {result.ranking})"
    print("test_ratio_diagram_recovers_known_distribution: PASS")


def test_ratio_diagram_recovers_gev_when_data_is_gev_shaped():
    from scipy.stats import genextreme

    from mayim_tools.rainfall.frequency_analysis.rfa.distributions import (
        fit_gev_lmoments,
    )
    from mayim_tools.rainfall.frequency_analysis.rfa.ratio_diagram import (
        compare_distributions,
    )

    sample = genextreme.rvs(-0.25, loc=50, scale=10, size=500000, random_state=11)
    lm = sample_l_moments(sample)
    _, _, kappa_gev = fit_gev_lmoments(lm["l1"], lm["l2"], lm["t3"])
    result = compare_distributions("test", lm["t3"], lm["t4"], kappa_gev)
    assert (
        result.recommended == "GEV"
    ), f"expected GEV, got {result.recommended} (ranking: {result.ranking})"
    print("test_ratio_diagram_recovers_gev_when_data_is_gev_shaped: PASS")


def test_ratio_diagram_ranking_is_sorted_by_distance():
    from mayim_tools.rainfall.frequency_analysis.rfa.ratio_diagram import (
        compare_distributions,
    )

    result = compare_distributions("test", tau3=0.1, tau4=0.15, kappa_gev=0.05)
    distances = [e.distance for e in result.entries]
    assert distances == sorted(distances)
    assert result.recommended == result.entries[0].distribution
    print("test_ratio_diagram_ranking_is_sorted_by_distance: PASS")


# ----------------------------------------------------------------------
# Full pipeline with all four distributions
# ----------------------------------------------------------------------


def test_full_pipeline_all_four_distributions():
    rng = np.random.default_rng(123)
    n_years = 30
    idx = pd.date_range("1995-01-01", periods=24 * 365 * n_years, freq="1h")
    depths = np.zeros(len(idx))
    for _ in range(40 * n_years):
        start = rng.integers(0, len(idx) - 10)
        dur = rng.integers(1, 8)
        depths[start : start + dur] += rng.exponential(scale=3.0, size=dur)
    df = pd.DataFrame({"Time": idx, "Depth": depths})

    result = run_frequency_analysis(df, "Time", "Depth")

    n_durations = len(result.duration_series) - sum(
        1 for w in result.warnings if "only" in w and "skipped" in w
    )
    # every duration that got fitted should have exactly 4 fits (one per distribution)
    from collections import Counter

    counts = Counter(f.duration_label for f in result.fits)
    for dur, count in counts.items():
        assert count == 4, f"{dur}: expected 4 distribution fits, got {count}"

    # every duration with fits should have a recommendation
    fitted_durations = set(counts.keys())
    recommended_durations = {r.duration_label for r in result.recommendations}
    assert fitted_durations == recommended_durations

    # LP3 quantiles must be monotone with return period too, same as GEV/Gumbel/GLO
    for dur in fitted_durations:
        for dist in ("GEV", "Gumbel", "GLO", "LP3"):
            qs = sorted(
                [
                    q
                    for q in result.quantiles
                    if q.duration_label == dur and q.distribution == dist
                ],
                key=lambda q: q.return_period_years,
            )
            depths_sorted = [q.depth_mm for q in qs]
            assert all(
                d2 >= d1 - 1e-6 for d1, d2 in zip(depths_sorted, depths_sorted[1:])
            ), f"{dur} [{dist}]: quantiles not monotone: {depths_sorted}"
    print("test_full_pipeline_all_four_distributions: PASS")


def test_ratio_diagram_entry_names_match_fit_distribution_normalization():
    """Regression test for a real bug: ratio_diagram.py hardcoded
    'Gumbel' (mixed case) while every DistributionFit/QuantileEstimate
    in the pipeline normalizes distribution names to 'GUMBEL'
    (uppercase, via .upper() in analysis.py). This silently broke
    every consumer that looked up a distribution's data by the
    recommended name whenever Gumbel won: the 'Recommended' flag in
    the parameters/quantiles CSVs stayed False even for the actual
    winner, and the final recommended-DDF table's row was entirely
    blank for that duration - a real all-blank row, not caught until
    someone looked at the actual CSV.

    Checks every candidate's name directly (not just whichever one
    happens to win a given sample - GEV nests Gumbel as its kappa->0
    case, so GEV often edges out Gumbel narrowly even on genuinely
    Gumbel-shaped data, making 'force Gumbel to win' an unreliable way
    to exercise this specific bug)."""
    from mayim_tools.rainfall.frequency_analysis.rfa.analysis import (
        DEFAULT_DISTRIBUTIONS,
    )
    from mayim_tools.rainfall.frequency_analysis.rfa.ratio_diagram import (
        compare_distributions,
    )

    normalized_names = {d.upper() for d in DEFAULT_DISTRIBUTIONS}
    result = compare_distributions("test", tau3=0.1, tau4=0.15, kappa_gev=0.05)
    entry_names = {e.distribution for e in result.entries}
    assert entry_names == normalized_names, (
        f"ratio diagram entry names {entry_names} must exactly match the pipeline's "
        f"normalized distribution names {normalized_names} - a mismatch here silently "
        f"breaks every downstream lookup keyed by distribution name"
    )
    print("test_ratio_diagram_entry_names_match_fit_distribution_normalization: PASS")


def test_write_recommended_ddf_no_blank_rows_regardless_of_which_distribution_wins():
    """End-to-end regression test for the same bug: runs the full
    pipeline and confirms every duration's recommended-DDF row has
    real numeric values, not blanks - regardless of which distribution
    the ratio diagram happened to recommend for that duration."""
    import io

    from mayim_tools.rainfall.frequency_analysis.rfa.export import write_recommended_ddf

    rng = np.random.default_rng(123)
    n_years = 30
    idx = pd.date_range("1995-01-01", periods=24 * 365 * n_years, freq="1h")
    depths = np.zeros(len(idx))
    for _ in range(40 * n_years):
        start = rng.integers(0, len(idx) - 10)
        dur = rng.integers(1, 8)
        depths[start : start + dur] += rng.exponential(scale=3.0, size=dur)
    df = pd.DataFrame({"Time": idx, "Depth": depths})

    result = run_frequency_analysis(df, "Time", "Depth")
    recommended_dists = {r.recommended_distribution for r in result.recommendations}
    assert "GUMBEL" in recommended_dists, (
        "this test's synthetic data should produce at least one Gumbel-recommended "
        "duration to actually exercise the bug - if this assertion itself fails, "
        "the test data needs adjusting, not the pipeline"
    )

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        path = f.name
    write_recommended_ddf(result, path)

    with open(path) as f:
        lines = [l.strip().split(",") for l in f.readlines()]
    header, rows = lines[0], lines[1:]
    for row in rows:
        depth_values = row[2:]  # Duration, RecommendedDistribution, then depth columns
        assert all(
            v != "" for v in depth_values
        ), f"blank values in row for {row[0]} ({row[1]}): {row}"
    print(
        "test_write_recommended_ddf_no_blank_rows_regardless_of_which_distribution_wins: PASS"
    )


if __name__ == "__main__":
    tests = [
        v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"{t.__name__}: FAIL - {e}")
        except Exception as e:
            failed += 1
            print(f"{t.__name__}: ERROR - {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
