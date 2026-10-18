"""
Tests for core/stochastic_classification.py.

Statistical in nature, so tests use fixed RNG seeds (for
reproducibility) and wide tolerance bands (since exact values are not
meaningful for a Monte Carlo method) rather than precise expected
numbers.

Run: python3 tests/test_stochastic_classification.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.depression_hierarchy import (
    build_depression_hierarchy,
)
from mayim_tools.hydrology.dem_hydro_processing.core.stochastic_classification import (
    _smooth_control_surface,
    compute_persistence_probability,
    compute_persistence_with_baseline,
    generate_correlated_noise,
)


def test_noise_mean_approximately_zero():
    """A single realisation's own mean is not a reliable check here -
    spatial correlation reduces the effective sample size, so one
    realisation's mean has real sampling variability even for an
    unbiased generator (confirmed by an initial version of this test
    with too tight a tolerance failing on a perfectly valid draw).
    Averaging across several independent realisations is the correct
    way to check for systematic bias."""
    rng = np.random.default_rng(1)
    means = [
        generate_correlated_noise(
            (200, 200), std=1.0, correlation_length_cells=3.0, rng=rng
        ).mean()
        for _ in range(20)
    ]
    grand_mean = np.mean(means)
    assert (
        abs(grand_mean) < 0.03
    ), f"mean across realisations should be near zero, got {grand_mean}"
    print("test_noise_mean_approximately_zero: PASS")


def test_noise_std_matches_target():
    rng = np.random.default_rng(2)
    target_std = 0.15
    noise = generate_correlated_noise(
        (300, 300), std=target_std, correlation_length_cells=3.0, rng=rng
    )
    assert (
        abs(noise.std() - target_std) / target_std < 0.1
    ), f"expected std~{target_std}, got {noise.std()}"
    print("test_noise_std_matches_target: PASS")


def test_noise_is_spatially_correlated():
    """Adjacent cells should be more similar (higher correlation
    across many independent realisations) than cells far apart -
    confirms the smoothing actually induces spatial structure, not
    just independent noise at a different scale."""
    rng = np.random.default_rng(3)
    n_realisations = 300
    shape = (20, 20)
    adjacent_pairs = []
    distant_pairs = []
    for _ in range(n_realisations):
        noise = generate_correlated_noise(
            shape, std=1.0, correlation_length_cells=3.0, rng=rng
        )
        adjacent_pairs.append((noise[10, 10], noise[10, 11]))
        distant_pairs.append((noise[10, 10], noise[10, 19]))
    adjacent_pairs = np.array(adjacent_pairs)
    distant_pairs = np.array(distant_pairs)
    adjacent_corr = np.corrcoef(adjacent_pairs[:, 0], adjacent_pairs[:, 1])[0, 1]
    distant_corr = np.corrcoef(distant_pairs[:, 0], distant_pairs[:, 1])[0, 1]
    assert (
        adjacent_corr > distant_corr
    ), f"adjacent correlation ({adjacent_corr}) should exceed distant ({distant_corr})"
    assert (
        adjacent_corr > 0.3
    ), f"adjacent cells should show clear positive correlation, got {adjacent_corr}"
    print("test_noise_is_spatially_correlated: PASS")


def test_noise_reproducible_with_same_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    noise1 = generate_correlated_noise(
        (50, 50), std=1.0, correlation_length_cells=3.0, rng=rng1
    )
    noise2 = generate_correlated_noise(
        (50, 50), std=1.0, correlation_length_cells=3.0, rng=rng2
    )
    assert np.array_equal(noise1, noise2)
    print("test_noise_reproducible_with_same_seed: PASS")


def test_obviously_artificial_pit_gets_low_persistence():
    """A single-cell pit only 0.01m deep, with a DEM vertical error of
    0.1m (10x the pit's own depth), on FLAT ground. The RAW score is
    not low here (~0.35) - on flat terrain noise alone puts a cell
    inside some depression a large fraction of the time (see
    compute_persistence_with_baseline) - an earlier version of this
    test asserted raw < 0.3 and was correct to fail once the
    flat-terrain baseline was measured. The chance-corrected score is
    what should be low."""
    size = 30
    dem = np.full((size, size), 10.0)
    dem[15, 15] = 10.0 - 0.01
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[15, 15])
    r = compute_persistence_with_baseline(
        dem,
        h,
        leaf_id,
        vertical_error_std=0.1,
        n_iterations=200,
        rng=np.random.default_rng(100),
    )
    assert r.adjusted < 0.25, f"expected low chance-corrected persistence, got {r}"
    assert r.raw > r.adjusted, "correction must only ever reduce the score"
    print("test_obviously_artificial_pit_gets_low_persistence: PASS")


def test_baseline_high_on_flat_low_on_steep():
    """The measured phenomenon that motivated the baseline: with no
    real feature at all, noise alone puts a cell inside a depression
    far more often on flat ground than on a steep slope."""

    def baseline_for(slope):
        dem = np.fromfunction(lambda r, c: 20.0 - slope * c, (40, 40))
        dem[20, 20] -= 0.02  # a token feature, so the leaf exists to be assessed
        if slope > 0:
            dem[20, 20] = dem[20, 21] - 0.02
        h = build_depression_hierarchy(dem)
        leaf = int(h.leaf_label[20, 20])
        return compute_persistence_with_baseline(
            dem, h, leaf, 0.10, n_iterations=80, rng=np.random.default_rng(5)
        ).baseline

    flat, steep = baseline_for(0.0), baseline_for(0.10)
    assert flat > 0.15, f"flat-terrain baseline should be substantial, got {flat}"
    assert steep < 0.10, f"steep-terrain baseline should be small, got {steep}"
    assert flat > steep + 0.1
    print("test_baseline_high_on_flat_low_on_steep: PASS")


def test_adjusted_persistence_deep_feature_stays_high():
    dem = np.full((30, 30), 10.0)
    dem[15, 15] = 5.0
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[15, 15])
    r = compute_persistence_with_baseline(
        dem, h, leaf_id, 0.05, n_iterations=60, rng=np.random.default_rng(101)
    )
    assert r.adjusted > 0.9, r
    print("test_adjusted_persistence_deep_feature_stays_high: PASS")


def test_control_surface_reproduces_a_slope():
    """Harmonic interpolation across a hole in a uniformly tilted
    plane must give back the plane - the property that makes it a
    fair 'terrain without the feature', and the reason it replaced
    the filled-surface control (which left a tie with the downhill
    neighbour)."""
    plane = np.fromfunction(lambda r, c: 20.0 - 0.1 * c, (21, 21))
    damaged = plane.copy()
    hole = np.zeros_like(plane, dtype=bool)
    hole[9:12, 9:12] = True
    damaged[hole] = 0.0
    valid = np.ones_like(hole)
    repaired = _smooth_control_surface(damaged, hole, valid)
    assert np.abs(repaired[hole] - plane[hole]).max() < 0.02, np.abs(
        repaired[hole] - plane[hole]
    ).max()
    print("test_control_surface_reproduces_a_slope: PASS")


def test_adjusted_is_a_valid_probability_and_never_exceeds_raw():
    for depth in [0.005, 0.05, 0.5]:
        dem = np.full((25, 25), 10.0)
        dem[12, 12] = 10.0 - depth
        h = build_depression_hierarchy(dem)
        leaf_id = int(h.leaf_label[12, 12])
        r = compute_persistence_with_baseline(
            dem, h, leaf_id, 0.2, n_iterations=40, rng=np.random.default_rng(3)
        )
        assert (
            0.0 <= r.adjusted <= 1.0
            and 0.0 <= r.raw <= 1.0
            and 0.0 <= r.baseline <= 1.0
        )
        assert r.adjusted <= r.raw + 1e-12
    print("test_adjusted_is_a_valid_probability_and_never_exceeds_raw: PASS")


def test_obviously_real_feature_gets_high_persistence():
    """A single-cell pit 5m deep, with a DEM vertical error of only
    0.05m (100x smaller than the pit's own depth) - such a deep
    feature should survive essentially every plausible perturbation."""
    size = 30
    dem = np.full((size, size), 10.0)
    dem[15, 15] = 5.0  # a genuinely deep pit, far exceeding the assumed DEM noise
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[15, 15])
    rng = np.random.default_rng(101)
    p = compute_persistence_probability(
        dem,
        h,
        leaf_id,
        vertical_error_std=0.05,
        n_iterations=200,
        rng=rng,
    )
    assert (
        p > 0.9
    ), f"expected a deep pit under small noise to have high persistence, got p={p}"
    print("test_obviously_real_feature_gets_high_persistence: PASS")


def test_persistence_deterministic_with_same_seed():
    dem = np.full((30, 30), 10.0)
    dem[15, 15] = 8.0
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[15, 15])
    p1 = compute_persistence_probability(
        dem,
        h,
        leaf_id,
        vertical_error_std=0.1,
        n_iterations=50,
        rng=np.random.default_rng(7),
    )
    p2 = compute_persistence_probability(
        dem,
        h,
        leaf_id,
        vertical_error_std=0.1,
        n_iterations=50,
        rng=np.random.default_rng(7),
    )
    assert p1 == p2, f"same seed must give identical results: {p1} vs {p2}"
    print("test_persistence_deterministic_with_same_seed: PASS")


def test_persistence_range_is_valid_probability():
    """Structural check across several depths: the result must always
    be a valid probability in [0, 1]."""
    for depth in [0.001, 0.1, 1.0, 5.0]:
        dem = np.full((25, 25), 10.0)
        dem[12, 12] = 10.0 - depth
        h = build_depression_hierarchy(dem)
        leaf_id = int(h.leaf_label[12, 12])
        p = compute_persistence_probability(
            dem,
            h,
            leaf_id,
            vertical_error_std=0.2,
            n_iterations=40,
            rng=np.random.default_rng(abs(hash(depth)) % 1000),
        )
        assert 0.0 <= p <= 1.0, f"depth={depth}: p={p} out of valid range"
    print("test_persistence_range_is_valid_probability: PASS")


def test_persistence_with_nodata_does_not_crash():
    nd = -9999.0
    size = 25
    dem = np.full((size, size), nd)
    dem[5:20, 5:20] = 10.0
    dem[12, 12] = 8.0
    h = build_depression_hierarchy(dem, nodata=nd)
    leaf_id = int(h.leaf_label[12, 12])
    p = compute_persistence_probability(
        dem,
        h,
        leaf_id,
        vertical_error_std=0.3,
        n_iterations=30,
        nodata=nd,
        rng=np.random.default_rng(5),
    )
    assert 0.0 <= p <= 1.0
    print("test_persistence_with_nodata_does_not_crash: PASS")


def test_window_margin_override_respected():
    """An explicitly small window_margin must not cause a crash, even
    when it clips against the array's own edge."""
    dem = np.full((15, 15), 10.0)
    dem[7, 7] = 9.0
    h = build_depression_hierarchy(dem)
    leaf_id = int(h.leaf_label[7, 7])
    p = compute_persistence_probability(
        dem,
        h,
        leaf_id,
        vertical_error_std=0.1,
        n_iterations=20,
        window_margin=2,
        rng=np.random.default_rng(9),
    )
    assert 0.0 <= p <= 1.0
    print("test_window_margin_override_respected: PASS")


if __name__ == "__main__":
    tests = [
        test_noise_mean_approximately_zero,
        test_noise_std_matches_target,
        test_noise_is_spatially_correlated,
        test_noise_reproducible_with_same_seed,
        test_obviously_artificial_pit_gets_low_persistence,
        test_baseline_high_on_flat_low_on_steep,
        test_adjusted_persistence_deep_feature_stays_high,
        test_control_surface_reproduces_a_slope,
        test_adjusted_is_a_valid_probability_and_never_exceeds_raw,
        test_obviously_real_feature_gets_high_persistence,
        test_persistence_deterministic_with_same_seed,
        test_persistence_range_is_valid_probability,
        test_persistence_with_nodata_does_not_crash,
        test_window_margin_override_respected,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
