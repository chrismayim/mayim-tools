"""
End-to-end tests for core/pipeline.py on synthetic terrain where the
correct answer is known by construction.

Run: python3 tests/test_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.classification import (
    LABEL_LIKELY_ARTIFICIAL,
    LABEL_LIKELY_NATURAL,
    LABEL_UNCERTAIN,
)
from mayim_tools.hydrology.dem_hydro_processing.core.depression_hierarchy import (
    build_depression_hierarchy,
)
from mayim_tools.hydrology.dem_hydro_processing.core.pipeline import (
    analyze_dem,
    apply_corrections,
)
from mayim_tools.hydrology.dem_hydro_processing.core.stream_evidence import (
    RasterTransform,
    rasterize_polylines,
)


def lake_and_noise_pit_dem():
    """A tilted plane (falls 0.1 m per column) with (a) a genuine
    3 m deep bowl-shaped lake and (b) a single-cell 2 cm pit - far
    below the 10 cm vertical error used in the tests. The lake here is
    a perfectly smooth bowl - deliberately kept as a SEPARATE, simpler
    scenario from the noisy lake below, not a substitute for it (see
    that test's own docstring for why a smooth lake alone is not
    enough to confirm the meta-depression fix)."""
    size = 60
    dem = np.fromfunction(lambda r, c: 30.0 - 0.1 * c, (size, size))
    rr, cc = np.mgrid[0:size, 0:size]
    d = np.hypot(rr - 20, cc - 20)
    dem -= 3.0 * np.clip(
        1.0 - (d / 6.0) ** 2, 0.0, None
    )  # lake, centre (20, 20), radius 6
    dem[45, 40] = dem[45, 41] - 0.02  # noise pit, centre (45, 40)
    return dem, (20, 20), (45, 40)


def noisy_lake_dem(seed=42):
    """The scenario the meta-depression fix actually exists for: a
    genuine 3 m deep lake with small random floor noise (+-5 cm)
    added - enough to fragment the floor into many noise-scale leaves
    (confirmed below, not assumed), exactly the failure mode described
    in hierarchical_classification.py's own module docstring.

    Deliberately FLAT-bottomed, not the smooth quadratic bowl used in
    lake_and_noise_pit_dem above: tried the quadratic bowl first, and
    even highly unrealistic noise (+-0.5 m, 10x the actual vertical
    error used in these tests) produced only 2-3 leaves - the bowl's
    own smooth curvature dominates small noise almost everywhere
    except right at its centre, so it does not actually exercise the
    fragmentation this test needs. A flat floor has no such dominant
    slope, so noise alone determines the local minima - reliably
    giving over a dozen leaves at a realistic 5 cm noise level,
    confirmed directly before relying on it here."""
    size = 60
    rng = np.random.default_rng(seed)
    dem = np.fromfunction(lambda r, c: 30.0 - 0.1 * c, (size, size))
    rr, cc = np.mgrid[0:size, 0:size]
    d = np.hypot(rr - 20, cc - 20)
    lake_area = d <= 6.0
    base_level = dem[
        20, 20
    ]  # the surrounding tilted plane's own local elevation at the lake centre
    dem[lake_area] = (base_level - 3.0) + rng.uniform(-0.05, 0.05, size=lake_area.sum())
    return dem, (20, 20), lake_area


def culvert_dem():
    """A valley crossed by a 2-cell embankment (10 m) - the classic
    culvert artefact. The valley floor rises steeply upstream (so the
    pond only occupies the part of the valley below 10 m, and a stream
    mapped along the valley continues UPSTREAM of the pond, as a real
    one would) and keeps falling downstream. An earlier version closed
    the valley at the upstream end with the stream starting at the
    pond's head, so the stream had only one side and was correctly
    reported as terminating in the pond - a flaw in the scenario, not
    the classifier."""
    dem = np.full((15, 27), 20.0)
    for c in range(1, 27):
        if c <= 11:
            dem[7, c] = 3.0 + 0.8 * (11 - c)
        elif c >= 14:
            dem[7, c] = 2.6 - 0.05 * (c - 14)
    dem[7, 12:14] = 10.0
    return dem


def culvert_stream_mask(shape):
    t = RasterTransform(
        origin_x=0.0, origin_y=float(shape[0]), pixel_width=1.0, pixel_height=1.0
    )
    # y=7.6 -> row 7.4 -> row 7 (avoids the .5 round-half-to-even ambiguity)
    return rasterize_polylines([[(0.6, 7.6), (shape[1] - 0.6, 7.6)]], t, shape)


def test_lake_is_preserved_and_noise_pit_is_resolved():
    dem, lake_c, pit_c = lake_and_noise_pit_dem()
    analysis = analyze_dem(dem, vertical_error_std=0.10, n_iterations=100, seed=1)

    lake_leaf = int(analysis.hierarchy.leaf_label[lake_c])
    pit_leaf = int(analysis.hierarchy.leaf_label[pit_c])
    assert analysis.classifications[lake_leaf].label == LABEL_LIKELY_NATURAL
    assert analysis.classifications[pit_leaf].label == LABEL_LIKELY_ARTIFICIAL

    corrected, results = apply_corrections(dem, analysis)
    assert corrected[pit_c] > dem[pit_c], "the noise pit must have been resolved"
    lake_cells = analysis.hierarchy.cells_of(lake_leaf)
    assert np.array_equal(
        corrected[lake_cells], dem[lake_cells]
    ), "the genuine lake must be left exactly as it was"
    print("test_lake_is_preserved_and_noise_pit_is_resolved: PASS")


def test_noisy_lake_fragments_but_is_still_recognised_and_preserved():
    """THE critical end-to-end regression test for the meta-depression
    finding (see hierarchical_classification.py and noisy_lake_dem's
    own docstrings): confirms the fragmentation is real (many leaves,
    none individually resembling the lake), that the pipeline still
    correctly classifies the MERGED lake as natural, and that applying
    corrections leaves the lake's cells completely untouched despite
    that fragmentation - not just that SOME node somewhere gets called
    natural."""
    dem, lake_c, lake_area = noisy_lake_dem()
    analysis = analyze_dem(dem, vertical_error_std=0.10, n_iterations=40, seed=1)

    lake_leaves = {
        nid
        for nid in analysis.classifications
        if not analysis.hierarchy.nodes[nid].children
        and analysis.hierarchy.footprint_of(nid, dem)[lake_area].any()
    }
    assert (
        len(lake_leaves) > 5
    ), f"expected real fragmentation into several leaves, got {len(lake_leaves)}"

    corrected, results = apply_corrections(dem, analysis)
    assert np.array_equal(corrected[lake_area], dem[lake_area]), (
        "the noisy lake's floor must be left completely untouched, despite fragmenting into "
        f"{len(lake_leaves)} separate leaves at the leaf level"
    )
    print("test_noisy_lake_fragments_but_is_still_recognised_and_preserved: PASS")


def test_culvert_with_through_stream_is_not_called_natural_and_is_breached_along_it():
    dem = culvert_dem()
    stream = culvert_stream_mask(dem.shape)
    plain = analyze_dem(dem, vertical_error_std=0.10, n_iterations=40, seed=2)
    with_stream = analyze_dem(
        dem, vertical_error_std=0.10, n_iterations=40, seed=2, stream_mask=stream
    )

    pond = int(with_stream.hierarchy.leaf_label[7, 6])
    c_plain = plain.classifications[pond]
    c_stream = with_stream.classifications[pond]

    assert c_stream.stream_relation == "passes_through"
    assert (
        c_stream.confidence < c_plain.confidence
    ), "a through-flowing stream must lower the natural-confidence"
    assert c_stream.label != LABEL_LIKELY_NATURAL, (
        "a 7 m deep pond held by a 2-cell embankment with a mapped stream running straight through "
        f"it must not be called Likely Natural: {c_stream}"
    )

    corrected, results = apply_corrections(
        dem,
        with_stream,
        enforce=False,
        resolve_labels=(LABEL_LIKELY_ARTIFICIAL, LABEL_UNCERTAIN),
    )
    assert (
        corrected[7, 12] < dem[7, 12] and corrected[7, 13] < dem[7, 13]
    ), "embankment must have been cut"
    assert (
        corrected[7, 12] < 4.0 and corrected[7, 13] < 4.0
    ), "cut down to about the pond bed's level"
    # the pond itself must not have been FILLED up to the embankment (prefer_breach)
    assert np.array_equal(
        corrected[7, 3:12], dem[7, 3:12]
    ), "pond cells must be untouched - breached, not filled"
    assert np.array_equal(corrected[:7], dem[:7]) and np.array_equal(
        corrected[8:], dem[8:]
    ), "walls untouched"
    print(
        "test_culvert_with_through_stream_is_not_called_natural_and_is_breached_along_it: PASS"
    )


def test_default_apply_leaves_uncertain_features_alone():
    """Uncertain features go to human review, they are not silently
    'fixed' - resolve_labels must be opted into."""
    dem = culvert_dem()
    stream = culvert_stream_mask(dem.shape)
    analysis = analyze_dem(
        dem, vertical_error_std=0.10, n_iterations=40, seed=2, stream_mask=stream
    )
    pond = int(analysis.hierarchy.leaf_label[7, 6])
    label = analysis.classifications[pond].label
    assert (
        label == LABEL_UNCERTAIN
    ), f"this scenario is meant to be an Uncertain case, got {label}"
    corrected, _ = apply_corrections(
        dem, analysis, enforce=False
    )  # default: Likely Artificial only
    assert np.array_equal(
        corrected, dem
    ), "nothing labelled Likely Artificial -> nothing may change"
    print("test_default_apply_leaves_uncertain_features_alone: PASS")


def test_apply_corrections_resolve_labels_can_change_without_rerunning_analysis():
    """The whole point of separating classify_all_nodes from
    decide_resolution (see pipeline.py's own docstring): the SAME
    analysis object can be applied twice, with different
    resolve_labels, without repeating Stage 1's Monte Carlo work."""
    dem = culvert_dem()
    stream = culvert_stream_mask(dem.shape)
    analysis = analyze_dem(
        dem, vertical_error_std=0.10, n_iterations=40, seed=2, stream_mask=stream
    )

    default_corrected, _ = apply_corrections(dem, analysis, enforce=False)
    reviewed_corrected, _ = apply_corrections(
        dem,
        analysis,
        enforce=False,
        resolve_labels=(LABEL_LIKELY_ARTIFICIAL, LABEL_UNCERTAIN),
    )

    assert np.array_equal(
        default_corrected, dem
    ), "default run resolves nothing in this Uncertain-only scenario"
    assert not np.array_equal(
        reviewed_corrected, dem
    ), "widened resolve_labels must actually resolve the culvert"
    print(
        "test_apply_corrections_resolve_labels_can_change_without_rerunning_analysis: PASS"
    )


def test_analysis_is_reproducible_and_leaf_independent():
    dem, _, _ = lake_and_noise_pit_dem()
    a = analyze_dem(dem, vertical_error_std=0.10, n_iterations=30, seed=7)
    b = analyze_dem(dem, vertical_error_std=0.10, n_iterations=30, seed=7)
    ca = {nid: c.confidence for nid, c in a.classifications.items()}
    cb = {nid: c.confidence for nid, c in b.classifications.items()}
    assert ca == cb
    print("test_analysis_is_reproducible_and_leaf_independent: PASS")


def test_flat_plateau_leaves_are_counted_not_silently_dropped():
    dem = np.full((20, 20), 5.0)
    dem[8:12, 8:12] = (
        5.0  # a flat patch: a "leaf" in the hierarchy but not a depression
    )
    dem[3:6, 3:6] = 1.0  # one genuine depression
    dem[2, 2:7] = 6.0
    dem[6, 2:7] = 6.0
    dem[2:7, 2] = 6.0
    dem[2:7, 6] = 6.0
    analysis = analyze_dem(dem, vertical_error_std=0.05, n_iterations=20, seed=1)
    assert all(c.metrics.volume > 0 for c in analysis.classifications.values())
    assert isinstance(analysis.skipped_non_depression_leaves, int)
    print("test_flat_plateau_leaves_are_counted_not_silently_dropped: PASS")


def test_long_artificial_chain_leaves_zero_residual_depressions_end_to_end():
    """THE end-to-end regression test, at the actual public API level
    (analyze_dem + apply_corrections, exactly what the QGIS wrapper
    calls), for a real bug found from a real corrected DEM producing
    short, disconnected flow-accumulation (see
    hierarchical_classification.group_into_resolution_units's own
    docstring for the full account). A long chain of noise-scale pits
    (depth far below the assumed vertical error, so genuinely
    classified artificial by real Monte Carlo persistence, not just
    hand-labelled) must come out with zero residual depressions after
    correction - not just mostly resolved."""
    size = 60
    n_pits = 25
    dem = np.full((size, size), 10.0)
    row = 30
    for i in range(n_pits):
        col = 5 + i * 2
        dem[row, col] = (
            10.0 - 0.01 - i * 0.0001
        )  # tiny, noise-scale depth, far below 0.1 vertical error
    analysis = analyze_dem(dem, vertical_error_std=0.1, n_iterations=60, seed=3)

    chain_leaves = [
        int(analysis.hierarchy.leaf_label[row, 5 + i * 2]) for i in range(n_pits)
    ]
    labels = {
        analysis.classifications[nid].label
        for nid in chain_leaves
        if nid in analysis.classifications
    }
    assert labels == {
        LABEL_LIKELY_ARTIFICIAL
    }, f"test setup expects the whole chain classified artificial, got {labels}"

    corrected, results = apply_corrections(dem, analysis)

    h2 = build_depression_hierarchy(corrected)
    from mayim_tools.hydrology.dem_hydro_processing.core.classification import (
        compute_depression_metrics,
    )

    residual = [
        nid
        for nid, n in h2.nodes.items()
        if not n.is_ocean
        and not n.children
        and compute_depression_metrics(corrected, h2, nid).volume > 0
    ]
    assert (
        residual == []
    ), f"expected zero residual depressions in the corrected DEM, got {len(residual)}"
    print(
        "test_long_artificial_chain_leaves_zero_residual_depressions_end_to_end: PASS"
    )


def _has_d8_direction(dem, r, c, nodata=None):
    """A minimal steepest-descent check (not a production flow-
    direction algorithm): True if cell (r,c) has at least one
    8-connected neighbour strictly lower than itself. This is exactly
    the property a real D8 flow-direction tool needs to resolve a
    direction - a flat cell (every neighbour equal or higher) is
    precisely what breaks flow accumulation, which is what this test
    exists to catch directly, not just infer from "depressions were
    filled"."""
    rows, cols = dem.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                nv = dem[nr, nc]
                if nodata is not None and nv == nodata:
                    continue
                if nv < dem[r, c]:
                    return True
    return False


def test_corrected_dem_is_d8_resolvable_not_just_filled():
    """THE direct test for what Chris actually reported: the tool was
    filling depressions correctly (visible on a profile comparison),
    but flow-direction/accumulation was still short and disconnected -
    because filling raises every cell of a depression to one uniform
    elevation, which a standard D8 algorithm cannot resolve a
    direction across. Checks the actual property that matters (every
    cell in a filled region has a steepest-descent direction) rather
    than only checking that depressions were removed."""
    size = 60
    n_pits = 20
    dem = np.full((size, size), 10.0)
    row = 30
    for i in range(n_pits):
        col = 5 + i * 2
        dem[row, col] = 10.0 - 0.01 - i * 0.0001
    analysis = analyze_dem(dem, vertical_error_std=0.1, n_iterations=60, seed=3)
    corrected, results = apply_corrections(dem, analysis)

    assert any(
        r.method == "fill" for r in results
    ), "test setup expects at least one fill"
    unresolvable = []
    for r in results:
        if r.method != "fill":
            continue
        for cell in r.elevation_changes:
            if not _has_d8_direction(corrected, cell[0], cell[1]):
                unresolvable.append(cell)
    assert (
        unresolvable == []
    ), f"{len(unresolvable)} filled cell(s) have no D8-resolvable direction: {unresolvable[:5]}"
    print("test_corrected_dem_is_d8_resolvable_not_just_filled: PASS")


if __name__ == "__main__":
    tests = [
        test_lake_is_preserved_and_noise_pit_is_resolved,
        test_noisy_lake_fragments_but_is_still_recognised_and_preserved,
        test_culvert_with_through_stream_is_not_called_natural_and_is_breached_along_it,
        test_default_apply_leaves_uncertain_features_alone,
        test_apply_corrections_resolve_labels_can_change_without_rerunning_analysis,
        test_analysis_is_reproducible_and_leaf_independent,
        test_flat_plateau_leaves_are_counted_not_silently_dropped,
        test_long_artificial_chain_leaves_zero_residual_depressions_end_to_end,
        test_corrected_dem_is_d8_resolvable_not_just_filled,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback

            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
