"""
Tests for core/report.py.

Run: python3 tests/test_report.py
"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.classification import (
    LABEL_LIKELY_ARTIFICIAL,
    LABEL_LIKELY_NATURAL,
)
from mayim_tools.hydrology.dem_hydro_processing.core.depression_hierarchy import (
    build_depression_hierarchy,
)
from mayim_tools.hydrology.dem_hydro_processing.core.hierarchical_classification import (
    classify_all_nodes,
    decide_resolution,
    decide_resolution_from_labels,
)
from mayim_tools.hydrology.dem_hydro_processing.core.impact_reduction import (
    apply_ira_to_all_leaves,
)
from mayim_tools.hydrology.dem_hydro_processing.core.report import (
    FIELDNAMES,
    read_classification_report,
    write_classification_report,
)


def _two_pit_dem():
    return np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0],
            [10.0, 8.0, 2.0, 8.0, 4.0, 8.0, 10.0],
            [10.0, 8.0, 8.0, 8.0, 8.0, 8.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    )


def _read_report(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_report_columns_and_basic_content():
    dem = _two_pit_dem()
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.2, n_iterations=20, seed=1
    )

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        n = write_classification_report(out, dem, h, classifications)
        assert n == len(classifications)
        rows = _read_report(out)
        assert len(rows) == len(classifications)
        assert set(rows[0].keys()) == set(FIELDNAMES)
    print("test_report_columns_and_basic_content: PASS")


def test_report_sorted_by_volume_descending():
    dem = _two_pit_dem()
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.2, n_iterations=20, seed=1
    )

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        write_classification_report(out, dem, h, classifications)
        rows = _read_report(out)
        volumes = [float(r["volume"]) for r in rows]
        assert volumes == sorted(
            volumes, reverse=True
        ), f"not sorted descending: {volumes}"
    print("test_report_sorted_by_volume_descending: PASS")


def test_report_cell_area_scales_area_and_volume():
    dem = _two_pit_dem()
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.2, n_iterations=20, seed=1
    )

    with tempfile.TemporaryDirectory() as d:
        out1 = Path(d) / "unit.csv"
        out2 = Path(d) / "scaled.csv"
        write_classification_report(out1, dem, h, classifications, cell_area=1.0)
        write_classification_report(out2, dem, h, classifications, cell_area=4.0)
        rows1 = {r["feature_id"]: r for r in _read_report(out1)}
        rows2 = {r["feature_id"]: r for r in _read_report(out2)}
        for fid in rows1:
            assert (
                abs(float(rows2[fid]["area"]) - 4.0 * float(rows1[fid]["area"])) < 1e-6
            )
            assert (
                abs(float(rows2[fid]["volume"]) - 4.0 * float(rows1[fid]["volume"]))
                < 1e-6
            )
    print("test_report_cell_area_scales_area_and_volume: PASS")


def test_report_feature_type_leaf_vs_meta():
    dem = _two_pit_dem()
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.2, n_iterations=20, seed=1
    )
    leaf_a = int(h.leaf_label[2, 2])
    meta_id = h.nodes[leaf_a].parent

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        write_classification_report(out, dem, h, classifications)
        rows = {int(r["feature_id"]): r for r in _read_report(out)}
        assert rows[leaf_a]["feature_type"] == "Leaf"
        assert rows[meta_id]["feature_type"] == "Meta-depression"
    print("test_report_feature_type_leaf_vs_meta: PASS")


def test_report_protected_by_column():
    dem = _two_pit_dem()
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.2, n_iterations=20, seed=1
    )
    resolution = decide_resolution(h, classifications)
    leaf_a = int(h.leaf_label[2, 2])

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        write_classification_report(
            out, dem, h, classifications, protected_by=resolution.protected_by
        )
        rows = {int(r["feature_id"]): r for r in _read_report(out)}
        if leaf_a in resolution.protected_by:
            assert int(rows[leaf_a]["protected_by"]) == resolution.protected_by[leaf_a]
        else:
            assert rows[leaf_a]["protected_by"] == ""
    print("test_report_protected_by_column: PASS")


def test_report_resolution_columns_populated_from_ira_results():
    dem = np.full((20, 20), 10.0)
    dem[10, 10] = 10.0 - 0.005  # a tiny, obviously artificial pit
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.1, n_iterations=50, seed=2
    )
    resolution = decide_resolution(h, classifications)
    assert resolution.resolve_leaf_ids, "test setup expects at least one resolved leaf"
    corrected, ira_results = apply_ira_to_all_leaves(
        dem, h, leaf_ids=resolution.resolve_leaf_ids
    )

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        write_classification_report(
            out,
            dem,
            h,
            classifications,
            protected_by=resolution.protected_by,
            ira_results=ira_results,
        )
        rows = {int(r["feature_id"]): r for r in _read_report(out)}
        resolved_leaf = resolution.resolve_leaf_ids[0]
        assert rows[resolved_leaf]["resolution"] in ("fill", "breach")
        assert float(rows[resolved_leaf]["resolution_impact"]) > 0
    print("test_report_resolution_columns_populated_from_ira_results: PASS")


def test_report_without_ira_results_leaves_resolution_blank():
    dem = _two_pit_dem()
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.2, n_iterations=20, seed=1
    )

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        write_classification_report(out, dem, h, classifications)
        rows = _read_report(out)
        assert all(r["resolution"] == "" and r["resolution_impact"] == "" for r in rows)
    print("test_report_without_ira_results_leaves_resolution_blank: PASS")


def test_report_creates_parent_directories():
    dem = _two_pit_dem()
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.2, n_iterations=20, seed=1
    )

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "nested" / "subdir" / "report.csv"
        write_classification_report(out, dem, h, classifications)
        assert out.exists()
    print("test_report_creates_parent_directories: PASS")


def test_report_centroid_within_footprint_bounds():
    dem = _two_pit_dem()
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.2, n_iterations=20, seed=1
    )
    leaf_a = int(h.leaf_label[2, 2])

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        write_classification_report(out, dem, h, classifications)
        rows = {int(r["feature_id"]): r for r in _read_report(out)}
        # the single-cell leaf's centroid must be exactly its own cell
        assert abs(float(rows[leaf_a]["centroid_row"]) - 2.0) < 1e-6
        assert abs(float(rows[leaf_a]["centroid_col"]) - 2.0) < 1e-6
    print("test_report_centroid_within_footprint_bounds: PASS")


def test_read_classification_report_round_trip():
    dem = _two_pit_dem()
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.2, n_iterations=20, seed=1
    )

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        write_classification_report(out, dem, h, classifications)
        labels = read_classification_report(out)
        assert labels == {nid: c.label for nid, c in classifications.items()}
    print("test_read_classification_report_round_trip: PASS")


def test_cross_session_workflow_matches_same_session_result():
    """THE critical end-to-end test for the real QGIS workflow: Stage 1
    writes a CSV; Stage 2, in a genuinely SEPARATE call with no shared
    Python objects (the hierarchy is rebuilt fresh, exactly as a real
    second Processing algorithm invocation would have to), reads that
    CSV back and must reach the IDENTICAL resolution decision as the
    same-session path that already has the ClassificationResult
    objects in memory."""
    dem = _two_pit_dem()
    h1 = build_depression_hierarchy(dem)  # "Stage 1's" own hierarchy
    classifications = classify_all_nodes(
        dem, h1, vertical_error_std=0.3, n_iterations=30, seed=5
    )
    same_session = decide_resolution(h1, classifications)

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        write_classification_report(out, dem, h1, classifications)

        # "Stage 2": a fresh hierarchy from the same DEM, no access to `classifications` at all
        h2 = build_depression_hierarchy(dem)
        labels = read_classification_report(out)
        cross_session = decide_resolution_from_labels(h2, labels)

    assert sorted(cross_session.resolve_leaf_ids) == sorted(
        same_session.resolve_leaf_ids
    )
    assert cross_session.protected_by == same_session.protected_by
    print("test_cross_session_workflow_matches_same_session_result: PASS")


def test_human_edited_label_changes_the_resolution_decision():
    """The actual point of the CSV round-trip: a reviewer's manual
    correction in the classification column must be honoured exactly
    as if it had been the tool's own original verdict.

    Uses an ISOLATED leaf with no natural ancestor, deliberately - an
    earlier version of this test edited a leaf that turned out to
    already be protected by its OWN natural meta-depression parent,
    where the edit correctly had no visible effect (ancestor
    protection is checked before a leaf's own label is ever looked at
    - by design, see hierarchical_classification.py's own module
    docstring; a leaf-level edit cannot "pierce through" a genuinely
    natural ancestor's protection). That is real, working, intentional
    behaviour, not something this test should paper over - but it
    meant the WRONG scenario to demonstrate a straightforward edit
    actually taking effect. An isolated leaf has no such ancestor, so
    the edit's effect is direct and unambiguous."""
    dem = np.full((20, 20), 10.0)
    dem[10, 10] = (
        10.0 - 0.005
    )  # tiny, isolated, obviously artificial pit - no natural ancestor
    h = build_depression_hierarchy(dem)
    classifications = classify_all_nodes(
        dem, h, vertical_error_std=0.1, n_iterations=50, seed=9
    )
    leaf_id = int(h.leaf_label[10, 10])
    assert (
        classifications[leaf_id].label == LABEL_LIKELY_ARTIFICIAL
    ), "test setup expects this to start Artificial"

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "report.csv"
        write_classification_report(out, dem, h, classifications)
        labels = read_classification_report(out)

        original_result = decide_resolution_from_labels(h, dict(labels))
        assert leaf_id in original_result.resolve_leaf_ids

        # a reviewer overrides this specific leaf to Likely Natural by hand
        edited_labels = dict(labels)
        edited_labels[leaf_id] = LABEL_LIKELY_NATURAL
        edited_result = decide_resolution_from_labels(h, edited_labels)
        assert (
            leaf_id not in edited_result.resolve_leaf_ids
        ), "the hand edit must be honoured"
        assert (
            edited_result.protected_by.get(leaf_id) == leaf_id
        ), "an edited-to-Natural leaf protects itself"
    print("test_human_edited_label_changes_the_resolution_decision: PASS")


if __name__ == "__main__":
    tests = [
        test_report_columns_and_basic_content,
        test_report_sorted_by_volume_descending,
        test_report_cell_area_scales_area_and_volume,
        test_report_feature_type_leaf_vs_meta,
        test_report_protected_by_column,
        test_report_resolution_columns_populated_from_ira_results,
        test_report_without_ira_results_leaves_resolution_blank,
        test_report_creates_parent_directories,
        test_report_centroid_within_footprint_bounds,
        test_read_classification_report_round_trip,
        test_cross_session_workflow_matches_same_session_result,
        test_human_edited_label_changes_the_resolution_decision,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{t.__name__}: FAIL - {e}")
    print(f"\n{passed}/{len(tests)} passed")
