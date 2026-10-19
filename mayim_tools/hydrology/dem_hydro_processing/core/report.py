"""
CSV report generation.

The first concrete output format this project produces - directly
answering the original request for something usable as a design-
report annexure (project scoping discussion, 2026-09-23), and a
natural point to have reached now that hierarchical_classification.py
classifies every node (leaf and meta-depression) with a full,
auditable trail of evidence.

WHY CSV, NOT A VECTOR LAYER, FOR THIS FIRST PASS: producing real
polygon geometry from a boolean footprint mask is a substantially
larger and more particular piece of work (multi-part features, holes,
simplification) than anything built so far, and QGIS itself already
has robust, well-tested polygonization tools available once this runs
inside the QGIS wrapper (still to be built) - reimplementing that
independently here would duplicate mature, GIS-standard functionality
for comparatively little benefit, unlike the line rasterization in
stream_evidence.py, which had no equivalent already available to a
pure-Python core. Deferred deliberately, not overlooked - see
dev/README.md's own "Still to build" list. In the meantime, each row
below includes its feature's own centroid (row, col), so a feature is
still locatable directly from the CSV alone.

UNITS: `cell_area` (the real-world area of one DEM cell, e.g. in m2)
is an explicit, required-to-think-about parameter, not a silent
assumption - passing 1.0 (the default) reports area/volume in raw
cell-count units, which is honest but not what a design-report
annexure needs; a caller working with real DEM data should pass the
DEM's own actual cell size squared.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .classification import ClassificationResult
from .depression_hierarchy import DepressionHierarchy
from .impact_reduction import IRAResult

FIELDNAMES = [
    "feature_id",
    "feature_type",
    "parent_id",
    "classification",
    "confidence",
    "persistence_probability",
    "stream_relation",
    "num_cells",
    "area",
    "volume",
    "max_depth",
    "mean_depth",
    "centroid_row",
    "centroid_col",
    "protected_by",
    "resolution",
    "resolution_impact",
]


def _feature_type(hierarchy: DepressionHierarchy, node_id: int) -> str:
    node = hierarchy.nodes[node_id]
    return "Leaf" if not node.children else "Meta-depression"


def _centroid(
    hierarchy: DepressionHierarchy, node_id: int, dem: np.ndarray
) -> tuple[float, float]:
    mask = hierarchy.footprint_of(node_id, dem)
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return float("nan"), float("nan")
    return float(rows.mean()), float(cols.mean())


def write_classification_report(
    path: str | Path,
    dem: np.ndarray,
    hierarchy: DepressionHierarchy,
    classifications: dict[int, ClassificationResult],
    protected_by: dict[int, int] | None = None,
    ira_results: list[IRAResult] | None = None,
    cell_area: float = 1.0,
) -> int:
    """Writes one row per classified node (leaf or meta-depression) to
    a CSV at `path`, sorted by volume descending (the most significant
    features first, a standard reporting convention) - see module
    docstring for the full schema and unit handling.

    `protected_by` (from hierarchical_classification.decide_resolution)
    and `ira_results` (from impact_reduction.apply_ira_to_all_leaves,
    via pipeline.apply_corrections) are both optional: a report can be
    written straight after Stage 1 (classification only, nothing
    resolved yet) with neither, or after Stage 2 with both, showing
    the full audit trail of what was actually changed and why.

    Returns the number of rows written.
    """
    protected_by = protected_by or {}
    ira_by_leaf = {r.leaf_id: r for r in (ira_results or [])}

    rows = []
    for node_id, c in classifications.items():
        centroid_r, centroid_c = _centroid(hierarchy, node_id, dem)
        ira = ira_by_leaf.get(node_id)
        if ira is not None:
            impact = (
                ira.breach_impact.total_abs_change
                if ira.method == "breach"
                else ira.fill_impact.total_abs_change
            )
            resolution_impact = round(impact * cell_area, 4)
        else:
            resolution_impact = ""
        rows.append(
            {
                "feature_id": node_id,
                "feature_type": _feature_type(hierarchy, node_id),
                "parent_id": hierarchy.nodes[node_id].parent,
                "classification": c.label,
                "confidence": round(c.confidence, 4),
                "persistence_probability": round(c.persistence_probability, 4),
                "stream_relation": (
                    c.stream_relation if c.stream_relation is not None else ""
                ),
                "num_cells": c.metrics.num_cells,
                "area": round(c.metrics.num_cells * cell_area, 4),
                "volume": round(c.metrics.volume * cell_area, 4),
                "max_depth": round(c.metrics.max_depth, 4),
                "mean_depth": round(c.metrics.mean_depth, 4),
                "centroid_row": round(centroid_r, 2),
                "centroid_col": round(centroid_c, 2),
                "protected_by": protected_by.get(node_id, ""),
                "resolution": ira.method if ira is not None else "",
                "resolution_impact": resolution_impact,
            }
        )

    rows.sort(key=lambda r: r["volume"], reverse=True)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # BOM for Excel
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def read_classification_report(path: str | Path) -> dict[int, str]:
    """Reads a CSV written by write_classification_report() back into
    {feature_id: classification} pairs - the read half of the round-
    trip that makes the CSV report double as the human-in-the-loop
    review mechanism: a reviewer opens the CSV, edits the
    `classification` column for any feature they disagree with (e.g.
    correcting a feature they know is genuine from "Likely Artificial"
    to "Likely Natural"), and this function is how Stage 2 picks those
    corrections up - see
    hierarchical_classification.decide_resolution_from_labels for how
    the result is used.

    Reads only `feature_id` and `classification` - every other column
    is informational/diagnostic and not needed to decide resolution
    (see decide_resolution_from_labels's own docstring for why the
    walk needs nothing more than each node's own label).
    """
    labels: dict[int, str] = {}
    # utf-8-sig reads files with and without the byte-order mark (older runs
    # wrote none; a table saved from Excel may have one)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row["feature_id"])] = row["classification"]
    return labels
