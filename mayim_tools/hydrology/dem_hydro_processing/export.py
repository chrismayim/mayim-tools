"""Sectioned CSV for the DEM Hydrological Processing tool (no QGIS
dependency). Numbers are written plainly, without thousands separators."""

from __future__ import annotations

import csv
import math

from .core.conditioning import ConditioningResult

UNIT_FIELDS = [
    ("unit_id", "resolution unit (depression) identifier"),
    ("type", "Leaf (one local minimum) or Meta-depression (merged leaves)"),
    ("method", "fill or breach, whichever disturbs the DEM less"),
    ("footprint_cells", "cells below the depression's spill elevation"),
    ("area_m2", "footprint area"),
    ("max_depth_m", "greatest depth below the spill elevation"),
    ("volume_m3", "volume the depression holds up to its spill elevation"),
    ("cells_changed", "cells whose elevation the fill or breach changed"),
    ("max_change_m", "largest elevation change made by this correction"),
    ("stream_relation", "none, terminates_in or passes_through (needs a stream layer)"),
    ("classification", "Likely Artificial / Uncertain - Review / Likely Natural"),
    ("confidence", "0 = confidently artificial, 1 = confidently natural"),
    ("persistence", "share of noisy DEM realisations in which the depression persists"),
    ("centroid_row", "footprint centroid, raster row"),
    ("centroid_col", "footprint centroid, raster column"),
]
KEPT_FIELDS = [
    ("leaf_id", "depression identifier"),
    ("reason", "why it was left in the DEM"),
    ("footprint_cells", "cells below the spill elevation"),
    ("area_m2", "footprint area"),
    ("max_depth_m", "greatest depth"),
    ("volume_m3", "volume held"),
    ("classification", "classification, when classification was run"),
    ("centroid_row", "footprint centroid, raster row"),
    ("centroid_col", "footprint centroid, raster column"),
]
SINK_FIELDS = [
    ("sink_id", "cluster number"),
    ("kind", "kept (a confirmed sink) or unresolved (should be investigated)"),
    ("reason", "why the cluster is a sink"),
    ("cells", "cells in the cluster with no downhill neighbour"),
    ("row", "raster row of the cluster's lowest cell"),
    ("col", "raster column of the cluster's lowest cell"),
    ("elevation_m", "elevation of the cluster's lowest cell"),
]
MATCH_FIELDS = [
    ("feature", "label of the input sink feature"),
    ("geometry", "polygon or point"),
    ("result", "what the feature matched"),
    ("depressions_kept", "depression identifiers kept because of this feature"),
    ("footprint_cells", "footprint cells of the kept depressions under the feature"),
    ("sink_row", "raster row of the permitted sink cell"),
    ("sink_col", "raster column of the permitted sink cell"),
    ("detail", "extra information (snap distance, cells covered)"),
]

STAT_LABELS = [
    ("mode", "Mode"),
    ("valid_cells", "Valid DEM cells"),
    ("depressions_found", "Depressions found (leaves holding water)"),
    ("resolution_units", "Resolution units (merged where adjacent)"),
    ("units_filled", "Units filled"),
    ("units_breached", "Units breached"),
    ("kept_depressions", "Depressions kept"),
    ("pits_after_fill_breach", "Cells with no downhill neighbour after fill/breach"),
    ("cells_raised_by_enforcement", "Cells raised by the drainage-enforcement pass"),
    ("remaining_sink_clusters", "Sink clusters remaining"),
    ("unresolved_sink_clusters", "Of which not confirmed as kept"),
    ("cells_changed", "Cells changed"),
    ("pct_cells_changed", "Cells changed (% of valid)"),
    ("max_raise_m", "Largest raise (m)"),
    ("max_lowering_m", "Largest lowering (m)"),
    ("mean_abs_change_changed_cells_m", "Mean absolute change of changed cells (m)"),
    ("fill_volume_m3", "Volume added by raising (m3)"),
    ("cut_volume_m3", "Volume removed by lowering (m3)"),
]


def fmt(value) -> str:
    """Plain number text: no thousands separators, no stray precision."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}" if abs(value) < 1e15 else f"{value:.0f}"
    return str(value)


def _section(w, title, fields, rows):
    w.writerow([title])
    w.writerow([name for name, _desc in fields])
    for row in rows:
        w.writerow([fmt(row.get(name)) for name, _desc in fields])
    w.writerow([])


def unit_rows(result: ConditioningResult) -> list[dict]:
    rows = []
    for u in sorted(result.units, key=lambda x: -x.volume_m3):
        rows.append(
            {
                "unit_id": u.unit_id,
                "type": u.node_type,
                "method": u.method,
                "footprint_cells": u.footprint_cells,
                "area_m2": u.area_m2,
                "max_depth_m": u.max_depth_m,
                "volume_m3": u.volume_m3,
                "cells_changed": u.cells_changed,
                "max_change_m": u.max_change_m,
                "stream_relation": u.stream_relation,
                "classification": u.classification,
                "confidence": u.confidence,
                "persistence": u.persistence,
                "centroid_row": u.centroid_row,
                "centroid_col": u.centroid_col,
            }
        )
    return rows


def write_sectioned_csv(path: str, result: ConditioningResult, context: dict) -> None:
    s = result.settings
    st = result.stats
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["DEM HYDROLOGICAL PROCESSING - RESULTS"])
        w.writerow([])
        w.writerow(["1. RUN SUMMARY"])
        w.writerow(["Item", "Value"])
        for label, value in [
            (
                "Tool",
                f"{context.get('tool', 'DEM Hydrological Processing')} "
                f"{context.get('version', '')}",
            ),
            ("Run", context.get("run_time", "")),
            ("Input DEM", context.get("dem_path", "")),
            ("CRS", context.get("crs", "")),
            ("Cell size (m)", context.get("cell_size", "")),
            ("Grid (columns x rows)", context.get("grid", "")),
            ("Stream centreline supplied", context.get("stream_layer", "no")),
            ("Confirmed-sink layer supplied", context.get("keep_layer", "no")),
            ("Classification table supplied", context.get("review_table", "no")),
        ]:
            w.writerow([label, fmt(value)])
        w.writerow([])
        w.writerow(["2. PARAMETERS"])
        w.writerow(["Parameter", "Value"])
        for label, value in [
            (
                "Mode",
                (
                    "Continuous drainage"
                    if s.mode == "continuous"
                    else "Preserve Likely Natural depressions"
                ),
            ),
            ("Classification run", "yes" if st.get("classified") else "no"),
            ("DEM source type", context.get("dem_source", "")),
            ("Vertical error std (m)", s.vertical_error_std),
            ("Monte Carlo iterations", s.n_iterations),
            ("Correlation length (cells)", s.correlation_length_cells),
            (
                "Confidence thresholds (artificial below, natural above)",
                f"{s.confidence_thresholds[0]} / {s.confidence_thresholds[1]}",
            ),
            ("Resolve Uncertain features (preserve mode)", s.resolve_uncertain),
            ("Maximum breach length (cells)", s.max_breach_length),
            ("Maximum breach depth (m)", s.max_breach_depth),
            ("Point search radius for confirmed sinks (cells)", s.keep_search_cells),
            ("Drainage enforcement", s.enforce),
        ]:
            w.writerow([label, fmt(value)])
        w.writerow([])
        w.writerow(["3. RESULTS"])
        w.writerow(["Item", "Value"])
        for key, label in STAT_LABELS:
            w.writerow([label, fmt(st.get(key))])
        for message in result.warnings:
            w.writerow(["Warning", message])
        w.writerow([])
        _section(
            w,
            "4. CONFIRMED SINKS SUPPLIED BY THE USER AND WHAT EACH MATCHED",
            MATCH_FIELDS,
            [
                {
                    "feature": m.label,
                    "geometry": m.kind,
                    "result": m.status,
                    "depressions_kept": " ".join(str(i) for i in m.leaf_ids),
                    "footprint_cells": m.n_footprint_cells,
                    "sink_row": m.sink_cell[0] if m.sink_cell else "",
                    "sink_col": m.sink_cell[1] if m.sink_cell else "",
                    "detail": m.detail,
                }
                for m in result.keep_matches
            ],
        )
        _section(
            w,
            "5. DEPRESSIONS LEFT IN THE DEM",
            KEPT_FIELDS,
            [vars(k) for k in result.kept],
        )
        _section(
            w,
            "6. SINK CLUSTERS REMAINING IN THE CORRECTED DEM",
            SINK_FIELDS,
            [
                {
                    "sink_id": k.sink_id,
                    "kind": k.kind,
                    "reason": k.reason,
                    "cells": k.cells,
                    "row": k.row,
                    "col": k.col,
                    "elevation_m": k.elevation,
                }
                for k in result.sinks
            ],
        )
        _section(
            w,
            "7. DEPRESSIONS RESOLVED (largest volume first)",
            UNIT_FIELDS,
            unit_rows(result),
        )
        w.writerow(["8. FIELD DEFINITIONS"])
        w.writerow(["Section", "Field", "Meaning"])
        for title, fields in [
            ("4", MATCH_FIELDS),
            ("5", KEPT_FIELDS),
            ("6", SINK_FIELDS),
            ("7", UNIT_FIELDS),
        ]:
            for name, desc in fields:
                w.writerow([title, name, desc])
