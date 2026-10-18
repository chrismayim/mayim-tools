"""The single DEM Hydrological Processing pipeline.

Builds the depression hierarchy, decides what to resolve, resolves it
(fill or breach, whichever disturbs the DEM less), enforces drainage and
collects everything the CSV and Word reports need.

Modes
-----
continuous (default): every depression is resolved, so flow routes
    continuously across the DEM. The only exceptions are sinks the user
    confirms as genuine (polygons / points, see keep_sinks.py). The Monte
    Carlo classification is optional here (report information only).
preserve: the earlier behaviour - depressions classified Likely Natural
    are kept; Likely Artificial (and optionally Uncertain) are resolved.
    A classification table from an earlier run can be reused to skip the
    slow classification step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .classification import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    LABEL_LIKELY_ARTIFICIAL,
    LABEL_LIKELY_NATURAL,
    LABEL_UNCERTAIN,
    ClassificationResult,
    compute_depression_metrics,
)
from .depression_hierarchy import DepressionHierarchy, build_depression_hierarchy
from .diagnostics import find_unresolvable_cells
from .drainage_enforcement import enforce_drainage
from .hierarchical_classification import (
    classify_all_nodes,
    decide_resolution,
    decide_resolution_from_labels,
    group_into_resolution_units,
)
from .impact_reduction import IRAResult, apply_ira_to_all_leaves
from .keep_sinks import KeepFeature, KeepMatch, match_keep_features
from .stochastic_classification import (
    DEFAULT_CORRELATION_LENGTH_CELLS,
    DEFAULT_N_ITERATIONS,
)
from .stream_evidence import (
    RasterTransform,
    compute_stream_relation,
    rasterize_polylines,
)
from .windowed import NodeWindows

MODE_CONTINUOUS = "continuous"
MODE_PRESERVE = "preserve"


class ConditioningError(Exception):
    """A problem the user can act on (bad input), not a bug."""


@dataclass
class ConditioningSettings:
    mode: str = MODE_CONTINUOUS
    classify: bool = False  # always done in preserve mode unless a table is supplied
    vertical_error_std: float = 1.5
    n_iterations: int = DEFAULT_N_ITERATIONS
    correlation_length_cells: float = DEFAULT_CORRELATION_LENGTH_CELLS
    confidence_thresholds: tuple[float, float] = DEFAULT_CONFIDENCE_THRESHOLDS
    stream_buffer_cells: int = 1
    seed: int = 0
    resolve_uncertain: bool = True
    max_breach_length: int | None = None
    max_breach_depth: float | None = None
    keep_search_cells: int = 3
    enforce: bool = True


@dataclass
class UnitRecord:
    unit_id: int
    node_type: str  # "Leaf" or "Meta-depression"
    method: str  # "fill" or "breach"
    footprint_cells: int
    area_m2: float
    max_depth_m: float
    volume_m3: float
    cells_changed: int
    max_change_m: float
    stream_relation: str
    classification: str
    confidence: float | None
    persistence: float | None
    centroid_row: float
    centroid_col: float


@dataclass
class KeptRecord:
    leaf_id: int
    reason: str
    footprint_cells: int
    area_m2: float
    max_depth_m: float
    volume_m3: float
    classification: str
    centroid_row: float
    centroid_col: float


@dataclass
class SinkRecord:
    sink_id: int
    kind: str  # "kept" or "unresolved"
    reason: str
    cells: int
    row: int
    col: int
    elevation: float


@dataclass
class ConditioningResult:
    corrected: np.ndarray
    original: np.ndarray
    settings: ConditioningSettings
    units: list[UnitRecord] = field(default_factory=list)
    kept: list[KeptRecord] = field(default_factory=list)
    keep_matches: list[KeepMatch] = field(default_factory=list)
    sinks: list[SinkRecord] = field(default_factory=list)
    classifications: dict[int, ClassificationResult] | None = None
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    hierarchy: DepressionHierarchy | None = None
    ira_results: list[IRAResult] = field(default_factory=list)


def _tick(progress, fraction: float, message: str) -> None:
    if progress is not None:
        progress(fraction, message)


def _leaf_ids_with_volume(hierarchy: DepressionHierarchy, dem: np.ndarray) -> list[int]:
    """Leaves that hold water (some cell below the leaf's spill elevation).
    Flat leaves with no volume are not depressions to fill or breach; the
    drainage-enforcement pass tilts any such flat towards an outlet."""
    labels = hierarchy.leaf_label
    n = int(labels.max()) + 1 if labels.size else 0
    min_elev = np.full(n, np.inf)
    ok = labels >= 0
    np.minimum.at(min_elev, labels[ok], dem[ok])
    leaves = []
    for node_id, node in hierarchy.nodes.items():
        if node.is_ocean or node.children or node.spill_elevation is None:
            continue
        if node_id < n and min_elev[node_id] < node.spill_elevation - 1e-12:
            leaves.append(node_id)
    return leaves


def _pit_cell(
    hierarchy: DepressionHierarchy, dem: np.ndarray, leaf_id: int, windows=None
):
    windows = windows or NodeWindows(hierarchy)
    r0, r1, c0, c1 = windows.window(leaf_id, 0)
    sub = windows.sub_hierarchy((r0, r1, c0, c1))
    rr, cc = np.nonzero(sub.cells_of(leaf_id))
    if not len(rr):
        return None
    k = int(np.argmin(dem[r0:r1, c0:c1][rr, cc]))
    return int(rr[k]) + r0, int(cc[k]) + c0


def _is_local_minimum(dem: np.ndarray, cell: tuple[int, int], nodata) -> bool:
    """True when no valid 8-neighbour is strictly lower (float32 compare)."""
    r, c = cell
    z = np.float32(dem[r, c])
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if not (0 <= rr < dem.shape[0] and 0 <= cc < dem.shape[1]):
                continue
            v = dem[rr, cc]
            if not np.isfinite(v) or (nodata is not None and v == nodata):
                continue
            if np.float32(v) < z:
                return False
    return True


def _metrics(hierarchy, dem, node_id, cell_area, windows=None):
    windows = windows or NodeWindows(hierarchy)
    win = windows.window(node_id, 2)
    r0, r1, c0, c1 = win
    sub = windows.sub_hierarchy(win)
    crop = dem[r0:r1, c0:c1]
    m = compute_depression_metrics(crop, sub, node_id)
    fp = sub.footprint_of(node_id, crop)
    rr, cc = np.nonzero(fp)
    if len(rr):
        crow, ccol = float(rr.mean()) + r0, float(cc.mean()) + c0
    else:
        crow = ccol = float("nan")
    return m, crow, ccol, m.volume * cell_area


def run_conditioning(
    dem: np.ndarray,
    nodata: float | None,
    transform: RasterTransform,
    cell_size: tuple[float, float],
    settings: ConditioningSettings | None = None,
    stream_polylines: list | None = None,
    keep_features: list[KeepFeature] | None = None,
    review_labels: dict[int, str] | None = None,
    progress: Callable[[float, str], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> ConditioningResult:
    s = settings or ConditioningSettings()
    if s.mode not in (MODE_CONTINUOUS, MODE_PRESERVE):
        raise ConditioningError(f"Unknown mode {s.mode!r}.")
    original = np.asarray(dem, dtype=np.float64)
    cell_area = float(cell_size[0]) * float(cell_size[1])
    result = ConditioningResult(
        corrected=original.copy(), original=original, settings=s
    )
    canceled = is_canceled or (lambda: False)

    _tick(progress, 0.02, "Building the depression hierarchy...")
    hierarchy = build_depression_hierarchy(original, nodata=nodata)
    result.hierarchy = hierarchy
    windows = NodeWindows(hierarchy)

    stream_mask = None
    if stream_polylines:
        stream_mask = rasterize_polylines(stream_polylines, transform, original.shape)
        if not stream_mask.any():
            result.warnings.append(
                "The stream centreline covers no DEM cells (check that its CRS "
                "and extent match the DEM); it was ignored."
            )
            stream_mask = None

    # ---- classification (optional in continuous mode) -------------------
    need_labels = s.mode == MODE_PRESERVE
    classifications = None
    if review_labels is not None and need_labels:
        pass  # table supplied: no Monte Carlo needed to decide
    elif s.classify or need_labels:
        _tick(
            progress,
            0.08,
            "Classifying depressions (Monte Carlo persistence; the slow step)...",
        )
        classifications = classify_all_nodes(
            original,
            hierarchy,
            s.vertical_error_std,
            nodata=nodata,
            stream_mask=stream_mask,
            n_iterations=s.n_iterations,
            correlation_length_cells=s.correlation_length_cells,
            confidence_thresholds=s.confidence_thresholds,
            stream_buffer_cells=s.stream_buffer_cells,
            seed=s.seed,
        )
        result.classifications = classifications
    if canceled():
        return result

    # ---- what to resolve -----------------------------------------------
    _tick(progress, 0.6, "Deciding what to resolve...")
    if s.mode == MODE_CONTINUOUS:
        resolve_ids = _leaf_ids_with_volume(hierarchy, original)
        protected_by_class: set[int] = set()
    else:
        labels_wanted = (LABEL_LIKELY_ARTIFICIAL,) + (
            (LABEL_UNCERTAIN,) if s.resolve_uncertain else ()
        )
        if review_labels is not None:
            decision = decide_resolution_from_labels(
                hierarchy, review_labels, resolve_labels=labels_wanted
            )
        else:
            decision = decide_resolution(
                hierarchy, classifications, resolve_labels=labels_wanted
            )
        resolve_ids = list(decision.resolve_leaf_ids)
        resolvable = set(_leaf_ids_with_volume(hierarchy, original))
        protected_by_class = resolvable - set(resolve_ids)

    keep_reasons: dict[int, str] = {}
    sink_reasons: dict[tuple[int, int], str] = {}
    extra_sinks: list[tuple[int, int]] = []
    if keep_features:
        matches, protected, extra_sinks = match_keep_features(
            hierarchy,
            original,
            transform,
            keep_features,
            nodata=nodata,
            search_cells=s.keep_search_cells,
        )
        result.keep_matches = matches
        for m in matches:
            if m.sink_cell is not None:
                sink_reasons[m.sink_cell] = f"kept: {m.label}"
            for lid in m.leaf_ids:
                keep_reasons.setdefault(lid, f"kept: {m.label}")
            if m.status != "matched":
                result.warnings.append(f"Kept sink '{m.label}': {m.status}.")
        resolve_ids = [i for i in resolve_ids if i not in protected]
    for lid in protected_by_class:
        keep_reasons.setdefault(lid, "classified Likely Natural")

    units = group_into_resolution_units(hierarchy, resolve_ids)
    stream_relations = None
    if stream_mask is not None and units:
        stream_relations = {}
        pad = s.stream_buffer_cells + 3
        for u in units:
            win = windows.window(u, pad)
            r0, r1, c0, c1 = win
            stream_relations[u] = compute_stream_relation(
                windows.sub_hierarchy(win),
                u,
                stream_mask[r0:r1, c0:c1],
                buffer_cells=s.stream_buffer_cells,
                dem=original[r0:r1, c0:c1],
            )

    # ---- resolve ---------------------------------------------------------
    _tick(progress, 0.7, "Resolving depressions (fill or breach)...")
    _tick(progress, 0.7, f"{len(units)} resolution units to process.")
    corrected, ira = apply_ira_to_all_leaves(
        original,
        hierarchy,
        nodata=nodata,
        max_breach_length=s.max_breach_length,
        max_breach_depth=s.max_breach_depth,
        leaf_ids=units,
        progress=(
            None
            if progress is None
            else lambda d, t, m: progress(0.7 + 0.15 * d / max(t, 1), m)
        ),
        is_canceled=is_canceled,
        stream_mask=stream_mask,
        stream_relations=stream_relations,
    )
    if canceled():
        return result

    # ---- enforce drainage ----------------------------------------------
    pits_before = int(
        find_unresolvable_cells(corrected.astype(np.float32), nodata).sum()
    )
    raised = 0
    if s.enforce:
        _tick(progress, 0.85, "Enforcing drainage...")
        sinks_allowed = list(extra_sinks)
        for lid in keep_reasons:
            cell = _pit_cell(hierarchy, original, lid, windows)
            if cell is not None:
                sinks_allowed.append(cell)
                sink_reasons.setdefault(cell, keep_reasons[lid])
        # a permitted sink only needs seeding if it is a true local minimum;
        # seeding a cell that already drains would raise its lower neighbours
        sinks_allowed = [
            c for c in sinks_allowed if _is_local_minimum(corrected, c, nodata)
        ]
        corrected, raised = enforce_drainage(
            corrected, nodata=nodata, sink_cells=sinks_allowed
        )
    result.corrected = corrected
    result.ira_results = ira

    # ---- records ---------------------------------------------------------
    _tick(progress, 0.92, "Collecting results...")
    cls = classifications or {}
    for r in ira:
        m, crow, ccol, vol = _metrics(
            hierarchy, original, r.leaf_id, cell_area, windows
        )
        node = hierarchy.nodes[r.leaf_id]
        c = cls.get(r.leaf_id)
        changes = r.elevation_changes
        max_change = (
            max(abs(v - original[k]) for k, v in changes.items()) if changes else 0.0
        )
        result.units.append(
            UnitRecord(
                unit_id=r.leaf_id,
                node_type="Meta-depression" if node.children else "Leaf",
                method=r.method,
                footprint_cells=m.num_cells,
                area_m2=m.num_cells * cell_area,
                max_depth_m=m.max_depth,
                volume_m3=vol,
                cells_changed=len(changes),
                max_change_m=float(max_change),
                stream_relation=(stream_relations or {}).get(r.leaf_id, ""),
                classification=c.label if c else "",
                confidence=c.confidence if c else None,
                persistence=c.persistence_probability if c else None,
                centroid_row=crow,
                centroid_col=ccol,
            )
        )
    for lid, reason in sorted(keep_reasons.items()):
        m, crow, ccol, vol = _metrics(hierarchy, original, lid, cell_area, windows)
        c = cls.get(lid)
        label = c.label if c else (review_labels or {}).get(lid, "")
        result.kept.append(
            KeptRecord(
                lid,
                reason,
                m.num_cells,
                m.num_cells * cell_area,
                m.max_depth,
                vol,
                label,
                crow,
                ccol,
            )
        )

    pit_mask = find_unresolvable_cells(corrected.astype(np.float32), nodata)
    lab, n = ndimage.label(pit_mask, structure=np.ones((3, 3)))
    c32 = corrected.astype(np.float32)
    boxes = ndimage.find_objects(lab)
    for cid in range(1, n + 1):
        sl = boxes[cid - 1]
        local = np.argwhere(lab[sl] == cid)
        cells = local + np.array([sl[0].start, sl[1].start])
        z = c32[cells[:, 0], cells[:, 1]]
        k = int(np.argmin(z))
        reason = ""
        for cell in map(tuple, cells.tolist()):
            if cell in sink_reasons:
                reason = sink_reasons[cell]
                break
        if not reason:
            # a kept depression's pit may sit next to the cluster's lowest cell
            for cell, why in sink_reasons.items():
                if lab[cell] == cid:
                    reason = why
                    break
        result.sinks.append(
            SinkRecord(
                cid,
                "kept" if reason else "unresolved",
                reason or "no downhill neighbour after correction",
                len(cells),
                int(cells[k, 0]),
                int(cells[k, 1]),
                float(z[k]),
            )
        )
    result.sinks.sort(key=lambda x: (x.kind != "unresolved", -x.cells))

    # ---- statistics ------------------------------------------------------
    valid = np.isfinite(original)
    if nodata is not None:
        valid &= original != nodata
    delta = np.where(valid, corrected - original, 0.0)
    changed = np.abs(delta) > 0
    n_valid = int(valid.sum())
    result.stats = {
        "mode": s.mode,
        "valid_cells": n_valid,
        "depressions_found": len(_leaf_ids_with_volume(hierarchy, original)),
        "resolution_units": len(units),
        "units_filled": sum(1 for r in ira if r.method == "fill"),
        "units_breached": sum(1 for r in ira if r.method == "breach"),
        "kept_depressions": len(keep_reasons),
        "pits_after_fill_breach": pits_before,
        "cells_raised_by_enforcement": raised,
        "remaining_sink_clusters": len(result.sinks),
        "unresolved_sink_clusters": sum(
            1 for x in result.sinks if x.kind == "unresolved"
        ),
        "cells_changed": int(changed.sum()),
        "pct_cells_changed": 100.0 * float(changed.sum()) / max(1, n_valid),
        "max_raise_m": float(delta.max()) if delta.size else 0.0,
        "max_lowering_m": float(-delta.min()) if delta.size else 0.0,
        "mean_abs_change_changed_cells_m": (
            float(np.abs(delta[changed]).mean()) if changed.any() else 0.0
        ),
        "fill_volume_m3": float(delta[delta > 0].sum()) * cell_area,
        "cut_volume_m3": float(-delta[delta < 0].sum()) * cell_area,
        "classified": classifications is not None or review_labels is not None,
    }
    if result.stats["unresolved_sink_clusters"]:
        result.warnings.append(
            f"{result.stats['unresolved_sink_clusters']} cell cluster(s) still have "
            "no downhill neighbour and are not kept sinks; see the sink points layer."
        )
    _tick(progress, 1.0, "Done.")
    return result


__all__ = [
    "MODE_CONTINUOUS",
    "MODE_PRESERVE",
    "ConditioningError",
    "ConditioningResult",
    "ConditioningSettings",
    "IRAResult",
    "KeepFeature",
    "LABEL_LIKELY_NATURAL",
    "run_conditioning",
]
