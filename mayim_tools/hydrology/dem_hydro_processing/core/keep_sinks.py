"""Matching user-confirmed natural sinks (pans, regional sinks, lakes, dams
to keep) to the depressions found in the DEM.

By default the tool resolves every depression so the corrected DEM drains
continuously. The user can supply polygons and/or points marking sinks that
are genuine and must stay. Matching works on the DEM's own depression
footprints (cells below a depression's spill elevation), never on flow
accumulation, so a feature does not have to lie on a stream.

Polygon: protects every depression whose footprint overlaps it, and marks
the lowest cell inside it as a permanent sink (so a pan the DEM shows as
flat is still not tilted towards an outlet).
Point: snapped to the nearest depression footprint within a search radius
(cells). If none is within reach the lowest cell inside the radius becomes
the sink. Nothing is ignored silently: every feature gets a KeepMatch
record that is written to the log and the report.

Zero QGIS dependency: geometries arrive as plain coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .depression_hierarchy import DepressionHierarchy
from .stream_evidence import RasterTransform

STATUS_MATCHED = "matched"
STATUS_SINK_ONLY = "no depression found - lowest cell kept as a sink"
STATUS_OUTSIDE = "outside the DEM"
STATUS_NO_DATA = "no valid DEM cells under the feature"


@dataclass
class KeepFeature:
    """One user-supplied sink. kind is "polygon" or "point".
    polygon: rings = [exterior, hole, ...], each a list of (x, y) in the DEM's CRS.
    point: xy = (x, y) in the DEM's CRS."""

    label: str
    kind: str
    rings: list[list[tuple[float, float]]] | None = None
    xy: tuple[float, float] | None = None


@dataclass
class KeepMatch:
    label: str
    kind: str
    status: str
    leaf_ids: list[int] = field(default_factory=list)
    n_footprint_cells: int = 0
    sink_cell: tuple[int, int] | None = None
    detail: str = ""


def _valid(dem: np.ndarray, nodata: float | None) -> np.ndarray:
    ok = np.isfinite(dem)
    if nodata is not None:
        ok &= dem != nodata
    return ok


def rasterize_polygon(
    rings: list[list[tuple[float, float]]],
    transform: RasterTransform,
    shape: tuple[int, int],
) -> np.ndarray:
    """Boolean mask of the cells whose centre lies inside the polygon
    (even-odd rule over all rings, so holes work). A polygon smaller than a
    cell, which would otherwise cover no cell centre, falls back to the cell
    containing its mean vertex."""
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)
    pts = [p for ring in rings for p in ring]
    if len(pts) < 3:
        return mask
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    r_a, c_a = transform.world_to_pixel(float(xs.min()), float(ys.max()))
    r_b, c_b = transform.world_to_pixel(float(xs.max()), float(ys.min()))
    r0 = max(0, int(np.floor(min(r_a, r_b))) - 1)
    r1 = min(rows, int(np.ceil(max(r_a, r_b))) + 2)
    c0 = max(0, int(np.floor(min(c_a, c_b))) - 1)
    c1 = min(cols, int(np.ceil(max(c_a, c_b))) + 2)
    if r0 < r1 and c0 < c1:
        rr, cc = np.mgrid[r0:r1, c0:c1]
        px = transform.origin_x + (cc + 0.5) * transform.pixel_width
        py = transform.origin_y - (rr + 0.5) * transform.pixel_height
        inside = np.zeros(px.shape, dtype=bool)
        for ring in rings:
            n = len(ring)
            for i in range(n):
                x1, y1 = ring[i]
                x2, y2 = ring[(i + 1) % n]
                if y1 == y2:
                    continue
                crosses = (y1 > py) != (y2 > py)
                x_int = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                inside ^= crosses & (px < x_int)
        mask[r0:r1, c0:c1] = inside
    if not mask.any():
        r, c = transform.world_to_pixel(float(xs.mean()), float(ys.mean()))
        r, c = int(np.floor(r)), int(np.floor(c))
        if 0 <= r < rows and 0 <= c < cols:
            mask[r, c] = True
    return mask


def _spill_lookup(hierarchy: DepressionHierarchy) -> np.ndarray:
    """spill_by_label[leaf_id] = the leaf's spill elevation (-inf where the
    label is not a depression leaf, so nothing is ever below it)."""
    max_id = max(hierarchy.nodes) if hierarchy.nodes else 0
    max_label = int(hierarchy.leaf_label.max()) if hierarchy.leaf_label.size else 0
    table = np.full(max(max_id, max_label) + 1, -np.inf)
    for node_id, node in hierarchy.nodes.items():
        if node.is_ocean or node.children or node.spill_elevation is None:
            continue
        table[node_id] = node.spill_elevation
    return table


def _footprint_hits(
    hierarchy: DepressionHierarchy,
    dem: np.ndarray,
    region: np.ndarray,
    spill_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For the cells in `region` (boolean mask) that lie inside some leaf's
    footprint, returns (hit_mask_over_region_cells_as_full_mask, leaf labels
    of the hit cells in row-major order). A cell is a hit when its elevation
    is below its own leaf's spill elevation."""
    labels = hierarchy.leaf_label
    safe = np.clip(labels, 0, len(spill_table) - 1)
    spill = np.where(labels >= 0, spill_table[safe], -np.inf)
    hit = region & (dem < spill - 1e-12)
    return hit, labels[hit]


def match_keep_features(
    hierarchy: DepressionHierarchy,
    dem: np.ndarray,
    transform: RasterTransform,
    features: list[KeepFeature],
    nodata: float | None = None,
    search_cells: int = 3,
) -> tuple[list[KeepMatch], set[int], list[tuple[int, int]]]:
    """Returns (one KeepMatch per feature, protected leaf ids, extra sink
    cells). The extra sink cells are the lowest cell of each feature and are
    handed to the drainage-enforcement pass as permitted sinks."""
    dem64 = np.asarray(dem, dtype=np.float64)
    shape = dem64.shape
    valid = _valid(dem64, nodata)
    spill_table = _spill_lookup(hierarchy)
    protected: set[int] = set()
    sinks: list[tuple[int, int]] = []
    matches: list[KeepMatch] = []

    def lowest_cell(region: np.ndarray) -> tuple[int, int] | None:
        cand = region & valid
        if not cand.any():
            return None
        masked = np.where(cand, dem64, np.inf)
        idx = int(np.argmin(masked))
        return divmod(idx, shape[1])

    for feat in features:
        if feat.kind == "polygon" and feat.rings:
            region = rasterize_polygon(feat.rings, transform, shape)
            if not region.any():
                matches.append(KeepMatch(feat.label, feat.kind, STATUS_OUTSIDE))
                continue
            hit, hit_labels = _footprint_hits(hierarchy, dem64, region, spill_table)
            leaf_ids = sorted({int(v) for v in np.unique(hit_labels)})
            sink = lowest_cell(region)
            if sink is None:
                matches.append(KeepMatch(feat.label, feat.kind, STATUS_NO_DATA))
                continue
            sinks.append(sink)
            protected.update(leaf_ids)
            matches.append(
                KeepMatch(
                    feat.label,
                    feat.kind,
                    STATUS_MATCHED if leaf_ids else STATUS_SINK_ONLY,
                    leaf_ids,
                    int(hit.sum()),
                    sink,
                    f"{int(region.sum())} cell(s) under the polygon",
                )
            )
        elif feat.kind == "point" and feat.xy is not None:
            r_f, c_f = transform.world_to_pixel(*feat.xy)
            r, c = int(np.floor(r_f)), int(np.floor(c_f))
            if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                matches.append(KeepMatch(feat.label, feat.kind, STATUS_OUTSIDE))
                continue
            rad = max(0, int(search_cells))
            rr, cc = np.ogrid[: shape[0], : shape[1]]
            disk = ((rr - r) ** 2 + (cc - c) ** 2) <= rad * rad + 1e-9
            window = np.zeros(shape, dtype=bool)
            window[max(0, r - rad) : r + rad + 1, max(0, c - rad) : c + rad + 1] = True
            region = disk & window
            hit, hit_labels = _footprint_hits(hierarchy, dem64, region, spill_table)
            if hit.any():
                hr, hc = np.nonzero(hit)
                d2 = (hr - r) ** 2 + (hc - c) ** 2
                k = int(np.argmin(d2))
                leaf = int(hierarchy.leaf_label[hr[k], hc[k]])
                dist = float(np.sqrt(d2[k]))
                sink = lowest_cell(hit & (hierarchy.leaf_label == leaf))
                protected.add(leaf)
                if sink is not None:
                    sinks.append(sink)
                matches.append(
                    KeepMatch(
                        feat.label,
                        feat.kind,
                        STATUS_MATCHED,
                        [leaf],
                        int((hit & (hierarchy.leaf_label == leaf)).sum()),
                        sink,
                        f"snapped {dist:.1f} cell(s)",
                    )
                )
            else:
                sink = lowest_cell(region)
                if sink is None:
                    matches.append(KeepMatch(feat.label, feat.kind, STATUS_NO_DATA))
                    continue
                sinks.append(sink)
                matches.append(
                    KeepMatch(
                        feat.label,
                        feat.kind,
                        STATUS_SINK_ONLY,
                        [],
                        0,
                        sink,
                        f"no depression within {rad} cell(s)",
                    )
                )
        else:
            matches.append(KeepMatch(feat.label, feat.kind, STATUS_NO_DATA))
    return matches, protected, sinks
