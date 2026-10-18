"""
Terrain Profile - core logic (no QGIS dependency, independently testable;
see tests/test_terrain_profile_core.py).

A section line drawn by the user is sampled along its length on a DEM and,
optionally, on one or more water-surface elevation (WSE) rasters (one per
event, e.g. 1:50 and 1:100). Where a property polygon is given, the points
where the section crosses its boundary are found exactly, and the
relationship between the flood and the property is measured:

- property dry: horizontal distance along the section from the nearest
  property boundary crossing to the flood edge, the vertical difference
  between ground at that boundary crossing and the water surface, and the
  lowest ground inside the property relative to the water surface;
- property inundated: wet length inside the property, maximum depth
  inside the property and the water surface there.

Independently of the sections, the shortest plan distance from the
property boundary to the nearest flooded cell is found for every event.

Flood edges (waterlines). A cell is wet where WSE > ground. WSE rasters
(HEC-RAS results, DEM Inundation Boundary) hold values on wet cells only,
so the edge falls inside the last, partly wet cell. Like RAS Mapper, the
WSE is carried a short distance past the last valid cell (1.5 WSE cells by
default plus one sample step, holding the nearest value) and the waterline
is placed where the water surface meets the ground, interpolated linearly
between samples.

Chainage runs from the first vertex of the drawn line, so the direction
the line is drawn sets the left-to-right order of the figure.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.hydrology._common.d8_network import GridError

ProfileError = GridError

DEFAULT_EDGE_EXTENSION_CELLS = 1.5
DEFAULT_SEARCH_RADIUS_M = 500.0
_EPS = 1e-9

EDGE_WATERLINE = "waterline"
EDGE_SECTION_END = "section end"
EDGE_WSE_LIMIT = "limit of WSE data"

STATUS_NO_PROPERTY = "no property"
STATUS_NOT_CROSSED = "section does not cross the property"
STATUS_NO_FLOOD = "no flooding on the section"
STATUS_DRY = "property dry"
STATUS_INUNDATED = "property inundated"


# ----------------------------------------------------------------------
# Rasters


@dataclass
class Grid:
    """A raster window as float64 (NoData -> NaN) with its geotransform."""

    array: np.ndarray
    geotransform: tuple

    @classmethod
    def from_raw(cls, array, geotransform, nodata=None) -> Grid:
        gt = tuple(float(v) for v in geotransform)
        if abs(gt[2]) > _EPS or abs(gt[4]) > _EPS:
            raise ProfileError(
                "Rotated rasters are not supported - warp the raster to a "
                "north-up grid first."
            )
        a = np.asarray(array, dtype=np.float64).copy()
        if nodata is not None and not (
            isinstance(nodata, float) and math.isnan(nodata)
        ):
            a[np.isclose(a, float(nodata), rtol=0.0, atol=1e-6)] = np.nan
        a[~np.isfinite(a)] = np.nan
        return cls(a, gt)

    @property
    def cell_size(self) -> float:
        return max(abs(self.geotransform[1]), abs(self.geotransform[5]))

    def cell_centres(self, rows, cols):
        gt = self.geotransform
        return gt[0] + (cols + 0.5) * gt[1], gt[3] + (rows + 0.5) * gt[5]

    def sample(self, x, y) -> np.ndarray:
        """Bilinear between cell centres where all four neighbours are
        valid, otherwise the value of the cell containing the point (which
        keeps stepped WSE rasters valid right up to their last wet cell).
        NaN outside the raster or on NoData."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        gt = self.geotransform
        rows, cols = self.array.shape
        u = (x - gt[0]) / gt[1]  # fractional column (cell edges at integers)
        v = (y - gt[3]) / gt[5]
        out = np.full(x.shape, np.nan)

        cn = np.floor(u).astype(np.int64)
        rn = np.floor(v).astype(np.int64)
        inside = (cn >= 0) & (cn < cols) & (rn >= 0) & (rn < rows)
        out[inside] = self.array[rn[inside], cn[inside]]

        uc, vc = u - 0.5, v - 0.5
        c0 = np.floor(uc).astype(np.int64)
        r0 = np.floor(vc).astype(np.int64)
        fx, fy = uc - c0, vc - r0
        ok = (c0 >= 0) & (c0 + 1 < cols) & (r0 >= 0) & (r0 + 1 < rows)
        if ok.any():
            a = self.array
            r, c = r0[ok], c0[ok]
            q00, q01 = a[r, c], a[r, c + 1]
            q10, q11 = a[r + 1, c], a[r + 1, c + 1]
            fxo, fyo = fx[ok], fy[ok]
            bil = (
                q00 * (1 - fxo) * (1 - fyo)
                + q01 * fxo * (1 - fyo)
                + q10 * (1 - fxo) * fyo
                + q11 * fxo * fyo
            )
            good = np.isfinite(bil)
            idx = np.flatnonzero(ok)[good]
            out[idx] = bil[good]
        return out


# ----------------------------------------------------------------------
# Line geometry


def clean_vertices(vertices) -> np.ndarray:
    """Nx2 array without repeated consecutive vertices."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    if len(v) == 0:
        raise ProfileError("The section line has no vertices.")
    keep = np.ones(len(v), bool)
    keep[1:] = np.hypot(*(v[1:] - v[:-1]).T) > _EPS
    v = v[keep]
    if len(v) < 2:
        raise ProfileError("The section line has zero length.")
    return v


def cumulative_length(vertices: np.ndarray) -> np.ndarray:
    seg = np.hypot(*(vertices[1:] - vertices[:-1]).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def point_at(vertices: np.ndarray, chainage):
    """x, y at chainage(s) along the polyline."""
    cum = cumulative_length(vertices)
    ch = np.clip(np.asarray(chainage, dtype=np.float64), 0.0, cum[-1])
    return np.interp(ch, cum, vertices[:, 0]), np.interp(ch, cum, vertices[:, 1])


def stations(vertices: np.ndarray, interval: float, extra=()) -> np.ndarray:
    """Chainages at a regular interval, plus every vertex, the end and any
    extra chainages (e.g. property boundary crossings), sorted and unique."""
    if not interval or interval <= 0:
        raise ProfileError("The sampling interval must be greater than zero.")
    cum = cumulative_length(vertices)
    length = cum[-1]
    n = int(math.floor(length / interval))
    if n > 200000:
        raise ProfileError(
            f"{n} samples needed along a {length:.0f} m section - increase the "
            "sampling interval."
        )
    ch = np.concatenate(
        [np.arange(n + 1) * interval, cum, [length], np.asarray(extra, float)]
    )
    ch = ch[(ch >= 0) & (ch <= length + _EPS)]
    ch = np.unique(np.round(np.clip(ch, 0, length), 6))
    return ch


def nice_step(length: float, target: int = 10) -> float:
    """A round chainage-marker interval giving about ``target`` markers."""
    if length <= 0:
        return 1.0
    raw = length / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


# ----------------------------------------------------------------------
# Polygons (list of polygons; each a list of rings; each ring [(x, y), ...])


def _ring_array(ring) -> np.ndarray:
    a = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
    if len(a) and (a[0] != a[-1]).any():
        a = np.vstack([a, a[:1]])
    return a


def _edges(polygons) -> np.ndarray:
    """All ring edges as (x0, y0, x1, y1)."""
    out = []
    for poly in polygons or ():
        for ring in poly:
            a = _ring_array(ring)
            if len(a) >= 4:
                out.append(np.hstack([a[:-1], a[1:]]))
    return np.vstack(out) if out else np.zeros((0, 4))


def points_in_polygons(x, y, polygons) -> np.ndarray:
    """Even-odd test over all rings (holes and multiple parts work)."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    inside = np.zeros(x.shape, bool)
    for x0, y0, x1, y1 in _edges(polygons):
        if y0 == y1:
            continue
        cond = (y0 > y) != (y1 > y)
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= cond & (x < xi)
    return inside


def polygon_crossings(vertices: np.ndarray, polygons) -> list[float]:
    """Chainages where the section line crosses a polygon ring."""
    edges = _edges(polygons)
    if len(edges) == 0:
        return []
    cum = cumulative_length(vertices)
    q = edges[:, :2]
    s = edges[:, 2:] - edges[:, :2]
    found = []
    for i in range(len(vertices) - 1):
        p = vertices[i]
        r = vertices[i + 1] - p
        denom = r[0] * s[:, 1] - r[1] * s[:, 0]
        ok = np.abs(denom) > _EPS
        qp = q - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / denom
            u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / denom
        hit = ok & (t >= -_EPS) & (t <= 1 + _EPS) & (u >= -_EPS) & (u <= 1 + _EPS)
        seg_len = cum[i + 1] - cum[i]
        found.extend((cum[i] + np.clip(t[hit], 0, 1) * seg_len).tolist())
    found = sorted(found)
    merged = []
    for c in found:
        if not merged or c - merged[-1] > 1e-6:
            merged.append(round(c, 6))
    return merged


def inside_intervals(
    vertices: np.ndarray, polygons, crossings: Sequence[float]
) -> list[tuple[float, float]]:
    """Chainage intervals of the section that lie inside the polygons."""
    length = cumulative_length(vertices)[-1]
    cuts = sorted({0.0, float(length), *[float(c) for c in crossings]})
    out: list[list[float]] = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        if b - a <= 1e-6:
            continue
        mx, my = point_at(vertices, 0.5 * (a + b))
        if points_in_polygons(mx, my, polygons)[0]:
            if out and abs(out[-1][1] - a) <= 1e-6:
                out[-1][1] = b
            else:
                out.append([a, b])
    return [(a, b) for a, b in out]


def _point_segment_distance(px, py, edges):
    """Distance from every point to every edge, chunked; returns
    (min distance per point, nearest x, nearest y on the edges)."""
    best = np.full(px.shape, np.inf)
    bx = np.zeros(px.shape)
    by = np.zeros(px.shape)
    for x0, y0, x1, y1 in edges:
        dx, dy = x1 - x0, y1 - y0
        ll = dx * dx + dy * dy
        if ll <= 0:
            t = np.zeros(px.shape)
        else:
            t = np.clip(((px - x0) * dx + (py - y0) * dy) / ll, 0, 1)
        cx, cy = x0 + t * dx, y0 + t * dy
        d = np.hypot(px - cx, py - cy)
        better = d < best
        best[better] = d[better]
        bx[better] = cx[better]
        by[better] = cy[better]
    return best, bx, by


# ----------------------------------------------------------------------
# Water surface along the section


@dataclass
class WetInterval:
    start: float
    end: float
    start_kind: str
    end_kind: str
    max_depth: float
    max_depth_ch: float
    ch: np.ndarray  # chainages incl. the interpolated edges
    ground: np.ndarray
    wse: np.ndarray

    @property
    def length(self) -> float:
        return self.end - self.start

    def edge(self, which: str) -> tuple[float, float, str]:
        """(chainage, water level, kind) at the 'start' or 'end' edge."""
        if which == "start":
            return self.start, float(self.wse[0]), self.start_kind
        return self.end, float(self.wse[-1]), self.end_kind


def extend_wse(chainage: np.ndarray, wse: np.ndarray, distance: float) -> np.ndarray:
    """Fills NaN water levels from the nearest valid sample along the
    section when it lies within ``distance`` (holds the value)."""
    out = np.asarray(wse, dtype=np.float64).copy()
    valid = np.flatnonzero(np.isfinite(out))
    if len(valid) == 0 or distance <= 0:
        return out
    gaps = np.flatnonzero(~np.isfinite(out))
    if len(gaps) == 0:
        return out
    vch = chainage[valid]
    pos = np.searchsorted(vch, chainage[gaps])
    left = np.clip(pos - 1, 0, len(valid) - 1)
    right = np.clip(pos, 0, len(valid) - 1)
    dl = np.abs(chainage[gaps] - vch[left])
    dr = np.abs(vch[right] - chainage[gaps])
    use_right = dr < dl
    src = np.where(use_right, valid[right], valid[left])
    dist = np.where(use_right, dr, dl)
    fill = dist <= distance + _EPS
    out[gaps[fill]] = out[src[fill]]
    return out


def wet_intervals(
    chainage: np.ndarray, ground: np.ndarray, wse: np.ndarray
) -> list[WetInterval]:
    """Runs of samples with water above the ground, with the waterlines
    interpolated linearly between the last wet and first dry sample."""
    depth = wse - ground
    wet = np.isfinite(depth) & (depth > 0)
    n = len(chainage)
    out = []
    i = 0
    while i < n:
        if not wet[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and wet[j + 1]:
            j += 1
        ch = list(chainage[i : j + 1])
        gr = list(ground[i : j + 1])
        ws = list(wse[i : j + 1])
        # start edge
        if i == 0:
            start_kind = EDGE_SECTION_END
        elif np.isfinite(depth[i - 1]):
            t = depth[i] / (depth[i] - depth[i - 1])
            c = chainage[i] - t * (chainage[i] - chainage[i - 1])
            z = ground[i] + t * (ground[i - 1] - ground[i])
            ch.insert(0, c)
            gr.insert(0, z)
            ws.insert(0, z)
            start_kind = EDGE_WATERLINE
        else:
            start_kind = EDGE_WSE_LIMIT
        # end edge
        if j == n - 1:
            end_kind = EDGE_SECTION_END
        elif np.isfinite(depth[j + 1]):
            t = depth[j] / (depth[j] - depth[j + 1])
            c = chainage[j] + t * (chainage[j + 1] - chainage[j])
            z = ground[j] + t * (ground[j + 1] - ground[j])
            ch.append(c)
            gr.append(z)
            ws.append(z)
            end_kind = EDGE_WATERLINE
        else:
            end_kind = EDGE_WSE_LIMIT
        k = i + int(np.argmax(depth[i : j + 1]))
        out.append(
            WetInterval(
                start=float(ch[0]),
                end=float(ch[-1]),
                start_kind=start_kind,
                end_kind=end_kind,
                max_depth=float(depth[k]),
                max_depth_ch=float(chainage[k]),
                ch=np.asarray(ch, float),
                ground=np.asarray(gr, float),
                wse=np.asarray(ws, float),
            )
        )
        i = j + 1
    return out


# ----------------------------------------------------------------------
# Sections


@dataclass
class EventProfile:
    label: str
    wse_sampled: np.ndarray  # as read from the raster
    wse: np.ndarray  # after the edge extension
    depth: np.ndarray  # NaN where dry
    intervals: list[WetInterval]

    @property
    def wet_length(self) -> float:
        return float(sum(w.length for w in self.intervals))

    @property
    def max_depth(self) -> float | None:
        d = self.depth[np.isfinite(self.depth)]
        return float(d.max()) if d.size else None

    def waterlines(self) -> list[float]:
        out = []
        for w in self.intervals:
            if w.start_kind == EDGE_WATERLINE:
                out.append(w.start)
            if w.end_kind == EDGE_WATERLINE:
                out.append(w.end)
        return out


@dataclass
class SectionProfile:
    label: str
    vertices: np.ndarray
    chainage: np.ndarray
    x: np.ndarray
    y: np.ndarray
    ground: np.ndarray
    property_crossings: list[float]
    property_intervals: list[tuple[float, float]]
    has_property: bool
    events: list[EventProfile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(self.chainage[-1])

    def ground_at(self, ch) -> float:
        ok = np.isfinite(self.ground)
        if not ok.any():
            return float("nan")
        return float(np.interp(ch, self.chainage[ok], self.ground[ok]))


def section_labels(n: int) -> list[str]:
    """A, B, ... Z, AA, AB, ..."""
    out = []
    for i in range(n):
        s, k = "", i
        while True:
            s = chr(ord("A") + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        out.append(s)
    return out


def build_section(
    label: str,
    vertices,
    dem: Grid,
    wse_grids: Sequence[tuple[str, Grid]] = (),
    interval: float | None = None,
    polygons=None,
    edge_extension_cells: float = DEFAULT_EDGE_EXTENSION_CELLS,
) -> SectionProfile:
    v = clean_vertices(vertices)
    interval = interval or dem.cell_size
    crossings = polygon_crossings(v, polygons) if polygons else []
    ch = stations(v, interval, crossings)
    x, y = point_at(v, ch)
    ground = dem.sample(x, y)
    warnings = []
    if not np.isfinite(ground).any():
        raise ProfileError(f"Section {label} does not overlap the DEM.")
    missing = int((~np.isfinite(ground)).sum())
    if missing:
        warnings.append(
            f"Section {label}: no DEM value at {missing} of {len(ch)} sample(s) "
            "(off the DEM or NoData) - the ground line has gaps there."
        )
    section = SectionProfile(
        label=label,
        vertices=v,
        chainage=ch,
        x=x,
        y=y,
        ground=ground,
        property_crossings=crossings,
        property_intervals=inside_intervals(v, polygons, crossings) if polygons else [],
        has_property=bool(polygons),
        warnings=warnings,
    )
    for ev_label, grid in wse_grids:
        raw = grid.sample(x, y)
        # carried 1.5 cells past the last wet cell; the extra sample step
        # covers the offset between the samples and the cell edges
        reach = edge_extension_cells * grid.cell_size + interval
        ext = extend_wse(ch, raw, reach)
        d = ext - ground
        depth = np.where(np.isfinite(d) & (d > 0), d, np.nan)
        section.events.append(
            EventProfile(
                label=ev_label,
                wse_sampled=raw,
                wse=ext,
                depth=depth,
                intervals=wet_intervals(ch, ground, ext),
            )
        )
    return section


# ----------------------------------------------------------------------
# Measurements


MEASUREMENT_FIELDS = [
    ("section", "section label"),
    ("event", "water-surface (event) label"),
    ("status", "property dry / property inundated / no flooding / ..."),
    ("section_length_m", "length of the section line, m"),
    ("section_wet_length_m", "total wet length along the section, m"),
    ("section_max_depth_m", "maximum depth along the section, m"),
    ("waterlines_ch_m", "chainages of the waterlines on the section, m"),
    ("prop_start_ch_m", "chainage where the section enters the property, m"),
    ("prop_end_ch_m", "chainage where the section leaves the property, m"),
    ("prop_length_m", "length of the section inside the property, m"),
    ("boundary_ch_m", "property boundary crossing nearest the flood, m"),
    ("ground_at_boundary_m", "ground level at that boundary crossing, m"),
    ("flood_edge_ch_m", "chainage of the nearest flood edge, m"),
    ("flood_edge_kind", "waterline, section end or limit of WSE data"),
    ("wse_m", "water level used for the vertical measurements, m"),
    ("horiz_dist_m", "boundary crossing to flood edge along the section, m"),
    ("vert_dist_m", "ground at the boundary crossing minus the water level, m"),
    ("lowest_ground_in_prop_m", "lowest ground inside the property, m"),
    ("lowest_minus_wse_m", "lowest ground inside the property minus water level, m"),
    ("wet_length_in_prop_m", "wet length inside the property, m"),
    ("wet_fraction_in_prop", "wet length / length inside the property"),
    ("max_depth_in_prop_m", "maximum depth inside the property, m"),
    ("max_depth_ch_m", "chainage of that maximum depth, m"),
    ("plan_dist_m", "shortest plan distance property to flooded cell, m"),
    ("plan_status", "result of the plan-distance search"),
    ("wet_area_in_prop_m2", "flooded area inside the property (cells), m2"),
]


def _overlap(a: tuple[float, float], b: tuple[float, float]):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if hi - lo > 1e-6 else None


def measure(section: SectionProfile, event: EventProfile) -> dict:
    """One measurement row for a section and an event (see
    MEASUREMENT_FIELDS). Also returns 'dimension' (what to draw)."""
    row: dict = {name: None for name, _ in MEASUREMENT_FIELDS}
    row.update(
        section=section.label,
        event=event.label,
        section_length_m=section.length,
        section_wet_length_m=event.wet_length,
        section_max_depth_m=event.max_depth,
        waterlines_ch_m=event.waterlines(),
    )
    props = section.property_intervals
    dimension = None
    if props:
        row["prop_start_ch_m"] = props[0][0]
        row["prop_end_ch_m"] = props[-1][1]
        row["prop_length_m"] = float(sum(b - a for a, b in props))
        in_prop = np.zeros(section.chainage.shape, bool)
        for a, b in props:
            in_prop |= (section.chainage >= a - 1e-6) & (section.chainage <= b + 1e-6)
        g = section.ground[in_prop]
        g = g[np.isfinite(g)]
        lowest = float(g.min()) if g.size else None
        row["lowest_ground_in_prop_m"] = lowest

    if not section.has_property:
        row["status"] = STATUS_NO_PROPERTY if event.intervals else STATUS_NO_FLOOD
    elif not props:
        row["status"] = STATUS_NOT_CROSSED
    elif not event.intervals:
        row["status"] = STATUS_NO_FLOOD
    else:
        pieces = []
        for w in event.intervals:
            for p in props:
                o = _overlap(p, (w.start, w.end))
                if o:
                    pieces.append((o, w))
        if pieces:
            wet_len = float(sum(o[1] - o[0] for o, _ in pieces))
            row["status"] = STATUS_INUNDATED
            row["wet_length_in_prop_m"] = wet_len
            row["wet_fraction_in_prop"] = wet_len / row["prop_length_m"]
            d = np.where(in_prop, event.depth, np.nan)
            if np.isfinite(d).any():
                k = int(np.nanargmax(d))
                row["max_depth_in_prop_m"] = float(d[k])
                row["max_depth_ch_m"] = float(section.chainage[k])
                row["wse_m"] = float(event.wse[k])
            else:  # wet only between samples
                o, w = pieces[0]
                mid = 0.5 * (o[0] + o[1])
                row["max_depth_in_prop_m"] = 0.0
                row["max_depth_ch_m"] = mid
                row["wse_m"] = float(np.interp(mid, w.ch, w.wse))
            if row["lowest_ground_in_prop_m"] is not None:
                row["lowest_minus_wse_m"] = (
                    row["lowest_ground_in_prop_m"] - row["wse_m"]
                )
            longest = max(pieces, key=lambda pw: pw[0][1] - pw[0][0])[0]
            dimension = {
                "kind": "inundated",
                "wet_from": longest[0],
                "wet_to": longest[1],
                "depth_ch": row["max_depth_ch_m"],
                "depth": row["max_depth_in_prop_m"],
                "wse": row["wse_m"],
            }
        else:
            best = None
            for w in event.intervals:
                for p0, p1 in props:
                    if w.end <= p0:
                        cand = (p0 - w.end, p0, w, "end")
                    elif w.start >= p1:
                        cand = (w.start - p1, p1, w, "start")
                    else:
                        continue
                    if best is None or cand[0] < best[0]:
                        best = cand
            gap, boundary, w, which = best
            edge_ch, wse, kind = w.edge(which)
            gb = section.ground_at(boundary)
            row.update(
                status=STATUS_DRY,
                boundary_ch_m=boundary,
                ground_at_boundary_m=gb,
                flood_edge_ch_m=edge_ch,
                flood_edge_kind=kind,
                wse_m=wse,
                horiz_dist_m=gap,
                vert_dist_m=gb - wse,
                wet_length_in_prop_m=0.0,
                wet_fraction_in_prop=0.0,
            )
            if row["lowest_ground_in_prop_m"] is not None:
                row["lowest_minus_wse_m"] = row["lowest_ground_in_prop_m"] - wse
            dimension = {
                "kind": "dry",
                "boundary_ch": boundary,
                "ground_at_boundary": gb,
                "edge_ch": edge_ch,
                "edge_kind": kind,
                "wse": wse,
                "horiz": gap,
                "vert": gb - wse,
            }
    row["dimension"] = dimension
    return row


# ----------------------------------------------------------------------
# Plan distance


@dataclass
class PlanDistance:
    event: str
    status: str
    distance_m: float | None = None
    property_xy: tuple[float, float] | None = None
    flood_xy: tuple[float, float] | None = None
    wet_area_in_property_m2: float = 0.0
    accuracy_m: float = 0.0


def plan_distance(
    event: str,
    polygons,
    wse: Grid,
    ground_at: Callable[[np.ndarray, np.ndarray], np.ndarray],
    search_radius: float = DEFAULT_SEARCH_RADIUS_M,
) -> PlanDistance:
    """Shortest plan distance from the property boundary to the centre of
    the nearest flooded WSE cell (WSE > ground at the cell centre). Cells
    inside the property make it 'inundated' with the flooded area counted.
    Accuracy: about half a WSE cell (cell centre vs. true waterline)."""
    gt = wse.geotransform
    accuracy = 0.5 * wse.cell_size
    edges = _edges(polygons)
    if len(edges) == 0:
        raise ProfileError("The property polygon has no rings.")
    xs = np.concatenate([edges[:, 0], edges[:, 2]])
    ys = np.concatenate([edges[:, 1], edges[:, 3]])
    x0, x1 = xs.min() - search_radius, xs.max() + search_radius
    y0, y1 = ys.min() - search_radius, ys.max() + search_radius
    rows, cols = wse.array.shape
    c_lo = max(int(math.floor((min(x0, x1) - gt[0]) / gt[1])), 0)
    c_hi = min(int(math.ceil((max(x0, x1) - gt[0]) / gt[1])), cols)
    ra = (y0 - gt[3]) / gt[5]
    rb = (y1 - gt[3]) / gt[5]
    r_lo = max(int(math.floor(min(ra, rb))), 0)
    r_hi = min(int(math.ceil(max(ra, rb))), rows)
    none = PlanDistance(
        event, "no flooding within the search radius", accuracy_m=accuracy
    )
    if r_hi <= r_lo or c_hi <= c_lo:
        return none
    sub = wse.array[r_lo:r_hi, c_lo:c_hi]
    rr, cc = np.nonzero(np.isfinite(sub))
    if rr.size == 0:
        return none
    cx, cy = wse.cell_centres(rr + r_lo, cc + c_lo)
    level = sub[rr, cc]
    gz = ground_at(cx, cy)
    wet_sel = np.isfinite(gz) & (level > gz)
    if not wet_sel.any():
        return none
    wet = np.zeros(sub.shape, bool)
    wet[rr[wet_sel], cc[wet_sel]] = True
    wr, wc = np.nonzero(wet)
    wx, wy = wse.cell_centres(wr + r_lo, wc + c_lo)

    # flooded cells inside the property
    bx0, bx1 = xs.min(), xs.max()
    by0, by1 = ys.min(), ys.max()
    near = (wx >= bx0) & (wx <= bx1) & (wy >= by0) & (wy <= by1)
    inside = np.zeros(wx.shape, bool)
    if near.any():
        inside[near] = points_in_polygons(wx[near], wy[near], polygons)
    if inside.any():
        cell_area = abs(gt[1] * gt[5])
        return PlanDistance(
            event,
            "property inundated",
            distance_m=0.0,
            wet_area_in_property_m2=float(inside.sum() * cell_area),
            accuracy_m=accuracy,
        )

    # only the outer cells of each flooded area can be nearest
    pad = np.pad(wet, 1, constant_values=False)
    interior = (
        pad[1:-1, 1:-1]
        & pad[:-2, 1:-1]
        & pad[2:, 1:-1]
        & pad[1:-1, :-2]
        & pad[1:-1, 2:]
    )
    border = wet[wr, wc] & ~interior[wr, wc]
    px, py = wx[border], wy[border]
    best_d, best = np.inf, None
    chunk = 200000
    for s in range(0, len(px), chunk):
        d, nx, ny = _point_segment_distance(px[s : s + chunk], py[s : s + chunk], edges)
        k = int(np.argmin(d))
        if d[k] < best_d:
            best_d = float(d[k])
            best = ((float(nx[k]), float(ny[k])), (float(px[s + k]), float(py[s + k])))
    if best is None or best_d > search_radius:
        return none
    return PlanDistance(
        event,
        "property dry",
        distance_m=best_d,
        property_xy=best[0],
        flood_xy=best[1],
        accuracy_m=accuracy,
    )
