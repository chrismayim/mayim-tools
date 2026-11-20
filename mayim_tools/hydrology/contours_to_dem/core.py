"""
Contours to DEM - the parts with no QGIS: heights along stream and ridge
lines from their contour crossings, the contour interval, and the check for
flat stretches along the lines.

Why: a TIN built from contours alone has flat triangles wherever a contour
bends sharply (valley floors, ridge crests). All three corners of such a
triangle lie on one contour, so the valley floor comes out as a staircase:
flat for the length of each contour's V, then a full contour interval down
at the next one. A stream line with heights that fall steadily between its
contour crossings, added to the TIN as a break line, cuts those triangles
and turns the staircase into a slope (and a ridge line does the same on a
crest).

Heights along a line:
- at each contour crossing, the contour's level;
- between two crossings, linear in distance along the line;
- before the first and after the last crossing, the gradient of the
  nearest two crossings carried on, limited so the line stays within one
  contour interval of the last crossing (it crosses no further contour);
- a line with one crossing is flat at that level beyond it (no gradient),
  a line with none is left out.

Recommended by Claude for Chris, 2026-11-19 (his profile through a TIN from
2 m contours showed 200-250 m flats along the valley floor).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOOL_VERSION = "1.0.0"
FLAT_TOL_FRACTION = 0.001  # of the contour interval: a step smaller is flat


class ContourError(ValueError):
    pass


@dataclass
class Crossing:
    chainage: float  # along the line, m
    z: float  # contour level
    x: float
    y: float


@dataclass
class LineHeights:
    """Heights along one stream or ridge line."""

    name: str
    kind: str  # "stream" or "ridge"
    xyz: np.ndarray  # (n, 3) vertices with heights, crossings included
    chainage: np.ndarray  # (n,)
    crossings: list[Crossing]
    length: float
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return len(self.crossings) >= 1 and len(self.xyz) >= 2

    @property
    def fall(self) -> float:
        return float(self.xyz[0, 2] - self.xyz[-1, 2]) if self.usable else np.nan


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _chainages(xy: np.ndarray) -> np.ndarray:
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return np.r_[0.0, np.cumsum(seg)]


def segment_crossings(line_xy, contour_xy):
    """Points where the polyline ``line_xy`` crosses the polyline
    ``contour_xy``: [(chainage along the line, x, y)], vectorised over all
    segment pairs. Touching at a vertex counts once."""
    a = np.asarray(line_xy, float)
    b = np.asarray(contour_xy, float)
    if len(a) < 2 or len(b) < 2:
        return []
    p, r = a[:-1], np.diff(a, axis=0)  # line segments p + t r
    q, s = b[:-1], np.diff(b, axis=0)  # contour segments q + u s
    # quick reject on bounding boxes
    amin = np.minimum(a[:-1], a[1:])
    amax = np.maximum(a[:-1], a[1:])
    bmin = np.minimum(b[:-1], b[1:])
    bmax = np.maximum(b[:-1], b[1:])
    i, j = np.nonzero(
        (amin[:, None, 0] <= bmax[None, :, 0])
        & (amax[:, None, 0] >= bmin[None, :, 0])
        & (amin[:, None, 1] <= bmax[None, :, 1])
        & (amax[:, None, 1] >= bmin[None, :, 1])
    )
    if not len(i):
        return []
    rr, ss = r[i], s[j]
    qp = q[j] - p[i]
    den = rr[:, 0] * ss[:, 1] - rr[:, 1] * ss[:, 0]
    ok = np.abs(den) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * ss[:, 1] - qp[:, 1] * ss[:, 0]) / den
        u = (qp[:, 0] * rr[:, 1] - qp[:, 1] * rr[:, 0]) / den
    eps = 1e-9
    hit = ok & (t >= -eps) & (t <= 1 + eps) & (u >= -eps) & (u <= 1 + eps)
    if not hit.any():
        return []
    ch = _chainages(a)
    seg_len = np.diff(ch)
    out = []
    for k in np.flatnonzero(hit):
        tk = float(np.clip(t[k], 0.0, 1.0))
        x, y = p[i[k]] + tk * r[i[k]]
        out.append((float(ch[i[k]] + tk * seg_len[i[k]]), float(x), float(y)))
    out.sort()
    # a crossing exactly on a shared vertex is found on both segments
    merged = []
    for c in out:
        if merged and abs(c[0] - merged[-1][0]) < 1e-6:
            continue
        merged.append(c)
    return merged


def line_crossings(line_xy, contours) -> list[Crossing]:
    """All crossings of a line with the contours [(xy, z)], sorted along the
    line. Where two contours meet the line at the same point (touching
    contours), the first one found is kept."""
    out = []
    for xy, z in contours:
        for ch, x, y in segment_crossings(line_xy, xy):
            out.append(Crossing(ch, float(z), x, y))
    out.sort(key=lambda c: c.chainage)
    kept = []
    for c in out:
        if kept and abs(c.chainage - kept[-1].chainage) < 1e-6:
            continue
        kept.append(c)
    return kept


def contour_interval(levels) -> float:
    """The contour interval: the most common positive difference between
    the distinct contour levels (rounded to mm)."""
    z = np.unique(np.round(np.asarray(levels, float), 3))
    if len(z) < 2:
        raise ContourError("The contours have fewer than two different levels.")
    d = np.round(np.diff(z), 3)
    d = d[d > 0]
    vals, counts = np.unique(d, return_counts=True)
    return float(vals[np.argmax(counts)])


def densify(xy: np.ndarray, step: float, extra_chainages=()) -> tuple:
    """Vertices every ``step`` along the line (keeping the original
    vertices) plus points at ``extra_chainages``: (xy, chainage)."""
    xy = np.asarray(xy, float)
    ch = _chainages(xy)
    marks = [ch]
    for c0, c1 in zip(ch[:-1], ch[1:], strict=True):
        n = int(np.floor((c1 - c0) / step))
        if n > 1:
            marks.append(c0 + step * np.arange(1, n))
    marks.append(np.asarray(list(extra_chainages), float))
    allc = np.unique(np.round(np.concatenate(marks), 6))
    allc = allc[(allc >= 0) & (allc <= ch[-1] + 1e-9)]
    x = np.interp(allc, ch, xy[:, 0])
    y = np.interp(allc, ch, xy[:, 1])
    return np.c_[x, y], allc


# --------------------------------------------------------------------------
# Heights along a line
# --------------------------------------------------------------------------


def line_heights(
    name: str,
    line_xy,
    contours,
    interval: float,
    step: float,
    kind: str = "stream",
) -> LineHeights:
    """Heights along a stream or ridge line from its contour crossings."""
    xy = np.asarray(line_xy, float)
    crossings = line_crossings(xy, contours)
    length = float(_chainages(xy)[-1]) if len(xy) >= 2 else 0.0
    warnings: list[str] = []
    if not crossings:
        warnings.append("crosses no contour, so it is left out")
        return LineHeights(name, kind, np.zeros((0, 3)), np.zeros(0), [], length,
                           warnings)  # fmt: skip
    pts, ch = densify(xy, step, [c.chainage for c in crossings])
    cc = np.array([c.chainage for c in crossings])
    cz = np.array([c.z for c in crossings])
    z = np.interp(ch, cc, cz)
    if len(crossings) == 1:
        warnings.append(
            "crosses one contour only, so it is flat at that level (no "
            "gradient); extend it across a second contour"
        )
    else:
        # carry the end gradients on, within one interval of the end level
        g0 = (cz[1] - cz[0]) / max(cc[1] - cc[0], 1e-9)
        g1 = (cz[-1] - cz[-2]) / max(cc[-1] - cc[-2], 1e-9)
        lim = 0.999 * interval
        before = ch < cc[0]
        z[before] = cz[0] + np.clip(g0 * (ch[before] - cc[0]), -lim, lim)
        after = ch > cc[-1]
        z[after] = cz[-1] + np.clip(g1 * (ch[after] - cc[-1]), -lim, lim)
        dz = np.diff(cz)
        if kind == "stream" and (dz > 0).any() and (dz < 0).any():
            warnings.append(
                "its contour levels go both up and down along it - check that "
                "it follows the valley floor and does not climb over a ridge"
            )
        same = int(np.sum(np.abs(dz) < 1e-6))
        if same:
            warnings.append(
                f"crosses the same contour twice in a row {same} time(s) "
                "(a bend across one contour); it is flat between those crossings"
            )
        if np.any(np.abs(dz) > 1.5 * interval):
            warnings.append(
                "skips a contour between two crossings (a gap or a missing "
                "contour) - the height is interpolated across it"
            )
    return LineHeights(name, kind, np.c_[pts, z], ch, crossings, length, warnings)


def oriented_downstream(line: LineHeights) -> LineHeights:
    """A stream line runs downhill in the report: reversed when its end is
    higher than its start (the heights are unchanged)."""
    if not line.usable or line.kind != "stream" or line.xyz[0, 2] >= line.xyz[-1, 2]:
        return line
    ch = line.length - line.chainage[::-1]
    crossings = [
        Crossing(line.length - c.chainage, c.z, c.x, c.y)
        for c in reversed(line.crossings)
    ]
    return LineHeights(line.name, line.kind, line.xyz[::-1].copy(), ch, crossings,
                       line.length, line.warnings)  # fmt: skip


# --------------------------------------------------------------------------
# Flat stretches along a line (the check)
# --------------------------------------------------------------------------


@dataclass
class FlatCheck:
    longest_flat_m: float  # longest stretch with no change in height
    flat_share: float  # share of the sampled length that is flat
    sampled_m: float


def flat_stretches(chainage, z, interval: float) -> FlatCheck:
    """Longest flat stretch and flat share of a sampled profile (NaN cells
    left out). Flat = a step smaller than 0.1 % of the contour interval."""
    ch = np.asarray(chainage, float)
    zz = np.asarray(z, float)
    ok = np.isfinite(zz)
    pair = ok[:-1] & ok[1:]  # steps between two sampled cells
    if not pair.any():
        return FlatCheck(np.nan, np.nan, 0.0)
    seg = np.abs(np.diff(ch))
    with np.errstate(invalid="ignore"):
        flat = pair & (np.abs(np.diff(zz)) < FLAT_TOL_FRACTION * interval)
    longest = run = 0.0
    for f, p, s in zip(flat, pair, seg, strict=True):
        run = run + s if f else 0.0
        if not p:
            run = 0.0
        longest = max(longest, run)
    total = float(seg[pair].sum())
    return FlatCheck(float(longest), float(seg[flat].sum() / total) if total else 0.0,
                     total)  # fmt: skip


def profile_samples(line: LineHeights, step: float):
    """(chainage, x, y) every ``step`` along the line, for the check."""
    n = max(int(np.ceil(line.length / step)), 1)
    ch = np.linspace(0.0, line.length, n + 1)
    x = np.interp(ch, line.chainage, line.xyz[:, 0])
    y = np.interp(ch, line.chainage, line.xyz[:, 1])
    return ch, x, y
