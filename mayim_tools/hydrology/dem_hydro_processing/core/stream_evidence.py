"""
Stream centreline rasterization and crossing-evidence scoring.

Produces the one remaining input classification.py's own
combine_evidence() already accepts but that nothing has, until now,
actually computed: stream_crossing. This module has no direct basis
in the Lindsay & Creed literature - it is this project's own
implementation of the general idea, agreed in the original scoping
discussion (2026-09-23), of using whatever reference hydrography a
project has available as an additional, independent line of evidence
- a modern equivalent of the "examine the source data" approach
Lindsay & Creed's own 2006 paper lists among the five ways of
distinguishing real depressions from artefacts, automated here rather
than left as manual inspection.

WHY THIS IS BUILT INDEPENDENTLY OF QGIS: a stream centreline arrives,
in practice, as a QGIS vector layer - but rasterizing a polyline onto
a grid is a standard, well-established GIS operation in its own
right (not something inherently tied to QGIS), so it belongs in this
project's zero-QGIS-dependency core, tested on its own, exactly like
every other module here. The QGIS-facing wrapper (not yet built) will
be responsible for extracting vertex coordinates and the raster's own
georeferencing from QGIS objects and handing them to this module as
plain numbers.

Rasterization uses Bresenham's line algorithm (the standard, well-
established method for determining which grid cells a line passes
through - not specific to GIS, originally from computer graphics,
long-established for exactly this purpose). Only north-up (no
rotation) rasters are supported, via RasterTransform - matching every
other raster this whole plugin suite has ever handled; a rotated
raster has never come up anywhere else in this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label

from .depression_hierarchy import DepressionHierarchy


@dataclass
class RasterTransform:
    """A simple, north-up (no rotation) affine georeferencing
    transform - the standard convention for every raster this whole
    plugin suite works with.

    origin_x/origin_y: real-world coordinates of the raster's own
    top-left corner (the outer edge of row 0, column 0 - not that
    cell's centre).
    pixel_width/pixel_height: cell size in each direction, both given
    as POSITIVE magnitudes (row increases as y DEcreases, matching
    standard raster convention - handled internally, not by requiring
    a signed pixel_height the way some GIS libraries do, since that is
    an easy sign mistake to make when wiring this up later).
    """

    origin_x: float
    origin_y: float
    pixel_width: float
    pixel_height: float

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Returns (row, col) as FRACTIONAL pixel coordinates (not yet
        rounded to a specific cell) - kept fractional so a caller
        rasterizing a line can round consistently at the point that
        actually matters."""
        col = (x - self.origin_x) / self.pixel_width
        row = (self.origin_y - y) / self.pixel_height
        return row, col


def _bresenham_line(r0: int, c0: int, r1: int, c1: int):
    """Yields every (row, col) grid cell along the line from (r0,c0)
    to (r1,c1) inclusive of both endpoints - the standard Bresenham
    line algorithm."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        yield (r, c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def rasterize_polylines(
    polylines: list[list[tuple[float, float]]],
    transform: RasterTransform,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterizes one or more polylines (each a list of (x, y)
    real-world vertex coordinates - a stream network is typically
    several separate line features, hence a list of polylines rather
    than just one) onto a boolean grid of the given shape. Vertices
    outside the raster are handled gracefully (a segment that crosses
    the boundary contributes only the cells actually within the
    raster, not an error) - the classic case being a stream that
    continues beyond the DEM's own extent.
    """
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)

    for vertices in polylines:
        if len(vertices) < 2:
            continue
        pixel_coords = [transform.world_to_pixel(x, y) for x, y in vertices]
        for (r0f, c0f), (r1f, c1f) in zip(pixel_coords[:-1], pixel_coords[1:]):
            r0, c0 = int(round(r0f)), int(round(c0f))
            r1, c1 = int(round(r1f)), int(round(c1f))
            for r, c in _bresenham_line(r0, c0, r1, c1):
                if 0 <= r < rows and 0 <= c < cols:
                    mask[r, c] = True

    return mask


def compute_stream_crossing(
    hierarchy: DepressionHierarchy,
    leaf_id: int,
    stream_mask: np.ndarray,
    buffer_cells: int = 1,
) -> bool:
    """Whether a rasterized stream passes through - or immediately
    alongside - a given leaf's own cells. A small buffer (default 1
    cell) is applied around the depression before checking overlap,
    rather than requiring the stream to intersect the depression's
    exact interior cells: a mapped stream centreline is a
    simplification of the real channel's own width and position, and
    a stream that clearly feeds a depression without its rasterized
    centreline landing exactly inside it should still count as a
    crossing.
    """
    node = hierarchy.nodes[leaf_id]
    if node.is_ocean or node.children:
        raise ValueError(
            f"compute_stream_crossing expects a LEAF id, got node {leaf_id}"
        )

    mask = hierarchy.cells_of(leaf_id)
    if buffer_cells > 0:
        mask = _dilate(mask, buffer_cells)

    return bool(np.any(mask & stream_mask))


def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Expands a boolean mask outward by n cells (8-connected).
    Uses scipy.ndimage (already a dependency of this package via
    stochastic_classification.py) - an earlier version of this module
    hand-rolled a pure-Python dilation to avoid that import, which was
    correct but far too slow to run per-depression on a real DEM."""
    if n <= 0:
        return mask.copy()
    return binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=n)


STREAM_NONE = "none"
STREAM_TERMINATES_IN = "terminates_in"
STREAM_PASSES_THROUGH = "passes_through"


def compute_stream_relation(
    hierarchy: DepressionHierarchy,
    leaf_id: int,
    stream_mask: np.ndarray,
    buffer_cells: int = 1,
    dem: np.ndarray | None = None,
) -> str:
    """Classifies how a rasterized stream relates to a depression:

    "none": no stream within the depression (plus buffer).
    "terminates_in": a stream reaches the depression from at most ONE
        side and does not continue out the other - consistent with a
        genuine waterbody (lake, pan, wetland) being fed by that stream.
    "passes_through": stream continues on BOTH sides of the depression
        - i.e. the stream is mapped as flowing straight across a place
        the DEM says is a closed basin. Water that really flowed along
        that stream would not be trapped, so the barrier holding the
        depression closed is very likely an artefact the DEM failed to
        capture (a culvert, bridge or road embankment) - the classic
        case this whole tool exists to handle.

    Implemented by removing the (buffered) depression from the stream
    mask and counting the separate stream pieces left touching it: two
    or more pieces means the stream enters one side and leaves another.

    Despite the parameter's name (kept for minimal disruption to
    existing callers), this now accepts ANY non-ocean node - a leaf OR
    a meta-depression - for the same reason as
    classification.compute_depression_metrics (see hierarchical
    classification's own module docstring for the full finding).

    KNOWN LIMITATION: a stream that genuinely passes through but has
    one side beyond the DEM's own edge, or one side simply not mapped
    in the supplied stream layer, appears as a single piece and is
    reported "terminates_in" - the safe direction to err (it can only
    make the tool more cautious about calling something an artefact).
    """
    node = hierarchy.nodes[leaf_id]
    if node.is_ocean:
        raise ValueError(
            f"compute_stream_relation expects a non-ocean node, got the ocean node {leaf_id}"
        )

    # With `dem` supplied, the depression is its actual footprint (cells below
    # its spill elevation). Without it this falls back to cells_of(), which is
    # the leaf's CATCHMENT and includes uphill cells - so a stream on the
    # hillside above a pond would wrongly count as inside it. Always pass dem.
    base = (
        hierarchy.footprint_of(leaf_id, dem)
        if dem is not None
        else hierarchy.cells_of(leaf_id)
    )
    if dem is not None and not base.any():
        base = hierarchy.cells_of(leaf_id)
    region = _dilate(base, buffer_cells)
    if not np.any(region & stream_mask):
        return STREAM_NONE

    outside = stream_mask & ~region
    if not outside.any():
        return STREAM_TERMINATES_IN

    labels, n_pieces = label(outside, structure=np.ones((3, 3), dtype=bool))
    touching_ring = _dilate(region, 1) & ~region
    touching_labels = set(np.unique(labels[touching_ring & outside]).tolist()) - {0}
    return STREAM_PASSES_THROUGH if len(touching_labels) >= 2 else STREAM_TERMINATES_IN
