"""Local windows around a depression, so per-depression work costs time in
proportion to the depression and not to the whole raster.

On a multi-million-cell DEM with tens of thousands of depressions, doing
whole-grid operations (boolean masks, np.isin, copies) for each depression
takes days. Everything per depression happens on a crop around the node's
cells instead; results are shifted back to grid coordinates.

Zero QGIS dependency.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .depression_hierarchy import DepressionHierarchy


class NodeWindows:
    """Bounding boxes of every leaf (built once) and of any node (union of
    its descendant leaves)."""

    def __init__(self, hierarchy: DepressionHierarchy):
        self.hierarchy = hierarchy
        lab = np.where(hierarchy.leaf_label > 0, hierarchy.leaf_label, 0)
        self.shape = lab.shape
        self._leaf_box: dict[int, tuple[int, int, int, int]] = {}
        for i, sl in enumerate(ndimage.find_objects(lab.astype(np.int32)), start=1):
            if sl is not None:
                self._leaf_box[i] = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)
        self._cache: dict[int, tuple[int, int, int, int] | None] = {}

    def bbox(self, node_id: int):
        """(r0, r1, c0, c1) half-open, or None when the node has no cells."""
        if node_id in self._cache:
            return self._cache[node_id]
        boxes = [
            self._leaf_box[i]
            for i in self.hierarchy._descendant_leaf_ids(node_id)
            if i in self._leaf_box
        ]
        if not boxes:
            box = None
        else:
            box = (
                min(b[0] for b in boxes),
                max(b[1] for b in boxes),
                min(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
        self._cache[node_id] = box
        return box

    def window(self, node_id: int, margin: int):
        """Returns (r0, r1, c0, c1) padded by `margin` and clipped to the
        grid, or the full grid when the node has no cells."""
        rows, cols = self.shape
        box = self.bbox(node_id)
        if box is None:
            return 0, rows, 0, cols
        r0, r1, c0, c1 = box
        return (
            max(0, r0 - margin),
            min(rows, r1 + margin),
            max(0, c0 - margin),
            min(cols, c1 + margin),
        )

    def default_margin(self, node_id: int) -> int:
        box = self.bbox(node_id)
        if box is None:
            return 0
        return max(8, box[1] - box[0], box[3] - box[2])

    def sub_hierarchy(self, win) -> DepressionHierarchy:
        r0, r1, c0, c1 = win
        h = self.hierarchy
        return DepressionHierarchy(
            nodes=h.nodes,
            leaf_label=h.leaf_label[r0:r1, c0:c1],
            filled_dem=h.filled_dem[r0:r1, c0:c1],
        )


def edge_flags(win, shape):
    """Which sides of the window are NOT the grid edge (top, bottom, left,
    right): a search that reaches such a side may have been clipped."""
    r0, r1, c0, c1 = win
    return (r0 > 0, r1 < shape[0], c0 > 0, c1 < shape[1])
