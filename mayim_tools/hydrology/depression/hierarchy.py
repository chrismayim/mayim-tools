"""
Mayim Tools - Depression Hierarchy
=====================================

Implements an in-house nested-graph structure representing the
parent-child relationships among depressions in a DEM.

The hierarchy is constructed following the approach described in:

    Barnes, R., Callaghan, K. L., and Wickert, A. D. (2020).
    Computing water flow through complex landscapes - Part 2:
    Finding hierarchies in depressions and morphological
    segmentations.
    Earth Surface Dynamics, 8(2), 431-445.

IP Status
---------
Original Mayim implementation.
Uses an in-house nested-graph structure — NOT networkx — so that
this core IP component carries no external graph-library dependency,
consistent with Section 6.4 of the Mayim Tools Research Paper Rev 1.
Written solely from the published paper listed above.
No WhiteboxTools, RichDEM or any other third-party hydrological
source was consulted by the implementing engineer.
Runtime dependencies: none beyond Python standard library and numpy.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DepressionNode:
    """
    A single node in the Mayim depression hierarchy graph.

    Each node represents one depression, identified by its unique
    integer ID assigned during the Priority-Flood traversal in
    detection.py.

    Attributes
    ----------
    depression_id : int
        Unique integer identifier (1-indexed).
    pit_row : int
        Row index of the pit cell (lowest point).
    pit_col : int
        Column index of the pit cell.
    pit_elevation : float
        Elevation of the pit cell.
    spill_elevation : float
        Elevation at which water escapes this depression into an
        adjacent drainage area or a parent depression.
    depth : float
        Vertical distance from pit to spill.
        depth = spill_elevation - pit_elevation.
    area_cells : int
        Number of raster cells belonging to this depression.
    area_map_units : float
        Planimetric area in CRS map units squared.
    perimeter_cells : int
        Number of cells on the outer boundary of the depression.
    volume_estimate : float
        Approximate volume of the depression in map units cubed.
        Computed as the sum of (spill_elevation - cell_elevation)
        for all cells in the depression footprint, multiplied by
        the cell area.
    touches_boundary : bool
        True if any cell of this depression touches the DEM edge.
        Boundary-connected depressions cannot have a confirmed
        spill point within the DEM and must be treated with caution.
    parent_id : int or None
        ID of the parent depression in the hierarchy.
        None if this is a top-level depression.
    child_ids : list[int]
        IDs of all immediate child depressions nested within this
        depression.
    """

    depression_id:    int
    pit_row:          int
    pit_col:          int
    pit_elevation:    float
    spill_elevation:  float
    depth:            float
    area_cells:       int
    area_map_units:   float
    perimeter_cells:  int
    volume_estimate:  float
    touches_boundary: bool
    parent_id:        int | None       = None
    child_ids:        list[int]        = field(default_factory=list)

    # ── Derived properties ─────────────────────────────────────────── #

    @property
    def is_root(self) -> bool:
        """True if this depression has no parent."""
        return self.parent_id is None

    @property
    def is_leaf(self) -> bool:
        """True if this depression has no children."""
        return len(self.child_ids) == 0

    @property
    def elongation_index(self) -> float:
        """
        Simple shape-regularity proxy.

        Defined here as the ratio of area_cells to the square of
        perimeter_cells, normalised to [0, 1].

        A value close to 1 indicates a compact, roughly circular
        depression. A value close to 0 indicates a long, narrow
        or irregular depression.

        A long, narrow, linear depression aligned with a road or
        watercourse is characteristic of a culvert, bridge or
        embankment artifact, consistent with the classification
        criteria in Section 5.3 (Stage 4) of the research paper.
        """
        if self.perimeter_cells <= 0:
            return 0.0
        ratio = (4.0 * self.area_cells) / max(
            self.perimeter_cells ** 2, 1
        )
        return float(min(ratio, 1.0))

    def to_dict(self) -> dict:
        """
        Serialise this node to a plain dictionary.

        Used for JSON export by the hierarchy and provenance systems.

        :returns: Dictionary representation of this node.
        """
        return {
            "depression_id":    self.depression_id,
            "pit_row":          self.pit_row,
            "pit_col":          self.pit_col,
            "pit_elevation":    self.pit_elevation,
            "spill_elevation":  self.spill_elevation,
            "depth":            self.depth,
            "area_cells":       self.area_cells,
            "area_map_units":   self.area_map_units,
            "perimeter_cells":  self.perimeter_cells,
            "volume_estimate":  self.volume_estimate,
            "touches_boundary": self.touches_boundary,
            "elongation_index": self.elongation_index,
            "parent_id":        self.parent_id,
            "child_ids":        self.child_ids,
            "is_root":          self.is_root,
            "is_leaf":          self.is_leaf,
        }


class DepressionHierarchy:
    """
    In-house nested-graph structure representing the parent-child
    relationships among depressions in a DEM.

    This is a deliberate, proprietary implementation — NOT a wrapper
    around networkx or any other graph library — consistent with
    Section 6.4 of the Mayim Tools Research Paper Rev 1 (2026):

        "DepressionScope itself builds via an in-house nested-graph
         structure, not networkx, so that this core IP component
         carries no external graph-library dependency either."

    The graph is a forest (collection of trees). Each tree represents
    a meta-depression and its nested sub-depressions. Roots are
    top-level depressions with no parent.

    References
    ----------
    Barnes, R., Callaghan, K. L., and Wickert, A. D. (2020).
    Computing water flow through complex landscapes - Part 2:
    Finding hierarchies in depressions and morphological
    segmentations. Earth Surface Dynamics, 8(2), 431-445.
    """

    def __init__(self) -> None:
        """Initialise an empty hierarchy."""
        self._nodes: dict[int, DepressionNode] = {}

    # ── Node management ────────────────────────────────────────────── #

    def add(self, node: DepressionNode) -> None:
        """
        Add a DepressionNode to the hierarchy.

        :param node: DepressionNode to add.
        :raises ValueError: If a node with the same ID already exists.
        """
        if node.depression_id in self._nodes:
            raise ValueError(
                f"Depression ID {node.depression_id} already exists "
                f"in the hierarchy."
            )
        self._nodes[node.depression_id] = node

    def get(self, depression_id: int) -> DepressionNode | None:
        """
        Retrieve a node by depression ID.

        :param depression_id: Integer depression ID.
        :returns: DepressionNode or None if not found.
        """
        return self._nodes.get(depression_id)

    def set_parent(
        self,
        child_id: int,
        parent_id: int,
    ) -> None:
        """
        Establish a parent-child relationship.

        :param child_id: ID of the child depression.
        :param parent_id: ID of the parent depression.
        :raises KeyError: If either ID does not exist.
        """
        child  = self._nodes[child_id]
        parent = self._nodes[parent_id]

        child.parent_id = parent_id

        if child_id not in parent.child_ids:
            parent.child_ids.append(child_id)

    # ── Traversal ──────────────────────────────────────────────────── #

    def roots(self) -> list[DepressionNode]:
        """
        Return all root nodes (depressions with no parent).

        :returns: List of root DepressionNodes.
        """
        return [
            node for node in self._nodes.values()
            if node.is_root
        ]

    def leaves(self) -> list[DepressionNode]:
        """
        Return all leaf nodes (depressions with no children).

        :returns: List of leaf DepressionNodes.
        """
        return [
            node for node in self._nodes.values()
            if node.is_leaf
        ]

    def children(self, depression_id: int) -> list[DepressionNode]:
        """
        Return the immediate children of a depression.

        :param depression_id: Integer depression ID.
        :returns: List of child DepressionNodes.
        """
        node = self._nodes.get(depression_id)
        if node is None:
            return []
        return [
            self._nodes[cid]
            for cid in node.child_ids
            if cid in self._nodes
        ]

    def ancestors(self, depression_id: int) -> list[DepressionNode]:
        """
        Return all ancestors of a depression in order from
        immediate parent to root.

        :param depression_id: Integer depression ID.
        :returns: List of ancestor DepressionNodes.
        """
        result = []
        current = self._nodes.get(depression_id)
        while current is not None and current.parent_id is not None:
            parent = self._nodes.get(current.parent_id)
            if parent is None:
                break
            result.append(parent)
            current = parent
        return result

    def descendants(self, depression_id: int) -> list[DepressionNode]:
        """
        Return all descendants of a depression (all nested children,
        grandchildren, etc.) using breadth-first traversal.

        :param depression_id: Integer depression ID.
        :returns: List of descendant DepressionNodes.
        """
        result   = []
        frontier = list(self.children(depression_id))
        while frontier:
            node = frontier.pop(0)
            result.append(node)
            frontier.extend(self.children(node.depression_id))
        return result

    def iter_all(self) -> Iterator[DepressionNode]:
        """
        Iterate over all nodes in the hierarchy.

        :returns: Iterator of DepressionNode instances.
        """
        return iter(self._nodes.values())

    # ── Statistics ─────────────────────────────────────────────────── #

    def __len__(self) -> int:
        """Return the total number of depressions."""
        return len(self._nodes)

    def __contains__(self, depression_id: int) -> bool:
        """Return True if the depression ID exists."""
        return depression_id in self._nodes

    @property
    def total_depressions(self) -> int:
        """Total number of depressions in the hierarchy."""
        return len(self._nodes)

    @property
    def root_count(self) -> int:
        """Number of top-level (root) depressions."""
        return len(self.roots())

    @property
    def max_depth(self) -> int:
        """
        Maximum nesting depth of the hierarchy.

        A flat hierarchy (all roots, no children) has depth 0.
        A hierarchy with one level of children has depth 1, etc.
        """
        if not self._nodes:
            return 0
        depths = []
        for node in self._nodes.values():
            depth = len(self.ancestors(node.depression_id))
            depths.append(depth)
        return max(depths) if depths else 0

    # ── Serialisation ──────────────────────────────────────────────── #

    def to_dict(self) -> dict:
        """
        Serialise the full hierarchy to a plain dictionary.

        Suitable for JSON export as the depression hierarchy output
        file.

        :returns: Dictionary with summary statistics and all nodes.
        """
        return {
            "summary": {
                "total_depressions": self.total_depressions,
                "root_count":        self.root_count,
                "max_depth":         self.max_depth,
            },
            "nodes": {
                str(did): node.to_dict()
                for did, node in self._nodes.items()
            },
        }

    def to_list(self) -> list[dict]:
        """
        Return all nodes as a flat list of dictionaries.

        Used when writing the depression inventory to GeoPackage.

        :returns: List of node dictionaries.
        """
        return [
            node.to_dict()
            for node in sorted(
                self._nodes.values(),
                key=lambda n: n.depression_id,
            )
        ]


def build_hierarchy(
    dem: np.ndarray,
    depression_ids: np.ndarray,
    pit_cells: np.ndarray,
    spill_points: dict[int, float],
    nodata: float,
    cell_size: float,
) -> DepressionHierarchy:
    """
    Construct the depression hierarchy from detection outputs.

    For each depression, computes geometric features and establishes
    parent-child relationships based on spill connectivity.

    A depression A is the parent of depression B if the spill point
    of B leads into A — that is, B must fill to its spill elevation
    before it can drain, and when it does, it drains into A.

    This follows the nested-basin concept of Barnes, Callaghan and
    Wickert (2020), where sub-depressions merge into meta-depressions
    as the water level rises.

    Parameters
    ----------
    dem : np.ndarray
        2-D float64 elevation array.
    depression_ids : np.ndarray
        Depression ID array from detect_depressions().
    pit_cells : np.ndarray
        Pit-cell boolean array from detect_depressions().
    spill_points : dict[int, float]
        Spill elevation per depression from identify_spill_points().
    nodata : float
        Sentinel value for invalid cells.
    cell_size : float
        Mean cell size in CRS map units (used for area and volume).

    Returns
    -------
    DepressionHierarchy
        Populated hierarchy with all depressions as nodes.

    References
    ----------
    Barnes, R., Callaghan, K. L., and Wickert, A. D. (2020).
    Computing water flow through complex landscapes - Part 2:
    Finding hierarchies in depressions and morphological
    segmentations. Earth Surface Dynamics, 8(2), 431-445.
    """
    rows, cols  = dem.shape
    cell_area   = cell_size ** 2
    hierarchy   = DepressionHierarchy()

    unique_ids = [
        int(did) for did in np.unique(depression_ids)
        if did > 0
    ]

    # ── Build one node per depression ─────────────────────────────── #
    for did in unique_ids:
        mask = depression_ids == did

        # Pit location
        pit_mask = pit_cells & mask
        pit_locs = np.argwhere(pit_mask)

        if len(pit_locs) == 0:
            # Fallback: use the lowest cell in the depression
            masked_dem = np.where(mask, dem, np.inf)
            pit_loc = np.unravel_index(
                np.argmin(masked_dem), dem.shape
            )
        else:
            pit_loc = tuple(pit_locs[0])

        pit_row = int(pit_loc[0])
        pit_col = int(pit_loc[1])
        pit_elev = float(dem[pit_row, pit_col])

        spill_elev = float(spill_points.get(did, pit_elev))
        depth      = max(0.0, spill_elev - pit_elev)

        # Area and volume
        area_cells = int(np.sum(mask))
        area_map   = area_cells * cell_area

        # Volume estimate: sum of (spill - cell_elev) for all cells
        cells_in = dem[mask]
        vol_est  = float(
            np.sum(np.maximum(0.0, spill_elev - cells_in)) * cell_area
        )

        # Perimeter: cells in the depression that have at least one
        # neighbour outside the depression
        perimeter = 0
        dep_rows, dep_cols = np.where(mask)
        for r, c in zip(dep_rows.tolist(), dep_cols.tolist()):
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if (
                    not (0 <= nr < rows and 0 <= nc < cols)
                    or depression_ids[nr, nc] != did
                ):
                    perimeter += 1
                    break

        # Boundary contact
        touches = bool(
            np.any(dep_rows == 0)
            or np.any(dep_rows == rows - 1)
            or np.any(dep_cols == 0)
            or np.any(dep_cols == cols - 1)
        )

        node = DepressionNode(
            depression_id=did,
            pit_row=pit_row,
            pit_col=pit_col,
            pit_elevation=pit_elev,
            spill_elevation=spill_elev,
            depth=depth,
            area_cells=area_cells,
            area_map_units=area_map,
            perimeter_cells=perimeter,
            volume_estimate=vol_est,
            touches_boundary=touches,
        )

        hierarchy.add(node)

    # ── Establish parent-child relationships ───────────────────────── #
    #
    # A depression is considered nested within the smallest other
    # depression whose spill elevation is higher than the child's spill
    # elevation and whose footprint contains the child's pit cell.
    #
    # This relationship is established from the depression footprints
    # rather than from an external graph library.

    nodes = list(hierarchy.iter_all())

    for child in nodes:
        candidate_parents = []

        for parent in nodes:
            if parent.depression_id == child.depression_id:
                continue

            if parent.spill_elevation <= child.spill_elevation:
                continue

            parent_mask = depression_ids == parent.depression_id

            if parent_mask[child.pit_row, child.pit_col]:
                candidate_parents.append(parent)

        if candidate_parents:
            immediate_parent = min(
                candidate_parents,
                key=lambda candidate: candidate.area_cells,
            )

            hierarchy.set_parent(
                child_id=child.depression_id,
                parent_id=immediate_parent.depression_id,
            )

    # Validate that the hierarchy contains no cycles.
    for node in hierarchy.iter_all():
        visited_ids = set()
        current = node

        while current.parent_id is not None:
            if current.parent_id in visited_ids:
                raise ValueError(
                    "Cycle detected in depression hierarchy involving "
                    f"depression {node.depression_id}."
                )

            visited_ids.add(current.parent_id)
            parent = hierarchy.get(current.parent_id)

            if parent is None:
                raise ValueError(
                    f"Depression {current.depression_id} references "
                    f"missing parent {current.parent_id}."
                )

            current = parent

    return hierarchy


