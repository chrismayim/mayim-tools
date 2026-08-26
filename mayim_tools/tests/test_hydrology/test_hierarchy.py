"""Tests for the native Mayim depression hierarchy.

These tests validate the in-house hierarchy structure only.
They do not use networkx, WhiteboxTools or RichDEM.
"""

import pytest

from mayim_tools.hydrology.depression.hierarchy import (
    DepressionHierarchy,
    DepressionNode,
)


def make_node(
    depression_id: int,
    pit_elevation: float = 1.0,
    spill_elevation: float = 5.0,
    area_cells: int = 10,
) -> DepressionNode:
    """Create a small test node."""
    return DepressionNode(
        depression_id=depression_id,
        pit_row=2,
        pit_col=2,
        pit_elevation=pit_elevation,
        spill_elevation=spill_elevation,
        depth=spill_elevation - pit_elevation,
        area_cells=area_cells,
        area_map_units=float(area_cells),
        perimeter_cells=8,
        volume_estimate=10.0,
        touches_boundary=False,
    )


class TestDepressionNode:
    """Tests for DepressionNode."""

    def test_node_properties(self):
        """A node reports its root and leaf state correctly."""
        node = make_node(1)

        assert node.depression_id == 1
        assert node.depth == 4.0
        assert node.is_root is True
        assert node.is_leaf is True
        assert node.parent_id is None
        assert node.child_ids == []

    def test_node_to_dict_contains_expected_fields(self):
        """Node serialisation contains the required fields."""
        node = make_node(1)
        data = node.to_dict()

        assert data["depression_id"] == 1
        assert data["pit_row"] == 2
        assert data["pit_col"] == 2
        assert data["pit_elevation"] == 1.0
        assert data["spill_elevation"] == 5.0
        assert data["depth"] == 4.0
        assert data["area_cells"] == 10
        assert data["parent_id"] is None
        assert data["child_ids"] == []
        assert data["is_root"] is True
        assert data["is_leaf"] is True

    def test_elongation_index_is_bounded(self):
        """The elongation index is between zero and one."""
        node = make_node(1)
        assert 0.0 <= node.elongation_index <= 1.0


class TestDepressionHierarchy:
    """Tests for DepressionHierarchy."""

    def test_empty_hierarchy(self):
        """An empty hierarchy has no nodes, roots or leaves."""
        hierarchy = DepressionHierarchy()

        assert len(hierarchy) == 0
        assert hierarchy.total_depressions == 0
        assert hierarchy.root_count == 0
        assert hierarchy.max_depth == 0
        assert hierarchy.roots() == []
        assert hierarchy.leaves() == []

    def test_add_and_get_node(self):
        """A node can be added and retrieved by ID."""
        hierarchy = DepressionHierarchy()
        node = make_node(1)

        hierarchy.add(node)

        assert len(hierarchy) == 1
        assert hierarchy.get(1) is node
        assert hierarchy.get(999) is None
        assert 1 in hierarchy
        assert 999 not in hierarchy

    def test_duplicate_node_id_is_rejected(self):
        """Duplicate depression IDs must not be accepted."""
        hierarchy = DepressionHierarchy()
        hierarchy.add(make_node(1))

        with pytest.raises(ValueError, match="already exists"):
            hierarchy.add(make_node(1))

    def test_single_node_is_root_and_leaf(self):
        """A single node is both a root and a leaf."""
        hierarchy = DepressionHierarchy()
        node = make_node(1)

        hierarchy.add(node)

        assert hierarchy.roots() == [node]
        assert hierarchy.leaves() == [node]
        assert hierarchy.children(1) == []
        assert hierarchy.ancestors(1) == []
        assert hierarchy.descendants(1) == []
        assert hierarchy.max_depth == 0

    def test_set_parent_creates_relationship(self):
        """Setting a parent updates both sides of the relationship."""
        hierarchy = DepressionHierarchy()
        parent = make_node(1, area_cells=100)
        child = make_node(2, area_cells=10)

        hierarchy.add(parent)
        hierarchy.add(child)
        hierarchy.set_parent(child_id=2, parent_id=1)

        assert child.parent_id == 1
        assert parent.child_ids == [2]
        assert child.is_root is False
        assert child.is_leaf is True
        assert parent.is_root is True
        assert parent.is_leaf is False

    def test_children_and_roots(self):
        """The hierarchy returns the correct roots and children."""
        hierarchy = DepressionHierarchy()
        parent = make_node(1, area_cells=100)
        child_a = make_node(2, area_cells=10)
        child_b = make_node(3, area_cells=20)

        for node in (parent, child_a, child_b):
            hierarchy.add(node)

        hierarchy.set_parent(2, 1)
        hierarchy.set_parent(3, 1)

        assert hierarchy.roots() == [parent]
        assert hierarchy.children(1) == [child_a, child_b]
        assert hierarchy.leaves() == [child_a, child_b]

    def test_ancestors_and_descendants(self):
        """Nested relationships can be traversed."""
        hierarchy = DepressionHierarchy()
        root = make_node(1, area_cells=100)
        child = make_node(2, area_cells=50)
        grandchild = make_node(3, area_cells=10)

        for node in (root, child, grandchild):
            hierarchy.add(node)

        hierarchy.set_parent(2, 1)
        hierarchy.set_parent(3, 2)

        assert hierarchy.ancestors(3) == [child, root]
        assert hierarchy.descendants(1) == [child, grandchild]
        assert hierarchy.descendants(2) == [grandchild]
        assert hierarchy.max_depth == 2

    def test_set_parent_missing_child_is_rejected(self):
        """A missing child ID must raise KeyError."""
        hierarchy = DepressionHierarchy()
        hierarchy.add(make_node(1))

        with pytest.raises(KeyError):
            hierarchy.set_parent(child_id=999, parent_id=1)

    def test_set_parent_missing_parent_is_rejected(self):
        """A missing parent ID must raise KeyError."""
        hierarchy = DepressionHierarchy()
        hierarchy.add(make_node(1))

        with pytest.raises(KeyError):
            hierarchy.set_parent(child_id=1, parent_id=999)

    def test_to_dict_contains_summary_and_nodes(self):
        """The complete hierarchy can be serialised."""
        hierarchy = DepressionHierarchy()
        parent = make_node(1, area_cells=100)
        child = make_node(2, area_cells=10)

        hierarchy.add(parent)
        hierarchy.add(child)
        hierarchy.set_parent(2, 1)

        data = hierarchy.to_dict()

        assert data["summary"]["total_depressions"] == 2
        assert data["summary"]["root_count"] == 1
        assert data["summary"]["max_depth"] == 1
        assert "1" in data["nodes"]
        assert "2" in data["nodes"]

    def test_to_list_is_sorted_by_depression_id(self):
        """Flat serialisation is deterministic."""
        hierarchy = DepressionHierarchy()
        hierarchy.add(make_node(5))
        hierarchy.add(make_node(2))
        hierarchy.add(make_node(9))

        result = hierarchy.to_list()

        assert [item["depression_id"] for item in result] == [2, 5, 9]
