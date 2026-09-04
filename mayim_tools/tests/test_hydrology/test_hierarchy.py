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
    pit_row: int = 2,
    pit_col: int = 2,
    touches_boundary: bool = False,
) -> DepressionNode:
    """Create a representative test depression node."""
    return DepressionNode(
        depression_id=depression_id,
        pit_row=pit_row,
        pit_col=pit_col,
        pit_elevation=pit_elevation,
        spill_elevation=spill_elevation,
        depth=spill_elevation - pit_elevation,
        area_cells=area_cells,
        area_map_units=float(area_cells),
        perimeter_cells=8,
        volume_estimate=10.0,
        touches_boundary=touches_boundary,
    )


class TestDepressionNode:
    """Tests for DepressionNode."""

    def test_node_creation(self):
        """A node stores its supplied attributes."""
        node = make_node(
            depression_id=1,
            pit_elevation=2.0,
            spill_elevation=7.0,
            area_cells=25,
        )

        assert node.depression_id == 1
        assert node.pit_elevation == 2.0
        assert node.spill_elevation == 7.0
        assert node.depth == 5.0
        assert node.area_cells == 25
        assert node.parent_id is None
        assert node.child_ids == []

    def test_root_and_leaf_detection(self):
        """A standalone node is both a root and a leaf."""
        node = make_node(1)

        assert node.is_root is True
        assert node.is_leaf is True

        node.parent_id = 99
        assert node.is_root is False

        node.parent_id = None
        node.child_ids.append(2)
        assert node.is_leaf is False

    def test_node_serialisation(self):
        """A node serialises to a dictionary with expected fields."""
        node = make_node(
            depression_id=4,
            pit_elevation=2.0,
            spill_elevation=8.0,
            area_cells=20,
        )

        data = node.to_dict()

        assert data["depression_id"] == 4
        assert data["pit_row"] == 2
        assert data["pit_col"] == 2
        assert data["pit_elevation"] == 2.0
        assert data["spill_elevation"] == 8.0
        assert data["depth"] == 6.0
        assert data["area_cells"] == 20
        assert data["parent_id"] is None
        assert data["child_ids"] == []
        assert data["is_root"] is True
        assert data["is_leaf"] is True

    def test_elongation_index_is_bounded(self):
        """The elongation index must remain between zero and one."""
        node = make_node(1)

        assert 0.0 <= node.elongation_index <= 1.0

    def test_boundary_flag_is_stored(self):
        """Boundary-connected status is stored on the node."""
        node = make_node(
            depression_id=1,
            touches_boundary=True,
        )

        assert node.touches_boundary is True


class TestDepressionHierarchy:
    """Tests for DepressionHierarchy."""

    def test_empty_hierarchy(self):
        """An empty hierarchy has no nodes or relationships."""
        hierarchy = DepressionHierarchy()

        assert len(hierarchy) == 0
        assert hierarchy.total_depressions == 0
        assert hierarchy.root_count == 0
        assert hierarchy.max_depth == 0
        assert hierarchy.roots() == []
        assert hierarchy.leaves() == []

    def test_adding_and_retrieving_nodes(self):
        """Nodes can be added and retrieved by depression ID."""
        hierarchy = DepressionHierarchy()
        node = make_node(1)

        hierarchy.add(node)

        assert len(hierarchy) == 1
        assert hierarchy.total_depressions == 1
        assert hierarchy.get(1) is node
        assert hierarchy.get(999) is None
        assert 1 in hierarchy
        assert 999 not in hierarchy

    def test_duplicate_id_is_rejected(self):
        """Two nodes cannot have the same depression ID."""
        hierarchy = DepressionHierarchy()

        hierarchy.add(make_node(1))

        with pytest.raises(ValueError, match="already exists"):
            hierarchy.add(make_node(1))

    def test_root_and_leaf_discovery(self):
        """A standalone node appears in both roots and leaves."""
        hierarchy = DepressionHierarchy()
        node = make_node(1)

        hierarchy.add(node)

        assert hierarchy.roots() == [node]
        assert hierarchy.leaves() == [node]

    def test_parent_child_relationship(self):
        """Setting a parent updates both parent and child nodes."""
        hierarchy = DepressionHierarchy()
        parent = make_node(1, area_cells=100)
        child = make_node(2, area_cells=10)

        hierarchy.add(parent)
        hierarchy.add(child)
        hierarchy.set_parent(
            child_id=2,
            parent_id=1,
        )

        assert child.parent_id == 1
        assert parent.child_ids == [2]
        assert child in hierarchy.children(1)
        assert child.is_root is False
        assert parent.is_root is True
        assert parent.is_leaf is False
        assert child.is_leaf is True

    def test_repeated_parent_assignment_is_not_duplicated(self):
        """The same child is not added twice to the parent's child list."""
        hierarchy = DepressionHierarchy()
        parent = make_node(1)
        child = make_node(2)

        hierarchy.add(parent)
        hierarchy.add(child)

        hierarchy.set_parent(2, 1)
        hierarchy.set_parent(2, 1)

        assert parent.child_ids == [2]

    def test_root_and_leaf_discovery_after_relationship(self):
        """Parent and child are correctly classified after linking."""
        hierarchy = DepressionHierarchy()
        parent = make_node(1, area_cells=100)
        child_a = make_node(2, area_cells=10)
        child_b = make_node(3, area_cells=20)

        hierarchy.add(parent)
        hierarchy.add(child_a)
        hierarchy.add(child_b)

        hierarchy.set_parent(2, 1)
        hierarchy.set_parent(3, 1)

        assert hierarchy.roots() == [parent]
        assert hierarchy.leaves() == [child_a, child_b]
        assert hierarchy.children(1) == [child_a, child_b]

    def test_ancestors_and_descendants(self):
        """Nested relationships can be traversed in both directions."""
        hierarchy = DepressionHierarchy()
        root = make_node(1, area_cells=100)
        child = make_node(2, area_cells=50)
        grandchild = make_node(3, area_cells=10)

        hierarchy.add(root)
        hierarchy.add(child)
        hierarchy.add(grandchild)

        hierarchy.set_parent(2, 1)
        hierarchy.set_parent(3, 2)

        assert hierarchy.ancestors(3) == [child, root]
        assert hierarchy.ancestors(2) == [root]
        assert hierarchy.ancestors(1) == []

        assert hierarchy.descendants(1) == [child, grandchild]
        assert hierarchy.descendants(2) == [grandchild]
        assert hierarchy.descendants(3) == []

    def test_hierarchy_depth(self):
        """Maximum depth reflects the deepest nested relationship."""
        hierarchy = DepressionHierarchy()
        root = make_node(1)
        child = make_node(2)
        grandchild = make_node(3)
        great_grandchild = make_node(4)

        for node in (root, child, grandchild, great_grandchild):
            hierarchy.add(node)

        hierarchy.set_parent(2, 1)
        hierarchy.set_parent(3, 2)
        hierarchy.set_parent(4, 3)

        assert hierarchy.max_depth == 3

    def test_missing_child_is_rejected(self):
        """A missing child ID raises KeyError."""
        hierarchy = DepressionHierarchy()
        hierarchy.add(make_node(1))

        with pytest.raises(KeyError):
            hierarchy.set_parent(
                child_id=999,
                parent_id=1,
            )

    def test_missing_parent_is_rejected(self):
        """A missing parent ID raises KeyError."""
        hierarchy = DepressionHierarchy()
        hierarchy.add(make_node(1))

        with pytest.raises(KeyError):
            hierarchy.set_parent(
                child_id=1,
                parent_id=999,
            )

    def test_self_parent_is_rejected(self):
        """A node cannot be its own parent."""
        hierarchy = DepressionHierarchy()
        hierarchy.add(make_node(1))

        with pytest.raises(ValueError, match="own parent"):
            hierarchy.set_parent(
                child_id=1,
                parent_id=1,
            )

    def test_cycle_prevention(self):
        """The hierarchy rejects relationships that create a cycle."""
        hierarchy = DepressionHierarchy()
        node_a = make_node(1)
        node_b = make_node(2)
        node_c = make_node(3)

        hierarchy.add(node_a)
        hierarchy.add(node_b)
        hierarchy.add(node_c)

        hierarchy.set_parent(2, 1)
        hierarchy.set_parent(3, 2)

        with pytest.raises(ValueError, match="cycle"):
            hierarchy.set_parent(
                child_id=1,
                parent_id=3,
            )

    def test_hierarchy_serialisation(self):
        """The hierarchy serialises to a summary and node dictionary."""
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
        assert data["nodes"]["2"]["parent_id"] == 1

    def test_sorted_serialisation(self):
        """Flat serialisation is deterministic by depression ID."""
        hierarchy = DepressionHierarchy()

        hierarchy.add(make_node(5))
        hierarchy.add(make_node(2))
        hierarchy.add(make_node(9))

        serialised = hierarchy.to_list()

        assert [item["depression_id"] for item in serialised] == [2, 5, 9]
