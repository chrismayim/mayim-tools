"""
Tests for native Stage 7 hydrography topology preparation.

These tests use Shapely only as generic geometry infrastructure.
They do not use WhiteboxTools, RichDEM or TauDEM.
"""

import pytest
from shapely.geometry import LineString, MultiLineString, Point

from mayim_tools.hydrology.hydrography.topology import (
    prepare_hydrography_topology,
)

DEM_BOUNDS = (0.0, 0.0, 10.0, 10.0)


class TestPrepareHydrographyTopology:
    """Tests for prepare_hydrography_topology()."""

    def test_single_line_creates_one_feature_and_one_component(self):
        """A single valid line is represented as one network component."""
        geometries = [
            LineString([(1.0, 1.0), (9.0, 9.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["valid"] is True
        assert result["feature_count"] == 1
        assert result["valid_feature_count"] == 1
        assert result["invalid_feature_count"] == 0
        assert result["component_count"] == 1
        assert result["components"][0]["feature_ids"] == [1]
        assert result["edge_count"] == 1

    def test_connected_lines_form_one_component(self):
        """Lines sharing an endpoint form one component."""
        geometries = [
            LineString([(0.0, 5.0), (5.0, 5.0)]),
            LineString([(5.0, 5.0), (10.0, 5.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["valid"] is True
        assert result["component_count"] == 1
        assert result["components"][0]["feature_ids"] == [1, 2]
        assert result["node_count"] == 3

    def test_disconnected_lines_form_separate_components(self):
        """Lines without shared endpoints remain separate."""
        geometries = [
            LineString([(0.0, 1.0), (1.0, 1.0)]),
            LineString([(8.0, 8.0), (9.0, 9.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["valid"] is True
        assert result["component_count"] == 2
        assert result["components"][0]["feature_ids"] == [1]
        assert result["components"][1]["feature_ids"] == [2]

    def test_endpoint_tolerance_connects_nearby_lines(self):
        """A positive endpoint tolerance connects nearby endpoints."""
        geometries = [
            LineString([(0.0, 5.0), (5.0, 5.0)]),
            LineString([(5.001, 5.0), (10.0, 5.0)]),
        ]

        without_tolerance = prepare_hydrography_topology(
            geometries=geometries,
            endpoint_tolerance=0.0,
        )

        with_tolerance = prepare_hydrography_topology(
            geometries=geometries,
            endpoint_tolerance=0.01,
        )

        assert without_tolerance["component_count"] == 2
        assert with_tolerance["component_count"] == 1

    def test_crossing_lines_are_reported_as_crossings(self):
        """Crossing lines are reported separately from endpoint connections."""
        geometries = [
            LineString([(2.0, 5.0), (8.0, 5.0)]),
            LineString([(5.0, 2.0), (5.0, 8.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["valid"] is True
        assert len(result["intersections"]) == 1
        assert result["intersections"][0]["intersection_type"] == ("crossing")
        assert result["topology_conflicts"]

    def test_crossing_lines_are_not_automatically_connected(self):
        """Interior crossings do not create endpoint connectivity."""
        geometries = [
            LineString([(2.0, 5.0), (8.0, 5.0)]),
            LineString([(5.0, 2.0), (5.0, 8.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["component_count"] == 2

    def test_overlapping_lines_are_reported(self):
        """Overlapping lines are reported as overlap intersections."""
        geometries = [
            LineString([(1.0, 1.0), (8.0, 8.0)]),
            LineString([(3.0, 3.0), (6.0, 6.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert len(result["intersections"]) == 1
        assert result["intersections"][0]["intersection_type"] == ("overlap")

    def test_duplicate_segments_are_reported(self):
        """Equivalent duplicate segments are identified."""
        geometries = [
            LineString([(1.0, 1.0), (8.0, 8.0)]),
            LineString([(1.0, 1.0), (8.0, 8.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["duplicate_segments"] == [[1, 2]]
        assert result["topology_conflicts"]

    def test_reversed_duplicate_segments_are_reported(self):
        """Reversed copies of the same line are duplicates."""
        geometries = [
            LineString([(1.0, 1.0), (8.0, 8.0)]),
            LineString([(8.0, 8.0), (1.0, 1.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["duplicate_segments"] == [[1, 2]]

    def test_multilinestring_is_supported(self):
        """A valid MultiLineString is accepted."""
        geometries = [
            MultiLineString(
                [
                    [(1.0, 1.0), (2.0, 2.0)],
                    [(4.0, 4.0), (5.0, 5.0)],
                ]
            ),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["valid"] is True
        assert result["valid_feature_count"] == 1
        assert result["edge_count"] == 1
        assert len(result["endpoints"]) == 4

    def test_non_line_geometry_is_invalid(self):
        """Point geometry is rejected."""
        geometries = [
            Point(5.0, 5.0),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["valid"] is False
        assert result["invalid_feature_count"] == 1
        assert result["errors"]

    def test_null_geometry_is_invalid(self):
        """Null geometry is rejected."""
        result = prepare_hydrography_topology(
            geometries=[None],
        )

        assert result["valid"] is False
        assert result["invalid_feature_count"] == 1
        assert result["errors"]

    def test_empty_geometry_collection_is_invalid(self):
        """No geometries produces an invalid result and warning."""
        result = prepare_hydrography_topology(
            geometries=[],
        )

        assert result["valid"] is False
        assert result["feature_count"] == 0
        assert result["component_count"] == 0
        assert result["warnings"]

    def test_feature_ids_are_preserved(self):
        """Supplied feature identifiers appear in the topology result."""
        geometries = [
            LineString([(1.0, 1.0), (2.0, 2.0)]),
            LineString([(2.0, 2.0), (3.0, 3.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
            feature_ids=["main", "tributary"],
        )

        assert result["valid"] is True
        assert result["components"][0]["feature_ids"] == [
            "main",
            "tributary",
        ]

    def test_stream_order_and_upstream_area_create_priorities(self):
        """Higher-order and larger-area features receive higher priority."""
        geometries = [
            LineString([(1.0, 1.0), (2.0, 2.0)]),
            LineString([(3.0, 3.0), (4.0, 4.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
            feature_ids=["minor", "major"],
            stream_order={
                "minor": 1,
                "major": 3,
            },
            upstream_area={
                "minor": 10.0,
                "major": 100.0,
            },
        )

        priorities = result["feature_priorities"]

        assert priorities["major"]["priority_rank"] == 1
        assert priorities["minor"]["priority_rank"] == 2
        assert priorities["major"]["stream_order"] == 3.0
        assert priorities["major"]["upstream_area"] == 100.0

    def test_missing_priority_attributes_do_not_crash(self):
        """Missing stream attributes receive zero priority values."""
        geometries = [
            LineString([(1.0, 1.0), (2.0, 2.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        priority = result["feature_priorities"]["1"]

        assert priority["stream_order"] == 0.0
        assert priority["upstream_area"] == 0.0
        assert priority["priority_rank"] == 1

    def test_invalid_endpoint_tolerance_is_rejected(self):
        """Negative or non-finite tolerance is rejected."""
        geometries = [
            LineString([(1.0, 1.0), (2.0, 2.0)]),
        ]

        with pytest.raises(ValueError, match="endpoint_tolerance"):
            prepare_hydrography_topology(
                geometries=geometries,
                endpoint_tolerance=-1.0,
            )

    def test_mismatched_feature_ids_are_rejected(self):
        """Feature IDs must match the geometry count."""
        geometries = [
            LineString([(1.0, 1.0), (2.0, 2.0)]),
            LineString([(3.0, 3.0), (4.0, 4.0)]),
        ]

        with pytest.raises(ValueError, match="one identifier"):
            prepare_hydrography_topology(
                geometries=geometries,
                feature_ids=["only-one"],
            )

    def test_results_are_deterministic(self):
        """Repeated topology preparation produces identical results."""
        geometries = [
            LineString([(0.0, 5.0), (5.0, 5.0)]),
            LineString([(5.0, 5.0), (10.0, 5.0)]),
        ]

        first = prepare_hydrography_topology(
            geometries=geometries,
            endpoint_tolerance=0.0,
        )

        second = prepare_hydrography_topology(
            geometries=geometries,
            endpoint_tolerance=0.0,
        )

        assert first == second

    def test_dem_bounds_argument_is_not_required(self):
        """Topology preparation is independent of DEM spatial bounds."""
        geometries = [
            LineString([(100.0, 100.0), (200.0, 200.0)]),
        ]

        result = prepare_hydrography_topology(
            geometries=geometries,
        )

        assert result["valid"] is True
        assert result["feature_count"] == 1
