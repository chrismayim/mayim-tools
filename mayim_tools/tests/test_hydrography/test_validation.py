"""
Tests for native Stage 7 hydrography validation.

No WhiteboxTools, RichDEM or TauDEM is used.
"""

from types import SimpleNamespace

from shapely.geometry import LineString, Point

from mayim_tools.hydrology.hydrography.validation import (
    validate_hydrography,
)

DEM_BOUNDS = (0.0, 0.0, 10.0, 10.0)


class TestValidateHydrography:
    """Tests for validate_hydrography()."""

    def test_valid_line_inside_dem_extent(self):
        """A valid line with matching CRS is accepted."""
        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is True
        assert result["geometry_count"] == 1
        assert result["invalid_geometry_count"] == 0
        assert result["intersecting_dem_count"] == 1
        assert result["outside_dem_count"] == 0
        assert result["crs_match"] is True
        assert result["errors"] == []

    def test_missing_hydrography_is_invalid(self):
        """Missing hydrography is rejected."""
        result = validate_hydrography(
            geometries=None,
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert result["geometry_count"] == 0
        assert result["errors"]

    def test_empty_geometry_is_invalid(self):
        """Empty geometry is rejected."""
        result = validate_hydrography(
            geometries=[
                LineString(),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert result["empty_geometry_count"] == 1
        assert result["invalid_geometry_count"] == 1

    def test_invalid_geometry_is_invalid(self):
        """An invalid geometry is reported."""
        invalid_line = LineString([(0.0, 0.0), (0.0, 0.0)])

        assert invalid_line.is_valid is False

        result = validate_hydrography(
            geometries=[invalid_line],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert result["invalid_geometry_count"] == 1
        assert any("invalid" in error.lower() for error in result["errors"])

    def test_non_line_geometry_is_invalid(self):
        """Point geometry is rejected."""
        result = validate_hydrography(
            geometries=[
                Point(5.0, 5.0),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert result["non_line_count"] == 1

    def test_mismatched_crs_is_invalid(self):
        """Different CRS values are rejected."""
        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:4326",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert result["crs_match"] is False
        assert any("CRS" in error for error in result["errors"])

    def test_feature_outside_dem_is_reported(self):
        """A valid line outside the DEM extent is warned about."""
        result = validate_hydrography(
            geometries=[
                LineString([(20.0, 20.0), (30.0, 30.0)]),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is True
        assert result["outside_dem_count"] == 1
        assert result["intersecting_dem_count"] == 0
        assert result["warnings"]

    def test_mixed_inside_and_outside_features_are_counted(self):
        """Inside and outside features are counted separately."""
        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
                LineString([(20.0, 20.0), (30.0, 30.0)]),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is True
        assert result["geometry_count"] == 2
        assert result["intersecting_dem_count"] == 1
        assert result["outside_dem_count"] == 1

    def test_missing_dem_crs_is_invalid(self):
        """Missing DEM CRS is rejected."""
        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
            ],
            dem_crs=None,
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert any("DEM CRS" in error for error in result["errors"])

    def test_missing_hydrography_crs_is_invalid(self):
        """Missing hydrography CRS is rejected."""
        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs=None,
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert any("Hydrography CRS" in error for error in result["errors"])

    def test_object_bounds_are_supported(self):
        """Bounds objects with named attributes are supported."""
        bounds = SimpleNamespace(
            left=0.0,
            bottom=0.0,
            right=10.0,
            top=10.0,
        )

        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=bounds,
        )

        assert result["valid"] is True
        assert result["intersecting_dem_count"] == 1
        assert result["outside_dem_count"] == 0

    def test_invalid_bounds_are_reported(self):
        """Malformed DEM bounds are reported as an error."""
        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=(0.0, 0.0, 10.0),
        )

        assert result["valid"] is False
        assert any("bounds" in error.lower() for error in result["errors"])

    def test_reversed_bounds_are_invalid(self):
        """Bounds with inverted coordinates are rejected."""
        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=(10.0, 10.0, 0.0, 0.0),
        )

        assert result["valid"] is False
        assert any("bounds" in error.lower() for error in result["errors"])

    def test_non_iterable_geometries_are_invalid(self):
        """A non-iterable geometry input is rejected."""
        result = validate_hydrography(
            geometries=123,
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert result["geometry_count"] == 0
        assert result["errors"]

    def test_empty_geometry_collection_is_warned(self):
        """An empty geometry collection produces a warning."""
        result = validate_hydrography(
            geometries=[],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert result["geometry_count"] == 0
        assert any(
            "no hydrography" in warning.lower() for warning in result["warnings"]
        )

    def test_multipoint_is_rejected_as_non_line_geometry(self):
        """Non-line geometry types are not accepted."""
        result = validate_hydrography(
            geometries=[
                Point(1.0, 1.0),
                Point(2.0, 2.0),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is False
        assert result["non_line_count"] == 2

    def test_crs_objects_with_authid_are_supported(self):
        """CRS-like objects exposing authid() can be compared."""
        dem_crs = SimpleNamespace(
            authid=lambda: "EPSG:32735",
        )
        hydrography_crs = SimpleNamespace(
            authid=lambda: "EPSG:32735",
        )

        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
            ],
            dem_crs=dem_crs,
            hydrography_crs=hydrography_crs,
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is True
        assert result["crs_match"] is True

    def test_lines_touching_dem_boundary_are_intersecting(self):
        """Lines touching the DEM boundary are counted as intersecting."""
        result = validate_hydrography(
            geometries=[
                LineString([(-1.0, 5.0), (0.0, 5.0)]),
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["valid"] is True
        assert result["intersecting_dem_count"] == 1
        assert result["outside_dem_count"] == 0

    def test_validation_counts_are_consistent(self):
        """Feature counts are internally consistent."""
        result = validate_hydrography(
            geometries=[
                LineString([(1.0, 1.0), (9.0, 9.0)]),
                LineString([(20.0, 20.0), (30.0, 30.0)]),
                Point(5.0, 5.0),
                None,
            ],
            dem_crs="EPSG:32735",
            hydrography_crs="EPSG:32735",
            dem_bounds=DEM_BOUNDS,
        )

        assert result["geometry_count"] == 4
        assert result["intersecting_dem_count"] == 1
        assert result["outside_dem_count"] == 1
        assert result["non_line_count"] == 1
        assert result["invalid_geometry_count"] == 1
        assert (
            result["intersecting_dem_count"]
            + result["outside_dem_count"]
            + result["invalid_geometry_count"]
            + result["non_line_count"]
            == result["geometry_count"]
        )
