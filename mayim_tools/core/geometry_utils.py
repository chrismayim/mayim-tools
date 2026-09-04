"""
Mayim Tools – Geometry Utilities
Shared helpers for geometry validation, transformation, and inspection.
"""

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
)


class GeometryUtils:
    """Static utility class for common geometry operations."""

    @staticmethod
    def is_valid(geometry: QgsGeometry) -> bool:
        """
        Check if a geometry is valid and non-empty.

        :param geometry: QgsGeometry to validate
        :returns: True if valid, False otherwise
        """
        return (
            geometry is not None
            and not geometry.isNull()
            and not geometry.isEmpty()
            and geometry.isGeosValid()
        )

    @staticmethod
    def fix_geometry(geometry: QgsGeometry) -> QgsGeometry:
        """
        Attempt to fix an invalid geometry using GEOS buffer(0) trick.

        :param geometry: Potentially invalid QgsGeometry
        :returns: Fixed QgsGeometry
        """
        if not geometry.isGeosValid():
            geometry = geometry.buffer(0, 5)
        return geometry

    @staticmethod
    def reproject(
        geometry: QgsGeometry,
        source_crs: QgsCoordinateReferenceSystem,
        target_crs: QgsCoordinateReferenceSystem,
    ) -> QgsGeometry:
        """
        Reproject a geometry from one CRS to another.

        :param geometry: Source geometry
        :param source_crs: Source coordinate reference system
        :param target_crs: Target coordinate reference system
        :returns: Reprojected QgsGeometry
        """
        transform = QgsCoordinateTransform(
            source_crs, target_crs, QgsProject.instance()
        )
        geometry.transform(transform)
        return geometry

    @staticmethod
    def area_sq_meters(geometry: QgsGeometry) -> float:
        """
        Calculate the area of a polygon geometry in square metres.

        :param geometry: Polygon QgsGeometry
        :returns: Area in square metres
        """
        return geometry.area()

    @staticmethod
    def length_meters(geometry: QgsGeometry) -> float:
        """
        Calculate the length of a line geometry in metres.

        :param geometry: Line QgsGeometry
        :returns: Length in metres
        """
        return geometry.length()
