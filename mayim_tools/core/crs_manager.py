"""
Mayim Tools – CRS Manager
Utilities for CRS detection, validation, and transformation setup.
"""

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsMapLayer,
    QgsProject,
)


class CRSManager:
    """Static utility class for CRS operations."""

    @staticmethod
    def from_epsg(epsg_code: int) -> QgsCoordinateReferenceSystem:
        """
        Create a CRS from an EPSG code.

        :param epsg_code: Integer EPSG code (e.g., 4326, 32735)
        :returns: QgsCoordinateReferenceSystem
        """
        crs = QgsCoordinateReferenceSystem(f"EPSG:{epsg_code}")
        return crs

    @staticmethod
    def project_crs() -> QgsCoordinateReferenceSystem:
        """Return the current QGIS project CRS."""
        return QgsProject.instance().crs()

    @staticmethod
    def layer_crs(layer: QgsMapLayer) -> QgsCoordinateReferenceSystem:
        """
        Return the CRS of a given layer.

        :param layer: Any QgsMapLayer
        :returns: Layer CRS
        """
        return layer.crs()

    @staticmethod
    def needs_reprojection(
        source_crs: QgsCoordinateReferenceSystem,
        target_crs: QgsCoordinateReferenceSystem,
    ) -> bool:
        """
        Check if two CRS objects differ and reprojection is needed.

        :param source_crs: Source CRS
        :param target_crs: Target CRS
        :returns: True if they differ, False if they are the same
        """
        return source_crs != target_crs

    @staticmethod
    def get_transform(
        source_crs: QgsCoordinateReferenceSystem,
        target_crs: QgsCoordinateReferenceSystem,
    ) -> QgsCoordinateTransform:
        """
        Build a coordinate transform between two CRS objects.

        :param source_crs: Source CRS
        :param target_crs: Target CRS
        :returns: QgsCoordinateTransform ready to use
        """
        return QgsCoordinateTransform(source_crs, target_crs, QgsProject.instance())

    @staticmethod
    def is_geographic(crs: QgsCoordinateReferenceSystem) -> bool:
        """
        Check if a CRS is geographic (lat/lon) vs projected (metres).

        :param crs: CRS to check
        :returns: True if geographic, False if projected
        """
        return crs.isGeographic()
