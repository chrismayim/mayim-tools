# -*- coding: utf-8 -*-
"""
Mayim Tools – Layer Utilities
Shared helpers for layer inspection, access, and manipulation.
"""

from typing import Optional

from qgis.core import (
    QgsMapLayer,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsWkbTypes,
)


class LayerUtils:
    """Static utility class for common layer operations."""

    @staticmethod
    def get_all_layers() -> list[QgsMapLayer]:
        """Return all layers currently loaded in the QGIS project."""
        return list(QgsProject.instance().mapLayers().values())

    @staticmethod
    def get_vector_layers() -> list[QgsVectorLayer]:
        """Return all vector layers in the current project."""
        return [
            layer for layer in QgsProject.instance().mapLayers().values()
            if isinstance(layer, QgsVectorLayer)
        ]

    @staticmethod
    def get_raster_layers() -> list[QgsRasterLayer]:
        """Return all raster layers in the current project."""
        return [
            layer for layer in QgsProject.instance().mapLayers().values()
            if isinstance(layer, QgsRasterLayer)
        ]

    @staticmethod
    def get_layer_by_name(name: str) -> Optional[QgsMapLayer]:
        """
        Find a layer by its name.

        :param name: Layer name to search for
        :returns: First matching layer, or None
        """
        layers = QgsProject.instance().mapLayersByName(name)
        return layers[0] if layers else None

    @staticmethod
    def is_polygon_layer(layer: QgsVectorLayer) -> bool:
        """Check if a vector layer contains polygon geometries."""
        return layer.geometryType() == QgsWkbTypes.GeometryType.PolygonGeometry

    @staticmethod
    def is_line_layer(layer: QgsVectorLayer) -> bool:
        """Check if a vector layer contains line geometries."""
        return layer.geometryType() == QgsWkbTypes.GeometryType.LineGeometry

    @staticmethod
    def is_point_layer(layer: QgsVectorLayer) -> bool:
        """Check if a vector layer contains point geometries."""
        return layer.geometryType() == QgsWkbTypes.GeometryType.PointGeometry

    @staticmethod
    def get_field_names(layer: QgsVectorLayer) -> list[str]:
        """
        Return a list of field names for a vector layer.

        :param layer: The vector layer to inspect
        :returns: List of field name strings
        """
        return [field.name() for field in layer.fields()]

    @staticmethod
    def layer_is_valid(layer: Optional[QgsMapLayer]) -> bool:
        """
        Check if a layer exists and is valid.

        :param layer: Layer to check
        :returns: True if valid, False otherwise
        """
        return layer is not None and layer.isValid()
