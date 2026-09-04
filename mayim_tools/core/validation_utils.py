"""
Mayim Tools – Validation Utilities
Input validation helpers used across all tools and categories.
"""

from typing import Any

from qgis.core import QgsRasterLayer, QgsVectorLayer


class ValidationUtils:
    """Static utility class for input validation."""

    @staticmethod
    def is_not_none(value: Any, label: str = "Value") -> bool:
        """Check that a value is not None."""
        if value is None:
            return False
        return True

    @staticmethod
    def is_not_empty_string(value: str, label: str = "Value") -> bool:
        """
        Check that a string is not None or empty.

        :param value: String to check
        :param label: Label used in error messages
        :returns: True if valid, False otherwise
        """
        if not value or not value.strip():
            return False
        return True

    @staticmethod
    def is_positive_number(value: Any, label: str = "Value") -> bool:
        """
        Check that a value is a positive number (int or float).

        :param value: Value to check
        :param label: Label used in error messages
        :returns: True if positive number, False otherwise
        """
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def is_in_range(
        value: float,
        min_val: float,
        max_val: float,
        label: str = "Value",
    ) -> bool:
        """
        Check that a numeric value falls within a specified range.

        :param value: Value to check
        :param min_val: Minimum allowed value (inclusive)
        :param max_val: Maximum allowed value (inclusive)
        :param label: Label used in error messages
        :returns: True if in range, False otherwise
        """
        try:
            return min_val <= float(value) <= max_val
        except (TypeError, ValueError):
            return False

    @staticmethod
    def is_valid_vector_layer(layer: Any) -> bool:
        """
        Check that the input is a valid, loaded QgsVectorLayer.

        :param layer: Object to check
        :returns: True if valid vector layer, False otherwise
        """
        return (
            layer is not None and isinstance(layer, QgsVectorLayer) and layer.isValid()
        )

    @staticmethod
    def is_valid_raster_layer(layer: Any) -> bool:
        """
        Check that the input is a valid, loaded QgsRasterLayer.

        :param layer: Object to check
        :returns: True if valid raster layer, False otherwise
        """
        return (
            layer is not None and isinstance(layer, QgsRasterLayer) and layer.isValid()
        )

    @staticmethod
    def is_valid_file_path(path: str, must_exist: bool = True) -> bool:
        """
        Check that a file path is valid and optionally that it exists.

        :param path: File path string to check
        :param must_exist: If True, also checks that the file exists on disk
        :returns: True if valid, False otherwise
        """
        from pathlib import Path

        if not path or not path.strip():
            return False
        if must_exist:
            return Path(path).exists()
        return True

    @staticmethod
    def is_valid_epsg(epsg_code: Any) -> bool:
        """
        Check that a value is a valid EPSG code (positive integer).

        :param epsg_code: Value to check
        :returns: True if valid EPSG code format, False otherwise
        """
        try:
            code = int(epsg_code)
            return 1024 <= code <= 32767 or 4000 <= code <= 5000
        except (TypeError, ValueError):
            return False

    @staticmethod
    def layer_has_features(layer: QgsVectorLayer) -> bool:
        """
        Check that a vector layer contains at least one feature.

        :param layer: QgsVectorLayer to check
        :returns: True if layer has features, False if empty
        """
        return ValidationUtils.is_valid_vector_layer(layer) and layer.featureCount() > 0

    @staticmethod
    def layer_has_field(layer: QgsVectorLayer, field_name: str) -> bool:
        """
        Check that a vector layer contains a specific field by name.

        :param layer: QgsVectorLayer to inspect
        :param field_name: Field name to look for
        :returns: True if field exists, False otherwise
        """
        if not ValidationUtils.is_valid_vector_layer(layer):
            return False
        return layer.fields().indexFromName(field_name) != -1
