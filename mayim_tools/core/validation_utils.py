"""
Mayim Tools - Validation utilities.
"""

from __future__ import annotations

from typing import Any


class ValidationUtils:
    """
    Shared input validation helpers.
    """

    @staticmethod
    def is_not_none(value: Any, label: str = "Value") -> bool:
        """
        Check that a value is not None.

        Parameters
        ----------
        value : Any
            Value to test.
        label : str
            Optional label for future compatibility.

        Returns
        -------
        bool
            True if the value is not None.
        """
        _ = label
        return value is not None

    @staticmethod
    def is_non_empty_string(value: str | None) -> bool:
        """
        Check that a string is non-empty after stripping whitespace.

        Parameters
        ----------
        value : str | None
            Value to test.

        Returns
        -------
        bool
            True if the string contains non-whitespace characters.
        """
        return not (not value or not value.strip())

    @staticmethod
    def is_positive_number(value: float) -> bool:
        """
        Check that a number is positive.

        Parameters
        ----------
        value : int | float
            Number to test.

        Returns
        -------
        bool
            True if the number is greater than zero.
        """
        return value > 0

    @staticmethod
    def is_in_range(
        value: float,
        minimum: float,
        maximum: float,
    ) -> bool:
        """
        Check that a value lies within an inclusive numeric range.

        Parameters
        ----------
        value : int | float
            Value to test.
        minimum : int | float
            Minimum permitted value.
        maximum : int | float
            Maximum permitted value.

        Returns
        -------
        bool
            True if minimum <= value <= maximum.
        """
        return minimum <= value <= maximum

    @staticmethod
    def is_valid_raster_layer(layer: Any) -> bool:
        """
        Check that a raster layer object exists and is valid.

        Parameters
        ----------
        layer : Any
            Candidate QGIS raster layer.

        Returns
        -------
        bool
            True if the layer exists and reports valid.
        """
        return layer is not None and hasattr(layer, "isValid") and layer.isValid()

    @staticmethod
    def is_valid_vector_layer(layer: Any) -> bool:
        """
        Check that a vector layer object exists and is valid.

        Parameters
        ----------
        layer : Any
            Candidate QGIS vector layer.

        Returns
        -------
        bool
            True if the layer exists and reports valid.
        """
        return layer is not None and hasattr(layer, "isValid") and layer.isValid()
