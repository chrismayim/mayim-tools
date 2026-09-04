"""
Mayim Tools — Rainfall Analysis Category

Descriptor class for the Rainfall Analysis tool category.
"""

from __future__ import annotations

from mayim_tools.categories.base_category import BaseCategory


class RainfallCategory(BaseCategory):
    """
    Rainfall Analysis Tools category.

    Contains tools for design rainfall estimation, frequency analysis,
    and regional rainfall statistics.
    """

    @property
    def id(self) -> str:
        return "rainfall"

    @property
    def name(self) -> str:
        return "Rainfall Analysis Tools"

    @property
    def description(self) -> str:
        return (
            "Tools for design rainfall estimation and regional rainfall "
            "analysis, including the Smithers & Schulze L-moment methodology "
            "(WRC Report K5/1060, 2002)."
        )

    @property
    def icon_path(self) -> str:
        from mayim_tools.resources_rc import get_icon_path

        return get_icon_path("rainfall.png")

    def get_algorithms(self) -> list:
        """Return all Rainfall Analysis processing algorithms."""
        try:
            from mayim_tools.categories.rainfall.tools.design_rainfall_point import (
                MayimDesignRainfallPoint,
            )
            from mayim_tools.categories.rainfall.tools.huff_curves_from_csv import (
                MayimHuffCurvesFromCSV,
            )

            return [
                MayimDesignRainfallPoint(),
                MayimHuffCurvesFromCSV(),
            ]

        except Exception as error:  # noqa: BLE001
            from mayim_tools.core.logger import MayimLogger

            MayimLogger.critical(
                f"Rainfall Analysis Tools: Failed to load algorithms: {error}"
            )
            return []
