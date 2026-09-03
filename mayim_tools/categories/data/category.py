"""
Mayim Tools — Data Tools Category

Descriptor class for the Data Tools category.
"""

from __future__ import annotations

from mayim_tools.categories.base_category import BaseCategory


class DataCategory(BaseCategory):
    """
    Data Tools category.

    Contains tools for data conversion, export and preparation,
    including GRIB, CSV, and other geospatial data formats.
    """

    @property
    def id(self) -> str:
        return "data"

    @property
    def name(self) -> str:
        return "Data Tools"

    @property
    def description(self) -> str:
        return (
            "Tools for data conversion, export and preparation. "
            "Includes GRIB to CSV export for ERA5 and other "
            "reanalysis or forecast products."
        )

    @property
    def icon_path(self) -> str:
        from pathlib import Path

        from mayim_tools.resources_rc import get_icon_path

        data_icon = get_icon_path("data.png")

        if data_icon and Path(data_icon).exists():
            return data_icon

        return get_icon_path("mayim_logo.png")

    def get_algorithms(self) -> list:
        """Return all Data Tools processing algorithms."""
        try:
            from mayim_tools.categories.data.tools.grib_to_csv import (
                MayimGribToCsv,
            )

            return [
                MayimGribToCsv(),
            ]

        except Exception as error:  # noqa: BLE001
            from mayim_tools.core.logger import MayimLogger

            MayimLogger.critical(
                f"Data Tools: Failed to load algorithms: {error}"
            )
            return []
