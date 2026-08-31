"""
Mayim Tools — Hydrology Category
Descriptor class for the Hydrology tool category.
"""

from mayim_tools.categories.base_category import BaseCategory


class HydrologyCategory(BaseCategory):
    """
    Hydrology Tools category.
    Contains tools for DEM conditioning, catchment analysis,
    flow routing, and rainfall processing.
    """

    @property
    def id(self) -> str:
        return "hydrology"

    @property
    def name(self) -> str:
        return "Hydrology Tools"

    @property
    def description(self) -> str:
        return (
            "Tools for hydrological analysis including DEM conditioning, "
            "catchment delineation, flow accumulation, rainfall analysis, "
            "and water balance calculations."
        )

    @property
    def icon_path(self) -> str:
        from mayim_tools.resources_rc import get_icon_path
        return get_icon_path("hydrology.png")

    def get_algorithms(self) -> list:
        """
        Return all Hydrology processing algorithms.
        Each tool is imported here to avoid circular imports
        at module load time.
        """
        try:
            from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
                DEMHydrologicalScreening,
            )
            from mayim_tools.categories.hydrology.tools.dem_hydrological_smoothing import (
                DEMHydrologicalSmoothing,
            )
            from mayim_tools.categories.hydrology.tools.dem_depression_analysis import (
                DEMDepressionAnalysis,
            )
            from mayim_tools.categories.hydrology.tools.dem_hydrological_filling import (
                DEMHydrologicalFilling,
            )

            return [
                DEMHydrologicalScreening(),
                DEMHydrologicalSmoothing(),
                DEMDepressionAnalysis(),
                DEMHydrologicalFilling(),
            ]

        except Exception as e:
            from mayim_tools.core.logger import MayimLogger
            MayimLogger.critical(
                f"Hydrology Tools: Failed to load algorithms: {e}"
            )
            return []
