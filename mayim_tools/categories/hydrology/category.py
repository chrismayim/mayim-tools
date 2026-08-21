# -*- coding: utf-8 -*-
"""
Mayim Tools – Hydrology Category
Descriptor class for the Hydrology tool category.
"""

from mayim_tools.categories.base_category import BaseCategory


class HydrologyCategory(BaseCategory):
    """
    Hydrology Tools category.
    Contains tools for catchment analysis, flow routing, and rainfall processing.
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
            "Tools for hydrological analysis including catchment delineation, "
            "flow accumulation, rainfall analysis, and water balance calculations."
        )

    @property
    def icon_path(self) -> str:
        return ":/icons/hydrology.png"

    def get_algorithms(self) -> list:
        """
        Return all Hydrology processing algorithms.
        Add new tools here as they are developed.
        """
        # Import here to avoid circular imports at module load time
        # from mayim_tools.categories.hydrology.tools.catchment_delineation \
        #     import CatchmentDelineationAlgorithm
        # return [CatchmentDelineationAlgorithm()]

        # Returning empty list until tools are implemented:
        return []
