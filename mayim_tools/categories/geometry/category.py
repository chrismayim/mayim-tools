# -*- coding: utf-8 -*-
"""
Mayim Tools – Geometry Category
Descriptor class for the Geometry tool category.
"""

from mayim_tools.categories.base_category import BaseCategory


class GeometryCategory(BaseCategory):
    """
    Geometry Tools category.
    Contains tools for geometric operations and spatial analysis.
    """

    @property
    def id(self) -> str:
        return "geometry"

    @property
    def name(self) -> str:
        return "Geometry Tools"

    @property
    def description(self) -> str:
        return (
            "Tools for geometric operations including polygon splitting, "
            "centreline extraction, buffer analysis, and shape simplification."
        )

    @property
    def icon_path(self) -> str:
        from mayim_tools.resources_rc import get_icon_path
        return get_icon_path("geometry.png")

    def get_algorithms(self) -> list:
        """
        Return all Geometry processing algorithms.
        Add new tools here as they are developed.
        """
        return []
