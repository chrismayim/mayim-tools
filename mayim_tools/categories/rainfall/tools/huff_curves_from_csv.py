# huff_curves_from_csv.py
#
# Rainfall Analysis Tools — Huff Curves from CSV Wrapper
# Mayim Tools | Rainfall Analysis Category
#
# Thin wrapper around HuffCurvesAlgorithm that assigns it to the
# Mayim Tools Rainfall Analysis Tools group without modifying the
# original huff_curves package.
#
# Author  : Mayim Tools Development Team
# Created : 2025
# License : Proprietary — Zutari / Mayim

"""
Huff Curves from CSV — Mayim Tools wrapper.

Wraps HuffCurvesAlgorithm from the bundled huff_curves subpackage,
overriding only the group, groupId, createInstance and icon methods
to place the tool inside the Mayim Tools Rainfall Analysis Tools
category.

All calculation logic remains in huff_curves/huffrain/ — this wrapper
does not reimplement any algorithm logic.
"""

from __future__ import annotations

from mayim_tools.huff_curves.huff_curves_algorithm import (
    HuffCurvesAlgorithm,
)


class MayimHuffCurvesFromCSV(HuffCurvesAlgorithm):
    """
    Huff Curves from CSV — Mayim Tools edition.

    Inherits all algorithm logic from HuffCurvesAlgorithm.
    Overrides group, groupId, createInstance and icon to place
    the tool in the Rainfall Analysis Tools category within
    Mayim Tools.
    """

    def group(self) -> str:
        return "Rainfall Analysis Tools"

    def groupId(self) -> str:
        return "rainfallanalysistools"

    def createInstance(self):
        return MayimHuffCurvesFromCSV()

    def icon(self):
        """Return the Mayim Tools icon."""
        from qgis.PyQt.QtGui import QIcon

        from mayim_tools.resources_rc import get_icon_path

        icon = QIcon(get_icon_path("rainfall.png"))

        if icon.isNull():
            icon = QIcon(get_icon_path("mayim_logo.png"))

        if icon.isNull():
            return super().icon()

        return icon
