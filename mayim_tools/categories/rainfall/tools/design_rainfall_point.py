# design_rainfall_point.py
#
# Rainfall Analysis Tools — Design Rainfall at Point(s) Wrapper
# Mayim Tools | Rainfall Analysis Category
#
# Thin wrapper around DesignRainfallPointAlgorithm that assigns it
# to the Mayim Tools Rainfall Analysis Tools group without modifying
# the original design_rainfall package.
#
# Author  : Mayim Tools Development Team
# Created : 2025
# License : Proprietary — Zutari / Mayim

"""
Design Rainfall at Point(s) — Mayim Tools wrapper.

Wraps DesignRainfallPointAlgorithm from the bundled design_rainfall
subpackage, overriding only the group and groupId methods to place
the tool inside the Mayim Tools Rainfall Analysis Tools category.

All calculation logic remains in design_rainfall/core.py and
design_rainfall/report.py — this wrapper does not reimplement
any algorithm logic.
"""

from __future__ import annotations

from mayim_tools.design_rainfall.design_rainfall_algorithm import (
    DesignRainfallPointAlgorithm,
)


class MayimDesignRainfallPoint(DesignRainfallPointAlgorithm):
    """
    Design Rainfall at Point(s) — Mayim Tools edition.

    Inherits all algorithm logic from DesignRainfallPointAlgorithm.
    Overrides group and groupId to place the tool in the
    Rainfall Analysis Tools category within Mayim Tools.
    """

    def icon(self):  # noqa: N802
        """Return the Mayim Tools icon."""
        from qgis.PyQt.QtGui import QIcon

        from mayim_tools.resources_rc import get_icon_path

        icon = QIcon(get_icon_path("mayim_logo.png"))

        if icon.isNull():
            return super().icon()

        return icon

    def group(self) -> str:  # noqa: N802
        return "Rainfall Analysis Tools"

    def groupId(self) -> str:  # noqa: N802
        return "rainfallanalysistools"

    def createInstance(self):  # noqa: N802
        return MayimDesignRainfallPoint()
