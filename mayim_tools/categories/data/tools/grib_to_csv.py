# grib_to_csv.py
#
# Data Tools — Convert GRIB to CSV Wrapper
# Mayim Tools | Data Tools Category
#
# Thin wrapper around GribToCsvAlgorithm that assigns it to the
# Mayim Tools Data Tools group without modifying the original
# grib_to_csv package.
#
# IMPORTANT: This tool depends on xarray, cfgrib and eccodes.
# The in-process compatibility of eccodes with QGIS's bundled GDAL
# has been verified before integration. If QGIS crashes or hangs
# after this tool loads, disable this tool and investigate the
# eccodes in-process conflict documented in the technical reference.
#
# Author  : Mayim Tools Development Team
# Created : 2025
# License : Proprietary — Zutari / Mayim

"""
Convert GRIB to CSV — Mayim Tools wrapper.

Wraps GribToCsvAlgorithm from the bundled grib_to_csv subpackage,
overriding only the group, groupId, createInstance and icon methods
to place the tool inside the Mayim Tools Data Tools category.

All processing logic remains in grib_to_csv/core.py — this wrapper
does not reimplement any algorithm logic.

Dependencies: xarray, cfgrib, eccodes.
"""

from __future__ import annotations

from mayim_tools.grib_to_csv.grib_to_csv_algorithm import (
    GribToCsvAlgorithm,
)


class MayimGribToCsv(GribToCsvAlgorithm):
    """
    Convert GRIB to CSV — Mayim Tools edition.

    Inherits all algorithm logic from GribToCsvAlgorithm.
    Overrides group, groupId, createInstance and icon to place
    the tool in the Data Tools category within Mayim Tools.
    """

    def group(self) -> str:  # noqa: N802
        return "Data Tools"

    def groupId(self) -> str:  # noqa: N802
        return "datatools"

    def createInstance(self):  # noqa: N802
        return MayimGribToCsv()

    def icon(self):  # noqa: N802
        """Return the Mayim Tools icon."""
        from qgis.PyQt.QtGui import QIcon

        from mayim_tools.resources_rc import get_icon_path

        icon = QIcon(get_icon_path("data.png"))

        if icon.isNull():
            icon = QIcon(get_icon_path("mayim_logo.png"))

        if icon.isNull():
            return super().icon()

        return icon
