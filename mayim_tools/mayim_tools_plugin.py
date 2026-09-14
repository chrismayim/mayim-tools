# -*- coding: utf-8 -*-
"""
Mayim Tools - Processing Provider
===================================

Registers the Mayim Tools group inside the QGIS Processing Toolbox.
No algorithms are registered yet. Tool categories are added back
deliberately during the rebuild.
"""

from __future__ import annotations

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon


class MayimToolsProvider(QgsProcessingProvider):
    """
    Mayim Tools Processing provider.

    Currently registers zero algorithms. This is intentional: the
    plugin is being rebuilt from a clean, empty state, and tool
    categories will be added back one at a time.
    """

    def id(self) -> str:  # noqa: A003 - QGIS API requires this name.
        """Return the unique provider ID used in algorithm IDs."""
        return "mayimtools"

    def name(self) -> str:
        """Return the provider's display name in the Processing Toolbox."""
        return "Mayim Tools"

    def icon(self) -> QIcon:
        """Return the provider icon, or a blank icon if unavailable."""
        return QIcon()

    def loadAlgorithms(self) -> None:  # noqa: N802 - QGIS API requires this name.
        """
        Register algorithms with the provider.

        Intentionally empty. Algorithms are added back deliberately
        during the rebuild.
        """
        return
