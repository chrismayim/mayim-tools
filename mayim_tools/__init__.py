"""
Mayim Tools - QGIS Processing Plugin
=====================================

Plugin entry point required by QGIS. QGIS calls ``classFactory`` when the
plugin is loaded, and expects it to return an instance of the main plugin
class.
"""

from __future__ import annotations


def classFactory(iface):  # noqa: N802 - QGIS requires this exact name.
    """
    Instantiate the Mayim Tools plugin.

    Parameters
    ----------
    iface:
        QGIS application interface, supplied by QGIS at load time.

    Returns
    -------
    MayimToolsPlugin
        The plugin instance.
    """
    from mayim_tools.mayim_tools_plugin import MayimToolsPlugin

    return MayimToolsPlugin(iface)
