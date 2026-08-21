# -*- coding: utf-8 -*-
"""
Mayim Tools – QGIS 4+ Plugin
Entry point: classFactory() is called by QGIS on plugin load.
"""


def classFactory(iface):
    """
    Load MayimToolsPlugin class.

    :param iface: A QGIS interface instance (QgisInterface).
    :type iface: QgisInterface
    :returns: MayimToolsPlugin instance
    """
    from mayim_tools.mayim_tools_plugin import MayimToolsPlugin
    return MayimToolsPlugin(iface)
