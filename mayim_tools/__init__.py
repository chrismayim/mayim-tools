"""
Mayim Tools – QGIS 4+ Plugin
Entry point: classFactory() is called by QGIS on plugin load.
"""

__version__ = "0.3.0"
__author__ = "Chris Etsebeth / Mayim Consulting Engineers"
__email__ = "chris@mayimconsulting.com"
__licence__ = "GPL-2.0+"


def classFactory(iface):
    """
    Load MayimToolsPlugin class.

    :param iface: A QGIS interface instance (QgisInterface).
    :type iface: QgisInterface
    :returns: MayimToolsPlugin instance
    """
    from mayim_tools.mayim_tools_plugin import MayimToolsPlugin

    return MayimToolsPlugin(iface)
