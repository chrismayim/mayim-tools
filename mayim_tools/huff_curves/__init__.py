def classFactory(iface):
    from .huff_curves_plugin import HuffCurvesPlugin

    return HuffCurvesPlugin(iface)
