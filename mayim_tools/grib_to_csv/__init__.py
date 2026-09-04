def classFactory(iface):
    from .grib_to_csv_plugin import GribToCsvPlugin

    return GribToCsvPlugin(iface)
