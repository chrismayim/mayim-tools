"""Contour shapefile output of the DEM Hydrological Processing tool."""

from __future__ import annotations

import numpy as np
import pytest

from mayim_tools.hydrology.dem_hydro_processing.contours_export import (
    contour_levels,
    is_major,
    write_contours,
)

GT = (500000.0, 10.0, 0.0, 7100000.0, 0.0, -10.0)


def test_levels_and_major():
    assert contour_levels(100.2, 104.9, 1.0) == 4  # 101 .. 104
    assert contour_levels(100.0, 105.0, 1.0) == 6
    assert contour_levels(0.0, 1.0, 0.0) == 0
    assert is_major(105.0, 1.0) and not is_major(104.0, 1.0)
    assert is_major(2.5, 0.5) and not is_major(2.0, 0.5)


def _cone():
    y, x = np.mgrid[0:60, 0:80]
    return 100.0 + 0.2 * np.hypot(x - 40, y - 30)  # bowl, ~100 to ~110 m


def test_write_contours_fields_types_and_nodata(tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    from osgeo import ogr, osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32735)
    dem = _cone()
    dem[:5, :5] = np.nan
    path = str(tmp_path / "M1_Contours.shp")
    n = write_contours(path, dem, GT, srs.ExportToWkt(), 1.0)
    assert n > 0
    ds = ogr.Open(path)
    layer = ds.GetLayer(0)
    names = [layer.GetLayerDefn().GetFieldDefn(i).GetName() for i in range(3)]
    assert names == ["id", "elev_m", "type"]
    levels = {}
    for f in layer:
        levels[f.GetField("elev_m")] = f.GetField("type")
    assert {float(v) for v in range(101, 110)} <= set(levels)
    assert all(v == round(v) for v in levels)  # multiples of 1 m
    assert levels[105.0] == "Major" and levels[106.0] == "Minor"
    assert levels[101.0] == "Minor"
    ext = layer.GetExtent()  # within the DEM
    assert ext[0] >= GT[0] and ext[1] <= GT[0] + 80 * GT[1]
    assert (tmp_path / "M1_Contours.cpg").read_text() == "UTF-8"
    ds = None
    # replaced on a second run
    assert write_contours(path, dem, GT, srs.ExportToWkt(), 5.0) >= 1
    del gdal


def test_too_many_levels_raises(tmp_path):
    pytest.importorskip("osgeo.gdal")
    with pytest.raises(ValueError, match="larger contour interval"):
        write_contours(str(tmp_path / "c.shp"), _cone(), GT, "", 0.001)
    with pytest.raises(ValueError, match="greater than 0"):
        write_contours(str(tmp_path / "c.shp"), _cone(), GT, "", 0.0)
