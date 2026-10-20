"""Tests for Extract: Land Cover - the SANLC product (local GeoTIFF)."""

from __future__ import annotations

import numpy as np
import pytest

from mayim_tools.land_cover.land_cover_extract import core
from mayim_tools.land_cover.land_cover_extract import sanlc as sl
from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.grid import TargetGrid

ALBERS = (
    "+proj=aea +lat_0=0 +lon_0=25 +lat_1=-24 +lat_2=-33 +x_0=0 +y_0=0 "
    "+datum=WGS84 +units=m +no_defs"
)


def test_legend_matches_the_2022_attribute_table():
    assert [c.code for c in sl.CLASSES] == list(range(1, 74))
    c = sl.CLASS_BY_CODE
    assert c[13].name == "Natural grassland" and c[13].salcc1 == "Grassland"
    assert c[19].salcc2 == "Artificial Waterbodies"
    assert c[9].salcc2 == "Karoo & Fynbos Shrubland"  # VAT spelling slip fixed
    assert c[73].name == "Fallow land & old fields (wetlands)"
    assert {k.salcc1 for k in sl.CLASSES} == {
        "Forested land",
        "Shrubland",
        "Grassland",
        "Waterbodies",
        "Wetlands",
        "Barren land",
        "Cultivated",
        "Built-up",
        "Mines & quarries",
    }
    assert all(k.colour.startswith("#") and len(k.colour) == 7 for k in sl.CLASSES)
    assert sl.class_name(0) == "Unknown (0)" and sl.class_name(None) == ""


def test_same_codes_as_the_rational_c_lookup():
    from mayim_tools.land_cover.rational_c.lookup import sanlc_lookup

    assert set(sanlc_lookup().rules) == set(sl.CLASS_BY_CODE)


def test_edition_from_file_name():
    assert sl.year_in_name(r"C:\x\SA_NLC_2022_ALBERS.tif") == 2022
    assert sl.year_in_name("sanlc.tif") is None
    assert sl.edition_warnings("SA_NLC_2022_ALBERS.tif", 2022) == []
    assert "suggests SANLC 2020" in sl.edition_warnings("SANLC_2020.tif", 2022)[0]
    assert "does not show" in sl.edition_warnings("landcover.tif", 2018)[0]


def _grid():
    return TargetGrid(0.0, 0.0, 200.0, 100.0, 20.0, "LOCAL")


def test_extract_area_with_fakes(tmp_path):
    written = {}

    def warp(path, grid, cancel_fn):
        arr = np.full((grid.height, grid.width), 13, np.uint8)
        arr[:, :3] = 47
        arr[0, 0] = 0
        return arr

    def write(path, grid, arr, year):
        written[path] = (arr.copy(), year)

    r = sl.extract_area(
        2022,
        "SA_NLC_2022_ALBERS.tif",
        _grid(),
        str(tmp_path),
        check_fn=lambda p: {"type": "Byte", "res": 20.0},
        warp_fn=warp,
        write_fn=write,
    )
    rec = r.records[0]
    assert rec.counts == {13: 35, 47: 14} and rec.nodata_cells == 1
    assert rec.areas[13] == pytest.approx(35 * 0.0004)
    assert r.files[0].endswith("sanlc_2022_map.tif")
    assert r.warnings and "no class" in r.warnings[-1]
    meta = sl.build_metadata(r, {"x": 1}, "3.9", "tool")
    names = [s[0] for s in meta]
    assert names[:3] == ["Run", "Classes", "Class areas"]
    groups = dict((row[0], row[1]) for row in meta[3][2])
    assert set(groups) == {"Grassland", "Built-up"}


def test_extract_area_outside_raster(tmp_path):
    with pytest.raises(SoilDataError, match="outside South Africa"):
        sl.extract_area(
            2022,
            "SA_NLC_2022.tif",
            _grid(),
            str(tmp_path),
            check_fn=lambda p: {},
            warp_fn=lambda p, g, c: np.zeros((g.height, g.width), np.uint8),
            write_fn=lambda *a: None,
        )


def test_extract_area_warns_on_odd_rasters(tmp_path):
    r = sl.extract_area(
        2020,
        "lc_2022.tif",
        _grid(),
        str(tmp_path),
        check_fn=lambda p: {"type": "Float32", "res": 30.0},
        warp_fn=lambda p, g, c: np.full((g.height, g.width), 99, np.uint8),
        write_fn=lambda *a: None,
    )
    text = " ".join(r.warnings)
    assert "suggests SANLC 2022" in text
    assert "Float32" in text and "30-unit cells" in text
    assert "not SANLC classes" in text


def test_extract_points_with_fakes():
    sites = [core.Site("a", 28.0, -25.0), core.Site("b", 10.0, 0.0)]
    r = sl.extract_points(
        2022,
        "SA_NLC_2022.tif",
        sites,
        check_fn=lambda p: {"type": "Byte", "res": 20.0},
        sample_fn=lambda path, lonlats: [66.0, float("nan")],
    )
    a, b = r.rows
    assert a["ClassCode"] == 66 and a["ClassName"] == "Industrial"
    assert a["SALCC_1"] == "Built-up" and a["SALCC_2"] == "Industrial"
    assert b["ClassCode"] == "" and b["SALCC_1"] == ""
    assert list(a) == sl.POINT_COLUMNS
    assert r.records[0].nodata_cells == 1


def test_unknown_edition():
    with pytest.raises(SoilDataError):
        sl.extract_points(2019, "x.tif", [core.Site("a", 1, 1)])


def test_gdal_end_to_end(tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    from osgeo import osr

    from mayim_tools.soil._common.export import write_class_geotiff
    from mayim_tools.soil._common.grid import make_grid

    gdal.UseExceptions()
    alb = osr.SpatialReference()
    alb.ImportFromProj4(ALBERS)
    utm = osr.SpatialReference()
    utm.ImportFromEPSG(32735)
    utm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    ll = osr.SpatialReference()
    ll.ImportFromEPSG(4326)
    ll.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    # a "national" raster around Pretoria in Albers: west grassland, east
    # residential, a dam block, and 0 (no data) in the north-west corner
    to_alb = osr.CoordinateTransformation(ll, alb)
    cx, cy, _ = to_alb.TransformPoint(28.2, -25.75)
    n = 500
    x0, y0 = cx - n * 10, cy - n * 10
    arr = np.full((n, n), 13, np.uint8)
    arr[:, n // 2 :] = 49
    arr[200:260, 100:160] = 19
    arr[:40, :40] = 0
    src = tmp_path / "SA_NLC_2022_ALBERS.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(src), n, n, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((x0, 20, 0, y0 + n * 20, 0, -20))
    ds.SetProjection(alb.ExportToWkt())
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.GetRasterBand(1).WriteArray(arr)
    ds = None

    to_utm = osr.CoordinateTransformation(ll, utm)
    ux, uy, _ = to_utm.TransformPoint(28.2, -25.75)
    grid = make_grid(
        (ux - 3000, uy - 3000, ux + 3000, uy + 3000), 20.0, utm.ExportToWkt()
    )

    def write(path, g, a, year):
        write_class_geotiff(
            path,
            g,
            a,
            sl.band_description(year),
            [(c.code, c.name, c.colour) for c in sl.CLASSES],
            nodata=sl.NODATA,
        )

    r = sl.extract_area(2022, str(src), grid, str(tmp_path), write_fn=write)
    rec = r.records[0]
    assert set(rec.counts) == {13, 19, 49}  # never averaged
    assert rec.counts[13] == pytest.approx(rec.counts[49], rel=0.15)
    assert r.info["type"] == "Byte" and r.info["res"] == 20
    out = gdal.Open(str(tmp_path / "sanlc_2022_map.tif"))
    band = out.GetRasterBand(1)
    assert band.GetDescription() == "landcover_sanlc_2022_20m (class)"
    assert band.GetRasterColorTable().GetColorEntry(13)[:3] == (210, 180, 133)
    out = None

    pts = sl.extract_points(
        2022,
        str(src),
        [core.Site("res", 28.23, -25.75), core.Site("far", 18.4, -33.9)],
    )
    assert [p["ClassCode"] for p in pts.rows] == [49, ""]
