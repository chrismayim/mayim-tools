"""Tests for Extract: Land Cover (ESA WorldCover) - no QGIS, no network."""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest

from mayim_tools.land_cover.land_cover_extract import catalogue as cat
from mayim_tools.land_cover.land_cover_extract import core
from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.grid import TargetGrid

WGS84 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
)


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------


def test_tile_names_and_paths():
    assert cat.tile_name(51, 3) == "N51E003"
    assert cat.tile_name(-27, 30) == "S27E030"
    assert cat.tile_name(0, -93) == "N00W093"
    assert cat.tile_of(31.0, -25.7) == "S27E030"  # Mpumalanga
    assert cat.tile_of(18.4, -33.9) == "S36E018"  # Cape Town
    assert cat.tile_of(115.9, -31.95) == "S33E114"  # Perth
    assert cat.tile_of(3.0, 51.0) == "N51E003"  # on the corner
    assert cat.tile_key(2021, "N51E003") == (
        "v200/2021/map/ESA_WorldCover_10m_2021_v200_N51E003_Map.tif"
    )
    assert cat.tile_key(2020, "N21E030") == (
        "v100/2020/map/ESA_WorldCover_10m_2020_v100_N21E030_Map.tif"
    )
    assert cat.cog_path(2021, "S27E030").startswith(
        "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/"
    )
    assert cat.s3_path(2021, "S27E030") == (
        "/vsis3/esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_S27E030_Map.tif"
    )


def test_tiles_for_bounds():
    assert cat.tiles_for_bounds((30.5, -25.5, 31.0, -25.0)) == ["S27E030"]
    # Across a corner: four tiles.
    assert sorted(cat.tiles_for_bounds((29.5, -24.5, 30.5, -23.5))) == [
        "S24E027",
        "S24E030",
        "S27E027",
        "S27E030",
    ]
    # Exactly on a tile edge: the pad pulls in the neighbour (safe).
    assert "S27E030" in cat.tiles_for_bounds((30.2, -24.5, 30.4, -24.0))
    assert cat.tiles_for_bounds((179.5, 10.0, 180.0, 10.5)) == ["N09E177"]


def test_classes_and_editions():
    assert [c.code for c in cat.CLASSES] == [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        95,
        100,
    ]
    assert cat.CLASS_BY_CODE[10].colour == "#006400"
    assert cat.class_name(40) == "Cropland"
    assert cat.class_name(None) == ""
    assert cat.class_name(7) == "Unknown (7)"
    assert cat.EDITION_BY_YEAR[2021].version == "v200"
    assert "zenodo.7254221" in cat.EDITION_BY_YEAR[2021].citation
    assert cat.band_description(2021) == "landcover_worldcover_2021_v200_10m (class)"


def test_plan_selection():
    sel = core.plan_selection([2020, 2021, 2021])
    assert sel.years == [2021, 2020]
    assert any("different algorithms" in w for w in sel.warnings)
    assert core.plan_selection([2021]).warnings == []
    with pytest.raises(SoilDataError):
        core.plan_selection([])
    with pytest.raises(SoilDataError):
        core.plan_selection([2019])


# ----------------------------------------------------------------------
# Tile finding
# ----------------------------------------------------------------------


def test_find_tiles_skips_ocean_and_falls_back():
    tiles = ["S36E018", "S36E015"]  # second is open sea

    def https_only(p):
        return p.startswith("/vsicurl/") and "S36E018" in p

    ts = core.find_tiles(2021, tiles, https_only, workers=2)
    assert ts.route == "COG (https)" and ts.found == ["S36E018"]
    assert ts.missing == ["S36E015"]

    def s3_only(p):
        return p.startswith("/vsis3/") and "S36E018" in p

    ts = core.find_tiles(2021, tiles, s3_only, workers=1)
    assert ts.route == "S3 (unsigned)" and ts.found == ["S36E018"]


def test_find_tiles_all_ocean_vs_unreachable():
    def only_known(p):
        return cat.KNOWN_TILE in p

    ts = core.find_tiles(2021, ["S48E000"], only_known, workers=1)
    assert ts.found == [] and ts.missing == ["S48E000"]

    def broken(p):
        raise OSError("proxy")

    with pytest.raises(SoilDataError, match="Could not reach"):
        core.find_tiles(2021, ["S27E030"], broken, workers=4)


# ----------------------------------------------------------------------
# Areas
# ----------------------------------------------------------------------


def test_class_areas_projected_and_geographic():
    grid = TargetGrid(0, 0, 40, 20, 10, "")
    arr = np.array([[10, 10, 30, 0], [40, 40, 40, 10]], dtype=np.uint8)
    counts, areas = core.class_areas(arr, core.row_cell_areas_km2(grid, False))
    assert counts == {0: 1, 10: 3, 30: 1, 40: 3}
    assert areas[10] == pytest.approx(3e-4)
    g = TargetGrid(0.0, -1.0, 1.0, 1.0, 1.0, WGS84)
    row = core.row_cell_areas_km2(g, True)
    assert row[0] == pytest.approx(row[1])
    assert row.sum() == pytest.approx(2 * 111.19**2, rel=0.01)


# ----------------------------------------------------------------------
# Area mode (fakes)
# ----------------------------------------------------------------------


class Recorder:
    def __init__(self):
        self.written = {}

    def __call__(self, path, grid, arr, year):
        self.written[os.path.basename(path)] = arr.copy()


def _grid():
    return TargetGrid(30.0, -25.02, 30.04, -25.0, 0.01, WGS84)


def test_extract_area_with_fakes(tmp_path):
    calls = []

    def warp(paths, grid, cancel_fn):
        calls.append(list(paths))
        if paths[0].startswith("/vsicurl/"):
            raise OSError("https blocked")
        arr = np.full((grid.height, grid.width), 30, dtype=np.uint8)
        arr[0, 0] = 0
        arr[1, :2] = 10
        return arr

    rec = Recorder()
    sel = core.plan_selection([2021, 2020])
    result = core.extract_area(
        sel,
        _grid(),
        (30.0, -25.02, 30.04, -25.0),
        str(tmp_path),
        geographic=True,
        exists_fn=lambda p: "S27E030" in p,
        warp_fn=warp,
        write_fn=rec,
        workers=1,
    )
    assert sorted(rec.written) == ["worldcover_2020_map.tif", "worldcover_2021_map.tif"]
    assert len(calls) == 4  # https then s3, for each year
    y = result.years[0]
    assert y.year == 2021 and y.route == "S3 (unsigned)"
    assert y.counts == {10: 2, 30: 5} and y.nodata_cells == 1
    assert any("no class" in w for w in result.warnings)
    sections = core.build_metadata(result, sel, {"Output CRS": "EPSG:4326"})
    titles = [s[0] for s in sections]
    assert titles == ["Run", "Editions", "Classes", "Tiles", "Class areas", "Warnings"]
    areas = dict((s[0], s[2]) for s in sections)["Class areas"]
    shares = [r[6] for r in areas if r[0] == 2021 and r[2] in (10, 30)]
    assert sum(shares) == pytest.approx(100.0, abs=0.02)


def test_extract_area_all_ocean(tmp_path):
    rec = Recorder()
    result = core.extract_area(
        core.plan_selection([2021]),
        _grid(),
        (30.0, -25.02, 30.04, -25.0),
        str(tmp_path),
        exists_fn=lambda p: cat.KNOWN_TILE in p,
        warp_fn=lambda *a: pytest.fail("should not warp"),
        write_fn=rec,
        workers=1,
    )
    assert (rec.written["worldcover_2021_map.tif"] == 0).all()
    assert any("open ocean" in w for w in result.warnings)


def test_extract_area_needs_writer(tmp_path):
    with pytest.raises(SoilDataError):
        core.extract_area(core.plan_selection([2021]), _grid(), (0, 0, 1, 1), "")


def test_extract_area_cancel(tmp_path):
    with pytest.raises(InterruptedError):
        core.extract_area(
            core.plan_selection([2021]),
            _grid(),
            (30.0, -25.02, 30.04, -25.0),
            str(tmp_path),
            exists_fn=lambda p: True,
            warp_fn=lambda *a: None,
            write_fn=Recorder(),
            cancel_fn=lambda: True,
        )


# ----------------------------------------------------------------------
# Points mode (fakes)
# ----------------------------------------------------------------------


def test_extract_points_with_fakes():
    sites = [
        core.Site("A", 30.5, -25.5),
        core.Site("B", 31.0, -26.0),
        core.Site("Sea", 16.0, -35.0),
    ]
    sampled = []

    def sample(path, lonlats):
        sampled.append((path, list(lonlats)))
        return [40.0, 0.0][: len(lonlats)]

    result = core.extract_points(
        core.plan_selection([2021]),
        sites,
        exists_fn=lambda p: "S27E030" in p or cat.KNOWN_TILE in p,
        sample_fn=sample,
        workers=1,
    )
    assert len(sampled) == 1 and "S27E030" in sampled[0][0]
    rows = {r["Site"]: r for r in result.rows}
    assert rows["A"]["ClassCode"] == 40 and rows["A"]["ClassName"] == "Cropland"
    assert rows["B"]["ClassCode"] == "" and rows["B"]["ClassName"] == ""
    assert rows["Sea"]["ClassCode"] == "" and rows["Sea"]["Tile"] == "S36E015"
    assert result.years[0].nodata_cells == 2
    sections = core.build_metadata(result, core.plan_selection([2021]), {})
    assert "Class counts" in [s[0] for s in sections]


def test_extract_points_requires_sites():
    with pytest.raises(SoilDataError):
        core.extract_points(core.plan_selection([2021]), [])


# ----------------------------------------------------------------------
# Real GDAL with synthetic local tiles (skipped without GDAL)
# ----------------------------------------------------------------------


def _make_tile(path, lat0, lon0, value, res=0.01):
    from osgeo import gdal

    n = int(round(3 / res))
    ds = gdal.GetDriverByName("GTiff").Create(str(path), n, n, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((lon0, res, 0, lat0 + 3, 0, -res))
    ds.SetProjection(WGS84)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    arr = np.full((n, n), value, dtype=np.uint8)
    arr[:5, :5] = 0
    band.WriteArray(arr)
    ds = None


def test_gdal_end_to_end(tmp_path, monkeypatch):
    gdal = pytest.importorskip("osgeo.gdal")
    gdal.UseExceptions()
    from mayim_tools.soil._common.export import write_class_geotiff

    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    # S27E030 = cropland, S27E027 = grassland; S24E030 missing (as if sea).
    _make_tile(tiles_dir / "S27E030.tif", -27, 30, 40)
    _make_tile(tiles_dir / "S27E027.tif", -27, 27, 30)
    _make_tile(tiles_dir / f"{cat.KNOWN_TILE}.tif", 51, 3, 10)

    def local(year, tile):
        return str(tiles_dir / f"{tile}.tif")

    monkeypatch.setattr(core, "ROUTES", [("Local", local)])
    utm36s = gdal.osr.SpatialReference()
    utm36s.ImportFromEPSG(32736)
    # About 30.0 E, -24.05 to -23.95 (crosses into the missing S24E030 tile)
    # and 29.95-30.05 E (crosses into S27E027).
    tr = gdal.osr.CoordinateTransformation(_srs(4326, gdal), utm36s)
    x0, y0, _ = tr.TransformPoint(29.95, -24.05)
    x1, y1, _ = tr.TransformPoint(30.05, -23.95)
    from mayim_tools.soil._common.grid import make_grid

    grid = make_grid((x0, y0, x1, y1), 100.0, utm36s.ExportToWkt())

    def write(path, g, arr, year):
        write_class_geotiff(
            path,
            g,
            arr,
            cat.band_description(year),
            [(c.code, c.name, c.colour) for c in cat.CLASSES],
            metadata={"TOOL_VERSION": core.TOOL_VERSION},
        )

    result = core.extract_area(
        core.plan_selection([2021]),
        grid,
        (29.95, -24.05, 30.05, -23.95),
        str(tmp_path),
        exists_fn=os.path.exists,
        write_fn=write,
        workers=2,
        options_ctx=_null(),
    )
    rec = result.years[0]
    assert sorted(rec.tiles.found) == ["S27E027", "S27E030"]
    assert "S24E030" in rec.tiles.missing
    assert set(rec.counts) == {30, 40} and rec.nodata_cells > 0
    ds = gdal.Open(str(tmp_path / "worldcover_2021_map.tif"))
    band = ds.GetRasterBand(1)
    assert band.DataType == gdal.GDT_Byte and band.GetNoDataValue() == 0
    assert band.GetDescription() == "landcover_worldcover_2021_v200_10m (class)"
    assert band.GetRasterColorTable().GetColorEntry(40)[:3] == (240, 150, 255)
    assert "40=Cropland" in ds.GetMetadataItem("CLASSES")
    assert not os.path.exists(str(tmp_path / "worldcover_2021_map.tif.aux.xml"))
    arr = band.ReadAsArray()
    assert set(np.unique(arr)) == {0, 30, 40}  # never averaged
    ds = None

    # Points: one on each tile, one in the missing tile.
    pts = core.extract_points(
        core.plan_selection([2021]),
        [
            core.Site("crop", 30.5, -25.0),
            core.Site("grass", 28.0, -25.0),
            core.Site("gap", 30.5, -23.0),
        ],
        exists_fn=os.path.exists,
        options_ctx=_null(),
        workers=1,
    )
    codes = {r["Site"]: r["ClassCode"] for r in pts.rows}
    assert codes == {"crop": 40, "grass": 30, "gap": ""}
    from mayim_tools.soil._common.export import write_points_csv

    out = tmp_path / "pts.csv"
    write_points_csv(pts.rows, out, core.POINT_COLUMNS)
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["ClassCode"] == "40" and rows[2]["ClassCode"] == ""


def _srs(epsg, gdal):
    srs = gdal.osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(gdal.osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


def _null():
    from contextlib import nullcontext

    return nullcontext()
