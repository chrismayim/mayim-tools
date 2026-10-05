"""
Tests for mayim_tools/soil/openlandmap_extract (catalogue.py, core.py).

No network: the server is replaced by injected fake readers, or by local
GeoTIFFs written with the same scale metadata OpenLandMap uses (real
GDAL tests, skipped when osgeo is unavailable).
"""

import contextlib
import csv
import math
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from mayim_tools.soil._common.export import write_points_csv
from mayim_tools.soil.openlandmap_extract import catalogue as cat
from mayim_tools.soil.openlandmap_extract import core

CATALOGUE_CSV = Path(__file__).parent / "data" / "openlandmap_soildb_cogs.csv"

# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------


def _official_urls():
    with open(CATALOGUE_CSV, newline="", encoding="utf-8") as f:
        return {row["s3_path"] for row in csv.DictReader(f)}


def _built_urls(scheme="https"):
    urls = set()
    for v in cat.VARIABLES:
        for s in cat.STATISTICS:
            periods = v.periods_30m if s.resolution_m == 30 else v.periods_120m
            for depth, _, _ in cat.DEPTHS:
                for period in periods:
                    urls.add(cat.cog_url(v.code, s.code, depth, period, scheme))
    return urls


def test_catalogue_matches_official_layer_table_exactly():
    """Every property layer in openlandmap/soildb's OpenLandMap_soildb_COGS.csv
    (264) is built exactly, and nothing else."""
    official = _official_urls()
    assert len(official) == 264
    assert _built_urls() == official


def test_catalogue_scales_match_official_table():
    scale = {
        "clay": "",
        "sand": "",
        "silt": "",
        "soc": "0.1",
        "socd": "0.1",
        "bd.core": "0.01",
        "ph.h2o": "0.1",
    }
    with open(CATALOGUE_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assert row["scaler"] == scale[row["property"]]
    to_code = {"bd.core": "bdod", "ph.h2o": "phh2o"}
    for prop, value in scale.items():
        v = cat.VARIABLE_BY_CODE[to_code.get(prop, prop)]
        assert v.scale == (float(value) if value else 1.0)


def test_http_variant():
    url = cat.cog_url("clay", "mean30", "0-30cm", "2020-2022", scheme="http")
    assert url.startswith("http://s3.opengeohub.org/global-soil/")
    assert url.endswith(
        "clay.tot_iso.11277.2020.wpct_m_30m_b0cm..30cm_"
        "20200101_20221231_g_epsg.4326_v20250523.tif"
    )


def test_resolve_period_rules():
    # Texture: static, 2020-2022 for any period.
    assert cat.resolve_period("clay", "mean30", "2000-2005") == "2020-2022"
    assert cat.resolve_period("clay", "p16", "2015-2020") == "2020-2022"
    # Bulk density: 30 m only for 2020-2022, 120 m for every period.
    assert cat.resolve_period("bdod", "mean30", "2015-2020") is None
    assert cat.resolve_period("bdod", "mean30", "2020-2022") == "2020-2022"
    assert cat.resolve_period("bdod", "p84", "2005-2010") == "2005-2010"
    # SOC, pH: every period.
    for period in cat.PERIODS:
        assert cat.resolve_period("soc", "mean30", period) == period
        assert cat.resolve_period("phh2o", "mean120", period) == period


def test_names():
    assert cat.output_file_name("clay", "mean_30m", "2020-2022") == (
        "olm_clay_mean_30m_2020-2022.tif"
    )
    assert cat.band_description("soc", "30-60cm", "p16_120m", "2010-2015", "g/kg") == (
        "soc_30-60cm_p16_120m_2010-2015 (g/kg)"
    )


def test_texture_is_iso_not_usda():
    assert "0.063" in cat.VARIABLE_BY_CODE["silt"].label
    assert "ISO 11277" in cat.VARIABLE_BY_CODE["sand"].label


# ----------------------------------------------------------------------
# Selection and job planning
# ----------------------------------------------------------------------


def test_plan_selection_orders_and_validates():
    sel = core.plan_selection(
        ["soc", "clay"],
        ["60-100cm", "0-30cm"],
        ["2020-2022", "2000-2005"],
        ["p84", "mean30"],
    )
    assert sel.variables == ["clay", "soc"]
    assert sel.depths == ["0-30cm", "60-100cm"]
    assert sel.periods == ["2000-2005", "2020-2022"]
    assert sel.statistics == ["mean30", "p84"]


def test_plan_selection_ru68_adds_inputs():
    sel = core.plan_selection(["clay"], ["0-30cm"], ["2020-2022"], ["mean30"], True)
    assert sel.statistics == ["mean30", "mean120", "p16", "p84"]
    assert sel.written_statistics == ["mean30"]
    assert sel.warnings


@pytest.mark.parametrize(
    "args, message",
    [
        ((["x"], ["0-30cm"], ["2020-2022"], ["mean30"]), "Unknown OpenLandMap"),
        ((["clay"], ["0-5cm"], ["2020-2022"], ["mean30"]), "Unknown depth"),
        ((["clay"], ["0-30cm"], ["1999"], ["mean30"]), "Unknown period"),
        ((["clay"], ["0-30cm"], ["2020-2022"], ["Q0.5"]), "Unknown statistic"),
        (([], ["0-30cm"], ["2020-2022"], ["mean30"]), "at least one variable"),
        ((["clay"], [], ["2020-2022"], ["mean30"]), "at least one depth"),
        ((["clay"], ["0-30cm"], [], ["mean30"]), "at least one period"),
        ((["clay"], ["0-30cm"], ["2020-2022"], []), "at least one statistic"),
    ],
)
def test_plan_selection_errors(args, message):
    with pytest.raises(core.SoilDataError, match=message):
        core.plan_selection(*args)


def test_plan_jobs_static_texture_read_once_and_noted():
    sel = core.plan_selection(
        ["clay"], ["0-30cm"], ["2000-2005", "2020-2022"], ["mean30"]
    )
    jobs, notes = core.plan_jobs(sel)
    assert jobs == [core.Job("clay", "mean30", "2020-2022", "0-30cm")]
    assert any("static" in n for n in notes)


def test_plan_jobs_skips_unpublished_and_notes():
    sel = core.plan_selection(["bdod"], ["0-30cm"], ["2015-2020"], ["mean30", "p16"])
    jobs, notes = core.plan_jobs(sel)
    assert jobs == [core.Job("bdod", "p16", "2015-2020", "0-30cm")]
    assert any("not published" in n and "120 m mean" in n for n in notes)


# ----------------------------------------------------------------------
# Scale factors
# ----------------------------------------------------------------------


def test_resolve_scale():
    assert core.resolve_scale(0.1, 0.0, 0.1) == (0.1, 0.0, "file metadata")
    scale, offset, note = core.resolve_scale(None, None, 0.1)
    assert (scale, offset) == (0.1, 0.0) and note.startswith("catalogue")
    scale, offset, note = core.resolve_scale(1.0, 0.0, 0.01)
    assert scale == 0.01 and note.startswith("catalogue")
    assert core.resolve_scale(1.0, 0.0, 1) == (1.0, 0.0, "none needed")
    scale, _, note = core.resolve_scale(0.5, 0.0, 0.1)
    assert scale == 0.5 and "differs" in note
    assert core.resolve_scale(1.0, 5.0, 1)[:2] == (1.0, 5.0)


def test_apply_scale_keeps_nan():
    out = core.apply_scale(np.array([250.0, np.nan]), 0.1, 0.0)
    assert out[0] == pytest.approx(25.0) and math.isnan(out[1])


# ----------------------------------------------------------------------
# Area mode (fake reader / writer)
# ----------------------------------------------------------------------

RAW = {"mean30": 300.0, "mean120": 310.0, "p16": 200.0, "p84": 500.0}
TEXTURE = {"sand": 60.0, "silt": 25.0, "clay": 15.0}


def fake_value(url):
    for v in cat.VARIABLES:
        if f"/{v.filename}_" in url:
            for s in cat.STATISTICS:
                if f"_{s.file_code}_{s.resolution_m}m_" in url:
                    if v.code in TEXTURE and s.code == "mean30":
                        return TEXTURE[v.code], None, None
                    if v.code in TEXTURE:
                        return RAW[s.code] / 10, None, None
                    return RAW[s.code], v.scale, 0.0
    raise AssertionError(url)


def make_reader(fail_https=False, fail_all=False, calls=None, empty=False):
    lock = threading.Lock()

    def reader(source, grid, with_scale=False):
        assert with_scale is True
        if calls is not None:
            with lock:
                calls.append(source)
        if fail_all or (fail_https and source.startswith("/vsicurl/https")):
            raise RuntimeError("HTTP 503")
        value, scale, offset = fake_value(source)
        arr = np.full((grid.height, grid.width), np.nan if empty else value)
        if not empty:
            arr[0, 0] = np.nan
        return arr, scale, offset

    return reader


class FakeWriter:
    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()

    def __call__(self, path, grid, bands, units):
        with self.lock:
            self.files[os.path.basename(path)] = (bands, units)


@pytest.fixture
def grid():
    from mayim_tools.soil._common.grid import make_grid

    return make_grid((0, 0, 120, 90), 30.0, "WKT", buffer_cells=0)


def run_area(grid, tmp_path, selection, **kw):
    writer = FakeWriter()
    result = core.extract_area(
        selection,
        grid,
        str(tmp_path),
        read_grid_fn=kw.pop("reader", make_reader()),
        write_fn=writer,
        options_ctx=contextlib.nullcontext(),
        **kw,
    )
    return result, writer


def test_area_files_bands_and_scaling(grid, tmp_path):
    sel = core.plan_selection(
        ["clay", "soc", "bdod"],
        ["0-30cm", "60-100cm"],
        ["2020-2022"],
        ["mean30", "p16"],
    )
    result, writer = run_area(grid, tmp_path, sel)
    assert set(writer.files) == {
        "olm_clay_mean_30m_2020-2022.tif",
        "olm_clay_p16_120m_2020-2022.tif",
        "olm_soc_mean_30m_2020-2022.tif",
        "olm_soc_p16_120m_2020-2022.tif",
        "olm_bdod_mean_30m_2020-2022.tif",
        "olm_bdod_p16_120m_2020-2022.tif",
    }
    bands, units = writer.files["olm_soc_mean_30m_2020-2022.tif"]
    assert units == "g/kg"
    assert [b[0] for b in bands] == [
        "soc_0-30cm_mean_30m_2020-2022 (g/kg)",
        "soc_60-100cm_mean_30m_2020-2022 (g/kg)",
    ]
    assert bands[0][1][1, 1] == pytest.approx(30.0)  # 300 x 0.1 (file scale)
    assert math.isnan(bands[0][1][0, 0])
    bd, bd_units = writer.files["olm_bdod_mean_30m_2020-2022.tif"]
    assert bd_units == "g/cm3" and bd[0][1][1, 1] == pytest.approx(3.0)
    clay, _ = writer.files["olm_clay_mean_30m_2020-2022.tif"]
    assert clay[0][1][1, 1] == pytest.approx(15.0)  # scale 1 (none needed)
    recs = {r.layer: r for r in result.layers}
    assert recs["soc_0-30cm_mean_30m_2020-2022"].scale_source == "file metadata"
    assert recs["clay_0-30cm_mean_30m_2020-2022"].scale_source == "none needed"
    assert recs["soc_0-30cm_p16_120m_2020-2022"].resolution_m == 120
    assert {r.route for r in result.layers} == {"COG (https)"}
    assert sorted(os.path.basename(p) for p in result.load_files) == [
        "olm_bdod_mean_30m_2020-2022.tif",
        "olm_clay_mean_30m_2020-2022.tif",
        "olm_soc_mean_30m_2020-2022.tif",
    ]


def test_area_catalogue_scale_used_when_file_has_none(grid, tmp_path):
    def reader(source, grid, with_scale=False):
        arr, _, _ = make_reader()(source, grid, True)
        return arr, None, None

    sel = core.plan_selection(["phh2o"], ["0-30cm"], ["2020-2022"], ["mean30"])
    result, writer = run_area(grid, tmp_path, sel, reader=reader)
    bands, _ = writer.files["olm_phh2o_mean_30m_2020-2022.tif"]
    assert bands[0][1][1, 1] == pytest.approx(30.0)  # 300 x 0.1 catalogue
    assert result.layers[0].scale_source.startswith("catalogue")


def test_area_multi_period_soc_and_static_texture(grid, tmp_path):
    sel = core.plan_selection(
        ["clay", "soc"], ["0-30cm"], ["2010-2015", "2020-2022"], ["mean30"]
    )
    result, writer = run_area(grid, tmp_path, sel)
    assert set(writer.files) == {
        "olm_clay_mean_30m_2020-2022.tif",
        "olm_soc_mean_30m_2010-2015.tif",
        "olm_soc_mean_30m_2020-2022.tif",
    }
    assert any("static" in n for n in result.notes)
    loaded = sorted(os.path.basename(p) for p in result.load_files)
    assert loaded == [
        "olm_clay_mean_30m_2020-2022.tif",
        "olm_soc_mean_30m_2020-2022.tif",
    ]


def test_area_ru68(grid, tmp_path):
    sel = core.plan_selection(["soc"], ["0-30cm"], ["2020-2022"], ["mean30"], True)
    result, writer = run_area(grid, tmp_path, sel)
    assert set(writer.files) == {
        "olm_soc_mean_30m_2020-2022.tif",
        "olm_soc_RU68_120m_2020-2022.tif",
    }
    bands, units = writer.files["olm_soc_RU68_120m_2020-2022.tif"]
    assert units == "ratio"
    assert bands[0][1][1, 1] == pytest.approx((50.0 - 20.0) / 31.0)
    assert result.uncertainty[0]["median_ru68"] == pytest.approx(30.0 / 31.0)


def test_area_texture_check(grid, tmp_path):
    sel = core.plan_selection(
        ["sand", "silt", "clay"], ["0-30cm"], ["2020-2022"], ["mean30"]
    )
    result, _ = run_area(grid, tmp_path, sel)
    assert result.texture == [
        {
            "period": "2020-2022",
            "depth": "0-30cm",
            "checked": grid.width * grid.height - 1,
            "flagged": 0,
            "max_abs_dev": 0.0,
        }
    ]


def test_area_https_failure_falls_back_to_http(grid, tmp_path):
    calls, logs = [], []
    sel = core.plan_selection(["soc"], ["0-30cm"], ["2020-2022"], ["mean30"])
    result, _ = run_area(
        grid,
        tmp_path,
        sel,
        reader=make_reader(fail_https=True, calls=calls),
        log_fn=logs.append,
    )
    assert [c.split(":")[0] for c in calls] == ["/vsicurl/https", "/vsicurl/http"]
    assert result.layers[0].route == "COG (http)"
    assert logs and "HTTP 503" in logs[0]


def test_area_all_routes_fail(grid, tmp_path):
    sel = core.plan_selection(["soc"], ["0-30cm"], ["2020-2022"], ["mean30"])
    with pytest.raises(core.SoilDataError, match="All access routes failed"):
        run_area(grid, tmp_path, sel, reader=make_reader(fail_all=True))


def test_area_nothing_published(grid, tmp_path):
    sel = core.plan_selection(["bdod"], ["0-30cm"], ["2000-2005"], ["mean30"])
    with pytest.raises(core.SoilDataError, match="Nothing to read"):
        run_area(grid, tmp_path, sel)


def test_area_empty_layer_warned(grid, tmp_path):
    sel = core.plan_selection(["soc"], ["0-30cm"], ["2020-2022"], ["mean30"])
    result, _ = run_area(grid, tmp_path, sel, reader=make_reader(empty=True))
    assert any("no valid values" in w and "desert" in w for w in result.warnings)


def test_area_cancel_and_progress(grid, tmp_path):
    sel = core.plan_selection(["soc"], ["0-30cm", "30-60cm"], ["2020-2022"], ["mean30"])
    with pytest.raises(InterruptedError):
        run_area(grid, tmp_path, sel, cancel_fn=lambda: True)
    seen = []
    run_area(grid, tmp_path, sel, progress_fn=lambda f, m: seen.append(f))
    assert seen[-1] == 1.0 and max(seen[:-1]) < 1.0


def test_area_parallel_equals_sequential(grid, tmp_path):
    sel = core.plan_selection(
        [v.code for v in cat.VARIABLES],
        [d[0] for d in cat.DEPTHS],
        list(cat.PERIODS),
        [s.code for s in cat.STATISTICS],
        True,
    )
    _, w1 = run_area(grid, tmp_path, sel, workers=1)
    _, w8 = run_area(grid, tmp_path, sel, workers=8)
    assert set(w1.files) == set(w8.files)
    for name, (bands, _) in w1.files.items():
        other = w8.files[name][0]
        assert [b[0] for b in bands] == [b[0] for b in other]
        assert all(
            np.array_equal(a[1], b[1], equal_nan=True)
            for a, b in zip(bands, other, strict=True)
        )


# ----------------------------------------------------------------------
# Point mode
# ----------------------------------------------------------------------

SITES = [core.Site("A", 28.10, -25.80), core.Site("B", 28.12, -25.81)]


def make_sampler(missing_second=False):
    def sampler(source, lonlats, with_scale=False):
        assert with_scale is True
        value, scale, offset = fake_value(source)
        values = [value] * len(lonlats)
        if missing_second:
            values[1] = math.nan
        return values, scale, offset

    return sampler


def run_points(selection, **kw):
    return core.extract_points(
        selection,
        kw.pop("sites", SITES),
        sample_fn=kw.pop("sampler", make_sampler()),
        options_ctx=contextlib.nullcontext(),
        **kw,
    )


def test_points_rows(tmp_path):
    sel = core.plan_selection(["soc"], ["0-30cm"], ["2020-2022"], ["mean30", "p84"])
    result = run_points(sel)
    assert len(result.rows) == 4
    row = result.rows[0]
    assert row["Product"] == "OpenLandMap-soildb"
    assert (row["DepthTop_cm"], row["DepthBottom_cm"]) == (0, 30)
    assert row["Period"] == "2020-2022"
    assert row["Value"] == pytest.approx(30.0)
    assert {r["Resolution_m"] for r in result.rows} == {30, 120}
    path = tmp_path / "p.csv"
    assert write_points_csv(result.rows, path, core.POINT_COLUMNS) == (4, 0)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == core.POINT_COLUMNS


def test_points_missing_and_ru68():
    sel = core.plan_selection(["soc"], ["0-30cm"], ["2020-2022"], [], True)
    result = run_points(sel, sampler=make_sampler(missing_second=True))
    ru = [r for r in result.rows if r["Statistic"] == cat.RU68]
    assert len(ru) == 2
    assert ru[0]["Value"] == pytest.approx(30.0 / 31.0)
    assert math.isnan(ru[1]["Value"])


def test_points_texture_check():
    sel = core.plan_selection(
        ["sand", "silt", "clay"], ["0-30cm"], ["2010-2015"], ["mean30"]
    )
    result = run_points(sel)
    assert result.texture[0]["period"] == "2020-2022"
    assert result.texture[0]["flagged"] == 0


def test_points_requires_sites():
    sel = core.plan_selection(["soc"], ["0-30cm"], ["2020-2022"], ["mean30"])
    with pytest.raises(core.SoilDataError, match="No points"):
        run_points(sel, sites=[])


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


def test_metadata(grid, tmp_path):
    sel = core.plan_selection(
        ["sand", "silt", "clay", "bdod"], ["0-30cm"], ["2015-2020"], ["mean30"], True
    )
    result, _ = run_area(grid, tmp_path, sel)
    sections = core.build_metadata(result, sel, {"Output CRS": "EPSG:2049"}, "3.13")
    titles = [s[0] for s in sections]
    assert titles[0] == "Run" and "Notes (periods)" in titles
    run = dict(sections[0][2])
    assert "Hengl" in run["Citation"] and run["Licence"] == "CC-BY 4.0"
    assert "ISO 11277" in run["Note - texture limits"]
    assert "120 m only" in run["Note - statistics"]
    notes = next(s for s in sections if s[0] == "Notes (periods)")[2]
    assert any("bdod" in n[0] and "not published" in n[0] for n in notes)
    layers = next(s for s in sections if s[0] == "Layers")
    assert layers[1][10] == "Scale source"


# ----------------------------------------------------------------------
# Real GDAL with OpenLandMap-style local files (no network)
# ----------------------------------------------------------------------


def _make_cog_like(gdal, osr, path, arr, scale, nodata, dtype):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), arr.shape[1], arr.shape[0], 1, dtype, options=["TILED=YES"]
    )
    ds.SetGeoTransform((28.0, 0.00025, 0, -25.70, 0, -0.00025))  # 30 m-ish
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    if scale is not None:
        band.SetScale(scale)
        band.SetOffset(0.0)
    band.WriteArray(arr)
    ds = None


def test_real_gdal_reads_file_scale_and_nodata(tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    osr = pytest.importorskip("osgeo.osr")
    gdal.UseExceptions()
    osr.UseExceptions()
    from mayim_tools.soil._common.gdal_io import gdal_read_grid, gdal_sample_points

    soc = np.full((400, 400), 253, dtype=np.uint16)  # 25.3 g/kg at scale 0.1
    soc[10, 10] = 32767
    path = tmp_path / "soc.tif"
    _make_cog_like(gdal, osr, path, soc, 0.1, 32767, gdal.GDT_UInt16)

    utm = osr.SpatialReference()
    utm.ImportFromEPSG(32735)
    t = osr.CoordinateTransformation(_ll(osr), utm)
    x, y, _ = t.TransformPoint(28.04, -25.74)
    from mayim_tools.soil._common.grid import make_grid

    grid = make_grid((x - 1500, y - 1500, x + 1500, y + 1500), 30.0, utm.ExportToWkt())
    arr, scale, offset = gdal_read_grid(str(path), grid, with_scale=True)
    assert scale == pytest.approx(0.1) and offset == pytest.approx(0.0)
    values = core.apply_scale(arr, *core.resolve_scale(scale, offset, 0.1)[:2])
    assert np.nanmax(values) == pytest.approx(25.3)

    lon_nd = 28.0 + 10.5 * 0.00025
    lat_nd = -25.70 - 10.5 * 0.00025
    vals, s2, _ = gdal_sample_points(
        str(path), [(28.04, -25.74), (lon_nd, lat_nd)], with_scale=True
    )
    assert vals[0] == 253.0 and math.isnan(vals[1]) and s2 == pytest.approx(0.1)
    # Default (no with_scale) keeps the old return type.
    assert isinstance(gdal_sample_points(str(path), [(28.04, -25.74)]), list)


def test_real_gdal_end_to_end_area(tmp_path):
    """extract_area with a reader that maps the s3 URLs to local files."""
    gdal = pytest.importorskip("osgeo.gdal")
    osr = pytest.importorskip("osgeo.osr")
    gdal.UseExceptions()
    osr.UseExceptions()
    from mayim_tools.soil._common.export import write_multiband_geotiff
    from mayim_tools.soil._common.gdal_io import gdal_read_grid
    from mayim_tools.soil._common.grid import make_grid

    local = {}
    for depth, value in (("0-30cm", 200), ("30-60cm", 250)):
        clay = np.full((400, 400), value // 10, dtype=np.uint8)
        p = tmp_path / f"clay_{depth}.tif"
        _make_cog_like(gdal, osr, p, clay, None, 255, gdal.GDT_Byte)
        local[cat.cog_url("clay", "mean30", depth, "2020-2022")] = str(p)
        ph = np.full((400, 400), 65 + value // 100, dtype=np.uint8)
        q = tmp_path / f"ph_{depth}.tif"
        _make_cog_like(gdal, osr, q, ph, 0.1, 255, gdal.GDT_Byte)
        local[cat.cog_url("phh2o", "mean30", depth, "2020-2022")] = str(q)

    def reader(source, grid, with_scale=False):
        url = source.replace("/vsicurl/", "")
        return gdal_read_grid(local[url], grid, with_scale=with_scale)

    utm = osr.SpatialReference()
    utm.ImportFromEPSG(32735)
    t = osr.CoordinateTransformation(_ll(osr), utm)
    x, y, _ = t.TransformPoint(28.04, -25.74)
    grid = make_grid((x - 900, y - 900, x + 900, y + 900), 30.0, utm.ExportToWkt())
    sel = core.plan_selection(
        ["clay", "phh2o"], ["0-30cm", "30-60cm"], ["2020-2022"], ["mean30"]
    )
    out = tmp_path / "out"
    out.mkdir()
    result = core.extract_area(
        sel,
        grid,
        str(out),
        read_grid_fn=reader,
        write_fn=write_multiband_geotiff,
        workers=4,
        options_ctx=contextlib.nullcontext(),
    )
    assert not result.warnings
    ds = gdal.Open(str(out / "olm_phh2o_mean_30m_2020-2022.tif"))
    assert ds.RasterCount == 2
    assert ds.GetRasterBand(2).GetDescription() == (
        "phh2o_30-60cm_mean_30m_2020-2022 (pH)"
    )
    assert np.allclose(ds.GetRasterBand(1).ReadAsArray(), 6.7)  # 67 x 0.1
    assert np.allclose(ds.GetRasterBand(2).ReadAsArray(), 6.7)
    ds = None
    ds = gdal.Open(str(out / "olm_clay_mean_30m_2020-2022.tif"))
    assert np.allclose(ds.GetRasterBand(1).ReadAsArray(), 20.0)
    assert np.allclose(ds.GetRasterBand(2).ReadAsArray(), 25.0)
    ds = None
    assert (out / "olm_clay_mean_30m_2020-2022.qml").exists()


def _ll(osr):
    s = osr.SpatialReference()
    s.ImportFromEPSG(4326)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


# ----------------------------------------------------------------------
# USDA subgroups
# ----------------------------------------------------------------------

from mayim_tools.soil.openlandmap_extract.soil_types import SUBGROUP_NAMES  # noqa: E402

SOIL_TYPES_CSV = Path(__file__).parent / "data" / "openlandmap_soil_types.csv"


def test_subgroup_urls_match_official_table_exactly():
    with open(SOIL_TYPES_CSV, newline="", encoding="utf-8") as f:
        official = {row["s3_path"] for row in csv.DictReader(f)}
    built = {cat.subgroup_url(n) for n in SUBGROUP_NAMES}
    assert len(SUBGROUP_NAMES) == 818
    assert built == official


def test_subgroup_names_and_orders():
    assert cat.subgroup_label("typic.haplustalfs") == "Typic Haplustalfs"
    assert cat.subgroup_label("ruptic-histic.aquiturbels") == (
        "Ruptic-histic Aquiturbels"
    )
    assert cat.great_group("aquic.dystruderts") == "Dystruderts"
    expected = {
        "typic.haplustalfs": "Alfisols",
        "typic.hapludands": "Andisols",
        "typic.haplocambids": "Aridisols",
        "typic.ustorthents": "Entisols",
        "typic.haplorthels": "Gelisols",
        "typic.haplosaprists": "Histosols",
        "typic.haplustepts": "Inceptisols",
        "typic.haplustolls": "Mollisols",
        "typic.hapludox": "Oxisols",
        "typic.haplorthods": "Spodosols",
        "typic.haplustults": "Ultisols",
        "typic.haplusterts": "Vertisols",
    }
    for name, order in expected.items():
        assert cat.soil_order(name) == order, name
    assert all(cat.soil_order(n) != "Unknown" for n in SUBGROUP_NAMES)


def test_top2_update_nan_and_ties():
    shape = (4,)
    state = {
        "p1": np.full(shape, -1.0),
        "p2": np.full(shape, -1.0),
        "code1": np.zeros(shape, dtype=np.int32),
        "code2": np.zeros(shape, dtype=np.int32),
        "psum": np.zeros(shape),
        "n": np.zeros(shape, dtype=np.int32),
    }
    core.top2_update(state, np.array([10.0, 50.0, np.nan, 30.0]), 1)
    core.top2_update(state, np.array([40.0, 50.0, np.nan, 20.0]), 2)
    core.top2_update(state, np.array([20.0, 10.0, np.nan, 25.0]), 3)
    assert state["code1"].tolist() == [2, 1, 0, 1]  # tie at cell 1: lower code
    assert state["code2"].tolist() == [3, 2, 0, 3]
    assert state["p1"].tolist()[:2] == [40.0, 50.0]
    assert state["p2"].tolist()[3] == 25.0
    assert state["psum"].tolist() == [70.0, 110.0, 0.0, 75.0]
    assert state["n"].tolist() == [3, 3, 0, 3]


HAPLUSTALFS = SUBGROUP_NAMES.index("typic.haplustalfs") + 1
HAPLORTHODS = SUBGROUP_NAMES.index("typic.haplorthods") + 1


def _subgroup_field(url, shape):
    """Left half: Haplustalfs 60 %, Haplorthods 30 %; right half the reverse
    (50 / 35); every other subgroup 0 %; top-left cell unmapped."""
    left = np.zeros(shape, dtype=bool)
    left[..., : shape[-1] // 2] = True
    arr = np.zeros(shape)
    if "typic.haplustalfs_p_" in url:
        arr = np.where(left, 60.0, 35.0)
    elif "typic.haplorthods_p_" in url:
        arr = np.where(left, 30.0, 50.0)
    arr = arr.astype(float)
    arr.flat[0] = np.nan
    return arr


def subgroup_reader(source, grid, with_scale=False):
    if "soil.types_ensemble." in source:
        return _subgroup_field(source, (grid.height, grid.width)), None, None
    return make_reader()(source, grid, with_scale)


def test_plan_selection_subgroups_only():
    sel = core.plan_selection([], [], [], [], subgroups=True)
    assert sel.subgroups and sel.variables == []
    with pytest.raises(
        core.SoilDataError, match="USDA subgroup or water-content option"
    ):
        core.plan_selection([], [], [], [])


def test_area_subgroups(grid, tmp_path):
    sel = core.plan_selection([], [], [], [], subgroups=True)
    seen = []
    result, writer = run_area(
        grid,
        tmp_path,
        sel,
        reader=subgroup_reader,
        progress_fn=lambda f, m: seen.append(f),
    )
    bands, units = writer.files[core.SUBGROUP_FILE]
    assert [b[0].split(" (")[0] for b in bands] == [
        "usda_subgroup_code_rank1_2000-2022",
        "usda_subgroup_probability_rank1_2000-2022",
        "usda_subgroup_code_rank2_2000-2022",
        "usda_subgroup_probability_rank2_2000-2022",
        "usda_subgroup_probability_sum_2000-2022",
    ]
    code1, p1, code2, p2, psum = (b[1] for b in bands)
    assert math.isnan(code1[0, 0]) and math.isnan(p1[0, 0])
    assert code1[1, 0] == HAPLUSTALFS and p1[1, 0] == 60.0
    assert code2[1, 0] == HAPLORTHODS and p2[1, 0] == 30.0
    assert code1[1, -1] == HAPLORTHODS and p1[1, -1] == 50.0
    assert psum[1, 0] == 90.0
    table = result.subgroup["table"]
    assert {r["code"] for r in table} == {HAPLUSTALFS, HAPLORTHODS}
    assert table[0]["order"] in ("Alfisols", "Spodosols")
    assert sum(r["cells"] for r in table) == result.subgroup["mapped"]
    assert result.load_files[-1].endswith(core.SUBGROUP_FILE)
    assert len(seen) > 800 and seen[-1] == 1.0


def test_area_subgroups_with_properties_and_parallel_equals_sequential(grid, tmp_path):
    sel = core.plan_selection(
        ["soc"], ["0-30cm"], ["2020-2022"], ["mean30"], subgroups=True
    )
    r1, w1 = run_area(grid, tmp_path, sel, reader=subgroup_reader, workers=1)
    r8, w8 = run_area(grid, tmp_path, sel, reader=subgroup_reader, workers=8)
    assert set(w1.files) == {"olm_soc_mean_30m_2020-2022.tif", core.SUBGROUP_FILE}
    a = [b[1] for b in w1.files[core.SUBGROUP_FILE][0]]
    b = [b[1] for b in w8.files[core.SUBGROUP_FILE][0]]
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b, strict=True))
    assert r1.subgroup["table"] == r8.subgroup["table"]


def test_points_subgroups():
    def sampler(source, lonlats, with_scale=False):
        if "soil.types_ensemble." in source:
            arr = _subgroup_field(source, (1, 2 * len(lonlats)))
            # site 0 -> left column 0 (unmapped by construction), site 1 -> col 1
            values = [float(arr[0, i]) for i in range(len(lonlats))]
            return values, None, None
        return make_sampler()(source, lonlats, with_scale)

    sel = core.plan_selection([], [], [], [], subgroups=True)
    result = run_points(sel, sampler=sampler)
    rows = [r for r in result.rows if r["Variable"] == "usda_subgroup"]
    assert len(rows) == 4  # 2 ranks x 2 sites
    site_b = [r for r in rows if r["Site"] == "B"]
    assert "Typic Haplustalfs" in site_b[0]["Description"]
    assert "Alfisols" in site_b[0]["Description"]
    assert site_b[0]["Value"] == 60.0 and site_b[1]["Value"] == 30.0
    assert site_b[0]["DepthTop_cm"] == "" and site_b[0]["Period"] == "2000-2022"
    site_a = [r for r in rows if r["Site"] == "A"]
    assert site_a[0]["Description"] == "not mapped"
    assert math.isnan(site_a[0]["Value"])


def test_metadata_subgroup_section(grid, tmp_path):
    sel = core.plan_selection([], [], [], [], subgroups=True)
    result, _ = run_area(grid, tmp_path, sel, reader=subgroup_reader)
    sections = core.build_metadata(result, sel, {}, "3.13")
    sub = next(s for s in sections if s[0].startswith("USDA subgroups"))
    labels = [row[1] for row in sub[2]]
    assert "Typic Haplustalfs" in labels and "(all mapped cells)" in labels
    run = dict(sections[0][2])
    assert "818" in run["USDA subgroups"]


def test_paletted_style(tmp_path):
    import xml.etree.ElementTree as ET

    from mayim_tools.soil._common.export import class_colour, write_paletted_style

    qml = write_paletted_style(
        tmp_path / "s.tif", [(5, 'Typic "A" & B'), (812, "Lithic C")]
    )
    root = ET.parse(qml).getroot()
    renderer = root.find("pipe/rasterrenderer")
    assert renderer.get("type") == "paletted" and renderer.get("band") == "1"
    entries = [(e.get("value"), e.get("label")) for e in root.iter("paletteEntry")]
    assert entries == [("5", 'Typic "A" & B'), ("812", "Lithic C")]
    assert class_colour(5) == class_colour(5) != class_colour(6)


# ----------------------------------------------------------------------
# Legacy 250 m water content (33 / 1500 kPa) and AWC
# ----------------------------------------------------------------------


def test_legacy_urls_as_confirmed_live():
    """Exact paths confirmed by the live probe (2026-10-05) for 33 kPa; the
    1500 kPa names follow the same pattern (catalogue / Zenodo record)."""
    arco, zenodo = cat.legacy_urls("wc33", 0)
    assert arco == (
        "https://s3.openlandmap.org/arco/watercontent.33kPa_usda.4b1c_m_250m_b0cm_"
        "19500101_20171231_go_epsg.4326_v0.1.tif"
    )
    assert zenodo == (
        "https://zenodo.org/records/2784001/files/sol_watercontent.33kPa_usda.4b1c"
        "_m_250m_b0..0cm_1950..2017_v0.1.tif"
    )
    arco, zenodo = cat.legacy_urls("wc1500", 60)
    assert "watercontent.1500kPa_usda.3c2a1a_m_250m_b60cm_" in arco
    assert zenodo.endswith(
        "sol_watercontent.1500kPa_usda.3c2a1a_m_250m_b60..60cm_1950..2017_v0.1.tif"
    )
    assert cat.legacy_points_needed(["30-60cm"]) == [30, 60]
    assert cat.legacy_points_needed([d[0] for d in cat.DEPTHS]) == [0, 30, 60, 100]


def test_interval_mean_and_awc():
    pts = {0: np.array([20.0, np.nan]), 30: np.array([30.0, 10.0])}
    assert core.interval_mean(pts, "0-30cm")[0] == 25.0
    assert math.isnan(core.interval_mean(pts, "0-30cm")[1])
    # FC 30 %, WP 12 % over 0-30 cm (300 mm): 0.18 x 300 = 54 mm
    out = core.awc_mm(
        np.array([30.0, 10.0, np.nan]), np.array([12.0, 15.0, 5.0]), "0-30cm"
    )
    assert out[0] == pytest.approx(54.0)
    assert out[1] == 0.0  # negative difference clipped
    assert math.isnan(out[2])
    # 60-100 cm is 400 mm thick
    assert core.awc_mm(np.array([25.0]), np.array([10.0]), "60-100cm")[0] == 60.0


WATER = {("wc33", 0): 20, ("wc33", 30): 24, ("wc1500", 0): 8, ("wc1500", 30): 10}


def water_reader(fail_arco=False, calls=None):
    def reader(source, grid, with_scale=False):
        if "watercontent" not in source:
            return make_reader()(source, grid, with_scale)
        if calls is not None:
            calls.append(source)
        if fail_arco and "s3.openlandmap.org" in source:
            raise RuntimeError("HTTP 404")
        code = "wc33" if "33kPa" in source else "wc1500"
        depth = 0 if ("_b0cm_" in source or "_b0..0cm_" in source) else 30
        arr = np.full((grid.height, grid.width), float(WATER[(code, depth)]))
        arr[0, 0] = np.nan
        return arr, None, None

    return reader


def test_area_water_content_and_awc(grid, tmp_path):
    sel = core.plan_selection([], ["0-30cm"], [], [], water=True)
    result, writer = run_area(grid, tmp_path, sel, reader=water_reader())
    assert set(writer.files) == {
        "olm_wc33_250m_1950-2017.tif",
        "olm_wc1500_250m_1950-2017.tif",
        "olm_awc_250m_1950-2017.tif",
    }
    fc, units = writer.files["olm_wc33_250m_1950-2017.tif"]
    assert units == "vol %"
    assert fc[0][0] == "wc33_0-30cm_mean_250m_1950-2017 (vol %)"
    assert fc[0][1][1, 1] == 22.0  # (20 + 24) / 2
    wp, _ = writer.files["olm_wc1500_250m_1950-2017.tif"]
    assert wp[0][1][1, 1] == 9.0
    awc, awc_units = writer.files["olm_awc_250m_1950-2017.tif"]
    assert awc_units == "mm" and awc[0][1][1, 1] == pytest.approx(39.0)  # 13 % x 300
    assert math.isnan(awc[0][1][0, 0])
    routes = {r.variable: r.route for r in result.layers}
    assert routes == {
        "wc33": "COG (s3.openlandmap.org)",
        "wc1500": "COG (s3.openlandmap.org)",
        "awc": "derived",
    }


def test_area_water_falls_back_to_zenodo(grid, tmp_path):
    calls, logs = [], []
    sel = core.plan_selection([], ["0-30cm"], [], [], water=True)
    result, _ = run_area(
        grid,
        tmp_path,
        sel,
        reader=water_reader(fail_arco=True, calls=calls),
        log_fn=logs.append,
    )
    assert any("zenodo.org" in c for c in calls)
    assert {r.route for r in result.layers if r.variable == "wc33"} == {
        "Zenodo (slower)"
    }
    assert logs and "HTTP 404" in logs[0]


def test_points_water(tmp_path):
    def sampler(source, lonlats, with_scale=False):
        if "watercontent" in source:
            code = "wc33" if "33kPa" in source else "wc1500"
            depth = 0 if "_b0cm_" in source else 30
            return [float(WATER[(code, depth)])] * len(lonlats), None, None
        return make_sampler()(source, lonlats, with_scale)

    sel = core.plan_selection([], ["0-30cm"], [], [], water=True)
    result = run_points(sel, sampler=sampler)
    by = {(r["Site"], r["Variable"]): r for r in result.rows}
    assert by[("A", "wc33")]["Value"] == 22.0
    assert by[("A", "awc")]["Value"] == pytest.approx(39.0)
    assert by[("A", "awc")]["Units"] == "mm"
    assert by[("A", "wc1500")]["Period"] == "1950-2017"
    assert by[("A", "wc33")]["Resolution_m"] == 250


def test_metadata_water(grid, tmp_path):
    sel = core.plan_selection([], ["0-30cm"], [], [], water=True)
    result, _ = run_area(grid, tmp_path, sel, reader=water_reader())
    run = dict(core.build_metadata(result, sel, {}, "3.13")[0][2])
    assert "CC BY-SA 4.0" in run["Licence (water content)"]
    assert "2784001" in run["Data DOI (water content)"]
    assert "measured" in run["Note - water content"]
