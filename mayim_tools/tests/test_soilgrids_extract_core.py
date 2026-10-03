"""
Tests for mayim_tools/soil/soilgrids_extract (core.py and export.py).

No network is used anywhere: the ISRIC server is replaced by injected
fake readers/samplers. The default GDAL readers are exercised against
LOCAL rasters (including one in the same Interrupted Goode Homolosine
projection SoilGrids uses), skipped when osgeo is not available.
"""

import csv
import math
import os

import numpy as np
import pytest

from mayim_tools.soil.soilgrids_extract import core
from mayim_tools.soil.soilgrids_extract.export import (
    metadata_path_for,
    write_metadata_csv,
    write_points_csv,
)

# ----------------------------------------------------------------------
# Catalogue, units, names
# ----------------------------------------------------------------------


def test_conversion_factors_match_isric():
    factors = {v.code: v.factor for v in core.VARIABLES}
    assert factors == {
        "sand": 10,
        "silt": 10,
        "clay": 10,
        "soc": 10,
        "bdod": 100,
        "cfvo": 10,  # ISRIC FAQ (one third-party catalogue says 100 - wrong)
        "phh2o": 10,
        "cec": 10,
        "wv0010": 10,
        "wv0033": 10,
        "wv1500": 10,
    }
    assert core.VARIABLE_BY_CODE["bdod"].units == "g/cm3"
    assert core.VARIABLE_BY_CODE["clay"].units == "%"


def test_convert_examples_and_nodata_stays_nan():
    raw = np.array([250.0, np.nan, 0.0])
    out = core.convert(raw, core.VARIABLE_BY_CODE["clay"].factor)
    assert out[0] == pytest.approx(25.0)
    assert math.isnan(out[1])  # nodata is never turned into zero
    assert out[2] == 0.0
    assert core.convert(np.array([135.0]), 100)[0] == pytest.approx(1.35)


def test_defaults_exclude_optional_layers():
    defaults = [v.code for v in core.VARIABLES if v.default]
    assert defaults == ["sand", "silt", "clay", "soc", "bdod", "cfvo", "phh2o", "cec"]


def test_webdav_path_pattern():
    assert core.webdav_path("clay", "0-5cm", "Q0.5") == (
        "/vsicurl/https://files.isric.org/soilgrids/latest/data/"
        "clay/clay_0-5cm_Q0.5.vrt"
    )
    assert core.webdav_path("soc", "15-30cm", "mean").endswith(
        "soc/soc_15-30cm_mean.vrt"
    )


def test_every_layer_name_is_well_formed():
    for v in core.VARIABLES:
        for depth, _, _ in core.DEPTHS:
            for stat in core.STATISTICS:
                name = core.layer_name(v.code, depth, stat)
                assert name == f"{v.code}_{depth}_{stat}"
                assert " " not in core.webdav_path(v.code, depth, stat)


def test_wcs_path_contents():
    path = core.wcs_path("sand", "30-60cm", "Q0.95", (28.1, -25.9, 28.4, -25.6))
    assert path.startswith("/vsicurl/https://maps.isric.org/mapserv?map=/map/sand.map")
    assert "COVERAGEID=sand_30-60cm_Q0.95" in path
    assert "SUBSET=long(28.100000,28.400000)" in path
    assert "SUBSET=lat(-25.900000,-25.600000)" in path
    assert "EPSG/0/4326" in path


def test_bedrock_paths():
    assert core.bedrock_path("BDTICM") == (
        "/vsicurl/https://files.isric.org/soilgrids/former/2017-03-10/data/"
        "BDTICM_M_250m_ll.tif"
    )
    assert [b.code for b in core.BEDROCK_LAYERS] == ["BDTICM", "BDRICM", "BDRLOG"]


def test_stat_tag_and_file_name():
    assert core.output_file_name("clay", "Q0.5") == "soilgrids_clay_Q0.50.tif"
    assert core.output_file_name("clay", "mean") == "soilgrids_clay_mean.tif"
    assert core.output_file_name("clay", core.RU90) == "soilgrids_clay_RU90.tif"


def test_band_description():
    assert (
        core.band_description("clay", "30-60cm", "Q0.5", "%") == "clay_30-60cm_Q0.5 (%)"
    )


# ----------------------------------------------------------------------
# Selection planning
# ----------------------------------------------------------------------


def test_plan_selection_orders_by_catalogue():
    sel = core.plan_selection(["clay", "sand"], ["30-60cm", "0-5cm"], ["Q0.95", "mean"])
    assert sel.variables == ["sand", "clay"]
    assert sel.depths == ["0-5cm", "30-60cm"]
    assert sel.statistics == ["mean", "Q0.95"]
    assert sel.written_statistics == ["mean", "Q0.95"]


def test_plan_selection_ru90_fetches_missing_quantiles():
    sel = core.plan_selection(["clay"], ["0-5cm"], ["mean"], ru90=True)
    assert sel.statistics == ["mean", "Q0.05", "Q0.5", "Q0.95"]
    assert sel.written_statistics == ["mean"]
    assert sel.warnings and "Q0.05" in sel.warnings[0]


@pytest.mark.parametrize(
    "args, message",
    [
        ((["nope"], ["0-5cm"], ["mean"]), "Unknown SoilGrids variable"),
        ((["clay"], ["0-7cm"], ["mean"]), "Unknown depth"),
        ((["clay"], ["0-5cm"], ["Q0.25"]), "Unknown statistic"),
        ((["clay"], [], ["mean"]), "at least one depth"),
        ((["clay"], ["0-5cm"], []), "at least one statistic"),
        (([], ["0-5cm"], ["mean"]), "at least one variable"),
    ],
)
def test_plan_selection_errors(args, message):
    with pytest.raises(core.SoilGridsError, match=message):
        core.plan_selection(*args)


def test_plan_selection_bedrock_only_is_allowed():
    sel = core.plan_selection([], ["0-5cm"], ["mean"], bedrock=True)
    assert sel.bedrock and sel.variables == []


# ----------------------------------------------------------------------
# Area and grid
# ----------------------------------------------------------------------


def test_area_km2_one_degree_at_equator():
    assert core.area_km2((0, 0, 1, 1)) == pytest.approx(12364, rel=0.002)


def test_check_area_limit_and_message():
    with pytest.raises(core.SoilGridsError, match="Maximum area per run"):
        core.check_area((0, 0, 3, 3), 50000)
    assert core.check_area((0, 0, 3, 3), 0) > 50000  # 0 = no limit
    assert core.check_area((28, -26, 28.2, -25.8), 50000) < 500


def test_check_area_invalid_extent():
    with pytest.raises(core.SoilGridsError, match="invalid or empty"):
        core.check_area((10, 0, 5, 1), 50000)


def test_make_grid_buffers_and_snaps():
    g = core.make_grid((1010.0, 2020.0, 1990.0, 2980.0), 250.0, "WKT", buffer_cells=2)
    assert (g.xmin, g.ymin, g.xmax, g.ymax) == (500.0, 1500.0, 2500.0, 3500.0)
    assert (g.width, g.height) == (8, 8)
    assert g.geotransform == (500.0, 250.0, 0.0, 3500.0, 0.0, -250.0)


def test_make_grid_rejects_bad_inputs():
    with pytest.raises(core.SoilGridsError):
        core.make_grid((0, 0, 1, 1), 0, "WKT")
    with pytest.raises(core.SoilGridsError):
        core.make_grid((5, 0, 1, 1), 1, "WKT")


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------


def test_relative_width():
    ru = core.relative_width(
        np.array([10.0, 5.0, np.nan, 1.0]),
        np.array([20.0, 0.0, 3.0, 4.0]),
        np.array([30.0, 9.0, 4.0, 9.0]),
    )
    assert ru[0] == pytest.approx(1.0)
    assert math.isnan(ru[1])  # Q0.5 = 0
    assert math.isnan(ru[2])  # NaN input
    assert ru[3] == pytest.approx(2.0)


def test_texture_sum_check():
    out = core.texture_sum_check(
        np.array([40.0, 40.0, np.nan]),
        np.array([40.0, 40.0, 10.0]),
        np.array([20.0, 25.0, 10.0]),
        tol=2.0,
    )
    assert out == {"checked": 2, "flagged": 1, "max_abs_dev": 5.0}


def test_summarise_counts_nodata():
    s = core.summarise(np.array([[1.0, np.nan], [3.0, 2.0]]))
    assert s["valid"] == 3 and s["nodata"] == 1
    assert (s["min"], s["median"], s["max"]) == (1.0, 2.0, 3.0)
    empty = core.summarise(np.array([np.nan]))
    assert empty["valid"] == 0 and math.isnan(empty["median"])


# ----------------------------------------------------------------------
# Area mode orchestration (fake reader / writer)
# ----------------------------------------------------------------------

RAW = {"mean": 300.0, "Q0.05": 200.0, "Q0.5": 300.0, "Q0.95": 500.0}
TEXTURE_MEAN = {"sand": 400.0, "silt": 400.0, "clay": 200.0}


def fake_raw_value(path):
    for stat, value in RAW.items():
        if path.endswith(f"_{stat}.vrt") or f"_{stat}&" in path:
            for code, tex in TEXTURE_MEAN.items():
                if stat == "mean" and f"/{code}_" in path:
                    return tex
            return value
    if "BDTICM" in path:
        return 1500.0
    if "BDRICM" in path:
        return 200.0
    if "BDRLOG" in path:
        return 12.0
    raise AssertionError(f"unexpected path {path}")


def make_reader(fail_webdav=False, fail_all=False, calls=None):
    def reader(path, grid):
        if calls is not None:
            calls.append(path)
        if fail_all or (fail_webdav and "files.isric.org/soilgrids/latest" in path):
            raise RuntimeError("HTTP 503")
        arr = np.full((grid.height, grid.width), fake_raw_value(path))
        arr[0, 0] = np.nan  # one nodata cell
        return arr

    return reader


class FakeWriter:
    def __init__(self):
        self.files = {}

    def __call__(self, path, grid, bands, units):
        self.files[path] = (bands, units)


@pytest.fixture
def grid():
    return core.make_grid((0, 0, 1000, 750), 250.0, "WKT", buffer_cells=0)


def run_area(grid, tmp_path, selection, **kw):
    writer = FakeWriter()
    result = core.extract_area(
        selection,
        grid,
        (28.0, -26.0, 28.01, -25.99),
        str(tmp_path),
        read_grid_fn=kw.pop("reader", make_reader()),
        write_fn=writer,
        **kw,
    )
    return result, writer


def test_area_bands_units_and_nodata(grid, tmp_path):
    sel = core.plan_selection(["clay", "bdod"], ["0-5cm", "30-60cm"], ["Q0.5"])
    result, writer = run_area(grid, tmp_path, sel)
    clay = str(tmp_path / "soilgrids_clay_Q0.50.tif")
    bands, units = writer.files[clay]
    assert units == "%"
    assert [b[0] for b in bands] == ["clay_0-5cm_Q0.5 (%)", "clay_30-60cm_Q0.5 (%)"]
    assert bands[0][1][1, 1] == pytest.approx(30.0)  # 300 g/kg -> 30 %
    assert math.isnan(bands[0][1][0, 0])  # nodata kept
    bdod_bands, bdod_units = writer.files[str(tmp_path / "soilgrids_bdod_Q0.50.tif")]
    assert bdod_units == "g/cm3"
    assert bdod_bands[0][1][1, 1] == pytest.approx(3.0)  # /100
    assert result.load_files == [clay, str(tmp_path / "soilgrids_bdod_Q0.50.tif")]
    assert len(result.layers) == 4
    assert all(rec.route == "WebDAV" for rec in result.layers)
    assert result.layers[0].stats["nodata"] == 1


def test_area_one_file_per_variable_statistic(grid, tmp_path):
    sel = core.plan_selection(
        ["sand", "silt", "clay"],
        [d[0] for d in core.DEPTHS],
        list(core.STATISTICS),
    )
    result, writer = run_area(grid, tmp_path, sel)
    assert len(writer.files) == 3 * 4
    for bands, _ in writer.files.values():
        assert len(bands) == 6


def test_area_loads_mean_when_no_median(grid, tmp_path):
    sel = core.plan_selection(["clay"], ["0-5cm"], ["mean", "Q0.95"])
    result, _ = run_area(grid, tmp_path, sel)
    assert result.load_files == [str(tmp_path / "soilgrids_clay_mean.tif")]


def test_area_ru90_written_and_extra_quantiles_not_written(grid, tmp_path):
    sel = core.plan_selection(["clay"], ["0-5cm"], ["mean"], ru90=True)
    result, writer = run_area(grid, tmp_path, sel)
    names = sorted(os.path.basename(p) for p in writer.files)
    assert names == ["soilgrids_clay_RU90.tif", "soilgrids_clay_mean.tif"]
    ru_bands, ru_units = writer.files[str(tmp_path / "soilgrids_clay_RU90.tif")]
    assert ru_units == "ratio"
    assert ru_bands[0][0] == "clay_0-5cm_RU90 (ratio)"
    assert ru_bands[0][1][1, 1] == pytest.approx((50.0 - 20.0) / 30.0)
    assert result.uncertainty[0]["median_ru90"] == pytest.approx(1.0)


def test_area_texture_check(grid, tmp_path):
    sel = core.plan_selection(["sand", "silt", "clay"], ["0-5cm"], ["mean"])
    result, _ = run_area(grid, tmp_path, sel)
    assert result.texture == [
        {"depth": "0-5cm", "checked": 11, "flagged": 0, "max_abs_dev": 0.0}
    ]


def test_area_fallback_to_wcs_is_logged(grid, tmp_path):
    calls, logs = [], []
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    result, writer = run_area(
        grid,
        tmp_path,
        sel,
        reader=make_reader(fail_webdav=True, calls=calls),
        log_fn=logs.append,
    )
    assert len(calls) == 2 and "maps.isric.org" in calls[1]
    assert result.layers[0].route == "WCS (EPSG:4326)"
    assert logs and "HTTP 503" in logs[0]
    assert result.warnings == logs


def test_area_all_routes_fail_names_layer(grid, tmp_path):
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    with pytest.raises(core.SoilGridsError, match="All access routes failed"):
        run_area(grid, tmp_path, sel, reader=make_reader(fail_all=True))


def test_area_bedrock_file(grid, tmp_path):
    sel = core.plan_selection([], ["0-5cm"], ["mean"], bedrock=True)
    result, writer = run_area(grid, tmp_path, sel)
    bands, _ = writer.files[str(tmp_path / core.BEDROCK_FILE_NAME)]
    assert [b[0].split()[0] for b in bands] == ["BDTICM", "BDRICM", "BDRLOG"]
    assert bands[0][1][1, 1] == 1500.0  # cm, no conversion
    assert result.load_files == [str(tmp_path / core.BEDROCK_FILE_NAME)]
    assert {rec.route for rec in result.layers} == {"WebDAV (2017 archive)"}


def test_area_cancel(grid, tmp_path):
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    with pytest.raises(InterruptedError):
        run_area(grid, tmp_path, sel, cancel_fn=lambda: True)


def test_area_progress_reaches_one(grid, tmp_path):
    seen = []
    sel = core.plan_selection(["clay"], ["0-5cm", "5-15cm"], ["Q0.5"])
    run_area(grid, tmp_path, sel, progress_fn=lambda f, m: seen.append(f))
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert seen == sorted(seen)


def test_area_requires_writer(grid, tmp_path):
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    with pytest.raises(core.SoilGridsError, match="writer"):
        core.extract_area(sel, grid, (0, 0, 1, 1), str(tmp_path))


# ----------------------------------------------------------------------
# Point mode orchestration (fake sampler)
# ----------------------------------------------------------------------

SITES = [core.Site("A", 28.10, -25.80), core.Site("B", 28.12, -25.81)]


def make_sampler(fail_webdav=False, missing_second=False, calls=None):
    def sampler(path, lonlats):
        if calls is not None:
            calls.append((path, len(lonlats)))
        if fail_webdav and "files.isric.org/soilgrids/latest" in path:
            raise RuntimeError("timeout")
        value = fake_raw_value(path)
        values = [value] * len(lonlats)
        if missing_second and len(values) > 1:
            values[1] = math.nan
        return values

    return sampler


def test_points_rows_long_format():
    sel = core.plan_selection(["clay"], ["0-5cm", "100-200cm"], ["Q0.5"])
    result = core.extract_points(sel, SITES, sample_fn=make_sampler())
    assert len(result.rows) == 2 * 2
    row = result.rows[0]
    assert row["Site"] == "A" and row["Variable"] == "clay"
    assert (row["DepthTop_cm"], row["DepthBottom_cm"]) == (0, 5)
    assert row["Value"] == pytest.approx(30.0)
    assert row["Units"] == "%" and row["Route"] == "WebDAV"
    assert result.rows[2]["DepthBottom_cm"] == 200


def test_points_missing_value_stays_nan():
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    result = core.extract_points(
        sel, SITES, sample_fn=make_sampler(missing_second=True)
    )
    assert math.isnan(result.rows[1]["Value"])


def test_points_wcs_fallback_groups_close_sites():
    calls = []
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    result = core.extract_points(
        sel, SITES, sample_fn=make_sampler(fail_webdav=True, calls=calls)
    )
    assert len(calls) == 2  # WebDAV attempt, then ONE WCS request for both
    assert "maps.isric.org" in calls[1][0] and calls[1][1] == 2
    assert result.rows[0]["Route"] == "WCS (EPSG:4326)"
    assert result.warnings


def test_point_groups_far_apart_split():
    far = [core.Site("A", 10, 0), core.Site("B", 20, 0)]
    assert core.point_groups(far) == [[0], [1]]
    assert core.point_groups(SITES) == [[0, 1]]


def test_points_ru90_rows():
    sel = core.plan_selection(["clay"], ["0-5cm"], [], ru90=True)
    result = core.extract_points(sel, SITES, sample_fn=make_sampler())
    assert {r["Statistic"] for r in result.rows} == {core.RU90}
    assert result.rows[0]["Value"] == pytest.approx(1.0)
    assert result.rows[0]["Route"] == "derived"


def test_points_bedrock_rows():
    sel = core.plan_selection([], ["0-5cm"], ["mean"], bedrock=True)
    result = core.extract_points(sel, SITES, sample_fn=make_sampler())
    assert [r["Variable"] for r in result.rows[::2]] == ["BDTICM", "BDRICM", "BDRLOG"]
    assert result.rows[0]["DepthTop_cm"] == ""


def test_points_texture_check():
    sel = core.plan_selection(["sand", "silt", "clay"], ["0-5cm"], ["mean"])
    result = core.extract_points(sel, SITES, sample_fn=make_sampler())
    assert result.texture[0]["checked"] == 2 and result.texture[0]["flagged"] == 0


def test_points_requires_sites():
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    with pytest.raises(core.SoilGridsError, match="No points"):
        core.extract_points(sel, [], sample_fn=make_sampler())


# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------


def test_metadata_contents(grid, tmp_path):
    sel = core.plan_selection(
        ["sand", "silt", "clay"], ["0-5cm"], ["mean"], ru90=True, bedrock=True
    )
    result, _ = run_area(grid, tmp_path, sel)
    sections = core.build_metadata(result, sel, {"Output CRS": "EPSG:2049"}, "3.11")
    titles = [s[0] for s in sections]
    assert titles[0] == "Run" and "Layers" in titles and titles[-1] == "Warnings"
    run = dict(sections[0][2])
    assert "Poggio" in run["Citation"] and "CC-BY 4.0" in run["Licence"]
    assert run["Output CRS"] == "EPSG:2049"
    assert "Shangguan" in run["Citation (depth to bedrock)"]
    assert "marginal" in run["Note - quantiles"]
    layers = next(s for s in sections if s[0] == "Layers")[2]
    assert {row[7] for row in layers} >= {"WebDAV", "derived"}
    ru = next(s for s in sections if s[0].startswith("Uncertainty"))[2]
    assert len(ru) == 3


def test_metadata_csv_and_paths(tmp_path):
    sections = [("Run", ["Item", "Value"], [["a", 1]]), ("Warnings", ["W"], [])]
    path = tmp_path / "m.csv"
    write_metadata_csv(sections, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["# Run", "Item,Value", "a,1"]
    assert metadata_path_for(tmp_path, "area").name == "soilgrids_metadata.csv"
    assert (
        metadata_path_for(tmp_path / "soil.csv", "points").name == "soil_metadata.csv"
    )


def test_points_csv_missing_is_empty(tmp_path):
    rows = [
        {
            "Site": "A",
            "Longitude": 28.1,
            "Latitude": -25.8,
            "Variable": "clay",
            "Description": "Clay",
            "DepthTop_cm": 0,
            "DepthBottom_cm": 5,
            "Statistic": "Q0.5",
            "Value": v,
            "Units": "%",
            "Route": "WebDAV",
        }
        for v in (30.0, math.nan)
    ]
    path = tmp_path / "p.csv"
    assert write_points_csv(rows, path) == (2, 1)
    with open(path, newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    assert data[0]["Value"] == "30.0" and data[1]["Value"] == ""


# ----------------------------------------------------------------------
# Real GDAL on local files (no network)
# ----------------------------------------------------------------------

IGH = "+proj=igh +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def _make_source(gdal, osr, path, srs_text, x0, y0, res, arr, nodata):
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), arr.shape[1], arr.shape[0], 1, gdal.GDT_Int16
    )
    ds.SetGeoTransform((x0, res, 0, y0, 0, -res))
    srs = osr.SpatialReference()
    srs.SetFromUserInput(srs_text)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    ds = None


def test_gdal_read_grid_and_sample_from_homolosine(tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    osr = pytest.importorskip("osgeo.osr")
    osr.UseExceptions()
    # A 40 x 40 cell, 250 m source grid in Homolosine around 28E, 25.8S,
    # value = 100 + column index, with one nodata cell.
    tr = osr.CoordinateTransformation(_srs(osr, "EPSG:4326"), _srs(osr, IGH))
    cx, cy, _ = tr.TransformPoint(28.1, -25.8)
    x0, y0 = math.floor(cx / 250) * 250 - 5000, math.floor(cy / 250) * 250 + 5000
    arr = np.tile(np.arange(40, dtype=np.int16) + 100, (40, 1))
    arr[20, 20] = -32768
    src = tmp_path / "clay_igh.tif"
    _make_source(gdal, osr, src, IGH, x0, y0, 250.0, arr, -32768)

    # Sample the cell centres of column 5 and the nodata cell, from lon/lat.
    inv = osr.CoordinateTransformation(_srs(osr, IGH), _srs(osr, "EPSG:4326"))
    lon5, lat5, _ = inv.TransformPoint(x0 + 5.5 * 250, y0 - 10.5 * 250)
    lon_nd, lat_nd, _ = inv.TransformPoint(x0 + 20.5 * 250, y0 - 20.5 * 250)
    values = core.gdal_sample_points(str(src), [(lon5, lat5), (lon_nd, lat_nd)])
    assert values[0] == 105.0
    assert math.isnan(values[1])
    far = core.gdal_sample_points(str(src), [(0.0, 0.0)])
    assert math.isnan(far[0])  # outside the raster

    # Warp to a grid in the SAME CRS and alignment: values must be exact.
    grid = core.TargetGrid(
        x0 + 1000, y0 - 3000, x0 + 3000, y0 - 1000, 250.0, _srs(osr, IGH).ExportToWkt()
    )
    out = core.gdal_read_grid(str(src), grid)
    assert out.shape == (8, 8)
    assert out[0, 0] == 104.0 and out[0, 7] == 111.0
    # Grid in UTM: nearest neighbour keeps values within the source's range.
    utm = _srs(osr, "EPSG:32735")
    t2 = osr.CoordinateTransformation(_srs(osr, "EPSG:4326"), utm)
    ux, uy, _ = t2.TransformPoint(lon5, lat5)
    g2 = core.make_grid((ux, uy, ux + 2000, uy + 2000), 250.0, utm.ExportToWkt())
    out2 = core.gdal_read_grid(str(src), g2)
    valid = out2[np.isfinite(out2)]
    assert valid.size and valid.min() >= 100 and valid.max() <= 139
    assert np.all(valid == np.round(valid))  # nearest: no interpolated values


def _srs(osr, text):
    s = osr.SpatialReference()
    s.SetFromUserInput(text)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def test_write_multiband_geotiff_roundtrip(tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    from mayim_tools.soil.soilgrids_extract.export import write_multiband_geotiff

    grid = core.TargetGrid(0, 0, 750, 500, 250.0, _wkt_utm())
    a = np.array([[1.5, np.nan, 3.0], [4.0, 5.0, 6.0]])
    b = a * 2
    path = tmp_path / "soilgrids_clay_Q0.50.tif"
    write_multiband_geotiff(
        path, grid, [("clay_0-5cm_Q0.5 (%)", a), ("clay_5-15cm_Q0.5 (%)", b)], "%"
    )
    ds = gdal.Open(str(path))
    assert ds.RasterCount == 2
    assert ds.GetGeoTransform() == grid.geotransform
    band = ds.GetRasterBand(1)
    assert band.GetDescription() == "clay_0-5cm_Q0.5 (%)"
    assert band.GetNoDataValue() == core.NODATA_OUT
    assert band.GetUnitType() == "%"
    data = band.ReadAsArray()
    assert data[0, 1] == core.NODATA_OUT and data[0, 0] == pytest.approx(1.5)
    assert ds.GetRasterBand(2).ReadAsArray()[1, 2] == pytest.approx(12.0)
    assert ds.GetMetadata("IMAGE_STRUCTURE").get("COMPRESSION") == "DEFLATE"


def test_write_multiband_geotiff_shape_mismatch(tmp_path):
    pytest.importorskip("osgeo.gdal")
    from mayim_tools.soil.soilgrids_extract.export import write_multiband_geotiff

    grid = core.TargetGrid(0, 0, 500, 500, 250.0, _wkt_utm())
    with pytest.raises(ValueError, match="shape"):
        write_multiband_geotiff(
            tmp_path / "x.tif",
            grid,
            [("a", np.zeros((2, 2))), ("b", np.zeros((3, 3)))],
            "%",
        )


def _wkt_utm():
    from osgeo import osr

    s = osr.SpatialReference()
    s.ImportFromEPSG(32735)
    return s.ExportToWkt()


def test_http_options_restored():
    gdal = pytest.importorskip("osgeo.gdal")
    gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", None)
    with core.http_options():
        assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") == "120"
    assert gdal.GetConfigOption("GDAL_HTTP_TIMEOUT") is None
