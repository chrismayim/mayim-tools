"""
Tests for mayim_tools/soil/soilgrids_extract (core.py and export.py).

No network is used anywhere: the ISRIC server is replaced by injected
fake readers/samplers. The default GDAL readers are exercised against
LOCAL rasters (including one in the same Interrupted Goode Homolosine
projection SoilGrids uses), skipped when osgeo is not available.
"""

import contextlib
import csv
import math
import os
import re
import threading

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


def test_every_layer_name_is_well_formed():
    for v in core.VARIABLES:
        for depth, _, _ in core.DEPTHS:
            for stat in core.STATISTICS:
                name = core.layer_name(v.code, depth, stat)
                assert name == f"{v.code}_{depth}_{stat}"
                assert " " not in core.layer_vrt(v.code, depth, stat)


def test_bedrock_paths():
    assert core.bedrock_path("BDTICM") == (
        "/vsicurl/https://files.isric.org/soilgrids/former/2017-03-10/data/"
        "BDTICM_M_250m_ll.tif"
    )
    assert [b.code for b in core.BEDROCK_LAYERS] == ["BDTICM", "BDRICM", "BDRLOG"]


def test_stat_tag_and_file_name():
    assert core.output_file_name("clay", "Q0.5") == "soilgrids_clay_Q0.50.tif"
    assert core.output_file_name("clay", "mean") == "soilgrids_clay_mean.tif"


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
# Paths, 2017 helpers, tile index
# ----------------------------------------------------------------------


def test_layer_vrt_pattern():
    assert core.layer_vrt("clay", "0-5cm", "Q0.5") == (
        "/vsicurl/https://files.isric.org/soilgrids/latest/data/"
        "clay/clay_0-5cm_Q0.5.vrt"
    )


def test_tile_paths_follow_isric_layout():
    # Layout as reported by GDAL LocationInfo on the live server (2026-10-03).
    index = core.TileIndex(
        ("tileSG-025-051/tileSG-025-051_2-2.tif",), "clay_0-5cm_mean"
    )
    assert index.paths_for("soc", "0-5cm", "mean") == [
        "/vsicurl/https://files.isric.org/soilgrids/latest/data/soc/./"
        "soc_0-5cm_mean/tileSG-025-051/tileSG-025-051_2-2.tif"
    ]


def test_layer_sources_order():
    index = core.TileIndex(("t/t_1-1.tif",), "clay_0-5cm_mean")
    srcs = core.layer_sources(index, "sand", "5-15cm", "Q0.95")
    assert [s.route for s in srcs] == ["Tiles", "VRT"]
    assert srcs[0].path[0].endswith("sand/./sand_5-15cm_Q0.95/t/t_1-1.tif")
    assert srcs[1].path.endswith("sand/sand_5-15cm_Q0.95.vrt")
    assert [s.route for s in core.layer_sources(None, "sand", "0-5cm", "mean")] == [
        "VRT"
    ]


VRT_XML = """<VRTDataset rasterXSize="400" rasterYSize="400">
  <VRTRasterBand dataType="Int16" band="1">
    <ComplexSource>
      <SourceFilename relativeToVRT="1">./clay_0-5cm_mean/tA/tA_1-1.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="200" ySize="200" />
      <DstRect xOff="0" yOff="0" xSize="200" ySize="200" />
    </ComplexSource>
    <ComplexSource>
      <SourceFilename relativeToVRT="1">./clay_0-5cm_mean/tA/tA_2-1.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="200" ySize="200" />
      <DstRect xOff="200" yOff="0" xSize="200" ySize="200" />
    </ComplexSource>
    <ComplexSource>
      <SourceFilename relativeToVRT="1">./clay_0-5cm_mean/tB/tB_1-2.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="200" ySize="200" />
      <DstRect xOff="0" yOff="200" xSize="200" ySize="200" />
    </ComplexSource>
  </VRTRasterBand>
</VRTDataset>"""


def test_tiles_in_window_selects_intersecting_tiles():
    assert core.tiles_in_window(VRT_XML, "clay_0-5cm_mean", (10, 10, 20, 20)) == (
        "tA/tA_1-1.tif",
    )
    assert core.tiles_in_window(VRT_XML, "clay_0-5cm_mean", (190, 190, 210, 210)) == (
        "tA/tA_1-1.tif",
        "tA/tA_2-1.tif",
        "tB/tB_1-2.tif",
    )
    # Window edge touching a tile edge does not select it (exclusive bounds).
    assert core.tiles_in_window(VRT_XML, "clay_0-5cm_mean", (0, 0, 200, 200)) == (
        "tA/tA_1-1.tif",
    )


def test_tiles_in_window_accepts_windows_backslash_paths():
    # GDAL on Windows writes relative VRT paths with backslashes.
    xml = VRT_XML.replace(
        "./clay_0-5cm_mean/tA/tA_1-1.tif", "clay_0-5cm_mean\\tA\\tA_1-1.tif"
    )
    assert core.tiles_in_window(xml, "clay_0-5cm_mean", (10, 10, 20, 20)) == (
        "tA/tA_1-1.tif",
    )


def test_tiles_in_window_errors():
    with pytest.raises(core.SoilGridsError, match="No SoilGrids tile"):
        core.tiles_in_window(VRT_XML, "clay_0-5cm_mean", (500, 500, 600, 600))
    with pytest.raises(core.SoilGridsError, match="Unexpected tile path"):
        core.tiles_in_window(VRT_XML, "sand_0-5cm_mean", (10, 10, 20, 20))
    with pytest.raises(core.SoilGridsError, match="lists no tiles"):
        core.tiles_in_window("<VRTDataset/>", "clay_0-5cm_mean", (0, 0, 1, 1))


def test_sg2017_paths_and_points():
    assert core.sg2017_path("clay", 0) == (
        "/vsicurl/https://files.isric.org/soilgrids/former/2017-03-10/data/"
        "CLYPPT_M_sl1_250m_ll.tif"
    )
    assert core.sg2017_path("bdod", 6).endswith("BLDFIE_M_sl7_250m_ll.tif")
    assert core.sg2017_points_needed(["0-5cm"]) == [0, 1]
    assert core.sg2017_points_needed(["30-60cm", "100-200cm"]) == [3, 4, 5, 6]
    assert core.sg2017_points_needed([d[0] for d in core.DEPTHS]) == list(range(7))


def test_vars_2017_cover_all_but_water_content():
    assert set(core.VARS_2017) == {
        "sand",
        "silt",
        "clay",
        "soc",
        "bdod",
        "cfvo",
        "phh2o",
        "cec",
    }
    assert core.VARS_2017["bdod"].divide_by == 1000  # kg/m3 -> g/cm3
    assert core.VARS_2017["phh2o"].divide_by == 10


def test_plan_selection_sg2017_warning_for_water_content_only():
    sel = core.plan_selection(["wv0033"], ["0-5cm"], ["mean"], sg2017=True)
    assert sel.sg2017 and any("2017" in w for w in sel.warnings)


# ----------------------------------------------------------------------
# Parallel job runner
# ----------------------------------------------------------------------


def test_run_jobs_parallel_and_sequential_agree():
    done = []
    par = core.run_jobs(list(range(20)), lambda k: k * k, 8, lambda: False, done.append)
    seq = core.run_jobs(list(range(20)), lambda k: k * k, 1, lambda: False, _noop)
    assert par == seq == {k: k * k for k in range(20)}
    assert sorted(done) == list(range(20))


def test_run_jobs_propagates_errors_and_cancel():
    def boom(k):
        if k == 3:
            raise core.SoilGridsError("bad layer")
        return k

    with pytest.raises(core.SoilGridsError, match="bad layer"):
        core.run_jobs(list(range(6)), boom, 4, lambda: False, _noop)
    with pytest.raises(InterruptedError):
        core.run_jobs(list(range(6)), lambda k: k, 4, lambda: True, _noop)
    with pytest.raises(InterruptedError):
        core.run_jobs(list(range(6)), lambda k: k, 1, lambda: True, _noop)


def _noop(*_a, **_k):
    return None


# ----------------------------------------------------------------------
# Area mode orchestration (fake index / reader / writer)
# ----------------------------------------------------------------------

RAW = {"mean": 300.0, "Q0.05": 200.0, "Q0.5": 300.0, "Q0.95": 500.0}
TEXTURE_MEAN = {"sand": 400.0, "silt": 400.0, "clay": 200.0}
RAW_2017 = {"CLYPPT": 25.0, "SNDPPT": 40.0, "SLTPPT": 35.0, "BLDFIE": 1400.0}
LAYER_RE = re.compile(r"/(\w+?)_(\d+-\d+cm)_(mean|Q0\.05|Q0\.5|Q0\.95)")
SL_RE = re.compile(r"/([A-Z]+)_M_sl(\d)_250m_ll\.tif$")


def fake_raw_value(source):
    path = source[0] if isinstance(source, tuple) else source
    m = LAYER_RE.search(path)
    if m:
        var, _, stat = m.groups()
        if stat == "mean" and var in TEXTURE_MEAN:
            return TEXTURE_MEAN[var]
        return RAW[stat]
    m = SL_RE.search(path)
    if m:
        code, k = m.group(1), int(m.group(2))
        return RAW_2017[code] + k  # value grows with depth point
    for code, value in (("BDTICM", 1500.0), ("BDRICM", 200.0), ("BDRLOG", 12.0)):
        if code in path:
            return value
    raise AssertionError(f"unexpected path {path}")


def fake_index(bounds_ll, points=None, base=core.BASE_DIR, ref=core.REF_LAYER):
    return core.TileIndex(("tA/tA_1-1.tif", "tA/tA_2-1.tif"), "clay_0-5cm_mean")


def failing_index(*_a, **_k):
    raise RuntimeError("index server down")


def make_reader(fail_tiles=False, fail_all=False, fail_2017=False, calls=None):
    lock = threading.Lock()

    def reader(source, grid):
        if calls is not None:
            with lock:
                calls.append(source)
        if fail_all:
            raise RuntimeError("HTTP 503")
        if fail_tiles and isinstance(source, tuple):
            raise RuntimeError("tile 404")
        if fail_2017 and "_M_sl" in str(source):
            raise RuntimeError("2017 down")
        arr = np.full((grid.height, grid.width), fake_raw_value(source))
        arr[0, 0] = np.nan  # one nodata cell
        return arr

    return reader


class FakeWriter:
    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()

    def __call__(self, path, grid, bands, units):
        with self.lock:
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
        index_fn=kw.pop("index_fn", fake_index),
        write_fn=writer,
        options_ctx=kw.pop("options_ctx", contextlib.nullcontext()),
        **kw,
    )
    return result, writer


def test_area_bands_units_and_nodata(grid, tmp_path):
    sel = core.plan_selection(["clay", "bdod"], ["0-5cm", "30-60cm"], ["Q0.5"])
    result, writer = run_area(grid, tmp_path, sel)
    clay = os.path.join(str(tmp_path), "soilgrids_clay_Q0.50.tif")
    bands, units = writer.files[clay]
    assert units == "%"
    assert [b[0] for b in bands] == ["clay_0-5cm_Q0.5 (%)", "clay_30-60cm_Q0.5 (%)"]
    assert bands[0][1][1, 1] == pytest.approx(30.0)  # 300 g/kg -> 30 %
    assert math.isnan(bands[0][1][0, 0])  # nodata kept
    bdod = os.path.join(str(tmp_path), "soilgrids_bdod_Q0.50.tif")
    bdod_bands, bdod_units = writer.files[bdod]
    assert bdod_units == "g/cm3"
    assert bdod_bands[0][1][1, 1] == pytest.approx(3.0)  # /100
    assert result.load_files == [clay, bdod]
    assert len(result.layers) == 4
    assert all(rec.route == "Tiles" for rec in result.layers)
    assert result.layers[0].stats["nodata"] == 1
    assert result.tiles == {"clay": 2, "bdod": 2}


def test_area_tiles_route_reads_layer_specific_tile_paths(grid, tmp_path):
    calls = []
    sel = core.plan_selection(["soc"], ["15-30cm"], ["Q0.95"])
    run_area(grid, tmp_path, sel, reader=make_reader(calls=calls))
    assert calls == [
        tuple(
            core.BASE_DIR + f"soc/./soc_15-30cm_Q0.95/{rel}"
            for rel in ("tA/tA_1-1.tif", "tA/tA_2-1.tif")
        )
    ]


def test_area_one_file_per_variable_statistic_parallel(grid, tmp_path):
    sel = core.plan_selection(
        ["sand", "silt", "clay"], [d[0] for d in core.DEPTHS], list(core.STATISTICS)
    )
    for workers in (1, 8):
        result, writer = run_area(grid, tmp_path, sel, workers=workers)
        assert len(writer.files) == 3 * 4
        for bands, _ in writer.files.values():
            assert [b[0].split("_")[1] for b in bands] == [d[0] for d in core.DEPTHS]


def test_area_loads_mean_when_no_median(grid, tmp_path):
    sel = core.plan_selection(["clay"], ["0-5cm"], ["mean", "Q0.95"])
    result, _ = run_area(grid, tmp_path, sel)
    assert result.load_files == [os.path.join(str(tmp_path), "soilgrids_clay_mean.tif")]


def test_area_texture_check(grid, tmp_path):
    sel = core.plan_selection(["sand", "silt", "clay"], ["0-5cm"], ["mean"])
    result, _ = run_area(grid, tmp_path, sel)
    assert result.texture == [
        {"depth": "0-5cm", "checked": 11, "flagged": 0, "max_abs_dev": 0.0}
    ]


def test_area_tile_failure_falls_back_to_vrt_and_logs(grid, tmp_path):
    logs = []
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    result, _ = run_area(
        grid, tmp_path, sel, reader=make_reader(fail_tiles=True), log_fn=logs.append
    )
    assert result.layers[0].route == "VRT"
    assert logs and "tile 404" in logs[0]
    assert result.warnings == logs


def test_area_index_failure_uses_vrt_for_every_layer(grid, tmp_path):
    logs = []
    sel = core.plan_selection(["clay"], ["0-5cm", "5-15cm"], ["Q0.5"])
    result, _ = run_area(
        grid, tmp_path, sel, index_fn=failing_index, log_fn=logs.append
    )
    assert {rec.route for rec in result.layers} == {"VRT"}
    assert result.tiles == {"clay": 0}
    assert len(logs) == 1 and "index server down" in logs[0]


def test_area_builds_one_index_per_variable(grid, tmp_path):
    refs = []
    lock = threading.Lock()

    def index_fn(bounds_ll, points=None, base=core.BASE_DIR, ref=core.REF_LAYER):
        with lock:
            refs.append(ref)
        return fake_index(bounds_ll)

    sel = core.plan_selection(["clay", "bdod", "phh2o"], ["5-15cm"], ["Q0.5"])
    run_area(grid, tmp_path, sel, index_fn=index_fn)
    assert sorted(refs) == [
        ("bdod", "0-5cm", "mean"),
        ("clay", "0-5cm", "mean"),
        ("phh2o", "0-5cm", "mean"),
    ]


def test_area_tiles_with_no_data_fall_back_to_vrt(grid, tmp_path):
    """Live bug 2026-10-05: tiles that exist but cover another area return
    all-nodata without an error. Such a layer must be re-read via its VRT."""
    logs = []

    def reader(source, grid):
        if isinstance(source, tuple) and "/bdod/" in source[0]:
            return np.full((grid.height, grid.width), np.nan)
        return make_reader()(source, grid)

    sel = core.plan_selection(["clay", "bdod"], ["0-5cm"], ["mean"])
    result, writer = run_area(grid, tmp_path, sel, reader=reader, log_fn=logs.append)
    routes = {r.variable: r.route for r in result.layers}
    assert routes == {"clay": "Tiles", "bdod": "VRT"}
    bands, _ = writer.files[os.path.join(str(tmp_path), "soilgrids_bdod_mean.tif")]
    assert bands[0][1][1, 1] == pytest.approx(3.0)
    assert any("no data in the area" in m for m in logs)


def test_area_layer_empty_on_every_route_is_warned(grid, tmp_path):
    def reader(source, grid):
        return np.full((grid.height, grid.width), np.nan)

    logs = []
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    result, _ = run_area(grid, tmp_path, sel, reader=reader, log_fn=logs.append)
    assert result.layers[0].route == "VRT"
    assert any("has no valid values" in m for m in result.warnings)


def test_read_with_fallback_accepts_empty_last_route():
    srcs = [core.Source("Tiles", ("a",)), core.Source("VRT", "b")]
    out, route, errors = core.read_with_fallback(srcs, lambda p: [math.nan, math.nan])
    assert route == "VRT" and errors == ["Tiles: no data in the area"]
    only = [core.Source("Tiles", ("a",))]
    out, route, _ = core.read_with_fallback(only, lambda p: [math.nan])
    assert route == "Tiles"


def test_area_all_routes_fail_names_routes(grid, tmp_path):
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    with pytest.raises(core.SoilGridsError, match="All access routes failed"):
        run_area(grid, tmp_path, sel, reader=make_reader(fail_all=True))


def test_area_sg2017_file_native_depth_points(grid, tmp_path):
    sel = core.plan_selection(
        ["clay", "bdod", "wv0033"], ["0-5cm", "100-200cm"], ["Q0.5"], sg2017=True
    )
    result, writer = run_area(grid, tmp_path, sel)
    path = os.path.join(str(tmp_path), "soilgrids2017_clay_mean.tif")
    bands, units = writer.files[path]
    assert units == "%"
    # Points bounding the selected intervals, stored as published (no averaging)
    assert [b[0] for b in bands] == [
        "clay_0cm_mean [SG2017] (%)",
        "clay_5cm_mean [SG2017] (%)",
        "clay_100cm_mean [SG2017] (%)",
        "clay_200cm_mean [SG2017] (%)",
    ]
    # CLYPPT fake = 25 + k (k = sl number): sl1 26, sl2 27, sl6 31, sl7 32
    assert [b[1][1, 1] for b in bands] == [26.0, 27.0, 31.0, 32.0]
    bd, bd_units = writer.files[
        os.path.join(str(tmp_path), "soilgrids2017_bdod_mean.tif")
    ]
    assert bd_units == "g/cm3"
    assert bd[0][1][1, 1] == pytest.approx(1401 / 1000)
    assert not any(
        os.path.basename(p).startswith("soilgrids2017_wv") for p in writer.files
    )
    recs = [r for r in result.layers if r.product == core.PRODUCT_SG2017]
    assert {r.depth for r in recs} == {"0cm", "5cm", "100cm", "200cm"}
    assert {r.route for r in recs} == {"2017 archive"}
    assert path not in result.load_files


def test_area_sg2017_failure_is_a_warning(grid, tmp_path):
    logs = []
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"], sg2017=True)
    result, writer = run_area(
        grid, tmp_path, sel, reader=make_reader(fail_2017=True), log_fn=logs.append
    )
    assert not any("soilgrids2017" in p for p in writer.files)
    assert any("CLYPPT" in w and "skipped" in w for w in result.warnings)


def test_area_bedrock_file(grid, tmp_path):
    sel = core.plan_selection([], ["0-5cm"], ["mean"], bedrock=True)
    result, writer = run_area(grid, tmp_path, sel)
    bands, _ = writer.files[os.path.join(str(tmp_path), core.BEDROCK_FILE_NAME)]
    assert [b[0].split()[0] for b in bands] == ["BDTICM", "BDRICM", "BDRLOG"]
    assert bands[0][1][1, 1] == 1500.0  # cm, no conversion
    assert result.load_files == [os.path.join(str(tmp_path), core.BEDROCK_FILE_NAME)]
    assert {rec.route for rec in result.layers} == {"2017 archive"}
    assert result.tiles == {}  # no 2.0 variables -> no index needed


def test_area_cancel(grid, tmp_path):
    sel = core.plan_selection(["clay"], ["0-5cm", "5-15cm"], ["Q0.5"])
    with pytest.raises(InterruptedError):
        run_area(grid, tmp_path, sel, cancel_fn=lambda: True)


def test_area_progress_reaches_one(grid, tmp_path):
    seen = []
    sel = core.plan_selection(
        ["clay"], ["0-5cm", "5-15cm"], ["Q0.5"], sg2017=True, bedrock=True
    )
    run_area(grid, tmp_path, sel, progress_fn=lambda f, m: seen.append(f))
    assert seen[0] == 0.0 and seen[-1] == 1.0
    assert max(seen[:-1]) < 1.0
    assert core.count_jobs(sel) == 2 + 3 + 3  # 2.0 layers + 2017 points + bedrock


def test_area_requires_writer(grid, tmp_path):
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    with pytest.raises(core.SoilGridsError, match="writer"):
        core.extract_area(sel, grid, (0, 0, 1, 1), str(tmp_path))


# ----------------------------------------------------------------------
# Point mode orchestration (fake index / sampler)
# ----------------------------------------------------------------------

SITES = [core.Site("A", 28.10, -25.80), core.Site("B", 28.12, -25.81)]


def make_sampler(fail_tiles=False, missing_second=False, calls=None):
    lock = threading.Lock()

    def sampler(source, lonlats):
        if calls is not None:
            with lock:
                calls.append((source, len(lonlats)))
        if fail_tiles and isinstance(source, tuple):
            raise RuntimeError("timeout")
        values = [fake_raw_value(source)] * len(lonlats)
        if missing_second and len(values) > 1:
            values[1] = math.nan
        return values

    return sampler


def run_points(selection, **kw):
    return core.extract_points(
        selection,
        kw.pop("sites", SITES),
        sample_fn=kw.pop("sampler", make_sampler()),
        index_fn=kw.pop("index_fn", fake_index),
        options_ctx=contextlib.nullcontext(),
        **kw,
    )


def test_points_rows_long_format():
    sel = core.plan_selection(["clay"], ["0-5cm", "100-200cm"], ["Q0.5"])
    result = run_points(sel)
    assert len(result.rows) == 2 * 2
    row = result.rows[0]
    assert row["Site"] == "A" and row["Variable"] == "clay"
    assert row["Product"] == core.PRODUCT_SG2
    assert (row["DepthTop_cm"], row["DepthBottom_cm"]) == (0, 5)
    assert row["Value"] == pytest.approx(30.0)
    assert row["Units"] == "%" and row["Route"] == "Tiles"
    assert result.rows[2]["DepthBottom_cm"] == 200


def test_points_index_gets_point_locations():
    seen = {}

    def index_fn(bounds_ll, points=None, base=core.BASE_DIR, ref=core.REF_LAYER):
        seen["bounds"], seen["points"] = bounds_ll, points
        return fake_index(bounds_ll)

    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    run_points(sel, index_fn=index_fn)
    assert seen["points"] == [(28.10, -25.80), (28.12, -25.81)]
    assert seen["bounds"] == (28.10, -25.81, 28.12, -25.80)


def test_points_missing_value_stays_nan():
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    result = run_points(sel, sampler=make_sampler(missing_second=True))
    assert math.isnan(result.rows[1]["Value"])


def test_points_tile_failure_falls_back_to_vrt():
    calls = []
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    result = run_points(sel, sampler=make_sampler(fail_tiles=True, calls=calls))
    assert len(calls) == 2 and calls[1][0].endswith("clay_0-5cm_Q0.5.vrt")
    assert result.rows[0]["Route"] == "VRT"
    assert result.warnings


def test_points_sg2017_rows_native_points():
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"], sg2017=True)
    result = run_points(sel)
    rows17 = [r for r in result.rows if r["Product"] == core.PRODUCT_SG2017]
    assert len(rows17) == 4  # 2 depth points x 2 sites
    first = rows17[0]
    assert first["Value"] == pytest.approx(26.0)  # sl1 (0 cm), no averaging
    assert (first["DepthTop_cm"], first["DepthBottom_cm"]) == (0, 0)
    assert rows17[2]["DepthTop_cm"] == 5
    assert first["Statistic"] == "mean" and first["Route"] == "2017 archive"


def test_no_derived_statistics_offered():
    assert core.STATISTICS == ("mean", "Q0.05", "Q0.5", "Q0.95")
    assert not hasattr(core, "RU90")
    assert core.point_label_2017(2) == "15cm"


def test_points_bedrock_rows():
    sel = core.plan_selection([], ["0-5cm"], ["mean"], bedrock=True)
    result = run_points(sel)
    assert [r["Variable"] for r in result.rows[::2]] == ["BDTICM", "BDRICM", "BDRLOG"]
    assert result.rows[0]["DepthTop_cm"] == ""
    assert result.rows[0]["Product"] == core.PRODUCT_SG2017


def test_points_texture_check():
    sel = core.plan_selection(["sand", "silt", "clay"], ["0-5cm"], ["mean"])
    result = run_points(sel)
    assert result.texture[0]["checked"] == 2 and result.texture[0]["flagged"] == 0


def test_points_requires_sites():
    sel = core.plan_selection(["clay"], ["0-5cm"], ["Q0.5"])
    with pytest.raises(core.SoilGridsError, match="No points"):
        run_points(sel, sites=[])


# ----------------------------------------------------------------------
# Metadata and CSV writers
# ----------------------------------------------------------------------


def test_metadata_contents(grid, tmp_path):
    sel = core.plan_selection(
        ["sand", "silt", "clay"],
        ["0-5cm"],
        ["mean"],
        bedrock=True,
        sg2017=True,
    )
    result, _ = run_area(grid, tmp_path, sel)
    sections = core.build_metadata(result, sel, {"Output CRS": "EPSG:2049"}, "3.13")
    titles = [s[0] for s in sections]
    assert titles[0] == "Run" and "Layers" in titles and titles[-1] == "Warnings"
    run = dict(sections[0][2])
    assert "Poggio" in run["Citation"] and "CC-BY 4.0" in run["Licence"]
    assert "Hengl" in run["Citation (2017)"] and "ODbL" in run["Licence (2017)"]
    assert run["Output CRS"] == "EPSG:2049"
    assert "Shangguan" in run["Citation (depth to bedrock)"]
    assert "marginal" in run["Note - quantiles"]
    assert "urban" in run["Note - masked areas"]
    assert run["Access route"].startswith("Tiles (one index per variable")
    assert "clay 2" in run["Access route"]
    layers = next(s for s in sections if s[0] == "Layers")
    assert layers[1][0] == "Product"
    assert {row[8] for row in layers[2]} >= {"Tiles", "2017 archive"}
    assert not any(s[0].startswith("Uncertainty") for s in sections)
    assert "native depth POINTS" in run["Note - SoilGrids 2017"]


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
            "Product": core.PRODUCT_SG2,
            "Variable": "clay",
            "Description": "Clay",
            "DepthTop_cm": 0,
            "DepthBottom_cm": 5,
            "Statistic": "Q0.5",
            "Value": v,
            "Units": "%",
            "Route": "Tiles",
        }
        for v in (30.0, math.nan)
    ]
    path = tmp_path / "p.csv"
    assert write_points_csv(rows, path) == (2, 1)
    with open(path, newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    assert data[0]["Value"] == "30.0" and data[1]["Value"] == ""
    assert data[0]["Product"] == "SoilGrids 2.0"


# ----------------------------------------------------------------------
# Real GDAL on a local mock of ISRIC's server layout (no network)
# ----------------------------------------------------------------------

IGH = "+proj=igh +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def _srs(osr, text):
    s = osr.SpatialReference()
    s.SetFromUserInput(text)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def _make_tif(gdal, path, wkt, x0, y0, res, arr, nodata=-32768):
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), arr.shape[1], arr.shape[0], 1, gdal.GDT_Int16
    )
    ds.SetGeoTransform((x0, res, 0, y0, 0, -res))
    ds.SetProjection(wkt)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    ds = None


@pytest.fixture
def mock_server(tmp_path):
    """ISRIC-like layout: <base>/<var>/<layer>.vrt indexing 2 x 2 tiles at
    <base>/<var>/<layer>/<tile>/<tile>_i-j.tif, for clay (reference) and soc.
    Tile values encode layer and position so misplacement is detectable."""
    gdal = pytest.importorskip("osgeo.gdal")
    osr = pytest.importorskip("osgeo.osr")
    gdal.UseExceptions()
    osr.UseExceptions()
    igh = _srs(osr, IGH)
    tr = osr.CoordinateTransformation(_srs(osr, "EPSG:4326"), igh)
    cx, cy, _ = tr.TransformPoint(28.2, -25.75)
    res, n = 250.0, 60
    x0 = math.floor(cx / res) * res - n * res
    y0 = math.floor(cy / res) * res + n * res
    base = tmp_path / "data"
    for var, offset in (("clay", 1000), ("soc", 2000), ("bdod", 3000)):
        name = f"{var}_0-5cm_mean"
        tiles = []
        for i in range(2):
            for j in range(2):
                tile = base / var / name / "tileT" / f"tileT_{i + 1}-{j + 1}.tif"
                arr = np.full((n, n), offset + 10 * i + j, dtype=np.int16)
                if (i, j) == (1, 1):
                    arr[5, 5] = -32768
                if var == "bdod":
                    # Same tile NAMES as clay, different placement: shifted
                    # half a tile east, like bdod/phh2o on the live server.
                    tile_x0 = x0 + (i + 0.5) * n * res
                else:
                    tile_x0 = x0 + i * n * res
                _make_tif(
                    gdal,
                    tile,
                    igh.ExportToWkt(),
                    tile_x0,
                    y0 - j * n * res,
                    res,
                    arr,
                )
                tiles.append(str(tile))
        gdal.BuildVRT(str(base / var / f"{name}.vrt"), tiles).FlushCache()
    return {
        "base": str(base) + "/",
        "x0": x0,
        "y0": y0,
        "res": res,
        "n": n,
        "igh": igh,
        "osr": osr,
    }


def test_gdal_tile_index_and_direct_read_match_full_vrt(mock_server):
    ms = mock_server
    osr = ms["osr"]
    inv = osr.CoordinateTransformation(ms["igh"], _srs(osr, "EPSG:4326"))
    # A small area in the top-left tile only.
    lon0, lat0, _ = inv.TransformPoint(
        ms["x0"] + 10 * ms["res"], ms["y0"] - 30 * ms["res"]
    )
    lon1, lat1, _ = inv.TransformPoint(
        ms["x0"] + 30 * ms["res"], ms["y0"] - 10 * ms["res"]
    )
    index = core.gdal_tile_index((lon0, lat0, lon1, lat1), base=ms["base"])
    assert index.rel_paths == ("tileT/tileT_1-1.tif",)
    # An area spanning the four-tile corner selects all four tiles.
    mx, my = ms["x0"] + ms["n"] * ms["res"], ms["y0"] - ms["n"] * ms["res"]
    a, b, _ = inv.TransformPoint(mx - 5 * ms["res"], my - 5 * ms["res"])
    c, d, _ = inv.TransformPoint(mx + 5 * ms["res"], my + 5 * ms["res"])
    index4 = core.gdal_tile_index((a, b, c, d), base=ms["base"])
    assert len(index4.rel_paths) == 4

    grid = core.make_grid(
        (mx - 2000, my - 2000, mx + 2000, my + 2000), 250.0, ms["igh"].ExportToWkt()
    )
    direct = core.gdal_read_grid(
        tuple(index4.paths_for("soc", "0-5cm", "mean", ms["base"])), grid
    )
    full = core.gdal_read_grid(core.layer_vrt("soc", "0-5cm", "mean", ms["base"]), grid)
    assert np.array_equal(direct, full, equal_nan=True)
    assert set(np.unique(direct[np.isfinite(direct)])) == {2000, 2001, 2010, 2011}


def test_gdal_tile_index_for_points_and_sampling(mock_server):
    ms = mock_server
    osr = ms["osr"]
    inv = osr.CoordinateTransformation(ms["igh"], _srs(osr, "EPSG:4326"))
    pts = []
    for col, row in ((5.5, 5.5), (66.5, 66.5), (65.5, 5.5)):
        lon, lat, _ = inv.TransformPoint(
            ms["x0"] + col * ms["res"], ms["y0"] - row * ms["res"]
        )
        pts.append((lon, lat))
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    index = core.gdal_tile_index(
        (min(lons), min(lats), max(lons), max(lats)), points=pts, base=ms["base"]
    )
    assert set(index.rel_paths) == {
        "tileT/tileT_1-1.tif",
        "tileT/tileT_2-2.tif",
        "tileT/tileT_2-1.tif",
    }
    values = core.gdal_sample_points(
        tuple(index.paths_for("soc", "0-5cm", "mean", ms["base"])), pts
    )
    assert values == [2000.0, 2011.0, 2010.0]
    # The nodata cell in tile (2,2) is at local (5,5) -> global (65,65).
    lon, lat, _ = inv.TransformPoint(
        ms["x0"] + 65.5 * ms["res"], ms["y0"] - 65.5 * ms["res"]
    )
    nd = core.gdal_sample_points(
        core.layer_vrt("soc", "0-5cm", "mean", ms["base"]), [(lon, lat)]
    )
    assert math.isnan(nd[0])


def test_extract_area_end_to_end_with_real_gdal(mock_server, tmp_path):
    ms = mock_server
    osr = ms["osr"]
    from mayim_tools.soil.soilgrids_extract.export import write_multiband_geotiff

    gdal = pytest.importorskip("osgeo.gdal")
    inv = osr.CoordinateTransformation(ms["igh"], _srs(osr, "EPSG:4326"))
    mx, my = ms["x0"] + ms["n"] * ms["res"], ms["y0"] - ms["n"] * ms["res"]
    lo = inv.TransformPoint(mx - 3000, my - 3000)
    hi = inv.TransformPoint(mx + 3000, my + 3000)
    bounds_ll = (lo[0], lo[1], hi[0], hi[1])
    utm = _srs(osr, "EPSG:32735")
    t = osr.CoordinateTransformation(_srs(osr, "EPSG:4326"), utm)
    ux0, uy0, _ = t.TransformPoint(lo[0], lo[1])
    ux1, uy1, _ = t.TransformPoint(hi[0], hi[1])
    grid = core.make_grid(
        (min(ux0, ux1), min(uy0, uy1), max(ux0, ux1), max(uy0, uy1)),
        250.0,
        utm.ExportToWkt(),
    )
    sel = core.plan_selection(["soc"], ["0-5cm"], ["mean"])
    out = tmp_path / "out"
    out.mkdir()
    result = core.extract_area(
        sel,
        grid,
        bounds_ll,
        str(out),
        write_fn=write_multiband_geotiff,
        base=ms["base"],
        workers=4,
    )
    assert result.tiles == {"soc": 4}
    assert [r.route for r in result.layers] == ["Tiles"]
    ds = gdal.Open(str(out / "soilgrids_soc_mean.tif"))
    data = ds.GetRasterBand(1).ReadAsArray()
    valid = data[data != core.NODATA_OUT]
    # soc factor 10: tile values 2000..2011 -> 200.0..201.1 g/kg
    found = np.unique(valid)
    assert found.size and all(
        np.isclose(v, [200.0, 200.1, 201.0, 201.1], atol=1e-4).any() for v in found
    )
    assert ds.GetRasterBand(1).GetDescription() == "soc_0-5cm_mean (g/kg)"


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


def test_windowed_reader_matches_plain_warp_and_is_thread_consistent(mock_server):
    """The thread-safe reader (download native block, warp in memory under
    a lock) must give exactly what a plain full gdal.Warp gives, and
    parallel reads must equal sequential ones."""
    gdal = pytest.importorskip("osgeo.gdal")
    ms = mock_server
    osr = ms["osr"]
    utm = _srs(osr, "EPSG:32735")
    t = osr.CoordinateTransformation(ms["igh"], utm)
    mx, my = ms["x0"] + ms["n"] * ms["res"], ms["y0"] - ms["n"] * ms["res"]
    ux, uy, _ = t.TransformPoint(mx, my)
    grid = core.make_grid(
        (ux - 7000, uy - 7000, ux + 7000, uy + 7000), 200.0, utm.ExportToWkt()
    )
    path = core.layer_vrt("clay", "0-5cm", "mean", ms["base"])
    ours = core.gdal_read_grid(path, grid)
    plain = gdal.Warp(
        "",
        path,
        format="MEM",
        outputBounds=(grid.xmin, grid.ymin, grid.xmax, grid.ymax),
        width=grid.width,
        height=grid.height,
        dstSRS=grid.crs_wkt,
        resampleAlg="near",
        outputType=gdal.GDT_Float32,
        dstNodata=float("nan"),
    )
    ref = plain.GetRasterBand(1).ReadAsArray().astype(np.float64)
    assert np.isfinite(ours).sum() > 1000
    assert np.array_equal(ours, ref, equal_nan=True)

    keys = list(range(24))
    seq = core.run_jobs(
        keys, lambda k: core.gdal_read_grid(path, grid), 1, lambda: False, _noop
    )
    par = core.run_jobs(
        keys, lambda k: core.gdal_read_grid(path, grid), 8, lambda: False, _noop
    )
    assert all(np.array_equal(seq[k], par[k], equal_nan=True) for k in keys)


def test_mismatched_tile_layout_is_detected_and_handled(mock_server, tmp_path):
    """bdod tiles carry clay's tile names but sit elsewhere (live bug).
    Clay's index applied to bdod must raise (coverage check), and the
    per-variable index must read bdod correctly."""
    ms = mock_server
    osr = ms["osr"]
    inv = osr.CoordinateTransformation(ms["igh"], _srs(osr, "EPSG:4326"))
    lo = inv.TransformPoint(ms["x0"] + 10 * ms["res"], ms["y0"] - 30 * ms["res"])
    hi = inv.TransformPoint(ms["x0"] + 30 * ms["res"], ms["y0"] - 10 * ms["res"])
    bounds_ll = (lo[0], lo[1], hi[0], hi[1])
    grid = core.make_grid(
        (
            ms["x0"] + 10 * ms["res"],
            ms["y0"] - 30 * ms["res"],
            ms["x0"] + 30 * ms["res"],
            ms["y0"] - 10 * ms["res"],
        ),
        250.0,
        ms["igh"].ExportToWkt(),
        buffer_cells=0,
    )
    clay_index = core.gdal_tile_index(bounds_ll, base=ms["base"])
    wrong = tuple(clay_index.paths_for("bdod", "0-5cm", "mean", ms["base"]))
    with pytest.raises(core.SoilGridsError, match="do not cover"):
        core.gdal_read_grid(wrong, grid)

    # In bdod's own layout, cells 100-120 lie in its second tile.
    lo = inv.TransformPoint(ms["x0"] + 100 * ms["res"], ms["y0"] - 30 * ms["res"])
    hi = inv.TransformPoint(ms["x0"] + 120 * ms["res"], ms["y0"] - 10 * ms["res"])
    bdod_index = core.gdal_tile_index(
        (lo[0], lo[1], hi[0], hi[1]), base=ms["base"], ref=("bdod", "0-5cm", "mean")
    )
    assert bdod_index.rel_paths == ("tileT/tileT_2-1.tif",)


def test_extract_area_mismatched_layout_end_to_end(mock_server, tmp_path):
    """Full run where bdod's layout differs from clay's: with per-variable
    indices both come out right via Tiles; no empty layers."""
    from mayim_tools.soil.soilgrids_extract.export import write_multiband_geotiff

    gdal = pytest.importorskip("osgeo.gdal")
    ms = mock_server
    osr = ms["osr"]
    inv = osr.CoordinateTransformation(ms["igh"], _srs(osr, "EPSG:4326"))
    # Cells 85-95: clay tile 2-1 (cells 60-120); in bdod's layout the same
    # area straddles its tiles 1-1 (30-90) and 2-1 (90-150).
    xa, xb = ms["x0"] + 95 * ms["res"], ms["x0"] + 85 * ms["res"]
    lo = inv.TransformPoint(xb, ms["y0"] - 30 * ms["res"])
    hi = inv.TransformPoint(xa, ms["y0"] - 10 * ms["res"])
    grid = core.make_grid(
        (xb, ms["y0"] - 30 * ms["res"], xa, ms["y0"] - 10 * ms["res"]),
        250.0,
        ms["igh"].ExportToWkt(),
        buffer_cells=0,
    )
    sel = core.plan_selection(["clay", "bdod"], ["0-5cm"], ["mean"])
    out = tmp_path / "o"
    out.mkdir()
    result = core.extract_area(
        sel,
        grid,
        (lo[0], lo[1], hi[0], hi[1]),
        str(out),
        write_fn=write_multiband_geotiff,
        base=ms["base"],
        workers=4,
    )
    assert not result.warnings
    assert {r.variable: r.route for r in result.layers} == {
        "clay": "Tiles",
        "bdod": "Tiles",
    }
    expected = {"clay": {101.0}, "bdod": {30.0, 30.1}}
    for name, values in expected.items():
        ds = gdal.Open(str(out / f"soilgrids_{name}_mean.tif"))
        data = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        found = {round(float(v), 3) for v in np.unique(data)}
        assert found == values, (name, found)
    assert not list(out.glob("*.qml"))  # no style files any more
