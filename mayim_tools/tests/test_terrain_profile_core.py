"""Tests for hydrology/terrain_profile (core, export, figure) - no QGIS.

The synthetic valley has ground z = 98 + 0.03 |x - 300| (capped at 104) on
a 1 m grid, so waterlines, distances and depths are known exactly.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from mayim_tools.hydrology.terrain_profile import core
from mayim_tools.hydrology.terrain_profile.core import (
    EDGE_SECTION_END,
    EDGE_WATERLINE,
    STATUS_DRY,
    STATUS_INUNDATED,
    STATUS_NO_FLOOD,
    STATUS_NO_PROPERTY,
    STATUS_NOT_CROSSED,
    Grid,
    ProfileError,
)
from mayim_tools.hydrology.terrain_profile.export import (
    column_tag,
    write_measurements_csv,
    write_profile_csv,
)

GT = (0.0, 1.0, 0.0, 200.0, 0.0, -1.0)
NODATA = -9999.0
PROPERTY = [[[(420, 50), (520, 50), (520, 150), (420, 150), (420, 50)]]]
LINE = [(150, 100), (600, 100)]


def _ground():
    x = np.arange(1000) + 0.5
    xx = np.broadcast_to(x, (200, 1000))
    return 98 + np.minimum(np.abs(xx - 300) * 0.03, 6.0)


@pytest.fixture()
def dem():
    return Grid.from_raw(_ground(), GT, NODATA)


def _wse(level):
    """A stepped WSE raster like a RAS result: the level on wet cells only."""
    z = _ground()
    return Grid.from_raw(np.where(z < level, level, NODATA), GT, NODATA)


def _row(section, i=0):
    return core.measure(section, section.events[i])


# ----------------------------------------------------------------- grid


def test_grid_nodata_and_bilinear(dem):
    assert math.isclose(dem.sample(200.0, 100.0)[0], 101.0, abs_tol=1e-9)
    g = Grid.from_raw(
        np.array([[1.0, NODATA], [3.0, 4.0]]), (0, 1, 0, 2, 0, -1), NODATA
    )
    assert np.isnan(g.array[0, 1])
    # bilinear impossible next to NoData -> value of the containing cell
    assert g.sample(0.9, 1.9)[0] == 1.0
    assert np.isnan(g.sample(1.5, 1.5)[0])
    assert np.isnan(g.sample(-5, -5)[0])


def test_rotated_raster_rejected():
    with pytest.raises(ProfileError):
        Grid.from_raw(np.zeros((2, 2)), (0, 1, 0.1, 2, 0, -1), None)


# ----------------------------------------------------------------- geometry


def test_stations_include_vertices_extras_and_end():
    v = core.clean_vertices([(0, 0), (10, 0), (10, 5.5)])
    ch = core.stations(v, 2.0, extra=[3.3])
    assert ch[0] == 0 and ch[-1] == pytest.approx(15.5)
    for c in (10.0, 3.3, 15.5):
        assert np.any(np.isclose(ch, c))
    assert np.all(np.diff(ch) > 0)


def test_zero_length_line_rejected():
    with pytest.raises(ProfileError):
        core.clean_vertices([(1, 1), (1, 1)])


def test_polygon_crossings_and_inside_intervals():
    v = core.clean_vertices(LINE)
    cr = core.polygon_crossings(v, PROPERTY)
    assert cr == pytest.approx([270.0, 370.0])
    assert core.inside_intervals(v, PROPERTY, cr) == pytest.approx([(270.0, 370.0)])


def test_polygon_with_hole_gives_two_intervals():
    hole = [(450, 80), (480, 80), (480, 120), (450, 120)]
    poly = [[PROPERTY[0][0], hole]]
    v = core.clean_vertices(LINE)
    cr = core.polygon_crossings(v, poly)
    iv = core.inside_intervals(v, poly, cr)
    assert iv == pytest.approx([(270.0, 300.0), (330.0, 370.0)])


def test_section_labels():
    assert core.section_labels(28)[:3] == ["A", "B", "C"]
    assert core.section_labels(28)[26:] == ["AA", "AB"]


def test_nice_step():
    assert core.nice_step(450) == 50
    assert core.nice_step(2300) == 250


# ----------------------------------------------------------------- water


def test_waterlines_are_exact(dem):
    s = core.build_section("A", LINE, dem, [("1:100", _wse(101.0))], 1.0, PROPERTY)
    ev = s.events[0]
    assert len(ev.intervals) == 1
    w = ev.intervals[0]
    # |x - 300| = 100 -> x = 200 and 400 -> chainage 50 and 250
    assert w.start == pytest.approx(50.0, abs=1e-6)
    assert w.end == pytest.approx(250.0, abs=1e-6)
    assert w.start_kind == w.end_kind == EDGE_WATERLINE
    assert ev.max_depth == pytest.approx(3.0, abs=0.02)


def test_waterline_found_between_cell_centres(dem):
    """100.5 m meets the ground at x = 383.33, inside a partly wet cell."""
    s = core.build_section("A", LINE, dem, [("1:50", _wse(100.5))], 1.0, PROPERTY)
    w = s.events[0].intervals[0]
    assert w.end == pytest.approx(233.333, abs=1e-3)
    assert w.end_kind == EDGE_WATERLINE


def test_extend_wse_holds_nearest_within_distance():
    ch = np.arange(6.0)
    out = core.extend_wse(ch, np.array([np.nan, 1, 2, np.nan, np.nan, np.nan]), 1.5)
    assert np.isnan(out[5]) and np.isnan(out[4])
    assert out[0] == 1 and out[3] == 2


def test_wet_to_section_end():
    ch = np.arange(5.0)
    ground = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    ws = np.full(5, 1.25)
    (w,) = core.wet_intervals(ch, ground, ws)
    assert w.start_kind == EDGE_SECTION_END and w.start == 0
    assert w.end == pytest.approx(2.5)


# ----------------------------------------------------------------- measurements


def test_property_dry_distances(dem):
    s = core.build_section("A", LINE, dem, [("1:100", _wse(101.0))], 1.0, PROPERTY)
    r = _row(s)
    assert r["status"] == STATUS_DRY
    assert r["boundary_ch_m"] == pytest.approx(270.0)
    assert r["flood_edge_ch_m"] == pytest.approx(250.0)
    assert r["horiz_dist_m"] == pytest.approx(20.0, abs=1e-6)
    assert r["ground_at_boundary_m"] == pytest.approx(101.6, abs=1e-6)
    assert r["vert_dist_m"] == pytest.approx(0.6, abs=1e-6)
    assert r["lowest_ground_in_prop_m"] == pytest.approx(101.6, abs=1e-6)
    assert r["wet_length_in_prop_m"] == 0
    assert r["dimension"]["kind"] == "dry"


def test_property_inundated(dem):
    s = core.build_section("B", LINE, dem, [("1:100", _wse(102.2))], 1.0, PROPERTY)
    r = _row(s)
    assert r["status"] == STATUS_INUNDATED
    # waterline at x = 440 -> 20 m inside the property (x 420 to 440)
    assert r["wet_length_in_prop_m"] == pytest.approx(20.0, abs=1e-6)
    assert r["wet_fraction_in_prop"] == pytest.approx(0.2, abs=1e-6)
    assert r["max_depth_in_prop_m"] == pytest.approx(0.6, abs=1e-6)
    assert r["max_depth_ch_m"] == pytest.approx(270.0)
    assert r["lowest_minus_wse_m"] == pytest.approx(-0.6, abs=1e-6)
    d = r["dimension"]
    assert (d["wet_from"], d["wet_to"]) == pytest.approx((270.0, 290.0), abs=1e-6)


def test_statuses(dem):
    s = core.build_section("C", LINE, dem, [("dry", _wse(97.0))], 1.0, PROPERTY)
    assert _row(s)["status"] == STATUS_NO_FLOOD
    s = core.build_section(
        "D", [(150, 10), (600, 10)], dem, [("e", _wse(101))], 1.0, PROPERTY
    )
    assert _row(s)["status"] == STATUS_NOT_CROSSED
    s = core.build_section("E", LINE, dem, [("e", _wse(101))], 1.0, None)
    r = _row(s)
    assert r["status"] == STATUS_NO_PROPERTY
    assert r["section_wet_length_m"] == pytest.approx(200.0, abs=1e-6)


def test_section_off_dem_raises(dem):
    with pytest.raises(ProfileError):
        core.build_section("Z", [(5000, 5000), (6000, 5000)], dem)


# ----------------------------------------------------------------- plan distance


def test_plan_distance_dry(dem):
    pd = core.plan_distance("1:100", PROPERTY, _wse(101.0), dem.sample, 500)
    assert pd.status == "property dry"
    # true waterline at x = 400, nearest wet cell centre at 399.5
    assert pd.distance_m == pytest.approx(20.0, abs=pd.accuracy_m + 1e-9)
    assert pd.property_xy[0] == pytest.approx(420.0)


def test_plan_distance_inundated_and_out_of_range(dem):
    pd = core.plan_distance("big", PROPERTY, _wse(102.2), dem.sample, 500)
    assert pd.status == "property inundated"
    # x 420..440, y 50..150 -> 20 x 100 cells
    assert pd.wet_area_in_property_m2 == pytest.approx(2000.0, rel=0.05)
    far = core.plan_distance("1:100", PROPERTY, _wse(101.0), dem.sample, 10)
    assert far.distance_m is None


# ----------------------------------------------------------------- export


def test_csv_outputs(tmp_path, dem):
    s = core.build_section(
        "A", LINE, dem, [("1:100 (RAS)", _wse(101.0))], 1.0, PROPERTY
    )
    p1 = tmp_path / "profile.csv"
    write_profile_csv(str(p1), [s])
    raw = p1.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 with BOM
    lines = raw.decode("utf-8-sig").splitlines()
    assert lines[0] == "# Column key"
    rows = list(csv.DictReader(ln for ln in lines if not ln.startswith("#")))
    assert len(rows) == len(s.chainage)
    assert "wse_1_100_RAS_m" in rows[0]
    p2 = tmp_path / "m.csv"
    write_measurements_csv(str(p2), [_row(s)])
    text = p2.read_text(encoding="utf-8-sig")
    assert text.startswith("# Column key\n# section: section label")
    assert "20.00" in text and "," in text
    assert column_tag("  ") == "event"


def test_no_thousands_separators(tmp_path):
    x = np.arange(3000) + 0.5
    z = np.broadcast_to(1500 + 0.001 * x, (10, 3000))
    big = Grid.from_raw(z, (0, 1, 0, 10, 0, -1), None)
    s = core.build_section("A", [(0, 5), (2999, 5)], big, [], 1.0)
    p = tmp_path / "p.csv"
    write_profile_csv(str(p), [s])
    last = p.read_text(encoding="utf-8-sig").strip().splitlines()[-1]
    assert "2999.000" in last and "1,5" not in last


# ----------------------------------------------------------------- figure


def test_figure_is_16_5_cm_wide_and_keeps_ve(tmp_path, dem):
    pytest.importorskip("matplotlib")
    from mayim_tools._common.report_figures import save_report_figure
    from mayim_tools.hydrology.terrain_profile import figure

    s = core.build_section("A", LINE, dem, [("1:100", _wse(101.0))], 1.0, PROPERTY)
    fig, lay = figure.draw_section(s, _row(s)["dimension"], ve=10)
    w_cm, h_cm = (v * 2.54 for v in fig.get_size_inches())
    assert w_cm == pytest.approx(16.5, abs=1e-9)
    assert 6.0 - 1e-9 <= h_cm <= 14.0 + 1e-9
    assert lay.ve == 10 and not lay.ve_reduced
    # drawn to scale: vertical cm per m = 10 x horizontal cm per m
    axes_w = 16.5 - figure.LEFT_CM - figure.RIGHT_CM
    assert lay.axes_h_cm / (lay.y1 - lay.y0) == pytest.approx(10 * axes_w / s.length)
    paths = save_report_figure(fig, tmp_path / "A", formats=("png", "svg"))
    from matplotlib.image import imread

    img = imread(paths[0])
    assert abs(img.shape[1] - 16.5 / 2.54 * 300) < 1.0  # default 300 dpi
    p200 = save_report_figure(fig, tmp_path / "B", formats=("png",), dpi=200)
    assert abs(imread(p200[0]).shape[1] - 16.5 / 2.54 * 200) < 1.0
    assert "font-family" in (tmp_path / "A.svg").read_text(encoding="utf-8")


def test_steep_section_reduces_ve(dem):
    pytest.importorskip("matplotlib")
    from mayim_tools.hydrology.terrain_profile import figure

    s = core.build_section("A", [(250, 100), (400, 100)], dem, [], 1.0)
    lay = figure.layout(s, ve=50, max_height_cm=8)
    assert lay.ve_reduced and lay.ve < 50
    assert lay.height_cm == pytest.approx(8)


def test_fonts_fall_back_with_warning():
    pytest.importorskip("matplotlib")
    from mayim_tools._common.report_figures import resolve_font

    family, warning = resolve_font("No Such Font Family XYZ")
    assert family and warning


def test_plan_view_fits_section_and_stacks_above_profile(dem):
    pytest.importorskip("matplotlib")
    from mayim_tools.hydrology.terrain_profile import figure

    s = core.build_section("A", LINE, dem, [("1:100", _wse(101.0))], 1.0, PROPERTY)
    pv = figure.plan_view(s)
    x0, y0, x1, y1 = pv.extent
    # the whole section is inside the extent, with the plan's aspect
    assert x0 < 150 and x1 > 600 and y0 < 100 < y1
    assert (x1 - x0) / (y1 - y0) == pytest.approx(pv.width_cm / pv.height_cm)
    assert figure.PLAN_MIN_H_CM <= pv.height_cm <= figure.PLAN_MAX_H_CM
    w, h = pv.pixels
    pv.image = np.full((h, w, 4), 255, np.uint8)
    pv.polygons = PROPERTY
    profile, lay = figure.draw_section(s, _row(s)["dimension"])
    both, lay2 = figure.draw_section(s, _row(s)["dimension"], plan=pv)
    w_cm, h_cm = (v * 2.54 for v in both.get_size_inches())
    assert w_cm == pytest.approx(16.5)
    expected = lay.height_cm + figure.PLAN_GAP_CM + pv.height_cm + figure.PLAN_TOP_CM
    assert h_cm == pytest.approx(expected)
    assert lay2.ve == lay.ve


def test_plan_view_of_north_south_section_is_tall(dem):
    from mayim_tools.hydrology.terrain_profile import figure

    s = core.build_section("B", [(300, 20), (300, 180)], dem, [], 1.0)
    pv = figure.plan_view(s)
    assert pv.height_cm == pytest.approx(figure.PLAN_MAX_H_CM)
    assert pv.extent[1] < 20 and pv.extent[3] > 180
