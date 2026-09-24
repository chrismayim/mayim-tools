"""
Tests for mayim_tools/hydrology/catchment_delineation/core.py and
export.py. Zero QGIS dependency - run under plain pytest.

Flow accumulation is cross-checked against this suite's own
D8 Flow Accumulation core (itself a verified WhiteboxTools replication),
and polygon validity is checked with shapely when it is installed.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from mayim_tools.hydrology.catchment_delineation.core import (
    CatchmentError,
    Outlet,
    _ring_signed_area_idx,
    build_downstream_index,
    check_geotransform,
    check_same_grid,
    delineate_catchments,
    equal_area_slope,
    fill_holes,
    flow_accumulation_cells,
    group_polygons,
    horn_slope,
    hypsometric_curve,
    line_cells,
    pointer_to_direction_index,
    resolve_terminals,
    rings_to_multipolygon_wkt,
    simplify_ring,
    slope_10_85,
    snap_to_max_accumulation,
    tc_bransby_williams_minutes,
    tc_kirpich_minutes,
    tc_usbr_minutes,
    trace_rings,
)
from mayim_tools.hydrology.catchment_delineation.export import (
    write_hypsometric_csv,
    write_parameters_csv,
)
from mayim_tools.hydrology.d8_flow_accumulation.core import (
    accumulate_flow_cells,
    compute_d8_pointer,
)
from mayim_tools.hydrology.d8_flow_accumulation.core import (
    pointer_to_direction_index as fa_pointer_to_direction_index,
)

PTR_NODATA = -32768.0
CELL = 10.0
GT = (1000.0, CELL, 0.0, 5000.0, 0.0, -CELL)


def cxy(row, col):
    """Map coordinates of a cell centre on GT."""
    return GT[0] + (col + 0.5) * CELL, GT[3] - (row + 0.5) * CELL


def valley_dem(rows=40, cols=50, seed=0):
    """V-shaped valley draining south along the centre column."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    mid = cols // 2
    return np.abs(x - mid) * 0.5 + (rows - y) * 1.0 + rng.random((rows, cols)) * 1e-3


def pointer_for(dem):
    return compute_d8_pointer(dem, np.nan, CELL, CELL)


def upstream_mask_bruteforce(pointer, seeds, esri=False):
    """Slow reference: a cell is in the catchment if walking downstream
    from it hits a seed cell."""
    di, _ = pointer_to_direction_index(pointer, PTR_NODATA, esri)
    rows, cols = di.shape
    d_row = [-1, 0, 1, 1, 1, 0, -1, -1]
    d_col = [1, 1, 1, 0, -1, -1, -1, 0]
    seeds = set(seeds)
    out = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            rr, cc, steps = r, c, 0
            while steps < rows * cols:
                if (rr, cc) in seeds:
                    out[r, c] = True
                    break
                d = di[rr, cc]
                if d < 0:
                    break
                rr, cc = rr + d_row[d], cc + d_col[d]
                if not (0 <= rr < rows and 0 <= cc < cols) or di[rr, cc] == -2:
                    break
                steps += 1
    return out


# ----------------------------------------------------------------------
# Pointer decoding and network
# ----------------------------------------------------------------------


def test_pointer_decoding_wbt_and_esri():
    wbt = np.array([[1, 2, 4, 8, 16, 32, 64, 128, 0, PTR_NODATA]])
    di, n_bad = pointer_to_direction_index(wbt, PTR_NODATA)
    assert di.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, -1, -2]]
    assert n_bad == 0

    esri = np.array([[128, 1, 2, 4, 8, 16, 32, 64]])
    di, _ = pointer_to_direction_index(esri, PTR_NODATA, esri_style=True)
    assert di.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_invalid_pointer_values_are_counted_and_masked():
    di, n_bad = pointer_to_direction_index(np.array([[3, 2, 255]]), PTR_NODATA)
    assert n_bad == 2
    assert di.tolist() == [[-2, 1, -2]]


def test_downstream_index_terminals():
    # E, E, off-grid E ; the middle cell points into a NoData cell below
    pointer = np.array([[2, 8, 2], [0, PTR_NODATA, 0]])
    di, _ = pointer_to_direction_index(pointer, PTR_NODATA)
    nxt = build_downstream_index(di)
    assert nxt.tolist() == [1, 1, 2, 3, 4, 5]


def test_resolve_terminals_distance_and_loop_detection():
    nxt = np.array([1, 2, 3, 3])
    term, dist, unresolved = resolve_terminals(nxt, np.array([1.0, 2.0, 3.0, 0.0]))
    assert term.tolist() == [3, 3, 3, 3]
    assert dist.tolist() == [6.0, 5.0, 3.0, 0.0]
    assert not unresolved.any()

    loop = np.array([1, 0, 1, 3])  # 0 <-> 1, 2 drains into the loop, 3 pit
    _, _, unresolved = resolve_terminals(loop)
    assert unresolved.tolist() == [True, True, True, False]
    loop3 = np.array([1, 2, 0])  # 3-cell loop
    assert resolve_terminals(loop3)[2].all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_flow_accumulation_matches_d8_flow_accumulation_tool(seed):
    rng = np.random.default_rng(seed)
    dem = rng.random((30, 35)) * 5 + np.arange(35)[None, :] * 0.2
    dem[3, 4] = np.nan
    pointer = compute_d8_pointer(dem, np.nan, CELL, CELL)
    di, _ = pointer_to_direction_index(pointer, PTR_NODATA)
    acc, n_loop = flow_accumulation_cells(
        build_downstream_index(di), (di != -2).ravel()
    )
    ref = accumulate_flow_cells(fa_pointer_to_direction_index(pointer, PTR_NODATA))
    assert n_loop == 0
    np.testing.assert_allclose(acc.reshape(di.shape), ref, equal_nan=True)


# ----------------------------------------------------------------------
# Outlet placement
# ----------------------------------------------------------------------


def test_snap_picks_highest_accumulation_then_nearest():
    acc = np.ones((7, 7))
    acc[3, 5] = 50
    acc[3, 1] = 50
    assert snap_to_max_accumulation(3, 4, acc, 2) == (3, 5)
    assert snap_to_max_accumulation(3, 3, acc, 1) == (3, 3)
    assert snap_to_max_accumulation(3, 3, acc, 0) == (3, 3)
    acc[0, 0] = np.nan
    assert snap_to_max_accumulation(0, 0, acc, 0) is None
    assert snap_to_max_accumulation(-10, -10, acc, 2) is None


def test_line_cells_are_four_connected():
    shape = (20, 20)
    x0, y0 = cxy(2, 2)
    x1, y1 = cxy(15, 17)
    cells = line_cells([(x0, y0), (x1, y1)], GT, shape)
    assert cells[0] == (2, 2) and cells[-1] == (15, 17)
    for (r0, c0), (r1, c1) in zip(cells[:-1], cells[1:], strict=True):
        assert abs(r1 - r0) + abs(c1 - c0) == 1


def test_diagonal_flow_cannot_leak_through_a_diagonal_pour_line():
    # Plane falling to the SW: every D8 step is diagonal (SW).
    rows = cols = 12
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    dem = x - y + 100.0
    pointer = pointer_for(dem)
    # pour line running NW -> SE across the flow
    line = [cxy(1, 1), cxy(10, 10)]
    res = delineate_catchments(
        pointer, PTR_NODATA, dem, np.nan, GT, [Outlet("L", "line", [line])]
    )
    cells = line_cells(line, GT, (rows, cols))
    expected = upstream_mask_bruteforce(pointer, cells)
    # everything NE of the line drains into it
    assert res.catchments[0].attributes["area_km2"] == pytest.approx(
        expected.sum() * CELL * CELL / 1e6
    )
    assert expected[0, 11]  # a far-upstream corner cell is captured


# ----------------------------------------------------------------------
# Boundary tracing / hole filling
# ----------------------------------------------------------------------


def _ring_area(ring):
    return -_ring_signed_area_idx(ring)


def test_single_cell_ring_is_anticlockwise_on_map():
    rings = trace_rings(np.array([[True]]))
    assert len(rings) == 1
    assert _ring_signed_area_idx(rings[0]) == -1.0  # outer ring
    assert simplify_ring(rings[0])[0] == simplify_ring(rings[0])[-1]
    assert len(simplify_ring(rings[0])) == 5


def test_hole_is_filled_and_island_absorbed():
    mask = np.zeros((7, 7), dtype=bool)
    mask[0:7, 0:7] = True
    mask[2:5, 2:5] = False  # hole
    mask[3, 3] = True  # island inside the hole
    filled, rings = fill_holes(mask)
    assert filled.all()
    assert len(rings) == 1
    assert _ring_area(rings[0]) == 49


def test_diagonal_contact_gives_two_parts():
    mask = np.array([[True, False], [False, True]])
    filled, rings = fill_holes(mask)
    assert (filled == mask).all()
    assert len(rings) == 2
    assert all(_ring_signed_area_idx(r) < 0 for r in rings)


def test_hole_enclosed_only_through_diagonal_contacts_is_filled():
    # a diamond of cells touching at corners encloses the centre cell
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 1] = mask[1, 0] = mask[1, 2] = mask[2, 1] = True
    filled, rings = fill_holes(mask)
    assert filled[1, 1]
    assert sum(_ring_area(r) for r in rings) == filled.sum()


@pytest.mark.parametrize("seed", range(40))
def test_keep_out_cells_stay_as_valid_holes(seed):
    rng = np.random.default_rng(500 + seed)
    mask = rng.random((25, 25)) > 0.35
    keep_out = ~mask & (rng.random((25, 25)) > 0.5)
    filled, rings = fill_holes(mask, keep_out)
    assert not (filled & keep_out).any()
    assert (filled | ~mask).all()
    polys = group_polygons(rings)
    total = sum(
        -_ring_signed_area_idx(p[0]) - sum(_ring_signed_area_idx(h) for h in p[1:])
        for p in polys
    )
    assert total == filled.sum()

    pytest.importorskip("shapely")
    from shapely import wkt

    geom = wkt.loads(
        rings_to_multipolygon_wkt(
            [[[(x, -y) for x, y in simplify_ring(r)] for r in p] for p in polys]
        )
    )
    assert geom.is_valid
    assert geom.area == pytest.approx(filled.sum())


def reference_fill(mask):
    """Slow reference: background cells not 4-connected to the border."""
    rows, cols = mask.shape
    outside = np.zeros_like(mask)
    stack = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r in (0, rows - 1) or c in (0, cols - 1)) and not mask[r, c]
    ]
    while stack:
        r, c = stack.pop()
        if outside[r, c] or mask[r, c]:
            continue
        outside[r, c] = True
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= r + dr < rows and 0 <= c + dc < cols:
                stack.append((r + dr, c + dc))
    return ~outside


@pytest.mark.parametrize("seed", range(20))
def test_fill_holes_matches_flood_fill_reference(seed):
    rng = np.random.default_rng(100 + seed)
    mask = rng.random((30, 30)) > rng.uniform(0.3, 0.7)
    filled, _ = fill_holes(mask)
    assert (filled == reference_fill(mask)).all()


@pytest.mark.parametrize("seed", range(40))
def test_random_masks_area_and_validity(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((25, 25)) > 0.45
    filled, rings = fill_holes(mask)
    assert (filled | ~mask).all()  # never drops a cell
    assert sum(_ring_area(r) for r in rings) == filled.sum()
    assert all(_ring_signed_area_idx(r) < 0 for r in rings)

    pytest.importorskip("shapely")
    from shapely import wkt

    polys = [
        [[(x, -y) for x, y in simplify_ring(r)] for r in poly]
        for poly in group_polygons(rings)
    ]
    geom = wkt.loads(rings_to_multipolygon_wkt(polys))
    assert geom.is_valid  # valid before the QGIS wrapper's makeValid
    assert geom.area == pytest.approx(filled.sum())


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------


def test_profile_slopes_on_straight_profile():
    x = np.linspace(0, 1000, 51)
    z = 100 + 0.02 * x
    assert slope_10_85(x, z) == pytest.approx(0.02)
    assert equal_area_slope(x, z) == pytest.approx(0.02)


def test_equal_area_slope_concave_profile():
    x = np.linspace(0, 1000, 1001)
    z = 1e-5 * x**2  # area under = 1e-5 * L^3 / 3
    assert equal_area_slope(x, z) == pytest.approx(2 * (1e-5 * 1000**3 / 3) / 1000**2)


def test_tc_formulas_hand_checked():
    # L = 1 km, S = 0.01 m/m
    assert tc_kirpich_minutes(1000.0, 0.01) == pytest.approx(
        0.0195 * 1000**0.77 * 0.01**-0.385
    )
    assert tc_kirpich_minutes(1000.0, 0.01) == pytest.approx(23.45, abs=0.05)
    assert tc_usbr_minutes(1.0, 0.01) == pytest.approx(
        60 * (0.87 / 10.0) ** 0.385, rel=1e-9
    )
    assert tc_usbr_minutes(1.0, 0.01) == pytest.approx(23.44, abs=0.05)
    # Bransby-Williams: 58 L / (A^0.1 Se^0.2), Se in m/km
    assert tc_bransby_williams_minutes(2.0, 4.0, 0.005) == pytest.approx(
        58 * 2.0 / (4.0**0.1 * 5.0**0.2)
    )
    assert tc_kirpich_minutes(1000.0, 0.0) is None
    assert tc_usbr_minutes(1.0, None) is None


def test_horn_slope_on_plane():
    y, x = np.mgrid[0:6, 0:6].astype(float)
    z = 0.03 * x * CELL + 0.04 * y * CELL
    s = horn_slope(np.pad(z, 1, constant_values=np.nan), CELL, CELL)
    np.testing.assert_allclose(s[1:-1, 1:-1], 0.05)


def test_hypsometric_curve_linear():
    z = np.arange(0, 101, dtype=float)
    curve = hypsometric_curve(z, 100.0)
    assert curve[0]["rel_area"] == 1.0
    assert curve[-1]["rel_area"] == pytest.approx(1 / 101)
    assert curve[10]["rel_height"] == 0.5
    assert curve[10]["rel_area"] == pytest.approx(51 / 101)


# ----------------------------------------------------------------------
# End-to-end delineation
# ----------------------------------------------------------------------


def test_single_outlet_matches_bruteforce_and_attributes():
    dem = valley_dem()
    pointer = pointer_for(dem)
    x, y = cxy(39, 26)  # one cell east of the valley outlet: snapping fixes it
    res = delineate_catchments(
        pointer, PTR_NODATA, dem, np.nan, GT, [Outlet("A", "point", [[(x, y)]])]
    )
    assert not res.warnings
    c = res.catchments[0]
    a = c.attributes
    assert (a["x_out"], a["y_out"]) == cxy(39, 25)
    assert a["snap_m"] == pytest.approx(CELL)
    expected = upstream_mask_bruteforce(pointer, [(39, 25)])
    assert a["area_km2"] == pytest.approx(expected.sum() * CELL * CELL / 1e6)
    assert a["area_km2"] == pytest.approx(40 * 50 * 100 / 1e6)
    assert a["hole_ha"] == 0 and a["n_parts"] == 1
    assert a["z_min"] == pytest.approx(float(np.min(dem)), abs=1e-3)
    assert a["relief_m"] == pytest.approx(float(np.ptp(dem)), abs=1e-3)
    assert a["perim_km"] == pytest.approx(2 * (400 + 500) / 1000)
    # LFP from a top corner: diagonal run to the valley, then straight down
    assert a["lfp_km"] > 0.39
    assert 0 < a["s_1085"] < 1.5 and a["s_ea"] > 0 and a["s_avg"] > 0
    assert a["tc_kirp_mn"] > 0 and a["tc_usbr_mn"] > 0 and a["tc_bw_mn"] > 0
    assert 0 < a["hyps_int"] < 1
    assert 0 < a["lca_km"] < a["lfp_km"]
    assert c.flow_path[-1] == pytest.approx(cxy(39, 25))
    assert res.dem_array.shape == (40, 50)


def test_total_and_incremental_nested_outlets():
    dem = valley_dem()
    pointer = pointer_for(dem)
    outlets = [
        Outlet("down", "point", [[cxy(39, 25)]]),
        Outlet("up", "point", [[cxy(20, 25)]]),
    ]
    tot = delineate_catchments(
        pointer, PTR_NODATA, dem, np.nan, GT, outlets, snap_radius_cells=0
    )
    inc = delineate_catchments(
        pointer,
        PTR_NODATA,
        dem,
        np.nan,
        GT,
        outlets,
        snap_radius_cells=0,
        mode="incremental",
    )
    t = {c.outlet_id: c.attributes for c in tot.catchments}
    i = {c.outlet_id: c.attributes for c in inc.catchments}
    up_ref = upstream_mask_bruteforce(pointer, [(20, 25)]).sum() * 1e-4
    assert t["up"]["area_km2"] == pytest.approx(up_ref)
    assert i["up"]["area_km2"] == pytest.approx(up_ref)
    assert i["down"]["area_km2"] == pytest.approx(
        t["down"]["area_km2"] - t["up"]["area_km2"]
    )
    assert t["up"]["ds_ids"] == "down" and t["down"]["ds_ids"] == ""


def test_pour_line_across_valley():
    dem = valley_dem()
    pointer = pointer_for(dem)
    line = [cxy(30, 10), cxy(30, 40)]
    res = delineate_catchments(
        pointer, PTR_NODATA, dem, np.nan, GT, [Outlet("L", "line", [line])]
    )
    expected = upstream_mask_bruteforce(pointer, line_cells(line, GT, dem.shape))
    a = res.catchments[0].attributes
    assert a["area_km2"] == pytest.approx(expected.sum() * 1e-4)
    assert a["snap_m"] is None and a["n_outcells"] == 31


def test_crater_creates_filled_hole_and_warning():
    # a mound on the valley side with a small crater on top: the crater
    # drains to its own pit, surrounded by cells that do reach the outlet
    dem = valley_dem()
    y, x = np.mgrid[0:40, 0:50]
    d = np.hypot(y - 15, x - 12)
    bump = np.where(d <= 10, 30 - 3 * d, 0.0)
    dem = dem + np.where(d < 2, 18 + 3 * d, bump)
    pointer = pointer_for(dem)
    res = delineate_catchments(
        pointer,
        PTR_NODATA,
        dem,
        np.nan,
        GT,
        [Outlet("A", "point", [[cxy(39, 25)]])],
        snap_radius_cells=0,
    )
    a = res.catchments[0].attributes
    drains = upstream_mask_bruteforce(pointer, [(39, 25)])
    assert a["area_km2"] == pytest.approx(drains.sum() * 1e-4)
    assert a["hole_ha"] == pytest.approx(0.2)  # the 20-cell crater
    assert a["poly_km2"] == pytest.approx(a["area_km2"] + a["hole_ha"] / 100)
    assert a["n_parts"] == 1
    assert any("interior holes" in w for w in res.warnings)
    # the catchment DEM covers the filled outline, crater included
    n_dem = int(np.sum(res.dem_array != res.dem_nodata))
    assert n_dem * 1e-4 == pytest.approx(a["poly_km2"])


def test_outlet_off_grid_is_skipped_with_warning():
    dem = valley_dem()
    res = delineate_catchments(
        pointer_for(dem),
        PTR_NODATA,
        dem,
        np.nan,
        GT,
        [Outlet("far", "point", [[(0.0, 0.0)]]), Outlet("ok", "point", [[cxy(5, 5)]])],
    )
    assert [c.outlet_id for c in res.catchments] == ["ok"]
    assert any("far" in w for w in res.warnings)


def test_esri_pointer_gives_same_result():
    dem = valley_dem()
    wbt = pointer_for(dem)
    esri = compute_d8_pointer(dem, np.nan, CELL, CELL, esri_style=True)
    out = [Outlet("A", "point", [[cxy(30, 25)]])]
    a = delineate_catchments(wbt, PTR_NODATA, dem, np.nan, GT, out)
    b = delineate_catchments(esri, PTR_NODATA, dem, np.nan, GT, out, esri_style=True)
    assert a.catchments[0].attributes == b.catchments[0].attributes


def test_bad_inputs_raise():
    with pytest.raises(CatchmentError):
        check_geotransform((0, 1, 0.5, 0, 0, -1))
    with pytest.raises(CatchmentError):
        check_same_grid((2, 2), GT, (2, 3), GT)
    with pytest.raises(CatchmentError):
        check_same_grid((2, 2), GT, (2, 2), (1001.0, CELL, 0, 5000.0, 0, -CELL))
    dem = valley_dem()
    with pytest.raises(CatchmentError):
        delineate_catchments(
            pointer_for(dem), PTR_NODATA, dem, np.nan, GT, [], mode="bogus"
        )


def test_csv_exports(tmp_path):
    dem = valley_dem()
    res = delineate_catchments(
        pointer_for(dem),
        PTR_NODATA,
        dem,
        np.nan,
        GT,
        [Outlet("A", "point", [[cxy(39, 25)]])],
    )
    report = tmp_path / "report.csv"
    hypso = tmp_path / "hypso.csv"
    write_parameters_csv(str(report), res.catchments)
    write_hypsometric_csv(str(hypso), res.catchments)
    rows = list(csv.DictReader(report.open()))
    assert rows[0]["outlet_id"] == "A"
    assert math.isclose(float(rows[0]["area_km2"]), 0.2)
    h = list(csv.DictReader(hypso.open()))
    assert len(h) == 21 and h[0]["rel_area"] == "1.0"


def test_file_roundtrip_with_gdal(tmp_path):
    pytest.importorskip("osgeo.gdal")
    from mayim_tools.hydrology.catchment_delineation.core import (
        read_raster,
        write_raster,
    )

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = str(tmp_path / "t.tif")
    write_raster(path, arr, GT, "", -9999.0)
    back, gt, _, nodata = read_raster(path)
    np.testing.assert_allclose(back, arr)
    assert tuple(gt) == GT and nodata == -9999.0


def test_incremental_polygons_are_valid_and_do_not_overlap():
    pytest.importorskip("shapely")
    from shapely import wkt
    from shapely.ops import unary_union

    rows = cols = 120
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    dem = np.abs(x - 60) * 0.3 + (rows - y) * 0.5 + 2 * np.sin(x / 7) * np.cos(y / 9)
    pointer = pointer_for(dem)
    rng = np.random.default_rng(7)
    outlets = [
        Outlet(str(i), "point", [[cxy(*rng.integers(0, rows, 2))]]) for i in range(40)
    ]
    res = delineate_catchments(
        pointer, PTR_NODATA, dem, np.nan, GT, outlets, mode="incremental"
    )
    geoms = [wkt.loads(rings_to_multipolygon_wkt(c.polygons)) for c in res.catchments]
    assert all(g.is_valid for g in geoms)
    total = sum(g.area for g in geoms)
    assert unary_union(geoms).area == pytest.approx(total)
    polys = sum(c.attributes["poly_km2"] for c in res.catchments)
    assert total / 1e6 == pytest.approx(polys)


# ----------------------------------------------------------------------
# Geomorphometry, description and report
# ----------------------------------------------------------------------


def test_strahler_order_y_network():
    from mayim_tools.hydrology.catchment_delineation.core import (
        bifurcation_ratio,
        segment_starts,
        strahler_order,
        stream_order_summary,
    )

    # cells 0-1 and 2-3 are two headwater branches joining at 4, then 5, 6
    # a single first-order side branch 7 joins at 5
    nxt = np.array([1, 4, 3, 4, 5, 6, 6, 5])
    stream = np.ones(8, dtype=bool)
    order = strahler_order(nxt, stream)
    assert order.tolist() == [1, 1, 1, 1, 2, 2, 2, 1]
    summary = stream_order_summary(
        order, segment_starts(nxt, order), nxt, np.ones(8) * 1000.0
    )
    assert [s["n_segments"] for s in summary] == [3, 1]
    assert [s["length_km"] for s in summary] == [5.0, 2.0]
    assert bifurcation_ratio(summary) == 3.0


def test_aspect_and_slope_classes():
    from mayim_tools.hydrology.catchment_delineation.core import (
        aspect_distribution,
        horn_aspect_deg,
        horn_gradient,
        slope_class_distribution,
    )

    y, x = np.mgrid[0:5, 0:5].astype(float)
    z = -y * CELL * 0.1  # falls towards the south (row increases)
    dzdx, dzdy = horn_gradient(np.pad(z, 1, constant_values=np.nan), CELL, CELL)
    aspect = horn_aspect_deg(dzdx, dzdy)[1:-1, 1:-1]
    np.testing.assert_allclose(aspect, 180.0)
    dist = aspect_distribution(aspect.ravel(), np.full(9, 0.1))
    assert {d["label"]: d["pct"] for d in dist}["S"] == 100.0
    classes = slope_class_distribution(np.array([1.0, 3.0, 20.0, 70.0]))
    assert [c["pct"] for c in classes] == [25.0, 25.0, 0.0, 0.0, 25.0, 0.0, 25.0]


def test_geomorphometric_attributes_consistent():
    dem = valley_dem()
    res = delineate_catchments(
        pointer_for(dem),
        PTR_NODATA,
        dem,
        np.nan,
        GT,
        [Outlet("A", "point", [[cxy(39, 25)]])],
        stream_threshold_km2=0.002,
    )
    a = res.catchments[0].attributes
    h = a["z_max"] - a["z_outlet"]
    assert a["melton_r"] == pytest.approx(h / 1000 / math.sqrt(a["area_km2"]), rel=1e-3)
    assert a["lo_km"] == pytest.approx(1 / (2 * a["dd_kmkm2"]), abs=1e-4)
    assert a["c_maint"] == pytest.approx(1 / a["dd_kmkm2"], abs=1e-4)
    assert a["strm_ord"] >= 1 and a["n_strm"] >= 1
    assert a["sinuosity"] >= 1.0
    assert a["lb_km"] == pytest.approx(math.hypot(250, 390) / 1000, rel=0.01)
    detail = res.catchments[0].detail
    assert sum(c["pct"] for c in detail["slope_classes"]) == pytest.approx(100.0)
    assert sum(c["pct"] for c in detail["aspect"]) == pytest.approx(100.0)
    assert detail["streams"].shape[1] == 5


def test_description_classes():
    from mayim_tools.hydrology.catchment_delineation.description import (
        ELONGATION_CLASSES,
        SIZE_CLASSES,
        classify,
        hypsometric_stage,
        melton_screening,
    )

    assert classify(1.0, SIZE_CLASSES) == "small"
    assert classify(2.5, SIZE_CLASSES) == "midsize"
    assert classify(300.0, SIZE_CLASSES) == "large"
    assert classify(0.95, ELONGATION_CLASSES) == "circular"
    assert classify(0.45, ELONGATION_CLASSES) == "more elongated"
    assert hypsometric_stage(0.7).startswith("youthful")
    assert hypsometric_stage(0.5).startswith("mature")
    assert hypsometric_stage(0.2).startswith("old")
    assert melton_screening(0.2, 5.0) == "flood"
    assert melton_screening(0.5, 1.0) == "debris flood"
    assert melton_screening(0.8, 1.0) == "debris flow"
    assert melton_screening(0.8, 3.0) == "debris flood"


def test_describe_catchment_text():
    from mayim_tools.hydrology.catchment_delineation.description import (
        describe_catchment,
    )

    dem = valley_dem()
    res = delineate_catchments(
        pointer_for(dem),
        PTR_NODATA,
        dem,
        np.nan,
        GT,
        [Outlet("A", "point", [[cxy(39, 25)]])],
        stream_threshold_km2=0.002,
    )
    c = res.catchments[0]
    text = describe_catchment(c.attributes, c.detail)
    assert "Catchment A drains 0.200 km²" in text["overview"]
    joined = " ".join(text["geomorphology"])
    assert "small catchment" in joined and "Strahler order" in joined
    assert any("Time of concentration" in p for p in text["modelling"])


def test_word_report(tmp_path):
    docx = pytest.importorskip("docx")
    from mayim_tools.hydrology.catchment_delineation.report import write_report_docx

    dem = valley_dem()
    outlets = [
        Outlet("down", "point", [[cxy(39, 25)]]),
        Outlet("up", "line", [[cxy(20, 20), cxy(20, 30)]]),
    ]
    res = delineate_catchments(
        pointer_for(dem), PTR_NODATA, dem, np.nan, GT, outlets, mode="incremental"
    )
    path = tmp_path / "report.docx"
    include = True
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        include = False
    write_report_docx(
        str(path),
        res.catchments,
        {"version": "test", "mode": "incremental", "dem": {"cell_x": 10}},
        include_figures=include,
    )
    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Catchment Delineation and Characterisation" in text
    assert "5.2 Catchment up" in text and "References" in text
    if include:
        assert len(doc.inline_shapes) == 8  # four figures per catchment


def test_stream_links_split_at_junctions():
    from mayim_tools.hydrology.catchment_delineation.core import (
        strahler_order,
        stream_links,
    )

    # two branches 0-1 and 2-3 join at 4, side branch 7 joins at 5, 6 outlet
    nxt = np.array([1, 4, 3, 4, 5, 6, 6, 5])
    stream = np.ones(8, dtype=bool)
    order = strahler_order(nxt, stream)
    links = {
        tuple(lk["cells"]): lk for lk in stream_links(nxt, stream, order, np.ones(8))
    }
    assert set(links) == {(0, 1, 4), (2, 3, 4), (4, 5), (7, 5), (5, 6)}
    ids = {lk["link_id"]: cells for cells, lk in links.items()}
    assert ids[links[(0, 1, 4)]["ds_link"]] == (4, 5)
    assert ids[links[(7, 5)]["ds_link"]] == (5, 6)
    assert links[(5, 6)]["ds_link"] is None
    assert links[(4, 5)]["order"] == 2 and links[(7, 5)]["order"] == 1
    assert links[(0, 1, 4)]["length_m"] == 2.0
    assert links[(5, 6)]["length_m"] == 1.0  # the outlet cell adds no step


def test_stream_links_and_centroid_in_results():
    dem = valley_dem()
    res = delineate_catchments(
        pointer_for(dem),
        PTR_NODATA,
        dem,
        np.nan,
        GT,
        [Outlet("A", "point", [[cxy(39, 25)]])],
        stream_threshold_km2=0.002,
    )
    c = res.catchments[0]
    links = c.detail["stream_links"]
    ids = {lk["link_id"] for lk in links}
    assert all(lk["ds_link"] in ids for lk in links if lk["ds_link"] is not None)
    assert sum(lk["ds_link"] is None for lk in links) == 1  # one outlet link
    total_km = sum(lk["length_m"] for lk in links) / 1000
    assert total_km == pytest.approx(c.attributes["strm_km"], abs=1e-3)
    assert max(lk["strahler"] for lk in links) == c.attributes["strm_ord"]
    outlet_link = next(lk for lk in links if lk["ds_link"] is None)
    assert outlet_link["coords"][-1] == pytest.approx(cxy(39, 25))
    assert outlet_link["us_km2"] == pytest.approx(c.attributes["area_km2"])
    assert c.detail["centroid_inside"] is True
    assert c.detail["centroid_z"] is not None
