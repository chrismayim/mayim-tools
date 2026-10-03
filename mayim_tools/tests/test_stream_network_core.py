"""
Tests for mayim_tools/hydrology/stream_network (core, export, report) and
the shared hydrology/_common/d8_network module. Zero QGIS dependency.
"""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from mayim_tools.hydrology._common import d8_network
from mayim_tools.hydrology.catchment_delineation import core as catchment_core
from mayim_tools.hydrology.d8_flow_accumulation.core import compute_d8_pointer
from mayim_tools.hydrology.stream_network.core import (
    PourPoint,
    RoutingSettings,
    StreamNetworkError,
    _loglinear,
    constant_drop_test,
    extract_stream_network,
    horton_statistics,
    order_reaches,
    reach_routing,
    segment_drops,
    strahler_streams,
    threshold_series,
    welch_t,
)
from mayim_tools.hydrology.stream_network.export import (
    REACH_FIELDS,
    write_sectioned_csv,
)

PTR_NODATA = -32768.0
CELL = 10.0
GT = (1000.0, CELL, 0.0, 5000.0, 0.0, -CELL)


def cxy(row, col):
    return GT[0] + (col + 0.5) * CELL, GT[3] - (row + 0.5) * CELL


def dendritic_dem(rows=60, cols=60, seed=3):
    """Valley falling south with rough sides, pits filled (priority flood)."""
    import heapq

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    z = (rows - y) * 1.0 + np.abs(x - cols / 2) * 0.4 + rng.random((rows, cols)) * 3
    out = np.full_like(z, np.inf)
    seen = np.zeros_like(z, dtype=bool)
    heap = []
    for r in range(rows):
        for c in range(cols):
            if r in (0, rows - 1) or c in (0, cols - 1):
                heapq.heappush(heap, (z[r, c], r, c))
                seen[r, c] = True
                out[r, c] = z[r, c]
    while heap:
        v, r, c = heapq.heappop(heap)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and not seen[rr, cc]:
                    seen[rr, cc] = True
                    out[rr, cc] = max(z[rr, cc], v + 1e-3)
                    heapq.heappush(heap, (out[rr, cc], rr, cc))
    return out


def pointer_for(dem):
    return compute_d8_pointer(dem, np.nan, CELL, CELL)


def reach(lid, ds, strahler, length, area, net="N1", z0=None, z1=None):
    return {
        "link_id": lid,
        "ds_link": ds,
        "strahler": strahler,
        "length_m": length,
        "us_km2": area,
        "network_id": net,
        "z_start": z0,
        "z_end": z1,
    }


# ----------------------------------------------------------------------
# Shared module
# ----------------------------------------------------------------------


def test_catchment_core_reexports_shared_network_functions():
    for name in (
        "pointer_to_direction_index",
        "build_downstream_index",
        "resolve_terminals",
        "flow_accumulation_cells",
        "strahler_order",
        "stream_links",
        "read_raster",
    ):
        assert getattr(catchment_core, name) is getattr(d8_network, name)
    assert catchment_core.CatchmentError is d8_network.GridError


def test_stream_links_break_cells():
    # straight chain 0 -> 1 -> 2 -> 3 -> 4 (outlet); break at cell 2
    nxt = np.array([1, 2, 3, 4, 4])
    stream = np.ones(5, dtype=bool)
    order = np.ones(5, dtype=np.int32)
    breaks = np.zeros(5, dtype=bool)
    breaks[2] = True
    links = d8_network.stream_links(nxt, stream, order, np.ones(5), break_cells=breaks)
    cells = sorted(tuple(lk["cells"]) for lk in links)
    assert cells == [(0, 1, 2), (2, 3, 4)]


# ----------------------------------------------------------------------
# Ordering on the reach graph
# ----------------------------------------------------------------------


def _y_network():
    """Reaches: 1 & 2 (order 1) join -> 4 (order 2); 3 (order 1, long)
    joins 4 -> 5 (order 2, outlet). Lengths chosen so 3 is the longest
    upstream path."""
    return [
        reach(1, 4, 1, 100.0, 0.1),
        reach(2, 4, 1, 50.0, 0.1),
        reach(3, 5, 1, 400.0, 0.3),
        reach(4, 5, 2, 100.0, 0.3),
        reach(5, None, 2, 100.0, 0.7),
    ]


def test_order_reaches_systems():
    rs = _y_network()
    order_reaches(rs)
    by = {r["link_id"]: r for r in rs}
    assert [by[i]["shreve"] for i in range(1, 6)] == [1, 1, 1, 2, 3]
    # Horton: at the outlet the order-2 branch (4) is the main stream, and
    # above 4 the longer branch (1) continues order 2
    assert by[5]["horton"] == 2 and by[4]["horton"] == 2
    assert by[1]["horton"] == 2 and by[2]["horton"] == 1 and by[3]["horton"] == 1
    # Hack: main stem follows the longest path (3 is 400 m > 4 + 1 = 200 m)
    assert by[5]["hack"] == 1 and by[3]["hack"] == 1
    assert by[4]["hack"] == 2 and by[1]["hack"] == 2 and by[2]["hack"] == 3
    assert by[2]["links_to_out"] == 3 and by[5]["links_to_out"] == 1
    assert by[1]["dist_out_m"] == 200.0 and by[5]["dist_out_m"] == 0.0
    assert by[5]["up_len_m"] == 500.0
    assert by[3]["is_main"] == 1 and by[4]["is_main"] == 0


def test_strahler_streams_and_horton_statistics():
    rs = _y_network()
    for r in rs:
        r["z_start"], r["z_end"] = 10.0, 0.0
    order_reaches(rs)
    streams = strahler_streams(rs)
    by_order = {}
    for s in streams:
        by_order.setdefault(s["order"], []).append(s)
    assert len(by_order[1]) == 3 and len(by_order[2]) == 1
    assert by_order[2][0]["length_m"] == 200.0  # reaches 4 + 5 chained
    rows, fit = horton_statistics(streams, "N1")
    assert [r["n_streams"] for r in rows] == [3, 1]
    assert rows[0]["rb_next"] == 3.0
    assert fit["rb_fit"] == pytest.approx(3.0)


def test_loglinear_recovers_geometric_ratios():
    orders = [1, 2, 3, 4]
    n = [64, 16, 4, 1]
    ratio, r2 = _loglinear(orders, n, decreasing=True)
    assert ratio == pytest.approx(4.0) and r2 == pytest.approx(1.0)
    ratio, _ = _loglinear(orders, [0.5, 1.0, 2.0, 4.0])
    assert ratio == pytest.approx(2.0)
    assert _loglinear([1], [3])[0] is None


# ----------------------------------------------------------------------
# Drop test, routing
# ----------------------------------------------------------------------


def test_welch_t_matches_formula():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0])
    expected = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / 4 + b.var(ddof=1) / 3)
    assert welch_t(a, b) == pytest.approx(expected)
    assert welch_t(a[:1], b) is None


def test_segment_drops_on_cell_network():
    # cells 0,1 (order 1) -> 4 ; cells 2,3 (order 1) -> 4 ; 4 -> 5 (order 2)
    nxt = np.array([1, 4, 3, 4, 5, 5])
    stream = np.ones(6, dtype=bool)
    order = np.array([1, 1, 1, 1, 2, 2], dtype=np.int32)
    z = np.array([10.0, 8.0, 12.0, 9.0, 5.0, 1.0])
    drops, orders = segment_drops(nxt, stream, order, z)
    got = sorted(zip(orders.tolist(), drops.tolist(), strict=True))
    # first-order streams drop to the junction cell 4 (z = 5)
    assert got == [(1, 5.0), (1, 7.0), (2, 4.0)]


def test_threshold_series_and_validation():
    s = threshold_series(0.01, 1.0, 3)
    assert s == pytest.approx([0.01, 0.1, 1.0])
    with pytest.raises(StreamNetworkError):
        threshold_series(1.0, 0.5, 3)


def test_reach_routing():
    st = RoutingSettings(method="manning", manning_n=0.04, hydraulic_radius_m=1.0)
    out = reach_routing(1200.0, 0.01, st)
    v = 1.0 * math.sqrt(0.01) / 0.04
    assert out["v_ms"] == pytest.approx(v)
    assert out["tt_min"] == pytest.approx(1200.0 / v / 60.0)
    assert out["musk_k_h"] == pytest.approx(out["tt_min"] / 60.0)
    # flat reach falls back to the minimum slope
    flat = reach_routing(100.0, -0.1, st)
    assert flat["v_ms"] == pytest.approx(math.sqrt(st.min_slope) / 0.04)
    const = reach_routing(
        600.0, 0.02, RoutingSettings(method="constant", velocity_ms=2.0)
    )
    assert const["tt_min"] == pytest.approx(5.0)
    assert const["n_sub"] == 1 and const["dt_check"] == "ok"
    short = reach_routing(
        60.0, 0.02, RoutingSettings(method="constant", velocity_ms=2.0)
    )
    assert short["dt_check"].startswith("dt above")


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def terrain():
    dem = dendritic_dem()
    return dem, pointer_for(dem)


def test_whole_raster_network(terrain):
    dem, ptr = terrain
    res = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0.01,
        drop_test_range=(0.002, 0.1, 6), min_network_km2=0.05,
    )  # fmt: skip
    assert res.networks and res.reaches
    ids = {r["link_id"] for r in res.reaches}
    assert all(r["ds_link"] in ids for r in res.reaches if r["ds_link"] is not None)
    for net in res.networks:
        mine = [r for r in res.reaches if r["network_id"] == net["network_id"]]
        root = [r for r in mine if r["ds_link"] is None]
        assert len(root) == 1
        heads = [n for n in res.nodes if n["network_id"] == net["network_id"]
                 and n["node_type"] == "channel head"]  # fmt: skip
        assert root[0]["shreve"] == len(heads) == net["n_heads"]
        assert net["max_strahler"] == max(r["strahler"] for r in mine)
        assert net["total_length_km"] == pytest.approx(
            sum(r["length_m"] for r in mine) / 1000, abs=1e-3
        )
        main = [r for r in mine if r["hack"] == 1]
        assert sum(r["length_m"] for r in main) / 1000 == pytest.approx(
            net["main_length_km"], abs=1e-3
        )
        assert all(r["horton"] >= r["strahler"] for r in mine)
        prof = res.profiles[net["network_id"]]
        d = [p["dist_from_outlet_m"] for p in prof]
        assert d == sorted(d, reverse=True) and d[-1] == pytest.approx(0.0)
    assert len(res.drop_test) == 6
    for r in res.reaches:
        assert set(name for name, *_ in REACH_FIELDS) <= set(r)


def test_strahler_matches_cell_level_ordering(terrain):
    dem, ptr = terrain
    res = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0.01, drop_test_range=None
    )
    di, _ = d8_network.pointer_to_direction_index(ptr, PTR_NODATA)
    nxt = d8_network.build_downstream_index(di)
    acc, _ = d8_network.flow_accumulation_cells(nxt, (di != -2).ravel())
    stream = np.nan_to_num(acc) * 1e-4 >= 0.01
    order = d8_network.strahler_order(nxt, stream)
    cols = dem.shape[1]
    for r in res.reaches:
        x, y = r["coords"][0]
        row, col = d8_network.world_to_cell(x, y, GT)
        assert order[row * cols + col] == r["strahler"]


def test_pour_points_split_reaches_and_limit_domain(terrain):
    dem, ptr = terrain
    full = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0.01, drop_test_range=None
    )
    biggest = max(full.networks, key=lambda n: n["area_km2"])
    main = [r for r in full.reaches
            if r["network_id"] == biggest["network_id"] and r["hack"] == 1]  # fmt: skip
    mid = max(main, key=lambda r: r["length_m"])
    k = len(mid["coords"]) // 2
    pp = [
        PourPoint("Outlet", biggest["outlet_x"], biggest["outlet_y"]),
        PourPoint("Mid", *mid["coords"][k]),
    ]
    res = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, pour_points=pp, snap_radius_cells=0,
        threshold_km2=0.01, drop_test_range=None,
    )  # fmt: skip
    assert [n["network_id"] for n in res.networks] == ["Outlet"]
    assert res.networks[0]["area_km2"] == pytest.approx(biggest["area_km2"])
    assert (
        len(res.reaches)
        == sum(1 for r in full.reaches if r["network_id"] == biggest["network_id"]) + 1
    )  # the mid pour point splits one reach
    assert {r["outlet_id"] for r in res.reaches} == {"Outlet", "Mid"}
    assert all(r["length_m"] > 0 for r in res.reaches)
    kinds = [n["node_type"] for n in res.nodes]
    assert kinds.count("pour point") == 1 and kinds.count("outlet") == 1


def test_bad_inputs(terrain):
    dem, ptr = terrain
    with pytest.raises(StreamNetworkError):
        extract_stream_network(ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0)
    with pytest.raises(StreamNetworkError):
        extract_stream_network(
            ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=1e6, drop_test_range=None
        )
    with pytest.raises(StreamNetworkError):
        extract_stream_network(
            ptr, PTR_NODATA, dem, np.nan, GT,
            pour_points=[PourPoint("far", 0.0, 0.0)], drop_test_range=None,
        )  # fmt: skip


def test_drop_test_detects_overextended_streams():
    # plane hillslope draining to a valley: very small thresholds make
    # parallel hillslope "streams" whose drops differ from channel drops
    dem = dendritic_dem(seed=7)
    ptr = pointer_for(dem)
    di, _ = d8_network.pointer_to_direction_index(ptr, PTR_NODATA)
    nxt = d8_network.build_downstream_index(di)
    valid = (di != -2).ravel()
    acc, _ = d8_network.flow_accumulation_cells(nxt, valid)
    rows, best = constant_drop_test(
        nxt, valid, np.nan_to_num(acc) * 1e-4, dem.ravel(), [0.0005, 0.005, 0.05]
    )
    assert len(rows) == 3
    assert rows[0]["n_first_order"] > rows[-1]["n_first_order"]
    assert best is None or best in [r["threshold_km2"] for r in rows]


def test_sectioned_csv(terrain, tmp_path):
    dem, ptr = terrain
    res = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0.01,
        drop_test_range=(0.002, 0.1, 4),
    )  # fmt: skip
    path = tmp_path / "net.csv"
    write_sectioned_csv(str(path), res, {"tool": "Stream Network", "version": "t"})
    text = path.read_text(encoding="utf-8")
    assert text.index("ABBREVIATIONS USED IN COLUMN HEADINGS") < text.index(
        "1. RUN SUMMARY"
    )
    for heading in (
        "1. RUN SUMMARY",
        "2. CONSTANT-DROP TEST",
        "3. NETWORK SUMMARY",
        "4. HORTON ORDER STATISTICS",
        "5. HORTON RATIOS",
        "6. REACHES",
        "7. NODES",
        "8. MAIN-STEM LONGITUDINAL PROFILES",
        "9. FIELD DEFINITIONS",
    ):
        assert heading in text
    rows = list(csv.reader(path.open(encoding="utf-8")))
    i = next(k for k, r in enumerate(rows) if r and r[0].startswith("6. REACHES"))
    assert rows[i + 1][:3] == ["link_id", "ds_link", "network_id"]
    assert len(rows[i + 2]) == len(REACH_FIELDS)
    assert "," not in "".join(rows[i + 2][10:14])  # plain numbers
    assert not any("e+" in c for r in rows for c in r)


def test_word_report(terrain, tmp_path):
    docx = pytest.importorskip("docx")
    from mayim_tools.hydrology.stream_network.report import (
        describe_network,
        write_report_docx,
    )

    dem, ptr = terrain
    res = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0.01,
        drop_test_range=(0.002, 0.1, 4),
    )  # fmt: skip
    paras = describe_network(res.networks[0], res.horton_fits[0], res.reaches)
    assert paras[0].startswith(f"Network {res.networks[0]['network_id']} drains")
    path = tmp_path / "net.docx"
    write_report_docx(str(path), res, {"tool": "Stream Network", "version": "t"})
    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    for heading in (
        "Stream Network Extraction and Classification",
        "4. Stream threshold and constant-drop test",
        "5. Stream ordering systems",
        "10. References",
    ):
        assert heading in text


def test_catchment_outlets_give_same_network_as_pour_points(terrain):
    dem, ptr = terrain
    full = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0.01, drop_test_range=None
    )
    big = max(full.networks, key=lambda n: n["area_km2"])
    outlets = [
        catchment_core.Outlet("A", "point", [[(big["outlet_x"], big["outlet_y"])]])
    ]
    delin = catchment_core.delineate_catchments(
        ptr, PTR_NODATA, dem, np.nan, GT, outlets, snap_radius_cells=0
    )
    via_catchment = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0.01, drop_test_range=None,
        outlet_cells=delin.outlet_cells,
    )  # fmt: skip
    via_points = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0.01, drop_test_range=None,
        pour_points=[PourPoint("A", big["outlet_x"], big["outlet_y"])],
        snap_radius_cells=0,
    )  # fmt: skip
    keys = ("strahler", "shreve", "horton", "hack", "length_m", "us_km2", "slope")
    a = sorted(tuple(r[k] for k in keys) for r in via_catchment.reaches)
    b = sorted(tuple(r[k] for k in keys) for r in via_points.reaches)
    assert a == b and len(a) > 3
    assert via_catchment.networks[0]["network_id"] == "A"


def test_catchment_pour_line_network(terrain):
    dem, ptr = terrain
    rows, cols = dem.shape
    y = GT[3] - (rows - 5.5) * CELL
    line = [(GT[0] + 2 * CELL, y), (GT[0] + (cols - 2) * CELL, y)]
    delin = catchment_core.delineate_catchments(
        ptr, PTR_NODATA, dem, np.nan, GT, [catchment_core.Outlet("L", "line", [line])]
    )
    net = extract_stream_network(
        ptr, PTR_NODATA, dem, np.nan, GT, threshold_km2=0.01, drop_test_range=None,
        outlet_cells=delin.outlet_cells,
    )  # fmt: skip
    line_cells = set(int(c) for c in delin.outlet_cells[0][1])
    assert net.reaches and all(r["outlet_id"] == "L" for r in net.reaches)
    for r in net.reaches:
        if r["ds_link"] is None:  # every network ends on the pour line
            row, col = d8_network.world_to_cell(*r["coords"][-1], GT)
            assert row * cols + col in line_cells
