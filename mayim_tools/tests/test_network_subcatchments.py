"""Tests of the network sub-catchments (Catchment Delineation incremental
mode): stream names, a sub-catchment per reach and per channel head, and
the optional split along the stream centreline."""

from __future__ import annotations

import numpy as np
import pytest

from mayim_tools.hydrology.catchment_delineation.subcatchments import (
    SUB_HEAD,
    SUB_STREAM,
    delineate_network_subcatchments,
    natural_key,
    split_along_centreline,
)
from mayim_tools.hydrology.stream_network.core import (
    extract_stream_network,
    name_network,
    split_positions,
    stem_letters,
)
from mayim_tools.tests.test_stream_network_core import (
    CELL,
    GT,
    PTR_NODATA,
    dendritic_dem,
    pointer_for,
)


def r(lid, ds, area, length=100.0):
    return {"link_id": lid, "ds_link": ds, "us_km2": area, "up_len_m": length}


def test_stem_letters():
    assert [stem_letters(i) for i in (0, 1, 25, 26, 27, 51, 52)] == [
        "A", "B", "Z", "AA", "AB", "AZ", "BA",
    ]  # fmt: skip


def test_names_follow_main_stem_and_branches():
    # outlet reach 1; 2 above it; 3 above 2 is the main stem (larger area),
    # 4 is a branch joining at the top of reach 2 (= A2), 5 above 4
    reaches = [r(1, None, 10), r(2, 1, 9), r(3, 2, 6), r(4, 2, 2), r(5, 4, 1)]
    nodes = [
        {"node_type": "outlet", "link_id": 1},
        {"node_type": "confluence", "link_id": 1},
        {"node_type": "confluence", "link_id": 2},
        {"node_type": "channel head", "link_id": 3},
        {"node_type": "stream breakpoint", "link_id": 4},
        {"node_type": "channel head", "link_id": 5},
    ]
    heads = name_network(reaches, nodes)
    names = {x["link_id"]: x["name"] for x in reaches}
    assert names == {1: "A1", 2: "A2", 3: "A3", 4: "A2A1", 5: "A2A2"}
    assert {x["link_id"]: x["ds_name"] for x in reaches} == {
        1: None, 2: "A1", 3: "A2", 4: "A2", 5: "A2A1",
    }  # fmt: skip
    assert heads == {3: "A4", 5: "A2A3"}
    # nodes are named after the stream that drains to them
    assert [nd["name"] for nd in nodes] == ["A1", "A2", "A3", "A4", "A2A2", "A2A3"]


def test_several_branches_and_systems_are_lettered_by_area():
    reaches = [
        r(1, None, 5), r(2, 1, 3), r(3, 1, 1), r(4, 1, 0.5),  # three inflows
        r(9, None, 20),  # a second, larger system
    ]  # fmt: skip
    name_network(reaches, [])
    names = {x["link_id"]: x["name"] for x in reaches}
    assert names[9] == "A1" and names[1] == "B1"
    assert names[2] == "B2" and names[3] == "B1A1" and names[4] == "B1B1"


def test_natural_order():
    names = ["A10", "A2", "A3A1", "A1b", "A1a", "A4", "A1A1a", "A3A2b", "B1", "A3A2a"]
    assert sorted(names, key=natural_key) == [
        "A1a", "A1b", "A1A1a", "A2", "A3A1", "A3A2a", "A3A2b", "A4", "A10", "B1",
    ]  # fmt: skip


def box_grid(rows=20, cols=21):
    from types import SimpleNamespace

    gt = (0.0, 10.0, 0.0, rows * 10.0, 0.0, -10.0)
    return SimpleNamespace(geotransform=gt, rows=rows, cols=cols, sx=10.0, sy=10.0)


def test_split_along_a_straight_centreline():
    g = box_grid()
    cells = np.arange(g.rows * g.cols)
    line = np.arange(g.rows) * g.cols + 10  # column 10, flowing south
    out = split_along_centreline(g, cells, line, np.ones(cells.size, bool))
    banks = dict(out["banks"])
    cols_a = set((banks["a"] % g.cols).tolist())
    cols_b = set((banks["b"] % g.cols).tolist())
    # looking south, east is the left bank; the line cells go with it
    assert cols_a == set(range(10, 21)) and cols_b == set(range(0, 10))
    assert banks["a"].size + banks["b"].size == cells.size
    xs = [p[0] for p in out["line_xy"]]
    ys = [p[1] for p in out["line_xy"]]
    assert set(xs) == {105.0}  # the cut runs through the cell centres
    assert max(ys) > 200.0 and min(ys) < 0.0  # extended past both ends
    assert out["n_pieces"] == 2 and out["n_merged"] == 0


def test_split_tolerance_area_and_width():
    g = box_grid()
    cells = np.arange(g.rows * g.cols)
    any_sub = np.ones(cells.size, bool)
    # a one-cell-wide strip west of the line (20 cells, width 1): not split
    line = np.arange(g.rows) * g.cols + 1
    assert split_along_centreline(g, cells, line, any_sub) is None
    # two cells wide passes the 2-cell width at the default tolerance ...
    line = np.arange(g.rows) * g.cols + 2
    assert split_along_centreline(g, cells, line, any_sub) is not None
    # ... but not with a stricter width or area tolerance
    assert split_along_centreline(g, cells, line, any_sub, min_width_cells=3) is None
    assert split_along_centreline(g, cells, line, any_sub, min_cells=41) is None


def test_small_pieces_join_their_neighbour():
    g = box_grid()
    rows, cols = g.rows, g.cols
    # a notch from the west boundary at row 16 cuts the west bank (left of a
    # line at column 2) into a large piece (rows 0-15) and a small one
    # (rows 17-19, 6 cells), which joins the east bank across the line
    keep = np.ones((rows, cols), bool)
    keep[16, :2] = False
    cells = np.flatnonzero(keep)
    line = np.arange(rows) * cols + 2
    out = split_along_centreline(g, cells, line, np.ones(rows * cols, bool))
    assert out is not None and out["n_pieces"] == 3 and out["n_merged"] == 1
    banks = dict(out["banks"])
    west_rows = set((banks["b"] // cols).tolist())
    assert west_rows == set(range(16))
    assert banks["a"].size + banks["b"].size == cells.size


def test_remainder_percentage():
    seg = [10.0] * 56  # 560 m
    assert split_positions(seg, 500, 0.10) == [6]  # remainder 60 m > 50 m
    assert split_positions(seg, 500, 0.20) == []  # 60 m < 100 m


@pytest.fixture(scope="module")
def setup():
    dem = dendritic_dem()
    ptr = pointer_for(dem)
    kw = dict(threshold_km2=0.005, drop_test_range=None)
    whole = extract_stream_network(ptr, PTR_NODATA, dem, None, GT, **kw)
    root = max(
        (x for x in whole.reaches if x["ds_link"] is None),
        key=lambda x: x["us_km2"],
    )
    outlet = [("Outlet", [root["_end_cell"]])]
    net = extract_stream_network(
        ptr, PTR_NODATA, dem, None, GT, outlet_cells=outlet,
        max_reach_length_m=80.0, **kw,
    )  # fmt: skip
    plain = delineate_network_subcatchments(
        ptr, PTR_NODATA, dem, None, GT, net, outlet, stream_threshold_km2=0.005
    )
    banks = delineate_network_subcatchments(
        ptr, PTR_NODATA, dem, None, GT, net, outlet, stream_threshold_km2=0.005,
        split_banks=True,
    )  # fmt: skip
    return net, plain, banks, root


def test_one_subcatchment_per_reach_and_channel_head(setup):
    net, plain, _banks, root = setup
    names = [c.outlet_id for c in plain.catchments]
    assert len(names) == len(set(names))
    streams = {c.outlet_id for c in plain.catchments
               if c.attributes["sub_type"] == SUB_STREAM}  # fmt: skip
    heads = {c.outlet_id for c in plain.catchments
             if c.attributes["sub_type"] == SUB_HEAD}  # fmt: skip
    assert streams == {x["name"] for x in net.reaches}
    # a channel head one step above a node has no sub-catchment of its own
    merged = [w for w in plain.warnings if "single step below" in w]
    all_heads = {x["head_name"] for x in net.reaches if x.get("head_name")}
    assert heads <= all_heads and len(all_heads - heads) == len(merged)
    n_heads = sum(1 for nd in net.nodes if nd["node_type"] == "channel head")
    assert len(all_heads) == n_heads
    # the sub-catchments cover the outlet's whole catchment, without overlap
    total = sum(c.attributes["area_km2"] for c in plain.catchments)
    assert total == pytest.approx(root["us_km2"], rel=1e-6)
    assert names == sorted(names, key=natural_key)
    assert names[0] == "A1"


def test_links_between_subcatchments(setup):
    net, plain, _banks, _root = setup
    reach = {x["name"]: x for x in net.reaches}
    node_names = {nd["name"] for nd in net.nodes}
    for c in plain.catchments:
        a = c.attributes
        assert a["node"] in node_names
        if a["sub_type"] == SUB_STREAM:
            assert a["outlet_id"] == a["stream"]
            x = reach[a["stream"]]
            assert a["ds_stream"] == x["ds_name"]
            ds = reach.get(x["ds_name"])
            on_stem = ds is None or ds["stem"] == x["stem"]
            # a stream on the stem drains to its own node; a branch to the
            # confluence, named after the main-stem stream above it
            assert a["node"] == (
                x["name"] if on_stem else f"{ds['stem']}{ds['seg'] + 1}"
            )
        else:  # channel head: drains into the top reach of its stem
            top = next(x for x in net.reaches if x.get("head_name") == a["node"])
            assert a["ds_stream"] == top["name"]
        assert a["system"] == "A"
        assert a["lfp_km"] > 0 and c.flow_path


def test_head_catchments_are_about_the_threshold_area(setup):
    _net, plain, _banks, _root = setup
    cell_km2 = CELL * CELL / 1e6
    for c in plain.catchments:
        if c.attributes["sub_type"] == SUB_HEAD:
            # the channel head is the first cell with at least the threshold
            # area; it gathers up to eight inflows below the threshold
            assert 0.005 - 1e-9 <= c.attributes["area_km2"] <= 8 * 0.005 + cell_km2


def test_banks_split_each_stream_subcatchment(setup):
    _net, plain, banks, _root = setup
    whole = {c.outlet_id: c.attributes["area_km2"] for c in plain.catchments}
    parts: dict[str, float] = {}
    cell_km2 = CELL * CELL / 1e6
    n_split = 0
    for c in banks.catchments:
        a = c.attributes
        if a["sub_type"] == SUB_STREAM and a["bank"]:
            assert a["bank"] in ("a", "b")
            assert c.outlet_id == a["stream"] + a["bank"]
            assert a["area_km2"] >= 10 * cell_km2 - 1e-12  # the area tolerance
            assert c.detail["split"]["line"] and c.detail["split"]["whole"]
            n_split += a["bank"] == "b"
            key = a["stream"]
        else:
            assert a["bank"] is None and c.outlet_id in whole
            key = c.outlet_id
        parts[key] = parts.get(key, 0.0) + a["area_km2"]
    assert parts == pytest.approx(whole, rel=1e-6)
    assert n_split > 0
    # a bank's flow path runs down the stream to the same node
    for c in banks.catchments:
        if c.attributes["bank"] == "b":
            full = next(
                p for p in plain.catchments if p.outlet_id == c.attributes["stream"]
            )
            assert c.outlet_xy == pytest.approx(full.outlet_xy)


def test_nodes_sit_at_the_start_of_their_reach(setup):
    net, _plain, _banks, _root = setup
    by_id = {x["link_id"]: x for x in net.reaches}
    for nd in net.nodes:
        x = by_id[nd["link_id"]]
        end = x["coords"][-1] if nd["node_type"] == "outlet" else x["coords"][0]
        assert (nd["x"], nd["y"]) == pytest.approx(end)


def test_incremental_report_is_compact(setup, tmp_path):
    docx = pytest.importorskip("docx")
    from mayim_tools.hydrology.catchment_delineation.report import write_report_docx

    _net, _plain, banks, _root = setup
    ctx = {"mode": "incremental", "split_banks": True, "min_final_reach_pct": 10.0}
    path = tmp_path / "r.docx"
    write_report_docx(str(path), banks.catchments, ctx, include_figures=False)
    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "split along the stream centreline" in text
    assert "5.1 Sub-catchment parameters" in text
    assert "5.2 Sub-catchment" not in text  # no per-sub-catchment sections
    cells = {c.text for t in doc.tables for r in t.rows for c in r.cells}
    assert {c.outlet_id for c in banks.catchments} <= cells
    assert any(x.startswith("node A2, then stream A1") for x in cells)
    ctx["detail_sections"] = True
    write_report_docx(str(path), banks.catchments[:3], ctx, include_figures=False)
    text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    assert "5.1 Sub-catchment A1a" in text and "drains to node A1 (the outlet)" in text
