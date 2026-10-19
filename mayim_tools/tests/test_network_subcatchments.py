"""Tests of the network sub-catchments (Catchment Delineation incremental
mode): stream names, a sub-catchment per reach and per channel head, and
the optional split along the stream centreline."""

from __future__ import annotations

import numpy as np
import pytest

from mayim_tools.hydrology.catchment_delineation.subcatchments import (
    SUB_HEAD,
    SUB_STREAM,
    bank_side,
    bank_sides,
    delineate_network_subcatchments,
    natural_key,
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


def test_bank_side_left_and_right_looking_downstream():
    cols = 10
    # stream flowing south along column 5: up (row 4), s (row 5), dn (row 6)
    up, s, dn = 4 * cols + 5, 5 * cols + 5, 6 * cols + 5
    east, west = 5 * cols + 6, 5 * cols + 4
    # looking south, east is on the left
    assert bank_side(east, s, up, dn, cols, 1.0, 1.0) == 1
    assert bank_side(west, s, up, dn, cols, 1.0, 1.0) == 2
    assert list(bank_sides([east, west], [s, s], [up, up], [dn, dn], cols, 1, 1)) == [
        1,
        2,
    ]


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
    halves: dict[str, float] = {}
    for c in banks.catchments:
        a = c.attributes
        if a["sub_type"] == SUB_STREAM:
            assert a["bank"] in ("a", "b")
            assert c.outlet_id == a["stream"] + a["bank"]
            halves[a["stream"]] = halves.get(a["stream"], 0.0) + a["area_km2"]
        else:
            assert a["bank"] is None and c.outlet_id in whole
    for name, area in halves.items():
        assert area == pytest.approx(whole[name], rel=1e-6)
    assert sum(1 for c in banks.catchments if c.attributes["bank"] == "b") > 0
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
