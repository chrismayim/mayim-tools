"""Tests of stream breakpoints: splitting reaches at a maximum stream length,
measured from each reach's downstream end, only where the rest of the reach
is longer than 10 % of the maximum."""

from __future__ import annotations

import numpy as np
import pytest

from mayim_tools.hydrology.stream_network.core import (
    BREAKPOINT_NODE,
    extract_stream_network,
    split_positions,
)
from mayim_tools.tests.test_stream_network_core import (
    GT,
    PTR_NODATA,
    dendritic_dem,
    pointer_for,
)

MAX_LEN = 60.0


def lengths_from_low_end(seg, breaks):
    d = np.concatenate([np.cumsum(np.asarray(seg)[::-1])[::-1], [0.0]])
    return [float(d[k]) for k in breaks]


def test_split_positions_every_max_length():
    seg = [10.0] * 132  # a 1320 m reach
    assert lengths_from_low_end(seg, split_positions(seg, 500)) == [500.0, 1000.0]


def test_split_only_when_remainder_over_ten_percent():
    assert split_positions([10.0] * 55, 500) == []  # 550 m: remainder 50 m
    assert lengths_from_low_end([10.0] * 56, split_positions([10.0] * 56, 500)) == [
        500.0
    ]  # 560 m: remainder 60 m
    seg = [10.0] * 104  # 1040 m: break at 500 m, not at 1000 m (remainder 40 m)
    assert lengths_from_low_end(seg, split_positions(seg, 500)) == [500.0]


def test_pieces_never_exceed_the_maximum():
    seg = [14.142] * 40  # diagonal steps
    breaks = split_positions(seg, 100.0)
    bounds = [len(seg)] + breaks + [0]
    pieces = [sum(seg[lo:hi]) for hi, lo in zip(bounds[:-1], bounds[1:], strict=True)]
    assert all(p <= 100.0 + 1e-9 for p in pieces[:-1])
    assert pieces[-1] > 10.0  # the top piece is the remainder (> 10 %)


def test_no_maximum_means_no_breaks():
    assert split_positions([10.0] * 50, 0) == []
    assert split_positions([10.0] * 50, None) == []


@pytest.fixture(scope="module")
def networks():
    dem = dendritic_dem()
    ptr = pointer_for(dem)
    kw = dict(threshold_km2=0.005, drop_test_range=None)
    plain = extract_stream_network(ptr, PTR_NODATA, dem, None, GT, **kw)
    split = extract_stream_network(
        ptr, PTR_NODATA, dem, None, GT, max_reach_length_m=MAX_LEN, **kw
    )
    return plain, split


def test_network_split_lengths_and_nodes(networks):
    plain, split = networks
    n_breaks = sum(1 for n in split.nodes if n["node_type"] == BREAKPOINT_NODE)
    assert n_breaks > 0
    assert split.settings["n_breakpoints"] == n_breaks
    assert len(split.reaches) == len(plain.reaches) + n_breaks
    # total channel length is unchanged
    assert sum(r["length_m"] for r in split.reaches) == pytest.approx(
        sum(r["length_m"] for r in plain.reaches), abs=0.001 * len(split.reaches)
    )  # lengths are rounded to 1 mm per reach
    diag = 10.0 * 2**0.5
    for r in split.reaches:
        if r.get("split_from") is not None and r["piece"] < max(
            q["piece"] for q in split.reaches if q.get("split_from") == r["split_from"]
        ):
            assert r["length_m"] <= MAX_LEN + 1e-6  # lower pieces at most the max
        else:  # unsplit reaches and top pieces: at most the max plus 10 %
            assert r["length_m"] <= 1.1 * MAX_LEN + diag + 1e-6


def test_topology_after_split(networks):
    plain, split = networks
    ids = {r["link_id"] for r in split.reaches}
    assert len(ids) == len(split.reaches)
    assert all(r["ds_link"] is None or r["ds_link"] in ids for r in split.reaches)
    roots = [r for r in split.reaches if r["ds_link"] is None]
    assert len(roots) == len([r for r in plain.reaches if r["ds_link"] is None])
    # every reach still reaches an outlet
    by_id = {r["link_id"]: r for r in split.reaches}
    for r in split.reaches:
        seen = 0
        while r["ds_link"] is not None:
            r = by_id[r["ds_link"]]
            seen += 1
            assert seen < len(by_id)
    # a split reach's pieces join end to end
    for r in split.reaches:
        if r.get("split_from") is not None and r["piece"] > 1:
            below = by_id[r["ds_link"]]
            assert below["split_from"] == r["split_from"]
            assert below["piece"] == r["piece"] - 1
            assert r["coords"][-1] == below["coords"][0]


def test_breakpoint_nodes_sit_on_piece_joints(networks):
    _plain, split = networks
    starts = {r["coords"][0] for r in split.reaches}
    for n in split.nodes:
        if n["node_type"] == BREAKPOINT_NODE:
            assert (n["x"], n["y"]) in starts
            assert n["n_inflow"] == 1


def test_network_statistics_unchanged(networks):
    plain, split = networks
    assert split.networks == plain.networks
    assert split.order_stats == plain.order_stats


def test_report_mentions_breakpoints(networks, tmp_path):
    docx = pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    from mayim_tools.hydrology.stream_network.report import (
        order_styles,
        write_report_docx,
    )

    _plain, split = networks
    path = tmp_path / "net.docx"
    write_report_docx(str(path), split, {"version": "t", "project_number": "P1"})
    doc = docx.Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs)
    n = split.settings["n_breakpoints"]
    assert f"split at {n} stream breakpoint(s)" in text
    rows = {
        r.cells[0].text: r.cells[1].text
        for t in doc.tables
        for r in t.rows
        if len(r.cells) == 2
    }
    assert rows["Maximum stream length"] == f"60 m ({n} stream breakpoints)"
    # Strahler colours and widths come from the QML style
    styles = order_styles(3)
    assert styles[1][0] == "#0001fd" and styles[3][1] > styles[1][1]
