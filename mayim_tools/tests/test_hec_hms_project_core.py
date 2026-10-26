"""Tests for Prepare HEC-HMS project (data/hec_hms_project) - no QGIS needed.

The HEC-HMS file layouts themselves were checked by opening, computing and
re-saving generated projects with HEC-HMS 4.13 and 4.14; these tests pin the logic
and the text the writers produce.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from mayim_tools.data.hec_hms_project import core
from mayim_tools.data.hec_hms_project import hms_files as hf
from mayim_tools.data.hec_hms_project import project as pj
from mayim_tools.data.hec_hms_project import results as rs
from mayim_tools.data.hec_hms_project import storms as stm

DDF_CSV = """Duration,2yr Depth (mm),10yr Depth (mm),50yr Depth (mm),100yr Depth (mm)
5 min,9.1,13.8,18.9,21.1
15 min,15.6,23.6,32.3,36.1
30 min,20.4,30.9,42.3,47.3
1 h,25.3,38.3,52.4,58.6
2 h,30.0,45.4,62.2,69.5
6 h,38.2,57.8,79.1,88.4
12 h,44.4,67.2,92.0,102.8
24 h,51.6,78.1,106.9,119.5
"""


@pytest.fixture
def ddf_path(tmp_path):
    p = tmp_path / "ddf.csv"
    p.write_text(DDF_CSV)
    return str(p)


def network_inputs():
    """Catchment Delineation incremental naming: stem A1..A4, channel head A5,
    branch A3A1 joining at node A4 (top of A3) with channel head A3A2; A3 split
    in banks a/b."""
    subs = [
        core.SubIn("A1", 1.2, 1000, 1000, node="A1", tc_min=25, cn=72),
        core.SubIn("A2", 0.8, 1100, 1600, node="A2", tc_min=20, cn=75),
        core.SubIn("A3a", 0.5, 900, 2200, node="A3", tc_min=18, cn=70, bank="a"),
        core.SubIn("A3b", 0.6, 1300, 2200, node="A3", tc_min=19, cn=78, bank="b"),
        core.SubIn("A4", 0.7, 1100, 3000, node="A4", tc_min=22, cn=74),
        core.SubIn("A5", 0.4, 1100, 3600, node="A5", tc_min=15, cn=68),
        core.SubIn("A3A1", 0.9, 1800, 2700, node="A4", tc_min=24, cn=80),
        core.SubIn("A3A2", 0.3, 2200, 3100, node="A3A2", tc_min=3, cn=66),
    ]
    reaches = [
        core.ReachIn("A1", None, [(1050, 1400), (1000, 800)], 650, 0.01, 0.2, 0.2),
        core.ReachIn("A2", "A1", [(1100, 2000), (1050, 1400)], 600, 0.012, 0.15, 0.2),
        core.ReachIn("A3", "A2", [(1100, 2600), (1100, 2000)], 600, 0.015, 0.01, 0.2),
        core.ReachIn("A4", "A3", [(1100, 3300), (1100, 2600)], 700, 0.02, 0.18, 0.2),
        core.ReachIn("A3A1", "A3", [(2000, 3000), (1100, 2600)], 950, 0.03, 0.25, 0.2),
    ]
    return subs, reaches


# ---------------------------------------------------------------- naming


def test_top_node_and_node_below():
    assert core.top_node("A1") == "A2"
    assert core.top_node("A3A1") == "A3A2"
    assert core.top_node("B12") == "B13"
    assert core.top_node("Main") is None
    assert core.node_below(core.ReachIn("A1")) == "A1"
    assert core.node_below(core.ReachIn("A2", "A1")) == "A2"
    # a branch ends at the confluence node at the top of the stream it joins
    assert core.node_below(core.ReachIn("A3A1", "A3")) == "A4"


def test_hms_name_and_file_stem():
    assert core.hms_name(" A1/b: x ") == "A1-b- x"
    assert core.file_stem("1in50 24h") == "1in50_24h"
    with pytest.raises(core.HmsProjectError):
        core.hms_name(" / ")


# ---------------------------------------------------------------- network


def test_build_network_topology():
    subs, reaches = network_inputs()
    net = core.build_network(subs, reaches, [], core.Settings())
    by = net.by_name()
    assert {e.name for e in net.of("Sink")} == {"Outlet-A1"}
    assert {e.name for e in net.of("Junction")} == {
        "J-A2",
        "J-A3",
        "J-A4",
        "J-A5",
        "J-A3A2",
    }
    assert by["A1"].downstream == "Outlet-A1"
    assert by["A3a"].downstream == by["A3b"].downstream == "J-A3"
    assert by["A3A1"].downstream == "J-A4"
    assert by["J-A4"].downstream == "R-A3"
    assert by["R-A3A1"].downstream == "J-A4"
    assert by["R-A1"].downstream == "Outlet-A1"
    assert by["J-A2"].downstream == "R-A1"
    assert by["J-A5"].downstream == "R-A4"
    # reaches are drawn from their top node to their bottom node
    assert by["R-A3"].from_xy == (1100, 2600) and (by["R-A3"].x, by["R-A3"].y) == (
        1100,
        2000,
    )
    # upstream elements come first, the sink last
    names = [e.name for e in net.elements]
    assert names[-1] == "Outlet-A1"
    assert names.index("A3A2") < names.index("J-A4") < names.index("R-A3")


def test_node_positions_from_node_layer():
    subs, reaches = network_inputs()
    nodes = [core.NodeIn("A4", 1111.0, 2601.0)]
    net = core.build_network(subs, reaches, nodes, core.Settings())
    j = net.by_name()["J-A4"]
    assert (j.x, j.y) == (1111.0, 2601.0)


def test_names_must_be_unique_across_types():
    subs, reaches = network_inputs()
    s = core.Settings(reach_prefix="", junction_prefix="", sink_prefix="")
    with pytest.raises(core.HmsProjectError, match="unique"):
        core.build_network(subs, reaches, [], s)


def test_no_node_field_gives_own_outlets():
    subs = [
        core.SubIn("C1", 1.0, 0, 0, tc_min=20, cn=70),
        core.SubIn("C2", 2.0, 10, 0, tc_min=30, cn=75),
    ]
    net = core.build_network(subs, [], [], core.Settings())
    assert {e.name for e in net.of("Sink")} == {"Outlet-C1", "Outlet-C2"}
    assert net.by_name()["C2"].downstream == "Outlet-C2"


def test_bad_reach_name_and_duplicate_start():
    subs, _ = network_inputs()
    with pytest.raises(core.HmsProjectError, match="naming"):
        core.build_network(subs[:1], [core.ReachIn("Main")], [], core.Settings())
    with pytest.raises(core.HmsProjectError, match="both start"):
        core.build_network(
            subs[:1],
            [core.ReachIn("A1"), core.ReachIn("A01")],
            [],
            core.Settings(),
        )


# ---------------------------------------------------------------- loss and time


def test_scs_runoff_and_runoff_equivalent_cn():
    # CN 80, lambda 0.1: S = 63.5, Ia = 6.35; P = 25.3 -> Q = 4.355 mm
    q = float(core.scs_runoff(25.3, 80, 0.1))
    assert q == pytest.approx(18.95**2 / (18.95 + 63.5), rel=1e-6)
    assert core.runoff_equivalent_cn(np.full(10, 75.0), 50, 0.2) == pytest.approx(
        75, abs=1e-6
    )
    cells = np.r_[np.full(50, 60.0), np.full(50, 90.0)]
    req = core.runoff_equivalent_cn(cells, 50, 0.2)
    assert req > cells.mean()  # averaging CNs underestimates runoff
    st = core.zone_cn(cells, core.CN_RUNOFF, 50, 0.2)
    assert st["cn"] == pytest.approx(req) and st["cn_mean"] == pytest.approx(75)


def test_muskingum_steps_and_choose_dt():
    # K = 60 min, X = 0.2, dt = 5: 2.4 <= n <= 19.2, nearest K/dt = 12
    assert core.muskingum_steps(60, 0.2, 5) == 12
    # K/(n dt) within [1/(2(1-X)), 1/(2X)]
    n = core.muskingum_steps(20, 0.3, 5)
    assert 1 / (2 * 0.7) <= 20 / (n * 5) <= 1 / (2 * 0.3)
    assert core.muskingum_steps(1.0, 0.2, 5) is None  # too short
    assert core.choose_dt(30, []) == 6.0  # 0.29 x 30 = 8.7
    assert core.choose_dt(200, []) == 15.0  # capped
    assert core.choose_dt(200, [5]) == 5.0  # not coarser than a hyetograph
    assert core.choose_dt(2, []) == 1.0


def test_assemble_parameters(ddf_path):
    subs, reaches = network_inputs()
    curves = stm.DdfCurves.from_csv(ddf_path)
    storms = [stm.frequency_storm(curves, 50, 1440, 50)]
    r = core.assemble(subs, reaches, [], storms, core.Settings())
    by = r.network.by_name()
    a1 = by["A1"].params
    assert a1["cn"] == 72 and a1["ia_mm"] == pytest.approx(0.1 * (25400 / 72 - 254))
    assert a1["lag_min"] == pytest.approx(15.0)
    assert "tc_floor" in by["A3A2"].flags and by["A3A2"].params["tc_min"] == 5.0
    # shortest lag 0.6 x 5 = 3 min -> 0.29 x 3 = 0.87 -> 1 min
    assert r.dt_min == 1.0
    assert by["R-A1"].params["method"] == core.ROUTE_MUSKINGUM
    assert by["R-A1"].params["steps"] == core.muskingum_steps(12, 0.2, 1.0)
    # 0.6 min K is too short at a 1 min step -> Lag
    assert by["R-A3"].params["method"] == core.ROUTE_LAG
    assert "lag_fallback" in by["R-A3"].flags
    assert r.tail_h >= 1 and r.controls[0].end > r.controls[0].start
    assert r.runs[0].name == "Freq 1in50 24h P50" and r.runs[0].gage_name is None


def test_assemble_muskingum_cunge_and_clark(ddf_path):
    subs, reaches = network_inputs()
    curves = stm.DdfCurves.from_csv(ddf_path)
    s = core.Settings(routing=core.ROUTE_MC, transform=core.TRANSFORM_CLARK, dt_min=5)
    r = core.assemble(subs, reaches, [], [stm.frequency_storm(curves, 2, 60, 50)], s)
    a1 = r.network.by_name()["A1"].params
    assert a1["clark_tc_h"] == pytest.approx(25 / 60)
    assert a1["clark_r_h"] == pytest.approx(25 / 60)  # R/(Tc+R) = 0.5
    p = r.network.by_name()["R-A3A1"].params
    assert p["method"] == core.ROUTE_MC and p["slope"] == 0.03 and p["length_m"] == 950
    text = hf.basin_text(r, hf.ProjectNames(r, "P"))
    assert "Route: Muskingum Cunge" in text and "Index Celerity: 1.5" in text
    assert "Transform: Clark" in text and "Clark Method: Specified" in text


# ---------------------------------------------------------------- storms


def test_ddf_depth_interpolation(ddf_path):
    c = stm.DdfCurves.from_csv(ddf_path)
    assert c.depth(60, 50)[0] == pytest.approx(52.4)
    # log-log between 2 h (62.2) and 6 h (79.1) at 3 h
    b = math.log(79.1 / 62.2) / math.log(3)
    assert c.depth(180, 50)[0] == pytest.approx(62.2 * 1.5**b)
    d, flags = c.depth(1, 10)
    assert "extrap_short" in flags and d < 13.8
    _, flags = c.depth(60, 200)
    assert "extrap_T" in flags


def test_frequency_storm(ddf_path):
    c = stm.DdfCurves.from_csv(ddf_path)
    st = stm.frequency_storm(c, 50, 360, 33)
    assert sorted(st.depths) == [5, 10, 15, 30, 60, 120, 180, 360]
    assert st.depths[360] == pytest.approx(79.1) and st.total_mm == pytest.approx(79.1)
    assert all(np.diff([st.depths[k] for k in sorted(st.depths)]) >= 0)
    with pytest.raises(stm.StormError, match="HEC-HMS durations"):
        stm.frequency_storm(c, 50, 90, 50)
    t, cum = stm.nested_cumulative(st.depths, 360, 50)
    assert t[0] == 0 and cum[-1] == pytest.approx(79.1)
    assert all(np.diff(cum) >= -1e-9)


def test_parse_lists():
    assert stm.parse_number_list("2, 5;10 50") == [2, 5, 10, 50]
    assert stm.parse_duration_list("60, 3h, 1 day") == [60, 180, 1440]
    assert stm.parse_duration_list("") == []
    with pytest.raises(stm.StormError):
        stm.parse_number_list("2, x")


def test_alternating_block_csv_from_the_real_tool(tmp_path):
    from mayim_tools.rainfall.ddf_to_hyetographs import export as abx
    from mayim_tools.rainfall.ddf_to_hyetographs.core import parse_idf_table

    df = pd.read_csv(pd.io.common.StringIO(DDF_CSV))
    tables = parse_idf_table(df)
    path = tmp_path / "ab.csv"
    abx.write_incremental_depth(
        [(50.0, 60.0), (50.0, 360.0), (2.0, 60.0)], 5, tables, path
    )
    rows = stm.read_alternating_block_csv(str(path))
    assert len(rows) == 3 and rows[0]["interval_min"] == 5
    storms, warns = stm.alternating_block_storms(rows, [50], [], 50, 1.0)
    assert len(storms) == 2 and not warns
    s60 = next(s for s in storms if s.duration_min == 60)
    assert len(s60.increments) == 12
    assert s60.total_mm == pytest.approx(52.4, rel=1e-3)
    assert s60.name == "AB 1in50 1h P50"
    # 67 % peak maps to the tool's 66 % column
    storms67, _ = stm.alternating_block_storms(rows, [2], [60], 67, 0.9)
    assert storms67[0].total_mm == pytest.approx(0.9 * 25.3, rel=1e-3)


def test_huff_storms(tmp_path, ddf_path):
    path = tmp_path / "huff.csv"
    x = np.linspace(0, 1, 11)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Quartile",
                "X",
                "NEvents",
                "InsufficientSample",
                "Mean",
                "Min",
                "Max",
                "P10",
                "P50",
                "P90",
            ]
        )
        for q in (1, 2):
            for xi in x:
                y = xi ** (0.5 if q == 1 else 1.0)
                w.writerow([q, xi, 20, False, y, y, y, y, y, y])
    huff = stm.read_huff_csv(str(path))
    assert (1, "P50") in huff and (2, "Mean") in huff
    c = stm.DdfCurves.from_csv(ddf_path)
    storms, warns = stm.huff_storms(c, huff, [50], [360, 720, 1440], 5, None, "P50")
    assert [s.name for s in storms] == ["Huff Q1 1in50 6h", "Huff Q2 1in50 12h"]
    assert "Q3" in warns[0]
    s6 = storms[0]
    assert len(s6.increments) == 72 and s6.total_mm == pytest.approx(79.1)
    assert s6.increments[0] > s6.increments[-1]  # first-quartile storm
    with pytest.raises(stm.StormError):
        stm.huff_storms(c, huff, [50], [62], 5, 1, "P50")


def test_ensemble_storms(tmp_path, ddf_path):
    from mayim_tools.rainfall.design_storm_ensembles.report import write_ensemble_csv

    rows = []
    for band in ("frequent", "intermediate", "rare"):
        for shape in ("Early", "Middle", "Late"):
            inc = (
                [10.0] * 12
                if shape != "Early"
                else [30, 20, 10, 10, 5, 5, 5, 5, 5, 2, 2, 1]
            )
            rows.append(
                {
                    "Duration": 60,
                    "TimeStep": 5,
                    "AEP": band,
                    "Shape": shape,
                    "Increments": inc if band != "rare" or shape != "Late" else [],
                }
            )
    path = tmp_path / "ens.csv"
    write_ensemble_csv(rows, str(path))
    ens = stm.read_ensemble_csv(str(path))
    c = stm.DdfCurves.from_csv(ddf_path)
    storms, warns = stm.ensemble_storms(c, ens, [2, 100], [60], ["Early", "Late"])
    assert [s.name for s in storms] == [
        "Ens 1in2 1h Early",
        "Ens 1in2 1h Late",
        "Ens 1in100 1h Early",
    ]
    assert len(warns) == 1 and "rare Late" in warns[0]
    assert storms[0].total_mm == pytest.approx(25.3)
    assert storms[0].increments[0] == pytest.approx(25.3 * 30 / 100)
    assert (
        stm.ensemble_band(5) == "frequent" and stm.ensemble_band(10) == "intermediate"
    )
    assert stm.ensemble_band(50) == "rare"


# ---------------------------------------------------------------- files


def _result(ddf_path, with_hyeto=True):
    subs, reaches = network_inputs()
    c = stm.DdfCurves.from_csv(ddf_path)
    storms = [stm.frequency_storm(c, t, 360, 50) for t in (2, 50)]
    if with_hyeto:
        x = np.linspace(0, 1, 5)
        storms += stm.huff_storms(
            c, {(1, "P50"): (x, x**0.5)}, [50], [360], 5, 1, "P50"
        )[0]
    return core.assemble(
        subs, reaches, [], storms, core.Settings(project_name="T1", dt_min=5)
    )


def test_text_files(ddf_path):
    r = _result(ddf_path)
    n = hf.ProjectNames(r, "T1")
    prj = hf.project_text(r, n)
    assert prj.startswith("Project: T1\r\n") and "DSS File Name: T1.dss" in prj
    assert (
        "Precipitation: Freq 1in50 6h P50\r\n     Filename: Freq_1in50_6h_P50.met"
        in prj
    )
    assert "Control: 6h storm\r\n     FileName: 6h_storm.control" in prj
    basin = hf.basin_text(r, n, "PROJCS[x]", "Terrain")
    assert "Subbasin: A3A1\r\n" in basin and "     Downstream: J-A4\r\n" in basin
    assert "     LossRate: SCS\r\n" in basin and "     Curve Number: 80\r\n" in basin
    assert "     Route: Muskingum\r\n" in basin and "     Terrain: Terrain\r\n" in basin
    assert "Sink: Outlet-A1\r\n" in basin and "Coordinate System: PROJCS[x]" in basin
    freq = hf.met_text(r, r.runs[1])
    assert "Precipitation Method: Frequency Based Hypothetical" in freq
    assert "     Depth 360: 79.100000\r\n" in freq and "     Depth 720: \r\n" in freq
    assert "     Time Interval: 5\r\n" in freq
    assert "Interpolation Method: Log\r\n     Depth-Area Reduction Method" in freq
    assert "Percent of Duration Before Peak Rainfall: 50" in freq
    hy = hf.met_text(r, r.runs[2])
    assert "Precipitation Method: Specified Average" in hy
    assert "Subbasin: A1\r\n" in hy and "     Gage: Huff Q1 1in50 6h\r\n" in hy
    gage = hf.gage_text(r, n)
    assert "Data Source Type: External DSS" in gage
    assert "Pathname: //HUFF Q1 1IN50 6H/PRECIP-INC/01JAN2000/5MIN/MAYIM/" in gage
    run = hf.run_text(r)
    assert "Run: Freq 1in2 6h P50\r\n" in run
    assert "     Precip: Freq 1in2 6h P50\r\n" in run
    assert "     Control: 6h storm\r\n" in run
    assert "Time Series Results Manager Start:\r\n" in run
    ctl = hf.control_text(r.controls[0])
    assert "Start Date: 1 January 2000" in ctl and "Time Interval: 5" in ctl
    assert "     Version: 4.14\r\n" in prj and "     Version: 4.14\r\n" in basin
    assert "     Version: 4.14\r\n" in freq and "Version: 4.14" in gage
    assert "Version: 4.13" in hf.control_text(r.controls[0], "4.13")
    ter = hf.terrain_text(n, "Terrain", "DEM")
    assert "Terrain Directory: terrain\\Terrain" in ter
    assert hf.terrain_relpath("Terrain") == "terrain/Terrain/00/elevation.tif"
    assert hf.intensity_duration(2) == 5 and hf.intensity_duration(10) == 10


def test_gage_series_covers_the_control_window(ddf_path):
    r = _result(ddf_path)
    run = r.runs[2]
    first, dt, vals = hf.gage_series(run)
    assert first == datetime(2000, 1, 1, 0, 5) and dt == 5
    span = (run.control.end - run.control.start).total_seconds() / 60
    assert len(vals) == span / 5
    assert sum(vals) == pytest.approx(79.1)


def test_write_project(tmp_path, ddf_path):
    r = _result(ddf_path)
    got = []

    def fake_dss(path, series):
        got.extend(series)
        open(path, "wb").close()
        return len(series)

    wkb = bytes.fromhex(
        "0103000000010000000400000000000000000000000000000000000000000000000000f03f000000000000f03f000000000000f03f00000000000000000000000000000000"
    )
    r.network.by_name()["A1"].wkb = wkb
    files = pj.write_project(
        r,
        tmp_path,
        "T1 Model",
        srid=32735,
        srtext="PROJCS[x]",
        dss_writer=fake_dss,
        map_writer=lambda *a: [],
    )
    folder = tmp_path / "T1_Model"
    assert (folder / "T1_Model.hms").is_file() and (folder / "Existing.basin").is_file()
    assert (folder / "Freq_1in2_6h_P50.met").is_file() and (
        folder / "6h_storm.control"
    ).is_file()
    assert (folder / "T1_Model.gage").is_file() and (
        folder / "run_all_simulations.cmd"
    ).is_file()
    assert len(got) == 1 and got[0][0].startswith("//HUFF Q1 1IN50 6H/")
    assert files["Basin geometry"].endswith("Existing.sqlite")
    con = sqlite3.connect(files["Basin geometry"])
    tables = {
        t for (t,) in con.execute("select name from sqlite_master where type='table'")
    }
    assert {"subbasin2d", "reach2d", "junction", "sink", "geometry_columns"} <= tables
    assert con.execute("select name from subbasin2d").fetchall() == [("A1",)]
    assert con.execute(
        "select srid from geometry_columns where f_table_name='subbasin2d'"
    ).fetchone() == (32735,)
    con.close()
    script = (folder / "run_all_simulations.py").read_text()
    assert "OpenProject(PROJECT, FOLDER)" in script and "'Huff Q1 1in50 6h'" in script


def test_real_dss_round_trip(tmp_path, ddf_path):
    pytest.importorskip("hecdss")
    from mayim_tools.data.hec_hms_project import dss

    r = _result(ddf_path)
    pj.write_project(r, tmp_path, "T2")
    run = r.runs[2]
    path = tmp_path / "T2" / "T2_design_storms.dss"
    times, vals = dss.read_gage(
        path, hf.gage_pathname(run.gage_name, run.control.start, 5)
    )
    assert times[0] == datetime(2000, 1, 1, 0, 5)
    assert sum(vals) == pytest.approx(79.1, rel=1e-6)


def test_zonal_cn_from_raster(tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    from osgeo import osr

    gdal.UseExceptions()
    path = str(tmp_path / "cn.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 10, 10, 2, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 10, 0, 100, 0, -10))
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(32735)
    ds.SetProjection(sr.ExportToWkt())
    a = np.full((10, 10), 60.0, dtype="float32")
    a[:, 5:] = 90.0
    ds.GetRasterBand(1).WriteArray(np.zeros((10, 10), dtype="float32") + 1)
    ds.GetRasterBand(1).SetDescription("other")
    ds.GetRasterBand(2).WriteArray(a)
    ds.GetRasterBand(2).SetDescription("cn_design")
    ds = None
    zones = [
        "POLYGON((0 0,50 0,50 100,0 100,0 0))",
        "POLYGON((0 0,100 0,100 100,0 100,0 0))",
        "POLYGON((71 71,72 71,72 72,71 72,71 71))",  # smaller than a cell
    ]
    stats, label = core.zonal_cn_from_raster(path, zones, core.CN_MEAN, 50, 0.2)
    assert "cn_design" in label
    assert stats[0]["cn"] == pytest.approx(60) and stats[0]["n_cells"] == 50
    assert stats[1]["cn_mean"] == pytest.approx(75)
    assert stats[1]["cn_runoff"] > 75
    assert stats[2]["centroid_only"] and stats[2]["cn"] == pytest.approx(90)


def test_read_results(tmp_path):
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "RUN_1in50_6h.results").write_text(
        """<?xml version="1.0" encoding="UTF-8"?><RunResults><RunName>1in50 6h</RunName>
<BasinElement name="Outlet-A1" type="Sink"><Statistics>
<StatisticMeasure type="Outflow Maximum" value="48.67" units="M3/S" />
<StatisticMeasure type="Outflow Maximum Time" value="01Jan2000, 03:38" />
<StatisticMeasure type="Outflow Volume" value="167576.9" units="M3" />
</Statistics></BasinElement></RunResults>"""
    )
    rows, missing = rs.read_results(str(tmp_path), ["1in50 6h", "1in2 6h"])
    assert missing == ["1in2 6h"]
    assert rows[0].element == "Outlet-A1" and rows[0].peak_m3s == pytest.approx(48.67)
    assert rows[0].peak_time == "01Jan2000, 03:38"


def test_export_csv_and_report(tmp_path, ddf_path):
    from mayim_tools.data.hec_hms_project.export import peak_table, write_csv

    r = _result(ddf_path)
    res = [
        rs.ElementResult(
            r.runs[0].name, "Outlet-A1", "Sink", 12.5, "t", 1000.0, 1.0, None
        )
    ]
    p = write_csv(str(tmp_path / "setup.csv"), r, [("DDF", "x.csv")], res)
    text = open(p).read()
    assert "## Sub-basins" in text and "## Computed peak flows (HEC-HMS)" in text
    assert "1000" in text and "1,000" not in text
    hdr, rows = peak_table(r, res)
    assert hdr[0] == "Element" and ["Outlet-A1", 12.5, None, None] in rows
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    from mayim_tools.data.hec_hms_project.report import write_report

    geom = {("Subbasin", "A1"): [[(900, 900), (1100, 900), (1100, 1100), (900, 900)]]}
    notes = write_report(
        str(tmp_path / "r.docx"), r, {"inputs": [], "outputs": []}, geom, res
    )
    assert notes == [] and (tmp_path / "r.docx").stat().st_size > 10000


def test_hms_version_413(ddf_path):
    subs, reaches = network_inputs()
    c = stm.DdfCurves.from_csv(ddf_path)
    s = core.Settings(project_name="T3", hms_version="4.13", dt_min=5)
    r = core.assemble(subs, reaches, [], [stm.frequency_storm(c, 2, 60, 50)], s)
    n = hf.ProjectNames(r, "T3")
    for text in (hf.project_text(r, n), hf.basin_text(r, n), hf.met_text(r, r.runs[0])):
        assert "     Version: 4.13\r\n" in text and "4.14" not in text
    assert "Interpolation Method" not in hf.met_text(r, r.runs[0])


# ---------------------------------------------------------------- 0.3: names, peaks,
# ensembles, map layers, curve number grid


def test_hms_safe_project_names():
    # HEC-HMS turns every non-alphanumeric character of the project name into
    # '_' and looks for <that>.run etc.; a hyphen made the project open empty
    assert core.project_stem("HEC-HMS_Model") == "HEC_HMS_Model"
    assert core.project_stem("MC2026-01 HEC-HMS Model") == "MC2026_01_HEC_HMS_Model"
    assert core.folder_stem("MC2026-01 HEC-HMS Model") == "MC2026-01_HEC-HMS_Model"
    assert core.file_stem("Freq 1in50 6h P50") == "Freq_1in50_6h_P50"
    assert "-" not in core.file_stem("AB-1in2")
    assert hf.hms_component_file("Frequency 1in50", ".ens") == "Frequency_1in50.ens"


def test_multiple_peak_positions_and_ensembles(ddf_path):
    subs, reaches = network_inputs()
    c = stm.DdfCurves.from_csv(ddf_path)
    storms = [
        stm.frequency_storm(c, t, d, p)
        for t in (2, 50)
        for d in (60, 360)
        for p in (25, 50, 75)
    ]
    r = core.assemble(subs, reaches, [], storms, core.Settings(dt_min=5))
    assert [e.name for e in r.ensembles] == ["Frequency 1in2", "Frequency 1in50"]
    assert len(r.ensembles[1].runs) == 6
    assert {run.name for run in r.ensembles[1].runs} >= {"Freq 1in50 6h P75"}
    # ensemble members need the same time window: one shared control
    assert len(r.controls) == 1 and r.controls[0].name == "Design storms"
    assert r.controls[0].duration_min == 360
    text = hf.ensemble_text(r.ensembles[1], r)
    assert text.startswith("Ensemble Analysis: Frequency 1in50\r\n")
    assert "     Ensemble Member Type: Simulation Run\r\n" in text
    assert "          Ensemble Member: Freq 1in50 1h P25\r\n" in text
    assert (
        "     Analysis Point: Outlet-A1\r\n     Time Series Type: Outflow\r\n" in text
    )
    assert "     End Analysis Point: \r\n" in text  # HEC-HMS needs the space
    assert text.endswith("End Ensemble Analysis: Frequency 1in50\r\n")
    # without ensembles: one control per duration
    r2 = core.assemble(
        subs, reaches, [], storms, core.Settings(dt_min=5, ensembles=False)
    )
    assert not r2.ensembles and [x.name for x in r2.controls] == [
        "1h storm",
        "6h storm",
    ]


def test_alternating_block_several_peaks(tmp_path):
    from mayim_tools.rainfall.ddf_to_hyetographs import export as abx
    from mayim_tools.rainfall.ddf_to_hyetographs.core import parse_idf_table

    tables = parse_idf_table(pd.read_csv(pd.io.common.StringIO(DDF_CSV)))
    path = tmp_path / "ab.csv"
    abx.write_incremental_depth([(50.0, 60.0)], 5, tables, path)
    rows = stm.read_alternating_block_csv(str(path))
    storms, warns = stm.alternating_block_storms(rows, [50], [60], [25, 50, 67], 1.0)
    assert [s.name for s in storms] == [
        "AB 1in50 1h P25",
        "AB 1in50 1h P50",
        "AB 1in50 1h P66",
    ]
    assert not warns and all(
        s.total_mm == pytest.approx(52.4, rel=1e-3) for s in storms
    )


def test_project_folder_names_and_ensemble_files(tmp_path, ddf_path):
    subs, reaches = network_inputs()
    c = stm.DdfCurves.from_csv(ddf_path)
    storms = [stm.frequency_storm(c, 50, d, p) for d in (60, 360) for p in (33, 50)]
    r = core.assemble(subs, reaches, [], storms, core.Settings(dt_min=5))
    files = pj.write_project(
        r, tmp_path, "MC-1_HEC-HMS Model", map_writer=lambda *a: []
    )
    folder = tmp_path / "MC-1_HEC-HMS_Model"
    assert files["Project"] == str(folder / "MC_1_HEC_HMS_Model.hms")
    prj = (folder / "MC_1_HEC_HMS_Model.hms").read_text()
    assert prj.startswith("Project: MC_1_HEC_HMS_Model\n")
    assert (folder / "MC_1_HEC_HMS_Model.run").is_file()
    assert "DSS File Name: MC_1_HEC_HMS_Model.dss" in prj
    assert (folder / "ensemble" / "Frequency_1in50.ens").is_file()
    assert "MC_1_HEC_HMS_Model" in (folder / "run_all_simulations.py").read_text()


def test_grid_and_map_layer_text(ddf_path):
    r = _result(ddf_path)
    n = hf.ProjectNames(r, "T1")
    grid = hf.grid_text(n, [("Curve Numbers", "Curve Number", "maps/cn.tif", "CN")])
    assert "Grid: Curve Numbers\r\n     Grid: Curve Numbers\r\n" in grid
    assert "     Grid Type: Curve Number\r\n" in grid
    assert "     Data Source Type: GeoTIFF\r\n     Filename: maps\\cn.tif\r\n" in grid
    assert "     Last Modified Date: " in grid  # HEC-HMS fails to open without it
    basin = hf.basin_text(r, n, map_layers=[("Subbasins", "maps/T1_Subbasins.shp")])
    assert (
        "     Map Layer:\r\n          Filename: maps\\T1_Subbasins.shp\r\n"
        "          Layer:0\r\n          Display name: Subbasins\r\n"
    ) in basin


def test_map_layers_and_cn_grid_files(tmp_path, ddf_path):
    gdal = pytest.importorskip("osgeo.gdal")
    from osgeo import ogr, osr

    r = _result(ddf_path)
    sq = ogr.CreateGeometryFromWkt("POLYGON((0 0,100 0,100 100,0 100,0 0))")
    ln = ogr.CreateGeometryFromWkt("LINESTRING(50 100,50 0)")
    by = r.network.by_name()
    by["A1"].wkb = bytes(sq.ExportToWkb())
    by["R-A1"].wkb = bytes(ln.ExportToWkb())
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(32735)
    wkt = sr.ExportToWkt()
    layers = pj.write_map_layers(r, tmp_path / "maps", wkt, "T1")
    assert layers == [
        ("Subbasins", "maps/T1_Subbasins.shp"),
        ("Reaches", "maps/T1_Reaches.shp"),
    ]
    ds = ogr.Open(str(tmp_path / "maps" / "T1_Subbasins.shp"))
    f = ds.GetLayer(0).GetNextFeature()
    assert f["name"] == "A1" and f["cn"] == 72 and f["ds"] == "Outlet-A1"
    ds = None
    # CN raster: band 2 of a two-band raster
    src = tmp_path / "cn.tif"
    d = gdal.GetDriverByName("GTiff").Create(str(src), 20, 20, 2, gdal.GDT_Float32)
    d.SetGeoTransform((-500, 50, 0, 600, 0, -50))
    d.SetProjection(wkt)
    d.GetRasterBand(1).Fill(60)
    d.GetRasterBand(2).Fill(77)
    d = None
    out = pj.write_cn_grid(
        str(src), 2, tmp_path / "maps" / "curve_numbers.tif", wkt, (0, 0, 100, 100)
    )
    g = gdal.Open(out)
    arr = g.GetRasterBand(1).ReadAsArray()
    assert g.RasterCount == 1 and float(arr.max()) == 77.0
