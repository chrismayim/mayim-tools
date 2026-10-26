"""Tests for Prepare SWMM project (core, inp writer, runner, CSV)."""

from __future__ import annotations

import numpy as np
import pytest

from mayim_tools.data.hec_hms_project import core as hms
from mayim_tools.data.hec_hms_project import storms as stm
from mayim_tools.data.swmm_project import core as sc
from mayim_tools.data.swmm_project import export as ex
from mayim_tools.data.swmm_project import inp
from mayim_tools.data.swmm_project import run as rn
from mayim_tools.tests.test_hec_hms_project_core import DDF_CSV, network_inputs


@pytest.fixture
def ddf_path(tmp_path):
    p = tmp_path / "ddf.csv"
    p.write_text(DDF_CSV)
    return str(p)


def _model(ddf_path, settings=None, dem=None, peaks=(33, 50), extra=None):
    subs, reaches = network_inputs()
    c = stm.DdfCurves.from_csv(ddf_path)
    storms = [
        stm.frequency_storm(c, t, d, p)
        for t in (2, 50)
        for d in (60, 360)
        for p in peaks
    ]
    x = np.linspace(0, 1, 5)
    storms += stm.huff_storms(
        c, {(1, "P50"): (x, x**0.5)}, [50], [360], 10, 1, "P50"
    )[0]
    r = hms.assemble(subs, reaches, [], storms, hms.Settings(ensembles=False))
    return sc.build_model(
        r,
        settings or sc.SwmmSettings(),
        sub_extra=extra,
        dem=dem,
        reach_slopes={x.name: x.slope for x in reaches},
        reach_points={x.name: x.xy for x in reaches},
    )


def test_names_steps_and_cumulative_storms(ddf_path):
    assert sc.swmm_id("Freq 1in50 6h P50") == "Freq_1in50_6h_P50"
    c = stm.DdfCurves.from_csv(ddf_path)
    f = stm.frequency_storm(c, 50, 360, 50)
    t, cum = sc.storm_cumulative(f, 5, 5)
    # the peak moves by less than one block so the 5 min burst fills one block
    assert t[0] == 0 and 360 <= t[-1] < 370 and cum[0] == 0
    assert cum[-1] == pytest.approx(79.1)
    assert all(np.diff(cum) >= 0)
    # the 5 min nested peak block holds the 5 min DDF depth
    assert max(np.diff(cum)) == pytest.approx(18.9, rel=1e-3)
    inc = np.diff(cum)
    assert max(np.convolve(inc, np.ones(3), "valid")) == pytest.approx(32.3, rel=0.01)
    x = np.linspace(0, 1, 5)
    h = stm.huff_storms(c, {(1, "P50"): (x, x**0.5)}, [50], [360], 10, 1, "P50")[0][0]
    assert sc.common_step([f, h], 5) == 5
    t2, cum2 = sc.storm_cumulative(h, 5)
    assert cum2[-1] == pytest.approx(79.1) and cum2[2] == pytest.approx(h.increments[0])


def test_model_without_dem(ddf_path):
    m = _model(ddf_path, extra={"A1": {"w_flow_m": 3000.0, "s_mean_pc": 4.5}})
    subs = {x.name: x for x in m.subcatchments}
    assert subs["A1"].width_m == 3000 and subs["A1"].slope_pc == 4.5
    assert subs["A1"].area_ha == pytest.approx(120)
    assert subs["A2"].width_m == pytest.approx((0.8e6) ** 0.5)  # no field: sqrt(A)
    assert subs["A3a"].outlet == subs["A3b"].outlet == "J-A3"
    assert any("sqrt(area)" in w for w in m.warnings)
    nodes = m.node_map()
    assert nodes["Outlet-A1"].kind == "outfall" and nodes["Outlet-A1"].invert == 100
    con = {c.name: c for c in m.conduits}
    assert (con["R-A1"].from_node, con["R-A1"].to_node) == ("J-A2", "Outlet-A1")
    assert (con["R-A3A1"].from_node, con["R-A3A1"].to_node) == ("J-A3A2", "J-A4")
    # inverts from the reach slopes, so every conduit falls downstream
    assert nodes["J-A2"].invert == pytest.approx(100 + 0.01 * 650)
    assert all(c.slope(nodes) > 0 for c in m.conduits)
    assert len(m.series) == 9 and m.step_min == 5
    # run-out: 2 x the HEC-HMS run-out, at least 6 h
    assert (m.end - m.start).total_seconds() / 3600 >= 6 + 6


def _dem(drain_ok=True):
    """10 m DEM over the test network falling to the south (outlet), or not."""
    ny, nx = 400, 300
    yy, xx = np.mgrid[0:ny, 0:nx]
    north = 4000.0 - yy * 10.0  # y of each row
    z = (
        100 + 0.01 * north + 0.002 * np.abs(xx * 10.0 - 1100)
        if drain_ok
        else 140 - 0.01 * north
    )
    return sc.DemSampler(z, (0.0, 10.0, 0.0, 4000.0, 0.0, -10.0))


def test_model_with_dem_inverts_and_transects(ddf_path):
    m = _model(ddf_path, sc.SwmmSettings(xs_method=sc.XS_TRANSECT), dem=_dem())
    nodes = m.node_map()
    assert nodes["Outlet-A1"].dem_invert is not None
    assert all(c.slope(nodes) >= 0.001 - 1e-9 for c in m.conduits)
    assert all(c.shape == "IRREGULAR" and c.transect is not None for c in m.conduits)
    t = m.conduits[0].transect
    assert len(t.stations) == 41 and t.stations[0] == 0 and t.stations[-1] == 100
    assert t.left_bank < t.right_bank
    # a DEM falling the wrong way: nodes lowered and flagged
    m2 = _model(ddf_path, dem=_dem(False))
    assert any("lowered" in n.flags for n in m2.nodes)
    assert any("lowered below the DEM" in w for w in m2.warnings)


def test_inp_text(ddf_path):
    m = _model(ddf_path)
    text = inp.inp_text(m)
    for sec in (
        "[OPTIONS]",
        "[RAINGAGES]",
        "[SUBCATCHMENTS]",
        "[SUBAREAS]",
        "[INFILTRATION]",
        "[JUNCTIONS]",
        "[OUTFALLS]",
        "[CONDUITS]",
        "[XSECTIONS]",
        "[TIMESERIES]",
        "[MAP]",
        "[COORDINATES]",
        "[Polygons]",
        "[SYMBOLS]",
    ):
        assert sec in text
    assert " FLOW_ROUTING          DYNWAVE" in text
    assert "INFILTRATION          CURVE_NUMBER" in text
    assert (
        "RG1        CUMULATIVE 0:05       1.0        TIMESERIES Freq_1in2_1h_P33"
        in text
    )
    assert "Freq_1in50_6h_P50 01/01/2000       0:00             0" in text
    assert "\r\n" in text
    other = inp.inp_text(m, "Huff_Q1_1in50_6h")
    assert "TIMESERIES Huff_Q1_1in50_6h" in other


def test_report_parser():
    text = """
  *******************
  Node Inflow Summary
  *******************

  -------------------------------------------------------------------------------------------------
                                  Maximum  Maximum                  Lateral       Total        Flow
                                  Lateral    Total  Time of Max      Inflow      Inflow     Balance
                                   Inflow   Inflow   Occurrence      Volume      Volume       Error
  Node                 Type           CMS      CMS  days hr:min    10^6 ltr    10^6 ltr     Percent
  -------------------------------------------------------------------------------------------------
  J-A2                 JUNCTION     1.234    5.678     0  03:12       12.3        45.6       0.010
  Outlet-A1            OUTFALL      0.000    6.500     0  03:20        0.0        50.1       0.000

  Continuity Error (%) .....        -0.012
  Continuity Error (%) .....         1.500
  ERROR 200: one or more errors in input file.
"""
    peaks, ro, rt, errs, _ = rn.parse_report(text)
    assert peaks["Outlet-A1"] == (6.5, "0 03:20") and peaks["J-A2"][0] == 5.678
    assert ro == -0.012 and rt == 1.5 and errs and errs[0].startswith("ERROR 200")


def test_swmm_engine_runs_every_storm(tmp_path, ddf_path):
    pytest.importorskip("swmm.toolkit")
    m = _model(
        ddf_path,
        extra={
            k: {"w_flow_m": 4000.0}
            for k in ("A1", "A2", "A3a", "A3b", "A4", "A5", "A3A1", "A3A2")
        },
    )
    inp.write_inp(tmp_path / "m.inp", m)
    res = rn.run_all(m, str(tmp_path / "runs"))
    assert len(res) == len(m.series)
    assert all(not r.errors for r in res)
    by = {r.run: r for r in res}
    big = by["Freq 1in50 6h P50"]
    assert big.peaks["Outlet-A1"][0] > by["Freq 1in2 6h P50"].peaks["Outlet-A1"][0] > 0
    assert abs(big.routing_error_pct) < 5 and abs(big.runoff_error_pct) < 1
    crit = ex.critical_rows(m, res)
    methods = {(row[0], row[1]) for row in crit}
    assert ("Frequency", "1in50") in methods and ("Huff", "1in50") in methods
    row = next(x for x in crit if x[:2] == ["Frequency", "1in50"])
    assert row[2] == 4 and row[4] == max(
        by[n].peaks["Outlet-A1"][0] for n in by if n.startswith("Freq 1in50")
    )
    path = ex.write_csv(str(tmp_path / "m.csv"), m, [("x", "y")], res)
    body = open(path, encoding="utf-8-sig").read()
    assert "## Critical storm per method and return period (SWMM)" in body


def test_transect_model_runs(tmp_path, ddf_path):
    pytest.importorskip("swmm.toolkit")
    m = _model(
        ddf_path, sc.SwmmSettings(xs_method=sc.XS_TRANSECT), dem=_dem(), peaks=(50,)
    )
    text = inp.inp_text(m)
    assert "[TRANSECTS]" in text and "IRREGULAR    R-A1" in text
    res = rn.run_all(m, str(tmp_path / "runs"))
    assert all(not r.errors for r in res)
    assert max(r.peaks["Outlet-A1"][0] for r in res) > 0
