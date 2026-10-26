"""Areal reduction factors (rainfall/_common/arf.py) and their use in the
storm builders of Prepare HEC-HMS / SWMM project and the Alternating Block
tool - no QGIS needed."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from mayim_tools.data.hec_hms_project import core
from mayim_tools.data.hec_hms_project import storms as stm
from mayim_tools.rainfall._common import arf as arfm
from mayim_tools.tests.test_hec_hms_project_core import DDF_CSV, network_inputs


@pytest.fixture
def ddf_path(tmp_path):
    p = tmp_path / "ddf.csv"
    p.write_text(DDF_CSV)
    return str(p)


# ---------------------------------------------------------------- equations


def test_eq310_matches_the_sanral_worked_examples():
    # SANRAL Application Guide examples 3.1 and 3.2 (96.0 % and 80.0 %)
    assert arfm.arf_eq310(28.5, 1.338 * 60) == pytest.approx(0.960, abs=0.001)
    assert arfm.arf_eq310(4318, 31.16 * 60) == pytest.approx(0.800, abs=0.003)
    assert arfm.arf_eq310(1.0, 15) == 1.0


def test_feh_values_and_band_continuity():
    # FEH Vol 2 / Keers and Wescott: 500 km2, 12 h -> 1 - 0.304 x 12^-0.390
    assert arfm.arf_feh(500, 720) == pytest.approx(0.8847, abs=0.0005)
    # ARF falls with area and rises with duration
    assert arfm.arf_feh(10, 60) > arfm.arf_feh(100, 60) > arfm.arf_feh(1000, 60)
    assert arfm.arf_feh(100, 60) < arfm.arf_feh(100, 1440)
    # the area bands join up (within 1 %)
    for a in (20, 100, 500, 1000):
        lo, hi = arfm.arf_feh(a * 0.9999, 60), arfm.arf_feh(a * 1.0001, 60)
        assert abs(lo - hi) < 0.01
    # Eq 3.10 (Alexander's fit) is gentler than the UK curves for small areas
    assert arfm.arf_eq310(10, 60) == 1.0 and arfm.arf_feh(10, 60) < 0.92


def test_pietersen_with_eq310_below_24h():
    m = arfm.ArfModel(arfm.METHOD_PIETERSEN, area_km2=500, region=2)
    short = m.factor(360, 50)
    assert short == pytest.approx(arfm.arf_eq310(500, 360))
    assert "pietersen_fallback" in m.flags and m.notes()
    long = m.factor(1440, 50)
    assert long == pytest.approx(arfm.arf_pietersen(2, 500, 1440, 50))
    with pytest.raises(arfm.ArfError):
        arfm.ArfModel(arfm.METHOD_PIETERSEN, area_km2=500)  # no region


def test_model_validation_and_labels():
    with pytest.raises(arfm.ArfError):
        arfm.ArfModel(arfm.METHOD_EQ310)  # no area
    with pytest.raises(arfm.ArfError):
        arfm.ArfModel(arfm.METHOD_CONSTANT, constant=1.5)
    assert arfm.ArfModel(arfm.METHOD_NONE).factor(60) == 1.0
    assert arfm.ArfModel(arfm.METHOD_NONE).is_unity
    c = arfm.ArfModel(arfm.METHOD_CONSTANT, constant=0.9)
    assert c(5) == c(1440) == 0.9 and not c.varies_with_duration
    m = arfm.ArfModel(arfm.METHOD_EQ310, area_km2=45.2)
    assert m.label() == "SANRAL DM Eq 3.10, A = 45.2 km2"
    assert m(5) < m(60) < m(1440) <= 1.0


def test_user_table(tmp_path):
    p = tmp_path / "arf.csv"
    p.write_text("DurationMin,ARF\n60,80\n360,90\n1440,95\n")
    t = arfm.ArfTable.from_csv(str(p))
    m = arfm.ArfModel(arfm.METHOD_TABLE, table=t)
    assert not m.needs_area  # no area column
    assert m(60) == pytest.approx(0.80) and m(1440) == pytest.approx(0.95)
    # linear in ln D between rows
    assert m(np.sqrt(60 * 360)) == pytest.approx(0.85)
    assert m(15) == pytest.approx(0.80) and "table_extrap" in m.flags
    # with areas
    p2 = tmp_path / "arf2.csv"
    p2.write_text(
        "AreaKm2,Duration,ARF\n10,1h,0.95\n10,24h,0.99\n100,1h,0.85\n100,24h,0.95\n"
    )
    m2 = arfm.ArfModel(
        arfm.METHOD_TABLE, area_km2=np.sqrt(1000), table=arfm.ArfTable.from_csv(str(p2))
    )
    assert m2(60) == pytest.approx(0.90)
    p3 = tmp_path / "bad.csv"
    p3.write_text("Duration,Depth\n60,10\n")
    with pytest.raises(arfm.ArfError):
        arfm.ArfTable.from_csv(str(p3))


# ---------------------------------------------------------------- storms


def test_frequency_storm_per_duration_arf(ddf_path):
    c = stm.DdfCurves.from_csv(ddf_path)
    m = arfm.ArfModel(arfm.METHOD_EQ310, area_km2=200)
    f = stm.frequency_storm(c, 50, 360, 50, m)
    assert f.depths[5] == pytest.approx(18.9 * m(5))
    assert f.depths[360] == pytest.approx(79.1 * m(360))
    assert f.arf_peak == pytest.approx(m(5)) and f.arf_total == pytest.approx(m(360))
    assert f.arf_peak < f.arf_total
    assert "ARF" in f.description
    # a constant still scales every depth alike; none leaves them alone
    g = stm.frequency_storm(c, 50, 360, 50, 0.9)
    assert g.depths[5] == pytest.approx(0.9 * 18.9)
    h = stm.frequency_storm(c, 50, 360, 50, arfm.ArfModel())
    assert h.depths[5] == pytest.approx(18.9) and h.arf_values == {}


def test_areal_alternating_block_keeps_the_ddf_structure():
    inc = [1.0, 2.0, 5.0, 9.0, 4.0, 3.0]  # peak at index 3
    arf = {k * 5.0: 0.70 + 0.04 * k for k in range(1, 7)}
    out, used = stm.areal_alternating_block(inc, 5.0, lambda d: arf[d])
    assert used == arf
    order = sorted(range(6), key=lambda i: -inc[i])
    cum = np.cumsum([inc[i] for i in order])
    got = np.cumsum([out[i] for i in order])
    # the k largest blocks add up to the (point depth for k steps) x ARF(k steps)
    assert np.allclose(got, [c * arf[(k + 1) * 5.0] for k, c in enumerate(cum)])
    assert int(np.argmax(out)) == 3  # peak position kept
    # constant: plain scaling
    out2, _ = stm.areal_alternating_block(inc, 5.0, lambda d: 0.9)
    assert np.allclose(out2, np.array(inc) * 0.9)


def test_alternating_block_storms_from_the_tool(tmp_path, ddf_path):
    from mayim_tools.rainfall.ddf_to_hyetographs import export as abx
    from mayim_tools.rainfall.ddf_to_hyetographs.core import parse_idf_table

    tables = parse_idf_table(pd.read_csv(pd.io.common.StringIO(DDF_CSV)))
    point = tmp_path / "ab.csv"
    abx.write_incremental_depth([(50.0, 360.0)], 5, tables, point)
    m = arfm.ArfModel(arfm.METHOD_EQ310, area_km2=200)
    rows = stm.read_alternating_block_csv(str(point))
    st = stm.alternating_block_storms(rows, [50], [], 50, m)[0][0]
    assert st.total_mm == pytest.approx(79.1 * m(360), rel=1e-3)
    assert max(st.increments) == pytest.approx(18.9 * m(5), rel=1e-3)
    # the Alternating Block tool reducing the hyetographs itself gives the same
    # storm, flagged so the model tools do not reduce it twice
    areal = tmp_path / "ab_areal.csv"
    abx.write_incremental_depth([(50.0, 360.0)], 5, tables, areal, arf_model=m)
    rows2 = stm.read_alternating_block_csv(str(areal))
    assert rows2[0]["arf_method"].startswith("SANRAL DM Eq 3.10")
    st2 = stm.alternating_block_storms(rows2, [50], [], 50, m)[0][0]
    assert np.allclose(st2.increments, st.increments, rtol=1e-3, atol=1e-3)
    assert st2.arf_values == {} and "in the CSV" in st2.description


def test_ab_tool_core_with_arf():
    from mayim_tools.rainfall.ddf_to_hyetographs.core import (
        build_hyetograph,
        parse_idf_table,
    )

    tables = parse_idf_table(pd.read_csv(pd.io.common.StringIO(DDF_CSV)))
    m = arfm.ArfModel(arfm.METHOD_FEH, area_km2=50)
    h = build_hyetograph(60, 5, tables[50.0], False, lambda t: m(t, 50))
    inc = h[0.5]["incremental"]
    assert sum(inc) == pytest.approx(52.4 * m(60), rel=1e-6)
    assert max(inc) == pytest.approx(18.9 * m(5), rel=1e-6)
    # still an alternating block storm (peak in the middle)
    assert int(np.argmax(inc)) == round(0.5 * (len(inc) - 1))


def test_huff_and_ensemble_scale_with_arf_at_d(tmp_path, ddf_path):
    c = stm.DdfCurves.from_csv(ddf_path)
    m = arfm.ArfModel(arfm.METHOD_EQ310, area_km2=200)
    x = np.linspace(0, 1, 5)
    h = stm.huff_storms(c, {(1, "P50"): (x, x**0.5)}, [50], [360], 10, 1, "P50", m)
    st = h[0][0]
    assert st.total_mm == pytest.approx(79.1 * m(360))
    assert st.arf_values == {360.0: pytest.approx(m(360))}
    rows = [
        {
            "duration_min": 360,
            "interval_min": 30,
            "band": stm.ensemble_band(50),
            "shape": "Middle",
            "increments_pct": [5.0] * 4 + [30.0] * 2 + [5.0] * 6,
        }
    ]
    e = stm.ensemble_storms(c, rows, [50], [360], ["Middle"], m)[0][0]
    assert e.total_mm == pytest.approx(79.1 * m(360))


def test_burst_check(ddf_path):
    c = stm.DdfCurves.from_csv(ddf_path)
    m = arfm.ArfModel(arfm.METHOD_EQ310, area_km2=200)
    # a uniform 6 h storm is far smoother than the DDF
    x = np.linspace(0, 1, 5)
    smooth = stm.huff_storms(c, {(1, "P50"): (x, x)}, [50], [360], 5, 1, "P50", m)[0]
    # an alternating block storm built from the DDF matches it (100 %)
    from mayim_tools.rainfall.ddf_to_hyetographs.core import (
        build_hyetograph,
        parse_idf_table,
    )

    tables = parse_idf_table(pd.read_csv(pd.io.common.StringIO(DDF_CSV)))
    h = build_hyetograph(360, 5, tables[50.0], False, lambda t: m(t, 50))
    ab = stm.Storm(stm.SOURCE_AB, 50, 360, "AB", "", 50, None, 5, h[0.5]["incremental"])
    storms = smooth + [ab]
    stm.add_burst_checks(storms, c, m)
    by = {b[0]: b[3] for b in ab.burst}
    assert set(by) == {15, 30, 60, 120, 180}
    for d in (15, 30, 60, 120):  # table durations: exact
        assert by[d] == pytest.approx(100, abs=0.5)
    low = {b[0]: b[3] for b in smooth[0].burst}
    assert low[15] < 30 and low[180] < 100
    from mayim_tools.data.hec_hms_project.export import burst_min, burst_table

    v, at = burst_min(smooth[0])
    assert at == "15min" and v == pytest.approx(low[15])
    head, rows = burst_table([("Huff", smooth[0]), ("AB", ab)])
    assert head[:3] == ["run", "pattern", "15min"] and len(rows) == 2


# ---------------------------------------------------------------- model tools


def test_system_areas():
    subs, reaches = network_inputs()
    areas = core.system_areas(subs, reaches, [], core.Settings())
    assert list(areas) == ["Outlet-A1"]
    assert areas["Outlet-A1"] == pytest.approx(sum(x.area_km2 for x in subs))
    # without reaches every sub-catchment is its own system, largest first
    own = core.system_areas(
        [core.SubIn(x.name, x.area_km2, x.cx, x.cy, tc_min=20, cn=70) for x in subs],
        [],
        [],
        core.Settings(),
    )
    assert len(own) == len(subs) and next(iter(own.values())) == 1.2


def test_csv_and_report_list_the_arf(tmp_path, ddf_path):
    from mayim_tools.data.hec_hms_project.export import arf_table, write_csv

    subs, reaches = network_inputs()
    c = stm.DdfCurves.from_csv(ddf_path)
    area = sum(x.area_km2 for x in subs)
    m = arfm.ArfModel(arfm.METHOD_EQ310, area_km2=area, area_note="total area")
    storms = [stm.frequency_storm(c, t, 360, 50, m) for t in (2, 50)]
    x = np.linspace(0, 1, 5)
    storms += stm.huff_storms(
        c, {(1, "P50"): (x, x**0.5)}, [2, 50], [360], 10, 1, "P50", m
    )[0]
    stm.add_burst_checks(storms, c, m)
    r = core.assemble(subs, reaches, [], storms, core.Settings(ensembles=False))
    r.arf = m
    head, rows = arf_table(r.storms, m)
    assert head == ["duration", "1in2", "1in50"] and rows[0][0] == "5min"
    assert rows[0][1] == pytest.approx(m(5), abs=1e-4)
    p = write_csv(str(tmp_path / "s.csv"), r, [])
    text = open(p, encoding="utf-8").read()
    assert 'Areal reduction,"SANRAL DM Eq 3.10, A = 5.4 km2"' in text
    assert "## Areal reduction factors" in text and "## Burst check" in text
    with open(p, encoding="utf-8") as f:
        body = list(csv.reader(f))
    hdr = next(row for row in body if row[:2] == ["run", "source"])
    assert "arf_storm" in hdr and "burst_min_pct" in hdr
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    from mayim_tools.data.hec_hms_project.report import write_report

    notes = write_report(str(tmp_path / "r.docx"), r, {"inputs": [], "outputs": []}, {})
    assert notes == []
    import docx

    doc = docx.Document(str(tmp_path / "r.docx"))
    txt = "\n".join(p.text for p in doc.paragraphs)
    assert "Areal reduction" in txt and "Burst check" in txt
    assert "Eq 3.10" in txt
