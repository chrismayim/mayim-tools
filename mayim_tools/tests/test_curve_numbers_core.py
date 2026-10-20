"""Tests for SCS Curve Numbers (no QGIS; the end-to-end test needs GDAL)."""

from __future__ import annotations

import csv
import math
import os

import numpy as np
import pytest

from mayim_tools.land_cover.curve_numbers import core
from mayim_tools.land_cover.curve_numbers import lookup as lk
from mayim_tools.land_cover.curve_numbers import tables as tb
from mayim_tools.soil._common.errors import SoilDataError

# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------


def test_scs_sa_spot_values():
    r = tb.ROWS
    assert r["scs_sa:veld_2"].values == (49, 61, 69, 75, 79, 82, 84)
    assert r["scs_sa:urban_3"].values == (89, 91, 92, 93, 94, 95, 95)
    assert r["scs_sa:sugarcane_8"].values == (6, 14, 35, 59, 70, 75, 79)
    assert r["scs_sa:forest_12"].values == (18, 33, 47, 52, 57, 61, 65)
    assert r["scs_sa:row_crops_9"].values[-1] == 82
    assert len(tb.SCS_SA_ROWS) == 83


def test_tr55_agrees_with_scs_sa_where_rows_overlap():
    pairs = [
        ("scs_sa:veld_1", "tr55:pasture_poor"),
        ("scs_sa:veld_2", "tr55:pasture_fair"),
        ("scs_sa:veld_3", "tr55:pasture_good"),
        ("scs_sa:woods_1", "tr55:woods_poor"),
        ("scs_sa:woods_2", "tr55:woods_fair"),
        ("scs_sa:row_crops_1", "tr55:row_sr_poor"),
        ("scs_sa:row_crops_6", "tr55:row_c_good"),
        ("scs_sa:small_grain_1", "tr55:small_grain_sr_poor"),
        ("scs_sa:urban_5", "tr55:residential_500"),
        ("scs_sa:urban_6", "tr55:residential_1000"),
        ("scs_sa:fallow_1", "tr55:fallow_bare"),
    ]
    for a, b in pairs:
        va = [tb.ROWS[a].values[i] for i in (0, 2, 4, 6)]
        vb = [tb.ROWS[b].values[i] for i in (0, 2, 4, 6)]
        assert va == vb, (a, b)


def test_row_values_mean_and_constant():
    v = tb.row_values("scs_sa:row_crops_5 + scs_sa:row_crops_6")
    assert v[0] == pytest.approx(67.5) and v[-1] == pytest.approx(87.0)
    assert tb.row_values("const:100") == (100.0,) * 7
    assert tb.row_values("tr55:woods_fair")[1] is None
    with pytest.raises(KeyError):
        tb.row_values("scs_sa:nothing")
    assert tb.describe("const:98") == "Constant CN 98"


# ----------------------------------------------------------------------
# Lookup
# ----------------------------------------------------------------------


def test_default_lookups_complete():
    for key in ("worldcover_scs_sa", "worldcover_tr55"):
        look = lk.default_lookup(key)
        assert sorted(look.classes) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
        for e in look.classes.values():
            for case in lk.CASES:
                assert np.isfinite(e.values[case]).all()
            assert (e.values["low"] <= e.values["design"] + 1e-9).all()
            assert (e.values["high"] >= e.values["design"] - 1e-9).all()
    tr = lk.default_lookup("worldcover_tr55")
    # interpolated intermediate: woods fair A/B = (36 + 60) / 2
    assert tr.classes[10].values["design"][1] == pytest.approx(48.0)
    assert tr.classes[90].override == "D"


def test_lookup_csv_round_trip(tmp_path):
    look = lk.default_lookup("worldcover_scs_sa")
    path = tmp_path / "lk.csv"
    lk.write_lookup_csv(look, path)
    back = lk.read_lookup_csv(str(path))
    for code, e in look.classes.items():
        for case in lk.CASES:
            np.testing.assert_allclose(e.values[case], back.classes[code].values[case])
        assert back.classes[code].override == e.override


def test_lookup_csv_design_only_and_interpolation(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text(
        "code,class,A,B,C,D,group override\n"
        "1,Veld,49,69,79,84,\n"
        "2,Vlei,,,,84,d\n",
        encoding="utf-8",
    )
    look = lk.read_lookup_csv(str(path))
    v = look.classes[1].values
    assert v["design"][1] == pytest.approx(59.0)  # A/B from A and B
    np.testing.assert_allclose(v["low"], v["design"])
    assert look.classes[2].override == "D"
    assert any("no condition range" in n for n in look.notes)


@pytest.mark.parametrize(
    "body, message",
    [
        ("code,case,A,B,C,D\n1,mean,1,2,3,4\n", "case must be"),
        ("code,case,A,B,C,D\n1,design,1,2,,4\n", "no curve number"),
        ("code,case,A,B,C,D\n1,design,1,2,3,140\n", "between 0 and 100"),
        ("code,case,A,B,C,D,group override\n1,design,1,2,3,4,X\n", "override"),
        ("code,case,A,B,C,D\n1,low,1,2,3,4\n", "no 'design' row"),
        ("code,case,A,B,C\n1,design,1,2,3\n", "no 'D' column"),
    ],
)
def test_lookup_csv_errors(tmp_path, body, message):
    path = tmp_path / "bad.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(SoilDataError, match=message):
        lk.read_lookup_csv(str(path))


# ----------------------------------------------------------------------
# Pure functions
# ----------------------------------------------------------------------


def test_group_columns():
    codes = np.array([1, 21, 2, 22, 3, 23, 4, 11, 12, 13, 0, np.nan])
    und = core.group_columns(codes, "undrained")
    assert und.tolist() == [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, -1, -1]
    dr = core.group_columns(codes, "drained")
    assert dr[7:10].tolist() == [0, 2, 4]


def test_amc_adjust():
    cn = np.array([40.0, 70.0, 90.0, 100.0])
    np.testing.assert_allclose(core.amc_adjust(cn, "none"), cn)
    a1, a3 = core.amc_adjust(cn, "arc1"), core.amc_adjust(cn, "arc3")
    assert (a1[:3] < cn[:3]).all() and (a3[:3] > cn[:3]).all()
    assert a1[1] == pytest.approx(70 / (2.281 - 0.01281 * 70))
    assert core.amc_adjust(100.0, "sobhani") == pytest.approx(100.0)
    np.testing.assert_allclose(core.amc_adjust(cn, "median", 0.0), cn)
    up = core.amc_adjust(cn, "median", 25.4)
    assert up[1] == pytest.approx(1100 / (1100 / 70 - 1.0))
    assert (up <= 100).all()
    with pytest.raises(SoilDataError):
        core.amc_adjust(cn, "bogus")


def test_runoff_and_equivalent_cn():
    # CN 80, P 50 mm, Ia = 0.2 S: S = 63.5, Ia = 12.7
    assert float(core.runoff_mm(50, 80, 0.2)) == pytest.approx(37.3**2 / (37.3 + 63.5))
    assert float(core.runoff_mm(50, 100, 0.1)) == pytest.approx(50.0)
    assert float(core.runoff_mm(5, 50, 0.2)) == 0.0
    q = float(core.runoff_mm(80, 72.0, 0.1))
    assert core.equivalent_cn(80, q, 0.1) == pytest.approx(72.0, abs=1e-6)
    # a mix of 60 and 90 gives more runoff than CN 75
    q_mix = float(np.mean(core.runoff_mm(50, np.array([60.0, 90.0]), 0.1)))
    assert core.equivalent_cn(50, q_mix, 0.1) > 75.0


def test_conf_class():
    assert core.conf_class(np.array([5.0, 10.0, 19.9, 20.0, np.nan])).tolist() == [
        1,
        2,
        2,
        3,
        0,
    ]


def test_compute_block_by_hand():
    look = lk.default_lookup("worldcover_scs_sa")
    lc = np.array([[30, 90, 7, 30]])
    rec = np.array([[22.0, 2.0, 2.0, np.nan]])  # B/C, B, B, no soil
    p05 = np.array([[3.0, 3.0, 3.0, np.nan]])  # C (more runoff)
    p95 = np.array([[2.0, 1.0, 1.0, np.nan]])  # B (less runoff)
    out = core.compute_block(lc, rec, p05, p95, look, "undrained")
    # grassland = veld fair 75 on B/C; soil 69-79; condition good 68 / poor 83
    assert out["design"][0, 0] == 75
    assert out["p5"][0, 0] == pytest.approx(75 - math.hypot(6, 7))
    assert out["p95"][0, 0] == pytest.approx(75 + math.hypot(4, 8))
    assert out["share"][0, 0] == pytest.approx(100 * (36 + 16) / (36 + 16 + 49 + 64))
    # wetland: group D whatever the soil map says, no soil uncertainty
    assert (
        out["design"][0, 1] == 84 and out["p5"][0, 1] == 80 and out["p95"][0, 1] == 89
    )
    assert out["share"][0, 1] == 0
    assert np.isnan(out["design"][0, 2]) and np.isnan(out["design"][0, 3])
    # without the Ksat range only the condition range remains
    out2 = core.compute_block(lc, rec, None, None, look, "undrained")
    assert (
        out2["p5"][0, 0] == 68 and out2["p95"][0, 0] == 83 and out2["share"][0, 0] == 0
    )


# ----------------------------------------------------------------------
# End to end with GDAL (synthetic rasters)
# ----------------------------------------------------------------------


def _make_inputs(folder, method="scssa"):
    gdal = pytest.importorskip("osgeo.gdal")
    from osgeo import osr

    gdal.UseExceptions()
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32736)
    wkt = srs.ExportToWkt()

    def tif(path, arrs, res, x0, y1, descs, dtype, nd):
        h, w = arrs[0].shape
        ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, len(arrs), dtype)
        ds.SetGeoTransform((x0, res, 0, y1, 0, -res))
        ds.SetProjection(wkt)
        for i, (a, d) in enumerate(zip(arrs, descs, strict=True), 1):
            b = ds.GetRasterBand(i)
            b.SetNoDataValue(nd)
            b.SetDescription(d)
            b.WriteArray(a)
        ds = None

    lc = np.full((120, 200), 30, np.uint8)
    lc[:, :60] = 10
    lc[:, 120:160] = 40
    lc[:30, 160:] = 50
    lc[30:, 160:190] = 90
    lc[:, 190:] = 0
    lc[0:2, 0:2] = 7
    lcp = folder / "worldcover_2021_map.tif"
    tif(lcp, [lc], 10, 500000, 7160000, ["lc"], gdal.GDT_Byte, 0)
    g = np.full((12, 20), 3.0)
    g[:, :8] = 22
    g[:, 14:] = 2
    p05 = np.where(g == 2, 3.0, 4.0)
    p95 = np.where(g == 22, 2.0, 1.0)
    rec = folder / f"hsg_{method}_recommended.tif"
    tif(
        rec,
        [g],
        100,
        500000,
        7160000,
        [f"hsg_{method}_recommended (code)"],
        gdal.GDT_Float32,
        -9999,
    )
    tif(
        folder / f"hsg_{method}_by_ksat.tif",
        [p05, g, p95],
        100,
        500000,
        7160000,
        [
            f"hsg_{method}_at_ksat_P05 (code)",
            f"hsg_{method}_at_ksat_P50 (code)",
            f"hsg_{method}_at_ksat_P95 (code)",
        ],
        gdal.GDT_Float32,
        -9999,
    )
    zones = [
        (
            "West",
            "POLYGON((500000 7158800,501000 7158800,501000 7160000,500000 7160000,500000 7158800))",
        ),
        (
            "East",
            "POLYGON((501000 7158800,502000 7158800,502000 7160000,501000 7160000,501000 7158800))",
        ),
    ]
    return str(lcp), str(rec), zones


def test_end_to_end(tmp_path):
    lcp, rec, zones = _make_inputs(tmp_path)
    out = tmp_path / "out"
    s = core.RunSettings(
        lcp, rec, str(out), zones=zones, block_rows=37, write_report=False
    )
    r = core.run(s)
    core.write_outputs(r, "test")
    assert r.hsg.method == "SCSSA" and r.hsg.by_ksat_path.endswith(
        "hsg_scssa_by_ksat.tif"
    )
    assert r.ia_ratio == 0.1
    for f in (
        "curve_numbers.tif",
        "cn_zone_summary.csv",
        "cn_lookup_used.csv",
        "cn_table_scs_sa.csv",
        "cn_metadata.csv",
    ):
        assert (out / f).exists(), f
    from osgeo import gdal

    ds = gdal.Open(str(out / "curve_numbers.tif"))
    descs = [ds.GetRasterBand(i).GetDescription() for i in range(1, ds.RasterCount + 1)]
    assert descs[0] == "cn_design (CN)" and descs[4].startswith("cn_soil_share")
    design = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    assert design[50, 70] == 75  # grassland (veld fair) on B/C
    assert design[50, 100] == 79  # grassland on C
    assert design[60, 175] == 84  # wetland forced to D
    assert design[0, 0] == core.NODATA_OUT  # code 7 not in the lookup
    assert any("not in the lookup" in w for w in r.warnings)
    names = [z["Zone"] for z in r.zone_rows]
    assert names == ["Whole raster", "West", "East"]
    key = next(k for k in r.zone_rows[0] if k.startswith("Runoff-equivalent CN design"))
    for z in r.zone_rows:
        assert z["Mean CN P05"] <= z["Mean CN design"] <= z["Mean CN P95"]
        assert z[key] >= z["Mean CN design"] - 1e-9
    assert r.total.n == int(r.histogram.sum())
    shares = sum(c["Share (%)"] for c in r.class_rows)
    assert shares == pytest.approx(100.0)

    # The lookup written by the run reproduces the run as a custom lookup.
    s2 = core.RunSettings(
        lcp,
        rec,
        str(tmp_path / "out2"),
        lookup_key="custom",
        lookup_csv=str(out / "cn_lookup_used.csv"),
        write_report=False,
    )
    r2 = core.run(s2)
    assert r2.zone_rows[0]["Mean CN design"] == pytest.approx(
        r.zone_rows[0]["Mean CN design"]
    )


def test_end_to_end_neh630_with_scs_sa_warns(tmp_path):
    lcp, rec, _ = _make_inputs(tmp_path, method="neh630")
    r = core.run(core.RunSettings(lcp, rec, str(tmp_path / "o"), write_report=False))
    assert r.hsg.method == "NEH630"
    assert any("NEH 630 soil groups with the SCS-SA table" in w for w in r.warnings)


def test_end_to_end_without_range_and_amc(tmp_path):
    lcp, rec, _ = _make_inputs(tmp_path)
    base = core.run(
        core.RunSettings(
            lcp, rec, str(tmp_path / "a"), use_ksat_range=False, write_report=False
        )
    )
    assert (
        any("condition only" in w for w in base.warnings) or not base.hsg.by_ksat_path
    )
    wet = core.run(
        core.RunSettings(lcp, rec, str(tmp_path / "b"), amc="arc3", write_report=False)
    )
    assert wet.zone_rows[0]["Mean CN design"] > base.zone_rows[0]["Mean CN design"]


def test_end_to_end_report(tmp_path):
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    lcp, rec, zones = _make_inputs(tmp_path)
    r = core.run(core.RunSettings(lcp, rec, str(tmp_path / "o"), zones=zones))
    core.write_outputs(r, "test")
    assert os.path.exists(r.report_path)
    assert not [w for w in r.warnings if "not drawn" in w]


def test_no_overlap_raises(tmp_path):
    lcp, rec, _ = _make_inputs(tmp_path)
    path = tmp_path / "other.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Code", "Case", "A", "B", "C", "D"])
        w.writerow([999, "design", 50, 60, 70, 80])
    with pytest.raises(SoilDataError, match="No cell has both"):
        core.run(
            core.RunSettings(
                lcp,
                rec,
                str(tmp_path / "o"),
                lookup_key="custom",
                lookup_csv=str(path),
                write_report=False,
            )
        )


# ----------------------------------------------------------------------
# SANLC 2018 / 2020 lookup (WRC 2747/1/20 + SCS-SA Table 5.1)
# ----------------------------------------------------------------------


def test_wrc_rows_and_corrections():
    r = tb.ROWS
    assert r["wrc2747:48"].values == (51, 59, 65, 69, 73, 76, 77)
    assert r["wrc2747:47"].values == (78, 82, 86, 88, 90, 91, 92)
    assert r["wrc2747:6"].values[3] == 72  # printed 75
    assert r["wrc2747:61"].values[4] == pytest.approx(85.5)  # printed 76
    assert r["wrc2747:68"].values[6] == 98  # printed 95
    for row in tb.WRC_ROWS:  # corrected rows never decrease across groups
        v = row.values
        assert all(v[i] <= v[i + 1] for i in range(6)), row.key


def test_sanlc_lookup():
    look = lk.default_lookup("sanlc_2020")
    assert sorted(look.classes) == list(range(1, 74))
    for e in look.classes.values():
        for case in lk.CASES:
            assert np.isfinite(e.values[case]).all()
            assert ((e.values[case] >= 0) & (e.values[case] <= 100)).all()
        assert e.colour.startswith("#")
    d = {c: tuple(e.values["design"]) for c, e in look.classes.items()}
    w = {
        k.split(":")[1]: v.values for k, v in tb.ROWS.items() if k.startswith("wrc2747")
    }
    # formal residential -> DEA urban residential, informal -> DEA urban informal
    assert [d[c] for c in (47, 48, 49, 50)] == [w[str(n)] for n in (48, 49, 50, 51)]
    assert [d[c] for c in (51, 52, 53, 54)] == [w[str(n)] for n in (44, 45, 46, 47)]
    # recreational fields -> DEA 57-60; villages -> DEA 68
    assert [d[c] for c in (61, 62, 63, 64)] == [w[str(n)] for n in (57, 58, 59, 60)]
    assert d[55] == d[56] == w["68"]
    assert d[14] == (100.0,) * 7 and d[25] == (98.0,) * 7
    assert d[8][1] == 36  # composite A/B 39 corrected
    assert look.classes[22].override == "D" and look.classes[23].override == "D"
    assert look.classes[24].override == ""  # kept as in the composite
    # the high end flags a possibly low design value for annual crops
    assert look.classes[40].values["high"][2] == tb.ROWS["scs_sa:row_crops_1"].values[2]


def test_sanlc_end_to_end(tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    from osgeo import osr

    gdal.UseExceptions()
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32735)
    lc = np.array([[47, 51, 22, 14, 40, 50]], dtype=np.uint8).repeat(4, axis=0)
    path = tmp_path / "sanlc_2020.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 6, 4, 1, gdal.GDT_Byte)
    ds.SetGeoTransform((600000, 20, 0, 7150000, 0, -20))
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds.GetRasterBand(1).WriteArray(lc)
    ds = None
    hsg = tmp_path / "hsg_scssa_recommended.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(hsg), 2, 2, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((600000, 60, 0, 7150000, 0, -60))
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).SetDescription("hsg_scssa_recommended (code)")
    ds.GetRasterBand(1).WriteArray(np.full((2, 2), 2.0))  # group B everywhere
    ds = None
    out = tmp_path / "o"
    r = core.run(
        core.RunSettings(
            str(path), str(hsg), str(out), lookup_key="sanlc_2020", write_report=False
        )
    )
    core.write_outputs(r, "test")
    ds = gdal.Open(str(out / "curve_numbers.tif"))
    design = ds.GetRasterBand(1).ReadAsArray()[0]
    ds = None
    # formal res., informal res., wetland on D, river, rain-fed crops, formal bare
    assert design.tolist() == [65, 74, 80, 100, 69, 77]
    assert (out / "cn_table_wrc2747.csv").exists() and (
        out / "cn_table_scs_sa.csv"
    ).exists()
    assert not any("Built-up land covers" in w for w in r.warnings)
