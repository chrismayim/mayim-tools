"""Phase 1 loss / transform / routing methods of Prepare HEC-HMS project
(data/hec_hms_project/methods.py) - no QGIS needed.

The basin-file layouts below were checked by computing and re-saving
generated projects in HEC-HMS 4.14 (the saved file keeps every key, and the
results respond to the parameters in the expected units)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mayim_tools.data.hec_hms_project import core
from mayim_tools.data.hec_hms_project import hms_files as hf
from mayim_tools.data.hec_hms_project import methods as mth
from mayim_tools.data.hec_hms_project import storms as stm
from mayim_tools.tests.test_hec_hms_project_core import DDF_CSV, network_inputs

SOIL = {
    "ksat": 12.0,
    "psi_f": 110.0,
    "theta_s": 0.45,
    "theta_fc": 0.25,
    "theta_wp": 0.12,
}


@pytest.fixture
def ddf_path(tmp_path):
    p = tmp_path / "ddf.csv"
    p.write_text(DDF_CSV)
    return str(p)


def _assemble(ddf_path, subs=None, reaches=None, **kw):
    s_subs, s_reaches = network_inputs()
    c = stm.DdfCurves.from_csv(ddf_path)
    s = core.Settings(project_name="T1", dt_min=5, **kw)
    return core.assemble(
        subs or s_subs,
        s_reaches if reaches is None else reaches,
        [],
        [stm.frequency_storm(c, 50, 360, 50)],
        s,
    )


# ---------------------------------------------------------------- loss


def test_initial_constant_sources():
    ms = mth.MethodSettings(loss=mth.LOSS_IC, ksat_factor=0.5)
    p, src, w = mth.loss_parameters("A", None, SOIL, 72, 9.88, 98.8, ms)
    assert p["initial_loss_mm"] == 9.88 and src["initial_loss_mm"] == "Ia from the CN"
    assert p["constant_rate_mmh"] == 6.0 and "x 0.5" in src["constant_rate_mmh"]
    # fields win; without CN, soil or fields the defaults are used (and said)
    p, src, _ = mth.loss_parameters(
        "A", {"il_mm": 15, "cl_mmh": 2}, SOIL, 72, 9.88, 98.8, ms
    )
    assert (p["initial_loss_mm"], p["constant_rate_mmh"]) == (15, 2)
    assert src == {
        "initial_loss_mm": "field il_mm",
        "constant_rate_mmh": "field cl_mmh",
    }
    p, src, _ = mth.loss_parameters("A", None, None, None, None, None, ms)
    assert (p["initial_loss_mm"], p["constant_rate_mmh"]) == (10, 5)
    assert set(src.values()) == {"default"}


def test_deficit_constant_from_soil_and_cn():
    ms = mth.MethodSettings(loss=mth.LOSS_DC, soil_depth_mm=300)
    p, src, _ = mth.loss_parameters("A", None, SOIL, 72, 9.88, 98.8, ms)
    assert p["max_deficit_mm"] == pytest.approx((0.45 - 0.12) * 300)
    # moist start: theta_i = field capacity
    assert p["initial_deficit_mm"] == pytest.approx((0.45 - 0.25) * 300)
    assert "moist" in src["initial_deficit_mm"] and p["recovery_factor"] == 1.0
    dry = mth.MethodSettings(loss=mth.LOSS_DC, antecedent=mth.ANTECEDENT_DRY)
    p, _, _ = mth.loss_parameters("A", None, SOIL, 72, 9.88, 98.8, dry)
    assert p["initial_deficit_mm"] == pytest.approx(p["max_deficit_mm"])
    # no soil: S and Ia from the CN
    p, src, _ = mth.loss_parameters("A", None, None, 72, 9.88, 98.8, ms)
    assert (p["max_deficit_mm"], p["initial_deficit_mm"]) == (98.8, 9.88)
    assert src["max_deficit_mm"] == "S from the CN"
    # an initial deficit above the maximum is limited
    p, _, w = mth.loss_parameters("A", {"id_mm": 500}, SOIL, 72, 9.88, 98.8, ms)
    assert p["initial_deficit_mm"] == p["max_deficit_mm"] and w
    with pytest.raises(mth.MethodError):
        mth.loss_parameters("A", None, None, None, None, None, ms)


def test_green_ampt():
    ms = mth.MethodSettings(loss=mth.LOSS_GA)
    p, src, _ = mth.loss_parameters("A", None, SOIL, None, None, None, ms)
    assert (p["ga_ks_mmh"], p["ga_psi_mm"], p["ga_ts"], p["ga_ti"]) == (
        12.0,
        110.0,
        0.45,
        0.25,
    )
    assert "theta_fc" in src["ga_ti"]
    p, src, _ = mth.loss_parameters("A", {"ga_ks_mmh": 3}, SOIL, None, None, None, ms)
    assert p["ga_ks_mmh"] == 3 and src["ga_ks_mmh"] == "field ga_ks_mmh"
    with pytest.raises(mth.MethodError, match="Green and Ampt"):
        mth.loss_parameters("A", None, {"ksat": 5}, None, None, None, ms)
    # initial content at or above saturation is limited
    p, _, w = mth.loss_parameters("A", {"ga_ti": 0.6}, SOIL, None, None, None, ms)
    assert p["ga_ti"] == pytest.approx(0.99 * 0.45) and w


# ---------------------------------------------------------------- transform / routing


def test_snyder_and_routing_estimates():
    ms = mth.MethodSettings()
    p, src, _ = mth.snyder_parameters("A", None, 30, ms)
    assert p["snyder_tp_h"] == pytest.approx(0.3) and p["snyder_cp"] == 0.6
    ct = mth.MethodSettings(snyder_tp=mth.SNYDER_TP_CT, snyder_ct=2.0)
    p, src, _ = mth.snyder_parameters("A", {"lfp_km": 4.0, "lca_km": 2.0}, 30, ct)
    assert p["snyder_tp_h"] == pytest.approx(0.75 * 2.0 * 8.0**0.3)
    p, _, w = mth.snyder_parameters("A", None, 30, ct)
    assert p["snyder_tp_h"] == pytest.approx(0.3) and w
    # HEC-HMS rejects Tp < 0.1 h: raised and said
    p, src, w = mth.snyder_parameters("A", None, 5, ms)
    assert p["snyder_tp_h"] == 0.1 and w and "HEC-HMS minimum" in src["snyder_tp_h"]
    p, src = mth.lagk_parameters(None, 60, mth.MethodSettings(lagk_lag_share=0.25))
    assert (p["lagk_lag_h"], p["lagk_k_h"]) == (0.25, 0.75)
    p, src = mth.lagk_parameters({"lagk_lag_h": 0.1}, 60, ms)
    assert p["lagk_lag_h"] == 0.1 and src["lagk_lag_h"] == "field lagk_lag_h"
    assert mth.kw_subreaches(900, 1.5, 5) == 2 and mth.kw_subreaches(10, 1.5, 5) == 1


# ---------------------------------------------------------------- basin file


def test_basin_text_for_each_method(ddf_path):
    subs, _ = network_inputs()
    for x in subs:
        x.soil = dict(SOIL)
    r = _assemble(
        ddf_path,
        subs=subs,
        transform=core.TRANSFORM_SNYDER,
        routing=core.ROUTE_LAGK,
        methods=mth.MethodSettings(loss=mth.LOSS_GA),
    )
    text = hf.basin_text(r, hf.ProjectNames(r, "T1"))
    for line in (
        "     LossRate: Green and Ampt\r\n",
        "     Initial Variable: Initial Content\r\n",
        "     Capillarity Method: SUCTION\r\n",
        "     Wetting Front Suction: 110\r\n",
        "     Hydraulic Conductivity: 12\r\n",
        "     Transform: Snyder\r\n     Snyder Method: Standard\r\n",
        "     Route: Lag & K\r\n",
        "     Use Constant Lag: Yes\r\n",
    ):
        assert line in text
    assert "Curve Number:" not in text
    a1 = r.network.by_name()["A1"].params
    assert a1["lag_min"] == pytest.approx(15.0)  # Tp = 0.6 Tc = 0.25 h
    assert a1["sources"]["ga_ks_mmh"].startswith("Ksat")

    r = _assemble(
        ddf_path,
        routing=core.ROUTE_KW,
        methods=mth.MethodSettings(loss=mth.LOSS_IC),
    )
    text = hf.basin_text(r, hf.ProjectNames(r, "T1"))
    assert (
        "     LossRate: Initial+Constant\r\n     Percent Impervious Area: 0\r\n" in text
    )
    assert "     Initial Loss: 9.878\r\n" in text  # Ia of CN 72 (A1)
    assert "     Route: Kinematic Wave\r\n     Channel: Kinematic Wave\r\n" in text
    assert "     Shape: Trapezoid\r\n" in text and "Number of Subreaches:" in text
    rk = r.network.by_name()["R-A3A1"].params
    assert rk["method"] == core.ROUTE_KW and rk["subreaches"] == 2

    r = _assemble(ddf_path, methods=mth.MethodSettings(loss=mth.LOSS_DC))
    text = hf.basin_text(r, hf.ProjectNames(r, "T1"))
    assert "     LossRate: Deficit Constant\r\n" in text
    assert "     Recovery Factor: 1\r\n" in text


def test_cn_only_needed_for_scs(ddf_path):
    subs, _ = network_inputs()
    for x in subs:
        x.cn = None
        x.soil = dict(SOIL)
    r = _assemble(ddf_path, subs=subs, methods=mth.MethodSettings(loss=mth.LOSS_GA))
    assert "cn" not in r.network.by_name()["A1"].params
    with pytest.raises(core.HmsProjectError, match="curve number"):
        _assemble(ddf_path, subs=subs)


def test_csv_and_report_show_methods_and_sources(tmp_path, ddf_path):
    from mayim_tools.data.hec_hms_project.export import write_csv

    subs, _ = network_inputs()
    for x in subs:
        x.soil = dict(SOIL)
    r = _assemble(
        ddf_path,
        subs=subs,
        transform=core.TRANSFORM_SNYDER,
        routing=core.ROUTE_KW,
        methods=mth.MethodSettings(loss=mth.LOSS_DC),
    )
    p = write_csv(str(tmp_path / "s.csv"), r, [])
    text = open(p, encoding="utf-8").read()
    assert "Loss method,Deficit and Constant" in text
    assert "Snyder unit hydrograph (standard)" in text
    assert "max_deficit_mm: (theta_s - theta_wp) x 300 mm (soil rasters)" in text
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    import docx

    from mayim_tools.data.hec_hms_project.report import write_report

    write_report(str(tmp_path / "r.docx"), r, {"inputs": [], "outputs": []}, {})
    body = "\n".join(p.text for p in docx.Document(str(tmp_path / "r.docx")).paragraphs)
    assert "Losses: Deficit and Constant" in body.replace("LOSSES", "Losses")
    assert "Snyder" in body and "Kinematic wave" in body


# ---------------------------------------------------------------- soil rasters


def _write_raster(path, arr, desc):
    from osgeo import gdal, osr

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(str(path), arr.shape[1], arr.shape[0], len(desc), gdal.GDT_Float32)
    ds.SetGeoTransform((0, 100, 0, 1000, 0, -100))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32735)
    ds.SetProjection(srs.ExportToWkt())
    for i, d in enumerate(desc, start=1):
        b = ds.GetRasterBand(i)
        b.WriteArray(arr * i)
        b.SetDescription(d)
        b.SetNoDataValue(-9999)
    ds = None


def test_find_soil_rasters_and_zonal_values(tmp_path):
    pytest.importorskip("osgeo")
    arr = np.full((10, 10), 4.0, dtype=float)
    arr[:, 5:] = 16.0
    _write_raster(
        tmp_path / "P1_Ksat.tif",
        arr,
        ["ksat_0-30cm_P50 (mm/h)", "ksat_30-60cm_P50 (mm/h)"],
    )
    _write_raster(tmp_path / "P1_Suction.tif", arr * 10, ["psi_f_0-30cm_P50 (mm)"])
    (tmp_path / "notes.txt").write_text("x")
    found, notes = mth.find_soil_rasters(str(tmp_path))
    assert set(found) == {"ksat", "psi_f"} and found["ksat"][1] == 1 and not notes
    zone = "POLYGON ((0 0, 1000 0, 1000 1000, 0 1000, 0 0))"
    tiny = "POLYGON ((610 410, 620 410, 620 420, 610 420, 610 410))"
    k = mth.zonal_values(found["ksat"][0], 1, [zone, tiny], geometric=True)
    assert k[0] == pytest.approx(8.0)  # geometric mean of 4 and 16
    assert k[1] == pytest.approx(16.0)  # centroid cell
    psi = mth.zonal_values(*found["psi_f"], [zone])
    assert psi[0] == pytest.approx(100.0)  # arithmetic mean of 40 and 160
    with pytest.raises(mth.MethodError):
        mth.find_soil_rasters(os.path.join(str(tmp_path), "missing"))
