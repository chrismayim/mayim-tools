"""Tests for Regional soil parameterisation (no QGIS needed).

Method checks reproduce published values (Saxton & Rawls 2006, Table 3);
the end-to-end tests build small synthetic extraction folders with the
exact band descriptions of the two extraction tools (need GDAL).
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil.regional_parameterisation import fill as fill_mod
from mayim_tools.soil.regional_parameterisation import inputs as inp
from mayim_tools.soil.regional_parameterisation import uncertainty as unc
from mayim_tools.soil.regional_parameterisation.ptf import rawls_1983
from mayim_tools.soil.regional_parameterisation.ptf.saxton_rawls import (
    FLAG_AIR_ENTRY,
    FLAG_CLAY,
    FLAG_OM,
    KPA_TO_MM,
    YE_MIN_KPA,
    saxton_rawls,
)
from mayim_tools.soil.regional_parameterisation.texture import (
    CLASS_NAME,
    ISO_TO_USDA_SILT_FACTOR,
    iso_to_usda,
    normalise_texture,
    ternary_xy,
    usda_class,
)

# Saxton & Rawls (2006) Table 3: class, sand, clay, WP, FC, SAT, PAW (%v),
# Ksat (mm/h), matric density (g/cm3) at 2.5 % OM.
TABLE3 = [
    ("Sand", 88, 5, 5, 10, 46, 5, 108.1, 1.43),
    ("Loamy sand", 80, 5, 5, 12, 46, 7, 96.7, 1.43),
    ("Sandy loam", 65, 10, 8, 18, 45, 10, 50.3, 1.46),
    ("Loam", 40, 20, 14, 28, 46, 14, 15.5, 1.43),
    ("Silt loam", 20, 15, 11, 31, 48, 20, 16.1, 1.38),
    ("Silt", 10, 5, 6, 30, 48, 25, 22.0, 1.38),
    ("Sandy clay loam", 60, 25, 17, 27, 43, 10, 11.3, 1.50),
    ("Clay loam", 30, 35, 22, 36, 48, 14, 4.3, 1.39),
    ("Silty clay loam", 10, 35, 22, 38, 51, 17, 5.7, 1.30),
    ("Silty clay", 10, 45, 27, 41, 52, 14, 3.7, 1.26),
    ("Sandy clay", 50, 40, 25, 36, 44, 11, 1.4, 1.47),
    ("Clay", 25, 50, 30, 42, 50, 12, 1.1, 1.33),
]


# ----------------------------------------------------------------------
# Saxton & Rawls (2006)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("row", TABLE3, ids=[r[0] for r in TABLE3])
def test_saxton_rawls_reproduces_table3(row):
    _, sand, clay, wp, fc, sat, paw, ksat, rho = row
    r = saxton_rawls(sand, clay, 2.5)
    assert round(float(r["theta_wp"]) * 100) == wp
    assert round(float(r["theta_fc"]) * 100) == fc
    assert round(float(r["theta_s"]) * 100) == sat
    assert round(float(r["paw"]) * 100) == paw
    assert round(float(r["ksat_mm_h"]), 1) == ksat
    assert round(float(r["rho_normal"]), 2) == rho


def test_saxton_rawls_vectorised_matches_scalar():
    sand = np.array([r[1] for r in TABLE3], float)
    clay = np.array([r[2] for r in TABLE3], float)
    v = saxton_rawls(sand, clay, np.full(sand.shape, 2.5))
    for i, row in enumerate(TABLE3):
        s = saxton_rawls(row[1], row[2], 2.5)
        assert v["ksat_mm_h"][i] == pytest.approx(float(s["ksat_mm_h"]))


def test_saxton_rawls_green_ampt_suction_formula():
    r = saxton_rawls(40, 20, 2.5)
    lam = float(r["lambda"])
    expected = (2 + 3 * lam) / (1 + 3 * lam) * float(r["psi_e_kpa"]) / 2 * KPA_TO_MM
    assert float(r["psi_f_mm"]) == pytest.approx(expected)
    assert float(r["lambda"]) == pytest.approx(1.0 / float(r["B"]))


def test_saxton_rawls_air_entry_bounded_for_sand():
    r = saxton_rawls(88, 5, 2.5)
    assert float(r["psi_e_raw_kpa"]) < 0  # Eq. 4 is negative for sand
    assert float(r["psi_e_kpa"]) == pytest.approx(YE_MIN_KPA)
    assert int(r["flags"]) & FLAG_AIR_ENTRY
    assert float(r["psi_f_mm"]) > 0


def test_saxton_rawls_flags_outside_calibration():
    r = saxton_rawls(10, 65, 9.0)
    assert int(r["flags"]) & FLAG_CLAY
    assert int(r["flags"]) & FLAG_OM
    assert int(saxton_rawls(40, 20, 2.5)["flags"]) == 0


def test_saxton_rawls_gravel_and_density_options():
    base = saxton_rawls(40, 20, 2.5)
    g0 = saxton_rawls(40, 20, 2.5, gravel_vol_pct=0.0, use_gravel=True)
    assert float(g0["ksat_mm_h"]) == pytest.approx(float(base["ksat_mm_h"]))
    g = saxton_rawls(40, 20, 2.5, gravel_vol_pct=30.0, use_gravel=True)
    assert float(g["ksat_mm_h"]) < float(base["ksat_mm_h"])
    assert float(g["paw"]) == pytest.approx(float(base["paw"]) * 0.7)
    # Denser than normal -> lower porosity and Ksat; DF limited to 1.3
    dense = saxton_rawls(40, 20, 2.5, bulk_density=2.5, use_density=True)
    assert float(dense["density_factor"]) == pytest.approx(1.3)
    assert float(dense["theta_s"]) < float(base["theta_s"])
    assert float(dense["ksat_mm_h"]) < float(base["ksat_mm_h"])
    # Unknown bulk density -> no adjustment
    nan = saxton_rawls(40, 20, 2.5, bulk_density=np.nan, use_density=True)
    assert float(nan["theta_s"]) == pytest.approx(float(base["theta_s"]))


def test_saxton_rawls_nan_propagates():
    r = saxton_rawls(np.array([40.0, np.nan]), np.array([20.0, 20.0]), 2.5)
    assert np.isfinite(r["ksat_mm_h"][0]) and np.isnan(r["ksat_mm_h"][1])


# ----------------------------------------------------------------------
# Texture
# ----------------------------------------------------------------------


def _canonical_class(sa, si, cl):
    """USDA rules as published by NRCS (independent of the implementation)."""
    rules = {
        1: (si + 1.5 * cl) < 15,
        2: ((si + 1.5 * cl) >= 15) & ((si + 2 * cl) < 30),
        3: ((cl >= 7) & (cl < 20) & (sa > 52) & ((si + 2 * cl) >= 30))
        | ((cl < 7) & (si < 50) & ((si + 2 * cl) >= 30)),
        4: (cl >= 7) & (cl < 27) & (si >= 28) & (si < 50) & (sa <= 52),
        5: ((si >= 50) & (cl >= 12) & (cl < 27)) | ((si >= 50) & (si < 80) & (cl < 12)),
        6: (si >= 80) & (cl < 12),
        7: (cl >= 20) & (cl < 35) & (si < 28) & (sa > 45),
        8: (cl >= 27) & (cl < 40) & (sa > 20) & (sa <= 45),
        9: (cl >= 27) & (cl < 40) & (sa <= 20),
        10: (cl >= 35) & (sa > 45),
        11: (cl >= 40) & (si >= 40),
        12: (cl >= 40) & (sa <= 45) & (si < 40),
    }
    out = np.zeros(sa.shape, int)
    count = np.zeros(sa.shape, int)
    for code, mask in rules.items():
        out[mask] = code
        count += mask
    return out, count


def test_usda_class_matches_canonical_rules_everywhere():
    g = np.arange(0, 100.01, 0.5)
    sa, cl = np.meshgrid(g, g)
    keep = sa + cl <= 100
    sa, cl = sa[keep], cl[keep]
    si = 100 - sa - cl
    ref, count = _canonical_class(sa, si, cl)
    assert (count == 1).all()  # the rules partition the triangle
    assert (usda_class(sa, si, cl) == ref).all()


def test_usda_class_table3_examples_and_nan():
    for name, sand, clay, *_ in TABLE3:
        code = int(usda_class(sand, 100 - sand - clay, clay))
        assert CLASS_NAME[code] == name
    assert int(usda_class(np.nan, 20, 20)) == 0


def test_iso_to_usda_conversion():
    assert ISO_TO_USDA_SILT_FACTOR == pytest.approx(math.log(25) / math.log(31.5))
    sand, silt, clay = iso_to_usda(40.0, 30.0, 30.0)
    assert float(clay) == 30.0
    assert float(silt) == pytest.approx(30.0 * ISO_TO_USDA_SILT_FACTOR)
    assert float(sand + silt + clay) == pytest.approx(100.0)
    assert float(sand) > 40.0


def test_normalise_texture_and_ternary():
    s, si, c = normalise_texture(50.0, 20.0, 30.0 * 1.1)
    assert float(s + si + c) == pytest.approx(100.0)
    x, y = ternary_xy(100.0, 0.0)
    assert (float(x), float(y)) == (0.0, 0.0)
    x, y = ternary_xy(0.0, 100.0)
    assert float(x) == pytest.approx(50.0)


# ----------------------------------------------------------------------
# Harmonisation
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "desc,kind,var,top,bottom,stat",
    [
        ("clay_15-30cm_Q0.5 (%)", "sg", "clay", 15, 30, "Q0.5"),
        ("bdod_0-5cm_mean (g/cm3)", "sg", "bdod", 0, 5, "mean"),
        ("cec_0-5cm_Q0.5 (cmol(c)/kg)", "sg", "cec", 0, 5, "Q0.5"),
        ("cec_15cm_mean [SG2017] (cmol(c)/kg)", "sg2017", "cec", 15, None, "mean"),
        ("clay_15cm_mean [SG2017] (%)", "sg2017", "clay", 15, None, "mean"),
        ("clay_0-30cm_mean_30m_2020-2022 (%)", "olm", "clay", 0, 30, "mean_30m"),
        ("soc_30-60cm_p84_120m_2015-2020 (g/kg)", "olm", "soc", 30, 60, "p84_120m"),
        (
            "field_capacity_33kPa_30cm_250m_1950-2017 (vol %)",
            "water",
            "fc",
            30,
            None,
            "mean",
        ),
    ],
)
def test_parse_description(desc, kind, var, top, bottom, stat):
    info = inp.parse_description(desc)
    assert info["kind"] == kind
    assert info["var"] == var
    assert info["top"] == top
    assert info["bottom"] == bottom
    assert info["stat"] == stat


def test_parse_description_unknown():
    assert inp.parse_description("Band 1") is None
    assert inp.parse_description("BDTICM Absolute depth to bedrock (cm)") is None


def test_intervals_to_layer_thickness_weighted():
    vals = {
        (0, 5): np.array([10.0]),
        (5, 15): np.array([20.0]),
        (15, 30): np.array([40.0]),
    }
    out = inp.intervals_to_layer(vals, 0, 30)
    assert out[0] == pytest.approx((5 * 10 + 10 * 20 + 15 * 40) / 30)
    assert inp.intervals_to_layer({(0, 5): np.array([1.0])}, 0, 30) is None
    # 30-60 from 30-60 only (other intervals have zero weight)
    vals[(30, 60)] = np.array([7.0])
    assert inp.intervals_to_layer(vals, 30, 60)[0] == pytest.approx(7.0)


def test_points_to_layer_trapezoid():
    vals = {0: np.array([10.0]), 15: np.array([20.0]), 30: np.array([40.0])}
    out = inp.points_to_layer(vals, 0, 30)
    assert out[0] == pytest.approx((15 * 15 + 15 * 30) / 30)
    assert inp.points_to_layer({0: np.array([1.0])}, 0, 30) is None


def test_dist_from_quantiles_and_p16_p84():
    d = inp.dist_from_quantiles(
        "clay", np.array([20.0]), np.array([10.0]), np.array([40.0])
    )
    assert d.sig_lo[0] == pytest.approx(math.log(2) / inp.Z90)
    assert d.sig_hi[0] == pytest.approx(math.log(2) / inp.Z90)
    d = inp.dist_from_p16_p84("bdod", np.array([1.4]), np.array([1.3]), np.array([1.5]))
    assert d.sig_lo[0] == pytest.approx(0.2 / (2 * inp.Z68))
    d = inp.dist_from_quantiles("clay", np.array([20.0, np.nan]))
    assert d.sig_lo[0] == 0.0 and np.isnan(d.sig_lo[1])


# ----------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------


def _cells(system, n=50, sand=65.0, silt=25.0, clay=10.0, width=0.2):
    def d(var, c):
        c = np.full(n, c)
        return inp.dist_from_quantiles(var, c, c * (1 - width), c * (1 + width))

    dists = {
        "sand": d("sand", sand),
        "silt": d("silt", silt),
        "clay": d("clay", clay),
        "soc": d("soc", 10.0),
        "bdod": d("bdod", 1.4),
    }
    return unc.ProductCells(system, system, dists)


def test_sample_inputs_are_real_soils():
    rng = np.random.default_rng(0)
    s = unc.sample_inputs(_cells("USDA"), 300, rng, 1.724)
    total = s["sand"] + s["silt"] + s["clay"]
    assert np.allclose(total, 100.0)
    assert (s["sand"] > 0).all() and (s["clay"] > 0).all()
    assert np.median(s["om"]) == pytest.approx(10.0 / 10 * 1.724, rel=0.05)
    assert np.isnan(s["gravel"]).all()  # no cfvo given


def test_sample_inputs_iso_converted_to_usda():
    rng = np.random.default_rng(0)
    iso = unc.sample_inputs(_cells("ISO 11277", width=0.0), 5, rng, 1.724)
    assert np.allclose(iso["silt"], 25.0 * ISO_TO_USDA_SILT_FACTOR)
    assert np.allclose(iso["sand"] + iso["silt"] + iso["clay"], 100.0)


def test_variance_split_known_case():
    # 1 method, 2 products with means 0 and 2, within variance 1 -> product 1
    rng = np.random.default_rng(1)
    g = np.empty((1, 2, 20000, 1))
    g[0, 0, :, 0] = rng.standard_normal(20000)
    g[0, 1, :, 0] = rng.standard_normal(20000) + 2.0
    v = unc.variance_split(g)
    assert v["input"][0] == pytest.approx(1.0, rel=0.03)
    assert v["product"][0] == pytest.approx(1.0, rel=0.03)
    assert v["method"][0] == pytest.approx(0.0)
    assert v["total"][0] == pytest.approx(2.0, rel=0.03)


def test_run_layer_reproducible_and_ordered():
    settings = unc.Settings(draws=100)
    a = unc.run_layer([_cells("USDA")], settings, np.random.default_rng(7))
    b = unc.run_layer([_cells("USDA")], settings, np.random.default_rng(7))
    assert np.array_equal(a.stats["ksat"], b.stats["ksat"])
    p5, p50, p95 = a.stats["ksat"]
    assert (p5 <= p50).all() and (p50 <= p95).all()
    assert (a.texture_class == 3).all()  # sandy loam
    # Narrow inputs -> top confidence classes; wide inputs -> lower
    from mayim_tools.soil.regional_parameterisation import core

    narrow = unc.run_layer(
        [_cells("USDA", width=0.01)], settings, np.random.default_rng(7)
    )
    f = core.ksat_factor(narrow.stats["ksat"][0], narrow.stats["ksat"][2])
    assert (core.ksat_class(f) == 1).all()
    assert (core.texture_conf_class(narrow.texture_share) == 1).all()
    fw = core.ksat_factor(a.stats["ksat"][0], a.stats["ksat"][2])
    assert (core.ksat_class(fw) >= 1).all() and np.nanmedian(fw) > np.nanmedian(f)


def test_confidence_classes():
    from mayim_tools.soil.regional_parameterisation import core

    f = core.ksat_factor(
        np.array([1.0, 1.0, 1.0, 1.0, np.nan]), np.array([3.9, 16.0, 99.0, 101.0, 1.0])
    )
    assert core.ksat_class(f).tolist() == [1, 2, 3, 4, 0]
    sh = np.array([0.95, 0.8, 0.7, 0.5, 0.2, np.nan])
    assert core.texture_conf_class(sh).tolist() == [1, 1, 2, 3, 4, 0]


def test_rawls_lookup_units():
    out = rawls_1983.lookup(np.array([4, 6, 0]))
    assert out["k"][0] == pytest.approx(3.4)  # loam 0.34 cm/h
    assert out["psi_f"][0] == pytest.approx(88.9)
    assert np.isnan(out["k"][1]) and np.isnan(out["k"][2])  # silt, none


# ----------------------------------------------------------------------
# Gap filling
# ----------------------------------------------------------------------


def _product(name, system, clay, sand=50.0, silt=30.0):
    p = inp.ProductInputs(name, "", system, "")
    shape = clay.shape

    def d(var, c):
        c = np.where(np.isfinite(clay), c, np.nan) if var != "clay" else clay
        return inp.dist_from_quantiles(var, c, c * 0.8, c * 1.2)

    p.layers = {
        "0-30cm": {
            "sand": d("sand", np.full(shape, sand)),
            "silt": d("silt", np.full(shape, silt)),
            "clay": d("clay", clay),
            "soc": d("soc", np.full(shape, 8.0)),
        }
    }
    return p


def test_fill_order_other_then_sg2017_then_neighbour():
    layers = (("0-30cm", 0, 30),)
    clay = np.array([[20.0, np.nan, np.nan, np.nan]])
    sg = _product(inp.PRODUCT_SG, "USDA", clay)
    olm_clay = np.array([[30.0, 31.0, np.nan, np.nan]])
    olm = _product(inp.PRODUCT_OLM, "ISO 11277", olm_clay)
    sg.sg2017 = {"0-30cm": {"clay": np.array([[25.0, 25.0, 26.0, np.nan]])}}

    def fake_neighbour(arr, max_cells):
        return np.where(np.isfinite(arr), arr, 99.0)

    src = fill_mod.fill_product(sg, olm, 2.0, layers, fill_fn=fake_neighbour)
    out = sg.layers["0-30cm"]["clay"].centre
    assert out.tolist() == [[20.0, 31.0, 26.0, 99.0]]
    assert src["0-30cm"].tolist() == [[1, 2, 3, 4]]


def test_fill_converts_texture_between_systems():
    layers = (("0-30cm", 0, 30),)
    sg = _product(inp.PRODUCT_SG, "USDA", np.array([[np.nan]]))
    olm = _product(inp.PRODUCT_OLM, "ISO 11277", np.array([[20.0]]), sand=50, silt=30)
    fill_mod.fill_product(sg, olm, 0.0, layers)
    k = ISO_TO_USDA_SILT_FACTOR
    assert sg.layers["0-30cm"]["silt"].centre[0, 0] == pytest.approx(30 * k)
    assert sg.layers["0-30cm"]["sand"].centre[0, 0] == pytest.approx(50 + 30 * (1 - k))


# ----------------------------------------------------------------------
# End to end (synthetic extraction folders; needs GDAL)
# ----------------------------------------------------------------------


def _write_folders(base):
    pytest.importorskip("osgeo.gdal")
    from osgeo import osr

    from mayim_tools.soil._common.export import write_multiband_geotiff
    from mayim_tools.soil._common.grid import TargetGrid

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32735)
    wkt = srs.ExportToWkt()
    x0, y0, size = 500000.0, 7150000.0, 1500.0

    def grid(res):
        return TargetGrid(x0, y0, x0 + size, y0 + size, res, wkt)

    def field(g, mean, amp):
        yy, xx = np.mgrid[0 : g.height, 0 : g.width]
        return mean + amp * np.sin(xx / 3.0) * np.cos(yy / 4.0)

    sg_dir, olm_dir = os.path.join(base, "sg"), os.path.join(base, "olm")
    os.makedirs(sg_dir)
    os.makedirs(olm_dir)
    g250 = grid(250.0)
    sg_depths = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]
    means = {
        "sand": 60,
        "silt": 20,
        "clay": 20,
        "soc": 8,
        "bdod": 1.4,
        "cfvo": 5,
        "phh2o": 6.5,
        "cec": 12,
    }
    units = {
        "soc": "g/kg",
        "bdod": "g/cm3",
        "cfvo": "vol %",
        "phh2o": "pH",
        "cec": "cmol(c)/kg",
    }
    for var, m in means.items():
        for stat, tag, f in (
            ("Q0.05", "Q0.05", 0.75),
            ("Q0.5", "Q0.50", 1.0),
            ("Q0.95", "Q0.95", 1.3),
        ):
            if var == "bdod":
                f = 1.0 + (f - 1.0) * 0.1
            bands = []
            for top, bot in sg_depths:
                v = field(g250, m, m * 0.1) * f
                v[0, 0] = np.nan  # masked (e.g. built-up) cell
                bands.append((f"{var}_{top}-{bot}cm_{stat} ({units.get(var, '%')})", v))
            write_multiband_geotiff(
                os.path.join(sg_dir, f"soilgrids_{var}_{tag}.tif"), g250, bands, "%"
            )
    for code, m in (("wv0033", 25), ("wv1500", 13)):
        bands = [
            (f"{code}_{t}-{b}cm_Q0.5 (vol %)", field(g250, m, 2)) for t, b in sg_depths
        ]
        write_multiband_geotiff(
            os.path.join(sg_dir, f"soilgrids_{code}_Q0.50.tif"), g250, bands, "vol %"
        )
    g30, g120 = grid(30.0), grid(120.0)
    olm_means = {"sand": 45, "silt": 23, "clay": 32, "soc": 9, "bdod": 1.35}
    for var, m in olm_means.items():
        for tag, g, f in (
            ("mean_30m", g30, 1.0),
            ("p16_120m", g120, 0.85),
            ("p84_120m", g120, 1.15),
        ):
            bands = [
                (
                    f"{var}_{t}-{b}cm_{tag}_2020-2022 ({units.get(var, '%')})",
                    field(g, m, m * 0.1) * f,
                )
                for t, b in ((0, 30), (30, 60), (60, 100))
            ]
            write_multiband_geotiff(
                os.path.join(olm_dir, f"olm_{var}_{tag}_2020-2022.tif"), g, bands, "%"
            )
    for name, m in (("field_capacity_33kPa", 27), ("wilting_point_1500kPa", 15)):
        bands = [
            (f"{name}_{p}cm_250m_1950-2017 (vol %)", field(g250, m, 2))
            for p in (0, 30, 60, 100)
        ]
        write_multiband_geotiff(
            os.path.join(olm_dir, f"olm_{name}_250m_1950-2017.tif"),
            g250,
            bands,
            "vol %",
        )
    with open(
        os.path.join(sg_dir, "soilgrids_metadata.csv"), "w", encoding="utf-8"
    ) as fh:
        fh.write(
            "# Run\nItem,Value\nTool,Extract: SoilGrids 2.0\nAccess date,2026-10-05\n\n"
        )
    return sg_dir, olm_dir


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    return _write_folders(str(tmp_path_factory.mktemp("rsp_inputs")))


def _run(tmp_path, folders, **kw):
    from mayim_tools.soil.regional_parameterisation import core

    sg, olm = folders
    s = core.RunSettings(
        out_dir=str(tmp_path),
        sg_folder=kw.pop("sg", sg),
        olm_folder=kw.pop("olm", olm),
        write_report=kw.pop("report", False),
        **kw,
    )
    s.mc.draws = 40
    return core.run(s)


def _bands(path):
    from osgeo import gdal

    ds = gdal.Open(path)
    return [ds.GetRasterBand(i).GetDescription() for i in range(1, ds.RasterCount + 1)]


def test_end_to_end_both_products(tmp_path, folders):
    r = _run(tmp_path, folders)
    assert r.grid.res == 30.0  # finest input grid
    names = sorted(os.path.basename(f) for f, _ in r.files)
    for expected in (
        "rsp_ksat.tif",
        "rsp_psi_f.tif",
        "rsp_theta_s.tif",
        "rsp_texture_class.tif",
        "rsp_quality.tif",
        "rsp_inputs_P50.tif",
        "rsp_product_difference.tif",
        "rsp_zone_summary.csv",
        "rsp_metadata.csv",
    ):
        assert expected in names
    bands = _bands(str(tmp_path / "rsp_ksat.tif"))
    assert bands[0] == "ksat_0-30cm_P50 (mm/h)"
    assert len(bands) == 9 and bands[3] == "ksat_0-30cm_P05 (mm/h)"
    assert {row["Layer"] for row in r.zone_rows} == {"0-30cm", "30-60cm", "60-100cm"}
    ks = [
        x for x in r.zone_rows if x["Parameter"] == "ksat" and x["Layer"] == "0-30cm"
    ][0]
    assert ks["Median cell P5"] < ks["Median of P50"] < ks["Median cell P95"]
    shares = sum(ks[f"Share {k}"] for k in unc.VARIANCE_PARTS)
    assert shares == pytest.approx(1.0)
    assert ks["Share product"] > 0  # two products disagree
    assert r.comparison and r.checks and r.reference_rows
    # The masked SoilGrids cell was filled from OpenLandMap
    src = r.sources[inp.PRODUCT_SG]["0-30cm"]
    assert (src == fill_mod.SOURCE_OTHER).any()


def test_end_to_end_reproducible_across_workers(tmp_path, folders):
    a = _run(tmp_path / "a", folders, workers=1)
    b = _run(tmp_path / "b", folders, workers=3)
    assert np.array_equal(
        a.stats["ksat"]["0-30cm"], b.stats["ksat"]["0-30cm"], equal_nan=True
    )


def test_end_to_end_single_product_and_zones(tmp_path, folders):
    r = _run(
        tmp_path,
        folders,
        olm="",
        zones=[
            (
                "Z1",
                "POLYGON((500100 7150100,500900 7150100,500900 7150900,500100 7150900,500100 7150100))",
            )
        ],
    )
    assert r.grid.res == 250.0
    assert r.zone_names == ["Z1"]
    ks = [x for x in r.zone_rows if x["Parameter"] == "ksat"][0]
    assert ks["Share product"] == pytest.approx(0.0)
    # only cells inside the zone are processed
    outside = r.zone_raster == 0
    assert np.isnan(r.stats["ksat"]["0-30cm"][1][outside]).all()


def test_end_to_end_report(tmp_path, folders):
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    r = _run(tmp_path, folders, report=True)
    assert r.report_path and os.path.getsize(r.report_path) > 20000


def test_wrong_folder_and_missing_inputs(tmp_path, folders):
    sg, olm = folders
    with pytest.raises(SoilDataError, match="contains OpenLandMap"):
        _run(tmp_path, folders, sg=olm, olm="")
    with pytest.raises(SoilDataError, match="Give the output folder"):
        _run(tmp_path, folders, sg="", olm="")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SoilDataError, match="No SoilGrids or OpenLandMap"):
        _run(tmp_path, folders, sg=str(empty), olm="")


# ----------------------------------------------------------------------
# Tóth et al. (2015) / HiHydroSoil and van Genuchten helpers
# ----------------------------------------------------------------------

from mayim_tools.soil.regional_parameterisation.ptf import (  # noqa: E402
    AVAILABLE_METHODS,
    METHOD_BY_CODE,
    active_parameters,
    toth2015,
)
from mayim_tools.soil.regional_parameterisation.ptf import (  # noqa: E402
    van_genuchten as vg,
)


@pytest.mark.parametrize(
    "alpha,n",
    [
        (0.145, 2.68),
        (0.124, 2.28),
        (0.075, 1.89),
        (0.036, 1.56),
        (0.02, 1.41),
        (0.01, 1.23),
        (0.05, 1.3),
    ],
)
def test_capillary_drive_matches_numerical_integral(alpha, n):
    # Integral of the Mualem Kr(h) over h from 0 to infinity (log-spaced
    # trapezoid; the tail beyond 1e7 cm is negligible).
    h = np.concatenate([[0.0], np.logspace(-6, 7, 200001)])
    kr = vg.mualem_kr(h, alpha, n)
    numeric = float(np.sum((kr[1:] + kr[:-1]) / 2 * np.diff(h)))
    assert float(vg.capillary_drive_cm(alpha, n)) == pytest.approx(numeric, rel=0.02)


def test_theta_at_limits():
    assert float(vg.theta_at(0.0, 0.05, 0.45, 0.03, 1.5)) == pytest.approx(0.45)
    assert float(vg.theta_at(1e12, 0.05, 0.45, 0.03, 1.5)) == pytest.approx(
        0.05, abs=1e-4
    )
    assert vg.H33_CM == pytest.approx(336.5, abs=0.1)


def test_toth2015_hand_calculation():
    # Equations as printed in the HiHydroSoil v2.0 report (pp. 7-8), typed
    # out independently of the coefficient table.
    bd, cl, si, oc, ph, cec = 1.4, 20.0, 20.0, 1.0, 6.5, 12.0
    r = toth2015.toth2015(60.0, si, cl, oc, bd, True, ph=ph, cec=cec)
    ths = 0.83080 - 0.28217 * bd + 0.0002728 * cl + 0.000187 * si
    la = (
        -0.43348
        - 0.41729 * bd
        - 0.04762 * oc
        + 0.21810 * 1
        - 0.01581 * cl
        - 0.01207 * si
    )
    ln1 = (
        0.22236
        - 0.30189 * bd
        - 0.05558 * 1
        - 0.005306 * cl
        - 0.003084 * si
        - 0.01072 * oc
    )
    lk = (
        0.40220
        + 0.26122 * ph
        + 0.44565 * 1
        - 0.02329 * cl
        - 0.01265 * si
        - 0.01038 * cec
    )
    assert float(r["theta_s"]) == pytest.approx(ths)
    assert float(r["alpha"]) == pytest.approx(10**la)
    assert float(r["n_vg"]) == pytest.approx(1 + 10**ln1)
    assert float(r["ksat"]) == pytest.approx(10**lk * 10 / 24)  # cm/d -> mm/h
    assert float(r["theta_r"]) == 0.041
    t33 = vg.theta_at(vg.H33_CM, 0.041, ths, 10**la, 1 + 10**ln1)
    assert float(r["theta_fc"]) == pytest.approx(float(t33))
    assert 0.041 < float(r["theta_wp"]) < float(r["theta_fc"]) < ths
    assert int(r["flags"]) == 0


def test_toth2015_topsoil_flag_and_theta_r_tree():
    top = toth2015.toth2015(60.0, 20.0, 20.0, 1.0, 1.4, True, ph=6.5, cec=12.0)
    sub = toth2015.toth2015(60.0, 20.0, 20.0, 1.0, 1.4, False, ph=6.5, cec=12.0)
    assert float(top["alpha"]) / float(sub["alpha"]) == pytest.approx(10**0.21810)
    assert float(top["ksat"]) / float(sub["ksat"]) == pytest.approx(10**0.44565)
    assert float(toth2015.toth2015(1.0, 40.0, 59.0, 1.0, 1.2, True)["theta_r"]) == 0.179


def test_toth2015_without_ph_cec_flags_ksat():
    r = toth2015.toth2015(np.array([60.0]), 20.0, 20.0, 1.0, 1.4, True)
    assert np.isnan(r["ksat"]).all()
    assert int(r["flags"][0]) & toth2015.FLAG_NO_KSAT_INPUTS
    assert np.isfinite(r["theta_fc"]).all()


def test_method_registry():
    codes = [m.code for m in AVAILABLE_METHODS]
    assert codes == ["SR2006", "TOTH2015"]
    params = {p.code for p in active_parameters(["SR2006"])}
    assert "alpha" not in params and "lambda" in params
    params = {p.code for p in active_parameters(["TOTH2015"])}
    assert "alpha" in params and "lambda" not in params


def test_run_layer_two_methods_split_and_missing_ksat():
    settings = unc.Settings(draws=100, methods=("SR2006", "TOTH2015"))
    a = unc.run_layer(
        [_cells("USDA")], settings, np.random.default_rng(1), topsoil=True
    )
    assert "alpha" in a.stats and "lambda" in a.stats
    # Without pH/CEC Tóth Ksat is left out: pooled Ksat equals S&R alone
    sr = unc.run_layer(
        [_cells("USDA")],
        unc.Settings(draws=100, methods=("SR2006",)),
        np.random.default_rng(1),
        topsoil=True,
    )
    assert np.allclose(a.stats["ksat"], sr.stats["ksat"])
    assert np.nanmean(a.variance["psi_f"]["method"]) > 0


def test_end_to_end_methods(tmp_path, folders):
    r = _run(tmp_path / "both", folders)
    assert r.settings.mc.methods == ("SR2006", "TOTH2015")
    names = {os.path.basename(f) for f, _ in r.files}
    assert {"rsp_alpha.tif", "rsp_n_vg.tif", "rsp_lambda.tif"} <= names
    ks = [
        x for x in r.zone_rows if x["Parameter"] == "ksat" and x["Layer"] == "0-30cm"
    ][0]
    assert ks["Share method"] > 0  # Tóth Ksat computed (pH and CEC parsed)
    assert np.isfinite(ks["SG central (TOTH2015)"])
    one = _run(tmp_path / "toth", folders, olm="")
    one.settings.mc.methods  # noqa: B018
    from mayim_tools.soil.regional_parameterisation import core

    s = core.RunSettings(
        out_dir=str(tmp_path / "x"), sg_folder=folders[0], write_report=False
    )
    s.mc.methods = ()
    with pytest.raises(SoilDataError, match="at least one method"):
        core.run(s)
