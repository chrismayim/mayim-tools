"""Run-off Coefficients (SANRAL Deterministic): tables, per-cell classes, the catchment
arithmetic (SANRAL Application Guide example 3.1) and a synthetic GDAL run."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from mayim_tools.land_cover.rational_c import cells as cl
from mayim_tools.land_cover.rational_c import coefficients as co
from mayim_tools.land_cover.rational_c import lookup as lk
from mayim_tools.land_cover.rational_c import tables as tb

# ---------------------------------------------------------------- tables


def test_table_3_7_values():
    assert tb.CS[0] == (0.01, 0.03, 0.05) and tb.CS[3] == (0.22, 0.26, 0.30)
    assert tb.CP[2] == (0.12, 0.16, 0.20) and tb.CP[3] == (0.21, 0.26, 0.30)
    assert tb.CV[1] == (0.07, 0.11, 0.15) and tb.CV[3] == (0.26, 0.28, 0.30)
    assert tb.URBAN_ROWS["streets"][1:] == (0.70, 0.95)
    assert tb.DOLOMITE_FACTORS == (0.10, 0.20, 0.35, 0.50)
    assert tb.FT["flat_permeable"] == (0.50, 0.55, 0.60, 0.67, 0.83, 1.00)
    assert tb.FT["steep_impermeable"] == (0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


@pytest.mark.parametrize(
    "mm,band", [(599.9, 0), (600, 1), (900, 1), (900.1, 2), (300, 0), (1500, 2)]
)
def test_map_band_edges(mm, band):
    assert tb.map_band(mm) == band


def test_map_band_rejects_nonsense():
    with pytest.raises(ValueError):
        tb.map_band(0)


def test_urban_position():
    assert tb.urban_value("houses", "upper") == 0.50
    assert tb.urban_value("houses", "lower") == 0.30
    assert tb.urban_value("houses", "middle") == pytest.approx(0.40)


def test_neighbour_band():
    assert co.neighbour_band(560) == 1
    assert co.neighbour_band(640) == 0
    assert co.neighbour_band(880) == 2
    assert co.neighbour_band(950) == 1
    assert co.neighbour_band(746.6) is None


# ---------------------------------------------------------------- cells


def test_slope_class_edges():
    s = np.array([0, 2.99, 3.0, 9.99, 10.0, 29.9, 30.0, 80, np.nan])
    assert cl.slope_class(s).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, -1]


def test_perm_from_hsg_codes():
    codes = np.array([1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 9, np.nan])
    assert cl.perm2_from_hsg(codes).tolist() == [0, 2, 4, 6, 6, 6, 6, 1, 3, 5, -1, -1]


def test_perm_shares_split_intermediate():
    sh = cl.perm_shares(np.array([0, 1, 2, 3, -1]))
    assert sh.tolist() == pytest.approx([0.375, 0.5, 0.125, 0.0])
    assert sum(sh) == pytest.approx(1)


def test_cp_value_intermediate_is_average():
    v = cl.cp_value(np.array([1, 5]), 1)
    assert v.tolist() == pytest.approx([0.06, 0.21])


def test_perm_quantiles():
    prob = np.zeros((4, 1, 3))
    prob[:, 0, 0] = [2, 90, 6, 2]  # P05 B, P95 C
    prob[:, 0, 1] = [10, 80, 10, 0]  # P05 A, P95 C
    prob[:, 0, 2] = np.nan
    design = np.array([[2, 2, 4]], dtype=np.int8)
    lo, hi = cl.perm2_quantiles(prob, design)
    assert lo.tolist() == [[2, 0, 4]]
    assert hi.tolist() == [[4, 4, 4]]


def test_lawn_interpolation():
    row = np.array([cl.LAWN_IDX] * 4)
    slope = np.array([1.0, 4.5, 8.0, 1.0])
    perm = np.array([2, 2, 2, 4])
    d, lo, hi = cl.urban_c2(row, slope, perm, "upper")
    assert d.tolist() == pytest.approx([0.10, 0.15, 0.20, 0.17])
    assert lo.tolist() == pytest.approx([0.05, 0.10, 0.15, 0.13])


def test_classify_roles_and_overrides():
    look = lk.sanlc_lookup()
    lc = np.array([[13, 47, 19, 18], [25, 0, 61, np.nan]])
    slope = np.full(lc.shape, 5.0)
    hsg = np.full(lc.shape, 2.0)
    out = cl.classify(lc, slope, hsg, look, cl.CellOptions(1))
    assert out["role"].tolist() == [[1, 2, 3, 4], [1, 0, 2, 0]]
    assert out["perm"][1, 0] == 6  # rock forced impermeable
    assert out["veg"][0, 0] == lk.GR
    assert out["c2"][0, 1] == pytest.approx(0.50)
    assert out["c1"][0, 0] == pytest.approx(0.08 + 0.08 + 0.21)
    ov = {
        "cv": np.array([[0, -1, -1, -1], [-1, -1, -1, -1]], dtype=np.int8),
        "urban": np.full(lc.shape, -1, dtype=np.int16),
        "c2": np.full(lc.shape, np.nan),
    }
    ov["c2"][0, 1] = 0.33
    out = cl.classify(lc, slope, hsg, look, cl.CellOptions(1), override=ov)
    assert out["veg"][0, 0] == lk.TB
    assert out["c2"][0, 1] == pytest.approx(0.33)


def test_wetlands_impermeable_option():
    look = lk.sanlc_lookup()
    lc = np.array([[22.0, 13.0]])
    out = cl.classify(
        lc, np.ones((1, 2)), np.ones((1, 2)), look, cl.CellOptions(1, "upper", True)
    )
    assert out["perm"].tolist() == [[6, 0]]


# ---------------------------------------------------------------- lookup


def test_sanlc_lookup_complete():
    look = lk.sanlc_lookup()
    assert sorted(look.rules) == list(range(1, 74))
    roles = {r.role for r in look.rules.values()}
    assert roles == set(lk.ROLES)
    assert {c for c, r in look.rules.items() if r.role == lk.WATER} == {14, 15, 17, 19}
    assert {c for c, r in look.rules.items() if r.role == lk.INEFFECTIVE} == {
        16,
        18,
        20,
        21,
        26,
    }
    for r in look.rules.values():
        if r.role == lk.RURAL:
            assert r.cv_low <= r.cv <= r.cv_high


def test_lookup_csv_round_trip(tmp_path):
    p = tmp_path / "lookup.csv"
    lk.write_lookup_csv(str(p), lk.sanlc_lookup())
    back = lk.read_lookup_csv(str(p))
    for code, r in lk.sanlc_lookup().rules.items():
        b = back.rules[code]
        assert (b.role, b.cv, b.cv_low, b.cv_high, b.urban) == (
            r.role,
            r.cv,
            r.cv_low,
            r.cv_high,
            r.urban,
        )


def test_lookup_csv_errors(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("code,role,cv_design\n1,forest,\n")
    with pytest.raises(lk.CLookupError):
        lk.read_lookup_csv(str(p))


# ---------------------------------------------------------------- AG example 3.1

AG = co.Shares((0.2, 0.7, 0.1, 0.0), (0.0, 0.5, 0.5, 0.0), (0.0, 0.45, 0.5, 0.05))
AG_C2 = (
    0.2 * 0.10 + 0.1 * 0.20 + 0.4 * 0.50 + 0.05 * 0.70 + 0.05 * 0.80 + 0.1 * 0.70
) + 0.1 * 0.95


def test_application_guide_example_3_1():
    assert AG_C2 == pytest.approx(0.48)
    r = co.catchment_c(
        AG, 0.4, 0.6, 0.0, 746.6, (AG_C2,) * 3, d_fraction=0.10,
        ft_override="flat_permeable",
    )  # fmt: skip
    assert r.c1 == pytest.approx(0.3665)
    assert r.d_factor == pytest.approx(0.195)
    assert round(r.c1d, 3) == 0.337
    assert r.c_t[20] == pytest.approx(0.3783, abs=5e-5)
    assert r.c_t[50] == pytest.approx(0.3999, abs=5e-5)
    # the score agrees with the guide's choice of flat and permeable
    assert r.ft_score == pytest.approx(0.3)
    assert tb.ft_class_from_score(r.ft_score) == "flat_permeable"


def test_pmf_and_alpha_beta_gamma():
    r = co.catchment_c(AG, 0.5, 0.3, 0.2, 746.6, (0.5, 0.4, 0.6), c3=1.0)
    assert r.alpha + r.beta + r.gamma == pytest.approx(1)
    c1p = 0.2 * 0.03 + 0.7 * 0.08 + 0.1 * 0.16 + 0.26 + 0.28
    assert r.c_t["PMF"] == pytest.approx(0.5 * c1p + 0.3 + 0.2)
    assert r.c_t[100] == pytest.approx(0.5 * r.c1d + 0.3 * 0.5 + 0.2)


def test_uncertainty_root_sum_square():
    sh = co.Shares(
        (0, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0),
        perm_lo=(1, 0, 0, 0), perm_hi=(0, 0, 1, 0),
        veg_lo=(1, 0, 0, 0), veg_hi=(0, 0, 1, 0),
    )  # fmt: skip
    r = co.catchment_c(sh, 1, 0, 0, 700)
    dlo = math.hypot(0.08 - 0.04, 0.11 - 0.04)
    dhi = math.hypot(0.16 - 0.08, 0.21 - 0.11)
    assert r.c1_p05 == pytest.approx(r.c1 - dlo)
    assert r.c1_p95 == pytest.approx(r.c1 + dhi)
    assert 0 < r.soil_share < 1


def test_ft_borderline_gives_both_rows():
    sh = co.Shares((0, 0.5, 0.5, 0), (0, 0.5, 0.5, 0), (0, 1, 0, 0))
    r = co.catchment_c(sh, 1, 0, 0, 700)
    assert r.ft_score == pytest.approx(0.5)
    assert r.ft_class == "steep_impermeable" and r.ft_borderline
    assert r.c_t_other_ft[20] == pytest.approx(r.c1d * 0.67)


def test_override_wins():
    sh = co.Shares((0, 0, 0, 1), (0, 0, 0, 1), (0, 1, 0, 0))
    r = co.catchment_c(sh, 1, 0, 0, 700, ft_override="flat_permeable")
    assert r.ft_class == "flat_permeable" and r.ft_source == "override"


def test_ft_on_c2_option():
    r = co.catchment_c(None, 0, 1, 0, 700, (0.5, 0.5, 0.5), ft_on_c2=True)
    assert r.c_t[2] == pytest.approx(0.5 * 0.50)
    r = co.catchment_c(None, 0, 1, 0, 700, (0.5, 0.5, 0.5))
    assert r.c_t[2] == pytest.approx(0.5)


# ---------------------------------------------------------------- GDAL run

EPSG = 32735


def _tif(path, arr, x0, y0, res, nodata, dtype, descs=None):
    from osgeo import gdal, osr

    gdal.UseExceptions()
    bands = arr[None] if arr.ndim == 2 else arr
    nb, h, w = bands.shape
    ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, nb, dtype)
    ds.SetGeoTransform((x0, res, 0, y0 + h * res, 0, -res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    ds.SetProjection(srs.ExportToWkt())
    for i in range(nb):
        b = ds.GetRasterBand(i + 1)
        if nodata is not None:
            b.SetNoDataValue(nodata)
        if descs:
            b.SetDescription(descs[i])
        b.WriteArray(bands[i])
    ds = None
    return str(path), srs.ExportToWkt()


def _box(x0, y0, x1, y1):
    return f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"


@pytest.fixture
def synthetic(tmp_path):
    pytest.importorskip("osgeo.gdal")
    from osgeo import gdal

    x0, y0 = 500000.0, 7000000.0
    # DEM 10 m, 200 x 200 (2 km), slope 5 % in x (flat class)
    xx = np.arange(200) * 10.0
    dem = np.tile(0.05 * xx, (200, 1)).astype(np.float32)
    dem_p, wkt = _tif(tmp_path / "dem.tif", dem, x0, y0, 10, -9999, gdal.GDT_Float32)
    # land cover 20 m, 100 x 100: west half grassland 13, east half: houses 47,
    # a dam 19 strip and a dry pan 26 block
    lc = np.full((100, 100), 13, dtype=np.uint8)
    lc[:, 50:] = 47
    lc[0:10, 0:50] = 19
    lc[90:100, 0:10] = 26
    lc_p, _ = _tif(tmp_path / "lc.tif", lc, x0, y0, 20, 0, gdal.GDT_Byte)
    hsg = np.full((40, 40), 2, dtype=np.uint8)  # 50 m, all B
    hsg_p, _ = _tif(
        tmp_path / "hsg_neh630_recommended.tif",
        hsg,
        x0,
        y0,
        50,
        0,
        gdal.GDT_Byte,
        ["hsg_neh630_recommended (code)"],
    )
    prob = np.zeros((4, 40, 40), dtype=np.float32)
    prob[0], prob[1], prob[2] = 3, 90, 7  # P05 B, P95 C
    _tif(
        tmp_path / "hsg_neh630_probability.tif",
        prob,
        x0,
        y0,
        50,
        -9999,
        gdal.GDT_Float32,
        [f"hsg_neh630_probability_{g} (%)" for g in "ABCD"],
    )
    return tmp_path, wkt, dem_p, lc_p, hsg_p, (x0, y0)


def test_run_synthetic(synthetic):
    from mayim_tools.land_cover.rational_c import core

    d, wkt, dem_p, lc_p, hsg_p, (x0, y0) = synthetic
    cats = [
        core.Catchment("W", _box(x0, y0, x0 + 1000, y0 + 2000)),
        core.Catchment("ALL", _box(x0, y0, x0 + 2000, y0 + 2000), map_mm=950),
    ]
    rasters = {k: str(d / f"{k}.tif") for k in core.RASTERS}
    s = core.RunSettings(
        cats, wkt, dem_p, lc_p, hsg_p, 700.0, rasters=rasters,
        dolomite=[_box(x0, y0 + 1000, x0 + 1000, y0 + 2000)],
    )  # fmt: skip
    r = core.run(s)
    w, a = r.catchments
    assert w.area_km2 == pytest.approx(2.0)
    # west: 0.2 km2 dam, 0.04 km2 pan (ineffective), rest grassland
    assert w.a_ineff_km2 == pytest.approx(0.04, abs=0.005)
    k = w.coeff
    assert k.gamma == pytest.approx(0.2 / 1.96, abs=0.01)
    assert k.beta == 0
    assert w.shares.slope[1] == pytest.approx(1.0, abs=0.02)
    assert w.shares.perm == pytest.approx((0, 1, 0, 0))
    assert w.shares.perm_hi == pytest.approx((0, 0, 1, 0))
    assert w.shares.veg == pytest.approx((0, 0, 1, 0))
    assert k.c1 == pytest.approx(0.08 + 0.08 + 0.21, abs=0.002)
    assert k.c1_p95 > k.c1 == pytest.approx(k.c1_p05)  # grassland, B: no lower class
    assert w.d_fraction == pytest.approx(2000 / 4400, abs=0.01)  # dam in the north half
    assert k.c1d < k.c1
    assert any("open water" in f for f in w.flags)
    assert any("excluded" in f for f in w.flags)
    assert w.slope_native is not None and w.slope_native[1] == pytest.approx(
        1, abs=0.03
    )
    # whole area: half urban (houses, upper end 0.50), MAP band > 900
    k = a.coeff
    assert k.band == 2
    assert k.beta == pytest.approx(0.5 / 0.98, abs=0.01)
    assert k.c2 == pytest.approx(0.50)
    assert k.c2_p05 == pytest.approx(0.30)
    from osgeo import gdal

    ds = gdal.Open(rasters["role"])
    arr = ds.GetRasterBand(1).ReadAsArray()
    assert set(np.unique(arr)) <= {1, 2, 3, 4, 255}
    assert r.unknown == {}


def test_run_rejects_geographic(synthetic):
    from osgeo import osr

    from mayim_tools.land_cover.rational_c import core

    d, wkt, dem_p, lc_p, hsg_p, _ = synthetic
    g = osr.SpatialReference()
    g.ImportFromEPSG(4326)
    s = core.RunSettings(
        [core.Catchment("A", _box(0, 0, 1, 1))],
        g.ExportToWkt(),
        dem_p,
        lc_p,
        hsg_p,
        700,
    )
    with pytest.raises(core.RationalCError):
        core.run(s)


def test_field_names_fit_shapefile():
    from mayim_tools.land_cover.rational_c import export as ex

    assert all(len(f) <= 10 for f in ex.FIELDS)
    assert len(set(f.lower() for f in ex.FIELDS)) == len(ex.FIELDS)


def test_c_csv_round_trip_and_results_csv(synthetic):
    from mayim_tools.hydrology.sanral_peak_flows.inputs import read_c_csv
    from mayim_tools.land_cover.rational_c import core
    from mayim_tools.land_cover.rational_c import export as ex

    d, wkt, dem_p, lc_p, hsg_p, (x0, y0) = synthetic
    cats = [core.Catchment("ALL", _box(x0, y0, x0 + 2000, y0 + 2000))]
    r = core.run(core.RunSettings(cats, wkt, dem_p, lc_p, hsg_p, 700.0))
    p = d / "c.csv"
    ex.write_c_csv(str(p), r)
    c = read_c_csv(str(p))["ALL"]
    k = r.catchments[0].coeff
    assert c.alpha == pytest.approx(k.alpha, abs=1e-4)
    assert c.c1 == pytest.approx(k.c1d, abs=1e-4)
    assert c.c2 == pytest.approx(k.c2, abs=1e-4)
    assert c.alpha + c.beta + c.gamma == pytest.approx(1, abs=1e-3)
    # C1 breakdown for the SANRAL calculation sheets
    sh = r.catchments[0].shares
    assert c.detail["c1_base"] == pytest.approx(k.c1, abs=1e-4)
    assert c.detail["cs"] + c.detail["cp"] + c.detail["cv"] == pytest.approx(
        k.c1, abs=1e-3
    )
    assert c.detail["cs1_pct"] == pytest.approx(100 * sh.slope[0], abs=0.01)
    assert sum(c.detail[f"cv{i}_pct"] for i in range(1, 5)) == pytest.approx(
        100, abs=0.1
    )
    assert c.detail["map_mm"] == pytest.approx(700)
    q = d / "results.csv"
    ex.write_results_csv(str(q), r, [("MAP (mm)", 700)])
    text = q.read_text(encoding="utf-8-sig")
    assert "[catchments]" in text and "[lookup used]" in text
    row = ex.attributes(r.catchments[0])
    assert set(row) == set(ex.FIELDS)


def test_report_written(synthetic):
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    from mayim_tools.land_cover.rational_c import core
    from mayim_tools.land_cover.rational_c.report import write_report

    d, wkt, dem_p, lc_p, hsg_p, (x0, y0) = synthetic
    cats = [
        core.Catchment("A", _box(x0, y0, x0 + 1000, y0 + 2000)),
        core.Catchment("B", _box(x0 + 1000, y0, x0 + 2000, y0 + 2000)),
    ]
    r = core.run(core.RunSettings(cats, wkt, dem_p, lc_p, hsg_p, 620.0))
    path = d / "report.docx"
    notes = write_report(str(path), r, {"inputs": [("DEM", dem_p)], "outputs": []})
    assert path.exists() and path.stat().st_size > 20000
    assert not [n for n in notes if "could not" in n]
    import docx

    text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    assert "within 50 mm of a band edge" in text


def test_code_zero_is_no_data():
    look = lk.sanlc_lookup()
    out = cl.classify(
        np.array([[0.0, 13.0]]),
        np.ones((1, 2)),
        np.ones((1, 2)),
        look,
        cl.CellOptions(1),
    )
    assert out["role"].tolist() == [[0, 1]]
    assert not out["unknown"].any()


def test_styles_are_valid_xml():
    import xml.etree.ElementTree as ET
    from pathlib import Path

    styles = Path(cl.__file__).parent / "styles"
    files = sorted(styles.glob("*.qml"))
    assert len(files) == 5
    for f in files:
        ET.parse(f)
