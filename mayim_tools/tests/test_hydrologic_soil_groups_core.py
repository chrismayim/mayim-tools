"""Tests for Hydrologic soil groups (no QGIS)."""

from __future__ import annotations

import csv
import os
from statistics import NormalDist

import numpy as np
import pytest

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil.hydrologic_soil_groups import core, neh630, scs_sa
from mayim_tools.soil.hydrologic_soil_groups import probability as pr

LAYERS = ("0-30cm", "30-60cm", "60-100cm")


def _k(values):
    """{layer: array} from per-layer scalars (shape (1,))."""
    return {lab: np.array([float(v)]) for lab, v in zip(LAYERS, values, strict=True)}


# ----------------------------------------------------------------------
# NEH 630 Table 7-1
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "imp, wt, case",
    [
        (30, 500, 1),
        (49.9, 30, 1),
        (50, 40, 2),
        (100, 59.9, 2),
        (70, 60, 3),
        (100, 300, 3),
        (150, 40, 4),
        (150, 60, 5),
        (150, 100, 5),
        (150, 101, 6),
        (np.nan, 300, 0),
    ],
)
def test_case_of(imp, wt, case):
    assert neh630.case_of(np.array([imp]), np.array([wt]))[0] == case


@pytest.mark.parametrize(
    "case, ksat_um_s, expected",
    [
        # 50-100 cm, water table >= 60: 0-50 cm, 40 / 10 / 1 um/s
        (3, 40.1, neh630.A),
        (3, 40.0, neh630.B),
        (3, 10.1, neh630.B),
        (3, 10.0, neh630.C),
        (3, 1.01, neh630.C),
        (3, 1.0, neh630.D),
        # 50-100 cm, water table < 60: dual, 0-60 cm
        (2, 50, neh630.AD),
        (2, 20, neh630.BD),
        (2, 5, neh630.CD),
        (2, 0.5, neh630.D),
        # > 100 cm, water table < 60: dual, 0-100 cm, 10 / 4 / 0.4
        (4, 10.1, neh630.AD),
        (4, 10.0, neh630.BD),
        (4, 4.0, neh630.CD),
        (4, 0.4, neh630.D),
        # > 100 cm, water table 60-100: 0-50 cm, 40 / 10 / 1
        (5, 20, neh630.B),
        # > 100 cm, water table > 100: 0-100 cm, 10 / 4 / 0.4
        (6, 20, neh630.A),
        (6, 5, neh630.B),
        (6, 1, neh630.C),
        (6, 0.3, neh630.D),
        # impermeable < 50 cm: always D
        (1, 500, neh630.D),
    ],
)
def test_table_7_1(case, ksat_um_s, expected):
    k = ksat_um_s * neh630.UM_S_TO_MM_H
    g = neh630.classify(_k([k, k, k]), np.array([case]))
    assert g[0] == expected


def test_least_transmissive_layer_and_depth_range():
    # 0-50 cm uses 0-30 and 30-60; the deep layer only counts for 0-100 cm
    k = _k([200.0, 150.0, 1.0])
    assert neh630.classify(k, np.array([3]))[0] == neh630.A  # 0-50: min 150
    assert neh630.classify(k, np.array([6]))[0] == neh630.D  # 0-100: min 1 <= 1.44
    k = _k([200.0, 20.0, 300.0])
    assert neh630.classify(k, np.array([3]))[0] == neh630.C  # 20 mm/h: 3.6-36


def test_unit_conversion():
    assert neh630.SHALLOW_MM_H == pytest.approx((144.0, 36.0, 3.6))
    assert neh630.DEEP_MM_H == pytest.approx((36.0, 14.4, 1.44))


# ----------------------------------------------------------------------
# Probabilities
# ----------------------------------------------------------------------


def test_cdf_reproduces_quantiles_and_normal():
    p05, p50, p95 = np.array([2.0]), np.array([10.0]), np.array([80.0])
    assert pr.cdf(2.0, p05, p50, p95)[0] == pytest.approx(0.05, abs=1e-6)
    assert pr.cdf(10.0, p05, p50, p95)[0] == pytest.approx(0.5, abs=1e-9)
    assert pr.cdf(80.0, p05, p50, p95)[0] == pytest.approx(0.95, abs=1e-6)
    x = np.linspace(-6, 6, 25)
    ref = np.array([NormalDist().cdf(v) for v in x])
    assert np.max(np.abs(pr._phi(x) - ref)) < 2e-7


def test_group_probabilities_sum_to_one_and_degenerate():
    p = pr.group_probabilities(
        np.array([2.0]), np.array([10.0]), np.array([80.0]), neh630.SHALLOW_MM_H
    )
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p >= 0)
    # no spread: all probability in the median's group
    p = pr.group_probabilities(
        np.array([20.0]), np.array([20.0]), np.array([20.0]), neh630.SHALLOW_MM_H
    )
    assert p[:, 0] == pytest.approx([0, 0, 1, 0])
    # missing bounds collapse to the median as well
    nan = np.array([np.nan])
    p = pr.group_probabilities(nan, np.array([20.0]), nan, neh630.SHALLOW_MM_H)
    assert p[:, 0] == pytest.approx([0, 0, 1, 0])


def test_probabilities_follow_case():
    q = {t: _k([20.0, 20.0, 20.0]) for t in ("P05", "P50", "P95")}
    case = np.array([1, 3, 6])
    q = {t: {lab: np.full(3, 20.0) for lab in LAYERS} for t in q}
    p = pr.probabilities(q, case)
    assert p[:, 0] == pytest.approx([0, 0, 0, 1])  # impermeable < 50 cm
    assert p[:, 1] == pytest.approx([0, 0, 1, 0])  # C at 20 mm/h, shallow
    assert p[:, 2] == pytest.approx([0, 1, 0, 0])  # B at 20 mm/h, deep


# ----------------------------------------------------------------------
# SCS-SA
# ----------------------------------------------------------------------


def test_scs_sa_basic_groups():
    k = np.array([8.0, 7.6, 4.0, 3.8, 2.0, 1.3, 0.5])
    assert list(scs_sa.group_at(k)) == [1, 2, 2, 3, 3, 4, 4]


def test_scs_sa_intermediate_when_straddling():
    prob = np.array([[0.40], [0.45], [0.10], [0.05]])
    code, conf = scs_sa.recommended(prob, np.array([2]))
    assert code[0] == scs_sa.AB and conf[0] == pytest.approx(0.85)
    prob = np.array([[0.10], [0.70], [0.15], [0.05]])
    code, conf = scs_sa.recommended(prob, np.array([2]))
    assert code[0] == neh630.B and conf[0] == pytest.approx(0.70)
    # median group taken even if another group is more probable
    prob = np.array([[0.34], [0.30], [0.10], [0.26]])
    code, _ = scs_sa.recommended(prob, np.array([2]))
    assert code[0] == neh630.B


def test_scs_sa_adjustments():
    codes = np.array([1, 2, 22, 4, 2])
    imp = np.array([40.0, 200.0, 40.0, 40.0, 40.0])
    wt = np.array([500.0, 30.0, 30.0, 30.0, 500.0])
    out, steps = scs_sa.adjust(codes, imp, wt)
    assert list(out) == [scs_sa.AB, scs_sa.BC, scs_sa.CD_SA, 4, scs_sa.BC]
    assert list(steps) == [1, 1, 2, 2, 1]
    out, _ = scs_sa.adjust(codes, imp, wt, shallow=False, water=False)
    assert list(out) == list(codes)


def test_texture_group():
    sand = np.array([95, 70, 30, 20, 70])
    clay = np.array([5, 15, 30, 50, 30])
    assert list(core.texture_group(sand, clay)) == [1, 2, 3, 4, 0]


# ----------------------------------------------------------------------
# End to end on a synthetic Regional output folder
# ----------------------------------------------------------------------


def _write_regional(folder, k50, spread=3.0):
    from osgeo import osr

    from mayim_tools.soil._common.export import write_multiband_geotiff
    from mayim_tools.soil._common.grid import TargetGrid

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32735)
    rows, cols = k50.shape
    grid = TargetGrid(
        500000,
        7000000,
        500000 + cols * 30,
        7000000 + rows * 30,
        30.0,
        srs.ExportToWkt(),
    )
    bands = []
    for tag, f in (("P50", 1.0), ("P05", 1 / spread), ("P95", spread)):
        for i, lab in enumerate(LAYERS):
            bands.append((f"ksat_{lab}_{tag} (mm/h)", k50 * f * (1.0 - 0.1 * i)))
    os.makedirs(folder, exist_ok=True)
    write_multiband_geotiff(os.path.join(folder, "rsp_ksat.tif"), grid, bands, "mm/h")
    sand = np.full(k50.shape, 70.0)
    clay = np.full(k50.shape, 15.0)
    write_multiband_geotiff(
        os.path.join(folder, "rsp_inputs_P50.tif"),
        grid,
        [("sand_0-30cm_P50 (%)", sand), ("clay_0-30cm_P50 (%)", clay)],
        "%",
    )
    return grid


@pytest.fixture()
def regional(tmp_path):
    pytest.importorskip("osgeo")
    k50 = np.full((20, 30), 20.0)
    k50[:, 15:] = 5.0
    k50[0, 0] = np.nan
    folder = str(tmp_path / "rsp")
    grid = _write_regional(folder, k50)
    return folder, grid


def _run(tmp_path, folder, **kw):
    from mayim_tools.soil._common.export import write_multiband_geotiff

    s = core.RunSettings(
        regional_folder=folder,
        out_dir=str(tmp_path / "out"),
        bedrock_source=kw.pop("bedrock_source", "none"),
        write_report=kw.pop("report", False),
        **kw,
    )
    return core.run(s, write_fn=write_multiband_geotiff)


def test_end_to_end(tmp_path, regional):
    folder, grid = regional
    r = _run(tmp_path, folder)
    neh = r.methods["NEH630"]
    sa = r.methods["SCSSA"]
    # deep soil, no water table: 0-100 cm, 36 / 14.4 / 1.44 (least = 0.8 x P50)
    assert neh.recommended[5, 2] == neh630.B  # 16 mm/h
    assert neh.recommended[5, 20] == neh630.C  # 4 mm/h
    assert sa.recommended[5, 2] == neh630.A  # 16 mm/h > 7.6
    # 4 mm/h: median in B (3.8-7.6) but P(B) 0.36 and P(C) 0.42 -> B/C
    assert sa.recommended[5, 20] == scs_sa.BC
    assert neh.recommended[0, 0] == 0 and not r.mask[0, 0]
    assert np.nanmax(neh.confidence) <= 100.0
    for name in (
        "hsg_neh630_recommended.tif",
        "hsg_scssa_by_ksat.tif",
        "hsg_neh630_probability.tif",
        "hsg_conditions.tif",
        "hsg_zone_summary.csv",
        "hsg_metadata.csv",
    ):
        assert os.path.isfile(os.path.join(tmp_path, "out", name)), name
    with open(
        os.path.join(tmp_path, "out", "hsg_zone_summary.csv"), encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    assert {r_["Method"] for r_ in rows} == {"NEH630", "SCSSA"}
    neh_row = next(r_ for r_ in rows if r_["Method"] == "NEH630")
    total = sum(
        float(neh_row[k]) for k in neh_row if k.startswith("Share ") and neh_row[k]
    )
    assert total == pytest.approx(100.0, abs=0.05)
    # texture 70 % sand / 15 % clay indicates B
    assert float(neh_row["Texture indicates a group (%)"]) == pytest.approx(100.0)


def test_end_to_end_depths_and_zones(tmp_path, regional):
    folder, grid = regional
    zone = (
        f"POLYGON(({grid.xmin} {grid.ymin},{grid.xmin + 450} {grid.ymin},"
        f"{grid.xmin + 450} {grid.ymax},{grid.xmin} {grid.ymax},{grid.xmin} {grid.ymin}))"
    )
    r = _run(
        tmp_path,
        folder,
        bedrock_source="constant",
        bedrock_constant_m=0.8,
        water_source="constant",
        water_constant_m=0.4,
        zones=[("West", zone)],
    )
    assert r.zone_names == ["West"]
    assert not r.mask[:, 20].any()
    neh = r.methods["NEH630"]
    # 50-100 cm with water table < 60: dual, 0-60 cm, 144 / 36 / 3.6
    assert neh.recommended[5, 2] == neh630.CD  # 18 mm/h -> C, dual
    sa = r.methods["SCSSA"]
    assert sa.recommended[5, 2] == scs_sa.AB  # A, one step down for water table


def test_report_written(tmp_path, regional):
    pytest.importorskip("docx")
    pytest.importorskip("matplotlib")
    folder, _ = regional
    r = _run(tmp_path, folder, report=True)
    assert r.report_path and os.path.getsize(r.report_path) > 20000


def test_errors(tmp_path, regional):
    folder, _ = regional
    with pytest.raises(SoilDataError):
        _run(tmp_path, str(tmp_path / "missing"))
    with pytest.raises(SoilDataError):
        _run(tmp_path, folder, bedrock_source="BDRICM", sg_folder=str(tmp_path))
    with pytest.raises(SoilDataError):
        _run(tmp_path, folder, methods=())


# ----------------------------------------------------------------------
# iSDAsoil depth to bedrock with uncertainty
# ----------------------------------------------------------------------


def test_depth_class_probability():
    p = core.depth_class_probability(
        np.array([70.0, 30.0, 300.0, 70.0]),
        np.array([30.0, 10.0, 50.0, np.nan]),
        (50.0, 100.0),
    )
    nd = NormalDist()
    assert p[0] == pytest.approx(nd.cdf(1.0) - nd.cdf(-2 / 3), abs=1e-6)
    assert p[1] == pytest.approx(nd.cdf(2.0), abs=1e-6)
    assert p[2] == pytest.approx(1 - nd.cdf(-4.0), abs=1e-6)
    assert p[3] == 1.0  # no sd: no extra uncertainty


def _write_isda_bedrock(folder, grid, mean, sd):
    from mayim_tools.soil._common.export import write_multiband_geotiff

    os.makedirs(folder, exist_ok=True)
    shape = (grid.height, grid.width)
    write_multiband_geotiff(
        os.path.join(folder, "isda_bedrock_mean.tif"),
        grid,
        [("bedrock_0-200cm_mean_30m_isda (cm)", np.full(shape, mean))],
        "cm",
    )
    write_multiband_geotiff(
        os.path.join(folder, "isda_bedrock_sd.tif"),
        grid,
        [("bedrock_0-200cm_sd_30m_isda (cm)", np.full(shape, sd))],
        "cm",
    )


def test_end_to_end_isda_bedrock(tmp_path, regional):
    folder, grid = regional
    isda = str(tmp_path / "isda")
    _write_isda_bedrock(isda, grid, 70.0, 30.0)
    base = _run(
        tmp_path / "a", folder, bedrock_source="constant", bedrock_constant_m=0.7
    )
    r = _run(tmp_path / "b", folder, bedrock_source="ISDA", isda_folder=isda)
    nd = NormalDist()
    factor = nd.cdf(1.0) - nd.cdf(-2 / 3)
    for m in ("NEH630", "SCSSA"):
        a = base.methods[m]
        b = r.methods[m]
        assert np.array_equal(a.recommended, b.recommended)
    a, b = base.methods["NEH630"], r.methods["NEH630"]
    assert b.confidence[5, 2] == pytest.approx(a.confidence[5, 2] * factor, rel=1e-6)
    sa_factor = 1 - nd.cdf(-2 / 3)  # SCS-SA: same side of 50 cm
    a, b = base.methods["SCSSA"], r.methods["SCSSA"]
    assert b.confidence[5, 2] == pytest.approx(a.confidence[5, 2] * sa_factor, rel=1e-6)
    assert np.nanmax(r.p_shallow) == pytest.approx(nd.cdf(-2 / 3), abs=1e-6)
    row = r.zone_rows[0]
    assert row["Mean P(impermeable layer < 50 cm) (%)"] == pytest.approx(
        100 * nd.cdf(-2 / 3), abs=0.01
    )
    with pytest.raises(SoilDataError):
        _run(tmp_path / "c", folder, bedrock_source="ISDA", isda_folder=str(tmp_path))
