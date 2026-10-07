"""Tests for Extract: iSDAsoil (Africa) (no QGIS, no network)."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from mayim_tools.soil._common.errors import SoilDataError
from mayim_tools.soil._common.grid import TargetGrid
from mayim_tools.soil._common.jobs import Source
from mayim_tools.soil.isda_extract import catalogue as cat
from mayim_tools.soil.isda_extract import core

WGS84 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
)


def test_back_transforms():
    v = cat.VARIABLE_BY_CODE
    assert cat.to_value(30, v["soc"], "mean") == pytest.approx(math.expm1(3.0))
    assert cat.to_value(30, v["soc"], "sd") == pytest.approx(3.0)
    assert cat.to_value(135, v["bdod"], "mean") == pytest.approx(1.35)
    assert cat.to_value(12, v["bdod"], "sd") == pytest.approx(0.12)
    assert cat.to_value(65, v["phh2o"], "mean") == pytest.approx(6.5)
    assert cat.to_value(42, v["clay"], "mean") == pytest.approx(42.0)
    assert cat.to_value(20, v["cfvo"], "mean") == pytest.approx(math.expm1(2.0))
    assert cat.units_of(v["soc"], "sd") == "ln(1+g/kg)"
    assert cat.units_of(v["clay"], "sd") == "%"


def test_band_order_and_descriptions():
    assert cat.default_band("clay", "mean", "0-20cm") == 1
    assert cat.default_band("clay", "mean", "20-50cm") == 2
    assert cat.default_band("clay", "sd", "0-20cm") == 3
    assert cat.default_band("clay", "sd", "20-50cm") == 4
    assert cat.default_band("bedrock", "sd", "0-200cm") == 2
    descs = ["stdev_20_50", "mean_0_20", "stdev_0_20", "mean_20_50"]
    assert cat.band_from_descriptions(descs, "mean", "20-50cm") == 4
    assert cat.band_from_descriptions(descs, "sd", "20-50cm") == 1
    assert cat.band_from_descriptions(["", ""], "mean", "0-20cm") is None


def test_resolve_bands_fallback_and_descriptions():
    bands, note = core.resolve_bands("clay", lambda p: ["", "", "", ""])
    assert note.startswith("published") and bands[("sd", "20-50cm")] == 4

    def fail(p):
        raise OSError("offline")

    bands, note = core.resolve_bands("clay", fail)
    assert note.startswith("published")
    descs = ["mean_20_50", "mean_0_20", "stdev_20_50", "stdev_0_20"]
    bands, note = core.resolve_bands("clay", lambda p: descs)
    assert note.startswith("band descriptions")
    assert bands[("mean", "0-20cm")] == 2 and bands[("sd", "0-20cm")] == 4


def test_plan_selection():
    sel = core.plan_selection(["clay", "sand"], ["0-20cm"], ["mean", "sd"])
    assert sel.variables == ["sand", "clay"]
    jobs = core.plan_jobs(sel)
    assert len(jobs) == 4
    sel = core.plan_selection(["bedrock"], [], ["mean", "sd"])
    assert [j.depth for j in core.plan_jobs(sel)] == ["0-200cm", "0-200cm"]
    with pytest.raises(SoilDataError):
        core.plan_selection([], ["0-20cm"], ["mean"])
    with pytest.raises(SoilDataError):
        core.plan_selection(["clay"], ["0-20cm"], [])
    with pytest.raises(SoilDataError):
        core.plan_selection(["clay"], [], ["mean"])
    sel = core.plan_selection(["ecec"], ["0-20cm"], ["mean"])
    assert any("EFFECTIVE" in w for w in sel.warnings)
    assert any("Standard deviations" in w for w in sel.warnings)


GRID = TargetGrid(28.0, -26.0, 28.01, -25.99, 0.001, WGS84)


def _fake_reader(calls):
    raw = {
        "sand_content": 50,
        "silt_content": 20,
        "clay_content": 30,
        "carbon_organic": 20,
        "bulk_density": 140,
        "ph": 60,
        "stone_content": 10,
        "bedrock_depth": 120,
    }

    def read(path, grid, with_scale, band):
        calls.append((path, band))
        prop = path.rsplit("/", 1)[-1][:-4]
        base = raw[prop]
        value = base if band in (1, 2) or prop == "bedrock_depth" and band == 1 else 5
        if prop == "bedrock_depth" and band == 2:
            value = 25
        return np.full((grid.height, grid.width), float(value))

    return read


def test_extract_area_with_fakes(tmp_path):
    written = {}

    def write(path, grid, bands, units):
        written[os.path.basename(path)] = (bands, units)

    calls = []
    sel = core.plan_selection(
        ["sand", "silt", "clay", "soc", "bdod", "phh2o", "cfvo", "bedrock"],
        ["0-20cm", "20-50cm"],
        ["mean", "sd"],
    )
    r = core.extract_area(
        sel,
        GRID,
        str(tmp_path),
        read_grid_fn=_fake_reader(calls),
        describe_fn=lambda p: [],
        write_fn=write,
        workers=1,
    )
    assert "isda_clay_mean.tif" in written and "isda_soc_sd.tif" in written
    bands, units = written["isda_soc_mean.tif"]
    assert [b[0] for b in bands] == [
        "soc_0-20cm_mean_30m_isda (g/kg)",
        "soc_20-50cm_mean_30m_isda (g/kg)",
    ]
    assert bands[0][1][0, 0] == pytest.approx(math.expm1(2.0))
    bands, units = written["isda_soc_sd.tif"]
    assert units == "ln(1+g/kg)" and bands[0][1][0, 0] == pytest.approx(0.5)
    bands, _ = written["isda_bdod_mean.tif"]
    assert bands[1][1][0, 0] == pytest.approx(1.4)
    bands, _ = written["isda_bedrock_sd.tif"]
    assert bands[0][0] == "bedrock_0-200cm_sd_30m_isda (cm)"
    assert bands[0][1][0, 0] == pytest.approx(25.0)
    assert r.texture and r.texture[0]["flagged"] == 0
    assert all(
        p.startswith("/vsicurl/https://isdasoil.s3.amazonaws.com/") for p, _ in calls
    )
    assert {b for p, b in calls if "clay" in p} == {1, 2, 3, 4}
    meta = core.build_metadata(r, sel, {"Area": "test"}, "3.8")
    titles = [s[0] for s in meta]
    assert "Layers" in titles and "Texture check (mean sand + silt + clay)" in titles


def test_fallback_route(tmp_path):
    def read(path, grid, with_scale, band):
        if path.startswith("/vsicurl/"):
            raise OSError("blocked")
        return np.full((grid.height, grid.width), 30.0)

    r = core.extract_area(
        core.plan_selection(["clay"], ["0-20cm"], ["mean"]),
        GRID,
        str(tmp_path),
        read_grid_fn=read,
        describe_fn=lambda p: [],
        write_fn=lambda *a: None,
        workers=1,
    )
    assert r.layers[0].route == "S3 (unsigned)"
    assert any("primary route failed" in w for w in r.warnings)


def test_extract_points_with_fakes():
    def sample(path, lonlats, with_scale, band):
        return [30.0 if band in (1, 2) else 4.0 for _ in lonlats]

    sites = [core.Site("A", 28.0, -26.0), core.Site("B", 28.1, -26.1)]
    r = core.extract_points(
        core.plan_selection(["clay"], ["0-20cm", "20-50cm"], ["mean", "sd"]),
        sites,
        sample_fn=sample,
        describe_fn=lambda p: [],
        workers=1,
    )
    assert len(r.rows) == 2 * 2 * 2
    row = next(x for x in r.rows if x["Statistic"] == "Standard deviation")
    assert row["Value"] == pytest.approx(4.0) and row["Product"] == "iSDAsoil"


def test_real_gdal_band_selection(tmp_path, monkeypatch):
    gdal = pytest.importorskip("osgeo.gdal")
    gdal.UseExceptions()
    path = str(tmp_path / "clay_content.tif")
    ds = gdal.GetDriverByName("GTiff").Create(path, 40, 40, 4, gdal.GDT_Int16)
    ds.SetGeoTransform((27.98, 0.001, 0, -25.97, 0, -0.001))
    ds.SetProjection(WGS84)
    for i, (v, d) in enumerate(
        ((31, "mean_0_20"), (35, "mean_20_50"), (6, "stdev_0_20"), (7, "stdev_20_50")),
        start=1,
    ):
        b = ds.GetRasterBand(i)
        b.WriteArray(np.full((40, 40), v, dtype=np.int16))
        b.SetDescription(d)
    ds = None
    monkeypatch.setattr(core, "layer_sources", lambda var: [Source("local", path)])
    got = {}

    def write(p, grid, bands, units):
        got[os.path.basename(p)] = bands

    core.extract_area(
        core.plan_selection(["clay"], ["0-20cm", "20-50cm"], ["mean", "sd"]),
        GRID,
        str(tmp_path),
        write_fn=write,
        workers=2,
        options_ctx=__import__("contextlib").nullcontext(),
    )
    assert np.nanmean(got["isda_clay_mean.tif"][1][1]) == pytest.approx(35.0)
    assert np.nanmean(got["isda_clay_sd.tif"][0][1]) == pytest.approx(6.0)
