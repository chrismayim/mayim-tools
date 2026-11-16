"""HEC-RAS land cover parameters written by Extract: Land Cover (area mode)."""

from __future__ import annotations

import pandas as pd
import pytest

from mayim_tools.land_cover.land_cover_extract import catalogue as cat
from mayim_tools.land_cover.land_cover_extract import hec_ras_parameters as hp
from mayim_tools.land_cover.land_cover_extract import sanlc as sl


def test_every_class_has_parameters():
    assert set(hp.SANLC) == {c.code for c in sl.CLASSES}
    assert set(hp.WORLDCOVER) == {c.code for c in cat.CLASSES}
    for table in (hp.SANLC, hp.WORLDCOVER):
        for n, rng, imp, spiral in table.values():
            lo, hi = (float(v) for v in rng.split("-"))
            assert lo <= n <= hi
            assert 0 <= imp <= 100 and spiral in (0, 1)


def test_spiral_only_in_river_channels():
    assert [c for c, v in hp.SANLC.items() if v[3] == 1] == [14, 15, 30]
    assert all(v[3] == 0 for v in hp.WORLDCOVER.values())


@pytest.mark.parametrize("product", ["SANLC", "ESA WorldCover"])
def test_csv(tmp_path, product):
    classes = sl.CLASSES if product == "SANLC" else cat.CLASSES
    names = {c.code: c.name for c in classes}
    codes = [classes[0].code, classes[-1].code]
    areas = {codes[0]: 3.0, codes[1]: 1.0}
    path = tmp_path / "lc.csv"
    hp.write_hecras_csv(str(path), product, names, areas)
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    df = pd.read_csv(path, comment="#", encoding="utf-8-sig")
    assert list(df.columns) == [c for c, _ in hp.COLUMN_KEY]
    assert df["Value"].tolist() == [0, *codes]
    assert df["Area_pct"].tolist()[1:] == [75.0, 25.0]
    table = hp.SANLC if product == "SANLC" else hp.WORLDCOVER
    assert df["ManningN"].tolist()[1] == table[codes[0]][0]
