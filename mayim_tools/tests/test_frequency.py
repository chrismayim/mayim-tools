"""Tests for _common.frequency against the Mayim drainage terminology table
(MCE - Drainage terminology, 1 March 2025)."""

from __future__ import annotations

import pytest

from mayim_tools._common.frequency import (
    aep_from_ari,
    aep_from_ey,
    aep_from_one_in_x,
    aep_label,
    aep_label_from_one_in_x,
    ari_from_aep,
    ey_from_aep,
    format_aep,
    one_in_x_from_aep,
)

# (EY, AEP %, AEP 1 in x, ARI) rows of the terminology table; the table
# rounds ARI to AEP (1 in x) from 5% AEP down (20 for 19.5), hence 3%
TABLE = [
    (6, 99.75, 1.002, 0.17),
    (4, 98.17, 1.02, 0.25),
    (3, 95.02, 1.05, 0.33),
    (2, 86.47, 1.16, 0.5),
    (1, 63.21, 1.58, 1),
    (0.69, 50, 2, 1.44),
    (0.5, 39.35, 2.54, 2),
    (0.22, 20, 5, 4.48),
    (0.2, 18.13, 5.52, 5),
    (0.11, 10, 10, 9.49),
    (0.05, 5, 20, 20),
    (0.02, 2, 50, 50),
    (0.01, 1, 100, 100),
    (0.005, 0.5, 200, 200),
    (0.001, 0.1, 1000, 1000),
]


@pytest.mark.parametrize("ey,aep,one_in_x,ari", TABLE)
def test_terminology_table(ey, aep, one_in_x, ari):
    assert aep_from_ey(ey) == pytest.approx(aep, abs=0.6)
    assert one_in_x_from_aep(aep) == pytest.approx(one_in_x, rel=0.01)
    assert ari_from_aep(aep) == pytest.approx(ari, rel=0.03, abs=0.01)
    assert aep_from_ari(ari) == pytest.approx(aep, rel=0.03, abs=0.05)
    assert ey_from_aep(aep) == pytest.approx(1.0 / ari_from_aep(aep))


def test_annual_maximum_return_period_is_one_in_x():
    assert aep_from_one_in_x(2) == 50
    assert aep_from_one_in_x(5) == 20
    assert aep_from_one_in_x(100) == 1


def test_labels():
    assert aep_label(5) == "5% AEP"
    assert aep_label_from_one_in_x(50) == "2% AEP"
    assert aep_label_from_one_in_x(200) == "0.5% AEP"
    assert aep_label_from_one_in_x(3) == "33.33% AEP"
    assert format_aep(18.126) == "18.13"
    assert format_aep(0.05) == "0.05"
    assert format_aep(0.0333) == "0.0333"


def test_bad_values():
    with pytest.raises(ValueError):
        aep_from_one_in_x(0)
    with pytest.raises(ValueError):
        ari_from_aep(100)
