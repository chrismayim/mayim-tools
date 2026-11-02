"""House fonts for report figures: width variants such as Arial Narrow,
whose typographic family is 'Arial' (style 'Narrow'), are found under
their own family name."""

from __future__ import annotations

import pytest

from mayim_tools._common import report_figures as rf

SOURCE = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"


def _narrow_variant(tmp_path):
    """A copy of Liberation Sans named like Arial Narrow: legacy family
    'Mayim Test Narrow' (ID 1), typographic family 'Mayim Test' (ID 16)
    with style 'Narrow' (ID 17)."""
    ttlib = pytest.importorskip("fontTools.ttLib")
    import os

    if not os.path.exists(SOURCE):
        pytest.skip("Liberation Sans not installed")
    font = ttlib.TTFont(SOURCE)
    names = font["name"]
    for rec in list(names.names):
        if rec.nameID in (1, 4, 6, 16, 17):
            names.removeNames(nameID=rec.nameID)
    for pid, eid, lid in ((3, 1, 0x409), (1, 0, 0)):
        names.setName("Mayim Test Narrow", 1, pid, eid, lid)
        names.setName("Mayim Test Narrow", 4, pid, eid, lid)
        names.setName("MayimTest-Narrow", 6, pid, eid, lid)
        names.setName("Mayim Test", 16, pid, eid, lid)
        names.setName("Narrow", 17, pid, eid, lid)
    folder = tmp_path / "fonts"
    folder.mkdir()
    path = folder / "MAYIMTN.TTF"
    font.save(str(path))
    return folder


def test_width_variant_found_under_its_own_name(tmp_path, monkeypatch):
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    folder = _narrow_variant(tmp_path)
    ft = FT2Font(str(folder / "MAYIMTN.TTF"))
    assert "Mayim Test Narrow" in rf._sfnt_family_names(ft)

    monkeypatch.setattr(rf, "_font_dirs", lambda: [str(folder)])
    monkeypatch.setattr(rf, "_REGISTER_TRIED", set())
    family, warning = rf.resolve_font("Mayim Test Narrow")
    assert (family, warning) == ("Mayim Test Narrow", None)
    found = font_manager.findfont(
        font_manager.FontProperties(family="Mayim Test Narrow"),
        fallback_to_default=False,
    )
    assert found.endswith("MAYIMTN.TTF")
    # once registered, it is listed and resolved without a warning
    assert rf.resolve_font("Mayim Test Narrow") == ("Mayim Test Narrow", None)


def test_missing_font_still_substituted(monkeypatch):
    monkeypatch.setattr(rf, "_font_dirs", lambda: [])
    monkeypatch.setattr(rf, "_REGISTER_TRIED", set())
    family, warning = rf.resolve_font("No Such Font Narrow")
    assert family == "sans-serif" and "not installed" in warning
