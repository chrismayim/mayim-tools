"""Tests for _common.data_sources."""

from __future__ import annotations

from mayim_tools._common.data_sources import SOURCES, data_source_help, log_data_source


def test_every_source_has_citation_and_licence():
    for key, src in SOURCES.items():
        assert src.citations, key
        assert src.licence, key


def test_help_lists_citations_and_licence():
    text = data_source_help("era5")
    assert text.startswith("DATA SOURCE:")
    assert "doi:10.24381/cds.adbb2d47" in text
    assert "CC BY 4.0" in text
    assert "Copernicus Climate Change Service information <year>" in text


def test_log_writes_attribution_with_year():
    lines = []

    class Feedback:
        def pushInfo(self, text):  # noqa: N802
            lines.append(text)

    log_data_source(Feedback(), "era5", year=2026)
    assert any("information 2026." in line for line in lines)
    lines.clear()
    log_data_source(Feedback(), "chirps")
    assert not any("Attribution" in line for line in lines)
