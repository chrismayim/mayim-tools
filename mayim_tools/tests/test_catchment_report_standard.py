"""Catchment Delineation Word report against the Mayim report standard v2.1:
title page, three-column parameter table, inputs (maximum stream length,
contours) and the OpenStreetMap licence note."""

from __future__ import annotations

import numpy as np
import pytest

docx = pytest.importorskip("docx")
pytest.importorskip("matplotlib")

from mayim_tools.hydrology.catchment_delineation.core import (  # noqa: E402
    Outlet,
    delineate_catchments,
)
from mayim_tools.hydrology.catchment_delineation.report import (  # noqa: E402
    OSM_ATTRIBUTION,
    PlanMap,
    contour_text,
    plan_extent,
    write_report_docx,
)
from mayim_tools.tests.test_catchment_delineation_core import (  # noqa: E402
    GT,
    PTR_NODATA,
    cxy,
    pointer_for,
    valley_dem,
)


@pytest.fixture(scope="module")
def single():
    dem = valley_dem()
    return delineate_catchments(
        pointer_for(dem),
        PTR_NODATA,
        dem,
        np.nan,
        GT,
        [Outlet("A", "point", [[cxy(39, 25)]])],
    ).catchments


def write(tmp_path, catchments, **ctx):
    path = tmp_path / "r.docx"
    messages = write_report_docx(
        str(path), catchments, {"version": "0.7.0", **ctx}, plan_map=ctx.get("pm")
    )
    return docx.Document(str(path)), messages


def test_title_page(tmp_path, single):
    doc, _ = write(
        tmp_path,
        single,
        project_number="MC2025_014",
        project_name="Riverside Estate",
        software="Mayim Tools 0.18.4, QGIS 4.0.3",
    )
    assert doc.paragraphs[0].style.name == "Title"
    assert doc.paragraphs[0].text == "Catchment Delineation and Characterisation"
    assert doc.paragraphs[1].text == "Riverside Estate"
    details = {r.cells[0].text: r.cells[1].text for r in doc.tables[0].rows}
    assert details["Project number"] == "MC2025_014"
    assert details["Software"] == "Mayim Tools 0.18.4, QGIS 4.0.3"
    assert details["Document status"] == "Draft"
    assert doc.paragraphs[4].text == "1. Summary"


def test_parameter_table_three_columns_with_values(tmp_path, single):
    doc, _ = write(tmp_path, single)
    t = next(
        t for t in doc.tables if t.rows[0].cells[0].text == "Parameter (symbol, unit)"
    )
    assert [c.text for c in t.rows[0].cells] == [
        "Parameter (symbol, unit)",
        "Value",
        "Definition / equation (source)",
    ]
    rows = {r.cells[0].text: r.cells[1].text for r in t.rows[1:]}
    area = single[0].attributes["area_km2"]
    assert rows["Contributing area, A (km²)"] == f"{area:.4f}"
    assert "(Schumm (1956))" in next(
        r.cells[2].text for r in t.rows if r.cells[0].text.startswith("Elevation gain")
    )


def test_inputs_rows_for_breakpoints_and_contours(tmp_path, single):
    doc, _ = write(
        tmp_path,
        single,
        max_stream_length_m=500.0,
        n_breakpoints=7,
        contour_interval=1.0,
        study_area="Site boundary",
    )
    inputs = next(t for t in doc.tables if t.rows[0].cells[0].text == "Item")
    rows = {r.cells[0].text: r.cells[1].text for r in inputs.rows[1:]}
    assert rows["Maximum stream length"] == "500 m (7 stream breakpoints)"
    assert rows["Contours on the layout map"] == "1 m minor, 5 m major"
    assert rows["Study area on the layout map"] == "Site boundary"
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "stream breakpoints" in text  # method step
    assert contour_text(0) == "none"


def test_osm_note_in_notes_and_map_messages(tmp_path, single):
    def pm(c, width_cm, dpi):
        return PlanMap(np.ones((60, 80, 3)), plan_extent(c), [], OSM_ATTRIBUTION)

    doc, _ = write(
        tmp_path, single, pm=pm, map_messages=["Minor contours (1 m) not drawn"]
    )
    bullets = [p.text for p in doc.paragraphs if p.style.name == "List Bullet"]
    assert any("Open Database License" in b for b in bullets)
    assert "Minor contours (1 m) not drawn" in bullets
