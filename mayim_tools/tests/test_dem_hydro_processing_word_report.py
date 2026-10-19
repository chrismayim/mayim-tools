"""Tests of the DEM Hydrological Processing Word report in the Mayim report
and figure standard (no QGIS needed): title page, house styles, section
structure, number formatting, missing values, sink coordinates and the
layout map (QGIS-rendered when supplied, simplified fallback otherwise)."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

docx = pytest.importorskip("docx")
pytest.importorskip("matplotlib")

from mayim_tools.hydrology.dem_hydro_processing.core.conditioning import (  # noqa: E402
    MODE_PRESERVE,
    ConditioningSettings,
    run_conditioning,
)
from mayim_tools.hydrology.dem_hydro_processing.core.keep_sinks import (  # noqa: E402
    KeepFeature,
)
from mayim_tools.hydrology.dem_hydro_processing.core.stream_evidence import (  # noqa: E402
    RasterTransform,
)
from mayim_tools.hydrology.dem_hydro_processing.report import (  # noqa: E402
    CHANGE_CLASSES,
    change_class_label,
    change_classes_present,
    elevation_change,
    write_report_docx,
)

NODATA = -9999.0
N = 60
GT = (500000.0, 5.0, 0.0, 7100300.0, 0.0, -5.0)


def make_dem():
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:N, 0:N]
    dem = 1500 - 0.05 * x - 0.03 * y + rng.normal(0, 0.3, (N, N))
    dem -= 6.0 * np.exp(-(((x - 30) ** 2 + (y - 30) ** 2) / 30.0))  # a pan
    dem[:3, :3] = NODATA
    return dem


def run(keep=True, settings=None):
    keep_features = []
    if keep:  # polygon around the pan (cells 25..35), in map coordinates
        x0, x1 = GT[0] + 25 * 5.0, GT[0] + 35 * 5.0
        y0, y1 = GT[3] - 35 * 5.0, GT[3] - 25 * 5.0
        keep_features = [
            KeepFeature(
                "pan",
                "polygon",
                rings=[[(x0, y1), (x1, y1), (x1, y0), (x0, y0), (x0, y1)]],
            )
        ]
    return run_conditioning(
        make_dem(),
        NODATA,
        RasterTransform(GT[0], GT[3], 5.0, 5.0),
        (5.0, 5.0),
        settings=settings or ConditioningSettings(),
        keep_features=keep_features,
    )


@pytest.fixture(scope="module")
def result():
    return run()


def write(tmp_path, result, **ctx):
    path = tmp_path / "report.docx"
    context = {"version": "1.1.0", "nodata": NODATA, **ctx}
    messages = write_report_docx(path, result, context, plan_map=ctx.pop("pm", None))
    return docx.Document(str(path)), messages


def all_text(document) -> str:
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(c.text for c in row.cells)
    return "\n".join(parts)


def headings(document):
    return [p.text for p in document.paragraphs if p.style.name.startswith("Heading")]


def test_title_page(tmp_path, result):
    doc, _ = write(
        tmp_path,
        result,
        project_number="MC2025_014",
        project_name="Riverside Estate",
        software="Mayim Tools 0.18.3, QGIS 4.0.3",
        report_date="2026-10-19",
    )
    assert doc.paragraphs[0].style.name == "Title"
    assert doc.paragraphs[0].text == "DEM Hydrological Processing"
    font = doc.styles["Title"].font
    assert font.name == "Arial" and font.size.pt == 20 and font.bold
    assert str(font.color.rgb) == "0F0F36" and not font.all_caps
    assert doc.paragraphs[1].text == "Riverside Estate"
    assert "flow direction" in doc.paragraphs[2].text
    details = {r.cells[0].text: r.cells[1].text for r in doc.tables[0].rows}
    assert details["Project number"] == "MC2025_014"
    assert details["Tool and version"] == "DEM Hydrological Processing, version 1.1.0"
    assert details["Software"] == "Mayim Tools 0.18.3, QGIS 4.0.3"
    assert details["Report date"] == "2026-10-19"
    assert details["Document status"] == "Draft"
    assert 'w:type="page"' in doc.paragraphs[3]._p.xml
    assert doc.paragraphs[4].text == "1. Summary"


def test_missing_details_show_dash(tmp_path, result):
    doc, _ = write(tmp_path, result)
    details = {r.cells[0].text: r.cells[1].text for r in doc.tables[0].rows}
    assert details["Project number"] == "-"
    assert details["Project / site"] == "-"
    assert details["Software"] == "-"


def test_section_structure_and_house_styles(tmp_path, result):
    doc, _ = write(tmp_path, result, outputs=[("Corrected DEM", "D:/out/a.tif")])
    h = headings(doc)
    main = [x for x in h if re.match(r"^\d+\. ", x)]
    assert main == [
        "1. Summary",
        "2. Tool and method",
        "3. Inputs and settings",
        "4. Parameters and calculations",
        "5. Results",
        "6. Outputs",
        "7. Notes and limitations",
        "8. References",
    ]
    assert "5.3 Sinks remaining in the corrected DEM" in h
    h1 = doc.styles["Heading 1"].font
    assert h1.name == "Arial" and h1.size.pt == 12 and h1.bold and h1.all_caps
    assert str(h1.color.rgb) == "0F0F36"
    assert doc.styles["Normal"].font.size.pt == 10
    for table in doc.tables[1:]:
        assert table.style.name == "Table Grid"
    # parameter table: three columns, Parameter | Value | Definition
    params = next(
        t for t in doc.tables if t.rows[0].cells[0].text == "Parameter (symbol, unit)"
    )
    assert len(params.columns) == 3


def test_no_outputs_section_without_outputs(tmp_path, result):
    doc, _ = write(tmp_path, result)
    assert "6. Notes and limitations" in headings(doc)


def test_no_thousands_separators_and_dashes(tmp_path, result):
    doc, _ = write(tmp_path, result, geotransform=GT)
    text = all_text(doc)
    assert not re.search(r"\d,\d{3}", text)
    assert "None" not in text and "nan" not in text
    # classification not run: Monte Carlo iterations shown as "-"
    params = next(
        t for t in doc.tables if t.rows[0].cells[0].text == "Parameter (symbol, unit)"
    )
    mc = next(r for r in params.rows if r.cells[0].text.startswith("Monte Carlo"))
    assert mc.cells[1].text == "-"


def test_sink_table_uses_coordinates_with_geotransform(tmp_path, result):
    assert result.sinks, "test setup expects kept sinks in the pan"
    doc, _ = write(tmp_path, result, geotransform=GT)
    sinks = next(t for t in doc.tables if t.rows[0].cells[0].text == "Id")
    assert [c.text for c in sinks.rows[0].cells][4:6] == ["Easting (m)", "Northing (m)"]
    k = result.sinks[0]
    assert sinks.rows[1].cells[4].text == f"{GT[0] + (k.col + 0.5) * 5.0:.1f}"
    doc2, _ = write(tmp_path, result)
    sinks2 = next(t for t in doc2.tables if t.rows[0].cells[0].text == "Id")
    assert sinks2.rows[0].cells[4].text == "Row"


def test_fallback_map_and_histogram(tmp_path, result):
    doc, _ = write(tmp_path, result, geotransform=GT)
    assert len(doc.inline_shapes) == 2
    text = all_text(doc)
    assert "Figure 1: Layout of the elevation changes" in text
    assert "Figure 2: Distribution of the elevation changes" in text
    assert "OpenStreetMap" not in text


def test_plan_map_used_and_osm_note(tmp_path, result):
    from mayim_tools._common.report_maps import OSM_ATTRIBUTION, PlanMap

    calls = []

    def pm(label, width_cm, dpi):
        calls.append((label, width_cm, dpi))
        image = np.ones((80, 80, 3))
        return PlanMap(image, (0.0, 0.0, 400.0, 400.0), [], OSM_ATTRIBUTION)

    doc, _ = write(tmp_path, result, pm=pm)
    assert calls and calls[0][1] == 14.0 and calls[0][2] == 150
    text = all_text(doc)
    assert "(background: OpenStreetMap)" in text
    assert "Open Database License" in text


def test_plan_map_failure_falls_back(tmp_path, result):
    def broken(label, width_cm, dpi):
        raise RuntimeError("no QGIS")

    doc, messages = write(tmp_path, result, pm=broken)
    assert any("could not be rendered" in m for m in messages)
    assert len(doc.inline_shapes) == 2
    assert "could not be rendered" in all_text(doc)  # listed under the notes


def test_preserve_mode_reports_classification(tmp_path):
    r = run(
        keep=False, settings=ConditioningSettings(mode=MODE_PRESERVE, n_iterations=20)
    )
    doc, _ = write(tmp_path, r)
    text = all_text(doc)
    assert "Preserve Likely Natural depressions" in text
    params = next(
        t for t in doc.tables if t.rows[0].cells[0].text == "Parameter (symbol, unit)"
    )
    mc = next(row for row in params.rows if row.cells[0].text.startswith("Monte Carlo"))
    assert mc.cells[1].text == "20"


def test_change_classes():
    assert change_class_label(-3.0) == "Lowered more than 2 m"
    assert change_class_label(-0.05) == "Lowered less than 0.1 m"
    assert change_class_label(0.0) == "Unchanged"
    assert change_class_label(0.3) == "Raised 0.1 to 0.5 m"
    assert change_class_label(5.0) == "Raised more than 2 m"
    diff = np.array([[0.0, 0.05, -0.7], [np.nan, 0.0, 0.0]])
    assert change_classes_present(diff) == [
        "Lowered 0.5 to 2 m",
        "Raised less than 0.1 m",
    ]
    assert CHANGE_CLASSES[-1][0] == math.inf


def test_elevation_change_nan_at_nodata(result):
    diff = elevation_change(result, NODATA)
    assert np.isnan(diff[0, 0])
    assert np.nanmax(np.abs(diff)) > 0
