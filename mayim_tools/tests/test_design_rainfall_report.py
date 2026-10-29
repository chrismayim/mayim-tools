"""Tests of the Design Rainfall Word report (no QGIS needed): house
structure, the nearest-stations section, number formatting and the
layout map extent."""

from __future__ import annotations

import re

import pytest

docx = pytest.importorskip("docx")
pytest.importorskip("matplotlib")
gpd = pytest.importorskip("geopandas")

from shapely.geometry import Point  # noqa: E402

from mayim_tools.rainfall.design_rainfall.core import (  # noqa: E402
    DesignRainfallResult,
    PointEstimate,
)
from mayim_tools.rainfall.design_rainfall.report import (  # noqa: E402
    STATIONS_HEADING,
    map_extent,
    station_rows,
    write_report_docx,
)


def make_estimate(depth_scale: float = 1.0) -> PointEstimate:
    results = []
    for dur in ("5 min", "1.5 h", "24 h", "1 day", "3 day"):
        for rt in (2, 50, 200):
            base = {"5 min": 10, "1.5 h": 40, "24 h": 75, "1 day": 65, "3 day": 1100}
            d = base[dur] * (1 + rt / 100) * depth_scale
            results.append(DesignRainfallResult(dur, rt, round(d, 1), None, d * 1.2))
    return PointEstimate(
        latitude=-25.75,
        longitude=28.23333,
        map_mm=765.0,
        altitude_m=1370.0,
        cluster=16,
        s_cluster=1,
        av_cluster=1,
        source="grid",
        results=results,
        requested_latitude=-25.7479,
        requested_longitude=28.2293,
        distance_km=0.47,
        l1_1day=70.14,
        ratio_24h=1.202,
    )


def make_stations():
    rows = [
        {
            "FSTN_NO": "0513405_A",
            "STN_NAME": "PRETORIA-WB.",
            "LATITUDE": -25.75,
            "LONGITUDE": 28.2333,
            "MAP": 765,
            "ALTITUDE": 1370,
            "FYR": 1923,
            "LYR": 1971,
            "REC_LEN": 49,
            "DIST_KM": 0.46,
            "D1_M2": 60.9,
            "D1_M50": 147.6,
            "D1_M200": 1195.5,
        },
        {
            "FSTN_NO": "0513404_W",
            "STN_NAME": "PRETORIA-BRYNTIRION-1",
            "LATITUDE": -25.7383,
            "LONGITUDE": 28.2283,
            "MAP": 677,
            "ALTITUDE": 1340,
            "FYR": 1905,
            "LYR": 1997,
            "REC_LEN": 94,
            "DIST_KM": 1.07,
            "D1_M2": 60.0,
            "D1_M50": float("nan"),
            "D1_M200": 192.7,
        },
    ]
    geoms = [Point(r["LONGITUDE"], r["LATITUDE"]) for r in rows]
    # no CRS: the report does not use it, and setting one needs the PROJ
    # database, which is not found when the tests run outside QGIS
    return gpd.GeoDataFrame(rows, geometry=geoms)


def all_text(document) -> str:
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(c.text for c in row.cells)
    return "\n".join(parts)


def write(tmp_path, estimates, stations=None, **ctx):
    path = tmp_path / "report.docx"
    messages = write_report_docx(
        path,
        estimates,
        {"version": "1.1", "num_stations": 2, **ctx},
        stations_by_site=stations,
    )
    return docx.Document(str(path)), messages


def test_single_site_structure_and_stations_section(tmp_path):
    doc, _ = write(
        tmp_path,
        {"Site A": make_estimate()},
        {"Site A": make_stations()},
        outputs=[("CSV table", "D:/out/a.csv")],
    )
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert headings[0] == "1. Summary"  # the title is on the title page only
    assert f"6. {STATIONS_HEADING}" in headings
    assert headings[-1].endswith("References")
    assert (
        STATIONS_HEADING == "Nearest Rainfall Stations "
        "(reference only - not used in the grid-based calculation)"
    )
    text = all_text(doc)
    assert "PRETORIA-BRYNTIRION-1" in text
    assert "1923 to 1971" in text
    # one layout figure, with its caption
    assert len(doc.inline_shapes) == 1
    assert "Figure 1: Site A: layout of the point of interest" in text


def test_house_styles(tmp_path):
    doc, _ = write(tmp_path, {"Site A": make_estimate()})
    h1 = doc.styles["Heading 1"].font
    assert h1.name == "Arial" and h1.size.pt == 12 and h1.all_caps and h1.bold
    assert str(h1.color.rgb) == "0F0F36"
    assert doc.styles["Normal"].font.name == "Arial"
    assert doc.styles["Normal"].font.size.pt == 10
    for table in doc.tables[1:]:  # [0] is the borderless title-page table
        assert table.style.name == "Table Grid"


def test_no_thousands_separators_and_dash_for_missing(tmp_path):
    doc, _ = write(tmp_path, {"Site A": make_estimate()}, {"Site A": make_stations()})
    text = all_text(doc)
    assert not re.search(r"\d,\d{3}", text)
    assert "1195.5" in text  # large station value, no separator
    # lower bounds are None in the test data -> "-" in the lower-bound table
    lower = next(
        t
        for t in doc.tables
        if t.rows[0].cells[0].text == "Duration"
        and all(c.text == "-" for c in t.rows[1].cells[1:])
    )
    assert lower is not None
    # NaN station value -> "-"
    station_1day = next(
        t for t in doc.tables if t.rows[0].cells[0].text == "Station name"
    )
    assert station_1day.rows[2].cells[2].text == "-"


def test_durations_in_time_order(tmp_path):
    doc, _ = write(tmp_path, {"Site A": make_estimate()})
    table = next(t for t in doc.tables if t.rows[0].cells[0].text == "Duration")
    durations = [r.cells[0].text for r in table.rows[1:]]
    assert durations == ["5 min", "1.5 h", "24 h", "1 day", "3 day"]


def test_multi_site_one_map_per_site(tmp_path):
    estimates = {"Site A": make_estimate(), "Site B": make_estimate(1.1)}
    stations = {"Site A": make_stations(), "Site B": make_stations()}
    doc, _ = write(tmp_path, estimates, stations)
    assert len(doc.inline_shapes) == 2
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert "5.1 Site A" in headings and "5.2 Site B" in headings


def test_no_stations_section_without_stations(tmp_path):
    doc, _ = write(tmp_path, {"Site A": make_estimate()}, num_stations=0)
    headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert not any(STATIONS_HEADING in h for h in headings)


def test_plan_map_failure_falls_back(tmp_path):
    def broken(site, width_cm, dpi):
        raise RuntimeError("no QGIS")

    path = tmp_path / "r.docx"
    messages = write_report_docx(
        path, {"Site A": make_estimate()}, {"version": "1.1"}, plan_map=broken
    )
    assert any("could not be rendered" in m for m in messages)
    assert len(docx.Document(str(path)).inline_shapes) == 1


def test_station_rows_columns():
    rows = station_rows(make_stations())
    assert rows[0] == [
        "0513405_A",
        "PRETORIA-WB.",
        "0.46",
        "765",
        "1370",
        "1923 to 1971",
        "49",
    ]


def test_map_extent_padding_minimum_and_aspect():
    # single point -> minimum size, square limited by aspect range
    xmin, ymin, xmax, ymax = map_extent([(1000.0, 2000.0)])
    assert xmax - xmin >= 1500 and ymax - ymin >= 1500 * 0.6
    # wide spread -> height raised to at least 0.6 x width
    xmin, ymin, xmax, ymax = map_extent([(0.0, 0.0), (10000.0, 100.0)])
    assert (ymax - ymin) / (xmax - xmin) == pytest.approx(0.6)
    assert xmin < 0 and xmax > 10000  # padded
    # tall spread -> width raised to height
    xmin, ymin, xmax, ymax = map_extent([(0.0, 0.0), (100.0, 10000.0)])
    assert (ymax - ymin) / (xmax - xmin) == pytest.approx(1.0)


def test_title_page(tmp_path):
    doc, _ = write(
        tmp_path,
        {"Site A": make_estimate()},
        project_number="MC2025_014",
        project_name="Riverside Estate",
        software="Mayim Tools 0.17.2, QGIS 4.0.3",
        report_date="2026-10-16",
    )
    title = doc.paragraphs[0]
    assert title.style.name == "Title"
    assert title.text == "Design Rainfall Estimation (South Africa)"
    font = doc.styles["Title"].font
    assert font.name == "Arial" and font.size.pt == 20 and font.bold
    assert not font.all_caps and str(font.color.rgb) == "0F0F36"
    assert title.alignment == 1  # centre
    assert doc.paragraphs[1].text == "Riverside Estate"  # subtitle
    assert "as input to stormwater and flood design" in doc.paragraphs[2].text
    details = {r.cells[0].text: r.cells[1].text for r in doc.tables[0].rows}
    assert details["Project number"] == "MC2025_014"
    assert details["Project / site"] == "Riverside Estate"
    assert details["Prepared by"] == "Mayim Consulting"
    assert details["Tool and version"] == (
        "Design Rainfall Estimation (South Africa), version 1.1"
    )
    assert details["Software"] == "Mayim Tools 0.17.2, QGIS 4.0.3"
    assert details["Report date"] == "2026-10-16"
    assert details["Document status"] == "Draft"
    # page break before the body
    xml = doc.paragraphs[3]._p.xml
    assert 'w:type="page"' in xml
    assert doc.paragraphs[4].text == "1. Summary"


def test_title_page_missing_details_show_dash(tmp_path):
    estimates = {"Site A": make_estimate(), "Site B": make_estimate()}
    doc, _ = write(tmp_path, estimates)
    details = {r.cells[0].text: r.cells[1].text for r in doc.tables[0].rows}
    assert details["Project number"] == "-"
    assert details["Project / site"] == "-"  # several sites, no project name
    assert details["Software"] == "-"


def test_csv_is_utf8_with_bom_and_still_reads_as_ddf(tmp_path):
    """Mayim output standard: the depth table CSV is UTF-8 with a BOM; the
    DDF readers of the other tools (Adjust to DDF, HEC-HMS / SWMM project)
    still read it."""
    from mayim_tools.data.hec_hms_project import storms as stm
    from mayim_tools.rainfall._common.ddf import read_ddf_csv
    from mayim_tools.rainfall.design_rainfall.report import write_csv, write_csv_multi

    path = tmp_path / "ddf.csv"
    write_csv(make_estimate(), path, stations=make_stations())
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert path.read_text(encoding="utf-8-sig").startswith("Latitude,")
    ddf = read_ddf_csv(str(path))
    assert ddf is not None
    curves = stm.DdfCurves.from_csv(str(path))
    assert curves is not None

    multi = tmp_path / "multi.csv"
    write_csv_multi({"A": make_estimate(), "B": make_estimate(1.1)}, multi)
    assert multi.read_bytes().startswith(b"\xef\xbb\xbf")
    assert multi.read_text(encoding="utf-8-sig").startswith("Site,")


def test_ddf_table_csv_depths_only(tmp_path):
    """DDF table output: the depth columns of the depth table only - no
    lower / upper bounds, site header or stations."""
    import csv

    from mayim_tools.rainfall.design_rainfall.report import write_ddf_csv

    est = make_estimate()
    path = tmp_path / "M1234_Design_Rainfall_DDF_Table.csv"
    write_ddf_csv({"Site 1": est}, path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    depths = ["2yr Depth (mm)", "50yr Depth (mm)", "200yr Depth (mm)"]
    assert rows[0] == ["Duration", *depths]
    assert [r[0] for r in rows[1:]] == ["5 min", "1.5 h", "24 h", "1 day", "3 day"]
    lookup = {(r.duration_label, r.return_period): r.depth for r in est.results}
    assert rows[1][1:] == [str(lookup[("5 min", rt)]) for rt in (2, 50, 200)]
    text = path.read_text(encoding="utf-8-sig")
    assert "Lower" not in text and "Upper" not in text and "Latitude" not in text


def test_ddf_table_csv_multi_site(tmp_path):
    import csv

    from mayim_tools.rainfall.design_rainfall.report import write_ddf_csv

    path = tmp_path / "ddf.csv"
    write_ddf_csv({"A": make_estimate(), "B": make_estimate(1.1)}, path)
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    assert rows[0][:2] == ["Site", "Duration"]
    assert len(rows) == 1 + 2 * 5
    assert {r[0] for r in rows[1:]} == {"A", "B"}
