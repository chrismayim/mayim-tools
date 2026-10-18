"""Tests for Extract: DWS and WR2012 data (core.py, no QGIS)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from mayim_tools.hydrology.dws_wr2012 import core

# ---------------------------------------------------------------------------
# catalogue
# ---------------------------------------------------------------------------


def test_catalogue_keys_unique_and_urls_are_layers():
    keys = [layer.key for layer in core.DWS_LAYERS] + ["OUT_WMA"]
    assert len(keys) == len(set(keys))
    for layer in list(core.DWS_LAYERS) + list(core.WMA_LAYERS.values()):
        assert layer.urls
        for url in layer.urls:
            assert url.startswith("https://")
            assert url.rsplit("/", 1)[-1].isdigit()
        assert layer.geometry in ("polygon", "line", "point")
    assert core.WMA_VERSIONS[0] == "2023"
    assert len(core.WMA_OPTIONS) == len(core.WMA_VERSIONS)


def test_wr_field_names_fit_shapefiles():
    names = [f.name for f in core.WR_FIELDS]
    assert len(names) == len(set(names))
    assert all(len(n) <= 10 for n in names)
    assert all(len(n) <= 10 for n, _ in core.STATION_FIELDS)
    assert {f for f, _ in core.GRA2_GRIDS.values()} <= set(names)


# ---------------------------------------------------------------------------
# Esri JSON -> WKT
# ---------------------------------------------------------------------------

OUTER = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]  # clockwise
HOLE = [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]  # counter-clockwise
OTHER = [[20, 0], [20, 5], [25, 5], [25, 0], [20, 0]]  # clockwise


def test_polygon_with_hole_and_second_part():
    wkt = core.esri_geometry_to_wkt({"rings": [OUTER, OTHER, HOLE]})
    assert wkt.startswith("MULTIPOLYGON (((")
    polys = core.polygons_from_rings([OUTER, OTHER, HOLE])
    assert len(polys) == 2
    assert len(polys[0]) == 2  # the hole went to the polygon containing it
    assert len(polys[1]) == 1


def test_counter_clockwise_only_rings_are_outers():
    polys = core.polygons_from_rings([HOLE])
    assert len(polys) == 1 and len(polys[0]) == 1


def test_lines_points_and_empty():
    assert core.esri_geometry_to_wkt({"paths": [[[0, 0], [1, 1]]]}).startswith(
        "MULTILINESTRING ((0.00000000 0.00000000"
    )
    assert core.esri_geometry_to_wkt({"x": 28.1, "y": -25.7}) == (
        "POINT (28.10000000 -25.70000000)"
    )
    assert core.esri_geometry_to_wkt({"x": None, "y": None}) is None
    assert core.esri_geometry_to_wkt({"paths": [[[0, 0]]]}) is None
    assert core.esri_geometry_to_wkt({}) is None


# ---------------------------------------------------------------------------
# REST querying with a fake server
# ---------------------------------------------------------------------------


class FakeServer:
    def __init__(self, n):
        self.n = n
        self.calls = []

    def __call__(self, url, params):
        self.calls.append((url, dict(params)))
        if params.get("returnIdsOnly") == "true":
            return {
                "objectIdFieldName": "OBJECTID",
                "objectIds": list(range(1, self.n + 1)),
            }
        ids = [int(i) for i in params["objectIds"].split(",")]
        return {
            "geometryType": "esriGeometryPoint",
            "fields": [
                {"name": "OBJECTID", "type": "esriFieldTypeOID"},
                {"name": "QUATERNARY", "type": "esriFieldTypeString"},
                {"name": "Area_KM2", "type": "esriFieldTypeDouble"},
                {"name": "Shape__Area", "type": "esriFieldTypeDouble"},
                {"name": "Updated", "type": "esriFieldTypeDate"},
            ],
            "features": [
                {
                    "attributes": {
                        "OBJECTID": i,
                        "QUATERNARY": f" A21A{i} ",
                        "Area_KM2": 1.5 * i,
                        "Shape__Area": 99,
                        "Updated": 0,
                    },
                    "geometry": {"x": 28.0 + i / 1000, "y": -25.0},
                }
                for i in ids
            ],
        }


def test_fetch_layer_in_chunks():
    server = FakeServer(450)
    progress = []
    data = core.fetch_layer(
        "https://x/MapServer/0/",
        (27, -26, 29, -24),
        server,
        lambda a, b: progress.append((a, b)),
    )
    assert len(data.features) == 450
    assert server.calls[0][0] == "https://x/MapServer/0/query"
    assert server.calls[0][1]["geometry"] == "27.000000,-26.000000,29.000000,-24.000000"
    assert len(server.calls) == 1 + 3  # ids + 200 + 200 + 50
    assert progress[-1] == (3, 3)
    assert data.fields == [
        ("QUATERNARY", "string"),
        ("Area_KM2", "double"),
        ("Updated", "date"),
    ]
    wkt, attrs = data.features[0]
    assert wkt.startswith("POINT")
    assert attrs["QUATERNARY"] == "A21A1"
    assert attrs["Updated"] == "1970-01-01"
    assert "Shape__Area" not in attrs and "OBJECTID" not in attrs


def test_fetch_layer_empty_and_errors():
    assert (
        core.fetch_layer("u", (0, 0, 1, 1), lambda u, p: {"objectIds": None}).features
        == []
    )
    with pytest.raises(core.RestError, match="Invalid URL"):
        core.fetch_layer(
            "u",
            (0, 0, 1, 1),
            lambda u, p: {
                "error": {"code": 400, "message": "Invalid URL", "details": []}
            },
        )
    with pytest.raises(core.RestError):
        core.check_response([1, 2])
    with pytest.raises(core.RestError, match="cancelled"):
        core.fetch_layer("u", (0, 0, 1, 1), FakeServer(5), is_cancelled=lambda: True)


# ---------------------------------------------------------------------------
# dBASE
# ---------------------------------------------------------------------------


def write_dbf(path: Path, fields, records, deleted=()):
    header_len = 32 + 32 * len(fields) + 1
    rec_len = 1 + sum(f[2] for f in fields)
    out = bytearray(
        struct.pack("<BBBBIHH20x", 3, 126, 10, 17, len(records), header_len, rec_len)
    )
    for name, ftype, length, dec in fields:
        out += name.encode().ljust(11, b"\x00") + ftype.encode() + b"\x00" * 4
        out += bytes([length, dec]) + b"\x00" * 14
    out += b"\x0d"
    for i, rec in enumerate(records):
        out += b"*" if i in deleted else b" "
        for (_, ftype, length, _), value in zip(fields, rec, strict=True):
            text = "" if value is None else str(value)
            out += (text.rjust(length) if ftype in "NF" else text.ljust(length)).encode(
                "latin1"
            )
    out += b"\x1a"
    path.write_bytes(bytes(out))


def test_read_dbf(tmp_path):
    p = tmp_path / "t.dbf"
    write_dbf(
        p,
        [
            ("CATNUM", "C", 7, 0),
            ("MAR", "N", 6, 1),
            ("CURVE", "N", 3, 0),
            ("D", "D", 8, 0),
        ],
        [
            ("A21A", "12.5", "3", "20120101"),
            ("A21B", None, "4", ""),
            ("X", "1", "1", ""),
        ],
        deleted={2},
    )
    rows = core.read_dbf(p)
    assert rows == [
        {"CATNUM": "A21A", "MAR": 12.5, "CURVE": 3, "D": "2012-01-01"},
        {"CATNUM": "A21B", "MAR": None, "CURVE": 4, "D": None},
    ]


def test_read_dbf_rejects_corrupt(tmp_path):
    p = tmp_path / "bad.dbf"
    p.write_bytes(b"\x03" + bytes(range(200)))
    with pytest.raises(core.DbfError):
        core.read_dbf(p)


def test_find_path_case_insensitive(tmp_path):
    (tmp_path / "MAP").mkdir()
    (tmp_path / "MAP" / "Rain Zones WR90.dbf").write_bytes(b"")
    assert (
        core.find_path(tmp_path, "map/rain zones wr90.dbf").name
        == "Rain Zones WR90.dbf"
    )
    assert core.find_path(tmp_path, "map/none.dbf") is None
    assert core.find_path(tmp_path / "nothing", "a.dbf") is None


def test_load_wr2012_tables(tmp_path):
    (tmp_path / "catchments").mkdir()
    (tmp_path / "MAP").mkdir()
    write_dbf(
        tmp_path / "catchments" / "dquat.dbf",
        [("QUATERNARY", "C", 4, 0), ("CMAP", "N", 8, 2), ("TERTIARY", "C", 3, 0)],
        [("A21A", "650.25", "A21")],
    )
    write_dbf(
        tmp_path / "MAP" / "rsa_mar_wr2012.dbf",
        [
            ("CATNUM", "C", 7, 0),
            ("MAR", "N", 6, 1),
            ("CURVE", "N", 3, 0),
            ("HYDROZ", "C", 1, 0),
        ],
        [("a21a", "45.5", "6", "B")],
    )
    messages = []
    data = core.load_wr2012_tables(tmp_path, messages)
    assert data["A21A"] == {
        "MAP_MM": 650.25,
        "TERTIARY": "A21",
        "MAR_MM": 45.5,
        "CURVE": 6,
        "HYDROZ": "B",
    }
    assert any("rsa_calibration_12" in m for m in messages)


_WR_CANDIDATES = [  # Chris's local WR2012 copy (Windows, or staged)
    Path(
        "D:/MC Library/01 Tegniese data en hulpbronne/01 Tegniese hulpbronne/"
        "12 GIS data/WRC - WR2012/Coverages"
    ),
    Path("/mnt/user-data/uploads/12 GIS data/WRC - WR2012/Coverages"),
]
WR = next((p for p in _WR_CANDIDATES if p.is_dir()), _WR_CANDIDATES[0])


@pytest.mark.skipif(not WR.is_dir(), reason="WR2012 tables not available")
def test_real_wr2012_tables():
    messages = []
    data = core.load_wr2012_tables(WR, messages)
    assert len(data) >= 1940
    a71l = data["A71L"]
    assert a71l["MAP_MM"] == pytest.approx(287.84)
    assert a71l["MAR_MM"] == pytest.approx(2.9)
    assert a71l["EISC"] == "LOW/MARGINAL"
    assert a71l["TDS_P95"] == pytest.approx(327.6)
    assert a71l["SED_YLD"] == 89
    assert a71l["ST"] == 200


# ---------------------------------------------------------------------------
# WR2012 rows and the summary
# ---------------------------------------------------------------------------


def test_mae_midpoint_and_dominant():
    assert core.mae_midpoint("1400-1500mm") == 1450
    assert core.mae_midpoint("2200-2600mm") == 2400
    assert core.mae_midpoint("<1200mm") == 1200
    assert core.mae_midpoint(">2600mm") == 2600
    assert core.mae_midpoint("") is None
    assert core.dominant([("A", 2.0), ("B", 1.5), ("B", 1.0), (None, 9)]) == "B"
    assert core.dominant([]) is None


def test_finish_row():
    row = core.finish_row(
        {
            "QUAT": "A21A",
            "AREA_KM2": 500.0,
            "AOI_KM2": 50.0,
            "MAP_MM": 700.0,
            "MAR_MM": 35.0,
            "MAE_CLASS": "1600-1700mm",
        }
    )
    assert row["MAR_MCM"] == pytest.approx(17.5)
    assert row["RUNOFF_PC"] == pytest.approx(5.0)
    assert row["MAE_MM"] == 1650
    assert row["AOI_PCT"] == pytest.approx(10.0)
    assert set(f.name for f in core.WR_FIELDS) <= set(row)


def test_water_balance_summary_weights_by_area_inside():
    rows = [
        core.finish_row(
            {
                "QUAT": "A21A",
                "AREA_KM2": 400.0,
                "AOI_KM2": 30.0,
                "MAP_MM": 600.0,
                "MAR_MM": 30.0,
                "RECH_MM": 20.0,
            }
        ),
        core.finish_row(
            {
                "QUAT": "A21B",
                "AREA_KM2": 300.0,
                "AOI_KM2": 10.0,
                "MAP_MM": 800.0,
                "MAR_MM": 70.0,
                "RECH_MM": None,
            }
        ),
    ]
    summary = {s.item: s for s in core.water_balance_summary(rows, 40.0)}
    assert summary["Mean annual precipitation (MAP)"].value == pytest.approx(650.0)
    assert summary["Naturalised mean annual runoff (MAR)"].value == pytest.approx(40.0)
    assert summary[
        "Naturalised MAR generated in the study area"
    ].value == pytest.approx(1.6)
    assert summary["Runoff as a share of MAP"].value == pytest.approx(100 * 40 / 650)
    rech = summary["Groundwater recharge (GRA II)"]
    assert rech.value == pytest.approx(20.0)
    assert "75 %" in rech.note
    assert summary["Naturalised MAR of the intersected quaternaries (whole)"].value == (
        pytest.approx(12.0 + 21.0)
    )
    assert summary["Quaternary catchments intersected"].value == "A21A, A21B"
    assert "Study area outside the WR2012 quaternaries" not in summary


def test_summary_flags_area_outside_quaternaries():
    rows = [core.finish_row({"QUAT": "A21A", "AREA_KM2": 10.0, "AOI_KM2": 5.0})]
    summary = {s.item: s for s in core.water_balance_summary(rows, 10.0)}
    assert summary["Study area outside the WR2012 quaternaries"].value == pytest.approx(
        5.0
    )
    assert summary["Mean annual precipitation (MAP)"].value is None


# ---------------------------------------------------------------------------
# stations
# ---------------------------------------------------------------------------


def test_bundled_station_table():
    rows = core.load_stations()
    assert len(rows) == 10702
    types = {}
    for r in rows:
        types[r["TYPE"]] = types.get(r["TYPE"], 0) + 1
    assert types == {
        "GW": 6707,
        "River": 2905,
        "Met": 678,
        "Reservoir": 343,
        "Tidal": 69,
    }
    assert all(-35.5 < r["LAT"] < -21.5 and 15.5 < r["LON"] < 33.5 for r in rows)


def test_station_filters():
    rows = [
        {"STATION": "A2H001", "TYPE": "River", "OPEN": "Y"},
        {"STATION": "A2E001", "TYPE": "Met", "RAIN": "Y", "EVAP_A": "Y", "OPEN": "N"},
        {"STATION": "A2E002", "TYPE": "Met", "RAIN": "N", "EVAP_S": "Y", "OPEN": "Y"},
        {"STATION": "A2N001", "TYPE": "GW", "OPEN": "Y"},
    ]
    sel = core.filter_stations(rows, ["Rainfall", "Evaporation"])
    assert [r["STATION"] for r in sel] == ["A2E001", "A2E002"]
    assert sel[0]["CATEGORY"] == "Rainfall / Evaporation"
    assert [
        r["STATION"] for r in core.filter_stations(rows, ["Evaporation"], True)
    ] == ["A2E002"]
    sel = core.filter_stations(
        rows, core.STATION_TYPE_KEYS, keep=lambda r: r["STATION"] != "A2N001"
    )
    assert [r["STATION"] for r in sel] == ["A2H001", "A2E001", "A2E002"]
    counts = core.station_counts(sel)
    assert counts == [
        ("River flow gauges", 1, 1),
        ("Meteorological - rainfall", 1, 0),
        ("Meteorological - evaporation", 2, 1),
    ]


def test_haversine():
    assert core.haversine_km(-25.0, 28.0, -26.0, 28.0) == pytest.approx(111.2, abs=0.1)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_fmt_has_no_thousands_separators():
    assert core.fmt(1234567.891, 1) == "1234567.9"
    assert core.fmt(1234567) == "1234567"
    assert core.fmt(12345.0) == "12345"
    assert core.fmt(None) == ""
    assert core.fmt(float("nan"), 2) == ""
    assert core.fmt("A21A") == "A21A"


def test_csv_files_have_bom(tmp_path):
    rows = [core.finish_row({"QUAT": "A21A", "AREA_KM2": 1234.5, "MAR_MM": 10.0})]
    p = core.write_wr2012_csv(tmp_path / "w.csv", rows)
    raw = Path(p).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert "AREA_KM2 (km²)" in text
    assert "1234.50" in text and "12.345" in text
    s = core.write_summary_csv(
        tmp_path / "s.csv", core.water_balance_summary(rows, 0.0)
    )
    assert Path(s).read_bytes().startswith(b"\xef\xbb\xbf")
    st = core.write_stations_csv(
        tmp_path / "st.csv", [{"STATION": "A2H001", "LAT": -25.5, "LON": 28.25}]
    )
    assert "-25.500000" in Path(st).read_text(encoding="utf-8-sig")


def test_summary_for_a_point_study_area_uses_equal_weights():
    rows = [
        core.finish_row(
            {"QUAT": "A21A", "AREA_KM2": 400.0, "AOI_KM2": 0.0, "MAP_MM": 600.0}
        ),
        core.finish_row(
            {"QUAT": "A21B", "AREA_KM2": 300.0, "AOI_KM2": 0.0, "MAP_MM": 800.0}
        ),
    ]
    summary = {s.item: s for s in core.water_balance_summary(rows, 0.0)}
    assert summary["Mean annual precipitation (MAP)"].value == pytest.approx(700.0)
    assert summary["Naturalised MAR generated in the study area"].value is None


def test_features_params_generalisation_and_url_order():
    assert "maxAllowableOffset" not in core.features_params([1, 2])
    p = core.features_params([1, 2], 0.0005)
    assert p["maxAllowableOffset"] == "0.0005" and p["objectIds"] == "1,2"
    by_key = {layer.key: layer for layer in core.DWS_LAYERS}
    # the ArcGIS Online copies answer; the gia root MapServers did not
    for key in ("OUT_PRIMARY", "OUT_TERTIARY", "OUT_QUATERNARY", "OUT_RIVERS"):
        assert by_key[key].urls[0].startswith(core.AGOL)
    assert core.WMA_LAYERS["2023"].max_offset > 0
    assert by_key["OUT_QUATERNARY"].max_offset == 0
