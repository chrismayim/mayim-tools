"""Extract: South African spatial data - catalogue, province matching, layer
lookup by name and the paged REST reading (no QGIS, no network)."""

from __future__ import annotations

import pytest

from mayim_tools.data.sa_spatial_data import core


def test_catalogue():
    keys = [i.key for i in core.CATALOGUE]
    assert len(keys) == len(set(keys))
    assert {i.group for i in core.CATALOGUE} == set(core.GROUPS)
    assert all(i.source in ("dffe", "nda") for i in core.CATALOGUE)
    # the cadastre comes from the CSG national service on DFFE (Chris's choice)
    cad = [i for i in core.CATALOGUE if i.group == "Cadastre"]
    assert all(i.source == "dffe" and "CSG_Cadast" in i.service for i in cad)
    assert [i.key for i in core.CATALOGUE if i.merge] == ["sapad", "sacad"]
    assert len(core.option_labels()) == len(core.CATALOGUE)
    assert core.default_indices()


def test_provinces_in():
    assert core.provinces_in(["North West", "x", "GAUTENG ", "North West"]) == [
        "North_West",
        "Gauteng",
    ]
    assert core.provinces_in(["KwaZulu-Natal"]) == ["KwaZulu_Natal"]
    assert core.provinces_in(["NW", "GT"]) == ["North_West", "Gauteng"]
    assert core.provinces_in([None, 3, "Lesotho"]) == []


def test_resolve_layer_ids_by_name():
    svc = {
        "layers": [{"id": 100, "name": "Rivers"}, {"id": 454, "name": "Tributaries"}]
    }
    assert core.resolve_layer_ids(svc, ("tributaries",), (156,)) == [
        (454, "Tributaries")
    ]
    # a name not in a listed service is not there: no fallback
    assert core.resolve_layer_ids(svc, ("Dams",), (103,)) == []
    # a service that lists no layers: the fallback numbers
    assert core.resolve_layer_ids({}, ("Dams",), (103,)) == [(103, "Dams")]
    # alternative spellings give one layer
    mdb = {"layers": [{"id": 1, "name": "GISDB.LDR.LocalMunicipalites"}]}
    item = core.ITEM_BY_KEY["local_munic"]
    assert core.resolve_layer_ids(mdb, item.names, item.fallback_ids) == [
        (1, "GISDB.LDR.LocalMunicipalites")
    ]


def test_service_urls():
    assert core.service_url(core.ITEM_BY_KEY["rivers"], "Gauteng").endswith(
        "/Gauteng/MapServer"
    )
    assert core.service_url(core.ITEM_BY_KEY["erven"]).endswith(
        "CSG_Cadaster/CSG_Cadastral_Data/MapServer"
    )


def test_latest_date():
    feats = [
        (None, {"DATE_STAMP": "2023-11-14"}),
        (None, {"date_stamp": "2024-10-27"}),
        (None, {"DATE_STAMP": None, "NAME": "2099-01-01"}),
    ]
    assert core.latest_date(feats) == "2024-10-27"
    assert core.latest_date([(None, {"NAME": "x"})]) == ""


def _fake_server(n):
    """fetch(url, params) for a layer of n square polygons."""

    def fetch(url, params):
        if params.get("returnIdsOnly") == "true":
            return {"objectIds": list(range(1, n + 1))}
        ids = [int(i) for i in params["objectIds"].split(",")]
        assert len(ids) <= 200
        return {
            "geometryType": "esriGeometryPolygon",
            "fields": [
                {"name": "OBJECTID", "type": "esriFieldTypeOID"},
                {"name": "NAME", "type": "esriFieldTypeString"},
                {"name": "DATE_STAMP", "type": "esriFieldTypeDate"},
                {"name": "Shape__Area", "type": "esriFieldTypeDouble"},
            ],
            "features": [
                {
                    "attributes": {
                        "OBJECTID": i,
                        "NAME": f"P{i}",
                        "DATE_STAMP": 1700000000000,
                    },
                    "geometry": {"rings": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
                }
                for i in ids
            ],
        }

    return fetch


def test_fetch_features_pages_and_limits():
    fields, feats, kind, warns = core.fetch_features(
        "http://x/MapServer/1", (0, 0, 1, 1), _fake_server(450)
    )
    assert kind == "polygon" and len(feats) == 450 and not warns
    assert [n for n, _ in fields] == ["NAME", "DATE_STAMP"]  # no ID, no Shape__
    assert feats[0][1]["DATE_STAMP"] == "2023-11-14"
    assert feats[0][0].startswith("MULTIPOLYGON") or feats[0][0].startswith("POLYGON")
    with pytest.raises(core.TooManyFeaturesError) as e:
        core.fetch_features(
            "http://x/MapServer/1", (0, 0, 1, 1), _fake_server(450), max_features=100
        )
    assert e.value.count == 450


def test_fetch_features_reports_server_errors():
    from mayim_tools.hydrology.dws_wr2012.core import RestError

    def fetch(url, params):
        return {"error": {"code": 400, "message": "Invalid layer"}}

    with pytest.raises(RestError, match="Invalid layer"):
        core.fetch_features("http://x/MapServer/9", (0, 0, 1, 1), fetch)


def test_source_codes_and_file_names():
    from mayim_tools._common.output_files import output_file_name

    codes = {core.source_code(i) for i in core.CATALOGUE}
    assert codes == set(core.SOURCES)
    name = lambda k: output_file_name(  # noqa: E731
        "M2643", core.file_description(core.ITEM_BY_KEY[k]), "shapefile"
    )
    assert name("farm_portions") == "M2643_CSG_Farm_Portions.shp"
    assert name("wards") == "M2643_MDB_Wards.shp"
    assert name("sapad") == "M2643_DFFE_Protected_Areas_SAPAD.shp"
    assert name("quaternary") == "M2643_DoA_Quaternary_Catchments.shp"
    assert "2021 Q3" in core.data_date(core.ITEM_BY_KEY["sacad"])
    assert "DATE_STAMP" in core.data_date(core.ITEM_BY_KEY["erven"])
    assert core.data_date(core.ITEM_BY_KEY["rivers"]) == "not stated"
