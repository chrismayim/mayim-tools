"""
Extract: South African spatial data - the catalogue and the parts with no
QGIS (built 2026-11-21 at Chris's request: "a tool that will download the
latest set of data for an area we specify").

Sources (ArcGIS REST map services whose layers are vector feature layers
that answer queries, even where QGIS's browser only offers them as images):

- DFFE geoportal (dffeportal.environment.gov.za):
  - CSG_Cadaster/CSG_Cadastral_Data: the Chief Surveyor-General's national
    cadastre (parent farms, farm portions, erven, holdings, public places);
  - MDB_Boundaries/Wards: Municipal Demarcation Board provinces, district
    and local municipalities and wards;
  - PACA/SAPAD and PACA/SACAD: protected and conservation areas.
- Department of Agriculture AGIS (ndagis.nda.agric.za), one service per
  province: rivers, tributaries, dams, catchments, water management areas,
  land types, soils, geology, terrain, veld and vegetation types,
  cultivated fields, protected agricultural areas, grazing capacity,
  critical biodiversity and ecological support areas, gullies. The layer
  numbers differ between provinces, so layers are found by name.

Chris's choices: all four groups; the DFFE national CSG cadastre; features
that touch the area are kept whole (not clipped); the built-in list only.

The REST reading (object IDs by bounding box, features in batches of 200,
Esri JSON to WKT) is shared with Extract: DWS and WR2012 data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TOOL_VERSION = "1.0.0"

DFFE = "https://dffeportal.environment.gov.za/hosting/rest/services"
NDA = "https://ndagis.nda.agric.za/arcgis/rest/services"

# MDB province names -> NDA AGIS service names
PROVINCES = {
    "eastern cape": "Eastern_Cape",
    "free state": "Free_State",
    "gauteng": "Gauteng",
    "kwazulu-natal": "KwaZulu_Natal",
    "kwazulu natal": "KwaZulu_Natal",
    "limpopo": "Limpopo",
    "mpumalanga": "Mpumalanga",
    "northern cape": "Northern_Cape",
    "north west": "North_West",
    "north-west": "North_West",
    "western cape": "Western_Cape",
    # MDB province codes
    "ec": "Eastern_Cape",
    "fs": "Free_State",
    "gt": "Gauteng",
    "gp": "Gauteng",
    "kzn": "KwaZulu_Natal",
    "lim": "Limpopo",
    "lp": "Limpopo",
    "mp": "Mpumalanga",
    "nc": "Northern_Cape",
    "nw": "North_West",
    "wc": "Western_Cape",
}
PROVINCE_LAYER = f"{DFFE}/MDB_Boundaries/Provinces/MapServer/0"

GROUPS = [
    "Cadastre",
    "Admin boundaries",
    "Water and catchments",
    "Environment and agriculture",
]


@dataclass(frozen=True)
class Item:
    """One output layer.

    ``source`` is "dffe" (``service`` is a path under the DFFE services) or
    "nda" (``service`` is empty: the provincial AGIS service is used).
    ``names`` are the server layer names that make up this output (several
    are merged, e.g. the SAPAD protected-area classes, with a ``layer``
    field naming the class); ``fallback_ids`` are used when a name is not
    found (North West numbering)."""

    key: str
    group: str
    label: str
    description: str  # file name part
    source: str
    service: str
    names: tuple[str, ...]
    fallback_ids: tuple[int, ...] = ()
    default: bool = False
    merge: bool = False  # several server layers into one, with a "layer" field


CSG = "CSG_Cadaster/CSG_Cadastral_Data/MapServer"
MDB = "MDB_Boundaries/Wards/MapServer"
SAPAD_NAMES = (
    "ENV_DB.LDR.World_Heritage_Site",
    "ENV_DB.LDR.National_Park",
    "ENV_DB.LDR.Nature_Reserve",
    "ENV_DB.LDR.Forest_Nature_Reserve",
    "ENV_DB.LDR.Forest_Wilderness_Area",
    "ENV_DB.LDR.Marine_Protected_Area",
    "ENV_DB.LDR.Mountain_Catchment _Area",
    "ENV_DB.LDR.Protected_Environment",
    "ENV_DB.LDR.Special_Nature_Reserve",
)
SACAD_NAMES = (
    "ENV_DB.LDR.Ramsar_Site",
    "ENV_DB.LDR.Botanical_Garden",
    "ENV_DB.LDR.Biosphere_Reserve",
)

CATALOGUE: list[Item] = [
    # --- cadastre (Chief Surveyor-General, via DFFE)
    Item(
        "parent_farm",
        "Cadastre",
        "Parent farms",
        "SG Parent Farms",
        "dffe",
        CSG,
        ("Parent Farm",),
        (0,),
        True,
    ),
    Item(
        "farm_portions",
        "Cadastre",
        "Farm portions",
        "SG Farm Portions",
        "dffe",
        CSG,
        ("Farm Portions",),
        (1,),
        True,
    ),
    Item("erven", "Cadastre", "Erven", "SG Erven", "dffe", CSG, ("Erven",), (2,), True),
    Item(
        "holdings",
        "Cadastre",
        "Agricultural holdings",
        "SG Holdings",
        "dffe",
        CSG,
        ("Holdings",),
        (3,),
        True,
    ),
    Item(
        "public_places",
        "Cadastre",
        "Public places",
        "SG Public Places",
        "dffe",
        CSG,
        ("Public Places",),
        (4,),
        False,
    ),
    # --- administrative boundaries (Municipal Demarcation Board, via DFFE)
    Item(
        "provinces",
        "Admin boundaries",
        "Provinces",
        "MDB Provinces",
        "dffe",
        MDB,
        ("GISDB.LDR.Provinces",),
        (3,),
    ),
    Item(
        "district_munic",
        "Admin boundaries",
        "District municipalities",
        "MDB District Municipalities",
        "dffe",
        MDB,
        ("GISDB.LDR.DistrictMunicipalities",),
        (2,),
    ),
    Item(
        "local_munic",
        "Admin boundaries",
        "Local municipalities",
        "MDB Local Municipalities",
        "dffe",
        MDB,
        ("GISDB.LDR.LocalMunicipalites", "GISDB.LDR.LocalMunicipalities"),
        (1,),
        True,
    ),
    Item(
        "wards",
        "Admin boundaries",
        "Wards",
        "MDB Wards",
        "dffe",
        MDB,
        ("GISDB.LDR.Wards",),
        (0,),
        True,
    ),
    # --- water and catchments (Department of Agriculture AGIS, per province)
    Item(
        "rivers",
        "Water and catchments",
        "Rivers",
        "Rivers",
        "nda",
        "",
        ("Rivers",),
        (100,),
        True,
    ),
    Item(
        "tributaries",
        "Water and catchments",
        "Tributaries",
        "Tributaries",
        "nda",
        "",
        ("Tributaries",),
        (156,),
        True,
    ),
    Item(
        "dams",
        "Water and catchments",
        "Dams",
        "Dams",
        "nda",
        "",
        ("Dams",),
        (103,),
        True,
    ),
    Item(
        "quaternary",
        "Water and catchments",
        "Quaternary catchments",
        "Quaternary Catchments",
        "nda",
        "",
        ("Quaternary Catchments",),
        (108,),
        True,
    ),
    Item(
        "tertiary",
        "Water and catchments",
        "Tertiary catchments",
        "Tertiary Catchments",
        "nda",
        "",
        ("Tertiary Catchments",),
        (110,),
    ),
    Item(
        "wma",
        "Water and catchments",
        "Water management areas",
        "Water Management Areas",
        "nda",
        "",
        ("WMA Area",),
        (118,),
    ),
    # --- environment and agriculture
    Item(
        "sapad",
        "Environment and agriculture",
        "Protected areas (SAPAD, all classes)",
        "Protected Areas SAPAD",
        "dffe",
        "PACA/SAPAD/MapServer",
        SAPAD_NAMES,
        tuple(range(9)),
        True,
        merge=True,
    ),
    Item(
        "sacad",
        "Environment and agriculture",
        "Conservation areas (SACAD: Ramsar, botanical gardens, biosphere)",
        "Conservation Areas SACAD",
        "dffe",
        "PACA/SACAD/MapServer",
        SACAD_NAMES,
        (0, 1, 2),
        merge=True,
    ),
    Item(
        "cba",
        "Environment and agriculture",
        "Critical biodiversity areas",
        "Critical Biodiversity Areas",
        "nda",
        "",
        ("Critical Biodiversity Areas",),
        (148,),
        True,
    ),
    Item(
        "esa",
        "Environment and agriculture",
        "Ecological support areas",
        "Ecological Support Areas",
        "nda",
        "",
        ("Ecological Support Areas",),
        (149,),
        True,
    ),
    Item(
        "landtypes",
        "Environment and agriculture",
        "Land types",
        "Land Types",
        "nda",
        "",
        ("Landtypes Area",),
        (76,),
        True,
    ),
    Item(
        "soil_patterns",
        "Environment and agriculture",
        "General soil patterns",
        "General Soil Patterns",
        "nda",
        "",
        ("General Soil Patterns",),
        (73,),
    ),
    Item(
        "soil_classes",
        "Environment and agriculture",
        "Soil classification",
        "Soil Classification",
        "nda",
        "",
        ("Soil Classification Classes",),
        (78,),
    ),
    Item(
        "geology",
        "Environment and agriculture",
        "Geology",
        "Geology",
        "nda",
        "",
        ("Geology",),
        (60,),
    ),
    Item(
        "terrain",
        "Environment and agriculture",
        "Terrain types",
        "Terrain Types",
        "nda",
        "",
        ("Terrain Types",),
        (64,),
    ),
    Item(
        "acocks",
        "Environment and agriculture",
        "Acocks veld types",
        "Acocks Veld Types",
        "nda",
        "",
        ("Acocks Veld Types",),
        (90,),
    ),
    Item(
        "vegmap",
        "Environment and agriculture",
        "Vegetation types (VegMap)",
        "VegMap Vegetation Types",
        "nda",
        "",
        ("Vegmap Vegetation Types",),
        (96,),
    ),
    Item(
        "cultivated",
        "Environment and agriculture",
        "Cultivated fields (2021)",
        "Cultivated Fields",
        "nda",
        "",
        ("Cultivated Fields",),
        (123,),
    ),
    Item(
        "paa",
        "Environment and agriculture",
        "Protected agricultural areas (2020)",
        "Protected Agricultural Areas",
        "nda",
        "",
        ("PAA",),
        (139,),
    ),
    Item(
        "grazing",
        "Environment and agriculture",
        "Grazing capacity (2018)",
        "Grazing Capacity",
        "nda",
        "",
        ("Grazing Capacity",),
        (142,),
    ),
    Item(
        "gullies",
        "Environment and agriculture",
        "Gullies",
        "Gullies",
        "nda",
        "",
        ("Gullies",),
        (144,),
    ),
]
ITEM_BY_KEY = {i.key: i for i in CATALOGUE}

SOURCE_NAMES = {
    "dffe": "DFFE geoportal",
    "nda": "Department of Agriculture AGIS",
}
CUSTODIANS = {
    "Cadastre": "Chief Surveyor-General (via DFFE)",
    "Admin boundaries": "Municipal Demarcation Board (via DFFE)",
}
TERMS = (
    "Government data: credit the custodian, do not sell it or pass the raw "
    "data on beyond the project, verify before relying on it (DFFE GIS data "
    "user agreement; gov.za terms). Cadastral copies are not the legal "
    "record: the SG diagram or general plan governs."
)


def option_labels() -> list[str]:
    return [f"{i.group}: {i.label}" for i in CATALOGUE]


def default_indices() -> list[int]:
    return [k for k, i in enumerate(CATALOGUE) if i.default]


def norm(text) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def provinces_in(values) -> list[str]:
    """NDA service names of the provinces named in any of ``values`` (the
    attribute values of the MDB province features that touch the area)."""
    out = []
    for v in values:
        key = norm(v)
        svc = PROVINCES.get(key)
        if svc and svc not in out:
            out.append(svc)
    return out


def resolve_layer_ids(service_json: dict, names, fallback_ids) -> list[tuple[int, str]]:
    """[(layer id, name)] of the named layers in a map service's JSON
    ('layers' list), matched without regard to case and spacing; the
    fallback ids are used only when the service lists no layers (a name
    that is not in a listed service is not there)."""
    layers = [ly for ly in service_json.get("layers") or [] if "id" in ly]
    by_name = {
        norm(ly.get("name")): (int(ly["id"]), ly.get("name"))
        for ly in layers
        if "id" in ly
    }
    out = []
    for k, name in enumerate(names):
        hit = by_name.get(norm(name))
        # the fallback numbers only when the service lists no layers at all
        if hit is None and not layers and k < len(fallback_ids):
            hit = (fallback_ids[k], name)
        if hit is not None and hit not in out:
            out.append(hit)
    return out


def service_url(item: Item, province: str | None = None) -> str:
    if item.source == "nda":
        return f"{NDA}/{province}/MapServer"
    return f"{DFFE}/{item.service}"


def latest_date(
    features, field_names=("DATE_STAMP", "DATESTAMP", "LAST_EDIT", "UPDATED", "DATE")
) -> str:
    """Latest ISO date in the first of ``field_names`` that the features
    carry ('' when none): the currency of the downloaded data."""
    best = ""
    for _, attrs in features:
        for name in field_names:
            for key, value in attrs.items():
                if key.upper() == name and value:
                    text = str(value)[:10]
                    if re.match(r"\d{4}-\d{2}-\d{2}$", text) and text > best:
                        best = text
    return best


# --------------------------------------------------------------------------
# Reading one layer (REST helpers shared with Extract: DWS and WR2012 data)
# --------------------------------------------------------------------------

GEOMETRY_KINDS = {
    "esriGeometryPolygon": "polygon",
    "esriGeometryPolyline": "line",
    "esriGeometryPoint": "point",
    "esriGeometryMultipoint": "point",
}


class TooManyFeaturesError(Exception):
    def __init__(self, count: int, limit: int):
        super().__init__(f"{count} features (limit {limit})")
        self.count = count
        self.limit = limit


def fetch_features(
    layer_url, bbox, fetch, max_features=0, is_cancelled=None, progress=None
):
    """All features of one REST layer that intersect ``bbox`` (WGS 84,
    xmin ymin xmax ymax): (fields [(name, kind)], features [(wkt, attrs)],
    geometry kind, warnings). Raises TooManyFeaturesError above
    ``max_features`` (0 = no limit) before downloading them."""
    from mayim_tools.hydrology.dws_wr2012 import core as rest

    query = layer_url.rstrip("/") + "/query"
    ids_reply = rest.check_response(fetch(query, rest.ids_params(bbox)))
    ids = sorted(int(i) for i in ids_reply.get("objectIds") or [])
    if max_features and len(ids) > max_features:
        raise TooManyFeaturesError(len(ids), max_features)
    fields: dict[str, str] = {}
    features = []
    geometry = ""
    warnings = []
    batches = rest.chunks(ids, rest.ID_CHUNK)
    for n, batch in enumerate(batches, start=1):
        if is_cancelled is not None and is_cancelled():
            raise rest.RestError("cancelled")
        reply = rest.check_response(fetch(query, rest.features_params(batch)))
        geometry = reply.get("geometryType") or geometry
        for f in reply.get("fields") or []:
            name, ftype = f.get("name"), f.get("type", "")
            if name and rest.keep_field(name, ftype) and name not in fields:
                fields[name] = rest.field_kind(ftype)
        if reply.get("exceededTransferLimit"):
            warnings.append("the server returned fewer features than asked for")
        for feat in reply.get("features") or []:
            geom = feat.get("geometry")
            wkt = rest.esri_geometry_to_wkt(geom, geometry) if geom else None
            if wkt:
                attrs = feat.get("attributes") or {}
                features.append(
                    (
                        wkt,
                        {
                            k: rest.convert_value(attrs.get(k), v)
                            for k, v in fields.items()
                        },
                    )
                )
        if progress is not None:
            progress(n, len(batches))
    for _, attrs in features:
        for name in fields:
            attrs.setdefault(name, None)
    if len(features) < len(ids) and not warnings:
        warnings.append(f"{len(ids) - len(features)} features had no geometry")
    return list(fields.items()), features, GEOMETRY_KINDS.get(geometry, ""), warnings
