"""
Extract: DWS and WR2012 data - logic with zero QGIS dependency.

* The catalogue of Department of Water and Sanitation (DWS) ArcGIS REST
  layers (water management areas, drainage regions, quinary catchments,
  1:50 000 rivers and dams, Reserve status), each with a primary URL on
  gia.dws.gov.za and, where one exists, the DWS ArcGIS Online mirror.
* ArcGIS REST querying through a caller-supplied ``fetch(url, params)``
  (object IDs first, then the features in chunks, so it works on MapServer
  and FeatureServer layers alike) and conversion of Esri JSON features to
  WKT and plain attribute dictionaries.
* A small dBASE reader for the WR2012 attribute tables, the WR2012
  per-quaternary data set (MAP, MAR, Pitman parameters, ecological
  importance and present ecological state, TDS, sediment yield, zones and
  GRA II groundwater values) and its area-weighted water-balance summary.
* The bundled DWS hydrological station list (locations only) and its
  filters by station type.
* CSV writing (UTF-8 with BOM, no thousands separators).
"""

from __future__ import annotations

import csv
import datetime as _dt
import math
import re
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

TOOL_VERSION = "0.1.1"
STATION_TABLE = "DWS_Hydrological_Stations_20260920.csv"
STATION_LIST_DATE = "2026-09-20"
DATA_DIR = Path(__file__).resolve().parent / "data"

GIA = "https://gia.dws.gov.za/hosting/rest/services"
AGOL = "https://services3.arcgis.com/QdLJLZBqzVAhCil8/ArcGIS/rest/services"


class ExtractError(Exception):
    """A problem the user can act on (shown as the tool's error message)."""


class RestError(Exception):
    """An ArcGIS REST request failed or returned an error object."""


# ---------------------------------------------------------------------------
# DWS layer catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RestLayer:
    """One DWS ArcGIS REST layer: output key, label, file-name description,
    geometry kind ('polygon', 'line', 'point'), candidate layer URLs (tried
    in order), whether the output is ticked by default, whether features
    are clipped to the area (lines and points) or kept whole (catchments)
    and a one-line source note for the report. ``max_offset`` lets the
    server simplify large polygons that are only used as context."""

    key: str
    label: str
    description: str
    geometry: str
    urls: tuple[str, ...]
    default: bool = True
    clip: bool = False
    source: str = ""
    max_offset: float = 0.0  # geometry generalisation on the server, degrees


WMA_VERSIONS = ["2023", "2012", "2004"]
WMA_OPTIONS = [
    "2023 (NWRS III, gazetted 1 September 2023 - 9 WMAs)",
    "2012 (NWRS 2 - 9 WMAs)",
    "2004 (NWRS 1 - 19 WMAs)",
]
WMA_LAYERS = {
    "2023": RestLayer(
        "OUT_WMA",
        "Water management areas",
        "DWS WMA 2023",
        "polygon",
        (
            f"{GIA}/Hosted/Water_Management_Areas_2023/FeatureServer/8",
            f"{AGOL}/Water_Management_Areas_2023/FeatureServer/8",
        ),
        max_offset=0.0005,
        source="DWS Water Management Areas 2023 (NWRS III, Government Gazette "
        "49225, Notice 3855, 1 September 2023)",
    ),
    "2012": RestLayer(
        "OUT_WMA",
        "Water management areas",
        "DWS WMA 2012",
        "polygon",
        (f"{AGOL}/WMA_2012/FeatureServer/0", f"{GIA}/WMA_2012/MapServer/0"),
        max_offset=0.0005,
        source="DWS Water Management Areas 2012 (NWRS 2)",
    ),
    "2004": RestLayer(
        "OUT_WMA",
        "Water management areas",
        "DWS WMA 2004",
        "polygon",
        (f"{AGOL}/WMA_2004/FeatureServer/0", f"{GIA}/WMA_2004/MapServer/0"),
        max_offset=0.0005,
        source="DWS Water Management Areas 2004 (NWRS 1)",
    ),
}

_DRAINAGE_SOURCE = (
    "Weepener HL, van den Berg HM, Metz M and Hamandawana H (2012), WRC "
    "Report 1908/1/11 (DWS drainage regions, SRTM-based)"
)

DWS_LAYERS = [
    RestLayer(
        "OUT_PRIMARY",
        "Primary drainage regions",
        "DWS Primary Drainage Regions",
        "polygon",
        (
            f"{AGOL}/Primary_Catchments/FeatureServer/0",
            f"{GIA}/Primary_Drainage/MapServer/0",
        ),
        default=False,
        max_offset=0.0005,
        source=_DRAINAGE_SOURCE,
    ),
    RestLayer(
        "OUT_SECONDARY",
        "Secondary drainage regions",
        "DWS Secondary Drainage Regions",
        "polygon",
        (
            f"{AGOL}/Secondary_Catchments/FeatureServer/0",
            f"{GIA}/Secondary_Drainage_Region/MapServer/0",
        ),
        default=False,
        max_offset=0.0005,
        source=_DRAINAGE_SOURCE,
    ),
    RestLayer(
        "OUT_TERTIARY",
        "Tertiary drainage regions",
        "DWS Tertiary Drainage Regions",
        "polygon",
        (
            f"{AGOL}/Tertiary_Catchments/FeatureServer/0",
            f"{GIA}/Tertiary_Drainage_Region/MapServer/0",
        ),
        source=_DRAINAGE_SOURCE,
    ),
    RestLayer(
        "OUT_QUATERNARY",
        "Quaternary catchments (DWS)",
        "DWS Quaternary Catchments",
        "polygon",
        (
            f"{AGOL}/Quaternary_Catchments/FeatureServer/0",
            f"{GIA}/Quaternary_Drainage_Region/MapServer/0",
        ),
        source=_DRAINAGE_SOURCE,
    ),
    RestLayer(
        "OUT_QUINARY",
        "Quinary catchments",
        "DWS Quinary Catchments",
        "polygon",
        (f"{GIA}/Hosted/Quinary_Catchments/FeatureServer/0",),
        default=False,
        source="DWS quinary catchments (WRC 1562/1/10, 1843/2/11, NFEPA "
        "1801/2/11, March 2013)",
    ),
    RestLayer(
        "OUT_RIVERS",
        "Rivers 1:50 000",
        "DWS Rivers 50k",
        "line",
        (f"{AGOL}/DWS_Rivers_50k/FeatureServer/0", f"{GIA}/DWS_Rivers_50k/MapServer/0"),
        clip=True,
        source="DWS 1:50 000 rivers (NGI topographical data)",
    ),
    RestLayer(
        "OUT_DAMS",
        "Dams 1:50 000",
        "DWS Dams 50k",
        "polygon",
        (f"{AGOL}/DWS_Dams_50k/FeatureServer/0", f"{GIA}/DWS_Dams_50k/MapServer/0"),
        clip=True,
        source="DWS 1:50 000 dams (NGI topographical data)",
    ),
    RestLayer(
        "OUT_SW_RESERVE",
        "Surface water Reserve status (Sept 2017)",
        "DWS SW Reserve Status 2017",
        "polygon",
        (f"{GIA}/Hosted/RDM_Surface_Water_Reserve_Status_WFL1/FeatureServer/23",),
        default=False,
        source="DWS Resource Directed Measures, surface water Reserve status, "
        "September 2017",
    ),
    RestLayer(
        "OUT_GW_RESERVE",
        "Groundwater Reserve status (2018)",
        "DWS GW Reserve Status 2018",
        "polygon",
        (f"{GIA}/Hosted/RDM_Groundwater_Reserve_Status_WFL1/FeatureServer/18",),
        default=False,
        source="DWS Resource Directed Measures, groundwater Reserve status, 2018",
    ),
    RestLayer(
        "OUT_EWR_SITES",
        "EWR sites (Sept 2016)",
        "DWS EWR Sites 2016",
        "point",
        (f"{GIA}/Hosted/RDM_Surface_Water_Reserve_Status_WFL1/FeatureServer/18",),
        default=False,
        clip=True,
        source="DWS Resource Directed Measures, ecological water requirement "
        "(EWR) sites, September 2016",
    ),
]


# ---------------------------------------------------------------------------
# ArcGIS REST
# ---------------------------------------------------------------------------

Fetch = Callable[[str, dict], dict]
ID_CHUNK = 200


def ids_params(bbox: tuple[float, float, float, float]) -> dict:
    """Query parameters for the object IDs of the features that intersect a
    WGS 84 bounding box (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = bbox
    return {
        "where": "1=1",
        "geometry": f"{xmin:.6f},{ymin:.6f},{xmax:.6f},{ymax:.6f}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnIdsOnly": "true",
        "f": "json",
    }


def features_params(ids: Iterable[int], max_offset: float = 0.0) -> dict:
    """Query parameters for the features with the given object IDs, all
    fields, geometry in WGS 84 (simplified on the server by
    ``max_offset`` degrees when that is above zero)."""
    params = {
        "objectIds": ",".join(str(int(i)) for i in ids),
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "json",
    }
    if max_offset > 0:
        params["maxAllowableOffset"] = f"{max_offset:g}"
    return params


def chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def check_response(data) -> dict:
    """Raises RestError for an ArcGIS error object or a non-object reply."""
    if not isinstance(data, dict):
        raise RestError("unexpected reply (not a JSON object)")
    if "error" in data:
        err = data["error"] or {}
        details = "; ".join(str(d) for d in err.get("details") or [] if d)
        message = err.get("message") or "error"
        raise RestError(f"{message} ({details})" if details else message)
    return data


_SKIP_FIELD_TYPES = {
    "esriFieldTypeOID",
    "esriFieldTypeGlobalID",
    "esriFieldTypeGeometry",
    "esriFieldTypeBlob",
    "esriFieldTypeRaster",
    "esriFieldTypeXML",
}


def keep_field(name: str, ftype: str) -> bool:
    """Attribute fields worth keeping: not the object ID or geometry
    bookkeeping (Shape__Area, Shape_Leng, SHAPE.STArea() ...)."""
    if ftype in _SKIP_FIELD_TYPES:
        return False
    low = name.lower()
    return not (low.startswith("shape") or low.startswith("st_"))


def field_kind(ftype: str) -> str:
    """'int', 'double', 'date' or 'string' for an Esri field type."""
    if ftype in (
        "esriFieldTypeInteger",
        "esriFieldTypeSmallInteger",
        "esriFieldTypeBigInteger",
    ):
        return "int"
    if ftype in ("esriFieldTypeDouble", "esriFieldTypeSingle"):
        return "double"
    if ftype == "esriFieldTypeDate":
        return "date"
    return "string"


def convert_value(value, kind: str):
    if value is None:
        return None
    if kind == "date":
        try:  # Esri dates are milliseconds since 1970 (UTC)
            stamp = _dt.datetime(1970, 1, 1) + _dt.timedelta(milliseconds=value)
            return stamp.date().isoformat()
        except (TypeError, ValueError, OverflowError):
            return str(value)
    if kind == "string":
        return str(value).strip()
    return value


@dataclass
class LayerData:
    """A downloaded layer: fields [(name, kind)], features [(wkt, attrs)],
    the URL that answered and any warnings."""

    fields: list[tuple[str, str]] = field(default_factory=list)
    features: list[tuple[str, dict]] = field(default_factory=list)
    url: str = ""
    warnings: list[str] = field(default_factory=list)


def fetch_layer(
    layer_url: str,
    bbox: tuple[float, float, float, float],
    fetch: Fetch,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    max_offset: float = 0.0,
) -> LayerData:
    """All features of one REST layer that intersect ``bbox`` (WGS 84)."""
    query = layer_url.rstrip("/") + "/query"
    ids_reply = check_response(fetch(query, ids_params(bbox)))
    ids = sorted(int(i) for i in ids_reply.get("objectIds") or [])
    out = LayerData(url=layer_url)
    if not ids:
        return out
    fields_seen: dict[str, str] = {}
    geometry_type = ""
    batches = chunks(ids, ID_CHUNK)
    for n, batch in enumerate(batches, start=1):
        if is_cancelled is not None and is_cancelled():
            raise RestError("cancelled")
        reply = check_response(fetch(query, features_params(batch, max_offset)))
        geometry_type = reply.get("geometryType") or geometry_type
        for f in reply.get("fields") or []:
            name, ftype = f.get("name"), f.get("type", "")
            if name and keep_field(name, ftype) and name not in fields_seen:
                fields_seen[name] = field_kind(ftype)
        if reply.get("exceededTransferLimit"):
            out.warnings.append(
                "The server returned fewer features than requested in one batch."
            )
        for feat in reply.get("features") or []:
            geom = feat.get("geometry")
            wkt = esri_geometry_to_wkt(geom, geometry_type) if geom else None
            if not wkt:
                continue
            attrs = feat.get("attributes") or {}
            out.features.append(
                (
                    wkt,
                    {k: convert_value(attrs.get(k), v) for k, v in fields_seen.items()},
                )
            )
        if progress is not None:
            progress(n, len(batches))
    out.fields = list(fields_seen.items())
    # features of early batches may miss fields first seen later
    for _, attrs in out.features:
        for name in fields_seen:
            attrs.setdefault(name, None)
    return out


# ---------------------------------------------------------------------------
# Esri JSON geometry -> WKT
# ---------------------------------------------------------------------------


def ring_area(ring) -> float:
    """Signed shoelace area (positive = counter-clockwise)."""
    s = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True):
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def point_in_ring(x: float, y: float, ring) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                inside = not inside
    return inside


def _closed(ring):
    ring = [(float(p[0]), float(p[1])) for p in ring]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _coords(ring) -> str:
    return ", ".join(f"{x:.8f} {y:.8f}" for x, y in ring)


def polygons_from_rings(rings) -> list[list[list[tuple[float, float]]]]:
    """Groups Esri rings into polygons: clockwise rings are outer rings,
    counter-clockwise rings are holes of the outer ring that contains them."""
    polygons: list[list] = []
    holes = []
    for raw in rings:
        ring = _closed(raw)
        if len(ring) < 4:
            continue
        if ring_area(ring[:-1]) <= 0:  # clockwise (or degenerate) = outer
            polygons.append([ring])
        else:
            holes.append(ring)
    if not polygons and holes:  # wrongly wound data: treat all as outers
        return [[h] for h in holes]
    for hole in holes:
        x, y = hole[0]
        owner = next(
            (p for p in polygons if point_in_ring(x, y, p[0][:-1])), polygons[-1]
        )
        owner.append(hole)
    return polygons


def esri_geometry_to_wkt(geom: dict, geometry_type: str = "") -> str | None:
    if not geom:
        return None
    if "rings" in geom:
        polys = polygons_from_rings(geom["rings"])
        if not polys:
            return None
        parts = ", ".join(
            "(" + ", ".join(f"({_coords(r)})" for r in poly) + ")" for poly in polys
        )
        return f"MULTIPOLYGON ({parts})"
    if "paths" in geom:
        paths = [
            [(float(p[0]), float(p[1])) for p in path]
            for path in geom["paths"]
            if len(path) >= 2
        ]
        if not paths:
            return None
        return "MULTILINESTRING (" + ", ".join(f"({_coords(p)})" for p in paths) + ")"
    if "points" in geom:
        pts = geom["points"]
        if not pts:
            return None
        return "MULTIPOINT (" + ", ".join(f"({p[0]:.8f} {p[1]:.8f})" for p in pts) + ")"
    if "x" in geom and "y" in geom and geom["x"] is not None:
        x, y = float(geom["x"]), float(geom["y"])
        if math.isnan(x) or math.isnan(y):
            return None
        return f"POINT ({x:.8f} {y:.8f})"
    return None


# ---------------------------------------------------------------------------
# dBASE (.dbf) reader for the WR2012 tables
# ---------------------------------------------------------------------------


class DbfError(Exception):
    pass


def read_dbf(path: str | Path, encoding: str = "latin1") -> list[dict]:
    """Records of a dBASE III/IV table as dictionaries (C, N, F, L, D
    fields; deleted records skipped). Numeric blanks become None."""
    data = Path(path).read_bytes()
    if len(data) < 32:
        raise DbfError(f"{Path(path).name}: not a dBASE file")
    n_rec, header_len, rec_len = struct.unpack("<IHH", data[4:12])
    fields = []
    pos = 32
    while pos < header_len - 1 and data[pos] != 0x0D:
        raw = data[pos : pos + 32]
        if len(raw) < 32:
            break
        name = raw[:11].split(b"\x00")[0].decode("ascii", "replace").strip()
        ftype = chr(raw[11])
        length, decimals = raw[16], raw[17]
        if not name or not name.replace("_", "").isalnum():
            raise DbfError(f"{Path(path).name}: corrupt field header")
        fields.append((name, ftype, length, decimals))
        pos += 32
    if sum(f[2] for f in fields) + 1 != rec_len:
        raise DbfError(f"{Path(path).name}: field lengths do not match the record")
    if header_len + n_rec * rec_len > len(data) + 1:
        raise DbfError(f"{Path(path).name}: file is shorter than its header says")
    records = []
    for i in range(n_rec):
        start = header_len + i * rec_len
        rec = data[start : start + rec_len]
        if not rec or rec[:1] == b"*":
            continue
        out = {}
        off = 1
        for name, ftype, length, _dec in fields:
            raw = rec[off : off + length]
            off += length
            text = raw.decode(encoding, "replace").strip().strip("\x00")
            if ftype in "NF":
                text = text.replace("*", "")
                if not text:
                    out[name] = None
                else:
                    try:
                        value = float(text)
                        out[name] = (
                            int(value) if ftype == "N" and "." not in text else value
                        )
                    except ValueError:
                        out[name] = None
            elif ftype == "L":
                out[name] = text.upper() in ("Y", "T") if text else None
            elif ftype == "D":
                out[name] = (
                    f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 else None
                )
            else:
                out[name] = text
        records.append(out)
    return records


def find_path(root: str | Path, relative: str) -> Path | None:
    """``root/relative`` matched case-insensitively, part by part (the
    WR2012 folders mix cases); None when absent."""
    current = Path(root)
    for part in Path(relative).parts:
        if not current.is_dir():
            return None
        # scan the folder (rather than exists()) so the result carries the
        # name as stored, also on case-insensitive Windows drives
        names = list(current.iterdir())
        low = part.lower()
        match = next((p for p in names if p.name == part), None) or next(
            (p for p in names if p.name.lower() == low), None
        )
        if match is None:
            return None
        current = match
    return current


# ---------------------------------------------------------------------------
# WR2012 per-quaternary data
# ---------------------------------------------------------------------------

WR2012_FILES = {
    "quaternaries": "catchments/dquat.shp",
    "mar": "MAP/rsa_mar_wr2012.dbf",
    "calibration": "Calibration Parameters/rsa_calibration_12.dbf",
    "ewr": "EWR/rsa_ewr.dbf",
    "tds": "TDS/rsa_tds_12.dbf",
    "sediment": "Sedimentation/sediment_yld_wr90.dbf",
    "rain_zones": "MAP/rain zones wr90.shp",
    "evap_zones": "Evaporation/evapo zones wr90.shp",
    "mae": "Evaporation/rsa_mae-apan.shp",
}
GRA2_GRID_DIR = "Groundwater_GRA2/GRIDS"


@dataclass(frozen=True)
class WrField:
    """One attribute of the WR2012 quaternary output: shapefile / CSV name
    (10 characters or fewer), label, unit, number of decimals (None for
    text) and source."""

    name: str
    label: str
    unit: str
    decimals: int | None
    source: str


WR_FIELDS = [
    WrField("QUAT", "Quaternary catchment", "", None, "WR2012 dquat"),
    WrField("TERTIARY", "Tertiary drainage region", "", None, "WR2012 dquat"),
    WrField("HYDROZ", "Hydrological zone", "", None, "WR2012 rsa_mar_wr2012"),
    WrField("AREA_KM2", "Quaternary area", "km²", 2, "WR2012 dquat (ellipsoidal)"),
    WrField("AOI_KM2", "Area inside the study area", "km²", 3, "computed"),
    WrField("AOI_PCT", "Share of the quaternary in the study area", "%", 1, "computed"),
    WrField("MAP_MM", "Mean annual precipitation", "mm/a", 0, "WR2012 dquat CMAP"),
    WrField(
        "MAR_MM", "Naturalised mean annual runoff", "mm/a", 1, "WR2012 rsa_mar_wr2012"
    ),
    WrField(
        "MAR_MCM", "Naturalised mean annual runoff", "million m³/a", 3, "MAR_MM x area"
    ),
    WrField("RUNOFF_PC", "Runoff as a share of MAP", "%", 1, "MAR_MM / MAP_MM"),
    WrField("CURVE", "Regional runoff curve", "", 0, "WR2012 rsa_mar_wr2012"),
    WrField("RAINZONE", "Rainfall zone (largest share)", "", None, "WR90 rain zones"),
    WrField(
        "EVAPZONE",
        "Evaporation zone (largest share)",
        "",
        None,
        "WR90 evaporation zones",
    ),
    WrField(
        "MAE_CLASS",
        "A-pan mean annual evaporation class",
        "",
        None,
        "WR2012 rsa_mae-apan",
    ),
    WrField(
        "MAE_MM",
        "A-pan mean annual evaporation (class midpoint)",
        "mm/a",
        0,
        "from MAE_CLASS",
    ),
    WrField("POW", "Pitman POW", "", 2, "WR2012 rsa_calibration_12 (POW12)"),
    WrField(
        "ST",
        "Pitman ST (soil moisture capacity)",
        "mm",
        0,
        "WR2012 rsa_calibration_12 (ST12)",
    ),
    WrField(
        "FT",
        "Pitman FT (runoff at ST)",
        "mm/month",
        1,
        "WR2012 rsa_calibration_12 (FT12)",
    ),
    WrField(
        "ZMIN",
        "Pitman ZMIN (minimum absorption)",
        "mm/month",
        0,
        "WR2012 rsa_calibration_12 (ZMIN12)",
    ),
    WrField(
        "ZMAX",
        "Pitman ZMAX (maximum absorption)",
        "mm/month",
        0,
        "WR2012 rsa_calibration_12 (ZMAX12)",
    ),
    WrField(
        "GPOW",
        "Pitman GPOW (recharge power)",
        "",
        2,
        "WR2012 rsa_calibration_12 (GPOW12)",
    ),
    WrField(
        "HGSL",
        "Pitman HGSL (groundwater slope)",
        "%",
        2,
        "WR2012 rsa_calibration_12 (HGSL12)",
    ),
    WrField(
        "HGGW",
        "Pitman HGGW (maximum recharge)",
        "mm/month",
        2,
        "WR2012 rsa_calibration_12 (HGGW12)",
    ),
    WrField(
        "EISC",
        "Ecological importance and sensitivity class",
        "",
        None,
        "WR2012 rsa_ewr",
    ),
    WrField("PES", "Present ecological state (desktop)", "", None, "WR2012 rsa_ewr"),
    WrField("TDS_P95", "TDS, 95th percentile", "mg/l", 0, "WR2012 rsa_tds_12 (TDSP95)"),
    WrField(
        "SED_YLD", "Sediment yield", "t/km²/a", 0, "WR90 sediment_yld_wr90 (YLD_1000)"
    ),
    WrField(
        "RECH_MM",
        "Groundwater recharge (GRA II, mean)",
        "mm/a",
        1,
        "GRA II grid recharge",
    ),
    WrField(
        "RECH_DRY",
        "Groundwater recharge, dry (GRA II, mean)",
        "mm/a",
        1,
        "GRA II grid recharge_dry (m³/km²/a ÷ 1000)",
    ),
    WrField(
        "BASEF_MM",
        "Groundwater baseflow (GRA II, mean)",
        "mm/a",
        1,
        "GRA II grid baseflow (m³/km²/a ÷ 1000)",
    ),
    WrField(
        "EXPL_FACT",
        "Groundwater exploitation factor (GRA II, mean)",
        "",
        2,
        "GRA II grid exploit_fact",
    ),
    WrField(
        "GEP_WET",
        "Groundwater exploitation potential, wet (GRA II, mean)",
        "m³/km²/a",
        0,
        "GRA II grid gep_wet",
    ),
    WrField(
        "GEP_DRY",
        "Groundwater exploitation potential, dry (GRA II, mean)",
        "m³/km²/a",
        0,
        "GRA II grid gep_dry",
    ),
]
WR_FIELD_BY_NAME = {f.name: f for f in WR_FIELDS}

# GRA II grid folder -> (output field, factor to the output unit). The
# recharge grid holds mm/a; recharge_dry and baseflow hold m³/km²/a (their
# means over South Africa are about 1000 times the recharge mean), converted
# to mm/a; the exploitation potential grids stay in m³/km²/a.
GRA2_GRIDS = {
    "recharge": ("RECH_MM", 1.0),
    "recharge_dry": ("RECH_DRY", 0.001),
    "baseflow": ("BASEF_MM", 0.001),
    "exploit_fact": ("EXPL_FACT", 1.0),
    "gep_wet": ("GEP_WET", 1.0),
    "gep_dry": ("GEP_DRY", 1.0),
}


def norm_quat(code) -> str:
    return str(code or "").strip().upper()


def mae_midpoint(label: str | None) -> float | None:
    """Midpoint of an MAE class such as '1400-1500mm'; the bound for the
    open classes '<1200mm' and '>2600mm'."""
    if not label:
        return None
    text = label.replace(" ", "").lower().replace("mm", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", text)
    if m:
        return 0.5 * (float(m.group(1)) + float(m.group(2)))
    m = re.fullmatch(r"[<>](\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def dominant(pieces: Iterable[tuple[object, float]]):
    """The value with the largest total area among (value, area) pieces."""
    totals: dict = {}
    for value, area in pieces:
        if value in (None, "") or area is None or area <= 0:
            continue
        totals[value] = totals.get(value, 0.0) + area
    if not totals:
        return None
    return max(totals.items(), key=lambda t: (t[1], str(t[0])))[0]


def _first(row: dict, *names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def load_wr2012_tables(root: str | Path, messages: list[str]) -> dict[str, dict]:
    """Per-quaternary attributes from the WR2012 dBASE tables, keyed by
    quaternary code. Missing or unreadable tables are reported in
    ``messages`` and skipped."""
    out: dict[str, dict] = {}

    def table(key: str) -> list[dict]:
        rel = WR2012_FILES[key]
        if rel.endswith(".shp"):
            rel = rel[:-4] + ".dbf"
        path = find_path(root, rel)
        if path is None:
            messages.append(f"WR2012 table not found: {rel}")
            return []
        try:
            return read_dbf(path)
        except (DbfError, OSError, struct.error) as e:
            messages.append(f"WR2012 table {rel} could not be read: {e}")
            return []

    for r in table("quaternaries"):
        q = norm_quat(r.get("QUATERNARY"))
        if q:
            d = out.setdefault(q, {})
            d["MAP_MM"] = _first(r, "CMAP")
            d["TERTIARY"] = _first(r, "TERTIARY")
    for r in table("mar"):
        q = norm_quat(r.get("CATNUM"))
        if q:
            d = out.setdefault(q, {})
            d["MAR_MM"] = _first(r, "MAR")
            d["CURVE"] = _first(r, "CURVE")
            d["HYDROZ"] = _first(r, "HYDROZ")
    for r in table("calibration"):
        q = norm_quat(_first(r, "Quaterna_2", "QUATERNARY"))
        if q:
            d = out.setdefault(q, {})
            for name in ("POW", "ST", "FT", "ZMIN", "ZMAX", "GPOW", "HGSL", "HGGW"):
                d[name] = _first(r, f"{name}12", name)
    for r in table("ewr"):
        q = norm_quat(r.get("QUATERNARY"))
        if q:
            d = out.setdefault(q, {})
            d["EISC"] = _first(r, "EISC")
            d["PES"] = _first(r, "PESC__DESK")
    for r in table("tds"):
        q = norm_quat(_first(r, "Quat", "QUATERNARY"))
        if q:
            out.setdefault(q, {})["TDS_P95"] = _first(r, "TDSP95", "TDS_P95")
    for r in table("sediment"):
        q = norm_quat(r.get("CATNUM"))
        if q:
            out.setdefault(q, {})["SED_YLD"] = _first(r, "YLD_1000")
    return out


def finish_row(row: dict) -> dict:
    """Derived values of one WR2012 quaternary row (MAR volume, runoff
    share, MAE midpoint, study area share); every WR field present."""
    area = row.get("AREA_KM2")
    mar = row.get("MAR_MM")
    map_ = row.get("MAP_MM")
    if area is not None and mar is not None:
        row["MAR_MCM"] = mar * area / 1000.0
    if mar is not None and map_:
        row["RUNOFF_PC"] = 100.0 * mar / map_
    if row.get("MAE_CLASS") and row.get("MAE_MM") is None:
        row["MAE_MM"] = mae_midpoint(row["MAE_CLASS"])
    if area and row.get("AOI_KM2") is not None:
        row["AOI_PCT"] = min(100.0, 100.0 * row["AOI_KM2"] / area)
    for f in WR_FIELDS:
        row.setdefault(f.name, None)
    return row


# ---------------------------------------------------------------------------
# Water-balance summary
# ---------------------------------------------------------------------------


@dataclass
class SummaryRow:
    item: str
    value: float | str | None
    unit: str
    note: str = ""


def weighted_mean(rows: list[dict], name: str, weight: str = "AOI_KM2"):
    """Area-weighted mean of ``name`` over rows with a value; returns (mean,
    share of the weight covered). When the study area has no area (points
    or lines without a buffer) the quaternaries count equally."""
    total = sum(r.get(weight) or 0.0 for r in rows)
    if total <= 0:
        values = [r[name] for r in rows if r.get(name) is not None]
        if not values:
            return None, 0.0
        return sum(values) / len(values), len(values) / len(rows)
    pairs = [
        (r[name], r.get(weight) or 0.0)
        for r in rows
        if r.get(name) is not None and (r.get(weight) or 0.0) > 0
    ]
    covered = sum(w for _, w in pairs)
    if not pairs or covered <= 0:
        return None, 0.0
    return sum(v * w for v, w in pairs) / covered, covered / total if total else 0.0


def water_balance_summary(rows: list[dict], aoi_area_km2: float) -> list[SummaryRow]:
    """Area-weighted WR2012 water-balance indicators for the study area. The
    study area takes each quaternary's values pro rata to the part of the
    quaternary inside it."""
    out = [
        SummaryRow("Study area", aoi_area_km2, "km²"),
        SummaryRow(
            "Quaternary catchments intersected",
            ", ".join(r["QUAT"] for r in rows) or "-",
            "",
        ),
        SummaryRow(
            "Total area of the intersected quaternaries",
            sum(r.get("AREA_KM2") or 0.0 for r in rows),
            "km²",
        ),
    ]
    within = sum(r.get("AOI_KM2") or 0.0 for r in rows)
    if aoi_area_km2 > 0 and within < 0.98 * aoi_area_km2:
        out.append(
            SummaryRow(
                "Study area outside the WR2012 quaternaries",
                aoi_area_km2 - within,
                "km²",
                "No WR2012 values there (outside South Africa, Lesotho and "
                "Eswatini or at the coast).",
            )
        )

    def mean_row(label, name, unit, decimals_note=""):
        value, share = weighted_mean(rows, name)
        note = decimals_note
        if value is not None and share < 0.999:
            note = (note + " " if note else "") + (
                f"Values cover {100 * share:.0f} % of the study area."
            )
        out.append(SummaryRow(label, value, unit, note))
        return value

    map_ = mean_row("Mean annual precipitation (MAP)", "MAP_MM", "mm/a")
    mar = mean_row("Naturalised mean annual runoff (MAR)", "MAR_MM", "mm/a")
    pairs = [
        (r["MAR_MM"], r.get("AOI_KM2") or 0.0)
        for r in rows
        if r.get("MAR_MM") is not None
    ]
    out.append(
        SummaryRow(
            "Naturalised MAR generated in the study area",
            sum(v * a for v, a in pairs) / 1000.0 if pairs else None,
            "million m³/a",
            "MAR (mm) x area inside the study area.",
        )
    )
    out.append(
        SummaryRow(
            "Runoff as a share of MAP",
            100.0 * mar / map_ if mar is not None and map_ else None,
            "%",
        )
    )
    mean_row(
        "A-pan mean annual evaporation (MAE)",
        "MAE_MM",
        "mm/a",
        "Midpoints of the WR2012 MAE classes.",
    )
    rech = mean_row("Groundwater recharge (GRA II)", "RECH_MM", "mm/a")
    pairs = [
        (r["RECH_MM"], r.get("AOI_KM2") or 0.0)
        for r in rows
        if r.get("RECH_MM") is not None
    ]
    out.append(
        SummaryRow(
            "Groundwater recharge volume in the study area",
            sum(v * a for v, a in pairs) / 1000.0 if pairs else None,
            "million m³/a",
            "Recharge (mm) x area inside the study area.",
        )
    )
    mean_row("Groundwater recharge, dry (GRA II)", "RECH_DRY", "mm/a")
    mean_row("Groundwater baseflow (GRA II)", "BASEF_MM", "mm/a")
    if rech is not None and map_:
        out.append(SummaryRow("Recharge as a share of MAP", 100.0 * rech / map_, "%"))
    out.append(
        SummaryRow(
            "Naturalised MAR of the intersected quaternaries (whole)",
            (
                sum(r["MAR_MCM"] for r in rows if r.get("MAR_MCM") is not None)
                if any(r.get("MAR_MCM") is not None for r in rows)
                else None
            ),
            "million m³/a",
            "Incremental MAR of each quaternary, not routed.",
        )
    )
    return out


# ---------------------------------------------------------------------------
# DWS hydrological stations (locations only)
# ---------------------------------------------------------------------------

STATION_TYPES = [
    ("River", "River flow gauges"),
    ("Reservoir", "Reservoirs (dam levels)"),
    ("Rainfall", "Meteorological - rainfall"),
    ("Evaporation", "Meteorological - evaporation"),
    ("Groundwater", "Groundwater (boreholes)"),
    ("Tidal", "Tidal / estuary"),
]
STATION_TYPE_KEYS = [k for k, _ in STATION_TYPES]
STATION_FIELDS = [
    ("STATION", "string"),
    ("NAME", "string"),
    ("TYPE", "string"),
    ("CATEGORY", "string"),
    ("WMA_NO", "int"),
    ("WMA_NAME", "string"),
    ("LAT", "double"),
    ("LON", "double"),
    ("QUAT", "string"),
    ("AREA_KM2", "double"),
    ("DESCR", "string"),
    ("START", "string"),
    ("END", "string"),
    ("OPEN", "string"),
    ("RAIN", "string"),
    ("EVAP_A", "string"),
    ("EVAP_S", "string"),
    ("FSL_MCM", "double"),
    ("PARAMS", "string"),
    ("DIST_KM", "double"),
]


def _num(text):
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return int(value) if value.is_integer() and "." not in text else value


def load_stations(path: str | Path | None = None) -> list[dict]:
    """The bundled DWS station table (one row per station)."""
    path = Path(path) if path else DATA_DIR / STATION_TABLE
    if not path.is_file():
        raise ExtractError(f"The DWS station table is missing: {path}")
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            row = {k: (v or "").strip() for k, v in r.items()}
            for k in ("WMA_NO", "LAT", "LON", "AREA_KM2", "FSL_MCM", "FSL_RL_M"):
                if k in row:
                    row[k] = _num(row[k])
            if row.get("LAT") is None or row.get("LON") is None:
                continue
            rows.append(row)
    return rows


def station_categories(row: dict) -> list[str]:
    """Selectable types a station belongs to (a meteorological station can
    be both rainfall and evaporation)."""
    t = (row.get("TYPE") or "").strip().lower()
    if t == "river":
        return ["River"]
    if t == "reservoir":
        return ["Reservoir"]
    if t == "gw":
        return ["Groundwater"]
    if t == "tidal":
        return ["Tidal"]
    if t == "met":
        cats = []
        if row.get("RAIN") == "Y":
            cats.append("Rainfall")
        if row.get("EVAP_A") == "Y" or row.get("EVAP_S") == "Y":
            cats.append("Evaporation")
        return cats or ["Rainfall"]
    return []


def filter_stations(
    rows: list[dict],
    types: Iterable[str],
    open_only: bool = False,
    keep: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Stations of the chosen types (and open ones only if asked) for which
    ``keep(row)`` is true; each kept row gets CATEGORY."""
    wanted = set(types)
    out = []
    for r in rows:
        cats = [c for c in station_categories(r) if c in wanted]
        if not cats:
            continue
        if open_only and r.get("OPEN") != "Y":
            continue
        if keep is not None and not keep(r):
            continue
        row = dict(r)
        row["CATEGORY"] = " / ".join(cats)
        out.append(row)
    out.sort(
        key=lambda r: (
            STATION_TYPE_KEYS.index(r["CATEGORY"].split(" / ")[0]),
            r.get("STATION", ""),
        )
    )
    return out


def station_counts(rows: list[dict]) -> list[tuple[str, int, int]]:
    """(type label, stations, open stations) per selectable type."""
    out = []
    for key, label in STATION_TYPES:
        sel = [r for r in rows if key in (r.get("CATEGORY") or "").split(" / ")]
        if sel:
            out.append((label, len(sel), sum(1 for r in sel if r.get("OPEN") == "Y")))
    return out


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def fmt(value, decimals: int | None = None) -> str:
    """Plain number text: no thousands separators, fixed decimals."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Y" if value else "N"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ""
        if decimals is None:
            if isinstance(value, int) or float(value).is_integer():
                return str(int(value))
            return f"{value:.6g}" if abs(value) < 1e15 else str(value)
        return f"{value:.{decimals}f}"
    return str(value)


def write_csv(path: str | Path, header: list[str], rows: Iterable[list]) -> str:
    """CSV in UTF-8 with BOM (Excel opens it with the right characters)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
    return str(path)


def write_wr2012_csv(path, rows: list[dict]) -> str:
    header = [f"{f.name} ({f.unit})" if f.unit else f.name for f in WR_FIELDS]
    return write_csv(
        path,
        header,
        ([fmt(r.get(f.name), f.decimals) for f in WR_FIELDS] for r in rows),
    )


SUMMARY_DECIMALS = {"km²": 2, "mm/a": 1, "%": 1, "million m³/a": 3}


def summary_text(s: SummaryRow) -> str:
    """Value text of a summary row, decimals by unit."""
    if isinstance(s.value, float):
        return fmt(s.value, SUMMARY_DECIMALS.get(s.unit, 2))
    return fmt(s.value)


def write_summary_csv(path, summary: list[SummaryRow]) -> str:
    return write_csv(
        path,
        ["Item", "Value", "Unit", "Note"],
        ([s.item, summary_text(s), s.unit, s.note] for s in summary),
    )


def write_stations_csv(path, rows: list[dict]) -> str:
    names = [n for n, _ in STATION_FIELDS]
    decimals = {"LAT": 6, "LON": 6, "AREA_KM2": 1, "FSL_MCM": 3, "DIST_KM": 2}
    return write_csv(
        path, names, ([fmt(r.get(n), decimals.get(n)) for n in names] for r in rows)
    )
