"""Data facts for "Extract: Land Cover" - no QGIS, no GDAL.

ESA WorldCover 10 m (Zanaga et al. 2021, 2022): 2020 (v100) and 2021
(v200), 3 x 3 degree cloud-optimised GeoTIFF tiles in EPSG:4326 on the
public Amazon S3 bucket ``esa-worldcover`` (eu-central-1), one "Map" layer
per tile with the class code (nodata 0). Tile names give the lower-left
corner at multiples of 3 degrees, e.g. S27E030 covers 27-24 S, 30-33 E.
Key pattern (from ESA's own esa-worldcover-datasets repository):
    v200/2021/map/ESA_WorldCover_10m_2021_v200_S27E030_Map.tif
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PRODUCT = "ESA WorldCover"
PRODUCT_LONG = "ESA WorldCover 10 m"
BUCKET = "esa-worldcover"
HOST = "esa-worldcover.s3.eu-central-1.amazonaws.com"
RESOLUTION_M = 10
TILE_DEG = 3
NODATA = 0
LICENCE = "CC BY 4.0"
# A tile that certainly exists (used to tell "no tiles here" from "server
# not reachable").
KNOWN_TILE = "N51E003"


@dataclass(frozen=True)
class Edition:
    year: int
    version: str
    algorithm: str
    accuracy: str
    citation: str
    attribution: str


EDITIONS = [
    Edition(
        2021,
        "v200",
        "v200 (improved algorithm)",
        "76.7 % overall accuracy (global); 76.5 % for Africa",
        "Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, W., "
        "Brockmann, C., Kirches, G., Wevers, J., Cartus, O., Santoro, M., "
        "Fritz, S., Lesiv, M., Herold, M., Tsendbazar, N.E., Xu, P., Ramoino, "
        "F., Arino, O. (2022). ESA WorldCover 10 m 2021 v200. Zenodo. "
        "https://doi.org/10.5281/zenodo.7254221",
        "(c) ESA WorldCover project 2021 / Contains modified Copernicus "
        "Sentinel data (2021) processed by ESA WorldCover consortium",
    ),
    Edition(
        2020,
        "v100",
        "v100 (first algorithm)",
        "74.4 % overall accuracy (global)",
        "Zanaga, D., Van De Kerchove, R., De Keersmaecker, W., Souverijns, N., "
        "Brockmann, C., Quast, R., Wevers, J., Grosu, A., Paccini, A., Vergnaud, "
        "S., Cartus, O., Santoro, M., Fritz, S., Georgieva, I., Lesiv, M., "
        "Carter, S., Herold, M., Li, L., Tsendbazar, N.E., Ramoino, F., Arino, "
        "O. (2021). ESA WorldCover 10 m 2020 v100. Zenodo. "
        "https://doi.org/10.5281/zenodo.5571936",
        "(c) ESA WorldCover project 2020 / Contains modified Copernicus "
        "Sentinel data (2020) processed by ESA WorldCover consortium",
    ),
]
EDITION_BY_YEAR = {e.year: e for e in EDITIONS}
DEFAULT_YEARS = [2021]


@dataclass(frozen=True)
class LandClass:
    code: int
    name: str
    colour: str
    note: str = ""


CLASSES = [
    LandClass(10, "Tree cover", "#006400"),
    LandClass(20, "Shrubland", "#ffbb22", "one of the least accurate classes"),
    LandClass(30, "Grassland", "#ffff4c"),
    LandClass(40, "Cropland", "#f096ff"),
    LandClass(50, "Built-up", "#fa0000", "buildings and roads; no imperviousness"),
    LandClass(60, "Bare / sparse vegetation", "#b4b4b4"),
    LandClass(70, "Snow and ice", "#f0f0f0"),
    LandClass(80, "Permanent water bodies", "#0064c8"),
    LandClass(90, "Herbaceous wetland", "#0096a0", "one of the least accurate classes"),
    LandClass(95, "Mangroves", "#00cf75"),
    LandClass(100, "Moss and lichen", "#fae6a0", "one of the least accurate classes"),
]
CLASS_BY_CODE = {c.code: c for c in CLASSES}


def class_name(code) -> str:
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return ""
    c = CLASS_BY_CODE.get(int(code))
    return c.name if c else f"Unknown ({int(code)})"


# ----------------------------------------------------------------------
# Tiles and paths
# ----------------------------------------------------------------------


def tile_name(lat0: int, lon0: int) -> str:
    """Name of the tile whose lower-left corner is (lat0, lon0)."""
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def tile_of(lon: float, lat: float) -> str:
    """Tile containing a point (a point on a tile edge belongs to the tile
    to its north / east)."""
    lat0 = int(math.floor(lat / TILE_DEG)) * TILE_DEG
    lon0 = int(math.floor(lon / TILE_DEG)) * TILE_DEG
    lat0 = min(max(lat0, -90), 90 - TILE_DEG)
    lon0 = min(max(lon0, -180), 180 - TILE_DEG)
    return tile_name(lat0, lon0)


def tiles_for_bounds(bounds_ll, pad_deg: float = 0.01) -> list[str]:
    """Every tile that touches a lon/lat box (padded slightly so the output
    grid's buffer cells are covered)."""
    lon0, lat0, lon1, lat1 = bounds_ll
    lon0, lat0 = max(lon0 - pad_deg, -180.0), max(lat0 - pad_deg, -90.0)
    lon1, lat1 = min(lon1 + pad_deg, 180.0), min(lat1 + pad_deg, 90.0)
    la = int(math.floor(lat0 / TILE_DEG)) * TILE_DEG
    la_stop = int(math.ceil(lat1 / TILE_DEG)) * TILE_DEG
    lo = int(math.floor(lon0 / TILE_DEG)) * TILE_DEG
    lo_stop = int(math.ceil(lon1 / TILE_DEG)) * TILE_DEG
    names = []
    for lat in range(la, max(la_stop, la + TILE_DEG), TILE_DEG):
        for lon in range(lo, max(lo_stop, lo + TILE_DEG), TILE_DEG):
            if -90 <= lat < 90 and -180 <= lon < 180:
                names.append(tile_name(lat, lon))
    return names


def tile_key(year: int, tile: str) -> str:
    ver = EDITION_BY_YEAR[year].version
    return f"{ver}/{year}/map/ESA_WorldCover_10m_{year}_{ver}_{tile}_Map.tif"


def cog_path(year: int, tile: str) -> str:
    return f"/vsicurl/https://{HOST}/{tile_key(year, tile)}"


def s3_path(year: int, tile: str) -> str:
    return f"/vsis3/{BUCKET}/{tile_key(year, tile)}"


ROUTES = [("COG (https)", cog_path), ("S3 (unsigned)", s3_path)]


def output_file_name(year: int) -> str:
    return f"worldcover_{year}_map.tif"


def band_description(year: int) -> str:
    return f"landcover_worldcover_{year}_{EDITION_BY_YEAR[year].version}_10m (class)"
