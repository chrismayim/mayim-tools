"""OpenLandMap-soildb layer catalogue (no QGIS, no GDAL).

Built from the official layer table, openlandmap/soildb
tables/OpenLandMap_soildb_COGS.csv (checked 2026-10-05; the test suite
compares every URL this module can build against that table).

Facts that shape the tool:
- Single global cloud-optimised GeoTIFFs in EPSG:4326 on s3.opengeohub.org,
  read window-by-window with GDAL /vsicurl/ (no index file needed).
- Mean predictions at 30 m; the 68 % prediction interval (P16 / P84) only
  at 120 m (a 120 m mean is also published).
- Depths 0-30, 30-60, 60-100 cm.
- Periods: SOC, SOC density and pH for 2000-2005 ... 2020-2022; texture
  for 2020-2022 only (static); bulk density 2020-2022 at 30 m, all
  periods at 120 m.
- Texture follows ISO 11277: silt 2-63 um, sand 63 um - 2 mm (USDA uses
  50 um). Convert before using the USDA texture triangle.
- Values are stored scaled (e.g. SOC x10); the scale is also written in
  each file's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

HOST = "s3.opengeohub.org"
PRODUCT = "OpenLandMap-soildb"

PERIODS: dict[str, str] = {
    "2000-2005": "20000101_20051231",
    "2005-2010": "20050101_20101231",
    "2010-2015": "20100101_20151231",
    "2015-2020": "20150101_20201231",
    "2020-2022": "20200101_20221231",
}
ALL_PERIODS = tuple(PERIODS)
LATEST_PERIOD = "2020-2022"

DEPTHS: tuple[tuple[str, int, int], ...] = (
    ("0-30cm", 0, 30),
    ("30-60cm", 30, 60),
    ("60-100cm", 60, 100),
)
DEPTH_BY_LABEL = {d[0]: d for d in DEPTHS}


@dataclass(frozen=True)
class OlmStatistic:
    code: str  # tool code
    file_code: str  # in the file name
    resolution_m: int
    tag: str  # in output file / band names
    label: str


STATISTICS: tuple[OlmStatistic, ...] = (
    OlmStatistic("mean30", "m", 30, "mean_30m", "Mean (30 m)"),
    OlmStatistic("mean120", "m", 120, "mean_120m", "Mean (120 m)"),
    OlmStatistic("p16", "p16", 120, "p16_120m", "P16, lower 68 % limit (120 m)"),
    OlmStatistic("p84", "p84", 120, "p84_120m", "P84, upper 68 % limit (120 m)"),
)
STAT_BY_CODE = {s.code: s for s in STATISTICS}
RU68 = "RU68"
RU68_TAG = "RU68_120m"
RU68_LABEL = "Relative 68% interval width (P84-P16)/mean, 120 m"
STATS_FOR_RU = ("p16", "mean120", "p84")


@dataclass(frozen=True)
class OlmVariable:
    code: str  # tool code (same names as the SoilGrids tool where possible)
    filename: str  # OpenLandMap variable-procedure name
    label: str
    units: str
    scale: float  # catalogue scaler: value = stored * scale
    folder: str
    version: str
    periods_30m: tuple[str, ...]
    periods_120m: tuple[str, ...]
    static: bool  # one period only, valid for any requested period
    default: bool
    purpose: str


_PROPS = ("global_soil_props_v20250204_mosaics", "v20250204")
_TEXTURE = ("global_soil_props_v20250523", "v20250523")
_RECENT = ("2020-2022",)

VARIABLES: tuple[OlmVariable, ...] = (
    OlmVariable(
        "sand",
        "sand.tot_iso.11277.2020.wpct",
        "Sand (0.063-2 mm, ISO 11277)",
        "%",
        1,
        *_TEXTURE,
        _RECENT,
        _RECENT,
        True,
        True,
        "texture; PTFs (convert ISO to USDA limits first)",
    ),
    OlmVariable(
        "silt",
        "silt.tot_iso.11277.2020.wpct",
        "Silt (0.002-0.063 mm, ISO 11277)",
        "%",
        1,
        *_TEXTURE,
        _RECENT,
        _RECENT,
        True,
        True,
        "texture; PTFs (convert ISO to USDA limits first)",
    ),
    OlmVariable(
        "clay",
        "clay.tot_iso.11277.2020.wpct",
        "Clay (< 0.002 mm)",
        "%",
        1,
        *_TEXTURE,
        _RECENT,
        _RECENT,
        True,
        True,
        "texture; all PTFs",
    ),
    OlmVariable(
        "soc",
        "oc_iso.10694.1995.wpml",
        "Soil organic carbon content",
        "g/kg",
        0.1,
        *_PROPS,
        ALL_PERIODS,
        ALL_PERIODS,
        False,
        True,
        "organic matter (SOC is not converted to OM here)",
    ),
    OlmVariable(
        "socd",
        "oc_iso.10694.1995.mg.cm3",
        "Soil organic carbon density",
        "kg/m3",
        0.1,
        *_PROPS,
        ALL_PERIODS,
        ALL_PERIODS,
        False,
        False,
        "carbon stocks (not needed for hydraulic properties)",
    ),
    OlmVariable(
        "bdod",
        "bd.core_iso.11272.2017.g.cm3",
        "Bulk density of the fine earth",
        "g/cm3",
        0.01,
        *_PROPS,
        _RECENT,
        ALL_PERIODS,
        False,
        True,
        "density factor; Rosetta/Wosten",
    ),
    OlmVariable(
        "phh2o",
        "ph.h2o_iso.10390.2021.index",
        "pH in water",
        "pH",
        0.1,
        *_PROPS,
        ALL_PERIODS,
        ALL_PERIODS,
        False,
        True,
        "euptf2 / Toth",
    ),
)
VARIABLE_BY_CODE = {v.code: v for v in VARIABLES}
TEXTURE_CODES = ("sand", "silt", "clay")


def resolve_period(var: str, stat: str, period: str) -> str | None:
    """Period actually read for a request, or None if not published.
    Static variables (texture) always use their single period."""
    v = VARIABLE_BY_CODE[var]
    available = (
        v.periods_30m if STAT_BY_CODE[stat].resolution_m == 30 else (v.periods_120m)
    )
    if period in available:
        return period
    if v.static:
        return available[0]
    return None


def cog_url(var: str, stat: str, depth: str, period: str, scheme: str = "https") -> str:
    """URL of one published layer (no availability check - see resolve_period)."""
    v = VARIABLE_BY_CODE[var]
    s = STAT_BY_CODE[stat]
    _, top, bottom = DEPTH_BY_LABEL[depth]
    return (
        f"{scheme}://{HOST}/global-soil/{v.folder}/{v.filename}_{s.file_code}_"
        f"{s.resolution_m}m_b{top}cm..{bottom}cm_{PERIODS[period]}_g_epsg.4326_"
        f"{v.version}.tif"
    )


def layer_label(var: str, depth: str, stat_tag: str, period: str) -> str:
    return f"{var}_{depth}_{stat_tag}_{period}"


def band_description(
    var: str, depth: str, stat_tag: str, period: str, units: str
) -> str:
    """Band description used to SELECT bands downstream (never band number)."""
    return f"{layer_label(var, depth, stat_tag, period)} ({units})"


def output_file_name(var: str, stat_tag: str, period: str) -> str:
    return f"olm_{var}_{stat_tag}_{period}.tif"
