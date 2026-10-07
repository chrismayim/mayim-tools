"""iSDAsoil facts (no QGIS, no GDAL).

iSDAsoil (Hengl et al., 2021; Miller et al., 2021): soil properties for
Africa at 30 m, predicted by two-scale ensemble machine learning from more
than 100000 African soil samples, for 0-20 and 20-50 cm (depth to bedrock
for 0-200 cm), period 2001-2017. Each property is one continent-wide
cloud-optimised GeoTIFF on a public Amazon S3 bucket (no login):

    https://isdasoil.s3.amazonaws.com/soil_data/<property>/<property>.tif

with four bands: mean 0-20 cm, mean 20-50 cm, standard deviation 0-20 cm,
standard deviation 20-50 cm (depth to bedrock: mean and standard deviation
0-200 cm). Band descriptions in the file are used when present; otherwise
this order (as published in the Google Earth Engine catalogue).

STORED VALUES (iSDA metadata):
- texture (sand, silt, clay; USDA limits) in %, as stored;
- bulk density x 100; pH x 10;
- organic carbon, stone content and effective CEC as 10 x ln(1 + x):
  back-transformed with exp(x / 10) - 1.

The standard deviation is the spread of the ensemble's individual learners
(bootstrapped 5-fold cross-validation), in the same space as the stored
mean: for the log-stored properties it is a standard deviation of ln(1 + x)
(written here as such, divided by 10), not of x itself.
"""

from __future__ import annotations

from dataclasses import dataclass

PRODUCT = "iSDAsoil"
HOST = "isdasoil.s3.amazonaws.com"
BUCKET = "isdasoil"
PERIOD = "2001-2017"
RESOLUTION_M = 30

CITATION = (
    "Hengl, T., Miller, M. A. E., Krizan, J., Shepherd, K. D., Sila, A., "
    "Kilibarda, M., Antonijevic, O., Glusica, L., Dobermann, A., Haefele, S. M., "
    "McGrath, S. P., Acquah, G. E., Collinson, J., Parente, L., Sheykhmousa, M., "
    "Saito, K., Johnson, J.-M., Chamberlin, J., Silatsa, F. B. T., Yemefack, M., "
    "Wendt, J., MacMillan, R. A., Wheeler, I. and Crouch, J. (2021). African soil "
    "properties and nutrients mapped at 30 m spatial resolution using two-scale "
    "ensemble machine learning. Scientific Reports, 11, 6130. "
    "https://doi.org/10.1038/s41598-021-85639-y"
)
CITATION_2 = (
    "Miller, M. A. E., Shepherd, K. D., Kisitu, B. and Collinson, J. (2021). "
    "iSDAsoil: The first continent-scale soil property map at 30 m resolution "
    "provides a soil information revolution for Africa. PLOS Biology, 19(11), "
    "e3001441. https://doi.org/10.1371/journal.pbio.3001441"
)
LICENCE = "CC-BY 4.0"

DEPTHS: tuple[tuple[str, int, int], ...] = (
    ("0-20cm", 0, 20),
    ("20-50cm", 20, 50),
)
DEPTH_BY_LABEL = {d[0]: d for d in DEPTHS}
BEDROCK_DEPTH = ("0-200cm", 0, 200)

STATISTICS = (("mean", "Mean"), ("sd", "Standard deviation"))


@dataclass(frozen=True)
class IsdaVariable:
    code: str  # tool code (same names as the other extract tools)
    prop: str  # iSDA property (file) name
    label: str
    units: str
    transform: str  # "none" | "scale" | "log1p"
    factor: float  # stored = value x factor ("scale"); 10 for "log1p"
    default: bool
    purpose: str


VARIABLES: tuple[IsdaVariable, ...] = (
    IsdaVariable(
        "sand",
        "sand_content",
        "Sand (0.05-2 mm, USDA)",
        "%",
        "none",
        1,
        True,
        "texture; all PTFs",
    ),
    IsdaVariable(
        "silt",
        "silt_content",
        "Silt (0.002-0.05 mm, USDA)",
        "%",
        "none",
        1,
        True,
        "texture; some PTFs",
    ),
    IsdaVariable(
        "clay",
        "clay_content",
        "Clay (< 0.002 mm)",
        "%",
        "none",
        1,
        True,
        "texture; all PTFs",
    ),
    IsdaVariable(
        "soc",
        "carbon_organic",
        "Soil organic carbon content",
        "g/kg",
        "log1p",
        10,
        True,
        "organic matter (SOC is not converted to OM here)",
    ),
    IsdaVariable(
        "bdod",
        "bulk_density",
        "Bulk density of the fine earth (< 2 mm)",
        "g/cm3",
        "scale",
        100,
        True,
        "density factor; Toth",
    ),
    IsdaVariable("phh2o", "ph", "pH in water", "pH", "scale", 10, True, "Toth Ksat"),
    IsdaVariable(
        "cfvo",
        "stone_content",
        "Stone content (coarse fragments)",
        "vol %",
        "log1p",
        10,
        True,
        "gravel correction",
    ),
    IsdaVariable(
        "ecec",
        "cation_exchange_capacity",
        "Effective cation exchange capacity (at soil pH)",
        "cmol(c)/kg",
        "log1p",
        10,
        False,
        "information only: effective CEC is not CEC at pH 7",
    ),
    IsdaVariable(
        "bedrock",
        "bedrock_depth",
        "Depth to bedrock (values of 200 mean " ">= 200 cm)",
        "cm",
        "none",
        1,
        True,
        "impermeable layer for hydrologic soil groups",
    ),
)
VARIABLE_BY_CODE = {v.code: v for v in VARIABLES}
TEXTURE_CODES = ("sand", "silt", "clay")


def cog_url(prop: str, scheme: str = "https") -> str:
    return f"{scheme}://{HOST}/soil_data/{prop}/{prop}.tif"


def s3_path(prop: str) -> str:
    return f"/vsis3/{BUCKET}/soil_data/{prop}/{prop}.tif"


def depths_of(code: str) -> tuple:
    return (BEDROCK_DEPTH,) if code == "bedrock" else DEPTHS


def default_band(code: str, stat: str, depth: str) -> int:
    """Band number in the published order (mean bands first)."""
    deps = [d[0] for d in depths_of(code)]
    i = deps.index(depth)
    return (i + 1) if stat == "mean" else (len(deps) + i + 1)


def band_from_descriptions(descriptions: list[str], stat: str, depth: str):
    """Band number from the file's band descriptions (e.g. 'mean_0_20' or
    'stdev_20_50'); None if they do not identify it."""
    top, bottom = depth[:-2].split("-")
    want = ("mean" if stat == "mean" else "stdev", f"{top}_{bottom}")
    for i, d in enumerate(descriptions, start=1):
        dl = (d or "").lower().replace("-", "_").replace(" ", "_")
        if want[0] in dl and want[1] in dl:
            return i
    return None


def to_value(raw, var: IsdaVariable, stat: str):
    """Stored value -> physical units. Means: exp(x/10) - 1 for the
    log-stored properties, x / factor for the scaled ones. Standard
    deviations: x / factor (for log-stored properties this is a standard
    deviation of ln(1 + value))."""
    import numpy as np

    raw = np.asarray(raw, dtype=np.float64)
    if var.transform == "log1p":
        return np.expm1(raw / var.factor) if stat == "mean" else raw / var.factor
    return raw / var.factor


def units_of(var: IsdaVariable, stat: str) -> str:
    if stat == "sd" and var.transform == "log1p":
        return f"ln(1+{var.units})"
    return var.units


def band_description(code: str, depth: str, stat: str, units: str) -> str:
    """Band description used to SELECT bands downstream (never band number)."""
    return f"{code}_{depth}_{stat}_30m_isda ({units})"


def output_file_name(code: str, stat: str) -> str:
    return f"isda_{code}_{stat}.tif"


def layer_label(code: str, depth: str, stat: str) -> str:
    return f"{code}_{depth}_{stat}"
