"""Monte Carlo propagation of input uncertainty through the method ensemble
(stages 3-4). No QGIS, no GDAL.

For every cell and depth layer, each product contributes ``draws`` samples
of its inputs from the per-cell distributions (inputs.Dist):

- each input t = T(centre) + z * sigma_lo (z < 0) or z * sigma_hi (z >= 0),
  z ~ N(0, 1), back-transformed and limited to the variable's range;
- sand, silt and clay are drawn independently in log space and closed to
  100 % - independent log-normal parts followed by closure, i.e. a
  logistic-normal distribution on the simplex - so every draw is a real
  soil (Aitchison, 1986). ISO texture is then converted to USDA limits;
- organic matter OM (%) = SOC (g/kg) / 10 x the OM factor.

Every method runs on every draw. The pooled set (methods x products x draws,
each product weighted equally) gives the P5 / P50 / P95 per cell. The
variance (log10 scale for Ksat and suctions) is split by the law of total
variance into input data (within method and product), product, method and
method x product interaction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .inputs import TEXTURE, VARIABLE_SPACE, Dist, from_space, to_space
from .ptf import METHOD_BY_CODE, PARAMETERS
from .texture import ISO_TO_USDA_SILT_FACTOR, normalise_texture, usda_class

PERCENTILES = (5.0, 50.0, 95.0)
STAT_TAGS = ("P05", "P50", "P95")
N_CLASSES = 12
VARIANCE_PARTS = ("input", "product", "method", "interaction")


@dataclass
class Settings:
    draws: int = 200
    seed: int = 12345
    om_factor: float = 1.724
    methods: tuple[str, ...] = ("SR2006", "TOTH2015")
    density: bool = False
    gravel: bool = False


@dataclass
class ProductCells:
    """One product's input distributions for the cells of a chunk and one
    depth layer (1-D arrays)."""

    name: str
    texture_system: str
    dists: dict[str, Dist]


# Inputs whose ensemble median is written (rsp_inputs_P50.tif)
MEDIAN_INPUTS = ("sand", "silt", "clay", "om", "bulk_density", "gravel")

# Optional inputs: (variable in the folders, key passed to the methods)
OPTIONAL_INPUTS = (
    ("bdod", "bulk_density"),
    ("cfvo", "gravel"),
    ("phh2o", "ph"),
    ("cec", "cec"),
)


def draw_variable(var: str, d: Dist, n: int, rng) -> np.ndarray:
    """``n`` draws per cell from a split-normal in the variable's space."""
    cells = d.centre.shape[0]
    z = rng.standard_normal((n, cells), dtype=np.float32)
    sig = np.where(z < 0, d.sig_lo[np.newaxis], d.sig_hi[np.newaxis])
    t = to_space(var, d.centre)[np.newaxis] + z * sig
    return from_space(var, t).astype(np.float32)


def sample_inputs(
    p: ProductCells, n: int, rng, om_factor: float, optional=None
) -> dict:
    """Draws (n, cells) of USDA sand/silt/clay, OM (%), bulk density and
    gravel (vol %); NaN where the product has no value."""
    parts = {v: draw_variable(v, p.dists[v], n, rng) for v in TEXTURE}
    sand, silt, clay = normalise_texture(parts["sand"], parts["silt"], parts["clay"])
    if p.texture_system != "USDA":
        silt_usda = silt * ISO_TO_USDA_SILT_FACTOR
        sand = 100.0 - clay - silt_usda
        silt = silt_usda
    out = {"sand": sand, "silt": silt, "clay": clay}
    oc_pct = draw_variable("soc", p.dists["soc"], n, rng) / 10.0
    out["oc_pct"] = oc_pct
    out["om"] = oc_pct * om_factor
    for var, key in OPTIONAL_INPUTS:
        d = p.dists.get(var)
        if d is None or (optional is not None and var not in optional):
            out[key] = np.full_like(sand, np.nan)
        else:
            out[key] = draw_variable(var, d, n, rng)
    return out


def needed_optional(settings) -> set:
    """Optional inputs the selected methods and options actually use (bulk
    density is always drawn: its median is written with the inputs)."""
    need = {"bdod"}
    if settings.gravel:
        need.add("cfvo")
    if "TOTH2015" in settings.methods:
        need.update({"phh2o", "cec"})
    return need


def central_inputs(p: ProductCells, om_factor: float) -> dict:
    """The product's central soil (no sampling), USDA limits."""
    c = {v: p.dists[v].centre for v in TEXTURE}
    sand, silt, clay = normalise_texture(c["sand"], c["silt"], c["clay"])
    if p.texture_system != "USDA":
        silt_usda = silt * ISO_TO_USDA_SILT_FACTOR
        sand = 100.0 - clay - silt_usda
        silt = silt_usda
    out = {"sand": sand, "silt": silt, "clay": clay}
    out["oc_pct"] = p.dists["soc"].centre / 10.0
    out["om"] = out["oc_pct"] * om_factor
    for var, key in OPTIONAL_INPUTS:
        d = p.dists.get(var)
        out[key] = (
            np.clip(d.centre, VARIABLE_SPACE[var][2], VARIABLE_SPACE[var][3])
            if d is not None
            else np.full_like(sand, np.nan)
        )
    return out


def pooled_quantiles(flat: np.ndarray) -> np.ndarray:
    """P5 / P50 / P95 over axis 0 of (draws, cells), identical to
    np.nanpercentile(..., method="linear"), vectorised: one partition of the
    transposed (cell-contiguous) array when there is no NaN; otherwise one
    sort (NaN sorts last) with per-cell positions from the valid count."""
    xt = np.ascontiguousarray(flat.T)
    q = np.asarray(PERCENTILES) / 100.0
    nan_mask = np.isnan(xt)
    if not nan_mask.any():
        n = xt.shape[1]
        pos = q * (n - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        w = pos - lo
        part = np.partition(xt, sorted(set(lo) | set(hi)), axis=1)
        return (part[:, lo] * (1.0 - w) + part[:, hi] * w).T
    srt = np.sort(xt, axis=1)
    valid = (~nan_mask).sum(axis=1)
    pos = q[np.newaxis] * np.maximum(valid - 1, 0)[:, np.newaxis]
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(valid - 1, 0)[:, np.newaxis])
    w = pos - lo
    a = np.take_along_axis(srt, lo, axis=1)
    b = np.take_along_axis(srt, hi, axis=1)
    out = a * (1.0 - w) + b * w
    out[valid == 0] = np.nan
    return out.T


def _transform(param_code: str, x: np.ndarray) -> np.ndarray:
    log = next(p.log for p in PARAMETERS if p.code == param_code)
    if log:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log10(np.maximum(x, 1e-6))
    return x


def variance_split(groups: np.ndarray) -> dict[str, np.ndarray]:
    """Law-of-total-variance split for draws arranged (methods, products,
    draws, cells). Returns per-cell variances: input (mean within-group
    variance), product and method main effects, interaction, total."""
    if not np.isnan(groups).any():
        mu = groups.mean(axis=2)  # (M, P, cells)
        within = groups.var(axis=2).mean(axis=(0, 1))
        between = mu.reshape(-1, mu.shape[-1]).var(axis=0)
        method = mu.mean(axis=1).var(axis=0)
        product = mu.mean(axis=0).var(axis=0)
    else:
        with warnings.catch_warnings():  # all-NaN groups (missing inputs)
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(groups, axis=2)
            within = np.nanmean(np.nanvar(groups, axis=2), axis=(0, 1))
            between = np.nanvar(mu.reshape(-1, mu.shape[-1]), axis=0)
            method = np.nanvar(np.nanmean(mu, axis=1), axis=0)
            product = np.nanvar(np.nanmean(mu, axis=0), axis=0)
    inter = np.maximum(between - method - product, 0.0)
    return {
        "input": within,
        "product": product,
        "method": method,
        "interaction": inter,
        "total": within + between,
    }


@dataclass
class LayerResult:
    """Results for the cells of one chunk and one depth layer."""

    stats: dict[str, np.ndarray] = field(default_factory=dict)  # (3, cells)
    inputs_p50: dict[str, np.ndarray] = field(default_factory=dict)
    texture_class: np.ndarray | None = None
    texture_share: np.ndarray | None = None
    flags: np.ndarray | None = None
    variance: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    central: dict[str, dict] = field(default_factory=dict)  # product -> values


def run_layer(
    products: list[ProductCells], settings: Settings, rng, topsoil: bool = False
) -> LayerResult:
    """Monte Carlo for one chunk of cells and one depth layer. ``topsoil``
    is passed to methods with a topsoil/subsoil term (Tóth et al., 2015)."""
    n = settings.draws
    options = {
        "density": settings.density,
        "gravel": settings.gravel,
        "topsoil": topsoil,
        "om_factor": settings.om_factor,
    }
    methods = [METHOD_BY_CODE[c] for c in settings.methods]
    res = LayerResult()
    optional = needed_optional(settings)
    samples = [sample_inputs(p, n, rng, settings.om_factor, optional) for p in products]
    cells = samples[0]["sand"].shape[1]

    # Each parameter is pooled over the methods that define it, arranged
    # (methods, products, draws, cells).
    results = [[m.function(s, options) for s in samples] for m in methods]
    for prm in PARAMETERS:
        # A method that returns nothing for this parameter here (e.g. Tóth
        # Ksat without CEC) is left out rather than turning the pool into NaN.
        idx = [
            i
            for i, m in enumerate(methods)
            if prm.code in m.defines
            and any(
                np.isfinite(results[i][j][prm.code]).any() for j in range(len(samples))
            )
        ]
        if not idx:
            if any(prm.code in m.defines for m in methods):
                res.stats[prm.code] = np.full((len(PERCENTILES), cells), np.nan)
            continue
        arr = np.stack(
            [
                np.stack([results[i][j][prm.code] for j in range(len(samples))])
                for i in idx
            ]
        )
        flat = arr.reshape(-1, cells)
        with np.errstate(invalid="ignore"):
            res.stats[prm.code] = pooled_quantiles(flat)
            y = _transform(prm.code, arr)
            res.variance[prm.code] = variance_split(y)

    # Inputs (pooled over products): medians; texture closed again.
    pooled = {}
    for k in MEDIAN_INPUTS:
        have = [s_[k] for s_ in samples if np.isfinite(s_[k]).any()]
        pooled[k] = np.concatenate(have) if have else samples[0][k]
    med = {k: pooled_quantiles(v)[1] for k, v in pooled.items()}
    sand, silt, clay = normalise_texture(med["sand"], med["silt"], med["clay"])
    res.inputs_p50 = dict(med, sand=sand, silt=silt, clay=clay)

    # Texture class over the draws
    codes = usda_class(pooled["sand"], pooled["silt"], pooled["clay"])
    counts = np.stack([(codes == k).sum(axis=0) for k in range(1, N_CLASSES + 1)])
    total = counts.sum(axis=0)
    res.texture_class = np.where(total > 0, counts.argmax(axis=0) + 1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        res.texture_share = np.where(total > 0, counts.max(axis=0) / total, np.nan)

    # Central soils per product: validity flags and product comparison.
    flags = np.zeros(cells, dtype=np.int16)
    for p in products:
        c = central_inputs(p, settings.om_factor)
        per_method = {}
        for m in methods:
            r = m.function(c, options)
            flags |= r["flags"].astype(np.int16)
            per_method[m.code] = r
        res.central[p.name] = {"inputs": c, "methods": per_method}
    res.flags = flags
    return res
