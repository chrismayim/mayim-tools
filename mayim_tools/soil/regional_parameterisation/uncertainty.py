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

from dataclasses import dataclass, field

import numpy as np

from .inputs import TEXTURE, VARIABLE_SPACE, Dist, from_space, to_space
from .ptf import METHOD_BY_CODE, PARAMETERS
from .texture import ISO_TO_USDA_SILT_FACTOR, normalise_texture, usda_class

PERCENTILES = (5.0, 50.0, 95.0)
STAT_TAGS = ("P05", "P50", "P95")
KSAT_ROBUST_RATIO = 4.0  # P95/P5 within one Ksat class (classes ~x4 apart)
TEXTURE_ROBUST_SHARE = 0.8  # modal texture class in >= 80 % of draws
N_CLASSES = 12
VARIANCE_PARTS = ("input", "product", "method", "interaction")


@dataclass
class Settings:
    draws: int = 200
    seed: int = 12345
    om_factor: float = 1.724
    methods: tuple[str, ...] = ("SR2006",)
    density: bool = False
    gravel: bool = False


@dataclass
class ProductCells:
    """One product's input distributions for the cells of a chunk and one
    depth layer (1-D arrays)."""

    name: str
    texture_system: str
    dists: dict[str, Dist]


def draw_variable(var: str, d: Dist, n: int, rng) -> np.ndarray:
    """``n`` draws per cell from a split-normal in the variable's space."""
    cells = d.centre.shape[0]
    z = rng.standard_normal((n, cells))
    sig = np.where(z < 0, d.sig_lo[np.newaxis], d.sig_hi[np.newaxis])
    t = to_space(var, d.centre)[np.newaxis] + z * sig
    return from_space(var, t)


def sample_inputs(p: ProductCells, n: int, rng, om_factor: float) -> dict:
    """Draws (n, cells) of USDA sand/silt/clay, OM (%), bulk density and
    gravel (vol %); NaN where the product has no value."""
    parts = {v: draw_variable(v, p.dists[v], n, rng) for v in TEXTURE}
    sand, silt, clay = normalise_texture(parts["sand"], parts["silt"], parts["clay"])
    if p.texture_system != "USDA":
        silt_usda = silt * ISO_TO_USDA_SILT_FACTOR
        sand = 100.0 - clay - silt_usda
        silt = silt_usda
    out = {"sand": sand, "silt": silt, "clay": clay}
    out["om"] = draw_variable("soc", p.dists["soc"], n, rng) / 10.0 * om_factor
    for var, key in (("bdod", "bulk_density"), ("cfvo", "gravel")):
        d = p.dists.get(var)
        if d is None:
            out[key] = np.full_like(sand, np.nan)
        else:
            out[key] = draw_variable(var, d, n, rng)
    return out


def central_inputs(p: ProductCells, om_factor: float) -> dict:
    """The product's central soil (no sampling), USDA limits."""
    c = {v: p.dists[v].centre for v in TEXTURE}
    sand, silt, clay = normalise_texture(c["sand"], c["silt"], c["clay"])
    if p.texture_system != "USDA":
        silt_usda = silt * ISO_TO_USDA_SILT_FACTOR
        sand = 100.0 - clay - silt_usda
        silt = silt_usda
    out = {"sand": sand, "silt": silt, "clay": clay}
    out["om"] = p.dists["soc"].centre / 10.0 * om_factor
    for var, key in (("bdod", "bulk_density"), ("cfvo", "gravel")):
        d = p.dists.get(var)
        out[key] = (
            np.clip(d.centre, VARIABLE_SPACE[var][2], VARIABLE_SPACE[var][3])
            if d is not None
            else np.full_like(sand, np.nan)
        )
    return out


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
    mu = groups.mean(axis=2)  # (M, P, cells)
    within = groups.var(axis=2).mean(axis=(0, 1))
    between = mu.reshape(-1, mu.shape[-1]).var(axis=0)
    method = mu.mean(axis=1).var(axis=0)
    product = mu.mean(axis=0).var(axis=0)
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
    robust: np.ndarray | None = None  # 0 none, 1 texture, 2 Ksat, 3 both
    flags: np.ndarray | None = None
    variance: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    central: dict[str, dict] = field(default_factory=dict)  # product -> values


def run_layer(products: list[ProductCells], settings: Settings, rng) -> LayerResult:
    """Monte Carlo for one chunk of cells and one depth layer."""
    n = settings.draws
    options = {"density": settings.density, "gravel": settings.gravel}
    methods = [METHOD_BY_CODE[c] for c in settings.methods]
    res = LayerResult()
    samples = [sample_inputs(p, n, rng, settings.om_factor) for p in products]
    cells = samples[0]["sand"].shape[1]

    # Outputs arranged (methods, products, draws, cells)
    out = {
        prm.code: np.empty((len(methods), len(products), n, cells))
        for prm in PARAMETERS
    }
    for i, m in enumerate(methods):
        for j, s in enumerate(samples):
            r = m.function(s, options)
            for prm in PARAMETERS:
                out[prm.code][i, j] = r[prm.code]

    for prm in PARAMETERS:
        arr = out[prm.code]
        flat = arr.reshape(-1, cells)
        with np.errstate(invalid="ignore"):
            res.stats[prm.code] = np.percentile(flat, PERCENTILES, axis=0)
            y = _transform(prm.code, arr)
            res.variance[prm.code] = variance_split(y)

    # Inputs (pooled over products): medians; texture closed again.
    pooled = {k: np.concatenate([s[k] for s in samples]) for k in samples[0]}
    med = {k: np.median(v, axis=0) for k, v in pooled.items()}
    sand, silt, clay = normalise_texture(med["sand"], med["silt"], med["clay"])
    res.inputs_p50 = dict(med, sand=sand, silt=silt, clay=clay)

    # Texture class over the draws
    codes = usda_class(pooled["sand"], pooled["silt"], pooled["clay"])
    counts = np.stack([(codes == k).sum(axis=0) for k in range(1, N_CLASSES + 1)])
    total = counts.sum(axis=0)
    res.texture_class = np.where(total > 0, counts.argmax(axis=0) + 1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        res.texture_share = np.where(total > 0, counts.max(axis=0) / total, np.nan)

    p5, _, p95 = res.stats["ksat"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ksat_ok = (p95 / p5) <= KSAT_ROBUST_RATIO
    tex_ok = res.texture_share >= TEXTURE_ROBUST_SHARE
    res.robust = tex_ok.astype(np.int16) + 2 * ksat_ok.astype(np.int16)
    res.robust = np.where(total > 0, res.robust, -1)

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
