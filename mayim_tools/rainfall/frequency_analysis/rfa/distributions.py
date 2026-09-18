"""
GEV and Gumbel: parameter estimation (L-moments and MLE) and quantile
functions.

Parameterization: Hosking's (Location Xi, Scale Alpha, Shape Kappa),
matching both the user's own terminology and the WRC K5/1060
methodology already used elsewhere in this plugin suite.
    kappa > 0: upper-bounded tail
    kappa < 0: unbounded/heavy tail
    kappa = 0: Gumbel (the GEV's limiting case)

Quantile function (Hosking & Wallis 1997, eq. 3.3), for F = the
NON-exceedance probability (i.e. F = 1 - AEP):
    x(F) = xi + (alpha/kappa) * [1 - (-ln F)^kappa]   for kappa != 0
    x(F) = xi - alpha * ln(-ln F)                      for kappa == 0

GEV L-moment fitting uses Hosking's approximation for kappa from the
L-skewness t3 (Hosking & Wallis 1997, eq. 3.8-3.9), accurate to about
9 decimal places for |kappa| < 0.5 - the practically relevant range for
rainfall frequency analysis.

MLE fitting uses scipy.stats.genextreme.fit(). scipy's shape parameter
`c` maps DIRECTLY to Hosking's kappa with NO sign flip - verified two
ways: (1) scipy's own PDF formula, f(x,c) = exp(-(1-cx)^(1/c))*(1-cx)^(1/c-1)
for x <= 1/c, c > 0, has the exact same functional form as Hosking's
CDF F(y) = exp(-(1-ky)^(1/k)) with c playing the role of kappa
directly (both positive = upper-bounded, at x=1/c and x=1/kappa
respectively); (2) empirically confirmed in tests/test_core.py by
fitting a large synthetic sample generated with a known scipy `c` via
both this module's L-moment method and scipy's own MLE, and checking
they recover the same sign. (scipy's own documentation warning about
"opposite sign conventions" refers to the discrepancy against the
Coles-textbook xi/gamma convention, not against Hosking's kappa - the
two are easy to conflate, which is exactly why this was checked two
ways rather than taken on faith from a single source.)
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import gammaln

from .lmoments import sample_l_moments

EULER_MASCHERONI = 0.5772156649015329


def fit_gev_lmoments(l1: float, l2: float, t3: float) -> tuple[float, float, float]:
    """Returns (xi, alpha, kappa) via Hosking's L-moment approximation."""
    c = 2.0 / (3.0 + t3) - np.log(2) / np.log(3)
    kappa = 7.8590 * c + 2.9554 * c**2

    if abs(kappa) < 1e-8:
        # numerically indistinguishable from Gumbel - fall back to the
        # exact Gumbel closed form rather than dividing by ~0.
        xi, alpha = fit_gumbel_lmoments(l1, l2)
        return xi, alpha, 0.0

    g = gamma_fn(1 + kappa)
    alpha = (l2 * kappa) / ((1 - 2 ** (-kappa)) * g)
    xi = l1 - alpha * (1 - g) / kappa
    return float(xi), float(alpha), float(kappa)


def fit_gumbel_lmoments(l1: float, l2: float) -> tuple[float, float]:
    """Returns (xi, alpha). Gumbel is GEV's kappa=0 limiting case."""
    alpha = l2 / np.log(2)
    xi = l1 - alpha * EULER_MASCHERONI
    return float(xi), float(alpha)


def fit_gev_mle(data) -> tuple[float, float, float]:
    """Returns (xi, alpha, kappa) via scipy's MLE fit. See module
    docstring for the sign-convention verification.

    IMPORTANT: always seeds scipy's optimizer with the L-moment fit as
    the initial guess. Empirically confirmed (see tests/test_core.py)
    that scipy.stats.genextreme.fit() can silently converge to a
    wrong-signed, wildly incorrect result from its own default initial
    guess on perfectly ordinary GEV data - not a rare edge case, this
    happened on the very first test sample tried while building this
    module. Seeding with the L-moment fit (itself fast and robust)
    fixes this reliably. If the MLE result still ends up far from the
    L-moment fit despite the good starting point, that's flagged as a
    warning rather than silently trusted either way.
    """
    from scipy.stats import genextreme

    data = np.asarray(data, dtype=float)
    lm = sample_l_moments(data)
    if lm["t3"] is None:
        raise ValueError("Need at least 3 values to fit GEV.")
    xi0, alpha0, kappa0 = fit_gev_lmoments(lm["l1"], lm["l2"], lm["t3"])

    c, loc, scale = genextreme.fit(data, kappa0, loc=xi0, scale=alpha0)

    warnings = []
    if abs(c - kappa0) > 0.15 or abs(scale - alpha0) / alpha0 > 0.3:
        warnings.append(
            f"MLE fit (kappa={c:.4f}) diverges substantially from the L-moment fit "
            f"(kappa={kappa0:.4f}) despite seeding the optimizer with it - treat the MLE "
            "result with caution, the L-moment fit is likely more reliable here."
        )
    return float(loc), float(scale), float(c), warnings


def gev_quantile(F: float, xi: float, alpha: float, kappa: float) -> float:
    if kappa is None or abs(kappa) < 1e-8:
        return xi - alpha * np.log(-np.log(F))
    return xi + (alpha / kappa) * (1 - (-np.log(F)) ** kappa)


def gev_cdf(x: float, xi: float, alpha: float, kappa: float) -> float:
    if kappa is None or abs(kappa) < 1e-8:
        return float(np.exp(-np.exp(-(x - xi) / alpha)))
    z = 1 - kappa * (x - xi) / alpha
    if z <= 0:
        return 1.0 if kappa > 0 else 0.0  # outside the distribution's support
    return float(np.exp(-(z ** (1 / kappa))))


def gev_tau3_tau4_exact(kappa: float) -> tuple[float, float]:
    """Exact (no approximation) GEV L-skewness/L-kurtosis as a function
    of kappa (Hosking & Wallis 1997). Verified empirically against
    large synthetic samples across multiple kappa values before being
    relied on - see dev/README.md."""
    if abs(kappa) < 1e-8:
        # Gumbel limit - evaluate at a tiny kappa rather than deriving
        # a separate symbolic limit, avoiding a second formula to
        # transcribe incorrectly.
        kappa = 1e-8
    t3 = 2 * (1 - 3 ** (-kappa)) / (1 - 2 ** (-kappa)) - 3
    t4 = (
        5 * (1 - 4 ** (-kappa)) - 10 * (1 - 3 ** (-kappa)) + 6 * (1 - 2 ** (-kappa))
    ) / (1 - 2 ** (-kappa))
    return t3, t4


GUMBEL_TAU3, GUMBEL_TAU4 = gev_tau3_tau4_exact(
    0.0
)  # fixed point - no free shape parameter


def fit_distribution(
    data, distribution: str = "GEV", method: str = "L-moments"
) -> dict:
    """Convenience wrapper covering all four distributions. Returns a
    dict with xi/alpha/kappa (kappa None for Gumbel) plus the L-moments
    used (for transparency/audit) and any warnings. LP3's dict also
    carries mu/sigma/gamma (the standard log-space reporting
    convention) in addition to the xi/beta/alpha Pearson reparameterization
    mapped onto the same xi/alpha/kappa field names as the other three
    distributions for schema consistency."""
    distribution = distribution.upper()
    method = method.lower()

    if distribution == "LP3":
        fit = fit_lp3_lmoments(data)
        return {
            "xi": fit["xi"],
            "alpha": fit["beta"],
            "kappa": fit["alpha"],
            "l_moments": fit["l_moments"],
            "warnings": [],
            "extra": {
                "mu_log10": fit["mu"],
                "sigma_log10": fit["sigma"],
                "gamma_log10": fit["gamma"],
            },
        }

    lm = sample_l_moments(data)

    if distribution == "GUMBEL":
        xi, alpha = fit_gumbel_lmoments(lm["l1"], lm["l2"])
        return {
            "xi": xi,
            "alpha": alpha,
            "kappa": None,
            "l_moments": lm,
            "warnings": [],
            "extra": {},
        }

    if distribution == "GLO":
        if lm["t3"] is None:
            raise ValueError("Need at least 3 values to fit GLO.")
        xi, alpha, kappa = fit_glo_lmoments(lm["l1"], lm["l2"], lm["t3"])
        return {
            "xi": xi,
            "alpha": alpha,
            "kappa": kappa,
            "l_moments": lm,
            "warnings": [],
            "extra": {},
        }

    if distribution == "GEV":
        if method.startswith("l"):
            if lm["t3"] is None:
                raise ValueError("Need at least 3 values to fit GEV via L-moments.")
            xi, alpha, kappa = fit_gev_lmoments(lm["l1"], lm["l2"], lm["t3"])
            return {
                "xi": xi,
                "alpha": alpha,
                "kappa": kappa,
                "l_moments": lm,
                "warnings": [],
                "extra": {},
            }
        elif method.startswith("m"):
            xi, alpha, kappa, mle_warnings = fit_gev_mle(data)
            return {
                "xi": xi,
                "alpha": alpha,
                "kappa": kappa,
                "l_moments": lm,
                "warnings": mle_warnings,
                "extra": {},
            }
        else:
            raise ValueError(
                f"Unknown method {method!r} - expected 'L-moments' or 'MLE'."
            )

    raise ValueError(
        f"Unknown distribution {distribution!r} - expected 'GEV', 'Gumbel', 'GLO', or 'LP3'."
    )


def quantile_for(
    distribution: str,
    F: float,
    xi: float,
    alpha: float,
    kappa,
    extra: dict | None = None,
) -> float:
    """Dispatches to the correct quantile function by distribution name."""
    distribution = distribution.upper()
    if distribution == "GEV" or distribution == "GUMBEL":
        return gev_quantile(F, xi, alpha, kappa if kappa is not None else 0.0)
    if distribution == "GLO":
        return glo_quantile(F, xi, alpha, kappa if kappa is not None else 0.0)
    if distribution == "LP3":
        extra = extra or {}
        return lp3_quantile(
            F, extra["mu_log10"], extra["sigma_log10"], extra["gamma_log10"]
        )
    raise ValueError(f"Unknown distribution {distribution!r}")


# ----------------------------------------------------------------------
# Generalized Logistic (GLO) - exact closed-form L-moment relationships,
# no approximation needed (Hosking & Wallis 1997). Verified empirically
# against Hosking's own quantile function with a large synthetic sample
# (see tests/test_core.py) before being relied on here.
# ----------------------------------------------------------------------


def fit_glo_lmoments(l1: float, l2: float, t3: float) -> tuple[float, float, float]:
    """Returns (xi, alpha, kappa). kappa = -t3 exactly.

    l2 = alpha * (kappa*pi) / sin(kappa*pi), so alpha = l2 * sin(kappa*pi) / (kappa*pi)
    - NOTE: an earlier version of this function had this reciprocal
    backwards (alpha = l2*kappa*pi/sin(kappa*pi)). That version passed
    a quick sanity check but failed a proper large-sample parameter-
    recovery test (kappa recovered correctly since its formula is
    separate and simple, but alpha/xi were off by up to 36% at
    |kappa|=0.3, growing with |kappa| - the signature of a wrong
    functional form, not just noise). Fixed and re-verified: the
    corrected formula recovers alpha and xi to within Monte Carlo
    sampling noise (<0.5%) across kappa in [-0.3, 0.3] - see
    tests/test_core.py and dev/README.md."""
    kappa = -t3
    if abs(kappa) < 1e-8:
        # logistic (kappa=0) closed form
        alpha = l2
        xi = l1
        return float(xi), float(alpha), 0.0
    sinkp = np.sin(kappa * np.pi)
    alpha = l2 * sinkp / (kappa * np.pi)
    xi = l1 - alpha * (1.0 / kappa - np.pi / sinkp)
    return float(xi), float(alpha), float(kappa)


def glo_quantile(F: float, xi: float, alpha: float, kappa: float) -> float:
    if kappa is None or abs(kappa) < 1e-8:
        return xi - alpha * np.log((1 - F) / F)
    return xi + (alpha / kappa) * (1 - ((1 - F) / F) ** kappa)


def glo_cdf(x: float, xi: float, alpha: float, kappa: float) -> float:
    if kappa is None or abs(kappa) < 1e-8:
        y = (x - xi) / alpha
        return float(1 / (1 + np.exp(-y)))
    z = 1 - kappa * (x - xi) / alpha
    if z <= 0:
        return 1.0 if kappa > 0 else 0.0
    y = -np.log(z) / kappa
    return float(1 / (1 + np.exp(-y)))


def glo_tau3_tau4_exact(kappa: float) -> tuple[float, float]:
    """Exact (no approximation): tau3=-kappa, tau4=(1+5*kappa^2)/6."""
    return -kappa, (1 + 5 * kappa**2) / 6


# ----------------------------------------------------------------------
# Log-Pearson Type III (LP3) - Pearson Type III fitted to log10-
# transformed data, matching USGS Bulletin 17C / standard hydrological
# convention. Shape parameter (Pearson's "alpha", degrees-of-freedom-
# like) estimated via Hosking's rational-function approximation from
# L-skewness (Hosking & Wallis 1997, pp. 201-202) - there is no simple
# closed form, confirmed by multiple independent sources. Quantile
# evaluation delegates to scipy.stats.pearson3's well-tested numerical
# implementation, given OUR OWN L-moment-derived (mean, std, skew) in
# log space - scipy is used here only as a numerical evaluator for a
# generic, well-known distribution, not as a wrapped hydrology-specific
# fitting algorithm (the actual parameter ESTIMATION, from L-moments,
# is this module's own clean-room implementation).
# ----------------------------------------------------------------------


def _pearson3_alpha_from_tau3(tau3: float) -> float | None:
    """Hosking's rational-function approximation for the Pearson III
    shape parameter alpha from |L-skewness|. Returns None at tau3=0
    (the degenerate Normal-distribution case, handled separately)."""
    at3 = abs(tau3)
    if at3 < 1e-6:
        return None
    if at3 < 1.0 / 3.0:
        z = 3 * np.pi * at3**2
        return (1 + 0.2906 * z) / (z + 0.1882 * z**2 + 0.0442 * z**3)
    z = 1 - at3
    return (0.36067 * z - 0.59567 * z**2 + 0.25361 * z**3) / (
        1 - 2.78861 * z + 2.56096 * z**2 - 0.77045 * z**3
    )


def fit_pearson3_lmoments(
    l1: float, l2: float, t3: float
) -> tuple[float, float, float]:
    """Returns (mu, sigma, gamma) - the log-space MEAN, STANDARD
    DEVIATION, and SKEWNESS (Pearson's "moment" parameterization,
    matching standard LP3 reporting convention, e.g. USGS practice).
    Fits directly to the (log-transformed) sample's L-moments passed
    in - the caller is responsible for the log10 transform (see
    fit_lp3_lmoments below, which does this end-to-end from raw data)."""
    alpha = _pearson3_alpha_from_tau3(t3)
    if alpha is None:
        # tau3 ~ 0: degenerate to Normal distribution
        return float(l1), float(l2 * np.sqrt(np.pi)), 0.0
    gamma = 2 * alpha**-0.5 * np.sign(t3)
    # Gamma(alpha)/Gamma(alpha+0.5) computed via log-gamma to avoid
    # overflow: Gamma(alpha) overflows for alpha > ~171 (a real case,
    # not just a theoretical edge - happens whenever L-skewness is
    # close to zero, since alpha ~ 4/gamma^2 grows without bound as
    # skewness -> 0). Found via a RuntimeWarning on real synthetic
    # test data, not a hypothetical - see tests/test_core.py.
    log_ratio = gammaln(alpha) - gammaln(alpha + 0.5)
    sigma = l2 * np.sqrt(np.pi) * np.sqrt(alpha) * np.exp(log_ratio)
    mu = l1
    return float(mu), float(sigma), float(gamma)


def fit_lp3_lmoments(data) -> dict:
    """End-to-end LP3 fit from raw (untransformed) annual maxima.
    Returns mu/sigma/gamma (log10 space) plus the Pearson (xi, beta,
    alpha) reparameterization for schema consistency with GEV/GLO's
    location/scale/shape reporting."""
    data = np.asarray(data, dtype=float)
    if np.any(data <= 0):
        raise ValueError(
            "LP3 requires strictly positive values (cannot take log of zero/negative depth)."
        )
    log_data = np.log10(data)
    lm = sample_l_moments(log_data)
    if lm["t3"] is None:
        raise ValueError("Need at least 3 values to fit LP3.")
    mu, sigma, gamma = fit_pearson3_lmoments(lm["l1"], lm["l2"], lm["t3"])

    if abs(gamma) < 1e-8:
        xi, beta, palpha = mu, sigma, None
    else:
        palpha = 4.0 / gamma**2
        beta = 0.5 * sigma * abs(gamma)
        xi = mu - 2 * sigma / gamma

    return {
        "mu": mu,
        "sigma": sigma,
        "gamma": gamma,
        "xi": xi,
        "beta": beta,
        "alpha": palpha,
        "l_moments": lm,
    }


def lp3_quantile(F: float, mu: float, sigma: float, gamma: float) -> float:
    """Quantile in ORIGINAL (untransformed) units - exponentiates the
    log-space Pearson III quantile back via 10**y."""
    from scipy.stats import pearson3

    if abs(gamma) < 1e-8:
        from scipy.stats import norm

        y = norm.ppf(F, loc=mu, scale=sigma)
    else:
        y = pearson3.ppf(F, skew=gamma, loc=mu, scale=sigma)
    return float(10**y)


_PT3_TAU4_CACHE: dict = {}


def pt3_tau4_from_tau3(tau3: float, n_mc: int = 200_000, seed: int = 0) -> float:
    """Monte Carlo estimate of Pearson III's theoretical L-kurtosis
    given a target L-skewness - no closed form exists (confirmed by
    multiple independent sources). Cached per rounded tau3 value since
    this is called repeatedly (once per duration) in the ratio-diagram
    diagnostic. Validated at tau3=0 against the known exact Normal-
    distribution tau4 constant (~0.1226) before being relied on - see
    dev/README.md for the validation numbers.
    """
    key = round(tau3, 3)
    if key in _PT3_TAU4_CACHE:
        return _PT3_TAU4_CACHE[key]

    from scipy.stats import norm, pearson3

    alpha = _pearson3_alpha_from_tau3(tau3)
    if alpha is None:
        sample = norm.rvs(size=n_mc, random_state=seed)
    else:
        gamma = 2 * alpha**-0.5 * np.sign(tau3)
        sample = pearson3.rvs(skew=gamma, loc=0, scale=1, size=n_mc, random_state=seed)
    t4 = sample_l_moments(sample)["t4"]
    _PT3_TAU4_CACHE[key] = t4
    return t4
