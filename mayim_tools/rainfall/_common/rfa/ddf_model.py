"""
Duration-consistent DDF model.

Fitting a distribution independently to each duration's annual maxima
(and even choosing a different distribution per duration) produces DDF
tables whose curves can cross: a longer duration can end up with a
smaller design depth than a shorter one at the same return period, which
is physically impossible. Published DDF/IFD products avoid this by
treating all durations as one consistent family (e.g. Smithers & Schulze
2003's scale-invariant approach for South Africa; ARR 2016 IFDs), and
the literature offers explicit duration-consistent models (Koutsoyiannis
et al. 1998; Overeem et al. 2008) and post-hoc consistency corrections
(Roksvag et al. 2021).

Model (index-flood across durations, Koutsoyiannis et al. 1998 form)
---------------------------------------------------------------------
For duration d (minutes) the annual maximum depth X_d is written as

    X_d = s(d) * Z,     s(d) = (d / 60) * ((d + theta) / (60 + theta)) ** (-eta)

i.e. a duration scaling s(d) (normalised to 1 at 60 min) times ONE
standardised variable Z with a single distribution (GEV by default)
shared by all durations. theta >= 0 (minutes) and 0 <= eta < 1 make
s(d) strictly increasing, so design depths ALWAYS increase with duration
and with return period - curves cannot cross.

Estimation (L-moments, Hosking & Wallis 1997 regional-style pooling):
  * per duration, sample L-moments l1_d, l2_d, tau3_d of the AMS;
  * (theta, eta) and the pooled mean / L-scale (lambda1, lambda2) of Z by
    weighted least squares on log l1_d and log l2_d (weights = record
    length);
  * the shape is set from the record-length-weighted mean tau3 (the
    regional average in Hosking & Wallis's sense); the standardised
    distribution is then fitted to (lambda1, lambda2, tau3).

Optional duration-dependent growth curve ("smoothed L-moments"): the
L-CV follows l2_d / l1_d = cv60 * (d / 60) ** beta and the L-skewness
tau3_d = tau3_60 + g * ln(d / 60), both fitted by weighted least squares
across durations - i.e. the at-site L-moments are SMOOTHED across
durations rather than fitted independently (in the spirit of the ARR
2016 IFDs, which smooth L-moments across durations by regression).
Short durations typically have a higher L-CV and L-skewness than long
ones, which a single growth curve would dampen. Monotonicity is then
checked numerically; remaining crossings are removed by isotonic
regression across durations and return periods (Roksvag et al. 2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .distributions import (
    fit_gev_lmoments,
    fit_glo_lmoments,
    fit_gumbel_lmoments,
    quantile_for,
)
from .lmoments import sample_l_moments

MODEL_DISTRIBUTIONS = ("GEV", "GLO", "GUMBEL")
THETA_BOUNDS = (0.0, 720.0)
ETA_BOUNDS = (0.0, 0.99)
BETA_BOUNDS = (-0.5, 0.5)
TAU3_BOUNDS = (-0.2, 0.5)


def duration_scale(d, theta, eta):
    d = np.asarray(d, float)
    return (d / 60.0) * ((d + theta) / (60.0 + theta)) ** (-eta)


def _growth_params(distribution, cv, tau3):
    """Parameters of the standardised variable with mean 1, L-CV cv and
    L-skewness tau3 (so quantiles are growth factors)."""
    dist = distribution.upper()
    if dist == "GEV":
        return fit_gev_lmoments(1.0, cv, tau3)
    if dist == "GLO":
        return fit_glo_lmoments(1.0, cv, tau3)
    if dist == "GUMBEL":
        xi, al = fit_gumbel_lmoments(1.0, cv)
        return xi, al, 0.0
    raise ValueError(f"Model distribution must be one of {MODEL_DISTRIBUTIONS}")


@dataclass
class DDFModel:
    distribution: str
    theta: float
    eta: float
    lambda1: float  # mean of Z (depth at 60 min scale)
    cv60: float  # L-CV at 60 min
    beta: float  # L-CV duration exponent (0 = constant)
    tau3: float  # pooled L-skewness (at 60 min when it varies with duration)
    durations: list
    n_years: dict
    per_duration: dict = field(default_factory=dict)  # d -> sample L-moments
    rmse_log_l1: float = float("nan")
    rmse_log_l2: float = float("nan")
    tau3_slope: float = 0.0  # d tau3 / d ln(d/60) (0 = constant)
    rmse_tau3: float = float("nan")
    notes: list = field(default_factory=list)

    def t3(self, d):
        v = self.tau3 + self.tau3_slope * np.log(np.asarray(d, float) / 60.0)
        return np.clip(v, TAU3_BOUNDS[0], TAU3_BOUNDS[1])

    def l1(self, d):
        return self.lambda1 * duration_scale(d, self.theta, self.eta)

    def cv(self, d):
        return self.cv60 * (np.asarray(d, float) / 60.0) ** self.beta

    def depth(self, d, return_period):
        F = 1.0 - 1.0 / float(return_period)
        cv = float(self.cv(d))
        xi, al, ka = _growth_params(self.distribution, cv, float(self.t3(d)))
        growth = quantile_for(self.distribution, F, xi, al, ka)
        return float(self.l1(d) * growth)

    def table(self, durations, return_periods):
        return {float(d): [self.depth(d, t) for t in return_periods] for d in durations}

    def is_monotone(self, durations, return_periods):
        ds = np.geomspace(min(durations), max(durations), 200)
        for t in return_periods:
            q = np.array([self.depth(d, t) for d in ds])
            if np.any(np.diff(q) < -1e-9):
                return False
        for d in ds:
            q = np.array([self.depth(d, t) for t in sorted(return_periods)])
            if np.any(np.diff(q) < -1e-9):
                return False
        return True


def fit_ddf_model(ams_by_duration: dict, distribution="GEV", vary_cv=False, start=None):
    """ams_by_duration: {duration_min: 1-D array of annual maxima (mm)}.
    Needs at least 3 durations with at least 5 values each. start:
    optional (theta, eta) starting point (used by the bootstrap to skip
    the multi-start search)."""
    dist = distribution.upper()
    if dist not in MODEL_DISTRIBUTIONS:
        raise ValueError(f"Model distribution must be one of {MODEL_DISTRIBUTIONS}")
    per, n = {}, {}
    for d, x in sorted(ams_by_duration.items()):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 5:
            continue
        lm = sample_l_moments(x)
        if lm["l2"] <= 0 or lm["t3"] is None:
            continue
        per[float(d)] = lm
        n[float(d)] = len(x)
    if len(per) < 3:
        raise ValueError(
            "The duration-consistent DDF model needs at least 3 durations with 5 or "
            "more complete years each."
        )
    ds = np.array(sorted(per))
    w = np.array([n[d] for d in ds], float)
    w = w / w.sum()
    ll1 = np.log([per[d]["l1"] for d in ds])
    lcv = np.log([per[d]["l2"] / per[d]["l1"] for d in ds])

    def fit_mean(theta, eta):
        ls = np.log(duration_scale(ds, theta, eta))
        a = np.sum(w * (ll1 - ls))  # log lambda1
        r = ll1 - ls - a
        return a, float(np.sum(w * r**2))

    best = None
    starts = (
        [start]
        if start is not None
        else [(t, e) for t in (0.0, 10.0, 60.0, 240.0) for e in (0.2, 0.5, 0.8)]
    )
    for th0, et0 in starts:
        res = minimize(
            lambda p: fit_mean(p[0], p[1])[1],
            [th0, et0],
            method="L-BFGS-B",
            bounds=[THETA_BOUNDS, ETA_BOUNDS],
        )
        if best is None or res.fun < best.fun:
            best = res
    theta, eta = float(best.x[0]), float(best.x[1])
    loga, sse1 = fit_mean(theta, eta)

    lx = np.log(ds / 60.0)
    if vary_cv:
        A = np.vstack([np.ones_like(lx), lx]).T
        W = np.diag(w)
        coef = np.linalg.solve(A.T @ W @ A, A.T @ W @ lcv)
        log_cv60, beta = float(coef[0]), float(np.clip(coef[1], *BETA_BOUNDS))
    else:
        log_cv60, beta = float(np.sum(w * lcv)), 0.0
    resid_cv = lcv - (log_cv60 + beta * lx)

    t3s = np.array([per[d]["t3"] for d in ds])
    if vary_cv:
        A = np.vstack([np.ones_like(lx), lx]).T
        W = np.diag(w)
        c3 = np.linalg.solve(A.T @ W @ A, A.T @ W @ t3s)
        tau3, tau3_slope = float(c3[0]), float(c3[1])
    else:
        tau3, tau3_slope = float(np.sum(w * t3s)), 0.0
    resid_t3 = t3s - (tau3 + tau3_slope * lx)
    model = DDFModel(
        distribution=dist,
        theta=theta,
        eta=eta,
        lambda1=float(np.exp(loga)),
        cv60=float(np.exp(log_cv60)),
        beta=beta,
        tau3=tau3,
        durations=[float(d) for d in ds],
        n_years={float(d): n[d] for d in ds},
        per_duration=per,
        rmse_log_l1=float(np.sqrt(sse1)),
        rmse_log_l2=float(np.sqrt(np.sum(w * resid_cv**2))),
        tau3_slope=tau3_slope,
        rmse_tau3=float(np.sqrt(np.sum(w * resid_t3**2))),
    )
    return model


def _pava(y, w=None):
    """Pool-adjacent-violators: least-squares non-decreasing fit."""
    y = [float(v) for v in y]
    w = [1.0] * len(y) if w is None else [float(v) for v in w]
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            a, b = blocks[i], blocks[i + 1]
            wt = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / wt, wt, a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, k in blocks:
        out += [v] * k
    return np.array(out)


def enforce_consistency(table, durations, return_periods):
    """Make depths non-decreasing in duration (per return period) and in
    return period (per duration) by isotonic regression (Roksvag et al.
    2021). table: {duration: [depth per return period]}. Returns
    (new_table, max_relative_change)."""
    ds = sorted(durations)
    order = np.argsort(return_periods)
    arr = np.array([table[d] for d in ds], float)
    orig = arr.copy()
    for _ in range(5):
        for k in range(arr.shape[1]):
            arr[:, k] = _pava(arr[:, k])
        for i in range(arr.shape[0]):
            arr[i, order] = _pava(arr[i, order])
        ok_d = np.all(np.diff(arr, axis=0) >= -1e-9)
        ok_t = np.all(np.diff(arr[:, order], axis=1) >= -1e-9)
        if ok_d and ok_t:
            break
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.nanmax(np.abs(arr - orig) / np.where(orig > 0, orig, np.nan))
    return {d: list(arr[i]) for i, d in enumerate(ds)}, float(np.nan_to_num(rel))
