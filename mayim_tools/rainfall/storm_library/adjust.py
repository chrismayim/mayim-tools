"""
Series adjustment to a reference DDF - pass 2 (within-storm shape).

Pass 1 (core.AnchorMapping) scales every storm so its anchor-duration
depth follows the reference DDF at the storm's own return period. That
fixes magnitude but keeps the gridded product's (typically too flat)
within-storm structure, so the short-duration annual maxima of the
pass-1 series still fall short of the DDF.

Pass 2 - nested peak-window scaling
-----------------------------------
Working down a ladder of calibration durations shorter than the anchor
(e.g. 24 h -> 12 h -> 3 h -> 1 h), for every storm:

  * the PARENT window is the storm's peak window at the previous
    (longer) rung - for the first rung, its anchor-duration window;
  * the CHILD window is the highest-depth window of the current rung's
    duration inside that parent;
  * the child is multiplied by alpha_e and the rest of the parent by
    beta_e = (P - alpha_e*C) / (P - C), so the parent total P is
    unchanged (mass-preserving at every coarser rung, hence the anchor
    magnitude from pass 1 is untouched), zeros stay zero, and the storm's
    timing is kept;
  * alpha_e = exp(a_k + c_k * (ln T_e - ln 10)), clipped to
    [ALPHA_MIN, P/C] (so beta_e >= 0), where T_e is the storm's return
    period from pass 1. c_k = 0 when rarity dependence is switched off.

(a_k, c_k) are fitted rung by rung, from the longest to the shortest,
by minimising the mean squared log-ratio between the adjusted series'
implied DDF at that rung's duration and the reference DDF (over all
reference return periods). Durations between the rungs, and durations
longer than the anchor, are never fitted: they are the independent
VALIDATION of the adjustment.

Optional stochastic ensemble: each realisation multiplies alpha_e by an
independent mean-one lognormal factor exp(sigma*z - sigma^2/2) at every
rung, and (a_k, c_k) are re-fitted so that the ENSEMBLE-MEDIAN implied
DDF matches the reference. sigma expresses within-storm variability
that a DDF table cannot itself constrain; the spread across
realisations shows the resulting structural uncertainty.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

ALPHA_MIN = 0.5
LN_T_CENTRE = np.log(10.0)
LN_T_MAX = np.log(1e4)
A_BOUNDS = (-1.0, 2.5)
C_BOUNDS = (-1.0, 1.0)


def split_durations(short_durations, anchor):
    """Default calibration / validation split of the durations shorter
    than the anchor: every other one counted from the shortest is a
    calibration rung (so the shortest - where the deficit is largest -
    is always fitted); those in between are held out for validation."""
    shorts = sorted(d for d in short_durations if d < anchor)
    return shorts[0::2], shorts[1::2]


def locate_children(v, parent_start, parent_len, child_len, ok):
    """For every storm with ok=True, the start of the highest-depth
    child_len window inside its parent window. Returns child_start
    (-1 where not applicable)."""
    child_start = np.full(len(parent_start), -1, dtype=np.int64)
    vf = np.nan_to_num(v, nan=0.0)
    for e in np.flatnonzero(ok):
        ps = parent_start[e]
        seg = vf[ps : ps + parent_len]
        if len(seg) < child_len:
            continue
        cs = np.concatenate([[0.0], np.cumsum(seg)])
        sums = cs[child_len:] - cs[:-child_len]
        child_start[e] = ps + int(np.argmax(sums))
    return child_start


class Rung:
    """Pre-computed index structure for one rung, so that evaluating a
    candidate (a, c) is a handful of vectorised numpy operations."""

    def __init__(self, v, parent_start, parent_len, child_start, child_len, ok):
        use = ok & (child_start >= 0)
        # drop storms whose parent overlaps an earlier storm's parent at
        # this rung (neighbouring storms sharing one anchor window) so no
        # step is scaled twice
        order = np.argsort(parent_start)
        last_end = -1
        keep = np.zeros(len(use), bool)
        for e in order:
            if not use[e]:
                continue
            if parent_start[e] > last_end:
                keep[e] = True
                last_end = parent_start[e] + parent_len - 1
        self.events = np.flatnonzero(keep)
        steps, ev, child = [], [], []
        for e in self.events:
            ps, cs0 = parent_start[e], child_start[e]
            rng = np.arange(ps, ps + parent_len)
            steps.append(rng)
            ev.append(np.full(parent_len, e))
            child.append((rng >= cs0) & (rng < cs0 + child_len))
        self.steps = np.concatenate(steps) if steps else np.array([], int)
        self.ev = np.concatenate(ev) if ev else np.array([], int)
        self.is_child = np.concatenate(child) if child else np.array([], bool)
        n = len(parent_start)
        vals = np.nan_to_num(v[self.steps], nan=0.0)
        self.P = np.bincount(self.ev, weights=vals, minlength=n)
        self.C = np.bincount(self.ev, weights=vals * self.is_child, minlength=n)
        self.child_start = child_start

    def apply(self, v, alpha):
        """Return a new series with this rung's scaling applied."""
        out = np.array(v, dtype=float, copy=True)
        if len(self.steps) == 0:
            return out
        P, C = self.P, self.C
        with np.errstate(divide="ignore", invalid="ignore"):
            a_max = np.where(C > 0, P / C, 1.0)
            a = np.clip(alpha, ALPHA_MIN, a_max)
            b = np.where(P - C > 1e-12, (P - a * C) / (P - C), 1.0)
        b = np.maximum(b, 0.0)  # guard against round-off below zero
        # nothing outside the child to borrow from (or give to): leave as is
        a = np.where((C > 0) & (P - C > 1e-12), a, 1.0)
        mult = np.where(self.is_child, a[self.ev], b[self.ev])
        out[self.steps] = out[self.steps] * mult
        return out


def alphas(ln_t, a, c, noise=None):
    al = np.exp(a + c * (ln_t - LN_T_CENTRE))
    if noise is not None:
        al = al * noise
    return al


def fit_rung(
    rungs_per_series,
    series_list,
    ln_t,
    implied_fn,
    ref_depths_d,
    rarity_dependent,
    noises=None,
    start=(0.3, 0.0),
):
    """Fit (a, c) for one rung. rungs_per_series/series_list: one Rung
    and one current series per realisation (a single one for the
    deterministic adjustment). implied_fn(series) -> implied depths at
    this rung's duration (array over reference return periods) or None.
    The median across realisations is matched to ref_depths_d."""

    def loss(p):
        a = p[0]
        c = p[1] if rarity_dependent else 0.0
        vals = []
        for j, (rg, s) in enumerate(zip(rungs_per_series, series_list, strict=True)):
            nz = None if noises is None else noises[j]
            q = implied_fn(rg.apply(s, alphas(ln_t, a, c, nz)))
            if q is None:
                return 1e6
            vals.append(q)
        med = np.median(np.vstack(vals), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log(med / ref_depths_d)
        if not np.all(np.isfinite(r)):
            return 1e6
        return float(np.mean(r**2))

    x0 = np.array(start if rarity_dependent else start[:1], float)
    bounds = [A_BOUNDS, C_BOUNDS] if rarity_dependent else [A_BOUNDS]
    res = minimize(
        loss,
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400},
    )
    a = float(res.x[0])
    c = float(res.x[1]) if rarity_dependent else 0.0
    return a, c, float(res.fun)


def lognormal_noise(n, sigma, rng):
    if sigma <= 0:
        return np.ones(n)
    return np.exp(sigma * rng.standard_normal(n) - 0.5 * sigma * sigma)


def nested_adjust(
    v1,
    anchor_start,
    anchor_len,
    ok,
    ln_t,
    ladder,
    native,
    implied_at,
    ref_depths,
    rarity_dependent=True,
    n_realisations=1,
    sigma=0.0,
    rng=None,
):
    """Run the full rung ladder. v1: pass-1 series. anchor_start: per
    storm start index of its anchor window. ladder: calibration
    durations (minutes, shorter than the anchor), any order.
    implied_at(series, d) -> implied depths at d. Returns
    (list_of_adjusted_series, rung_params, per_storm_alpha_first_series).
    With n_realisations == 1 and sigma == 0 the result is the single
    deterministic adjustment."""
    ladder = sorted(ladder, reverse=True)
    n_ev = len(anchor_start)
    stochastic = n_realisations > 1 or sigma > 0
    noises_all = None
    if stochastic:
        rng = rng or np.random.default_rng(0)
        noises_all = [
            [lognormal_noise(n_ev, sigma, rng) for _ in ladder]
            for _ in range(n_realisations)
        ]
    series = [np.array(v1, float, copy=True) for _ in range(n_realisations)]
    parents = [np.array(anchor_start, np.int64) for _ in range(n_realisations)]
    parent_len = int(anchor_len)
    params = []
    alpha_record = np.ones(n_ev)
    # calibrate on (at most) the first 8 realisations - common random numbers
    m = min(n_realisations, 8)
    prev = (0.3, 0.0)
    for k, d in enumerate(ladder):
        child_len = int(round(d / native))
        rungs = []
        for j in range(n_realisations):
            cs = locate_children(series[j], parents[j], parent_len, child_len, ok)
            rungs.append(Rung(series[j], parents[j], parent_len, cs, child_len, ok))
        nz = [noises_all[j][k] for j in range(m)] if stochastic else None
        a, c, loss = fit_rung(
            rungs[:m],
            series[:m],
            ln_t,
            lambda s, d=d: implied_at(s, d),
            ref_depths[d],
            rarity_dependent,
            noises=nz,
            start=prev,
        )
        prev = (a, c)
        for j in range(n_realisations):
            nzj = noises_all[j][k] if stochastic else None
            al = alphas(ln_t, a, c, nzj)
            if j == 0:
                rg = rungs[0]
                with np.errstate(divide="ignore", invalid="ignore"):
                    amax = np.where(rg.C > 0, rg.P / rg.C, 1.0)
                alpha_record = alpha_record * np.where(
                    rg.C > 0, np.clip(al, ALPHA_MIN, amax), 1.0
                )
            series[j] = rungs[j].apply(series[j], al)
            parents[j] = np.where(
                rungs[j].child_start >= 0, rungs[j].child_start, parents[j]
            )
        parent_len = child_len
        params.append(
            {"duration_min": d, "a": a, "c": c, "rmse_log": float(np.sqrt(loss))}
        )
    return series, params, alpha_record
