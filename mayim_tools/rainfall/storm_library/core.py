"""
Adjust Sub-daily Rainfall to DDF - calculation engine
(pass 1, diagnostics and storm library; pass 2 lives in adjust.py).

Zero QGIS/Qt dependency (pure numpy/pandas), same convention as every
core.py in this suite, so the whole method is unit-testable outside
QGIS (tests/test_storm_library_core.py).

Question answered
-----------------
"If I take this gridded product's storms and scale them to my
reference DDF at one anchor duration, do the shorter (and longer)
bursts inside them come out consistent with my DDF?"

Magnitude is taken from the reference DDF at the anchor duration and
storm structure from the gridded product, so what remains in the
comparison is the product's WITHIN-STORM structure only - the
"flattening" of short convective bursts that reanalysis and satellite
products are known for (Guerreiro et al. 2024; Lavers et al. 2022).

Pipeline
--------
1. prepare_series()          parse, optional site filter and time-zone
                             shift, regular grid (missing stays NaN,
                             never zero)
2. year completeness          shared rfa.compute_year_completeness
3. anchor AMS + fit           product's own frequency curve G at the
                             anchor duration (fixed-interval corrected)
4. event catalogue            shared IETD extract_events
5. magnitude mapping          every event rescaled so its anchor depth
                             follows the reference DDF at the event's
                             own AEP in G (AMS quantile mapping)
6. implied DDF                AMS of the rescaled series at every test
                             duration, fitted with the same
                             distribution
7. flatness index             F = implied / reference, year-block
                             bootstrap percentiles, per-duration
                             decision
8. storm library              normalised mass curves of the rescaled
                             storms, Early/Middle/Late, AEP band,
                             month and an ARR-style embedded-burst flag

Return-period convention: the reference DDF's '<T>yr' columns are
treated as annual-maximum-series return periods, AEP = 1/T - the
convention of both Design Rainfall (South Africa) and Precipitation
data to DDF, whose outputs this tool is designed to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mayim_tools.rainfall._common.ddf import DDFTable, aep_band
from mayim_tools.rainfall._common.events import (
    DEFAULT_IETD_HOURS,
    DEFAULT_MIN_EVENT_DEPTH_MM,
    dedupe_windows,
    extract_duration_window,
    extract_events,
)
from mayim_tools.rainfall._common.patterns import TARGET_DURATIONS, classify_shape
from mayim_tools.rainfall._common.rfa.ams import (
    build_regular_grid,
    compute_year_completeness,
)
from mayim_tools.rainfall._common.rfa.distributions import (
    fit_distribution,
    quantile_for,
)
from mayim_tools.rainfall._common.rfa.timebase import (
    detect_native_interval,
    duration_label,
)
from mayim_tools.rainfall._common.rfa.validation import parse_and_validate

from . import adjust as _adj

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFERENCE_TIERS = {
    1: "Tier 1 - published gauge-based DDF",
    2: "Tier 2 - GSDR-IDF / literature IDF / gauge-derived DDF",
    3: "Tier 3 - gridded-only reference",
}
INDICATIVE_STAMP = "INDICATIVE - NOT VALIDATED (Tier 3 reference)"
VALIDATED_STAMP = "Reference-based consistency check"

DISTRIBUTION_CHOICES = {
    "GEV (L-moments)": ("GEV", "L-moments"),
    "GEV (MLE)": ("GEV", "MLE"),
    "Gumbel": ("Gumbel", "L-moments"),
    "GLO": ("GLO", "L-moments"),
    "LP3": ("LP3", "L-moments"),
}

DEFAULT_ANCHOR_MIN = 1440.0
DEFAULT_MAX_TEST_DURATION_MIN = 4320.0
DEFAULT_MIN_COMPLETENESS = 0.90
DEFAULT_TOLERANCE_PCT = 10.0
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_RANDOM_SEED = 42
SELF_CHECK_TOLERANCE = 0.03
MIN_YEARS_ERROR = 5
MIN_YEARS_WARNING = 20
MIN_LIBRARY_STEPS = 3

DECISION_CONSISTENT = "consistent"
DECISION_INCONCLUSIVE = "inconclusive - interval includes 1"
DECISION_FLATTER = "flatter than reference - sharpening indicated"
DECISION_PEAKIER = "peakier than reference - review"
_DECISION_SEVERITY = {
    DECISION_CONSISTENT: 0,
    DECISION_INCONCLUSIVE: 1,
    DECISION_PEAKIER: 2,
    DECISION_FLATTER: 3,
}

TIME_COLUMN_CANDIDATES = ("ValidTime", "Time", "Timestamp", "DateTime", "Date")
DEPTH_COLUMN_CANDIDATES = ("PrecipitationMM", "Depth", "Rainfall", "Precip", "Value")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class StormLibraryConfig:
    anchor_min: float = DEFAULT_ANCHOR_MIN
    test_durations_min: tuple | None = None  # None -> automatic
    ietd_hours: float = DEFAULT_IETD_HOURS
    min_event_depth_mm: float = DEFAULT_MIN_EVENT_DEPTH_MM
    min_completeness: float = DEFAULT_MIN_COMPLETENESS
    fixed_interval_correction: bool = True
    distribution: str = "GEV (L-moments)"
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    random_seed: int = DEFAULT_RANDOM_SEED
    reference_tier: int = 1
    reference_is_areal: bool = False
    timezone_offset_h: float = 0.0
    build_library: bool = True
    # --- series adjustment (pass 2) ---
    adjust: bool = True
    rarity_dependent_sharpening: bool = True
    calibration_durations_min: tuple | None = None  # None -> automatic split
    n_realisations: int = 0  # stochastic ensemble size (0 = off)
    ensemble_sigma: float = 0.35
    # --- MAP constraint ---
    map_constraint: bool = True
    target_map_mm: float | None = None  # None -> retain the input series' MAP
    map_source: str = "input series"


@dataclass
class StormLibraryResult:
    consistency_rows: list = field(default_factory=list)
    duration_summary: list = field(default_factory=list)
    library_rows: list = field(default_factory=list)
    event_rows: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    overall_verdict: str = ""
    self_check_passed: bool = False
    status_stamp: str = VALIDATED_STAMP
    # --- series adjustment ---
    index: object = None  # DatetimeIndex of the (time-zone shifted) grid
    input_values: object = None
    pass1_values: object = None
    adjusted_values: object = None
    ensemble: list = field(default_factory=list)
    step_event_id: object = None
    step_scale: object = None
    step_exponent: object = None
    adjust_params: dict = field(default_factory=dict)
    after_rows: list = field(default_factory=list)
    ensemble_rows: list = field(default_factory=list)
    calibration_durations: list = field(default_factory=list)
    validation_durations: list = field(default_factory=list)
    verdict_after: str = ""
    map_info: dict = field(default_factory=dict)
    monthly_rows: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def weiss_factor(n_intervals: int) -> float:
    """Weiss (1964) ratio of the true (sliding) maximum to the maximum
    found from n_intervals fixed observation intervals:
    1 / (1 - 1/(8n)). n=1 -> 1.143, n=2 -> 1.067, n=24 -> 1.005.
    (The n=2 value, 1.0667, is the factor quoted for this relationship
    in the literature; Hershfield's 1961 empirical 1.13 for n=1 is the
    commonly used rounded equivalent.)"""
    n = int(n_intervals)
    if n < 1:
        raise ValueError("n_intervals must be >= 1")
    return 1.0 / (1.0 - 1.0 / (8.0 * n))


def _pick_column(columns, requested, candidates, what):
    if requested:
        if requested not in columns:
            raise ValueError(
                f"{what} column {requested!r} not found. Available: {list(columns)}"
            )
        return requested
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(
        f"Could not auto-detect the {what} column (tried {list(candidates)}). "
        f"Available: {list(columns)} - please name it explicitly."
    )


def prepare_series(
    df: pd.DataFrame,
    time_col: str | None = None,
    depth_col: str | None = None,
    site: str | None = None,
    site_col: str = "Site",
    timezone_offset_h: float = 0.0,
):
    """Returns (grid, native_minutes, site_used, warnings). grid is a
    regular pd.Series of incremental depth (mm) with NaN for missing -
    never zero. The optional time-zone offset is applied BEFORE years
    and months are assigned (ERA5 is UTC; South African daily gauges
    are read at 08:00 SAST)."""
    warnings = []
    site_used = None
    if site_col in df.columns:
        sites = [s for s in pd.unique(df[site_col].astype(str))]
        if site:
            if site not in sites:
                raise ValueError(f"Site {site!r} not found. Available: {sites}")
            site_used = site
        else:
            site_used = sites[0]
            if len(sites) > 1:
                warnings.append(
                    f"Input has {len(sites)} sites; using the first ({site_used!r}). "
                    "Run once per site."
                )
        df = df[df[site_col].astype(str) == site_used]

    tcol = _pick_column(df.columns, time_col, TIME_COLUMN_CANDIDATES, "time")
    dcol = _pick_column(df.columns, depth_col, DEPTH_COLUMN_CANDIDATES, "depth")
    parsed, diag = parse_and_validate(df, tcol, dcol)
    warnings += diag["warnings"]
    if len(parsed) < 2:
        raise ValueError("Fewer than 2 valid rows in the rainfall series.")
    if timezone_offset_h:
        parsed = parsed.copy()
        parsed["timestamp"] = parsed["timestamp"] + pd.Timedelta(
            hours=timezone_offset_h
        )
    native = detect_native_interval(parsed["timestamp"])
    grid = build_regular_grid(parsed, native).astype(float)
    return grid, native, site_used, warnings


def ams_by_year(values, years, valid_years, window):
    """Fast annual-maximum extraction (numpy cumulative sums) with the
    same rules as the shared rfa.extract_ams_for_duration: a window is
    labelled by its END time step (pandas rolling convention), any
    window touching a missing value is invalid, and only valid years
    contribute. Returns (years_out, maxima) as numpy arrays. Needed
    because the bootstrap recomputes AMS thousands of times; tests
    check it matches rfa exactly."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if window > n:
        return np.array([], dtype=int), np.array([], dtype=float)
    nan = np.isnan(values)
    cs = np.concatenate([[0.0], np.cumsum(np.where(nan, 0.0, values))])
    cn = np.concatenate([[0], np.cumsum(nan)])
    sums = cs[window:] - cs[:-window]
    sums[(cn[window:] - cn[:-window]) > 0] = np.nan
    end_years = np.asarray(years, dtype=np.int64)[window - 1 :]
    vy = np.asarray(valid_years, dtype=np.int64)
    lo = np.searchsorted(end_years, vy, side="left")
    hi = np.searchsorted(end_years, vy, side="right")
    years_out, maxima = [], []
    for y, a, b in zip(vy, lo, hi, strict=True):
        if b <= a:
            continue
        m = np.fmax.reduce(sums[a:b])
        if np.isnan(m):
            continue
        years_out.append(int(y))
        maxima.append(float(m))
    return np.array(years_out, dtype=int), np.array(maxima, dtype=float)


class FittedCurve:
    """A fitted annual-maximum distribution with vectorised quantile
    and (tabulated) non-exceedance functions, generic across the rfa
    distributions (including LP3, which has no closed-form CDF here)."""

    _Y = np.linspace(-np.log(-np.log(1e-4)), -np.log(-np.log(1 - 1e-6)), 4000)
    _F = np.exp(-np.exp(-_Y))

    def __init__(self, data, distribution: str, method: str):
        self.distribution = distribution
        self.method = method
        self.fit = fit_distribution(np.asarray(data, float), distribution, method)
        self._q = None  # quantile table, built on first return_period() call

    def _table(self):
        if self._q is None:
            f = self.fit
            if self.distribution.upper() in ("GEV", "GUMBEL", "GLO"):
                # closed-form quantiles are numpy-vectorised
                q = np.asarray(
                    quantile_for(
                        self.distribution, self._F, f["xi"], f["alpha"], f["kappa"]
                    ),
                    float,
                )
            else:
                q = np.array([self.quantile(F) for F in self._F])
            # guard against any numerically non-monotone tail
            self._q = np.maximum.accumulate(q)
        return self._q

    def quantile(self, prob):
        f = self.fit
        return float(
            quantile_for(
                self.distribution, prob, f["xi"], f["alpha"], f["kappa"], f["extra"]
            )
        )

    def depth_at_return_period(self, t_years):
        return self.quantile(1.0 - 1.0 / float(t_years))

    def return_period(self, x):
        """Vectorised T(x) = 1/(1-F(x)), clipped to the tabulated range."""
        F = np.interp(np.asarray(x, float), self._table(), self._F)
        return 1.0 / (1.0 - F)


# ---------------------------------------------------------------------------
# Magnitude mapping (AMS quantile mapping at the anchor duration)
# ---------------------------------------------------------------------------


DEFAULT_T0 = 1.1  # return period (yr) where the bulk segment of the transfer starts
GAMMA_BOUNDS = (0.05, 25.0)


def _fit_reference_gev(ref: DDFTable, duration_min: float):
    """GEV (Hosking parameterisation) fitted by least squares to the
    reference table's depths at one duration, used only to extend the
    reference below its most frequent tabulated return period. Falls
    back to Gumbel (kappa = 0) with fewer than three return periods."""
    from scipy.optimize import least_squares

    from mayim_tools.rainfall._common.rfa.distributions import gev_quantile

    t = np.array([float(x) for x in ref.aris])
    d = np.array([ref.depth_at_ari(duration_min, x)[0] for x in t])
    y = -np.log(-np.log(1.0 - 1.0 / t))
    alpha0, xi0 = np.polyfit(y, d, 1)
    alpha0 = max(alpha0, 1e-6)
    if len(t) < 3:
        return float(xi0), float(alpha0), 0.0

    def resid(p):
        return np.array([gev_quantile(1.0 - 1.0 / x, p[0], p[1], p[2]) for x in t]) - d

    fit = least_squares(
        resid,
        [xi0, alpha0, 0.0],
        bounds=([-np.inf, 1e-9, -0.5], [np.inf, np.inf, 0.5]),
    )
    return float(fit.x[0]), float(fit.x[1]), float(fit.x[2])


class AnchorTransfer:
    """Pass 1 - maps a storm's anchor-duration depth x (fixed-interval
    corrected) to a target depth f(x), and scales the whole storm by
    f(x)/x. Two segments, continuous at x0 = G^-1(T0):

    * annual-maximum range (x >= x0): f(x) = Q_ref(T_G(x)) - the
      reference DDF depth at the storm's return period in the product's
      own anchor AMS curve G. Q_ref is the reference table (ln T
      interpolation, linear-in-ln T extrapolation above the rarest T,
      flagged) and, between T0 and the table's most frequent T, a GEV
      fitted to the table (shifted to be continuous with it);
    * bulk (x < x0): f(x) = f(x0) * (x / x0) ** gamma. gamma is solved
      so the adjusted series reproduces the target mean annual
      precipitation (MAP). gamma > 1 reduces light rain relative to
      heavy rain - it concentrates the same annual total into fewer,
      heavier storms, which corrects the "drizzle" bias of reanalysis
      products at the storm scale. gamma = 1 is a constant ratio.

    Rain outside any storm (below the minimum storm depth) is scaled by
    the factor of the smallest storm, f(x_b)/x_b, so the transfer is
    continuous down to zero."""

    def __init__(
        self,
        grid_curve: FittedCurve,
        ref: DDFTable,
        anchor_min: float,
        gamma: float = 1.0,
        t0: float = DEFAULT_T0,
        x_background: float = DEFAULT_MIN_EVENT_DEPTH_MM,
    ):
        self.curve = grid_curve
        self.T_min = float(min(ref.aris))
        self.T_max = float(max(ref.aris))
        self.t0 = float(min(t0, self.T_min))
        from mayim_tools.rainfall._common.rfa.distributions import gev_quantile

        xi, al, ka = _fit_reference_gev(ref, anchor_min)
        shift = ref.depth_at_ari(anchor_min, self.T_min)[0] - gev_quantile(
            1.0 - 1.0 / self.T_min, xi, al, ka
        )
        self._lnT = np.linspace(np.log(self.t0), np.log(1e6), 800)
        vals = []
        for v in self._lnT:
            t = float(np.exp(v))
            if t >= self.T_min:
                vals.append(ref.depth_at_ari(anchor_min, t)[0])
            else:
                vals.append(gev_quantile(1.0 - 1.0 / t, xi, al, ka) + shift)
        self._ref = np.maximum.accumulate(np.maximum(np.array(vals), 0.0))
        self.reference_gev = (xi, al, ka)
        self.x0 = float(grid_curve.depth_at_return_period(self.t0))
        self.f0 = float(self._ref[0])
        self.gamma = float(gamma)
        self.x_background = float(max(x_background, 0.1))

    @property
    def s_floor(self):
        """Scale factor for rain outside any storm (continuous with the
        smallest storms)."""
        xb = self.x_background
        return float(self.f(np.array([xb]))[0] / xb)

    def f(self, x):
        x = np.asarray(x, float)
        T = self.curve.return_period(x)
        tail = np.interp(np.log(np.maximum(T, self.t0)), self._lnT, self._ref)
        with np.errstate(divide="ignore", invalid="ignore"):
            bulk = self.f0 * np.power(np.maximum(x, 0.0) / self.x0, self.gamma)
        return np.where(x >= self.x0, tail, bulk)

    def scale(self, x_corr):
        """Vectorised: returns (scale_factors, T_grid, flags)."""
        x = np.asarray(x_corr, float)
        T = self.curve.return_period(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(x > 0, self.f(x) / x, self.s_floor)
        flags = np.where(
            x < self.x0, "bulk", np.where(T > self.T_max, "extrap_aep", "in_range")
        )
        return s, T, flags


# Backward-compatible name (the pre-MAP version of the transfer)
AnchorMapping = AnchorTransfer


def embedded_burst_check(vals, native, duration, burst_t, ref, sub_durations, w_fn):
    """ARR-style embedded-burst test for one scaled storm window: is any
    shorter burst inside it (at a reference duration that the native
    step resolves) rarer than the window's own return period burst_t?
    Depths are compared on the same fixed-interval-corrected basis as
    the reference (w_fn(n) is the correction for an n-step window).
    Returns (flag, detail_text). A sub-burst deeper than the reference's
    rarest tabulated depth counts as rarer."""
    vals = np.asarray(vals, float)
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    flag, detail = False, []
    for dsub in sub_durations:
        k = int(round(dsub / native))
        if dsub >= duration or k < 1 or k > len(vals):
            continue
        sub = float(np.max(cs[k:] - cs[:-k])) * w_fn(k)
        t_sub, ok = ref.ari_of_depth(dsub, sub)
        top = ref.depth_at_ari(dsub, max(ref.aris))[0]
        if (ok and t_sub > burst_t * (1 + 1e-9)) or (not ok and sub > top):
            flag = True
            txt = f"1 in {t_sub:.3g} yr" if ok else f"> 1 in {max(ref.aris):g} yr"
            detail.append(
                f"{duration_label(dsub)} burst {txt} rarer than burst "
                f"(1 in {burst_t:.3g} yr)"
            )
    return flag, "; ".join(detail)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


def _auto_test_durations(ref: DDFTable, native: float, anchor: float, max_d: float):
    out = []
    for d in ref.durations:
        d = float(d)
        if d < native or d > max_d:
            continue
        ratio = d / native
        if abs(ratio - round(ratio)) > 1e-6:
            continue
        out.append(d)
    if anchor not in out:
        out.append(float(anchor))
    return sorted(set(out))


def _decide(p05, p50, p95, tol):
    if any(v is None or not np.isfinite(v) for v in (p05, p50, p95)):
        return DECISION_INCONCLUSIVE
    dev = abs(p50 - 1.0)
    if dev <= tol:
        return DECISION_CONSISTENT
    if p05 <= 1.0 <= p95:
        return DECISION_INCONCLUSIVE
    return DECISION_FLATTER if p95 < 1.0 else DECISION_PEAKIER


class StormLibraryEngine:
    def __init__(self, config: StormLibraryConfig | None = None):
        self.cfg = config or StormLibraryConfig()
        if self.cfg.distribution not in DISTRIBUTION_CHOICES:
            raise ValueError(
                f"Unknown distribution {self.cfg.distribution!r}; choose one of "
                f"{list(DISTRIBUTION_CHOICES)}"
            )
        if self.cfg.reference_tier not in REFERENCE_TIERS:
            raise ValueError("reference_tier must be 1, 2 or 3")

    # -- helpers -----------------------------------------------------------
    def _w(self, window_steps):
        return weiss_factor(window_steps) if self.cfg.fixed_interval_correction else 1.0

    def _fit(self, data):
        dist, method = DISTRIBUTION_CHOICES[self.cfg.distribution]
        return FittedCurve(data, dist, method)

    # -- run ---------------------------------------------------------------
    def run(
        self,
        grid: pd.Series,
        native: float,
        ref: DDFTable,
        progress=None,
        ref_long: pd.DataFrame | None = None,
        ref_info: dict | None = None,
    ):
        """ref_long / ref_info are the optional outputs of
        _common.ddf.read_ddf_csv(); when given, reference bounds
        (Design Rainfall Lower/Upper) are reported alongside F and the
        reader's conversions are recorded in the metadata."""
        cfg = self.cfg
        bounds = {}
        if ref_long is not None and "lower_mm" in ref_long:
            for r in ref_long.itertuples(index=False):
                bounds[(float(r.duration_min), float(r.ari_years))] = (
                    r.lower_mm,
                    r.upper_mm,
                )
        res = StormLibraryResult()
        tier_stamp = INDICATIVE_STAMP if cfg.reference_tier == 3 else VALIDATED_STAMP
        res.status_stamp = tier_stamp
        tier_label = REFERENCE_TIERS[cfg.reference_tier]

        anchor = float(cfg.anchor_min)
        a_steps = anchor / native
        if a_steps < 1 or abs(a_steps - round(a_steps)) > 1e-6:
            raise ValueError(
                f"Anchor duration {anchor:g} min is not a whole multiple of the "
                f"series' native step ({native:g} min)."
            )
        a_steps = int(round(a_steps))
        if anchor not in [float(d) for d in ref.durations]:
            res.warnings.append(
                f"Anchor duration {duration_label(anchor)} is not a tabulated "
                "reference DDF duration - reference depths there are interpolated "
                "in ln(duration). Prefer an anchor the table actually contains."
            )

        # 2. year completeness
        year_records = compute_year_completeness(grid, native, cfg.min_completeness)
        valid_years = np.array([r.year for r in year_records if r.included], int)
        excluded = [r for r in year_records if not r.included]
        if len(valid_years) < MIN_YEARS_ERROR:
            raise ValueError(
                f"Only {len(valid_years)} year(s) meet the {cfg.min_completeness:.0%} "
                f"completeness threshold - at least {MIN_YEARS_ERROR} are needed."
            )
        if len(valid_years) < MIN_YEARS_WARNING:
            res.warnings.append(
                f"Only {len(valid_years)} complete years - frequency fits and "
                "bootstrap intervals will be wide; treat results as low-confidence."
            )

        values = grid.values.astype(float)
        years = grid.index.year.values.astype(np.int64)
        filled = grid.fillna(0.0)
        nan_mask = grid.isna().values

        # test durations
        if cfg.test_durations_min:
            tests = sorted({float(d) for d in cfg.test_durations_min} | {anchor})
        else:
            tests = _auto_test_durations(
                ref, native, anchor, DEFAULT_MAX_TEST_DURATION_MIN
            )
        test_steps = {}
        for d in tests:
            r = d / native
            if r < 1 or abs(r - round(r)) > 1e-6:
                res.warnings.append(
                    f"Test duration {duration_label(d)} skipped - not a whole multiple "
                    f"of the native step ({native:g} min)."
                )
                continue
            test_steps[d] = int(round(r))
        tests = sorted(test_steps)

        # 3. anchor AMS of the product + fit G
        yrs_a, ams_a = ams_by_year(values, years, valid_years, a_steps)
        ams_a = ams_a * self._w(a_steps)
        g_curve = self._fit(ams_a)
        mapping = AnchorTransfer(
            g_curve, ref, anchor, x_background=cfg.min_event_depth_mm
        )

        # 4. events + their anchor window depth (on the NaN-filled grid;
        #    events touching missing data are flagged)
        events = extract_events(filled, cfg.ietd_hours, cfg.min_event_depth_mm)
        if not events:
            raise ValueError(
                "No storm events identified - check IETD / minimum event depth and "
                "that the series holds incremental (not cumulative) depths."
            )
        idx = grid.index
        ev_start = np.searchsorted(
            idx.values, np.array([e["start"] for e in events], dtype="datetime64[ns]")
        )
        ev_end = np.searchsorted(
            idx.values, np.array([e["end"] for e in events], dtype="datetime64[ns]")
        )
        x_raw = np.empty(len(events))
        ev_missing = np.zeros(len(events), bool)
        anchor_start = ev_start.astype(np.int64).copy()
        anchor_full = np.zeros(len(events), bool)
        for i, ev in enumerate(events):
            w = extract_duration_window(filled, native, ev, anchor)
            x_raw[i] = w["depth_mm"] if w is not None else ev["depth_mm"]
            if w is not None:
                s0 = idx.get_loc(w["start"])
                e0 = idx.get_loc(w["end"])
                anchor_start[i] = s0
                anchor_full[i] = (e0 - s0 + 1) == a_steps
                ev_missing[i] = bool(nan_mask[s0 : e0 + 1].any())
            ev_missing[i] |= bool(nan_mask[ev_start[i] : ev_end[i] + 1].any())
        x_corr = x_raw * self._w(a_steps)

        # step -> event index (-1 outside events)
        step_event = np.full(len(values), -1, dtype=int)
        for i in range(len(events)):
            step_event[ev_start[i] : ev_end[i] + 1] = i

        def rescaled_values(mp: AnchorTransfer):
            s, T, flags = mp.scale(x_corr)
            scale = np.where(step_event >= 0, s[np.maximum(step_event, 0)], mp.s_floor)
            return values * scale, s, T, flags

        # 4b. MAP constraint: solve the bulk exponent gamma so that the
        #     pass-1 series (and hence the adjusted series - pass 2 is
        #     mass-preserving) has the target mean annual precipitation
        in_valid = np.isin(years, valid_years)

        def map_of(vals):
            return float(np.nansum(np.where(in_valid, vals, 0.0))) / len(valid_years)

        map_input = map_of(values)
        map_target = float(cfg.target_map_mm) if cfg.target_map_mm else map_input
        gamma = 1.0
        if cfg.map_constraint:
            gamma = self._solve_gamma(mapping, rescaled_values, map_of, map_target)
        mapping.gamma = gamma
        res.map_info = {
            "map_input_mm": map_input,
            "map_target_mm": map_target,
            "map_target_source": cfg.map_source,
            "map_constraint": cfg.map_constraint,
            "transfer_gamma": gamma,
            "transfer_t0_yr": mapping.t0,
            "transfer_x0_mm": mapping.x0,
            "reference_gev_below_table": ",".join(
                f"{v:.4g}" for v in mapping.reference_gev
            ),
        }

        # 5/6. point estimate
        resc, s_e, T_e, f_e = rescaled_values(mapping)
        res.map_info["map_pass1_mm"] = map_of(resc)
        ref_aris = [float(t) for t in ref.aris]

        def implied_for(resc_vals, year_subset_idx=None):
            out = {}
            for d in tests:
                n = test_steps[d]
                yy, mx = ams_by_year(resc_vals, years, valid_years, n)
                mx = mx * self._w(n)
                if year_subset_idx is not None:
                    pos = {y: k for k, y in enumerate(yy)}
                    mx = np.array([mx[pos[y]] for y in year_subset_idx if y in pos])
                if len(mx) < 3:
                    out[d] = None
                    continue
                c = self._fit(mx)
                out[d] = np.array([c.depth_at_return_period(T) for T in ref_aris])
            return out

        implied = implied_for(resc)
        ref_depth = {
            d: np.array([ref.depth_at_ari(d, T)[0] for T in ref_aris]) for d in tests
        }
        ref_flag = {d: [ref.depth_at_ari(d, T)[1] for T in ref_aris] for d in tests}

        # 7. bootstrap (year blocks)
        rng = np.random.default_rng(cfg.random_seed)
        boot = {d: [] for d in tests}
        n_failed = 0
        ams_a_by_year = dict(zip(yrs_a.tolist(), ams_a.tolist(), strict=True))
        a_years = np.array(sorted(ams_a_by_year))
        for b in range(cfg.n_bootstrap):
            sample = rng.choice(a_years, size=len(a_years), replace=True)
            try:
                gb = self._fit(np.array([ams_a_by_year[y] for y in sample]))
                mb = AnchorTransfer(
                    gb, ref, anchor, gamma=gamma, x_background=cfg.min_event_depth_mm
                )
                rb, _, _, _ = rescaled_values(mb)
                ib = implied_for(rb, year_subset_idx=sample)
            except (ValueError, FloatingPointError, ZeroDivisionError):
                n_failed += 1
                continue
            for d in tests:
                if ib[d] is not None:
                    boot[d].append(ib[d] / ref_depth[d])
            if progress is not None and (b + 1) % 20 == 0:
                if progress(100.0 * (b + 1) / max(cfg.n_bootstrap, 1)) is False:
                    raise InterruptedError("Cancelled")
        if n_failed:
            res.warnings.append(
                f"{n_failed} of {cfg.n_bootstrap} bootstrap replicates failed to fit "
                "and were skipped."
            )

        tol = cfg.tolerance_pct / 100.0
        worst = DECISION_CONSISTENT
        self_dev = 0.0
        for d in tests:
            arr = np.vstack(boot[d]) if boot[d] else None
            d_worst = DECISION_CONSISTENT
            for k, T in enumerate(ref_aris):
                F_pt = (
                    implied[d][k] / ref_depth[d][k]
                    if implied[d] is not None and ref_depth[d][k] > 0
                    else np.nan
                )
                if arr is not None and arr.shape[0] >= 10:
                    p05, p50, p95 = np.nanpercentile(arr[:, k], [5, 50, 95])
                else:
                    p05 = p50 = p95 = np.nan
                is_anchor = d == anchor
                decision = (
                    "anchor (self-check)" if is_anchor else _decide(p05, p50, p95, tol)
                )
                if is_anchor and np.isfinite(F_pt):
                    self_dev = max(self_dev, abs(F_pt - 1.0))
                if not is_anchor and d < anchor:
                    if _DECISION_SEVERITY[decision] > _DECISION_SEVERITY[d_worst]:
                        d_worst = decision
                res.consistency_rows.append(
                    {
                        "duration_min": d,
                        "duration": duration_label(d),
                        "return_period_yr": T,
                        "aep_pct": 100.0 / T,
                        "reference_depth_mm": ref_depth[d][k],
                        "reference_flag": ref_flag[d][k],
                        "implied_depth_mm": (
                            implied[d][k] if implied[d] is not None else np.nan
                        ),
                        "reference_lower_mm": bounds.get((d, T), (np.nan, np.nan))[0],
                        "reference_upper_mm": bounds.get((d, T), (np.nan, np.nan))[1],
                        "F": F_pt,
                        "F_p05": p05,
                        "F_p50": p50,
                        "F_p95": p95,
                        "n_bootstrap_ok": 0 if arr is None else arr.shape[0],
                        "n_years": len(valid_years),
                        "decision": decision,
                        "is_anchor": is_anchor,
                        "reference_tier": tier_label,
                        "status": tier_stamp,
                    }
                )
            if d < anchor:
                res.duration_summary.append(
                    {
                        "duration_min": d,
                        "duration": duration_label(d),
                        "decision": d_worst,
                    }
                )
                if _DECISION_SEVERITY[d_worst] > _DECISION_SEVERITY[worst]:
                    worst = d_worst
            elif d > anchor:
                longer = [r for r in res.consistency_rows if r["duration_min"] == d]
                dec = max(
                    (r["decision"] for r in longer),
                    key=lambda x: _DECISION_SEVERITY.get(x, 0),
                )
                res.duration_summary.append(
                    {"duration_min": d, "duration": duration_label(d), "decision": dec}
                )

        res.self_check_passed = self_dev <= SELF_CHECK_TOLERANCE
        if not res.self_check_passed:
            res.warnings.append(
                f"Anchor self-check: implied/reference differs from 1 by up to "
                f"{self_dev:.1%} at the anchor duration (tolerance "
                f"{SELF_CHECK_TOLERANCE:.0%}). The event-level mapping is exact, but "
                "the rescaled annual maxima are not well described by the chosen "
                "distribution with the reference DDF's shape - try another "
                "distribution, or check the reference table at the anchor duration."
            )
        res.overall_verdict = (
            "flatter than reference at short durations (shape correction needed)"
            if worst == DECISION_FLATTER
            else {
                DECISION_CONSISTENT: "consistent - no sharpening needed",
                DECISION_INCONCLUSIVE: "inconclusive - record too short to decide",
                DECISION_PEAKIER: "product peakier than reference - review",
            }[worst]
        )

        # 7b. SERIES ADJUSTMENT (pass 2: within-storm shape)
        ln_t = np.log(np.clip(T_e, 1.0, np.exp(_adj.LN_T_MAX)))
        ev_ok = (~ev_missing) & anchor_full
        n_ev = len(events)
        b_e = np.ones(n_ev)
        adjusted = resc
        if cfg.adjust:
            adjusted, b_e = self._adjust(
                res,
                resc,
                anchor_start,
                a_steps,
                native,
                ev_ok,
                ln_t,
                tests,
                test_steps,
                anchor,
                years,
                valid_years,
                ref_aris,
                ref_depth,
                implied,
                progress,
            )
        res.index = grid.index
        res.input_values = values
        res.pass1_values = resc
        res.adjusted_values = adjusted
        res.step_event_id = np.where(
            step_event >= 0,
            np.array([e["event_id"] for e in events])[np.maximum(step_event, 0)],
            -1,
        )
        res.step_scale = np.where(
            step_event >= 0, s_e[np.maximum(step_event, 0)], mapping.s_floor
        )
        res.step_exponent = np.where(
            step_event >= 0, b_e[np.maximum(step_event, 0)], 1.0
        )

        # event audit rows
        for i, ev in enumerate(events):
            res.event_rows.append(
                {
                    "event_id": ev["event_id"],
                    "start": ev["start"],
                    "end": ev["end"],
                    "event_depth_mm": ev["depth_mm"],
                    "anchor_window_depth_mm": x_raw[i],
                    "anchor_depth_corrected_mm": x_corr[i],
                    "return_period_in_product_yr": T_e[i],
                    "scale_factor": s_e[i],
                    "rescaled_anchor_depth_mm": x_corr[i] * s_e[i],
                    "peak_concentration_factor": b_e[i],
                    "range_flag": f_e[i],
                    "touches_missing": bool(ev_missing[i]),
                    "status": tier_stamp,
                }
            )

        # 8. storm library
        if cfg.build_library:
            self._build_library(
                res,
                grid,
                adjusted,
                native,
                ref,
                events,
                ev_missing,
                s_e,
                step_event,
                tier_label,
                tier_stamp,
            )

        res.metadata = {
            "tool": "Adjust Sub-daily Rainfall to DDF",
            "reference_tier": tier_label,
            "status": tier_stamp,
            "reference_is_areal": cfg.reference_is_areal,
            "reference_return_periods_yr": ",".join(f"{t:g}" for t in ref_aris),
            "reference_durations_min": ",".join(f"{d:g}" for d in ref.durations),
            "return_period_convention": "AMS: AEP = 1/T",
            "native_step_min": native,
            "anchor_duration_min": anchor,
            "test_durations_min": ",".join(f"{d:g}" for d in tests),
            "distribution": cfg.distribution,
            "fixed_interval_correction": (
                "Weiss (1964) 1/(1-1/(8n))" if cfg.fixed_interval_correction else "off"
            ),
            "ietd_hours": cfg.ietd_hours,
            "min_event_depth_mm": cfg.min_event_depth_mm,
            "min_completeness": cfg.min_completeness,
            "timezone_offset_h": cfg.timezone_offset_h,
            "record_start": str(grid.index.min()),
            "record_end": str(grid.index.max()),
            "n_years_used": len(valid_years),
            "years_excluded": ",".join(str(r.year) for r in excluded),
            "n_events": len(events),
            "n_events_touching_missing": int(ev_missing.sum()),
            "n_events_bulk_segment": int((f_e == "bulk").sum()),
            "n_events_beyond_reference_range": int((f_e == "extrap_aep").sum()),
            "scale_factor_floor": mapping.s_floor,
            "n_bootstrap": cfg.n_bootstrap,
            "n_bootstrap_failed": n_failed,
            "random_seed": cfg.random_seed,
            "tolerance_pct": cfg.tolerance_pct,
            "self_check_max_deviation": self_dev,
            "self_check_passed": res.self_check_passed,
            "overall_verdict": res.overall_verdict,
            "product_anchor_fit_xi": g_curve.fit["xi"],
            "product_anchor_fit_alpha": g_curve.fit["alpha"],
            "product_anchor_fit_kappa": g_curve.fit["kappa"],
            "n_library_patterns": len(res.library_rows),
            "library_built_from": "adjusted series" if cfg.adjust else "pass-1 series",
        }
        res.metadata.update(res.adjust_params)

        # MAP / seasonal / wet-hour diagnostics
        res.map_info["map_output_mm"] = map_of(adjusted)
        months = grid.index.month.values
        ny = len(valid_years)
        for mth in range(1, 13):
            sel = in_valid & (months == mth)
            vin = float(np.nansum(values[sel])) / ny
            vout = float(np.nansum(adjusted[sel])) / ny
            res.monthly_rows.append(
                {
                    "month": mth,
                    "input_mean_mm": vin,
                    "adjusted_mean_mm": vout,
                    "ratio": vout / vin if vin > 0 else np.nan,
                }
            )
        thr = 0.1 * native / 60.0  # 0.1 mm/h wet-step threshold
        res.map_info["wet_step_threshold_mm"] = thr
        res.map_info["wet_steps_per_year_input"] = (
            float(np.sum(in_valid & (np.nan_to_num(values) > thr))) / ny
        )
        res.map_info["wet_steps_per_year_adjusted"] = (
            float(np.sum(in_valid & (np.nan_to_num(adjusted) > thr))) / ny
        )
        res.metadata.update(res.map_info)
        if (
            cfg.map_constraint
            and abs(res.map_info["map_output_mm"] - map_target) > 0.005 * map_target
        ):
            res.warnings.append(
                f"Adjusted MAP {res.map_info['map_output_mm']:.1f} mm differs from the "
                f"target {map_target:.1f} mm by more than 0.5%."
            )
        for row in res.duration_summary:
            res.metadata[f"decision_{row['duration']}"] = row["decision"]
        if ref_info:
            res.metadata["reference_site"] = ref_info.get("site") or ""
            res.metadata["reference_durations_dropped"] = ",".join(
                ref_info.get("dropped_durations", [])
            )
            res.metadata["reference_fixed_day_conversions"] = ";".join(
                ref_info.get("fixed_day_durations", [])
            )
            res.metadata["reference_has_bounds"] = ref_info.get("has_bounds", False)
        return res

    # -- MAP constraint ----------------------------------------------------
    @staticmethod
    def _solve_gamma(mapping, rescaled_values, map_of, target):
        lo, hi = GAMMA_BOUNDS

        def m(g):
            mapping.gamma = g
            return map_of(rescaled_values(mapping)[0])

        m_lo, m_hi = m(lo), m(hi)
        if target < m_hi:
            raise ValueError(
                f"Target MAP {target:.0f} mm/yr cannot be reached: storms matched to "
                f"the reference DDF alone already give about {m_hi:.0f} mm/yr in this "
                "series. The reference DDF, the target MAP and the rainfall series "
                "are not mutually consistent (check the site, the analysis period "
                "and the MAP source)."
            )
        if target > m_lo:
            raise ValueError(
                f"Target MAP {target:.0f} mm/yr cannot be reached: even with maximum "
                f"weight on light rain the series gives only {m_lo:.0f} mm/yr. Check "
                "the MAP value and that the series is complete and in mm."
            )
        a, b = np.log(lo), np.log(hi)
        for _ in range(80):
            mid = 0.5 * (a + b)
            if m(float(np.exp(mid))) > target:
                a = mid
            else:
                b = mid
            if b - a < 1e-7:
                break
        g = float(np.exp(0.5 * (a + b)))
        mapping.gamma = g
        return g

    # -- series adjustment (pass 2) --------------------------------------
    def _implied_subset(self, series, durations, test_steps, years, valid_years, aris):
        out = {}
        for d in durations:
            n = test_steps[d]
            _, mx = ams_by_year(series, years, valid_years, n)
            if len(mx) < 3:
                out[d] = None
                continue
            c = self._fit(mx * self._w(n))
            out[d] = np.array([c.depth_at_return_period(t) for t in aris])
        return out

    def _fit_boot(self, series, d, n, years, valid_years, aris, rng):
        _, mx = ams_by_year(series, years, valid_years, n)
        mx = mx * self._w(n)
        reps = []
        for _ in range(self.cfg.n_bootstrap):
            sample = rng.choice(mx, size=len(mx), replace=True)
            try:
                c = self._fit(sample)
            except (ValueError, FloatingPointError, ZeroDivisionError):
                continue
            reps.append([c.depth_at_return_period(t) for t in aris])
        return np.array(reps) if reps else None

    def _adjust(
        self,
        res,
        resc,
        anchor_start,
        a_steps,
        native,
        ev_ok,
        ln_t,
        tests,
        test_steps,
        anchor,
        years,
        valid_years,
        aris,
        ref_depth,
        implied_before,
        progress,
    ):
        cfg = self.cfg
        rd = cfg.rarity_dependent_sharpening
        shorts = [d for d in tests if d < anchor]
        if cfg.calibration_durations_min:
            calib = sorted(
                {float(d) for d in cfg.calibration_durations_min} & set(shorts)
            )
        else:
            calib, _ = _adj.split_durations(shorts, anchor)
        valid = [d for d in tests if d not in calib and d != anchor]
        res.calibration_durations = calib
        res.validation_durations = valid

        def implied_at(series, d):
            return self._implied_subset(
                series, [d], test_steps, years, valid_years, aris
            )[d]

        adjusted = resc
        concentration = np.ones(len(anchor_start))
        rung_params = []
        if calib:
            out, rung_params, concentration = _adj.nested_adjust(
                resc,
                anchor_start,
                a_steps,
                ev_ok,
                ln_t,
                calib,
                native,
                implied_at,
                ref_depth,
                rarity_dependent=rd,
            )
            adjusted = out[0]
        res.adjust_params = {
            "adjust_method": "nested peak-window scaling (mass-preserving)",
            "rarity_dependent": rd,
            "calibration_durations_min": ",".join(f"{d:g}" for d in calib),
            "validation_durations_min": ",".join(f"{d:g}" for d in valid),
            "n_storms_adjusted": int(ev_ok.sum()),
            "n_storms_not_adjusted": int((~ev_ok).sum()),
        }
        for rp in rung_params:
            key = duration_label(rp["duration_min"]).replace(" ", "")
            res.adjust_params[f"rung_{key}_a"] = rp["a"]
            res.adjust_params[f"rung_{key}_c"] = rp["c"]
            res.adjust_params[f"rung_{key}_rmse_log"] = rp["rmse_log"]
            if rp["a"] >= _adj.A_BOUNDS[1] - 1e-3:
                res.warnings.append(
                    f"Rung {duration_label(rp['duration_min'])}: concentration hit "
                    "its upper bound - the reference DDF may not be reachable at this "
                    "duration; check the inputs."
                )
        if progress is not None:
            progress(90.0)

        # after-diagnostics (fit bootstrap on the adjusted series)
        rng = np.random.default_rng(cfg.random_seed + 1)
        tol = cfg.tolerance_pct / 100.0
        implied_after = self._implied_subset(
            adjusted, tests, test_steps, years, valid_years, aris
        )
        worst = DECISION_CONSISTENT
        worst_valid = DECISION_CONSISTENT
        for d in tests:
            reps = self._fit_boot(
                adjusted, d, test_steps[d], years, valid_years, aris, rng
            )
            role = (
                "anchor"
                if d == anchor
                else ("calibration" if d in calib else "validation")
            )
            for k, t in enumerate(aris):
                ref_v = ref_depth[d][k]
                ia = implied_after[d][k] if implied_after[d] is not None else np.nan
                ib = implied_before[d][k] if implied_before[d] is not None else np.nan
                if reps is not None and len(reps) >= 10:
                    q05, q50, q95 = np.nanpercentile(reps[:, k] / ref_v, [5, 50, 95])
                else:
                    q05 = q50 = q95 = np.nan
                dec = _decide(q05, q50, q95, tol) if d != anchor else "anchor"
                if d != anchor and _DECISION_SEVERITY[dec] > _DECISION_SEVERITY[worst]:
                    worst = dec
                if role == "validation" and (
                    _DECISION_SEVERITY[dec] > _DECISION_SEVERITY[worst_valid]
                ):
                    worst_valid = dec
                res.after_rows.append(
                    {
                        "duration_min": d,
                        "duration": duration_label(d),
                        "role": role,
                        "return_period_yr": t,
                        "aep_pct": 100.0 / t,
                        "reference_depth_mm": ref_v,
                        "input_implied_depth_mm": ib,
                        "adjusted_implied_depth_mm": ia,
                        "F_before": ib / ref_v if ref_v > 0 else np.nan,
                        "F_after": ia / ref_v if ref_v > 0 else np.nan,
                        "F_after_p05": q05,
                        "F_after_p50": q50,
                        "F_after_p95": q95,
                        "decision_after": dec,
                        "status": res.status_stamp,
                    }
                )
        res.verdict_after = (
            "adjusted series consistent with the reference DDF at all durations"
            if worst == DECISION_CONSISTENT
            else f"adjusted series: {worst} at one or more durations"
        )
        res.adjust_params["verdict_after"] = res.verdict_after
        res.adjust_params["verdict_after_validation_durations"] = worst_valid

        # optional stochastic ensemble
        n_real = int(cfg.n_realisations)
        if n_real > 0 and calib:
            erng = np.random.default_rng(cfg.random_seed + 2)
            ens, eparams, _ = _adj.nested_adjust(
                resc,
                anchor_start,
                a_steps,
                ev_ok,
                ln_t,
                calib,
                native,
                implied_at,
                ref_depth,
                rarity_dependent=rd,
                n_realisations=n_real,
                sigma=cfg.ensemble_sigma,
                rng=erng,
            )
            res.ensemble = ens
            per = [
                self._implied_subset(r, tests, test_steps, years, valid_years, aris)
                for r in ens
            ]
            for d in tests:
                arr = np.vstack([q[d] for q in per if q.get(d) is not None])
                for k, t in enumerate(aris):
                    f = arr[:, k] / ref_depth[d][k]
                    res.ensemble_rows.append(
                        {
                            "duration_min": d,
                            "duration": duration_label(d),
                            "return_period_yr": t,
                            "F_ensemble_p05": float(np.percentile(f, 5)),
                            "F_ensemble_p50": float(np.percentile(f, 50)),
                            "F_ensemble_p95": float(np.percentile(f, 95)),
                            "n_realisations": arr.shape[0],
                        }
                    )
            res.adjust_params.update(
                {
                    "ensemble_n_realisations": n_real,
                    "ensemble_sigma": cfg.ensemble_sigma,
                    "ensemble_random_seed": cfg.random_seed + 2,
                }
            )
            for rp in eparams:
                key = duration_label(rp["duration_min"]).replace(" ", "")
                res.adjust_params[f"ensemble_rung_{key}_a"] = rp["a"]
                res.adjust_params[f"ensemble_rung_{key}_c"] = rp["c"]
        if progress is not None:
            progress(100.0)
        return adjusted, concentration

    # -- storm library -----------------------------------------------------
    def _build_library(
        self,
        res,
        grid,
        resc,
        native,
        ref,
        events,
        ev_missing,
        s_e,
        step_event,
        tier_label,
        tier_stamp,
    ):
        idx = grid.index
        resc_series = pd.Series(np.nan_to_num(resc, nan=0.0), index=idx)
        raw_vals = grid.values
        nan_mask = grid.isna().values
        sub_durations = [
            float(d)
            for d in ref.durations
            if d >= native and abs(d / native - round(d / native)) < 1e-6
        ]
        ev_index = {ev["event_id"]: i for i, ev in enumerate(events)}
        n_skipped_missing = 0
        storm_id = 0
        for duration, _ts, _n in TARGET_DURATIONS:
            steps = duration / native
            if abs(steps - round(steps)) > 1e-6 or steps < MIN_LIBRARY_STEPS:
                continue
            steps = int(round(steps))
            windows = []
            for ev in events:
                w = extract_duration_window(resc_series, native, ev, duration)
                if w is None or len(w["values"]) < steps:
                    continue
                windows.append(w)
            for w in dedupe_windows(windows):
                s0 = idx.get_loc(w["start"])
                e0 = idx.get_loc(w["end"])
                if nan_mask[s0 : e0 + 1].any():
                    n_skipped_missing += 1
                    continue
                vals = np.asarray(w["values"], float)
                total = float(vals.sum())
                if total <= 0:
                    continue
                w_corr = total * self._w(steps)
                T, in_range = ref.ari_of_depth(duration, w_corr)
                if in_range:
                    aep = 100.0 / T
                    band = aep_band(aep)
                    rflag = "in_range"
                else:
                    lo = ref.depth_at_ari(duration, min(ref.aris))[0]
                    aep = None
                    band = "frequent" if w_corr < lo else "rare"
                    rflag = (
                        "below_reference_range"
                        if w_corr < lo
                        else "above_reference_range"
                    )
                emb_flag, emb_detail = (
                    embedded_burst_check(
                        vals, native, duration, T, ref, sub_durations, self._w
                    )
                    if in_range
                    else (False, "")
                )
                storm_id += 1
                ev_i = ev_index.get(w["event_id"], -1)
                row = {
                    "storm_id": storm_id,
                    "event_id": w["event_id"],
                    "duration_min": duration,
                    "duration": duration_label(duration),
                    "start": w["start"],
                    "end": w["end"],
                    "month": pd.Timestamp(w["start"]).month,
                    "raw_depth_mm": float(np.nansum(raw_vals[s0 : e0 + 1])),
                    "scale_factor": float(s_e[ev_i]) if ev_i >= 0 else np.nan,
                    "rescaled_depth_mm": total,
                    "return_period_yr": T if in_range else np.nan,
                    "aep_pct": aep if aep is not None else np.nan,
                    "aep_band": band,
                    "range_flag": rflag,
                    "shape": classify_shape({"values": vals}),
                    "timestep_min": native,
                    "n_increments": steps,
                    "embedded_burst_flag": emb_flag,
                    "embedded_burst_detail": emb_detail,
                    "reference_tier": tier_label,
                    "status": tier_stamp,
                }
                pct = vals / total * 100.0
                for j, v in enumerate(pct, start=1):
                    row[f"inc_{j:03d}"] = float(v)
                res.library_rows.append(row)
        if n_skipped_missing:
            res.warnings.append(
                f"{n_skipped_missing} candidate storm window(s) touched missing data "
                "and were left out of the storm library."
            )
