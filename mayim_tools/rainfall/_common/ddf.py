"""
DDF (depth-duration-frequency) table handling shared by the rainfall
tools: duration-label parsing, AEP/ARI conversion, AEP band
classification and the DDFTable interpolation / inversion class.

Moved from design_storm_ensembles/core.py (behaviour unchanged). The
ARI-space helpers DDFTable.depth_at_ari() and DDFTable.ari_of_depth()
were added so callers working with annual-maximum-series return
periods (AEP = 1/T) can use the table without going through the exact
Poisson AEP<->ARI conversion that aep_from_ari()/ari_from_aep()
implement.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .csvio import read_csv_any_delimiter

AEP_BANDS = ["frequent", "intermediate", "rare"]

AEP_BAND_FREQUENT_MIN = 20.0  # >=20% AEP -> frequent
AEP_BAND_INTERMEDIATE_MIN = 5.0  # >=5% and <20% -> intermediate; <5% -> rare


DURATION_RE = re.compile(
    r"([\d.]+)\s*(day|days|d|hour|hours|hr|hrs|h|minute|minutes|min|mins|m)\b",
    re.IGNORECASE,
)

UNIT_TO_MINUTES = {
    "day": 1440,
    "days": 1440,
    "d": 1440,
    "hour": 60,
    "hours": 60,
    "hr": 60,
    "hrs": 60,
    "h": 60,
    "minute": 1,
    "minutes": 1,
    "min": 1,
    "mins": 1,
    "m": 1,
}

ARI_COL_RE = re.compile(r"([\d.]+)\s*yr", re.IGNORECASE)


def parse_duration_to_minutes(text):
    s = str(text).strip()
    m = DURATION_RE.search(s)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        return value * UNIT_TO_MINUTES[unit]
    try:
        return float(s)  # bare number -> assume minutes
    except ValueError as exc:
        raise ValueError(
            f"Could not parse duration '{text}'. Expected e.g. '3 h' or '180'."
        ) from exc


def aep_from_ari(ari_years):
    """Exact AEP/ARI relationship (ARR Book 1 s2.2.5 / Poisson process),
    not the 1/ARI approximation."""
    return (1.0 - np.exp(-1.0 / ari_years)) * 100.0


def ari_from_aep(aep_percent):
    p = aep_percent / 100.0
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -1.0 / np.log(1.0 - p)


def aep_band(aep_percent):
    if aep_percent >= AEP_BAND_FREQUENT_MIN:
        return "frequent"
    elif aep_percent >= AEP_BAND_INTERMEDIATE_MIN:
        return "intermediate"
    return "rare"


def load_ddf_table(path, duration_col=None):
    """Load a DDF CSV (Duration + one or more '<ARI>yr Depth (mm)'
    columns) and return a long-format DataFrame with columns
    [duration_min, ari_years, aep_percent, depth_mm]."""
    df = read_csv_any_delimiter(path)
    if duration_col is None:
        duration_col = df.columns[0]

    ari_cols = []
    for col in df.columns:
        if col == duration_col:
            continue
        m = ARI_COL_RE.search(str(col))
        if m:
            ari_cols.append((col, float(m.group(1))))

    if not ari_cols:
        raise ValueError(
            "No return-period depth columns found in the DDF table "
            "(expected column names like '10yr Depth (mm)')."
        )

    rows = []
    for _, row in df.iterrows():
        dur_min = parse_duration_to_minutes(row[duration_col])
        for col, ari in ari_cols:
            depth = row[col]
            if pd.notna(depth):
                rows.append(
                    {
                        "duration_min": dur_min,
                        "ari_years": ari,
                        "aep_percent": aep_from_ari(ari),
                        "depth_mm": float(depth),
                    }
                )

    long_df = (
        pd.DataFrame(rows)
        .sort_values(["duration_min", "ari_years"])
        .reset_index(drop=True)
    )
    if long_df.empty:
        raise ValueError("DDF table parsed to zero usable rows.")
    return long_df


class DDFTable:
    """Wraps a long-format DDF table and caches the expensive parts of
    AEP lookups: each table duration's depth-vs-ln(ARI) curve (built
    once at construction) and each target duration's interpolated
    depth-vs-ln(ARI) curve (built once, on first use, then cached).
    Without this caching, assigning an AEP to every storm re-filters
    and re-sorts the DDF table from scratch for every ARI column of
    every storm - fine for a handful of calls, but very slow across a
    multi-decade storm catalogue."""

    def __init__(self, long_df):
        self.long_df = long_df
        self.durations = sorted(long_df["duration_min"].unique())
        self.aris = sorted(long_df["ari_years"].unique())
        self._duration_curves = {}
        for d in self.durations:
            sub = long_df[long_df["duration_min"] == d].sort_values("ari_years")
            self._duration_curves[d] = (
                np.log(sub["ari_years"].values),
                sub["depth_mm"].values,
            )
        self._aep_curve_cache = {}

    def _depth_at_table_duration(self, table_duration_min, target_ari):
        x, y = self._duration_curves[table_duration_min]
        target_x = np.log(target_ari)
        if target_x <= x[0]:
            extrap = target_x < x[0]
            if len(x) > 1 and extrap:
                slope = (y[1] - y[0]) / (x[1] - x[0])
                return y[0] + slope * (target_x - x[0]), extrap
            return y[0], extrap
        if target_x >= x[-1]:
            extrap = target_x > x[-1]
            if len(x) > 1 and extrap:
                slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
                return y[-1] + slope * (target_x - x[-1]), extrap
            return y[-1], extrap
        return float(np.interp(target_x, x, y)), False

    def depth_at(self, duration_min, aep_percent):
        """Interpolated/extrapolated depth (mm) at an arbitrary
        duration/AEP. Returns (depth_mm, flag) where flag is one of
        'interp', 'extrap_aep', 'extrap_duration'.

        The AEP is converted to the table's ARI columns with the exact
        Poisson relationship (ari_from_aep). Callers working directly
        in annual-maximum return periods should use depth_at_ari()."""
        return self.depth_at_ari(duration_min, ari_from_aep(aep_percent))

    def depth_at_ari(self, duration_min, target_ari):
        """Same as depth_at(), but addressed directly by the table's own
        return period (the number in its '<T>yr' column headers), with
        no AEP<->ARI conversion. Interpolation is linear in ln(T) along
        a table duration and linear in ln(duration) between table
        durations; beyond the table's range it extrapolates linearly
        in ln(T) from the last two points and flags it."""
        if duration_min in self._duration_curves:
            depth, extrap = self._depth_at_table_duration(duration_min, target_ari)
            return depth, ("extrap_aep" if extrap else "interp")

        lower = max([d for d in self.durations if d < duration_min], default=None)
        upper = min([d for d in self.durations if d > duration_min], default=None)

        if lower is None and upper is None:
            raise ValueError("DDF table has no usable durations.")
        if lower is None:
            depth, _ = self._depth_at_table_duration(upper, target_ari)
            return depth, "extrap_duration"
        if upper is None:
            depth, _ = self._depth_at_table_duration(lower, target_ari)
            return depth, "extrap_duration"

        d_lo, extrap_lo = self._depth_at_table_duration(lower, target_ari)
        d_up, extrap_up = self._depth_at_table_duration(upper, target_ari)
        w = (np.log(duration_min) - np.log(lower)) / (np.log(upper) - np.log(lower))
        depth = d_lo + w * (d_up - d_lo)
        flag = "extrap_aep" if (extrap_lo or extrap_up) else "interp"
        return depth, flag

    def _aep_curve(self, duration_min):
        if duration_min not in self._aep_curve_cache:
            curve_x, curve_y = [], []
            for ari in self.aris:
                depth, _ = self.depth_at(duration_min, aep_from_ari(ari))
                curve_x.append(np.log(ari))
                curve_y.append(depth)
            curve_x = np.array(curve_x)
            curve_y = np.array(curve_y)
            order = np.argsort(curve_x)
            self._aep_curve_cache[duration_min] = (curve_x[order], curve_y[order])
        return self._aep_curve_cache[duration_min]

    def assign_aep(self, duration_min, obs_depth_mm):
        """Invert this duration's depth-vs-AEP curve to find the AEP
        matching an observed event depth. Returns (aep_percent,
        in_range). in_range is False when obs_depth_mm falls outside
        the table's covered depth range at this duration, signalling
        the caller to fall back to empirical ranking."""
        curve_x, curve_y = self._aep_curve(duration_min)
        if obs_depth_mm < curve_y[0] or obs_depth_mm > curve_y[-1]:
            return None, False
        ln_ari = np.interp(obs_depth_mm, curve_y, curve_x)
        ari = np.exp(ln_ari)
        return aep_from_ari(ari), True

    def ari_of_depth(self, duration_min, depth_mm):
        """Inverse of depth_at_ari(): the table return period T (years,
        in the table's own '<T>yr' units) at which this duration
        reaches depth_mm. Returns (T, in_range); T is None when
        depth_mm falls outside the depth range the table covers at this
        duration (no extrapolation - the caller decides how to treat
        out-of-range depths)."""
        curve_x, curve_y = self._aep_curve(duration_min)
        if depth_mm < curve_y[0] or depth_mm > curve_y[-1]:
            return None, False
        return float(np.exp(np.interp(depth_mm, curve_y, curve_x))), True


def assign_aep_empirical(population_depths, this_depth):
    """Weibull plotting-position AEP (%) for this_depth within
    population_depths."""
    n = len(population_depths)
    rank = 1 + sum(1 for d in population_depths if d > this_depth)
    return 100.0 * rank / (n + 1)


# ---------------------------------------------------------------------------
# Robust reader for the suite's DDF CSV layouts
# ---------------------------------------------------------------------------

_BOUND_RE = re.compile(r"\b(lower|upper)\b", re.IGNORECASE)
_DAY_LABEL_RE = re.compile(r"\bdays?\b|\bd\b", re.IGNORECASE)


def _weiss(n):
    return 1.0 / (1.0 - 1.0 / (8.0 * n))


def read_ddf_csv(path, site=None, fixed_day_to_continuous=True):
    """Read a DDF table in any of the layouts this suite writes, and
    return (long_df, info).

    Handles:
    - Precipitation data to DDF 'recommended DDF' CSV
      (Duration, RecommendedDistribution, '<T>yr Depth (mm)'...)
    - Design Rainfall (South Africa) single-site CSV: a location
      header block, a blank line, the table, then an optional
      nearest-stations block after another blank line
    - Design Rainfall multi-site CSV (Site column; one site is used)
    - any simple 'Duration' + '<T>yr ...' table

    Only the central-estimate '<T>yr' columns are used as depths;
    '<T>yr Lower'/'<T>yr Upper' bound columns are carried separately
    as lower_mm/upper_mm (load_ddf_table() does not make this
    distinction). Where a table gives both a continuous '24 h' and a
    fixed-interval '1 day' value, the continuous one is kept. Day-
    labelled durations (fixed 08:00-08:00 readings in the SA
    methodology) are converted to continuous-duration equivalents with
    the Weiss (1964) factor 1/(1-1/(8n)) when fixed_day_to_continuous
    is True, and reported in info['fixed_day_durations'].

    long_df columns: duration_min, duration_label, ari_years,
    aep_percent (= 100/T, annual-maximum convention), depth_mm,
    lower_mm, upper_mm, fixed_day_factor."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        lines = f.read().splitlines()

    import csv as _csv

    rows = list(_csv.reader(lines))
    header_i = None
    for i, r in enumerate(rows):
        cells = [c.strip() for c in r]
        if any(c.lower() == "duration" for c in cells) and any(
            ARI_COL_RE.search(c) for c in cells
        ):
            header_i = i
            break
    if header_i is None:
        raise ValueError(
            "Could not find the DDF table header (a 'Duration' column plus "
            "'<T>yr ...' depth columns) in the reference DDF file."
        )
    header = [c.strip() for c in rows[header_i]]
    body = []
    for r in rows[header_i + 1 :]:
        if not any(c.strip() for c in r):
            break  # a blank line ends the table (stations block follows)
        body.append(r + [""] * (len(header) - len(r)))
    df = pd.DataFrame([r[: len(header)] for r in body], columns=header)

    info = {"site": None, "dropped_durations": [], "fixed_day_durations": []}
    site_col = next((c for c in header if c.lower() == "site"), None)
    if site_col is not None and len(df):
        sites = list(dict.fromkeys(df[site_col].astype(str)))
        chosen = site if site else sites[0]
        if chosen not in sites:
            raise ValueError(f"Site {chosen!r} not in reference DDF. Found: {sites}")
        df = df[df[site_col].astype(str) == chosen]
        info["site"] = chosen
        info["sites_available"] = sites

    dur_col = next(c for c in header if c.lower() == "duration")
    depth_cols, bound_cols = {}, {}
    for c in header:
        m = ARI_COL_RE.search(c)
        if not m or c == dur_col:
            continue
        T = float(m.group(1))
        b = _BOUND_RE.search(c)
        if b:
            bound_cols[(T, b.group(1).lower())] = c
        else:
            depth_cols[T] = c
    if not depth_cols:
        raise ValueError("No central-estimate '<T>yr' depth columns found.")

    # choose one row per duration in minutes (continuous label preferred)
    chosen_rows = {}
    for _, r in df.iterrows():
        label = str(r[dur_col]).strip()
        if not label:
            continue
        minutes = parse_duration_to_minutes(label)
        is_day = bool(_DAY_LABEL_RE.search(label))
        prev = chosen_rows.get(minutes)
        if prev is None:
            chosen_rows[minutes] = (label, is_day, r)
        elif prev[1] and not is_day:
            info["dropped_durations"].append(prev[0])
            chosen_rows[minutes] = (label, is_day, r)
        else:
            info["dropped_durations"].append(label)

    def num(v):
        v = pd.to_numeric(v, errors="coerce")
        return float(v) if pd.notna(v) else np.nan

    out = []
    for minutes in sorted(chosen_rows):
        label, is_day, r = chosen_rows[minutes]
        factor = 1.0
        if is_day and fixed_day_to_continuous:
            n_days = max(1, int(round(minutes / 1440.0)))
            factor = _weiss(n_days)
            info["fixed_day_durations"].append(f"{label} x{factor:.4f}")
        for T, col in sorted(depth_cols.items()):
            d = num(r[col])
            if np.isnan(d):
                continue
            lo = (
                num(r[bound_cols[(T, "lower")]])
                if (T, "lower") in bound_cols
                else np.nan
            )
            up = (
                num(r[bound_cols[(T, "upper")]])
                if (T, "upper") in bound_cols
                else np.nan
            )
            out.append(
                {
                    "duration_min": minutes,
                    "duration_label": label,
                    "ari_years": T,
                    "aep_percent": 100.0 / T,
                    "depth_mm": d * factor,
                    "lower_mm": lo * factor,
                    "upper_mm": up * factor,
                    "fixed_day_factor": factor,
                }
            )
    long_df = pd.DataFrame(out)
    if long_df.empty:
        raise ValueError("Reference DDF parsed to zero usable rows.")
    info["has_bounds"] = bool(long_df["lower_mm"].notna().any())
    return long_df, info
