"""
Design Storm Ensembles - calculation engine.

Zero QGIS/Qt dependency by design (pure pandas/numpy) so it can be
unit-tested standalone before being wired into the QGIS Processing
algorithm - same pattern used by the Design Rainfall plugin in this
suite.

Pipeline
--------
1. load_timeseries()          - read rainfall CSV, detect native time step
2. load_ddf_table()            - read DDF CSV, parse durations + ARI columns
3. extract_events()             - IETD-based independent storm event catalogue
4. extract_duration_window()    - per event x target duration, best sub-window
5. DDFTable.assign_aep()        - hybrid DDF-interpolation / empirical-ranking
6. classify_shape()             - Early / Middle / Late peak-position class
7. build_direct_bin_curve()     - Duration x AEP x Shape median curve, where
                                   target TimeStep >= native step
8. disaggregate_bin_curve()     - Monte Carlo cascade for TimeStep < native

See the plugin's README for the full methodology writeup and its
literature basis (Huff 1967; ARR Project 3 temporal patterns; AEP/ARI
per ARR Book 1 s2.2.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mayim_tools.rainfall._common.csvio import read_csv_any_delimiter
from mayim_tools.rainfall._common.ddf import (
    AEP_BAND_FREQUENT_MIN,
    AEP_BAND_INTERMEDIATE_MIN,
    AEP_BANDS,
    ARI_COL_RE,
    DURATION_RE,
    UNIT_TO_MINUTES,
    DDFTable,
    aep_band,
    aep_from_ari,
    ari_from_aep,
    assign_aep_empirical,
    load_ddf_table,
    parse_duration_to_minutes,
)
from mayim_tools.rainfall._common.events import (
    DEFAULT_IETD_HOURS,
    DEFAULT_MIN_EVENT_DEPTH_MM,
    dedupe_windows,
    extract_duration_window,
    extract_events,
)
from mayim_tools.rainfall._common.patterns import (
    SHAPES,
    TARGET_DURATIONS,
    classify_shape,
    normalized_cumulative_curve,
)

# Names re-exported for backward compatibility: these building blocks
# now live in mayim_tools.rainfall._common (shared with other tools) but
# remain importable from this module exactly as before.
__all__ = [
    "AEP_BAND_FREQUENT_MIN",
    "AEP_BAND_INTERMEDIATE_MIN",
    "AEP_BANDS",
    "ARI_COL_RE",
    "DURATION_RE",
    "SHAPES",
    "TARGET_DURATIONS",
    "UNIT_TO_MINUTES",
    "DDFTable",
    "DesignStormEngine",
    "aep_band",
    "aep_from_ari",
    "ari_from_aep",
    "assign_aep_empirical",
    "build_direct_bin_curve",
    "classify_shape",
    "dedupe_windows",
    "disaggregate_bin_curve",
    "dirichlet_split",
    "extract_duration_window",
    "extract_events",
    "load_ddf_table",
    "load_timeseries",
    "normalized_cumulative_curve",
    "parse_duration_to_minutes",
    "read_csv_any_delimiter",
]

# ---------------------------------------------------------------------------
# Engine defaults (TARGET_DURATIONS, SHAPES, AEP bands, IETD and minimum
# event depth defaults now live in mayim_tools.rainfall._common)
# ---------------------------------------------------------------------------

DEFAULT_MIN_SAMPLE_SIZE = 5
DEFAULT_MC_SIMS = 500
DEFAULT_MC_ALPHA = 3.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_POOL_SPARSE_BINS = True


# ---------------------------------------------------------------------------
# 1. Time series loading
# ---------------------------------------------------------------------------


COMMON_DATETIME_FORMATS = [
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d-%m-%Y %H:%M",
    "%m/%d/%Y %H:%M",
]


def parse_datetime_series(col):
    """Try a handful of common formats first (fast, unambiguous) before
    falling back to pandas' generic per-element inference (slow on
    large files, and silently ambiguous between day-first/month-first
    layouts)."""
    for fmt in COMMON_DATETIME_FORMATS:
        parsed = pd.to_datetime(col, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.95:
            return parsed
    return pd.to_datetime(col, errors="coerce")


def detect_datetime_column(df):
    for col in df.columns:
        parsed = parse_datetime_series(df[col])
        if parsed.notna().mean() > 0.95:
            return col
    raise ValueError("Could not find a datetime column in the time series CSV.")


def detect_value_column(df, datetime_col):
    for col in df.columns:
        if col == datetime_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() > 0.95:
            return col
    raise ValueError(
        "Could not find a numeric rainfall depth column in the time series CSV."
    )


def detect_native_interval(index: pd.DatetimeIndex):
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError(
            "Time series has fewer than 2 rows; cannot detect its time step."
        )
    mode_delta = diffs.mode().iloc[0]
    minutes = mode_delta.total_seconds() / 60.0
    if minutes <= 0:
        raise ValueError(
            "Detected a non-positive time step; check the datetime column."
        )
    return minutes


def load_timeseries(path, datetime_col=None, value_col=None):
    """Load a rainfall time series CSV and return (pd.Series, native_minutes).

    The series is indexed by datetime and holds INCREMENTAL depth per
    interval (mm), not cumulative. If datetime_col / value_col are not
    given, the first parseable-as-datetime column and the first
    remaining numeric column are used.
    """
    df = read_csv_any_delimiter(path)
    if datetime_col is None:
        datetime_col = detect_datetime_column(df)
    if value_col is None:
        value_col = detect_value_column(df, datetime_col)

    dt = parse_datetime_series(df[datetime_col])
    if dt.isna().any():
        n_bad = int(dt.isna().sum())
        raise ValueError(
            f"{n_bad} row(s) in '{datetime_col}' could not be parsed as datetimes."
        )

    values = pd.to_numeric(df[value_col], errors="coerce")
    if values.isna().any():
        values = values.fillna(0.0)

    series = pd.Series(values.values, index=pd.DatetimeIndex(dt.values)).sort_index()
    series = series[~series.index.duplicated(keep="first")]
    native_minutes = detect_native_interval(series.index)
    return series, native_minutes


# ---------------------------------------------------------------------------
# 2-5. DDF table loading, IETD event catalogue, per-duration window
#      extraction and Early/Middle/Late classification: see
#      mayim_tools.rainfall._common (ddf.py, events.py, patterns.py)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Direct-from-data bin curve (target TimeStep >= native resolution)
# ---------------------------------------------------------------------------


def _renormalize(increments, total=100.0):
    increments = np.clip(increments, 0, None)
    s = increments.sum()
    if s <= 0:
        return increments
    return increments * (total / s)


def build_direct_bin_curve(windows, n_increments):
    """Median-of-normalized-curves aggregation (Huff-style) for a bin
    whose storms have native-resolution data at least as fine as
    required. Returns percentage increments (list of length
    n_increments, sums to 100) or None if the bin is empty."""
    if not windows:
        return None
    target_time = np.arange(1, n_increments + 1) / n_increments
    resampled = []
    for w in windows:
        cum_time, cum_depth = normalized_cumulative_curve(w["values"])
        resampled.append(np.interp(target_time, cum_time, cum_depth))
    median_curve = np.median(np.vstack(resampled), axis=0)
    median_curve = np.concatenate([[0.0], median_curve])
    increments = np.diff(median_curve) * 100.0
    increments = _renormalize(increments)
    return increments.tolist()


# ---------------------------------------------------------------------------
# 7. Monte Carlo cascade disaggregation (target TimeStep < native resolution)
# ---------------------------------------------------------------------------


def dirichlet_split(parent_values, k, alpha, rng):
    """Split each element of a 1D array into k children via a
    symmetric Dirichlet(alpha) multiplicative cascade, preserving each
    parent increment's mass. Returns a 1D array of length
    len(parent_values) * k."""
    out = np.empty(len(parent_values) * k)
    for i, v in enumerate(parent_values):
        weights = rng.dirichlet([alpha] * k)
        out[i * k : (i + 1) * k] = weights * v
    return out


def disaggregate_bin_curve(
    anchor_increments_pct,
    anchor_timestep_minutes,
    target_duration_minutes,
    target_timestep_minutes,
    target_n_increments,
    mc_sims,
    alpha,
    rng,
):
    """Derive a target Duration/TimeStep pattern from a coarser real
    'anchor' pattern via: (1) crop the anchor to the highest-mass
    contiguous slice covering >= target_duration_minutes, using a
    deterministic uniform sub-split to locate the crop window, then
    (2) run mc_sims realizations of a random Dirichlet cascade split
    of that slice down to target_timestep_minutes resolution, (3) crop
    each realization to the best target_n_increments-long window and
    take the median across realizations."""
    anchor = np.asarray(anchor_increments_pct, dtype=float)
    ratio = anchor_timestep_minutes / target_timestep_minutes
    if ratio < 1:
        raise ValueError("Anchor TimeStep must be coarser than the target TimeStep.")
    k = max(1, round(ratio))
    k_anchor = max(1, int(np.ceil(target_duration_minutes / anchor_timestep_minutes)))
    k_anchor = min(k_anchor, len(anchor))

    csum = np.concatenate([[0.0], np.cumsum(anchor)])
    if len(anchor) >= k_anchor:
        sums = csum[k_anchor:] - csum[:-k_anchor]
    else:
        sums = np.array([anchor.sum()])
    slice_start = int(np.argmax(sums)) if len(sums) > 0 else 0
    slice_vals = anchor[slice_start : slice_start + k_anchor]
    if len(slice_vals) < k_anchor:
        slice_vals = np.pad(slice_vals, (0, k_anchor - len(slice_vals)))

    uniform_full = np.repeat(slice_vals / k, k)
    n_target = target_n_increments
    if len(uniform_full) >= n_target:
        csum_u = np.concatenate([[0.0], np.cumsum(uniform_full)])
        sums_u = csum_u[n_target:] - csum_u[:-n_target]
        crop_start = int(np.argmax(sums_u))
    else:
        crop_start = 0

    realizations = np.empty((mc_sims, n_target))
    for s in range(mc_sims):
        full = dirichlet_split(slice_vals, k, alpha, rng)
        if len(full) < crop_start + n_target:
            full = np.pad(full, (0, crop_start + n_target - len(full)))
        cropped = full[crop_start : crop_start + n_target]
        realizations[s, :] = _renormalize(cropped)

    median_curve = np.median(realizations, axis=0)
    return _renormalize(median_curve).tolist()


# ---------------------------------------------------------------------------
# 8. Orchestration
# ---------------------------------------------------------------------------


class DesignStormEngine:
    def __init__(
        self,
        ietd_hours=DEFAULT_IETD_HOURS,
        min_event_depth_mm=DEFAULT_MIN_EVENT_DEPTH_MM,
        min_sample_size=DEFAULT_MIN_SAMPLE_SIZE,
        mc_sims=DEFAULT_MC_SIMS,
        mc_alpha=DEFAULT_MC_ALPHA,
        random_seed=DEFAULT_RANDOM_SEED,
        pool_sparse_bins=DEFAULT_POOL_SPARSE_BINS,
        target_durations=None,
    ):
        self.ietd_hours = ietd_hours
        self.min_event_depth_mm = min_event_depth_mm
        self.min_sample_size = min_sample_size
        self.mc_sims = mc_sims
        self.mc_alpha = mc_alpha
        self.random_seed = random_seed
        self.pool_sparse_bins = pool_sparse_bins
        self.target_durations = target_durations or TARGET_DURATIONS
        self.warnings = []

    @staticmethod
    def _bin_summary(
        duration_minutes,
        timestep_minutes,
        band,
        shape,
        n,
        status,
        pooled_from="",
        n_native=None,
    ):
        return {
            "Duration": duration_minutes,
            "TimeStep": timestep_minutes,
            "AEP": band,
            "Shape": shape,
            "n_storms": n,
            "n_storms_own_duration": n_native if n_native is not None else n,
            "status": status,
            "pooled_from_durations": pooled_from,
        }

    def run(self, timeseries_path, ddf_path, datetime_col=None, value_col=None):
        rng = np.random.default_rng(self.random_seed)

        series, native_minutes = load_timeseries(
            timeseries_path, datetime_col, value_col
        )
        ddf_long = load_ddf_table(ddf_path)
        ddf_table = DDFTable(ddf_long)

        events = extract_events(series, self.ietd_hours, self.min_event_depth_mm)
        if not events:
            raise ValueError(
                "No storm events were identified - check the IETD, minimum event "
                "depth threshold, and that the rainfall column contains incremental "
                "(not cumulative) depths."
            )

        per_duration_windows = {}
        catalogue_rows = []

        for duration_minutes, _timestep_minutes, _n_increments in self.target_durations:
            windows = []
            for event in events:
                w = extract_duration_window(
                    series, native_minutes, event, duration_minutes
                )
                if w is not None:
                    windows.append(w)
            windows = dedupe_windows(windows)
            per_duration_windows[duration_minutes] = windows

        for duration_minutes, _timestep_minutes, _n_increments in self.target_durations:
            windows = per_duration_windows[duration_minutes]
            population_depths = [w["depth_mm"] for w in windows]
            for w in windows:
                aep_pct, in_range = ddf_table.assign_aep(
                    duration_minutes, w["depth_mm"]
                )
                method = "ddf"
                if not in_range:
                    aep_pct = assign_aep_empirical(population_depths, w["depth_mm"])
                    method = "empirical"
                w["aep_percent"] = aep_pct
                w["aep_band"] = aep_band(aep_pct)
                w["shape"] = classify_shape(w)
                w["aep_method"] = method
                catalogue_rows.append(
                    {
                        "duration_minutes": duration_minutes,
                        "event_id": w["event_id"],
                        "start": w["start"],
                        "end": w["end"],
                        "depth_mm": round(w["depth_mm"], 3),
                        "aep_percent": round(aep_pct, 3),
                        "aep_band": w["aep_band"],
                        "aep_method": method,
                        "shape": w["shape"],
                    }
                )

        direct_curves = {}
        bin_summary_rows = []
        direct_tier = [
            (d, ts, n) for d, ts, n in self.target_durations if ts >= native_minutes
        ]
        direct_tier_durations = [d for d, ts, n in direct_tier]

        for duration_minutes, timestep_minutes, n_increments in direct_tier:
            windows = per_duration_windows[duration_minutes]
            for band in AEP_BANDS:
                for shape in SHAPES:
                    bin_windows = [
                        w
                        for w in windows
                        if w["aep_band"] == band and w["shape"] == shape
                    ]
                    n = len(bin_windows)
                    if n >= self.min_sample_size:
                        curve = build_direct_bin_curve(bin_windows, n_increments)
                        direct_curves[(duration_minutes, band, shape)] = curve
                        bin_summary_rows.append(
                            self._bin_summary(
                                duration_minutes,
                                timestep_minutes,
                                band,
                                shape,
                                n,
                                "direct_from_data",
                            )
                        )
                        continue

                    if not self.pool_sparse_bins:
                        bin_summary_rows.append(
                            self._bin_summary(
                                duration_minutes,
                                timestep_minutes,
                                band,
                                shape,
                                n,
                                "insufficient_data",
                            )
                        )
                        continue

                    pooled_windows = list(bin_windows)
                    pooled_from = []
                    this_idx = direct_tier_durations.index(duration_minutes)
                    neighbours = sorted(
                        [d for d in direct_tier_durations if d != duration_minutes],
                        key=lambda d: abs(direct_tier_durations.index(d) - this_idx),
                    )
                    for neighbour_duration in neighbours:
                        if len(pooled_windows) >= self.min_sample_size:
                            break
                        extra = [
                            w
                            for w in per_duration_windows[neighbour_duration]
                            if w["aep_band"] == band and w["shape"] == shape
                        ]
                        if extra:
                            pooled_windows.extend(extra)
                            pooled_from.append(neighbour_duration)

                    if len(pooled_windows) >= self.min_sample_size:
                        curve = build_direct_bin_curve(pooled_windows, n_increments)
                        direct_curves[(duration_minutes, band, shape)] = curve
                        bin_summary_rows.append(
                            self._bin_summary(
                                duration_minutes,
                                timestep_minutes,
                                band,
                                shape,
                                len(pooled_windows),
                                "direct_from_data_pooled",
                                pooled_from=",".join(str(d) for d in pooled_from),
                                n_native=n,
                            )
                        )
                    else:
                        bin_summary_rows.append(
                            self._bin_summary(
                                duration_minutes,
                                timestep_minutes,
                                band,
                                shape,
                                len(pooled_windows),
                                "insufficient_data_even_pooled",
                                pooled_from=",".join(str(d) for d in pooled_from),
                                n_native=n,
                            )
                        )

        candidates = [d for d, ts, n in self.target_durations if ts >= native_minutes]
        if candidates:
            anchor_duration = min(candidates)
        else:
            anchor_duration = max(d for d, ts, n in self.target_durations)
            self.warnings.append(
                f"Native resolution ({native_minutes:.0f} min) is coarser than every "
                f"target TimeStep; using {anchor_duration} min as the best-available "
                f"anchor. All results are Monte Carlo-derived and approximate."
            )
        anchor_ts = next(
            ts for d, ts, n in self.target_durations if d == anchor_duration
        )

        for duration_minutes, timestep_minutes, n_increments in self.target_durations:
            if timestep_minutes >= native_minutes:
                continue
            for band in AEP_BANDS:
                for shape in SHAPES:
                    anchor_curve = direct_curves.get((anchor_duration, band, shape))
                    if anchor_curve is None:
                        bin_summary_rows.append(
                            self._bin_summary(
                                duration_minutes,
                                timestep_minutes,
                                band,
                                shape,
                                0,
                                "insufficient_data_no_anchor",
                            )
                        )
                        continue
                    curve = disaggregate_bin_curve(
                        anchor_curve,
                        anchor_ts,
                        duration_minutes,
                        timestep_minutes,
                        n_increments,
                        self.mc_sims,
                        self.mc_alpha,
                        rng,
                    )
                    direct_curves[(duration_minutes, band, shape)] = curve
                    bin_summary_rows.append(
                        self._bin_summary(
                            duration_minutes,
                            timestep_minutes,
                            band,
                            shape,
                            len(per_duration_windows[duration_minutes]),
                            "monte_carlo_disaggregated",
                        )
                    )

        ensemble_rows = []
        for duration_minutes, timestep_minutes, _n_increments in self.target_durations:
            for band in AEP_BANDS:
                for shape in SHAPES:
                    curve = direct_curves.get((duration_minutes, band, shape))
                    ensemble_rows.append(
                        {
                            "Duration": duration_minutes,
                            "TimeStep": timestep_minutes,
                            "AEP": band,
                            "Shape": shape,
                            "Increments": [round(v, 2) for v in curve] if curve else [],
                        }
                    )

        return {
            "ensemble_rows": ensemble_rows,
            "catalogue_rows": catalogue_rows,
            "bin_summary_rows": bin_summary_rows,
            "native_minutes": native_minutes,
            "n_events": len(events),
            "warnings": self.warnings,
        }
