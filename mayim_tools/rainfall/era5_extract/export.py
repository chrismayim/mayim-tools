"""CSV export for ERA5/ERA5-Land point extraction results."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def write_era5_csv(results_by_site: dict, path: str | Path) -> tuple:
    """results_by_site: {site_label: Era5Result}. Writes a single
    combined CSV with a Site column. Returns (n_rows, n_missing).

    Uses itertuples(), NOT iterrows() - a real bug found on a live
    4-hour run: iterrows() returns each row as a single pd.Series,
    which must have one uniform dtype. A row mixing a datetime64 value
    (ValidTime) with a float that happens to be NaN gets silently
    coerced by pandas into NaT for THAT cell - confirmed by direct
    reproduction, not assumed - so round(row["PrecipitationMM"], 4)
    crashed with "type NaTType doesn't define __round__ method" the
    moment any row's precipitation value was genuinely missing.
    itertuples() preserves each column's own dtype correctly instead.

    A missing (NaN) value is written as an empty CSV cell, not
    silently dropped or defaulted to zero - and every missing value is
    counted and returned so the caller can surface it as a warning
    rather than this being invisible until a downstream crash, which
    is exactly what happened before this fix."""
    n_rows = 0
    n_missing = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Site", "ValidTime", "PrecipitationMM"])
        for site, result in results_by_site.items():
            for row in result.dataframe.itertuples(index=False):
                value = row.PrecipitationMM
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    n_missing += 1
                    w.writerow([site, row.ValidTime, ""])
                else:
                    w.writerow([site, row.ValidTime, round(value, 4)])
                n_rows += 1
    return n_rows, n_missing


def write_era5_anomalies(results_by_site: dict, path: str | Path) -> int:
    """Audit CSV: every large-negative-after-deaccumulation row across
    all sites, with full context (reference_time, step_hours,
    valid_time, the raw accumulated value, and the resulting negative
    increment) - added after a live run flagged 12,387 such values
    with only a bare count in the log, which wasn't enough to
    distinguish real GRIB-packing noise from a genuine bug. Returns
    the number of rows written (0 if there were no anomalies for any
    site)."""
    n_rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Site",
                "ReferenceTime",
                "StepHours",
                "ValidTime",
                "AccumulatedMM",
                "PrecipitationMMRaw",
            ]
        )
        for site, result in results_by_site.items():
            if result.anomalies is None or len(result.anomalies) == 0:
                continue
            for row in result.anomalies.itertuples(index=False):
                w.writerow(
                    [
                        site,
                        row.reference_time,
                        row.step_hours,
                        row.valid_time,
                        round(row.accumulated_mm, 4),
                        round(row.precipitation_mm_raw, 4),
                    ]
                )
                n_rows += 1
    return n_rows
