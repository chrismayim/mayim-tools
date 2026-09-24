"""
Input parsing and validation. Same principle as every other tool in
this suite: missing rainfall is never silently coerced to zero. A
blank/NaN cell stays NaN; only a genuine parsed 0.0 counts as a dry
reading.
"""

from __future__ import annotations

import pandas as pd


def parse_and_validate(
    df: pd.DataFrame,
    timestamp_col: str,
    depth_col: str,
    timestamp_format: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Returns (clean_df, diagnostics). clean_df has columns:
    timestamp (parsed, sorted, deduplicated), depth_mm (float, NaN
    kept as NaN)."""
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column {timestamp_col!r} not found. Available: {list(df.columns)}"
        )
    if depth_col not in df.columns:
        raise ValueError(
            f"Depth column {depth_col!r} not found. Available: {list(df.columns)}"
        )

    diagnostics: dict = {"warnings": []}

    work = pd.DataFrame(
        {
            "timestamp_raw": df[timestamp_col],
            "depth_raw": df[depth_col],
        }
    )
    work["timestamp"] = pd.to_datetime(
        work["timestamp_raw"], format=timestamp_format, errors="coerce"
    )
    n_bad_ts = int(work["timestamp"].isna().sum())
    if n_bad_ts:
        diagnostics["warnings"].append(
            f"{n_bad_ts} row(s) had unparseable timestamps and were dropped."
        )
    work = work.dropna(subset=["timestamp"]).copy()

    work["depth_mm"] = pd.to_numeric(work["depth_raw"], errors="coerce")
    n_nonnumeric = int(work["depth_raw"].notna().sum() - work["depth_mm"].notna().sum())
    if n_nonnumeric:
        diagnostics["warnings"].append(
            f"{n_nonnumeric} depth value(s) were non-numeric and are treated as missing (not zero)."
        )

    n_negative = int((work["depth_mm"] < 0).sum())
    if n_negative:
        diagnostics["warnings"].append(
            f"{n_negative} negative depth value(s) found - treated as missing (not zero)."
        )
        work.loc[work["depth_mm"] < 0, "depth_mm"] = pd.NA

    work = work.sort_values("timestamp").reset_index(drop=True)
    n_dupe = int(work["timestamp"].duplicated().sum())
    if n_dupe:
        diagnostics["warnings"].append(
            f"{n_dupe} duplicate timestamp(s) found - keeping the first occurrence of each."
        )
        work = work.drop_duplicates(subset="timestamp", keep="first").reset_index(
            drop=True
        )

    diagnostics["n_rows_in"] = len(df)
    diagnostics["n_rows_valid"] = len(work)
    diagnostics["timestamp_range"] = (
        (str(work["timestamp"].min()), str(work["timestamp"].max()))
        if len(work)
        else (None, None)
    )
    return work[["timestamp", "depth_mm"]], diagnostics
