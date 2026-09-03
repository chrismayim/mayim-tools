"""
Input parsing and validation (paper Section 3.1-3.2).

Key principle carried through everywhere in this pipeline: missing
rainfall is NEVER silently converted to zero. A blank/NaN cell stays
NaN and is tagged quality_code='missing'; only a genuine parsed 0.0 is
tagged 'zero'. This distinction matters because treating missing data
as dry can artificially split real storms across a data gap.
"""

from __future__ import annotations

import pandas as pd


def parse_and_validate(
    df: "pd.DataFrame",
    timestamp_col: str,
    depth_col: str,
    timestamp_format: str | None = None,
) -> tuple["pd.DataFrame", dict]:
    """Parses timestamps, sorts chronologically, and runs the paper's
    Section 3.2 checks. Returns (clean_df, diagnostics) where clean_df
    has columns: timestamp (parsed, sorted), depth_mm (float, NaN kept
    as NaN), source_row (original row index before sorting).

    Raises ValueError if the required columns are missing.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column {timestamp_col!r} not found. Available: {list(df.columns)}")
    if depth_col not in df.columns:
        raise ValueError(f"Depth column {depth_col!r} not found. Available: {list(df.columns)}")

    diagnostics: dict = {"warnings": []}

    work = pd.DataFrame({
        "source_row": df.index,
        "timestamp_raw": df[timestamp_col],
        "depth_raw": df[depth_col],
    })

    work["timestamp"] = pd.to_datetime(work["timestamp_raw"], format=timestamp_format, errors="coerce")
    n_bad_ts = int(work["timestamp"].isna().sum())
    if n_bad_ts:
        diagnostics["warnings"].append(f"{n_bad_ts} row(s) had unparseable timestamps and were dropped.")
    work = work.dropna(subset=["timestamp"]).copy()

    # depth: coerce to numeric, but NEVER fill NaN with 0 - a blank cell
    # or non-numeric token becomes NaN (missing), not zero.
    work["depth_mm"] = pd.to_numeric(work["depth_raw"], errors="coerce")
    n_nonnumeric = int(work["depth_raw"].notna().sum() - work["depth_mm"].notna().sum())
    if n_nonnumeric:
        diagnostics["warnings"].append(
            f"{n_nonnumeric} depth value(s) were non-numeric and are treated as missing (not zero)."
        )

    n_negative = int((work["depth_mm"] < 0).sum())
    if n_negative:
        diagnostics["warnings"].append(
            f"{n_negative} negative depth value(s) found - treated as missing (not zero), "
            "since negative rainfall is physically invalid and likely a sensor/encoding artefact."
        )
        work.loc[work["depth_mm"] < 0, "depth_mm"] = pd.NA

    work = work.sort_values("timestamp").reset_index(drop=True)

    n_dupe = int(work["timestamp"].duplicated().sum())
    if n_dupe:
        diagnostics["warnings"].append(
            f"{n_dupe} duplicate timestamp(s) found. Keeping the first occurrence of each; "
            "review the source file if this is unexpected."
        )
        work = work.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    diagnostics["n_rows_in"] = len(df)
    diagnostics["n_rows_valid"] = len(work)
    diagnostics["timestamp_range"] = (
        (str(work["timestamp"].min()), str(work["timestamp"].max())) if len(work) else (None, None)
    )

    return work[["source_row", "timestamp", "depth_mm"]], diagnostics
