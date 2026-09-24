"""Delimiter-tolerant CSV reading shared by the rainfall tools."""

from __future__ import annotations

import pandas as pd


def read_csv_any_delimiter(path):
    """Read a CSV that might actually be comma-, tab-, or semicolon-
    delimited (users regularly paste/export tab-separated data with a
    '.csv' extension) by sniffing the delimiter rather than assuming
    comma."""
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] == 1:
        for sep in ["\t", ";", ","]:
            candidate = pd.read_csv(path, sep=sep)
            if candidate.shape[1] > 1:
                return candidate
    return df
