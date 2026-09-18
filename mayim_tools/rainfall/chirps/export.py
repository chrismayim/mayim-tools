"""CSV export for CHIRPS point extraction results."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def write_chirps_csv(results_by_site: dict, path: str | Path) -> int:
    """results_by_site: {site_label: ChirpsResult}. Writes a single
    combined CSV with a Site column. Returns the row count."""
    n_rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Site", "Date", "PrecipitationMM"])
        for site, result in results_by_site.items():
            for _, row in result.dataframe.iterrows():
                value = row["PrecipitationMM"]
                w.writerow(
                    [site, row["Date"], round(value, 3) if _is_finite(value) else ""]
                )
                n_rows += 1
    return n_rows


def _is_finite(value) -> bool:
    try:
        return not math.isnan(value)
    except TypeError:
        return True
