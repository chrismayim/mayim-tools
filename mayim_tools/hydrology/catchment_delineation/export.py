"""CSV writers for the catchment delineation tool (no QGIS dependency)."""

from __future__ import annotations

import csv
from collections.abc import Sequence

from .core import CatchmentResult


def write_parameters_csv(path: str, catchments: Sequence[CatchmentResult]) -> None:
    """One row per catchment with every computed parameter - the same
    values as the polygon attribute table, for use as a report annexure."""
    if not catchments:
        fieldnames = ["outlet_id"]
    else:
        fieldnames = list(catchments[0].attributes.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in catchments:
            writer.writerow(
                {k: ("" if v is None else v) for k, v in c.attributes.items()}
            )


def write_hypsometric_csv(path: str, catchments: Sequence[CatchmentResult]) -> None:
    """Long-format hypsometric curves: one row per catchment per relative
    height step."""
    fieldnames = [
        "outlet_id",
        "rel_height",
        "rel_area",
        "elevation_m",
        "area_above_km2",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in catchments:
            for row in c.hypsometric_curve:
                writer.writerow(
                    {
                        "outlet_id": c.outlet_id,
                        "rel_height": row["rel_height"],
                        "rel_area": round(row["rel_area"], 6),
                        "elevation_m": round(row["elevation_m"], 3),
                        "area_above_km2": round(row["area_above_km2"], 6),
                    }
                )
