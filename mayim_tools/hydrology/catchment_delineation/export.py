"""CSV writers for the catchment delineation tool (no QGIS dependency).

Mayim output standard: UTF-8 with a byte-order mark (so Excel opens the
files cleanly) and, because the headings are abbreviated, a column key
above the header as '# column: meaning' lines (read with pandas
``comment='#'``)."""

from __future__ import annotations

import csv
from collections.abc import Sequence

from .core import CatchmentResult

CSV_ENCODING = "utf-8-sig"
KEY_PREFIX = "# "

HYPSOMETRIC_KEY = {
    "outlet_id": "outlet name (as in the catchments layer)",
    "rel_height": "relative height above the outlet (z - z_min) / (z_max - z_min), 0-1",
    "rel_area": "fraction of the catchment area above that height, 0-1",
    "elevation_m": "elevation at that relative height (m)",
    "area_above_km2": "catchment area above that elevation (km2)",
}


def _write_key(f, key, columns) -> None:
    rows = [(c, key[c]) for c in columns if key.get(c)]
    if not rows:
        return
    f.write(f"{KEY_PREFIX}Column key\n")
    for col, meaning in rows:
        f.write(f"{KEY_PREFIX}{col}: {meaning}\n")


def write_parameters_csv(
    path: str,
    catchments: Sequence[CatchmentResult],
    key: dict[str, str] | None = None,
) -> None:
    """One row per catchment with every computed parameter - the same
    values as the polygon attribute table, for use as a report annexure.
    ``key`` (field -> meaning) is written above the header for the fields
    present."""
    if not catchments:
        fieldnames = ["outlet_id"]
    else:
        fieldnames = list(catchments[0].attributes.keys())
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        _write_key(f, key or {}, fieldnames)
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
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        _write_key(f, HYPSOMETRIC_KEY, fieldnames)
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
