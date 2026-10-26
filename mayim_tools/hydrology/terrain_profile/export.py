"""CSV writers for the Terrain Profile tool (no QGIS dependency).
Numbers are written without thousands separators. The CSVs are UTF-8 with
a byte-order mark and start with a column key ('#' lines): read them in
pandas with comment='#'."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence

import numpy as np

from .core import MEASUREMENT_FIELDS, SectionProfile

CSV_ENCODING = "utf-8-sig"
PROFILE_KEY = {
    "section": "section label",
    "chainage_m": "chainage along the section from its start, m",
    "x": "easting of the sample, m (DEM CRS)",
    "y": "northing of the sample, m (DEM CRS)",
    "ground_m": "ground level from the DEM, m",
    "wse_<event>_m": "water-surface elevation for the event, m (blank where dry)",
    "depth_<event>_m": "water depth for the event, m (blank where dry)",
}


def _write_key(f, key) -> None:
    f.write("# Column key\n")
    for col, meaning in key:
        f.write(f"# {col}: {meaning}\n")


def _num(value, decimals: int) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(f"{v:.{decimals}f}" for v in value)
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return ""
    return f"{value:.{decimals}f}"


def column_tag(label: str) -> str:
    """A CSV-safe column fragment for an event label."""
    tag = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
    return tag or "event"


def write_profile_csv(path: str, sections: Sequence[SectionProfile]) -> None:
    """Long table: one row per sample per section, with the ground level and
    a water-level and depth column per event (blank where dry)."""
    labels = []
    for s in sections:
        for e in s.events:
            if e.label not in labels:
                labels.append(e.label)
    header = ["section", "chainage_m", "x", "y", "ground_m"]
    for lab in labels:
        tag = column_tag(lab)
        header += [f"wse_{tag}_m", f"depth_{tag}_m"]
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        _write_key(f, PROFILE_KEY.items())
        writer = csv.writer(f)
        writer.writerow(header)
        for s in sections:
            by_label = {e.label: e for e in s.events}
            for i, ch in enumerate(s.chainage):
                row = [
                    s.label,
                    _num(ch, 3),
                    _num(s.x[i], 3),
                    _num(s.y[i], 3),
                    _num(s.ground[i], 3),
                ]
                for lab in labels:
                    e = by_label.get(lab)
                    if e is None or not np.isfinite(e.depth[i]):
                        row += ["", ""]
                    else:
                        row += [_num(e.wse[i], 3), _num(e.depth[i], 3)]
                writer.writerow(row)


_DECIMALS = {
    "wet_fraction_in_prop": 3,
    "wet_area_in_prop_m2": 1,
}


def write_measurements_csv(path: str, rows: Sequence[dict]) -> None:
    """One row per section per event (MEASUREMENT_FIELDS)."""
    names = [n for n, _ in MEASUREMENT_FIELDS]
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        _write_key(f, MEASUREMENT_FIELDS)
        writer = csv.writer(f)
        writer.writerow(names)
        for r in rows:
            writer.writerow([_num(r.get(n), _DECIMALS.get(n, 2)) for n in names])
