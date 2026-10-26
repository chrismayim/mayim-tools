"""CSV writers for the bathymetry estimate of DEM depression stage-storage
(no QGIS dependency). UTF-8 with a byte-order mark, starting with a column
key ('#' lines)."""

from __future__ import annotations

import csv
from collections.abc import Sequence

from .bathymetry import SCENARIOS, BathymetryResult
from .export import CSV_ENCODING, write_key

SUMMARY_KEY = {
    "storage_id": "storage area name",
    "row_type": "scenario (low / central / high bed) or method",
    "name": "scenario or method name",
    "water_level_m": "surveyed water level, m",
    "water_area_m2": "surveyed water-surface area, m2",
    "max_depth_m": "maximum depth below the water level, m",
    "mean_depth_m": "mean depth below the water level, m",
    "volume_m3": "volume below the surveyed water surface, m3",
    "status_or_basis": "basis of the scenario, or the method's status",
}
PROFILE_KEY = {
    "storage_id": "storage area name",
    "chainage_m": "distance along the centreline from its start, m",
    "from_entry_m": "distance downstream from where the river enters the water, m",
    "x": "easting, m",
    "y": "northing, m",
    "in_water": "1 inside the surveyed water surface, else 0",
    "dem_z": "DEM level, m",
    "bed_low": "estimated bed level - low (shallow), m",
    "bed_central": "estimated bed level - central, m",
    "bed_high": "estimated bed level - high (deep), m",
}

METHOD_LABELS = {
    "slope": "Shoreline slope (Hollister et al., 2011)",
    "hypsometry": "Hypsometry above the water (after Heathcote et al., 2015)",
    "valley": "Valley profile along the stream centreline",
}


def _fmt(value, digits=3):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def write_bathymetry_summary_csv(
    path: str, results: Sequence[BathymetryResult]
) -> None:
    """One row per storage area and scenario, plus one per method."""
    fields = [
        "storage_id",
        "row_type",
        "name",
        "water_level_m",
        "water_area_m2",
        "max_depth_m",
        "mean_depth_m",
        "volume_m3",
        "status_or_basis",
    ]
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        write_key(f, SUMMARY_KEY, fields)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            common = {
                "storage_id": r.storage_id,
                "water_level_m": _fmt(r.water_level),
                "water_area_m2": _fmt(r.water_area_m2, 1),
            }
            for name in SCENARIOS:
                sc = r.scenarios[name]
                writer.writerow(
                    {
                        **common,
                        "row_type": "scenario",
                        "name": name,
                        "max_depth_m": _fmt(sc["max_depth"]),
                        "mean_depth_m": _fmt(sc["mean_depth"]),
                        "volume_m3": _fmt(sc["volume_m3"], 1),
                        "status_or_basis": r.basis,
                    }
                )
            for key, label in METHOD_LABELS.items():
                m = r.methods.get(key, {})
                writer.writerow(
                    {
                        **common,
                        "row_type": "method",
                        "name": label,
                        "max_depth_m": _fmt(m.get("max_depth")),
                        "mean_depth_m": "",
                        "volume_m3": "",
                        "status_or_basis": m.get("status", ""),
                    }
                )


def write_profile_csv(path: str, results: Sequence[BathymetryResult]) -> None:
    """Centreline profiles: DEM and estimated bed along each centreline."""
    fields = [
        "storage_id",
        "chainage_m",
        "from_entry_m",
        "x",
        "y",
        "in_water",
        "dem_z",
        "bed_low",
        "bed_central",
        "bed_high",
    ]
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        write_key(f, PROFILE_KEY, fields)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            for row in r.profile:
                writer.writerow(
                    {
                        "storage_id": r.storage_id,
                        **{k: ("" if v is None else v) for k, v in row.items()},
                    }
                )
