"""CSV writer for the inundation boundary tool (no QGIS dependency)."""

from __future__ import annotations

import csv

from .core import InundationResult

SUMMARY_FIELDS = [
    "level_m",
    "area_m2",
    "area_ha",
    "volume_m3",
    "max_depth_m",
    "mean_depth_m",
    "wet_cells",
    "n_polygons",
    "polygon_area_m2",
    "perimeter_m",
]


def write_summary_csv(path: str, result: InundationResult, geometry=None) -> None:
    """One row per level. Area/volume are counted on the DEM cells (inside the
    bounding polygon); ``geometry`` maps level -> (polygon area, perimeter,
    number of polygons) measured on the clipped output polygons, if given."""
    geometry = geometry or {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for lv in result.levels:
            g = geometry.get(lv.level)
            writer.writerow(
                {
                    "level_m": f"{lv.level:.3f}",
                    "area_m2": f"{lv.area_m2:.1f}",
                    "area_ha": f"{lv.area_m2 / 1e4:.4f}",
                    "volume_m3": f"{lv.volume_m3:.1f}",
                    "max_depth_m": f"{lv.max_depth:.3f}",
                    "mean_depth_m": (
                        f"{lv.volume_m3 / lv.area_m2:.3f}" if lv.area_m2 > 0 else ""
                    ),
                    "wet_cells": lv.wet_cells,
                    "n_polygons": "" if g is None else g[2],
                    "polygon_area_m2": "" if g is None else f"{g[0]:.1f}",
                    "perimeter_m": "" if g is None else f"{g[1]:.1f}",
                }
            )
