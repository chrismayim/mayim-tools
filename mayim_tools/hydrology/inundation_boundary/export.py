"""CSV writer for the inundation boundary tool (no QGIS dependency). UTF-8
with BOM, so Excel shows units correctly (Mayim output standard)."""

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
    """One row for the water level. Area/volume are counted on the DEM cells
    (inside the bounding polygon); ``geometry`` is (polygon area, perimeter,
    number of polygons) measured on the clipped output polygons, if given."""
    g = geometry
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "level_m": f"{result.level:.3f}",
                "area_m2": f"{result.area_m2:.1f}",
                "area_ha": f"{result.area_m2 / 1e4:.4f}",
                "volume_m3": f"{result.volume_m3:.1f}",
                "max_depth_m": f"{result.max_depth:.3f}",
                "mean_depth_m": (
                    "" if result.mean_depth is None else f"{result.mean_depth:.3f}"
                ),
                "wet_cells": result.wet_cells,
                "n_polygons": "" if g is None else g[2],
                "polygon_area_m2": "" if g is None else f"{g[0]:.1f}",
                "perimeter_m": "" if g is None else f"{g[1]:.1f}",
            }
        )
