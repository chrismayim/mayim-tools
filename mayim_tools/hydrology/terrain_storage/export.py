"""Table writers for DEM depression stage-storage (no QGIS dependency)."""

from __future__ import annotations

import csv
from collections.abc import Sequence

from .core import StorageResult

TABLE_FIELDS = [
    "stage_m",
    "depth_m",
    "area_m2",
    "area_ha",
    "volume_m3",
    "volume_ML",
    "inc_vol_m3",
    "wet_cells",
]
RANGE_FIELDS = ["area_low_m2", "volume_low_m3", "area_high_m2", "volume_high_m3"]


def write_stage_table_csv(path: str, results: Sequence[StorageResult]) -> None:
    """Long format: one row per storage area per stage. When the bed was
    estimated below a surveyed water surface, the low and high estimates
    add area/volume columns (central = the main columns)."""
    fields = ["storage_id", *TABLE_FIELDS]
    if any(r.range_tables for r in results):
        fields += RANGE_FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            for row in r.table:
                writer.writerow({"storage_id": r.storage_id, **row})


def write_hec_table(path: str, results: Sequence[StorageResult]) -> None:
    """Tab-delimited elevation-volume-area blocks, one per storage area,
    ready to paste into a HEC-RAS storage area (Elevation-Volume curve)
    or a HEC-HMS reservoir (Elevation-Storage / Elevation-Area paired
    data). SI units as both programs expect them: elevation m, volume
    1000 m3, area 1000 m2."""
    lines = []
    for r in results:
        lines.append(f"# Storage area: {r.storage_id}")
        lines.append(
            f"# Floor {r.floor_z:.3f} m, top {r.top_stage:.3f} m "
            f"({r.top_source}), mode: {r.mode}"
        )
        lines.append("Elevation (m)\tVolume (1000 m3)\tArea (1000 m2)")
        for row in r.table:
            lines.append(
                f"{row['stage_m']:.3f}\t{row['volume_m3'] / 1e3:.4f}"
                f"\t{row['area_m2'] / 1e3:.4f}"
            )
        lines.append("")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("\r\n".join(lines))
