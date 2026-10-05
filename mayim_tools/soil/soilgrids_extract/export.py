"""File writers for Extract: SoilGrids 2.0 - thin wrappers around the
shared soil writers (mayim_tools.soil._common.export)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mayim_tools.soil._common import export as _export
from mayim_tools.soil._common.export import (  # noqa: F401 - re-exported
    NODATA_OUT,
    write_metadata_csv,
    write_style_sidecar,
)

from .core import TOOL_VERSION, TargetGrid

POINT_COLUMNS = [
    "Site",
    "Longitude",
    "Latitude",
    "Product",
    "Variable",
    "Description",
    "DepthTop_cm",
    "DepthBottom_cm",
    "Statistic",
    "Value",
    "Units",
    "Route",
]


def write_multiband_geotiff(
    path: str | Path,
    grid: TargetGrid,
    bands: list[tuple[str, np.ndarray]],
    units: str,
) -> None:
    _export.write_multiband_geotiff(
        path,
        grid,
        bands,
        units,
        source="ISRIC SoilGrids (via mayim_tools)",
        tool_version=TOOL_VERSION,
    )


def write_points_csv(rows: list[dict], path: str | Path) -> tuple[int, int]:
    return _export.write_points_csv(rows, path, POINT_COLUMNS)


def metadata_path_for(output: str | Path, mode: str) -> Path:
    return _export.metadata_path_for(output, mode, "soilgrids_metadata.csv")
