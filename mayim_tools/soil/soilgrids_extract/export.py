"""File writers for Extract: SoilGrids 2.0 (GeoTIFF, point CSV, metadata CSV)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .core import NODATA_OUT, TOOL_VERSION, TargetGrid

POINT_COLUMNS = [
    "Site",
    "Longitude",
    "Latitude",
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
    """One Float32 GeoTIFF, one band per (description, array).

    Values are already in conventional units. NaN is written as the
    output nodata value (-9999); nodata is never written as zero.
    DEFLATE with the floating-point predictor keeps files small."""
    from osgeo import gdal

    gdal.UseExceptions()
    if not bands:
        raise ValueError("No bands to write.")
    rows, cols = bands[0][1].shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        str(path),
        cols,
        rows,
        len(bands),
        gdal.GDT_Float32,
        options=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
        ],
    )
    ds.SetGeoTransform(grid.geotransform)
    ds.SetProjection(grid.crs_wkt)
    ds.SetMetadataItem("SOURCE", "ISRIC SoilGrids (via mayim_tools)")
    ds.SetMetadataItem("TOOL_VERSION", TOOL_VERSION)
    for i, (description, arr) in enumerate(bands, start=1):
        if arr.shape != (rows, cols):
            raise ValueError(
                f"Band {i} ({description}) has shape {arr.shape}, "
                f"expected {(rows, cols)}."
            )
        out = np.where(np.isfinite(arr), arr, NODATA_OUT).astype(np.float32)
        band = ds.GetRasterBand(i)
        band.SetNoDataValue(NODATA_OUT)
        band.SetDescription(description)
        band.SetUnitType(units)
        band.WriteArray(out)
    ds.FlushCache()
    ds = None


def write_points_csv(rows: list[dict], path: str | Path) -> tuple[int, int]:
    """Long-format point CSV. Missing values are empty cells, never zero.
    Returns (n_rows, n_missing)."""
    n_missing = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(POINT_COLUMNS)
        for row in rows:
            value = row["Value"]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                n_missing += 1
                value = ""
            else:
                value = round(float(value), 4)
            w.writerow(
                [
                    row["Site"],
                    round(float(row["Longitude"]), 6),
                    round(float(row["Latitude"]), 6),
                    row["Variable"],
                    row["Description"],
                    row["DepthTop_cm"],
                    row["DepthBottom_cm"],
                    row["Statistic"],
                    value,
                    row["Units"],
                    row["Route"],
                ]
            )
    return len(rows), n_missing


def write_metadata_csv(
    sections: list[tuple[str, list[str], list[list]]], path: str | Path
) -> None:
    """Sectioned CSV: '# <title>' line, header, rows, blank line."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for title, header, rows in sections:
            w.writerow([f"# {title}"])
            w.writerow(header)
            for row in rows:
                w.writerow(row)
            w.writerow([])


def metadata_path_for(output: str | Path, mode: str) -> Path:
    """Area mode: <folder>/soilgrids_metadata.csv.
    Point mode: <csv stem>_metadata.csv next to the point CSV."""
    output = Path(output)
    if mode == "area":
        return output / "soilgrids_metadata.csv"
    return output.with_name(output.stem + "_metadata.csv")
