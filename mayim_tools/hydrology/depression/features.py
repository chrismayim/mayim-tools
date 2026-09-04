"""
Mayim Tools - Depression Feature Calculations
==============================================

Calculates descriptive properties for depressions identified by the
native Mayim depression-detection module.

This module implements feature extraction only. It does not:

    - Modify DEM elevations.
    - Fill depressions.
    - Breach depressions.
    - Resolve flat areas.
    - Classify depressions.
    - Call WhiteboxTools.
    - Call RichDEM.

The calculated features support Stage 4 depression classification.

IP status
---------
Original Mayim implementation using NumPy and Python standard-library
components. No third-party hydrological implementation is used.

The feature definitions follow the Mayim Tools DEM Hydrological
Conditioning methodology, including:

    - Pit elevation.
    - Spill elevation.
    - Depression depth.
    - Area.
    - Perimeter.
    - Approximate volume.
    - Boundary contact.
    - Compactness proxy.

These features are intended to be inspectable inputs to the Stage 4
artifact-likelihood classifier.
"""

from __future__ import annotations

import numpy as np

# Eight-connected neighbourhood used for depression adjacency.
_D8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

# Four-connected neighbourhood used for the perimeter-cell definition.
_CARDINAL = [
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 0),
]


def calculate_depression_features(
    dem: np.ndarray,
    depression_ids: np.ndarray,
    spill_points: dict[int, float],
    cell_size: float,
    nodata: float,
) -> dict[int, dict]:
    """
    Calculate descriptive features for every depression.

    A depression is represented by all cells in ``depression_ids`` that
    have the same positive integer ID.

    The approximate volume is calculated as:

        sum(max(spill_elevation - cell_elevation, 0))
        multiplied by cell area.

    This is a static volume estimate relative to the depression spill
    elevation. It is not a dynamic fill-spill simulation.

    Parameters
    ----------
    dem:
        Two-dimensional elevation array.
    depression_ids:
        Two-dimensional integer array. Positive values identify
        depression regions. Zero identifies non-depression cells.
        Negative values identify NoData cells.
    spill_points:
        Mapping from depression ID to spill elevation.
    cell_size:
        Mean cell size in map units. The current API assumes square
        cells.
    nodata:
        NoData sentinel used by the DEM.

    Returns
    -------
    dict[int, dict]
        Mapping from depression ID to a dictionary of calculated
        depression features.

    Raises
    ------
    ValueError
        If input arrays have incompatible shapes, the DEM is not
        two-dimensional, or cell_size is not positive.
    """
    _validate_inputs(
        dem=dem,
        depression_ids=depression_ids,
        cell_size=cell_size,
    )

    features: dict[int, dict] = {}

    depression_values = np.unique(depression_ids)
    depression_values = [
        int(depression_id) for depression_id in depression_values if depression_id > 0
    ]

    for depression_id in sorted(depression_values):
        mask = depression_ids == depression_id
        rows, cols = np.where(mask)

        if len(rows) == 0:
            continue

        valid_cells = mask & np.isfinite(dem)

        # Exclude cells matching the DEM NoData value.
        valid_cells &= dem != nodata
        valid_rows, valid_cols = np.where(valid_cells)

        if len(valid_rows) == 0:
            continue

        pit_index = int(np.argmin(dem[valid_cells]))
        pit_row = int(valid_rows[pit_index])
        pit_col = int(valid_cols[pit_index])
        pit_elevation = float(dem[pit_row, pit_col])

        spill_elevation = float(
            spill_points.get(
                depression_id,
                pit_elevation,
            )
        )

        depth = spill_elevation - pit_elevation

        if depth < 0:
            raise ValueError(
                f"Depression {depression_id} has a spill elevation "
                f"({spill_elevation}) below its pit elevation "
                f"({pit_elevation})."
            )

        area_cells = int(np.sum(valid_cells))
        cell_area = float(cell_size**2)
        area_map_units = float(area_cells * cell_area)

        depression_elevations = dem[valid_cells].astype(np.float64)
        cell_depths = np.maximum(
            spill_elevation - depression_elevations,
            0.0,
        )
        volume_estimate = float(np.sum(cell_depths) * cell_area)

        perimeter_cells = _count_perimeter_cells(
            depression_ids=depression_ids,
            depression_id=depression_id,
            rows=rows,
            cols=cols,
        )

        touches_boundary = _touches_boundary(
            rows=rows,
            cols=cols,
            height=dem.shape[0],
            width=dem.shape[1],
        )

        elongation_index = _calculate_compactness(
            area_cells=area_cells,
            perimeter_cells=perimeter_cells,
        )

        features[depression_id] = {
            "depression_id": depression_id,
            "pit_row": pit_row,
            "pit_col": pit_col,
            "pit_elevation": pit_elevation,
            "spill_elevation": spill_elevation,
            "depth": float(depth),
            "area_cells": area_cells,
            "area_map_units": area_map_units,
            "perimeter_cells": perimeter_cells,
            "volume_estimate": volume_estimate,
            "touches_boundary": touches_boundary,
            "elongation_index": elongation_index,
        }

    return features


def _validate_inputs(
    dem: np.ndarray,
    depression_ids: np.ndarray,
    cell_size: float,
) -> None:
    """
    Validate feature-calculation inputs.

    Parameters
    ----------
    dem:
        DEM array.
    depression_ids:
        Depression ID array.
    cell_size:
        Cell size in map units.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if dem.ndim != 2:
        raise ValueError(
            f"DEM must be two-dimensional; received {dem.ndim} dimensions."
        )

    if depression_ids.ndim != 2:
        raise ValueError("depression_ids must be a two-dimensional array.")

    if dem.shape != depression_ids.shape:
        raise ValueError("DEM and depression_ids must have identical shapes.")

    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive; received {cell_size}.")


def _count_perimeter_cells(
    depression_ids: np.ndarray,
    depression_id: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> int:
    """
    Count depression cells on the depression perimeter.

    A cell is a perimeter cell when at least one cardinal neighbour is:

    - Outside the raster.
    - Not part of the same depression.

    This counts perimeter cells, not perimeter edge length.
    """
    height, width = depression_ids.shape
    perimeter = 0

    for row, col in zip(rows.tolist(), cols.tolist()):
        for row_offset, col_offset in _CARDINAL:
            neighbour_row = row + row_offset
            neighbour_col = col + col_offset

            outside = not (0 <= neighbour_row < height and 0 <= neighbour_col < width)

            if outside:
                perimeter += 1
                break

            if depression_ids[neighbour_row, neighbour_col] != depression_id:
                perimeter += 1
                break

    return perimeter


def _touches_boundary(
    rows: np.ndarray,
    cols: np.ndarray,
    height: int,
    width: int,
) -> bool:
    """
    Return whether a depression touches the raster boundary.
    """
    return bool(
        np.any(rows == 0)
        or np.any(rows == height - 1)
        or np.any(cols == 0)
        or np.any(cols == width - 1)
    )


def _calculate_compactness(
    area_cells: int,
    perimeter_cells: int,
) -> float:
    """
    Calculate a compactness proxy between zero and one.

    The proxy is:

        4 * area / perimeter²

    It is not a formal geometric elongation measurement because it is
    based on raster-cell counts rather than vectorised geometry.
    """
    if area_cells <= 0 or perimeter_cells <= 0:
        return 0.0

    compactness = 4.0 * float(area_cells) / float(perimeter_cells**2)

    return float(np.clip(compactness, 0.0, 1.0))
