"""
Tests for mayim_tools/hydrology/d8_flow_accumulation/core.py. Zero
QGIS dependency - run under plain pytest.

Every direction-code assertion is checked against the exact encoding
shared with this suite's own d8_flow_direction plugin - not a generic/
assumed D8 convention. Includes the core proof-of-equivalence test:
the vectorised, topological-level (wave) accumulation implementation
actually used in core.py is checked against a naive, obviously-correct
one-cell-at-a-time reference implementation across several random
(but guaranteed-acyclic) synthetic drainage networks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mayim_tools.hydrology.d8_flow_accumulation.core import (
    D_COL,
    D_ROW,
    OUTPUT_NODATA,
    accumulate_flow_cells,
    cell_sizes_from_geotransform,
    compute_d8_flow_accumulation_from_file,
    compute_d8_pointer,
    compute_inflow_count,
    detect_interior_pits,
    pointer_to_direction_index,
    read_raster,
    scale_accumulation,
    write_raster,
)

NAN = np.nan


# ----------------------------------------------------------------------
# A naive, definitely-correct reference implementation of flow
# accumulation (one cell at a time via a stack - the same STRUCTURE as
# WhiteboxTools' own algorithm, just in slow, obviously-correct pure
# Python) - used ONLY to prove the vectorised, wave-based
# implementation in core.py produces IDENTICAL results, not just
# plausible-looking ones. This is deliberately NOT the implementation
# used in core.py itself.
# ----------------------------------------------------------------------


def naive_accumulate_flow_cells(direction_index: np.ndarray) -> np.ndarray:
    rows, cols = direction_index.shape
    is_nodata = direction_index == -2
    accumulation = np.where(is_nodata, np.nan, 1.0).astype(np.float64)
    inflow = compute_inflow_count(direction_index).astype(np.int64)
    inflow = np.where(is_nodata, -1, inflow)

    stack = [(r, c) for r in range(rows) for c in range(cols) if inflow[r, c] == 0]
    while stack:
        r, c = stack.pop()
        dir_idx = direction_index[r, c]
        if dir_idx >= 0:
            nr, nc = r + int(D_ROW[dir_idx]), c + int(D_COL[dir_idx])
            if 0 <= nr < rows and 0 <= nc < cols:
                accumulation[nr, nc] += accumulation[r, c]
                inflow[nr, nc] -= 1
                if inflow[nr, nc] == 0:
                    stack.append((nr, nc))
    return accumulation


def random_direction_index(rows, cols, seed):
    """Builds a genuinely random but VALID (acyclic, since every
    direction is chosen to point toward decreasing (row+col*cols)
    rank, guaranteeing no cycles - the same guarantee a properly
    computed D8 pointer on a real DEM provides) direction index grid,
    for equivalence-testing the accumulation algorithm at scale."""
    rng = np.random.default_rng(seed)
    direction_index = np.full((rows, cols), -1, dtype=np.int8)
    rank = rng.permutation(rows * cols).reshape(rows, cols)
    for r in range(rows):
        for c in range(cols):
            candidates = []
            for i in range(8):
                nr, nc = r + int(D_ROW[i]), c + int(D_COL[i])
                if 0 <= nr < rows and 0 <= nc < cols and rank[nr, nc] < rank[r, c]:
                    candidates.append(i)
            if candidates:
                direction_index[r, c] = rng.choice(candidates)
    return direction_index


def test_vectorized_accumulation_matches_naive_reference_random_grids():
    """The core proof of equivalence: several random (but guaranteed-
    acyclic) direction grids, comparing the vectorised wave-based
    implementation actually used in core.py against the naive,
    obviously-correct per-cell reference above."""
    for seed in range(8):
        direction_index = random_direction_index(10, 10, seed=seed)
        vectorized = accumulate_flow_cells(direction_index)
        naive = naive_accumulate_flow_cells(direction_index)
        assert np.allclose(
            vectorized, naive, equal_nan=True
        ), f"mismatch at seed={seed}"


def test_vectorized_accumulation_matches_naive_reference_with_nodata():
    direction_index = random_direction_index(8, 8, seed=42)
    direction_index[0, 0] = -2
    direction_index[3, 3] = -2
    # cells that pointed at now-NoData cells become pits instead, for a
    # valid test setup
    for r in range(8):
        for c in range(8):
            if direction_index[r, c] >= 0:
                nr = r + int(D_ROW[direction_index[r, c]])
                nc = c + int(D_COL[direction_index[r, c]])
                if 0 <= nr < 8 and 0 <= nc < 8 and direction_index[nr, nc] == -2:
                    direction_index[r, c] = -1

    vectorized = accumulate_flow_cells(direction_index)
    naive = naive_accumulate_flow_cells(direction_index)
    assert np.allclose(vectorized, naive, equal_nan=True)


# ----------------------------------------------------------------------
# Hand-verifiable accumulation cases
# ----------------------------------------------------------------------


def test_accumulation_simple_linear_chain():
    """A -> B -> C -> D (a straight line, each cell pointing E):
    accumulation must be exactly 1, 2, 3, 4."""
    direction_index = np.full((1, 4), 1, dtype=np.int8)  # all point E
    direction_index[0, 3] = -1  # last cell is the outlet (pit)
    result = accumulate_flow_cells(direction_index)
    assert list(result[0]) == [1.0, 2.0, 3.0, 4.0]


def test_accumulation_confluence_sums_both_branches():
    """Two headwater cells both draining into a third, which then
    drains onward - the confluence cell's accumulation must be the
    SUM of both upstream branches plus itself."""
    # 2 rows x 3 cols: (0,0) points SE into (1,1); (0,2) points SW into
    # (1,1); (1,1) points nowhere (pit)
    direction_index = np.full((2, 3), -1, dtype=np.int8)
    direction_index[0, 0] = 2  # SE
    direction_index[0, 2] = 4  # SW
    result = accumulate_flow_cells(direction_index)
    assert result[0, 0] == 1.0
    assert result[0, 2] == 1.0
    assert result[1, 1] == 3.0  # itself (1) + both upstream contributors (1+1)


def test_accumulation_isolated_pit_is_just_itself():
    direction_index = np.array([[-1]], dtype=np.int8)
    result = accumulate_flow_cells(direction_index)
    assert result[0, 0] == 1.0


def test_accumulation_nodata_cell_is_nan_and_excluded():
    direction_index = np.array([[1, -2], [-1, -1]], dtype=np.int8)
    result = accumulate_flow_cells(direction_index)
    assert np.isnan(result[0, 1])
    assert result[0, 0] == 1.0  # NoData neighbour doesn't count as a valid target


# ----------------------------------------------------------------------
# Inflow count
# ----------------------------------------------------------------------


def test_compute_inflow_count_confluence():
    direction_index = np.full((2, 3), -1, dtype=np.int8)
    direction_index[0, 0] = 2  # SE, points into (1,1)
    direction_index[0, 2] = 4  # SW, points into (1,1)
    inflow = compute_inflow_count(direction_index)
    assert inflow[1, 1] == 2
    assert inflow[0, 0] == 0
    assert inflow[0, 2] == 0


# ----------------------------------------------------------------------
# Pointer -> direction index mapping
# ----------------------------------------------------------------------


def test_pointer_to_direction_index_default_scheme():
    pointer = np.array([[1, 2, 4], [8, 16, 32], [64, 128, 0]], dtype=np.float64)
    idx = pointer_to_direction_index(pointer, pointer_nodata=NAN)
    assert idx[0, 0] == 0  # NE=1
    assert idx[0, 1] == 1  # E=2
    assert idx[0, 2] == 2  # SE=4
    assert idx[1, 0] == 3  # S=8
    assert idx[1, 1] == 4  # SW=16
    assert idx[1, 2] == 5  # W=32
    assert idx[2, 0] == 6  # NW=64
    assert idx[2, 1] == 7  # N=128
    assert idx[2, 2] == -1  # 0 -> pit


def test_pointer_to_direction_index_esri_scheme():
    pointer = np.array([[1, 64]], dtype=np.float64)  # ESRI: 1=E(idx1), 64=N(idx7)
    idx = pointer_to_direction_index(pointer, pointer_nodata=NAN, esri_style=True)
    assert idx[0, 0] == 1
    assert idx[0, 1] == 7


def test_pointer_to_direction_index_nodata():
    pointer = np.array([[OUTPUT_NODATA, 1]], dtype=np.float64)
    idx = pointer_to_direction_index(pointer, pointer_nodata=OUTPUT_NODATA)
    assert idx[0, 0] == -2
    assert idx[0, 1] == 0


# ----------------------------------------------------------------------
# Interior pit detection
# ----------------------------------------------------------------------


def test_detect_interior_pits_flags_genuine_interior_pit():
    # a 3x3 block of valid data, no NoData anywhere - the centre being
    # a pit (-1) is genuinely interior, not an edge artefact
    direction_index = np.full((3, 3), 1, dtype=np.int8)
    direction_index[1, 1] = -1
    assert detect_interior_pits(direction_index) == 1


def test_detect_interior_pits_ignores_pit_adjacent_to_nodata():
    direction_index = np.full((3, 3), 1, dtype=np.int8)
    direction_index[1, 1] = -1
    direction_index[1, 2] = -2  # NoData right next to the pit
    assert detect_interior_pits(direction_index) == 0


def test_detect_interior_pits_edge_pit_not_flagged():
    """A pit at the raster's own edge is adjacent to the padding
    (treated as NoData) - must not be flagged as a genuine interior
    problem."""
    direction_index = np.full((3, 3), 1, dtype=np.int8)
    direction_index[0, 0] = -1  # corner cell
    assert detect_interior_pits(direction_index) == 0


# ----------------------------------------------------------------------
# Output scaling - exact formulas confirmed from source
# ----------------------------------------------------------------------


def test_scale_accumulation_cells_no_area_scaling():
    accumulation = np.array([[1.0, 5.0]])
    direction_index = np.array([[1, -1]], dtype=np.int8)
    result = scale_accumulation(
        accumulation,
        direction_index,
        cell_size_x=10.0,
        cell_size_y=20.0,
        out_type="cells",
    )
    assert result[0, 0] == 1.0
    assert result[0, 1] == 5.0


def test_scale_accumulation_catchment_area_multiplies_by_cell_area():
    accumulation = np.array([[3.0]])
    direction_index = np.array([[-1]], dtype=np.int8)
    result = scale_accumulation(
        accumulation, direction_index, cell_size_x=10.0, cell_size_y=20.0, out_type="ca"
    )
    assert result[0, 0] == 3.0 * 10.0 * 20.0


def test_scale_accumulation_sca_divides_by_average_cell_size():
    accumulation = np.array([[3.0]])
    direction_index = np.array([[-1]], dtype=np.int8)
    result = scale_accumulation(
        accumulation,
        direction_index,
        cell_size_x=10.0,
        cell_size_y=20.0,
        out_type="sca",
    )
    expected = 3.0 * (10.0 * 20.0) / ((10.0 + 20.0) / 2.0)
    assert abs(result[0, 0] - expected) < 1e-6


def test_scale_accumulation_log_transform():
    accumulation = np.array([[np.e]])
    direction_index = np.array([[-1]], dtype=np.int8)
    result = scale_accumulation(
        accumulation,
        direction_index,
        cell_size_x=1.0,
        cell_size_y=1.0,
        out_type="cells",
        log_transform=True,
    )
    assert abs(result[0, 0] - 1.0) < 1e-5


def test_scale_accumulation_nodata_preserved():
    accumulation = np.array([[1.0, np.nan]])
    direction_index = np.array([[1, -2]], dtype=np.int8)
    result = scale_accumulation(
        accumulation,
        direction_index,
        cell_size_x=1.0,
        cell_size_y=1.0,
        out_type="cells",
    )
    assert result[0, 1] == np.float32(OUTPUT_NODATA)


def test_scale_accumulation_invalid_out_type_raises():
    accumulation = np.array([[1.0]])
    direction_index = np.array([[-1]], dtype=np.int8)
    with pytest.raises(ValueError):
        scale_accumulation(accumulation, direction_index, 1.0, 1.0, out_type="bogus")


# ----------------------------------------------------------------------
# D8 pointer computation (spot checks - full coverage already exists
# in this suite's own d8_flow_direction plugin's test suite; these
# confirm the duplicated copy here behaves identically, not a full
# re-verification of every rule)
# ----------------------------------------------------------------------


def test_compute_d8_pointer_simple_case():
    a = np.full((3, 3), 10.0)
    a[1, 1] = 10.0
    a[1, 2] = 5.0  # E neighbour, clearly lower
    pointer = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert pointer[1, 1] == 2  # E


def test_compute_d8_pointer_pit_is_zero():
    a = np.full((3, 3), 1.0)
    a[1, 1] = 1.0
    pointer = compute_d8_pointer(a, nodata_value=NAN, cell_size_x=1.0, cell_size_y=1.0)
    assert pointer[1, 1] == 0


# ----------------------------------------------------------------------
# Raster I/O and full orchestration
# ----------------------------------------------------------------------


def test_cell_sizes_from_geotransform_uses_absolute_values():
    geotransform = (0.0, 2.5, 0.0, 0.0, 0.0, -2.5)
    cx, cy = cell_sizes_from_geotransform(geotransform)
    assert cx == 2.5 and cy == 2.5


def test_full_pipeline_from_dem_end_to_end():
    pytest.importorskip("osgeo.gdal")

    # a simple tilted plane draining east - every interior cell should
    # accumulate strictly more than its upstream neighbour
    rows, cols = 5, 5
    dem = np.zeros((rows, cols))
    for c in range(cols):
        dem[:, c] = (cols - c) * 10.0

    tmp_dir = Path(tempfile.mkdtemp())
    in_path = str(tmp_dir / "dem.tif")
    out_path = str(tmp_dir / "accum.tif")
    geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    write_raster(in_path, dem.astype(np.float32), geotransform, "")

    info = compute_d8_flow_accumulation_from_file(in_path, out_path, out_type="cells")
    result, _, _, _ = read_raster(out_path)

    # the easternmost column (the outlet edge) should have the highest
    # accumulation in each row, since the whole plane drains that way
    assert np.all(result[:, -1] >= result[:, 0])
    assert isinstance(info["interior_pit_count"], int)


def test_full_pipeline_from_pointer_matches_from_dem():
    """Computing accumulation from a DEM directly, vs. computing the
    pointer first and feeding IT in with pntr_input=True, must give
    the same result - confirms the two code paths are consistent."""
    pytest.importorskip("osgeo.gdal")

    rows, cols = 4, 4
    dem = np.zeros((rows, cols))
    for c in range(cols):
        dem[:, c] = (cols - c) * 5.0

    tmp_dir = Path(tempfile.mkdtemp())
    dem_path = str(tmp_dir / "dem.tif")
    geotransform = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
    write_raster(dem_path, dem.astype(np.float32), geotransform, "")

    accum_from_dem_path = str(tmp_dir / "accum_from_dem.tif")
    compute_d8_flow_accumulation_from_file(
        dem_path, accum_from_dem_path, out_type="cells", pntr_input=False
    )

    pointer_path = str(tmp_dir / "pointer.tif")
    array, gt, proj, nodata = read_raster(dem_path)
    cx, cy = cell_sizes_from_geotransform(gt)
    pointer = compute_d8_pointer(array, nodata, cx, cy)
    write_raster(pointer_path, pointer.astype(np.float32), gt, proj)

    accum_from_pntr_path = str(tmp_dir / "accum_from_pntr.tif")
    compute_d8_flow_accumulation_from_file(
        pointer_path, accum_from_pntr_path, out_type="cells", pntr_input=True
    )

    result_dem, _, _, _ = read_raster(accum_from_dem_path)
    result_pntr, _, _, _ = read_raster(accum_from_pntr_path)
    assert np.allclose(result_dem, result_pntr, equal_nan=True)
