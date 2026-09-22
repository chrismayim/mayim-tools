"""
Core D8 flow-accumulation logic - a deliberate, exact replication of
WhiteboxTools' D8FlowAccumulation tool, not "a" flow accumulation
algorithm. Every behavioural detail below matches WhiteboxTools' own
Rust source (d8_flow_accum.rs, jblindsay/whitebox-tools, MIT licensed,
algorithm by Dr. John Lindsay) - not inferred from documentation.

INPUT: either a DEM (this tool computes its own D8 pointer internally,
using the exact same algorithm as this suite's own d8_flow_direction
plugin - duplicated here rather than imported, matching this suite's
convention of every plugin being self-contained) or a pre-computed D8
pointer raster (e.g. from d8_flow_direction's own output), with an
esri_style flag for the alternative ESRI pointer convention - matching
d8_flow_direction's own confirmed encodings exactly:

    Default: NE=1, E=2, SE=4, S=8, SW=16, W=32, NW=64, N=128
    ESRI:    E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128

A pointer value of 0 (this suite's own "no lower neighbour" code) maps
to a pit/outlet cell with no downstream propagation.

ALGORITHM (matching WhiteboxTools exactly):
- Every valid cell starts with an accumulation of exactly 1.0 (itself),
  not 0 - a headwater cell with no upstream contributors has an
  accumulation of 1, not 0 (WhiteboxTools:
  output.reinitialize_values(1.0)).
- For each cell, count how many of its 8 neighbours have a flow
  direction pointing BACK at it (in-degree/inflow count).
- Process cells in topological order (each cell's final accumulation
  depends on every upstream cell being fully resolved first):
  repeatedly identify all cells with zero REMAINING inflow, add each
  one's own accumulated value to whatever cell it points downstream
  to, decrement that downstream cell's remaining inflow count, and
  repeat until no cells remain. This implementation processes cells in
  vectorised BATCHES by topological level ("wave") using np.add.at
  (which correctly handles multiple same-level cells draining to the
  same downstream cell), rather than WhiteboxTools' own
  one-cell-at-a-time stack loop - a genuine algorithmic difference,
  but NOT a behavioural one: cells at the same topological level are,
  by construction, independent of each other (neither is upstream of
  the other), so the order they're processed in cannot affect the
  final accumulated values. This is verified directly by test (see
  tests/test_d8_flow_accumulation_core.py), not just argued.
- A cell with NO downstream neighbour (pointer value 0, i.e. a pit or
  outlet) does not propagate its accumulation anywhere - correctly
  representing a terminal point, not an error.
- A cell that is NoData in the input propagates NoData in the output.

OUTPUT TYPES (three options, matching WhiteboxTools exactly):
- "cells" (WhiteboxTools' own default): raw accumulated cell count, no
  area scaling at all (cell_area=1, flow_width=1 for every direction).
- "ca" (catchment area): accumulated cell count x true cell area
  (cell_size_x x cell_size_y) - a real area, e.g. m^2, with NO flow-
  width division.
- "sca" (specific contributing area): catchment area DIVIDED by a
  constant flow width - constant across all 8 directions (the AVERAGE
  of cell_size_x and cell_size_y), deliberately NOT direction-
  dependent. WhiteboxTools' own source comment explains why: "if flow
  width is allowed to vary by direction, the flow accumulation output
  will not increase continuously downstream and any applications
  involving stream network extraction will encounter issues with
  discontinuous streams." This is a deliberate reversion in
  WhiteboxTools' own history (an earlier, direction-dependent flow
  width was tried and abandoned) - replicated exactly here, not
  "improved" on, since doing so would silently break monotonic
  downstream accumulation.

Output data type is always float (float32 here), regardless of the
input pointer's own dtype - matching WhiteboxTools' own current
behaviour, which deliberately avoids a real, previously-reported
WhiteboxTools bug where an integer pointer's dtype was inherited by
the accumulation output, silently truncating any catchment large
enough to overflow that integer type.

Log-transform (log_transform=True) applies the natural logarithm AFTER
area/width scaling, matching source exactly - and, matching
WhiteboxTools' own documentation, a log-transformed output must not be
used to compute secondary terrain indices (e.g. wetness index).

Interior pit detection: a cell with no downslope neighbour that is NOT
adjacent to any NoData cell (i.e. genuinely interior to the valid data
region, not just a natural DEM-edge artefact) is flagged - matching
WhiteboxTools' own warning that such cells likely indicate the DEM
needs further hydrological conditioning (depression filling/breaching)
before this tool's output can be trusted.

REQUIRED PRECONDITION, same as this suite's own d8_flow_direction: if
a raw DEM is supplied (rather than a pre-computed pointer), it must
already be hydrologically corrected - this tool does not fill
depressions or resolve flats itself.
"""

from __future__ import annotations

import numpy as np

# Direction order: NE, E, SE, S, SW, W, NW, N - identical to
# d8_flow_direction's own confirmed encoding.
D_ROW = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
D_COL = np.array([1, 1, 1, 0, -1, -1, -1, 0])

POINTER_VALS_DEFAULT = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int64)
POINTER_VALS_ESRI = np.array([128, 1, 2, 4, 8, 16, 32, 64], dtype=np.int64)

# Matches WhiteboxTools' own source exactly - output is always float,
# not int, unlike d8_flow_direction's own int16 pointer output.
OUTPUT_NODATA = -32768.0

VALID_OUT_TYPES = ("cells", "ca", "sca")


# ---------------------------------------------------------------------------
# D8 pointer computation from a raw DEM - duplicated from this suite's
# own d8_flow_direction plugin (not imported, matching this suite's
# convention of self-contained plugins), identical algorithm.
# ---------------------------------------------------------------------------


def compute_d8_pointer(
    z: np.ndarray,
    nodata_value: float,
    cell_size_x: float,
    cell_size_y: float,
    esri_style: bool = False,
) -> np.ndarray:
    """Computes the D8 flow-direction (pointer) raster from an
    elevation array - identical algorithm to this suite's own
    d8_flow_direction plugin. See that plugin's core.py for the full,
    separately-confirmed behavioural documentation (tie-breaking rule,
    edge handling, etc.) - every detail is the same, since both were
    confirmed against the same WhiteboxTools source."""
    z = np.asarray(z, dtype=np.float64)
    rows, cols = z.shape

    if np.isnan(nodata_value):
        is_nodata = np.isnan(z)
    else:
        is_nodata = z == nodata_value

    z_pad = np.pad(z, 1, mode="constant", constant_values=np.nan)
    nodata_pad = np.pad(is_nodata, 1, mode="constant", constant_values=True)

    diag = float(np.hypot(cell_size_x, cell_size_y))
    distances = [
        diag,
        cell_size_x,
        diag,
        cell_size_y,
        diag,
        cell_size_x,
        diag,
        cell_size_y,
    ]

    max_slope = np.full((rows, cols), -np.inf, dtype=np.float64)
    found = np.zeros((rows, cols), dtype=bool)
    dir_index = np.zeros((rows, cols), dtype=np.int8)

    for i in range(8):
        dr, dc = int(D_ROW[i]), int(D_COL[i])
        neighbour_z = z_pad[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        neighbour_nodata = nodata_pad[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        slope = (z - neighbour_z) / distances[i]
        candidate = (~neighbour_nodata) & (slope > max_slope) & (slope > 0)
        max_slope = np.where(candidate, slope, max_slope)
        dir_index = np.where(candidate, i, dir_index)
        found = found | candidate

    pointer_vals = POINTER_VALS_ESRI if esri_style else POINTER_VALS_DEFAULT
    pointer = pointer_vals[dir_index]
    pointer = np.where(found, pointer, 0)
    pointer = np.where(is_nodata, OUTPUT_NODATA, pointer)
    return pointer.astype(np.int32)


# ---------------------------------------------------------------------------
# Pointer -> internal direction index (0-7, or -1 pit, or -2 NoData)
# ---------------------------------------------------------------------------


def pointer_to_direction_index(
    pointer: np.ndarray, pointer_nodata: float, esri_style: bool = False
) -> np.ndarray:
    """Maps raw pointer values (1/2/4/8/16/32/64/128, or the ESRI
    equivalents) to an internal 0-7 direction index matching D_ROW/
    D_COL's own order, -1 for a pit/no-flow cell (pointer value 0), or
    -2 for NoData."""
    pointer_vals = POINTER_VALS_ESRI if esri_style else POINTER_VALS_DEFAULT
    value_to_index = {int(v): i for i, v in enumerate(pointer_vals)}

    if np.isnan(pointer_nodata):
        is_nodata = np.isnan(pointer)
    else:
        is_nodata = pointer == pointer_nodata

    direction_index = np.full(pointer.shape, -2, dtype=np.int8)
    for val, idx in value_to_index.items():
        direction_index = np.where(
            (pointer == val) & (~is_nodata), idx, direction_index
        )

    is_pit = (pointer == 0) & (~is_nodata)
    direction_index = np.where(is_pit, -1, direction_index)
    direction_index = np.where(is_nodata, -2, direction_index)
    return direction_index


# ---------------------------------------------------------------------------
# Inflow count (in-degree) and topological-level flow accumulation
# ---------------------------------------------------------------------------


def compute_inflow_count(direction_index: np.ndarray) -> np.ndarray:
    """For each valid cell, how many of its 8 neighbours have a flow
    direction pointing back at it - the opposite direction index,
    (i + 4) % 8, for each of the 8 neighbour offsets."""
    rows, cols = direction_index.shape
    dir_pad = np.pad(direction_index, 1, mode="constant", constant_values=-2)
    inflow_count = np.zeros((rows, cols), dtype=np.int32)

    for i in range(8):
        dr, dc = int(D_ROW[i]), int(D_COL[i])
        neighbour_dir = dir_pad[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        opposite = (i + 4) % 8
        inflow_count += (neighbour_dir == opposite).astype(np.int32)

    return inflow_count


def accumulate_flow_cells(direction_index: np.ndarray) -> np.ndarray:
    """Topological-level (wave) flow accumulation in raw CELL COUNT
    units (every valid cell starts at 1.0 - see module docstring for
    why). Cells at the same topological level are processed together
    via vectorised NumPy operations (np.add.at, which correctly
    handles multiple same-level cells draining to the same downstream
    cell) rather than WhiteboxTools' own one-cell-at-a-time stack -
    see module docstring for why this produces identical results, not
    just faster ones."""
    rows, cols = direction_index.shape
    is_nodata = direction_index == -2
    accumulation = np.where(is_nodata, np.nan, 1.0).astype(np.float64)
    remaining_inflow = compute_inflow_count(direction_index).astype(np.int64)
    remaining_inflow = np.where(is_nodata, -1, remaining_inflow)
    processed = is_nodata.copy()

    row_idx, col_idx = np.indices((rows, cols))
    ready_mask = (~processed) & (remaining_inflow == 0)

    while np.any(ready_mask):
        r_ready = row_idx[ready_mask]
        c_ready = col_idx[ready_mask]
        dir_ready = direction_index[ready_mask]
        acc_ready = accumulation[ready_mask]

        has_downstream = dir_ready >= 0
        if np.any(has_downstream):
            dr = D_ROW[dir_ready[has_downstream]]
            dc = D_COL[dir_ready[has_downstream]]
            r_down = r_ready[has_downstream] + dr
            c_down = c_ready[has_downstream] + dc
            acc_down = acc_ready[has_downstream]

            valid_target = (
                (r_down >= 0) & (r_down < rows) & (c_down >= 0) & (c_down < cols)
            )
            np.add.at(
                accumulation,
                (r_down[valid_target], c_down[valid_target]),
                acc_down[valid_target],
            )
            np.subtract.at(
                remaining_inflow, (r_down[valid_target], c_down[valid_target]), 1
            )

        processed[ready_mask] = True
        ready_mask = (~processed) & (remaining_inflow == 0)

    return accumulation


def detect_interior_pits(direction_index: np.ndarray) -> int:
    """Counts pit cells (no downslope neighbour) that are NOT adjacent
    to any NoData cell - i.e. genuinely interior to the valid data
    region, not a natural DEM-edge artefact. Matches WhiteboxTools'
    own warning that such cells likely indicate the DEM needs further
    hydrological conditioning before this tool's output can be
    trusted."""
    rows, cols = direction_index.shape
    is_nodata = direction_index == -2
    is_pit = direction_index == -1
    nodata_pad = np.pad(is_nodata, 1, mode="constant", constant_values=True)

    adjacent_to_nodata = np.zeros((rows, cols), dtype=bool)
    for i in range(8):
        dr, dc = int(D_ROW[i]), int(D_COL[i])
        neighbour_nodata = nodata_pad[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        adjacent_to_nodata |= neighbour_nodata

    interior_pit = is_pit & (~adjacent_to_nodata)
    return int(np.sum(interior_pit))


# ---------------------------------------------------------------------------
# Output scaling (cells / catchment area / specific contributing area)
# ---------------------------------------------------------------------------


def scale_accumulation(
    accumulation_cells: np.ndarray,
    direction_index: np.ndarray,
    cell_size_x: float,
    cell_size_y: float,
    out_type: str = "cells",
    log_transform: bool = False,
) -> np.ndarray:
    """Scales raw cell-count accumulation into the requested output
    type - formulas matching WhiteboxTools' source exactly (see
    module docstring)."""
    if out_type not in VALID_OUT_TYPES:
        raise ValueError(f"out_type must be one of {VALID_OUT_TYPES}, got {out_type!r}")

    if out_type == "cells":
        cell_area = 1.0
        flow_width = 1.0
    elif out_type == "ca":
        cell_area = cell_size_x * cell_size_y
        flow_width = 1.0
    else:  # sca
        cell_area = cell_size_x * cell_size_y
        flow_width = (cell_size_x + cell_size_y) / 2.0

    is_nodata = direction_index == -2
    result = accumulation_cells * cell_area / flow_width

    if log_transform:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.log(result)

    result = np.where(is_nodata, OUTPUT_NODATA, result)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Raster I/O - identical approach to this suite's own d8_flow_direction
# ---------------------------------------------------------------------------


def read_raster(path: str):
    """Reads a raster's first band as a float64 array, along with its
    geotransform, projection (WKT), and NoData value. Identical
    approach to this suite's own d8_flow_direction plugin - GDAL is
    already a QGIS dependency."""
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    if ds is None:
        raise ValueError(f"Could not open raster: {path}")

    band = ds.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float64)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = np.nan

    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None
    return array, geotransform, projection, nodata_value


def write_raster(path: str, array: np.ndarray, geotransform, projection: str) -> None:
    """Writes a float32 flow-accumulation raster as GeoTIFF, matching
    WhiteboxTools' own output type and NoData value exactly."""
    from osgeo import gdal

    gdal.UseExceptions()
    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(path, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(float(OUTPUT_NODATA))
    out_band.FlushCache()
    out_ds = None


def cell_sizes_from_geotransform(geotransform) -> tuple:
    """GDAL's geotransform stores pixel height as NEGATIVE for a
    north-up raster - returns both as positive linear cell sizes."""
    cell_size_x = abs(geotransform[1])
    cell_size_y = abs(geotransform[5])
    return cell_size_x, cell_size_y


# ---------------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------------


def compute_d8_flow_accumulation_from_file(
    input_path: str,
    output_path: str,
    out_type: str = "cells",
    log_transform: bool = False,
    pntr_input: bool = False,
    esri_style: bool = False,
) -> dict:
    """Reads a DEM or D8 pointer raster, computes D8 flow accumulation
    exactly as WhiteboxTools' D8FlowAccumulation would, writes the
    result. Returns a small info dict (interior pit count) so a
    caller (e.g. the QGIS algorithm) can surface WhiteboxTools' own
    diagnostic warning.

    All the actual algorithm logic lives in the other functions in
    this module, which have zero file-I/O or QGIS dependency and are
    independently unit-tested - this is the thin orchestration layer.
    """
    array, geotransform, projection, nodata_value = read_raster(input_path)
    cell_size_x, cell_size_y = cell_sizes_from_geotransform(geotransform)

    if pntr_input:
        direction_index = pointer_to_direction_index(
            array, nodata_value, esri_style=esri_style
        )
    else:
        pointer = compute_d8_pointer(
            array, nodata_value, cell_size_x, cell_size_y, esri_style=esri_style
        )
        direction_index = pointer_to_direction_index(
            pointer, float(OUTPUT_NODATA), esri_style=esri_style
        )

    n_interior_pits = detect_interior_pits(direction_index)
    accumulation_cells = accumulate_flow_cells(direction_index)
    result = scale_accumulation(
        accumulation_cells,
        direction_index,
        cell_size_x,
        cell_size_y,
        out_type=out_type,
        log_transform=log_transform,
    )
    write_raster(output_path, result, geotransform, projection)
    return {"interior_pit_count": n_interior_pits}
