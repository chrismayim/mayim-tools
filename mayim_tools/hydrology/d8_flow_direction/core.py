"""
Core D8 flow-direction (flow pointer) logic - a deliberate, exact
replication of WhiteboxTools' `D8Pointer` tool, not "a" D8 algorithm.

Every behavioural detail below was confirmed directly against
WhiteboxTools' own Rust source (d8_pointer.rs, jblindsay/whitebox-tools,
MIT licensed, algorithm by Dr. John Lindsay), not inferred from
documentation or general D8 knowledge - documentation alone doesn't
specify the exact numeric encoding, tie-breaking rule, or nodata/edge
handling precisely enough for a genuine replication.

DIRECTION ENCODING (confirmed exactly from source):

Default (WhiteboxTools' own "clockwise, base-2" scheme):
    | NW=64 | N=128 | NE=1 |
    | W=32  |  0    | E=2  |
    | SW=16 | S=8   | SE=4 |

ESRI-style (--esri_pntr flag in WhiteboxTools; esri_style=True here):
    | NW=32 | N=64  | NE=128 |
    | W=16  |  0    | E=1    |
    | SW=8  | S=4   | SE=2   |
This matches the standard ESRI ArcGIS flow direction convention
exactly - confirmed directly from WhiteboxTools' own source, not
assumed to match from general knowledge.

ALGORITHM (confirmed exactly from source, not just "steepest descent"):
- For each cell, evaluate its 8 neighbours in a FIXED order: NE, E, SE,
  S, SW, W, NW, N.
- Diagonal neighbours use distance sqrt(cell_size_x^2 + cell_size_y^2);
  cardinal (N/S/E/W) neighbours use the cell size in that axis. This
  matters for non-square pixels.
- slope = (centre_elevation - neighbour_elevation) / distance.
- The neighbour is only a candidate if its slope is POSITIVE (strictly
  lower than the centre) - equal or higher elevation neighbours are
  never candidates.
- Ties are broken by iteration order: the comparison is a STRICT `>`
  against the running maximum, so among neighbours with an identical
  (tied) maximum slope, the FIRST one encountered in the NE,E,SE,S,SW,
  W,NW,N order wins - not the last, and not an arbitrary one.
- If NO neighbour has a positive slope (a local pit, a flat area where
  this cell isn't the lowest, or an edge/corner cell whose off-grid
  neighbours don't exist), the output is 0 - not NoData, not an
  arbitrary direction. WhiteboxTools' own documentation: "grid cells
  that have no lower neighbours are assigned a flow direction of
  zero."
- A neighbour that is NoData (including an off-grid neighbour at an
  edge/corner, which WhiteboxTools' own Array2D treats as NoData on
  out-of-bounds access) is simply excluded from consideration, not
  treated as very low or very high.
- If the CENTRE cell itself is NoData, the output is NoData (not 0),
  taken from the DEM's own NoData value on read, but written as this
  tool's own fixed output NoData value (-32768, matching WhiteboxTools
  exactly) rather than whatever the input DEM happened to use.
- Output data type: signed 16-bit integer, matching WhiteboxTools
  exactly (D8Pointer's own output is i16).

REQUIRED PRECONDITION, stated as clearly as WhiteboxTools states it,
not softened: this tool does NOT fill depressions or resolve flat
areas itself. WhiteboxTools' own documentation is explicit that the
input DEM "must have been hydrologically corrected to remove all
spurious depressions and flat areas" first (their own BreachDepressions
or FillDepressions tools). Feeding this tool a raw, uncorrected DEM
will produce direction=0 at every unresolved pit and will not produce
hydrologically connected flow paths through them - this is D8Pointer's
own documented behaviour being faithfully reproduced, not a bug in
this replication.
"""

from __future__ import annotations

import numpy as np

# Direction order used throughout: NE, E, SE, S, SW, W, NW, N -
# confirmed exactly from WhiteboxTools' own d_x/d_y arrays.
D_ROW = np.array([-1, 0, 1, 1, 1, 0, -1, -1])
D_COL = np.array([1, 1, 1, 0, -1, -1, -1, 0])

OUT_VALS_DEFAULT = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int16)
OUT_VALS_ESRI = np.array([128, 1, 2, 4, 8, 16, 32, 64], dtype=np.int16)

OUTPUT_NODATA = np.int16(-32768)  # confirmed exactly from WhiteboxTools' own source

# Direction order used throughout: NE, E, SE, S, SW, W, NW, N - matches
# OUT_VALS_DEFAULT/OUT_VALS_ESRI's own index order above.
DIRECTION_NAMES = ["NE", "E", "SE", "S", "SW", "W", "NW", "N"]

# Default styling colours/labels for this tool's output, keyed by
# compass direction (not by raw numeric value, since the numeric code
# for a given direction differs between the default and ESRI pointer
# schemes - see direction_palette() below, which resolves this
# correctly for whichever scheme a given run actually used).
DIRECTION_COLORS = {
    "E": "#0076fd",
    "SE": "#04c900",
    "S": "#047300",
    "SW": "#fff500",
    "W": "#ffb200",
    "NW": "#ff0000",
    "N": "#4f0d5b",
    "NE": "#787878",
}
PIT_COLOR = "#3f3f3f"
PIT_LABEL = "Pit / flat (no lower neighbour)"


def direction_palette(esri_style: bool = False) -> list:
    """Returns the [(value, color_hex, label), ...] palette for this
    tool's output raster, for whichever pointer scheme a given run
    actually used (default WhiteboxTools scheme or ESRI scheme) - the
    numeric code for a given compass direction differs between the
    two schemes, so this must be resolved per-run, not applied as one
    fixed value->colour mapping regardless of esri_style. Includes an
    entry for value 0 (pit/flat/no lower neighbour), which is a real,
    commonly-occurring output value distinct from every compass
    direction and from NoData."""
    values = OUT_VALS_ESRI if esri_style else OUT_VALS_DEFAULT
    entries = [(0, PIT_COLOR, PIT_LABEL)]
    for name, value in zip(DIRECTION_NAMES, values, strict=True):
        entries.append((int(value), DIRECTION_COLORS[name], name))
    return entries


def compute_d8_pointer(
    z: np.ndarray,
    nodata_value: float,
    cell_size_x: float,
    cell_size_y: float,
    esri_style: bool = False,
) -> np.ndarray:
    """Computes the D8 flow-direction (pointer) raster from an
    elevation array, replicating WhiteboxTools' D8Pointer exactly -
    see module docstring for the confirmed behavioural details this
    implementation reproduces.

    z: 2D array of elevations (any numeric dtype; converted to
       float64 internally so nodata comparisons and slope arithmetic
       are exact regardless of the input DEM's own dtype).
    nodata_value: the input DEM's NoData value. NaN is handled
       correctly (compared via isnan, not ==, since NaN != NaN).
    cell_size_x, cell_size_y: pixel size in the same linear units as
       the DEM's elevation values' horizontal reference (e.g. metres
       for a projected DEM) - required separately, not assumed
       square, since WhiteboxTools itself does not assume this
       either (its own grid_lengths array uses cell_size_x for E/W
       and cell_size_y for N/S specifically).
    esri_style: if True, output uses the ESRI ArcGIS flow-direction
       numeric convention instead of WhiteboxTools' own default -
       see module docstring for both encodings, confirmed exactly
       from WhiteboxTools' own source.

    Returns an int16 array, same shape as z, using OUTPUT_NODATA
    (-32768) for cells that were NoData in the input, 0 for cells
    with no valid downslope neighbour, and the appropriate direction
    code otherwise.
    """
    z = np.asarray(z, dtype=np.float64)
    rows, cols = z.shape

    if np.isnan(nodata_value):
        is_nodata = np.isnan(z)
    else:
        is_nodata = z == nodata_value

    # Pad with a 1-cell NoData border so off-grid neighbours at edges/
    # corners are automatically excluded, exactly matching
    # WhiteboxTools' own Array2D out-of-bounds-returns-NoData behaviour
    # - not wrapped, not zero-filled, not an IndexError.
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
        # exact tie-break: strict '>' against the running max, checked
        # in a fixed i=0..7 order - the first direction to reach a
        # given maximum wins, matching WhiteboxTools' own sequential
        # loop precisely (not an arbitrary numpy argmax over all 8 at
        # once, which could pick a different winner on ties).
        candidate = (~neighbour_nodata) & (slope > max_slope) & (slope > 0)
        max_slope = np.where(candidate, slope, max_slope)
        dir_index = np.where(candidate, i, dir_index)
        found = found | candidate

    out_vals = OUT_VALS_ESRI if esri_style else OUT_VALS_DEFAULT
    direction = out_vals[
        dir_index
    ]  # fancy-indexes every cell's chosen direction's code
    direction = np.where(found, direction, np.int16(0))
    direction = np.where(is_nodata, OUTPUT_NODATA, direction)

    return direction.astype(np.int16)


def read_raster(path: str):
    """Reads a raster's first band as a float64 array, along with its
    geotransform, projection (WKT), and NoData value. Uses GDAL
    directly (not rasterio) since GDAL is already a QGIS dependency -
    no extra package needed inside QGIS's own Python environment."""
    from osgeo import gdal

    # GDAL 4.0 will make this the default; opting in now avoids both
    # the deprecation warning and a silent behaviour change later -
    # errors raise cleanly either way.
    gdal.UseExceptions()

    ds = gdal.Open(path)
    if ds is None:
        raise ValueError(f"Could not open raster: {path}")

    band = ds.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float64)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = (
            np.nan
        )  # no NoData defined on the source - treat everything as valid data

    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None  # closes the GDAL dataset handle

    return array, geotransform, projection, nodata_value


def write_raster(path: str, array: np.ndarray, geotransform, projection: str) -> None:
    """Writes an int16 D8 pointer raster as GeoTIFF, matching
    WhiteboxTools' own output type and NoData value exactly (see
    module docstring)."""
    from osgeo import gdal

    # See read_raster's docstring for why.
    gdal.UseExceptions()

    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(path, cols, rows, 1, gdal.GDT_Int16)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(float(OUTPUT_NODATA))
    out_band.FlushCache()
    out_ds = None  # closes and finalizes the GDAL dataset handle


def cell_sizes_from_geotransform(geotransform) -> tuple:
    """GDAL's geotransform stores pixel width as index 1 (always
    positive for a north-up raster) and pixel height as index 5
    (conventionally NEGATIVE for a north-up raster, since Y decreases
    as row increases) - returns both as positive linear cell sizes."""
    cell_size_x = abs(geotransform[1])
    cell_size_y = abs(geotransform[5])
    return cell_size_x, cell_size_y


def compute_d8_pointer_from_file(
    input_path: str, output_path: str, esri_style: bool = False
) -> None:
    """End-to-end: read a DEM, compute its D8 pointer raster exactly
    as WhiteboxTools' D8Pointer would, write the result. The thin
    orchestration layer QGIS calls into - all the actual algorithm
    logic lives in compute_d8_pointer(), which has zero file-I/O or
    QGIS dependency and is independently unit-tested."""
    z, geotransform, projection, nodata_value = read_raster(input_path)
    cell_size_x, cell_size_y = cell_sizes_from_geotransform(geotransform)

    direction = compute_d8_pointer(
        z, nodata_value, cell_size_x, cell_size_y, esri_style=esri_style
    )

    write_raster(output_path, direction, geotransform, projection)
