"""File writers for DEM Channel Bathymetry (no QGIS dependency): the station
table (CSV, UTF-8 with a byte-order mark and a column key) and the bed DEM
(GDAL, bundled with QGIS)."""

from __future__ import annotations

import csv

import numpy as np

from mayim_tools.hydrology._common.d8_network import GridError
from mayim_tools.hydrology.terrain_storage.export import CSV_ENCODING, write_key

from .core import BASIS_TEXT, CAP_TEXT, ChannelResult

STATION_KEY = {
    "line": "centreline number (1 = first feature or part)",
    "chainage_m": "distance downstream along the centreline, m",
    "x": "easting of the station, m",
    "y": "northing of the station, m",
    "wse_m": "water level at the station (DEM water surface, smoothed), m",
    "width_m": "water-surface width W, m",
    "slope_left": "left bank slope sL (looking downstream), m/m",
    "slope_right": "right bank slope sR, m/m",
    "depth_banks_m": "depth from the bank slopes before the caps, m",
    "depth_low_m": "depth below the water - low (shallow) bed, m",
    "depth_central_m": "depth below the water - central bed, m",
    "depth_high_m": "depth below the water - high (deep) bed, m",
    "bed_central_m": "lowest bed level - central bed, m",
    "w_d_ratio": "width-to-depth ratio of the central bed",
    "thalweg_left": "distance of the deepest point from the left edge / W",
    "bank_h_left_m": "left bank-top height above the water, m",
    "bank_h_right_m": "right bank-top height above the water, m",
    "basis": "how the depth was set",
    "cap": "cap that limited the depth, if any",
    "flags": "notes on the station",
}
STATION_FIELDS = list(STATION_KEY)


def station_rows(result: ChannelResult) -> list[dict]:
    rows = []
    for st in result.stations:
        d = st.depth or {}
        dc = d.get("central", np.nan)
        rows.append(
            {
                "line": st.line + 1,
                "chainage_m": st.chainage,
                "x": st.x,
                "y": st.y,
                "wse_m": st.wse,
                "width_m": st.width,
                "slope_left": st.sl,
                "slope_right": st.sr,
                "depth_banks_m": st.depth_banks,
                "depth_low_m": d.get("low", np.nan),
                "depth_central_m": dc,
                "depth_high_m": d.get("high", np.nan),
                "bed_central_m": st.wse - dc,
                "w_d_ratio": st.width / dc if dc and np.isfinite(dc) else np.nan,
                "thalweg_left": st.split if d else np.nan,
                "bank_h_left_m": st.height_left,
                "bank_h_right_m": st.height_right,
                "basis": BASIS_TEXT[st.basis],
                "cap": CAP_TEXT[st.cap],
                "flags": "; ".join(dict.fromkeys(st.flags)),
            }
        )
    return rows


def _fmt(value, digits=3):
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else f"{value:.{digits}f}"
    return value


def write_station_csv(path: str, result: ChannelResult) -> None:
    digits = {"chainage_m": 1, "x": 2, "y": 2, "w_d_ratio": 1, "thalweg_left": 2}
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        write_key(f, STATION_KEY, STATION_FIELDS)
        writer = csv.DictWriter(f, fieldnames=STATION_FIELDS)
        writer.writeheader()
        for row in station_rows(result):
            writer.writerow({k: _fmt(v, digits.get(k, 3)) for k, v in row.items()})


def write_bed_dem(src_path: str, out_path: str, window, bed: np.ndarray) -> None:
    """Copies the input DEM to a float32 GeoTIFF and writes the bed into the
    window's water cells; the rest of the DEM is unchanged, so the file can
    go straight into a HEC-RAS terrain."""
    from osgeo import gdal

    gdal.UseExceptions()
    src = gdal.Open(src_path)
    if src is None:
        raise GridError(f"Could not open raster: {src_path}")
    ds = gdal.Translate(
        out_path,
        src,
        format="GTiff",
        outputType=gdal.GDT_Float32,
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    src = None
    band = ds.GetRasterBand(1)
    r0, c0, nr, nc = window
    current = band.ReadAsArray(c0, r0, nc, nr).astype(np.float64)
    updated = np.where(np.isfinite(bed), bed, current)
    band.WriteArray(updated.astype(np.float32), c0, r0)
    band.FlushCache()
    ds = None
