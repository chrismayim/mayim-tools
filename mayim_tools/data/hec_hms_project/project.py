"""
Writes the HEC-HMS project folder - zero QGIS dependency (GDAL is used for
the terrain, imported lazily).
"""

from __future__ import annotations

from pathlib import Path

from . import hms_files as hf
from .core import HmsProjectError, ProjectResult, folder_stem, project_stem
from .dss import write_gages

HMS_DIR_TEMPLATE = r"C:\Program Files\HEC\HEC-HMS\{version}"
DEFAULT_HMS_DIR = HMS_DIR_TEMPLATE.format(version="4.14")


def write_terrain(
    dem_path: str,
    out_path: str | Path,
    dst_wkt: str,
    bounds: tuple[float, float, float, float],
    buffer_m: float = 500.0,
) -> str:
    """DEM clipped to the basin (plus a buffer) and warped to the basin CRS,
    as a tiled, compressed float GeoTIFF - the elevation file of a HEC-HMS
    terrain data set."""
    from osgeo import gdal

    gdal.UseExceptions()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bounds
    opts = gdal.WarpOptions(
        dstSRS=dst_wkt,
        outputBounds=(
            xmin - buffer_m,
            ymin - buffer_m,
            xmax + buffer_m,
            ymax + buffer_m,
        ),
        resampleAlg="bilinear",
        outputType=gdal.GDT_Float32,
        dstNodata=-9999.0,
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
        format="GTiff",
    )
    ds = gdal.Warp(str(out_path), dem_path, options=opts)
    if ds is None:
        raise HmsProjectError("The DEM could not be written to the HEC-HMS terrain.")
    ds = None
    return str(out_path)


def write_cn_grid(
    cn_path: str,
    band: int,
    out_path: str | Path,
    dst_wkt: str,
    bounds: tuple[float, float, float, float],
    buffer_m: float = 500.0,
) -> str:
    """One band of the curve number raster, clipped to the basin (plus a
    buffer) and warped to its CRS (nearest neighbour), as a GeoTIFF for the
    HEC-HMS grid data (Curve Number grid)."""
    from osgeo import gdal

    gdal.UseExceptions()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bounds
    src = gdal.Translate("", cn_path, format="VRT", bandList=[band])
    ds = gdal.Warp(
        str(out_path),
        src,
        options=gdal.WarpOptions(
            dstSRS=dst_wkt,
            outputBounds=(
                xmin - buffer_m,
                ymin - buffer_m,
                xmax + buffer_m,
                ymax + buffer_m,
            ),
            resampleAlg="near",
            outputType=gdal.GDT_Float32,
            dstNodata=-9999.0,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE"],
            format="GTiff",
        ),
    )
    if ds is None:
        raise HmsProjectError("The CN raster could not be written to the HEC-HMS grid.")
    ds = None
    src = None
    return str(out_path)


SUB_FIELDS = [
    ("name", "str"),
    ("area_km2", "float"),
    ("cn", "float"),
    ("ia_mm", "float"),
    ("imperv_pc", "float"),
    ("tc_min", "float"),
    ("lag_min", "float"),
    ("ds", "str"),
]
REACH_FIELDS = [
    ("name", "str"),
    ("length_m", "float"),
    ("route", "str"),
    ("k_h", "float"),
    ("x", "float"),
    ("steps", "int"),
    ("lag_min", "float"),
    ("ds", "str"),
]


def write_map_layers(
    r: ProjectResult, maps_dir: str | Path, crs_wkt: str, stem: str
) -> list[tuple[str, str]]:
    """Sub-basin polygons and reach lines as shapefiles in maps/ for the
    HEC-HMS basin map, with the model parameters as attributes. Returns
    [(display name, path relative to the project folder)]."""
    from osgeo import ogr, osr

    ogr.UseExceptions()
    maps_dir = Path(maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    srs = osr.SpatialReference()
    if crs_wkt:
        srs.ImportFromWkt(crs_wkt)
    drv = ogr.GetDriverByName("ESRI Shapefile")
    types = {"str": ogr.OFTString, "float": ogr.OFTReal, "int": ogr.OFTInteger}
    out = []
    for kind, label, fields, gtype in (
        ("Subbasin", "Subbasins", SUB_FIELDS, ogr.wkbPolygon),
        ("Reach", "Reaches", REACH_FIELDS, ogr.wkbLineString),
    ):
        elements = [e for e in r.network.of(kind) if e.wkb]
        if not elements:
            continue
        path = maps_dir / f"{stem}_{label}.shp"
        if path.exists():
            drv.DeleteDataSource(str(path))
        ds = drv.CreateDataSource(str(path))
        lyr = ds.CreateLayer(path.stem, srs if crs_wkt else None, gtype)
        for name, t in fields:
            fd = ogr.FieldDefn(name, types[t])
            if t == "str":
                fd.SetWidth(80)
            lyr.CreateField(fd)
        for e in elements:
            p = e.params
            values = {
                "name": e.name,
                "ds": e.downstream or "",
                "imperv_pc": p.get("imperv_pct"),
                "route": p.get("method") or "",
            }
            f = ogr.Feature(lyr.GetLayerDefn())
            for name, t in fields:
                v = values.get(name, p.get(name))
                if v is None:
                    continue
                f.SetField(name, int(v) if t == "int" else v)
            f.SetGeometry(ogr.CreateGeometryFromWkb(bytes(e.wkb)))
            lyr.CreateFeature(f)
            f = None
        ds = None
        out.append((label, f"maps/{path.name}"))
    return out


def write_project(
    r: ProjectResult,
    parent_folder: str | Path,
    folder_name: str,
    crs_wkt: str = "",
    srid: int = 0,
    srtext: str = "",
    dem_path: str | None = None,
    terrain_name: str = "Terrain",
    bounds: tuple[float, float, float, float] | None = None,
    hms_dir: str = DEFAULT_HMS_DIR,
    dss_writer=write_gages,
    terrain_writer=write_terrain,
    log=print,
    cn_raster: str | None = None,
    cn_band: int = 1,
    srs_auth: str = "EPSG",
    map_writer=write_map_layers,
    grid_writer=write_cn_grid,
) -> dict:
    """Writes every project file into ``parent_folder/folder_name``; returns
    {label: path}. ``bounds`` (xmin, ymin, xmax, ymax) is the basin extent
    (the sub-catchment polygons), else the element positions. Existing
    component files of the same names are
    replaced; other files in the folder are left alone. ``cn_raster`` (band
    ``cn_band``) is added to the HEC-HMS grid data as the Curve Number grid;
    the sub-basin and reach geometry is written as basin-map shapefiles."""
    folder = Path(parent_folder) / folder_stem(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    # HEC-HMS finds the .run, .gage, .terrain ... files from the project name,
    # after turning every character other than A-Z, a-z, 0-9 into '_'; the
    # project name and the .hms stem are therefore written that way already.
    stem = project_stem(folder_name)
    r.settings.project_name = stem
    n = hf.ProjectNames(r, stem)
    files: dict[str, str] = {}

    if bounds is None:
        xs = [e.x for e in r.network.elements]
        ys = [e.y for e in r.network.elements]
        bounds = (min(xs), min(ys), max(xs), max(ys))

    terrain = None
    if dem_path:
        rel = hf.terrain_relpath(terrain_name)
        try:
            terrain_writer(dem_path, folder / rel, crs_wkt, bounds)
            hf.write_text(
                folder / f"{stem}.terrain",
                hf.terrain_text(n, terrain_name, "DEM prepared with Mayim Tools"),
            )
            files["Terrain elevation"] = str(folder / rel)
            terrain = terrain_name
        except Exception as e:  # noqa: BLE001 - the model must still be written
            r.warnings.append(f"Terrain not written: {e}")
            log(f"Terrain not written: {e}")

    layers: list[tuple[str, str]] = []
    try:
        layers = map_writer(r, folder / "maps", crs_wkt, stem) or []
        for label, rel in layers:
            files[f"Map layer {label.lower()}"] = str(folder / rel)
    except Exception as e:  # noqa: BLE001 - the model must still be written
        r.warnings.append(f"Basin map layers not written: {e}")
        log(f"Basin map layers not written: {e}")

    if cn_raster:
        rel = "maps/curve_numbers.tif"
        try:
            grid_writer(cn_raster, cn_band, folder / rel, crs_wkt, bounds)
            files["Curve number grid"] = str(folder / rel)
            files["Grid data"] = hf.write_text(
                folder / f"{stem}.grid",
                hf.grid_text(
                    n,
                    [
                        (
                            hf.CN_GRID_NAME,
                            "Curve Number",
                            rel,
                            "SCS curve numbers from Mayim Tools (for Gridded SCS loss)",
                        )
                    ],
                ),
            )
        except Exception as e:  # noqa: BLE001
            r.warnings.append(f"Curve number grid not written: {e}")
            log(f"Curve number grid not written: {e}")

    files["Project"] = hf.write_text(folder / f"{stem}.hms", hf.project_text(r, n))
    files["Basin model"] = hf.write_text(
        folder / n.basin_file,
        hf.basin_text(r, n, crs_wkt, terrain, bounds, map_layers=layers),
    )
    if srid:
        files["Basin geometry"] = hf.write_sqlite(
            folder / n.sqlite_file, r, srid, srtext, auth=srs_auth
        )
    for run in r.runs:
        hf.write_text(folder / n.met_files[run.met_name], hf.met_text(r, run))
    for c in r.controls:
        hf.write_text(
            folder / n.control_files[c.name], hf.control_text(c, r.settings.hms_version)
        )
    files["Simulation runs"] = hf.write_text(folder / f"{stem}.run", hf.run_text(r))
    ens_dir = folder / "ensemble"
    for ens in r.ensembles:
        hf.write_text(
            ens_dir / hf.hms_component_file(ens.name, ".ens"),
            hf.ensemble_text(ens, r),
        )
    if r.ensembles:
        files["Ensemble analyses"] = f"{ens_dir} ({len(r.ensembles)})"

    hyeto = [run for run in r.runs if run.gage_name]
    if hyeto:
        series = []
        for run in hyeto:
            first, dt, vals = hf.gage_series(run)
            series.append(
                (
                    hf.gage_pathname(run.gage_name, run.control.start, dt),
                    first,
                    dt,
                    vals,
                )
            )
        dss_path = folder / n.rain_dss
        dss_writer(dss_path, series)
        files["Design storm gages (DSS)"] = str(dss_path)
        files["Precipitation gages"] = hf.write_text(
            folder / f"{stem}.gage", hf.gage_text(r, n)
        )
    script = folder / "run_all_simulations.py"
    hf.write_text(script, hf.run_all_script(r, n, str(folder)))
    hf.write_text(folder / "run_all_simulations.cmd", hf.run_all_cmd(hms_dir))
    files["Run-all script"] = str(folder / "run_all_simulations.cmd")
    r.folder = str(folder)
    r.files = files
    return files
