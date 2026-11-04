"""Processing algorithm: Import LandXML Surface (Data Tools).

Reads a LandXML file (e.g. a Civil 3D export) and writes its surfaces as
TIN meshes (2DM, loaded as QGIS mesh layers) and DEM rasters (GeoTIFF,
linear interpolation on the TIN), with optional breaklines, TIN points
and contours, plus the other survey elements found in the file - survey
points (with their point groups), alignments, feature lines, parcels and
pipe networks - as shapefiles, and a Word report. Thin QGIS wrapper: the
reading is in core.py, the TIN outputs in tin.py.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import numpy as np
from osgeo import gdal
from qgis.core import (
    Qgis,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterCrs,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    RASTER,
    SHAPEFILE,
    WORD,
    clean_name_part,
    output_file_name,
)
from mayim_tools.hydrology.catchment_delineation.qgis_outputs import save_shapefile
from mayim_tools.hydrology.dem_hydro_processing.contours_export import (
    MAJOR_FACTOR,
    write_contours,
)
from mayim_tools.hydrology.dem_hydro_processing.qgis_map import copy_source_layer

from . import qgis_layers as ql
from .core import read_landxml
from .report import write_report
from .tin import NODATA, outline_rings, tin_to_grid, write_2dm

TOOL_VERSION = "1.1.0"

# Per surface: key, dialog label, file description after the surface name,
# folder kind, extension, ticked by default.
SURFACE_OUTPUTS = [
    ("OUTPUT_TIN", "Surface TIN (mesh, 2DM)", "TIN", RASTER, ".2dm", True),
    ("OUTPUT_DEM", "Surface DEM (GeoTIFF)", "DEM", RASTER, ".tif", True),
    (
        "OUTPUT_BREAKLINES",
        "Surface breaklines and boundaries",
        "Breaklines",
        SHAPEFILE,
        ".shp",
        False,
    ),
    ("OUTPUT_TIN_POINTS", "Surface TIN points", "TIN_Points", SHAPEFILE, ".shp", False),
    ("OUTPUT_CONTOURS", "Surface contours", "Contours", SHAPEFILE, ".shp", False),
]
# Whole-file elements (written only when the file has them).
ELEMENT_OUTPUTS = [
    ("OUTPUT_POINTS", "Survey points (with point groups)", "Survey Points"),
    ("OUTPUT_ALIGNMENTS", "Alignments", "Alignments"),
    ("OUTPUT_FEATURE_LINES", "Feature lines", "Feature Lines"),
    ("OUTPUT_PARCELS", "Parcels (sites)", "Parcels"),
    ("OUTPUT_PIPES", "Pipe networks (structures and pipes)", "Pipe Network"),
]
OUTPUT_REPORT = ("OUTPUT_DOCX", "Import report (Word)", "LandXML Import Report")
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for TIN meshes and DEM rasters"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}
LABELS = {k: lbl for k, lbl, *_ in SURFACE_OUTPUTS}
LABELS.update({k: lbl for k, lbl, _d in ELEMENT_OUTPUTS})
LABELS[OUTPUT_REPORT[0]] = OUTPUT_REPORT[1]


def _software_version() -> str:
    parts = []
    metadata = Path(__file__).resolve().parents[2] / "metadata.txt"
    try:
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.startswith("version="):
                parts.append(f"Mayim Tools {line.split('=', 1)[1].strip()}")
                break
    except OSError:
        pass
    parts.append(f"QGIS {Qgis.version()}")
    return ", ".join(parts)


def _path(folder, project_number, description, ext) -> str:
    name = output_file_name(project_number, description, SHAPEFILE)
    return str(Path(folder) / (Path(name).stem + ext))


def _remove(path) -> None:
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError as e:
            raise RuntimeError(
                f"{Path(path).name} exists and could not be replaced - it is "
                "probably open in QGIS. Remove it from the project and run again."
            ) from e


def _write_tif(path, array, gt, wkt) -> None:
    _remove(path)
    ds = gdal.GetDriverByName("GTiff").Create(
        path,
        array.shape[1],
        array.shape[0],
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    if ds is None:
        raise RuntimeError(f"{Path(path).name} could not be created.")
    ds.SetGeoTransform(gt)
    if wkt:
        ds.SetProjection(wkt)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(NODATA)
    band.WriteArray(array)
    band.FlushCache()
    ds = None


class LandXmlImportAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    SURFACES = "SURFACES"
    CRS = "CRS"
    CELL_SIZE = "CELL_SIZE"
    CONTOUR_INTERVAL = "CONTOUR_INTERVAL"
    SITE_BOUNDARY = "SITE_BOUNDARY"
    REPORT_OSM = "REPORT_OSM"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"

    def __init__(self):
        super().__init__()
        self._keep = []

    def createInstance(self):
        return LandXmlImportAlgorithm()

    def name(self):
        return "import_landxml_surface"

    def displayName(self):
        return "Import LandXML Surface"

    def group(self):
        return "Data Tools"

    def groupId(self):
        return "data_tools"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def shortHelpString(self):
        surf = "\n".join(
            f"  {lbl}: <project number>_<surface>_{d}{ext}"
            for _k, lbl, d, _kind, ext, _dflt in SURFACE_OUTPUTS
        )
        elems = "\n".join(
            f"  {lbl}: <project number>_{clean_name_part(d)}.shp"
            + (" and _Pipes.shp" if k == "OUTPUT_PIPES" else "")
            for k, lbl, d in ELEMENT_OUTPUTS
        ).replace(
            "Pipe_Network.shp and _Pipes.shp",
            "Pipe_Structures.shp and <project number>_Pipes.shp",
        )
        return (
            f"<b>Import LandXML Surface (version {TOOL_VERSION})</b><br><br>"
            "<b>PURPOSE</b><br>Imports a LandXML file (e.g. exported from Civil "
            "3D) into QGIS: each surface as a TIN mesh and a DEM raster, and the "
            "other survey elements in the file - survey points with their point "
            "groups, alignments, feature lines, parcels (sites) and pipe "
            "networks - as GIS layers, with an optional Word report (per "
            "surface a layout map of the elevation and survey elements, and "
            "the same map with the TIN points and triangles).<br><br>"
            "<b>METHOD</b><br>"
            "LandXML gives coordinates as northing, easting (and elevation); "
            "they are written as easting (x), northing (y), elevation (z). "
            "<i>TIN</i>: the surface's own points and triangles (hidden faces, "
            "e.g. holes, left out), written unchanged as a 2DM mesh, which QGIS "
            "opens as a mesh layer. <i>DEM</i>: the TIN sampled at every cell "
            "centre by linear interpolation within each triangle, so it "
            "reproduces the TIN; NoData outside it. <i>Contours</i>: traced from "
            "the DEM and smoothed; every 5th contour is major. Alignments, "
            "feature lines and parcels are built from their lines, arcs and "
            "spirals; pipes run between the centres of their structures. "
            "Alignment profiles, cross-sections and corridor assemblies are not "
            "imported (a corridor is exported as surfaces, alignments and "
            "feature lines, which are).<br><br>"
            "<b>PARAMETERS</b><br>"
            "<i>LandXML file</i>: .xml file.<br>"
            "<i>Surfaces</i>: names to import, separated by commas; blank = "
            "all.<br>"
            "<i>Coordinate system</i>: leave blank to use the one in the file; "
            "set it when the file has none or it is wrong.<br>"
            "<i>DEM cell size</i> (default 0.5 m) and <i>contour interval</i> "
            "(default 0.5 m), in the file's units.<br>"
            "<i>Site boundary</i> (optional): a polygon layer drawn on the "
            "report's layout maps.<br><br>"
            "<b>OUTPUTS</b><br>Tick the outputs wanted and choose a folder for "
            "each kind. Files are named &lt;project number&gt;_&lt;description&gt; "
            "and replace files of the same name; a file that is open cannot be "
            "replaced and is reported, and the other outputs are still written. "
            "Survey elements are written only when the file has them. Layers "
            "are loaded with the Mayim styles; the DEM and TIN with the HEC "
            "elevation colours over the surface's range.<br>"
            + (
                surf
                + "\n"
                + elems
                + "\n  "
                + OUTPUT_REPORT[1]
                + ": <project number>_"
                + clean_name_part(OUTPUT_REPORT[2])
                + ".docx"
            ).replace("\n", "<br>")
            + "<br><br><b>FIELDS</b><br>Survey points: name, code, desc, groups, "
            "x, y, z. Alignments: name, length, sta_start, desc. Feature lines: "
            "name, group, desc (3D lines). Parcels: name, site, type, area_m2, "
            "desc. Pipe structures: name, network, rim_m, sump_m, desc. Pipes: "
            "name, network, from, to, shape, size_m, length_m, slope, inv_start, "
            "inv_end, desc. Breaklines: desc, kind. Contours: id, elev_m, type "
            "(Minor / Major)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(self.INPUT, "LandXML file", extension="xml")
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.SURFACES,
                "Surfaces to import (names, comma-separated; blank = all)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.CRS,
                "Coordinate system (blank = from the file)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CELL_SIZE,
                "DEM cell size (file units, e.g. m)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.001,
                defaultValue=0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONTOUR_INTERVAL,
                f"Contour interval (file units; every {MAJOR_FACTOR}th contour "
                "is major)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.SITE_BOUNDARY,
                "Site boundary for the report layout maps (optional)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix and report, optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project / site name (report title page, optional)",
                optional=True,
            )
        )
        for key, label, *_rest, default in SURFACE_OUTPUTS:
            self.addParameter(
                QgsProcessingParameterBoolean(key, f"Output: {label}", default)
            )
        for key, label, _d in ELEMENT_OUTPUTS:
            self.addParameter(
                QgsProcessingParameterBoolean(key, f"Output: {label}", True)
            )
        self.addParameter(
            QgsProcessingParameterBoolean(
                OUTPUT_REPORT[0], f"Output: {OUTPUT_REPORT[1]}", True
            )
        )
        for name, label in FOLDER_PARAMS.values():
            self.addParameter(
                QgsProcessingParameterFolderDestination(name, label, optional=True)
            )
        osm = QgsProcessingParameterBoolean(
            self.REPORT_OSM,
            "Word report: OpenStreetMap background on the layout maps (needs "
            "internet)",
            defaultValue=True,
        )
        osm.setFlags(osm.flags() | Qgis.ProcessingParameterFlag.Advanced)
        self.addParameter(osm)

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        self._keep = []
        path = self.parameterAsFile(parameters, self.INPUT, context)
        if not path:
            raise QgsProcessingException("Choose a LandXML file.")
        want = {
            k
            for k, *_r in SURFACE_OUTPUTS + ELEMENT_OUTPUTS + [OUTPUT_REPORT]
            if self.parameterAsBoolean(parameters, k, context)
        }
        if not want:
            raise QgsProcessingException("Tick at least one output.")
        kinds = {kind for k, _l, _d, kind, _e, _f in SURFACE_OUTPUTS if k in want}
        if want & {k for k, _l, _d in ELEMENT_OUTPUTS}:
            kinds.add(SHAPEFILE)
        if OUTPUT_REPORT[0] in want:
            kinds.add(WORD)
        folders = {}
        for kind in kinds:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        pn = self.parameterAsString(parameters, self.PROJECT_NUMBER, context).strip()
        pname = self.parameterAsString(parameters, self.PROJECT_NAME, context).strip()
        cell = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        interval = self.parameterAsDouble(parameters, self.CONTOUR_INTERVAL, context)
        if "OUTPUT_CONTOURS" in want and interval <= 0:
            raise QgsProcessingException(
                "Enter a contour interval greater than 0 for the contours output."
            )

        feedback.setProgressText("Reading the LandXML file")
        try:
            data = read_landxml(path)
        except ValueError as e:
            raise QgsProcessingException(str(e)) from e
        wanted = [
            s.strip()
            for s in self.parameterAsString(parameters, self.SURFACES, context).split(
                ","
            )
            if s.strip()
        ]
        if wanted:
            missing = [w for w in wanted if w not in {s.name for s in data.surfaces}]
            for w in missing:
                feedback.pushWarning(f"Surface '{w}' is not in the file.")
            data.surfaces = [s for s in data.surfaces if s.name in wanted]
        for k, v in data.counts().items():
            feedback.pushInfo(f"{k}: {v}")
        for w in data.warnings:
            feedback.pushWarning(w)

        override = self.parameterAsCrs(parameters, self.CRS, context)
        crs, crs_source = ql.crs_from(data, override)
        if not crs.isValid():
            feedback.pushWarning(
                "The file has no coordinate system and none was set: the layers "
                "are written without one. Set the coordinate system and run again."
            )
        else:
            feedback.pushInfo(f"Coordinate system: {crs.authid()} ({crs_source}).")
        wkt = crs.toWkt() if crs.isValid() else ""
        tctx = context.transformContext()

        outputs, written = {}, []

        def done(key, label, p):
            outputs.setdefault(key, p)
            written.append((label, p))
            feedback.pushInfo(f"{label}: {p}")

        def failed(label, p, e):
            feedback.reportError(
                f"{label}: {Path(p).name} could not be written ({e}).",
                fatalError=False,
            )

        def shp(layer, key, label, desc, qml):
            if layer is None or layer.featureCount() == 0:
                return None
            p = _path(folders[SHAPEFILE], pn, desc, ".shp")
            try:
                save_shapefile(layer, p, tctx)
            except Exception as e:
                failed(label, p, e)
                return None
            ql.load_on_completion(
                context, p, Path(p).stem, ql.QmlStyler(qml), self._keep
            )
            done(key, label, p)
            return p

        dem_info, surf_ctx, map_layers = {}, {}, []
        n_surf = max(len(data.surfaces), 1)
        for k, s in enumerate(data.surfaces):
            if feedback.isCanceled():
                break
            feedback.setProgress(int(10 + 60 * k / n_surf))
            feedback.setProgressText(f"Surface {s.name}")
            zmin, zmax = s.elev_range
            sname = clean_name_part(s.name) or f"Surface{k + 1}"
            surf_ctx[s.name] = {"dem": None}
            rings = outline_rings(s.xyz, s.triangles, s.outline_edges())
            boundary = ql.tin_boundary_layer(s, rings, crs)
            map_layers.append(
                (boundary, f"TIN boundary ({s.name})", "LXML_tin_boundary.qml", s.name)
            )
            if OUTPUT_REPORT[0] in want:  # the TIN map (second figure)
                map_layers[:0] = [
                    (
                        ql.tin_points_layer(s, crs),
                        "TIN point",
                        "LXML_tin_points.qml",
                        (s.name, "tin"),
                    ),
                    (ql.tin_edges_layer(s, crs), "TIN triangle", None, (s.name, "tin")),
                ]
            if "OUTPUT_TIN" in want:
                p = _path(folders[RASTER], pn, f"{sname}_TIN", ".2dm")
                try:
                    _remove(p)
                    write_2dm(p, s.xyz, s.triangles, s.name)
                    ql.load_on_completion(
                        context,
                        p,
                        Path(p).stem,
                        ql.ElevationStyler(zmin, zmax, crs),
                        self._keep,
                        ql.MESH_HINT,
                    )
                    done("OUTPUT_TIN", f"TIN mesh {s.name}", p)
                except Exception as e:
                    failed(f"TIN mesh {s.name}", p, e)
            grid = None
            if want & {"OUTPUT_DEM", "OUTPUT_CONTOURS", OUTPUT_REPORT[0]}:
                try:
                    grid = tin_to_grid(s.xyz, s.triangles, cell)
                    surf_ctx[s.name]["dem"] = grid[0].shape
                except ValueError as e:
                    feedback.reportError(f"DEM {s.name}: {e}.", fatalError=False)
            dem_path = None
            if grid is not None and ("OUTPUT_DEM" in want or OUTPUT_REPORT[0] in want):
                if "OUTPUT_DEM" in want:
                    dem_path = _path(folders[RASTER], pn, f"{sname}_DEM", ".tif")
                else:  # the report map still needs the DEM as a file
                    from qgis.core import QgsProcessingUtils

                    dem_path = QgsProcessingUtils.generateTempFilename(
                        f"{sname}_DEM.tif", context
                    )
                try:
                    _write_tif(dem_path, grid[0], grid[1], wkt)
                    dem_info[s.name] = (dem_path, zmin, zmax)
                    if "OUTPUT_DEM" in want:
                        ql.load_on_completion(
                            context,
                            dem_path,
                            Path(dem_path).stem,
                            ql.ElevationStyler(zmin, zmax),
                            self._keep,
                        )
                        done("OUTPUT_DEM", f"DEM {s.name}", dem_path)
                except Exception as e:
                    failed(f"DEM {s.name}", dem_path, e)
            if "OUTPUT_BREAKLINES" in want:
                lyr = ql.breaklines_layer(s, crs)
                shp(
                    lyr,
                    "OUTPUT_BREAKLINES",
                    f"Breaklines {s.name}",
                    f"{sname}_Breaklines",
                    "LXML_breaklines.qml",
                )
                map_layers.insert(0, (lyr, "Breaklines", "LXML_breaklines.qml", s.name))
            if "OUTPUT_TIN_POINTS" in want:
                shp(
                    ql.tin_points_layer(s, crs),
                    "OUTPUT_TIN_POINTS",
                    f"TIN points {s.name}",
                    f"{sname}_TIN_Points",
                    "LXML_tin_points.qml",
                )
            if "OUTPUT_CONTOURS" in want and grid is not None:
                p = _path(folders[SHAPEFILE], pn, f"{sname}_Contours", ".shp")
                try:
                    surface = np.where(grid[0] == NODATA, np.nan, grid[0]).astype(float)
                    n = write_contours(p, surface, grid[1], wkt, interval)
                    feedback.pushInfo(f"Contours {s.name}: {n} line(s).")
                    ql.load_on_completion(
                        context,
                        p,
                        Path(p).stem,
                        ql.QmlStyler("LXML_contours.qml"),
                        self._keep,
                    )
                    done("OUTPUT_CONTOURS", f"Contours {s.name}", p)
                    contours = QgsVectorLayer(p, "Contours", "ogr")
                    if contours.isValid():
                        map_layers.append(
                            (contours, "Contours", "LXML_contours.qml", s.name)
                        )
                except Exception as e:
                    failed(f"Contours {s.name}", p, e)

        feedback.setProgress(75)
        feedback.setProgressText("Survey elements")
        element_layers = []
        if "OUTPUT_POINTS" in want and data.points:
            lyr = ql.survey_points_layer(data.points, crs)
            shp(
                lyr,
                "OUTPUT_POINTS",
                "Survey points",
                "Survey Points",
                "LXML_survey_points.qml",
            )
            element_layers.append((lyr, "Survey point", "LXML_survey_points.qml", None))
        if "OUTPUT_ALIGNMENTS" in want and data.alignments:
            lyr = ql.alignments_layer(data.alignments, crs)
            shp(
                lyr,
                "OUTPUT_ALIGNMENTS",
                "Alignments",
                "Alignments",
                "LXML_alignments.qml",
            )
            element_layers.append((lyr, "Alignment", "LXML_alignments.qml", None))
        if "OUTPUT_FEATURE_LINES" in want and data.feature_lines:
            lyr = ql.feature_lines_layer(data.feature_lines, crs)
            shp(
                lyr,
                "OUTPUT_FEATURE_LINES",
                "Feature lines",
                "Feature Lines",
                "LXML_feature_lines.qml",
            )
            element_layers.append((lyr, "Feature line", "LXML_feature_lines.qml", None))
        if "OUTPUT_PARCELS" in want and data.parcels:
            lyr = ql.parcels_layer(data.parcels, crs)
            shp(lyr, "OUTPUT_PARCELS", "Parcels", "Parcels", "LXML_parcels.qml")
            element_layers.append((lyr, "Parcel", "LXML_parcels.qml", None))
        if "OUTPUT_PIPES" in want and (data.structures or data.pipes):
            st = ql.structures_layer(data.structures, crs)
            pp = ql.pipes_layer(data.pipes, crs)
            shp(
                st,
                "OUTPUT_PIPES",
                "Pipe structures",
                "Pipe Structures",
                "LXML_structures.qml",
            )
            shp(pp, "OUTPUT_PIPES", "Pipes", "Pipes", "LXML_pipes.qml")
            element_layers += [
                (st, "Pipe structure", "LXML_structures.qml", None),
                (pp, "Pipe", "LXML_pipes.qml", None),
            ]

        if OUTPUT_REPORT[0] in want:
            feedback.setProgress(85)
            feedback.setProgressText("Writing the report")
            p = _path(folders[WORD], pn, OUTPUT_REPORT[2], ".docx")
            messages = []
            plan_map = None
            site_layers, site_name = [], None
            site = self.parameterAsSource(parameters, self.SITE_BOUNDARY, context)
            if site is not None and crs.isValid():
                site_name = self.parameterAsLayer(
                    parameters, self.SITE_BOUNDARY, context
                )
                site_name = site_name.name() if site_name is not None else "yes"
                layer = copy_source_layer(site, "Site boundary", crs, tctx)
                if layer.featureCount():
                    site_layers = [
                        (ql.site_boundary_layer(layer), "Site boundary", None, None)
                    ]
            if crs.isValid():
                plan_map = ql.make_plan_map(
                    crs,
                    tctx,
                    dem_info,
                    site_layers + element_layers + map_layers,
                    use_osm=self.parameterAsBoolean(
                        parameters, self.REPORT_OSM, context
                    ),
                    messages=messages,
                )
            ctx = {
                "tool": self.displayName(),
                "version": TOOL_VERSION,
                "software": _software_version(),
                "project_number": pn,
                "project_name": pname,
                "date": _dt.date.today().isoformat(),
                "crs": (
                    f"{crs.authid()} - {crs.description()}" if crs.isValid() else "-"
                ),
                "crs_source": crs_source,
                "cell_size": cell,
                "contour_interval": interval if "OUTPUT_CONTOURS" in want else None,
                "surfaces": surf_ctx,
                "outputs": list(written),
                "site_boundary": site_name,
            }
            try:
                _remove(p)
                notes = write_report(p, data, ctx, plan_map=plan_map)
                for m in dict.fromkeys(notes + messages):
                    feedback.pushWarning(m)
                done(OUTPUT_REPORT[0], OUTPUT_REPORT[1], p)
            except ImportError as e:
                feedback.pushWarning(
                    f"Report skipped: {e}. Install python-docx and matplotlib in "
                    "QGIS's Python."
                )
            except Exception as e:
                failed(OUTPUT_REPORT[1], p, e)

        feedback.setProgress(100)
        if not written:
            raise QgsProcessingException(
                "No output could be written - see the messages above (the file "
                "may not contain the elements ticked)."
            )
        for kind, (name, _label) in FOLDER_PARAMS.items():
            if kind in folders:
                outputs[name] = folders[kind]
        return outputs
