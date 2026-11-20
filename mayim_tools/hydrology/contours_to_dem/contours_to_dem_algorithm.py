"""Processing Toolbox algorithm: Contours to DEM.

Contours (with a level field or Z) and stream centrelines (optional ridge
lines and spot heights) in -> a DEM from a linear TIN in which the streams
fall steadily between their contour crossings, so valley floors slope
instead of stepping. Plus the lines with their heights (3D), the contour
crossings and a check of the flat stretches along each line, with and
without the lines.

The TIN is QGIS's own TIN mesh (native:tinmeshcreation, a constrained
Delaunay triangulation in which the contours and lines are fixed edges),
rasterised with native:meshrasterize. It is linear, like QGIS's TIN
interpolation, and much faster: about 2 s against 5 min for 300000 cells.

Outputs follow the Mayim output standard. The line heights and the check
are in core.py (no QGIS; see tests/test_contours_to_dem_core.py).
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsLineString,
    QgsMemoryProviderUtils,
    QgsPoint,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsVectorFileWriter,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    RASTER,
    SHAPEFILE,
    OutputSpec,
    clean_name_part,
)
from mayim_tools.hydrology.catchment_delineation.qgis_outputs import save_shapefile
from mayim_tools.hydrology.inundation_boundary.qgis_outputs import (
    load_on_completion,
    memory_layer,
)

from .core import (
    TOOL_VERSION,
    ContourError,
    contour_interval,
    flat_stretches,
    line_heights,
    oriented_downstream,
    profile_samples,
)

NODATA = -9999.0
_S = QMetaType.Type.QString
_I = QMetaType.Type.Int
_D = QMetaType.Type.Double

LINE_FIELDS = [
    ("name", _S, "line name (name field, else Stream n / Ridge n)"),
    ("kind", _S, "stream or ridge"),
    ("length_m", _D, "length of the line, m"),
    ("crossings", _I, "contour crossings"),
    ("z_start", _D, "height at the start (streams: the upstream end), m"),
    ("z_end", _D, "height at the end, m"),
    ("fall_m", _D, "fall from start to end, m"),
    ("slope_pct", _D, "mean slope, %"),
    ("flat_m", _D, "longest flat stretch along the line on the DEM, m"),
    ("flat_m_c", _D, "the same on a DEM from the contours alone, m"),
    ("notes", _S, "warnings for this line"),
]
CROSSING_FIELDS = [
    ("name", _S, "line name"),
    ("chain_m", _D, "distance along the line, m"),
    ("z", _D, "contour level, m"),
]

OUTPUT_SPECS = [
    OutputSpec("OUTPUT_DEM", "DEM (GeoTIFF)", "DEM", RASTER),
    OutputSpec(
        "OUTPUT_DEM_PLAIN",
        "DEM from the contours alone, for comparison",
        "DEM Contours Only",
        RASTER,
        False,
    ),
    OutputSpec(
        "OUTPUT_LINES",
        "Stream and ridge lines with heights (3D lines)",
        "DEM Lines 3D",
        SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_CROSSINGS",
        "Contour crossings (points)",
        "DEM Line Crossings",
        SHAPEFILE,
        False,
    ),
    OutputSpec("OUTPUT_CSV", "Line check (CSV)", "DEM Line Check", CSV),
]
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
}


def file_prefix(project_number: str | None, dem_description: str | None) -> str:
    """'<project number>_<DEM description>' (either may be empty), as in DEM
    Hydrological Processing, e.g. M1234_Contours2m_DEM.tif."""
    parts = (clean_name_part(project_number), clean_name_part(dem_description))
    return "_".join(p for p in parts if p)


class _DemStyler(QgsProcessingLayerPostProcessorInterface):
    """HEC elevation colours over the DEM's own range."""

    def __init__(self, zmin, zmax):
        super().__init__()
        self.zmin, self.zmax = zmin, zmax

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            from mayim_tools.data.landxml_import.qgis_layers import style_dem

            style_dem(layer, self.zmin, self.zmax)
            layer.triggerRepaint()
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style not applied to {layer.name()}: {e}")


def _r(value, digits=3):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def _parts(geom):
    """[(x, y) array] of each part of a line geometry."""
    if geom is None or geom.isNull() or geom.isEmpty():
        return []
    lines = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
    return [
        np.array([(p.x(), p.y()) for p in part], float)
        for part in lines
        if len(part) >= 2
    ]


def _run_child(alg_id, params, context, feedback):
    alg = QgsApplication.processingRegistry().createAlgorithmById(alg_id)
    if alg is None:
        raise QgsProcessingException(f"QGIS algorithm {alg_id} is not available.")
    results, ok = alg.run(params, context, feedback)
    if not ok:
        raise QgsProcessingException(f"QGIS algorithm {alg_id} failed.")
    return results


class ContoursToDemAlgorithm(QgsProcessingAlgorithm):
    CONTOURS = "CONTOURS"
    LEVEL_FIELD = "LEVEL_FIELD"
    STREAMS = "STREAMS"
    RIDGES = "RIDGES"
    NAME_FIELD = "NAME_FIELD"
    SPOTS = "SPOTS"
    SPOT_FIELD = "SPOT_FIELD"
    BOUNDARY = "BOUNDARY"
    CELL_SIZE = "CELL_SIZE"
    INTERVAL = "INTERVAL"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    DEM_DESCRIPTION = "DEM_DESCRIPTION"

    def __init__(self):
        super().__init__()
        self._post_processors = []

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return ContoursToDemAlgorithm()

    def name(self):
        return "contours_to_dem"

    def displayName(self):
        return "Contours to DEM"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        files = "\n".join(
            f"  {spec.label}: <project number>_<DEM description>_"
            f"{clean_name_part(spec.description)}{EXTENSIONS[spec.kind]}"
            for spec in OUTPUT_SPECS
        )
        fields = "\n".join(f"  {n}: {d}" for n, _, d in LINE_FIELDS)
        return (
            f"Contours to DEM (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tA DEM from contours that slopes along the valley floors. "
            "A TIN from contours alone joins points of the same contour across "
            "a valley, so the triangles there are flat and the valley floor "
            "becomes a staircase: flat for the length of each contour's V, then "
            "one contour interval down. Stream lines with heights cut those "
            "triangles and turn the staircase into a slope; ridge lines do the "
            "same on crests.\n"
            "\n"
            "METHOD:\t1. Each stream (and ridge) line gets a height at every "
            "contour it crosses, the contour's level. 2. Between crossings the "
            "height changes linearly with distance along the line; beyond the "
            "first and last crossing the end gradient is carried on, kept "
            "within one contour interval. 3. A linear TIN is built with the "
            "contours, the lines and any spot heights as fixed edges and points "
            "(QGIS TIN mesh), and rasterised at the cell size. 4. The DEM is "
            "sampled along each line to measure its longest flat stretch, also "
            "on a DEM from the contours alone for comparison.\n"
            "\n"
            "INPUTS:\tContours as lines, with their level in a field or as Z. "
            "Stream centrelines drawn along the valley floors (the direction "
            "does not matter; the Stream Network output can be used); each must "
            "cross at least two contours. Optional: ridge lines along crests, "
            "spot heights (points with a level field or Z) and a boundary "
            "polygon (the DEM is cut to it). A projected CRS in metres.\n"
            "\n"
            "CHECK:\tEach line's warnings say when it crosses one contour only, "
            "crosses the same contour twice in a row, skips a contour, or (for "
            "a stream) goes both up and down. 'flat_m' is the longest flat "
            "stretch along the line: near zero with the lines, often hundreds "
            "of metres without them.\n"
            "\n"
            "OUTPUTS:\tTick the outputs wanted and choose a folder for each kind "
            "used.\n" + files + "\n"
            "\n"
            "LINE FIELDS:\n" + fields
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.CONTOURS,
                "Contours (lines)",
                types=[QgsProcessing.SourceType.TypeVectorLine],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.LEVEL_FIELD,
                "Contour level field (leave empty to use the Z values)",
                parentLayerParameterName=self.CONTOURS,
                type=Qgis.ProcessingFieldParameterDataType.Numeric,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.STREAMS,
                "Stream centrelines along the valley floors",
                types=[QgsProcessing.SourceType.TypeVectorLine],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.RIDGES,
                "Ridge lines along crests (optional)",
                types=[QgsProcessing.SourceType.TypeVectorLine],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.NAME_FIELD,
                "Line name field in the stream and ridge layers (optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.SPOTS,
                "Spot heights (optional)",
                types=[QgsProcessing.SourceType.TypeVectorPoint],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.SPOT_FIELD,
                "Spot height level field (leave empty to use the Z values)",
                parentLayerParameterName=self.SPOTS,
                type=Qgis.ProcessingFieldParameterDataType.Numeric,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.BOUNDARY,
                "Boundary polygon (optional; default the extent of the contours)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CELL_SIZE,
                "Cell size (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.01,
                defaultValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.INTERVAL,
                "Contour interval (m; 0 = found from the contour levels)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix, optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DEM_DESCRIPTION,
                "DEM description (file names, after the project number; "
                "optional, e.g. Contours2m)",
                optional=True,
            )
        )
        for spec in OUTPUT_SPECS:
            self.addParameter(
                QgsProcessingParameterBoolean(
                    spec.key, f"Output: {spec.label}", defaultValue=spec.default
                )
            )
        for name, label in FOLDER_PARAMS.values():
            self.addParameter(
                QgsProcessingParameterFolderDestination(name, label, optional=True)
            )

    # ------------------------------------------------------------------

    def _read_contours(self, source, field_name, crs, context, feedback):
        """[(xy, z)] in ``crs`` from the contour layer."""
        transform = None
        if source.sourceCrs() != crs:
            transform = QgsCoordinateTransform(
                source.sourceCrs(), crs, context.transformContext()
            )
        has_z = QgsWkbTypes.hasZ(source.wkbType())
        if not field_name and not has_z:
            raise QgsProcessingException(
                "The contours have no Z values: choose the contour level field."
            )
        out, skipped = [], 0
        for feat in source.getFeatures():
            geom = QgsGeometry(feat.geometry())
            if transform is not None:
                geom.transform(transform)
            if field_name:
                value = feat[field_name]
                try:
                    z = float(value)
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                for xy in _parts(geom):
                    out.append((xy, z))
            else:
                for part in geom.constParts():
                    pts = [(v.x(), v.y(), v.z()) for v in part.vertices()]
                    if len(pts) >= 2 and math.isfinite(pts[0][2]):
                        arr = np.array(pts, float)
                        out.append((arr[:, :2], float(np.median(arr[:, 2]))))
        if skipped:
            feedback.pushWarning(f"{skipped} contour(s) without a level were left out.")
        if not out:
            raise QgsProcessingException("No contours with a level were found.")
        return out

    def _read_lines(self, source, kind, name_field, crs, context):
        if source is None:
            return []
        transform = None
        if source.sourceCrs() != crs:
            transform = QgsCoordinateTransform(
                source.sourceCrs(), crs, context.transformContext()
            )
        names = [f.name() for f in source.fields()]
        field = name_field if name_field in names else None
        out = []
        for k, feat in enumerate(source.getFeatures(), start=1):
            geom = QgsGeometry(feat.geometry())
            if transform is not None:
                geom.transform(transform)
            parts = _parts(geom)
            base = (
                str(feat[field])
                if field and feat[field] not in (None, "")
                else (f"{kind.title()} {k}")
            )
            for j, xy in enumerate(parts):
                out.append((base if len(parts) == 1 else f"{base} ({j + 1})", xy))
        return out

    def _write_tin_inputs(self, folder, crs, contours, lines, spots):
        """GeoPackages for the TIN mesh: contours (z field), lines (3D), spots
        (z field)."""
        ctx = QgsProcessingContext()
        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"

        def write(name, layer):
            path = os.path.join(folder, f"{name}.gpkg")
            err = QgsVectorFileWriter.writeAsVectorFormatV3(
                layer, path, ctx.transformContext(), opts
            )
            if err[0] != QgsVectorFileWriter.WriterError.NoError:
                raise QgsProcessingException(f"Could not write {name}: {err[1]}")
            return path

        fields = QgsFields()
        fields.append(QgsField("z", _D))
        c = QgsMemoryProviderUtils.createMemoryLayer(
            "contours", fields, Qgis.WkbType.LineString, crs
        )
        feats = []
        for xy, z in contours:
            f = QgsFeature(fields)
            f.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*p) for p in xy]))
            f["z"] = float(z)
            feats.append(f)
        c.dataProvider().addFeatures(feats)
        paths = {"contours": write("contours", c)}
        if lines:
            ll = QgsMemoryProviderUtils.createMemoryLayer(
                "lines", QgsFields(), Qgis.WkbType.LineStringZ, crs
            )
            feats = []
            for line in lines:
                f = QgsFeature()
                f.setGeometry(
                    QgsGeometry(
                        QgsLineString([QgsPoint(*map(float, p)) for p in line.xyz])
                    )
                )
                feats.append(f)
            ll.dataProvider().addFeatures(feats)
            paths["lines"] = write("lines", ll)
        if spots:
            sp = QgsMemoryProviderUtils.createMemoryLayer(
                "spots", fields, Qgis.WkbType.Point, crs
            )
            feats = []
            for x, y, z in spots:
                f = QgsFeature(fields)
                f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                f["z"] = float(z)
                feats.append(f)
            sp.dataProvider().addFeatures(feats)
            paths["spots"] = write("spots", sp)
        return paths

    def _tin_dem(
        self, sources, crs, extent, cell, out_path, folder, tag, context, feedback
    ):
        mesh_path = os.path.join(folder, f"{tag}.2dm")
        _run_child(
            "native:tinmeshcreation",
            {
                "SOURCE_DATA": sources,
                "MESH_FORMAT": 0,
                "CRS_OUTPUT": crs,
                "OUTPUT_MESH": mesh_path,
            },
            context,
            feedback,
        )
        from qgis.core import QgsMeshLayer

        mesh = QgsMeshLayer(mesh_path, tag, "mdal")
        if not mesh.isValid():
            raise QgsProcessingException("The TIN mesh could not be read back.")
        _run_child(
            "native:meshrasterize",
            {
                "INPUT": mesh,
                "DATASET_GROUPS": [0],
                "DATASET_TIME": {"type": "static"},
                "EXTENT": extent,
                "PIXEL_SIZE": cell,
                "CRS_OUTPUT": crs,
                "OUTPUT": out_path,
            },
            context,
            feedback,
        )
        return out_path

    @staticmethod
    def _finish_raster(src, dst, boundary_wkts, crs_wkt):
        """GeoTIFF with nodata -9999, LZW, cut to the boundary polygons."""
        from osgeo import gdal, ogr, osr

        gdal.UseExceptions()
        ds = gdal.Open(src)
        band = ds.GetRasterBand(1)
        a = band.ReadAsArray().astype(np.float32)
        nd = band.GetNoDataValue()
        bad = ~np.isfinite(a)
        if nd is not None:
            bad |= a == nd
        if boundary_wkts:
            mem = gdal.GetDriverByName("MEM").Create(
                "", ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte
            )
            mem.SetGeoTransform(ds.GetGeoTransform())
            vds = ogr.GetDriverByName("Memory").CreateDataSource("b")
            srs = osr.SpatialReference()
            srs.ImportFromWkt(crs_wkt)
            lyr = vds.CreateLayer("b", srs=srs, geom_type=ogr.wkbPolygon)
            for wkt in boundary_wkts:
                feat = ogr.Feature(lyr.GetLayerDefn())
                feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
                lyr.CreateFeature(feat)
            gdal.RasterizeLayer(mem, [1], lyr, burn_values=[1])
            bad |= mem.GetRasterBand(1).ReadAsArray() == 0
        a[bad] = NODATA
        out = gdal.GetDriverByName("GTiff").Create(
            dst,
            ds.RasterXSize,
            ds.RasterYSize,
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=LZW", "TILED=YES", "PREDICTOR=3"],
        )
        out.SetGeoTransform(ds.GetGeoTransform())
        out.SetProjection(crs_wkt or ds.GetProjection())
        ob = out.GetRasterBand(1)
        ob.SetNoDataValue(NODATA)
        ob.WriteArray(a)
        ob.FlushCache()
        out = None
        ds = None
        valid = a[a != NODATA]
        return (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)

    @staticmethod
    def _sampler(path):
        from osgeo import gdal

        ds = gdal.Open(path)
        gt = ds.GetGeoTransform()
        a = ds.GetRasterBand(1).ReadAsArray().astype(float)
        a[a == NODATA] = np.nan

        def sample(x, y):
            c = np.floor((np.asarray(x) - gt[0]) / gt[1]).astype(int)
            r = np.floor((np.asarray(y) - gt[3]) / gt[5]).astype(int)
            ok = (c >= 0) & (c < a.shape[1]) & (r >= 0) & (r < a.shape[0])
            out = np.full(len(c), np.nan)
            out[ok] = a[r[ok], c[ok]]
            return out

        return sample

    # ------------------------------------------------------------------

    def processAlgorithm(self, parameters, context, feedback):
        self._post_processors = []
        contours_src = self.parameterAsSource(parameters, self.CONTOURS, context)
        if contours_src is None:
            raise QgsProcessingException("Choose the contour layer.")
        crs = contours_src.sourceCrs()
        if not crs.isValid() or crs.isGeographic():
            raise QgsProcessingException(
                "The contours must be in a projected CRS in metres."
            )
        level_field = self.parameterAsString(parameters, self.LEVEL_FIELD, context)
        cell = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        prefix = file_prefix(
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context),
            self.parameterAsString(parameters, self.DEM_DESCRIPTION, context),
        )
        selected = [
            s
            for s in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, s.key, context)
        ]
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {s.kind for s in selected}:
            key, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, key, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {s.key: s.path(folders, prefix) for s in selected}

        feedback.setProgressText("Reading the contours and lines")
        contours = self._read_contours(
            contours_src, level_field, crs, context, feedback
        )
        interval = self.parameterAsDouble(parameters, self.INTERVAL, context)
        try:
            if interval <= 0:
                interval = contour_interval([z for _, z in contours])
                feedback.pushInfo(f"Contour interval found: {interval:g} m.")
        except ContourError as e:
            raise QgsProcessingException(str(e)) from e
        levels = [z for _, z in contours]
        feedback.pushInfo(
            f"{len(contours)} contour lines, levels {min(levels):g} to "
            f"{max(levels):g} m."
        )
        name_field = self.parameterAsString(parameters, self.NAME_FIELD, context)
        raw = [
            (n, xy, "stream")
            for n, xy in self._read_lines(
                self.parameterAsSource(parameters, self.STREAMS, context),
                "stream",
                name_field,
                crs,
                context,
            )
        ] + [
            (n, xy, "ridge")
            for n, xy in self._read_lines(
                self.parameterAsSource(parameters, self.RIDGES, context),
                "ridge",
                name_field,
                crs,
                context,
            )
        ]
        if not raw:
            feedback.pushWarning(
                "No stream or ridge lines: the DEM is built from the contours "
                "alone and will have flat valley floors."
            )
        step = max(cell, 0.5)
        lines = []
        for name, xy, kind in raw:
            if feedback.isCanceled():
                raise QgsProcessingException("Cancelled.")
            line = oriented_downstream(
                line_heights(name, xy, contours, interval, step, kind)
            )
            for w in line.warnings:
                feedback.pushWarning(f"{kind.title()} '{name}': {w}")
            lines.append(line)
        used = [ln for ln in lines if ln.usable]
        feedback.pushInfo(
            f"{len(used)} of {len(lines)} line(s) get heights from "
            f"{sum(len(ln.crossings) for ln in used)} contour crossings."
        )
        feedback.setProgress(20)

        spots = []
        spot_src = self.parameterAsSource(parameters, self.SPOTS, context)
        if spot_src is not None:
            sfield = self.parameterAsString(parameters, self.SPOT_FIELD, context)
            st = None
            if spot_src.sourceCrs() != crs:
                st = QgsCoordinateTransform(
                    spot_src.sourceCrs(), crs, context.transformContext()
                )
            for feat in spot_src.getFeatures():
                geom = QgsGeometry(feat.geometry())
                if st is not None:
                    geom.transform(st)
                for v in geom.vertices():
                    z = feat[sfield] if sfield else v.z()
                    try:
                        z = float(z)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(z):
                        spots.append((v.x(), v.y(), z))
            feedback.pushInfo(f"{len(spots)} spot heights.")

        boundary_wkts = []
        bsrc = self.parameterAsSource(parameters, self.BOUNDARY, context)
        if bsrc is not None:
            bt = None
            if bsrc.sourceCrs() != crs:
                bt = QgsCoordinateTransform(
                    bsrc.sourceCrs(), crs, context.transformContext()
                )
            bbox = None
            for feat in bsrc.getFeatures():
                geom = QgsGeometry(feat.geometry())
                if bt is not None:
                    geom.transform(bt)
                if geom.isNull() or geom.isEmpty():
                    continue
                boundary_wkts.append(geom.asWkt())
                bb = geom.boundingBox()
                if bbox is None:
                    bbox = bb
                else:
                    bbox.combineExtentWith(bb)
            if not boundary_wkts:
                raise QgsProcessingException("The boundary layer has no polygons.")
            xmin, xmax = bbox.xMinimum(), bbox.xMaximum()
            ymin, ymax = bbox.yMinimum(), bbox.yMaximum()
        else:
            allxy = np.vstack([xy for xy, _ in contours])
            xmin, ymin = allxy.min(axis=0)
            xmax, ymax = allxy.max(axis=0)
        # snap the extent to the cell size
        xmin = math.floor(xmin / cell) * cell
        ymin = math.floor(ymin / cell) * cell
        xmax = math.ceil(xmax / cell) * cell
        ymax = math.ceil(ymax / cell) * cell
        ncells = (xmax - xmin) * (ymax - ymin) / cell**2
        feedback.pushInfo(
            f"DEM grid {int(round((xmax - xmin) / cell))} x "
            f"{int(round((ymax - ymin) / cell))} cells at {cell:g} m "
            f"({ncells / 1e6:.1f} million cells)."
        )
        extent = f"{xmin},{xmax},{ymin},{ymax} [{crs.authid()}]"

        work = tempfile.mkdtemp(prefix="mayim_ctd_")
        inputs = self._write_tin_inputs(work, crs, contours, used, spots)
        base = [{"source": inputs["contours"], "type": 2, "attributeIndex": 1}]
        if "spots" in inputs:
            base.append({"source": inputs["spots"], "type": 0, "attributeIndex": 1})
        full = list(base)
        if "lines" in inputs:
            full.append({"source": inputs["lines"], "type": 2, "attributeIndex": -1})

        feedback.setProgressText("Building the TIN and the DEM")
        tin_full = self._tin_dem(
            full,
            crs,
            extent,
            cell,
            os.path.join(work, "dem_full.tif"),
            work,
            "full",
            context,
            feedback,
        )
        feedback.setProgress(60)
        written = []
        outputs = {}
        crs_wkt = (
            crs.toWkt(Qgis.CrsWktVariant.Preferred)
            if hasattr(Qgis, "CrsWktVariant")
            else crs.toWkt()
        )
        dem_check = os.path.join(work, "dem_full_final.tif")
        zr = self._finish_raster(tin_full, dem_check, boundary_wkts, crs_wkt)
        if "OUTPUT_DEM" in paths:
            try:
                self._finish_raster(
                    tin_full, paths["OUTPUT_DEM"], boundary_wkts, crs_wkt
                )
                written.append(paths["OUTPUT_DEM"])
                outputs["OUTPUT_DEM"] = paths["OUTPUT_DEM"]
                load_on_completion(
                    context,
                    paths["OUTPUT_DEM"],
                    Path(paths["OUTPUT_DEM"]).stem,
                    _DemStyler(*zr),
                    self._post_processors,
                )
            except Exception as e:
                self._failed(feedback, paths["OUTPUT_DEM"], e)

        plain_check = None
        if used:
            feedback.setProgressText("DEM from the contours alone, for the check")
            tin_plain = self._tin_dem(
                base,
                crs,
                extent,
                cell,
                os.path.join(work, "dem_plain.tif"),
                work,
                "plain",
                context,
                feedback,
            )
            plain_check = os.path.join(work, "dem_plain_final.tif")
            self._finish_raster(tin_plain, plain_check, boundary_wkts, crs_wkt)
            if "OUTPUT_DEM_PLAIN" in paths:
                try:
                    zp = self._finish_raster(
                        tin_plain, paths["OUTPUT_DEM_PLAIN"], boundary_wkts, crs_wkt
                    )
                    written.append(paths["OUTPUT_DEM_PLAIN"])
                    outputs["OUTPUT_DEM_PLAIN"] = paths["OUTPUT_DEM_PLAIN"]
                    load_on_completion(
                        context,
                        paths["OUTPUT_DEM_PLAIN"],
                        Path(paths["OUTPUT_DEM_PLAIN"]).stem,
                        _DemStyler(*zp),
                        self._post_processors,
                    )
                except Exception as e:
                    self._failed(feedback, paths["OUTPUT_DEM_PLAIN"], e)
        feedback.setProgress(80)

        # the check: flat stretches along each line, with and without lines
        rows = []
        s_full = self._sampler(dem_check)
        s_plain = self._sampler(plain_check) if plain_check else None
        for line in used:
            ch, x, y = profile_samples(line, cell)
            f_full = flat_stretches(ch, s_full(x, y), interval)
            f_plain = flat_stretches(ch, s_plain(x, y), interval) if s_plain else None
            z0, z1 = float(line.xyz[0, 2]), float(line.xyz[-1, 2])
            rows.append(
                {
                    "line": line,
                    "name": line.name,
                    "kind": line.kind,
                    "length_m": line.length,
                    "crossings": len(line.crossings),
                    "z_start": z0,
                    "z_end": z1,
                    "fall_m": z0 - z1,
                    "slope_pct": 100 * (z0 - z1) / line.length if line.length else None,
                    "flat_m": f_full.longest_flat_m,
                    "flat_m_c": f_plain.longest_flat_m if f_plain else None,
                    "notes": "; ".join(line.warnings),
                }
            )
            feedback.pushInfo(
                f"{line.kind.title()} '{line.name}': {len(line.crossings)} "
                f"crossings, fall {z0 - z1:.2f} m over {line.length:.0f} m; longest "
                f"flat stretch {f_full.longest_flat_m:.0f} m"
                + (
                    f" (contours alone: {f_plain.longest_flat_m:.0f} m)"
                    if f_plain
                    else ""
                )
            )
        for line in lines:
            if not line.usable:
                rows.append(
                    {
                        "line": line,
                        "name": line.name,
                        "kind": line.kind,
                        "length_m": line.length,
                        "crossings": 0,
                        "notes": "; ".join(line.warnings),
                    }
                )

        if "OUTPUT_LINES" in paths and used:
            try:
                lrows = []
                for r in rows:
                    if not r["line"].usable:
                        continue
                    geom = QgsGeometry(
                        QgsLineString([QgsPoint(*map(float, p)) for p in r["line"].xyz])
                    )
                    lrows.append(
                        (
                            geom,
                            {
                                n: (
                                    _r(r.get(n))
                                    if t == _D
                                    else (
                                        int(r[n])
                                        if t == _I
                                        else str(r.get(n) or "")[:254]
                                    )
                                )
                                for n, t, _ in LINE_FIELDS
                            },
                        )
                    )
                layer = memory_layer(
                    "lines", LINE_FIELDS, Qgis.WkbType.LineStringZ, crs, lrows
                )
                p = save_shapefile(
                    layer, paths["OUTPUT_LINES"], context.transformContext()
                )
                written.append(p)
                outputs["OUTPUT_LINES"] = p
                load_on_completion(
                    context, p, Path(p).stem, None, self._post_processors
                )
            except Exception as e:
                self._failed(feedback, paths["OUTPUT_LINES"], e)

        if "OUTPUT_CROSSINGS" in paths and used:
            try:
                prow = [
                    (
                        QgsGeometry.fromPointXY(QgsPointXY(c.x, c.y)),
                        {"name": ln.name, "chain_m": _r(c.chainage, 2), "z": _r(c.z)},
                    )
                    for ln in used
                    for c in ln.crossings
                ]
                layer = memory_layer(
                    "crossings", CROSSING_FIELDS, Qgis.WkbType.Point, crs, prow
                )
                p = save_shapefile(
                    layer, paths["OUTPUT_CROSSINGS"], context.transformContext()
                )
                written.append(p)
                outputs["OUTPUT_CROSSINGS"] = p
                load_on_completion(
                    context, p, Path(p).stem, None, self._post_processors
                )
            except Exception as e:
                self._failed(feedback, paths["OUTPUT_CROSSINGS"], e)

        if "OUTPUT_CSV" in paths:
            try:
                self._write_csv(
                    paths["OUTPUT_CSV"], rows, used, interval, cell, len(contours)
                )
                written.append(paths["OUTPUT_CSV"])
                outputs["OUTPUT_CSV"] = paths["OUTPUT_CSV"]
            except Exception as e:
                self._failed(feedback, paths["OUTPUT_CSV"], e)

        if not written:
            raise QgsProcessingException("No output could be written.")
        for p in written:
            feedback.pushInfo(f"Written: {p}")
        feedback.setProgress(100)
        return outputs

    @staticmethod
    def _failed(feedback, path, error):
        feedback.reportError(
            f"{Path(path).name} could not be written ({error}). If it is open in "
            "QGIS or another program, close it and run again.",
            fatalError=False,
        )

    @staticmethod
    def _write_csv(path, rows, used, interval, cell, n_contours):
        def v(x, d=3):
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                return "-"
            return f"{x:.{d}f}" if isinstance(x, float) else str(x)

        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["# Column key"])
            for n, _, d in LINE_FIELDS:
                w.writerow([f"# {n}", d])
            for n, _, d in CROSSING_FIELDS:
                w.writerow([f"# {n} (crossings)", d])
            w.writerow(["[run]"])
            w.writerow(["item", "value"])
            w.writerow(["tool", f"Contours to DEM {TOOL_VERSION}"])
            w.writerow(["contour lines", n_contours])
            w.writerow(["contour interval (m)", f"{interval:g}"])
            w.writerow(["cell size (m)", f"{cell:g}"])
            w.writerow(["flat = a step smaller than (m)", f"{0.001 * interval:g}"])
            w.writerow([])
            w.writerow(["[lines]"])
            w.writerow([n for n, _, _ in LINE_FIELDS])
            for r in rows:
                w.writerow(
                    [
                        v(r.get(n), 2 if n in ("length_m", "flat_m", "flat_m_c") else 3)
                        for n, _, _ in LINE_FIELDS
                    ]
                )
            w.writerow([])
            w.writerow(["[crossings]"])
            w.writerow([n for n, _, _ in CROSSING_FIELDS])
            for ln in used:
                for c in ln.crossings:
                    w.writerow([ln.name, v(c.chainage, 2), v(c.z)])
