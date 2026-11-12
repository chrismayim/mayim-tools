"""Processing algorithm: Data Boundary (mask polygon) (Data Tools).

Draws the polygon around a point, line or polygon layer - e.g. contours
or survey points - for use as a mask, e.g. to clip a DEM interpolated
from the data to the area the data covers. Thin QGIS wrapper: the
concave and convex hulls are in core.py, the QGIS geometry steps in
geometry.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import SHAPEFILE, output_file_name
from mayim_tools.hydrology.catchment_delineation.qgis_outputs import save_shapefile

from . import geometry as gm
from .core import TOOL_VERSION, concave_polygons, convex_polygon

CONCAVE, CONVEX, BUFFER = 0, 1, 2
METHODS = [
    "Concave (maximum edge length)",
    "Convex hull",
    "Buffer and dissolve",
]
METHOD_NAMES = ["concave", "convex", "buffer"]
DESCRIPTION = "Data Boundary"
STYLE = Path(__file__).resolve().parent / "styles" / "DATA_boundary.qml"
FIELDS = [
    ("id", "Int", "part number (1 = largest)"),
    ("area_m2", "Double", "area (m²)"),
    ("area_ha", "Double", "area (ha)"),
    ("perim_m", "Double", "perimeter (m)"),
    ("method", "String", "boundary method (concave, convex, buffer)"),
    ("param_m", "Double", "maximum edge length or buffer distance (m)"),
    ("offset_m", "Double", "outward offset (m)"),
    ("n_points", "Int", "data points used (concave and convex)"),
]


class _Styler(QgsProcessingLayerPostProcessorInterface):
    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            layer.loadNamedStyle(str(STYLE))
            layer.triggerRepaint()
        except Exception as e:  # styling never fails the run
            feedback.pushWarning(f"Style {STYLE.name} not applied: {e}")


class DataBoundaryAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    METHOD = "METHOD"
    MAX_EDGE = "MAX_EDGE"
    BUFFER = "BUFFER"
    OFFSET = "OFFSET"
    FILL_HOLES = "FILL_HOLES"
    MERGE = "MERGE"
    MIN_AREA = "MIN_AREA"
    SMOOTH = "SMOOTH"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    FOLDER_SHP = "FOLDER_SHP"

    def __init__(self):
        super().__init__()
        self._keep = []

    def createInstance(self):
        return DataBoundaryAlgorithm()

    def name(self):
        return "data_boundary"

    def displayName(self):
        return "Data Boundary (mask polygon)"

    def group(self):
        return "Data Tools"

    def groupId(self):
        return "data_tools"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def shortHelpString(self):
        fields = "<br>".join(f"&nbsp;&nbsp;{n}: {d}" for n, _t, d in FIELDS)
        return (
            f"<b>Data Boundary (mask polygon) (version {TOOL_VERSION})</b><br><br>"
            "<b>PURPOSE</b><br>Draws the boundary polygon around a point, line "
            "or polygon layer - e.g. contours, survey points, spot heights or "
            "breaklines - to use as a mask, e.g. to clip a DEM interpolated "
            "from the data to the area the data covers (Raster > Extraction > "
            "Clip raster by mask layer). An interpolated surface is otherwise "
            "filled out to the rectangle or the convex hull, including bays "
            "and corners with no data.<br><br>"
            "<b>METHOD</b><br>"
            "Lines and polygon outlines are first densified so their vertices "
            "are at most half the maximum edge length apart; points are used "
            "as they are. Duplicate points are removed.<br>"
            "<i>Concave (maximum edge length)</i>, the default: the points are "
            "triangulated (Delaunay) and every triangle with an edge longer "
            "than the maximum edge length is removed; the outline of the "
            "triangles left is the boundary. Long outer edges span areas with "
            "no data, so the boundary follows the data. A good starting value "
            "is 1.5 to 3 times the contour (or point) spacing; larger values "
            "give a fuller outline, smaller ones a tighter outline that may "
            "split into separate areas or get holes. This is the edge-length "
            "form of a concave hull (Duckham et al. 2008, chi-shapes; compare "
            "alpha shapes, Edelsbrunner et al. 1983).<br>"
            "<i>Convex hull</i>: the smallest convex polygon around the "
            "points.<br>"
            "<i>Buffer and dissolve</i>: every feature buffered by the buffer "
            "distance and the buffers merged.<br>"
            "Then, in this order: holes filled (ticked by default); the "
            "boundary moved outward by the offset (0 = none; a few metres keeps "
            "the outer data just inside the mask); parts smaller than the "
            "minimum area dropped (the largest part is always kept); the "
            "outline smoothed (optional); and the parts written as one feature "
            "(ticked by default) or one feature per part. Merging writes the "
            "separate areas as one multi-part feature; it does not bridge the "
            "gap between them - use a larger maximum edge length for that."
            "<br><br>"
            "<b>PARAMETERS</b><br>"
            "The layer must be in a projected coordinate system (metres); "
            "with 'Selected features only' ticked, only those are used.<br><br>"
            "<b>OUTPUT</b><br>"
            f"Boundary: &lt;project number&gt;_"
            f"{Path(output_file_name('', DESCRIPTION, SHAPEFILE)).name} in the "
            "output folder, in the input layer's coordinate system; loaded "
            "with the study-area style (black dashed outline, no fill). An "
            "existing file of that name is replaced; a file open in QGIS "
            "cannot be replaced and is reported.<br><br>"
            f"<b>FIELDS</b><br>{fields}<br><br>"
            "<b>REFERENCES</b><br>"
            "Duckham, M., Kulik, L., Worboys, M. and Galton, A. (2008). "
            "Efficient generation of simple polygons for characterizing the "
            "shape of a set of points in the plane. Pattern Recognition 41, "
            "3224-3236.<br>"
            "Edelsbrunner, H., Kirkpatrick, D.G. and Seidel, R. (1983). On the "
            "shape of a set of points in the plane. IEEE Transactions on "
            "Information Theory 29, 551-559."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Data layer (points, lines or polygons, e.g. contours)",
                types=[QgsProcessing.SourceType.TypeVectorAnyGeometry],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD, "Boundary method", options=METHODS, defaultValue=CONCAVE
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_EDGE,
                "Maximum edge length (m; concave method, about 1.5 to 3 x the "
                "contour spacing)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=50.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUFFER,
                "Buffer distance (m; buffer method)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=10.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OFFSET,
                "Outward offset (m; 0 = none)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(self.FILL_HOLES, "Fill holes", True)
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MERGE, "Merge separate areas into one feature", True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_AREA,
                "Drop separate areas smaller than (m²; 0 = keep all)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(self.SMOOTH, "Smooth the outline", False)
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix, optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.FOLDER_SHP, "Output folder for vector layers (shapefiles)"
            )
        )

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        self._keep = []
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Choose the data layer.")
        crs = source.sourceCrs()
        if not crs.isValid() or crs.isGeographic():
            raise QgsProcessingException(
                "The data layer must be in a projected coordinate system (metres), "
                "so the distances are in metres. Reproject it first."
            )
        method = self.parameterAsEnum(parameters, self.METHOD, context)
        max_edge = self.parameterAsDouble(parameters, self.MAX_EDGE, context)
        buf = self.parameterAsDouble(parameters, self.BUFFER, context)
        offset = self.parameterAsDouble(parameters, self.OFFSET, context)
        fill = self.parameterAsBoolean(parameters, self.FILL_HOLES, context)
        merge = self.parameterAsBoolean(parameters, self.MERGE, context)
        min_area = self.parameterAsDouble(parameters, self.MIN_AREA, context)
        smooth = self.parameterAsBoolean(parameters, self.SMOOTH, context)
        if method == CONCAVE and max_edge <= 0:
            raise QgsProcessingException("Enter a maximum edge length greater than 0.")
        if method == BUFFER and buf <= 0:
            raise QgsProcessingException("Enter a buffer distance greater than 0.")
        folder = self.parameterAsFileOutput(parameters, self.FOLDER_SHP, context)
        if not folder:
            raise QgsProcessingException("Choose the output folder for vector layers.")
        os.makedirs(folder, exist_ok=True)
        pn = self.parameterAsString(parameters, self.PROJECT_NUMBER, context).strip()
        path = str(Path(folder) / output_file_name(pn, DESCRIPTION, SHAPEFILE))

        feedback.setProgressText("Reading the data layer")
        total = source.featureCount() or 1
        geoms, chunks = [], []
        spacing = max_edge / 2.0 if method == CONCAVE else None
        for i, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                return {}
            g = f.geometry()
            if g is None or g.isEmpty():
                continue
            if method == BUFFER:
                geoms.append(QgsGeometry(g))
            else:
                chunks.append(gm.feature_points(g, spacing))
            feedback.setProgress(40 * (i + 1) / total)
        n_points = 0
        param = {CONCAVE: max_edge, CONVEX: None, BUFFER: buf}[method]

        feedback.setProgressText(f"Drawing the boundary ({METHOD_NAMES[method]})")
        try:
            if method == BUFFER:
                if not geoms:
                    raise ValueError("The data layer has no geometries.")
                geom = gm.buffer_union(geoms, buf)
            else:
                xy = np.vstack(chunks) if chunks else np.empty((0, 2))
                n_points = len(xy)
                feedback.pushInfo(f"{n_points} data points (before duplicates).")
                polys = (
                    concave_polygons(xy, max_edge)
                    if method == CONCAVE
                    else convex_polygon(xy)
                )
                geom = gm.polygons_to_geometry(polys)
        except ValueError as e:
            raise QgsProcessingException(str(e)) from e
        if feedback.isCanceled():
            return {}
        feedback.setProgress(70)

        boundary = gm.finish(geom, fill, offset, min_area, smooth)
        if not boundary:
            raise QgsProcessingException("No boundary area was produced.")
        if len(boundary) > 1:
            feedback.pushInfo(
                f"The boundary has {len(boundary)} separate areas"
                + (
                    " (written as one multi-part feature)."
                    if merge
                    else " (one feature each)."
                )
            )
        features = [gm.merged(boundary)] if merge else boundary

        fields = QgsFields()
        types = {
            "Int": QMetaType.Type.Int,
            "Double": QMetaType.Type.Double,
            "String": QMetaType.Type.QString,
        }
        for n, t, _d in FIELDS:
            fields.append(QgsField(n, types[t], len=10 if t == "String" else 0))
        layer = QgsVectorLayer(
            f"MultiPolygon?crs={crs.authid()}", DESCRIPTION, "memory"
        )
        if not crs.authid():
            layer.setCrs(crs)
        pr = layer.dataProvider()
        pr.addAttributes(fields.toList())
        layer.updateFields()
        out = []
        for k, g in enumerate(features, 1):
            f = QgsFeature(layer.fields())
            f.setGeometry(g)
            area = g.area()
            f.setAttributes(
                [
                    k,
                    round(area, 2),
                    round(area / 10000.0, 4),
                    round(g.length(), 2),
                    METHOD_NAMES[method],
                    None if param is None else float(param),
                    float(offset),
                    n_points if method != BUFFER else None,
                ]
            )
            out.append(f)
            feedback.pushInfo(
                f"Part {k}: {area / 10000.0:.4f} ha ({area:.0f} m²), "
                f"perimeter {g.length():.0f} m."
            )
        pr.addFeatures(out)
        try:
            save_shapefile(layer, path, context.transformContext())
        except RuntimeError as e:
            raise QgsProcessingException(str(e)) from e
        feedback.pushInfo(f"Boundary: {path}")
        styler = _Styler()
        self._keep.append(styler)
        details = QgsProcessingContext.LayerDetails(
            Path(path).stem, context.project(), Path(path).stem
        )
        details.setPostProcessor(styler)
        context.addLayerToLoadOnCompletion(path, details)
        feedback.setProgress(100)
        return {self.FOLDER_SHP: folder}
