"""Processing Toolbox algorithm: D8 pointer + DEM + pour point(s)/line(s)
in -> catchment polygon(s), catchment DEM, longest flow path(s),
snapped outlet(s) and a parameters report out.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_catchment_delineation_core.py).
"""

from datetime import datetime
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from .core import (
    CatchmentError,
    Outlet,
    check_same_grid,
    delineate_catchments,
    read_raster,
    rings_to_multipolygon_wkt,
    write_raster,
)
from .export import write_hypsometric_csv, write_parameters_csv
from .report import write_report_docx

TOOL_VERSION = "0.3.0"

MODE_OPTIONS = [
    "Total (full catchment per outlet; nested catchments overlap)",
    "Incremental (non-overlapping sub-catchments)",
]
MODE_KEYS = ["total", "incremental"]

_S = QMetaType.Type.QString
_I = QMetaType.Type.Int
_D = QMetaType.Type.Double

# (field name, type, description) - the description feeds the help text.
CATCHMENT_FIELDS = [
    ("outlet_id", _S, "outlet name (ID field, or feature id)"),
    ("src_fid", _I, "source feature id"),
    ("out_type", _S, "point or line"),
    ("mode", _S, "total or incremental"),
    ("ds_ids", _S, "outlet(s) this catchment drains into"),
    ("x_in", _D, "input point X (points only)"),
    ("y_in", _D, "input point Y (points only)"),
    ("x_out", _D, "outlet cell centre X (snapped)"),
    ("y_out", _D, "outlet cell centre Y (snapped)"),
    ("snap_m", _D, "snap distance, m (points only)"),
    ("n_outcells", _I, "number of outlet cells"),
    ("area_km2", _D, "contributing area, km2"),
    ("area_ha", _D, "contributing area, ha"),
    ("hole_ha", _D, "interior holes filled, ha"),
    ("poly_km2", _D, "polygon area incl. filled holes, km2"),
    ("perim_km", _D, "polygon perimeter (cell outline), km"),
    ("n_parts", _I, "polygon parts (diagonal contacts split)"),
    ("z_outlet", _D, "outlet elevation, m"),
    ("z_min", _D, "minimum elevation, m"),
    ("z_max", _D, "maximum elevation, m"),
    ("z_mean", _D, "mean elevation, m"),
    ("relief_m", _D, "z_max - z_min, m"),
    ("cx", _D, "centroid X"),
    ("cy", _D, "centroid Y"),
    ("lfp_km", _D, "longest flow path length, km"),
    ("lca_km", _D, "outlet to point on LFP nearest the centroid, km"),
    ("s_avg", _D, "LFP average slope (H/L), m/m"),
    ("s_1085", _D, "LFP 10-85 slope, m/m"),
    ("s_ea", _D, "LFP equal-area slope, m/m"),
    ("s_mean_pc", _D, "mean catchment slope (Horn), %"),
    ("form_f", _D, "form factor A/L^2"),
    ("circ_r", _D, "circularity ratio 4piA/P^2"),
    ("elong_r", _D, "elongation ratio 2sqrt(A/pi)/L"),
    ("gravel_kc", _D, "Gravelius compactness P/(2sqrt(piA))"),
    ("hyps_int", _D, "hypsometric integral"),
    ("strm_km", _D, "stream length above threshold, km"),
    ("dd_kmkm2", _D, "drainage density, km/km2"),
    ("lb_km", _D, "basin length (outlet to farthest point), km"),
    ("relief_r", _D, "relief ratio (z_max - z_outlet) / Lb"),
    ("melton_r", _D, "Melton ratio H / sqrt(A)"),
    ("rugged_n", _D, "ruggedness number H x Dd"),
    ("sinuosity", _D, "LFP length / straight-line distance"),
    ("strm_ord", _I, "highest Strahler stream order"),
    ("n_strm", _I, "number of stream segments"),
    ("bif_r", _D, "mean bifurcation ratio"),
    ("strm_freq", _D, "stream frequency, segments/km2"),
    ("lo_km", _D, "length of overland flow 1/(2 Dd), km"),
    ("c_maint", _D, "constant of channel maintenance 1/Dd, km2/km"),
    ("tc_kirp_mn", _D, "Tc Kirpich (L, s_avg), min"),
    ("tc_usbr_mn", _D, "Tc USBR / SANRAL (L, s_1085), min"),
    ("tc_bw_mn", _D, "Tc Bransby-Williams (L, A, s_ea), min"),
]


def _describe_destination(dest) -> str:
    """Readable location of a Processing output."""
    text = str(dest or "")
    if text.startswith("memory:") or "TEMPORARY_OUTPUT" in text or not text:
        return "Temporary layer (not saved)"
    return text


def _fields(specs) -> QgsFields:
    fields = QgsFields()
    for name, qtype, _ in specs:
        fields.append(QgsField(name, qtype))
    return fields


class CatchmentDelineationAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    POINTER = "POINTER"
    ESRI_STYLE = "ESRI_STYLE"
    OUTLETS = "OUTLETS"
    ID_FIELD = "ID_FIELD"
    SNAP_RADIUS = "SNAP_RADIUS"
    MODE = "MODE"
    STREAM_THRESHOLD = "STREAM_THRESHOLD"
    OUTPUT_CATCHMENTS = "OUTPUT_CATCHMENTS"
    OUTPUT_DEM = "OUTPUT_DEM"
    OUTPUT_FLOWPATHS = "OUTPUT_FLOWPATHS"
    OUTPUT_OUTLETS = "OUTPUT_OUTLETS"
    OUTPUT_STREAMS = "OUTPUT_STREAMS"
    OUTPUT_CENTROIDS = "OUTPUT_CENTROIDS"
    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_HYPSO = "OUTPUT_HYPSO"
    OUTPUT_DOCX = "OUTPUT_DOCX"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return CatchmentDelineationAlgorithm()

    def name(self):
        return "catchment_delineation"

    def displayName(self):
        return "Catchment Delineation"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        glossary = "\n".join(f"  {n}: {d}" for n, _, d in CATCHMENT_FIELDS)
        return (
            f"Catchment Delineation (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tDelineates the catchment draining to each pour "
            "point or pour line, and writes the catchment polygon(s) "
            "(geometry repaired), a catchment DEM, the longest flow "
            "path(s), the snapped outlet(s) and a set of hydrological "
            "modelling parameters.\n"
            "\n"
            "INPUTS:\tA D8 pointer raster (e.g. from D8 Flow Direction) "
            "and the DEM it was built from - both on exactly the same "
            "grid, in a projected CRS in metres. The DEM should be "
            "hydrologically conditioned. Pour points are snapped to the "
            "highest flow accumulation within the search radius. For "
            "pour lines, every cell the line crosses is an outlet cell "
            "(no snapping), and the catchment is everything draining to "
            "any point on the line.\n"
            "\n"
            "MODES:\tTotal gives each outlet its full catchment, so "
            "nested catchments overlap. Incremental splits the area at "
            "every outlet into non-overlapping sub-catchments (useful "
            "for HEC-HMS); ds_ids names the outlet each one drains into. "
            "The catchment DEM covers all catchments together.\n"
            "\n"
            "GEOMETRY:\tPolygons follow cell edges. Interior holes (cells "
            "inside the outline that do not drain to the outlet) are "
            "filled and reported as hole_ha. A large share of holes means "
            "the DEM needs conditioning. In incremental mode, a sub-"
            "catchment wholly enclosed by another stays a real hole, so "
            "polygons never overlap. Cells that touch only at a "
            "corner become separate parts of a valid MultiPolygon. Each "
            "polygon is passed through makeValid and checked. Perimeter "
            "is the stepped cell outline, so it runs longer than a "
            "smooth boundary, and circularity/Gravelius reflect that.\n"
            "\n"
            "TIME OF CONCENTRATION:\tKirpich (metric, with the average "
            "slope), USBR as used in the SANRAL Drainage Manual (defined "
            "watercourses, with the 10-85 slope) and Bransby-Williams "
            "(ARR form, with the equal-area slope). All three use the "
            "longest D8 flow path with no split between overland and "
            "channel flow. Check that each method suits the catchment "
            "before using the value.\n"
            "\n"
            "STREAM NETWORK:\tOptional line layer of the stream cells "
            "(upstream area at least the stream threshold), split into links "
            "at every junction. Each link carries its Strahler order, length, "
            "upstream area at its downstream end, start and end elevations, "
            "slope and the id of the link it flows into (ds_link), so the "
            "network can be traced. In total mode, nested catchments repeat "
            "their shared streams (select by outlet_id).\n"
            "\n"
            "CENTROIDS:\tOptional point layer of each catchment's area "
            "centroid, with its elevation, whether it falls inside the "
            "catchment (inside = 0 for strongly curved shapes), and the point "
            "on the longest flow path nearest to it (lca_x, lca_y) that "
            "defines Lca.\n"
            "\n"
            "REPORT:\tOptionally writes a Word (.docx) annexure: summary, "
            "tool and method, inputs, parameter definitions with sources, "
            "and for each catchment a description, parameter tables, slope, "
            "aspect and stream-order tables, figures (plan, longitudinal "
            "profile, hypsometric curve, elevation histogram), modelling "
            "considerations and references. It needs python-docx (and "
            "matplotlib for figures) in QGIS's Python.\n"
            "\n"
            "ATTRIBUTES:\n" + glossary
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.POINTER, "D8 pointer raster (flow direction)"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DEM, "DEM (same grid as the pointer; conditioned)"
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ESRI_STYLE,
                "Pointer uses ESRI encoding (default: WhiteboxTools)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.OUTLETS,
                "Pour point(s) or pour line(s)",
                types=[
                    QgsProcessing.SourceType.TypeVectorPoint,
                    QgsProcessing.SourceType.TypeVectorLine,
                ],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ID_FIELD,
                "Outlet name field (optional)",
                parentLayerParameterName=self.OUTLETS,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SNAP_RADIUS,
                "Pour point snap radius (cells; 0 = no snapping)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=0,
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE, "Catchment mode", options=MODE_OPTIONS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.STREAM_THRESHOLD,
                "Stream threshold for drainage density (upstream area, km2)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.1,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_CATCHMENTS,
                "Catchments",
                type=QgsProcessing.SourceType.TypeVectorPolygon,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(self.OUTPUT_DEM, "Catchment DEM")
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_FLOWPATHS,
                "Longest flow paths",
                type=QgsProcessing.SourceType.TypeVectorLine,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_OUTLETS,
                "Snapped outlets",
                type=QgsProcessing.SourceType.TypeVectorPoint,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_STREAMS,
                "Stream network",
                type=QgsProcessing.SourceType.TypeVectorLine,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_CENTROIDS,
                "Catchment centroids",
                type=QgsProcessing.SourceType.TypeVectorPoint,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_REPORT,
                "Catchment parameters report (CSV)",
                fileFilter="CSV files (*.csv)",
                optional=True,
                createByDefault=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_HYPSO,
                "Hypsometric curves (CSV)",
                fileFilter="CSV files (*.csv)",
                optional=True,
                createByDefault=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_DOCX,
                "Catchment report (Word annexure)",
                fileFilter="Word documents (*.docx)",
                optional=True,
                createByDefault=False,
            )
        )

    # ------------------------------------------------------------------

    def _read_outlets(self, source, id_field, target_crs, context, feedback):
        transform = None
        if source.sourceCrs() != target_crs:
            transform = QgsCoordinateTransform(
                source.sourceCrs(), target_crs, context.transformContext()
            )
        outlets = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            value = feature[id_field] if id_field else None
            if value is None or str(value) in ("", "NULL"):
                value = feature.id()
            name = str(value)
            gtype = geom.type()
            if gtype == Qgis.GeometryType.Point:
                pts = geom.asMultiPoint() if geom.isMultipart() else [geom.asPoint()]
                parts = [[(p.x(), p.y())] for p in pts]
                outlets.append(Outlet(name, "point", parts, int(feature.id())))
            elif gtype == Qgis.GeometryType.Line:
                geom.convertToStraightSegment()
                lines = (
                    geom.asMultiPolyline()
                    if geom.isMultipart()
                    else [geom.asPolyline()]
                )
                parts = [[(p.x(), p.y()) for p in line] for line in lines if line]
                outlets.append(Outlet(name, "line", parts, int(feature.id())))
            else:
                feedback.pushWarning(
                    f"Feature {feature.id()} skipped: not a point or line."
                )
        return outlets

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        pointer_layer = self.parameterAsRasterLayer(parameters, self.POINTER, context)
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        if pointer_layer is None or dem_layer is None:
            raise QgsProcessingException("Both a D8 pointer and a DEM are required.")
        crs = dem_layer.crs()
        if crs.isGeographic():
            raise QgsProcessingException(
                "The DEM is in a geographic CRS (degrees). Areas, lengths and "
                "slopes need a projected CRS in metres - reproject the DEM and "
                "pointer first."
            )
        if pointer_layer.crs() != crs:
            raise QgsProcessingException(
                "The D8 pointer and DEM have different CRSs - they must share "
                "the same grid."
            )

        esri_style = self.parameterAsBoolean(parameters, self.ESRI_STYLE, context)
        source = self.parameterAsSource(parameters, self.OUTLETS, context)
        if source is None:
            raise QgsProcessingException("No pour point/line layer provided.")
        id_field = self.parameterAsString(parameters, self.ID_FIELD, context)
        snap_radius = self.parameterAsInt(parameters, self.SNAP_RADIUS, context)
        mode = MODE_KEYS[self.parameterAsEnum(parameters, self.MODE, context)]
        threshold = self.parameterAsDouble(parameters, self.STREAM_THRESHOLD, context)
        dem_out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT_DEM, context)
        report_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_REPORT, context
        )
        hypso_path = self.parameterAsFileOutput(parameters, self.OUTPUT_HYPSO, context)
        docx_path = self.parameterAsFileOutput(parameters, self.OUTPUT_DOCX, context)

        outlets = self._read_outlets(source, id_field, crs, context, feedback)
        if not outlets:
            raise QgsProcessingException("The pour point/line layer has no features.")
        feedback.pushInfo(f"{len(outlets)} outlet(s), mode: {mode}")

        try:
            feedback.setProgressText("Reading rasters")
            pointer, p_gt, _, p_nodata = read_raster(pointer_layer.source())
            dem, d_gt, projection, d_nodata = read_raster(dem_layer.source())
            check_same_grid(pointer.shape, p_gt, dem.shape, d_gt)

            def progress(fraction, message):
                feedback.setProgress(int(100 * fraction))
                feedback.setProgressText(message)

            result = delineate_catchments(
                pointer,
                p_nodata,
                dem,
                d_nodata,
                d_gt,
                outlets,
                esri_style=esri_style,
                snap_radius_cells=snap_radius,
                mode=mode,
                stream_threshold_km2=threshold,
                progress=progress,
                is_canceled=feedback.isCanceled,
            )
        except CatchmentError as e:
            raise QgsProcessingException(str(e)) from e
        except Exception as e:
            raise QgsProcessingException(f"Catchment delineation failed: {e}") from e

        if feedback.isCanceled():
            return {}
        for message in result.warnings:
            feedback.pushWarning(message)
        if not result.catchments:
            raise QgsProcessingException(
                "No catchment could be delineated - check that the outlets lie "
                "on the pointer/DEM grid."
            )

        outputs = {}
        outputs[self.OUTPUT_CATCHMENTS] = self._write_catchments(
            parameters, context, feedback, result, crs
        )
        write_raster(
            dem_out_path,
            result.dem_array,
            result.dem_geotransform,
            projection,
            result.dem_nodata,
        )
        outputs[self.OUTPUT_DEM] = dem_out_path

        flow_id = self._write_flow_paths(parameters, context, result, crs)
        if flow_id is not None:
            outputs[self.OUTPUT_FLOWPATHS] = flow_id
        outlet_id = self._write_outlets(parameters, context, result, crs)
        if outlet_id is not None:
            outputs[self.OUTPUT_OUTLETS] = outlet_id
        streams_id = self._write_streams(parameters, context, result, crs)
        if streams_id is not None:
            outputs[self.OUTPUT_STREAMS] = streams_id
        centroids_id = self._write_centroids(parameters, context, result, crs)
        if centroids_id is not None:
            outputs[self.OUTPUT_CENTROIDS] = centroids_id
        if report_path:
            write_parameters_csv(report_path, result.catchments)
            outputs[self.OUTPUT_REPORT] = report_path
        if hypso_path:
            write_hypsometric_csv(hypso_path, result.catchments)
            outputs[self.OUTPUT_HYPSO] = hypso_path

        if docx_path:
            feedback.setProgressText("Writing report")
            context_info = {
                "tool": self.displayName(),
                "version": TOOL_VERSION,
                "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "dem": {
                    "path": dem_layer.source(),
                    "crs": f"{crs.authid()} - {crs.description()}",
                    "cell_x": abs(d_gt[1]),
                    "cell_y": abs(d_gt[5]),
                    "cols": dem.shape[1],
                    "rows": dem.shape[0],
                },
                "pointer": {
                    "path": pointer_layer.source(),
                    "encoding": "ESRI" if esri_style else "WhiteboxTools",
                },
                "outlets": {
                    "layer": source.sourceName(),
                    "n_features": len(outlets),
                    "id_field": id_field,
                },
                "snap_radius_cells": snap_radius,
                "mode": mode,
                "stream_threshold_km2": threshold,
                "outputs": [
                    (label, _describe_destination(outputs.get(key)))
                    for key, label in (
                        (self.OUTPUT_CATCHMENTS, "Catchments"),
                        (self.OUTPUT_DEM, "Catchment DEM"),
                        (self.OUTPUT_FLOWPATHS, "Longest flow paths"),
                        (self.OUTPUT_OUTLETS, "Snapped outlets"),
                        (self.OUTPUT_STREAMS, "Stream network"),
                        (self.OUTPUT_CENTROIDS, "Catchment centroids"),
                        (self.OUTPUT_REPORT, "Catchment parameters (CSV)"),
                        (self.OUTPUT_HYPSO, "Hypsometric curves (CSV)"),
                    )
                    if outputs.get(key)
                ]
                + [("This report", docx_path)],
                "warnings": result.warnings,
            }
            try:
                write_report_docx(docx_path, result.catchments, context_info)
                outputs[self.OUTPUT_DOCX] = docx_path
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs the python-docx package, which is not "
                    "installed in QGIS's Python. Install it from the OSGeo4W Shell "
                    "with: python -m pip install python-docx. All other outputs "
                    "were written."
                )
            except Exception as e:  # report failure must not lose the results
                feedback.pushWarning(f"The Word report could not be written: {e}")

        for c in result.catchments:
            a = c.attributes
            feedback.pushInfo(
                f"{c.outlet_id}: {a['area_km2']:.4f} km2, LFP {a['lfp_km']:.3f} km"
            )
        return outputs

    # ------------------------------------------------------------------

    def _write_catchments(self, parameters, context, feedback, result, crs):
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_CATCHMENTS,
            context,
            _fields(CATCHMENT_FIELDS),
            Qgis.WkbType.MultiPolygon,
            crs,
        )
        if sink is None:
            raise QgsProcessingException("Could not create the catchments output.")
        for c in result.catchments:
            geom = QgsGeometry.fromWkt(rings_to_multipolygon_wkt(c.polygons))
            geom = geom.makeValid()
            if QgsWkbTypes.flatType(geom.wkbType()) not in (
                Qgis.WkbType.Polygon,
                Qgis.WkbType.MultiPolygon,
            ):
                geom.convertGeometryCollectionToSubclass(Qgis.GeometryType.Polygon)
            geom.convertToMultiType()
            if geom.isEmpty() or not geom.isGeosValid():
                feedback.pushWarning(
                    f"Catchment '{c.outlet_id}': polygon could not be repaired "
                    "to a valid geometry."
                )
            feat = QgsFeature(_fields(CATCHMENT_FIELDS))
            feat.setGeometry(geom)
            for name, _, _ in CATCHMENT_FIELDS:
                feat[name] = c.attributes.get(name)
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_flow_paths(self, parameters, context, result, crs):
        specs = [
            ("outlet_id", _S, ""),
            ("lfp_km", _D, ""),
            ("z_start", _D, ""),
            ("z_outlet", _D, ""),
            ("s_avg", _D, ""),
            ("s_1085", _D, ""),
            ("s_ea", _D, ""),
        ]
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_FLOWPATHS,
            context,
            _fields(specs),
            Qgis.WkbType.LineString,
            crs,
        )
        if sink is None:
            return None
        for c in result.catchments:
            if len(c.flow_path) < 2:
                continue
            a = c.attributes
            feat = QgsFeature(_fields(specs))
            feat.setGeometry(
                QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in c.flow_path])
            )
            z_start = None
            if a["z_outlet"] is not None and a["s_avg"] is not None:
                z_start = round(a["z_outlet"] + a["s_avg"] * a["lfp_km"] * 1000.0, 3)
            feat["outlet_id"] = c.outlet_id
            feat["lfp_km"] = a["lfp_km"]
            feat["z_start"] = z_start
            feat["z_outlet"] = a["z_outlet"]
            feat["s_avg"] = a["s_avg"]
            feat["s_1085"] = a["s_1085"]
            feat["s_ea"] = a["s_ea"]
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_outlets(self, parameters, context, result, crs):
        specs = [
            ("outlet_id", _S, ""),
            ("out_type", _S, ""),
            ("x_in", _D, ""),
            ("y_in", _D, ""),
            ("snap_m", _D, ""),
            ("area_km2", _D, ""),
        ]
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_OUTLETS,
            context,
            _fields(specs),
            Qgis.WkbType.Point,
            crs,
        )
        if sink is None:
            return None
        for c in result.catchments:
            if c.outlet_xy is None:
                continue
            feat = QgsFeature(_fields(specs))
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*c.outlet_xy)))
            for name, _, _ in specs:
                feat[name] = c.attributes.get(name)
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_streams(self, parameters, context, result, crs):
        specs = [
            ("outlet_id", _S, ""),
            ("link_id", _I, ""),
            ("ds_link", _I, ""),
            ("strahler", _I, ""),
            ("length_m", _D, ""),
            ("us_km2", _D, ""),
            ("z_start", _D, ""),
            ("z_end", _D, ""),
            ("slope", _D, ""),
        ]
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_STREAMS,
            context,
            _fields(specs),
            Qgis.WkbType.LineString,
            crs,
        )
        if sink is None:
            return None
        for c in result.catchments:
            for link in c.detail.get("stream_links", []):
                if len(link["coords"]) < 2:
                    continue
                feat = QgsFeature(_fields(specs))
                feat.setGeometry(
                    QgsGeometry.fromPolylineXY(
                        [QgsPointXY(x, y) for x, y in link["coords"]]
                    )
                )
                feat["outlet_id"] = c.outlet_id
                for name, _, _ in specs[1:]:
                    feat[name] = link.get(name)
                sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_centroids(self, parameters, context, result, crs):
        specs = [
            ("outlet_id", _S, ""),
            ("cx", _D, ""),
            ("cy", _D, ""),
            ("z_m", _D, ""),
            ("inside", _I, ""),
            ("area_km2", _D, ""),
            ("lca_km", _D, ""),
            ("lca_x", _D, ""),
            ("lca_y", _D, ""),
        ]
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_CENTROIDS,
            context,
            _fields(specs),
            Qgis.WkbType.Point,
            crs,
        )
        if sink is None:
            return None
        for c in result.catchments:
            a = c.attributes
            lca = c.detail.get("lca_point") or (None, None)
            feat = QgsFeature(_fields(specs))
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(a["cx"], a["cy"])))
            feat["outlet_id"] = c.outlet_id
            feat["cx"] = a["cx"]
            feat["cy"] = a["cy"]
            feat["z_m"] = c.detail.get("centroid_z")
            feat["inside"] = int(bool(c.detail.get("centroid_inside")))
            feat["area_km2"] = a["area_km2"]
            feat["lca_km"] = a["lca_km"]
            feat["lca_x"] = lca[0]
            feat["lca_y"] = lca[1]
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id
