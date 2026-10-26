"""Processing Toolbox algorithm: D8 pointer + DEM (+ optional pour points)
in -> classified stream network (one reach layer carrying Strahler, Shreve,
Horton and Hack orders), nodes, main stems, a sectioned CSV and a Word
report out.

Outputs follow the Mayim output standard: a tick box per output,
shapefile, CSV and Word folders, files named <project number>_<description>,
layers loaded with the Mayim QML styles; a locked file is reported and the
other outputs are still written.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_stream_network_core.py).
"""

import os
from datetime import datetime
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsLineSymbol,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsRendererCategory,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    SHAPEFILE,
    WORD,
    OutputSpec,
    clean_name_part,
)
from mayim_tools.hydrology._common.d8_network import check_same_grid, read_raster

from .core import (
    PourPoint,
    RoutingSettings,
    StreamNetworkError,
    extract_stream_network,
)
from .export import NODE_FIELDS, REACH_FIELDS, write_sectioned_csv
from .report import write_report_docx

TOOL_VERSION = "0.2.0"
VELOCITY_OPTIONS = [
    "Manning (n and hydraulic radius, slope from the DEM)",
    "Constant velocity",
]
VELOCITY_KEYS = ["manning", "constant"]
_TYPES = {
    "s": QMetaType.Type.QString,
    "i": QMetaType.Type.Int,
    "d": QMetaType.Type.Double,
}
MAIN_STEM_FIELDS = [
    ("network_id", "s"),
    ("length_km", "d"),
    ("relief_m", "d"),
    ("slope_avg", "d"),
    ("slope_1085", "d"),
    ("sinuosity", "d"),
    ("tt_min", "d"),
    ("area_km2", "d"),
]
SNAP_FIELDS = [
    ("point_id", "s"),
    ("src_fid", "i"),
    ("x_in", "d"),
    ("y_in", "d"),
    ("snap_m", "d"),
    ("us_km2", "d"),
]
# Strahler symbology: one blue hue light -> dark, widening with order.
_BLUES = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]

# Selectable outputs (same file descriptions as the stream outputs of
# Catchment Delineation) and the Mayim QML style each layer is loaded with.
OUTPUT_SPECS = [
    OutputSpec(
        "OUTPUT_STREAMS", "Stream network (Strahler, Shreve, Horton, Hack orders)",
        "Stream Network", SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_NODES", "Stream nodes (channel heads, confluences, outlets)",
        "Stream Nodes", SHAPEFILE,
    ),
    OutputSpec("OUTPUT_MAIN_STEMS", "Main stems", "Main Stems", SHAPEFILE),
    OutputSpec(
        "OUTPUT_SNAPPED", "Snapped pour points", "Snapped Pour Points", SHAPEFILE,
        False,
    ),
    OutputSpec(
        "OUTPUT_CSV", "Stream network properties (CSV)", "Stream Network Properties",
        CSV,
    ),
    OutputSpec(
        "OUTPUT_DOCX", "Stream network report (Word)", "Stream Network Report", WORD
    ),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}
# QML style per layer (None: QGIS default symbology)
LAYER_STYLES = {
    "OUTPUT_STREAMS": "DRN_strahler_stream_order.qml",
    "OUTPUT_NODES": "DRN_stream_nodes.qml",
    "OUTPUT_MAIN_STEMS": None,
    "OUTPUT_SNAPPED": "DRN_outlets.qml",
}


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


def _fields(specs) -> QgsFields:
    fields = QgsFields()
    for spec in specs:
        fields.append(QgsField(spec[0], _TYPES[spec[1]]))
    return fields


def _advanced(param):
    param.setFlags(param.flags() | Qgis.ProcessingParameterFlag.Advanced)
    return param


class _StrahlerStyle(QgsProcessingLayerPostProcessorInterface):
    """Categorised Strahler symbology applied once the layer is loaded."""

    def __init__(self, max_order: int):
        super().__init__()
        self.max_order = max(1, int(max_order))

    def postProcessLayer(self, layer, context, feedback):
        try:
            cats = []
            for u in range(1, self.max_order + 1):
                color = _BLUES[min(u - 1 + max(0, 6 - self.max_order), 5)]
                symbol = QgsLineSymbol.createSimple(
                    {
                        "line_color": color,
                        "line_width": f"{0.25 + 0.3 * (u - 1):.2f}",
                        "capstyle": "round",
                        "joinstyle": "round",
                    }
                )
                cats.append(QgsRendererCategory(u, symbol, f"Order {u}"))
            layer.setRenderer(QgsCategorizedSymbolRenderer("strahler", cats))
            layer.triggerRepaint()
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Stream styling skipped: {e}")


class StreamNetworkAlgorithm(QgsProcessingAlgorithm):
    POINTER = "POINTER"
    DEM = "DEM"
    ESRI_STYLE = "ESRI_STYLE"
    POUR_POINTS = "POUR_POINTS"
    ID_FIELD = "ID_FIELD"
    SNAP_RADIUS = "SNAP_RADIUS"
    THRESHOLD = "THRESHOLD"
    RUN_DROP_TEST = "RUN_DROP_TEST"
    DROP_MIN = "DROP_MIN"
    DROP_MAX = "DROP_MAX"
    DROP_STEPS = "DROP_STEPS"
    MIN_NETWORK = "MIN_NETWORK"
    VELOCITY_METHOD = "VELOCITY_METHOD"
    VELOCITY = "VELOCITY"
    MANNING_N = "MANNING_N"
    HYD_RADIUS = "HYD_RADIUS"
    MIN_SLOPE = "MIN_SLOPE"
    MUSK_X = "MUSK_X"
    TIME_STEP = "TIME_STEP"
    OUTPUT_STREAMS = "OUTPUT_STREAMS"
    OUTPUT_NODES = "OUTPUT_NODES"
    OUTPUT_MAIN_STEMS = "OUTPUT_MAIN_STEMS"
    OUTPUT_SNAPPED = "OUTPUT_SNAPPED"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_DOCX = "OUTPUT_DOCX"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return StreamNetworkAlgorithm()

    def name(self):
        return "stream_network"

    def displayName(self):
        return "Stream Network"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        glossary = "\n".join(f"  {n}: {desc}" for n, _t, _u, desc in REACH_FIELDS)
        files = "\n".join(
            f"  {s.label}: <project number>_{clean_name_part(s.description)}"
            f"{EXTENSIONS[s.kind]}"
            for s in OUTPUT_SPECS
        )
        return (
            f"Stream Network (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tExtracts the stream network from a D8 pointer and its "
            "DEM and classifies every reach by Strahler, Shreve, Horton and "
            "Hack ordering in ONE line layer (one column per system). Also "
            "writes channel heads, confluences and outlets, main stems, a "
            "sectioned CSV with the network, Horton, reach and routing "
            "properties, and a Word report.\n"
            "\n"
            "EXTENT:\tWith pour points, the network upstream of them (each "
            "snapped to the highest flow accumulation within the radius; "
            "intermediate pour points also split reaches). Without pour "
            "points, every drainage network on the raster.\n"
            "\n"
            "THRESHOLD:\tA cell is a stream when its upstream area reaches "
            "the threshold (km²). The constant-drop test (Tarboton et al. "
            "1991) is run over a range of thresholds and reports the smallest "
            "one at which first-order streams drop as much as higher-order "
            "streams (|t| < 2). You keep your own threshold; the tool warns "
            "if it is below that value.\n"
            "\n"
            "ROUTING:\tEach reach gets a velocity (constant, or Manning with "
            "the given n and hydraulic radius and the reach slope, floored at "
            "the minimum slope), travel time, Muskingum K (= travel time), X, "
            "the number of sub-reaches for the time step and a stability "
            "check. These are first estimates for node-link models.\n"
            "\n"
            "INPUTS:\tPointer and DEM on the same grid in a projected CRS in "
            "metres; the DEM should be hydrologically conditioned. The Word "
            "report needs python-docx (and matplotlib for figures) in QGIS's "
            "Python.\n"
            "\n"
            "OUTPUTS:\tTick the outputs wanted and choose the shapefile, CSV "
            "and Word folders. Files are named <project number>_<description> "
            "and replace files of the same name; a file that is open in QGIS, "
            "Excel or Word cannot be replaced and is reported, and the other "
            "outputs are still written. The layers load with the Mayim styles "
            "(streams by Strahler order, nodes by type). The CSV is UTF-8 with "
            "a byte-order mark. Shapefile field names are at most 10 "
            "characters: links_to_out is written as links_to_o.\n"
            + files
            + "\n\nREACH ATTRIBUTES:\n"
            + glossary
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.POINTER, "D8 pointer raster")
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
                self.POUR_POINTS,
                "Pour points (optional - whole raster if empty)",
                types=[QgsProcessing.SourceType.TypeVectorPoint],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ID_FIELD,
                "Pour point name field (optional)",
                parentLayerParameterName=self.POUR_POINTS,
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
            QgsProcessingParameterNumber(
                self.THRESHOLD,
                "Stream threshold (upstream area, km²)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.1,
            )
        )
        add_network_parameters(self, include_min_network=True)

        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix, optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project name (report title page, optional)",
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

    def _read_pour_points(self, source, id_field, crs, context, feedback):
        if source is None:
            return []
        transform = None
        if source.sourceCrs() != crs:
            transform = QgsCoordinateTransform(
                source.sourceCrs(), crs, context.transformContext()
            )
        points = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            value = feature[id_field] if id_field else None
            if value is None or str(value) in ("", "NULL"):
                value = feature.id()
            pts = geom.asMultiPoint() if geom.isMultipart() else [geom.asPoint()]
            for k, p in enumerate(pts):
                name = str(value) if len(pts) == 1 else f"{value}_{k + 1}"
                points.append(PourPoint(name, p.x(), p.y(), int(feature.id())))
        if not points:
            feedback.pushWarning("The pour point layer has no usable points.")
        return points

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        pointer_layer = self.parameterAsRasterLayer(parameters, self.POINTER, context)
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        if pointer_layer is None or dem_layer is None:
            raise QgsProcessingException("Both a D8 pointer and a DEM are required.")
        crs = dem_layer.crs()
        if crs.isGeographic():
            raise QgsProcessingException(
                "The DEM is in a geographic CRS (degrees). Lengths, areas and "
                "slopes need a projected CRS in metres - reproject the DEM and "
                "pointer first."
            )
        if pointer_layer.crs() != crs:
            raise QgsProcessingException(
                "The D8 pointer and DEM have different CRSs - they must share "
                "the same grid."
            )
        esri = self.parameterAsBoolean(parameters, self.ESRI_STYLE, context)
        source = self.parameterAsSource(parameters, self.POUR_POINTS, context)
        id_field = self.parameterAsString(parameters, self.ID_FIELD, context)
        snap = self.parameterAsInt(parameters, self.SNAP_RADIUS, context)
        threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)
        drop_range, routing = read_network_settings(self, parameters, context)
        min_net = self.parameterAsDouble(parameters, self.MIN_NETWORK, context)
        project_number = (
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        ).strip()
        project_name = (
            self.parameterAsString(parameters, self.PROJECT_NAME, context) or ""
        ).strip()
        want = [
            spec.key
            for spec in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, spec.key, context)
        ]
        if not want:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {SPEC_BY_KEY[key].kind for key in want}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(
                    f"Choose the {label[0].lower() + label[1:]}."
                )
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {key: SPEC_BY_KEY[key].path(folders, project_number) for key in want}

        pour_points = self._read_pour_points(source, id_field, crs, context, feedback)
        feedback.pushInfo(
            f"{len(pour_points)} pour point(s)"
            if pour_points
            else "No pour points - extracting every network on the raster"
        )

        try:
            feedback.setProgressText("Reading rasters")
            pointer, p_gt, _, p_nodata = read_raster(pointer_layer.source())
            dem, d_gt, _, d_nodata = read_raster(dem_layer.source())
            check_same_grid(pointer.shape, p_gt, dem.shape, d_gt)

            def progress(fraction, message):
                feedback.setProgress(int(100 * fraction))
                feedback.setProgressText(message)

            result = extract_stream_network(
                pointer,
                p_nodata,
                dem,
                d_nodata,
                d_gt,
                pour_points=pour_points,
                esri_style=esri,
                snap_radius_cells=snap,
                threshold_km2=threshold,
                drop_test_range=drop_range,
                routing=routing,
                min_network_km2=min_net,
                progress=progress,
                is_canceled=feedback.isCanceled,
            )
        except StreamNetworkError as e:
            raise QgsProcessingException(str(e)) from e
        except Exception as e:
            raise QgsProcessingException(
                f"Stream network extraction failed: {e}"
            ) from e
        if feedback.isCanceled():
            return {}
        for message in result.warnings:
            feedback.pushWarning(message)

        # vector outputs: filled as memory layers, then written as shapefiles
        from mayim_tools.hydrology.catchment_delineation.qgis_outputs import (
            MemoryLayers,
            load_on_completion,
            save_shapefile,
        )

        memory = MemoryLayers()
        if self.OUTPUT_STREAMS in want:
            write_reaches(
                memory, parameters, self.OUTPUT_STREAMS, context, result, crs, []
            )
        if self.OUTPUT_NODES in want:
            write_nodes(memory, parameters, self.OUTPUT_NODES, context, result, crs)
        if self.OUTPUT_MAIN_STEMS in want:
            write_main_stems(
                memory, parameters, self.OUTPUT_MAIN_STEMS, context, result, crs
            )
        if self.OUTPUT_SNAPPED in want:
            if result.pour_points:
                self._write_snapped(memory, parameters, context, result, crs)
            else:
                feedback.pushWarning(
                    "Snapped pour points: no pour points were given - nothing to "
                    "write."
                )

        outputs = {name: folders[kind] for kind, (name, _) in FOLDER_PARAMS.items()
                   if kind in folders}  # fmt: skip
        written = []  # (label, path) for the report

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))
            feedback.pushInfo(f"Written: {path}")

        def failed(key, error):
            feedback.reportError(
                f"{SPEC_BY_KEY[key].label}: {Path(paths[key]).name} could not be "
                f"written ({error}). If it is open in QGIS, Excel or Word, close "
                "it and run again.",
                fatalError=False,
            )

        feedback.setProgressText("Writing layers")
        for key, qml in LAYER_STYLES.items():
            if key not in want:
                continue
            layer = memory.get(key)
            if layer is None:
                continue
            if layer.featureCount() == 0:
                feedback.pushWarning(f"{SPEC_BY_KEY[key].label}: nothing to write.")
                continue
            try:
                path = save_shapefile(layer, paths[key], context.transformContext())
            except Exception as e:
                failed(key, e)
                continue
            load_on_completion(
                context, path, Path(path).stem, qml, self._post_processors
            )
            done(key, path)

        context_info = {
            "tool": self.displayName(),
            "version": TOOL_VERSION,
            "software": _software_version(),
            "project_number": project_number,
            "project_name": project_name,
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dem_path": dem_layer.source(),
            "pointer_path": pointer_layer.source(),
            "encoding": "ESRI" if esri else "WhiteboxTools",
            "crs": f"{crs.authid()} - {crs.description()}",
        }
        if self.OUTPUT_CSV in want:
            path = paths[self.OUTPUT_CSV]
            try:
                write_sectioned_csv(path, result, context_info)
                done(self.OUTPUT_CSV, path)
            except Exception as e:
                failed(self.OUTPUT_CSV, e)

        if self.OUTPUT_DOCX in want:
            docx_path = paths[self.OUTPUT_DOCX]
            feedback.setProgressText("Writing report")
            context_info["outputs"] = list(written) + [("This report", docx_path)]
            try:
                for message in write_report_docx(docx_path, result, context_info):
                    feedback.pushInfo(message)
                done(self.OUTPUT_DOCX, docx_path)
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs the python-docx package, which is not "
                    "installed in QGIS's Python. Install it with: python -m pip "
                    "install python-docx. All other outputs were written."
                )
            except Exception as e:  # report failure must not lose the results
                failed(self.OUTPUT_DOCX, e)

        if not written:
            raise QgsProcessingException(
                "No output could be written - see the errors above."
            )
        for n in result.networks:
            feedback.pushInfo(
                f"{n['network_id']}: {n['area_km2']:.3f} km2, {n['n_reaches']} "
                f"reaches, Strahler order {n['max_strahler']}"
            )
        if result.drop_test_threshold_km2 is not None:
            feedback.pushInfo(
                f"Smallest threshold passing the drop test: "
                f"{result.drop_test_threshold_km2:.4g} km2"
            )
        return outputs

    # ------------------------------------------------------------------

    def _write_snapped(self, target, parameters, context, result, crs):
        fields = _fields(SNAP_FIELDS)
        sink, dest_id = target.parameterAsSink(
            parameters, self.OUTPUT_SNAPPED, context, fields, Qgis.WkbType.Point, crs
        )
        if sink is None:
            return None
        for p in result.pour_points:
            feat = QgsFeature(fields)
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(p["x"], p["y"])))
            for name, _t in SNAP_FIELDS:
                feat[name] = p.get(name)
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id


# ---------------------------------------------------------------------------
# Shared with Catchment Delineation: network parameters, settings and the
# reach / node / main-stem layer writers (so both tools write identical
# stream outputs).
# ---------------------------------------------------------------------------

RUN_DROP_TEST = "RUN_DROP_TEST"
DROP_MIN = "DROP_MIN"
DROP_MAX = "DROP_MAX"
DROP_STEPS = "DROP_STEPS"
MIN_NETWORK = "MIN_NETWORK"
VELOCITY_METHOD = "VELOCITY_METHOD"
VELOCITY = "VELOCITY"
MANNING_N = "MANNING_N"
HYD_RADIUS = "HYD_RADIUS"
MIN_SLOPE = "MIN_SLOPE"
MUSK_X = "MUSK_X"
TIME_STEP = "TIME_STEP"


def add_network_parameters(alg, include_min_network: bool = True) -> None:
    """Drop-test and routing parameters (routing ones under Advanced)."""
    alg.addParameter(
        QgsProcessingParameterBoolean(
            RUN_DROP_TEST, "Run the constant-drop test", defaultValue=True
        )
    )
    for key, label, kind, default, minv in (
        (DROP_MIN, "Drop test: smallest threshold (km²)", "d", 0.01, 0.0),
        (DROP_MAX, "Drop test: largest threshold (km²)", "d", 2.0, 0.0),
        (DROP_STEPS, "Drop test: number of thresholds (log-spaced)", "i", 12, 2),
    ):
        alg.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    key,
                    label,
                    type=(
                        QgsProcessingParameterNumber.Type.Integer
                        if kind == "i"
                        else QgsProcessingParameterNumber.Type.Double
                    ),
                    minValue=minv,
                    defaultValue=default,
                )
            )
        )
    if include_min_network:
        alg.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    MIN_NETWORK,
                    "Leave out networks smaller than (km²)",
                    type=QgsProcessingParameterNumber.Type.Double,
                    minValue=0.0,
                    defaultValue=0.0,
                )
            )
        )
    alg.addParameter(
        QgsProcessingParameterEnum(
            VELOCITY_METHOD,
            "Routing velocity method",
            options=VELOCITY_OPTIONS,
            defaultValue=0,
        )
    )
    for key, label, default, minv in (
        (MANNING_N, "Manning's n", 0.035, 0.001),
        (HYD_RADIUS, "Hydraulic radius (m)", 0.5, 0.001),
        (VELOCITY, "Constant velocity (m/s)", 1.0, 0.001),
        (MIN_SLOPE, "Minimum routing slope (m/m)", 0.0005, 0.0),
        (MUSK_X, "Muskingum X", 0.2, 0.0),
        (TIME_STEP, "Computational time step (min)", 5.0, 0.01),
    ):
        param = QgsProcessingParameterNumber(
            key,
            label,
            type=QgsProcessingParameterNumber.Type.Double,
            minValue=minv,
            maxValue=0.5 if key == MUSK_X else 1e12,
            defaultValue=default,
        )
        alg.addParameter(_advanced(param))


def read_network_settings(alg, parameters, context):
    """Returns (drop_test_range or None, RoutingSettings)."""
    drop_range = None
    if alg.parameterAsBoolean(parameters, RUN_DROP_TEST, context):
        drop_range = (
            alg.parameterAsDouble(parameters, DROP_MIN, context),
            alg.parameterAsDouble(parameters, DROP_MAX, context),
            alg.parameterAsInt(parameters, DROP_STEPS, context),
        )
    routing = RoutingSettings(
        method=VELOCITY_KEYS[alg.parameterAsEnum(parameters, VELOCITY_METHOD, context)],
        velocity_ms=alg.parameterAsDouble(parameters, VELOCITY, context),
        manning_n=alg.parameterAsDouble(parameters, MANNING_N, context),
        hydraulic_radius_m=alg.parameterAsDouble(parameters, HYD_RADIUS, context),
        muskingum_x=alg.parameterAsDouble(parameters, MUSK_X, context),
        time_step_min=alg.parameterAsDouble(parameters, TIME_STEP, context),
        min_slope=alg.parameterAsDouble(parameters, MIN_SLOPE, context),
    )
    return drop_range, routing


def write_reaches(alg, parameters, key, context, result, crs, keep_alive):
    """Reach layer with every REACH_FIELDS column, styled by Strahler order.
    Returns the destination id, or None when the output is not requested."""
    fields = _fields(REACH_FIELDS)
    sink, dest_id = alg.parameterAsSink(
        parameters, key, context, fields, Qgis.WkbType.LineString, crs
    )
    if sink is None:
        return None
    for r in result.reaches:
        if len(r["coords"]) < 2:
            continue
        feat = QgsFeature(fields)
        feat.setGeometry(
            QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in r["coords"]])
        )
        for name, *_ in REACH_FIELDS:
            feat[name] = r.get(name)
        sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
    if dest_id and context.willLoadLayerOnCompletion(dest_id):
        max_order = max((r["strahler"] for r in result.reaches), default=1)
        processor = _StrahlerStyle(max_order)
        context.layerToLoadOnCompletionDetails(dest_id).setPostProcessor(processor)
        keep_alive.append(processor)
    return dest_id


def write_nodes(alg, parameters, key, context, result, crs):
    fields = _fields(NODE_FIELDS)
    sink, dest_id = alg.parameterAsSink(
        parameters, key, context, fields, Qgis.WkbType.Point, crs
    )
    if sink is None:
        return None
    for n in result.nodes:
        feat = QgsFeature(fields)
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(n["x"], n["y"])))
        for name, *_ in NODE_FIELDS:
            feat[name] = n.get(name)
        sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
    return dest_id


def write_main_stems(alg, parameters, key, context, result, crs):
    fields = _fields(MAIN_STEM_FIELDS)
    sink, dest_id = alg.parameterAsSink(
        parameters, key, context, fields, Qgis.WkbType.LineString, crs
    )
    if sink is None:
        return None
    nets = {n["network_id"]: n for n in result.networks}
    for nid, coords in result.main_stems.items():
        if len(coords) < 2 or nid not in nets:
            continue
        n = nets[nid]
        feat = QgsFeature(fields)
        feat.setGeometry(
            QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in coords])
        )
        feat["network_id"] = nid
        feat["length_km"] = n["main_length_km"]
        feat["relief_m"] = n["main_relief_m"]
        feat["slope_avg"] = n["main_slope_avg"]
        feat["slope_1085"] = n["main_slope_1085"]
        feat["sinuosity"] = n["main_sinuosity"]
        feat["tt_min"] = n["main_tt_min"]
        feat["area_km2"] = n["area_km2"]
        sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
    return dest_id
