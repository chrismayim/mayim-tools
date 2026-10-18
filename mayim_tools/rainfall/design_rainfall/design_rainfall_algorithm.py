"""Processing Toolbox algorithm: point(s) of interest in -> report-ready
design rainfall output.

Point of interest may be supplied two ways (Phase 1):
    - a single point, either typed as coordinates or picked by clicking
      the map canvas (QgsProcessingParameterPoint's built-in widget), or
    - an existing vector point layer (loaded in the project, or browsed
      from disk) - every feature in it is processed and reported
      together.

Exactly one of the two must be supplied. A future phase can add a
polygon/area parameter alongside these two without changing this
structure - the per-site estimation loop below is already written to
handle an arbitrary list of (label, lat, lon) tuples, however they
were sourced.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
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
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterPoint,
    QgsProcessingParameterString,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from .core import (
    DAILY_DURATIONS,
    RETURN_PERIODS,
    SHORT_DURATIONS,
    DesignRainfallEngine,
    PointEstimate,
)
from .report import REPORT_TITLE, write_csv, write_csv_multi, write_report_docx

TOOL_VERSION = "1.1"

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

STYLES_DIR = Path(__file__).resolve().parent / "styles"


class _StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Loads a saved QML style (symbology + labeling) onto an
    output layer once Processing has finished loading it into the
    project."""

    def __init__(self, style_path: Path):
        super().__init__()
        self.style_path = str(style_path)

    def postProcessLayer(self, layer, context, feedback):
        layer.loadNamedStyle(self.style_path)
        layer.triggerRepaint()


def _qmetatype_for_dtype(dtype):
    """Maps a pandas/numpy dtype to the closest QMetaType for QgsField."""
    kind = getattr(dtype, "kind", "O")
    if kind in ("i", "u"):
        return QMetaType.Type.LongLong
    if kind == "f":
        return QMetaType.Type.Double
    return QMetaType.Type.QString


def _fields_from_dataframe(df, leading=()):
    """Builds a QgsFields from a (non-geometry) DataFrame's columns,
    optionally prefixed with extra (name, QMetaType) leading fields."""
    fields = QgsFields()
    for name, qtype in leading:
        fields.append(QgsField(name, qtype))
    for col in df.columns:
        if col == "geometry":
            continue
        fields.append(QgsField(str(col)[:80], _qmetatype_for_dtype(df[col].dtype)))
    return fields


def _software_version() -> str:
    """'Mayim Tools <version>, QGIS <version>' for the report title page."""
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


def _location(dest_id: str) -> str:
    """Report location of a sink output: its file path, or 'Temporary
    layer' for a memory layer."""
    path = str(dest_id).split("|")[0]
    return path if os.path.exists(path) else "Temporary layer (QGIS project)"


class DesignRainfallPointAlgorithm(QgsProcessingAlgorithm):

    POINT = "POINT"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    DURATIONS = "DURATIONS"
    RETURN_PERIODS = "RETURN_PERIODS"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_DOCX = "OUTPUT_DOCX"
    OUTPUT_LAYER = "OUTPUT_LAYER"
    OUTPUT_QUERY_POINTS = "OUTPUT_QUERY_POINTS"
    OUTPUT_STATIONS = "OUTPUT_STATIONS"
    NUM_STATIONS = "NUM_STATIONS"
    REPORT_OSM = "REPORT_OSM"

    # Duration choices shown in the UI, mapped to internal (family, value)
    _DURATION_CHOICES = (
        [
            (f"{d} min" if d < 60 else f"{d // 60} h", ("short", d))
            for d in SHORT_DURATIONS
        ]
        + [("1 day", ("daily", 1))]
        + [(f"{d} day", ("daily", d)) for d in DAILY_DURATIONS[1:]]
    )

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return DesignRainfallPointAlgorithm()

    def name(self):
        return "design_rainfall_point"

    def displayName(self):
        return "Design Rainfall Estimation (South Africa)"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        return (
            f"<b>{REPORT_TITLE} (version {TOOL_VERSION})</b><br><br>"
            "<b>PURPOSE</b><br>"
            "Estimates design rainfall depths (median and 90% prediction "
            "interval bounds) for durations of 5 min to 7 days and return "
            "periods of 2 to 200 years at any point in South Africa.<br><br>"
            "<b>METHOD</b><br>"
            "Each site is matched to the nearest point of the national 1 "
            "arc-minute grid. The index value (L1) of each duration is "
            "derived from the grid's 1-day L1 (short durations via the 24 h "
            "ratio and cluster regressions; 2 to 7 days via the regional "
            "power-law regressions) and scaled by the 1-day growth curve of "
            "the site's cluster.<br><br>"
            "<b>BACKGROUND</b><br>"
            "Regional L-moment method of Smithers &amp; Schulze (2002), WRC "
            "Report K5/1060, rebuilt from the rainfall3 dataset of the "
            "'Design Rainfall Estimation in South Africa' program. The "
            "nearest daily rainfall stations are listed for reference only "
            "and are NOT used in the grid-based calculation.<br><br>"
            "<b>PARAMETERS</b><br>"
            "<i>Point of interest</i> - click the map or type coordinates; "
            "OR <i>point layer</i> - every feature is processed (optional "
            "site name field).<br>"
            "<i>Project number / Project / site name</i> - optional, shown "
            "on the report's title page.<br>"
            "<i>Durations / Return periods</i> - the combinations to "
            "estimate.<br>"
            "<i>Output CSV</i> - duration x return period table (depth, "
            "lower, upper), with the nearest stations appended.<br>"
            "<i>Output Word report</i> - annexure in the Mayim report "
            "standard: title page, summary, method, inputs, parameters, "
            "depth tables, a "
            "layout map per site (point of interest, snapped grid point and "
            "nearest stations, drawn with the tool's layer styles), the "
            "nearest rainfall stations (reference only), outputs, notes and "
            "references.<br>"
            "<i>OpenStreetMap background</i> - adds OSM tiles behind the "
            "report map (needs internet access).<br>"
            "<i>Snapped grid point / POI / Nearest rainfall stations</i> - "
            "point layers loaded with their preset styles.<br>"
            "<i>Nearest rainfall stations to list</i> - number of stations "
            "per site (0 = none)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterPoint(
                self.POINT,
                "Point of interest (click map or type coordinates)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_POINTS,
                "OR: point layer (every feature processed)",
                types=[QgsProcessing.SourceType.TypeVectorPoint],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.NAME_FIELD,
                "Site name field (optional, from point layer)",
                parentLayerParameterName=self.INPUT_POINTS,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (optional, for the report)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project / site name (optional, for the report)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DURATIONS,
                "Durations",
                options=[c[0] for c in self._DURATION_CHOICES],
                allowMultiple=True,
                defaultValue=list(range(len(self._DURATION_CHOICES))),
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RETURN_PERIODS,
                "Return periods (years)",
                options=[str(rt) for rt in RETURN_PERIODS],
                allowMultiple=True,
                defaultValue=list(range(len(RETURN_PERIODS))),
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Output CSV",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_DOCX,
                "Output Word report",
                fileFilter="Word documents (*.docx)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REPORT_OSM,
                "Add OpenStreetMap background to the report map",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_LAYER,
                "Snapped grid point",
                optional=True,
                type=QgsProcessing.SourceType.TypeVectorPoint,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_QUERY_POINTS,
                "POI",
                optional=True,
                type=QgsProcessing.SourceType.TypeVectorPoint,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NUM_STATIONS,
                "Nearest rainfall stations to list "
                "(reference only, not used in calculation)",
                type=QgsProcessingParameterNumber.Integer,
                minValue=0,
                maxValue=20,
                defaultValue=6,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_STATIONS,
                "Nearest rainfall stations",
                optional=True,
                type=QgsProcessing.SourceType.TypeVectorPoint,
            )
        )

    # ------------------------------------------------------------------
    def _collect_sites(self, parameters, context, feedback):
        """Returns a list of (label, lat, lon) from whichever input was given."""
        source = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
        point = self.parameterAsPoint(parameters, self.POINT, context, crs=WGS84)

        has_layer = source is not None and source.featureCount() > 0
        has_point = not point.isEmpty()

        if has_layer and has_point:
            message = (
                "Both a point and a point layer were given - using the point layer."
            )
            feedback.pushWarning(message)
            self._warnings.append(message)
        if not has_layer and not has_point:
            raise QgsProcessingException(
                "Provide a point of interest "
                "(click the map canvas or type coordinates) "
                "OR a point layer - neither was supplied."
            )

        if has_layer:
            name_field = self.parameterAsString(parameters, self.NAME_FIELD, context)
            transform = QgsCoordinateTransform(
                source.sourceCrs(), WGS84, context.transformContext()
            )
            sites = []
            for feature in source.getFeatures():
                geom = feature.geometry()
                if geom is None or geom.isEmpty():
                    continue
                if (
                    QgsWkbTypes.geometryType(geom.wkbType())
                    != QgsWkbTypes.GeometryType.PointGeometry
                ):
                    feedback.pushWarning(
                        f"Feature {feature.id()} is not a point - skipped."
                    )
                    continue
                pt = geom.asPoint()
                if source.sourceCrs() != WGS84:
                    pt = transform.transform(pt)
                label = (
                    str(feature[name_field]) if name_field else f"Point {feature.id()}"
                )
                sites.append((label, pt.y(), pt.x()))
            return sites

        return [("Point of interest", point.y(), point.x())]

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        self._warnings = []
        sites = self._collect_sites(parameters, context, feedback)
        self._input_sites = sites
        feedback.pushInfo(f"{len(sites)} site(s) to process.")

        selected_idx = self.parameterAsEnums(parameters, self.DURATIONS, context)
        chosen = [self._DURATION_CHOICES[i][1] for i in selected_idx]
        short_durs = [v for fam, v in chosen if fam == "short"]
        daily_durs = [v for fam, v in chosen if fam == "daily" and v != 1]
        want_1day = ("daily", 1) in chosen

        selected_rt_idx = self.parameterAsEnums(
            parameters, self.RETURN_PERIODS, context
        )
        return_periods = [RETURN_PERIODS[i] for i in selected_rt_idx]

        engine = DesignRainfallEngine()
        estimates = {}
        for i, (label, lat, lon) in enumerate(sites):
            if feedback.isCanceled():
                break
            feedback.setProgress(100 * i / max(len(sites), 1))
            grid_row = engine.nearest_grid_point(lat, lon)

            results = []
            if short_durs:
                results += engine.estimate_short_durations(
                    grid_row, short_durs, return_periods
                )
            if want_1day:
                results += engine.estimate_1day(grid_row, return_periods)
            if daily_durs:
                results += engine.estimate_multiday(
                    grid_row, daily_durs, return_periods
                )

            estimates[label] = PointEstimate(
                latitude=float(grid_row.geometry.y),
                longitude=float(grid_row.geometry.x),
                map_mm=float(grid_row["MAP"]),
                altitude_m=float(grid_row["ALTITUDE"]),
                cluster=int(grid_row["CLUSTER"]),
                s_cluster=int(grid_row["S_CLUSTER"]),
                av_cluster=int(grid_row["AV_CLUSTER"]),
                source="grid",
                results=results,
                **engine.point_context(grid_row, lat, lon),
            )
            feedback.pushInfo(
                f"{label}: MAP={grid_row['MAP']}mm, {len(results)} estimates computed"
            )

        outputs = {}

        # Compute nearest stations once (used by both the CSV and the
        # dedicated stations output layer) - reference/comparison only,
        # not used in the grid-based calculation itself.
        num_stations = self.parameterAsInt(parameters, self.NUM_STATIONS, context)
        per_site_stations = {}
        if num_stations > 0:
            for label, lat, lon in sites:
                per_site_stations[label] = engine.nearest_stations(
                    lat, lon, n=num_stations
                )

        # Layers first, so the report can list every output.
        report_outputs = []
        sink_result = self._write_output_layer(
            parameters, context, estimates, return_periods
        )
        if sink_result:
            outputs[self.OUTPUT_LAYER] = sink_result
            self._register_style(context, sink_result, "snapped_grid_point.qml")
            report_outputs.append(("Snapped grid point layer", _location(sink_result)))

        query_points_result = self._write_query_points(parameters, context, sites)
        if query_points_result:
            outputs[self.OUTPUT_QUERY_POINTS] = query_points_result
            self._register_style(context, query_points_result, "poi.qml")
            report_outputs.append(("POI layer", _location(query_points_result)))

        if num_stations > 0:
            stations_result = self._write_stations_layer(
                parameters, context, per_site_stations, feedback
            )
            if stations_result:
                outputs[self.OUTPUT_STATIONS] = stations_result
                self._register_style(context, stations_result, "nearest_stations.qml")
                report_outputs.append(
                    ("Nearest rainfall stations layer", _location(stations_result))
                )

        csv_path = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)
        if csv_path:
            if len(estimates) == 1:
                label, est = next(iter(estimates.items()))
                write_csv(est, csv_path, stations=per_site_stations.get(label))
            else:
                write_csv_multi(
                    estimates, csv_path, stations_by_site=per_site_stations or None
                )
            outputs[self.OUTPUT_CSV] = csv_path
            report_outputs.append(("CSV table", csv_path))
            feedback.pushInfo(f"CSV written: {csv_path}")

        docx_path = self.parameterAsFileOutput(parameters, self.OUTPUT_DOCX, context)
        if docx_path and estimates:
            written = self._write_report(
                parameters,
                context,
                feedback,
                docx_path,
                estimates,
                per_site_stations,
                num_stations,
                report_outputs,
            )
            if written:
                outputs[self.OUTPUT_DOCX] = docx_path

        return outputs

    def _write_report(
        self,
        parameters,
        context,
        feedback,
        docx_path,
        estimates,
        per_site_stations,
        num_stations,
        report_outputs,
    ):
        """Word report with a QGIS-rendered layout map per site. Returns
        True when the report was written."""
        try:
            import docx  # noqa: F401
        except ImportError:
            feedback.reportError("python-docx not installed - skipping Word output.")
            return False
        source = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
        if source is not None and source.featureCount() > 0:
            input_text = f"Point layer: {source.sourceName()}"
        else:
            _label, lat, lon = self._input_sites[0]
            input_text = (
                f"Point of interest (latitude {lat:.5f}°, longitude {lon:.5f}°)"
            )

        map_messages = []
        plan_map = None
        try:
            from .qgis_map import make_plan_map

            plan_sites = {
                label: {
                    "poi": (est.requested_longitude, est.requested_latitude),
                    "grid": (est.longitude, est.latitude),
                    "stations": per_site_stations.get(label),
                }
                for label, est in estimates.items()
            }
            plan_map = make_plan_map(
                plan_sites,
                context.transformContext(),
                use_osm=self.parameterAsBoolean(parameters, self.REPORT_OSM, context),
                messages=map_messages,
            )
        except Exception as e:  # the report is still written without it
            map_messages.append(f"Layout map not rendered in QGIS ({e}).")

        report_context = {
            "tool": REPORT_TITLE,
            "version": TOOL_VERSION,
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "input": input_text,
            "software": _software_version(),
            "project_number": self.parameterAsString(
                parameters, self.PROJECT_NUMBER, context
            ).strip(),
            "project_name": self.parameterAsString(
                parameters, self.PROJECT_NAME, context
            ).strip(),
            "num_stations": num_stations,
            "outputs": report_outputs + [("This report", docx_path)],
            "warnings": list(self._warnings),
            # filled while the maps are rendered; read by the report's notes
            "map_messages": map_messages,
        }
        try:
            messages = write_report_docx(
                docx_path,
                estimates,
                report_context,
                stations_by_site=per_site_stations,
                plan_map=plan_map,
            )
        except PermissionError:
            feedback.reportError(
                f"Word report not written - the file is open or locked: {docx_path}"
            )
            return False
        for message in dict.fromkeys(map_messages + (messages or [])):
            feedback.pushWarning(message)
        feedback.pushInfo(f"Word report written: {docx_path}")
        return True

    def _register_style(self, context, dest_id, style_filename):
        """Attaches a saved QML style to an output layer via a
        post-processor, so it loads already symbolised/labeled -
        QGIS requires a Python-side reference to the post-processor
        to be kept alive for the duration of the run, hence
        self._post_processors."""
        if not dest_id:
            return
        details = context.layerToLoadOnCompletionDetails(dest_id)
        processor = _StylePostProcessor(STYLES_DIR / style_filename)
        details.setPostProcessor(processor)
        self._post_processors.append(processor)

    # ------------------------------------------------------------------
    def _write_query_points(self, parameters, context, sites):
        """Minimal layer: just the site label + the query point geometry -
        useful to see/symbolise the input location(s) independently of
        the results layer."""
        if not sites:
            return None
        fields = QgsFields()
        fields.append(QgsField("site", QMetaType.Type.QString))
        fields.append(QgsField("latitude", QMetaType.Type.Double))
        fields.append(QgsField("longitude", QMetaType.Type.Double))

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_QUERY_POINTS,
            context,
            fields,
            Qgis.WkbType.Point,
            WGS84,
        )
        if sink is None:
            return None
        for label, lat, lon in sites:
            feat = QgsFeature(fields)
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(lon, lat)))
            feat.setAttributes([label, lat, lon])
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_stations_layer(self, parameters, context, per_site_stations, feedback):
        """Nearest N daily rainfall stations per site, for reference/
        comparison only - NOT used in the grid-based calculation itself
        (mirrors the original tool's 'closest stations' listing).
        per_site_stations: dict[label, GeoDataFrame], precomputed by
        the caller so the same data feeds both this layer and the CSV."""
        if not per_site_stations or all(df.empty for df in per_site_stations.values()):
            feedback.pushWarning("No nearby stations found.")
            return None

        sample_df = next(df for df in per_site_stations.values() if not df.empty)
        fields = _fields_from_dataframe(
            sample_df, leading=[("site", QMetaType.Type.QString)]
        )
        data_cols = [c for c in sample_df.columns if c != "geometry"]

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_STATIONS,
            context,
            fields,
            Qgis.WkbType.Point,
            WGS84,
        )
        if sink is None:
            return None

        for label, nearby in per_site_stations.items():
            for _, row in nearby.iterrows():
                feat = QgsFeature(fields)
                feat.setGeometry(
                    QgsGeometry.fromPointXY(QgsPointXY(row.geometry.x, row.geometry.y))
                )
                attrs = [label] + [row[col] for col in data_cols]
                feat.setAttributes(attrs)
                sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)

        return dest_id

    # ------------------------------------------------------------------
    def _write_output_layer(self, parameters, context, estimates, return_periods):
        """Flattened point layer: one feature per site, one field per
        duration x return-period combo (median/lower/upper)."""
        fields = QgsFields()
        fields.append(QgsField("site", QMetaType.Type.QString))
        fields.append(QgsField("map_mm", QMetaType.Type.Double))
        fields.append(QgsField("altitude_m", QMetaType.Type.Double))
        fields.append(QgsField("cluster", QMetaType.Type.Int))

        # Build field list from the first estimate's results (all sites
        # share the same requested duration/RT combos)
        if not estimates:
            return None
        sample_results = next(iter(estimates.values())).results
        combo_keys = []
        for r in sample_results:
            key = f"{r.duration_label.replace(' ', '')}_{r.return_period}y"
            combo_keys.append(key)
            fields.append(QgsField(key, QMetaType.Type.Double))
            fields.append(QgsField(f"{key}_L", QMetaType.Type.Double))
            fields.append(QgsField(f"{key}_U", QMetaType.Type.Double))

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_LAYER,
            context,
            fields,
            Qgis.WkbType.Point,
            WGS84,
        )
        if sink is None:
            return None

        for label, est in estimates.items():
            feat = QgsFeature(fields)
            feat.setGeometry(
                QgsGeometry.fromPointXY(QgsPointXY(est.longitude, est.latitude))
            )
            attrs = [label, est.map_mm, est.altitude_m, est.cluster]
            lookup = {
                f"{r.duration_label.replace(' ', '')}_{r.return_period}y": r
                for r in est.results
            }
            for key in combo_keys:
                r = lookup.get(key)
                attrs += [
                    r.depth if r else None,
                    r.lower if r else None,
                    r.upper if r else None,
                ]
            feat.setAttributes(attrs)
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)

        return dest_id
