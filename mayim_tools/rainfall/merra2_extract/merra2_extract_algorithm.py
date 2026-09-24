"""Processing Toolbox algorithm: point of interest in -> MERRA-2
bias-corrected hourly precipitation (PRECTOTCORR) time series out, as
CSV.

Point input mirrors every other extraction plugin in this suite
exactly: either a single point (map click or typed coordinates) or a
point vector layer (every feature processed). Thin wrapper - all real
logic lives in core.py (zero QGIS dependency, independently testable;
see tests/test_core.py).
"""

from pathlib import Path

import pandas as pd
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
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterPoint,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from .core import (
    DEFAULT_GIOVANNI_VARIABLE,
    DEFAULT_MAX_WORKERS,
    RECORD_START,
    fetch_merra2_timeseries,
    get_earthdata_token,
    save_earthdata_credentials,
)

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

STYLES_DIR = Path(__file__).resolve().parent / "styles"


class _StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Loads a saved QML style (symbology + labeling) onto an output
    layer once Processing has finished loading it into the project -
    same pattern as era5_extract_algorithm.py's own post-processor."""

    def __init__(self, style_path: Path):
        super().__init__()
        self.style_path = str(style_path)

    def postProcessLayer(self, layer, context, feedback):
        layer.loadNamedStyle(self.style_path)
        layer.triggerRepaint()


class Merra2ExtractAlgorithm(QgsProcessingAlgorithm):

    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    VARIABLE = "VARIABLE"
    CHUNK_MONTHS = "CHUNK_MONTHS"
    MAX_WORKERS = "MAX_WORKERS"
    EARTHDATA_USERNAME = "EARTHDATA_USERNAME"
    EARTHDATA_PASSWORD = "EARTHDATA_PASSWORD"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_QUERY_POINTS = "OUTPUT_QUERY_POINTS"

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return Merra2ExtractAlgorithm()

    def name(self):
        return "merra2_point_extract"

    def displayName(self):

        return "Extract: MERRA2 precipitation"

    def group(self):

        return "Rainfall Tools"

    def groupId(self):

        return "rainfall_tools"

    def shortHelpString(self):
        return (
            "MERRA-2 Point Extractor (version 0.2.0)\n"
            "\n"
            "PURPOSE:\tExtracts the full available NASA MERRA-2 bias-"
            "corrected hourly precipitation (PRECTOTCORR) record "
            "(1980-01-01 to today, ~3-week processing latency) at a "
            "point, for a specified date range or the full record.\n"
            "\n"
            "METHOD:\t\tUses NASA GES DISC's Giovanni Time Series API - "
            "the same service already used by the IMERG Point Extractor "
            "plugin, confirmed to also serve MERRA-2 directly from "
            "Giovanni's own live WMS listing. PRECTOTCORR (variable ID "
            "M2T1NXFLX_5_12_4_PRECTOTCORR) is used, not the raw "
            "uncorrected PRECTOT - PRECTOTCORR is bias-corrected against "
            "gauge/satellite observations and is what the peer-reviewed "
            "hydrological literature actually uses. Requests are chunked "
            "by date range and fetched CONCURRENTLY - the same approach "
            "already used by the CHIRPS and IMERG Point Extractor "
            "plugins, since each chunk is an independent request. Unlike "
            "ERA5, no deaccumulation is needed: MERRA-2 precipitation is "
            "a time-averaged RATE (kg/m^2/s, i.e. mm/s) - converted to "
            "mm/hr by a simple x3600, confirmed via NASA's own Earthdata "
            "Forum, not a cumulative value needing differencing.\n"
            "\n"
            "BACKGROUND:\tRegional caveat worth knowing, not a generic "
            "disclaimer: per GMAO's own MERRA-2 file specification, "
            "PRECTOTCORR's gauge-based correction tapers back to raw "
            "model precipitation poleward of 42.5 degrees latitude (and "
            "is fully model-only poleward of 62.5 degrees); over "
            "continental Africa specifically, the correction source "
            "switches to the CMAP gauge-satellite product rather than "
            "the primary source used elsewhere, due to limited gauge "
            "density in the region - still a genuine bias correction, "
            "just from a coarser source for an African site. Requires a "
            "free NASA Earthdata Login account "
            "(https://urs.earthdata.nasa.gov) with 'NASA GESDISC DATA "
            "ARCHIVE' authorized under your profile's Applications tab. "
            "Enter your Earthdata username/password into this tool's "
            "parameters the FIRST time you run it - saved automatically "
            "(~/.netrc, and on Windows also ~/_netrc) and picked up on "
            "every run after that, so you only need to enter them once - "
            "same pattern as the IMERG Point Extractor plugin. IMPORTANT: "
            "the exact Giovanni response format for MERRA-2 was not "
            "independently verified against a live response when this "
            "tool was built - the same metadata-preamble-tolerant parser "
            "already hard-won for IMERG's Giovanni responses is applied "
            "here defensively, on the assumption that the same service "
            "behaves similarly, not a confirmation.\n"
            "\n"
            "PARAMETERS:\n"
            "  Start/End date: leave blank for the full available "
            "record.\n"
            "  Variable: the Giovanni variable ID to request - override "
            "if results come back empty.\n"
            "  Months per request chunk: default 12.\n"
            "  Concurrent requests: default 8 - higher is faster but "
            "places more load on the shared NASA service.\n"
            "  Earthdata Username/Password: enter these ONCE - saved "
            "automatically, leave blank after the first successful run.\n"
            "  Output: POI (query point(s)): echoes the site(s) actually "
            "requested (the point clicked/typed, or every feature from "
            "the input point layer) as its own vector layer, loaded with "
            "a preset style by default. NOTE: unlike this suite's other "
            "extraction tools, there is no separate 'matched grid cell' "
            "output here - MERRA-2 is served entirely through NASA's "
            "Giovanni API (a server-side point query), which never "
            "reports which grid cell it actually used internally, so "
            "there is nothing genuine to show beyond the requested point."
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
                self.START_DATE,
                f"Start date (blank = full record start, {RECORD_START})",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.END_DATE,
                "End date (blank = today; a bare date\n"
                "runs through the end of that day, not\n"
                "just its first instant - see description)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.VARIABLE,
                "Giovanni variable ID (override only if results are empty)",
                defaultValue=DEFAULT_GIOVANNI_VARIABLE,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CHUNK_MONTHS,
                "Months per API request chunk",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=12,
                minValue=1,
                maxValue=60,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_WORKERS,
                "Concurrent requests",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=DEFAULT_MAX_WORKERS,
                minValue=1,
                maxValue=25,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.EARTHDATA_USERNAME,
                "Earthdata Username (enter once - saved automatically)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.EARTHDATA_PASSWORD,
                "Earthdata Password (enter once - saved automatically)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Output CSV",
                fileFilter="CSV files (*.csv)",
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_QUERY_POINTS,
                "Output: POI (query point(s); optional)",
                optional=True,
                type=QgsProcessing.SourceType.TypeVectorPoint,
            )
        )

    def _collect_sites(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT_POINTS, context)
        point = self.parameterAsPoint(parameters, self.POINT, context, crs=WGS84)

        has_layer = source is not None and source.featureCount() > 0
        has_point = not point.isEmpty()

        if has_layer and has_point:
            feedback.pushWarning(
                "Both a point and a point layer were given - using the point layer."
            )
        if not has_layer and not has_point:
            raise QgsProcessingException(
                "Provide a point of interest (click the map canvas or "
                "type coordinates) OR a point layer - neither was supplied."
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
        sites = self._collect_sites(parameters, context, feedback)
        feedback.pushInfo(f"{len(sites)} site(s) to process.")

        start_date = (
            self.parameterAsString(parameters, self.START_DATE, context) or None
        )
        end_date = self.parameterAsString(parameters, self.END_DATE, context) or None
        variable = self.parameterAsString(parameters, self.VARIABLE, context)
        chunk_months = self.parameterAsInt(parameters, self.CHUNK_MONTHS, context)
        max_workers = self.parameterAsInt(parameters, self.MAX_WORKERS, context)
        earthdata_username = (
            self.parameterAsString(parameters, self.EARTHDATA_USERNAME, context) or None
        )
        earthdata_password = (
            self.parameterAsString(parameters, self.EARTHDATA_PASSWORD, context) or None
        )
        output_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        if earthdata_username and earthdata_password:
            saved_paths = save_earthdata_credentials(
                earthdata_username, earthdata_password
            )
            # never log the credentials themselves - only confirm where they were saved
            saved_paths_str = ", ".join(str(p) for p in saved_paths)
            feedback.pushInfo(
                f"Earthdata credentials saved to: {saved_paths_str} "
                "- leave these fields blank on future runs."
            )
        elif earthdata_username or earthdata_password:
            feedback.pushWarning(
                "Both Earthdata Username and Password are needed to save credentials - "
                "only one was provided, so nothing was saved. Falling back to ~/.netrc "
                "if it already exists."
            )

        try:
            token = get_earthdata_token()
        except Exception as e:
            raise QgsProcessingException(str(e)) from e

        all_rows = []
        for site_idx, (label, lat, lon) in enumerate(sites):
            feedback.pushInfo(f"[{label}] lat={lat:.4f}, lon={lon:.4f}")

            def _progress(pct, _label=label, _idx=site_idx):
                if feedback.isCanceled():
                    raise InterruptedError("Cancelled by user")
                overall = int(100 * (_idx + pct / 100) / len(sites))
                feedback.setProgress(overall)

            try:
                result = fetch_merra2_timeseries(
                    lat,
                    lon,
                    start_date=start_date,
                    end_date=end_date,
                    variable=variable,
                    chunk_months=chunk_months,
                    token=token,
                    progress_callback=_progress,
                    max_workers=max_workers,
                )
            except InterruptedError:
                feedback.pushInfo("Cancelled.")
                return {}

            for w in result.warnings:
                feedback.pushWarning(f"[{label}] {w}")
            feedback.pushInfo(f"[{label}] {len(result.dataframe)} rows retrieved.")

            df = result.dataframe.copy()
            df.insert(0, "Site", label)
            all_rows.append(df)

        if not all_rows or all(len(df) == 0 for df in all_rows):
            feedback.pushWarning("No data retrieved for any site.")

        combined = (
            pd.concat(all_rows, ignore_index=True)
            if all_rows
            else pd.DataFrame(columns=["Site", "Timestamp", "PrecipitationMMHR"])
        )
        combined.to_csv(output_csv, index=False)
        feedback.pushInfo(f"{len(combined)} total rows written to {output_csv}")

        outputs = {self.OUTPUT_CSV: output_csv}

        query_points_result = self._write_query_points(parameters, context, sites)
        if query_points_result:
            outputs[self.OUTPUT_QUERY_POINTS] = query_points_result
            self._register_style(context, query_points_result, "poi.qml")

        return outputs

    def _write_query_points(self, parameters, context, sites):
        """Minimal layer: just the site label + the query point geometry
        actually requested (the clicked/typed point, or every feature from
        the input point layer). MERRA-2 has no matched-grid-cell output
        (unlike this suite's other extraction tools) - it is served
        entirely through Giovanni, a server-side point query that never
        reports which grid cell it actually used internally, so there is
        nothing genuine to show beyond the requested point itself."""
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

    def _register_style(self, context, dest_id, style_filename):
        """Attaches a saved QML style to an output layer via a
        post-processor, so it loads already symbolised. The processor
        instance is kept alive on self._post_processors since Processing
        only holds a weak reference to it."""
        if not dest_id:
            return
        details = context.layerToLoadOnCompletionDetails(dest_id)
        processor = _StylePostProcessor(STYLES_DIR / style_filename)
        details.setPostProcessor(processor)
        self._post_processors.append(processor)
