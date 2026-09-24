"""Processing Toolbox algorithm: point of interest in -> full IMERG
Final Run 30-minute precipitation record out, as CSV.

Point input mirrors every other extraction plugin in this suite
exactly: either a single point (map click or typed coordinates) or a
point vector layer (every feature processed). Thin wrapper - all real
logic lives in core.py (zero QGIS dependency, independently testable;
see tests/test_core.py).
"""

from datetime import datetime
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
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
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
    fetch_giovanni_timeseries,
    fetch_granule_based_timeseries,
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


class ImergExtractAlgorithm(QgsProcessingAlgorithm):

    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    METHOD = "METHOD"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    VARIABLE = "VARIABLE"
    CHUNK_MONTHS = "CHUNK_MONTHS"
    MAX_WORKERS = "MAX_WORKERS"
    FORCE_GRANULE = "FORCE_GRANULE"
    EARTHDATA_USERNAME = "EARTHDATA_USERNAME"
    EARTHDATA_PASSWORD = "EARTHDATA_PASSWORD"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_MATCHED_POINTS = "OUTPUT_MATCHED_POINTS"
    OUTPUT_QUERY_POINTS = "OUTPUT_QUERY_POINTS"

    _METHOD_OPTIONS = ["giovanni", "granule"]

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return ImergExtractAlgorithm()

    def name(self):
        return "imerg_point_extract"

    def displayName(self):
        return "Extract: IMERG precipitation"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        return (
            "IMERG Point Extractor (version 0.4.0)\n"
            "\n"
            "PURPOSE:\tExtracts the full available NASA GPM IMERG Final Run "
            "30-minute precipitation record (2000-06-01 to today, ~455,000 "
            "half-hour timestamps) at a point, for a specified date range or "
            "the full record.\n"
            "\n"
            "METHOD:\t\tUses NASA GES DISC's official Giovanni Time Series "
            "API - purpose-built for single-point, long time series, "
            "avoiding downloading individual granules one at a time (the "
            "full record is ~455,000 files - never viable one at a time). "
            "Requests are chunked by date range and fetched CONCURRENTLY "
            "(see 'Concurrent requests' parameter) - each chunk is an "
            "independent request, the same latency-bound, embarrassingly-"
            "parallel shape as the CHIRPS Point Extractor plugin's per-date "
            "fetches, using the same approach and reasoning. A slower "
            "granule-based fallback method is also available for short "
            "validation windows only - it refuses to run for long ranges by "
            "default given the scale involved (~48 granules/day). Useful "
            "for cross-checking Giovanni's actual values against real "
            "downloaded IMERG files, since Giovanni's response format was "
            "only confirmed against live data after two rounds of real "
            "fixes (see dev/README.md) - the granule method reads the "
            "files directly with no dependency on Giovanni's API contract "
            "at all. Requires the 'netCDF4' or 'h5netcdf' package (xarray "
            "does not bundle a NetCDF/HDF5 reader itself) - checked "
            "upfront before any download starts, so a missing backend "
            "fails immediately rather than after downloading every "
            "granule first (a live run without this check wasted 25 "
            "minutes downloading 240 granules before failing to open any "
            "of them). Reads each granule with xr.open_dataset (not "
            "open_mfdataset, which unconditionally needs the 'dask' "
            "package even for a single file - a second real gap found on "
            "the very next live run after the netCDF4 fix).\n"
            "\n"
            "BACKGROUND:\tRequires a free NASA Earthdata Login account "
            "(https://urs.earthdata.nasa.gov) with 'NASA GESDISC DATA "
            "ARCHIVE' authorized under your profile's Applications tab. "
            "Enter your Earthdata username/password into this tool's "
            "parameters the FIRST time you run it - they'll be saved "
            "automatically (~/.netrc, and on Windows ~/_netrc as well - "
            "Python's own netrc handling looks for that underscore-"
            "prefixed name on Windows specifically) and picked up on every "
            "run after that, so you only need to enter them once - same "
            "pattern as the ERA5 Point Extractor plugin's CDS API Key. "
            "Unlike ERA5's ~/.cdsapirc (which exists for one purpose "
            "only), netrc files are shared and may already hold entries "
            "for other services - saving here preserves any such entries "
            "rather than overwriting the whole file. IMPORTANT: the exact "
            "Giovanni variable-ID string was not independently verified "
            "against a live response when this tool was built - if "
            "results come back empty, check the 'Variable' parameter "
            "first. END DATE: a bare date (e.g. '2005-03-05', the normal "
            "way to use this field) is extended to the very end of that "
            "day (23:59:59) before being sent to Giovanni - found via a "
            "real discrepancy where Giovanni took a bare end date "
            "literally as midnight of that day and stopped there, while "
            "the granule method's date-only query correctly covered the "
            "whole day; confirmed exactly (193 rows for a 4-day range was "
            "precisely 4 days x 48 half-hours + 1, i.e. midnight-to-"
            "midnight, not a full 5th day). An end date given WITH an "
            "explicit time is respected exactly as typed, not extended.\n"
            "\n"
            "PARAMETERS:\n"
            "  Extraction method: Giovanni Time Series API (recommended) or "
            "the slower granule-based fallback.\n"
            "  Start/End date: leave blank for the full available record.\n"
            "  Variable: the Giovanni variable ID to request - override if "
            "results come back empty (see BACKGROUND).\n"
            "  Months per request chunk: Giovanni method only.\n"
            "  Concurrent requests: how many date-range chunks to fetch at "
            "once (default 8) - higher is faster but places more load on "
            "the shared NASA service; diminishing returns and throttling "
            "risk are both plausible well above this, not independently "
            "confirmed against the live service.\n"
            "  Force granule method for long ranges: override the granule "
            "method's safety threshold if you really want to run it anyway.\n"
            "  Earthdata Username/Password: enter these ONCE - saved to "
            "~/.netrc automatically, leave blank after the first "
            "successful run.\n"
            "  Output: matched grid cell location: one point per site, at "
            "the actual nearest-gridpoint coordinate the data was pulled "
            "from - the same location reported in the 'matched to the "
            "nearest grid cell' log message, as a vector layer instead of "
            "just text. IMPORTANT: only ever populated by the granule-"
            "based method - the Giovanni method (the default/recommended "
            "one) is a server-side point query and never reports which "
            "grid cell it actually used internally, so this output is "
            "left empty when Giovanni is the method used.\n"
            "  Output: POI (query point(s)): echoes the site(s) actually "
            "requested (the point clicked/typed, or every feature from the "
            "input point layer) as its own vector layer - populated "
            "regardless of extraction method.\n"
            "Both point outputs load with a preset style by default."
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
            QgsProcessingParameterEnum(
                self.METHOD,
                "Extraction method",
                options=[
                    "Giovanni Time Series API (recommended - fast, for the full record)",
                    "Direct granule download (slow - short validation windows only)",
                ],
                defaultValue=0,
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
                "Giovanni variable ID (Giovanni method only - see tool description if results are empty)",
                defaultValue=DEFAULT_GIOVANNI_VARIABLE,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CHUNK_MONTHS,
                "Months per API request chunk (Giovanni method only)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=12,
                minValue=1,
                maxValue=60,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_WORKERS,
                "Concurrent requests (Giovanni method only)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=DEFAULT_MAX_WORKERS,
                minValue=1,
                maxValue=25,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FORCE_GRANULE,
                "Force the granule method to run even for a long date range (will be slow)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.EARTHDATA_USERNAME,
                "Earthdata Username (enter once - saved automatically for future runs)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.EARTHDATA_PASSWORD,
                "Earthdata Password (enter once - saved automatically for future runs)",
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
                self.OUTPUT_MATCHED_POINTS,
                "Output: matched grid cell location (vector point; optional; "
                "granule method only - see tool description)",
                optional=True,
                type=QgsProcessing.SourceType.TypeVectorPoint,
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
                "Provide a point of interest (click the map canvas or type coordinates) "
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

        method = self._METHOD_OPTIONS[
            self.parameterAsEnum(parameters, self.METHOD, context)
        ]
        start_date = (
            self.parameterAsString(parameters, self.START_DATE, context) or None
        )
        end_date = self.parameterAsString(parameters, self.END_DATE, context) or None
        variable = self.parameterAsString(parameters, self.VARIABLE, context)
        chunk_months = self.parameterAsInt(parameters, self.CHUNK_MONTHS, context)
        max_workers = self.parameterAsInt(parameters, self.MAX_WORKERS, context)
        force_granule = self.parameterAsBoolean(parameters, self.FORCE_GRANULE, context)
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
            feedback.pushInfo(
                f"Earthdata credentials saved to: {', '.join(str(p) for p in saved_paths)} "
                "- leave these fields blank on future runs."
            )
        elif earthdata_username or earthdata_password:
            feedback.pushWarning(
                "Both Earthdata Username and Password are needed to save credentials - "
                "only one was provided, so nothing was saved. Falling back to ~/.netrc "
                "if it already exists."
            )

        if method == "giovanni":
            try:
                token = get_earthdata_token()
            except Exception as e:
                raise QgsProcessingException(str(e)) from e
        else:
            token = None

        all_rows = []
        results_by_site = {}
        for site_idx, (label, lat, lon) in enumerate(sites):
            feedback.pushInfo(f"[{label}] lat={lat:.4f}, lon={lon:.4f}")

            def _progress(pct, _label=label, _idx=site_idx):
                if feedback.isCanceled():
                    raise InterruptedError("Cancelled by user")
                overall = int(100 * (_idx + pct / 100) / len(sites))
                feedback.setProgress(overall)

            try:
                if method == "giovanni":
                    result = fetch_giovanni_timeseries(
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
                else:
                    result = fetch_granule_based_timeseries(
                        lat,
                        lon,
                        start_date=start_date or RECORD_START,
                        end_date=end_date or datetime.now().strftime("%Y-%m-%d"),
                        force=force_granule,
                        progress_callback=_progress,
                    )
            except InterruptedError:
                feedback.pushInfo("Cancelled.")
                return {}

            for w in result.warnings:
                feedback.pushWarning(f"[{label}] {w}")
            feedback.pushInfo(
                f"[{label}] {len(result.dataframe)} rows retrieved via {result.method} method."
            )

            df = result.dataframe.copy()
            df.insert(0, "Site", label)
            all_rows.append(df)
            results_by_site[label] = result

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

        matched_points_result = self._write_matched_points(
            parameters, context, results_by_site
        )
        if matched_points_result:
            outputs[self.OUTPUT_MATCHED_POINTS] = matched_points_result
            self._register_style(
                context, matched_points_result, "snapped_grid_point.qml"
            )
        elif method == "giovanni":
            feedback.pushInfo(
                "Matched grid cell output left empty - the Giovanni method never "
                "reports which grid cell it used internally (see tool description)."
            )

        query_points_result = self._write_query_points(parameters, context, sites)
        if query_points_result:
            outputs[self.OUTPUT_QUERY_POINTS] = query_points_result
            self._register_style(context, query_points_result, "poi.qml")

        return outputs

    def _write_matched_points(self, parameters, context, results_by_site):
        """One point per site, at the actual nearest-gridpoint coordinate
        the data was extracted from (result.matched_lat/matched_lon - see
        core.py's FetchResult) - the same location already reported in the
        'matched to the nearest grid cell' warning message, now also
        available as a vector layer. Only ever populated for sites
        processed via the granule-based method (see FetchResult's
        docstring comment) - sites with no matched coordinate (Giovanni
        method, or no granule succeeded) are skipped rather than written
        with a missing geometry."""
        sites_with_match = [
            (label, result)
            for label, result in results_by_site.items()
            if result.matched_lat is not None and result.matched_lon is not None
        ]
        if not sites_with_match:
            return None

        fields = QgsFields()
        fields.append(QgsField("site", QMetaType.Type.QString))
        fields.append(QgsField("requested_lat", QMetaType.Type.Double))
        fields.append(QgsField("requested_lon", QMetaType.Type.Double))
        fields.append(QgsField("matched_lat", QMetaType.Type.Double))
        fields.append(QgsField("matched_lon", QMetaType.Type.Double))
        fields.append(QgsField("distance_km", QMetaType.Type.Double))

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT_MATCHED_POINTS,
            context,
            fields,
            Qgis.WkbType.Point,
            WGS84,
        )
        if sink is None:
            return None

        for label, result in sites_with_match:
            feat = QgsFeature(fields)
            feat.setGeometry(
                QgsGeometry.fromPointXY(
                    QgsPointXY(result.matched_lon, result.matched_lat)
                )
            )
            feat.setAttributes(
                [
                    label,
                    result.requested_lat,
                    result.requested_lon,
                    result.matched_lat,
                    result.matched_lon,
                    result.distance_km,
                ]
            )
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_query_points(self, parameters, context, sites):
        """Minimal layer: just the site label + the query point geometry
        actually requested (the clicked/typed point, or every feature from
        the input point layer) - populated regardless of extraction
        method, unlike the matched grid cell output above."""
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
