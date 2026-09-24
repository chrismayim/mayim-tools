"""Processing Toolbox algorithm: point of interest in -> CHIRPS v3
precipitation time series out, as CSV.

Point input mirrors the Design Rainfall / IMERG Point Extractor
plugins exactly: either a single point (map click or typed
coordinates) or a point vector layer (every feature processed). Thin
wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_core.py).

v0.2: added MAX_WORKERS (concurrent fetching) after a live run showed
~1.46s/file, projecting to ~6 hours for a full daily-record
extraction - see core.py's module docstring PERFORMANCE section.
"""

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

from .core import DEFAULT_MAX_WORKERS, PRODUCTS, fetch_chirps_timeseries
from .export import write_chirps_csv

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

STYLES_DIR = Path(__file__).resolve().parent / "styles"


class _StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Loads a saved QML style (symbology + labeling) onto an output
    layer once Processing has finished loading it into the project -
    same pattern as design_rainfall_algorithm.py's/era5_extract's own
    post-processor."""

    def __init__(self, style_path: Path):
        super().__init__()
        self.style_path = str(style_path)

    def postProcessLayer(self, layer, context, feedback):
        layer.loadNamedStyle(self.style_path)
        layer.triggerRepaint()


class ChirpsExtractAlgorithm(QgsProcessingAlgorithm):

    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    PRODUCT = "PRODUCT"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    MAX_WORKERS = "MAX_WORKERS"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_MATCHED_POINTS = "OUTPUT_MATCHED_POINTS"
    OUTPUT_QUERY_POINTS = "OUTPUT_QUERY_POINTS"

    _PRODUCT_OPTIONS = list(PRODUCTS)  # pentad, daily_rnl, daily_sat, monthly

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return ChirpsExtractAlgorithm()

    def name(self):
        return "chirps_point_extract"

    def displayName(self):
        return "Extract: CHIRPS precipitation"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        return (
            "CHIRPS Point Extractor (version 0.4.0)\n"
            "\n"
            "PURPOSE:\tExtracts CHIRPS v3 precipitation at a point for a specified "
            "date range or the full available record (1981-01-01 to today for "
            "Pentad/Monthly/Daily-ERA5; 1998-01-01 to today for Daily-IMERG - "
            "see BACKGROUND). CHIRPS is a thermal-infrared cold-cloud product, "
            "blended with rain gauge station data.\n"
            "\n"
            "METHOD:\t\tReads individual CHIRPS GeoTIFF/COG files directly from "
            "the Climate Hazards Center's public server via GDAL's HTTP "
            "range-request support - no login required (CHIRPS is public "
            "domain), no full-file downloads. Requests for different dates "
            "are fetched concurrently (see 'Concurrent requests' parameter) "
            "since each file's fetch is independent, latency-bound network "
            "work - a full daily-record extraction measured at ~6 hours "
            "fetched serially completed in under a minute per year once "
            "concurrency was added.\n"
            "\n"
            "BACKGROUND:\tCHIRPS is fundamentally a pentad (5-day) and monthly "
            "product; two derived daily variants are offered - 'daily_rnl' "
            "(ERA5-based disaggregation) and 'daily_sat' (IMERG-based "
            "disaggregation). Both split the SAME pentad total across its 5 "
            "days differently - the pentad total itself is identical either "
            "way, only the day-to-day timing differs. daily_sat's record "
            "starts in 1998, not 1981, because it depends on NASA IMERG, "
            "which has no data before that year - a genuine gap in the "
            "underlying satellite record, not a limitation of this tool. "
            "daily_rnl has no such limit since ERA5's own reanalysis record "
            "comfortably covers the full CHIRPS period. Monthly and daily URL "
            "patterns are confirmed exactly (daily was verified against the "
            "ropensci/chirps R package's own source code after an initial "
            "guess failed on first live use); the pentad URL pattern remains "
            "inferred and unconfirmed - if a pentad run returns no data, "
            "check dev/README.md and the exact URL shown in the log first.\n"
            "\n"
            "PARAMETERS:\n"
            "  CHIRPS Product: choose between Pentad (5-day, native resolution "
            "- recommended for long ranges), Daily (ERA5), Daily (IMERG), or "
            "Monthly.\n"
            "  Start/End date: leave blank for the full available record for "
            "the chosen product (see PURPOSE). An explicit date always "
            "overrides the default, even earlier than a product's real data "
            "availability, if you want to confirm a limit yourself.\n"
            "  Concurrent requests: a value between 1 and 25. Higher is "
            "faster but places more load on the public CHC server; 8 is a "
            "reasonable default. Diminishing returns and a real risk of "
            "server-side throttling are both plausible above roughly 16-20 - "
            "not independently confirmed against the live server, so treat "
            "anything above that range with caution rather than assuming it "
            "scales indefinitely.\n"
            "  Output: matched grid cell location: one point per site, at "
            "the centre of the actual CHIRPS pixel the data was read from - "
            "the same location reported in the 'matched to the nearest grid "
            "cell' log message, as a vector layer instead of just text. "
            "Attributes include the requested point, the matched grid cell, "
            "and the distance between them.\n"
            "  Output: POI (query point(s)): echoes the site(s) actually "
            "requested (the point clicked/typed, or every feature from the "
            "input point layer) as its own vector layer, for reference "
            "alongside the matched grid cell output above.\n"
            "Both point outputs load with a preset style by default "
            "(matching the style used by the Design Rainfall tool's "
            "equivalent outputs)."
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
                self.PRODUCT,
                "CHIRPS product",
                options=[
                    "Pentad (native 5-day resolution - recommended for long ranges)",
                    "Daily (ERA5-based disaggregation)",
                    "Daily (IMERG-based disaggregation)",
                    "Monthly",
                ],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.START_DATE,
                "Start date, YYYY-MM-DD (blank = full record start, 1981-01-01)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.END_DATE,
                "End date, YYYY-MM-DD (blank = today)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_WORKERS,
                "Concurrent requests (higher = faster, but more load on the "
                "public CHC server - 1 disables concurrency entirely)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=DEFAULT_MAX_WORKERS,
                minValue=1,
                maxValue=25,
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
                "Output: matched grid cell location (vector point; optional)",
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

        product = self._PRODUCT_OPTIONS[
            self.parameterAsEnum(parameters, self.PRODUCT, context)
        ]
        start_date = (
            self.parameterAsString(parameters, self.START_DATE, context) or None
        )
        end_date = self.parameterAsString(parameters, self.END_DATE, context) or None
        max_workers = self.parameterAsInt(parameters, self.MAX_WORKERS, context)
        output_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        feedback.pushInfo(f"Using {max_workers} concurrent request(s).")

        results_by_site = {}
        for site_idx, (label, lat, lon) in enumerate(sites):
            feedback.pushInfo(
                f"[{label}] lat={lat:.4f}, lon={lon:.4f}, product={product}"
            )

            def _progress(pct, _idx=site_idx):
                if feedback.isCanceled():
                    raise InterruptedError("Cancelled by user")
                feedback.setProgress(int(100 * (_idx + pct / 100) / len(sites)))

            try:
                result = fetch_chirps_timeseries(
                    lat,
                    lon,
                    start_date=start_date,
                    end_date=end_date,
                    product=product,
                    progress_callback=_progress,
                    max_workers=max_workers,
                )
            except InterruptedError:
                feedback.pushInfo("Cancelled.")
                return {}
            except ValueError as e:
                raise QgsProcessingException(str(e)) from e

            for w in result.warnings[:10]:
                feedback.pushWarning(f"[{label}] {w}")
            if len(result.warnings) > 10:
                feedback.pushWarning(
                    f"[{label}] ...and {len(result.warnings) - 10} more "
                    "warnings (see full log)."
                )

            feedback.pushInfo(f"[{label}] {len(result.dataframe)} rows retrieved.")
            if result.urls_attempted_sample:
                feedback.pushInfo(
                    f"[{label}] Sample URL attempted: {result.urls_attempted_sample[0]}"
                )

            results_by_site[label] = result

        n_rows = write_chirps_csv(results_by_site, output_csv)
        feedback.pushInfo(f"{n_rows} total rows written to {output_csv}")

        outputs = {self.OUTPUT_CSV: output_csv}

        matched_points_result = self._write_matched_points(
            parameters, context, results_by_site
        )
        if matched_points_result:
            outputs[self.OUTPUT_MATCHED_POINTS] = matched_points_result
            self._register_style(
                context, matched_points_result, "snapped_grid_point.qml"
            )

        query_points_result = self._write_query_points(parameters, context, sites)
        if query_points_result:
            outputs[self.OUTPUT_QUERY_POINTS] = query_points_result
            self._register_style(context, query_points_result, "poi.qml")

        return outputs

    def _write_matched_points(self, parameters, context, results_by_site):
        """One point per site, at the centre of the actual CHIRPS pixel
        the data was read from (result.matched_lat/matched_lon - see
        core.py's ChirpsResult) - the same location already reported in
        the 'matched to the nearest grid cell' warning message, now also
        available as a vector layer. Mirrors era5_extract_algorithm.py's
        _write_matched_points pattern. Sites where nothing succeeded
        (matched_lat/matched_lon still None) are skipped rather than
        written with a missing geometry."""
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
        the input point layer) - useful to see/symbolise the requested
        location(s) independently of the matched grid cell output above.
        Mirrors era5_extract_algorithm.py's/design_rainfall_algorithm.py's
        own _write_query_points."""
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
        post-processor, so it loads already symbolised - same pattern as
        era5_extract_algorithm.py's/design_rainfall_algorithm.py's own
        _register_style. The processor instance is kept alive on
        self._post_processors since Processing only holds a weak
        reference to it."""
        if not dest_id:
            return
        details = context.layerToLoadOnCompletionDetails(dest_id)
        processor = _StylePostProcessor(STYLES_DIR / style_filename)
        details.setPostProcessor(processor)
        self._post_processors.append(processor)
