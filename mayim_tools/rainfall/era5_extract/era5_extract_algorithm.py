"""Processing Toolbox algorithm: point of interest in -> ERA5 or
ERA5-Land precipitation time series out, as CSV.

Point input mirrors every other extraction plugin in this suite
exactly: either a single point (map click or typed coordinates) or a
point vector layer (every feature processed). Thin wrapper - all real
logic lives in core.py (zero QGIS dependency, independently testable;
see tests/test_era5_extract_core.py).
"""

import tempfile
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

from .core import (
    DATASETS,
    ERA5_RECORD_START,
    fetch_era5_point,
    fetch_era5_point_timeseries,
)
from .export import write_era5_anomalies, write_era5_csv

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

STYLES_DIR = Path(__file__).resolve().parent / "styles"


class _StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    """Loads a saved QML style (symbology + labeling) onto an output
    layer once Processing has finished loading it into the project -
    same pattern as design_rainfall_algorithm.py's own post-processor."""

    def __init__(self, style_path: Path):
        super().__init__()
        self.style_path = str(style_path)

    def postProcessLayer(self, layer, context, feedback):
        layer.loadNamedStyle(self.style_path)
        layer.triggerRepaint()


class Era5ExtractAlgorithm(QgsProcessingAlgorithm):
    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    PRODUCT = "PRODUCT"
    ACCESS_METHOD = "ACCESS_METHOD"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    YEARS_PER_CHUNK = "YEARS_PER_CHUNK"
    CDS_API_KEY = "CDS_API_KEY"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_ANOMALIES = "OUTPUT_ANOMALIES"
    OUTPUT_MATCHED_POINTS = "OUTPUT_MATCHED_POINTS"
    OUTPUT_QUERY_POINTS = "OUTPUT_QUERY_POINTS"

    PRODUCT_OPTIONS = list(DATASETS)  # ["era5", "era5_land"]
    ACCESS_METHOD_OPTIONS = [
        "Standard (robust, slower - hours for a long record)",
        "Fast (experimental - seconds to a couple of minutes for the full record)",
    ]

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return Era5ExtractAlgorithm()

    def name(self):
        return "era5_point_extract"

    def displayName(self):
        return "Extract: ERA5 precipitation"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        return (
            "Extract: ERA5 precipitation (version 0.5.0)\n"
            "\n"
            "PURPOSE:\tExtracts ERA5 or ERA5-Land total precipitation at a "
            "point for a specified date range or the full available record "
            "(1950-01-01 to today for both products). Both are ECMWF "
            "reanalysis products, not observations - ERA5 is a full "
            "atmospheric reanalysis (~31 km), ERA5-Land a higher-resolution "
            "(~9 km) land-surface-model rerun driven by ERA5's own "
            "atmospheric output. IMPORTANT: for precipitation specifically, "
            "ERA5-Land's finer grid is largely a linear interpolation of "
            "ERA5's own coarser field, not an independently higher-"
            "resolution estimate - unlike temperature/humidity, ERA5's "
            "precipitation forcing is NOT bias-corrected when downscaled "
            "to ERA5-Land. ERA5-Land's real value-add is land-surface "
            "variables (soil moisture, runoff, snow), not precipitation "
            "itself - plain ERA5 is the more defensible default for "
            "rainfall-focused work.\n"
            "\n"
            "METHOD:\t\tTwo selectable access methods, both requesting "
            "total_precipitation via the official ecmwf/cdsapi client:\n"
            "  Standard: the general-purpose CDS grid dataset, chunked by "
            "year (see 'Years per request'). Robust and thoroughly tested, "
            "but slow - a multi-decade record can take hours, since each "
            "request queues on CDS's batch processing backend.\n"
            "  Fast: a separate, purpose-built CDS dataset "
            "(reanalysis-era5-single-levels-timeseries / "
            "-land-timeseries) designed specifically for retrieving a long "
            "single-point time series in ONE request. Confirmed directly "
            "by live testing: the full 1950-2025 record for one point in "
            "40 seconds, versus multi-hour Standard runs - and its values "
            "matched a real CDS website reference exactly, hour for hour. "
            "It is also already the source used by ECMWF's own public "
            "ERA-Explorer web tool. HOWEVER, ECMWF's own documentation "
            "states plainly that this dataset is 'experimental... not "
            "recommended for use in operational systems' - offered here as "
            "a clearly-labelled, opt-in alternative, not the default, "
            "specifically so that caution is respected rather than quietly "
            "built past. Only the plain ERA5 product has been independently "
            "live-tested against a real reference with this method; "
            "ERA5-Land's equivalent dataset exists (per ECMWF's own "
            "announcement) but has not been separately confirmed by this "
            "tool.\n"
            "Both methods request each hour's value directly, with no "
            "de-accumulation applied - an earlier version of this tool "
            "de-accumulated Standard-method values against each preceding "
            "hour, on the assumption ERA5 delivers a running cumulative "
            "total needing differencing. That was a real, substantial bug "
            "(confirmed directly against the CDS website's own reported "
            "values) that silently understated true rainfall and produced "
            "spurious negative values whenever two consecutive hours both "
            "had real rain - fixed by removing the differencing step "
            "entirely, since each hour's delivered value is already "
            "correct on its own.\n"
            "\n"
            "BACKGROUND:\tRequires a free CDS account "
            "(https://cds.climate.copernicus.eu). Enter your Personal "
            "Access Token (from https://cds.climate.copernicus.eu/profile) "
            "into the 'CDS API Key' parameter the FIRST time you run this "
            "tool - it will be saved to ~/.cdsapirc automatically (the "
            "standard config file the official cdsapi client reads) and "
            "picked up on every run after that, so you only need to enter "
            "it once. Leave the field blank on subsequent runs. The key is "
            "never stored anywhere inside this tool or a QGIS project file "
            "- only in that one standard config file, exactly as if you'd "
            "created it yourself by hand. You must also accept the "
            "dataset's Terms of Use on the CDS website (logged in, visit "
            "the dataset page and accept the licence at the bottom of the "
            "download form) before the API will serve data - a one-time "
            "step this tool cannot do on your behalf; this applies "
            "separately to each dataset used, so switching access methods "
            "for the first time may need a separate one-time acceptance. "
            "Unlike IMERG or CHIRPS, ERA5 and ERA5-Land are combined into "
            "this single tool rather than kept as separate plugins, since "
            "both are served through the exact same CDS infrastructure - "
            "the only real difference is the dataset name requested.\n"
            "\n"
            "PARAMETERS:\n"
            "  Product: ERA5 (recommended default for precipitation) or "
            "ERA5-Land (see PURPOSE for why this isn't simply 'higher "
            "resolution' for rainfall specifically).\n"
            "  Access method: Standard or Fast - see METHOD above, and "
            "weigh the speed gain against ECMWF's own experimental/not-for-"
            "operational-use caveat before choosing Fast for design work.\n"
            "  Start/End date: leave blank for the full available record "
            "(1950-01-01 to today).\n"
            "  Years per request: Standard method only - how many years of "
            "hourly data to request from CDS in a single call (default 1) "
            "- a deliberate safety margin against undocumented CDS request "
            "size limits, not a strict requirement; increasing it reduces "
            "the number of requests but each one takes longer and risks a "
            "larger failure if something goes wrong partway through. Not "
            "used by the Fast method, which always makes one request "
            "covering the whole date range.\n"
            "  CDS API Key: enter this ONCE - it's saved to ~/.cdsapirc "
            "automatically and every later run reads it from there, so "
            "leave this blank after the first successful run.\n"
            "  Output: matched grid cell location: one point per site, at "
            "the actual nearest-gridpoint coordinate the data was pulled "
            "from - the same location reported in the 'matched to the "
            "nearest grid cell' log message, as a vector layer instead of "
            "just text. Attributes include the requested point, the "
            "matched grid cell, and the distance between them.\n"
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
                "Product",
                options=[
                    "ERA5 (recommended for precipitation - see tool description)",
                    "ERA5-Land",
                ],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ACCESS_METHOD,
                "Access method - see BACKGROUND below before choosing Fast",
                options=self.ACCESS_METHOD_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.START_DATE,
                f"Start date, YYYY-MM-DD (blank = full record start, "
                f"{ERA5_RECORD_START})",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.END_DATE, "End date, YYYY-MM-DD (blank = today)", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.YEARS_PER_CHUNK,
                "Years per CDS request (Standard access method only)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=1,
                minValue=1,
                maxValue=10,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.CDS_API_KEY,
                "CDS API Key (enter once - saved automatically for future runs)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV, "Output CSV", fileFilter="CSV files (*.csv)"
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_ANOMALIES,
                "Output: negative-value anomalies audit CSV (large negative "
                "values; optional)",
                fileFilter="CSV files (*.csv)",
                optional=True,
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

    def collect_sites(self, parameters, context, feedback):
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
                "Provide a point of interest (click the map canvas or type "
                "coordinates) OR a point layer - neither was supplied."
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
        sites = self.collect_sites(parameters, context, feedback)
        feedback.pushInfo(f"{len(sites)} site(s) to process.")

        product = self.PRODUCT_OPTIONS[
            self.parameterAsEnum(parameters, self.PRODUCT, context)
        ]
        access_method_idx = self.parameterAsEnum(
            parameters, self.ACCESS_METHOD, context
        )
        is_fast = access_method_idx == 1
        start_date = (
            self.parameterAsString(parameters, self.START_DATE, context) or None
        )
        end_date = self.parameterAsString(parameters, self.END_DATE, context) or None
        years_per_chunk = self.parameterAsInt(parameters, self.YEARS_PER_CHUNK, context)
        api_key = self.parameterAsString(parameters, self.CDS_API_KEY, context) or None
        output_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        work_dir = tempfile.mkdtemp(prefix="era5_extract_")
        feedback.pushInfo(
            f"Product: {product}. Access method: "
            f"{self.ACCESS_METHOD_OPTIONS[access_method_idx]}. "
            f"Working directory: {work_dir}"
        )
        if is_fast and product == "era5_land":
            feedback.pushWarning(
                "Fast access for ERA5-Land has not been independently live-tested by "
                "this tool (only plain ERA5 has been confirmed against a real "
                "reference) - proceed with extra caution and consider cross-checking "
                "a sample against the standard method."
            )

        if api_key:
            # Never log the key itself - only confirm that saving happened.
            feedback.pushInfo(
                "CDS API key provided - will be saved to ~/.cdsapirc for future runs "
                "(leave this field blank next time)."
            )

        results_by_site = {}
        for site_idx, (label, lat, lon) in enumerate(sites):
            feedback.pushInfo(f"[{label}] lat={lat:.4f}, lon={lon:.4f}")

            def progress(pct, idx=site_idx):
                if feedback.isCanceled():
                    raise InterruptedError("Cancelled by user")
                feedback.setProgress(int(100 * (idx + pct / 100) / len(sites)))

            try:
                if is_fast:
                    if feedback.isCanceled():
                        raise InterruptedError("Cancelled by user")
                    result = fetch_era5_point_timeseries(
                        lat,
                        lon,
                        start_date=start_date,
                        end_date=end_date,
                        product=product,
                        work_dir=work_dir,
                        api_key=api_key,
                    )
                    feedback.setProgress(int(100 * (site_idx + 1) / len(sites)))
                else:
                    result = fetch_era5_point(
                        lat,
                        lon,
                        start_date=start_date,
                        end_date=end_date,
                        product=product,
                        years_per_chunk=years_per_chunk,
                        progress_callback=progress,
                        work_dir=work_dir,
                        api_key=api_key,
                    )
            except InterruptedError:
                feedback.pushInfo("Cancelled.")
                return {}
            except (ValueError, RuntimeError) as e:
                raise QgsProcessingException(str(e)) from e

            for w in result.warnings[:10]:
                feedback.pushWarning(f"[{label}] {w}")
            if len(result.warnings) > 10:
                feedback.pushWarning(
                    f"[{label}] ...and {len(result.warnings) - 10} more warnings "
                    f"(see full log)."
                )
            feedback.pushInfo(f"[{label}] {len(result.dataframe)} rows retrieved.")
            results_by_site[label] = result

        n_rows, n_missing = write_era5_csv(results_by_site, output_csv)
        feedback.pushInfo(f"{n_rows} total rows written to {output_csv}")

        if n_missing:
            feedback.pushWarning(
                f"{n_missing} row(s) had a missing (NaN) PrecipitationMM value - "
                f"written as an empty cell, not zero or dropped. This means the "
                f"source GRIB data had a genuine gap for those specific "
                f"timestamps - worth checking which timestamps before treating "
                f"the output as complete."
            )

        outputs = {self.OUTPUT_CSV: output_csv}
        anomalies_path = self.parameterAsFileOutput(
            parameters, self.OUTPUT_ANOMALIES, context
        )
        if anomalies_path:
            n_anomalies = write_era5_anomalies(results_by_site, anomalies_path)
            feedback.pushInfo(
                f"{n_anomalies} anomaly row(s) written to {anomalies_path}"
            )
            outputs[self.OUTPUT_ANOMALIES] = anomalies_path

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
        """One point per site, at the actual nearest-gridpoint coordinate
        the data was extracted from (result.matched_lat/matched_lon - see
        core.py's Era5Result) - the same location already reported in the
        'matched to the nearest grid cell' warning message, now also
        available as a vector layer. Mirrors design_rainfall_algorithm.py's
        _write_query_points pattern. Sites where no chunk/request
        succeeded (matched_lat/matched_lon still None) are skipped rather
        than written with a missing geometry."""
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
        Mirrors design_rainfall_algorithm.py's own _write_query_points."""
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
        design_rainfall_algorithm.py's own _register_style. The
        processor instance is kept alive on self._post_processors since
        Processing only holds a weak reference to it."""
        if not dest_id:
            return
        details = context.layerToLoadOnCompletionDetails(dest_id)
        processor = _StylePostProcessor(STYLES_DIR / style_filename)
        details.setPostProcessor(processor)
        self._post_processors.append(processor)
