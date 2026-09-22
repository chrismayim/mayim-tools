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
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterPoint,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from .core import DATASETS, ERA5_RECORD_START, fetch_era5_point
from .export import write_era5_anomalies, write_era5_csv

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")


class Era5ExtractAlgorithm(QgsProcessingAlgorithm):
    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    PRODUCT = "PRODUCT"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    YEARS_PER_CHUNK = "YEARS_PER_CHUNK"
    CDS_API_KEY = "CDS_API_KEY"
    OUTPUT_CSV = "OUTPUT_CSV"
    OUTPUT_ANOMALIES = "OUTPUT_ANOMALIES"

    PRODUCT_OPTIONS = list(DATASETS)  # ["era5", "era5_land"]

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
            "Extract: ERA5 precipitation (version 0.4.0)\n"
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
            "METHOD:\t\tRequests total_precipitation from the Copernicus "
            "Climate Data Store (CDS) via the official ecmwf/cdsapi client, "
            "in GRIB format (not NetCDF - multiple currently-open ECMWF "
            "forum threads report their new GRIB-to-NetCDF conversion "
            "pipeline dropping or corrupting precipitation data "
            "specifically). A single CDS request can span many years of "
            "hourly data at once, so a full record needs only a handful of "
            "requests (see 'Years per request'), not one per timestep. "
            "Each hour's precipitation value, exactly as delivered by this "
            "specific request (queried by valid time, not forecast lead-"
            "time step), is already the correct, final figure for that "
            "hour - no further processing is applied to it. An earlier "
            "version of this tool de-accumulated every value against its "
            "own preceding hour, on the assumption that ERA5 delivers a "
            "running cumulative total needing differencing - a real, "
            "substantial bug (confirmed directly against the CDS website's "
            "own reported values for the same hours) that silently "
            "understated true rainfall and produced spurious negative "
            "values whenever two consecutive hours both had real rain.\n"
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
            "step this tool cannot do on your behalf. Unlike IMERG or "
            "CHIRPS, ERA5 and ERA5-Land are combined into this single tool "
            "rather than kept as separate plugins, since both are served "
            "through the exact same CDS infrastructure - the only real "
            "difference is the dataset name requested.\n"
            "\n"
            "PARAMETERS:\n"
            "  Product: ERA5 (recommended default for precipitation) or "
            "ERA5-Land (see PURPOSE for why this isn't simply 'higher "
            "resolution' for rainfall specifically).\n"
            "  Start/End date: leave blank for the full available record "
            "(1950-01-01 to today).\n"
            "  Years per request: how many years of hourly data to request "
            "from CDS in a single call (default 1) - a deliberate safety "
            "margin against undocumented CDS request size limits, not a "
            "strict requirement; increasing it reduces the number of "
            "requests but each one takes longer and risks a larger failure "
            "if something goes wrong partway through.\n"
            "  CDS API Key: enter this ONCE - it's saved to ~/.cdsapirc "
            "automatically and every later run reads it from there, so "
            "leave this blank after the first successful run."
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
                "Years per CDS request",
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
        sites = self.collect_sites(parameters, context, feedback)
        feedback.pushInfo(f"{len(sites)} site(s) to process.")

        product = self.PRODUCT_OPTIONS[
            self.parameterAsEnum(parameters, self.PRODUCT, context)
        ]
        start_date = (
            self.parameterAsString(parameters, self.START_DATE, context) or None
        )
        end_date = self.parameterAsString(parameters, self.END_DATE, context) or None
        years_per_chunk = self.parameterAsInt(parameters, self.YEARS_PER_CHUNK, context)
        api_key = self.parameterAsString(parameters, self.CDS_API_KEY, context) or None
        output_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        work_dir = tempfile.mkdtemp(prefix="era5_extract_")
        feedback.pushInfo(f"Product: {product}. Working directory: {work_dir}")

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

        return outputs
