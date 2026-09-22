"""Processing Toolbox algorithm: point of interest in -> PERSIANN
precipitation time series out, as CSV. Product parameter selects
between daily PERSIANN-CDR (Azure NODD, one file per day) and 3-hourly
PERSIANN-CCS-CDR (CHRS HTTPS, one file per 3 hours) - both direct
per-file fetches, the same architecture proven for CMORPH.

Point input mirrors every other extraction plugin in this suite
exactly: either a single point (map click or typed coordinates) or a
point vector layer (every feature processed). Thin wrapper - all real
logic lives in core.py (zero QGIS dependency, independently testable;
see tests/test_core.py).
"""

from pathlib import Path

import pandas as pd
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

from .core import (
    DEFAULT_MAX_WORKERS,
    RECORD_START,
    fetch_persiann_3hourly,
    fetch_persiann_daily,
)

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

PRODUCT_OPTIONS = [
    "Daily (PERSIANN-CDR, 0.25 deg)",
    "3-Hourly (PERSIANN-CCS-CDR, 0.04 deg / ~4km)",
]


class PersiannExtractAlgorithm(QgsProcessingAlgorithm):

    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    PRODUCT = "PRODUCT"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    MAX_WORKERS = "MAX_WORKERS"
    OUTPUT_CSV = "OUTPUT_CSV"

    def createInstance(self):
        return PersiannExtractAlgorithm()

    def name(self):
        return "persiann_point_extract"

    def displayName(self):
        return "Extract: PERSIANN CDR precipitation"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def shortHelpString(self):
        return (
            "PERSIANN Point Extractor (version 0.2.0)\n"
            "\n"
            "PURPOSE:\tExtracts PERSIANN precipitation at a point, for "
            "a specified date range or the full available record "
            "(1983-01-01 to today for both products). Two products, "
            "selectable: daily PERSIANN-CDR (0.25 deg) or 3-hourly "
            "PERSIANN-CCS-CDR (0.04 deg, ~4km).\n"
            "\n"
            "METHOD:\t\tBoth products use direct concurrent per-file "
            "fetching - the same architecture already proven for the "
            "CMORPH CDR Point Extractor plugin. Daily (PERSIANN-CDR) "
            "reads directly from NOAA's Open Data Dissemination (NODD) "
            "program on Azure Blob Storage: one file per DAY (roughly "
            "15,700 files for the full record - far fewer than the "
            "3-hourly product needs, since it's one file per day, not "
            "per hour). Each requested YEAR's file listing is fetched "
            "once and used to look up the real filename for each date "
            "(the exact filename isn't predictable from the date alone "
            "- see BACKGROUND). 3-Hourly (PERSIANN-CCS-CDR) reads "
            "directly from CHRS's own HTTPS server, one file per "
            "3-hour period - roughly 125,000 files for a full-record "
            "extraction. Both products use a 0-360 degree longitude "
            "convention internally - handled automatically.\n"
            "\n"
            "BACKGROUND:\tNo authentication required for either "
            "product. The daily product originally used NOAA NCEI's "
            "ERDDAP server (a single query can in principle answer a "
            "whole date range at once) - two live attempts, with an "
            "increasingly generous timeout and a proper browser "
            "User-Agent added between them, both failed identically: a "
            "read timeout at the full configured timeout, on the "
            "smallest possible query. Switched to direct Azure file "
            "access after that repeated, consistent failure, rather "
            "than continuing to tune ERDDAP parameters. IMPORTANT, "
            "stated honestly: the daily files' exact NetCDF variable "
            "name was not confirmed against a live download when this "
            "was built (handled defensively - tries 'precipitation' "
            "first, confirmed from this dataset's own ERDDAP metadata, "
            "then falls back to the file's sole data variable); the "
            "3-hourly product's missing-value convention is similarly "
            "unconfirmed. See this plugin's dev/README.md for full "
            "detail on both.\n"
            "\n"
            "PARAMETERS:\n"
            "  Product: Daily or 3-Hourly - see METHOD above for the "
            "scale difference.\n"
            "  Start/End date: leave blank for the full available "
            "record.\n"
            "  Concurrent requests: applies to both products - default 8."
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
                options=PRODUCT_OPTIONS,
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
                "End date (blank = today)",
                optional=True,
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
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Output CSV",
                fileFilter="CSV files (*.csv)",
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
                "Provide a point of interest (click the map canvas "
                "or type coordinates) OR a point layer - neither "
                "was supplied."
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
        sites = self._collect_sites(parameters, context, feedback)
        feedback.pushInfo(f"{len(sites)} site(s) to process.")

        product_idx = self.parameterAsEnum(parameters, self.PRODUCT, context)
        is_daily = product_idx == 0
        start_date = (
            self.parameterAsString(parameters, self.START_DATE, context) or None
        )
        end_date = self.parameterAsString(parameters, self.END_DATE, context) or None
        max_workers = self.parameterAsInt(parameters, self.MAX_WORKERS, context)
        output_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        feedback.pushInfo(f"Product: {PRODUCT_OPTIONS[product_idx]}")

        all_rows = []
        for site_idx, (label, lat, lon) in enumerate(sites):
            feedback.pushInfo(f"[{label}] lat={lat:.4f}, lon={lon:.4f}")

            def _progress(pct, _label=label, _idx=site_idx):
                if feedback.isCanceled():
                    raise InterruptedError("Cancelled by user")
                overall = int(100 * (_idx + pct / 100) / len(sites))
                feedback.setProgress(overall)

            try:
                if is_daily:
                    result = fetch_persiann_daily(
                        lat,
                        lon,
                        start_date=start_date,
                        end_date=end_date,
                        progress_callback=_progress,
                        max_workers=max_workers,
                    )
                else:
                    result = fetch_persiann_3hourly(
                        lat,
                        lon,
                        start_date=start_date,
                        end_date=end_date,
                        progress_callback=_progress,
                        max_workers=max_workers,
                    )
            except InterruptedError:
                feedback.pushInfo("Cancelled.")
                return {}

            for w in result.warnings[:10]:
                feedback.pushWarning(f"[{label}] {w}")
            if len(result.warnings) > 10:
                feedback.pushWarning(
                    f"[{label}] ...and {len(result.warnings) - 10} "
                    f"more warnings (see full log)."
                )

            feedback.pushInfo(f"[{label}] {len(result.dataframe)} rows retrieved.")

            df = result.dataframe.copy()
            df.insert(0, "Site", label)
            all_rows.append(df)

        if not all_rows or all(len(df) == 0 for df in all_rows):
            feedback.pushWarning("No data retrieved for any site.")

        time_col = "Date" if is_daily else "Timestamp"
        combined = (
            pd.concat(all_rows, ignore_index=True)
            if all_rows
            else pd.DataFrame(columns=["Site", time_col, "PrecipitationMM"])
        )
        combined.to_csv(output_csv, index=False)
        feedback.pushInfo(f"{len(combined)} total rows written to {output_csv}")

        return {self.OUTPUT_CSV: output_csv}
