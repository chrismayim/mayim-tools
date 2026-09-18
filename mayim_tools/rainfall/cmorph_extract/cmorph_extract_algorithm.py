"""Processing Toolbox algorithm: point of interest in -> NOAA CMORPH
CDR precipitation time series out, as CSV.

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
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterPoint,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from .core import DEFAULT_MAX_WORKERS, RECORD_START, fetch_cmorph_timeseries

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")


class CmorphExtractAlgorithm(QgsProcessingAlgorithm):

    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    MAX_WORKERS = "MAX_WORKERS"
    OUTPUT_CSV = "OUTPUT_CSV"

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return CmorphExtractAlgorithm()

    def name(self):
        return "cmorph_point_extract"

    def displayName(self):
        return "Extract: CMORPH precipitation"

    def group(self):
        return "Rainfall Tools"

    def groupId(self):
        return "rainfall_tools"

    def shortHelpString(self):
        return (
            "CMORPH CDR Point Extractor (version 0.3.0)\n"
            "\n"
            "PURPOSE:\tExtracts the full available NOAA CMORPH CDR "
            "(bias-adjusted, Climate Data Record quality) precipitation "
            "record (1998-01-01 to today, native 8km/30-minute "
            "resolution, 60S-60N coverage) at a point, for a specified "
            "date range or the full record.\n"
            "\n"
            "METHOD:\t\tReads directly from public, UNAUTHENTICATED "
            "files on Azure Blob Storage via NOAA's Open Data "
            "Dissemination (NODD) Program - confirmed via a real, live, "
            "unauthenticated fetch of the data container's listing "
            "before this tool was built, not assumed. No bulk time-"
            "series API exists for this NOAA product (unlike the NASA/"
            "GES-DISC Giovanni-based IMERG and MERRA-2 Point Extractor "
            "plugins), so requests are fetched CONCURRENTLY, one file "
            "per hour - the same architecture already used by the "
            "CHIRPS Point Extractor plugin. A full record request "
            "involves roughly 245,000 individual hourly files.\n"
            "\n"
            "BACKGROUND:\tNo authentication required - genuinely the "
            "simplest credential story of any plugin in this suite. "
            "CONFIRMED LIVE (v0.2.0): each hourly file holds exactly two "
            "timesteps, matching its '30min' resolution label - verified "
            "by an exact row-count match on a real 5-year extraction "
            "(87,744 rows = 1,828 days x 24 hours x 2, with zero fetch "
            "failures). The exact NetCDF variable name is still handled "
            "defensively (tries several plausible names, falling back to "
            "the file's sole data variable if none match) rather than "
            "hardcoded. CMORPH's own documentation states the output is "
            "already a rate in mm/hr for whatever interval it represents "
            "- no deaccumulation is applied, similar to MERRA-2, unlike "
            "ERA5. MISSING VALUES: a small fraction of timesteps (0.23% "
            "on that same real run) can have no value at the target "
            "point even though the file fetched and read successfully - "
            "the grid cell itself was missing/masked in the source data. "
            "Reported as a warning with a count and percentage, written "
            "as an empty CSV cell, not zero or dropped - plausibly a "
            "genuine characteristic of a satellite-derived product, but "
            "not confirmed as such; worth checking whether missing "
            "timestamps cluster on specific dates before assuming it's "
            "expected. PERFORMANCE (v0.3.0): a real 5-year run took ~4 "
            "hours - every fetch now reuses a shared, pooled HTTPS "
            "connection instead of opening a fresh one per file (a real "
            "measured inefficiency across the ~43,872 files a full-"
            "record run makes), which should meaningfully cut this. "
            "Concurrency above the default has NOT been tested against "
            "this server - Azure Blob Storage can throttle ('hot "
            "partition' 503s) under heavy sustained load against "
            "sequentially-named blobs, which this dataset's date-based "
            "naming resembles; raise Concurrent requests cautiously, "
            "not aggressively.\n"
            "\n"
            "PARAMETERS:\n"
            "  Start/End date: leave blank for the full available "
            "record.\n"
            "  Concurrent requests: default 8 - higher is faster but "
            "places more load on the shared public server."
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
        sites = self._collect_sites(parameters, context, feedback)
        feedback.pushInfo(f"{len(sites)} site(s) to process.")

        start_date = (
            self.parameterAsString(parameters, self.START_DATE, context) or None
        )
        end_date = self.parameterAsString(parameters, self.END_DATE, context) or None
        max_workers = self.parameterAsInt(parameters, self.MAX_WORKERS, context)
        output_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        all_rows = []
        for site_idx, (label, lat, lon) in enumerate(sites):
            feedback.pushInfo(f"[{label}] lat={lat:.4f}, lon={lon:.4f}")

            def _progress(pct, _label=label, _idx=site_idx):
                if feedback.isCanceled():
                    raise InterruptedError("Cancelled by user")
                overall = int(100 * (_idx + pct / 100) / len(sites))
                feedback.setProgress(overall)

            try:
                result = fetch_cmorph_timeseries(
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
                    f"[{label}] ...and {len(result.warnings) - 10} more "
                    f"warnings (see full log)."
                )

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

        return {self.OUTPUT_CSV: output_csv}
