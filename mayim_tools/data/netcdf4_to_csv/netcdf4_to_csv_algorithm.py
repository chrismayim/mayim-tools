"""Processing Toolbox algorithm: NetCDF4 file in -> CSV out, via
either a full flatten of one variable or point extraction. The
NetCDF4 counterpart to the existing GRIB to CSV plugin.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_netcdf4_to_csv_core.py).
"""

from pathlib import Path

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterPoint,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from .core import DEFAULT_MAX_FLATTEN_ROWS, convert_netcdf4_to_csv, list_variables

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")
MODE_OPTIONS = [
    "Flatten (every value, all dimensions)",
    "Points (nearest-neighbour extraction)",
]


class Netcdf4ToCsvAlgorithm(QgsProcessingAlgorithm):
    INPUT_NC = "INPUT_NC"
    VARIABLE = "VARIABLE"
    MODE = "MODE"
    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    MAX_ROWS = "MAX_ROWS"
    DROP_NA = "DROP_NA"
    OUTPUT_CSV = "OUTPUT_CSV"

    def icon(self):
        logo = Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def createInstance(self):
        return Netcdf4ToCsvAlgorithm()

    def name(self):
        return "netcdf4_to_csv"

    def displayName(self):
        return "NetCDF4 to CSV"

    def group(self):
        return "Data Tools"

    def groupId(self):
        return "data_tools"

    def shortHelpString(self):
        return (
            "Converts a NetCDF4 file to CSV - the NetCDF4 counterpart to "
            "the GRIB to CSV plugin. Two modes: Flatten (every value in "
            "one variable, across all its dimensions, as a long-format "
            "table) or Points (nearest-neighbour extraction at one or "
            "more points, matching the pattern used by every other "
            "extraction plugin in this suite).\n"
            "\n"
            "Unlike every point-extraction plugin in this suite, this "
            "isn't tied to a specific known data source, so variable and "
            "coordinate names are resolved defensively (tries several "
            "common conventions - lat/latitude/y, lon/longitude/x) "
            "rather than assumed. The file's longitude convention "
            "(-180/180 vs 0-360) is detected automatically: if a "
            "requested point's longitude doesn't fall within the file's "
            "own coordinate range, the 0-360-converted equivalent is "
            "tried instead.\n"
            "\n"
            "Not sure which variable you want? Run this provider's other "
            "tool, 'List NetCDF4 Variables', first - it reports every "
            "variable's name, dimensions, units, and description without "
            "needing to guess or trigger an error. Flatten mode refuses "
            "BEFORE attempting a conversion that would produce an "
            "unusably large row count (default limit "
            f"{DEFAULT_MAX_FLATTEN_ROWS:,} rows, adjustable) rather than "
            "exhausting memory or hanging; Points mode is usually the "
            "better fit for time-series-style files.\n"
            "\n"
            "PARAMETERS:\n"
            "  Variable: leave blank to auto-detect if the file has "
            "exactly one; otherwise required.\n"
            "  Mode: Flatten or Points.\n"
            "  Point of interest / point layer: Points mode only.\n"
            "  Max rows (Flatten only): safety limit, default "
            f"{DEFAULT_MAX_FLATTEN_ROWS:,}.\n"
            "  Drop missing values (Flatten only): off by default."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_NC,
                "Input NetCDF4 file",
                extension="nc",
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE,
                "Mode (choose first - the fields below apply to one or the other)",
                options=MODE_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.VARIABLE,
                "Variable (blank = auto-detect if only one; use "
                "'List NetCDF4 Variables' to see options)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterPoint(
                self.POINT,
                "Point of interest (Points mode; click map or type coordinates)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_POINTS,
                "OR: point layer (Points mode; every feature processed)",
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
            QgsProcessingParameterNumber(
                self.MAX_ROWS,
                "Max rows (Flatten mode only)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=DEFAULT_MAX_FLATTEN_ROWS,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DROP_NA,
                "Drop missing values (Flatten mode only)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Output CSV",
                fileFilter="CSV files (*.csv)",
            )
        )

    def collect_points(self, parameters, context, feedback):
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
                "Points mode requires a point of interest (click the map "
                "canvas or type coordinates) OR a point layer - neither "
                "was supplied."
            )

        if has_layer:
            name_field = self.parameterAsString(parameters, self.NAME_FIELD, context)
            transform = QgsCoordinateTransform(
                source.sourceCrs(), WGS84, context.transformContext()
            )
            points = []
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
                points.append((label, pt.y(), pt.x()))
            return points

        return [("Point of interest", point.y(), point.x())]

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        input_path = self.parameterAsFile(parameters, self.INPUT_NC, context)
        variable = self.parameterAsString(parameters, self.VARIABLE, context) or None
        mode_idx = self.parameterAsEnum(parameters, self.MODE, context)
        is_flatten = mode_idx == 0
        max_rows = self.parameterAsInt(parameters, self.MAX_ROWS, context)
        drop_na = self.parameterAsBoolean(parameters, self.DROP_NA, context)
        output_path = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)

        feedback.pushInfo(f"Input file: {input_path}")
        feedback.pushInfo(f"Mode: {MODE_OPTIONS[mode_idx]}")

        if is_flatten:
            point = self.parameterAsPoint(parameters, self.POINT, context, crs=WGS84)
            point_source = self.parameterAsSource(
                parameters, self.INPUT_POINTS, context
            )
            has_point_input = (not point.isEmpty()) or (
                point_source is not None and point_source.featureCount() > 0
            )
            if has_point_input:
                feedback.pushInfo(
                    "Flatten mode selected - the Point of interest / point "
                    "layer / site name field you provided are ignored in "
                    "this mode (they only apply to Points mode). Switch "
                    "Mode to 'Points' to use them."
                )

        if not variable:
            try:
                available = list_variables(input_path)
            except Exception as e:
                raise QgsProcessingException(f"Could not open input file: {e}") from e
            if len(available) > 1:
                raise QgsProcessingException(
                    f"No variable specified, and this file has "
                    f"{len(available)} data variables - specify one "
                    f"explicitly. Available variables: {available}"
                )
            feedback.pushInfo(
                f"Auto-detected sole variable: "
                f"{available[0] if available else '(none found)'}"
            )

        points = None
        if not is_flatten:
            points = self.collect_points(parameters, context, feedback)
            feedback.pushInfo(f"{len(points)} point(s) to extract.")

        feedback.setProgress(10)
        try:
            info = convert_netcdf4_to_csv(
                input_path,
                output_path,
                variable=variable,
                mode="flatten" if is_flatten else "points",
                points=points,
                max_rows=max_rows,
                drop_na=drop_na,
            )
        except Exception as e:
            raise QgsProcessingException(str(e)) from e
        feedback.setProgress(100)

        feedback.pushInfo(
            f"Variable used: {info['variable']}. {info['rows_written']} "
            f"row(s) written to {output_path}"
        )
        return {self.OUTPUT_CSV: output_path}
