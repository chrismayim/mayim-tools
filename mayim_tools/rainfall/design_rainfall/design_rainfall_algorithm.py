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

Outputs follow the Mayim report and figure standard: the user ticks the
outputs wanted and chooses one folder per kind used (vector layers,
CSV, Word); files are named <project number>_<description> with
underscores; every layer written is loaded into the project with its
style.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from qgis.core import (
    Qgis,
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
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterPoint,
    QgsProcessingParameterString,
    QgsWkbTypes,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    SHAPEFILE,
    WORD,
    OutputSpec,
    clean_name_part,
)

from .core import (
    DAILY_DURATIONS,
    RETURN_PERIODS,
    SHORT_DURATIONS,
    DesignRainfallEngine,
    PointEstimate,
)
from .qgis_outputs import (
    grid_points_layer,
    load_on_completion,
    poi_layer,
    save_shapefile,
    stations_layer,
)
from .report import (
    REPORT_TITLE,
    write_csv,
    write_csv_multi,
    write_ddf_csv,
    write_report_docx,
)

TOOL_VERSION = "1.3.0"

WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

# Selectable outputs: key, dialog label, file-name description, folder
# kind, ticked by default.
OUTPUT_SPECS = [
    OutputSpec(
        "OUTPUT_LAYER", "Snapped grid point(s)", "Design Rainfall Grid Points",
        SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_QUERY_POINTS", "Point(s) of interest (POI)", "Design Rainfall POI",
        SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_STATIONS", "Nearest rainfall stations", "Design Rainfall Stations",
        SHAPEFILE, False,
    ),
    OutputSpec("OUTPUT_CSV", "Depth table (CSV)", "Design Rainfall Table", CSV),
    OutputSpec(
        "OUTPUT_DDF_CSV", "DDF table (CSV)", "Design Rainfall DDF Table", CSV,
    ),
    OutputSpec(
        "OUTPUT_DOCX", "Design rainfall report (Word)", "Design Rainfall Report",
        WORD, False,
    ),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}

# QML style file per vector output, in this tool's own styles/ folder.
STYLE_BY_KEY = {
    "OUTPUT_LAYER": "snapped_grid_point.qml",
    "OUTPUT_QUERY_POINTS": "poi.qml",
    "OUTPUT_STATIONS": "nearest_stations.qml",
}


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


class DesignRainfallPointAlgorithm(QgsProcessingAlgorithm):

    POINT = "POINT"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    DURATIONS = "DURATIONS"
    RETURN_PERIODS = "RETURN_PERIODS"
    REPORT_OSM = "REPORT_OSM"
    NUM_STATIONS = "NUM_STATIONS"

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
        files = "\n".join(
            f"  {spec.label}: <project number>_"
            f"{clean_name_part(spec.description)}{EXTENSIONS[spec.kind]}"
            for spec in OUTPUT_SPECS
        )
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
            "<i>Project number</i> - optional, used as the output file-name "
            "prefix and shown on the report's title page.<br>"
            "<i>Project / site name</i> - optional, shown on the report's "
            "title page only (not used in file names).<br>"
            "<i>Durations / Return periods</i> - the combinations to "
            "estimate.<br>"
            "<i>Nearest rainfall stations to list</i> - number of stations "
            "per site, reference only, not used in the calculation (0 = "
            "none).<br>"
            "<i>OpenStreetMap background</i> - adds OSM tiles behind the "
            "report map (needs internet access).<br><br>"
            "<b>OUTPUTS</b><br>"
            "Tick the outputs wanted (the snapped grid points, POI, "
            "depth table and DDF table are ticked by default) and choose an "
            "output folder for each kind used: vector layers (shapefiles), "
            "CSV and Word. "
            "Files are named &lt;project number&gt;_&lt;description&gt; with "
            "underscores (just &lt;description&gt; without a project "
            "number) and replace files of the same name; a file that is "
            "open in QGIS cannot be replaced - remove it from the project "
            "first (the other outputs are still written). Every layer "
            "written is loaded into the project with its style.<br>"
            + files.replace("\n", "<br>")
            + "<br><br>"
            "<i>Snapped grid point(s)</i> - the grid point(s) used, with the "
            "duration x return-period depths as attributes (field names "
            "over 10 characters are shortened for the shapefile).<br>"
            "<i>Point(s) of interest (POI)</i> - the requested location(s), "
            "independently of the snapped grid points.<br>"
            "<i>Nearest rainfall stations</i> - the stations listed above, "
            "as a point layer (reference only).<br>"
            "<i>Depth table (CSV)</i> - duration x return period table "
            "(depth, lower, upper), with the nearest stations appended.<br>"
            "<i>DDF table (CSV)</i> - the duration-depth-frequency table "
            "only: Duration and one '&lt;T&gt;yr Depth (mm)' column per return "
            "period, without the lower and upper bounds, the site details "
            "or the stations (a leading Site column with several sites).<br>"
            "<i>Design rainfall report (Word)</i> - annexure in the Mayim "
            "report standard: title page, summary, method, inputs, "
            "parameters, depth tables, a layout map per site (point of "
            "interest, snapped grid point and nearest stations, drawn with "
            "the tool's layer styles), the nearest rainfall stations "
            "(reference only), outputs, notes and references."
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
                "Project number (file name prefix, optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project / site name (optional, for the report title page)",
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
        report_osm = QgsProcessingParameterBoolean(
            self.REPORT_OSM,
            "Word report: OpenStreetMap background on the layout map "
            "(needs internet)",
            defaultValue=True,
        )
        report_osm.setFlags(report_osm.flags() | Qgis.ProcessingParameterFlag.Advanced)
        self.addParameter(report_osm)

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

        project_number = (
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        ).strip()

        selected = [
            spec
            for spec in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, spec.key, context)
        ]
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {spec.kind for spec in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {spec.key: spec.path(folders, project_number) for spec in selected}
        want = set(paths)

        engine = DesignRainfallEngine()
        estimates = {}
        for i, (label, lat, lon) in enumerate(sites):
            if feedback.isCanceled():
                break
            feedback.setProgress(50 * i / max(len(sites), 1))
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

        outputs = {
            name: folders[kind] for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }  # fmt: skip
        written = []  # (label, path) for the log and the report

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))

        def failed(key, path, error):
            feedback.reportError(
                f"{SPEC_BY_KEY[key].label}: {Path(path).name} could not be "
                f"written ({error}). If it is open in QGIS, remove it from "
                "the project and run again.",
                fatalError=False,
            )

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

        feedback.setProgress(50)
        feedback.setProgressText("Writing outputs")
        transform_context = context.transformContext()

        vector_layers = (
            ("OUTPUT_LAYER", lambda: grid_points_layer(estimates, WGS84)),
            ("OUTPUT_QUERY_POINTS", lambda: poi_layer(sites, WGS84)),
            ("OUTPUT_STATIONS", lambda: stations_layer(per_site_stations, WGS84)),
        )
        for key, build in vector_layers:
            if key not in want:
                continue
            layer = build()
            if layer is None:
                feedback.pushWarning(f"{SPEC_BY_KEY[key].label}: nothing to write.")
                continue
            path = paths[key]
            try:
                save_shapefile(layer, path, transform_context)
            except Exception as e:
                failed(key, path, e)
                continue
            load_on_completion(
                context, path, Path(path).stem, STYLE_BY_KEY[key],
                self._post_processors,
            )  # fmt: skip
            done(key, path)

        if "OUTPUT_CSV" in want:
            path = paths["OUTPUT_CSV"]
            try:
                if len(estimates) == 1:
                    label, est = next(iter(estimates.items()))
                    write_csv(est, path, stations=per_site_stations.get(label))
                else:
                    write_csv_multi(
                        estimates, path, stations_by_site=per_site_stations or None
                    )
                done("OUTPUT_CSV", path)
            except Exception as e:
                failed("OUTPUT_CSV", path, e)
        if "OUTPUT_DDF_CSV" in want and estimates:
            path = paths["OUTPUT_DDF_CSV"]
            try:
                write_ddf_csv(estimates, path)
                done("OUTPUT_DDF_CSV", path)
            except Exception as e:
                failed("OUTPUT_DDF_CSV", path, e)
        feedback.setProgress(70)

        if "OUTPUT_DOCX" in want and estimates:
            self._write_report(
                parameters,
                context,
                feedback,
                paths["OUTPUT_DOCX"],
                estimates,
                per_site_stations,
                num_stations,
                written,
                project_number,
                done,
                failed,
            )

        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        feedback.setProgress(100)
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
        written,
        project_number,
        done,
        failed,
    ):
        """Word report with a QGIS-rendered layout map per site."""
        try:
            import docx  # noqa: F401
        except ImportError:
            feedback.reportError(
                "python-docx not installed - skipping Word output. Install "
                "it from the OSGeo4W Shell with: python -m pip install "
                "python-docx."
            )
            return
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
            "project_number": project_number,
            "project_name": self.parameterAsString(
                parameters, self.PROJECT_NAME, context
            ).strip(),
            "num_stations": num_stations,
            "outputs": written + [("This report", docx_path)],
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
        except PermissionError as e:
            failed("OUTPUT_DOCX", docx_path, e)
            return
        except Exception as e:  # a report failure must not lose the other outputs
            feedback.pushWarning(f"The Word report could not be written: {e}")
            return
        for message in dict.fromkeys(map_messages + (messages or [])):
            feedback.pushWarning(message)
        done("OUTPUT_DOCX", docx_path)
