"""Processing Toolbox algorithm: Curve numbers.

Land-cover raster + hydrologic soil groups (Hydrologic soil groups output,
or any group raster) + a land-cover-to-CN lookup -> a curve-number raster
(design, P05, P95, range, soil share of the uncertainty), zone summary with
runoff-equivalent curve numbers, metadata and a Word report. Thin wrapper -
the logic is in core.py (see tests/test_curve_numbers_core.py).
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
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools.soil._common import qgis_ui
from mayim_tools.soil._common.errors import SoilDataError

from . import core

LOOKUP_OPTIONS = [
    ("worldcover_scs_sa", "ESA WorldCover - SCS-SA Table 5.1 (southern Africa)"),
    ("worldcover_tr55", "ESA WorldCover - TR-55 / NEH 630"),
    ("custom", "Custom lookup CSV (below)"),
]
AMC_CODES = list(core.AMC_OPTIONS)
DUAL_OPTIONS = [
    ("undrained", "Undrained: use D"),
    ("drained", "Drained: use the first letter"),
]


class CurveNumbersAlgorithm(QgsProcessingAlgorithm):
    LANDCOVER = "LANDCOVER"
    HSG = "HSG"
    LOOKUP = "LOOKUP"
    LOOKUP_CSV = "LOOKUP_CSV"
    ZONES = "ZONES"
    ZONE_FIELD = "ZONE_FIELD"
    DESIGN_DEPTH = "DESIGN_DEPTH"
    AMC = "AMC"
    DELTA_S = "DELTA_S"
    USE_KSAT_RANGE = "USE_KSAT_RANGE"
    DUAL = "DUAL"
    IA_RATIO = "IA_RATIO"
    REPORT = "REPORT"
    LOAD_LAYERS = "LOAD_LAYERS"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    def __init__(self):
        super().__init__()
        self._post_processors = []

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return CurveNumbersAlgorithm()

    def name(self):
        return "curve_numbers"

    def displayName(self):
        return "Curve numbers"

    def group(self):
        return "Land Cover Tools"

    def groupId(self):
        return "land_cover_tools"

    def shortHelpString(self):
        return (
            f"Curve numbers (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tMaps SCS curve numbers (CN) from a land-cover raster and "
            "hydrologic soil groups through a published curve-number table, with "
            "the range the inputs allow, and gives zone values for lumped models.\n"
            "\n"
            "INPUTS:\tLand cover: e.g. worldcover_2021_map.tif from Extract: Land "
            "cover (or any class raster with a custom lookup). Soil groups: "
            "hsg_scssa_recommended.tif or hsg_neh630_recommended.tif from "
            "Hydrologic soil groups; the matching hsg_*_by_ksat.tif beside it is "
            "found automatically and adds the soil uncertainty. Any raster with "
            "codes 1-4 = A-D also works.\n"
            "\n"
            "LOOKUP:\tBuilt in: ESA WorldCover with SCS-SA Table 5.1 (Schulze, "
            "Schmidt & Smithers 2004; groups A, A/B, B, B/C, C, C/D, D) or with "
            "TR-55 / NEH 630 (A-D). Each class has a design row and a low / high "
            "row (e.g. veld fair; good and poor). Every run writes "
            "cn_lookup_used.csv: edit it and choose 'Custom lookup CSV' to use your "
            "own rows (also for SANLC or a hand-made land-cover map). Wetlands and "
            "mangroves are set to group D.\n"
            "\n"
            "UNCERTAINTY:\tP05 / P95 combine the soil range (CN at the group of the "
            "Ksat P5 and P95) and the condition range (low / high rows) by root-sum-"
            "square. The report says which dominates.\n"
            "\n"
            "ZONES:\tOptional polygons (sub-catchments). Each zone gets the area-"
            "weighted mean CN and the runoff-equivalent CN: the single CN that "
            "gives the zone's mean stormflow for the design rainfall depth (a "
            "plain mean CN understates runoff from mixed catchments).\n"
            "\n"
            "ANTECEDENT MOISTURE:\tNone (CN-II, default); ARC I / III (NEH 630 "
            "Ch. 10); Sobhani wet (SCS-SA Eq. 4); SCS-SA Median Condition (Eq. 3) "
            "with the change in storage for the catchment.\n"
            "\n"
            "OUTPUTS:\tcurve_numbers.tif (bands: design, P05, P95, range, soil "
            "share of the uncertainty %), cn_zone_summary.csv, cn_lookup_used.csv, "
            "cn_table_<table>.csv (the full table, for custom lookups), "
            "cn_metadata.csv, cn_layers.qlr and curve_numbers_report.docx. The "
            "output grid is the land-cover grid.\n"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.LANDCOVER, "Land cover raster (e.g. worldcover_2021_map.tif)"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.HSG,
                "Hydrologic soil groups raster (hsg_scssa_recommended.tif or "
                "hsg_neh630_recommended.tif)",
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.LOOKUP,
                "Land cover to curve number lookup",
                options=[label for _, label in LOOKUP_OPTIONS],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.LOOKUP_CSV,
                "Custom lookup CSV (layout of cn_lookup_used.csv)",
                extension="csv",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.ZONES,
                "Zones (polygons, e.g. sub-catchments) [optional]",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ZONE_FIELD,
                "Zone name field",
                parentLayerParameterName=self.ZONES,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DESIGN_DEPTH,
                "Design rainfall depth for the runoff-equivalent CN (mm)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=50.0,
                minValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.AMC,
                "Antecedent moisture adjustment",
                options=list(core.AMC_OPTIONS.values()),
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DELTA_S,
                "Change in storage dS for the Median Condition option (mm)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            qgis_ui.advanced(
                QgsProcessingParameterBoolean(
                    self.USE_KSAT_RANGE,
                    "Use the soil-group range (hsg_*_by_ksat.tif) when present",
                    defaultValue=True,
                )
            )
        )
        self.addParameter(
            qgis_ui.advanced(
                QgsProcessingParameterEnum(
                    self.DUAL,
                    "NEH 630 dual groups (A/D, B/D, C/D)",
                    options=[label for _, label in DUAL_OPTIONS],
                    defaultValue=0,
                )
            )
        )
        self.addParameter(
            qgis_ui.advanced(
                QgsProcessingParameterNumber(
                    self.IA_RATIO,
                    "Initial abstraction ratio Ia / S (0 = table default: 0.1 SCS-SA, "
                    "0.2 TR-55)",
                    type=QgsProcessingParameterNumber.Type.Double,
                    defaultValue=0.0,
                    minValue=0.0,
                    maxValue=0.5,
                )
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REPORT, "Write the Word report", defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.LOAD_LAYERS,
                "Load outputs into the project",
                options=qgis_ui.LOAD_OPTIONS,
                defaultValue=qgis_ui.LOAD_ALL,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, "Output folder")
        )

    # ------------------------------------------------------------------

    def _zones(self, parameters, context, crs_wkt, feedback):
        source = self.parameterAsSource(parameters, self.ZONES, context)
        if source is None or source.featureCount() == 0:
            return []
        name_field = self.parameterAsString(parameters, self.ZONE_FIELD, context)
        dst = QgsCoordinateReferenceSystem.fromWkt(crs_wkt)
        tr = QgsCoordinateTransform(source.sourceCrs(), dst, context.transformContext())
        zones, seen = [], {}
        for feat in source.getFeatures():
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            geom.transform(tr)
            name = str(feat[name_field]) if name_field else f"Zone {feat.id()}"
            if name in seen:
                seen[name] += 1
                name = f"{name} ({seen[name]})"
            else:
                seen[name] = 1
            zones.append((name, geom.asWkt()))
        if len(zones) > 50:
            feedback.pushWarning(f"{len(zones)} zones: the report tables will be long.")
        return zones

    def _raster_path(self, parameters, name, context):
        layer = self.parameterAsRasterLayer(parameters, name, context)
        if layer is None:
            raise QgsProcessingException(f"Choose a raster for '{name}'.")
        return layer.source()

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ia = self.parameterAsDouble(parameters, self.IA_RATIO, context)
        settings = core.RunSettings(
            landcover_path=self._raster_path(parameters, self.LANDCOVER, context),
            hsg_path=self._raster_path(parameters, self.HSG, context),
            out_dir=out_dir,
            lookup_key=LOOKUP_OPTIONS[
                self.parameterAsEnum(parameters, self.LOOKUP, context)
            ][0],
            lookup_csv=self.parameterAsFile(parameters, self.LOOKUP_CSV, context),
            use_ksat_range=self.parameterAsBoolean(
                parameters, self.USE_KSAT_RANGE, context
            ),
            dual=DUAL_OPTIONS[self.parameterAsEnum(parameters, self.DUAL, context)][0],
            amc=AMC_CODES[self.parameterAsEnum(parameters, self.AMC, context)],
            delta_s_mm=self.parameterAsDouble(parameters, self.DELTA_S, context),
            design_depth_mm=self.parameterAsDouble(
                parameters, self.DESIGN_DEPTH, context
            ),
            ia_ratio=ia if ia > 0 else None,
            write_report=self.parameterAsBoolean(parameters, self.REPORT, context),
        )
        if settings.amc == "median" and settings.delta_s_mm == 0:
            feedback.pushWarning(
                "Median Condition selected with a change in storage of 0 mm: the "
                "curve numbers are unchanged. Enter dS for the catchment (from Visual "
                "SCS-SA)."
            )
        from mayim_tools.soil._common.gdal_io import gdal_version

        try:
            grid = core.raster_grid(settings.landcover_path)
            settings.zones = self._zones(parameters, context, grid.crs_wkt, feedback)
            feedback.pushInfo(
                f"Land-cover grid {grid.width} x {grid.height} cells of {grid.res:g}; "
                f"{len(settings.zones)} zone(s)."
            )

            def progress(fraction, text=""):
                feedback.setProgress(int(100 * fraction))
                if text:
                    feedback.setProgressText(text)

            result = core.run(
                settings,
                progress_fn=progress,
                log_fn=feedback.pushWarning,
                cancel_fn=feedback.isCanceled,
            )
            core.write_outputs(result, gdal_version(), feedback.pushWarning)
        except SoilDataError as exc:
            raise QgsProcessingException(str(exc)) from exc
        except InterruptedError:
            feedback.pushWarning("Cancelled - outputs may be incomplete.")
            return {}
        eq_key = next(
            k
            for k in result.zone_rows[0]
            if k.startswith("Runoff-equivalent CN design")
        )
        for r in result.zone_rows:
            feedback.pushInfo(
                f"{r['Zone']}: runoff-equivalent CN {r[eq_key]:.1f} (P05 "
                f"{r['Runoff-equivalent CN P05']:.1f}, P95 {r['Runoff-equivalent CN P95']:.1f}); "
                f"mean CN {r['Mean CN design']:.1f}; confidence {r['Confidence']}."
            )
        feedback.pushInfo(f"Finished in {result.seconds:.0f} s; outputs in {out_dir}")
        if result.report_path:
            feedback.pushInfo(f"Report: {result.report_path}")
        outputs = [
            qgis_ui.OutputLayer(
                path=str(Path(out_dir) / core.OUT_FILE),
                name="Curve numbers",
                main=True,
                ramp=core.CN_RAMP,
            )
        ]
        qlr = str(Path(out_dir) / "cn_layers.qlr")
        if qgis_ui.write_layer_file(outputs, qlr, feedback):
            feedback.pushInfo(f"Layer file (reloads the output, styled): {qlr}")
        choice = self.parameterAsEnum(parameters, self.LOAD_LAYERS, context)
        qgis_ui.load_outputs(context, outputs, choice, self._post_processors)
        return {self.OUTPUT_FOLDER: out_dir}
