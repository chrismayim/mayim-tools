"""Processing Toolbox algorithm: Hydrologic soil groups.

Reads the output folder of 'Regional soil parameterisation' (Ksat P5 / P50
/ P95 per layer) and writes hydrologic soil groups by USDA-NRCS NEH 630
Chapter 7 (2009) and / or the SCS-SA groups (Schulze, Schmidt & Smithers,
2004), with group probabilities, a recommended group and its confidence,
a zone summary, metadata and a Word report. Thin wrapper - the logic is in
core.py (no QGIS; see tests/test_hydrologic_soil_groups_core.py).
"""

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
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools.soil._common import qgis_ui
from mayim_tools.soil._common.errors import SoilDataError

from . import core

METHOD_CODES = ["NEH630", "SCSSA"]
BEDROCK_CODES = ["BDRICM", "BDTICM", "raster", "constant", "none"]
WATER_CODES = ["none", "raster", "constant"]


def _folder_behavior():
    try:
        return Qgis.ProcessingFileParameterBehavior.Folder
    except AttributeError:  # pragma: no cover - older QGIS
        return QgsProcessingParameterFile.Behavior.Folder


class HydrologicSoilGroupsAlgorithm(QgsProcessingAlgorithm):
    REGIONAL_FOLDER = "REGIONAL_FOLDER"
    METHODS = "METHODS"
    BEDROCK_SOURCE = "BEDROCK_SOURCE"
    SG_FOLDER = "SG_FOLDER"
    BEDROCK_RASTER = "BEDROCK_RASTER"
    BEDROCK_CONSTANT = "BEDROCK_CONSTANT"
    WATER_SOURCE = "WATER_SOURCE"
    WATER_RASTER = "WATER_RASTER"
    WATER_CONSTANT = "WATER_CONSTANT"
    SA_SHALLOW = "SA_SHALLOW"
    SA_WATER = "SA_WATER"
    ZONES = "ZONES"
    ZONE_FIELD = "ZONE_FIELD"
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
        return HydrologicSoilGroupsAlgorithm()

    def name(self):
        return "hydrologic_soil_groups"

    def displayName(self):
        return "Hydrologic soil groups"

    def group(self):
        return "Soil Tools"

    def groupId(self):
        return "soil_tools"

    def shortHelpString(self):
        return (
            f"Hydrologic soil groups (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tAssigns hydrologic soil groups for curve-number runoff "
            "methods from the saturated hydraulic conductivity (Ksat) of "
            "'Regional soil parameterisation', the depth to a water impermeable "
            "layer and the depth to the water table - with the probability of "
            "each group, a recommended group and its confidence. Curve numbers "
            "are a separate step.\n"
            "\n"
            "METHODS:\tNEH 630: USDA-NRCS National Engineering Handbook Part 630, "
            "Chapter 7 (2009), Table 7-1 - impermeable layer < 50 cm gives D; "
            "otherwise the Ksat of the least transmissive layer in 0-50, 0-60 or "
            "0-100 cm against 144 / 36 / 3.6 mm/h (or 36 / 14.4 / 1.44 mm/h for "
            "deep soils); a water table < 60 cm gives the dual groups A/D, B/D, "
            "C/D (drained / undrained).\n"
            "SCS-SA: Schulze, Schmidt & Smithers (2004), Visual SCS-SA User "
            "Manual, Table 2.1 - permeability A > 7.6, B 3.8-7.6, C 1.3-3.8, "
            "D < 1.3 mm/h (least transmissive layer within 0-100 cm), with the "
            "southern African intermediate groups A/B, B/C and C/D where the "
            "Ksat range straddles a boundary (both groups at least 35 % likely), "
            "and optional field adjustments of Section 2.2.3(d): one step down "
            "for a shallow phase (impermeable layer < 50 cm) and for a water "
            "table < 60 cm. Where the soil form and series are known, the "
            "manual's own series assignments take precedence.\n"
            "The two methods use very different Ksat limits, so the same soil "
            "can fall one or two groups apart: use the method that matches the "
            "curve-number tables you will apply.\n"
            "\n"
            "UNCERTAINTY:\tThe Ksat P5 / P50 / P95 give the probability of each "
            "group. The recommended group is the group of the median Ksat; its "
            "confidence is the probability of that group (High >= 80 %, Medium "
            "60-80 %, Low < 60 %). The groups at the Ksat P5 (more runoff) and "
            "P95 (less runoff) are written too.\n"
            "\n"
            "INPUTS:\tRegional soil parameterisation output folder (rsp_ksat.tif; "
            "rsp_inputs_P50.tif for an NEH 630 texture cross-check). Depth to an "
            "impermeable layer: SoilGrids 2017 depth to bedrock (run 'Extract: "
            "SoilGrids 2.0' with the depth-to-bedrock layer and give its folder), "
            "a raster in metres, a constant, or none (> 100 cm). Depth to the "
            "water table: a raster in metres, a constant, or none (> 100 cm; no "
            "global map is reliable enough). Optional zones (sub-catchments).\n"
            "\n"
            "OUTPUTS:\thsg_<method>_recommended.tif (group, confidence %, "
            "confidence class), hsg_<method>_by_ksat.tif (group at Ksat P5, P50, "
            "P95), hsg_<method>_probability.tif (P of A, B, C, D), "
            "hsg_conditions.tif (depths, NEH 630 case, least transmissive Ksat, "
            "texture group), hsg_zone_summary.csv (group shares per zone - area "
            "fractions for curve numbers), hsg_metadata.csv, the Word report and "
            "hsg_layers.qlr. Group codes: 1-4 A-D, 11-13 NEH 630 dual A/D-C/D, "
            "21-23 SCS-SA intermediate A/B, B/C, C/D.\n"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.REGIONAL_FOLDER,
                "Regional soil parameterisation output folder",
                behavior=_folder_behavior(),
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHODS,
                "Methods",
                options=[core.METHODS[c] for c in METHOD_CODES],
                allowMultiple=True,
                defaultValue=[0, 1],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.BEDROCK_SOURCE,
                "Depth to a water impermeable layer",
                options=[core.BEDROCK_SOURCES[c] for c in BEDROCK_CODES],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.SG_FOLDER,
                "SoilGrids output folder with the depth-to-bedrock layer",
                behavior=_folder_behavior(),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.BEDROCK_RASTER,
                "Impermeable-layer depth raster (m) [optional]",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BEDROCK_CONSTANT,
                "Constant impermeable-layer depth (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=2.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WATER_SOURCE,
                "Depth to the high water table",
                options=[core.WATER_SOURCES[c] for c in WATER_CODES],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.WATER_RASTER,
                "Water-table depth raster (m) [optional]",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.WATER_CONSTANT,
                "Constant water-table depth (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=2.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SA_SHALLOW,
                "SCS-SA: one group down for an impermeable layer < 50 cm",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SA_WATER,
                "SCS-SA: one group down for a water table < 60 cm",
                defaultValue=True,
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
            QgsProcessingParameterBoolean(
                self.REPORT, "Write the Word report", defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.LOAD_LAYERS,
                "Load outputs into the project",
                options=qgis_ui.LOAD_OPTIONS,
                defaultValue=qgis_ui.LOAD_MAIN,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(self.OUTPUT_FOLDER, "Output folder")
        )

    # ------------------------------------------------------------------

    def _zones(self, parameters, context, grid, feedback):
        source = self.parameterAsSource(parameters, self.ZONES, context)
        if source is None or source.featureCount() == 0:
            return []
        name_field = self.parameterAsString(parameters, self.ZONE_FIELD, context)
        dst = QgsCoordinateReferenceSystem.fromWkt(grid.crs_wkt)
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
        return layer.source() if layer is not None else ""

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        chosen = self.parameterAsEnums(parameters, self.METHODS, context)
        if not chosen:
            raise QgsProcessingException("Select at least one method.")
        settings = core.RunSettings(
            regional_folder=self.parameterAsFile(
                parameters, self.REGIONAL_FOLDER, context
            ),
            out_dir=out_dir,
            methods=tuple(METHOD_CODES[i] for i in chosen),
            bedrock_source=BEDROCK_CODES[
                self.parameterAsEnum(parameters, self.BEDROCK_SOURCE, context)
            ],
            sg_folder=self.parameterAsFile(parameters, self.SG_FOLDER, context),
            bedrock_raster=self._raster_path(parameters, self.BEDROCK_RASTER, context),
            bedrock_constant_m=self.parameterAsDouble(
                parameters, self.BEDROCK_CONSTANT, context
            ),
            water_source=WATER_CODES[
                self.parameterAsEnum(parameters, self.WATER_SOURCE, context)
            ],
            water_raster=self._raster_path(parameters, self.WATER_RASTER, context),
            water_constant_m=self.parameterAsDouble(
                parameters, self.WATER_CONSTANT, context
            ),
            sa_adjust_shallow=self.parameterAsBoolean(
                parameters, self.SA_SHALLOW, context
            ),
            sa_adjust_water=self.parameterAsBoolean(parameters, self.SA_WATER, context),
            write_report=self.parameterAsBoolean(parameters, self.REPORT, context),
        )
        try:
            ksat_path = str(Path(settings.regional_folder) / core.KSAT_FILE)
            if not Path(ksat_path).is_file():
                raise SoilDataError(
                    f"{core.KSAT_FILE} not found in {settings.regional_folder}. "
                    "Point the tool at the output folder of 'Regional soil "
                    "parameterisation'."
                )
            grid = core.grid_of(ksat_path)
            settings.zones = self._zones(parameters, context, grid, feedback)

            def progress(fraction, text=""):
                feedback.setProgress(int(100 * fraction))

            from mayim_tools.soil._common.export import write_multiband_geotiff

            result = core.run(
                settings,
                log_fn=feedback.pushInfo,
                progress_fn=progress,
                write_fn=write_multiband_geotiff,
            )
        except SoilDataError as exc:
            raise QgsProcessingException(str(exc)) from exc
        for w in result.warnings:
            feedback.pushWarning(w)
        feedback.pushInfo(f"Finished in {result.seconds:.0f} s; outputs in {out_dir}")
        if result.report_path:
            feedback.pushInfo(f"Report: {result.report_path}")

        outputs = [
            qgis_ui.OutputLayer(
                path=o["path"], name=o["name"], main=o["main"], classes=o["classes"]
            )
            for o in core.output_layers(result)
            if Path(o["path"]).is_file()
        ]
        qlr = str(Path(out_dir) / "hsg_layers.qlr")
        if qgis_ui.write_layer_file(outputs, qlr, feedback):
            feedback.pushInfo(f"Layer file (reloads all outputs, styled): {qlr}")
        choice = self.parameterAsEnum(parameters, self.LOAD_LAYERS, context)
        qgis_ui.load_outputs(context, outputs, choice, self._post_processors)
        return {self.OUTPUT_FOLDER: out_dir}
