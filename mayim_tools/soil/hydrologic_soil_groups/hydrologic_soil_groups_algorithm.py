"""Processing Toolbox algorithm: Hydrological Soil Groups (HSG).

Reads the output folder of 'Regional Soil Parameterisation' (Ksat P5 / P50
/ P95 per layer) and writes hydrologic soil groups by USDA-NRCS NEH 630
Chapter 7 (2009) and / or the SCS-SA groups (Schulze, Schmidt & Smithers,
2004), with group probabilities, a recommended group and its confidence,
a zone summary, metadata and a Word report. Thin wrapper - the logic is in
core.py (no QGIS; see tests/test_hydrologic_soil_groups_core.py).

Outputs follow the Mayim output standard: a tick box per output, raster,
CSV and Word folders, files named <project number>_<description>.
"""

import os
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
    QgsProcessingParameterString,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    RASTER,
    WORD,
    OutputSpec,
    clean_name_part,
)
from mayim_tools.soil._common import qgis_ui
from mayim_tools.soil._common.errors import SoilDataError

from . import core

METHOD_CODES = ["NEH630", "SCSSA"]
BEDROCK_CODES = ["BDRICM", "BDTICM", "ISDA", "raster", "constant", "none"]
WATER_CODES = ["none", "raster", "constant"]

# Selectable outputs: core output key, dialog label, file-name description,
# folder kind, ticked by default. The per-method rasters are named
# <project number>_HSG_<METHOD>_<description>.tif for each chosen method.
OUTPUT_SPECS = [
    OutputSpec(
        core.OUT_RECOMMENDED, "Recommended groups and confidence (raster, per method)",
        "Recommended", RASTER,
    ),
    OutputSpec(
        core.OUT_BY_KSAT,
        "Groups at Ksat P5 / P50 / P95 (raster, per method; used by SCS Curve Numbers)",
        "By Ksat", RASTER,
    ),
    OutputSpec(
        core.OUT_PROBABILITY, "Group probabilities (raster, per method)",
        "Probability", RASTER,
    ),
    OutputSpec(
        core.OUT_CONDITIONS, "Conditions: depths, table case, least transmissive "
        "Ksat (raster)", "HSG Conditions", RASTER,
    ),
    OutputSpec(core.OUT_ZONES, "Zone summary (CSV)", "HSG Zone Summary", CSV),
    OutputSpec(core.OUT_METADATA, "Run metadata (CSV)", "HSG Metadata", CSV),
    OutputSpec(core.OUT_REPORT, "Word report", "HSG Report", WORD),
]  # fmt: skip
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}


def output_param(key: str) -> str:
    return f"OUTPUT_{key.upper()}"


def method_spec(spec: OutputSpec, method: str) -> OutputSpec:
    """The per-method version of a per-method spec."""
    short = core.METHOD_SHORT[method].upper()
    return OutputSpec(
        spec.key, spec.label, f"HSG {short} {spec.description}", spec.kind
    )


def _software_version() -> str:
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
    ISDA_FOLDER = "ISDA_FOLDER"
    BEDROCK_RASTER = "BEDROCK_RASTER"
    BEDROCK_CONSTANT = "BEDROCK_CONSTANT"
    WATER_SOURCE = "WATER_SOURCE"
    WATER_RASTER = "WATER_RASTER"
    WATER_CONSTANT = "WATER_CONSTANT"
    SA_SHALLOW = "SA_SHALLOW"
    SA_WATER = "SA_WATER"
    ZONES = "ZONES"
    ZONE_FIELD = "ZONE_FIELD"
    LOAD_LAYERS = "LOAD_LAYERS"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"

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
        return "Hydrological Soil Groups (HSG)"

    def group(self):
        return "Soil Tools"

    def groupId(self):
        return "soil_tools"

    def shortHelpString(self):
        files = "\n".join(
            f"  {s.label}: <project number>_"
            + (
                f"HSG_<METHOD>_{clean_name_part(s.description)}"
                if s.key in core.PER_METHOD
                else clean_name_part(s.description)
            )
            + EXTENSIONS[s.kind]
            for s in OUTPUT_SPECS
        )
        return (
            f"Hydrological Soil Groups (HSG), version {core.TOOL_VERSION}\n"
            "\n"
            "PURPOSE:\tAssigns hydrologic soil groups for curve-number runoff "
            "methods from the saturated hydraulic conductivity (Ksat) of "
            "'Regional Soil Parameterisation', the depth to a water impermeable "
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
            "INPUTS:\tRegional Soil Parameterisation raster output folder. The "
            "Ksat raster (e.g. <project>_Soil_Ksat.tif, or rsp_ksat.tif from "
            "older runs) and the harmonised inputs raster (for an NEH 630 texture "
            "cross-check) are found by their band descriptions; when several runs "
            "share the folder the newest is used. Depth to an "
            "impermeable layer: SoilGrids 2017 depth to bedrock (run 'Extract: "
            "SoilGrids 2.0' with the depth-to-bedrock layer and give its folder), "
            "iSDAsoil depth to bedrock with its standard deviation (Africa; the "
            "confidence then includes the chance that bedrock lies in another "
            "depth class), "
            "a raster in metres, a constant, or none (> 100 cm). Depth to the "
            "water table: a raster in metres, a constant, or none (> 100 cm; no "
            "global map is reliable enough). Optional zones (sub-catchments).\n"
            "\n"
            "OUTPUTS:\tTick the outputs wanted and choose the raster, CSV and "
            "Word folders. Per method: the recommended group (bands: group, "
            "confidence %, confidence class), the groups at Ksat P5, P50 and P95 "
            "(SCS Curve Numbers finds this raster beside the recommended one) and "
            "the probabilities of A, B, C and D. Also the conditions raster "
            "(depths, NEH 630 case, least transmissive Ksat, texture group), the "
            "zone summary (group shares per zone - area fractions for curve "
            "numbers), run metadata, the Word report (Mayim report standard) and "
            "a <project number>_HSG_Layers.qlr layer file beside the rasters. "
            "CSVs are UTF-8 with a byte-order mark. Files are named <project "
            "number>_<description> and replace files of the same name; a file "
            "that is open cannot be replaced and is reported, and the other "
            "outputs are still written. Group codes: 1-4 A-D, 11-13 NEH 630 dual "
            "A/D-C/D, 21-23 SCS-SA intermediate A/B, B/C, C/D.\n" + files
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.REGIONAL_FOLDER,
                "Regional Soil Parameterisation raster folder",
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
            QgsProcessingParameterFile(
                self.ISDA_FOLDER,
                "iSDAsoil output folder with the depth-to-bedrock layer (Africa)",
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
            QgsProcessingParameterEnum(
                self.LOAD_LAYERS,
                "Load outputs into the project",
                options=qgis_ui.LOAD_OPTIONS,
                defaultValue=qgis_ui.LOAD_MAIN,
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
                "Project name (report title page, optional)",
                optional=True,
            )
        )
        for spec in OUTPUT_SPECS:
            self.addParameter(
                QgsProcessingParameterBoolean(
                    output_param(spec.key),
                    f"Output: {spec.label}",
                    defaultValue=spec.default,
                )
            )
        for name, label in FOLDER_PARAMS.values():
            self.addParameter(
                QgsProcessingParameterFolderDestination(name, label, optional=True)
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
        chosen = self.parameterAsEnums(parameters, self.METHODS, context)
        if not chosen:
            raise QgsProcessingException("Select at least one method.")
        methods = tuple(METHOD_CODES[i] for i in chosen)
        project_number = (
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        ).strip()
        project_name = (
            self.parameterAsString(parameters, self.PROJECT_NAME, context) or ""
        ).strip()
        selected = [
            spec
            for spec in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, output_param(spec.key), context)
        ]
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {spec.kind for spec in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(
                    f"Choose the {label[0].lower() + label[1:]}."
                )
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths, labels = {}, {}
        for spec in selected:
            if spec.key in core.PER_METHOD:
                paths[spec.key] = {}
                for m in methods:
                    p = method_spec(spec, m).path(folders, project_number)
                    paths[spec.key][m] = p
                    labels[p] = f"{spec.label.split(' (')[0]}, {m}"
            else:
                p = spec.path(folders, project_number)
                paths[spec.key] = p
                labels[p] = spec.label
        settings = core.RunSettings(
            regional_folder=self.parameterAsFile(
                parameters, self.REGIONAL_FOLDER, context
            ),
            out_dir="",
            methods=methods,
            bedrock_source=BEDROCK_CODES[
                self.parameterAsEnum(parameters, self.BEDROCK_SOURCE, context)
            ],
            sg_folder=self.parameterAsFile(parameters, self.SG_FOLDER, context),
            isda_folder=self.parameterAsFile(parameters, self.ISDA_FOLDER, context),
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
            write_report=core.OUT_REPORT in paths,
            outputs=paths,
            report_context={
                "project_number": project_number,
                "project_name": project_name,
                "software": _software_version(),
                "labels": labels,
            },
        )
        try:
            ksat_path = core.find_regional_files(settings.regional_folder)["ksat"]
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
        for path, reason in result.failed:
            feedback.reportError(
                f"{Path(path).name} could not be written ({reason}). If it is open "
                "in QGIS, Excel or Word, close it and run again.",
                fatalError=False,
            )
        for path in result.files:
            feedback.pushInfo(f"Written: {path}")
        feedback.pushInfo(f"Finished in {result.seconds:.0f} s.")
        if not result.files:
            raise QgsProcessingException(
                "No output could be written - see the errors above."
            )
        outputs = [
            qgis_ui.OutputLayer(
                path=o["path"], name=o["name"], main=o["main"],
                classes=o["classes"],
            )
            for o in core.output_layers(result)
            if o["path"] in result.files
        ]  # fmt: skip
        if outputs:
            stem = f"{clean_name_part(project_number)}_" if project_number else ""
            qlr = str(Path(folders[RASTER]) / f"{stem}HSG_Layers.qlr")
            if qgis_ui.write_layer_file(outputs, qlr, feedback):
                feedback.pushInfo(f"Layer file (reloads all outputs, styled): {qlr}")
            choice = self.parameterAsEnum(parameters, self.LOAD_LAYERS, context)
            qgis_ui.load_outputs(context, outputs, choice, self._post_processors)
        out = {name: folders[kind] for kind, (name, _) in FOLDER_PARAMS.items()
               if kind in folders}  # fmt: skip
        if result.report_path:
            out[output_param(core.OUT_REPORT)] = result.report_path
        return out
