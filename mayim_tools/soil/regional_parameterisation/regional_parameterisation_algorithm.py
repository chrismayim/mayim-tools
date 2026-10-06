"""Processing Toolbox algorithm: Regional soil parameterisation.

Reads the output folders of 'Extract: SoilGrids 2.0' and/or 'Extract:
OpenLandMap Soils' and writes hydraulic soil parameters (median and 90 %
range per depth layer), quality rasters, a zone summary, metadata and a Word
report. Thin wrapper - the logic is in core.py and the modules it uses (no
QGIS; see tests/test_regional_parameterisation_core.py).
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
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools.soil._common import qgis_ui
from mayim_tools.soil._common.errors import SoilDataError

from . import core
from .inputs import TARGET_LAYERS
from .ptf import AVAILABLE_METHODS
from .texture import TEXTURE_CLASSES
from .uncertainty import Settings


def _folder_behavior():
    """Folder behaviour of QgsProcessingParameterFile (QGIS 4 scoped enum,
    older fallback)."""
    try:
        return Qgis.ProcessingFileParameterBehavior.Folder
    except AttributeError:  # pragma: no cover - older QGIS
        return QgsProcessingParameterFile.Behavior.Folder


MAIN_FILES = {"rsp_ksat", "rsp_psi_f", "rsp_theta_s", "rsp_texture_class"}


class RegionalSoilParameterisationAlgorithm(QgsProcessingAlgorithm):
    METHODS = "METHODS"
    SG_FOLDER = "SG_FOLDER"
    OLM_FOLDER = "OLM_FOLDER"
    ZONES = "ZONES"
    ZONE_FIELD = "ZONE_FIELD"
    RESOLUTION = "RESOLUTION"
    OM_FACTOR = "OM_FACTOR"
    DENSITY = "DENSITY"
    GRAVEL = "GRAVEL"
    FILL_RADIUS = "FILL_RADIUS"
    DRAWS = "DRAWS"
    SEED = "SEED"
    WORKERS = "WORKERS"
    MAX_CELLS = "MAX_CELLS"
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
        return RegionalSoilParameterisationAlgorithm()

    def name(self):
        return "regional_soil_parameterisation"

    def displayName(self):
        return "Regional soil parameterisation"

    def group(self):
        return "Soil Tools"

    def groupId(self):
        return "soil_tools"

    def shortHelpString(self):
        layers = ", ".join(lab for lab, _, _ in TARGET_LAYERS)
        return (
            f"Regional soil parameterisation (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tEstimates hydraulic soil parameters for any area from the "
            "outputs of 'Extract: SoilGrids 2.0' and/or 'Extract: OpenLandMap "
            "Soils': water content at saturation, field capacity (33 kPa) and "
            "wilting point (1500 kPa), plant-available water, saturated "
            "hydraulic conductivity (Ksat), Brooks-Corey and van Genuchten "
            "parameters and the Green-Ampt wetting-front suction - each as a "
            "median (P50) with a "
            "90 % range (P05-P95). A detailed Word report documents the data, "
            "methods, uncertainty and checks.\n"
            "\n"
            "INPUTS:\tGive the SoilGrids folder, the OpenLandMap folder, or both "
            "(both is best: each product is an ensemble member, their "
            "disagreement is reported and included in the uncertainty, and each "
            "fills the other's gaps). Needed in the folders: sand, silt, clay and "
            "SOC; bulk density is needed by Tóth et al., and pH and CEC by its "
            "Ksat (CEC: SoilGrids only); coarse fragments (SoilGrids) are used by "
            "the optional gravel correction; mapped water contents (SoilGrids wv0033/"
            "wv1500, OpenLandMap 250 m) are used for checks. Bands are found by "
            "their descriptions, so any depth/statistic selection works if the "
            "depths cover the target layers.\n"
            "\n"
            f"METHOD:\t(1) Harmonise to {layers} (thickness-weighted; ISO texture "
            "converted to USDA limits). (2) Fill gaps: other product, then "
            "SoilGrids 2017 means, then neighbours. (3-4) Monte Carlo: each cell "
            "draws inputs from the products' own uncertainty (texture sampled as "
            "a composition so every draw sums to 100 %) and runs every method "
            "chosen: Saxton & Rawls (2006), Tóth et al. (2015) as used in "
            "HiHydroSoil v2.0, or both (default; the method spread is then part "
            "of the uncertainty). "
            "Rawls, Brakensiek & Miller (1983) class values are a reference "
            "check.\n"
            "\n"
            "ZONES:\tOptional polygons (e.g. catchments) for per-zone tables. "
            "Only cells inside the zones are processed when zones are given.\n"
            "\n"
            "OUTPUTS:\tOne GeoTIFF per parameter (bands: P50, P05, P95 per "
            "layer - select bands by description), harmonised inputs, USDA "
            "texture class, quality raster (Ksat uncertainty factor and class, texture "
            "confidence, validity flags, input "
            "source), product difference (both products), zone summary CSV, "
            "metadata CSV, Word report and one .qlr layer file that reloads every "
            "output styled.\n"
            "\n"
            "LIMITATIONS:\tTexture-based methods ignore soil structure and "
            "macropores; global maps have modest site-level accuracy; the "
            "ensemble spread is a lower bound. Local measurements take "
            "precedence."
        )

    def initAlgorithm(self, config=None):
        folder = _folder_behavior()
        self.addParameter(
            QgsProcessingParameterFile(
                self.SG_FOLDER,
                "SoilGrids output folder (from Extract: SoilGrids 2.0)",
                behavior=folder,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.OLM_FOLDER,
                "OpenLandMap output folder (from Extract: OpenLandMap Soils)",
                behavior=folder,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHODS,
                "Methods (both = ensemble with the method spread in the uncertainty)",
                options=[m.name for m in AVAILABLE_METHODS],
                allowMultiple=True,
                defaultValue=list(range(len(AVAILABLE_METHODS))),
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.ZONES,
                "Zones (polygons, e.g. catchments) [optional]",
                [QgsProcessing.SourceType.TypeVectorPolygon],
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
                self.RESOLUTION,
                "Processing resolution in map units (0 = finest input grid)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.DENSITY,
                "Saxton & Rawls density adjustment from mapped bulk density",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.GRAVEL,
                "Gravel correction from coarse fragments (bulk-soil values)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.REPORT, "Write the Word report", defaultValue=True
            )
        )
        for param in (
            QgsProcessingParameterNumber(
                self.OM_FACTOR,
                "Organic matter factor (OM = SOC x factor)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=1.724,
                minValue=1.0,
                maxValue=2.5,
            ),
            QgsProcessingParameterNumber(
                self.FILL_RADIUS,
                "Neighbour gap-fill radius (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=1000.0,
                minValue=0.0,
            ),
            QgsProcessingParameterNumber(
                self.DRAWS,
                "Monte Carlo draws per cell and product",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=200,
                minValue=10,
                maxValue=2000,
            ),
            QgsProcessingParameterNumber(
                self.SEED,
                "Random seed (same seed = same results)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=12345,
                minValue=0,
            ),
            QgsProcessingParameterNumber(
                self.WORKERS,
                "Parallel workers (0 = automatic)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=0,
                minValue=0,
                maxValue=32,
            ),
            QgsProcessingParameterNumber(
                self.MAX_CELLS,
                "Maximum cells in the processing grid",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=core.MAX_CELLS_DEFAULT,
                minValue=1000,
            ),
        ):
            self.addParameter(qgis_ui.advanced(param))
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

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        out_dir = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        sg = self.parameterAsFile(parameters, self.SG_FOLDER, context)
        olm = self.parameterAsFile(parameters, self.OLM_FOLDER, context)
        chosen = self.parameterAsEnums(parameters, self.METHODS, context)
        if not chosen:
            raise QgsProcessingException("Select at least one method.")
        mc = Settings(
            methods=tuple(AVAILABLE_METHODS[i].code for i in chosen),
            draws=self.parameterAsInt(parameters, self.DRAWS, context),
            seed=self.parameterAsInt(parameters, self.SEED, context),
            om_factor=self.parameterAsDouble(parameters, self.OM_FACTOR, context),
            density=self.parameterAsBoolean(parameters, self.DENSITY, context),
            gravel=self.parameterAsBoolean(parameters, self.GRAVEL, context),
        )
        settings = core.RunSettings(
            out_dir=out_dir,
            sg_folder=sg,
            olm_folder=olm,
            resolution=self.parameterAsDouble(parameters, self.RESOLUTION, context),
            fill_radius_m=self.parameterAsDouble(parameters, self.FILL_RADIUS, context),
            mc=mc,
            workers=self.parameterAsInt(parameters, self.WORKERS, context),
            max_cells=self.parameterAsInt(parameters, self.MAX_CELLS, context),
            write_report=self.parameterAsBoolean(parameters, self.REPORT, context),
        )
        try:
            folders = [f for f in (sg, olm) if f]
            if not folders:
                raise SoilDataError(
                    "Give the SoilGrids output folder, the OpenLandMap output "
                    "folder, or both."
                )
            grid = core.processing_grid(folders, settings.resolution)
            settings.zones = self._zones(parameters, context, grid, feedback)

            def progress(fraction):
                feedback.setProgress(int(100 * fraction))

            result = core.run(
                settings,
                log_fn=feedback.pushInfo,
                progress_fn=progress,
                cancel_fn=feedback.isCanceled,
            )
        except SoilDataError as exc:
            raise QgsProcessingException(str(exc)) from exc
        for w in result.warnings:
            feedback.pushWarning(w)
        for row in result.zone_texture:
            if row["Layer"] == TARGET_LAYERS[0][0]:
                ksat = next(
                    (
                        r
                        for r in result.zone_rows
                        if r["Zone"] == row["Zone"]
                        and r["Layer"] == row["Layer"]
                        and r["Parameter"] == "ksat"
                    ),
                    None,
                )
                feedback.pushInfo(
                    f"{row['Zone']} {row['Layer']}: {row['Dominant class']} "
                    f"({row['Dominant share (%)']:.0f} % of cells)"
                    + (
                        f"; Ksat median {ksat['Median of P50']:.3g} mm/h "
                        f"(cell P5-P95 {ksat['Median cell P5']:.3g}-"
                        f"{ksat['Median cell P95']:.3g})"
                        if ksat
                        else ""
                    )
                )
        feedback.pushInfo(f"Finished in {result.seconds:.0f} s; outputs in {out_dir}")
        if result.report_path:
            feedback.pushInfo(f"Report: {result.report_path}")

        texture_classes = [(c, f"{n} ({a})") for c, n, a in TEXTURE_CLASSES]
        ksat_classes = list(core.KSAT_CLASS_LABELS.items())
        outputs = []
        for path, _ in result.files:
            if not path.lower().endswith(".tif"):
                continue
            stem = Path(path).stem
            classes = []
            if stem == "rsp_texture_class":
                classes = texture_classes
            elif stem == "rsp_quality":
                classes = ksat_classes
            outputs.append(
                qgis_ui.OutputLayer(
                    path=path, name=stem, main=stem in MAIN_FILES, classes=classes
                )
            )
        qlr = str(Path(out_dir) / core.LAYER_FILE)
        if qgis_ui.write_layer_file(outputs, qlr, feedback):
            feedback.pushInfo(f"Layer file (reloads all outputs, styled): {qlr}")
        choice = self.parameterAsEnum(parameters, self.LOAD_LAYERS, context)
        qgis_ui.load_outputs(context, outputs, choice, self._post_processors)
        return {self.OUTPUT_FOLDER: out_dir}
