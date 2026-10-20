"""Processing Toolbox algorithm: Regional Soil Parameterisation.

Reads the output folders of 'Extract: SoilGrids 2.0', 'Extract: OpenLandMap
Soils' and/or 'Extract: iSDAsoil (Africa)' and writes hydraulic soil
parameters (median and 90 % range per depth layer), USDA texture class per
layer, quality rasters, a zone summary, design parameters, metadata and a
Word report. Outputs follow the Mayim output standard: a tick box per output,
raster / CSV / Word folders and <project number>_<description> file names.
Thin wrapper - the logic is in core.py and the modules it uses (no QGIS; see
tests/test_regional_parameterisation_core.py).
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
    QgsProcessingParameterString,
    QgsProcessingParameterVectorLayer,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import CSV, RASTER, WORD, OutputSpec
from mayim_tools.soil._common import qgis_ui
from mayim_tools.soil._common.errors import SoilDataError

from . import core
from .inputs import TARGET_LAYERS
from .ptf import AVAILABLE_METHODS
from .texture import TEXTURE_STYLE
from .uncertainty import Settings

STYLE_TEXTURE = str(Path(__file__).resolve().parent / "styles" / TEXTURE_STYLE)


def _folder_behavior():
    """Folder behaviour of QgsProcessingParameterFile (QGIS 4 scoped enum,
    older fallback)."""
    try:
        return Qgis.ProcessingFileParameterBehavior.Folder
    except AttributeError:  # pragma: no cover - older QGIS
        return QgsProcessingParameterFile.Behavior.Folder


def _param(code, label, description, default=True):
    return OutputSpec(core.param_key(code), label, description, RASTER, default)


# One tick box per output (Mayim output standard). File names are
# <project number>_<description>.<ext>; texture is one raster per layer
# (<..>_Soil_USDA_Texture_0-30cm.tif).
OUTPUT_SPECS = [
    _param("ksat", "Ksat (P50, P05, P95 per layer)", "Soil Ksat"),
    _param("psi_f", "Wetting-front suction head ψf", "Soil Suction Head"),
    _param("theta_s", "Saturated water content θs", "Soil Theta S"),
    _param("theta_fc", "Field capacity θ33", "Soil Theta 33"),
    _param("theta_wp", "Wilting point θ1500", "Soil Theta 1500"),
    _param("paw", "Plant-available water", "Soil PAW"),
    _param("ima_dry", "Initial deficit, dry start", "Soil Initial Deficit Dry"),
    _param("ima_moist", "Initial deficit, moist start", "Soil Initial Deficit Moist"),
    _param("t_dry", "Drying time (SWMM)", "Soil Drying Time"),
    OutputSpec(
        core.OUT_TEXTURE,
        "USDA texture class (one raster per layer)",
        "Soil USDA Texture",
        RASTER,
    ),
    OutputSpec(
        core.OUT_QUALITY,
        "Quality (Ksat uncertainty, texture confidence, flags, sources)",
        "Soil Quality",
        RASTER,
    ),
    OutputSpec(
        core.OUT_INPUTS,
        "Harmonised inputs (used by Hydrological Soil Groups (HSG))",
        "Soil Harmonised Inputs",
        RASTER,
    ),
    OutputSpec(
        core.OUT_DIFFERENCE,
        "Product differences",
        "Soil Product Differences",
        RASTER,
        False,
    ),
    _param("psi_b", "Air-entry suction ψb (Saxton & Rawls)", "Soil Air Entry", False),
    _param("lambda", "Pore-size distribution index λ", "Soil Lambda", False),
    _param("theta_r", "van Genuchten θr", "Soil VG Theta R", False),
    _param("alpha", "van Genuchten α", "Soil VG Alpha", False),
    _param("n_vg", "van Genuchten n", "Soil VG N", False),
    OutputSpec(core.OUT_ZONES_CSV, "Zone summary", "Soil Zone Summary", CSV),
    OutputSpec(
        core.OUT_DESIGN_CSV,
        "Candidate design parameters",
        "Soil Design Parameters",
        CSV,
    ),
    OutputSpec(core.OUT_METADATA_CSV, "Run metadata", "Soil Parameters Metadata", CSV),
    OutputSpec(core.OUT_REPORT, "Word report", "Soil Parameters Report", WORD),
]
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}
MAIN_KEYS = {
    core.param_key("ksat"),
    core.param_key("psi_f"),
    core.param_key("ima_dry"),
    core.param_key("t_dry"),
    core.OUT_TEXTURE,
}


def output_param(key: str) -> str:
    """Processing parameter name of an output's tick box."""
    return "OUTPUT_" + key.upper()


class RegionalSoilParameterisationAlgorithm(QgsProcessingAlgorithm):
    METHODS = "METHODS"
    SG_FOLDER = "SG_FOLDER"
    OLM_FOLDER = "OLM_FOLDER"
    ISDA_FOLDER = "ISDA_FOLDER"
    ZONES = "ZONES"
    ZONE_FIELD = "ZONE_FIELD"
    SITE_BOUNDARY = "SITE_BOUNDARY"
    CATCHMENT_BOUNDARY = "CATCHMENT_BOUNDARY"
    RESOLUTION = "RESOLUTION"
    OM_FACTOR = "OM_FACTOR"
    DENSITY = "DENSITY"
    GRAVEL = "GRAVEL"
    FILL_RADIUS = "FILL_RADIUS"
    DRAWS = "DRAWS"
    SEED = "SEED"
    WORKERS = "WORKERS"
    MAX_CELLS = "MAX_CELLS"
    REPORT_BASEMAP = "REPORT_BASEMAP"
    LOAD_LAYERS = "LOAD_LAYERS"
    PROJECT_NUMBER = "PROJECT_NUMBER"

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
        return "Regional Soil Parameterisation"

    def group(self):
        return "Soil Tools"

    def groupId(self):
        return "soil_tools"

    def shortHelpString(self):
        layers = ", ".join(lab for lab, _, _ in TARGET_LAYERS)
        return (
            f"Regional Soil Parameterisation (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tEstimates hydraulic soil parameters for any area from the "
            "outputs of 'Extract: SoilGrids 2.0', 'Extract: OpenLandMap Soils' "
            "and/or 'Extract: iSDAsoil (Africa)': water content at saturation, "
            "field capacity (33 kPa) and "
            "wilting point (1500 kPa), plant-available water, saturated "
            "hydraulic conductivity (Ksat), Brooks-Corey and van Genuchten "
            "parameters, the Green-Ampt wetting-front suction head and the "
            "Green-Ampt (SWMM) initial deficit (dry and moist start) and drying "
            "time - each as a median (P50) with a 90 % range (P05-P95) - and the "
            "USDA texture class per layer. A Word report documents the data, "
            "methods, uncertainty, checks and candidate design values per zone, "
            "with a USDA texture layout map per layer.\n"
            "\n"
            "INPUTS:\tGive one, two or three product folders (more is better: "
            "each product is an ensemble member, their disagreement is reported "
            "and included in the uncertainty, and each fills the others' gaps). "
            "iSDAsoil (Africa only) is mapped to 50 cm: it is used for 0-30 and "
            "30-60 cm, not 60-100 cm. Needed in the folders: sand, silt, clay and "
            "SOC; bulk density is needed by Tóth et al., and pH and CEC by its "
            "Ksat (CEC: SoilGrids only); coarse fragments (SoilGrids) are used by "
            "the optional gravel correction; mapped water contents (SoilGrids wv0033/"
            "wv1500, OpenLandMap 250 m) are used for checks. Bands are found by "
            "their descriptions, so any depth/statistic selection works if the "
            "depths cover the target layers.\n"
            "\n"
            f"METHOD:\t(1) Harmonise to {layers} (thickness-weighted; ISO texture "
            "converted to USDA limits). (2) Fill gaps: other product(s), then "
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
            "REPORT MAPS:\tThe USDA texture layout maps show the texture raster "
            "at 70 % transparency over OpenStreetMap. Optional site boundary and "
            "catchment boundary polygon layers are drawn above it in their "
            "symbology from the project. With a catchment layer the map fits the "
            "catchment with a margin; without it the map fits the texture raster "
            "(and the zones are outlined as the study area).\n"
            "\n"
            "OUTPUTS:\tTick the outputs you need and choose a folder for "
            "rasters, CSV files and Word reports. Files are named <project "
            "number>_<description> with underscores (e.g. MC2025_014_Soil_Ksat."
            "tif); without a project number just the description. Rasters: one "
            "GeoTIFF per parameter (bands: P50, P05, P95 per layer - select bands "
            "by description), the USDA texture class per layer "
            "(<..>_Soil_USDA_Texture_0-30cm.tif, styled with "
            "SOIL_USDA_classification.qml), quality (Ksat uncertainty, texture "
            "confidence and share, validity flags, input source), harmonised "
            "inputs and product differences. CSV: zone summary, candidate design "
            "parameters, run metadata (UTF-8 with BOM). Word: the report. "
            "Existing files are overwritten; a file that is open elsewhere is "
            "reported and the other outputs are still written.\n"
            "\n"
            "HYDROLOGIC SOIL GROUPS:\tGive that tool the raster folder: it finds "
            "the Ksat and harmonised-inputs rasters by their band descriptions "
            "(keep 'Ksat' and 'Harmonised inputs' ticked).\n"
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
            QgsProcessingParameterFile(
                self.ISDA_FOLDER,
                "iSDAsoil output folder (from Extract: iSDAsoil (Africa))",
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
        for name, label in (
            (self.SITE_BOUNDARY, "Site boundary for the report maps [optional]"),
            (
                self.CATCHMENT_BOUNDARY,
                "Catchment boundary for the report maps [optional]",
            ),
        ):
            self.addParameter(
                QgsProcessingParameterVectorLayer(
                    name,
                    label,
                    [QgsProcessing.SourceType.TypeVectorPolygon],
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
        basemap = QgsProcessingParameterBoolean(
            self.REPORT_BASEMAP,
            "Report maps: OpenStreetMap background (needs internet)",
            defaultValue=True,
        )
        self.addParameter(qgis_ui.advanced(basemap))
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
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix, optional)",
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

    def _output_paths(self, parameters, context):
        """{core output key: path} for the ticked outputs, and the folders."""
        project_number = self.parameterAsString(
            parameters, self.PROJECT_NUMBER, context
        )
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
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        return {s.key: s.path(folders, project_number) for s in selected}, folders

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        paths, folders = self._output_paths(parameters, context)
        sg = self.parameterAsFile(parameters, self.SG_FOLDER, context)
        olm = self.parameterAsFile(parameters, self.OLM_FOLDER, context)
        isda = self.parameterAsFile(parameters, self.ISDA_FOLDER, context)
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
            out_dir="",
            sg_folder=sg,
            olm_folder=olm,
            isda_folder=isda,
            resolution=self.parameterAsDouble(parameters, self.RESOLUTION, context),
            fill_radius_m=self.parameterAsDouble(parameters, self.FILL_RADIUS, context),
            mc=mc,
            workers=self.parameterAsInt(parameters, self.WORKERS, context),
            max_cells=self.parameterAsInt(parameters, self.MAX_CELLS, context),
            write_report=core.OUT_REPORT in paths,
            outputs=paths,
        )
        map_fn = None
        if core.OUT_REPORT in paths:
            try:
                from .texture_map import make_texture_map, snapshot_layer

                site = snapshot_layer(
                    self.parameterAsVectorLayer(
                        parameters, self.SITE_BOUNDARY, context
                    ),
                    "Site boundary",
                )
                catchment = snapshot_layer(
                    self.parameterAsVectorLayer(
                        parameters, self.CATCHMENT_BOUNDARY, context
                    ),
                    "Catchment boundary",
                )
                map_fn = make_texture_map(
                    context.transformContext(),
                    use_osm=self.parameterAsBoolean(
                        parameters, self.REPORT_BASEMAP, context
                    ),
                    site_layer=site,
                    catchment_layer=catchment,
                )
            except Exception as exc:  # noqa: BLE001 - the report still gets maps
                feedback.pushWarning(
                    f"QGIS layout maps unavailable ({exc}); the report draws "
                    "simplified texture maps."
                )
        try:
            folders_in = [f for f in (sg, olm, isda) if f]
            if not folders_in:
                raise SoilDataError(
                    "Give at least one product folder: SoilGrids, OpenLandMap "
                    "or iSDAsoil."
                )
            grid = core.processing_grid(folders_in, settings.resolution)
            settings.zones = self._zones(parameters, context, grid, feedback)

            def progress(fraction):
                feedback.setProgress(int(100 * fraction))

            result = core.run(
                settings,
                log_fn=feedback.pushInfo,
                progress_fn=progress,
                cancel_fn=feedback.isCanceled,
                map_fn=map_fn,
            )
        except SoilDataError as exc:
            raise QgsProcessingException(str(exc)) from exc
        for w in result.warnings:
            feedback.pushWarning(w)
        for n in result.notes:
            feedback.pushInfo(n)
        for path, reason in result.failed:
            feedback.reportError(f"Not written: {path} ({reason})", fatalError=False)
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
        feedback.pushInfo(f"Finished in {result.seconds:.0f} s.")
        for path, desc in result.files:
            feedback.pushInfo(f"Written: {path} ({desc})")

        ksat_classes = list(core.KSAT_CLASS_LABELS.items())
        layers = []
        for key, written in result.outputs.items():
            for path in written:
                if not path.lower().endswith(".tif"):
                    continue
                stem = Path(path).stem
                layers.append(
                    qgis_ui.OutputLayer(
                        path=path,
                        name=stem,
                        main=key in MAIN_KEYS,
                        classes=ksat_classes if key == core.OUT_QUALITY else [],
                        qml=STYLE_TEXTURE if key == core.OUT_TEXTURE else "",
                    )
                )
        choice = self.parameterAsEnum(parameters, self.LOAD_LAYERS, context)
        qgis_ui.load_outputs(context, layers, choice, self._post_processors)
        out = {name: folders[kind] for kind, (name, _) in FOLDER_PARAMS.items()
               if kind in folders}  # fmt: skip
        return out
