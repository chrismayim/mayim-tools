"""Processing Toolbox algorithm: Extract: SoilGrids 2.0.

Area mode (default): an extent or polygon layer in, one multi-band
GeoTIFF per variable x statistic out (one band per depth). Point mode:
a point or point layer in, a long-format CSV out. Both modes always
write a metadata CSV (sources, routes, units, QA and uncertainty
statistics). Thin wrapper - all real logic lives in core.py (no QGIS
dependency; see tests/test_soilgrids_extract_core.py).
"""

from pathlib import Path

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterPoint,
    QgsUnitTypes,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools.soil._common import qgis_ui

from . import core
from .export import (
    metadata_path_for,
    write_metadata_csv,
    write_multiband_geotiff,
    write_points_csv,
)

STAT_OPTIONS = [
    ("mean", "Mean"),
    ("Q0.05", "Q0.05 (5% quantile)"),
    ("Q0.5", "Q0.50 (median)"),
    ("Q0.95", "Q0.95 (95% quantile)"),
]


class SoilGridsExtractAlgorithm(QgsProcessingAlgorithm):
    MODE = "MODE"
    EXTENT = "EXTENT"
    AREA_LAYER = "AREA_LAYER"
    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    VARIABLES = "VARIABLES"
    DEPTHS = "DEPTHS"
    STATISTICS = "STATISTICS"
    OUTPUT_CRS = "OUTPUT_CRS"
    OUTPUT_RES = "OUTPUT_RES"
    MAX_AREA_KM2 = "MAX_AREA_KM2"
    LOAD_LAYERS = "LOAD_LAYERS"
    SG2017 = "SG2017"
    WORKERS = "WORKERS"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    OUTPUT_CSV = "OUTPUT_CSV"

    MODE_OPTIONS = ["Area (multi-band GeoTIFFs)", "Points (CSV)"]

    def __init__(self):
        super().__init__()
        self._post_processors = []
        # Variable checklist: SoilGrids 2.0 variables, then depth to bedrock.
        self._var_codes = [v.code for v in core.VARIABLES] + [core.BEDROCK_CODE]
        self._var_labels = [
            f"{v.code} - {v.label} ({v.units})" + ("" if v.default else " [optional]")
            for v in core.VARIABLES
        ] + [f"{core.BEDROCK_LABEL} [optional]"]

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return SoilGridsExtractAlgorithm()

    def name(self):
        return "soilgrids_extract"

    def displayName(self):
        return "Extract: SoilGrids 2.0"

    def group(self):
        return "Soil Tools"

    def groupId(self):
        return "soil_tools"

    def shortHelpString(self):
        return (
            f"Extract: SoilGrids 2.0 (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tDownloads ISRIC SoilGrids 2.0 soil properties (250 m, "
            "global, six depths from 0-5 to 100-200 cm) for an area or for "
            "points, with SoilGrids' own prediction quantiles so uncertainty "
            "can be carried into later calculations (soil texture, hydraulic "
            "properties, hydrologic soil groups, curve numbers). Works anywhere "
            "in the world. The tool is method-agnostic: by default it extracts "
            "every input the common pedotransfer functions use (sand, silt, "
            "clay, organic carbon, bulk density, coarse fragments, pH, CEC).\n"
            "\n"
            "METHOD:\t\tReads directly from ISRIC's file server with QGIS's own "
            "GDAL - no account, no API key, no extra Python packages. ISRIC "
            "indexes each layer with a large virtual-raster file (about 6 MB, "
            "slow to fetch), so the tool reads ONE index per run, finds the "
            "tiles that cover the area, then reads those tiles directly for every "
            "layer, several layers at once. If that fails for a layer it falls "
            "back to that layer's full index (slow but always correct); the "
            "route used is logged per layer. A 500 km2 area with the default "
            "selection typically takes under a minute or two, depending on your "
            "connection. Values are resampled with nearest neighbour only (no further "
            "smoothing) and converted to conventional units (%, g/kg, g/cm3, "
            "vol %, pH, cmol(c)/kg) before writing.\n"
            "\n"
            "OUTPUTS:\tArea mode: one multi-band GeoTIFF per variable and "
            "statistic, e.g. soilgrids_clay_Q0.50.tif, with one band per depth. "
            "Each band carries a description such as 'clay_30-60cm_Q0.5 (%)'; "
            "later tools select bands by that description, never by band number. "
            "Loaded layers show band 1 (the shallowest depth) as a single grey "
            "band; pick another depth under Layer Properties > Symbology > Band. "
            "One layer file, soilgrids_layers.qlr, is written per run: drag it "
            "into any QGIS project to reload every output with its style (no "
            "style file per raster). Point mode: one long-format CSV (Site, "
            "Longitude, Latitude, Product, Variable, Description, depth, "
            "Statistic, Value, Units, Route). Both modes write a metadata CSV "
            "(soilgrids_metadata.csv "
            "in the folder, or <name>_metadata.csv next to the CSV) with sources, "
            "access date, routes, units, per-layer statistics, a sand+silt+clay "
            "check and all warnings.\n"
            "\n"
            "UNCERTAINTY:\tQ0.05 and Q0.95 bound SoilGrids' 90% prediction "
            "interval. This tool only acquires data; uncertainty measures and "
            "all derived quantities are computed by Regional soil "
            "parameterisation. "
            "Texture quantiles are marginal: Q0.05 sand together with Q0.05 clay "
            "is not a real soil, so uncertainty must be propagated by sampling, "
            "not by combining quantiles.\n"
            "\n"
            "BACKGROUND:\tSoilGrids 2.0 (Poggio et al. 2021, SOIL 7:217-240) is "
            "a machine-learning prediction from about 230000 soil profiles. Its "
            "site-level accuracy is modest, especially for very sandy or very "
            "clayey soils, so use local data where you have it. SOC is not "
            "converted to organic matter here (that factor is a method choice). "
            "Depth to bedrock is not part of SoilGrids 2.0; the optional layer "
            "comes from the SoilGrids 2017 archive (Shangguan et al. 2017) and "
            "gives depth to bedrock only, not hardpans or other restrictive "
            "layers. Data licence: CC-BY 4.0 - cite ISRIC SoilGrids in reports.\n"
            "\n"
            "PARAMETERS:\n"
            "  Mode: Area (default) or Points.\n"
            "  Area: an extent, OR a polygon layer (its extent is used). The "
            "area is buffered by two cells so edge values are complete.\n"
            "  Points: a single point, OR a point layer (every feature), with an "
            "optional name field.\n"
            "  Variables / Depths / Statistics: checklists.\n"
            "  SoilGrids 2017 means (optional): SoilGrids 2.0 has no predictions "
            "for urban, water, glacier and bare-surface areas. The 2017 product "
            "(Hengl et al. 2017, ODbL licence) does cover urban and bare areas; "
            "its mean values are stored at their native depth points (0, 5, 15, "
            "30, 60, 100, 200 cm - those bounding the selected intervals), one "
            "band per point, in soilgrids2017_<variable>_mean.tif, for filling "
            "those gaps in later tools. Mean only, no quantiles. Each 2017 layer is a "
            "single global file and reads more slowly (several seconds each).\n"
            "  Output CRS and resolution (area mode): default project CRS and "
            "250 m (converted to degrees for a geographic CRS).\n"
            "  Maximum area per run (advanced, default 50000 km2): the run stops "
            "if the area's bounding box is larger; raise it deliberately for "
            "large studies.\n"
            "  Parallel downloads (advanced, default 8): layers read at the same "
            "time. Lower it on a slow or metered connection.\n"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE, "Mode", options=self.MODE_OPTIONS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterExtent(
                self.EXTENT, "Area of interest (extent)", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.AREA_LAYER,
                "OR: area of interest polygon layer (its extent is used)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterPoint(
                self.POINT,
                "Point of interest (points mode; click map or type coordinates)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_POINTS,
                "OR: point layer (points mode; every feature processed)",
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
        defaults = [
            i for i, v in enumerate(core.VARIABLES) if v.default
        ]  # bedrock and water content are opt-in
        self.addParameter(
            QgsProcessingParameterEnum(
                self.VARIABLES,
                "Variables",
                options=self._var_labels,
                allowMultiple=True,
                defaultValue=defaults,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEPTHS,
                "Depth intervals",
                options=[d[0] for d in core.DEPTHS],
                allowMultiple=True,
                defaultValue=list(range(len(core.DEPTHS))),
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STATISTICS,
                "Statistics",
                options=[label for _, label in STAT_OPTIONS],
                allowMultiple=True,
                defaultValue=[0, 1, 2, 3],
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SG2017,
                "Also extract SoilGrids 2017 means (cover urban and bare areas "
                "that SoilGrids 2.0 leaves empty)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.OUTPUT_CRS, "Output CRS (area mode)", defaultValue="ProjectCrs"
            )
        )
        self.addParameter(
            qgis_ui.advanced(
                QgsProcessingParameterNumber(
                    self.OUTPUT_RES,
                    "Output resolution in metres (area mode; converted for a "
                    "geographic CRS)",
                    type=QgsProcessingParameterNumber.Type.Double,
                    defaultValue=250.0,
                    minValue=1.0,
                )
            )
        )
        self.addParameter(
            qgis_ui.advanced(
                QgsProcessingParameterNumber(
                    self.MAX_AREA_KM2,
                    "Maximum area per run (km2, bounding box; 0 = no limit)",
                    type=QgsProcessingParameterNumber.Type.Double,
                    defaultValue=core.DEFAULT_MAX_AREA_KM2,
                    minValue=0.0,
                )
            )
        )
        self.addParameter(
            qgis_ui.advanced(
                QgsProcessingParameterNumber(
                    self.WORKERS,
                    "Parallel downloads",
                    type=QgsProcessingParameterNumber.Type.Integer,
                    defaultValue=core.DEFAULT_WORKERS,
                    minValue=1,
                    maxValue=16,
                )
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.LOAD_LAYERS,
                "Load outputs into the project (area mode)",
                options=qgis_ui.LOAD_OPTIONS,
                defaultValue=qgis_ui.LOAD_ALL,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER, "Output folder (area mode)", optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_CSV,
                "Output CSV (points mode)",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
        )

    # ------------------------------------------------------------------
    # Input collection
    # ------------------------------------------------------------------

    def _selection(self, parameters, context):
        var_idx = self.parameterAsEnums(parameters, self.VARIABLES, context)
        codes = [self._var_codes[i] for i in var_idx]
        bedrock = core.BEDROCK_CODE in codes
        variables = [c for c in codes if c != core.BEDROCK_CODE]
        depths = [
            core.DEPTHS[i][0]
            for i in self.parameterAsEnums(parameters, self.DEPTHS, context)
        ]
        stat_codes = [
            STAT_OPTIONS[i][0]
            for i in self.parameterAsEnums(parameters, self.STATISTICS, context)
        ]
        stats = stat_codes
        try:
            sg2017 = self.parameterAsBoolean(parameters, self.SG2017, context)
            return core.plan_selection(variables, depths, stats, bedrock, sg2017=sg2017)
        except core.SoilGridsError as exc:
            raise QgsProcessingException(str(exc)) from exc

    def _area_bounds(self, parameters, context, out_crs, feedback):
        return qgis_ui.area_bounds(
            self, parameters, context, out_crs, feedback, self.AREA_LAYER, self.EXTENT
        )

    def _sites(self, parameters, context, feedback):
        return [
            core.Site(label, lon, lat)
            for label, lon, lat in qgis_ui.point_sites(
                self,
                parameters,
                context,
                feedback,
                self.POINT,
                self.INPUT_POINTS,
                self.NAME_FIELD,
            )
        ]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        mode_idx = self.parameterAsEnum(parameters, self.MODE, context)
        selection = self._selection(parameters, context)
        for warning in selection.warnings:
            feedback.pushWarning(warning)

        def progress(fraction, message):
            feedback.setProgress(int(100 * fraction))
            feedback.setProgressText(message)

        common = {
            "workers": self.parameterAsInt(parameters, self.WORKERS, context),
            "progress_fn": progress,
            "log_fn": feedback.pushWarning,
            "cancel_fn": feedback.isCanceled,
        }
        gdal_ver = core.gdal_version()
        feedback.pushInfo(f"GDAL {gdal_ver}; reading from {core.BASE_URL}")

        try:
            if mode_idx == 0:
                return self._run_area(
                    parameters, context, feedback, selection, common, gdal_ver
                )
            return self._run_points(
                parameters, context, feedback, selection, common, gdal_ver
            )
        except core.SoilGridsError as exc:
            raise QgsProcessingException(str(exc)) from exc
        except InterruptedError:
            feedback.pushWarning("Cancelled - outputs may be incomplete.")
            return {}

    def _run_area(self, parameters, context, feedback, selection, common, gdal_ver):
        folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        if not folder:
            raise QgsProcessingException("Area mode needs an output folder.")
        Path(folder).mkdir(parents=True, exist_ok=True)
        out_crs = self.parameterAsCrs(parameters, self.OUTPUT_CRS, context)
        if not out_crs.isValid():
            raise QgsProcessingException("Choose a valid output CRS.")
        res_m = self.parameterAsDouble(parameters, self.OUTPUT_RES, context)
        res = qgis_ui.resolution_in_crs_units(res_m, out_crs)
        max_area = self.parameterAsDouble(parameters, self.MAX_AREA_KM2, context)

        bounds_out, bounds_ll, desc = self._area_bounds(
            parameters, context, out_crs, feedback
        )
        area = core.check_area(bounds_ll, max_area)
        grid = core.make_grid(bounds_out, res, qgis_ui.crs_wkt(out_crs))
        feedback.pushInfo(
            f"Area of interest ({desc}): {area:.1f} km2 (bounding box); output "
            f"grid {grid.width} x {grid.height} cells at {res:g} "
            f"{QgsUnitTypes.toString(out_crs.mapUnits())} in {out_crs.authid()}."
        )

        result = core.extract_area(
            selection,
            grid,
            bounds_ll,
            folder,
            write_fn=write_multiband_geotiff,
            **common,
        )
        settings = {
            "Area of interest": desc,
            "Bounds (output CRS)": ", ".join(f"{v:.3f}" for v in bounds_out),
            "Bounds (lon/lat)": ", ".join(f"{v:.6f}" for v in bounds_ll),
            "Area (km2, bounding box)": round(area, 1),
            "Output CRS": out_crs.authid() or out_crs.description(),
            "Output resolution": f"{res:g} ({res_m:g} m requested)",
            "Grid size (cells)": f"{grid.width} x {grid.height}",
            "Buffer (cells)": core.BUFFER_CELLS,
            "Resampling": "nearest neighbour",
        }
        meta_path = metadata_path_for(folder, "area")
        write_metadata_csv(
            core.build_metadata(result, selection, settings, gdal_ver), meta_path
        )
        self._report(result, feedback)
        feedback.pushInfo(f"{len(result.files)} raster(s) written to {folder}")
        feedback.pushInfo(f"Metadata: {meta_path}")

        outputs = [
            qgis_ui.OutputLayer(
                path=path,
                name=Path(path).stem
                + (" (median)" if Path(path).stem.endswith("_Q0.50") else ""),
                main=path in result.load_files,
            )
            for path in result.files
        ]
        qlr = str(Path(folder) / "soilgrids_layers.qlr")
        if qgis_ui.write_layer_file(outputs, qlr, feedback):
            feedback.pushInfo(f"Layer file (reloads all outputs, styled): {qlr}")
        choice = self.parameterAsEnum(parameters, self.LOAD_LAYERS, context)
        qgis_ui.load_outputs(context, outputs, choice, self._post_processors)
        return {self.OUTPUT_FOLDER: folder}

    def _run_points(self, parameters, context, feedback, selection, common, gdal_ver):
        out_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)
        if not out_csv:
            raise QgsProcessingException("Points mode needs an output CSV.")
        sites = self._sites(parameters, context, feedback)
        feedback.pushInfo(f"{len(sites)} site(s) to sample.")
        result = core.extract_points(selection, sites, **common)
        n_rows, n_missing = write_points_csv(result.rows, out_csv)
        if n_missing:
            feedback.pushWarning(
                f"{n_missing} of {n_rows} values are missing (no SoilGrids data at "
                "the point, e.g. water, urban or outside coverage); written as "
                "empty cells, never as zero."
            )
        settings = {"Sites": len(sites), "Sampling": "nearest cell"}
        meta_path = metadata_path_for(out_csv, "points")
        write_metadata_csv(
            core.build_metadata(result, selection, settings, gdal_ver), meta_path
        )
        self._report(result, feedback)
        feedback.pushInfo(f"{n_rows} rows written to {out_csv}")
        feedback.pushInfo(f"Metadata: {meta_path}")
        return {self.OUTPUT_CSV: out_csv}

    @staticmethod
    def _report(result, feedback):
        if any(result.tiles.values()):
            per_var = ", ".join(f"{v} {n}" for v, n in result.tiles.items())
            feedback.pushInfo(
                f"Read via SoilGrids tiles (per variable: {per_var}); "
                f"finished in {result.seconds:.0f} s."
            )
        else:
            feedback.pushInfo(f"Finished in {result.seconds:.0f} s.")
        for tex in result.texture:
            if tex["flagged"]:
                feedback.pushWarning(
                    f"Texture check {tex['depth']}: {tex['flagged']} of "
                    f"{tex['checked']} cells/points have mean sand+silt+clay more "
                    f"than {core.TEXTURE_SUM_TOLERANCE:g}% from 100%."
                )
