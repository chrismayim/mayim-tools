"""Processing Toolbox algorithm: Extract: iSDAsoil (Africa).

Area mode (default): an extent or polygon layer in, one multi-band GeoTIFF
per variable x statistic out (one band per depth). Point mode: a point or
point layer in, a long-format CSV out. Both modes always write a metadata
CSV. Thin wrapper - the logic is in core.py and catalogue.py (no QGIS; see
tests/test_isda_extract_core.py).
"""

from pathlib import Path

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
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
from mayim_tools.soil._common.export import (
    metadata_path_for,
    write_metadata_csv,
    write_multiband_geotiff,
    write_points_csv,
)
from mayim_tools.soil._common.grid import check_area, make_grid

from . import catalogue as cat
from . import core

METADATA_NAME = "isda_metadata.csv"


def _write_geotiff(path, grid, bands, units):
    write_multiband_geotiff(
        path,
        grid,
        bands,
        units,
        source="iSDAsoil (via mayim_tools)",
        tool_version=core.TOOL_VERSION,
    )


class IsdaExtractAlgorithm(QgsProcessingAlgorithm):
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
    WORKERS = "WORKERS"
    LOAD_LAYERS = "LOAD_LAYERS"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    OUTPUT_CSV = "OUTPUT_CSV"

    MODE_OPTIONS = ["Area (multi-band GeoTIFFs)", "Points (CSV)"]

    def __init__(self):
        super().__init__()
        self._post_processors = []

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return IsdaExtractAlgorithm()

    def name(self):
        return "isda_extract"

    def displayName(self):
        return "Extract: iSDAsoil (Africa)"

    def group(self):
        return "Soil Tools"

    def groupId(self):
        return "soil_tools"

    def shortHelpString(self):
        return (
            f"Extract: iSDAsoil (Africa) (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tDownloads iSDAsoil soil properties for Africa (Hengl et al. "
            "2021; Miller et al. 2021) at 30 m for 0-20 and 20-50 cm - sand, silt, "
            "clay (USDA limits), organic carbon, bulk density, pH and stone "
            "content, plus depth to bedrock (0-200 cm) - each with its mean and "
            "standard deviation, for an area or for points. iSDAsoil was trained "
            "on more than 100000 African soil samples and is a third ensemble "
            "member for Regional soil parameterisation (0-30 and 30-60 cm) and a "
            "bedrock source for Hydrologic soil groups. Africa only.\n"
            "\n"
            "METHOD:\t\tReads only the requested window of each continent-wide "
            "cloud-optimised GeoTIFF from the public iSDAsoil Amazon S3 bucket with "
            "QGIS's own GDAL (no account, no extra packages), several layers at "
            "once, with an unsigned S3 fallback. Means are back-transformed to "
            "physical units (exp(x/10)-1 for organic carbon and stone content; "
            "x/100 bulk density; x/10 pH). Nearest-neighbour resampling only.\n"
            "\n"
            "UNCERTAINTY:\tThe standard deviation is the spread of the model's "
            "learners, not a calibrated prediction interval, and is likely "
            "narrower than the true error. For organic carbon and stone content "
            "it is a standard deviation of ln(1 + x). This tool only acquires "
            "data; uncertainty is propagated by Regional soil parameterisation.\n"
            "\n"
            "OUTPUTS:\tArea mode: isda_<variable>_<mean|sd>.tif, one band per "
            "depth, each described like 'clay_20-50cm_mean_30m_isda (%)' (later "
            "tools select bands by description), and isda_layers.qlr. Point mode: "
            "long-format CSV. Both modes write isda_metadata.csv (or "
            "<name>_metadata.csv) with sources, transforms, the band order used, "
            "per-layer statistics, a sand+silt+clay check and warnings.\n"
            "\n"
            "PARAMETERS:\n"
            "  Area: an extent OR a polygon layer (its extent, buffered by two "
            "cells). Points: a point OR a point layer with an optional name field.\n"
            "  Variables: effective CEC is optional and information only (it is "
            "not the CEC at pH 7 used by Toth et al. 2015).\n"
            "  Output resolution (advanced): default 30 m.\n"
            "  Maximum area per run (advanced, default 5000 km2).\n"
            "  Parallel downloads (advanced, default 8).\n"
            "\n"
            "Data: CC-BY 4.0. Cite Hengl et al. (2021), Scientific Reports 11:6130.\n"
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
        self.addParameter(
            QgsProcessingParameterEnum(
                self.VARIABLES,
                "Variables",
                options=[
                    f"{v.code} - {v.label} ({v.units})"
                    + ("" if v.default else " [optional]")
                    for v in cat.VARIABLES
                ],
                allowMultiple=True,
                defaultValue=[i for i, v in enumerate(cat.VARIABLES) if v.default],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEPTHS,
                "Depth intervals",
                options=[d[0] for d in cat.DEPTHS],
                allowMultiple=True,
                defaultValue=list(range(len(cat.DEPTHS))),
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STATISTICS,
                "Statistics",
                options=[label for _, label in cat.STATISTICS],
                allowMultiple=True,
                defaultValue=[0, 1],
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
                    defaultValue=30.0,
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

    def _selection(self, parameters, context):
        variables = [
            cat.VARIABLES[i].code
            for i in self.parameterAsEnums(parameters, self.VARIABLES, context)
        ]
        depths = [
            cat.DEPTHS[i][0]
            for i in self.parameterAsEnums(parameters, self.DEPTHS, context)
        ]
        stats = [
            cat.STATISTICS[i][0]
            for i in self.parameterAsEnums(parameters, self.STATISTICS, context)
        ]
        try:
            return core.plan_selection(variables, depths, stats)
        except core.SoilDataError as exc:
            raise QgsProcessingException(str(exc)) from exc

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
        from mayim_tools.soil._common.gdal_io import gdal_version

        gdal_ver = gdal_version()
        feedback.pushInfo(f"GDAL {gdal_ver}; reading from https://{cat.HOST}/")
        try:
            if mode_idx == 0:
                return self._run_area(
                    parameters, context, feedback, selection, common, gdal_ver
                )
            return self._run_points(
                parameters, context, feedback, selection, common, gdal_ver
            )
        except core.SoilDataError as exc:
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
        bounds_out, bounds_ll, desc = qgis_ui.area_bounds(
            self, parameters, context, out_crs, feedback, self.AREA_LAYER, self.EXTENT
        )
        area = check_area(bounds_ll, max_area)
        grid = make_grid(bounds_out, res, qgis_ui.crs_wkt(out_crs))
        feedback.pushInfo(
            f"Area of interest ({desc}): {area:.1f} km2 (bounding box); output "
            f"grid {grid.width} x {grid.height} cells at {res:g} "
            f"{QgsUnitTypes.toString(out_crs.mapUnits())} in {out_crs.authid()}."
        )
        result = core.extract_area(
            selection, grid, folder, write_fn=_write_geotiff, **common
        )
        settings = {
            "Area of interest": desc,
            "Bounds (output CRS)": ", ".join(f"{v:.3f}" for v in bounds_out),
            "Bounds (lon/lat)": ", ".join(f"{v:.6f}" for v in bounds_ll),
            "Area (km2, bounding box)": round(area, 1),
            "Output CRS": out_crs.authid() or out_crs.description(),
            "Output resolution": f"{res:g} ({res_m:g} m requested)",
            "Grid size (cells)": f"{grid.width} x {grid.height}",
            "Resampling": "nearest neighbour",
        }
        meta_path = metadata_path_for(folder, "area", METADATA_NAME)
        write_metadata_csv(
            core.build_metadata(result, selection, settings, gdal_ver), meta_path
        )
        self._report(result, feedback)
        feedback.pushInfo(f"{len(result.files)} raster(s) written to {folder}")
        feedback.pushInfo(f"Metadata: {meta_path}")
        outputs = [
            qgis_ui.OutputLayer(
                path=path, name=Path(path).stem, main=path in result.load_files
            )
            for path in result.files
        ]
        qlr = str(Path(folder) / "isda_layers.qlr")
        if qgis_ui.write_layer_file(outputs, qlr, feedback):
            feedback.pushInfo(f"Layer file (reloads all outputs, styled): {qlr}")
        choice = self.parameterAsEnum(parameters, self.LOAD_LAYERS, context)
        qgis_ui.load_outputs(context, outputs, choice, self._post_processors)
        return {self.OUTPUT_FOLDER: folder}

    def _run_points(self, parameters, context, feedback, selection, common, gdal_ver):
        out_csv = self.parameterAsFileOutput(parameters, self.OUTPUT_CSV, context)
        if not out_csv:
            raise QgsProcessingException("Points mode needs an output CSV.")
        sites = [
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
        feedback.pushInfo(f"{len(sites)} site(s) to sample.")
        result = core.extract_points(selection, sites, **common)
        n_rows, n_missing = write_points_csv(result.rows, out_csv, core.POINT_COLUMNS)
        if n_missing:
            feedback.pushWarning(
                f"{n_missing} of {n_rows} values are missing (outside Africa, "
                "water or masked); written as empty cells, never as zero."
            )
        meta_path = metadata_path_for(out_csv, "points")
        write_metadata_csv(
            core.build_metadata(
                result,
                selection,
                {"Sites": len(sites), "Sampling": "nearest cell"},
                gdal_ver,
            ),
            meta_path,
        )
        self._report(result, feedback)
        feedback.pushInfo(f"{n_rows} rows written to {out_csv}")
        feedback.pushInfo(f"Metadata: {meta_path}")
        return {self.OUTPUT_CSV: out_csv}

    @staticmethod
    def _report(result, feedback):
        feedback.pushInfo(f"Finished in {result.seconds:.0f} s.")
        for tex in result.texture:
            if tex["flagged"]:
                feedback.pushWarning(
                    f"Texture check {tex['depth']}: {tex['flagged']} of "
                    f"{tex['checked']} cells/points have mean sand+silt+clay more "
                    "than 2% from 100%."
                )
