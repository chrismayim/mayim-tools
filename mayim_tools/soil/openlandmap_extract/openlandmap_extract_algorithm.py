"""Processing Toolbox algorithm: Extract: OpenLandMap Soils.

Area mode (default): an extent or polygon layer in, one multi-band GeoTIFF
per variable x statistic x period out (one band per depth). Point mode: a
point or point layer in, a long-format CSV out. Both modes always write a
metadata CSV. Thin wrapper - the logic is in core.py and catalogue.py (no
QGIS; see tests/test_openlandmap_extract_core.py).
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
from mayim_tools.soil._common.export import (
    metadata_path_for,
    write_metadata_csv,
    write_multiband_geotiff,
    write_paletted_style,
    write_points_csv,
)
from mayim_tools.soil._common.grid import check_area, make_grid

from . import catalogue as cat
from . import core

STAT_OPTIONS = [(s.code, s.label) for s in cat.STATISTICS] + [
    (cat.RU68, "RU68 - relative 68% interval width (derived, 120 m)")
]
METADATA_NAME = "openlandmap_metadata.csv"


def _write_geotiff(path, grid, bands, units):
    write_multiband_geotiff(
        path,
        grid,
        bands,
        units,
        source="OpenLandMap-soildb (via mayim_tools)",
        tool_version=core.TOOL_VERSION,
    )


class OpenLandMapExtractAlgorithm(QgsProcessingAlgorithm):
    MODE = "MODE"
    EXTENT = "EXTENT"
    AREA_LAYER = "AREA_LAYER"
    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    VARIABLES = "VARIABLES"
    DEPTHS = "DEPTHS"
    PERIODS = "PERIODS"
    STATISTICS = "STATISTICS"
    SUBGROUPS = "SUBGROUPS"
    WATER = "WATER"
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
        return OpenLandMapExtractAlgorithm()

    def name(self):
        return "openlandmap_soils_extract"

    def displayName(self):
        return "Extract: OpenLandMap Soils"

    def group(self):
        return "Soil Tools"

    def groupId(self):
        return "soil_tools"

    def shortHelpString(self):
        return (
            f"Extract: OpenLandMap Soils (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tDownloads OpenLandMap-soildb soil properties (Hengl et al. "
            "2026) for an area or for points: sand, silt, clay, organic carbon, "
            "bulk density and pH (SOC density optional) for 0-30, 30-60 and "
            "60-100 cm, with prediction intervals, for later soil texture, "
            "hydraulic-property, hydrologic soil group and curve-number tools. "
            "Global, except deserts and permanent ice.\n"
            "\n"
            "RESOLUTION AND STATISTICS:\tMeans are published at 30 m. The "
            "uncertainty layers - P16 and P84, the 68% prediction interval "
            "(about one standard deviation either side) - exist only at 120 m, "
            "together with a 120 m mean. RU68 = (P84 - P16) / mean is derived "
            "from the three 120 m layers. On a 30 m output grid the 120 m layers "
            "repeat in 4 x 4 blocks.\n"
            "\n"
            "PERIODS:\tSOC, SOC density and pH are mapped for 2000-2005, "
            "2005-2010, 2010-2015, 2015-2020 and 2020-2022. Texture is mapped "
            "for 2020-2022 only and is used for any period selected. Bulk "
            "density is 2020-2022 at 30 m but every period at 120 m. Layers that "
            "are not published are skipped and listed in the log and metadata.\n"
            "\n"
            "IMPORTANT:\tTexture follows ISO 11277 (silt 2-63 um, sand from 63 "
            "um), not USDA (50 um). Convert before using the USDA texture "
            "triangle or USDA-based pedotransfer functions. Values are means, "
            "not medians. There are no coarse-fragment or CEC layers - use "
            "Extract: SoilGrids 2.0 for those.\n"
            "\n"
            "METHOD:\t\tReads only the requested window of each global "
            "cloud-optimised GeoTIFF from s3.opengeohub.org with QGIS's own GDAL "
            "(no account, no extra packages), several layers at once, with an "
            "http fallback. Stored values are converted with the scale written "
            "in each file (catalogue scale as a fallback). Nearest-neighbour "
            "resampling only.\n"
            "\n"
            "OUTPUTS:\tArea mode: one multi-band GeoTIFF per variable, statistic "
            "and period, e.g. olm_clay_mean_30m_2020-2022.tif, one band per "
            "depth, each band described like 'clay_30-60cm_mean_30m_2020-2022 "
            "(%)' (later tools select bands by description). Each GeoTIFF has a "
            ".qml style beside it (single band). The 30 m means of the latest "
            "selected period are loaded. Point mode: long-format CSV. Both "
            "modes write openlandmap_metadata.csv (or <name>_metadata.csv) with "
            "sources, scales, routes, per-layer statistics, RU68 summaries, a "
            "sand+silt+clay check, period notes and warnings.\n"
            "\n"
            "PARAMETERS:\n"
            "  Area: an extent OR a polygon layer (its extent, buffered by two "
            "cells). Points: a point OR a point layer with an optional name "
            "field.\n"
            "  Output resolution (advanced): default 30 m (converted to degrees "
            "for a geographic CRS).\n"
            "  Maximum area per run (advanced, default 5000 km2 - 30 m data is "
            "16 times denser than 120 m). Raise it deliberately, or use 120 m "
            "output, for large areas.\n"
            "  Parallel downloads (advanced, default 8).\n"
            "  Water content (optional): the older OpenLandMap 250 m maps of "
            "volumetric water content at 33 kPa (field capacity) and 1500 kPa "
            "(wilting point), 1950-2017, mapped from measured values rather "
            "than pedotransfer functions - an independent check on computed "
            "field capacity and wilting point. Published at depths 0, 30, 60 "
            "and 100 cm; each interval is the average of its two bounding "
            "depths. Available water capacity (mm) = (FC - WP) x layer "
            "thickness is derived. Outputs olm_wc33_*, olm_wc1500_* and "
            "olm_awc_*. Licence CC BY-SA 4.0 (share-alike).\n"
            "  USDA subgroup (optional): OpenLandMap publishes the probability of "
            "each of 818 USDA soil-taxonomy subgroups at 30 m. The tool keeps "
            "the most probable and second most probable subgroup per cell, with "
            "their probabilities and the sum of all probabilities, in "
            "olm_usda_subgroup_2000-2022.tif (band 1 = subgroup code, styled by "
            "name) plus a lookup CSV. Reading 818 layers takes noticeably "
            "longer. Treat as indicative where the top probability is low.\n"
            "\n"
            "Data: CC-BY 4.0. Cite Hengl et al. (2026), ESSD 18:989.\n"
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
                self.PERIODS,
                "Periods",
                options=list(cat.PERIODS),
                allowMultiple=True,
                defaultValue=[list(cat.PERIODS).index(cat.LATEST_PERIOD)],
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STATISTICS,
                "Statistics",
                options=[label for _, label in STAT_OPTIONS],
                allowMultiple=True,
                defaultValue=[0, 2, 3],
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.WATER,
                "Also extract water content at 33 and 1500 kPa and available "
                "water capacity (OpenLandMap 250 m, 1950-2017, measured-data "
                "maps)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SUBGROUPS,
                "Also map the most probable USDA soil subgroup (reads 818 "
                "probability layers; slower)",
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
            QgsProcessingParameterBoolean(
                self.LOAD_LAYERS,
                "Load the 30 m mean rasters into the project (area mode)",
                defaultValue=True,
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

    def _selection(self, parameters, context):
        variables = [
            cat.VARIABLES[i].code
            for i in self.parameterAsEnums(parameters, self.VARIABLES, context)
        ]
        depths = [
            cat.DEPTHS[i][0]
            for i in self.parameterAsEnums(parameters, self.DEPTHS, context)
        ]
        periods = [
            list(cat.PERIODS)[i]
            for i in self.parameterAsEnums(parameters, self.PERIODS, context)
        ]
        stat_codes = [
            STAT_OPTIONS[i][0]
            for i in self.parameterAsEnums(parameters, self.STATISTICS, context)
        ]
        ru68 = cat.RU68 in stat_codes
        stats = [s for s in stat_codes if s != cat.RU68]
        subgroups = self.parameterAsBoolean(parameters, self.SUBGROUPS, context)
        water = self.parameterAsBoolean(parameters, self.WATER, context)
        try:
            return core.plan_selection(
                variables,
                depths,
                periods,
                stats,
                ru68,
                subgroups=subgroups,
                water=water,
            )
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
        sub_path = result.subgroup.get("file")
        if sub_path:
            self._subgroup_outputs(result, sub_path, feedback)
        if self.parameterAsBoolean(parameters, self.LOAD_LAYERS, context):
            qgis_ui.load_rasters(
                context,
                [p for p in result.load_files if p != sub_path],
                self._post_processors,
            )
            if sub_path:
                qgis_ui.load_rasters(
                    context, [sub_path], self._post_processors, restyle=False
                )
        return {self.OUTPUT_FOLDER: folder}

    @staticmethod
    def _subgroup_outputs(result, sub_path, feedback):
        """Categorised style (band 1 = most probable subgroup code) and a
        lookup table of the subgroups present."""
        table = result.subgroup["table"]
        write_paletted_style(
            sub_path,
            [(r["code"], f"{r['label']} ({r['order']})") for r in table],
        )
        lookup = Path(sub_path).with_name(Path(sub_path).stem + "_lookup.csv")
        write_metadata_csv(
            [
                (
                    "USDA subgroups present (most probable), largest first",
                    [
                        "Code",
                        "Subgroup",
                        "Great group",
                        "Order",
                        "Cells",
                        "Share (%)",
                        "Mean top probability (%)",
                    ],
                    [
                        [
                            r["code"],
                            r["label"],
                            r["great_group"],
                            r["order"],
                            r["cells"],
                            round(r["share_pct"], 2),
                            round(r["mean_probability"], 1),
                        ]
                        for r in table
                    ],
                )
            ],
            lookup,
        )
        if table:
            main = table[0]
            feedback.pushInfo(
                f"USDA subgroups: {len(table)} present; most common "
                f"{main['label']} ({main['share_pct']:.0f}% of cells, mean top "
                f"probability {main['mean_probability']:.0f}%). Lookup: {lookup}"
            )

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
                f"{n_missing} of {n_rows} values are missing (not mapped at the "
                "point, e.g. water, desert or outside coverage); written as empty "
                "cells, never as zero."
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
                    f"Texture check {tex['period']} {tex['depth']}: "
                    f"{tex['flagged']} of {tex['checked']} cells/points have mean "
                    "sand+silt+clay more than 2% from 100%."
                )
        for unc in result.uncertainty:
            feedback.pushInfo(
                f"RU68 {unc['variable']} {unc['period']} {unc['depth']}: median "
                f"{unc['median_ru68']:.2f}"
            )
