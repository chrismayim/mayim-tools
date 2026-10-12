"""Processing Toolbox algorithm: Extract: Land cover.

Area mode (default): an extent or polygon layer in, one class GeoTIFF per
year out (Byte, embedded colour table and class names). Point mode: a
point or point layer in, a CSV of the class at each site out. Both modes
write a metadata CSV with sources, citations, tiles and class areas. Thin
wrapper - the logic is in core.py and catalogue.py (no QGIS; see
tests/test_land_cover_extract_core.py).

Products: ESA WorldCover 10 m (2020 v100, 2021 v200). The South African
National Land Cover is planned as a second product.
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
    write_class_geotiff,
    write_metadata_csv,
    write_points_csv,
)
from mayim_tools.soil._common.grid import check_area, make_grid

from . import catalogue as cat
from . import core

METADATA_NAME = "worldcover_metadata.csv"
PRODUCT_OPTIONS = ["ESA WorldCover 10 m (global)"]
YEAR_OPTIONS = [
    (2021, "2021 (v200, improved algorithm - recommended)"),
    (2020, "2020 (v100)"),
]
CLASS_STYLE = [(c.code, f"{c.code} {c.name}", c.colour) for c in cat.CLASSES]


def _write_classes(path, grid, arr, year):
    ed = cat.EDITION_BY_YEAR[year]
    write_class_geotiff(
        path,
        grid,
        arr,
        cat.band_description(year),
        [(c.code, c.name, c.colour) for c in cat.CLASSES],
        nodata=cat.NODATA,
        metadata={
            "SOURCE": f"{cat.PRODUCT_LONG} {year} {ed.version} (via mayim_tools)",
            "ATTRIBUTION": ed.attribution,
            "LICENCE": cat.LICENCE,
            "TOOL_VERSION": core.TOOL_VERSION,
        },
    )


class LandCoverExtractAlgorithm(QgsProcessingAlgorithm):
    MODE = "MODE"
    PRODUCT = "PRODUCT"
    YEARS = "YEARS"
    EXTENT = "EXTENT"
    AREA_LAYER = "AREA_LAYER"
    POINT = "POINT"
    INPUT_POINTS = "INPUT_POINTS"
    NAME_FIELD = "NAME_FIELD"
    OUTPUT_CRS = "OUTPUT_CRS"
    OUTPUT_RES = "OUTPUT_RES"
    MAX_AREA_KM2 = "MAX_AREA_KM2"
    WORKERS = "WORKERS"
    LOAD_LAYERS = "LOAD_LAYERS"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"
    OUTPUT_CSV = "OUTPUT_CSV"

    MODE_OPTIONS = ["Area (class GeoTIFF)", "Points (CSV)"]

    def __init__(self):
        super().__init__()
        self._post_processors = []

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return LandCoverExtractAlgorithm()

    def name(self):
        return "land_cover_extract"

    def displayName(self):
        return "Extract: Land cover"

    def group(self):
        return "Land Cover Tools"

    def groupId(self):
        return "land_cover_tools"

    def shortHelpString(self):
        return (
            f"Extract: Land cover (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tDownloads land cover for an area or for points, as the "
            "land-use input for curve numbers and other runoff parameters. "
            "Product: ESA WorldCover 10 m (Zanaga et al. 2021, 2022), global, for "
            "2021 (v200, recommended) and 2020 (v100), with 11 classes: tree "
            "cover, shrubland, grassland, cropland, built-up, bare / sparse "
            "vegetation, snow and ice, permanent water, herbaceous wetland, "
            "mangroves, and moss and lichen.\n"
            "\n"
            "METHOD:\t\tFinds the 3 x 3 degree WorldCover tiles that touch the "
            "area and reads only the parts it needs from the public cloud-"
            "optimised GeoTIFFs on Amazon S3 with QGIS's own GDAL (no account, no "
            "extra packages), with an unsigned S3 fallback. Tiles that do not "
            "exist (open sea) are skipped. Tiles are mosaicked and reprojected "
            "with nearest neighbour, so class codes are never averaged.\n"
            "\n"
            "ACCURACY:\tAbout 77 % overall for 2021 (76.5 % for Africa) and 74 % "
            "for 2020. Shrubland, herbaceous wetland and moss and lichen are the "
            "least accurate classes. The two years used different algorithms: "
            "do not use their difference as a change map. The classes say "
            "nothing about hydrological condition or the impervious share of "
            "built-up land; those are set in the curve-number step.\n"
            "\n"
            "OUTPUTS:\tArea mode: worldcover_<year>_map.tif (Byte, nodata 0, with "
            "the official colours and class names embedded) and "
            "worldcover_layers.qlr. Point mode: a CSV with the class code and "
            "name at each site per year. Both modes write worldcover_metadata.csv "
            "(or <name>_metadata.csv) with citations, the tiles used, class areas "
            "(or counts) and warnings.\n"
            "\n"
            "PARAMETERS:\n"
            "  Area: an extent OR a polygon layer (its extent, buffered by two "
            "cells). Points: a point OR a point layer with an optional name field.\n"
            "  Output resolution (advanced): default 10 m (native).\n"
            "  Maximum area per run (advanced, default 5000 km2).\n"
            "\n"
            "Data: CC BY 4.0. (c) ESA WorldCover project / Contains modified "
            "Copernicus Sentinel data processed by ESA WorldCover consortium.\n"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE, "Mode", options=self.MODE_OPTIONS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PRODUCT,
                "Land cover product",
                options=PRODUCT_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.YEARS,
                "Year(s)",
                options=[label for _, label in YEAR_OPTIONS],
                allowMultiple=True,
                defaultValue=[0],
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
                    defaultValue=10.0,
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
                    "Parallel tile checks",
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
        years = [
            YEAR_OPTIONS[i][0]
            for i in self.parameterAsEnums(parameters, self.YEARS, context)
        ]
        try:
            return core.plan_selection(years)
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
        if res_m > cat.RESOLUTION_M * 1.5:
            feedback.pushWarning(
                f"Output cells of {res_m:g} m are coarser than the native 10 m: "
                "each cell takes the class of the nearest source cell, so small "
                "patches can be lost or exaggerated."
            )
        result = core.extract_area(
            selection,
            grid,
            bounds_ll,
            folder,
            geographic=out_crs.isGeographic(),
            write_fn=_write_classes,
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
                path=path, name=Path(path).stem, main=True, classes=CLASS_STYLE
            )
            for path in result.files
        ]
        qlr = str(Path(folder) / "worldcover_layers.qlr")
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
        n_rows, _ = write_points_csv(result.rows, out_csv, core.POINT_COLUMNS)
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
        for rec in result.years:
            if result.mode != "area" or not rec.areas:
                continue
            total = sum(rec.areas.values())
            top = sorted(rec.areas.items(), key=lambda kv: -kv[1])[:4]
            text = "; ".join(
                f"{cat.class_name(c)} {100 * a / total:.0f} %" for c, a in top
            )
            feedback.pushInfo(f"WorldCover {rec.year}: {text}")
