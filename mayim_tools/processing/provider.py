"""
Mayim Tools - Processing Provider
=================================

Registers Mayim Tools algorithms in the QGIS Processing Toolbox.

Each tool is imported on its own (``ALGORITHMS`` below). When a tool
cannot be imported, usually because a Python package it needs is not
installed, only that tool is affected: it is listed in the toolbox as
"<name> (unavailable)", its help names the missing package and how to
install it, and running it stops with the same message. All other tools
load normally. The failure is also written to the QGIS message log
(Mayim Tools tab).
"""

from __future__ import annotations

import importlib
import traceback
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsMessageLog,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingProvider,
)
from qgis.PyQt.QtGui import QIcon

LOG_TAG = "Mayim Tools"

# Groups (display name, group id).
DATA = ("Data Tools", "data_tools")
HECRAS = ("HEC-RAS Tools", "hec_ras_tools")
HYDRO = ("Hydrological Tools", "hydrological_tools")
LANDCOVER = ("Land Cover Tools", "land_cover_tools")
RAIN = ("Rainfall Tools", "rainfall_tools")
SOIL = ("Soil Tools", "soil_tools")

# (module, class, algorithm id, display name, group). The id, name and
# group are only used for the placeholder of a tool that cannot load;
# a loaded tool reports its own.
ALGORITHMS = [
    (
        "rainfall.design_rainfall.design_rainfall_algorithm",
        "DesignRainfallPointAlgorithm",
        "design_rainfall_point",
        "Design Rainfall Estimation (South Africa)",
        RAIN,
    ),
    (
        "rainfall.chirps.chirps_extract_algorithm",
        "ChirpsExtractAlgorithm",
        "chirps_point_extract",
        "Extract: CHIRPS precipitation",
        RAIN,
    ),
    (
        "rainfall.cmorph_extract.cmorph_extract_algorithm",
        "CmorphExtractAlgorithm",
        "cmorph_point_extract",
        "Extract: CMORPH precipitation",
        RAIN,
    ),
    (
        "rainfall.imerg_extract.imerg_extract_algorithm",
        "ImergExtractAlgorithm",
        "imerg_point_extract",
        "Extract: IMERG precipitation",
        RAIN,
    ),
    (
        "rainfall.merra2_extract.merra2_extract_algorithm",
        "Merra2ExtractAlgorithm",
        "merra2_point_extract",
        "Extract: MERRA2 precipitation",
        RAIN,
    ),
    (
        "rainfall.frequency_analysis.rainfall_frequency_algorithm",
        "RainfallFrequencyAlgorithm",
        "rainfall_frequency_stage1",
        "Precipitation data to DDF",
        RAIN,
    ),
    (
        "rainfall.huff_curves.huff_curves_algorithm",
        "HuffCurvesAlgorithm",
        "huff_curves",
        "Precipitation data to Huff Curves",
        RAIN,
    ),
    (
        "data.grib_to_csv.grib_to_csv_algorithm",
        "GribToCsvAlgorithm",
        "grib_to_csv",
        "GRIB to CSV",
        DATA,
    ),
    (
        "data.landxml_import.landxml_import_algorithm",
        "LandXmlImportAlgorithm",
        "import_landxml_surface",
        "Import LandXML Surface",
        DATA,
    ),
    (
        "rainfall.ddf_to_hyetographs.ddf_to_hyetographs_algorithm",
        "DdfToHyetographsAlgorithm",
        "ddf_to_hyetographs",
        "DDF to Alternating Block Design Hyetographs",
        RAIN,
    ),
    (
        "rainfall.persiann_extract.persiann_extract_algorithm",
        "PersiannExtractAlgorithm",
        "persiann_point_extract",
        "Extract: PERSIANN CDR precipitation",
        RAIN,
    ),
    (
        "hydrology.d8_flow_direction.d8_flow_direction_algorithm",
        "D8FlowDirectionAlgorithm",
        "d8_flow_direction",
        "D8 Flow Direction (WBT)",
        HYDRO,
    ),
    (
        "hydrology.d8_flow_accumulation.d8_flow_accumulation_algorithm",
        "D8FlowAccumulationAlgorithm",
        "d8_flow_accumulation",
        "D8 Flow Accumulation (WBT)",
        HYDRO,
    ),
    (
        "data.netcdf4_to_csv.netcdf4_to_csv_algorithm",
        "Netcdf4ToCsvAlgorithm",
        "netcdf4_to_csv",
        "NetCDF4 to CSV",
        DATA,
    ),
    (
        "data.netcdf4_to_csv.netcdf4_list_variables_algorithm",
        "Netcdf4ListVariablesAlgorithm",
        "netcdf4_list_variables",
        "List NetCDF4 Variables",
        DATA,
    ),
    (
        "rainfall.design_storm_ensembles.design_storm_ensembles_algorithm",
        "DesignStormEnsembleAlgorithm",
        "generate_design_storm_ensembles",
        "Design Storm Ensembles",
        RAIN,
    ),
    (
        "rainfall.era5_extract.era5_extract_algorithm",
        "Era5ExtractAlgorithm",
        "era5_point_extract",
        "Extract: ERA5 precipitation",
        RAIN,
    ),
    (
        "hydrology.catchment_delineation.catchment_delineation_algorithm",
        "CatchmentDelineationAlgorithm",
        "catchment_delineation",
        "Catchments: 1. Delineation",
        HYDRO,
    ),
    (
        "rainfall.storm_library.storm_library_algorithm",
        "AdjustToDdfAlgorithm",
        "adjust_subdaily_to_ddf",
        "Adjust Sub-daily Rainfall to DDF",
        RAIN,
    ),
    (
        "hec_ras.import_ras_2d_data.import_ras_2d_data_algorithm",
        "ImportRas2dDataAlgorithm",
        "import_ras_2d_data",
        "Import RAS 2D data",
        HECRAS,
    ),
    (
        "hydrology.terrain_storage.terrain_storage_algorithm",
        "DemDepressionStageStorageAlgorithm",
        "dem_depression_stage_storage",
        "DEM depression stage-storage",
        HYDRO,
    ),
    (
        "hydrology.stream_network.stream_network_algorithm",
        "StreamNetworkAlgorithm",
        "stream_network",
        "Stream Network",
        HYDRO,
    ),
    (
        "soil.soilgrids_extract.soilgrids_extract_algorithm",
        "SoilGridsExtractAlgorithm",
        "soilgrids_extract",
        "Extract: SoilGrids 2.0",
        SOIL,
    ),
    (
        "soil.openlandmap_extract.openlandmap_extract_algorithm",
        "OpenLandMapExtractAlgorithm",
        "openlandmap_soils_extract",
        "Extract: OpenLandMap Soils",
        SOIL,
    ),
    (
        "soil.isda_extract.isda_extract_algorithm",
        "IsdaExtractAlgorithm",
        "isda_extract",
        "Extract: iSDAsoil (Africa)",
        SOIL,
    ),
    (
        "soil.regional_parameterisation.regional_parameterisation_algorithm",
        "RegionalSoilParameterisationAlgorithm",
        "regional_soil_parameterisation",
        "Regional Soil Parameterisation",
        SOIL,
    ),
    (
        "soil.hydrologic_soil_groups.hydrologic_soil_groups_algorithm",
        "HydrologicSoilGroupsAlgorithm",
        "hydrologic_soil_groups",
        "Hydrological Soil Groups (HSG)",
        SOIL,
    ),
    (
        "land_cover.land_cover_extract.land_cover_extract_algorithm",
        "LandCoverExtractAlgorithm",
        "land_cover_extract",
        "Extract: Land Cover",
        LANDCOVER,
    ),
    (
        "land_cover.curve_numbers.curve_numbers_algorithm",
        "CurveNumbersAlgorithm",
        "curve_numbers",
        "SCS Curve Numbers",
        LANDCOVER,
    ),
    (
        "land_cover.rational_c.rational_c_algorithm",
        "RationalCAlgorithm",
        "rational_c",
        "Run-off Coefficients (SANRAL Deterministic)",
        LANDCOVER,
    ),
    (
        "hydrology.dem_hydro_processing.dem_hydro_processing_algorithm",
        "DemHydroProcessingAlgorithm",
        "dem_hydro_processing",
        "DEM Hydrological Processing",
        HYDRO,
    ),
    (
        "hydrology.sanral_peak_flows.sanral_peak_flows_algorithm",
        "SanralPeakFlowsAlgorithm",
        "sanral_peak_flows",
        "SANRAL Deterministic Peak Flows",
        HYDRO,
    ),
    (
        "hydrology.time_of_concentration.time_of_concentration_algorithm",
        "TimeOfConcentrationAlgorithm",
        "time_of_concentration",
        "Catchments: 2. Time of Concentration",
        HYDRO,
    ),
    (
        "hydrology.inundation_boundary.inundation_boundary_algorithm",
        "InundationBoundaryAlgorithm",
        "inundation_boundary",
        "DEM Inundation Boundary",
        HYDRO,
    ),
    (
        "hydrology.channel_bathymetry.channel_bathymetry_algorithm",
        "ChannelBathymetryAlgorithm",
        "channel_bathymetry",
        "DEM Channel Bathymetry",
        HYDRO,
    ),
    (
        "hydrology.terrain_profile.terrain_profile_algorithm",
        "TerrainProfileAlgorithm",
        "terrain_profile",
        "Flood Extent Terrain Profile",
        HYDRO,
    ),
    (
        "hydrology.dws_wr2012.dws_wr2012_algorithm",
        "DwsWr2012Algorithm",
        "dws_wr2012_extract",
        "Extract: DWS and WR2012 data",
        HYDRO,
    ),
    (
        "data.hec_hms_project.prepare_hec_hms_project_algorithm",
        "PrepareHecHmsProjectAlgorithm",
        "prepare_hec_hms_project",
        "Prepare HEC-HMS project",
        DATA,
    ),
    (
        "data.swmm_project.prepare_swmm_project_algorithm",
        "PrepareSwmmProjectAlgorithm",
        "prepare_swmm_project",
        "Prepare SWMM project",
        DATA,
    ),
    (
        "data.data_boundary.data_boundary_algorithm",
        "DataBoundaryAlgorithm",
        "data_boundary",
        "Data Boundary (mask polygon)",
        DATA,
    ),
]

# Import name -> pip package name, where they differ.
PIP_NAMES = {
    "docx": "python-docx",
    "osgeo": "GDAL (part of QGIS)",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
}


def missing_package(exc: BaseException) -> str | None:
    """The top-level name of the package that could not be imported, or
    None when the failure is not a missing package."""
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        return exc.name.split(".")[0]
    return None


def unavailable_message(display_name: str, exc: BaseException) -> str:
    """User-facing explanation of why a tool could not be loaded."""
    package = missing_package(exc)
    if package is not None and not package.startswith("mayim_tools"):
        pip = PIP_NAMES.get(package, package)
        return (
            f"{display_name} needs the Python package '{pip}', which is "
            f"not installed in this QGIS.\n\n"
            "To install it on Windows, close QGIS, open the OSGeo4W Shell "
            "(Start menu, QGIS folder) and run:\n\n"
            f"    python -m pip install {pip}\n\n"
            "then restart QGIS. The other Mayim Tools are not affected."
        )
    return (
        f"{display_name} could not be loaded because of an error in the "
        f"tool:\n\n{type(exc).__name__}: {exc}\n\n"
        "The full traceback is in the QGIS message log (Mayim Tools tab). "
        "The other Mayim Tools are not affected."
    )


class UnavailableAlgorithm(QgsProcessingAlgorithm):
    """Placeholder listed for a tool that could not be imported."""

    def __init__(self, alg_id, display_name, group, message):
        super().__init__()
        self._id = alg_id
        self._display_name = display_name
        self._group, self._group_id = group
        self._message = message

    def createInstance(self):  # noqa: N802
        return UnavailableAlgorithm(
            self._id,
            self._display_name,
            (self._group, self._group_id),
            self._message,
        )

    def name(self):
        return self._id

    def displayName(self):  # noqa: N802
        return f"{self._display_name} (unavailable)"

    def group(self):
        return self._group

    def groupId(self):  # noqa: N802
        return self._group_id

    def shortHelpString(self):  # noqa: N802
        return self._message.replace("\n", "<br>")

    def initAlgorithm(self, config=None):  # noqa: N802
        pass

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        raise QgsProcessingException(self._message)


def load_algorithm(module, class_name, alg_id, display_name, group):
    """An instance of the tool, or an UnavailableAlgorithm when its module
    cannot be imported or the tool cannot be created."""
    try:
        mod = importlib.import_module(f"mayim_tools.{module}")
        return getattr(mod, class_name)()
    except Exception as exc:  # noqa: BLE001 - any failure isolates one tool
        QgsMessageLog.logMessage(
            f"{display_name} could not be loaded:\n{traceback.format_exc()}",
            LOG_TAG,
            Qgis.Warning,
        )
        return UnavailableAlgorithm(
            alg_id, display_name, group, unavailable_message(display_name, exc)
        )


class MayimToolsProvider(QgsProcessingProvider):
    """Mayim Tools Processing provider."""

    def id(self) -> str:  # noqa: A003 - required by the QGIS API.
        """Return the unique provider ID."""
        return "mayimtools"

    def name(self) -> str:
        """Return the provider display name."""
        return "Mayim Tools"

    def icon(self) -> QIcon:
        """Return the provider icon."""
        logo = Path(__file__).resolve().parents[1] / "icons" / "mayim_logo.png"
        return QIcon(str(logo))

    def loadAlgorithms(self) -> None:  # noqa: N802 - required by QGIS.
        """Register Mayim Tools algorithms, each one independently."""
        for entry in ALGORITHMS:
            self.addAlgorithm(load_algorithm(*entry))
