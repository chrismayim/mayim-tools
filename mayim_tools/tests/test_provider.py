"""
Tests for the Mayim Tools Processing provider.

These tests require the qgis package. When run outside the QGIS Python
environment, they are skipped rather than failing collection.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qgis")


@pytest.fixture(scope="module", autouse=True)
def qgis_app():
    """A QgsApplication, which addAlgorithm needs outside QGIS."""
    from qgis.core import QgsApplication

    app = QgsApplication.instance()
    if app is None:
        app = QgsApplication([], False)
        app.initQgis()
    yield app


def loaded_provider():
    """Create a provider with all Mayim Tools algorithms loaded."""
    from mayim_tools.processing.provider import MayimToolsProvider

    provider = MayimToolsProvider()
    provider.loadAlgorithms()
    return provider


def test_provider_id() -> None:
    provider = loaded_provider()

    assert provider.id() == "mayimtools"


def test_provider_name() -> None:
    provider = loaded_provider()

    assert provider.name() == "Mayim Tools"


def test_provider_registers_all_algorithms() -> None:
    provider = loaded_provider()

    algorithms = provider.algorithms()
    assert len(algorithms) == 42

    by_name = {algorithm.name(): algorithm for algorithm in algorithms}

    assert set(by_name) == {
        "design_rainfall_point",
        "chirps_point_extract",
        "rainfall_frequency_stage1",
        "huff_curves",
        "grib_to_csv",
        "import_landxml_surface",
        "ddf_to_hyetographs",
        "imerg_point_extract",
        "merra2_point_extract",
        "cmorph_point_extract",
        "persiann_point_extract",
        "d8_flow_direction",
        "netcdf4_to_csv",
        "netcdf4_list_variables",
        "generate_design_storm_ensembles",
        "d8_flow_accumulation",
        "era5_point_extract",
        "catchment_delineation",
        "adjust_subdaily_to_ddf",
        "import_ras_2d_data",
        "import_ras_results",
        "dem_depression_stage_storage",
        "inundation_boundary",
        "stream_network",
        "soilgrids_extract",
        "openlandmap_soils_extract",
        "regional_soil_parameterisation",
        "hydrologic_soil_groups",
        "isda_extract",
        "land_cover_extract",
        "curve_numbers",
        "dem_hydro_processing",
        "contours_to_dem",
        "sanral_peak_flows",
        "time_of_concentration",
        "terrain_profile",
        "dws_wr2012_extract",
        "rational_c",
        "prepare_hec_hms_project",
        "prepare_swmm_project",
        "data_boundary",
        "channel_bathymetry",
    }

    expected = {
        "design_rainfall_point": (
            "Design Rainfall Estimation (South Africa)",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "chirps_point_extract": (
            "Extract: CHIRPS precipitation",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "rainfall_frequency_stage1": (
            "Precipitation data to DDF",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "huff_curves": (
            "Precipitation data to Huff Curves",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "grib_to_csv": (
            "GRIB to CSV",
            "Data Tools",
            "data_tools",
        ),
        "import_landxml_surface": (
            "Import LandXML Surface",
            "Data Tools",
            "data_tools",
        ),
        "ddf_to_hyetographs": (
            "DDF to Alternating Block Design Hyetographs",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "imerg_point_extract": (
            "Extract: IMERG precipitation",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "merra2_point_extract": (
            "Extract: MERRA2 precipitation",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "cmorph_point_extract": (
            "Extract: CMORPH precipitation",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "persiann_point_extract": (
            "Extract: PERSIANN CDR precipitation",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "d8_flow_direction": (
            "D8 Flow Direction (WBT)",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "netcdf4_to_csv": (
            "NetCDF4 to CSV",
            "Data Tools",
            "data_tools",
        ),
        "netcdf4_list_variables": (
            "List NetCDF4 Variables",
            "Data Tools",
            "data_tools",
        ),
        "generate_design_storm_ensembles": (
            "Design Storm Ensembles",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "d8_flow_accumulation": (
            "D8 Flow Accumulation (WBT)",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "era5_point_extract": (
            "Extract: ERA5 precipitation",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "catchment_delineation": (
            "Catchments: 1. Delineation",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "adjust_subdaily_to_ddf": (
            "Adjust Sub-daily Rainfall to DDF",
            "Rainfall Tools",
            "rainfall_tools",
        ),
        "import_ras_2d_data": (
            "Import RAS 2D geometry data",
            "HEC-RAS Tools",
            "hec_ras_tools",
        ),
        "import_ras_results": (
            "Import RAS result data",
            "HEC-RAS Tools",
            "hec_ras_tools",
        ),
        "dem_depression_stage_storage": (
            "DEM depression stage-storage",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "stream_network": (
            "Stream Network",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "soilgrids_extract": (
            "Extract: SoilGrids 2.0",
            "Soil Tools",
            "soil_tools",
        ),
        "openlandmap_soils_extract": (
            "Extract: OpenLandMap Soils",
            "Soil Tools",
            "soil_tools",
        ),
        "regional_soil_parameterisation": (
            "Regional Soil Parameterisation",
            "Soil Tools",
            "soil_tools",
        ),
        "isda_extract": (
            "Extract: iSDAsoil (Africa)",
            "Soil Tools",
            "soil_tools",
        ),
        "hydrologic_soil_groups": (
            "Hydrological Soil Groups (HSG)",
            "Soil Tools",
            "soil_tools",
        ),
        "land_cover_extract": (
            "Extract: Land Cover",
            "Land Cover Tools",
            "land_cover_tools",
        ),
        "curve_numbers": (
            "SCS Curve Numbers",
            "Land Cover Tools",
            "land_cover_tools",
        ),
        "rational_c": (
            "Run-off Coefficients (SANRAL Deterministic)",
            "Land Cover Tools",
            "land_cover_tools",
        ),
        "dem_hydro_processing": (
            "DEM Hydrological Processing",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "contours_to_dem": (
            "Contours to DEM",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "sanral_peak_flows": (
            "SANRAL Deterministic Peak Flows",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "time_of_concentration": (
            "Catchments: 2. Time of Concentration",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "inundation_boundary": (
            "DEM Inundation Boundary",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "channel_bathymetry": (
            "DEM Channel Bathymetry",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "terrain_profile": (
            "Flood Extent Terrain Profile",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "dws_wr2012_extract": (
            "Extract: DWS and WR2012 data",
            "Hydrological Tools",
            "hydrological_tools",
        ),
        "prepare_hec_hms_project": (
            "Prepare HEC-HMS project",
            "Data Tools",
            "data_tools",
        ),
        "prepare_swmm_project": (
            "Prepare SWMM project",
            "Data Tools",
            "data_tools",
        ),
        "data_boundary": (
            "Data Boundary (mask polygon)",
            "Data Tools",
            "data_tools",
        ),
    }

    for name, (display_name, group, group_id) in expected.items():
        algorithm = by_name[name]
        assert algorithm.displayName() == display_name
        assert algorithm.group() == group
        assert algorithm.groupId() == group_id


def test_provider_lists_every_algorithm_once() -> None:
    from mayim_tools.processing.provider import ALGORITHMS

    ids = [entry[2] for entry in ALGORITHMS]
    assert len(ids) == len(set(ids)) == 42


def test_missing_package_gives_placeholder() -> None:
    from mayim_tools.processing.provider import (
        UnavailableAlgorithm,
        load_algorithm,
    )

    alg = load_algorithm(
        "no_such_package_for_test.module",
        "Nothing",
        "grib_to_csv",
        "GRIB to CSV",
        ("Data Tools", "data_tools"),
    )
    assert isinstance(alg, UnavailableAlgorithm)
    assert alg.name() == "grib_to_csv"
    assert alg.displayName() == "GRIB to CSV (unavailable)"
    assert alg.groupId() == "data_tools"
    # A missing mayim_tools module is reported as a tool error, not a
    # package to install.
    assert "error in the tool" in alg.shortHelpString()


def test_unavailable_message_names_pip_package() -> None:
    from mayim_tools.processing.provider import unavailable_message

    exc = ModuleNotFoundError("No module named 'docx'", name="docx")
    msg = unavailable_message("Some tool", exc)
    assert "python -m pip install python-docx" in msg
    exc = ModuleNotFoundError("No module named 'xarray.core'", name="xarray.core")
    assert "pip install xarray" in unavailable_message("GRIB to CSV", exc)
