"""
Tests for the Mayim Tools Processing provider.

These tests require the qgis package. When run outside the QGIS Python
environment, they are skipped rather than failing collection.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qgis")


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


def test_provider_registers_seven_algorithms() -> None:
    provider = loaded_provider()

    algorithms = provider.algorithms()
    assert len(algorithms) == 16

    by_name = {algorithm.name(): algorithm for algorithm in algorithms}

    assert set(by_name) == {
        "design_rainfall_point",
        "chirps_point_extract",
        "rainfall_frequency_stage1",
        "huff_curves",
        "grib_to_csv",
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
    }

    for name, (display_name, group, group_id) in expected.items():
        algorithm = by_name[name]
        assert algorithm.displayName() == display_name
        assert algorithm.group() == group
        assert algorithm.groupId() == group_id
