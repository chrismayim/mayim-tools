"""Tests for the DEM Gradient Resolution Processing adapter.

These tests validate metadata and parameter definitions only.
Raster execution is tested separately in the native Stage 6 tests.
"""

import pytest

pytest.importorskip("qgis")


class TestDEMGradientResolution:
    """Tests for DEMGradientResolution."""

    @pytest.fixture
    def algorithm(self):
        """Return a DEMGradientResolution instance."""
        from mayim_tools.categories.hydrology.tools.dem_gradient_resolution import (
            DEMGradientResolution,
        )

        return DEMGradientResolution()

    def test_algorithm_name(self, algorithm):
        """The Processing algorithm ID is correct."""
        assert algorithm.name() == "demgradientresolution"

    def test_display_name(self, algorithm):
        """The user-facing name is correct."""
        assert algorithm.displayName() == "DEM Gradient Resolution"

    def test_group(self, algorithm):
        """The algorithm belongs to Hydrology Tools."""
        assert algorithm.group() == "Hydrology Tools"
        assert algorithm.groupId() == "hydrology"

    def test_create_instance(self, algorithm):
        """createInstance returns a new adapter instance."""
        instance = algorithm.createInstance()

        assert isinstance(instance, algorithm.__class__)
        assert instance is not algorithm

    def test_help_contains_stage_6(self, algorithm):
        """Help text identifies Stage 6."""
        help_text = algorithm.shortHelpString()

        assert "Stage 6" in help_text
        assert "flat" in help_text.lower()
        assert "Garbrecht" in help_text

    def test_required_parameters_are_defined(self, algorithm):
        """Required adapter parameters are present."""
        algorithm.initAlgorithm()

        parameter_names = {
            parameter.name()
            for parameter in algorithm.parameterDefinitions()
        }

        assert "INPUT_DEM" in parameter_names
        assert "INPUT_MANIFEST" in parameter_names
        assert "VERTICAL_ACCURACY" in parameter_names
        assert "CELL_SIZE" in parameter_names
        assert "OUTPUT_FOLDER" in parameter_names

    def test_load_parameters_are_defined(self, algorithm):
        """Independent output-loading options are present."""
        algorithm.initAlgorithm()

        parameter_names = {
            parameter.name()
            for parameter in algorithm.parameterDefinitions()
        }

        assert "LOAD_RESOLVED_DEM" in parameter_names
        assert "LOAD_FLAT_MASK" in parameter_names
        assert "LOAD_DIFFERENCE" in parameter_names
        assert "LOAD_REGION_IDS" in parameter_names

    def test_required_outputs_are_defined(self, algorithm):
        """Expected Stage 6 outputs are present."""
        algorithm.initAlgorithm()

        output_names = {
            output.name()
            for output in algorithm.outputDefinitions()
        }

        assert "OUTPUT_RESOLVED_DEM" in output_names
        assert "OUTPUT_FLAT_MASK" in output_names
        assert "OUTPUT_REGION_IDS" in output_names
        assert "OUTPUT_DIFFERENCE" in output_names
        assert "OUTPUT_REPORT" in output_names
        assert "OUTPUT_PROVENANCE" in output_names
        assert "OUTPUT_MANIFEST" in output_names
