# -*- coding: utf-8 -*-
"""
Tests for DEMHydrologicalScreening tool.
Tests the tool registration, parameter definition,
and algorithm metadata — without requiring a live QGIS instance.
"""

import pytest


class TestDEMHydrologicalScreeningMetadata:
    """Test tool metadata and registration."""

    def test_tool_name(self):
        """Tool name must match the defined constant."""
        from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
            DEMHydrologicalScreening,
        )
        tool = DEMHydrologicalScreening()
        assert tool.name() == "demhydrologicalscreening"

    def test_tool_display_name(self):
        """Display name must be human readable."""
        from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
            DEMHydrologicalScreening,
        )
        tool = DEMHydrologicalScreening()
        assert tool.displayName() == "DEM Hydrological Screening"

    def test_tool_group(self):
        """Tool must belong to Hydrology Tools group."""
        from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
            DEMHydrologicalScreening,
        )
        tool = DEMHydrologicalScreening()
        assert tool.group() == "Hydrology Tools"
        assert tool.groupId() == "hydrology"

    def test_tool_create_instance(self):
        """createInstance must return a new instance of the same class."""
        from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
            DEMHydrologicalScreening,
        )
        tool = DEMHydrologicalScreening()
        instance = tool.createInstance()
        assert isinstance(instance, DEMHydrologicalScreening)
        assert instance is not tool  # must be a NEW instance

    def test_tool_tags(self):
        """Tool must include expected tags."""
        from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
            DEMHydrologicalScreening,
        )
        tool = DEMHydrologicalScreening()
        tags = tool.tags()
        assert "dem" in tags
        assert "hydrology" in tags
        assert "screening" in tags

    def test_dem_source_type_labels(self):
        """DEMSourceType must have correct number of labels."""
        from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
            DEMSourceType,
        )
        assert len(DEMSourceType.LABELS) == 8
        assert DEMSourceType.LABELS[0] == "Auto-detect"
        assert DEMSourceType.LABELS[3] == "SRTM (30m / 90m)"

    def test_dem_source_type_rmse_defaults(self):
        """All source types must have a default RMSE value."""
        from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
            DEMSourceType,
        )
        for key, value in DEMSourceType.DEFAULT_RMSE.items():
            if key != DEMSourceType.AUTO_DETECT:
                assert value is not None, (
                    f"Source type {key} has no default RMSE"
                )
                assert value > 0, (
                    f"Source type {key} RMSE must be positive"
                )

    def test_void_class_values(self):
        """VoidClass values must be distinct and correct."""
        from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
            VoidClass,
        )
        values = [
            VoidClass.VALID,
            VoidClass.SMALL,
            VoidClass.MEDIUM,
            VoidClass.LARGE,
        ]
        assert len(values) == len(set(values)), (
            "VoidClass values must all be unique"
        )
        assert VoidClass.VALID == 0

    def test_category_registers_tool(self):
        """HydrologyCategory.get_algorithms() must return the tool."""
        from mayim_tools.categories.hydrology.category import HydrologyCategory
        from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
            DEMHydrologicalScreening,
        )
        cat = HydrologyCategory()
        algorithms = cat.get_algorithms()
        assert len(algorithms) == 1
        assert isinstance(algorithms[0], DEMHydrologicalScreening)
