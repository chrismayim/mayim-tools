"""
Mayim Tools — Hydrology Category.

Descriptor class for the Hydrology tool category. Each tool is imported
inside its own try/except block so that a broken tool does not prevent
the remaining tools from loading.
"""

from __future__ import annotations

from mayim_tools.categories.base_category import BaseCategory
from mayim_tools.core.logger import MayimLogger


class HydrologyCategory(BaseCategory):
    """
    Hydrology Tools category.

    Contains tools for DEM conditioning, catchment analysis,
    flow routing and rainfall processing.
    """

    @property
    def id(self) -> str:
        """Return the unique category identifier."""
        return "hydrology"

    @property
    def name(self) -> str:
        """Return the human-readable category name."""
        return "Hydrology Tools"

    @property
    def description(self) -> str:
        """Return a short description of the category."""
        return (
            "Tools for hydrological analysis including DEM conditioning, "
            "catchment delineation, flow accumulation, rainfall analysis, "
            "and water balance calculations."
        )

    @property
    def icon_path(self) -> str:
        """Return the path to the category icon."""
        from mayim_tools.resources_rc import get_icon_path

        return get_icon_path("hydrology.png")

    def get_algorithms(self) -> list:
        """
        Return all Hydrology Tools processing algorithms.

        Each tool is imported inside its own try/except block so that
        an import error in one tool is logged clearly without preventing
        the remaining tools from loading.

        Returns
        -------
        list
            List of instantiated QgsProcessingAlgorithm objects.
        """
        algorithms: list = []

        # ── DEM Hydrological Screening ────────────────────────────────
        try:
            from mayim_tools.categories.hydrology.tools.dem_hydrological_screening import (
                DEMHydrologicalScreening,
            )

            algorithms.append(DEMHydrologicalScreening())
            MayimLogger.info("Registered: DEM Hydrological Screening")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register DEM Hydrological Screening: {e}")

        # ── DEM Hydrological Smoothing ────────────────────────────────
        try:
            from mayim_tools.categories.hydrology.tools.dem_hydrological_smoothing import (
                DEMHydrologicalSmoothing,
            )

            algorithms.append(DEMHydrologicalSmoothing())
            MayimLogger.info("Registered: DEM Hydrological Smoothing")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register DEM Hydrological Smoothing: {e}")

        # ── DEM Depression Analysis ───────────────────────────────────
        try:
            from mayim_tools.categories.hydrology.tools.dem_depression_analysis import (
                DEMDepressionAnalysis,
            )

            algorithms.append(DEMDepressionAnalysis())
            MayimLogger.info("Registered: DEM Depression Analysis")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register DEM Depression Analysis: {e}")

        # ── DEM Hydrological Filling ──────────────────────────────────
        try:
            from mayim_tools.categories.hydrology.tools.dem_hydrological_filling import (
                DEMHydrologicalFilling,
            )

            algorithms.append(DEMHydrologicalFilling())
            MayimLogger.info("Registered: DEM Hydrological Filling")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register DEM Hydrological Filling: {e}")

        # ── DEM Gradient Resolution ───────────────────────────────────
        try:
            from mayim_tools.categories.hydrology.tools.dem_gradient_resolution import (
                DEMGradientResolution,
            )

            algorithms.append(DEMGradientResolution())
            MayimLogger.info("Registered: DEM Gradient Resolution")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register DEM Gradient Resolution: {e}")

        # ── DEM Hydrography Enforcement ───────────────────────────────
        try:
            from mayim_tools.categories.hydrology.tools.dem_hydrography_enforcement import (
                DEMHydrographyEnforcement,
            )

            algorithms.append(DEMHydrographyEnforcement())
            MayimLogger.info("Registered: DEM Hydrography Enforcement")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register DEM Hydrography Enforcement: {e}")

        # ── DEM Conditioning Workflow ─────────────────────────────────
        try:
            from mayim_tools.categories.hydrology.tools.dem_conditioning_workflow import (
                DEMConditioningWorkflow,
            )

            algorithms.append(DEMConditioningWorkflow())
            MayimLogger.info("Registered: DEM Conditioning Workflow")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register DEM Conditioning Workflow: {e}")

        # ── D8 Flow Direction ─────────────────────────────────────────
        try:
            from mayim_tools.categories.hydrology.tools.dem_d8_flow_direction import (
                D8FlowDirection,
            )

            algorithms.append(D8FlowDirection())
            MayimLogger.info("Registered: D8 Flow Direction")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register D8 Flow Direction: {e}")

        # ── D8 Flow Accumulation ──────────────────────────────────────
        try:
            from mayim_tools.categories.hydrology.tools.dem_d8_flow_accumulation import (
                D8FlowAccumulation,
            )

            algorithms.append(D8FlowAccumulation())
            MayimLogger.info("Registered: D8 Flow Accumulation")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register D8 Flow Accumulation: {e}")

        # ── Add future tools below this line ──────────────────────────

        return algorithms
