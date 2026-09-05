# dem_conditioning_workflow.py
#
# DEM Conditioning Workflow — Orchestration Adapter
# Mayim Tools | Hydrology Category
#
# Combines all six DEM conditioning tools into a single sequential
# pipeline. Stage 6 (hydrography enforcement) is optional.
#
# Pipeline:
#   Stage 1 — DEM Hydrological Screening      (mandatory)
#   Stage 2 — DEM Hydrological Smoothing      (mandatory)
#   Stage 3 — DEM Depression Analysis         (mandatory)
#   Stage 4 — DEM Hydrological Filling        (mandatory)
#   Stage 5 — DEM Gradient Resolution         (mandatory)
#   Stage 6 — DEM Hydrography Enforcement     (optional)
#
# IP STATUS: CLEAR WITH CLEAN-ROOM RECORD KEEPING
# Uses QGIS Processing framework as infrastructure only.
# Does not reimplement any algorithm logic.
#
# Author  : Mayim Tools Development Team
# Created : 2025
# License : Proprietary — Zutari / Mayim

"""
DEM Conditioning Workflow.

Orchestrates the full six-stage DEM conditioning pipeline as a single
QGIS Processing tool. Each stage is executed in sequence. Outputs from
each stage are passed forward as inputs to the next stage automatically.

Stage 6 (DEM Hydrography Enforcement) is optional and requires
additional hydrography and flow-evidence inputs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import processing
from qgis.core import (
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFile,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
)

from mayim_tools.core.logger import MayimLogger
from mayim_tools.processing.algorithms.base_algorithm import (
    MayimBaseAlgorithm,
)


class DEMConditioningWorkflow(MayimBaseAlgorithm):
    """
    DEM Conditioning Workflow.

    Runs the full six-stage DEM conditioning pipeline sequentially.
    Stage 6 (hydrography enforcement) is optional.
    """

    # ── Parameter identifiers ─────────────────────────────────────────────────

    # Stage 1 — Screening
    INPUT_DEM = "INPUT_DEM"
    DEM_SOURCE_TYPE = "DEM_SOURCE_TYPE"
    USER_RMSE = "USER_RMSE"
    SMALL_VOID_THRESHOLD = "SMALL_VOID_THRESHOLD"
    LARGE_VOID_THRESHOLD = "LARGE_VOID_THRESHOLD"
    MAD_WINDOW_SIZE = "MAD_WINDOW_SIZE"
    MAD_THRESHOLD = "MAD_THRESHOLD"

    # Stage 2 — Smoothing
    ITERATIONS = "ITERATIONS"
    DIFFUSION_STRENGTH = "DIFFUSION_STRENGTH"
    EDGE_THRESHOLD = "EDGE_THRESHOLD"
    RESOLUTION_SCALE = "RESOLUTION_SCALE"

    # Stage 3 — Depression Analysis
    CLASSIFICATION_THRESHOLD = "CLASSIFICATION_THRESHOLD"
    REVIEW_MARGIN = "REVIEW_MARGIN"

    # Stage 4 — Filling
    MAX_BREACH_LENGTH = "MAX_BREACH_LENGTH"
    MAX_BREACH_DEPTH = "MAX_BREACH_DEPTH"
    CONNECTIVITY = "CONNECTIVITY"

    # Stage 5 — Gradient Resolution
    CELL_SIZE = "CELL_SIZE"

    # Stage 6 — Hydrography Enforcement (optional)
    RUN_ENFORCEMENT = "RUN_ENFORCEMENT"
    INPUT_HYDROGRAPHY = "INPUT_HYDROGRAPHY"
    INPUT_FLOW_EVIDENCE = "INPUT_FLOW_EVIDENCE"
    ENDPOINT_TOLERANCE = "ENDPOINT_TOLERANCE"
    POSITIONAL_TOLERANCE_CELLS = "POSITIONAL_TOLERANCE_CELLS"
    MAXIMUM_BURN_DEPTH = "MAXIMUM_BURN_DEPTH"

    # Shared
    VERTICAL_ACCURACY = "VERTICAL_ACCURACY"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    # ── Output identifiers ────────────────────────────────────────────────────

    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_PROVENANCE = "OUTPUT_PROVENANCE"
    OUTPUT_DEM = "OUTPUT_DEM"

    # ── Defaults ──────────────────────────────────────────────────────────────

    DEM_SOURCE_TYPES: ClassVar[list[str]] = [
        "Unknown",
        "LiDAR DTM",
        "LiDAR DSM",
        "SRTM",
        "Copernicus GLO-30",
        "FABDEM",
        "Aerial Photogrammetry",
        "Other",
    ]

    # ── Algorithm identity ────────────────────────────────────────────────────

    def name(self) -> str:
        return "demconditioningworkflow"

    def createInstance(self):
        return DEMConditioningWorkflow()

    def displayName(self) -> str:
        return "DEM Conditioning Workflow"

    def group(self) -> str:
        return "Hydrology Tools"

    def groupId(self) -> str:
        return "hydrologytools"

    def shortHelpString(self) -> str:
        return (
            "DEM Conditioning Workflow — full six-stage pipeline.\n\n"
            "Runs the following stages sequentially:\n"
            "  Stage 1 — DEM Hydrological Screening      (mandatory)\n"
            "  Stage 2 — DEM Hydrological Smoothing      (mandatory)\n"
            "  Stage 3 — DEM Depression Analysis         (mandatory)\n"
            "  Stage 4 — DEM Hydrological Filling        (mandatory)\n"
            "  Stage 5 — DEM Gradient Resolution         (mandatory)\n"
            "  Stage 6 — DEM Hydrography Enforcement     (optional)\n\n"
            "Outputs from each stage are passed automatically to the next. "
            "All intermediate and final outputs are written to the selected "
            "output folder. A unified workflow provenance record is produced.\n\n"
            "Stage 6 requires a vector hydrography layer and a DEM-derived "
            "flow-evidence raster. Enable it only when these inputs are "
            "available.\n\n"
            "─── SHARED PARAMETERS ───────────────────────────────────────\n"
            "Vertical accuracy (elevation units)\n"
            "  The DEM vertical accuracy (RMSE or LE90).\n"
            "  Suggested: 0.1–0.3 m for LiDAR | 3–5 m for SRTM/GLO-30\n"
            "  Default: 0.5\n\n"
            "─── STAGE 1 — DEM HYDROLOGICAL SCREENING ────────────────────\n"
            "DEM source type\n"
            "  Select the DEM data source. Used to estimate vertical\n"
            "  accuracy when no formal value is available.\n\n"
            "User RMSE override\n"
            "  Supply a measured RMSE to override the source estimate.\n"
            "  Set to 0 to use the source-type estimate.\n"
            "  Range: 0.0 and above | Default: 0 (use estimate)\n\n"
            "Small void threshold (cells)\n"
            "  Voids smaller than this are interpolated automatically.\n"
            "  Range: 1–10 000 | Suggested: 100\n\n"
            "Large void threshold (cells)\n"
            "  Voids larger than this are reported to the analyst\n"
            "  and not filled.\n"
            "  Range: > small void threshold | Suggested: 10 000\n\n"
            "MAD window size (cells)\n"
            "  Local window for median absolute deviation artifact\n"
            "  screening. Must be odd.\n"
            "  Range: 3–21 | Suggested: 7\n\n"
            "MAD threshold (multiples of MAD)\n"
            "  Cells exceeding this multiple of the local MAD are\n"
            "  flagged as artifacts.\n"
            "  Range: 1.0–10.0 | Suggested: 3.0\n\n"
            "─── STAGE 2 — DEM HYDROLOGICAL SMOOTHING ────────────────────\n"
            "Diffusion iterations\n"
            "  Number of Perona-Malik anisotropic diffusion passes.\n"
            "  More iterations increase cumulative smoothing.\n"
            "  Range: 1–100 | Suggested: 3–5 (light) | 5–10 (persistent noise)\n\n"
            "Diffusion strength\n"
            "  Per-iteration update strength. Higher is not always better.\n"
            "  Range: 0.01–0.25 | Suggested: 0.20\n\n"
            "Edge threshold (elevation units)\n"
            "  Gradient magnitude below which smoothing is applied.\n"
            "  Lower values preserve terrain edges more strongly.\n"
            "  Range: > 0 | Suggested: 1.0 x vertical accuracy\n\n"
            "Enable resolution scaling\n"
            "  Scales smoothing parameters to the DEM resolution.\n"
            "  Recommended: enabled\n\n"
            "─── STAGE 3 — DEM DEPRESSION ANALYSIS ───────────────────────\n"
            "Classification threshold (elevation units)\n"
            "  Depressions shallower than this are classified as\n"
            "  likely artifacts.\n"
            "  Range: 0.0 and above | Suggested: 1 x vertical accuracy\n\n"
            "Review margin (elevation units)\n"
            "  Buffer around the classification threshold within which\n"
            "  depressions are flagged for analyst review.\n"
            "  Range: 0.0 and above | Suggested: 0.1 x vertical accuracy\n\n"
            "─── STAGE 4 — DEM HYDROLOGICAL FILLING ──────────────────────\n"
            "Maximum breach length (cells)\n"
            "  Maximum path length for least-cost breaching before\n"
            "  falling back to filling.\n"
            "  Range: 1–500 | Suggested: 50\n\n"
            "Maximum breach depth (elevation units)\n"
            "  Maximum permitted lowering for a breach path.\n"
            "  Range: 0.0 and above | Suggested: 0.5\n\n"
            "─── STAGE 4 / 5 — SHARED ────────────────────────────────────\n"
            "Connectivity (4 or 8)\n"
            "  Raster neighbourhood connectivity for flow routing.\n"
            "  8-connected is recommended for most terrain types.\n"
            "  Options: 4 | 8 | Suggested: 8\n\n"
            "─── STAGE 5 — DEM GRADIENT RESOLUTION ───────────────────────\n"
            "Cell size override\n"
            "  Override the DEM cell size used in gradient calculations.\n"
            "  Set to 0 to read directly from the DEM.\n"
            "  Range: 0.0 and above | Default: 0 (read from DEM)\n\n"
            "─── STAGE 6 — DEM HYDROGRAPHY ENFORCEMENT (OPTIONAL) ────────\n"
            "Run hydrography enforcement\n"
            "  Enable Stage 6. Requires hydrography and flow-evidence\n"
            "  inputs. Leave disabled if these are not yet available.\n\n"
            "Hydrography endpoint tolerance (map units)\n"
            "  Snapping tolerance for network node matching.\n"
            "  Range: 0.0 and above | Suggested: 1 x cell size\n\n"
            "Positional tolerance (cells)\n"
            "  Maximum cell offset between mapped and DEM-derived\n"
            "  flow paths before divergence is declared.\n"
            "  Range: 0–20 | Suggested: 3\n\n"
            "Maximum burn depth (elevation units)\n"
            "  Absolute maximum permitted lowering per cell.\n"
            "  Range: > 0 | Suggested: 2–5 x vertical accuracy"
        )

    # ── Parameter definition ──────────────────────────────────────────────────

    def initAlgorithm(self, config=None) -> None:

        # ── Shared inputs ─────────────────────────────────────────────────────

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                "Input DEM",
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.VERTICAL_ACCURACY,
                "DEM vertical accuracy (elevation units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.001,
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER,
                "Output folder",
            )
        )

        # ── Stage 1 — Screening ───────────────────────────────────────────────

        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEM_SOURCE_TYPE,
                "Stage 1 — DEM source type",
                options=self.DEM_SOURCE_TYPES,
                defaultValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.USER_RMSE,
                "Stage 1 — User RMSE override (0 = use source estimate)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.SMALL_VOID_THRESHOLD,
                "Stage 1 — Small void threshold (cells)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=100,
                minValue=1,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.LARGE_VOID_THRESHOLD,
                "Stage 1 — Large void threshold (cells)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10000,
                minValue=1,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAD_WINDOW_SIZE,
                "Stage 1 — MAD window size (cells)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=7,
                minValue=3,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAD_THRESHOLD,
                "Stage 1 — MAD threshold (multiples of MAD)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=3.0,
                minValue=0.1,
            )
        )

        # ── Stage 2 — Smoothing ───────────────────────────────────────────────

        self.addParameter(
            QgsProcessingParameterNumber(
                self.ITERATIONS,
                "Stage 2 — Diffusion iterations",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1,
                maxValue=100,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.DIFFUSION_STRENGTH,
                "Stage 2 — Diffusion strength",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.20,
                minValue=0.01,
                maxValue=0.25,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.EDGE_THRESHOLD,
                "Stage 2 — Edge threshold (elevation units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.001,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RESOLUTION_SCALE,
                "Stage 2 — Enable resolution scaling",
                defaultValue=True,
            )
        )

        # ── Stage 3 — Depression Analysis ────────────────────────────────────

        self.addParameter(
            QgsProcessingParameterNumber(
                self.CLASSIFICATION_THRESHOLD,
                "Stage 3 — Classification threshold (elevation units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.1,
                minValue=0.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.REVIEW_MARGIN,
                "Stage 3 — Review margin (elevation units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.05,
                minValue=0.0,
            )
        )

        # ── Stage 4 — Filling ─────────────────────────────────────────────────

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_BREACH_LENGTH,
                "Stage 4 — Maximum breach length (cells)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=50,
                minValue=1,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_BREACH_DEPTH,
                "Stage 4 — Maximum breach depth (elevation units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONNECTIVITY,
                "Stage 4/5 — Connectivity (4 or 8)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                minValue=4,
                maxValue=8,
            )
        )

        # ── Stage 5 — Gradient Resolution ────────────────────────────────────

        self.addParameter(
            QgsProcessingParameterNumber(
                self.CELL_SIZE,
                "Stage 5 — Cell size override (0 = read from DEM)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                optional=True,
            )
        )

        # ── Stage 6 — Hydrography Enforcement (optional) ─────────────────────

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RUN_ENFORCEMENT,
                "Stage 6 — Run hydrography enforcement (optional)",
                defaultValue=False,
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_HYDROGRAPHY,
                "Stage 6 — Input vector hydrography",
                types=[QgsProcessing.TypeVectorLine],
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FLOW_EVIDENCE,
                "Stage 6 — DEM-derived flow evidence raster",
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.ENDPOINT_TOLERANCE,
                "Stage 6 — Hydrography endpoint tolerance (map units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.0,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.POSITIONAL_TOLERANCE_CELLS,
                "Stage 6 — Positional tolerance (cells)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=0,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAXIMUM_BURN_DEPTH,
                "Stage 6 — Maximum burn depth (elevation units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=2.0,
                minValue=0.001,
                optional=True,
            )
        )

        # ── Outputs ───────────────────────────────────────────────────────────

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_DEM,
                "Final conditioned DEM",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_REPORT,
                "Workflow report",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_PROVENANCE,
                "Workflow provenance",
            )
        )

    # ── Main processing ───────────────────────────────────────────────────────

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        logger = MayimLogger()
        logger.info("DEM Conditioning Workflow started.")

        run_timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        warnings: list[str] = []
        stage_provenance: dict = {}

        # ── Resolve shared inputs ─────────────────────────────────────────────

        dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        if dem_layer is None or not dem_layer.isValid():
            raise QgsProcessingException(
                "Input DEM is not valid or could not be loaded."
            )
        dem_source = dem_layer.source()

        vertical_accuracy = self.parameterAsDouble(
            parameters, self.VERTICAL_ACCURACY, context
        )

        output_folder = Path(
            self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        )
        output_folder.mkdir(parents=True, exist_ok=True)

        dem_stem = Path(dem_source).stem

        feedback.pushInfo("DEM Conditioning Workflow started.")
        feedback.pushInfo(f"Input DEM       : {dem_source}")
        feedback.pushInfo(f"Vertical accuracy: {vertical_accuracy}")
        feedback.pushInfo(f"Output folder   : {output_folder}")

        # ── Stage 1 — DEM Hydrological Screening ─────────────────────────────

        feedback.setProgress(2)
        feedback.pushInfo("── Stage 1: DEM Hydrological Screening ──────────────────")

        stage1_params = {
            "INPUT_DEM": dem_source,
            "DEM_SOURCE_TYPE": self.parameterAsInt(
                parameters, self.DEM_SOURCE_TYPE, context
            ),
            "USER_RMSE": self.parameterAsDouble(parameters, self.USER_RMSE, context),
            "SMALL_VOID_THRESHOLD": self.parameterAsInt(
                parameters, self.SMALL_VOID_THRESHOLD, context
            ),
            "LARGE_VOID_THRESHOLD": self.parameterAsInt(
                parameters, self.LARGE_VOID_THRESHOLD, context
            ),
            "MAD_WINDOW_SIZE": self.parameterAsInt(
                parameters, self.MAD_WINDOW_SIZE, context
            ),
            "MAD_THRESHOLD": self.parameterAsDouble(
                parameters, self.MAD_THRESHOLD, context
            ),
            "OUTPUT_FOLDER": str(output_folder / "stage1_screening"),
            "LOAD_SCREENED_DEM": False,
            "LOAD_VOID_MASK": False,
            "LOAD_ARTIFACT_MASK": False,
        }

        try:
            stage1_result = processing.run(
                "mayimtools:demhydrologicalscreening",
                stage1_params,
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
        except Exception as exc:
            raise QgsProcessingException(
                f"Stage 1 (DEM Hydrological Screening) failed: {exc}"
            ) from exc

        screened_dem_path = stage1_result["OUTPUT_DEM"]
        stage1_manifest = None

        stage_provenance["stage_1_screening"] = {
            "status": "complete",
            "output_dem": screened_dem_path,
            "output_folder": str(output_folder / "stage1_screening"),
        }

        feedback.pushInfo(f"Stage 1 complete. Screened DEM: {screened_dem_path}")

        # ── Stage 2 — DEM Hydrological Smoothing ─────────────────────────────

        feedback.setProgress(18)
        feedback.pushInfo("── Stage 2: DEM Hydrological Smoothing ──────────────────")

        stage2_params = {
            "INPUT_DEM": screened_dem_path,
            "INPUT_MANIFEST": stage1_manifest,
            "ITERATIONS": self.parameterAsInt(parameters, self.ITERATIONS, context),
            "DIFFUSION_STRENGTH": self.parameterAsDouble(
                parameters, self.DIFFUSION_STRENGTH, context
            ),
            "EDGE_THRESHOLD": self.parameterAsDouble(
                parameters, self.EDGE_THRESHOLD, context
            ),
            "RESOLUTION_SCALE": self.parameterAsBool(
                parameters, self.RESOLUTION_SCALE, context
            ),
            "OUTPUT_FOLDER": str(output_folder / "stage2_smoothing"),
            "LOAD_SMOOTHED_DEM": False,
            "LOAD_DIFFERENCE": False,
            "LOAD_SMOOTHING_MASK": False,
        }

        try:
            stage2_result = processing.run(
                "mayimtools:demhydrologicalsmoothing",
                stage2_params,
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
        except Exception as exc:
            raise QgsProcessingException(
                f"Stage 2 (DEM Hydrological Smoothing) failed: {exc}"
            ) from exc

        smoothed_dem_path = stage2_result["OUTPUT_SMOOTHED"]

        stage_provenance["stage_2_smoothing"] = {
            "status": "complete",
            "output_dem": smoothed_dem_path,
            "output_folder": str(output_folder / "stage2_smoothing"),
        }

        feedback.pushInfo(f"Stage 2 complete. Smoothed DEM: {smoothed_dem_path}")

        # ── Stage 3 — DEM Depression Analysis ────────────────────────────────

        feedback.setProgress(34)
        feedback.pushInfo("── Stage 3: DEM Depression Analysis ─────────────────────")

        stage3_params = {
            "INPUT_DEM": smoothed_dem_path,
            "INPUT_MANIFEST": stage1_manifest,
            "VERTICAL_ACCURACY": vertical_accuracy,
            "CLASSIFICATION_THRESHOLD": self.parameterAsDouble(
                parameters, self.CLASSIFICATION_THRESHOLD, context
            ),
            "REVIEW_MARGIN": self.parameterAsDouble(
                parameters, self.REVIEW_MARGIN, context
            ),
            "OUTPUT_FOLDER": str(output_folder / "stage3_depression"),
            "LOAD_DEPRESSION_IDS": False,
            "LOAD_CLASSIFICATION": False,
        }

        try:
            stage3_result = processing.run(
                "mayimtools:demdepressionanalysis",
                stage3_params,
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
        except Exception as exc:
            raise QgsProcessingException(
                f"Stage 3 (DEM Depression Analysis) failed: {exc}"
            ) from exc

        depression_manifest_path = stage3_result["OUTPUT_MANIFEST"]
        depression_inventory_path = stage3_result["OUTPUT_INVENTORY"]

        stage_provenance["stage_3_depression"] = {
            "status": "complete",
            "output_manifest": depression_manifest_path,
            "output_inventory": depression_inventory_path,
            "output_folder": str(output_folder / "stage3_depression"),
        }

        feedback.pushInfo(f"Stage 3 complete. Manifest: {depression_manifest_path}")

        # ── Stage 4 — DEM Hydrological Filling ───────────────────────────────

        feedback.setProgress(50)
        feedback.pushInfo("── Stage 4: DEM Hydrological Filling ────────────────────")

        stage4_params = {
            "INPUT_DEM": smoothed_dem_path,
            "INPUT_MANIFEST": depression_manifest_path,
            "INPUT_INVENTORY": depression_inventory_path,
            "MAX_BREACH_LENGTH": self.parameterAsInt(
                parameters, self.MAX_BREACH_LENGTH, context
            ),
            "MAX_BREACH_DEPTH": self.parameterAsDouble(
                parameters, self.MAX_BREACH_DEPTH, context
            ),
            "CONNECTIVITY": self.parameterAsInt(parameters, self.CONNECTIVITY, context),
            "OUTPUT_FOLDER": str(output_folder / "stage4_filling"),
            "LOAD_PRESERVED_DEM": False,
            "LOAD_READY_DEM": False,
            "LOAD_DECISION_RASTER": False,
        }

        try:
            stage4_result = processing.run(
                "mayimtools:demhydrologicalfilling",
                stage4_params,
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
        except Exception as exc:
            raise QgsProcessingException(
                f"Stage 4 (DEM Hydrological Filling) failed: {exc}"
            ) from exc

        filled_dem_path = stage4_result["OUTPUT_READY"]
        filling_manifest_path = stage4_result["OUTPUT_MANIFEST"]

        stage_provenance["stage_4_filling"] = {
            "status": "complete",
            "output_dem": filled_dem_path,
            "output_manifest": filling_manifest_path,
            "output_folder": str(output_folder / "stage4_filling"),
        }

        feedback.pushInfo(f"Stage 4 complete. Filled DEM: {filled_dem_path}")

        # ── Stage 5 — DEM Gradient Resolution ────────────────────────────────

        feedback.setProgress(66)
        feedback.pushInfo("── Stage 5: DEM Gradient Resolution ─────────────────────")

        # Derive cell size from the filled DEM if the user has not
        # supplied an override. The Gradient Resolution tool requires
        # a positive cell size and does not accept zero.

        user_cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)

        if user_cell_size > 0.0:
            resolved_cell_size = user_cell_size
        else:
            import rasterio

            with rasterio.open(filled_dem_path) as _ds:
                resolved_cell_size = float((_ds.res[0] + _ds.res[1]) / 2.0)

            feedback.pushInfo(
                f"Cell size read from DEM: {resolved_cell_size:.4f} " "map units."
            )

        stage5_params = {
            "INPUT_DEM": filled_dem_path,
            "INPUT_MANIFEST": filling_manifest_path,
            "VERTICAL_ACCURACY": vertical_accuracy,
            "CELL_SIZE": resolved_cell_size,
            "CONNECTIVITY": self.parameterAsInt(parameters, self.CONNECTIVITY, context),
            "OUTPUT_FOLDER": str(output_folder / "stage5_gradient"),
            "LOAD_RESOLVED_DEM": False,
            "LOAD_FLAT_MASK": False,
            "LOAD_DIFFERENCE": False,
            "LOAD_REGION_IDS": False,
        }

        try:
            stage5_result = processing.run(
                "mayimtools:demgradientresolution",
                stage5_params,
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )
        except Exception as exc:
            raise QgsProcessingException(
                f"Stage 5 (DEM Gradient Resolution) failed: {exc}"
            ) from exc

        resolved_dem_path = stage5_result["OUTPUT_RESOLVED_DEM"]

        stage_provenance["stage_5_gradient"] = {
            "status": "complete",
            "output_dem": resolved_dem_path,
            "output_folder": str(output_folder / "stage5_gradient"),
        }

        feedback.pushInfo(f"Stage 5 complete. Resolved DEM: {resolved_dem_path}")

        # ── Stage 6 — DEM Hydrography Enforcement (optional) ─────────────────

        run_enforcement = self.parameterAsBool(
            parameters, self.RUN_ENFORCEMENT, context
        )

        final_dem_path = resolved_dem_path

        if run_enforcement:
            feedback.setProgress(78)
            feedback.pushInfo(
                "── Stage 6: DEM Hydrography Enforcement ─────────────────"
            )

            hydrography_layer = self.parameterAsVectorLayer(
                parameters, self.INPUT_HYDROGRAPHY, context
            )
            flow_evidence_layer = self.parameterAsRasterLayer(
                parameters, self.INPUT_FLOW_EVIDENCE, context
            )

            if hydrography_layer is None or not hydrography_layer.isValid():
                raise QgsProcessingException(
                    "Stage 6 is enabled but no valid hydrography layer "
                    "was provided. Supply a line hydrography layer or "
                    "disable Stage 6."
                )

            if flow_evidence_layer is None or not flow_evidence_layer.isValid():
                raise QgsProcessingException(
                    "Stage 6 is enabled but no valid flow evidence raster "
                    "was provided. Supply a flow evidence raster or "
                    "disable Stage 6."
                )

            stage6_params = {
                "INPUT_DEM": resolved_dem_path,
                "INPUT_HYDROGRAPHY": hydrography_layer,
                "INPUT_FLOW_EVIDENCE": flow_evidence_layer.source(),
                "ENDPOINT_TOLERANCE": self.parameterAsDouble(
                    parameters, self.ENDPOINT_TOLERANCE, context
                ),
                "POSITIONAL_TOLERANCE_CELLS": self.parameterAsInt(
                    parameters, self.POSITIONAL_TOLERANCE_CELLS, context
                ),
                "VERTICAL_ACCURACY": vertical_accuracy,
                "MAXIMUM_BURN_DEPTH": self.parameterAsDouble(
                    parameters, self.MAXIMUM_BURN_DEPTH, context
                ),
                "OUTPUT_FOLDER": str(output_folder / "stage6_enforcement"),
            }

            try:
                stage6_result = processing.run(
                    "mayimtools:demhydrographyenforcement",
                    stage6_params,
                    context=context,
                    feedback=feedback,
                    is_child_algorithm=True,
                )
            except Exception as exc:
                raise QgsProcessingException(
                    f"Stage 6 (DEM Hydrography Enforcement) failed: {exc}"
                ) from exc

            stage_provenance["stage_6_enforcement"] = {
                "status": "complete",
                "output_report": stage6_result["OUTPUT_REPORT"],
                "output_provenance": stage6_result["OUTPUT_PROVENANCE"],
                "output_folder": str(output_folder / "stage6_enforcement"),
            }

            feedback.pushInfo("Stage 6 complete.")

        else:
            stage_provenance["stage_6_enforcement"] = {
                "status": "skipped",
                "reason": "RUN_ENFORCEMENT parameter is False.",
            }
            feedback.pushInfo("Stage 6 skipped — hydrography enforcement not enabled.")

        # ── Copy final conditioned DEM to workflow output folder ──────────────

        import shutil

        final_dem_output_path = output_folder / f"{dem_stem}_conditioned.tif"

        shutil.copy2(final_dem_path, final_dem_output_path)

        feedback.pushInfo(f"Final conditioned DEM copied to: {final_dem_output_path}")

        # ── Build workflow report ─────────────────────────────────────────────

        feedback.setProgress(90)

        # ── Build workflow report ─────────────────────────────────────────────

        feedback.setProgress(90)

        report_path = output_folder / f"{dem_stem}_conditioning_workflow_report.txt"
        provenance_path = (
            output_folder / f"{dem_stem}_conditioning_workflow_provenance.json"
        )

        report_lines = [
            "═" * 72,
            "  MAYIM TOOLS — DEM Conditioning Workflow",
            "  Full Pipeline Report",
            "═" * 72,
            "",
            f"  Run timestamp (UTC) : {run_timestamp}",
            f"  Input DEM           : {dem_source}",
            f"  Vertical accuracy   : {vertical_accuracy}",
            f"  Output folder       : {output_folder}",
            "",
            "── Stage Summary ────────────────────────────────────────────────",
            (
                f"  Stage 1 — DEM Hydrological Screening  : "
                f"{stage_provenance['stage_1_screening']['status']}"
            ),
            (
                f"  Stage 2 — DEM Hydrological Smoothing  : "
                f"{stage_provenance['stage_2_smoothing']['status']}"
            ),
            (
                f"  Stage 3 — DEM Depression Analysis     : "
                f"{stage_provenance['stage_3_depression']['status']}"
            ),
            (
                f"  Stage 4 — DEM Hydrological Filling    : "
                f"{stage_provenance['stage_4_filling']['status']}"
            ),
            (
                f"  Stage 5 — DEM Gradient Resolution     : "
                f"{stage_provenance['stage_5_gradient']['status']}"
            ),
            (
                f"  Stage 6 — DEM Hydrography Enforcement : "
                f"{stage_provenance['stage_6_enforcement']['status']}"
            ),
            "",
            "── Stage 1: DEM Hydrological Screening ──────────────────────────",
            (
                f"  Output DEM    : "
                f"{stage_provenance['stage_1_screening']['output_dem']}"
            ),
            (
                f"  Output folder : "
                f"{stage_provenance['stage_1_screening']['output_folder']}"
            ),
            "",
            "── Stage 2: DEM Hydrological Smoothing ──────────────────────────",
            (
                f"  Output DEM    : "
                f"{stage_provenance['stage_2_smoothing']['output_dem']}"
            ),
            (
                f"  Output folder : "
                f"{stage_provenance['stage_2_smoothing']['output_folder']}"
            ),
            "",
            "── Stage 3: DEM Depression Analysis ─────────────────────────────",
            (
                f"  Output manifest   : "
                f"{stage_provenance['stage_3_depression']['output_manifest']}"
            ),
            (
                f"  Output inventory  : "
                f"{stage_provenance['stage_3_depression']['output_inventory']}"
            ),
            (
                f"  Output folder     : "
                f"{stage_provenance['stage_3_depression']['output_folder']}"
            ),
            "",
            "── Stage 4: DEM Hydrological Filling ────────────────────────────",
            (
                f"  Output DEM    : "
                f"{stage_provenance['stage_4_filling']['output_dem']}"
            ),
            (
                f"  Output folder : "
                f"{stage_provenance['stage_4_filling']['output_folder']}"
            ),
            "",
            "── Stage 5: DEM Gradient Resolution ─────────────────────────────",
            (
                f"  Output DEM    : "
                f"{stage_provenance['stage_5_gradient']['output_dem']}"
            ),
            (
                f"  Output folder : "
                f"{stage_provenance['stage_5_gradient']['output_folder']}"
            ),
            "",
            "── Stage 6: DEM Hydrography Enforcement ─────────────────────────",
        ]

        if stage_provenance["stage_6_enforcement"]["status"] == "complete":
            report_lines.extend(
                [
                    (
                        f"  Output report     : "
                        f"{stage_provenance['stage_6_enforcement']['output_report']}"
                    ),
                    (
                        f"  Output provenance : "
                        f"{stage_provenance['stage_6_enforcement']['output_provenance']}"
                    ),
                    (
                        f"  Output folder     : "
                        f"{stage_provenance['stage_6_enforcement']['output_folder']}"
                    ),
                ]
            )
        else:
            report_lines.append(
                f"  Status : " f"{stage_provenance['stage_6_enforcement']['reason']}"
            )

        report_lines.extend(
            [
                "",
                "── Final Output ─────────────────────────────────────────────────",
                f"  Final conditioned DEM : {final_dem_output_path}",
                "",
                "── Warnings ─────────────────────────────────────────────────────",
            ]
        )

        if warnings:
            for warning in warnings:
                report_lines.append(f"  WARNING: {warning}")
        else:
            report_lines.append("  No warnings.")

        report_lines.extend(
            [
                "",
                "═" * 72,
                "  End of report.",
                "═" * 72,
            ]
        )

        report_text = "\n".join(report_lines)
        report_path.write_text(report_text, encoding="utf-8")
        feedback.pushInfo(f"Workflow report written: {report_path}")

        # ── Build workflow provenance ──────────────────────────────────────────

        provenance = {
            "tool": "DEMConditioningWorkflow",
            "run_timestamp_utc": run_timestamp,
            "inputs": {
                "dem_source": dem_source,
                "vertical_accuracy": vertical_accuracy,
                "output_folder": str(output_folder),
            },
            "stages": stage_provenance,
            "final_dem": final_dem_path,
            "warnings": warnings,
            "outputs": {
                "final_dem": str(final_dem_output_path),
                "report": str(report_path),
                "provenance": str(provenance_path),
            },
        }

        provenance_path.write_text(
            json.dumps(provenance, indent=4),
            encoding="utf-8",
        )
        feedback.pushInfo(f"Workflow provenance written: {provenance_path}")

        feedback.setProgress(100)
        feedback.pushInfo(
            "DEM Conditioning Workflow complete. "
            f"Final conditioned DEM: {final_dem_path}"
        )

        # ── Load final conditioned DEM into QGIS project ──────────────────────

        from qgis.core import QgsProject, QgsRasterLayer

        final_layer = QgsRasterLayer(
            str(final_dem_output_path),
            f"{dem_stem}_conditioned",
        )

        if final_layer.isValid():
            QgsProject.instance().addMapLayer(final_layer)
            feedback.pushInfo(
                f"Final conditioned DEM loaded into project: " f"{dem_stem}_conditioned"
            )
        else:
            feedback.pushWarning(
                "Final conditioned DEM could not be loaded into the "
                "project automatically. Open it manually from: "
                f"{final_dem_output_path}"
            )

        return {
            self.OUTPUT_DEM: str(final_dem_output_path),
            self.OUTPUT_REPORT: str(report_path),
            self.OUTPUT_PROVENANCE: str(provenance_path),
        }
