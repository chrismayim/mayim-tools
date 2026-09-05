"""
Mayim Tools - DEM Depression Analysis
=====================================

Implements Stage 3 and Stage 4 of the Mayim Tools DEM
hydrological-conditioning methodology.

Stage 3:
    Depression delineation and hierarchy construction.

Stage 4:
    Depression classification.

This tool analyses the DEM but does not modify terrain elevations.
It identifies depressions, calculates their features, builds the
in-house hierarchy structure, classifies depressions as artifact or
real-feature candidates, and exports inspectable analysis outputs.

Outputs include:

    - Depression ID raster.
    - Depression classification raster.
    - Depression hierarchy JSON.
    - Depression inventory JSON.
    - Human-readable text report.
    - JSON provenance record.
    - Derived MayimManifest.

IP status
---------
Original Mayim implementation using only:

    - Python standard-library components,
    - NumPy,
    - rasterio,
    - QGIS Processing API.

No WhiteboxTools, RichDEM, TauDEM or other third-party hydrological
runtime implementation is used.

The underlying methodology follows the revised Mayim research paper
and published literature, not third-party source code.

Important
---------
This tool does not fill, breach or otherwise modify the DEM.
Depression enforcement belongs to a later tool.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFile,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)

from mayim_tools.contract import MayimManifest
from mayim_tools.core.logger import MayimLogger
from mayim_tools.core.validation_utils import ValidationUtils
from mayim_tools.hydrology.depression.classification import (
    ARTIFACT,
    REAL_FEATURE,
    REVIEW_REQUIRED,
    classify_depressions,
)
from mayim_tools.hydrology.depression.detection import (
    detect_depressions,
    identify_spill_points,
)
from mayim_tools.hydrology.depression.features import (
    calculate_depression_features,
)
from mayim_tools.hydrology.depression.hierarchy import (
    build_hierarchy,
)
from mayim_tools.processing.algorithms.base_algorithm import (
    MayimBaseAlgorithm,
)


class DEMDepressionAnalysis(MayimBaseAlgorithm):
    """
    QGIS Processing adapter for native Mayim Stage 3 and Stage 4
    depression analysis.

    This tool:

    - Detects depressions.
    - Identifies spill elevations.
    - Builds the in-house depression hierarchy.
    - Calculates depression features.
    - Classifies depressions using the Stage 4 artifact-likelihood
      model.
    - Writes raster, JSON and text outputs.
    - Does NOT modify DEM elevations.

    The tool is analysis-only. Terrain modification belongs to the
    later selective enforcement tool.
    """

    # ── Parameter identifiers ─────────────────────────────────────── #
    PARAM_DEM = "INPUT_DEM"
    PARAM_MANIFEST = "INPUT_MANIFEST"
    PARAM_VERTICAL_ACCURACY = "VERTICAL_ACCURACY"
    PARAM_CLASSIFICATION_THRESHOLD = "CLASSIFICATION_THRESHOLD"
    PARAM_REVIEW_MARGIN = "REVIEW_MARGIN"
    PARAM_OUTPUT_FOLDER = "OUTPUT_FOLDER"
    PARAM_LOAD_DEPRESSION_IDS = "LOAD_DEPRESSION_IDS"
    PARAM_LOAD_CLASSIFICATION = "LOAD_CLASSIFICATION"

    # ── Output identifiers ────────────────────────────────────────── #
    OUTPUT_DEPRESSION_IDS = "OUTPUT_DEPRESSION_IDS"
    OUTPUT_CLASSIFICATION = "OUTPUT_CLASSIFICATION"
    OUTPUT_INVENTORY = "OUTPUT_INVENTORY"
    OUTPUT_HIERARCHY = "OUTPUT_HIERARCHY"
    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_PROVENANCE = "OUTPUT_PROVENANCE"
    OUTPUT_MANIFEST = "OUTPUT_MANIFEST"

    TOOL_VERSION = "dem-depression-analysis-0.2.0"

    def name(self) -> str:
        """Return the unique Processing algorithm identifier."""
        return "demdepressionanalysis"

    def displayName(self) -> str:
        """Return the human-readable algorithm name."""
        return "DEM Depression Analysis"

    def group(self) -> str:
        """Return the Processing Toolbox group name."""
        return "Hydrology Tools"

    def groupId(self) -> str:
        """Return the unique Processing Toolbox group identifier."""
        return "hydrology"

    def createInstance(self) -> DEMDepressionAnalysis:
        """Return a new instance of this algorithm."""
        return DEMDepressionAnalysis()

    def shortHelpString(self) -> str:
        """Return the Processing Toolbox help text."""
        return (
            "<b>DEM Depression Analysis</b><br><br>"
            "Implements Stage 3 and Stage 4 of the Mayim Tools DEM "
            "hydrological-conditioning methodology.<br><br>"
            "<b>Stage 3:</b> delineates depressions and constructs the "
            "native in-house depression hierarchy.<br>"
            "<b>Stage 4:</b> classifies each depression as "
            "ARTIFACT, REAL_FEATURE or REVIEW_REQUIRED using an "
            "inspectable multi-criteria artifact-likelihood score.<br><br>"
            "<b>This tool does not modify terrain elevations.</b> "
            "It is analysis-only and produces diagnostic rasters, "
            "JSON outputs, a text report, provenance, and a derived "
            "MayimManifest.<br><br>"
            "<b>IP:</b> Native Mayim implementation using only Python, "
            "NumPy, rasterio and the Mayim contract. No WhiteboxTools "
            "or RichDEM runtime dependency."
        )

    def helpUrl(self) -> str:
        """Return the project documentation URL."""
        return "https://github.com/chrismayim/mayim-tools"

    def tags(self) -> list[str]:
        """Return searchable Processing Toolbox tags."""
        return [
            "mayim",
            "dem",
            "hydrology",
            "depression",
            "classification",
            "analysis",
            "priority-flood",
            "hierarchy",
            "stage-3",
            "stage-4",
        ]

    def initAlgorithm(self, config=None) -> None:
        """Define Processing Toolbox parameters and outputs."""
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.PARAM_DEM,
                "Input DEM",
            )
        )

        self.addParameter(
            QgsProcessingParameterFile(
                self.PARAM_MANIFEST,
                "Input MayimManifest from previous tool (optional)",
                behavior=QgsProcessingParameterFile.File,
                extension="json",
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_VERTICAL_ACCURACY,
                "Vertical accuracy override in metres "
                "(recommended: use manifest; set -1 to use manifest/default)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=-1.0,
                minValue=-1.0,
                maxValue=100.0,
                optional=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_CLASSIFICATION_THRESHOLD,
                "Classification threshold "
                "(recommended: 0.60; valid range: 0.10-0.90)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.60,
                minValue=0.10,
                maxValue=0.90,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_REVIEW_MARGIN,
                "Review margin around threshold "
                "(recommended: 0.15; valid range: 0.05-0.40)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.15,
                minValue=0.05,
                maxValue=0.40,
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.PARAM_OUTPUT_FOLDER,
                "Output folder",
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_LOAD_DEPRESSION_IDS,
                "Load depression ID raster into project",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_LOAD_CLASSIFICATION,
                "Load classification raster into project",
                defaultValue=True,
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_DEPRESSION_IDS,
                "Depression ID raster",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_CLASSIFICATION,
                "Depression classification raster",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_INVENTORY,
                "Depression inventory",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_HIERARCHY,
                "Depression hierarchy",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_REPORT,
                "Depression analysis report",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_PROVENANCE,
                "Depression analysis provenance",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_MANIFEST,
                "Depression analysis MayimManifest",
            )
        )

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """
        Execute Stage 3 and Stage 4 depression analysis.

        Does not modify DEM elevations.

        :param parameters: Processing parameters.
        :param context: QGIS processing context.
        :param feedback: Processing feedback object.
        :returns: Dictionary of output paths.
        """
        try:
            import rasterio
        except ImportError as error:
            raise QgsProcessingException(
                "The rasterio library is required for this tool. "
                "Install it with: pip install rasterio"
            ) from error

        # ── Read parameters ───────────────────────────────────────── #

        dem_layer = self.parameterAsRasterLayer(
            parameters,
            self.PARAM_DEM,
            context,
        )

        if not ValidationUtils.is_valid_raster_layer(dem_layer):
            raise QgsProcessingException("The input DEM is missing or invalid.")

        manifest_path = self.parameterAsString(
            parameters,
            self.PARAM_MANIFEST,
            context,
        )

        vertical_accuracy_override = self.parameterAsDouble(
            parameters,
            self.PARAM_VERTICAL_ACCURACY,
            context,
        )

        classification_threshold = self.parameterAsDouble(
            parameters,
            self.PARAM_CLASSIFICATION_THRESHOLD,
            context,
        )

        review_margin = self.parameterAsDouble(
            parameters,
            self.PARAM_REVIEW_MARGIN,
            context,
        )

        output_folder = self.parameterAsString(
            parameters,
            self.PARAM_OUTPUT_FOLDER,
            context,
        )

        load_depression_ids = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_DEPRESSION_IDS,
            context,
        )

        load_classification = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_CLASSIFICATION,
            context,
        )

        # ── Handle temporary output folder ────────────────────────── #

        if not output_folder or output_folder == "TEMPORARY_OUTPUT":
            import tempfile

            output_folder = tempfile.mkdtemp(prefix="mayim_depression_")
            self.log(
                f"Using temporary folder: {output_folder}",
                feedback,
            )

        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        dem_stem = Path(dem_layer.source()).stem

        paths = {
            "depression_ids": output_dir / f"{dem_stem}_depression_ids.tif",
            "classification": output_dir / f"{dem_stem}_depression_classification.tif",
            "inventory": output_dir / f"{dem_stem}_depression_inventory.json",
            "hierarchy": output_dir / f"{dem_stem}_depression_hierarchy.json",
            "report": output_dir / f"{dem_stem}_depression_report.txt",
            "provenance": output_dir / f"{dem_stem}_depression_provenance.json",
            "manifest": output_dir / f"{dem_stem}_depression_analysis.manifest.json",
        }

        # ── Read input manifest if supplied ───────────────────────── #

        input_manifest = None

        if manifest_path and Path(manifest_path).exists():
            try:
                input_manifest = MayimManifest.read(manifest_path)
                errors = input_manifest.validate()
                if errors:
                    self.log_warning(
                        f"Input manifest validation issues: " f"{'; '.join(errors)}",
                        feedback,
                    )
                else:
                    self.log(
                        f"Input manifest loaded: " f"{input_manifest.summary()}",
                        feedback,
                    )
            except Exception as manifest_error:  # noqa: BLE001
                self.log_warning(
                    f"Could not read input manifest: " f"{manifest_error}",
                    feedback,
                )

        # ── Determine vertical accuracy ───────────────────────────── #

        if vertical_accuracy_override > 0:
            vertical_accuracy = float(vertical_accuracy_override)
            va_source = "user override"
        elif input_manifest is not None and input_manifest.vertical_accuracy > 0:
            vertical_accuracy = float(input_manifest.vertical_accuracy)
            va_source = "input manifest"
        else:
            vertical_accuracy = 5.0
            va_source = "conservative default (unknown source)"

        self.log(
            f"Vertical accuracy: {vertical_accuracy:.3f} m " f"({va_source})",
            feedback,
        )

        # ── Initialise provenance ─────────────────────────────────── #

        provenance = {
            "tool": "DEM Depression Analysis",
            "algorithm": "Mayim native Priority-Flood-style "
            "detection and classification",
            "algorithm_ref": (
                "Barnes, R., Lehman, C., and Mulla, D. (2014). "
                "Priority-flood. Computers and Geosciences, 62, "
                "117-127. "
                "Barnes, R., Callaghan, K. L., and Wickert, A. D. "
                "(2020). Computing water flow through complex "
                "landscapes - Part 2. Earth Surface Dynamics, 8(2), "
                "431-445."
            ),
            "ip_status": (
                "Original Mayim implementation. No WhiteboxTools or "
                "RichDEM runtime dependency."
            ),
            "version": self.TOOL_VERSION,
            "stages": [3, 4],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "input_dem": dem_layer.source(),
            "input_manifest": manifest_path or None,
            "parameters": {
                "vertical_accuracy": vertical_accuracy,
                "vertical_accuracy_source": va_source,
                "classification_threshold": classification_threshold,
                "review_margin": review_margin,
            },
            "outputs": {k: str(v) for k, v in paths.items()},
            "statistics": {},
            "warnings": [],
        }

        # ── Log run header ────────────────────────────────────────── #

        self.log("=" * 60, feedback)
        self.log("STAGE 3 - DEPRESSION DELINEATION AND HIERARCHY", feedback)
        self.log("=" * 60, feedback)
        self.log(f"Input DEM          : {dem_layer.source()}", feedback)
        self.log(f"Vertical accuracy  : {vertical_accuracy:.3f} m", feedback)
        self.log(f"Class. threshold   : {classification_threshold}", feedback)
        self.log(f"Review margin      : {review_margin}", feedback)

        feedback.setProgress(5)

        if self.is_cancelled(feedback):
            return {}

        # ══════════════════════════════════════════════════════════════
        # READ DEM
        # ══════════════════════════════════════════════════════════════

        dem_path = dem_layer.source()

        try:
            with rasterio.open(dem_path) as source:

                if source.count < 1:
                    raise QgsProcessingException(
                        "The input raster contains no valid bands."
                    )

                profile = source.profile.copy()
                dem = source.read(1).astype(np.float64)
                nodata = source.nodata if source.nodata is not None else -9999.0
                height, width = dem.shape
                res_x = abs(float(source.transform.a))
                res_y = abs(float(source.transform.e))
                cell_size = (res_x + res_y) / 2.0
                crs_string = (
                    source.crs.to_string() if source.crs is not None else "Unknown"
                )

                if source.crs is None:
                    warning = (
                        "The input DEM has no assigned CRS. "
                        "Depression area and volume estimates will be "
                        "in raster-cell units, not metric units."
                    )
                    provenance["warnings"].append(warning)
                    self.log_warning(warning, feedback)

                provenance["statistics"].update(
                    {
                        "crs": crs_string,
                        "resolution_x": res_x,
                        "resolution_y": res_y,
                        "cell_size": cell_size,
                        "width": width,
                        "height": height,
                        "nodata": str(nodata),
                        "dtype": str(source.dtypes[0]),
                    }
                )

                self.log(
                    f"DEM dimensions     : {width} x {height} cells",
                    feedback,
                )
                self.log(
                    f"Cell size          : {cell_size:.4f} map units",
                    feedback,
                )

        except QgsProcessingException:
            raise
        except Exception as error:
            MayimLogger.critical(
                f"DEM Depression Analysis: failed to read DEM: {error}"
            )
            raise QgsProcessingException(f"Failed to read DEM: {error}") from error

        feedback.setProgress(10)

        if self.is_cancelled(feedback):
            return {}

        # ══════════════════════════════════════════════════════════════
        # STAGE 3 — DEPRESSION DELINEATION AND HIERARCHY
        # ══════════════════════════════════════════════════════════════

        try:
            self.log("", feedback)
            self.log("Running depression detection...", feedback)

            depression_ids, pit_cells, count = detect_depressions(
                dem=dem,
                nodata=nodata,
            )

            self.log(
                f"Depressions detected: {count}",
                feedback,
            )

            if count == 0:
                warning = (
                    "No depressions were detected in the input DEM. "
                    "The DEM may already be fully conditioned, or "
                    "the NoData value may be incorrect."
                )
                provenance["warnings"].append(warning)
                self.log_warning(warning, feedback)

            provenance["statistics"]["depression_count"] = count

            feedback.setProgress(25)

            if self.is_cancelled(feedback):
                return {}

            self.log("Identifying spill elevations...", feedback)

            spill_points = identify_spill_points(
                dem=dem,
                depression_ids=depression_ids,
                nodata=nodata,
            )

            feedback.setProgress(35)

            if self.is_cancelled(feedback):
                return {}

            self.log("Building depression hierarchy...", feedback)

            hierarchy = build_hierarchy(
                dem=dem,
                depression_ids=depression_ids,
                pit_cells=pit_cells,
                spill_points=spill_points,
                nodata=nodata,
                cell_size=cell_size,
            )

            self.log(
                f"Hierarchy depth    : {hierarchy.max_depth}",
                feedback,
            )

            provenance["statistics"]["hierarchy_depth"] = hierarchy.max_depth
            provenance["statistics"]["root_depressions"] = hierarchy.root_count

            feedback.setProgress(45)

            if self.is_cancelled(feedback):
                return {}

        except QgsProcessingException:
            raise
        except Exception as error:
            MayimLogger.critical(f"Stage 3 detection failed: {error}")
            raise QgsProcessingException(
                f"Stage 3 depression delineation failed: {error}"
            ) from error

        # ══════════════════════════════════════════════════════════════
        # STAGE 4 — FEATURE EXTRACTION AND CLASSIFICATION
        # ══════════════════════════════════════════════════════════════

        try:
            self.log("", feedback)
            self.log("=" * 60, feedback)
            self.log("STAGE 4 - DEPRESSION CLASSIFICATION", feedback)
            self.log("=" * 60, feedback)

            self.log("Calculating depression features...", feedback)

            features = calculate_depression_features(
                dem=dem,
                depression_ids=depression_ids,
                spill_points=spill_points,
                cell_size=cell_size,
                nodata=nodata,
            )

            feedback.setProgress(55)

            if self.is_cancelled(feedback):
                return {}

            self.log("Classifying depressions...", feedback)

            thresholds = {
                "threshold": classification_threshold,
                "margin": review_margin,
            }

            results = classify_depressions(
                depression_features=features,
                vertical_accuracy=vertical_accuracy,
                thresholds=thresholds,
            )

            artifact_count = sum(
                1 for r in results.values() if r.classification == ARTIFACT
            )

            real_count = sum(
                1 for r in results.values() if r.classification == REAL_FEATURE
            )

            review_count = sum(
                1 for r in results.values() if r.classification == REVIEW_REQUIRED
            )

            self.log(
                "Classification results:",
                feedback,
            )
            self.log(
                f"  ARTIFACT        : {artifact_count}",
                feedback,
            )
            self.log(
                f"  REAL_FEATURE    : {real_count}",
                feedback,
            )
            self.log(
                f"  REVIEW_REQUIRED : {review_count}",
                feedback,
            )

            if review_count > 0:
                warning = (
                    f"{review_count} depression(s) require analyst "
                    "review before Stage 5 enforcement. These have "
                    "been preserved pending review."
                )
                provenance["warnings"].append(warning)
                self.log_warning(warning, feedback)

            provenance["statistics"].update(
                {
                    "artifact_count": artifact_count,
                    "real_count": real_count,
                    "review_count": review_count,
                }
            )

            feedback.setProgress(65)

            if self.is_cancelled(feedback):
                return {}

        except QgsProcessingException:
            raise
        except Exception as error:
            MayimLogger.critical(f"Stage 4 classification failed: {error}")
            raise QgsProcessingException(
                f"Stage 4 depression classification failed: {error}"
            ) from error

        # ══════════════════════════════════════════════════════════════
        # WRITE RASTER OUTPUTS
        # ══════════════════════════════════════════════════════════════

        try:
            self.log("", feedback)
            self.log("Writing raster outputs...", feedback)

            base_profile = profile

            # ── Depression ID raster ──────────────────────────────── #

            id_profile = base_profile.copy()
            id_profile.update(
                dtype="int32",
                count=1,
                compress="lzw",
                nodata=-1,
            )

            with rasterio.open(
                paths["depression_ids"],
                "w",
                **id_profile,
            ) as dst:
                dst.write(depression_ids.astype(np.int32), 1)

            self.log(
                f"Depression ID raster: " f"{paths['depression_ids'].name}",
                feedback,
            )

            # ── Classification raster ─────────────────────────────── #
            # Values:
            #   0   = not a depression
            #   1   = ARTIFACT
            #   2   = REAL_FEATURE
            #   3   = REVIEW_REQUIRED
            #   255 = NoData

            _classification_codes = {
                ARTIFACT: 1,
                REAL_FEATURE: 2,
                REVIEW_REQUIRED: 3,
            }

            classification_array = np.zeros(
                (height, width),
                dtype=np.uint8,
            )

            valid_mask = depression_ids > 0
            classification_array[~valid_mask] = 0

            nodata_mask = depression_ids == -1
            classification_array[nodata_mask] = 255

            for depression_id, result in results.items():
                cell_mask = depression_ids == depression_id
                code = _classification_codes.get(
                    result.classification,
                    3,
                )
                classification_array[cell_mask] = code

            class_profile = base_profile.copy()
            class_profile.update(
                dtype="uint8",
                count=1,
                compress="lzw",
                nodata=255,
            )

            with rasterio.open(
                paths["classification"],
                "w",
                **class_profile,
            ) as dst:
                dst.write(classification_array, 1)

            self.log(
                f"Classification raster: " f"{paths['classification'].name}",
                feedback,
            )

            feedback.setProgress(75)

        except QgsProcessingException:
            raise
        except Exception as error:
            MayimLogger.critical(f"Failed to write raster outputs: {error}")
            raise QgsProcessingException(
                f"Failed to write raster outputs: {error}"
            ) from error

        # ══════════════════════════════════════════════════════════════
        # WRITE JSON OUTPUTS
        # ══════════════════════════════════════════════════════════════

        try:
            self.log("Writing JSON outputs...", feedback)

            # ── Depression inventory ──────────────────────────────── #

            inventory = {
                str(did): {
                    "features": features.get(did, {}),
                    "classification": (
                        results[did].to_dict() if did in results else None
                    ),
                }
                for did in sorted(features)
            }

            with open(
                paths["inventory"],
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(inventory, file, indent=2, default=str)

            self.log(
                f"Depression inventory: " f"{paths['inventory'].name}",
                feedback,
            )

            # ── Depression hierarchy ──────────────────────────────── #

            with open(
                paths["hierarchy"],
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    hierarchy.to_dict(),
                    file,
                    indent=2,
                    default=str,
                )

            self.log(
                f"Depression hierarchy: " f"{paths['hierarchy'].name}",
                feedback,
            )

            feedback.setProgress(82)

        except QgsProcessingException:
            raise
        except Exception as error:
            MayimLogger.critical(f"Failed to write JSON outputs: {error}")
            raise QgsProcessingException(
                f"Failed to write JSON outputs: {error}"
            ) from error

        # ══════════════════════════════════════════════════════════════
        # WRITE TEXT REPORT
        # ══════════════════════════════════════════════════════════════

        self._write_text_report(
            paths=paths,
            provenance=provenance,
            features=features,
            results=results,
            hierarchy=hierarchy,
            feedback=feedback,
        )

        # ══════════════════════════════════════════════════════════════
        # WRITE PROVENANCE
        # ══════════════════════════════════════════════════════════════

        with open(
            paths["provenance"],
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                provenance,
                file,
                indent=4,
                default=str,
            )

        self.log(
            f"Provenance written    : {paths['provenance'].name}",
            feedback,
        )

        feedback.setProgress(88)

        # ══════════════════════════════════════════════════════════════
        # WRITE MAYIMMANIFEST
        # ══════════════════════════════════════════════════════════════

        try:
            if input_manifest is not None:
                manifest = input_manifest.derive(
                    produced_by=self.TOOL_VERSION,
                    raster_path=str(paths["depression_ids"]),
                    stage=4,
                    audit_log_path=str(paths["provenance"]),
                    warnings=(
                        provenance["warnings"] if provenance["warnings"] else None
                    ),
                    width=width,
                    height=height,
                )
            else:
                manifest = MayimManifest.create(
                    raster_path=str(paths["depression_ids"]),
                    crs=crs_string,
                    cell_size=cell_size,
                    vertical_accuracy=vertical_accuracy,
                    nodata=float(nodata),
                    produced_by=self.TOOL_VERSION,
                    stage=4,
                    audit_log_path=str(paths["provenance"]),
                    warnings=(
                        provenance["warnings"] if provenance["warnings"] else None
                    ),
                    width=width,
                    height=height,
                )

            manifest.write(str(paths["manifest"]))

            self.log(
                f"MayimManifest written : {paths['manifest'].name}",
                feedback,
            )

        except Exception as manifest_error:  # noqa: BLE001
            self.log_warning(
                f"Could not write MayimManifest: {manifest_error}",
                feedback,
            )

        feedback.setProgress(92)

        # ══════════════════════════════════════════════════════════════
        # LOAD SELECTED LAYERS INTO QGIS
        # ══════════════════════════════════════════════════════════════

        self._load_layers_into_project(
            paths=paths,
            dem_stem=dem_stem,
            load_depression_ids=load_depression_ids,
            load_classification=load_classification,
            feedback=feedback,
        )

        feedback.setProgress(98)

        report_path = Path(paths["report"]).resolve()
        report_uri = report_path.as_uri()

        self.log(
            f"Depression analysis report: {report_path}",
            feedback,
        )
        feedback.pushInfo(f"Open report: {report_uri}")

        self.log("", feedback)
        self.log("=" * 60, feedback)
        self.log("STAGE 3/4 DEPRESSION ANALYSIS COMPLETE", feedback)
        self.log("=" * 60, feedback)

        feedback.setProgress(100)

        return {
            self.OUTPUT_DEPRESSION_IDS: str(paths["depression_ids"]),
            self.OUTPUT_CLASSIFICATION: str(paths["classification"]),
            self.OUTPUT_INVENTORY: str(paths["inventory"]),
            self.OUTPUT_HIERARCHY: str(paths["hierarchy"]),
            self.OUTPUT_REPORT: str(paths["report"]),
            self.OUTPUT_PROVENANCE: str(paths["provenance"]),
            self.OUTPUT_MANIFEST: str(paths["manifest"]),
        }

    # ── Private helper methods ────────────────────────────────────── #

    def _load_layers_into_project(
        self,
        paths: dict,
        dem_stem: str,
        load_depression_ids: bool,
        load_classification: bool,
        feedback: QgsProcessingFeedback,
    ) -> None:
        """
        Load selected analysis rasters into the QGIS project.

        No layer group is created. Outputs are added directly to
        the project layer tree.

        :param paths: Output path dictionary.
        :param dem_stem: Input DEM filename stem.
        :param load_depression_ids: Load depression ID raster.
        :param load_classification: Load classification raster.
        :param feedback: QGIS processing feedback object.
        """
        try:
            from qgis.core import QgsProject, QgsRasterLayer

            project = QgsProject.instance()
            loaded = 0

            def load_raster(
                file_path: Path,
                layer_name: str,
            ) -> None:
                nonlocal loaded
                layer = QgsRasterLayer(
                    str(file_path),
                    layer_name,
                    "gdal",
                )
                if not layer.isValid():
                    self.log_warning(
                        f"Could not load: {layer_name}",
                        feedback,
                    )
                    return
                project.addMapLayer(layer, True)
                loaded += 1
                self.log(
                    f"Loaded into project: {layer_name}",
                    feedback,
                )

            if load_depression_ids:
                load_raster(
                    paths["depression_ids"],
                    f"{dem_stem} - Depression IDs",
                )

            if load_classification:
                load_raster(
                    paths["classification"],
                    f"{dem_stem} - Depression Classification",
                )

            self.log(
                f"{loaded} analysis layer(s) loaded into project.",
                feedback,
            )

        except Exception as error:  # noqa: BLE001
            self.log_warning(
                f"Could not load analysis layers: {error}",
                feedback,
            )

    def _write_text_report(
        self,
        paths: dict,
        provenance: dict,
        features: dict,
        results: dict,
        hierarchy,
        feedback: QgsProcessingFeedback,
    ) -> None:
        """
        Write the Stage 3/4 human-readable text report.

        :param paths: Output path dictionary.
        :param provenance: Provenance record dictionary.
        :param features: Depression feature dictionary.
        :param results: Classification result dictionary.
        :param hierarchy: Depression hierarchy instance.
        :param feedback: QGIS processing feedback object.
        """
        stats = provenance.get("statistics", {})
        params = provenance.get("parameters", {})
        warns = provenance.get("warnings", [])

        lines = []
        a = lines.append

        a("=" * 70)
        a("MAYIM TOOLS - DEM DEPRESSION ANALYSIS REPORT")
        a("=" * 70)
        a(f"Generated  : {provenance.get('timestamp', '')}")
        a(f"Tool       : {provenance.get('tool', '')}")
        a(f"Version    : {provenance.get('version', '')}")
        a(f"Stages     : {provenance.get('stages', [3, 4])}")
        a(f"Input DEM  : {provenance.get('input_dem', '')}")
        a(f"Algorithm  : {provenance.get('algorithm', '')}")
        a(f"IP status  : {provenance.get('ip_status', '')}")
        a("")

        a("-" * 70)
        a("PARAMETERS")
        a("-" * 70)
        a(
            f"Vertical accuracy      : "
            f"{params.get('vertical_accuracy', 'Unknown')} m "
            f"({params.get('vertical_accuracy_source', '')})"
        )
        a(
            f"Classification threshold: "
            f"{params.get('classification_threshold', 0.60)}"
        )
        a(f"Review margin          : " f"{params.get('review_margin', 0.15)}")
        a("")

        a("-" * 70)
        a("DEM METADATA")
        a("-" * 70)
        a(f"CRS                    : {stats.get('crs', 'Unknown')}")
        a(f"Resolution X           : {stats.get('resolution_x', 'Unknown')}")
        a(f"Resolution Y           : {stats.get('resolution_y', 'Unknown')}")
        a(f"Cell size              : {stats.get('cell_size', 'Unknown')}")
        a(f"Width                  : {stats.get('width', 'Unknown')} cells")
        a(f"Height                 : {stats.get('height', 'Unknown')} cells")
        a(f"NoData value           : {stats.get('nodata', 'Unknown')}")
        a(f"Data type              : {stats.get('dtype', 'Unknown')}")
        a("")

        a("-" * 70)
        a("STAGE 3 - DEPRESSION DELINEATION")
        a("-" * 70)
        a(f"Depressions detected   : " f"{stats.get('depression_count', 0)}")
        a(f"Root depressions       : " f"{stats.get('root_depressions', 0)}")
        a(f"Hierarchy depth        : " f"{stats.get('hierarchy_depth', 0)}")
        a("")

        a("-" * 70)
        a("STAGE 4 - DEPRESSION CLASSIFICATION")
        a("-" * 70)
        a(f"ARTIFACT               : " f"{stats.get('artifact_count', 0)}")
        a(f"REAL_FEATURE           : " f"{stats.get('real_count', 0)}")
        a(f"REVIEW_REQUIRED        : " f"{stats.get('review_count', 0)}")
        a("")

        a("-" * 70)
        a("CLASSIFICATION LEGEND")
        a("-" * 70)
        a("  Raster value 0   = not a depression")
        a("  Raster value 1   = ARTIFACT")
        a("  Raster value 2   = REAL_FEATURE")
        a("  Raster value 3   = REVIEW_REQUIRED")
        a("  Raster value 255 = NoData")
        a("")

        a("-" * 70)
        a("DEPRESSION SUMMARY")
        a("-" * 70)

        if features:
            for did in sorted(features):
                feat = features[did]
                result = results.get(did)

                classification_label = (
                    result.classification if result is not None else "Unknown"
                )

                score = f"{result.artifact_score:.3f}" if result is not None else "N/A"

                a(
                    f"  Depression {did:>4} : "
                    f"{classification_label:<16} "
                    f"score={score:>6}  "
                    f"depth={feat.get('depth', 0.0):.3f}  "
                    f"area={feat.get('area_cells', 0):>5} cells"
                )
        else:
            a("  No depressions detected.")

        a("")

        if warns:
            a("-" * 70)
            a("WARNINGS")
            a("-" * 70)
            for i, warning in enumerate(warns, 1):
                words = warning.split()
                line = f"  {i}. "
                for word in words:
                    if len(line) + len(word) + 1 > 68:
                        a(line)
                        line = "     " + word + " "
                    else:
                        line += word + " "
                if line.strip():
                    a(line)
            a("")

        a("-" * 70)
        a("OUTPUT FILES")
        a("-" * 70)
        a(f"Depression IDs         : {paths['depression_ids'].name}")
        a(f"Classification raster  : {paths['classification'].name}")
        a(f"Depression inventory   : {paths['inventory'].name}")
        a(f"Depression hierarchy   : {paths['hierarchy'].name}")
        a(f"Report (this file)     : {paths['report'].name}")
        a(f"Provenance             : {paths['provenance'].name}")
        a(f"MayimManifest          : {paths['manifest'].name}")
        a("")

        a("-" * 70)
        a("RECOMMENDED NEXT STEPS")
        a("-" * 70)

        review_count = stats.get("review_count", 0)
        artifact_count = stats.get("artifact_count", 0)

        if review_count > 0:
            a(
                f"  {review_count} depression(s) require analyst review "
                "before Stage 5 enforcement."
            )
            a("  Inspect the classification raster and inventory " "before proceeding.")

        if artifact_count > 0:
            a(
                f"  {artifact_count} depression(s) classified as ARTIFACT "
                "are candidates for Stage 5 selective enforcement."
            )

        if artifact_count == 0 and review_count == 0:
            a(
                "  No artifacts or review cases detected. "
                "DEM may not require Stage 5 enforcement."
            )

        a("")
        a("  Next tool: DEM Hydrological Filling (Stage 5)")
        a("")

        a("-" * 70)
        a("REFERENCES")
        a("-" * 70)
        a(
            "Barnes, R., Lehman, C., and Mulla, D. (2014). "
            "Priority-flood: An optimal depression-filling and "
            "watershed-labeling algorithm for digital elevation "
            "models. Computers and Geosciences, 62, 117-127."
        )
        a("")
        a(
            "Barnes, R., Callaghan, K. L., and Wickert, A. D. (2020). "
            "Computing water flow through complex landscapes - Part 2: "
            "Finding hierarchies in depressions and morphological "
            "segmentations. Earth Surface Dynamics, 8(2), 431-445."
        )
        a("")
        a(
            "Lindsay, J. B., and Creed, I. F. (2006). "
            "Distinguishing actual and artefact depressions in digital "
            "elevation data. Computers and Geosciences, 32(8), "
            "1192-1204."
        )
        a("")
        a(
            "Mayim Tools DEM Hydrological Conditioning Research Paper "
            "(Rev 1, August 2026). Stages 3 and 4."
        )
        a("")

        a("-" * 70)
        a("IP STATEMENT")
        a("-" * 70)
        a(
            "This tool uses an original Mayim implementation of the "
            "depression-detection and classification methodology. "
            "No WhiteboxTools, RichDEM or other third-party "
            "hydrological runtime is used. Only Python, NumPy and "
            "rasterio are used as external dependencies."
        )
        a("")

        a("-" * 70)
        a("REPORT LOCATION")
        a("-" * 70)
        a(str(Path(paths["report"]).resolve()))
        a("")

        a("=" * 70)
        a("End of DEM Depression Analysis report")
        a("=" * 70)
        a("")

        report_text = "\n".join(lines)

        with open(
            paths["report"],
            "w",
            encoding="utf-8",
        ) as file:
            file.write(report_text)

        self.log(
            f"Report written        : {paths['report'].name}",
            feedback,
        )
