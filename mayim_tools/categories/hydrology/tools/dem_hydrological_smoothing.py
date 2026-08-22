"""
Mayim Tools - DEM Hydrological Smoothing
========================================

Implements Stage 2 of the Mayim Tools DEM hydrological-conditioning
pipeline: controlled, edge-preserving smoothing.

This implementation uses Perona-Malik-style anisotropic diffusion.
The method reduces high-frequency noise while limiting diffusion across
strong elevation gradients. Unlike a uniform low-pass filter, it does
not apply the same smoothing strength indiscriminately across the DEM.

The tool:

    - Reads a single-band DEM.
    - Preserves the original input.
    - Scales the effective processing according to DEM resolution.
    - Applies edge-preserving anisotropic diffusion.
    - Produces a smoothed DEM.
    - Produces a cell-level difference raster.
    - Produces a cell-level smoothing mask.
    - Writes a text report.
    - Writes a JSON provenance record.
    - Optionally loads generated rasters directly into QGIS.

This tool does not perform:

    - Void interpolation.
    - Artifact classification.
    - Depression filling.
    - Depression breaching.
    - Flow-direction calculation.
    - Flow-accumulation calculation.

Those operations belong to other pipeline stages.

Reference:

    Perona, P. and Malik, J. (1990).
    Scale-space and edge detection using anisotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7), 629-639.

    DEM Hydrological Conditioning research paper:
    Stage 2 - Controlled Smoothing.

License:
    GPL-2.0+
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFile,
    QgsProcessingOutputRasterLayer,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)

from mayim_tools.core.logger import MayimLogger
from mayim_tools.core.validation_utils import ValidationUtils
from mayim_tools.processing.algorithms.base_algorithm import MayimBaseAlgorithm


class DEMHydrologicalSmoothing(MayimBaseAlgorithm):
    """
    Processing algorithm for Stage 2 controlled DEM smoothing.

    Anisotropic diffusion is applied to reduce high-frequency noise while
    preserving stronger terrain edges. The input DEM is never overwritten.
    """

    PARAM_DEM = "INPUT_DEM"
    PARAM_ITERATIONS = "ITERATIONS"
    PARAM_DIFFUSION = "DIFFUSION_STRENGTH"
    PARAM_EDGE_THRESHOLD = "EDGE_THRESHOLD"
    PARAM_RESOLUTION_SCALE = "RESOLUTION_SCALE"
    PARAM_OUTPUT_FOLDER = "OUTPUT_FOLDER"

    PARAM_LOAD_SMOOTHED = "LOAD_SMOOTHED_DEM"
    PARAM_LOAD_DIFFERENCE = "LOAD_DIFFERENCE"
    PARAM_LOAD_MASK = "LOAD_SMOOTHING_MASK"

    OUTPUT_SMOOTHED = "OUTPUT_SMOOTHED"
    OUTPUT_DIFFERENCE = "OUTPUT_DIFFERENCE"
    OUTPUT_MASK = "OUTPUT_MASK"
    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_PROVENANCE = "OUTPUT_PROVENANCE"

    def name(self) -> str:
        """Return the unique Processing algorithm identifier."""
        return "demhydrologicalsmoothing"

    def displayName(self) -> str:  # noqa: N802
        """Return the human-readable algorithm name."""
        return "DEM Hydrological Smoothing"

    def group(self) -> str:
        """Return the Processing Toolbox group name."""
        return "Hydrology Tools"

    def groupId(self) -> str:  # noqa: N802
        """Return the unique Processing Toolbox group identifier."""
        return "hydrology"

    def createInstance(self) -> DEMHydrologicalSmoothing:  # noqa: N802
        """Return a new instance of this algorithm."""
        return DEMHydrologicalSmoothing()

    def shortHelpString(self) -> str:  # noqa: N802
        """Return the Processing Toolbox help text."""
        return (
            "<b>DEM Hydrological Smoothing</b><br><br>"
            "Implements Stage 2 controlled smoothing using an "
            "edge-preserving anisotropic diffusion filter.<br><br>"
            "<b>Recommended starting values:</b><br>"
            "<ul>"
            "<li><b>Diffusion iterations:</b> 5 "
            "(valid range: 1-100). Use 3-5 for light noise and "
            "5-10 for more persistent high-frequency noise.</li>"
            "<li><b>Diffusion strength:</b> 0.20 "
            "(valid range: 0.01-0.25). Values above 0.25 are not "
            "permitted by this implementation.</li>"
            "<li><b>Edge threshold:</b> 1.0 elevation unit. "
            "Lower values preserve stronger terrain breaks; higher "
            "values allow smoothing across larger gradients.</li>"
            "</ul><br>"
            "<b>Practical guidance:</b><br>"
            "Start with the recommended values and inspect the "
            "smoothing difference raster and smoothing mask. Increase "
            "iterations only if residual high-frequency noise remains. "
            "Do not increase smoothing merely to remove genuine "
            "terrain features.<br><br>"
            "<b>Edge threshold guidance:</b><br>"
            "The edge threshold is expressed in the elevation units of "
            "the input DEM. Where a reliable RMSE is available, use a "
            "threshold near one to two times the RMSE as an initial "
            "sensitivity range, then review the results. This is a "
            "starting heuristic, not an automatic accuracy correction."
            "<br><br>"
            "<b>Important:</b> Use this tool only after reviewing the "
            "Stage 1 screening report and artifact mask. The input DEM "
            "is not overwritten.<br><br>"
            "This tool produces a smoothed DEM, a cell-level elevation "
            "difference raster, a smoothing mask, a text report, and "
            "a JSON provenance record.<br><br>"
            "This tool does not fill or breach depressions and does not "
            "calculate flow direction."
        )

    def helpUrl(self) -> str:  # noqa: N802
        """Return the project documentation URL."""
        return "https://github.com/chrismayim/mayim-tools"

    def tags(self) -> list[str]:
        """Return searchable Processing Toolbox tags."""
        return [
            "mayim",
            "dem",
            "hydrology",
            "smoothing",
            "anisotropic",
            "diffusion",
            "noise",
            "edge-preserving",
            "stage-2",
        ]

    def initAlgorithm(self, config=None) -> None:  # noqa: N802
        """Define the Processing Toolbox parameters and outputs."""

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.PARAM_DEM,
                "Input DEM",
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_ITERATIONS,
                "Diffusion iterations " "(recommended: 5; valid range: 1-100)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1,
                maxValue=100,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_DIFFUSION,
                "Diffusion strength " "(recommended: 0.20; valid range: 0.01-0.25)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.20,
                minValue=0.01,
                maxValue=0.25,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_EDGE_THRESHOLD,
                "Edge threshold in elevation units "
                "(recommended: 1.0; lower values preserve edges)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.000001,
                maxValue=100000.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_RESOLUTION_SCALE,
                "Scale diffusion according to DEM resolution",
                defaultValue=True,
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
                self.PARAM_LOAD_SMOOTHED,
                "Load smoothed DEM into project",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_LOAD_DIFFERENCE,
                "Load smoothing difference raster into project",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_LOAD_MASK,
                "Load smoothing mask into project",
                defaultValue=True,
            )
        )

        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_SMOOTHED,
                "Smoothed DEM",
            )
        )

        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_DIFFERENCE,
                "Smoothing difference raster",
            )
        )

        self.addOutput(
            QgsProcessingOutputRasterLayer(
                self.OUTPUT_MASK,
                "Smoothing mask",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_REPORT,
                "Smoothing report",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_PROVENANCE,
                "Smoothing provenance",
            )
        )

    def processAlgorithm(  # noqa: N802
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """
        Execute Stage 2 controlled smoothing.

        :param parameters: Processing parameters.
        :param context: QGIS processing context.
        :param feedback: Processing feedback object.
        :returns: Dictionary of output paths.
        """
        try:
            import rasterio
        except ImportError as error:
            raise QgsProcessingException(
                "The rasterio library is required for this tool."
            ) from error

        dem_layer = self.parameterAsRasterLayer(
            parameters,
            self.PARAM_DEM,
            context,
        )

        if not ValidationUtils.is_valid_raster_layer(dem_layer):
            raise QgsProcessingException("The input DEM is missing or invalid.")

        iterations = self.parameterAsInt(
            parameters,
            self.PARAM_ITERATIONS,
            context,
        )

        diffusion_strength = self.parameterAsDouble(
            parameters,
            self.PARAM_DIFFUSION,
            context,
        )

        edge_threshold = self.parameterAsDouble(
            parameters,
            self.PARAM_EDGE_THRESHOLD,
            context,
        )

        resolution_scale = self.parameterAsBoolean(
            parameters,
            self.PARAM_RESOLUTION_SCALE,
            context,
        )

        output_folder = self.parameterAsString(
            parameters,
            self.PARAM_OUTPUT_FOLDER,
            context,
        )

        if iterations < 1:
            raise QgsProcessingException("Diffusion iterations must be at least 1.")

        if diffusion_strength <= 0 or diffusion_strength > 0.25:
            raise QgsProcessingException(
                "Diffusion strength must be greater than 0 and no more " "than 0.25."
            )

        if edge_threshold <= 0:
            raise QgsProcessingException("Edge threshold must be greater than zero.")

        if not output_folder or output_folder == "TEMPORARY_OUTPUT":
            import tempfile

            output_folder = tempfile.mkdtemp(
                prefix="mayim_smoothing_",
            )

        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        dem_stem = Path(dem_layer.source()).stem

        paths = {
            "smoothed": output_dir / f"{dem_stem}_smoothed.tif",
            "difference": output_dir / f"{dem_stem}_smoothing_difference.tif",
            "mask": output_dir / f"{dem_stem}_smoothing_mask.tif",
            "report": output_dir / f"{dem_stem}_smoothing_report.txt",
            "provenance": output_dir / f"{dem_stem}_smoothing_provenance.json",
        }

        provenance = {
            "tool": "DEM Hydrological Smoothing",
            "algorithm_id": self.name(),
            "version": "0.1.0",
            "stage": 2,
            "timestamp": datetime.now().isoformat(),
            "input_dem": dem_layer.source(),
            "parameters": {
                "iterations": iterations,
                "diffusion_strength": diffusion_strength,
                "edge_threshold": edge_threshold,
                "resolution_scale": resolution_scale,
            },
            "outputs": {key: str(value) for key, value in paths.items()},
            "statistics": {},
            "warnings": [],
        }

        self.log("=" * 70, feedback)
        self.log("STAGE 2 - CONTROLLED DEM SMOOTHING", feedback)
        self.log("=" * 70, feedback)
        self.log(f"Input DEM: {dem_layer.source()}", feedback)
        self.log(
            f"Iterations: {iterations} "
            "(recommended starting value: 5; valid range: 1-100)",
            feedback,
        )
        self.log(
            f"Diffusion strength: {diffusion_strength} "
            "(recommended starting value: 0.20; valid range: 0.01-0.25)",
            feedback,
        )
        self.log(
            f"Edge threshold: {edge_threshold} "
            "(recommended starting value: 1.0 elevation unit)",
            feedback,
        )
        self.log(
            "Guidance: inspect the smoothing difference and smoothing "
            "mask before using the smoothed DEM downstream.",
            feedback,
        )

        dem_path = dem_layer.source()

        try:
            with rasterio.open(dem_path) as source:
                if source.count < 1:
                    raise QgsProcessingException(
                        "The input raster does not contain a valid band."
                    )

                if source.crs is None:
                    warning = (
                        "The input DEM has no assigned CRS. The smoothing "
                        "calculation can proceed because it is local, but "
                        "the output should not be used for distance-based "
                        "hydrological calculations until a CRS is assigned."
                    )
                    provenance["warnings"].append(warning)
                    self.log_warning(warning, feedback)

                profile = source.profile.copy()
                original = source.read(1).astype(np.float64)
                nodata = source.nodata
                height, width = original.shape

                if nodata is not None:
                    valid_mask = original != nodata
                else:
                    valid_mask = np.isfinite(original)

                valid_mask &= np.isfinite(original)

                if not np.any(valid_mask):
                    raise QgsProcessingException(
                        "The input DEM contains no valid elevation cells."
                    )

                working = original.copy()

                valid_values = working[valid_mask]
                mean_elevation = float(np.mean(valid_values))
                median_elevation = float(np.median(valid_values))

                provenance["statistics"]["mean_elevation"] = mean_elevation
                provenance["statistics"]["median_elevation"] = median_elevation

                if resolution_scale:
                    resolution_x = abs(float(source.transform.a))
                    resolution_y = abs(float(source.transform.e))
                    mean_resolution = (resolution_x + resolution_y) / 2.0

                    scale_factor = self._resolution_scale_factor(
                        mean_resolution,
                    )
                else:
                    resolution_x = None
                    resolution_y = None
                    mean_resolution = None
                    scale_factor = 1.0

                effective_diffusion = diffusion_strength * scale_factor

                if effective_diffusion > 0.25:
                    warning = (
                        "Resolution scaling reduced the effective diffusion "
                        "strength to the maximum stable value of 0.25."
                    )
                    provenance["warnings"].append(warning)
                    self.log_warning(warning, feedback)
                    effective_diffusion = 0.25

                self.log(
                    f"Effective diffusion strength: " f"{effective_diffusion:.6f}",
                    feedback,
                )

                provenance["statistics"]["resolution_x"] = resolution_x
                provenance["statistics"]["resolution_y"] = resolution_y
                provenance["statistics"]["mean_resolution"] = mean_resolution
                provenance["statistics"]["resolution_scale_factor"] = scale_factor
                provenance["statistics"]["effective_diffusion"] = effective_diffusion
                provenance["statistics"]["valid_cells"] = int(np.sum(valid_mask))

                if self.is_cancelled(feedback):
                    return {}

                # Use NaN internally for invalid cells. The original NoData
                # cells are restored before writing the output rasters.
                working[~valid_mask] = np.nan

                self.log(
                    "Applying edge-preserving anisotropic diffusion...",
                    feedback,
                )

                smoothed = self._anisotropic_diffusion(
                    array=working,
                    valid_mask=valid_mask,
                    iterations=iterations,
                    diffusion_strength=effective_diffusion,
                    edge_threshold=edge_threshold,
                    feedback=feedback,
                )

                if self.is_cancelled(feedback):
                    return {}

                # Restore the original values in NoData cells. The smoothing
                # tool must never invent values in existing NoData regions.
                smoothed[~valid_mask] = original[~valid_mask]

                difference = np.zeros_like(original, dtype=np.float64)
                difference[valid_mask] = smoothed[valid_mask] - original[valid_mask]

                # A very small numerical difference is treated as unchanged.
                change_tolerance = max(
                    np.finfo(np.float64).eps * 100.0,
                    abs(edge_threshold) * 1e-9,
                )

                smoothing_mask = np.zeros(
                    original.shape,
                    dtype=np.uint8,
                )
                smoothing_mask[valid_mask & (np.abs(difference) > change_tolerance)] = 1

                changed_cells = int(np.sum(smoothing_mask == 1))
                valid_cell_count = int(np.sum(valid_mask))

                if valid_cell_count > 0:
                    changed_percentage = (changed_cells / valid_cell_count) * 100.0
                else:
                    changed_percentage = 0.0

                valid_difference = difference[valid_mask]

                provenance["statistics"].update(
                    {
                        "changed_cells": changed_cells,
                        "changed_percentage": round(
                            changed_percentage,
                            6,
                        ),
                        "maximum_absolute_change": float(
                            np.max(np.abs(valid_difference))
                        ),
                        "mean_absolute_change": float(
                            np.mean(np.abs(valid_difference))
                        ),
                        "minimum_change": float(np.min(valid_difference)),
                        "maximum_change": float(np.max(valid_difference)),
                    }
                )

                self.log(
                    f"Changed cells: {changed_cells:,} "
                    f"({changed_percentage:.4f}% of valid cells)",
                    feedback,
                )
                self.log(
                    "Writing Stage 2 output rasters...",
                    feedback,
                )

                # ----------------------------------------------------------
                # Write smoothed DEM
                # ----------------------------------------------------------

                smoothed_profile = profile.copy()
                smoothed_profile.update(
                    dtype="float32",
                    count=1,
                    compress="lzw",
                )

                if nodata is not None:
                    smoothed_profile["nodata"] = nodata

                smoothed_output = smoothed.astype(np.float32)

                with rasterio.open(
                    paths["smoothed"],
                    "w",
                    **smoothed_profile,
                ) as destination:
                    destination.write(smoothed_output, 1)

                # ----------------------------------------------------------
                # Write difference raster
                # ----------------------------------------------------------

                difference_profile = profile.copy()
                difference_profile.update(
                    dtype="float32",
                    count=1,
                    compress="lzw",
                    nodata=-9999.0,
                )

                difference_output = difference.astype(np.float32)
                difference_output[~valid_mask] = -9999.0

                with rasterio.open(
                    paths["difference"],
                    "w",
                    **difference_profile,
                ) as destination:
                    destination.write(difference_output, 1)

                # ----------------------------------------------------------
                # Write smoothing mask
                # ----------------------------------------------------------

                mask_profile = profile.copy()
                mask_profile.update(
                    dtype="uint8",
                    count=1,
                    compress="lzw",
                    nodata=255,
                )

                mask_output = smoothing_mask.copy()
                mask_output[~valid_mask] = 255

                with rasterio.open(
                    paths["mask"],
                    "w",
                    **mask_profile,
                ) as destination:
                    destination.write(mask_output, 1)

        except QgsProcessingException:
            raise

        except Exception as error:
            MayimLogger.critical(f"DEM Hydrological Smoothing failed: {error}")
            raise QgsProcessingException(
                f"DEM Hydrological Smoothing failed: {error}"
            ) from error

        # ------------------------------------------------------------------
        # Write report and provenance
        # ------------------------------------------------------------------

        self._write_text_report(
            paths=paths,
            provenance=provenance,
            feedback=feedback,
        )

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

        report_path = Path(paths["report"]).resolve()
        report_uri = report_path.as_uri()

        self.log(
            f"Stage 2 report: {report_path}",
            feedback,
        )
        feedback.pushInfo(f"Open Stage 2 report: {report_uri}")

        # ------------------------------------------------------------------
        # Load selected outputs directly into QGIS
        # ------------------------------------------------------------------

        load_smoothed = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_SMOOTHED,
            context,
        )

        load_difference = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_DIFFERENCE,
            context,
        )

        load_mask = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_MASK,
            context,
        )

        self._load_layers_into_project(
            paths=paths,
            dem_stem=dem_stem,
            load_smoothed=load_smoothed,
            load_difference=load_difference,
            load_mask=load_mask,
            feedback=feedback,
        )

        feedback.setProgress(100)

        self.log(
            "STAGE 2 - CONTROLLED DEM SMOOTHING COMPLETE",
            feedback,
        )

        return {
            self.OUTPUT_SMOOTHED: str(paths["smoothed"]),
            self.OUTPUT_DIFFERENCE: str(paths["difference"]),
            self.OUTPUT_MASK: str(paths["mask"]),
            self.OUTPUT_REPORT: str(paths["report"]),
            self.OUTPUT_PROVENANCE: str(paths["provenance"]),
        }

    def _resolution_scale_factor(self, resolution: float) -> float:
        """
        Calculate a conservative resolution scaling factor.

        The reference resolution is 1 metre. Coarser DEMs receive a lower
        factor because a coarse cell represents a larger ground area and
        generally requires less aggressive per-cell diffusion.

        The result is bounded to avoid unstable or negligible behaviour.

        :param resolution: Mean DEM cell size in map units.
        :returns: Resolution scaling factor.
        """
        if resolution <= 0:
            return 1.0

        reference_resolution = 1.0
        factor = reference_resolution / resolution

        return float(np.clip(factor, 0.1, 1.0))

    def _anisotropic_diffusion(
        self,
        array: np.ndarray,
        valid_mask: np.ndarray,
        iterations: int,
        diffusion_strength: float,
        edge_threshold: float,
        feedback: QgsProcessingFeedback,
    ) -> np.ndarray:
        """
        Apply Perona-Malik-style anisotropic diffusion.

        The calculation uses four-connected neighbours. Diffusion across
        large elevation gradients is reduced by the conductance function,
        preserving terrain edges more effectively than uniform smoothing.

        :param array: DEM array containing NaN in invalid cells.
        :param valid_mask: Boolean mask of valid cells.
        :param iterations: Number of diffusion iterations.
        :param diffusion_strength: Update strength per iteration.
        :param edge_threshold: Gradient threshold controlling edge
            preservation.
        :param feedback: Processing feedback object.
        :returns: Smoothed DEM array.
        """
        result = array.copy()
        height, width = result.shape

        for iteration in range(iterations):
            if self.is_cancelled(feedback):
                return result

            north = np.full_like(result, np.nan)
            south = np.full_like(result, np.nan)
            west = np.full_like(result, np.nan)
            east = np.full_like(result, np.nan)

            north[1:, :] = result[:-1, :]
            south[:-1, :] = result[1:, :]
            west[:, 1:] = result[:, :-1]
            east[:, :-1] = result[:, 1:]

            valid_north = np.zeros_like(valid_mask)
            valid_south = np.zeros_like(valid_mask)
            valid_west = np.zeros_like(valid_mask)
            valid_east = np.zeros_like(valid_mask)

            valid_north[1:, :] = valid_mask[:-1, :] & valid_mask[1:, :]
            valid_south[:-1, :] = valid_mask[1:, :] & valid_mask[:-1, :]
            valid_west[:, 1:] = valid_mask[:, :-1] & valid_mask[:, 1:]
            valid_east[:, :-1] = valid_mask[:, 1:] & valid_mask[:, :-1]

            result_valid = valid_mask & np.isfinite(result)
            update = np.zeros_like(result)

            neighbours = (
                (north, valid_north),
                (south, valid_south),
                (west, valid_west),
                (east, valid_east),
            )

            for neighbour, neighbour_mask in neighbours:
                gradient = neighbour - result

                conductance = np.exp(
                    -((np.abs(gradient) / max(edge_threshold, 1e-12)) ** 2)
                )

                contribution = np.where(
                    neighbour_mask & np.isfinite(gradient),
                    conductance * gradient,
                    0.0,
                )

                update += contribution

            result[result_valid] = (
                result[result_valid] + diffusion_strength * update[result_valid]
            )

            progress = int(((iteration + 1) / iterations) * 100)
            feedback.setProgress(progress)

            self.log(
                f"Diffusion iteration {iteration + 1} of " f"{iterations} complete.",
                feedback,
            )

        return result

    def _load_layers_into_project(
        self,
        paths: dict,
        dem_stem: str,
        load_smoothed: bool,
        load_difference: bool,
        load_mask: bool,
        feedback: QgsProcessingFeedback,
    ) -> None:
        """
        Load selected Stage 2 outputs directly into the QGIS project.

        No layer group is created.

        :param paths: Output path dictionary.
        :param dem_stem: Input DEM stem.
        :param load_smoothed: Whether to load the smoothed DEM.
        :param load_difference: Whether to load the difference raster.
        :param load_mask: Whether to load the smoothing mask.
        :param feedback: QGIS processing feedback object.
        """
        try:
            from qgis.core import QgsProject, QgsRasterLayer

            project = QgsProject.instance()
            loaded_count = 0

            def load_raster(
                file_path: Path,
                layer_name: str,
            ) -> None:
                nonlocal loaded_count

                layer = QgsRasterLayer(
                    str(file_path),
                    layer_name,
                    "gdal",
                )

                if not layer.isValid():
                    self.log_warning(
                        f"Could not load output layer: {layer_name}",
                        feedback,
                    )
                    return

                project.addMapLayer(layer, True)
                loaded_count += 1

                self.log(
                    f"Loaded directly into project: {layer_name}",
                    feedback,
                )

            if load_smoothed:
                load_raster(
                    paths["smoothed"],
                    f"{dem_stem} - Smoothed DEM",
                )

            if load_difference:
                load_raster(
                    paths["difference"],
                    f"{dem_stem} - Smoothing Difference",
                )

            if load_mask:
                load_raster(
                    paths["mask"],
                    f"{dem_stem} - Smoothing Mask",
                )

            self.log(
                f"{loaded_count} Stage 2 output layer(s) loaded "
                "directly into the project.",
                feedback,
            )

        except Exception as error:
            self.log_warning(
                f"Could not load Stage 2 outputs: {error}",
                feedback,
            )

    def _write_text_report(
        self,
        paths: dict,
        provenance: dict,
        feedback: QgsProcessingFeedback,
    ) -> None:
        """
        Write the Stage 2 text report.

        :param paths: Output path dictionary.
        :param provenance: Provenance dictionary.
        :param feedback: QGIS processing feedback object.
        """
        statistics = provenance.get("statistics", {})
        warnings = provenance.get("warnings", [])
        parameters = provenance.get("parameters", {})

        lines = []
        append = lines.append

        append("=" * 70)
        append("MAYIM TOOLS - DEM HYDROLOGICAL SMOOTHING REPORT")
        append("=" * 70)
        append(f"Generated: {provenance.get('timestamp', '')}")
        append(f"Input DEM: {provenance.get('input_dem', '')}")
        append(f"Stage: {provenance.get('stage', 2)}")
        append("")

        append("-" * 70)
        append("PARAMETERS")
        append("-" * 70)
        append(f"Iterations: " f"{parameters.get('iterations', 'Unknown')}")
        append(
            f"Diffusion strength: " f"{parameters.get('diffusion_strength', 'Unknown')}"
        )
        append(f"Edge threshold: " f"{parameters.get('edge_threshold', 'Unknown')}")
        append(
            f"Resolution scaling: " f"{parameters.get('resolution_scale', 'Unknown')}"
        )
        append("")

        append("-" * 70)
        append("RESULTS")
        append("-" * 70)
        append(f"Valid cells: " f"{statistics.get('valid_cells', 0):,}")
        append(f"Changed cells: " f"{statistics.get('changed_cells', 0):,}")
        append(
            f"Changed percentage: " f"{statistics.get('changed_percentage', 0.0):.4f}%"
        )
        append(
            f"Mean absolute change: "
            f"{statistics.get('mean_absolute_change', 0.0):.6f}"
        )
        append(
            f"Maximum absolute change: "
            f"{statistics.get('maximum_absolute_change', 0.0):.6f}"
        )
        append(f"Minimum change: " f"{statistics.get('minimum_change', 0.0):.6f}")
        append(f"Maximum change: " f"{statistics.get('maximum_change', 0.0):.6f}")
        append("")

        append("-" * 70)
        append("OUTPUT FILES")
        append("-" * 70)
        append(f"Smoothed DEM: {paths['smoothed'].name}")
        append(f"Smoothing difference: " f"{paths['difference'].name}")
        append(f"Smoothing mask: {paths['mask'].name}")
        append(f"Report: {paths['report'].name}")
        append(f"Provenance: " f"{paths['provenance'].name}")
        append("")

        append("-" * 70)
        append("INTERPRETATION")
        append("-" * 70)
        append(
            "The smoothed DEM is a Stage 2 product. It has been processed "
            "using edge-preserving anisotropic diffusion."
        )
        append(
            "This tool does not fill, breach, classify, or otherwise "
            "resolve depressions."
        )
        append(
            "Review the smoothing difference and smoothing mask before "
            "using the output in subsequent hydrological stages."
        )
        append("")

        if warnings:
            append("-" * 70)
            append("WARNINGS")
            append("-" * 70)
            for warning in warnings:
                append(f"- {warning}")
            append("")

        append("-" * 70)
        append("REFERENCES")
        append("-" * 70)
        append(
            "Perona, P. and Malik, J. (1990). Scale-space and edge "
            "detection using anisotropic diffusion. IEEE Transactions "
            "on Pattern Analysis and Machine Intelligence, 12(7), "
            "629-639."
        )
        append("")

        append("Mayim Tools DEM Hydrological Conditioning research paper.")
        append("Stage 2 - Controlled Smoothing.")
        append("")

        append("-" * 70)
        append("REPORT LOCATION")
        append("-" * 70)
        append(str(Path(paths["report"]).resolve()))
        append("")

        append("=" * 70)
        append("End of DEM Hydrological Smoothing report")
        append("=" * 70)

        report_text = "\n".join(lines)

        with open(
            paths["report"],
            "w",
            encoding="utf-8",
        ) as file:
            file.write(report_text)

        self.log(
            f"Smoothing report written: " f"{Path(paths['report']).resolve()}",
            feedback,
        )
