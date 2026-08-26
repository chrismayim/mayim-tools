"""
Mayim Tools - DEM Hydrological Smoothing
=========================================

Implements Stage 2 of the Mayim Tools DEM hydrological-conditioning
pipeline: controlled, edge-preserving smoothing.

This tool is the Mayim implementation of the TerraCorrect Stage 2
component described in Section 6.2 of the Mayim Tools DEM
Hydrological Conditioning Research Paper (Rev 1, August 2026).

Algorithm
---------
Perona-Malik-style anisotropic diffusion (four-connected neighbours).

The method reduces high-frequency noise while limiting diffusion across
strong elevation gradients. Unlike a uniform low-pass filter, it does
not apply the same smoothing strength indiscriminately across the DEM,
and therefore avoids the well-documented risk of uniform smoothing
introducing new synthetic depressions and ridges.

Reference:
    Perona, P. and Malik, J. (1990).
    Scale-space and edge detection using anisotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7), 629-639.

IP Status
---------
Original Mayim implementation.
Algorithm implemented solely from the published paper above.
No WhiteboxTools, RichDEM or any other hydrological package consulted.
External dependencies: rasterio (MIT), numpy (BSD), QGIS API (GPL-2+).
MayimManifest: original Mayim IP, standard library only.

This tool does not perform:
    - Void interpolation (Stage 0).
    - Artifact classification (Stage 1).
    - Depression filling or breaching (Stage 5).
    - Flow-direction calculation.
    - Flow-accumulation calculation.

Those operations belong to other pipeline stages.

Pipeline position
-----------------
    Stage 0  Ingestion and QA           <- DEM Hydrological Screening
    Stage 1  Artifact Correction        <- DEM Hydrological Screening
    Stage 2  Controlled Smoothing       <- THIS TOOL (optional)
    Stage 3  Depression Delineation     <- DEM Depression Analysis
    Stage 4  Depression Classification  <- DEM Depression Analysis
    Stage 5  Selective Enforcement      <- DEM Hydrological Filling
    Stage 6  Flat Resolution            <- DEM Hydrological Filling
    Stage 7  Hydrography Enforcement    <- DEM Hydrography Enforcement
    Stage 8  Validation and Provenance  <- DEM Pipeline Audit
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
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
)

from mayim_tools.contract import MayimManifest
from mayim_tools.core.logger import MayimLogger
from mayim_tools.core.validation_utils import ValidationUtils
from mayim_tools.processing.algorithms.base_algorithm import MayimBaseAlgorithm


class DEMHydrologicalSmoothing(MayimBaseAlgorithm):
    """
    QGIS Processing algorithm for Stage 2 controlled DEM smoothing.

    Applies Perona-Malik anisotropic diffusion to reduce high-frequency
    noise while preserving stronger terrain edges. The input DEM is
    never overwritten. A MayimManifest is produced alongside all raster
    outputs to support pipeline chain-of-custody.
    """

    # ── Parameter identifiers ──────────────────────────────────────── #
    PARAM_DEM              = "INPUT_DEM"
    PARAM_MANIFEST         = "INPUT_MANIFEST"
    PARAM_ITERATIONS       = "ITERATIONS"
    PARAM_DIFFUSION        = "DIFFUSION_STRENGTH"
    PARAM_EDGE_THRESHOLD   = "EDGE_THRESHOLD"
    PARAM_RESOLUTION_SCALE = "RESOLUTION_SCALE"
    PARAM_OUTPUT_FOLDER    = "OUTPUT_FOLDER"
    PARAM_LOAD_SMOOTHED    = "LOAD_SMOOTHED_DEM"
    PARAM_LOAD_DIFFERENCE  = "LOAD_DIFFERENCE"
    PARAM_LOAD_MASK        = "LOAD_SMOOTHING_MASK"

    # ── Output identifiers ─────────────────────────────────────────── #
    OUTPUT_SMOOTHED    = "OUTPUT_SMOOTHED"
    OUTPUT_DIFFERENCE  = "OUTPUT_DIFFERENCE"
    OUTPUT_MASK        = "OUTPUT_MASK"
    OUTPUT_REPORT      = "OUTPUT_REPORT"
    OUTPUT_PROVENANCE  = "OUTPUT_PROVENANCE"

    # ── Tool version — update on every release ─────────────────────── #
    TOOL_VERSION = "dem-hydrological-smoothing-0.2.0"

    # ── MayimBaseAlgorithm interface ───────────────────────────────── #

    def name(self) -> str:  # noqa: N802
        """Return the unique Processing algorithm identifier."""
        return "demhydrologicalsmoothing"

    def displayName(self) -> str:  # noqa: N802
        """Return the human-readable algorithm name."""
        return "DEM Hydrological Smoothing"

    def group(self) -> str:  # noqa: N802
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
            "<b>DEM Hydrological Smoothing</b> — Stage 2<br><br>"
            "Implements Stage 2 controlled smoothing using an "
            "edge-preserving Perona-Malik anisotropic diffusion "
            "filter.<br><br>"
            "<b>Use this tool only after reviewing the Stage 1 "
            "screening report and artifact mask.</b> The input "
            "DEM is never overwritten.<br><br>"
            "<b>Recommended starting values:</b><br>"
            "<ul>"
            "<li><b>Diffusion iterations:</b> 5 "
            "(valid range: 1-100). Use 3-5 for light noise; "
            "5-10 for persistent high-frequency noise.</li>"
            "<li><b>Diffusion strength:</b> 0.20 "
            "(valid range: 0.01-0.25).</li>"
            "<li><b>Edge threshold:</b> 1.0 elevation unit. "
            "Lower values preserve stronger terrain breaks.</li>"
            "</ul><br>"
            "<b>Outputs:</b> smoothed DEM, signed difference "
            "raster, smoothing mask, text report, JSON provenance "
            "record, and a MayimManifest for pipeline "
            "chain-of-custody.<br><br>"
            "<b>IP:</b> Perona-Malik algorithm implemented from "
            "the published paper (IEEE TPAMI, 1990). No "
            "WhiteboxTools or RichDEM used at runtime.<br><br>"
            "<b>Reference:</b> Perona and Malik (1990). "
            "Scale-space and edge detection using anisotropic "
            "diffusion. IEEE TPAMI, 12(7), 629-639."
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
            "perona-malik",
        ]

    # ── Parameter definition ───────────────────────────────────────── #

    def initAlgorithm(self, config=None) -> None:  # noqa: N802
        """Define all Processing Toolbox parameters and outputs."""

        # ── Required: input DEM ────────────────────────────────────── #
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.PARAM_DEM,
                "Input DEM",
            )
        )

        # ── Optional: input manifest from previous tool ────────────── #
        param_manifest = QgsProcessingParameterFile(
            self.PARAM_MANIFEST,
            "Input MayimManifest from previous tool (optional)",
            behavior=QgsProcessingParameterFile.File,
            extension="json",
            optional=True,
        )
        self.addParameter(param_manifest)

        # ── Diffusion iterations ───────────────────────────────────── #
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_ITERATIONS,
                "Diffusion iterations "
                "(recommended: 5; valid range: 1-100)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1,
                maxValue=100,
            )
        )

        # ── Diffusion strength ─────────────────────────────────────── #
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_DIFFUSION,
                "Diffusion strength "
                "(recommended: 0.20; valid range: 0.01-0.25)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.20,
                minValue=0.01,
                maxValue=0.25,
            )
        )

        # ── Edge threshold ─────────────────────────────────────────── #
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

        # ── Resolution scaling ─────────────────────────────────────── #
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_RESOLUTION_SCALE,
                "Scale diffusion strength according to DEM resolution",
                defaultValue=True,
            )
        )

        # ── Output folder ──────────────────────────────────────────── #
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.PARAM_OUTPUT_FOLDER,
                "Output folder",
            )
        )

        # ── Load options ───────────────────────────────────────────── #
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

        # ── Declared outputs ───────────────────────────────────────── #
        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_SMOOTHED,
                "Smoothed DEM",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_DIFFERENCE,
                "Smoothing difference raster",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
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

    # ── Main processing method ─────────────────────────────────────── #

    def processAlgorithm(  # noqa: N802
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """
        Execute Stage 2 controlled smoothing.

        Reads the input DEM, applies Perona-Malik anisotropic
        diffusion, writes all outputs, produces a MayimManifest
        for pipeline chain-of-custody, and optionally loads result
        layers into the QGIS project.

        IP: Perona-Malik algorithm implemented from the published
        paper only. No WhiteboxTools or RichDEM at runtime.

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

        # ── Read parameters ────────────────────────────────────────── #

        dem_layer = self.parameterAsRasterLayer(
            parameters, self.PARAM_DEM, context
        )

        if not ValidationUtils.is_valid_raster_layer(dem_layer):
            raise QgsProcessingException(
                "The input DEM is missing or invalid."
            )

        input_manifest_path = self.parameterAsString(
            parameters, self.PARAM_MANIFEST, context
        )

        iterations = self.parameterAsInt(
            parameters, self.PARAM_ITERATIONS, context
        )

        diffusion_strength = self.parameterAsDouble(
            parameters, self.PARAM_DIFFUSION, context
        )

        edge_threshold = self.parameterAsDouble(
            parameters, self.PARAM_EDGE_THRESHOLD, context
        )

        resolution_scale = self.parameterAsBoolean(
            parameters, self.PARAM_RESOLUTION_SCALE, context
        )

        output_folder = self.parameterAsString(
            parameters, self.PARAM_OUTPUT_FOLDER, context
        )

        load_smoothed = self.parameterAsBoolean(
            parameters, self.PARAM_LOAD_SMOOTHED, context
        )

        load_difference = self.parameterAsBoolean(
            parameters, self.PARAM_LOAD_DIFFERENCE, context
        )

        load_mask = self.parameterAsBoolean(
            parameters, self.PARAM_LOAD_MASK, context
        )

        # ── Validate parameters ────────────────────────────────────── #

        if iterations < 1:
            raise QgsProcessingException(
                "Diffusion iterations must be at least 1."
            )

        if not (0.01 <= diffusion_strength <= 0.25):
            raise QgsProcessingException(
                "Diffusion strength must be between 0.01 and 0.25."
            )

        if edge_threshold <= 0:
            raise QgsProcessingException(
                "Edge threshold must be greater than zero."
            )

        # ── Handle temporary output folder ─────────────────────────── #

        if not output_folder or output_folder == "TEMPORARY_OUTPUT":
            import tempfile
            output_folder = tempfile.mkdtemp(
                prefix="mayim_smoothing_"
            )
            self.log(
                f"Using temporary folder: {output_folder}",
                feedback,
            )

        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        dem_stem = Path(dem_layer.source()).stem

        # ── Set up output paths ────────────────────────────────────── #

        paths = {
            "smoothed":    output_dir / f"{dem_stem}_smoothed.tif",
            "difference":  output_dir / f"{dem_stem}_smoothing_difference.tif",
            "mask":        output_dir / f"{dem_stem}_smoothing_mask.tif",
            "report":      output_dir / f"{dem_stem}_smoothing_report.txt",
            "provenance":  output_dir / f"{dem_stem}_smoothing_provenance.json",
            "manifest":    output_dir / f"{dem_stem}_smoothed.manifest.json",
        }

        # ── Read input manifest if supplied ────────────────────────── #

        input_manifest = None

        if input_manifest_path and Path(input_manifest_path).exists():
            try:
                input_manifest = MayimManifest.read(input_manifest_path)
                errors = input_manifest.validate()
                if errors:
                    self.log_warning(
                        f"Input manifest validation issues: "
                        f"{'; '.join(errors)}",
                        feedback,
                    )
                else:
                    self.log(
                        f"Input manifest loaded: "
                        f"{input_manifest.summary()}",
                        feedback,
                    )
            except Exception as _e:
                self.log_warning(
                    f"Could not read input manifest: {_e}. "
                    f"A new manifest will be created.",
                    feedback,
                )

        # ── Initialise provenance record ───────────────────────────── #

        provenance = {
            "tool":          "DEM Hydrological Smoothing",
            "algorithm":     "Perona-Malik anisotropic diffusion",
            "algorithm_ref": (
                "Perona, P. and Malik, J. (1990). Scale-space and "
                "edge detection using anisotropic diffusion. "
                "IEEE TPAMI, 12(7), 629-639."
            ),
            "ip_status": (
                "Original Mayim implementation. Algorithm implemented "
                "solely from the published paper. No WhiteboxTools or "
                "RichDEM source consulted."
            ),
            "version":       self.TOOL_VERSION,
            "stage":         2,
            "timestamp":     datetime.now().isoformat(),
            "input_dem":     dem_layer.source(),
            "input_manifest": input_manifest_path or None,
            "parameters": {
                "iterations":        iterations,
                "diffusion_strength": diffusion_strength,
                "edge_threshold":    edge_threshold,
                "resolution_scale":  resolution_scale,
            },
            "outputs":   {k: str(v) for k, v in paths.items()},
            "statistics": {},
            "warnings":  [],
        }

        # ── Log run header ─────────────────────────────────────────── #

        self.log("=" * 60, feedback)
        self.log("STAGE 2 - CONTROLLED DEM SMOOTHING", feedback)
        self.log("=" * 60, feedback)
        self.log(f"Input DEM   : {dem_layer.source()}", feedback)
        self.log(
            f"Iterations  : {iterations} "
            "(recommended: 5; valid range: 1-100)",
            feedback,
        )
        self.log(
            f"Strength    : {diffusion_strength} "
            "(recommended: 0.20; valid range: 0.01-0.25)",
            feedback,
        )
        self.log(
            f"Edge thresh : {edge_threshold} "
            "(recommended: 1.0 elevation unit)",
            feedback,
        )
        self.log(
            "Review the smoothing difference and mask before "
            "using the smoothed DEM downstream.",
            feedback,
        )

        feedback.setProgress(5)

        # ══════════════════════════════════════════════════════════════
        # OPEN DEM AND APPLY SMOOTHING
        # ══════════════════════════════════════════════════════════════

        dem_path = dem_layer.source()

        try:
            with rasterio.open(dem_path) as source:

                if source.count < 1:
                    raise QgsProcessingException(
                        "The input raster contains no valid bands."
                    )

                # ── CRS check ──────────────────────────────────────── #
                if source.crs is None:
                    warning = (
                        "The input DEM has no assigned CRS. Smoothing "
                        "can proceed because it is a local cell "
                        "operation, but the output must not be used for "
                        "distance-dependent calculations until a CRS is "
                        "assigned."
                    )
                    provenance["warnings"].append(warning)
                    self.log_warning(warning, feedback)

                profile  = source.profile.copy()
                original = source.read(1).astype(np.float64)
                nodata   = source.nodata
                height, width = original.shape

                res_x = abs(float(source.transform.a))
                res_y = abs(float(source.transform.e))
                mean_res = (res_x + res_y) / 2.0

                crs_string = (
                    source.crs.to_string()
                    if source.crs is not None
                    else "Unknown"
                )

                # ── Build valid-cell mask ──────────────────────────── #
                if nodata is not None:
                    valid_mask = (original != nodata) & np.isfinite(original)
                else:
                    valid_mask = np.isfinite(original)

                if not np.any(valid_mask):
                    raise QgsProcessingException(
                        "The input DEM contains no valid elevation cells."
                    )

                valid_values = original[valid_mask]
                mean_elevation   = float(np.mean(valid_values))
                median_elevation = float(np.median(valid_values))

                provenance["statistics"].update(
                    {
                        "crs":             crs_string,
                        "resolution_x":    res_x,
                        "resolution_y":    res_y,
                        "mean_resolution": mean_res,
                        "width":           width,
                        "height":          height,
                        "nodata":          str(nodata),
                        "dtype":           str(source.dtypes[0]),
                        "valid_cells":     int(np.sum(valid_mask)),
                        "mean_elevation":  mean_elevation,
                        "median_elevation": median_elevation,
                    }
                )

                # ── Resolution scaling ─────────────────────────────── #
                if resolution_scale:
                    scale_factor = self._resolution_scale_factor(
                        mean_res
                    )
                else:
                    scale_factor = 1.0

                effective_diffusion = diffusion_strength * scale_factor

                if effective_diffusion > 0.25:
                    effective_diffusion = 0.25
                    warning = (
                        "Resolution scaling clamped effective diffusion "
                        "strength to the numerical stability limit of 0.25."
                    )
                    provenance["warnings"].append(warning)
                    self.log_warning(warning, feedback)

                provenance["statistics"].update(
                    {
                        "resolution_scale_factor": scale_factor,
                        "effective_diffusion":     effective_diffusion,
                    }
                )

                self.log(
                    f"Effective diffusion: {effective_diffusion:.6f} "
                    f"(scale factor: {scale_factor:.4f})",
                    feedback,
                )

                feedback.setProgress(15)

                if self.is_cancelled(feedback):
                    return {}

                # ── Apply anisotropic diffusion ────────────────────── #
                # Algorithm: Perona-Malik (1990).
                # Implemented from the published paper only.
                # No WhiteboxTools or RichDEM source consulted.

                working = original.copy()
                working[~valid_mask] = np.nan

                self.log(
                    "Applying Perona-Malik anisotropic diffusion...",
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

                # Restore original NoData cells — never invent values
                smoothed[~valid_mask] = original[~valid_mask]

                # ── Compute difference and modification mask ───────── #
                difference = np.zeros_like(original, dtype=np.float64)
                difference[valid_mask] = (
                    smoothed[valid_mask] - original[valid_mask]
                )

                change_tolerance = max(
                    np.finfo(np.float64).eps * 100.0,
                    abs(edge_threshold) * 1e-9,
                )

                smoothing_mask = np.zeros(
                    original.shape, dtype=np.uint8
                )
                smoothing_mask[
                    valid_mask
                    & (np.abs(difference) > change_tolerance)
                ] = 1

                changed_cells   = int(np.sum(smoothing_mask == 1))
                valid_cell_count = int(np.sum(valid_mask))

                changed_pct = (
                    (changed_cells / valid_cell_count) * 100.0
                    if valid_cell_count > 0
                    else 0.0
                )

                valid_diff = difference[valid_mask]

                provenance["statistics"].update(
                    {
                        "changed_cells":           changed_cells,
                        "changed_percentage":      round(changed_pct, 6),
                        "mean_absolute_change":    float(
                            np.mean(np.abs(valid_diff))
                        ),
                        "max_absolute_change":     float(
                            np.max(np.abs(valid_diff))
                        ),
                        "minimum_change":          float(
                            np.min(valid_diff)
                        ),
                        "maximum_change":          float(
                            np.max(valid_diff)
                        ),
                    }
                )

                self.log(
                    f"Changed cells: {changed_cells:,} "
                    f"({changed_pct:.4f}% of valid cells)",
                    feedback,
                )

                # ── Warn if high modification rate ─────────────────── #
                if changed_pct > 25.0:
                    warning = (
                        f"{changed_pct:.1f}% of valid cells were "
                        "modified. This is unusually high. Review the "
                        "difference raster and consider reducing "
                        "iterations or diffusion strength."
                    )
                    provenance["warnings"].append(warning)
                    self.log_warning(warning, feedback)
                elif changed_pct > 10.0:
                    warning = (
                        f"{changed_pct:.1f}% of valid cells were "
                        "modified. Carefully review the smoothing "
                        "outputs before proceeding."
                    )
                    provenance["warnings"].append(warning)
                    self.log_warning(warning, feedback)

                feedback.setProgress(80)

                # ── Write smoothed DEM ─────────────────────────────── #
                smoothed_profile = profile.copy()
                smoothed_profile.update(
                    dtype="float32",
                    count=1,
                    compress="lzw",
                )

                if nodata is None:
                    smoothed_profile["nodata"] = -9999.0
                    smoothed_out = smoothed.astype(np.float32)
                    smoothed_out[~valid_mask] = -9999.0
                else:
                    smoothed_profile["nodata"] = nodata
                    smoothed_out = smoothed.astype(np.float32)
                    smoothed_out[~valid_mask] = nodata

                with rasterio.open(
                    paths["smoothed"], "w", **smoothed_profile
                ) as dst:
                    dst.write(smoothed_out, 1)

                # ── Write difference raster ────────────────────────── #
                diff_profile = profile.copy()
                diff_profile.update(
                    dtype="float32",
                    count=1,
                    compress="lzw",
                    nodata=-9999.0,
                )

                diff_out = difference.astype(np.float32)
                diff_out[~valid_mask] = -9999.0

                with rasterio.open(
                    paths["difference"], "w", **diff_profile
                ) as dst:
                    dst.write(diff_out, 1)

                # ── Write smoothing mask ───────────────────────────── #
                mask_profile = profile.copy()
                mask_profile.update(
                    dtype="uint8",
                    count=1,
                    compress="lzw",
                    nodata=255,
                )

                mask_out = smoothing_mask.copy()
                mask_out[~valid_mask] = 255

                with rasterio.open(
                    paths["mask"], "w", **mask_profile
                ) as dst:
                    dst.write(mask_out, 1)

                self.log(
                    "Stage 2 raster outputs written.",
                    feedback,
                )

                feedback.setProgress(88)

        except QgsProcessingException:
            raise

        except QgsProcessingException:
            raise

        except Exception as error:
            MayimLogger.critical(
                f"DEM Hydrological Smoothing failed: {error}"
            )
            raise QgsProcessingException(
                f"DEM Hydrological Smoothing failed: {error}"
            ) from error

        # ══════════════════════════════════════════════════════════════
        # WRITE REPORT AND PROVENANCE
        # ══════════════════════════════════════════════════════════════

        self._write_text_report(
            paths=paths,
            provenance=provenance,
            feedback=feedback,
        )

        with open(
            paths["provenance"], "w", encoding="utf-8"
        ) as f:
            json.dump(provenance, f, indent=4, default=str)

        report_path = Path(paths["report"]).resolve()
        report_uri  = report_path.as_uri()

        self.log(
            f"Stage 2 report    : {report_path}",
            feedback,
        )
        feedback.pushInfo(
            f"Open Stage 2 report: {report_uri}"
        )

        feedback.setProgress(92)

        # ══════════════════════════════════════════════════════════════
        # WRITE MAYIMMANIFEST
        # ══════════════════════════════════════════════════════════════

        try:
            if input_manifest is not None:
                # Derive from the input manifest — preserves the full
                # chain of custody back to Stage 0.
                manifest = input_manifest.derive(
                    produced_by=self.TOOL_VERSION,
                    raster_path=str(paths["smoothed"]),
                    stage=2,
                    audit_log_path=str(paths["provenance"]),
                    warnings=(
                        provenance["warnings"]
                        if provenance["warnings"]
                        else None
                    ),
                    width=provenance["statistics"].get("width"),
                    height=provenance["statistics"].get("height"),
                    dtype=provenance["statistics"].get("dtype"),
                )
            else:
                # No input manifest — create a fresh one.
                manifest = MayimManifest.create(
                    raster_path=str(paths["smoothed"]),
                    crs=provenance["statistics"].get("crs", "Unknown"),
                    cell_size=float(
                        provenance["statistics"].get("mean_resolution", 0.0)
                    ),
                    vertical_accuracy=5.0,
                    nodata=float(
                        provenance["statistics"].get("nodata", -9999.0)
                        if provenance["statistics"].get("nodata") != "None"
                        else -9999.0
                    ),
                    produced_by=self.TOOL_VERSION,
                    stage=2,
                    audit_log_path=str(paths["provenance"]),
                    warnings=(
                        provenance["warnings"]
                        if provenance["warnings"]
                        else None
                    ),
                    width=provenance["statistics"].get("width"),
                    height=provenance["statistics"].get("height"),
                    dtype=provenance["statistics"].get("dtype"),
                )

            manifest.write(str(paths["manifest"]))

            self.log(
                f"MayimManifest     : {paths['manifest'].name}",
                feedback,
            )

        except Exception as _manifest_error:
            self.log_warning(
                f"Could not write MayimManifest: {_manifest_error}",
                feedback,
            )

        feedback.setProgress(95)

        # ══════════════════════════════════════════════════════════════
        # LOAD SELECTED OUTPUTS INTO QGIS PROJECT
        # ══════════════════════════════════════════════════════════════

        self._load_layers_into_project(
            paths=paths,
            dem_stem=dem_stem,
            load_smoothed=load_smoothed,
            load_difference=load_difference,
            load_mask=load_mask,
            feedback=feedback,
        )

        feedback.setProgress(100)

        self.log("=" * 60, feedback)
        self.log("STAGE 2 COMPLETE", feedback)
        self.log("=" * 60, feedback)
        self.log(
            f"All outputs written to: {output_folder}",
            feedback,
        )

        return {
            self.OUTPUT_SMOOTHED:   str(paths["smoothed"]),
            self.OUTPUT_DIFFERENCE: str(paths["difference"]),
            self.OUTPUT_MASK:       str(paths["mask"]),
            self.OUTPUT_REPORT:     str(paths["report"]),
            self.OUTPUT_PROVENANCE: str(paths["provenance"]),
        }

    # ── Private helper methods ─────────────────────────────────────── #

    def _resolution_scale_factor(self, resolution: float) -> float:
        """
        Calculate a conservative resolution scaling factor.

        Reference resolution is 1 metre. Coarser DEMs receive a lower
        factor because a coarse cell represents a larger ground area
        and generally requires less aggressive per-cell diffusion.

        The result is bounded to the range [0.1, 1.0] to avoid
        unstable or negligible behaviour.

        :param resolution: Mean DEM cell size in map units.
        :returns: Resolution scaling factor in [0.1, 1.0].
        """
        if resolution <= 0:
            return 1.0
        factor = 1.0 / resolution
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
        Apply Perona-Malik anisotropic diffusion.

        Implements the four-connected-neighbour formulation of the
        anisotropic diffusion algorithm published in:

            Perona, P. and Malik, J. (1990).
            Scale-space and edge detection using anisotropic diffusion.
            IEEE Transactions on Pattern Analysis and Machine
            Intelligence, 12(7), 629-639.

        This implementation was written solely from the published paper.
        No WhiteboxTools, RichDEM or any other third-party hydrological
        source was consulted by the implementing engineer.

        The conductance function used is the Gaussian form from
        equation (13) of the paper:

            c(x, y, t) = exp(-(|grad I| / kappa)^2)

        where kappa is the edge_threshold parameter.

        Diffusion is reduced near strong elevation gradients, preserving
        terrain edges more effectively than a uniform low-pass filter.

        :param array: DEM array with NaN in invalid cells.
        :param valid_mask: Boolean mask — True where cells are valid.
        :param iterations: Number of diffusion iterations.
        :param diffusion_strength: Update strength per iteration (lambda
            in the paper). Must be <= 0.25 for numerical stability of
            the explicit four-connected scheme.
        :param edge_threshold: Gradient magnitude threshold controlling
            edge preservation (kappa in the paper). Expressed in the
            elevation units of the DEM.
        :param feedback: QGIS feedback object for cancellation checks
            and progress reporting.
        :returns: Smoothed DEM array. NaN cells are unchanged.
        """
        result = array.copy()

        for iteration in range(iterations):

            if self.is_cancelled(feedback):
                return result

            # ── Compute gradients to four-connected neighbours ─────── #
            north = np.full_like(result, np.nan)
            south = np.full_like(result, np.nan)
            west  = np.full_like(result, np.nan)
            east  = np.full_like(result, np.nan)

            north[1:,  :]  = result[:-1, :]
            south[:-1, :]  = result[1:,  :]
            west[ :,  1:]  = result[:,  :-1]
            east[ :, :-1]  = result[:,   1:]

            # ── Valid-neighbour masks ──────────────────────────────── #
            valid_north = np.zeros_like(valid_mask)
            valid_south = np.zeros_like(valid_mask)
            valid_west  = np.zeros_like(valid_mask)
            valid_east  = np.zeros_like(valid_mask)

            valid_north[1:,  :] = valid_mask[:-1, :] & valid_mask[1:,  :]
            valid_south[:-1, :] = valid_mask[1:,  :] & valid_mask[:-1, :]
            valid_west[ :,  1:] = valid_mask[:,  :-1] & valid_mask[:,   1:]
            valid_east[ :, :-1] = valid_mask[:,   1:] & valid_mask[:,  :-1]

            result_valid = valid_mask & np.isfinite(result)
            update = np.zeros_like(result)

            kappa_sq = max(edge_threshold, 1e-12) ** 2

            for neighbour, neighbour_valid in (
                (north, valid_north),
                (south, valid_south),
                (west,  valid_west),
                (east,  valid_east),
            ):
                gradient = neighbour - result

                # Gaussian conductance function — eq. (13), Perona &
                # Malik (1990):
                #   c = exp(-(|grad|^2) / kappa^2)
                conductance = np.exp(
                    -(gradient ** 2) / kappa_sq
                )

                contribution = np.where(
                    neighbour_valid & np.isfinite(gradient),
                    conductance * gradient,
                    0.0,
                )

                update += contribution

            # ── Update valid cells ─────────────────────────────────── #
            result[result_valid] = (
                result[result_valid]
                + diffusion_strength * update[result_valid]
            )

            progress = int(((iteration + 1) / iterations) * 60) + 15
            feedback.setProgress(progress)

            self.log(
                f"Diffusion iteration {iteration + 1} of "
                f"{iterations} complete.",
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

        No layer group is created. Outputs are added directly to the
        project layer tree.

        :param paths: Output path dictionary.
        :param dem_stem: Input DEM filename stem.
        :param load_smoothed: Whether to load the smoothed DEM.
        :param load_difference: Whether to load the difference raster.
        :param load_mask: Whether to load the smoothing mask.
        :param feedback: QGIS processing feedback object.
        """
        try:
            from qgis.core import QgsProject, QgsRasterLayer

            project = QgsProject.instance()
            loaded  = 0

            def load_raster(file_path: Path, layer_name: str) -> None:
                nonlocal loaded
                layer = QgsRasterLayer(
                    str(file_path), layer_name, "gdal"
                )
                if not layer.isValid():
                    self.log_warning(
                        f"Could not load layer: {layer_name}",
                        feedback,
                    )
                    return
                project.addMapLayer(layer, True)
                loaded += 1
                self.log(
                    f"Loaded into project: {layer_name}",
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
                f"{loaded} Stage 2 output layer(s) loaded into project.",
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
        Write the Stage 2 human-readable text report.

        The report summarises the input DEM, parameters, algorithm,
        processing statistics, warnings, output files and recommended
        next steps.

        :param paths: Output path dictionary.
        :param provenance: Provenance record dictionary.
        :param feedback: QGIS processing feedback object.
        """
        stats  = provenance.get("statistics", {})
        params = provenance.get("parameters", {})
        warns  = provenance.get("warnings",   [])

        lines  = []
        a      = lines.append

        a("=" * 70)
        a("MAYIM TOOLS - DEM HYDROLOGICAL SMOOTHING REPORT")
        a("=" * 70)
        a(f"Generated  : {provenance.get('timestamp', '')}")
        a(f"Tool       : {provenance.get('tool', '')}")
        a(f"Version    : {provenance.get('version', '')}")
        a(f"Stage      : {provenance.get('stage', 2)}")
        a(f"Algorithm  : {provenance.get('algorithm', '')}")
        a(f"Reference  : {provenance.get('algorithm_ref', '')}")
        a(f"IP status  : {provenance.get('ip_status', '')}")
        a(f"Input DEM  : {provenance.get('input_dem', '')}")
        a(f"Manifest in: {provenance.get('input_manifest', 'None')}")
        a("")

        a("-" * 70)
        a("PARAMETERS")
        a("-" * 70)
        a(
            f"Iterations         : "
            f"{params.get('iterations', 'Unknown')} "
            "(recommended: 5; valid range: 1-100)"
        )
        a(
            f"Diffusion strength : "
            f"{params.get('diffusion_strength', 'Unknown')} "
            "(recommended: 0.20; valid range: 0.01-0.25)"
        )
        a(
            f"Edge threshold     : "
            f"{params.get('edge_threshold', 'Unknown')} "
            "elevation units (recommended: 1.0)"
        )
        a(
            f"Resolution scaling : "
            f"{params.get('resolution_scale', 'Unknown')}"
        )
        a("")

        a("-" * 70)
        a("INPUT DEM METADATA")
        a("-" * 70)
        a(f"CRS                : {stats.get('crs', 'Unknown')}")
        a(f"Resolution X       : {stats.get('resolution_x', 'Unknown')}")
        a(f"Resolution Y       : {stats.get('resolution_y', 'Unknown')}")
        a(f"Mean resolution    : {stats.get('mean_resolution', 'Unknown')}")
        a(f"Width              : {stats.get('width', 'Unknown')} cells")
        a(f"Height             : {stats.get('height', 'Unknown')} cells")
        a(f"Data type          : {stats.get('dtype', 'Unknown')}")
        a(f"NoData value       : {stats.get('nodata', 'Unknown')}")
        a(f"Valid cells        : {stats.get('valid_cells', 0):,}")
        a(f"Mean elevation     : {stats.get('mean_elevation', 0.0):.4f}")
        a(f"Median elevation   : {stats.get('median_elevation', 0.0):.4f}")
        a("")

        a("-" * 70)
        a("DIFFUSION PARAMETERS (DERIVED)")
        a("-" * 70)
        a(
            f"Scale factor       : "
            f"{stats.get('resolution_scale_factor', 1.0):.6f}"
        )
        a(
            f"Effective diffusion: "
            f"{stats.get('effective_diffusion', 0.0):.6f}"
        )
        a("")

        a("-" * 70)
        a("SMOOTHING RESULTS")
        a("-" * 70)
        a(f"Changed cells      : {stats.get('changed_cells', 0):,}")
        a(
            f"Changed percentage : "
            f"{stats.get('changed_percentage', 0.0):.4f}%"
        )
        a(
            f"Mean abs change    : "
            f"{stats.get('mean_absolute_change', 0.0):.6f}"
        )
        a(
            f"Max abs change     : "
            f"{stats.get('max_absolute_change', 0.0):.6f}"
        )
        a(
            f"Minimum change     : "
            f"{stats.get('minimum_change', 0.0):.6f}"
        )
        a(
            f"Maximum change     : "
            f"{stats.get('maximum_change', 0.0):.6f}"
        )
        a("")

        # ── Interpretation ─────────────────────────────────────────── #
        a("-" * 70)
        a("INTERPRETATION")
        a("-" * 70)
        changed_pct = stats.get("changed_percentage", 0.0)
        if changed_pct == 0.0:
            a("No cells were modified. The DEM is unchanged.")
        elif changed_pct < 1.0:
            a("Very low modification rate. DEM is largely unchanged.")
            a("Review the difference raster for spatial distribution.")
        elif changed_pct < 10.0:
            a("Low to moderate modification rate.")
            a("Review the difference raster and smoothing mask.")
        elif changed_pct < 25.0:
            a("Moderate modification rate.")
            a("Carefully review the difference raster and smoothing mask")
            a("before using the smoothed DEM in downstream processing.")
        else:
            a("HIGH modification rate. A large proportion of the DEM")
            a("was changed. Review the smoothing parameters carefully.")
            a("Consider reducing iterations or diffusion strength.")
        a("")

        a("-" * 70)
        a("DIFFERENCE RASTER CONVENTION")
        a("-" * 70)
        a("  smoothed value - original value")
        a("  Negative: smoothing lowered the cell elevation.")
        a("  Positive: smoothing raised the cell elevation.")
        a("  Zero    : no effective change.")
        a("")

        a("-" * 70)
        a("SMOOTHING MASK VALUES")
        a("-" * 70)
        a("  0   = Unchanged or negligible change.")
        a("  1   = Cell modified by smoothing.")
        a("  255 = NoData (outside valid area).")
        a("")

        # ── Warnings ───────────────────────────────────────────────── #
        if warns:
            a("-" * 70)
            a("WARNINGS")
            a("-" * 70)
            for i, w in enumerate(warns, 1):
                words = w.split()
                line  = f"  {i}. "
                for word in words:
                    if len(line) + len(word) + 1 > 68:
                        a(line)
                        line = "     " + word + " "
                    else:
                        line += word + " "
                if line.strip():
                    a(line)
            a("")

        # ── Output files ───────────────────────────────────────────── #
        a("-" * 70)
        a("OUTPUT FILES")
        a("-" * 70)
        a(f"Smoothed DEM       : {paths['smoothed'].name}")
        a(f"Difference raster  : {paths['difference'].name}")
        a(f"Smoothing mask     : {paths['mask'].name}")
        a(f"Report (this file) : {paths['report'].name}")
        a(f"Provenance (JSON)  : {paths['provenance'].name}")
        a(f"Manifest (JSON)    : {paths['manifest'].name}")
        a("")

        # ── Recommended next steps ─────────────────────────────────── #
        a("-" * 70)
        a("RECOMMENDED NEXT STEPS")
        a("-" * 70)
        if changed_pct > 25.0:
            a("1. [REQUIRED] Review the difference raster and smoothing")
            a("   mask before proceeding. The modification rate is")
            a("   unusually high.")
            a("2. Consider reducing iterations or diffusion strength")
            a("   and re-running this tool.")
        else:
            a("The smoothed DEM is a Stage 2 product. It has been")
            a("processed using edge-preserving anisotropic diffusion.")
            a("")
            a("This tool does not fill, breach, classify, or otherwise")
            a("resolve depressions. Proceed to:")
            a("  -> Stage 3: DEM Depression Analysis")
            a("     (delineation and classification of depressions)")
        a("")

        # ── References ─────────────────────────────────────────────── #
        a("-" * 70)
        a("REFERENCES")
        a("-" * 70)
        a(
            "Perona, P. and Malik, J. (1990). Scale-space and edge "
            "detection using anisotropic diffusion. IEEE Transactions "
            "on Pattern Analysis and Machine Intelligence, 12(7), "
            "629-639."
        )
        a("")
        a(
            "Mayim Tools DEM Hydrological Conditioning Research Paper "
            "(Rev 1, August 2026). Stage 2 - Controlled Smoothing."
        )
        a("")

        # ── IP statement ───────────────────────────────────────────── #
        a("-" * 70)
        a("IP STATEMENT")
        a("-" * 70)
        a(
            "The Perona-Malik anisotropic diffusion algorithm used in "
            "this tool is an original Mayim implementation, written "
            "solely from the published paper listed above. No "
            "WhiteboxTools, RichDEM or any other third-party "
            "hydrological source was consulted by the implementing "
            "engineer. Only generic, non-hydrological libraries "
            "(NumPy, rasterio) are used at runtime."
        )
        a("")

        # ── Report location ────────────────────────────────────────── #
        a("-" * 70)
        a("REPORT LOCATION")
        a("-" * 70)
        a(str(Path(paths["report"]).resolve()))
        a("")

        # ── Footer ─────────────────────────────────────────────────── #
        a("=" * 70)
        a("Mayim Tools - DEM Hydrological Smoothing")
        a("https://github.com/chrismayim/mayim-tools")
        a("Licence: GPL-2.0+")
        a("=" * 70)
        a("")

        # ── Write to file ──────────────────────────────────────────── #
        report_text = "\n".join(lines)

        with open(
            paths["report"], "w", encoding="utf-8"
        ) as f:
            f.write(report_text)

        self.log(
            f"Stage 2 report written: {paths['report'].name}",
            feedback,
        )
