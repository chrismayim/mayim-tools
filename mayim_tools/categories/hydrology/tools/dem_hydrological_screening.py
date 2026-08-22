# -*- coding: utf-8 -*-
"""
Mayim Tools — DEM Hydrological Screening
=========================================
Implements Stages 0 and 1 of the Mayim Tools DEM Hydrological
Conditioning pipeline:

  Stage 0 — Ingestion & QA:
    Reads and validates a DEM, extracts metadata, detects and
    classifies void (no-data) cells, checks CRS suitability,
    and assigns vertical accuracy estimates.

  Stage 1 — Artifact Screening:
    Classifies the DEM source type and applies a Median Absolute
    Deviation (MAD) local outlier filter to detect speckle,
    striping, and scan-line noise characteristic of SAR/InSAR
    sources. LiDAR ground filtering and bare-earth substitution
    are flagged as coming in future versions.

References:
    Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-flood:
        An optimal depression-filling and watershed-labeling
        algorithm for digital elevation models.
        Computers & Geosciences, 62, 117-127.
    Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013).
        An improved simple morphological filter for the terrain
        classification of airborne LIDAR data.
        ISPRS Journal of Photogrammetry and Remote Sensing,
        77, 21-30.
    Wang, L., & Liu, H. (2006). An efficient method for
        identifying and filling surface depressions in digital
        elevation models for hydrologic analysis and modelling.
        International Journal of Geographical Information
        Science, 20(2), 193-213.
    Hawker, L. et al. (2022). A 30 m global map of elevation
        with forests and buildings removed.
        Environmental Research Letters, 17(2), 024016.

Author:     Mayim Tools
Version:    0.1.0
License:    GPL-2.0+
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingOutputFile,
    QgsProcessingOutputRasterLayer,
)

from mayim_tools.core.logger import MayimLogger
from mayim_tools.core.validation_utils import ValidationUtils
from mayim_tools.processing.algorithms.base_algorithm import MayimBaseAlgorithm


# -- DEM Source Type Constants ---------------------------------------------- #

class DEMSourceType:
    """Enumeration of supported DEM source types."""
    AUTO_DETECT          = 0
    LIDAR_DTM            = 1
    LIDAR_DSM            = 2
    SRTM                 = 3
    COPERNICUS_GLO30     = 4
    FABDEM_FATHOMDEM     = 5
    AERIAL_PHOTOGRAMMETRY = 6
    UNKNOWN              = 7

    LABELS = [
        "Auto-detect",
        "LiDAR bare-earth (DTM)",
        "LiDAR surface model (DSM, uncorrected)",
        "SRTM (30m / 90m)",
        "Copernicus GLO-30",
        "FABDEM / FathomDEM / GEDTM30",
        "Aerial photogrammetry",
        "Unknown / Other",
    ]

    # Default vertical accuracy RMSE (metres) per source type
    # Based on conservative estimates from the literature:
    # LiDAR DTM: ±0.1-0.3m, LiDAR DSM: ±0.3-0.5m
    # SRTM/Copernicus: ±3-5m, FABDEM: ±1-2m
    # Aerial: ±0.5-1.0m, Unknown: ±5m (conservative)
    DEFAULT_RMSE = {
        AUTO_DETECT:           None,
        LIDAR_DTM:             0.15,
        LIDAR_DSM:             0.40,
        SRTM:                  4.00,
        COPERNICUS_GLO30:      4.00,
        FABDEM_FATHOMDEM:      1.50,
        AERIAL_PHOTOGRAMMETRY: 0.75,
        UNKNOWN:               5.00,
    }


# -- Void Classification Constants ------------------------------------------ #

class VoidClass:
    """Void classification values written to the void mask raster."""
    VALID    = 0   # Valid data cell
    SMALL    = 1   # Small void — interpolated
    MEDIUM   = 2   # Medium void — flagged, not filled
    LARGE    = 3   # Large void — reported to analyst


# -- Main Tool Class -------------------------------------------------------- #

class DEMHydrologicalScreening(MayimBaseAlgorithm):
    """
    DEM Hydrological Screening tool.

    Implements Stages 0 (Ingestion & QA) and 1 (Artifact Screening)
    of the Mayim Tools DEM Hydrological Conditioning pipeline.

    This tool must be run before any depression processing, flow
    direction, or flow accumulation tools. Its outputs serve as
    the verified, quality-assured inputs for all subsequent
    conditioning steps.
    """

    # -- Parameter identifiers -------------------------------------------- #
    PARAM_DEM             = "INPUT_DEM"
    PARAM_SOURCE_TYPE     = "DEM_SOURCE_TYPE"
    PARAM_USER_RMSE       = "USER_RMSE"
    PARAM_SMALL_VOID      = "SMALL_VOID_THRESHOLD"
    PARAM_LARGE_VOID      = "LARGE_VOID_THRESHOLD"
    PARAM_MAD_WINDOW      = "MAD_WINDOW_SIZE"
    PARAM_MAD_THRESHOLD   = "MAD_THRESHOLD"
    PARAM_OUTPUT_FOLDER   = "OUTPUT_FOLDER"

    # -- Output identifiers ----------------------------------------------- #
    OUTPUT_DEM            = "OUTPUT_DEM"
    OUTPUT_VOID_MASK      = "OUTPUT_VOID_MASK"
    OUTPUT_ARTIFACT_MASK  = "OUTPUT_ARTIFACT_MASK"
    OUTPUT_QA_REPORT      = "OUTPUT_QA_REPORT"
    OUTPUT_PROVENANCE     = "OUTPUT_PROVENANCE"

    # -- MayimBaseAlgorithm interface ------------------------------------- #

    def name(self) -> str:
        return "demhydrologicalscreening"

    def displayName(self) -> str:
        return "DEM Hydrological Screening"

    def group(self) -> str:
        return "Hydrology Tools"

    def groupId(self) -> str:
        return "hydrology"

    def shortHelpString(self) -> str:
        return (
            "<b>DEM Hydrological Screening</b><br><br>"
            "Implements Stages 0 and 1 of the Mayim Tools DEM "
            "Hydrological Conditioning pipeline:<br><br>"
            "<b>Stage 0 — Ingestion &amp; QA:</b> Reads and validates "
            "the DEM, extracts metadata, detects and classifies void "
            "(no-data) cells, checks CRS suitability, and assigns "
            "vertical accuracy estimates.<br><br>"
            "<b>Stage 1 — Artifact Screening:</b> Classifies the DEM "
            "source type and applies a Median Absolute Deviation (MAD) "
            "local outlier filter to detect speckle, striping, and "
            "scan-line noise characteristic of SAR/InSAR sources.<br><br>"
            "<b>Run this tool first</b> before any depression "
            "processing or flow routing tools.<br><br>"
            "<b>References:</b> Barnes et al. (2014), "
            "Pingel et al. (2013), Wang &amp; Liu (2006), "
            "Hawker et al. (2022)"
        )

    def helpUrl(self) -> str:
        return "https://github.com/chrismayim/mayim-tools"

    def tags(self) -> list[str]:
        return [
            "dem", "hydrology", "screening", "qa", "quality",
            "artifact", "void", "conditioning", "mayim",
        ]

    def createInstance(self):
        return DEMHydrologicalScreening()

    # -- Parameter definition --------------------------------------------- #

    def initAlgorithm(self, config=None) -> None:
        """Define all input and output parameters."""

        # -- Input DEM --
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.PARAM_DEM,
                "Input DEM",
            )
        )

        # -- DEM Source Type --
        self.addParameter(
            QgsProcessingParameterEnum(
                self.PARAM_SOURCE_TYPE,
                "DEM Source Type",
                options=DEMSourceType.LABELS,
                defaultValue=DEMSourceType.AUTO_DETECT,
                optional=False,
            )
        )

        # -- Known vertical accuracy (optional override) --
        param_rmse = QgsProcessingParameterNumber(
            self.PARAM_USER_RMSE,
            "Known vertical accuracy RMSE (m) — leave -1 to use default",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=-1.0,
            minValue=-1.0,
            maxValue=100.0,
            optional=True,
        )
        self.addParameter(param_rmse)

        # -- Small void threshold --
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_SMALL_VOID,
                "Small void threshold (cells) — voids at or below "
                "this size will be interpolated",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
                minValue=1,
                maxValue=100,
            )
        )

        # -- Large void threshold --
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_LARGE_VOID,
                "Large void threshold (cells) — voids above this "
                "size will be reported to the analyst (not filled)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=100,
                minValue=10,
                maxValue=100000,
            )
        )

        # -- MAD filter window size --
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_MAD_WINDOW,
                "Artifact filter window size (cells, must be odd)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=3,
                maxValue=15,
            )
        )

        # -- MAD threshold (sigma) --
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_MAD_THRESHOLD,
                "Artifact detection threshold (sigma) — lower values "
                "flag more cells as suspected artifacts",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=3.0,
                minValue=1.0,
                maxValue=10.0,
            )
        )

        # -- Output folder --
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.PARAM_OUTPUT_FOLDER,
                "Output folder",
            )
        )

    # -- Main processing method ------------------------------------------- #

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """
        Execute the DEM Hydrological Screening tool.

        Runs Stage 0 (Ingestion & QA) and Stage 1 (Artifact Screening)
        sequentially, writing all outputs to the specified folder and
        returning output file paths to the QGIS Processing Framework.
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise QgsProcessingException(
                "The 'rasterio' library is required but not installed.\n"
                "Install it with: pip install rasterio"
            )

        try:
            from scipy.ndimage import label, generic_filter
        except ImportError:
            raise QgsProcessingException(
                "The 'scipy' library is required but not installed.\n"
                "Install it with: pip install scipy"
            )

        # -- Read parameters ---------------------------------------------- #

        dem_layer = self.parameterAsRasterLayer(
            parameters, self.PARAM_DEM, context
        )
        source_type_idx = self.parameterAsInt(
            parameters, self.PARAM_SOURCE_TYPE, context
        )
        user_rmse = self.parameterAsDouble(
            parameters, self.PARAM_USER_RMSE, context
        )
        small_void_thresh = self.parameterAsInt(
            parameters, self.PARAM_SMALL_VOID, context
        )
        large_void_thresh = self.parameterAsInt(
            parameters, self.PARAM_LARGE_VOID, context
        )
        mad_window = self.parameterAsInt(
            parameters, self.PARAM_MAD_WINDOW, context
        )
        mad_threshold = self.parameterAsDouble(
            parameters, self.PARAM_MAD_THRESHOLD, context
        )
        output_folder = self.parameterAsString(
            parameters, self.PARAM_OUTPUT_FOLDER, context
        )

        # -- Validate inputs ---------------------------------------------- #

        if not ValidationUtils.is_valid_raster_layer(dem_layer):
            raise QgsProcessingException(
                "Invalid or missing input DEM layer."
            )

        if small_void_thresh >= large_void_thresh:
            raise QgsProcessingException(
                f"Small void threshold ({small_void_thresh}) must be "
                f"less than large void threshold ({large_void_thresh})."
            )

        # Enforce odd MAD window size
        if mad_window % 2 == 0:
            mad_window += 1
            self.log(
                f"MAD window size adjusted to {mad_window} "
                f"(must be odd).", feedback
            )

        # -- Set up output paths ------------------------------------------ #

        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        dem_stem = Path(dem_layer.source()).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        paths = {
            "screened_dem":    output_dir / f"{dem_stem}_screened.tif",
            "void_mask":       output_dir / f"{dem_stem}_void_mask.tif",
            "artifact_mask":   output_dir / f"{dem_stem}_artifact_mask.tif",
            "qa_report_txt":   output_dir / f"{dem_stem}_screening_report.txt",
            "provenance":      output_dir / f"{dem_stem}_provenance.json",
        }

        # -- Initialise provenance log ------------------------------------ #

        provenance = {
            "tool":           "DEM Hydrological Screening",
            "version":        "0.1.0",
            "timestamp":      datetime.now().isoformat(),
            "input_dem":      dem_layer.source(),
            "parameters": {
                "source_type":       DEMSourceType.LABELS[source_type_idx],
                "user_rmse":         user_rmse,
                "small_void_thresh": small_void_thresh,
                "large_void_thresh": large_void_thresh,
                "mad_window":        mad_window,
                "mad_threshold":     mad_threshold,
            },
            "stage_0": {},
            "stage_1": {},
            "warnings":  [],
            "errors":    [],
        }

        # ==================================================================
        # STAGE 0 — INGESTION & QA
        # ==================================================================

        self.log("", feedback)
        self.log("=" * 60, feedback)
        self.log("  STAGE 0 — DEM INGESTION & QA", feedback)
        self.log("=" * 60, feedback)

        feedback.setProgress(5)
        if self.is_cancelled(feedback):
            return {}

        # -- Open DEM with rasterio --------------------------------------- #

        dem_path = dem_layer.source()

        try:
            with rasterio.open(dem_path) as src:

                # -- Extract metadata ------------------------------------- #
                crs        = src.crs
                transform  = src.transform
                res_x      = abs(transform.a)
                res_y      = abs(transform.e)
                nodata     = src.nodata
                width      = src.width
                height     = src.height
                dtype      = src.dtypes[0]
                bounds     = src.bounds
                band_count = src.count
                profile    = src.profile.copy()

                self.log(f"  DEM file    : {dem_path}", feedback)
                self.log(f"  Dimensions  : {width} x {height} cells", feedback)
                self.log(f"  Resolution  : {res_x:.4f} x {res_y:.4f} units", feedback)
                self.log(f"  Data type   : {dtype}", feedback)
                self.log(f"  Band count  : {band_count}", feedback)
                self.log(f"  NoData value: {nodata}", feedback)
                self.log(f"  Bounds      : {bounds}", feedback)
                self.log(f"  CRS         : {crs}", feedback)

                # -- CRS check ------------------------------------------- #
                crs_warning = None
                crs_is_geographic = False

                if crs is None:
                    crs_warning = (
                        "WARNING: DEM has no assigned CRS. "
                        "Assign a CRS before proceeding with "
                        "any conditioning steps."
                    )
                    crs_is_geographic = True
                elif crs.is_geographic:
                    crs_is_geographic = True
                    crs_warning = (
                        f"WARNING: DEM CRS is geographic "
                        f"({crs.to_string()}). "
                        f"Cell sizes are in degrees, not metres. "
                        f"Distance-dependent calculations (area, "
                        f"breach length, void size in m²) will be "
                        f"approximate. Consider reprojecting to a "
                        f"suitable projected CRS before conditioning."
                    )

                if crs_warning:
                    self.log_warning(crs_warning, feedback)
                    provenance["warnings"].append(crs_warning)
                else:
                    self.log(
                        f"  CRS check   : OK — projected CRS confirmed.",
                        feedback
                    )

                provenance["stage_0"]["crs"] = (
                    crs.to_string() if crs else "None"
                )
                provenance["stage_0"]["crs_is_geographic"] = crs_is_geographic
                provenance["stage_0"]["resolution_x"] = res_x
                provenance["stage_0"]["resolution_y"] = res_y
                provenance["stage_0"]["width"]  = width
                provenance["stage_0"]["height"] = height
                provenance["stage_0"]["nodata"] = str(nodata)
                provenance["stage_0"]["dtype"]  = dtype

                feedback.setProgress(15)

                # -- Read elevation data ---------------------------------- #
                self.log("", feedback)
                self.log("  Reading elevation data...", feedback)
                dem_array = src.read(1).astype(np.float64)

                # -- Vertical accuracy assignment ------------------------- #
                self.log("", feedback)
                self.log("  Assigning vertical accuracy...", feedback)

                if user_rmse > 0:
                    rmse = user_rmse
                    rmse_source = "user-supplied"
                else:
                    rmse = DEMSourceType.DEFAULT_RMSE.get(
                        source_type_idx,
                        DEMSourceType.DEFAULT_RMSE[DEMSourceType.UNKNOWN]
                    )
                    if rmse is None:
                        # Auto-detect: use unknown default
                        rmse = DEMSourceType.DEFAULT_RMSE[
                            DEMSourceType.UNKNOWN
                        ]
                        rmse_source = "auto-detect default (unknown source)"
                    else:
                        rmse_source = (
                            f"default for "
                            f"{DEMSourceType.LABELS[source_type_idx]}"
                        )

                self.log(
                    f"  Vertical accuracy (RMSE): {rmse:.3f} m "
                    f"({rmse_source})",
                    feedback
                )
                provenance["stage_0"]["vertical_accuracy_rmse"] = rmse
                provenance["stage_0"]["vertical_accuracy_source"] = rmse_source

                feedback.setProgress(20)

                # -- Void (no-data) detection ----------------------------- #
                self.log("", feedback)
                self.log("  Detecting void (no-data) cells...", feedback)

                if nodata is not None:
                    void_mask_bool = (dem_array == nodata)
                else:
                    void_mask_bool = ~np.isfinite(dem_array)

                total_cells   = width * height
                total_voids   = int(np.sum(void_mask_bool))
                void_pct      = (total_voids / total_cells) * 100

                self.log(
                    f"  Total cells : {total_cells:,}", feedback
                )
                self.log(
                    f"  Void cells  : {total_voids:,} "
                    f"({void_pct:.2f}% of total)",
                    feedback
                )

                # -- Classify voids by connected component size ----------- #
                void_class_array = np.zeros_like(dem_array, dtype=np.uint8)
                dem_screened     = dem_array.copy()

                small_void_count  = 0
                medium_void_count = 0
                large_void_count  = 0
                large_void_areas  = []

                if total_voids > 0:
                    # Label connected void regions
                    labeled_voids, num_regions = label(void_mask_bool)

                    self.log(
                        f"  Void regions: {num_regions} connected "
                        f"void region(s) detected",
                        feedback
                    )

                    for region_id in range(1, num_regions + 1):
                        region_mask = (labeled_voids == region_id)
                        region_size = int(np.sum(region_mask))

                        if region_size <= small_void_thresh:
                            # -- Small void — interpolate ----------------- #
                            void_class_array[region_mask] = VoidClass.SMALL
                            rows, cols = np.where(region_mask)
                            for r, c in zip(rows, cols):
                                # Mean of valid neighbours
                                r0 = max(0, r - 1)
                                r1 = min(height - 1, r + 1)
                                c0 = max(0, c - 1)
                                c1 = min(width - 1, c + 1)
                                neighbourhood = dem_array[r0:r1+1, c0:c1+1]
                                valid_vals = neighbourhood[
                                    ~void_mask_bool[r0:r1+1, c0:c1+1]
                                ]
                                if len(valid_vals) > 0:
                                    dem_screened[r, c] = np.mean(valid_vals)
                                else:
                                    # No valid neighbours — mark as medium
                                    void_class_array[r, c] = VoidClass.MEDIUM
                            small_void_count += region_size

                        elif region_size <= large_void_thresh:
                            # -- Medium void — flag, do not fill --------- #
                            void_class_array[region_mask] = VoidClass.MEDIUM
                            medium_void_count += region_size

                        else:
                            # -- Large void — report to analyst ---------- #
                            void_class_array[region_mask] = VoidClass.LARGE
                            large_void_count += region_size
                            # Record location of large void
                            rows, cols = np.where(region_mask)
                            large_void_areas.append({
                                "region_id":   int(region_id),
                                "size_cells":  region_size,
                                "row_min":     int(rows.min()),
                                "row_max":     int(rows.max()),
                                "col_min":     int(cols.min()),
                                "col_max":     int(cols.max()),
                            })

                    self.log(
                        f"  Small voids (interpolated) : "
                        f"{small_void_count:,} cells",
                        feedback
                    )
                    self.log(
                        f"  Medium voids (flagged)     : "
                        f"{medium_void_count:,} cells",
                        feedback
                    )
                    self.log(
                        f"  Large voids (analyst alert): "
                        f"{large_void_count:,} cells",
                        feedback
                    )

                    if large_void_areas:
                        self.log_warning(
                            f"ANALYST ALERT: {len(large_void_areas)} "
                            f"large void region(s) detected. These have "
                            f"NOT been filled. Review the void mask and "
                            f"address these gaps before proceeding with "
                            f"any conditioning steps.",
                            feedback
                        )
                        for lv in large_void_areas:
                            self.log_warning(
                                f"  Large void region {lv['region_id']}: "
                                f"{lv['size_cells']:,} cells — "
                                f"rows {lv['row_min']}-{lv['row_max']}, "
                                f"cols {lv['col_min']}-{lv['col_max']}",
                                feedback
                            )

                provenance["stage_0"]["void_summary"] = {
                    "total_void_cells":   total_voids,
                    "void_pct":           round(void_pct, 4),
                    "small_interpolated": small_void_count,
                    "medium_flagged":     medium_void_count,
                    "large_reported":     large_void_count,
                    "large_void_regions": large_void_areas,
                }

                feedback.setProgress(40)
                if self.is_cancelled(feedback):
                    return {}

                # -- Write screened DEM ----------------------------------- #
                self.log("", feedback)
                self.log("  Writing screened DEM...", feedback)

                out_profile = profile.copy()
                out_profile.update(
                    dtype=rasterio.float64,
                    count=1,
                    compress="lzw",
                    tiled=True,
                    blockxsize=256,
                    blockysize=256,
                )

                with rasterio.open(
                    paths["screened_dem"], "w", **out_profile
                ) as dst:
                    dst.write(dem_screened.astype(np.float64), 1)

                self.log(
                    f"  Screened DEM written: "
                    f"{paths['screened_dem'].name}",
                    feedback
                )

                # -- Write void classification mask ----------------------- #
                void_profile = profile.copy()
                void_profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    nodata=255,
                    compress="lzw",
                )

                with rasterio.open(
                    paths["void_mask"], "w", **void_profile
                ) as dst:
                    dst.write(void_class_array, 1)

                self.log(
                    f"  Void mask written   : "
                    f"{paths['void_mask'].name}",
                    feedback
                )

                feedback.setProgress(50)

                # ==========================================================
                # STAGE 1 — ARTIFACT SCREENING
                # ==========================================================

                self.log("", feedback)
                self.log("=" * 60, feedback)
                self.log("  STAGE 1 — ARTIFACT SCREENING", feedback)
                self.log("=" * 60, feedback)

                # -- Classify DEM source type ----------------------------- #
                self.log("", feedback)
                self.log(
                    f"  DEM source type: "
                    f"{DEMSourceType.LABELS[source_type_idx]}",
                    feedback
                )

                provenance["stage_1"]["source_type"] = (
                    DEMSourceType.LABELS[source_type_idx]
                )

                # -- Determine artifact screening pathway ---------------- #
                is_sar_insar = source_type_idx in (
                    DEMSourceType.SRTM,
                    DEMSourceType.COPERNICUS_GLO30,
                    DEMSourceType.UNKNOWN,
                    DEMSourceType.AUTO_DETECT,
                )
                is_lidar_dsm = (
                    source_type_idx == DEMSourceType.LIDAR_DSM
                )
                is_lidar_dtm = (
                    source_type_idx == DEMSourceType.LIDAR_DTM
                )
                is_corrected = (
                    source_type_idx == DEMSourceType.FABDEM_FATHOMDEM
                )

                # -- LiDAR DSM — ground filter (coming soon) ------------- #
                if is_lidar_dsm:
                    msg = (
                        "NOTE: LiDAR DSM ground filtering (SMRF / "
                        "Progressive TIN Densification) is not yet "
                        "implemented in this version. Your DSM will be "
                        "screened for statistical outliers only. Full "
                        "ground filtering will be available in a future "
                        "release. Consider supplying a bare-earth DTM "
                        "if one is available."
                    )
                    self.log_warning(msg, feedback)
                    provenance["stage_1"]["lidar_ground_filter"] = (
                        "not_implemented — coming in future release"
                    )

                # -- FABDEM/FathomDEM — already corrected ---------------- #
                if is_corrected:
                    self.log(
                        "  Source is a corrected bare-earth product "
                        "(FABDEM/FathomDEM/GEDTM30). Vegetation and "
                        "building bias correction already applied by "
                        "the data provider. Proceeding to statistical "
                        "outlier screening only.",
                        feedback
                    )
                    provenance["stage_1"]["bias_correction"] = (
                        "not_required — pre-corrected product"
                    )

                # -- LiDAR DTM — no bias correction needed --------------- #
                if is_lidar_dtm:
                    self.log(
                        "  Source is a LiDAR bare-earth DTM. "
                        "Vegetation/building bias correction not "
                        "required. Proceeding to statistical "
                        "outlier screening.",
                        feedback
                    )
                    provenance["stage_1"]["bias_correction"] = (
                        "not_required — bare-earth DTM"
                    )

                feedback.setProgress(55)
                if self.is_cancelled(feedback):
                    return {}

                # -- MAD Local Outlier Filter ----------------------------- #
                self.log("", feedback)
                self.log(
                    f"  Running MAD artifact filter "
                    f"(window={mad_window}x{mad_window}, "
                    f"threshold={mad_threshold}σ)...",
                    feedback
                )

                artifact_mask = self._run_mad_filter(
                    dem_array=dem_screened,
                    void_mask=void_mask_bool,
                    window_size=mad_window,
                    threshold=mad_threshold,
                    feedback=feedback,
                )

                total_artifacts = int(np.sum(artifact_mask))
                artifact_pct = (total_artifacts / total_cells) * 100

                self.log(
                    f"  Artifact cells detected: "
                    f"{total_artifacts:,} "
                    f"({artifact_pct:.3f}% of total)",
                    feedback
                )

                # -- Warn if high artifact percentage -------------------- #
                if artifact_pct > 5.0:
                    warn_msg = (
                        f"WARNING: {artifact_pct:.1f}% of cells flagged "
                        f"as suspected artifacts. This is unusually high. "
                        f"Consider reviewing the MAD threshold or "
                        f"inspecting the artifact mask before proceeding."
                    )
                    self.log_warning(warn_msg, feedback)
                    provenance["warnings"].append(warn_msg)
                elif artifact_pct > 1.0:
                    self.log(
                        f"  Note: {artifact_pct:.1f}% artifact rate. "
                        f"Review artifact mask before proceeding.",
                        feedback
                    )

                provenance["stage_1"]["mad_filter"] = {
                    "window_size":       mad_window,
                    "threshold_sigma":   mad_threshold,
                    "cells_flagged":     total_artifacts,
                    "cells_flagged_pct": round(artifact_pct, 4),
                }

                feedback.setProgress(75)
                if self.is_cancelled(feedback):
                    return {}

                # -- Write artifact mask raster --------------------------- #
                artifact_profile = profile.copy()
                artifact_profile.update(
                    dtype=rasterio.uint8,
                    count=1,
                    nodata=255,
                    compress="lzw",
                )

                with rasterio.open(
                    paths["artifact_mask"], "w", **artifact_profile
                ) as dst:
                    dst.write(
                        artifact_mask.astype(np.uint8), 1
                    )

                self.log(
                    f"  Artifact mask written: "
                    f"{paths['artifact_mask'].name}",
                    feedback
                )

                feedback.setProgress(80)

        # ── End rasterio context ──────────────────────────────────────── #

        except QgsProcessingException:
            # Re-raise QGIS processing exceptions directly
            raise

        except Exception as e:
            MayimLogger.critical(
                f"DEM Hydrological Screening failed: {e}"
            )
            raise QgsProcessingException(
                f"DEM Hydrological Screening failed: {e}"
            )

        # ==================================================================
        # WRITE REPORTS & PROVENANCE
        # ==================================================================

        self.log("", feedback)
        self.log("=" * 60, feedback)
        self.log("  WRITING REPORTS", feedback)
        self.log("=" * 60, feedback)

        # -- Write plain text QA report ----------------------------------- #
        self._write_text_report(
            paths=paths,
            provenance=provenance,
            dem_stem=dem_stem,
            timestamp=timestamp,
            feedback=feedback,
        )

        # -- Write provenance JSON ---------------------------------------- #
        with open(paths["provenance"], "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=4, default=str)

        self.log(
            f"  Provenance log written: "
            f"{paths['provenance'].name}",
            feedback
        )

        feedback.setProgress(95)

        # -- Final summary ------------------------------------------------ #
        self.log("", feedback)
        self.log("=" * 60, feedback)
        self.log("  SCREENING COMPLETE", feedback)
        self.log("=" * 60, feedback)
        self.log(
            f"  All outputs written to: {output_folder}", feedback
        )
        self.log(
            f"  Review the QA report before proceeding to "
            f"any conditioning steps.",
            feedback
        )
        self.log("", feedback)

        feedback.setProgress(100)

        return {
            self.OUTPUT_DEM:           str(paths["screened_dem"]),
            self.OUTPUT_VOID_MASK:     str(paths["void_mask"]),
            self.OUTPUT_ARTIFACT_MASK: str(paths["artifact_mask"]),
            self.OUTPUT_QA_REPORT:     str(paths["qa_report_txt"]),
            self.OUTPUT_PROVENANCE:    str(paths["provenance"]),
        }

    # -- Private helper methods ------------------------------------------- #

    def _run_mad_filter(
        self,
        dem_array: np.ndarray,
        void_mask: np.ndarray,
        window_size: int,
        threshold: float,
        feedback: QgsProcessingFeedback,
    ) -> np.ndarray:
        """
        Apply a Median Absolute Deviation (MAD) local outlier filter
        to detect speckle, striping, and scan-line noise.

        For each valid cell, computes the local median and MAD within
        a sliding window. Cells whose deviation from the local median
        exceeds (threshold * MAD) are flagged as suspected artifacts.

        This is a robust local outlier filter — it does NOT modify
        elevations. It only produces a binary artifact mask.

        Reference:
            Leys, C. et al. (2013). Detecting outliers: Do not use
            standard deviation around the mean, use absolute deviation
            around the median. Journal of Experimental Social
            Psychology, 49(4), 764-766.

        :param dem_array: 2D numpy array of elevation values
        :param void_mask: Boolean mask — True where cells are void
        :param window_size: Sliding window size (must be odd)
        :param threshold: Number of MAD units for outlier flagging
        :param feedback: QGIS feedback object for cancellation checks
        :returns: Binary uint8 array — 1 = artifact, 0 = clean
        """
        from scipy.ndimage import generic_filter

        height, width = dem_array.shape
        artifact_mask = np.zeros((height, width), dtype=np.uint8)
        half = window_size // 2

        # Working array — set voids to NaN so they are excluded
        work = dem_array.copy()
        work[void_mask] = np.nan

        self.log(
            f"    MAD filter: processing "
            f"{height:,} x {width:,} cells...",
            feedback
        )

        # Process row by row for progress reporting and cancellation
        for row in range(height):

            if self.is_cancelled(feedback):
                return artifact_mask

            # Report progress every 10% of rows
            if row % max(1, height // 10) == 0:
                pct = int((row / height) * 100)
                self.log(
                    f"    MAD filter: {pct}% complete...",
                    feedback
                )

            for col in range(width):

                # Skip void cells
                if void_mask[row, col]:
                    continue

                # Extract local window
                r0 = max(0, row - half)
                r1 = min(height, row + half + 1)
                c0 = max(0, col - half)
                c1 = min(width, col + half + 1)
                window = work[r0:r1, c0:c1]

                # Get valid (non-NaN) values in window
                valid = window[~np.isnan(window)]

                if len(valid) < 3:
                    # Not enough neighbours — skip
                    continue

                # Compute local median and MAD
                local_median = np.median(valid)
                mad = np.median(np.abs(valid - local_median))

                # Avoid division by zero in flat areas
                if mad < 1e-10:
                    continue

                # Flag if cell deviates beyond threshold
                cell_val = dem_array[row, col]
                outlier_score = abs(cell_val - local_median) / mad

                if outlier_score > threshold:
                    artifact_mask[row, col] = 1

        return artifact_mask

    def _write_text_report(
        self,
        paths: dict,
        provenance: dict,
        dem_stem: str,
        timestamp: str,
        feedback: QgsProcessingFeedback,
    ) -> None:
        """
        Write a plain text QA and artifact screening report.

        :param paths: Dictionary of output file paths
        :param provenance: Provenance log dictionary
        :param dem_stem: Input DEM filename stem
        :param timestamp: Run timestamp string
        :param feedback: QGIS feedback object
        """
        s0 = provenance.get("stage_0", {})
        s1 = provenance.get("stage_1", {})
        warnings = provenance.get("warnings", [])
        errors   = provenance.get("errors", [])
        params   = provenance.get("parameters", {})

        lines = []
        a = lines.append  # shorthand for readability

        # -- Report header ------------------------------------------------ #
        a("=" * 70)
        a("  MAYIM TOOLS — DEM HYDROLOGICAL SCREENING REPORT")
        a("=" * 70)
        a(f"  Generated : {provenance.get('timestamp', timestamp)}")
        a(f"  Tool      : {provenance.get('tool', 'DEM Hydrological Screening')}")
        a(f"  Version   : {provenance.get('version', '0.1.0')}")
        a(f"  Input DEM : {provenance.get('input_dem', 'Unknown')}")
        a("=" * 70)
        a("")

        # -- Parameters used ---------------------------------------------- #
        a("-" * 70)
        a("  PARAMETERS")
        a("-" * 70)
        a(f"  DEM source type       : {params.get('source_type', 'Unknown')}")
        a(f"  User RMSE override    : "
          f"{params.get('user_rmse', -1)} m "
          f"(-1 = use default)")
        a(f"  Small void threshold  : "
          f"{params.get('small_void_thresh', 10)} cells")
        a(f"  Large void threshold  : "
          f"{params.get('large_void_thresh', 100)} cells")
        a(f"  MAD window size       : "
          f"{params.get('mad_window', 3)} x "
          f"{params.get('mad_window', 3)} cells")
        a(f"  MAD threshold (sigma) : "
          f"{params.get('mad_threshold', 3.0)}")
        a("")

        # -- Stage 0 results ---------------------------------------------- #
        a("-" * 70)
        a("  STAGE 0 — INGESTION & QA")
        a("-" * 70)
        a(f"  CRS                   : {s0.get('crs', 'Unknown')}")
        a(f"  CRS is geographic     : {s0.get('crs_is_geographic', 'Unknown')}")
        a(f"  Resolution (X)        : {s0.get('resolution_x', 'Unknown')}")
        a(f"  Resolution (Y)        : {s0.get('resolution_y', 'Unknown')}")
        a(f"  Dimensions            : "
          f"{s0.get('width', '?')} x {s0.get('height', '?')} cells")
        a(f"  Data type             : {s0.get('dtype', 'Unknown')}")
        a(f"  NoData value          : {s0.get('nodata', 'None')}")
        a(f"  Vertical accuracy     : "
          f"{s0.get('vertical_accuracy_rmse', 'Unknown')} m RMSE "
          f"({s0.get('vertical_accuracy_source', 'Unknown')})")
        a("")

        # -- Void summary ------------------------------------------------- #
        void_summary = s0.get("void_summary", {})
        a(f"  VOID SUMMARY")
        a(f"  {'-' * 40}")
        a(f"  Total void cells      : "
          f"{void_summary.get('total_void_cells', 0):,} "
          f"({void_summary.get('void_pct', 0.0):.2f}% of total)")
        a(f"  Small (interpolated)  : "
          f"{void_summary.get('small_interpolated', 0):,} cells")
        a(f"  Medium (flagged)      : "
          f"{void_summary.get('medium_flagged', 0):,} cells")
        a(f"  Large (analyst alert) : "
          f"{void_summary.get('large_reported', 0):,} cells")
        a("")

        # -- Large void regions detail ------------------------------------ #
        large_regions = void_summary.get("large_void_regions", [])
        if large_regions:
            a(f"  LARGE VOID REGIONS — ANALYST ACTION REQUIRED")
            a(f"  {'-' * 40}")
            a(f"  These regions have NOT been filled.")
            a(f"  Address these data gaps before proceeding")
            a(f"  with any conditioning steps.")
            a("")
            for lv in large_regions:
                a(f"  Region {lv['region_id']:>3} : "
                  f"{lv['size_cells']:>8,} cells — "
                  f"rows {lv['row_min']}-{lv['row_max']}, "
                  f"cols {lv['col_min']}-{lv['col_max']}")
            a("")

        # -- Stage 1 results ---------------------------------------------- #
        a("-" * 70)
        a("  STAGE 1 — ARTIFACT SCREENING")
        a("-" * 70)
        a(f"  DEM source type       : "
          f"{s1.get('source_type', 'Unknown')}")
        a(f"  Bias correction       : "
          f"{s1.get('bias_correction', 'See notes below')}")
        a(f"  LiDAR ground filter   : "
          f"{s1.get('lidar_ground_filter', 'N/A')}")
        a("")

        # -- MAD filter results ------------------------------------------- #
        mad = s1.get("mad_filter", {})
        if mad:
            a(f"  MAD ARTIFACT FILTER RESULTS")
            a(f"  {'-' * 40}")
            a(f"  Window size           : "
              f"{mad.get('window_size', '?')} x "
              f"{mad.get('window_size', '?')} cells")
            a(f"  Detection threshold   : "
              f"{mad.get('threshold_sigma', '?')} sigma")
            a(f"  Cells flagged         : "
              f"{mad.get('cells_flagged', 0):,} cells "
              f"({mad.get('cells_flagged_pct', 0.0):.3f}%)")
            a("")

            # -- Interpretation guidance ---------------------------------- #
            pct = mad.get("cells_flagged_pct", 0.0)
            a(f"  INTERPRETATION")
            a(f"  {'-' * 40}")
            if pct == 0.0:
                a("  No artifacts detected. DEM appears clean.")
            elif pct < 0.1:
                a("  Very low artifact rate. DEM is in good condition.")
                a("  Minor noise may be present — review artifact mask.")
            elif pct < 1.0:
                a("  Low artifact rate. Some noise detected.")
                a("  Review artifact mask before conditioning.")
            elif pct < 5.0:
                a("  Moderate artifact rate. Noise is present.")
                a("  Carefully review artifact mask.")
                a("  Consider Stage 2 smoothing if noise is systematic.")
            else:
                a("  HIGH artifact rate. Significant noise detected.")
                a("  ACTION REQUIRED: Review artifact mask carefully.")
                a("  Consider adjusting MAD threshold or reviewing")
                a("  DEM source quality before proceeding.")
            a("")

        # -- Warnings ---------------------------------------------------- #
        if warnings:
            a("-" * 70)
            a("  WARNINGS")
            a("-" * 70)
            for i, w in enumerate(warnings, 1):
                # Word-wrap long warnings at 65 characters
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

        # -- Errors ------------------------------------------------------ #
        if errors:
            a("-" * 70)
            a("  ERRORS")
            a("-" * 70)
            for i, e in enumerate(errors, 1):
                a(f"  {i}. {e}")
            a("")

        # -- Output files ------------------------------------------------- #
        a("-" * 70)
        a("  OUTPUT FILES")
        a("-" * 70)
        a(f"  Screened DEM          : {paths['screened_dem'].name}")
        a(f"  Void mask             : {paths['void_mask'].name}")
        a(f"  Artifact mask         : {paths['artifact_mask'].name}")
        a(f"  QA report (this file) : {paths['qa_report_txt'].name}")
        a(f"  Provenance log (JSON) : {paths['provenance'].name}")
        a("")

        # -- Void mask legend --------------------------------------------- #
        a("-" * 70)
        a("  VOID MASK LEGEND")
        a("-" * 70)
        a("  Value 0 = Valid data cell")
        a("  Value 1 = Small void — interpolated by this tool")
        a("  Value 2 = Medium void — flagged, not filled")
        a("  Value 3 = Large void — reported to analyst, not filled")
        a("")

        # -- Artifact mask legend ----------------------------------------- #
        a("-" * 70)
        a("  ARTIFACT MASK LEGEND")
        a("-" * 70)
        a("  Value 0 = Clean cell")
        a("  Value 1 = Suspected artifact (MAD outlier)")
        a("")

        # -- Next steps --------------------------------------------------- #
        a("-" * 70)
        a("  RECOMMENDED NEXT STEPS")
        a("-" * 70)

        has_large_voids    = len(large_regions) > 0
        has_high_artifacts = mad.get("cells_flagged_pct", 0.0) > 1.0
        has_geo_crs        = s0.get("crs_is_geographic", False)

        if has_large_voids:
            a("  1. [REQUIRED] Address large void regions before")
            a("     proceeding. These represent genuine data gaps.")
        if has_geo_crs:
            a("  2. [RECOMMENDED] Reproject DEM to a suitable")
            a("     projected CRS before conditioning.")
        if has_high_artifacts:
            a("  3. [RECOMMENDED] Review the artifact mask raster.")
            a("     Consider Stage 2 (Edge-Preserving Smoothing)")
            a("     if noise is systematic.")
        if not has_large_voids and not has_high_artifacts:
            a("  DEM has passed screening. Proceed to the next")
            a("  conditioning stage:")
            a("  -> Stage 2: Edge-Preserving Smoothing (if needed)")
            a("  -> Stage 3: Depression Delineation & Hierarchy")
        a("")

        # -- References --------------------------------------------------- #
        a("-" * 70)
        a("  REFERENCES")
        a("-" * 70)
        a("  Barnes, R., Lehman, C., & Mulla, D. (2014).")
        a("    Priority-flood: An optimal depression-filling and")
        a("    watershed-labeling algorithm for digital elevation")
        a("    models. Computers & Geosciences, 62, 117-127.")
        a("")
        a("  Hawker, L. et al. (2022). A 30 m global map of")
        a("    elevation with forests and buildings removed.")
        a("    Environmental Research Letters, 17(2), 024016.")
        a("")
        a("  Leys, C. et al. (2013). Detecting outliers: Do not")
        a("    use standard deviation around the mean, use absolute")
        a("    deviation around the median. Journal of Experimental")
        a("    Social Psychology, 49(4), 764-766.")
        a("")
        a("  Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013).")
        a("    An improved simple morphological filter for the terrain")
        a("    classification of airborne LIDAR data. ISPRS Journal")
        a("    of Photogrammetry and Remote Sensing, 77, 21-30.")
        a("")
        a("  Wang, L., & Liu, H. (2006). An efficient method for")
        a("    identifying and filling surface depressions in digital")
        a("    elevation models for hydrologic analysis and modelling.")
        a("    International Journal of Geographical Information")
        a("    Science, 20(2), 193-213.")
        a("")

        # -- Footer ------------------------------------------------------- #
        a("=" * 70)
        a("  Mayim Tools — DEM Hydrological Screening")
        a("  https://github.com/chrismayim/mayim-tools")
        a("  License: GPL-2.0+")
        a("=" * 70)
        a("")

        # -- Write to file ------------------------------------------------ #
        report_text = "\n".join(lines)

        with open(
            paths["qa_report_txt"], "w", encoding="utf-8"
        ) as f:
            f.write(report_text)

        self.log(
            f"  QA report written     : "
            f"{paths['qa_report_txt'].name}",
            feedback
        )
