# dem_hydrography_enforcement.py
#
# Stage 7E — DEM Hydrography Enforcement Adapter
# Mayim Tools | Hydrology Category
#
# Phase 1: Grid alignment and rasterisation — complete.
# Phase 2: Divergence analysis — complete.
# Phase 3: Adaptive enforcement — connected in this version.
#
# IP STATUS: CLEAR WITH CLEAN-ROOM RECORD KEEPING
# Uses QGIS, rasterio, NumPy, Shapely and native Mayim modules as infrastructure.
# Does not call WhiteboxTools, RichDEM or TauDEM.
# Terrain-modification logic remains in the native Mayim enforcement module.
#
# References:
#   Soille, Vogt & Colombo (2003)
#   Lindsay (2016) — TopologicalBreachBurn
#   Hellweger (1997) — AGREE
#   Callow et al. (2007)
#
# Author  : Mayim Tools Development Team
# Created : 2025
# License : Proprietary — Zutari / Mayim

"""
Stage 7E — DEM Hydrography Enforcement Adapter.

Validates, prepares, grid-aligns, analyses divergence and applies
adaptive, topology-aware hydrography enforcement following Soille,
Vogt & Colombo (2003) and Lindsay's (2016) TopologicalBreachBurn.

Phase 1 (complete): Grid alignment and rasterisation.
Phase 2 (complete): Divergence analysis.
Phase 3 (this version): Adaptive enforcement.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qgis.core import (
    QgsProcessing,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsProject,
    QgsWkbTypes,
)

from mayim_tools.core.logger import MayimLogger
from mayim_tools.hydrology.hydrography.enforcement import (
    enforce_hydrography,
)
from mayim_tools.hydrology.hydrography.topology import (
    prepare_hydrography_topology,
)
from mayim_tools.hydrology.hydrography.validation import (
    validate_hydrography,
)
from mayim_tools.processing.algorithms.base_algorithm import (
    MayimBaseAlgorithm,
)


class DEMHydrographyEnforcement(MayimBaseAlgorithm):
    """
    Stage 7E — DEM Hydrography Enforcement.

    Adaptive, topology-aware hydrography enforcement following
    Soille, Vogt & Colombo (2003) and Lindsay's (2016)
    TopologicalBreachBurn.

    Burning is applied only where the DEM-derived flow path diverges
    materially from the mapped channel. Burn depth scales continuously
    with contributing area when upstream area is supplied.
    This stage is optional and off by default.
    """

    # ── Parameter identifiers ─────────────────────────────────────────────────

    INPUT_DEM = "INPUT_DEM"
    INPUT_HYDROGRAPHY = "INPUT_HYDROGRAPHY"
    INPUT_FLOW_EVIDENCE = "INPUT_FLOW_EVIDENCE"
    ENDPOINT_TOLERANCE = "ENDPOINT_TOLERANCE"
    POSITIONAL_TOLERANCE_CELLS = "POSITIONAL_TOLERANCE_CELLS"
    VERTICAL_ACCURACY = "VERTICAL_ACCURACY"
    MAXIMUM_BURN_DEPTH = "MAXIMUM_BURN_DEPTH"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    # ── Output identifiers ────────────────────────────────────────────────────

    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_PROVENANCE = "OUTPUT_PROVENANCE"

    # ── Defaults ──────────────────────────────────────────────────────────────

    DEFAULT_ENDPOINT_TOLERANCE = 1.0
    DEFAULT_POSITIONAL_TOLERANCE_CELLS = 3
    DEFAULT_VERTICAL_ACCURACY = 0.5
    DEFAULT_MAXIMUM_BURN_DEPTH = 2.0

    # ── Algorithm identity ────────────────────────────────────────────────────

    def name(self) -> str:
        return "demhydrographyenforcement"

    def createInstance(self):
        return DEMHydrographyEnforcement()

    def displayName(self) -> str:
        return "DEM Hydrography Enforcement"

    def group(self) -> str:
        return "Hydrology Tools"

    def groupId(self) -> str:
        return "hydrologytools"

    def shortHelpString(self) -> str:
        return (
            "Stage 7E — Adaptive, topology-aware hydrography enforcement.\n\n"
            "Burning is applied only where the DEM-derived flow path diverges "
            "materially from the mapped channel, following Soille, Vogt & Colombo "
            "(2003) and Lindsay's (2016) TopologicalBreachBurn.\n\n"
            "Burn depth is bounded by the lesser of vertical accuracy and maximum "
            "burn depth. Area-scaled burning is available when a flow accumulation "
            "raster is connected in a future phase.\n\n"
            "This stage is optional and off by default."
        )

    # ── Parameter definition ──────────────────────────────────────────────────

    def initAlgorithm(self, config=None) -> None:
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                "Input DEM",
            )
        )

        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT_HYDROGRAPHY,
                "Input vector hydrography",
                types=[QgsProcessing.TypeVectorLine],
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_FLOW_EVIDENCE,
                "DEM-derived flow evidence raster",
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.ENDPOINT_TOLERANCE,
                "Hydrography endpoint tolerance (map units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=self.DEFAULT_ENDPOINT_TOLERANCE,
                minValue=0.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.POSITIONAL_TOLERANCE_CELLS,
                "Positional tolerance (cells)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=self.DEFAULT_POSITIONAL_TOLERANCE_CELLS,
                minValue=0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.VERTICAL_ACCURACY,
                "DEM vertical accuracy (elevation units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=self.DEFAULT_VERTICAL_ACCURACY,
                minValue=0.001,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAXIMUM_BURN_DEPTH,
                "Maximum burn depth (elevation units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=self.DEFAULT_MAXIMUM_BURN_DEPTH,
                minValue=0.001,
            )
        )

        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER,
                "Output folder",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_REPORT,
                "Enforcement report",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_PROVENANCE,
                "Enforcement provenance",
            )
        )

    # ── Main processing ───────────────────────────────────────────────────────

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        logger = MayimLogger(__name__)
        logger.info("Stage 7E — DEM Hydrography Enforcement started.")

        run_timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

        warnings: list[str] = []

        # ── Resolve inputs ────────────────────────────────────────────────────

        dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        if dem_layer is None or not dem_layer.isValid():
            raise QgsProcessingException(
                "Input DEM is not valid or could not be loaded."
            )
        dem_source = dem_layer.source()

        hydrography_layer = self.parameterAsVectorLayer(
            parameters, self.INPUT_HYDROGRAPHY, context
        )
        if hydrography_layer is None or not hydrography_layer.isValid():
            raise QgsProcessingException(
                "Input hydrography layer is not valid or could not be loaded."
            )

        flow_evidence_layer = self.parameterAsRasterLayer(
            parameters, self.INPUT_FLOW_EVIDENCE, context
        )
        if flow_evidence_layer is None or not flow_evidence_layer.isValid():
            raise QgsProcessingException(
                "Flow evidence raster is not valid or could not be loaded."
            )
        flow_evidence_source = flow_evidence_layer.source()

        endpoint_tolerance = self.parameterAsDouble(
            parameters, self.ENDPOINT_TOLERANCE, context
        )

        positional_tolerance_cells = self.parameterAsInt(
            parameters, self.POSITIONAL_TOLERANCE_CELLS, context
        )

        vertical_accuracy = self.parameterAsDouble(
            parameters, self.VERTICAL_ACCURACY, context
        )

        maximum_burn_depth = self.parameterAsDouble(
            parameters, self.MAXIMUM_BURN_DEPTH, context
        )

        output_folder = Path(
            self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        )
        output_folder.mkdir(parents=True, exist_ok=True)

        feedback.pushInfo("Inputs resolved successfully.")
        feedback.pushInfo(f"DEM source                  : {dem_source}")
        feedback.pushInfo(f"Hydrography source          : {hydrography_layer.source()}")
        feedback.pushInfo(f"Flow evidence source        : {flow_evidence_source}")
        feedback.pushInfo(f"Endpoint tolerance          : {endpoint_tolerance}")
        feedback.pushInfo(f"Positional tolerance cells  : {positional_tolerance_cells}")
        feedback.pushInfo(f"Vertical accuracy           : {vertical_accuracy}")
        feedback.pushInfo(f"Maximum burn depth          : {maximum_burn_depth}")
        feedback.pushInfo(f"Output folder               : {output_folder}")

        # ── CRS check ─────────────────────────────────────────────────────────

        dem_crs = dem_layer.crs()
        hydrography_crs = hydrography_layer.crs()
        crs_match = dem_crs.authid() == hydrography_crs.authid()

        if not crs_match:
            warnings.append(
                f"CRS mismatch: DEM is {dem_crs.authid()}, "
                f"hydrography is {hydrography_crs.authid()}. "
                "An in-memory CRS transformation will be applied."
            )
            feedback.pushWarning(warnings[-1])

            from qgis.core import QgsCoordinateTransform

            crs_transform = QgsCoordinateTransform(
                hydrography_crs,
                dem_crs,
                QgsProject.instance(),
            )
        else:
            crs_transform = None
            feedback.pushInfo(f"CRS match confirmed: {dem_crs.authid()}")

        # ── Geometry type check ───────────────────────────────────────────────

        geometry_type = hydrography_layer.geometryType()
        if geometry_type != QgsWkbTypes.LineGeometry:
            raise QgsProcessingException(
                f"Hydrography layer must be a line layer. "
                f"Received geometry type: {geometry_type}."
            )

        feedback.pushInfo("Geometry type confirmed: line.")

        # ── Feature count ─────────────────────────────────────────────────────

        feature_count = hydrography_layer.featureCount()
        feedback.pushInfo(f"Hydrography feature count: {feature_count}")

        if feature_count == 0:
            raise QgsProcessingException("Hydrography layer contains no features.")

        # ── DEM extent check ──────────────────────────────────────────────────

        dem_extent = dem_layer.extent()
        features_inside = 0
        features_outside = 0

        for feature in hydrography_layer.getFeatures():
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue
            if crs_transform is not None:
                geometry.transform(crs_transform)
            if dem_extent.intersects(geometry.boundingBox()):
                features_inside += 1
            else:
                features_outside += 1

        feedback.pushInfo(f"Features intersecting DEM extent : {features_inside}")
        feedback.pushInfo(f"Features outside DEM extent      : {features_outside}")

        if features_inside == 0:
            raise QgsProcessingException(
                "No hydrography features intersect the DEM extent. "
                "Verify that the DEM and hydrography cover the same area."
            )

        if features_outside > 0:
            warnings.append(
                f"{features_outside} hydrography feature(s) fall outside "
                "the DEM extent and will be excluded from enforcement."
            )
            feedback.pushWarning(warnings[-1])

        # ── Hydrography validation ────────────────────────────────────────────

        feedback.pushInfo("Running hydrography geometry validation.")

        validation_result = validate_hydrography(
            layer=hydrography_layer,
            crs_transform=crs_transform,
            feedback=feedback,
        )

        feedback.pushInfo("Hydrography validation complete.")

        # ── Topology preparation ──────────────────────────────────────────────

        feedback.pushInfo("Running hydrography topology preparation.")

        topology_result = prepare_hydrography_topology(
            layer=hydrography_layer,
            crs_transform=crs_transform,
            endpoint_tolerance=endpoint_tolerance,
            feedback=feedback,
        )

        feedback.pushInfo("Topology preparation complete.")

        # ── Phase 1: DEM grid metadata ────────────────────────────────────────

        feedback.setProgress(10)
        feedback.pushInfo("── Phase 1: Grid alignment ──────────────────────")

        dem_metadata = self._read_dem_metadata(
            dem_path=dem_source,
            feedback=feedback,
        )

        # ── Phase 1: Read full DEM array ──────────────────────────────────────

        import rasterio

        feedback.pushInfo("Reading full DEM array for enforcement.")

        with rasterio.open(dem_source) as dem_dataset:
            dem_array = dem_dataset.read(1).astype(np.float64)

        feedback.pushInfo(
            f"DEM array read. Shape: " f"{dem_array.shape[0]} x {dem_array.shape[1]}."
        )

        # ── Phase 1: Hydrography rasterisation ───────────────────────────────

        feedback.setProgress(20)

        hydrography_raster = self._rasterise_hydrography(
            hydrography_layer=hydrography_layer,
            crs_transform=crs_transform,
            dem_metadata=dem_metadata,
            feedback=feedback,
        )

        burned_cell_count = int(np.sum(hydrography_raster > 0))

        # ── Phase 1: Flow-evidence raster ─────────────────────────────────────

        feedback.setProgress(30)

        flow_evidence_array = self._read_flow_evidence(
            flow_evidence_path=flow_evidence_source,
            dem_metadata=dem_metadata,
            feedback=feedback,
        )

        feedback.pushInfo("Phase 1 complete.")

        # ── Phase 2: Derive Boolean masks ─────────────────────────────────────

        feedback.setProgress(40)
        feedback.pushInfo("── Phase 2: Divergence analysis ─────────────────")

        hydrography_mask = hydrography_raster.astype(bool)
        dem_flow_mask = flow_evidence_array > 0

        nodata_mask = None
        if dem_metadata["nodata"] is not None:
            nodata_mask = flow_evidence_array == dem_metadata["nodata"]
            feedback.pushInfo(
                f"NoData mask derived. " f"NoData cells: {int(np.sum(nodata_mask))}."
            )

        # ── Phase 2: Divergence analysis ──────────────────────────────────────

        feedback.setProgress(50)
        feedback.pushInfo("Calling analyse_hydrography_divergence.")

        from mayim_tools.hydrology.hydrography.divergence import (
            analyse_hydrography_divergence,
        )

        try:
            divergence_result = analyse_hydrography_divergence(
                hydrography_mask=hydrography_mask,
                dem_flow_mask=dem_flow_mask,
                positional_tolerance_cells=positional_tolerance_cells,
                nodata_mask=nodata_mask,
            )
        except ValueError as exc:
            raise QgsProcessingException(f"Divergence analysis failed: {exc}") from exc

        divergence_stats = divergence_result["statistics"]
        conflict_mask = divergence_result["conflict_mask"].astype(bool)

        feedback.pushInfo(
            f"Divergence analysis complete. "
            f"Aligned cells        : {divergence_stats['aligned_cells']}."
        )
        feedback.pushInfo(
            f"Tolerated cells      : {divergence_stats['tolerated_cells']}."
        )
        feedback.pushInfo(
            f"Material divergence  : "
            f"{divergence_stats['material_divergence_cells']} cells."
        )
        feedback.pushInfo(
            f"Conflict cells       : {divergence_stats['conflict_cells']}."
        )

        feedback.pushInfo("Phase 2 complete.")

        # ── Phase 3: Build eligible mask ──────────────────────────────────────

        feedback.setProgress(60)
        feedback.pushInfo("── Phase 3: Enforcement ─────────────────────────")

        # Eligible cells are hydrography-only divergence cells.
        # Aligned, tolerated and conflict cells are excluded.
        # The native enforce_hydrography() further excludes conflict
        # cells even when eligible_mask is True.

        hydrography_only_mask = (
            divergence_result["divergence_mask"].astype(bool)
            & hydrography_mask
            & ~divergence_result["aligned_mask"].astype(bool)
            & ~divergence_result["tolerated_mask"].astype(bool)
        )

        eligible_mask = hydrography_only_mask & ~conflict_mask

        eligible_cell_count = int(np.sum(eligible_mask))
        feedback.pushInfo(f"Eligible cells for enforcement: {eligible_cell_count}.")

        if eligible_cell_count == 0:
            warnings.append(
                "No cells are eligible for enforcement. "
                "The enforced DEM will be identical to the input DEM. "
                "Consider adjusting positional tolerance or reviewing "
                "the hydrography and flow-evidence inputs."
            )
            feedback.pushWarning(warnings[-1])

        # ── Phase 3: Resolve NoData sentinel ─────────────────────────────────

        nodata_sentinel = (
            float(dem_metadata["nodata"])
            if dem_metadata["nodata"] is not None
            else -9999.0
        )

        # ── Phase 3: Cell size ────────────────────────────────────────────────

        cell_size = float(
            (dem_metadata["resolution_x"] + dem_metadata["resolution_y"]) / 2.0
        )

        # ── Phase 3: Call enforce_hydrography ─────────────────────────────────

        feedback.setProgress(70)
        feedback.pushInfo(
            f"Calling enforce_hydrography. "
            f"Cell size: {cell_size:.4f}. "
            f"Vertical accuracy: {vertical_accuracy}. "
            f"Maximum burn depth: {maximum_burn_depth}."
        )

        try:
            enforced_dem, difference, enforcement_mask, audit = enforce_hydrography(
                dem=dem_array,
                hydrography_mask=hydrography_mask,
                eligible_mask=eligible_mask,
                cell_size=cell_size,
                vertical_accuracy=vertical_accuracy,
                maximum_burn_depth=maximum_burn_depth,
                nodata=nodata_sentinel,
                upstream_area=None,
                reference_upstream_area=None,
                conflict_mask=conflict_mask,
            )
        except ValueError as exc:
            raise QgsProcessingException(
                f"Hydrography enforcement failed: {exc}"
            ) from exc

        feedback.pushInfo(
            f"Enforcement complete. " f"Modified cells   : {audit['modified_cells']}. "
        )
        feedback.pushInfo(f"Total lowering   : {audit['total_lowering']:.4f}. ")
        feedback.pushInfo(f"Maximum lowering : {audit['maximum_lowering']:.4f}. ")
        feedback.pushInfo(f"Mean lowering    : {audit['mean_lowering']:.4f}.")

        # ── Phase 3: Write enforced DEM raster ───────────────────────────────

        feedback.setProgress(80)

        dem_stem = Path(dem_source).stem

        enforced_dem_path = output_folder / f"{dem_stem}_hydrography_enforced.tif"
        difference_path = output_folder / f"{dem_stem}_hydrography_difference.tif"
        enforcement_mask_path = (
            output_folder / f"{dem_stem}_hydrography_enforcement_mask.tif"
        )

        with rasterio.open(dem_source) as dem_dataset:
            profile = dem_dataset.profile.copy()

        profile.update(dtype=np.float64, count=1, compress="deflate")

        with rasterio.open(enforced_dem_path, "w", **profile) as dst:
            dst.write(enforced_dem, 1)

        feedback.pushInfo(f"Enforced DEM written: {enforced_dem_path}")

        with rasterio.open(difference_path, "w", **profile) as dst:
            dst.write(difference, 1)

        feedback.pushInfo(f"Difference raster written: {difference_path}")

        mask_profile = profile.copy()
        mask_profile.update(dtype=np.uint8, nodata=255)

        with rasterio.open(enforcement_mask_path, "w", **mask_profile) as dst:
            dst.write(enforcement_mask, 1)

        feedback.pushInfo(f"Enforcement mask written: {enforcement_mask_path}")

        # ── Build report ──────────────────────────────────────────────────────

        feedback.setProgress(88)

        report_path = output_folder / f"{dem_stem}_hydrography_enforcement_report.txt"
        provenance_path = (
            output_folder / f"{dem_stem}_hydrography_enforcement_provenance.json"
        )

        report_lines = [
            "═" * 72,
            "  MAYIM TOOLS — Stage 7E: DEM Hydrography Enforcement",
            "  Full Enforcement Report",
            "═" * 72,
            "",
            f"  Run timestamp (UTC) : {run_timestamp}",
            f"  Output folder       : {output_folder}",
            "",
            "── Inputs ───────────────────────────────────────────────────────",
            f"  DEM source                  : {dem_source}",
            f"  Hydrography source          : {hydrography_layer.source()}",
            f"  Flow evidence source        : {flow_evidence_source}",
            f"  Endpoint tolerance          : {endpoint_tolerance}",
            f"  Positional tolerance cells  : {positional_tolerance_cells}",
            f"  Vertical accuracy           : {vertical_accuracy}",
            f"  Maximum burn depth          : {maximum_burn_depth}",
            "",
            "── CRS ──────────────────────────────────────────────────────────",
            f"  DEM CRS             : {dem_crs.authid()}",
            f"  Hydrography CRS     : {hydrography_crs.authid()}",
            f"  CRS match           : {crs_match}",
            "",
            "── Hydrography Features ─────────────────────────────────────────",
            f"  Total feature count             : {feature_count}",
            f"  Features inside DEM extent      : {features_inside}",
            f"  Features outside DEM extent     : {features_outside}",
            "",
            "── Validation ───────────────────────────────────────────────────",
        ]

        if validation_result:
            for key, value in validation_result.items():
                report_lines.append(f"  {key:<34}: {value}")
        else:
            report_lines.append("  No validation detail returned.")

        report_lines.extend(
            [
                "",
                "── Topology ─────────────────────────────────────────────────────",
            ]
        )

        if topology_result:
            for key, value in topology_result.items():
                report_lines.append(f"  {key:<34}: {value}")
        else:
            report_lines.append("  No topology detail returned.")

        report_lines.extend(
            [
                "",
                "── Phase 1: Grid Alignment ──────────────────────────────────────",
                (
                    f"  DEM grid shape      : {dem_metadata['height']} rows x "
                    f"{dem_metadata['width']} columns"
                ),
                f"  DEM resolution X    : {dem_metadata['resolution_x']:.6f}",
                f"  DEM resolution Y    : {dem_metadata['resolution_y']:.6f}",
                f"  DEM dtype           : {dem_metadata['dtype']}",
                f"  DEM NoData value    : {dem_metadata['nodata']}",
                f"  DEM CRS (rasterio)  : {dem_metadata['crs']}",
                f"  DEM transform       : {dem_metadata['transform']}",
                f"  Hydrography burned cells        : {burned_cell_count}",
                (
                    f"  Flow-evidence shape             : "
                    f"{flow_evidence_array.shape[0]} rows x "
                    f"{flow_evidence_array.shape[1]} columns"
                ),
                "  Flow-evidence alignment         : verified — matches DEM grid",
                "",
                "── Phase 2: Divergence Analysis ─────────────────────────────────",
                (
                    f"  Assessable cells            : "
                    f"{divergence_stats['assessable_cells']}"
                ),
                (
                    f"  NoData cells                : "
                    f"{divergence_stats['nodata_cells']}"
                ),
                (
                    f"  Hydrography cells           : "
                    f"{divergence_stats['hydrography_cells']}"
                ),
                (
                    f"  DEM flow cells              : "
                    f"{divergence_stats['dem_flow_cells']}"
                ),
                (
                    f"  Aligned cells (exact)       : "
                    f"{divergence_stats['aligned_cells']}"
                ),
                (
                    f"  Tolerated cells             : "
                    f"{divergence_stats['tolerated_cells']}"
                ),
                (
                    f"  Hydrography-only cells      : "
                    f"{divergence_stats['hydrography_only_cells']}"
                ),
                (
                    f"  DEM-only cells              : "
                    f"{divergence_stats['dem_only_cells']}"
                ),
                (
                    f"  Material divergence cells   : "
                    f"{divergence_stats['material_divergence_cells']}"
                ),
                (
                    f"  Conflict cells              : "
                    f"{divergence_stats['conflict_cells']}"
                ),
                (
                    f"  Positional tolerance used   : "
                    f"{divergence_stats['positional_tolerance_cells']} cells"
                ),
                "",
                "── Phase 3: Enforcement ─────────────────────────────────────────",
                f"  Cell size                       : {cell_size:.4f}",
                f"  Vertical accuracy               : {vertical_accuracy}",
                f"  Maximum burn depth configured   : {maximum_burn_depth}",
                (
                    f"  Base burn depth applied         : "
                    f"{audit['base_burn_depth']:.4f}"
                ),
                f"  Eligible cells                  : {eligible_cell_count}",
                (
                    f"  Authorised hydrography cells    : "
                    f"{audit['authorised_hydrography_cells']}"
                ),
                f"  Modified cells                  : {audit['modified_cells']}",
                (
                    f"  Conflict excluded cells         : "
                    f"{audit['conflict_excluded_cells']}"
                ),
                (
                    f"  Total lowering                  : "
                    f"{audit['total_lowering']:.4f}"
                ),
                (
                    f"  Maximum lowering                : "
                    f"{audit['maximum_lowering']:.4f}"
                ),
                (
                    f"  Mean lowering                   : "
                    f"{audit['mean_lowering']:.4f}"
                ),
                (
                    f"  Minimum lowering                : "
                    f"{audit['minimum_lowering']:.4f}"
                ),
                (
                    f"  Area scaling used               : "
                    f"{audit['area_scaling_used']}"
                ),
                "",
                "── Outputs ──────────────────────────────────────────────────────",
                f"  Enforced DEM        : {enforced_dem_path}",
                f"  Difference raster   : {difference_path}",
                f"  Enforcement mask    : {enforcement_mask_path}",
                "",
                "── Enforcement Mask Legend ──────────────────────────────────────",
                "  0   — unchanged non-hydrography cell",
                "  1   — hydrography enforced (elevation lowered)",
                "  2   — hydrography present but not enforced",
                "  3   — conflict/review excluded",
                "  255 — NoData",
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
                "── Enforcement Status ───────────────────────────────────────────",
                "  Elevation modification applied.",
                f"  {audit['modified_cells']} cell(s) were lowered.",
                "  Area-scaled burning: not yet connected.",
                "  Review modified_cell_records in provenance for cell detail.",
                "",
                "═" * 72,
                "  End of report.",
                "═" * 72,
            ]
        )

        report_text = "\n".join(report_lines)
        report_path.write_text(report_text, encoding="utf-8")
        feedback.pushInfo(f"Report written: {report_path}")

        # ── Build provenance record ───────────────────────────────────────────

        feedback.setProgress(94)

        provenance = {
            "tool": "DEMHydrographyEnforcement",
            "stage": "7E",
            "phase": "Phase 3 — Adaptive Enforcement",
            "run_timestamp_utc": run_timestamp,
            "inputs": {
                "dem_source": dem_source,
                "hydrography_source": hydrography_layer.source(),
                "flow_evidence_source": flow_evidence_source,
                "endpoint_tolerance": endpoint_tolerance,
                "positional_tolerance_cells": positional_tolerance_cells,
                "vertical_accuracy": vertical_accuracy,
                "maximum_burn_depth": maximum_burn_depth,
            },
            "crs": {
                "dem_crs": dem_crs.authid(),
                "hydrography_crs": hydrography_crs.authid(),
                "crs_match": crs_match,
            },
            "features": {
                "total_feature_count": feature_count,
                "features_inside_dem_extent": features_inside,
                "features_outside_dem_extent": features_outside,
            },
            "phase_1_grid_alignment": {
                "dem_height": dem_metadata["height"],
                "dem_width": dem_metadata["width"],
                "dem_resolution_x": dem_metadata["resolution_x"],
                "dem_resolution_y": dem_metadata["resolution_y"],
                "dem_dtype": dem_metadata["dtype"],
                "dem_nodata": dem_metadata["nodata"],
                "dem_crs": str(dem_metadata["crs"]),
                "dem_transform": list(dem_metadata["transform"]),
                "hydrography_burned_cells": burned_cell_count,
                "flow_evidence_height": int(flow_evidence_array.shape[0]),
                "flow_evidence_width": int(flow_evidence_array.shape[1]),
                "flow_evidence_alignment": "verified",
            },
            "phase_2_divergence": {
                "assessable_cells": divergence_stats["assessable_cells"],
                "nodata_cells": divergence_stats["nodata_cells"],
                "hydrography_cells": divergence_stats["hydrography_cells"],
                "dem_flow_cells": divergence_stats["dem_flow_cells"],
                "aligned_cells": divergence_stats["aligned_cells"],
                "tolerated_cells": divergence_stats["tolerated_cells"],
                "hydrography_only_cells": divergence_stats["hydrography_only_cells"],
                "dem_only_cells": divergence_stats["dem_only_cells"],
                "material_divergence_cells": divergence_stats[
                    "material_divergence_cells"
                ],
                "conflict_cells": divergence_stats["conflict_cells"],
                "positional_tolerance_cells": divergence_stats[
                    "positional_tolerance_cells"
                ],
            },
            "phase_3_enforcement": {
                "cell_size": cell_size,
                "vertical_accuracy": vertical_accuracy,
                "maximum_burn_depth_configured": maximum_burn_depth,
                "base_burn_depth_applied": audit["base_burn_depth"],
                "eligible_cells": eligible_cell_count,
                "authorised_hydrography_cells": audit["authorised_hydrography_cells"],
                "modified_cells": audit["modified_cells"],
                "conflict_excluded_cells": audit["conflict_excluded_cells"],
                "eligible_not_modified_cells": audit["eligible_not_modified_cells"],
                "total_lowering": audit["total_lowering"],
                "maximum_lowering": audit["maximum_lowering"],
                "mean_lowering": audit["mean_lowering"],
                "minimum_lowering": audit["minimum_lowering"],
                "total_signed_change": audit["total_signed_change"],
                "area_scaling_used": audit["area_scaling_used"],
                "modified_cell_records": audit["modified_cell_records"],
            },
            "enforcement_status": {
                "elevation_modified": audit["modified_cells"] > 0,
                "modified_cells": audit["modified_cells"],
                "area_scaling_connected": False,
            },
            "outputs": {
                "enforced_dem": str(enforced_dem_path),
                "difference_raster": str(difference_path),
                "enforcement_mask": str(enforcement_mask_path),
                "report": str(report_path),
                "provenance": str(provenance_path),
            },
            "warnings": warnings,
        }

        provenance_path.write_text(
            json.dumps(provenance, indent=4),
            encoding="utf-8",
        )
        feedback.pushInfo(f"Provenance written: {provenance_path}")

        feedback.setProgress(100)
        feedback.pushInfo(
            "Stage 7E complete. "
            f"{audit['modified_cells']} cell(s) enforced. "
            "Review the enforcement report and provenance before "
            "proceeding to downstream conditioning stages."
        )

        return {
            self.OUTPUT_REPORT: str(report_path),
            self.OUTPUT_PROVENANCE: str(provenance_path),
        }

    # ── Private methods ───────────────────────────────────────────────────────

    def _read_dem_metadata(
        self,
        dem_path: str,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Read and return DEM grid metadata."""
        import rasterio

        feedback.pushInfo("Reading DEM grid metadata.")

        with rasterio.open(dem_path) as dem_dataset:
            metadata = {
                "path": dem_path,
                "width": dem_dataset.width,
                "height": dem_dataset.height,
                "transform": dem_dataset.transform,
                "crs": dem_dataset.crs,
                "nodata": dem_dataset.nodata,
                "resolution_x": dem_dataset.res[0],
                "resolution_y": dem_dataset.res[1],
                "dtype": str(dem_dataset.dtypes[0]),
            }

        feedback.pushInfo(
            f"DEM grid: {metadata['width']} x {metadata['height']} "
            f"at {metadata['resolution_x']:.4f} resolution."
        )

        return metadata

    def _rasterise_hydrography(
        self,
        hydrography_layer,
        crs_transform,
        dem_metadata: dict,
        feedback: QgsProcessingFeedback,
    ) -> np.ndarray:
        """
        Burn vector hydrography lines onto the DEM grid.

        Returns a uint8 NumPy array with 1 where a hydrography line
        intersects the cell and 0 elsewhere. The output grid matches
        the DEM exactly: same shape, transform and CRS.
        """
        from rasterio.features import rasterize
        from shapely import wkt as shapely_wkt
        from shapely.geometry import mapping

        feedback.pushInfo("Rasterising vector hydrography onto DEM grid.")

        height = dem_metadata["height"]
        width = dem_metadata["width"]
        transform = dem_metadata["transform"]

        shapes = []

        for feature in hydrography_layer.getFeatures():
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue
            if crs_transform is not None:
                geometry.transform(crs_transform)
            wkt = geometry.asWkt()
            shapely_geom = shapely_wkt.loads(wkt)
            shapes.append((mapping(shapely_geom), 1))

        if not shapes:
            feedback.reportError("No valid hydrography geometries to rasterise.")
            return np.zeros((height, width), dtype=np.uint8)

        hydrography_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        burned_count = int(np.sum(hydrography_raster > 0))
        feedback.pushInfo(
            f"Hydrography rasterisation complete. " f"Burned cells: {burned_count}."
        )

        return hydrography_raster

    def _read_flow_evidence(
        self,
        flow_evidence_path: str,
        dem_metadata: dict,
        feedback: QgsProcessingFeedback,
    ) -> np.ndarray:
        """
        Read the flow-evidence raster and verify alignment with the DEM.

        Raises QgsProcessingException if dimensions, transform or CRS
        do not match the DEM grid.
        """
        import rasterio

        feedback.pushInfo("Reading flow-evidence raster.")

        with rasterio.open(flow_evidence_path) as flow_dataset:
            flow_width = flow_dataset.width
            flow_height = flow_dataset.height
            flow_transform = flow_dataset.transform
            flow_crs = flow_dataset.crs
            flow_array = flow_dataset.read(1).astype(np.float32)

        # --- Dimension check ---
        if flow_width != dem_metadata["width"] or flow_height != dem_metadata["height"]:
            raise QgsProcessingException(
                f"Flow-evidence raster dimensions "
                f"({flow_width} x {flow_height}) do not match "
                f"DEM dimensions "
                f"({dem_metadata['width']} x {dem_metadata['height']}). "
                "Resample the flow-evidence raster to match the DEM grid."
            )

        # --- Transform check ---
        dem_transform = dem_metadata["transform"]
        transform_tolerance = 1e-6

        transform_mismatch = any(
            abs(flow_transform[i] - dem_transform[i]) > transform_tolerance
            for i in range(6)
        )

        if transform_mismatch:
            raise QgsProcessingException(
                "Flow-evidence raster transform does not match the DEM. "
                "Ensure both rasters share the same origin, resolution "
                "and pixel alignment."
            )

        # --- CRS check ---
        dem_crs = dem_metadata["crs"]

        if flow_crs and dem_crs and flow_crs.to_epsg() != dem_crs.to_epsg():
            feedback.pushWarning(
                f"Flow-evidence CRS (EPSG:{flow_crs.to_epsg()}) "
                f"differs from DEM CRS (EPSG:{dem_crs.to_epsg()}). "
                "Verify that both rasters are in the same projection."
            )

        feedback.pushInfo(
            f"Flow-evidence raster verified. " f"Shape: {flow_height} x {flow_width}."
        )

        return flow_array
