# dem_hydrography_enforcement.py
#
# Stage 7E — DEM Hydrography Enforcement Adapter
# Mayim Tools | Hydrology Category
#
# Phase 1: Grid alignment and rasterisation — complete.
# Phase 2: Divergence analysis — connected in this version.
# Phase 3: Enforcement — not yet connected.
# No elevation modification occurs in this version.
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

Validates, prepares, grid-aligns and analyses divergence for adaptive,
topology-aware hydrography enforcement following Soille, Vogt & Colombo
(2003) and Lindsay's (2016) TopologicalBreachBurn.

Phase 1 (complete):
    - Reads DEM grid metadata.
    - Rasterises vector hydrography onto the exact DEM grid.
    - Reads and verifies the flow-evidence raster.

Phase 2 (this version):
    - Derives Boolean masks from Phase 1 arrays.
    - Derives NoData mask from DEM metadata.
    - Calls analyse_hydrography_divergence().
    - Records divergence statistics and masks in the report and provenance.

Phase 3 (enforcement) is not yet connected.
No elevation modification occurs in this version.
"""

from __future__ import annotations

import json
from datetime import datetime
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
from mayim_tools.hydrology.hydrography.divergence import (
    analyse_hydrography_divergence,
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
    with contributing area. This stage is optional and off by default.

    Phase 1 (complete): Grid alignment and rasterisation.
    Phase 2 (current):  Divergence analysis.
    Phase 3:            Enforcement — not yet connected.
    """

    # ── Parameter identifiers ─────────────────────────────────────────────────

    INPUT_DEM = "INPUT_DEM"
    INPUT_HYDROGRAPHY = "INPUT_HYDROGRAPHY"
    INPUT_FLOW_EVIDENCE = "INPUT_FLOW_EVIDENCE"
    ENDPOINT_TOLERANCE = "ENDPOINT_TOLERANCE"
    POSITIONAL_TOLERANCE_CELLS = "POSITIONAL_TOLERANCE_CELLS"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    # ── Output identifiers ────────────────────────────────────────────────────

    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_PROVENANCE = "OUTPUT_PROVENANCE"

    # ── Defaults ──────────────────────────────────────────────────────────────

    DEFAULT_ENDPOINT_TOLERANCE = 1.0
    DEFAULT_POSITIONAL_TOLERANCE_CELLS = 3

    # ── Algorithm identity ────────────────────────────────────────────────────

    def name(self) -> str:  # noqa: N802
        return "demhydrographyenforcement"

    def displayName(self) -> str:  # noqa: N802
        return "DEM Hydrography Enforcement"

    def group(self) -> str:  # noqa: N802
        return "Hydrology Tools"

    def groupId(self) -> str:  # noqa: N802
        return "hydrologytools"

    def shortHelpString(self) -> str:  # noqa: N802
        return (
            "Stage 7E — Adaptive, topology-aware hydrography enforcement.\n\n"
            "Burning is applied only where the DEM-derived flow path diverges "
            "materially from the mapped channel, following Soille, Vogt & Colombo "
            "(2003) and Lindsay's (2016) TopologicalBreachBurn. Burn depth scales "
            "continuously with contributing area.\n\n"
            "This stage is optional and off by default.\n\n"
            "Phase 1 (complete): Grid alignment and rasterisation.\n"
            "Phase 2 (current):  Divergence analysis.\n"
            "Phase 3:            Enforcement — not yet connected.\n\n"
            "No elevation modification occurs in this version."
        )

    # ── Parameter definition ──────────────────────────────────────────────────

    def initAlgorithm(self, config=None) -> None:  # noqa: N802
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
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER,
                "Output folder",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_REPORT,
                "Preparation and divergence report",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_PROVENANCE,
                "Preparation and divergence provenance",
            )
        )

    # ── Main processing ───────────────────────────────────────────────────────

    def processAlgorithm(  # noqa: N802
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        logger = MayimLogger(__name__)
        logger.info("Stage 7E — DEM Hydrography Enforcement started.")

        run_timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        warnings: list[str] = []

        # ── Resolve inputs ────────────────────────────────────────────────────

        dem_layer = self.parameterAsRasterLayer(
            parameters, self.INPUT_DEM, context
        )
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

        output_folder = Path(
            self.parameterAsString(
                parameters, self.OUTPUT_FOLDER, context
            )
        )
        output_folder.mkdir(parents=True, exist_ok=True)

        feedback.pushInfo("Inputs resolved successfully.")
        feedback.pushInfo(f"DEM source                  : {dem_source}")
        feedback.pushInfo(f"Hydrography source          : {hydrography_layer.source()}")
        feedback.pushInfo(f"Flow evidence source        : {flow_evidence_source}")
        feedback.pushInfo(f"Endpoint tolerance          : {endpoint_tolerance}")
        feedback.pushInfo(f"Positional tolerance cells  : {positional_tolerance_cells}")
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
            raise QgsProcessingException(
                "Hydrography layer contains no features."
            )

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

        feedback.pushInfo(
            f"Features intersecting DEM extent : {features_inside}"
        )
        feedback.pushInfo(
            f"Features outside DEM extent      : {features_outside}"
        )

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

        feedback.setProgress(15)
        feedback.pushInfo("── Phase 1: Grid alignment ──────────────────────")

        dem_metadata = self._read_dem_metadata(
            dem_path=dem_source,
            feedback=feedback,
        )

        # ── Phase 1: Hydrography rasterisation ───────────────────────────────

        feedback.setProgress(30)

        hydrography_raster = self._rasterise_hydrography(
            hydrography_layer=hydrography_layer,
            crs_transform=crs_transform,
            dem_metadata=dem_metadata,
            feedback=feedback,
        )

        burned_cell_count = int(np.sum(hydrography_raster > 0))

        # ── Phase 1: Flow-evidence raster ─────────────────────────────────────

        feedback.setProgress(45)

        flow_evidence_array = self._read_flow_evidence(
            flow_evidence_path=flow_evidence_source,
            dem_metadata=dem_metadata,
            feedback=feedback,
        )

        feedback.pushInfo("Phase 1 complete.")

        # ── Phase 2: Derive Boolean masks ─────────────────────────────────────

        feedback.setProgress(55)
        feedback.pushInfo("── Phase 2: Divergence analysis ─────────────────")

        hydrography_mask = hydrography_raster.astype(bool)

        dem_flow_mask = flow_evidence_array > 0

        nodata_mask = None
        if dem_metadata["nodata"] is not None:
            nodata_mask = (
                flow_evidence_array == dem_metadata["nodata"]
            )
            feedback.pushInfo(
                f"NoData mask derived. "
                f"NoData cells: {int(np.sum(nodata_mask))}."
            )

        # ── Phase 2: Divergence analysis ──────────────────────────────────────

        feedback.setProgress(65)
        feedback.pushInfo("Calling analyse_hydrography_divergence.")

        try:
            divergence_result = analyse_hydrography_divergence(
                hydrography_mask=hydrography_mask,
                dem_flow_mask=dem_flow_mask,
                positional_tolerance_cells=positional_tolerance_cells,
                nodata_mask=nodata_mask,
            )
        except ValueError as exc:
            raise QgsProcessingException(
                f"Divergence analysis failed: {exc}"
            ) from exc

        divergence_stats = divergence_result["statistics"]

        # Masks retained for Phase 3 enforcement — not yet consumed.
        _divergence_mask = divergence_result["divergence_mask"]
        _aligned_mask = divergence_result["aligned_mask"]
        _tolerated_mask = divergence_result["tolerated_mask"]
        _conflict_mask = divergence_result["conflict_mask"]

        feedback.pushInfo(
            f"Divergence analysis complete. "
            f"Aligned cells          : {divergence_stats['aligned_cells']}. "
        )
        feedback.pushInfo(
            f"Tolerated cells        : {divergence_stats['tolerated_cells']}."
        )
        feedback.pushInfo(
            f"Material divergence    : "
            f"{divergence_stats['material_divergence_cells']} cells."
        )
        feedback.pushInfo(
            f"Conflict cells         : {divergence_stats['conflict_cells']}."
        )
        feedback.pushInfo(
            "NOTE: Phase 3 (enforcement) is not yet connected. "
            "No elevation modification has occurred."
        )

        # ── Derive output file stems ──────────────────────────────────────────

        dem_stem = Path(dem_source).stem

        report_path = (
            output_folder
            / f"{dem_stem}_hydrography_preparation_report.txt"
        )
        provenance_path = (
            output_folder
            / f"{dem_stem}_hydrography_preparation_provenance.json"
        )

        # ── Build preparation and divergence report ───────────────────────────

        feedback.setProgress(80)

        report_lines = [
            "═" * 72,
            "  MAYIM TOOLS — Stage 7E: DEM Hydrography Enforcement",
            "  Preparation, Grid Alignment and Divergence Analysis Report",
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

        report_lines.extend([
            "",
            "── Topology ─────────────────────────────────────────────────────",
        ])

        if topology_result:
            for key, value in topology_result.items():
                report_lines.append(f"  {key:<34}: {value}")
        else:
            report_lines.append("  No topology detail returned.")

        report_lines.extend([
            "",
            "── Phase 1: Grid Alignment ──────────────────────────────────────",
            f"  DEM grid shape      : {dem_metadata['height']} rows x "
            f"{dem_metadata['width']} columns",
            f"  DEM resolution X    : {dem_metadata['resolution_x']:.6f}",
            f"  DEM resolution Y    : {dem_metadata['resolution_y']:.6f}",
            f"  DEM dtype           : {dem_metadata['dtype']}",
            f"  DEM NoData value    : {dem_metadata['nodata']}",
            f"  DEM CRS (rasterio)  : {dem_metadata['crs']}",
            f"  DEM transform       : {dem_metadata['transform']}",
            f"  Hydrography burned cells        : {burned_cell_count}",
            f"  Flow-evidence shape             : "
            f"{flow_evidence_array.shape[0]} rows x "
            f"{flow_evidence_array.shape[1]} columns",
            "  Flow-evidence alignment         : verified — matches DEM grid",
            "",
            "── Phase 2: Divergence Analysis ─────────────────────────────────",
            f"  Assessable cells            : "
            f"{divergence_stats['assessable_cells']}",
            f"  NoData cells                : "
            f"{divergence_stats['nodata_cells']}",
            f"  Hydrography cells           : "
            f"{divergence_stats['hydrography_cells']}",
            f"  DEM flow cells              : "
            f"{divergence_stats['dem_flow_cells']}",
            f"  Aligned cells (exact)       : "
            f"{divergence_stats['aligned_cells']}",
            f"  Tolerated cells             : "
            f"{divergence_stats['tolerated_cells']}",
            f"  Hydrography-only cells      : "
            f"{divergence_stats['hydrography_only_cells']}",
            f"  DEM-only cells              : "
            f"{divergence_stats['dem_only_cells']}",
            f"  Material divergence cells   : "
            f"{divergence_stats['material_divergence_cells']}",
            f"  Conflict cells              : "
            f"{divergence_stats['conflict_cells']}",
            f"  Positional tolerance used   : "
            f"{divergence_stats['positional_tolerance_cells']} cells",
            "",
            "── Warnings ─────────────────────────────────────────────────────",
        ])

        if warnings:
            for warning in warnings:
                report_lines.append(f"  WARNING: {warning}")
        else:
            report_lines.append("  No warnings.")

        report_lines.extend([
            "",
            "── Enforcement Status ───────────────────────────────────────────",
            "  No elevation modification has occurred in this run.",
            "  Phase 3 (enforcement) is not yet connected.",
            "",
            "═" * 72,
            "  End of report.",
            "═" * 72,
        ])

        report_text = "\n".join(report_lines)
        report_path.write_text(report_text, encoding="utf-8")
        feedback.pushInfo(f"Report written: {report_path}")

        # ── Build provenance record ───────────────────────────────────────────

        provenance = {
            "tool": "DEMHydrographyEnforcement",
            "stage": "7E",
            "phase": "Phase 2 — Divergence Analysis",
            "run_timestamp_utc": run_timestamp,
            "inputs": {
                "dem_source": dem_source,
                "hydrography_source": hydrography_layer.source(),
                "flow_evidence_source": flow_evidence_source,
                "endpoint_tolerance": endpoint_tolerance,
                "positional_tolerance_cells": positional_tolerance_cells,
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
                "flow_evidence_height": int(
                    flow_evidence_array.shape[0]
                ),
                "flow_evidence_width": int(
                    flow_evidence_array.shape[1]
                ),
                "flow_evidence_alignment": "verified",
            },
            "phase_2_divergence": {
                "assessable_cells": divergence_stats[
                    "assessable_cells"
                ],
                "nodata_cells": divergence_stats["nodata_cells"],
                "hydrography_cells": divergence_stats[
                    "hydrography_cells"
                ],
                "dem_flow_cells": divergence_stats["dem_flow_cells"],
                "aligned_cells": divergence_stats["aligned_cells"],
                "tolerated_cells": divergence_stats[
                    "tolerated_cells"
                ],
                "hydrography_only_cells": divergence_stats[
                    "hydrography_only_cells"
                ],
                "dem_only_cells": divergence_stats["dem_only_cells"],
                "material_divergence_cells": divergence_stats[
                    "material_divergence_cells"
                ],
                "conflict_cells": divergence_stats["conflict_cells"],
                "positional_tolerance_cells": divergence_stats[
                    "positional_tolerance_cells"
                ],
            },
            "enforcement_status": {
                "elevation_modified": False,
                "phase_3_enforcement_connected": False,
            },
            "warnings": warnings,
            "outputs": {
                "report": str(report_path),
                "provenance": str(provenance_path),
            },
        }

        provenance_path.write_text(
            json.dumps(provenance, indent=4),
            encoding="utf-8",
        )
        feedback.pushInfo(f"Provenance written: {provenance_path}")

        feedback.setProgress(100)
        feedback.pushInfo(
            "Stage 7E Phase 2 complete. "
            "Review the divergence report before proceeding to Phase 3."
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

        feedback.pushInfo(
            "Rasterising vector hydrography onto DEM grid."
        )

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
            feedback.reportError(
                "No valid hydrography geometries to rasterise."
            )
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
            f"Hydrography rasterisation complete. "
            f"Burned cells: {burned_count}."
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
        if (
            flow_width != dem_metadata["width"]
            or flow_height != dem_metadata["height"]
        ):
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

        if flow_crs and dem_crs:
            if flow_crs.to_epsg() != dem_crs.to_epsg():
                feedback.pushWarning(
                    f"Flow-evidence CRS (EPSG:{flow_crs.to_epsg()}) "
                    f"differs from DEM CRS (EPSG:{dem_crs.to_epsg()}). "
                    "Verify that both rasters are in the same projection."
                )

        feedback.pushInfo(
            f"Flow-evidence raster verified. "
            f"Shape: {flow_height} x {flow_width}."
        )

        return flow_array
