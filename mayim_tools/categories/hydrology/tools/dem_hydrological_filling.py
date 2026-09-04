"""
Mayim Tools - DEM Hydrological Filling
======================================

Implements Stage 5 of the Mayim Tools DEM hydrological-conditioning
methodology.

Stage 5:
    Selective Flow Enforcement

This tool consumes depression analysis outputs and applies selective
terrain modification according to the Stage 4 classification:

    ARTIFACT
        -> single-cell de-pitting, or
        -> constrained least-cost breaching, or
        -> confined filling fallback

    REAL_FEATURE
        -> preserved

    REVIEW_REQUIRED
        -> preserved pending analyst review

Outputs
-------
- Depression-preserving DEM
- Hydrology-ready DEM
- Enforcement decision raster
- Enforcement report
- Enforcement provenance
- Derived MayimManifest

Important
---------
This tool modifies terrain. It therefore follows the revised Mayim
research paper's clean-room rule: all terrain-modifying logic is
implemented in-house and does not call WhiteboxTools, RichDEM or
another hydrological runtime.

IP status
---------
Original Mayim implementation using:

    - Python standard-library components
    - NumPy
    - rasterio
    - QGIS Processing API
    - Native Mayim Stage 5 enforcement modules

No third-party hydrological runtime is used.
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
from mayim_tools.hydrology.enforcement.enforcement import (
    DECISION_BREACHED,
    DECISION_DEPITTED,
    DECISION_FILLED,
    DECISION_REAL_PRESERVED,
    DECISION_REVIEW_PRESERVED,
    enforce_selectively,
)
from mayim_tools.processing.algorithms.base_algorithm import (
    MayimBaseAlgorithm,
)


class DEMHydrologicalFilling(MayimBaseAlgorithm):
    """
    QGIS Processing adapter for native Stage 5 selective flow
    enforcement.

    This adapter reads the Stage 3/4 outputs produced by
    DEM Depression Analysis and passes them into the native
    Mayim enforcement modules.
    """

    PARAM_DEM = "INPUT_DEM"
    PARAM_MANIFEST = "INPUT_MANIFEST"
    PARAM_INVENTORY = "INPUT_INVENTORY"
    PARAM_MAX_BREACH_LENGTH = "MAX_BREACH_LENGTH"
    PARAM_MAX_BREACH_DEPTH = "MAX_BREACH_DEPTH"
    PARAM_CONNECTIVITY = "CONNECTIVITY"
    PARAM_OUTPUT_FOLDER = "OUTPUT_FOLDER"
    PARAM_LOAD_PRESERVED = "LOAD_PRESERVED_DEM"
    PARAM_LOAD_READY = "LOAD_READY_DEM"
    PARAM_LOAD_DECISIONS = "LOAD_DECISION_RASTER"

    OUTPUT_PRESERVED = "OUTPUT_PRESERVED"
    OUTPUT_READY = "OUTPUT_READY"
    OUTPUT_DECISIONS = "OUTPUT_DECISIONS"
    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_PROVENANCE = "OUTPUT_PROVENANCE"
    OUTPUT_MANIFEST = "OUTPUT_MANIFEST"

    TOOL_VERSION = "dem-hydrological-filling-0.2.0"

    def name(self) -> str:
        return "demhydrologicalfilling"

    def displayName(self) -> str:
        return "DEM Hydrological Filling"

    def group(self) -> str:
        return "Hydrology Tools"

    def groupId(self) -> str:
        return "hydrology"

    def createInstance(self) -> DEMHydrologicalFilling:
        return DEMHydrologicalFilling()

    def shortHelpString(self) -> str:
        return (
            "<b>DEM Hydrological Filling</b><br><br>"
            "Implements Stage 5 selective flow enforcement.<br><br>"
            "This tool reads a DEM together with the depression "
            "inventory produced by DEM Depression Analysis and applies "
            "the Stage 4 classification decisions:<br>"
            "ARTIFACT -> de-pit / breach / fill<br>"
            "REAL_FEATURE -> preserve<br>"
            "REVIEW_REQUIRED -> preserve pending review<br><br>"
            "Outputs a depression-preserving DEM, a hydrology-ready "
            "DEM, an enforcement decision raster, a text report, "
            "provenance, and a MayimManifest."
        )

    def helpUrl(self) -> str:
        return "https://github.com/chrismayim/mayim-tools"

    def tags(self) -> list[str]:
        return [
            "mayim",
            "dem",
            "hydrology",
            "filling",
            "breaching",
            "depitting",
            "stage-5",
            "enforcement",
        ]

    def initAlgorithm(self, config=None) -> None:
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
            QgsProcessingParameterFile(
                self.PARAM_INVENTORY,
                "Depression inventory JSON from DEM Depression Analysis",
                behavior=QgsProcessingParameterFile.File,
                extension="json",
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_MAX_BREACH_LENGTH,
                "Maximum breach length in cells (recommended: 250)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=250,
                minValue=1,
                maxValue=100000,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_MAX_BREACH_DEPTH,
                "Maximum breach depth in elevation units " "(recommended: 5.0)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=5.0,
                minValue=0.000001,
                maxValue=100000.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_CONNECTIVITY,
                "Connectivity (4 or 8; recommended: 8)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                minValue=4,
                maxValue=8,
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
                self.PARAM_LOAD_PRESERVED,
                "Load depression-preserving DEM into project",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_LOAD_READY,
                "Load hydrology-ready DEM into project",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_LOAD_DECISIONS,
                "Load decision raster into project",
                defaultValue=True,
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_PRESERVED,
                "Depression-preserving DEM",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_READY,
                "Hydrology-ready DEM",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_DECISIONS,
                "Enforcement decision raster",
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

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_MANIFEST,
                "Enforcement MayimManifest",
            )
        )

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
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

        manifest_path = self.parameterAsString(
            parameters,
            self.PARAM_MANIFEST,
            context,
        )

        inventory_path = self.parameterAsString(
            parameters,
            self.PARAM_INVENTORY,
            context,
        )

        max_breach_length = self.parameterAsInt(
            parameters,
            self.PARAM_MAX_BREACH_LENGTH,
            context,
        )

        max_breach_depth = self.parameterAsDouble(
            parameters,
            self.PARAM_MAX_BREACH_DEPTH,
            context,
        )

        connectivity = self.parameterAsInt(
            parameters,
            self.PARAM_CONNECTIVITY,
            context,
        )

        output_folder = self.parameterAsString(
            parameters,
            self.PARAM_OUTPUT_FOLDER,
            context,
        )

        load_preserved = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_PRESERVED,
            context,
        )

        load_ready = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_READY,
            context,
        )

        load_decisions = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_DECISIONS,
            context,
        )

        if connectivity not in (4, 8):
            raise QgsProcessingException("Connectivity must be either 4 or 8.")

        if not output_folder or output_folder == "TEMPORARY_OUTPUT":
            import tempfile

            output_folder = tempfile.mkdtemp(prefix="mayim_enforcement_")

        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        dem_stem = Path(dem_layer.source()).stem

        paths = {
            "preserved": output_dir / f"{dem_stem}_preserved_topology.tif",
            "ready": output_dir / f"{dem_stem}_hydrology_ready.tif",
            "decisions": output_dir / f"{dem_stem}_enforcement_decisions.tif",
            "difference": output_dir / f"{dem_stem}_enforcement_difference.tif",
            "report": output_dir / f"{dem_stem}_enforcement_report.txt",
            "provenance": output_dir / f"{dem_stem}_enforcement_provenance.json",
            "manifest": output_dir / f"{dem_stem}_enforcement.manifest.json",
        }

        input_manifest = None
        if manifest_path and Path(manifest_path).exists():
            try:
                input_manifest = MayimManifest.read(manifest_path)
            except Exception as error:
                self.log_warning(
                    f"Could not read input manifest: {error}",
                    feedback,
                )

        if not Path(inventory_path).exists():
            raise QgsProcessingException(
                f"Depression inventory not found: {inventory_path}"
            )

        self.log("=" * 60, feedback)
        self.log("STAGE 5 - SELECTIVE FLOW ENFORCEMENT", feedback)
        self.log("=" * 60, feedback)
        self.log(f"Input DEM            : {dem_layer.source()}", feedback)
        self.log(f"Inventory            : {inventory_path}", feedback)
        self.log(f"Max breach length    : {max_breach_length}", feedback)
        self.log(f"Max breach depth     : {max_breach_depth}", feedback)
        self.log(f"Connectivity         : {connectivity}", feedback)

        feedback.setProgress(5)

        with open(inventory_path, encoding="utf-8") as file:
            inventory = json.load(file)

        with rasterio.open(dem_layer.source()) as source:
            profile = source.profile.copy()
            dem = source.read(1).astype(np.float64)
            nodata = source.nodata if source.nodata is not None else -9999.0
            width = source.width
            height = source.height
            res_x = abs(float(source.transform.a))
            res_y = abs(float(source.transform.e))
            cell_size = (res_x + res_y) / 2.0
            crs_string = source.crs.to_string() if source.crs is not None else "Unknown"

        feedback.setProgress(15)

        if self.is_cancelled(feedback):
            return {}

        # ── Build depression records and masks from inventory ─────── #

        self.log("", feedback)
        self.log("Building depression records from inventory...", feedback)

        depression_records = {}
        depression_masks = {}

        for depression_id_str, entry in inventory.items():
            depression_id = int(depression_id_str)
            features = entry.get("features", {})
            classification_data = entry.get("classification", {})

            if not features:
                self.log_warning(
                    f"Depression {depression_id} has no features. " "Skipping.",
                    feedback,
                )
                continue

            if not classification_data:
                self.log_warning(
                    f"Depression {depression_id} has no classification. "
                    "Treating as REVIEW_REQUIRED.",
                    feedback,
                )
                classification_label = "REVIEW_REQUIRED"
            else:
                classification_label = classification_data.get(
                    "classification",
                    "REVIEW_REQUIRED",
                )

            pit_row = int(features.get("pit_row", 0))
            pit_col = int(features.get("pit_col", 0))
            spill_elevation = float(
                features.get("spill_elevation", float(dem[pit_row, pit_col]))
            )
            area_cells = int(features.get("area_cells", 1))

            depression_records[depression_id] = {
                "classification": classification_label,
                "pit_row": pit_row,
                "pit_col": pit_col,
                "spill_elevation": spill_elevation,
                "area_cells": area_cells,
            }

            # Rebuild the depression mask from the depression ID raster
            # stored in the inventory path's directory.
            # The depression ID raster is expected alongside the inventory.
            inventory_dir = Path(inventory_path).parent
            dem_stem_from_inventory = Path(inventory_path).stem.replace(
                "_depression_inventory", ""
            )
            id_raster_path = (
                inventory_dir / f"{dem_stem_from_inventory}_depression_ids.tif"
            )

            if not id_raster_path.exists():
                self.log_warning(
                    f"Depression ID raster not found at: "
                    f"{id_raster_path}. "
                    f"Depression {depression_id} will be skipped.",
                    feedback,
                )
                continue

            with rasterio.open(str(id_raster_path)) as id_source:
                id_array = id_source.read(1)

            depression_masks[depression_id] = id_array == depression_id

        if not depression_records:
            warning = (
                "No valid depression records were found in the inventory. "
                "The output surfaces will be identical to the input DEM."
            )
            self.log_warning(warning, feedback)

        self.log(
            f"Depression records loaded: {len(depression_records)}",
            feedback,
        )

        feedback.setProgress(30)

        if self.is_cancelled(feedback):
            return {}

        # ── Run selective enforcement ──────────────────────────────── #

        self.log("", feedback)
        self.log("=" * 60, feedback)
        self.log("STAGE 5 - SELECTIVE ENFORCEMENT", feedback)
        self.log("=" * 60, feedback)

        try:
            preserved_dem, hydrology_ready_dem, decision_codes, audits = (
                enforce_selectively(
                    dem=dem,
                    depression_records=depression_records,
                    depression_masks=depression_masks,
                    max_breach_length=max_breach_length,
                    max_breach_depth=max_breach_depth,
                    nodata=nodata,
                    connectivity=connectivity,
                )
            )
        except QgsProcessingException:
            raise
        except Exception as error:
            MayimLogger.critical(f"Stage 5 enforcement failed: {error}")
            raise QgsProcessingException(
                f"Stage 5 enforcement failed: {error}"
            ) from error

        # ── Summarise decisions ───────────────────────────────────── #

        depitted_count = sum(
            1 for a in audits if a.get("decision_code") == DECISION_DEPITTED
        )
        breached_count = sum(
            1 for a in audits if a.get("decision_code") == DECISION_BREACHED
        )
        filled_count = sum(
            1 for a in audits if a.get("decision_code") == DECISION_FILLED
        )
        real_count = sum(
            1 for a in audits if a.get("decision_code") == DECISION_REAL_PRESERVED
        )
        review_count = sum(
            1 for a in audits if a.get("decision_code") == DECISION_REVIEW_PRESERVED
        )

        self.log(f"  De-pitted    : {depitted_count}", feedback)
        self.log(f"  Breached     : {breached_count}", feedback)
        self.log(f"  Filled       : {filled_count}", feedback)
        self.log(f"  Preserved    : {real_count}", feedback)
        self.log(f"  Review       : {review_count}", feedback)

        if review_count > 0:
            self.log_warning(
                f"{review_count} depression(s) require analyst review "
                "before using the hydrology-ready surface.",
                feedback,
            )

        feedback.setProgress(65)

        if self.is_cancelled(feedback):
            return {}

        # ── Write raster outputs ──────────────────────────────────── #

        self.log("", feedback)
        self.log("Writing Stage 5 raster outputs...", feedback)

        base_profile = profile.copy()
        base_profile.update(
            dtype="float32",
            count=1,
            compress="lzw",
        )

        if nodata is not None:
            base_profile["nodata"] = nodata

        with rasterio.open(paths["preserved"], "w", **base_profile) as dst:
            dst.write(preserved_dem.astype(np.float32), 1)

        with rasterio.open(paths["ready"], "w", **base_profile) as dst:
            dst.write(hydrology_ready_dem.astype(np.float32), 1)

        difference = (hydrology_ready_dem - dem).astype(np.float32)
        difference_profile = base_profile.copy()
        difference_profile["nodata"] = -9999.0

        diff_out = difference.copy()
        nodata_mask = ~(np.isfinite(dem) & (dem != nodata))
        diff_out[nodata_mask] = -9999.0

        with rasterio.open(paths["difference"], "w", **difference_profile) as dst:
            dst.write(diff_out, 1)

        decision_profile = profile.copy()
        decision_profile.update(
            dtype="uint8",
            count=1,
            compress="lzw",
            nodata=255,
        )

        with rasterio.open(paths["decisions"], "w", **decision_profile) as dst:
            dst.write(decision_codes.astype(np.uint8), 1)

        self.log(
            "Raster outputs written.",
            feedback,
        )

        feedback.setProgress(78)

        # ── Write provenance ──────────────────────────────────────── #

        provenance = {
            "tool": "DEM Hydrological Filling",
            "algorithm": (
                "Selective enforcement: de-pit, constrained breach, "
                "confined fill, preserve"
            ),
            "ip_status": (
                "Original Mayim implementation. No WhiteboxTools or "
                "RichDEM runtime dependency."
            ),
            "version": self.TOOL_VERSION,
            "stage": 5,
            "timestamp": datetime.now().isoformat(),
            "input_dem": dem_layer.source(),
            "input_manifest": manifest_path or None,
            "input_inventory": inventory_path,
            "parameters": {
                "max_breach_length": max_breach_length,
                "max_breach_depth": max_breach_depth,
                "connectivity": connectivity,
            },
            "outputs": {k: str(v) for k, v in paths.items()},
            "statistics": {
                "total_depressions": len(audits),
                "depitted": depitted_count,
                "breached": breached_count,
                "filled": filled_count,
                "real_preserved": real_count,
                "review_preserved": review_count,
                "crs": crs_string,
                "cell_size": cell_size,
                "width": width,
                "height": height,
            },
            "audit_records": audits,
            "warnings": [],
        }

        if review_count > 0:
            provenance["warnings"].append(
                f"{review_count} depression(s) preserved pending " "analyst review."
            )

        with open(paths["provenance"], "w", encoding="utf-8") as file:
            json.dump(provenance, file, indent=4, default=str)

        self.log(
            f"Provenance written: {paths['provenance'].name}",
            feedback,
        )

        feedback.setProgress(84)

        # ── Write text report ─────────────────────────────────────── #

        self._write_text_report(
            paths=paths,
            provenance=provenance,
            feedback=feedback,
        )

        report_path = Path(paths["report"]).resolve()
        report_uri = report_path.as_uri()

        self.log(
            f"Report: {report_path}",
            feedback,
        )
        feedback.pushInfo(f"Open report: {report_uri}")

        feedback.setProgress(88)

        # ── Write MayimManifest ───────────────────────────────────── #

        try:
            if input_manifest is not None:
                manifest = input_manifest.derive(
                    produced_by=self.TOOL_VERSION,
                    raster_path=str(paths["preserved"]),
                    stage=5,
                    audit_log_path=str(paths["provenance"]),
                    warnings=(
                        provenance["warnings"] if provenance["warnings"] else None
                    ),
                    width=width,
                    height=height,
                )
            else:
                manifest = MayimManifest.create(
                    raster_path=str(paths["preserved"]),
                    crs=crs_string,
                    cell_size=cell_size,
                    vertical_accuracy=5.0,
                    nodata=float(nodata),
                    produced_by=self.TOOL_VERSION,
                    stage=5,
                    audit_log_path=str(paths["provenance"]),
                    warnings=(
                        provenance["warnings"] if provenance["warnings"] else None
                    ),
                    width=width,
                    height=height,
                )

            manifest.write(str(paths["manifest"]))

            self.log(
                f"MayimManifest written: {paths['manifest'].name}",
                feedback,
            )

        except Exception as manifest_error:
            self.log_warning(
                f"Could not write MayimManifest: {manifest_error}",
                feedback,
            )

        feedback.setProgress(92)

        # ── Load selected outputs into QGIS ──────────────────────── #

        self._load_layers_into_project(
            paths=paths,
            dem_stem=dem_stem,
            load_preserved=load_preserved,
            load_ready=load_ready,
            load_decisions=load_decisions,
            feedback=feedback,
        )

        feedback.setProgress(100)

        self.log("", feedback)
        self.log("=" * 60, feedback)
        self.log("STAGE 5 COMPLETE", feedback)
        self.log("=" * 60, feedback)

        return {
            self.OUTPUT_PRESERVED: str(paths["preserved"]),
            self.OUTPUT_READY: str(paths["ready"]),
            self.OUTPUT_DECISIONS: str(paths["decisions"]),
            self.OUTPUT_REPORT: str(paths["report"]),
            self.OUTPUT_PROVENANCE: str(paths["provenance"]),
            self.OUTPUT_MANIFEST: str(paths["manifest"]),
        }

    def _load_layers_into_project(
        self,
        paths: dict,
        dem_stem: str,
        load_preserved: bool,
        load_ready: bool,
        load_decisions: bool,
        feedback: QgsProcessingFeedback,
    ) -> None:
        """Load selected outputs directly into the QGIS project."""
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

            if load_preserved:
                load_raster(
                    paths["preserved"],
                    f"{dem_stem} - Preserved Topology",
                )

            if load_ready:
                load_raster(
                    paths["ready"],
                    f"{dem_stem} - Hydrology Ready",
                )

            if load_decisions:
                load_raster(
                    paths["decisions"],
                    f"{dem_stem} - Enforcement Decisions",
                )

            self.log(
                f"{loaded} Stage 5 layer(s) loaded into project.",
                feedback,
            )

        except Exception as error:
            self.log_warning(
                f"Could not load Stage 5 outputs: {error}",
                feedback,
            )

    def _write_text_report(
        self,
        paths: dict,
        provenance: dict,
        feedback: QgsProcessingFeedback,
    ) -> None:
        """Write the Stage 5 human-readable enforcement report."""
        stats = provenance.get("statistics", {})
        params = provenance.get("parameters", {})
        warns = provenance.get("warnings", [])
        audits = provenance.get("audit_records", [])

        lines = []
        a = lines.append

        a("=" * 70)
        a("MAYIM TOOLS - DEM HYDROLOGICAL FILLING REPORT")
        a("=" * 70)
        a(f"Generated  : {provenance.get('timestamp', '')}")
        a(f"Tool       : {provenance.get('tool', '')}")
        a(f"Version    : {provenance.get('version', '')}")
        a(f"Stage      : {provenance.get('stage', 5)}")
        a(f"Input DEM  : {provenance.get('input_dem', '')}")
        a(f"Inventory  : {provenance.get('input_inventory', '')}")
        a(f"IP status  : {provenance.get('ip_status', '')}")
        a("")

        a("-" * 70)
        a("PARAMETERS")
        a("-" * 70)
        a(f"Max breach length    : " f"{params.get('max_breach_length', 250)} cells")
        a(
            f"Max breach depth     : "
            f"{params.get('max_breach_depth', 5.0)} elevation units"
        )
        a(f"Connectivity         : " f"{params.get('connectivity', 8)}")
        a("")

        a("-" * 70)
        a("DEM METADATA")
        a("-" * 70)
        a(f"CRS                  : {stats.get('crs', 'Unknown')}")
        a(f"Cell size            : {stats.get('cell_size', 'Unknown')}")
        a(f"Width                : {stats.get('width', 'Unknown')} cells")
        a(f"Height               : {stats.get('height', 'Unknown')} cells")
        a("")

        a("-" * 70)
        a("STAGE 5 ENFORCEMENT SUMMARY")
        a("-" * 70)
        a(f"Total depressions    : " f"{stats.get('total_depressions', 0)}")
        a(f"De-pitted            : " f"{stats.get('depitted', 0)}")
        a(f"Breached             : " f"{stats.get('breached', 0)}")
        a(f"Filled (fallback)    : " f"{stats.get('filled', 0)}")
        a(f"Real feature preserved: " f"{stats.get('real_preserved', 0)}")
        a(f"Review required      : " f"{stats.get('review_preserved', 0)}")
        a("")

        a("-" * 70)
        a("DECISION RASTER LEGEND")
        a("-" * 70)
        a("  Value 0   = unchanged non-depression cell")
        a("  Value 1   = ARTIFACT de-pitted")
        a("  Value 2   = ARTIFACT breached")
        a("  Value 3   = ARTIFACT filled (fallback)")
        a("  Value 4   = REAL_FEATURE preserved")
        a("  Value 5   = REVIEW_REQUIRED preserved")
        a("  Value 255 = NoData")
        a("")

        a("-" * 70)
        a("PER-DEPRESSION DECISIONS")
        a("-" * 70)

        if audits:
            for audit in audits:
                did = audit.get("depression_id", "?")
                decision = audit.get("decision", "unknown")
                classification = audit.get("classification", "?")
                modified = audit.get("modified", False)
                method = audit.get("method", "none")

                a(
                    f"  Depression {did:>4} : "
                    f"{classification:<16} "
                    f"decision={decision:<22} "
                    f"modified={modified!s:<5} "
                    f"method={method}"
                )
        else:
            a("  No depressions were processed.")

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
        a(f"Preserved topology DEM  : " f"{paths['preserved'].name}")
        a(f"Hydrology-ready DEM     : " f"{paths['ready'].name}")
        a(f"Decision raster         : " f"{paths['decisions'].name}")
        a(f"Report (this file)      : " f"{paths['report'].name}")
        a(f"Provenance              : " f"{paths['provenance'].name}")
        a(f"MayimManifest           : " f"{paths['manifest'].name}")
        a("")

        a("-" * 70)
        a("IMPORTANT DISTINCTION")
        a("-" * 70)
        a(
            "The preserved-topology DEM retains REAL_FEATURE and "
            "REVIEW_REQUIRED depressions unchanged."
        )
        a(
            "The hydrology-ready DEM applies only explicitly authorised "
            "enforcement decisions."
        )
        a(
            "Neither surface silently fills genuine closed basins. "
            "REVIEW_REQUIRED cases must be resolved by the analyst "
            "before proceeding."
        )
        a("")

        a("-" * 70)
        a("RECOMMENDED NEXT STEPS")
        a("-" * 70)

        review_count = stats.get("review_preserved", 0)

        if review_count > 0:
            a(f"  [REQUIRED] {review_count} depression(s) require " "analyst review.")
            a(
                "  Inspect the decision raster and per-depression audit "
                "before using the hydrology-ready DEM."
            )
        else:
            a(
                "  All depressions were resolved or preserved without "
                "requiring analyst review."
            )

        a("  Next tool: DEM Gradient Resolution (Stage 6)")
        a("")

        a("-" * 70)
        a("REFERENCES")
        a("-" * 70)
        a(
            "Barnes, R., Lehman, C., and Mulla, D. (2014). "
            "Priority-flood: An optimal depression-filling and "
            "watershed-labeling algorithm for digital elevation models. "
            "Computers and Geosciences, 62, 117-127."
        )
        a("")
        a(
            "Lindsay, J. B. and Dhun, K. (2015). Modelling surface "
            "drainage patterns in altered landscapes using LiDAR. "
            "IJGIS, 29(3), 397-411."
        )
        a("")
        a(
            "Lindsay, J. B. (2016). Efficient hybrid breaching-filling "
            "sink removal methods for flow path enforcement in digital "
            "elevation models. Hydrological Processes, 30(6), 846-857."
        )
        a("")
        a(
            "Mayim Tools DEM Hydrological Conditioning Research Paper "
            "(Rev 1, August 2026). Stage 5 - Selective Flow Enforcement."
        )
        a("")

        a("-" * 70)
        a("IP STATEMENT")
        a("-" * 70)
        a(
            "Stage 5 terrain modification is performed by original Mayim "
            "implementations of published algorithms. No WhiteboxTools, "
            "RichDEM or other third-party hydrological runtime is used."
        )
        a("")

        a("-" * 70)
        a("REPORT LOCATION")
        a("-" * 70)
        a(str(Path(paths["report"]).resolve()))
        a("")

        a("=" * 70)
        a("End of DEM Hydrological Filling report")
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
            f"Report written: {paths['report'].name}",
            feedback,
        )
