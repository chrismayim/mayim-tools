"""
Mayim Tools - DEM Gradient Resolution
======================================

QGIS Processing adapter for Stage 6 flat-area resolution.

This adapter calls the native Mayim Stage 6 components:

    - flat_detection.detect_flats()
    - flat_regions.label_flat_regions()
    - gradient_resolution.resolve_flats()

The adapter performs input/output handling, reporting and provenance.
It does not contain a second implementation of the gradient algorithm.

Stage 6 resolves flat raster areas by applying a documented
Garbrecht-Martz-style dual-gradient correction. The input DEM is never
overwritten.

IP status
---------
Original Mayim adapter and domain integration.

No WhiteboxTools, RichDEM, TauDEM or other third-party hydrological
runtime is used.
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
from mayim_tools.hydrology.gradient.flat_detection import detect_flats
from mayim_tools.hydrology.gradient.flat_regions import label_flat_regions
from mayim_tools.hydrology.gradient.gradient_resolution import resolve_flats
from mayim_tools.processing.algorithms.base_algorithm import MayimBaseAlgorithm


class DEMGradientResolution(MayimBaseAlgorithm):
    """QGIS Processing adapter for native Stage 6 resolution."""

    PARAM_DEM = "INPUT_DEM"
    PARAM_MANIFEST = "INPUT_MANIFEST"
    PARAM_VERTICAL_ACCURACY = "VERTICAL_ACCURACY"
    PARAM_CELL_SIZE = "CELL_SIZE"
    PARAM_CONNECTIVITY = "CONNECTIVITY"
    PARAM_OUTPUT_FOLDER = "OUTPUT_FOLDER"

    PARAM_LOAD_RESOLVED = "LOAD_RESOLVED_DEM"
    PARAM_LOAD_FLAT_MASK = "LOAD_FLAT_MASK"
    PARAM_LOAD_DIFFERENCE = "LOAD_DIFFERENCE"
    PARAM_LOAD_REGION_IDS = "LOAD_REGION_IDS"

    OUTPUT_RESOLVED = "OUTPUT_RESOLVED_DEM"
    OUTPUT_FLAT_MASK = "OUTPUT_FLAT_MASK"
    OUTPUT_REGION_IDS = "OUTPUT_REGION_IDS"
    OUTPUT_DIFFERENCE = "OUTPUT_DIFFERENCE"
    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_PROVENANCE = "OUTPUT_PROVENANCE"
    OUTPUT_MANIFEST = "OUTPUT_MANIFEST"

    TOOL_VERSION = "dem-gradient-resolution-0.2.0"

    def name(self) -> str:
        """Return the Processing algorithm identifier."""
        return "demgradientresolution"

    def displayName(self) -> str:
        """Return the human-readable algorithm name."""
        return "DEM Gradient Resolution"

    def group(self) -> str:
        """Return the Processing Toolbox group."""
        return "Hydrology Tools"

    def groupId(self) -> str:
        """Return the Processing group identifier."""
        return "hydrology"

    def createInstance(self) -> DEMGradientResolution:
        """Return a new adapter instance."""
        return DEMGradientResolution()

    def shortHelpString(self) -> str:
        """Return the Processing help text."""
        return (
            "<b>DEM Gradient Resolution</b><br><br>"
            "Implements Stage 6 controlled flat-area resolution using "
            "a native Mayim Garbrecht-Martz-style dual-gradient "
            "method.<br><br>"
            "The tool identifies flat candidate regions, calculates "
            "region-specific boundaries and applies a small synthetic "
            "gradient below the supplied vertical-accuracy limit.<br><br>"
            "<b>Important:</b> The input DEM is never overwritten. "
            "Review the difference raster, flat mask and report before "
            "using the resolved DEM downstream."
        )

    def helpUrl(self) -> str:
        """Return the project documentation URL."""
        return "https://github.com/chrismayim/mayim-tools"

    def tags(self) -> list[str]:
        """Return searchable Processing tags."""
        return [
            "mayim",
            "dem",
            "hydrology",
            "flat",
            "flats",
            "gradient",
            "resolution",
            "garbrecht",
            "martz",
            "stage-6",
        ]

    def initAlgorithm(self, config=None) -> None:
        """Define Processing parameters and outputs."""
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
                "Vertical accuracy in metres "
                "(recommended: use manifest; otherwise 0.15)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.15,
                minValue=0.000001,
                maxValue=100000.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_CELL_SIZE,
                "Cell size in map units " "(recommended: use input DEM resolution)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.000001,
                maxValue=1000000.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_CONNECTIVITY,
                "Flat-region connectivity " "(recommended: 8; valid values: 4 or 8)",
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
                self.PARAM_LOAD_RESOLVED,
                "Load gradient-resolved DEM into project",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_LOAD_FLAT_MASK,
                "Load flat mask into project",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_LOAD_DIFFERENCE,
                "Load gradient difference raster into project",
                defaultValue=True,
            )
        )

        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PARAM_LOAD_REGION_IDS,
                "Load flat-region IDs into project",
                defaultValue=True,
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_RESOLVED,
                "Gradient-resolved DEM",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_FLAT_MASK,
                "Flat mask",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_REGION_IDS,
                "Flat-region IDs",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_DIFFERENCE,
                "Gradient difference raster",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_REPORT,
                "Gradient-resolution report",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_PROVENANCE,
                "Gradient-resolution provenance",
            )
        )

        self.addOutput(
            QgsProcessingOutputFile(
                self.OUTPUT_MANIFEST,
                "Gradient-resolution MayimManifest",
            )
        )

    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """Execute Stage 6 flat resolution."""
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

        vertical_accuracy = self.parameterAsDouble(
            parameters,
            self.PARAM_VERTICAL_ACCURACY,
            context,
        )

        cell_size = self.parameterAsDouble(
            parameters,
            self.PARAM_CELL_SIZE,
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

        load_resolved = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_RESOLVED,
            context,
        )

        load_flat_mask = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_FLAT_MASK,
            context,
        )

        load_difference = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_DIFFERENCE,
            context,
        )

        load_region_ids = self.parameterAsBoolean(
            parameters,
            self.PARAM_LOAD_REGION_IDS,
            context,
        )

        if vertical_accuracy <= 0:
            raise QgsProcessingException("Vertical accuracy must be greater than zero.")

        if cell_size <= 0:
            raise QgsProcessingException("Cell size must be greater than zero.")

        if connectivity not in (4, 8):
            raise QgsProcessingException("Connectivity must be either 4 or 8.")

        if not output_folder or output_folder == "TEMPORARY_OUTPUT":
            import tempfile

            output_folder = tempfile.mkdtemp(prefix="mayim_gradient_")

        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        dem_stem = Path(dem_layer.source()).stem

        paths = {
            "resolved": output_dir / f"{dem_stem}_gradient_resolved.tif",
            "flat_mask": output_dir / f"{dem_stem}_flat_mask.tif",
            "region_ids": output_dir / f"{dem_stem}_flat_region_ids.tif",
            "difference": output_dir / f"{dem_stem}_gradient_difference.tif",
            "report": output_dir / f"{dem_stem}_gradient_report.txt",
            "provenance": output_dir / f"{dem_stem}_gradient_provenance.json",
            "manifest": output_dir / f"{dem_stem}_gradient.manifest.json",
        }

        input_manifest = None

        if manifest_path and Path(manifest_path).exists():
            try:
                input_manifest = MayimManifest.read(manifest_path)
                manifest_errors = input_manifest.validate()

                if manifest_errors:
                    self.log_warning(
                        "Input manifest validation issues: "
                        + "; ".join(manifest_errors),
                        feedback,
                    )
                else:
                    vertical_accuracy = float(input_manifest.vertical_accuracy)
                    cell_size = float(input_manifest.cell_size)

                    self.log(
                        "Input manifest loaded; vertical accuracy and "
                        "cell size inherited from manifest.",
                        feedback,
                    )

            except Exception as error:  # noqa: BLE001
                self.log_warning(
                    f"Could not read input manifest: {error}",
                    feedback,
                )

        self.log("=" * 60, feedback)
        self.log("STAGE 6 - GRADIENT RESOLUTION", feedback)
        self.log("=" * 60, feedback)
        self.log(f"Input DEM: {dem_layer.source()}", feedback)
        self.log(
            f"Vertical accuracy: {vertical_accuracy:.6f}",
            feedback,
        )
        self.log(f"Cell size: {cell_size:.6f}", feedback)
        self.log(f"Connectivity: {connectivity}", feedback)

        provenance = {
            "tool": "DEM Gradient Resolution",
            "algorithm_id": self.name(),
            "algorithm": "Garbrecht-Martz-style dual-gradient resolution",
            "algorithm_reference": (
                "Garbrecht, J. and Martz, L. W. (1997). "
                "The assignment of drainage direction over flat "
                "surfaces in raster digital elevation models. "
                "Journal of Hydrology, 193(1-4), 204-213."
            ),
            "ip_status": (
                "Original Mayim implementation. No WhiteboxTools, "
                "RichDEM or TauDEM runtime dependency."
            ),
            "version": self.TOOL_VERSION,
            "stage": 6,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "input_dem": dem_layer.source(),
            "input_manifest": manifest_path or None,
            "parameters": {
                "vertical_accuracy": vertical_accuracy,
                "cell_size": cell_size,
                "connectivity": connectivity,
            },
            "outputs": {key: str(value) for key, value in paths.items()},
            "statistics": {},
            "warnings": [],
        }

        try:
            self.log(
                "Reading input DEM metadata and values...",
                feedback,
            )

            with rasterio.open(dem_layer.source()) as source:
                profile = source.profile.copy()
                dem = source.read(1).astype(np.float64)

                nodata = source.nodata if source.nodata is not None else -9999.0

                width = int(source.width)
                height = int(source.height)

                resolution_x = abs(float(source.transform.a))
                resolution_y = abs(float(source.transform.e))

                if cell_size <= 0.0:
                    cell_size = (resolution_x + resolution_y) / 2.0

                crs_string = (
                    source.crs.to_string() if source.crs is not None else "Unknown"
                )

            valid_mask = np.isfinite(dem) & (dem != nodata)

            if not np.any(valid_mask):
                raise QgsProcessingException(
                    "The input DEM contains no valid elevation cells."
                )

            provenance["statistics"].update(
                {
                    "crs": crs_string,
                    "resolution_x": resolution_x,
                    "resolution_y": resolution_y,
                    "cell_size": cell_size,
                    "width": width,
                    "height": height,
                    "nodata": str(nodata),
                    "dtype": str(dem.dtype),
                    "valid_cells": int(np.sum(valid_mask)),
                }
            )

            feedback.setProgress(15)

            if self.is_cancelled(feedback):
                return {}

            self.log(
                "Detecting flat candidate cells...",
                feedback,
            )

            flat_mask, higher_boundary, lower_boundary = detect_flats(
                dem=dem,
                nodata=nodata,
            )

            flat_cells = int(np.sum(flat_mask))

            self.log(
                f"Flat candidate cells: {flat_cells:,}",
                feedback,
            )

            feedback.setProgress(25)

            if self.is_cancelled(feedback):
                return {}

            self.log(
                "Labelling connected flat regions...",
                feedback,
            )

            region_ids, region_metadata = label_flat_regions(
                flat_mask=flat_mask,
                connectivity=connectivity,
            )

            region_count = len(region_metadata)

            self.log(
                f"Connected flat regions: {region_count}",
                feedback,
            )

            feedback.setProgress(40)

            if self.is_cancelled(feedback):
                return {}

            self.log(
                "Applying Garbrecht-Martz-style gradient resolution...",
                feedback,
            )

            resolved_dem, resolution_audit = resolve_flats(
                dem=dem,
                flat_mask=flat_mask,
                higher_boundary=higher_boundary,
                lower_boundary=lower_boundary,
                cell_size=cell_size,
                vertical_accuracy=vertical_accuracy,
                nodata=nodata,
                region_ids=region_ids,
                allow_unresolved=True,
            )

            unresolved_regions = resolution_audit.get(
                "unresolved_regions",
                [],
            )

            if unresolved_regions:
                warning = (
                    f"{len(unresolved_regions)} flat region(s) could "
                    "not be resolved because no valid lower boundary "
                    "was detected. They were preserved unchanged and "
                    "recorded for analyst review."
                )

                provenance["warnings"].append(warning)
                self.log_warning(warning, feedback)

            difference = resolved_dem - dem

            valid_mask = np.isfinite(dem) & (dem != nodata)

            difference[~valid_mask] = 0.0

            changed_mask = valid_mask & (np.abs(difference) > 0.0)

            changed_cells = int(np.sum(changed_mask))
            total_change = float(np.sum(np.abs(difference[changed_mask])))
            maximum_change = (
                float(np.max(np.abs(difference[changed_mask])))
                if changed_cells
                else 0.0
            )

            provenance["statistics"].update(
                {
                    "flat_cells": flat_cells,
                    "region_count": region_count,
                    "higher_boundary_cells": int(np.sum(higher_boundary)),
                    "lower_boundary_cells": int(np.sum(lower_boundary)),
                    "changed_cells": changed_cells,
                    "total_absolute_change": total_change,
                    "maximum_absolute_change": maximum_change,
                    "resolution_audit": resolution_audit,
                    "regions": region_metadata,
                }
            )

            self.log(
                f"Cells modified: {changed_cells:,}",
                feedback,
            )
            self.log(
                f"Maximum absolute correction: " f"{maximum_change:.8f}",
                feedback,
            )

            feedback.setProgress(60)

            if self.is_cancelled(feedback):
                return {}

            # ----------------------------------------------------------
            # Prepare output profiles
            # ----------------------------------------------------------

            resolved_profile = profile.copy()
            resolved_profile.update(
                dtype="float32",
                count=1,
                compress="lzw",
            )

            resolved_output = resolved_dem.astype(
                np.float32,
                copy=True,
            )

            if nodata is not None:
                resolved_profile["nodata"] = nodata
                resolved_output[~valid_mask] = nodata
            else:
                resolved_profile["nodata"] = -9999.0
                resolved_output[~valid_mask] = -9999.0

            flat_profile = profile.copy()
            flat_profile.update(
                dtype="uint8",
                count=1,
                compress="lzw",
                nodata=255,
            )

            flat_output = np.zeros(
                dem.shape,
                dtype=np.uint8,
            )
            flat_output[flat_mask] = 1
            flat_output[~valid_mask] = 255

            region_profile = profile.copy()
            region_profile.update(
                dtype="int32",
                count=1,
                compress="lzw",
                nodata=-1,
            )

            region_output = region_ids.astype(
                np.int32,
                copy=True,
            )
            region_output[~valid_mask] = -1

            difference_profile = profile.copy()
            difference_profile.update(
                dtype="float32",
                count=1,
                compress="lzw",
                nodata=-9999.0,
            )

            difference_output = difference.astype(
                np.float32,
                copy=True,
            )
            difference_output[~valid_mask] = -9999.0

            # ----------------------------------------------------------
            # Write raster outputs
            # ----------------------------------------------------------

            self.log(
                "Writing Stage 6 raster outputs...",
                feedback,
            )

            with rasterio.open(
                paths["resolved"],
                "w",
                **resolved_profile,
            ) as destination:
                destination.write(resolved_output, 1)

            with rasterio.open(
                paths["flat_mask"],
                "w",
                **flat_profile,
            ) as destination:
                destination.write(flat_output, 1)

            with rasterio.open(
                paths["region_ids"],
                "w",
                **region_profile,
            ) as destination:
                destination.write(region_output, 1)

            with rasterio.open(
                paths["difference"],
                "w",
                **difference_profile,
            ) as destination:
                destination.write(difference_output, 1)

            feedback.setProgress(75)

            # ----------------------------------------------------------
            # Write provenance
            # ----------------------------------------------------------

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

            # ----------------------------------------------------------
            # Write text report
            # ----------------------------------------------------------

            self._write_text_report(
                paths=paths,
                provenance=provenance,
                region_metadata=region_metadata,
                feedback=feedback,
            )

            report_path = Path(paths["report"]).resolve()
            report_uri = report_path.as_uri()

            self.log(
                f"Stage 6 report: {report_path}",
                feedback,
            )
            feedback.pushInfo(
                f"Open Stage 6 report: {report_uri}",
            )

            feedback.setProgress(88)

            # ----------------------------------------------------------
            # Write derived MayimManifest
            # ----------------------------------------------------------

            try:
                if input_manifest is not None:
                    manifest = input_manifest.derive(
                        produced_by=self.TOOL_VERSION,
                        raster_path=str(paths["resolved"]),
                        stage=6,
                        audit_log_path=str(paths["provenance"]),
                        warnings=(
                            provenance["warnings"] if provenance["warnings"] else None
                        ),
                        width=width,
                        height=height,
                        dtype="float32",
                    )
                else:
                    manifest = MayimManifest.create(
                        raster_path=str(paths["resolved"]),
                        crs=crs_string,
                        cell_size=cell_size,
                        vertical_accuracy=vertical_accuracy,
                        nodata=float(nodata if nodata is not None else -9999.0),
                        produced_by=self.TOOL_VERSION,
                        stage=6,
                        audit_log_path=str(paths["provenance"]),
                        warnings=(
                            provenance["warnings"] if provenance["warnings"] else None
                        ),
                        width=width,
                        height=height,
                        dtype="float32",
                    )

                manifest.write(str(paths["manifest"]))

                self.log(
                    f"MayimManifest written: " f"{paths['manifest'].name}",
                    feedback,
                )

            except Exception as manifest_error:  # noqa: BLE001
                warning = f"Could not write MayimManifest: " f"{manifest_error}"
                provenance["warnings"].append(warning)
                self.log_warning(warning, feedback)

            feedback.setProgress(92)

            # ----------------------------------------------------------
            # Load selected layers into the QGIS project
            # ----------------------------------------------------------

            self._load_layers_into_project(
                paths=paths,
                dem_stem=dem_stem,
                load_resolved=load_resolved,
                load_flat_mask=load_flat_mask,
                load_difference=load_difference,
                load_region_ids=load_region_ids,
                feedback=feedback,
            )

            feedback.setProgress(100)

            self.log(
                "STAGE 6 - GRADIENT RESOLUTION COMPLETE",
                feedback,
            )

            return {
                self.OUTPUT_RESOLVED: str(paths["resolved"]),
                self.OUTPUT_FLAT_MASK: str(paths["flat_mask"]),
                self.OUTPUT_REGION_IDS: str(paths["region_ids"]),
                self.OUTPUT_DIFFERENCE: str(paths["difference"]),
                self.OUTPUT_REPORT: str(paths["report"]),
                self.OUTPUT_PROVENANCE: str(paths["provenance"]),
                self.OUTPUT_MANIFEST: str(paths["manifest"]),
            }

        except QgsProcessingException:
            raise

        except Exception as error:
            MayimLogger.critical(
                f"DEM Gradient Resolution failed: {error}",
            )
            raise QgsProcessingException(
                f"DEM Gradient Resolution failed: {error}",
            ) from error

    def _load_layers_into_project(
        self,
        paths: dict,
        dem_stem: str,
        load_resolved: bool,
        load_flat_mask: bool,
        load_difference: bool,
        load_region_ids: bool,
        feedback: QgsProcessingFeedback,
    ) -> None:
        """
        Load selected Stage 6 outputs directly into QGIS.

        No layer group is created.
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
                        f"Could not load output layer: " f"{layer_name}",
                        feedback,
                    )
                    return

                project.addMapLayer(layer, True)
                loaded_count += 1

                self.log(
                    f"Loaded into project: {layer_name}",
                    feedback,
                )

            if load_resolved:
                load_raster(
                    paths["resolved"],
                    f"{dem_stem} - Gradient Resolved DEM",
                )

            if load_flat_mask:
                load_raster(
                    paths["flat_mask"],
                    f"{dem_stem} - Flat Mask",
                )

            if load_difference:
                load_raster(
                    paths["difference"],
                    f"{dem_stem} - Gradient Difference",
                )

            if load_region_ids:
                load_raster(
                    paths["region_ids"],
                    f"{dem_stem} - Flat Region IDs",
                )

            self.log(
                f"{loaded_count} Stage 6 output layer(s) loaded "
                "directly into the project.",
                feedback,
            )

        except Exception as error:  # noqa: BLE001
            self.log_warning(
                f"Could not load Stage 6 outputs: {error}",
                feedback,
            )

    def _write_text_report(
        self,
        paths: dict,
        provenance: dict,
        region_metadata: dict[int, dict],
        feedback: QgsProcessingFeedback,
    ) -> None:
        """
        Write the Stage 6 human-readable report.
        """
        statistics = provenance.get("statistics", {})
        parameters = provenance.get("parameters", {})
        warnings = provenance.get("warnings", [])

        lines = []
        append = lines.append

        append("=" * 70)
        append("MAYIM TOOLS - DEM GRADIENT RESOLUTION REPORT")
        append("=" * 70)
        append(f"Generated: {provenance.get('timestamp', '')}")
        append(f"Tool: {provenance.get('tool', '')}")
        append(f"Version: {provenance.get('version', '')}")
        append(f"Stage: {provenance.get('stage', 6)}")
        append(f"Input DEM: {provenance.get('input_dem', '')}")
        append("")

        append("-" * 70)
        append("PARAMETERS")
        append("-" * 70)
        append(
            f"Vertical accuracy: " f"{parameters.get('vertical_accuracy', 'Unknown')}"
        )
        append(f"Cell size: " f"{parameters.get('cell_size', 'Unknown')}")
        append(f"Connectivity: " f"{parameters.get('connectivity', 'Unknown')}")
        append("")

        append("-" * 70)
        append("RESULTS")
        append("-" * 70)
        append(f"Flat candidate cells: " f"{statistics.get('flat_cells', 0):,}")
        append(f"Connected flat regions: " f"{statistics.get('region_count', 0):,}")
        append(
            f"Higher-boundary cells: " f"{statistics.get('higher_boundary_cells', 0):,}"
        )
        append(
            f"Lower-boundary cells: " f"{statistics.get('lower_boundary_cells', 0):,}"
        )
        append(f"Changed cells: " f"{statistics.get('changed_cells', 0):,}")
        append(
            f"Total absolute change: "
            f"{statistics.get('total_absolute_change', 0.0):.8f}"
        )
        append(
            f"Maximum absolute change: "
            f"{statistics.get('maximum_absolute_change', 0.0):.8f}"
        )
        append("")

        append("-" * 70)
        append("FLAT REGION SUMMARY")
        append("-" * 70)

        if region_metadata:
            for region_id in sorted(region_metadata):
                metadata = region_metadata[region_id]
                append(
                    f"Region {region_id}: "
                    f"{metadata.get('cell_count', 0):,} cells; "
                    f"rows {metadata.get('row_min', '?')}-"
                    f"{metadata.get('row_max', '?')}; "
                    f"columns {metadata.get('col_min', '?')}-"
                    f"{metadata.get('col_max', '?')}"
                )
        else:
            append("No flat regions were detected.")

        append("")

        append("-" * 70)
        append("OUTPUT FILES")
        append("-" * 70)
        append(f"Gradient-resolved DEM: " f"{paths['resolved'].name}")
        append(f"Flat mask: " f"{paths['flat_mask'].name}")
        append(f"Flat-region IDs: " f"{paths['region_ids'].name}")
        append(f"Gradient difference: " f"{paths['difference'].name}")
        append(f"Report: " f"{paths['report'].name}")
        append(f"Provenance: " f"{paths['provenance'].name}")
        append(f"MayimManifest: " f"{paths['manifest'].name}")
        append("")

        append("-" * 70)
        append("INTERPRETATION")
        append("-" * 70)
        append("The gradient-resolved DEM is a Stage 6 product.")
        append(
            "A small synthetic gradient has been introduced across "
            "detected flat regions."
        )
        append(
            "The gradient difference raster records the change from " "the input DEM."
        )
        append("The flat mask identifies candidate cells considered for " "resolution.")
        append(
            "The flat-region ID raster identifies each connected " "candidate region."
        )
        append("Review the outputs before using the DEM for flow routing.")
        append("")

        if warnings:
            append("-" * 70)
            append("WARNINGS")
            append("-" * 70)
            for warning in warnings:
                append(f"- {warning}")
            append("")

        append("-" * 70)
        append("IP AND METHOD STATEMENT")
        append("-" * 70)
        append(
            "This is an original Mayim implementation of the Stage 6 "
            "flat-resolution workflow. No WhiteboxTools, RichDEM or "
            "TauDEM runtime implementation is used."
        )
        append("")

        append("-" * 70)
        append("REPORT LOCATION")
        append("-" * 70)
        append(str(Path(paths["report"]).resolve()))
        append("")

        append("=" * 70)
        append("End of DEM Gradient Resolution report")
        append("=" * 70)
        append("")

        report_text = "\n".join(lines)

        with open(
            paths["report"],
            "w",
            encoding="utf-8",
        ) as file:
            file.write(report_text)

        self.log(
            f"Enforcement report written: " f"{paths['report'].name}",
            feedback,
        )
