"""
D8 Flow Direction Tool.

Computes the D8 (deterministic eight-node) single-flow-direction raster
from a hydrologically corrected DEM using steepest-descent neighbour
analysis. Supports both Standard and ESRI classification schemes.

Methodology
-----------
For each valid raster cell the algorithm evaluates the eight cardinal and
diagonal neighbours and assigns the flow direction to the neighbour with
the greatest downslope gradient. Gradients to diagonal neighbours are
distance-corrected by dividing by the diagonal cell distance (√(dx²+dy²)).
Cells with no downslope neighbour are classified as flat cells.

Encoding
--------
Both schemes share the same directional power-of-two values:

    32   64  128
    16    *    1
     8    4    2

    Direction   Code
    ---------   ----
    East           1
    South-East     2
    South          4
    South-West     8
    West          16
    North-West    32
    North         64
    North-East   128

Flat / NoData encoding differs by scheme:
    Standard  →  flat = 0,   nodata = 0
    ESRI      →  flat = 255, nodata = 255

References
----------
O'Callaghan, J.F. and Mark, D.M. (1984). The extraction of drainage
networks from digital elevation data. Computer Vision, Graphics, and
Image Processing, 28(3), 323-344.

Garbrecht, J. and Martz, L.W. (1997). The assignment of drainage
direction over flat surfaces in raster digital elevation models.
Journal of Hydrology, 193(1-4), 204-213.

IP Status
---------
Clean-room implementation using rasterio and NumPy only.
Does not call WhiteboxTools, RichDEM or TauDEM.

Author  : Mayim Tools Development Team
Created : 2025
License : GNU General Public License v2.0 or later (GPL-2.0+)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from qgis.core import (
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingOutputFile,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)

from mayim_tools.core.logger import MayimLogger
from mayim_tools.processing.algorithms.base_algorithm import MayimBaseAlgorithm

# ---------------------------------------------------------------------------
# D8 directional encoding
# ---------------------------------------------------------------------------
# Neighbour offsets keyed by direction code (power-of-two).
# Row increases downward; col increases rightward.
#
#   Code   Direction   (dRow, dCol)
#   ----   ---------   -----------
#      1   E           ( 0, +1)
#      2   SE          (+1, +1)
#      4   S           (+1,  0)
#      8   SW          (+1, -1)
#     16   W           ( 0, -1)
#     32   NW          (-1, -1)
#     64   N           (-1,  0)
#    128   NE          (-1, +1)

D8_DIRECTIONS: dict[int, tuple[int, int]] = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}

D8_DIRECTION_LABELS: dict[int, str] = {
    1: "E",
    2: "SE",
    4: "S",
    8: "SW",
    16: "W",
    32: "NW",
    64: "N",
    128: "NE",
}

# Flat cell sentinel used internally during computation
_FLAT_SENTINEL: int = -1

# Output encoding by scheme
_SCHEME_NODATA: dict[str, int] = {
    "Standard": 0,
    "ESRI": 255,
}
_SCHEME_FLAT: dict[str, int] = {
    "Standard": 0,
    "ESRI": 255,
}

# Parameter / output keys
_PARAM_INPUT_DEM: str = "INPUT_DEM"
_PARAM_USE_ESRI: str = "USE_ESRI_ENCODING"
_PARAM_OUTPUT_RASTER: str = "OUTPUT_RASTER"
_PARAM_OUTPUT_REPORT: str = "OUTPUT_REPORT"
_OUT_PROVENANCE: str = "OUTPUT_PROVENANCE"


class D8FlowDirection(MayimBaseAlgorithm):
    """
    D8 Flow Direction tool.

    Computes the D8 single-flow-direction raster from a hydrologically
    corrected DEM. Each valid cell is assigned the direction of the
    steepest downslope gradient among its eight neighbours.

    Outputs
    -------
    - D8 flow direction raster (int16, GeoTIFF, DEFLATE compressed)
    - Plain-text summary report (.txt)
    - Provenance JSON record (auto-derived from report path)
    """

    # ------------------------------------------------------------------
    # Algorithm identity
    # ------------------------------------------------------------------

    def name(self) -> str:
        """Return the unique processing algorithm identifier."""
        return "d8flowdirection"

    def displayName(self) -> str:
        """Return the human-readable name shown in the Processing Toolbox."""
        return "D8 Flow Direction"

    def group(self) -> str:
        """Return the display name of the tool group."""
        return "Hydrology Tools"

    def groupId(self) -> str:
        """Return the unique identifier of the tool group."""
        return "hydrology"

    def shortHelpString(self) -> str:
        """Return the short help string shown in the Processing panel."""
        return (
            "Computes the D8 (deterministic eight-node) flow direction "
            "raster from a hydrologically corrected DEM.\n\n"
            "For each valid cell the algorithm identifies the neighbour "
            "with the steepest downslope gradient (distance-corrected for "
            "diagonal neighbours) and encodes the result using the standard "
            "powers-of-two scheme:\n\n"
            "    32   64  128\n"
            "    16    *    1\n"
            "     8    4    2\n\n"
            "Flat / sink cells and NoData cells are encoded as 0 (Standard) "
            "or 255 (ESRI).\n\n"
            "Input  : A hydrologically corrected (filled / conditioned) DEM.\n"
            "Outputs: Flow direction raster, plain-text report, provenance "
            "JSON (auto-named alongside the report).\n\n"
            "References:\n"
            "  O'Callaghan and Mark (1984)\n"
            "  Garbrecht and Martz (1997)\n\n"
            "NOTE: This tool does not replace professional engineering "
            "judgement, independent quality assurance or applicable design "
            "standards and regulations."
        )

    def createInstance(self) -> D8FlowDirection:
        """Return a new instance of this algorithm."""
        return D8FlowDirection()

    # ------------------------------------------------------------------
    # Parameter definition
    # ------------------------------------------------------------------

    def initAlgorithm(self, config: dict | None = None) -> None:
        """
        Define all input and output parameters.

        Parameters
        ----------
        config : dict | None
            Optional algorithm configuration (unused).
        """
        # Input: hydrologically corrected DEM
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=_PARAM_INPUT_DEM,
                description="Input hydrologically corrected DEM",
            )
        )

        # Option: ESRI encoding scheme
        self.addParameter(
            QgsProcessingParameterBoolean(
                name=_PARAM_USE_ESRI,
                description=(
                    "Use ESRI encoding scheme "
                    "(flat / NoData cells = 255; recommended for ArcGIS "
                    "compatibility)"
                ),
                defaultValue=False,
            )
        )

        # Output: flow direction raster — user specifies full path
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=_PARAM_OUTPUT_RASTER,
                description="Flow direction raster",
            )
        )

        # Output: plain-text report — user specifies full path
        self.addParameter(
            QgsProcessingParameterFileDestination(
                name=_PARAM_OUTPUT_REPORT,
                description="Summary report",
                fileFilter="Text files (*.txt)",
            )
        )

        # Output: provenance JSON — auto-derived, exposed as output only
        self.addOutput(
            QgsProcessingOutputFile(
                name=_OUT_PROVENANCE,
                description="Provenance JSON (auto-named alongside report)",
            )
        )

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        """
        Execute the D8 flow direction computation.

        Steps
        -----
        1. Validate and resolve inputs.
        2. Read the DEM via rasterio.
        3. Build the valid-cell mask.
        4. Compute D8 flow direction (steepest descent).
        5. Encode output array per selected scheme.
        6. Gather cell statistics.
        7. Write the flow direction raster.
        8. Write the plain-text report.
        9. Write the provenance JSON.

        Parameters
        ----------
        parameters : dict[str, Any]
            Algorithm parameter values supplied by QGIS.
        context : QgsProcessingContext
            QGIS processing context.
        feedback : QgsProcessingFeedback
            QGIS feedback object for progress reporting and cancellation.

        Returns
        -------
        dict[str, Any]
            Output keys mapped to their result file paths.

        Raises
        ------
        QgsProcessingException
            Raised on any unrecoverable error during processing.
        """
        try:
            run_timestamp: str = datetime.now(tz=timezone.utc).isoformat(
                timespec="seconds"
            )
            warnings: list[str] = []

            # ----------------------------------------------------------
            # Step 1 — Validate and resolve inputs
            # ----------------------------------------------------------
            feedback.setProgress(0)
            feedback.pushInfo("D8 Flow Direction — starting.")

            dem_layer = self.parameterAsRasterLayer(
                parameters, _PARAM_INPUT_DEM, context
            )
            if dem_layer is None or not dem_layer.isValid():
                raise QgsProcessingException(
                    "Input DEM is not valid or could not be loaded."
                )

            use_esri: bool = self.parameterAsBool(parameters, _PARAM_USE_ESRI, context)
            scheme: str = "ESRI" if use_esri else "Standard"
            out_nodata: int = _SCHEME_NODATA[scheme]
            out_flat: int = _SCHEME_FLAT[scheme]

            # Resolve user-specified output paths
            output_raster_path: str = self.parameterAsOutputLayer(
                parameters, _PARAM_OUTPUT_RASTER, context
            )
            output_report_path: str = self.parameterAsFileOutput(
                parameters, _PARAM_OUTPUT_REPORT, context
            )

            # Derive provenance path from report path (same folder, .json)
            report_path: Path = Path(output_report_path)
            provenance_path: Path = report_path.with_suffix(".json")

            dem_source: str = dem_layer.source()

            feedback.pushInfo(f"Input DEM        : {dem_source}")
            feedback.pushInfo(f"Encoding scheme  : {scheme}")
            feedback.pushInfo(f"Flat / NoData    : {out_nodata}")
            feedback.pushInfo(f"Output raster    : {output_raster_path}")
            feedback.pushInfo(f"Output report    : {output_report_path}")
            feedback.pushInfo(f"Output provenance: {provenance_path}")

            # ----------------------------------------------------------
            # Step 2 — Read DEM via rasterio
            # ----------------------------------------------------------
            feedback.setProgress(5)
            feedback.pushInfo("Reading DEM.")

            with rasterio.open(dem_source) as ds:
                dem_array: np.ndarray = ds.read(1).astype(np.float64)
                profile: dict = ds.profile.copy()
                nodata_value = ds.nodata
                cell_width: float = float(ds.res[0])
                cell_height: float = float(ds.res[1])
                crs = ds.crs
                transform = ds.transform

            n_rows: int = dem_array.shape[0]
            n_cols: int = dem_array.shape[1]

            feedback.pushInfo(f"DEM shape        : {n_rows} rows x {n_cols} columns")
            feedback.pushInfo(
                f"Cell size        : {cell_width:.4f} x {cell_height:.4f}"
            )
            feedback.pushInfo(f"CRS              : {crs}")
            feedback.pushInfo(f"NoData value     : {nodata_value}")

            # ----------------------------------------------------------
            # Step 3 — Build valid-cell mask
            # ----------------------------------------------------------
            feedback.setProgress(10)
            feedback.pushInfo("Building valid-cell mask.")

            if nodata_value is not None:
                nodata_mask: np.ndarray = np.isnan(dem_array) | (
                    dem_array == nodata_value
                )
            else:
                nodata_mask = np.isnan(dem_array)

            valid_mask: np.ndarray = ~nodata_mask
            valid_cell_count: int = int(np.sum(valid_mask))
            nodata_cell_count: int = int(np.sum(nodata_mask))

            feedback.pushInfo(f"Valid cells      : {valid_cell_count:,}")
            feedback.pushInfo(f"NoData cells     : {nodata_cell_count:,}")

            if valid_cell_count == 0:
                raise QgsProcessingException(
                    "The input DEM contains no valid cells. "
                    "Confirm the NoData value and raster integrity."
                )

            # ----------------------------------------------------------
            # Step 4 — Compute D8 flow direction
            # ----------------------------------------------------------
            feedback.setProgress(20)
            feedback.pushInfo("Computing D8 flow direction.")

            flow_dir: np.ndarray = self._compute_d8(
                dem_array=dem_array,
                valid_mask=valid_mask,
                cell_width=cell_width,
                cell_height=cell_height,
                feedback=feedback,
            )

            if feedback.isCanceled():
                feedback.pushInfo("Processing cancelled by user.")
                return {}

            # ----------------------------------------------------------
            # Step 5 — Encode output array per selected scheme
            # ----------------------------------------------------------
            feedback.setProgress(80)
            feedback.pushInfo(f"Encoding output array using {scheme} scheme.")

            output_array: np.ndarray = flow_dir.astype(np.int16)

            # Replace internal flat sentinel with scheme flat value
            flat_mask: np.ndarray = (flow_dir == _FLAT_SENTINEL) & valid_mask
            output_array[flat_mask] = out_flat
            output_array[nodata_mask] = out_nodata

            # ----------------------------------------------------------
            # Step 6 — Gather statistics
            # ----------------------------------------------------------
            directed_mask: np.ndarray = (flow_dir > 0) & valid_mask
            directed_count: int = int(np.sum(directed_mask))
            flat_count: int = int(np.sum(flat_mask))

            feedback.pushInfo(f"Directed cells   : {directed_count:,}")
            feedback.pushInfo(f"Flat cells       : {flat_count:,}")

            if flat_count > 0:
                flat_pct: float = flat_count / valid_cell_count * 100
                warning_msg: str = (
                    f"{flat_count:,} flat cells ({flat_pct:.2f}% of valid "
                    "cells) have no lower neighbour and could not be assigned "
                    "a flow direction. Run DEM Gradient Resolution before "
                    "this tool to minimise flat cells."
                )
                warnings.append(warning_msg)
                feedback.pushWarning(warning_msg)

            direction_counts: dict[str, int] = {
                label: int(np.sum((flow_dir == code) & valid_mask))
                for code, label in D8_DIRECTION_LABELS.items()
            }

            # ----------------------------------------------------------
            # Step 7 — Write flow direction raster
            # ----------------------------------------------------------
            feedback.setProgress(85)
            feedback.pushInfo("Writing flow direction raster.")

            out_profile: dict = profile.copy()
            out_profile.update(
                dtype=np.int16,
                count=1,
                nodata=out_nodata,
                compress="deflate",
            )

            with rasterio.open(output_raster_path, "w", **out_profile) as dst:
                dst.write(output_array, 1)

            feedback.pushInfo(f"Flow direction raster written: {output_raster_path}")

            # ----------------------------------------------------------
            # Step 8 — Write plain-text report
            # ----------------------------------------------------------
            feedback.setProgress(90)
            feedback.pushInfo("Writing summary report.")

            report_lines: list[str] = [
                "═" * 72,
                "  MAYIM TOOLS — D8 Flow Direction",
                "  Flow Direction Report",
                "═" * 72,
                "",
                f"  Run timestamp (UTC)  : {run_timestamp}",
                "",
                "── Inputs ───────────────────────────────────────────────────",
                f"  Input DEM            : {dem_source}",
                f"  Encoding scheme      : {scheme}",
                "",
                "── DEM Properties ───────────────────────────────────────────",
                f"  Grid shape           : {n_rows} rows x {n_cols} columns",
                f"  Cell size            : {cell_width:.4f} x {cell_height:.4f}",
                f"  CRS                  : {crs}",
                f"  NoData value (input) : {nodata_value}",
                "",
                "── Cell Statistics ──────────────────────────────────────────",
                f"  Total cells          : {n_rows * n_cols:,}",
                f"  Valid cells          : {valid_cell_count:,}",
                f"  NoData cells         : {nodata_cell_count:,}",
                f"  Directed cells       : {directed_count:,}",
                f"  Flat cells           : {flat_count:,}",
                "",
                "── Direction Frequency ───────────────────────────────────────",
                "  Direction    Code       Count        Percent",
                "  " + "─" * 52,
            ]

            for code, label in D8_DIRECTION_LABELS.items():
                count: int = direction_counts[label]
                pct: float = (
                    count / valid_cell_count * 100 if valid_cell_count > 0 else 0.0
                )
                report_lines.append(
                    f"  {label:<12} {code:<10} {count:>10,}   ({pct:>6.2f}%)"
                )

            report_lines.extend(
                [
                    "",
                    "── Encoding Scheme ──────────────────────────────────────────",
                    f"  Scheme               : {scheme}",
                    "  Direction values     :",
                    "      32   64  128",
                    "      16    *    1",
                    "       8    4    2",
                    f"  Flat cell value      : {out_flat}",
                    f"  NoData output value  : {out_nodata}",
                    "",
                    "── Outputs ──────────────────────────────────────────────────",
                    f"  Flow direction raster : {output_raster_path}",
                    f"  Report                : {output_report_path}",
                    f"  Provenance            : {provenance_path}",
                    "",
                    "── Warnings ─────────────────────────────────────────────────",
                ]
            )

            if warnings:
                for w in warnings:
                    report_lines.append(f"  WARNING : {w}")
            else:
                report_lines.append("  No warnings.")

            report_lines.extend(
                [
                    "",
                    "── Quality Assurance ────────────────────────────────────────",
                    "  Review the direction frequency table for unexpected",
                    "  asymmetry in flow directions across the catchment.",
                    "  A high flat-cell count indicates the DEM requires gradient",
                    "  resolution before flow-direction computation.",
                    "  Inspect the flow-direction raster visually in QGIS before",
                    "  proceeding to flow-accumulation computation.",
                    "  This tool does not replace professional engineering",
                    "  judgement, independent QA or applicable design standards.",
                    "",
                    "═" * 72,
                    "  End of report.",
                    "═" * 72,
                ]
            )

            report_path.write_text("\n".join(report_lines), encoding="utf-8")
            feedback.pushInfo(f"Report written: {report_path}")

            # ----------------------------------------------------------
            # Step 9 — Write provenance JSON
            # ----------------------------------------------------------
            feedback.setProgress(95)
            feedback.pushInfo("Writing provenance record.")

            provenance: dict[str, Any] = {
                "tool": "D8FlowDirection",
                "processing_id": "mayimtools:d8flowdirection",
                "run_timestamp_utc": run_timestamp,
                "inputs": {
                    "dem_source": dem_source,
                    "encoding_scheme": scheme,
                },
                "dem_properties": {
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "cell_width": cell_width,
                    "cell_height": cell_height,
                    "crs": str(crs),
                    "nodata_input": nodata_value,
                    "transform": list(transform),
                },
                "cell_statistics": {
                    "total_cells": n_rows * n_cols,
                    "valid_cells": valid_cell_count,
                    "nodata_cells": nodata_cell_count,
                    "directed_cells": directed_count,
                    "flat_cells": flat_count,
                },
                "direction_frequency": {
                    label: direction_counts[label]
                    for label in D8_DIRECTION_LABELS.values()
                },
                "encoding": {
                    "scheme": scheme,
                    "E": 1,
                    "SE": 2,
                    "S": 4,
                    "SW": 8,
                    "W": 16,
                    "NW": 32,
                    "N": 64,
                    "NE": 128,
                    "flat_value": out_flat,
                    "nodata_value": out_nodata,
                },
                "warnings": warnings,
                "outputs": {
                    "flow_direction_raster": str(output_raster_path),
                    "report": str(output_report_path),
                    "provenance": str(provenance_path),
                },
            }

            provenance_path.write_text(
                json.dumps(provenance, indent=4), encoding="utf-8"
            )
            feedback.pushInfo(f"Provenance written: {provenance_path}")

            feedback.setProgress(100)
            feedback.pushInfo(
                "D8 Flow Direction complete. Review the report and inspect "
                "the raster in QGIS before proceeding to flow-accumulation "
                "computation."
            )
            MayimLogger.success("D8 Flow Direction completed successfully.")

            return {
                _PARAM_OUTPUT_RASTER: output_raster_path,
                _PARAM_OUTPUT_REPORT: output_report_path,
                _OUT_PROVENANCE: str(provenance_path),
            }

        except QgsProcessingException:
            raise
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"D8 Flow Direction failed: {e}")
            raise QgsProcessingException(
                f"D8 Flow Direction encountered an unexpected error: {e}"
            )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _compute_d8(
        self,
        dem_array: np.ndarray,
        valid_mask: np.ndarray,
        cell_width: float,
        cell_height: float,
        feedback: QgsProcessingFeedback,
    ) -> np.ndarray:
        """
        Compute the D8 flow direction for every valid cell.

        For each valid cell the steepest downslope gradient is identified
        among the eight neighbours. The gradient is normalised by the
        distance to the neighbour so that diagonal and cardinal neighbours
        are compared on equal terms.

        Flat cells (no lower neighbour found) are assigned the internal
        sentinel value ``_FLAT_SENTINEL`` (-1) and re-encoded to the
        scheme-appropriate output value in ``processAlgorithm``.

        NoData cells retain the value 0 (default fill).

        Parameters
        ----------
        dem_array : np.ndarray
            2-D float64 elevation array.
        valid_mask : np.ndarray
            Boolean array, True where cells are valid (not NoData).
        cell_width : float
            Raster cell width in CRS units.
        cell_height : float
            Raster cell height in CRS units.
        feedback : QgsProcessingFeedback
            QGIS feedback object for progress and cancellation.

        Returns
        -------
        np.ndarray
            int16 array containing D8 direction codes, ``_FLAT_SENTINEL``
            for flat cells, and 0 for NoData cells.
        """
        n_rows, n_cols = dem_array.shape

        flow_dir: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.int16)

        diag_dist: float = float(np.sqrt(cell_width**2 + cell_height**2))

        # Distance to each neighbour by direction code
        dist_lookup: dict[int, float] = {
            1: cell_width,
            2: diag_dist,
            4: cell_height,
            8: diag_dist,
            16: cell_width,
            32: diag_dist,
            64: cell_height,
            128: diag_dist,
        }

        rows, cols = np.where(valid_mask)
        total_valid: int = len(rows)
        processed: int = 0
        last_progress: int = 20

        for row, col in zip(rows, cols):
            centre_elev: float = dem_array[row, col]
            best_code: int = _FLAT_SENTINEL
            best_gradient: float = 0.0

            for code, (dr, dc) in D8_DIRECTIONS.items():
                nr: int = row + dr
                nc: int = col + dc

                # Skip out-of-bounds neighbours
                if nr < 0 or nr >= n_rows or nc < 0 or nc >= n_cols:
                    continue

                # Skip NoData neighbours
                if not valid_mask[nr, nc]:
                    continue

                drop: float = centre_elev - dem_array[nr, nc]

                # Only consider downslope neighbours
                if drop <= 0.0:
                    continue

                gradient: float = drop / dist_lookup[code]

                if gradient > best_gradient:
                    best_gradient = gradient
                    best_code = code

            flow_dir[row, col] = best_code
            processed += 1

            # Update progress every 1% of valid cells processed
            if total_valid > 0:
                progress: int = 20 + int(processed / total_valid * 60)
                if progress > last_progress:
                    feedback.setProgress(progress)
                    last_progress = progress

            # Check for user cancellation inside the cell loop
            if feedback.isCanceled():
                raise QgsProcessingException("Processing was cancelled by the user.")

        feedback.pushInfo(
            f"D8 computation complete. " f"Processed {processed:,} valid cells."
        )

        return flow_dir
