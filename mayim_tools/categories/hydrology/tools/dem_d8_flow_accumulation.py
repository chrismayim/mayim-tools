"""
D8 Flow Accumulation Tool.

Computes the D8 flow accumulation raster from a D8 flow direction raster.
Each cell is assigned the number of upstream cells that drain into it,
following the D8 single-flow-direction encoding.

Methodology
-----------
The algorithm processes cells in topological order — strictly upstream to
downstream — using a dependency-count approach:

1. Parse the flow direction raster to build a downstream pointer for every
   valid cell and count how many upstream neighbours each cell has
   (in-degree).
2. Initialise a queue with all cells that have zero upstream neighbours
   (headwater cells).
3. Process the queue: for each cell, add its accumulated count (including
   itself) to its downstream neighbour's accumulation total and decrement
   the neighbour's remaining in-degree. When a neighbour's in-degree
   reaches zero it is added to the queue.
4. Cells with no valid downstream neighbour (edge cells, outlet cells)
   retain their own accumulated count.

This approach is a standard Kahn's algorithm topological sort and
guarantees that every cell is processed exactly once, in the correct
upstream-to-downstream order, with no iterative convergence required.

Encoding (expected input)
-------------------------
The input flow direction raster must use the standard D8 power-of-two
encoding produced by the Mayim Tools D8 Flow Direction tool:

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

Flat / NoData cells must be encoded as 0 (Standard) or 255 (ESRI).
Both are treated as NoData by this tool.

Output
------
Each valid output cell contains the number of upstream cells (including
itself) that drain into it. The minimum value for any valid cell is 1.
NoData cells are written as the output NoData value (-1).

References
----------
Kahn, A.B. (1962). Topological sorting of large networks.
Communications of the ACM, 5(11), 558-562.

O'Callaghan, J.F. and Mark, D.M. (1984). The extraction of drainage
networks from digital elevation data. Computer Vision, Graphics, and
Image Processing, 28(3), 323-344.

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
from collections import deque
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
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)

from mayim_tools.core.logger import MayimLogger
from mayim_tools.processing.algorithms.base_algorithm import MayimBaseAlgorithm

# ---------------------------------------------------------------------------
# D8 directional encoding
# ---------------------------------------------------------------------------
# Maps each direction code to the (row_delta, col_delta) of the downstream
# neighbour. Row increases downward; col increases rightward.

_D8_DIRECTIONS: dict[int, tuple[int, int]] = {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}

# Direction codes that indicate a valid flow direction
_VALID_CODES: frozenset[int] = frozenset(_D8_DIRECTIONS.keys())

# Output NoData value for the accumulation raster
_OUTPUT_NODATA: int = -1

# Parameter / output keys
_PARAM_INPUT_FLOW_DIR: str = "INPUT_FLOW_DIRECTION"
_PARAM_OUTPUT_RASTER: str = "OUTPUT_RASTER"
_PARAM_OUTPUT_REPORT: str = "OUTPUT_REPORT"
_OUT_PROVENANCE: str = "OUTPUT_PROVENANCE"


class D8FlowAccumulation(MayimBaseAlgorithm):
    """
    D8 Flow Accumulation tool.

    Computes the number of upstream cells draining into each cell of the
    raster, following the D8 single-flow-direction encoding. Processing
    uses a topological sort (Kahn's algorithm) to guarantee correct
    upstream-to-downstream ordering with a single pass over the data.

    Outputs
    -------
    - Flow accumulation raster (int32, GeoTIFF, DEFLATE compressed)
    - Plain-text summary report (.txt)
    - Provenance JSON record (auto-derived from report path)
    """

    # ------------------------------------------------------------------
    # Algorithm identity
    # ------------------------------------------------------------------

    def name(self) -> str:
        """Return the unique processing algorithm identifier."""
        return "d8flowaccumulation"

    def displayName(self) -> str:
        """Return the human-readable name shown in the Processing Toolbox."""
        return "D8 Flow Accumulation"

    def group(self) -> str:
        """Return the display name of the tool group."""
        return "Hydrology Tools"

    def groupId(self) -> str:
        """Return the unique identifier of the tool group."""
        return "hydrology"

    def shortHelpString(self) -> str:
        """Return the short help string shown in the Processing panel."""
        return (
            "Computes the D8 flow accumulation raster from a D8 flow "
            "direction raster.\n\n"
            "Each valid cell is assigned the number of upstream cells "
            "(including itself) that drain into it, following the D8 "
            "single-flow-direction encoding:\n\n"
            "    32   64  128\n"
            "    16    *    1\n"
            "     8    4    2\n\n"
            "Processing uses a topological sort (Kahn's algorithm) to "
            "guarantee correct upstream-to-downstream ordering in a "
            "single pass.\n\n"
            "Input  : A D8 flow direction raster produced by the Mayim "
            "Tools D8 Flow Direction tool (or any compatible tool using "
            "the standard power-of-two encoding).\n\n"
            "Outputs: Flow accumulation raster (int32), plain-text report, "
            "provenance JSON (auto-named alongside the report).\n\n"
            "The minimum accumulation value for any valid cell is 1. "
            "NoData cells are written as -1.\n\n"
            "References:\n"
            "  Kahn (1962) — topological sorting\n"
            "  O'Callaghan and Mark (1984) — D8 flow direction\n\n"
            "NOTE: This tool does not replace professional engineering "
            "judgement, independent quality assurance or applicable design "
            "standards and regulations."
        )

    def createInstance(self) -> D8FlowAccumulation:
        """Return a new instance of this algorithm."""
        return D8FlowAccumulation()

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
        # Input: D8 flow direction raster
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=_PARAM_INPUT_FLOW_DIR,
                description="Input D8 flow direction raster",
            )
        )

        # Output: flow accumulation raster — user specifies full path
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=_PARAM_OUTPUT_RASTER,
                description="Flow accumulation raster",
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
        Execute the D8 flow accumulation computation.

        Steps
        -----
        1. Validate and resolve inputs.
        2. Read the flow direction raster via rasterio.
        3. Build the valid-cell mask.
        4. Build downstream pointers and upstream in-degree counts.
        5. Compute flow accumulation via topological sort.
        6. Write the flow accumulation raster.
        7. Compute summary statistics.
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
            feedback.pushInfo("D8 Flow Accumulation — starting.")

            flow_dir_layer = self.parameterAsRasterLayer(
                parameters, _PARAM_INPUT_FLOW_DIR, context
            )
            if flow_dir_layer is None or not flow_dir_layer.isValid():
                raise QgsProcessingException(
                    "Input flow direction layer is not valid or could " "not be loaded."
                )

            output_raster_path: str = self.parameterAsOutputLayer(
                parameters, _PARAM_OUTPUT_RASTER, context
            )
            output_report_path: str = self.parameterAsFileOutput(
                parameters, _PARAM_OUTPUT_REPORT, context
            )

            report_path: Path = Path(output_report_path)
            provenance_path: Path = report_path.with_suffix(".json")

            flow_dir_source: str = flow_dir_layer.source()

            feedback.pushInfo(f"Input flow direction : {flow_dir_source}")
            feedback.pushInfo(f"Output raster        : {output_raster_path}")
            feedback.pushInfo(f"Output report        : {output_report_path}")
            feedback.pushInfo(f"Output provenance    : {provenance_path}")

            # ----------------------------------------------------------
            # Step 2 — Read flow direction raster via rasterio
            # ----------------------------------------------------------
            feedback.setProgress(5)
            feedback.pushInfo("Reading flow direction raster.")

            with rasterio.open(flow_dir_source) as ds:
                flow_dir_array: np.ndarray = ds.read(1).astype(np.int32)
                profile: dict = ds.profile.copy()
                nodata_value = ds.nodata
                cell_width: float = float(ds.res[0])
                cell_height: float = float(ds.res[1])
                crs = ds.crs
                transform = ds.transform

            n_rows: int = flow_dir_array.shape[0]
            n_cols: int = flow_dir_array.shape[1]

            feedback.pushInfo(
                f"Raster shape         : {n_rows} rows x {n_cols} columns"
            )
            feedback.pushInfo(
                f"Cell size            : {cell_width:.4f} x {cell_height:.4f}"
            )
            feedback.pushInfo(f"CRS                  : {crs}")
            feedback.pushInfo(f"NoData value (input) : {nodata_value}")

            # ----------------------------------------------------------
            # Step 3 — Build valid-cell mask
            # ----------------------------------------------------------
            feedback.setProgress(10)
            feedback.pushInfo("Building valid-cell mask.")

            # Valid cells have a recognised D8 direction code (1–128)
            # Flat / NoData cells have code 0 or 255 (both schemes)
            valid_mask: np.ndarray = np.isin(flow_dir_array, list(_VALID_CODES))

            valid_cell_count: int = int(np.sum(valid_mask))
            nodata_cell_count: int = int(n_rows * n_cols - valid_cell_count)

            feedback.pushInfo(f"Valid cells          : {valid_cell_count:,}")
            feedback.pushInfo(f"NoData / flat cells  : {nodata_cell_count:,}")

            if valid_cell_count == 0:
                raise QgsProcessingException(
                    "The input flow direction raster contains no valid "
                    "direction codes. Confirm the raster was produced by "
                    "the D8 Flow Direction tool using the standard "
                    "power-of-two encoding."
                )

            # ----------------------------------------------------------
            # Step 4 — Build downstream pointers and in-degree counts
            # ----------------------------------------------------------
            feedback.setProgress(15)
            feedback.pushInfo("Building downstream pointers and in-degree counts.")

            # downstream[row, col] = (downstream_row, downstream_col)
            # or None if the cell has no valid downstream neighbour.
            # in_degree[row, col] = number of valid upstream neighbours.

            in_degree: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.int32)

            # Store downstream targets as a flat index array.
            # -1 means no valid downstream neighbour (outlet / edge).
            downstream_flat: np.ndarray = np.full(
                n_rows * n_cols, fill_value=-1, dtype=np.int64
            )

            rows, cols = np.where(valid_mask)

            for row, col in zip(rows, cols):
                code: int = int(flow_dir_array[row, col])
                dr, dc = _D8_DIRECTIONS[code]
                nr: int = row + dr
                nc: int = col + dc

                # Check downstream neighbour is within bounds and valid
                if 0 <= nr < n_rows and 0 <= nc < n_cols and valid_mask[nr, nc]:
                    downstream_flat[row * n_cols + col] = nr * n_cols + nc
                    in_degree[nr, nc] += 1

            feedback.pushInfo("Downstream pointers and in-degree counts built.")

            # ----------------------------------------------------------
            # Step 5 — Compute flow accumulation via topological sort
            # ----------------------------------------------------------
            feedback.setProgress(25)
            feedback.pushInfo("Computing flow accumulation (topological sort).")

            accumulation: np.ndarray = np.zeros((n_rows, n_cols), dtype=np.int32)

            # Every valid cell contributes 1 to itself initially
            accumulation[valid_mask] = 1

            # Initialise queue with all headwater cells (in_degree == 0)
            # that are also valid (have a recognised direction code)
            queue: deque = deque()
            for row, col in zip(rows, cols):
                if in_degree[row, col] == 0:
                    queue.append((row, col))

            headwater_count: int = len(queue)
            feedback.pushInfo(f"Headwater cells      : {headwater_count:,}")

            processed: int = 0
            last_progress: int = 25

            while queue:
                if feedback.isCanceled():
                    raise QgsProcessingException(
                        "Processing was cancelled by the user."
                    )

                row, col = queue.popleft()
                flat_idx: int = row * n_cols + col
                ds_flat: int = int(downstream_flat[flat_idx])

                if ds_flat >= 0:
                    # Pass this cell's accumulated count downstream
                    ds_row: int = ds_flat // n_cols
                    ds_col: int = ds_flat % n_cols
                    accumulation[ds_row, ds_col] += accumulation[row, col]
                    in_degree[ds_row, ds_col] -= 1

                    if in_degree[ds_row, ds_col] == 0:
                        queue.append((ds_row, ds_col))

                processed += 1

                # Update progress between 25% and 80%
                if valid_cell_count > 0:
                    progress: int = 25 + int(processed / valid_cell_count * 55)
                    if progress > last_progress:
                        feedback.setProgress(progress)
                        last_progress = progress

            feedback.pushInfo(
                f"Topological sort complete. " f"Processed {processed:,} valid cells."
            )

            # Warn if any valid cells were not processed — indicates
            # cycles in the flow direction raster (should not occur with
            # a well-conditioned DEM but is worth flagging)
            unprocessed: int = valid_cell_count - processed
            if unprocessed > 0:
                warning_msg: str = (
                    f"{unprocessed:,} valid cells were not processed. "
                    "This may indicate cycles in the flow direction raster. "
                    "Ensure the input DEM was fully conditioned before "
                    "computing flow direction."
                )
                warnings.append(warning_msg)
                feedback.pushWarning(warning_msg)

            # ----------------------------------------------------------
            # Step 6 — Write flow accumulation raster
            # ----------------------------------------------------------
            feedback.setProgress(82)
            feedback.pushInfo("Writing flow accumulation raster.")

            # Set NoData cells to output NoData value
            output_array: np.ndarray = accumulation.copy()
            output_array[~valid_mask] = _OUTPUT_NODATA

            out_profile: dict = profile.copy()
            out_profile.update(
                dtype=np.int32,
                count=1,
                nodata=_OUTPUT_NODATA,
                compress="deflate",
            )

            with rasterio.open(output_raster_path, "w", **out_profile) as dst:
                dst.write(output_array, 1)

            feedback.pushInfo(f"Flow accumulation raster written: {output_raster_path}")

            # ----------------------------------------------------------
            # Step 7 — Compute summary statistics
            # ----------------------------------------------------------
            feedback.setProgress(88)
            feedback.pushInfo("Computing summary statistics.")

            valid_acc: np.ndarray = accumulation[valid_mask]
            acc_min: int = int(valid_acc.min())
            acc_max: int = int(valid_acc.max())
            acc_mean: float = float(valid_acc.mean())
            acc_median: float = float(np.median(valid_acc))

            # Estimate cell area in square metres / square degrees
            cell_area: float = cell_width * cell_height
            max_upstream_area: float = acc_max * cell_area

            feedback.pushInfo(f"Accumulation min     : {acc_min:,}")
            feedback.pushInfo(f"Accumulation max     : {acc_max:,}")
            feedback.pushInfo(f"Accumulation mean    : {acc_mean:,.1f}")
            feedback.pushInfo(f"Accumulation median  : {acc_median:,.1f}")
            feedback.pushInfo(
                f"Max upstream area    : {max_upstream_area:,.2f} " f"(CRS units²)"
            )

            # ----------------------------------------------------------
            # Step 8 — Write plain-text report
            # ----------------------------------------------------------
            feedback.setProgress(91)
            feedback.pushInfo("Writing summary report.")

            report_lines: list[str] = [
                "═" * 72,
                "  MAYIM TOOLS — D8 Flow Accumulation",
                "  Flow Accumulation Report",
                "═" * 72,
                "",
                f"  Run timestamp (UTC)  : {run_timestamp}",
                "",
                "── Inputs ───────────────────────────────────────────────────",
                f"  Flow direction raster : {flow_dir_source}",
                "",
                "── Raster Properties ────────────────────────────────────────",
                f"  Grid shape            : {n_rows} rows x {n_cols} columns",
                f"  Cell size             : {cell_width:.4f} x {cell_height:.4f}",
                f"  Cell area             : {cell_area:.6f} (CRS units²)",
                f"  CRS                   : {crs}",
                f"  NoData value (input)  : {nodata_value}",
                "",
                "── Cell Statistics ──────────────────────────────────────────",
                f"  Total cells           : {n_rows * n_cols:,}",
                f"  Valid cells           : {valid_cell_count:,}",
                f"  NoData / flat cells   : {nodata_cell_count:,}",
                f"  Headwater cells       : {headwater_count:,}",
                f"  Cells processed       : {processed:,}",
                "",
                "── Accumulation Statistics ──────────────────────────────────",
                f"  Minimum               : {acc_min:,} cells",
                f"  Maximum               : {acc_max:,} cells",
                f"  Mean                  : {acc_mean:,.1f} cells",
                f"  Median                : {acc_median:,.1f} cells",
                f"  Max upstream area     : {max_upstream_area:,.2f} (CRS units²)",
                "",
                "── Algorithm ────────────────────────────────────────────────",
                "  Method                : Topological sort (Kahn, 1962)",
                "  Weighting             : Unweighted (each cell = 1)",
                "  Output NoData value   : -1",
                "  Output data type      : int32",
                "",
                "── Outputs ──────────────────────────────────────────────────",
                f"  Flow accumulation raster : {output_raster_path}",
                f"  Report                   : {output_report_path}",
                f"  Provenance               : {provenance_path}",
                "",
                "── Warnings ─────────────────────────────────────────────────",
            ]

            if warnings:
                for w in warnings:
                    report_lines.append(f"  WARNING : {w}")
            else:
                report_lines.append("  No warnings.")

            report_lines.extend(
                [
                    "",
                    "── Quality Assurance ────────────────────────────────────────",
                    "  Verify the maximum accumulation value is consistent with",
                    "  the expected catchment size. The outlet cell should have",
                    "  the highest accumulation value in the catchment.",
                    "  Inspect the accumulation raster visually in QGIS — stream",
                    "  networks can be extracted by thresholding (e.g. cells with",
                    "  accumulation >= 1000 upstream cells).",
                    "  A high unprocessed cell count indicates cycles in the flow",
                    "  direction raster — re-run DEM conditioning if this occurs.",
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
                "tool": "D8FlowAccumulation",
                "processing_id": "mayimtools:d8flowaccumulation",
                "run_timestamp_utc": run_timestamp,
                "inputs": {
                    "flow_direction_source": flow_dir_source,
                },
                "raster_properties": {
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "cell_width": cell_width,
                    "cell_height": cell_height,
                    "cell_area": cell_area,
                    "crs": str(crs),
                    "nodata_input": nodata_value,
                    "transform": list(transform),
                },
                "cell_statistics": {
                    "total_cells": n_rows * n_cols,
                    "valid_cells": valid_cell_count,
                    "nodata_flat_cells": nodata_cell_count,
                    "headwater_cells": headwater_count,
                    "cells_processed": processed,
                    "unprocessed_cells": unprocessed,
                },
                "accumulation_statistics": {
                    "minimum": acc_min,
                    "maximum": acc_max,
                    "mean": round(acc_mean, 2),
                    "median": round(acc_median, 2),
                    "max_upstream_area_crs_units2": round(max_upstream_area, 4),
                },
                "algorithm": {
                    "method": "topological_sort_kahn_1962",
                    "weighting": "unweighted",
                    "output_nodata": _OUTPUT_NODATA,
                    "output_dtype": "int32",
                },
                "warnings": warnings,
                "outputs": {
                    "flow_accumulation_raster": str(output_raster_path),
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
                "D8 Flow Accumulation complete. Review the report and "
                "inspect the raster in QGIS. Stream networks can be "
                "extracted by thresholding the accumulation raster."
            )
            MayimLogger.success("D8 Flow Accumulation completed successfully.")

            return {
                _PARAM_OUTPUT_RASTER: output_raster_path,
                _PARAM_OUTPUT_REPORT: output_report_path,
                _OUT_PROVENANCE: str(provenance_path),
            }

        except QgsProcessingException:
            raise
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"D8 Flow Accumulation failed: {e}")
            raise QgsProcessingException(
                f"D8 Flow Accumulation encountered an unexpected error: {e}"
            )
