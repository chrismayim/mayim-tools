"""
Mayim Tools - Shared Raster Manifest Contract
=============================================

This module implements the shared interchange contract between all
Mayim Tools hydrological conditioning tools, as specified in:

  Section 6.3 of the Mayim Tools DEM Hydrological Conditioning
  Research Paper (Rev 1, August 2026).

The RasterManifest (here: MayimManifest) is the ONLY code shared
between all six tools. Each tool's algorithmic logic is independent.

IP STATUS: Original Mayim IP.
  - Uses Python standard library only (dataclasses, json, uuid, typing).
  - No third-party hydrological package imported.
  - Implemented from Section 6.3 of the Mayim research paper.
  - No WhiteboxTools or RichDEM source consulted.

Reference:
  Mayim Tools DEM Hydrological Conditioning Research Paper
  Rev 1, August 2026, Section 6.3.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass
class MayimManifest:
    """
    Shared interchange contract between all Mayim Tools conditioning
    tools.

    Every tool reads a MayimManifest from its predecessor and writes
    a new MayimManifest via .derive() for its successor. No tool
    imports another tool's internal code; each only reads and writes
    a MayimManifest, so any tool can be replaced, upgraded, or run
    in isolation without touching the others.

    Fields
    ------
    raster_path : str
        Absolute path to the output GeoTIFF raster produced by this
        tool.

    crs : str
        Coordinate reference system as a PROJ string, EPSG code or
        WKT string. Example: "EPSG:32735".

    cell_size : float
        Native raster cell size in the CRS linear unit (metres for
        projected CRS, degrees for geographic CRS). Where the raster
        is not square, this is the mean of x and y cell sizes.

    vertical_accuracy : float
        Vertical accuracy of the DEM in metres, expressed as RMSE
        or LE90. Where no formal accuracy figure is available, a
        conservative source-based estimate is used:
            LiDAR bare-earth DTM:    0.15 m
            LiDAR DSM uncorrected:   0.40 m
            SRTM / Copernicus:       4.00 m
            FABDEM / FathomDEM:      1.50 m
            Aerial photogrammetry:   0.75 m
            Unknown:                 5.00 m

    nodata : float
        NoData sentinel value used in the raster. All tools must
        preserve this value and never modify NoData cells.

    provenance_id : str
        UUID-4 string uniquely identifying this manifest instance.
        Generated automatically by .derive() when a tool produces
        a new output.

    produced_by : str
        Human-readable identifier of the tool and version that
        produced this manifest.
        Example: "dem-hydrological-screening-0.2.0"

    parent_provenance_id : Optional[str]
        The provenance_id of the manifest this tool received as
        input. Enables full chain-of-custody reconstruction by
        ProvenanceLedger (DEM Pipeline Audit).

    audit_log_path : Optional[str]
        Absolute path to the JSON audit log written by this tool.
        May be None if the tool does not produce a separate audit
        log (in addition to the manifest itself).

    stage : Optional[int]
        The pipeline stage number this tool implements.
        Stage 0 = Ingestion and QA
        Stage 1 = Artifact Correction
        Stage 2 = Controlled Smoothing
        Stage 3 = Depression Delineation
        Stage 4 = Depression Classification
        Stage 5 = Selective Flow Enforcement
        Stage 6 = Flat-Area Resolution
        Stage 7 = Hydrography Enforcement
        Stage 8 = Validation and Provenance Export

    warnings : Optional[list[str]]
        List of warning messages emitted by the producing tool.
        Preserved and accumulated across the chain.

    dem_source_type : Optional[str]
        Declared DEM source type. Passed forward so downstream
        tools can apply source-appropriate defaults.
        Example: "LiDAR DTM", "Copernicus GLO-30", "SRTM"

    width : Optional[int]
        Raster width in cells.

    height : Optional[int]
        Raster height in cells.

    bounds : Optional[dict]
        Raster spatial bounds as a dict with keys:
        left, bottom, right, top. In CRS units.

    dtype : Optional[str]
        NumPy dtype string of the raster data.
        Example: "float32", "uint8"
    """

    # ── Required fields ────────────────────────────────────────────── #
    raster_path: str
    crs: str
    cell_size: float
    vertical_accuracy: float
    nodata: float
    provenance_id: str
    produced_by: str

    # ── Optional fields ────────────────────────────────────────────── #
    parent_provenance_id: str | None = None
    audit_log_path: str | None = None
    stage: int | None = None
    warnings: list | None = None
    dem_source_type: str | None = None
    width: int | None = None
    height: int | None = None
    bounds: dict | None = None
    dtype: str | None = None

    # ── Core contract methods ──────────────────────────────────────── #

    def derive(
        self,
        produced_by: str,
        raster_path: str,
        stage: int | None = None,
        audit_log_path: str | None = None,
        warnings: list | None = None,
        width: int | None = None,
        height: int | None = None,
        bounds: dict | None = None,
        dtype: str | None = None,
    ) -> MayimManifest:
        """
        Create a new MayimManifest for this tool's output.

        Every tool calls this method to hand its output manifest
        to whatever runs next — another tool, the orchestrator,
        or nothing at all.

        The new manifest inherits all metadata from the parent
        (CRS, cell size, vertical accuracy, nodata, source type)
        and records the chain-of-custody link via
        parent_provenance_id.

        Parameters
        ----------
        produced_by : str
            Tool name and version string.
        raster_path : str
            Absolute path to this tool's output raster.
        stage : Optional[int]
            Pipeline stage number.
        audit_log_path : Optional[str]
            Path to this tool's JSON audit log.
        warnings : Optional[list]
            Warnings to carry forward.
        width : Optional[int]
            Output raster width in cells.
        height : Optional[int]
            Output raster height in cells.
        bounds : Optional[dict]
            Output raster spatial bounds.
        dtype : Optional[str]
            Output raster data type.

        Returns
        -------
        MayimManifest
            New manifest for the output of this tool.
        """
        combined_warnings = list(self.warnings or []) + list(warnings or [])

        return replace(
            self,
            produced_by=produced_by,
            raster_path=raster_path,
            parent_provenance_id=self.provenance_id,
            provenance_id=str(uuid.uuid4()),
            stage=stage if stage is not None else self.stage,
            audit_log_path=audit_log_path,
            warnings=combined_warnings if combined_warnings else None,
            width=width if width is not None else self.width,
            height=height if height is not None else self.height,
            bounds=bounds if bounds is not None else self.bounds,
            dtype=dtype if dtype is not None else self.dtype,
        )

    def write(self, path: str) -> None:
        """
        Serialise this manifest to a JSON file.

        Parameters
        ----------
        path : str
            Absolute path to write the JSON manifest file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @staticmethod
    def read(path: str) -> MayimManifest:
        """
        Deserialise a MayimManifest from a JSON file.

        Parameters
        ----------
        path : str
            Absolute path to the JSON manifest file.

        Returns
        -------
        MayimManifest
            Reconstructed manifest instance.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return MayimManifest(**data)

    # ── Factory methods ────────────────────────────────────────────── #

    @staticmethod
    def create(
        raster_path: str,
        crs: str,
        cell_size: float,
        vertical_accuracy: float,
        nodata: float,
        produced_by: str,
        stage: int | None = None,
        audit_log_path: str | None = None,
        warnings: list | None = None,
        dem_source_type: str | None = None,
        width: int | None = None,
        height: int | None = None,
        bounds: dict | None = None,
        dtype: str | None = None,
    ) -> MayimManifest:
        """
        Create the first MayimManifest in a processing chain.

        Called by the first tool in the pipeline (DEM Hydrological
        Screening / TerraCorrect) when it ingests a raw DEM with
        no prior manifest.

        Parameters
        ----------
        raster_path : str
            Absolute path to the input or output raster.
        crs : str
            CRS as PROJ, EPSG or WKT string.
        cell_size : float
            Native cell size in CRS linear units.
        vertical_accuracy : float
            Vertical accuracy in metres (RMSE or LE90).
        nodata : float
            NoData sentinel value.
        produced_by : str
            Tool name and version string.
        stage : Optional[int]
            Pipeline stage number.
        audit_log_path : Optional[str]
            Path to the JSON audit log.
        warnings : Optional[list]
            Initial warnings.
        dem_source_type : Optional[str]
            Declared DEM source type string.
            Example: "LiDAR DTM", "Copernicus GLO-30", "SRTM"
        width : Optional[int]
            Raster width in cells.
        height : Optional[int]
            Raster height in cells.
        bounds : Optional[dict]
            Raster spatial bounds dict with keys:
            left, bottom, right, top.
        dtype : Optional[str]
            NumPy dtype string. Example: "float32".

        Returns
        -------
        MayimManifest
            First manifest in the processing chain.
        """
        return MayimManifest(
            raster_path=raster_path,
            crs=crs,
            cell_size=cell_size,
            vertical_accuracy=vertical_accuracy,
            nodata=nodata,
            provenance_id=str(uuid.uuid4()),
            produced_by=produced_by,
            parent_provenance_id=None,
            audit_log_path=audit_log_path,
            stage=stage,
            warnings=warnings,
            dem_source_type=dem_source_type,
            width=width,
            height=height,
            bounds=bounds,
            dtype=dtype,
        )

    # ── Convenience properties ─────────────────────────────────────── #

    @property
    def manifest_filename(self) -> str:
        """
        Return a standard manifest filename derived from the
        raster path.

        Example:
            raster_path = "/output/dem_screened.tif"
            manifest_filename = "dem_screened.manifest.json"
        """
        stem = Path(self.raster_path).stem
        return f"{stem}.manifest.json"

    @property
    def manifest_path(self) -> str:
        """
        Return the full manifest path alongside the raster.

        Example:
            raster_path = "/output/dem_screened.tif"
            manifest_path = "/output/dem_screened.manifest.json"
        """
        raster = Path(self.raster_path)
        return str(raster.parent / self.manifest_filename)

    @property
    def has_parent(self) -> bool:
        """True if this manifest was derived from a prior tool."""
        return self.parent_provenance_id is not None

    @property
    def has_warnings(self) -> bool:
        """True if any warnings have been accumulated."""
        return bool(self.warnings)

    @property
    def warning_count(self) -> int:
        """Number of accumulated warnings."""
        return len(self.warnings) if self.warnings else 0

    # ── Validation ─────────────────────────────────────────────────── #

    def validate(self) -> list[str]:
        """
        Validate the manifest for internal consistency.

        Returns a list of validation error strings. An empty list
        means the manifest is valid.

        Called by each tool before accepting a manifest from its
        predecessor, so that a corrupt or incomplete manifest is
        caught at the tool boundary rather than mid-processing.

        Returns
        -------
        list[str]
            List of error descriptions. Empty if valid.
        """
        errors = []

        if not self.raster_path:
            errors.append("raster_path is empty or None.")

        if not Path(self.raster_path).exists():
            errors.append(f"raster_path does not exist: {self.raster_path}")

        if not self.crs:
            errors.append("crs is empty or None.")

        if self.cell_size <= 0:
            errors.append(f"cell_size must be positive, got: {self.cell_size}")

        if self.vertical_accuracy <= 0:
            errors.append(
                f"vertical_accuracy must be positive, " f"got: {self.vertical_accuracy}"
            )

        if not self.provenance_id:
            errors.append("provenance_id is empty or None.")

        if not self.produced_by:
            errors.append("produced_by is empty or None.")

        return errors

    def is_valid(self) -> bool:
        """
        Return True if validate() finds no errors.

        Returns
        -------
        bool
            True if the manifest is internally consistent.
        """
        return len(self.validate()) == 0

    # ── Display ────────────────────────────────────────────────────── #

    def summary(self) -> str:
        """
        Return a human-readable single-line summary of this manifest.

        Used in Processing feedback and text reports.

        Returns
        -------
        str
            Summary string.
        """
        return (
            f"[{self.produced_by}] "
            f"Stage {self.stage} | "
            f"CRS: {self.crs} | "
            f"Cell: {self.cell_size:.4f} | "
            f"VA: {self.vertical_accuracy:.3f} m | "
            f"Warnings: {self.warning_count} | "
            f"ID: {self.provenance_id[:8]}..."
        )

    def __str__(self) -> str:
        return self.summary()
