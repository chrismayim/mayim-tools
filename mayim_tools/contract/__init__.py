# mayim_tools/contract/__init__.py
"""
Mayim Tools - Shared Interface Contract
=======================================

This package implements the shared interchange contract between all
Mayim Tools hydrological conditioning tools, as specified in Section
6.3 of the Mayim Tools DEM Hydrological Conditioning Research Paper
(Rev 1, August 2026).

The MayimManifest is the ONLY code shared between all six tools.
Each tool's algorithmic logic is fully independent.

Usage
-----
Creating the first manifest (tool 1 only):

    from mayim_tools.contract import MayimManifest

    manifest = MayimManifest.create(
        raster_path="/output/dem_screened.tif",
        crs="EPSG:32735",
        cell_size=5.0,
        vertical_accuracy=0.15,
        nodata=-9999.0,
        produced_by="dem-hydrological-screening-0.2.0",
        stage=0,
    )
    manifest.write(manifest.manifest_path)

Deriving a manifest from a predecessor (tools 2-6):

    from mayim_tools.contract import MayimManifest

    manifest = MayimManifest.read("/output/dem_screened.manifest.json")
    errors = manifest.validate()
    if errors:
        raise ValueError(f"Invalid manifest: {errors}")

    new_manifest = manifest.derive(
        produced_by="dem-hydrological-smoothing-0.2.0",
        raster_path="/output/dem_smoothed.tif",
        stage=2,
    )
    new_manifest.write(new_manifest.manifest_path)

IP Status
---------
Original Mayim IP.
Python standard library only: dataclasses, json, uuid, typing, pathlib.
No third-party hydrological package imported.
Implemented from Section 6.3 of the Mayim research paper.
No WhiteboxTools or RichDEM source consulted.
"""

from mayim_tools.contract.manifest import MayimManifest

__all__ = ["MayimManifest"]
