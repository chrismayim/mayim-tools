# Changelog

All notable changes to Mayim Tools are documented in this file.

## [0.3.0] - 2026-09-02

### Added

#### Hydrology Tools

- Added DEM Depression Analysis.
- Added DEM Hydrological Filling.
- Added DEM Gradient Resolution.
- Added DEM Hydrography Enforcement.
- Added DEM Conditioning Workflow.
- Added final conditioned DEM output from the workflow tool.
- Added automatic loading of the final conditioned DEM into the QGIS project.
- Added workflow reports and workflow provenance records.
- Added optional hydrography enforcement in the workflow.

#### Rainfall Analysis Tools

- Integrated Design Rainfall at Point(s).
- Integrated Huff Curves from CSV.
- Added the Rainfall Analysis Tools category.
- Added rainfall category and algorithm icons.

#### Data Tools

- Integrated Convert GRIB to CSV.
- Added the Data Tools category.
- Added GRIB conversion using xarray, cfgrib and eccodes.
- Added a Data Tools category icon.

#### User Experience

- Added the Mayim logo to the Processing provider.
- Added Mayim icons to Processing algorithms.
- Added category icons to the Mayim Tools panel.
- Added version metadata for version 0.3.0.

### Changed

- Expanded the User Manual to cover all current categories and tools.
- Added parameter guidance, recommended values and known limitations.
- Expanded output naming conventions and glossary content.
- Updated the DEM conditioning workflow documentation.
- Updated deployment and dependency guidance.

### Known Limitations

- Native flow-direction and flow-accumulation tools are not yet available.
- Hydrography enforcement remains pending full testing with a suitable aligned
  flow-evidence raster.
- Huff Curves requires continued live-QGIS execution validation.
- Convert GRIB to CSV requires real ERA5-scale testing.
- The in-process compatibility of eccodes with QGIS native libraries remains
  an open validation item.

## [0.2.0] - 2026-08-22

Internal development release.

## [0.1.0] - 2026-08-22

Initial development release containing:

- DEM Hydrological Screening.
- DEM Hydrological Smoothing.
