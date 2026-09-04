# Changelog

All notable changes to Mayim Tools are documented here.

## [0.3.0] - 2026-09-02

### Added

#### Hydrology Tools

- Added DEM Depression Analysis.
- Added DEM Hydrological Filling.
- Added DEM Gradient Resolution.
- Added DEM Hydrography Enforcement.
- Added DEM Conditioning Workflow.
- Added final conditioned DEM output from the workflow tool.
- Added automatic loading of the final conditioned DEM into QGIS.
- Added workflow-level reports and provenance records.
- Added optional hydrography enforcement to the workflow.

#### Rainfall Analysis Tools

- Added the Rainfall Analysis Tools category.
- Integrated Design Rainfall at Point(s).
- Integrated Huff Curves from CSV.
- Added rainfall category and algorithm icons.

#### Data Tools

- Added the Data Tools category.
- Integrated Convert GRIB to CSV.
- Added GRIB conversion using `xarray`, `cfgrib` and `eccodes`.

#### User Interface

- Added the Mayim Tools provider icon.
- Added Mayim icons to Processing algorithms.
- Added category icons to the Mayim Tools panel.

### Changed

- Updated the plugin version to `0.3.0`.
- Expanded the User Manual to cover all current categories and tools.
- Added parameter ranges and recommended starting values.
- Expanded output naming conventions and glossary content.
- Added documentation for reports and provenance records.
- Updated installation and dependency guidance.

### Known Limitations

- Native flow-direction and flow-accumulation tools are not yet available.
- Hydrography enforcement cannot be fully validated until a suitable aligned
  flow-evidence raster is available.
- Area-scaled hydrography burn depth is not yet connected.
- Huff Curves requires further execution testing inside a live QGIS session.
- Convert GRIB to CSV requires testing against representative real-world
  ERA5-scale files.
- The interaction between `eccodes` native libraries and QGIS native
  libraries remains a specific compatibility risk.
- Design Rainfall coverage is limited to the bundled rainfall dataset.
- Design Rainfall mid-duration coefficient drift remains an open validation item.

## [0.2.0] - 2026-08-22

Internal development release.

## [0.1.0] - 2026-08-22

Initial development release containing:

- DEM Hydrological Screening.
- DEM Hydrological Smoothing.
