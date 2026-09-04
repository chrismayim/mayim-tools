# Changelog

All notable changes to Mayim Tools are documented in this file.

Format: [MAJOR.MINOR.PATCH] — YYYY-MM-DD
Type tags: Added | Changed | Fixed | Removed | Deprecated | Security

---

## [0.3.0] — 2025

### Added
- **D8 Flow Direction** (`mayimtools:d8flowdirection`)
  - Computes D8 single-flow-direction raster from a hydrologically
    corrected DEM using steepest-descent neighbour analysis
  - Supports Standard (flat = 0) and ESRI (flat = 255) encoding schemes
  - Outputs: flow direction raster (int16, GeoTIFF, DEFLATE), plain-text
    report, provenance JSON (auto-derived from report path)
  - Clean-room implementation using rasterio and NumPy only
  - References: O'Callaghan and Mark (1984); Garbrecht and Martz (1997)

- **D8 Flow Accumulation** (`mayimtools:d8flowaccumulation`)
  - Computes upstream cell count for each valid cell following D8
    single-flow-direction encoding
  - Uses topological sort (Kahn's algorithm) — single pass, no iteration
  - Outputs: flow accumulation raster (int32, GeoTIFF, DEFLATE),
    plain-text report, provenance JSON (auto-derived from report path)
  - Detects and warns on unprocessed cells (flow direction cycles)
  - Clean-room implementation using rasterio and NumPy only
  - References: Kahn (1962); O'Callaghan and Mark (1984)

### Changed
- **HydrologyCategory** (`category.py`) — refactored `get_algorithms()`
  to wrap each tool import in an individual `try/except` block, ensuring
  a broken tool does not prevent remaining tools from loading
- **D8 Flow Direction output strategy** — replaced
  `QgsProcessingParameterFolderDestination` with explicit
  `QgsProcessingParameterRasterDestination` and
  `QgsProcessingParameterFileDestination`, consistent with QGIS/GDAL
  tool conventions. Provenance JSON auto-derived from report path.

### Fixed
- `design_rainfall.gpkg` (139 MB) removed from Git history using
  `git filter-branch` and added to `.gitignore` to prevent future
  GitHub push rejections
- `HydrologyCategory.get_algorithms()` was returning an empty list when
  any single tool import failed — resolved by isolating each import

---

## [0.2.0] — 2025

### Added
- **DEM Hydrological Screening** (`mayimtools:demhydrologicalscreening`)
- **DEM Hydrological Smoothing** (`mayimtools:demhydrologicalsmoothing`)
- **DEM Depression Analysis** (`mayimtools:demdepressionanalysis`)
- **DEM Hydrological Filling** (`mayimtools:demhydrologicalfilling`)
- **DEM Gradient Resolution** (`mayimtools:demgradientresolution`)
- **DEM Hydrography Enforcement** (`mayimtools:demhydrographyenforcement`)
- **DEM Conditioning Workflow** (`mayimtools:demconditioningworkflow`)
  - Orchestrates Stages 1–5 automatically; Stage 6 optional
- Full hydrology domain library (`mayim_tools/hydrology/`)
  - `depression/` — classification, detection, features, hierarchy
  - `enforcement/` — breaching, depitting, enforcement, filling
  - `gradient/` — flat detection, flat regions, gradient resolution
  - `hydrography/` — divergence, enforcement, topology, validation
- MayimManifest provenance contract (`mayim_tools/contract/manifest.py`)

---

## [0.1.0] — 2025

### Added
- Full plugin architecture — QGIS 4.0+ Processing provider registered
- Core framework — 9 shared utility modules:
  - `MayimLogger`, `SettingsManager`, `EventBus`, `LayerUtils`
  - `GeometryUtils`, `CRSManager`, `FileIO`, `ValidationUtils`, `I18n`
- Category system — `BaseCategory`, `CategoryRegistry`
- Registered categories: `hydrology`, `geometry`, `rainfall`, `data`
- Processing provider: `mayimtools`
- Full UI — `MayimToolbar`, `MayimMenu`, `MayimDockWidget`,
  `AboutDialog`, `MayimBaseDialog`
- **Design Rainfall at Point(s)** (`mayimtools:design_rainfall_point`)
  - Smithers and Schulze regional L-moment methodology (WRC K5/1060)
- **Huff Curves from CSV** (`mayimtools:huff_curves`)
  - Dimensionless storm temporal-distribution curves from time series
- **Convert GRIB to CSV** (`mayimtools:grib_to_csv`)
  - GRIB1/GRIB2 processing using xarray, cfgrib, eccodes
- Deploy script (`scripts/deploy.py`)
- Test suite scaffolded (`tests/`)
- Plugin loading cleanly in QGIS 4.0.3 (Norrköping)

---

## Roadmap

| Version | Target | Description |
|---|---|---|
| 0.3.x | Next patch | Bug fixes to D8 tools if required |
| 0.4.0 | Planned | Stream network extraction, catchment delineation |
| 0.5.0 | Planned | Rational method peak flow, time of concentration |
| 1.0.0 | Target | Public release on QGIS Plugin Repository |
