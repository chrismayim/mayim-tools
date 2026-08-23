# Mayim Tools — Changelog

All notable changes to Mayim Tools are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

## Versioning Reference

| Part  | Meaning                          | When to increment                          |
|-------|----------------------------------|--------------------------------------------|
| MAJOR | Breaking changes                 | Tool renames, output format changes        |
| MINOR | New features (backwards compat)  | New tools, new categories                  |
| PATCH | Bug fixes and small improvements | Bug fixes, tweaks, performance             |

---

## Release Roadmap

| Version | Description                                              | Status      |
|---------|----------------------------------------------------------|-------------|
| 0.1.0   | Architecture complete, core framework, no tools yet      | ✅ Current  |
| 0.1.x   | Bug fixes to core framework and UI                       | 🔜 Next     |
| 0.2.0   | Full hydrology chain stable and tested                   | 🔜 Planned  |
| 0.3.0   | Additional tool categories added                         | 🔜 Planned  |
| 1.0.0   | Plugin mature, ready for QGIS Plugin Repository          | 🎯 Target   |

---

## [0.2.0] — 2026

### Added

- Hydrology Tools: DEM Hydrological Screening tool
  - Stage 0: DEM Ingestion and QA
    - CRS validation and geographic CRS warning
    - Void detection and classification (small/medium/large)
    - Small void interpolation using mean neighbourhood values
    - Medium void flagging
    - Large void analyst alert — not filled
    - Vertical accuracy assignment per DEM source type
      with optional user RMSE override
  - Stage 1: Artifact Screening
    - DEM source type classification (8 source types)
    - MAD local outlier filter for speckle and striping detection
    - LiDAR ground filter flagged as coming in future release
    - Bare-earth substitution flagged as coming in future release
  - Outputs: Screened DEM, void mask, artifact mask,
    QA text report, provenance JSON log
  - References: Barnes (2014), Pingel (2013),
    Wang and Liu (2006), Hawker (2022), Leys (2013)

- Hydrology Tools: DEM Hydrological Smoothing tool
  - Stage 2: Controlled edge-preserving smoothing
  - Method: Perona-Malik anisotropic diffusion
  - Resolution-adaptive diffusion strength scaling
  - Outputs: Smoothed DEM, signed difference raster,
    smoothing mask, smoothing text report, provenance JSON log
  - Reference: Perona and Malik (1990)

- Documentation
  - Architecture Reference Report v0.2.0
  - User Manual v0.1.0 (initial draft)
  - AI Context Document updated

### Changed

- Plugin menu removed — tool access via Processing Toolbox,
  dock panel, and toolbar only
- Output layers loaded directly into project without a group
- Ruff configuration updated to use lint section in pyproject.toml
- PowerShell profile updated with mayim-deploy pointing to
  correct QGIS4 profile folder

### Fixed

- Duplicate toolbar and menu entries on plugin reload
- QGIS4 profile folder path in deploy script
  (QGIS4 not QGIS3)
- Icon loading via get_icon_path() helper replacing
  Qt resource path approach
- Tool launch from dock panel and toolbar category buttons
- Encoding issues with box-drawing characters in log strings

### Known Issues

- pre-commit hooks remain disabled due to QGIS Python
  environment conflict
- resources_rc.py remains a development stub
- Icons remain placeholder PNG files
- LiDAR ground filtering not yet implemented in Stage 1
- Bare-earth substitution not yet implemented in Stage 1

## [0.1.0] — 2025

### Added
- Complete plugin architecture and folder structure
- Core framework: Logger, SettingsManager, EventBus, LayerUtils,
  GeometryUtils, CRSManager, FileIO, ValidationUtils
- Category system: BaseCategory, CategoryRegistry
- Hydrology Tools category (registered, no tools yet)
- Geometry Tools category (registered, no tools yet)
- Processing Provider (MayimToolsProvider) registered with QGIS
- UI components: Toolbar, Plugin Menu, Dock Panel, About Dialog
- Base dialog class (MayimBaseDialog)
- Base algorithm class (MayimBaseAlgorithm)
- Deploy script (scripts/deploy.py)
- Resource compiler script (scripts/compile_resources.py)
- Test suite scaffold (pytest + conftest)
- Initial GitHub repository setup

### Known Limitations
- No processing tools implemented yet
- Icons are placeholders (32x32 coloured PNG files)
- resources_rc.py is a development stub (pyrcc6 not available)
- pre-commit hooks disabled due to QGIS Python environment conflict
