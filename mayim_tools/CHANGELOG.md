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
