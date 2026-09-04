# Mayim Tools — Integration Guide

## Scope

Mayim Tools is a QGIS 4+ Processing plugin containing engineering and
geospatial tools organised into functional categories.

Version 0.3.0 contains:

- Hydrology Tools
- Rainfall Analysis Tools
- Data Tools

---

## Repository Structure

```text
mayim-tools/                          ← GitHub repository root
│
├── .gitignore
├── .pre-commit-config.yaml           (disabled — use ruff/black directly)
├── .vscode/
│   └── settings.json
├── ruff.toml                         (ruff configuration)
├── LICENSE                           (GPL-2.0+)
├── README.md
├── CHANGELOG.md
├── INTEGRATION.md                    (this file)
├── pyproject.toml                    (ruff, black, mypy, pytest config)
│
└── mayim_tools/                      ← Main plugin package
    │
    ├── __init__.py                   classFactory() entry point
    ├── mayim_tools_plugin.py         Main plugin class (lifecycle)
    ├── metadata.txt                  QGIS plugin metadata
    ├── resources.qrc                 Qt resource file (icons)
    ├── resources_rc.py               Development stub (pyrcc6 unavailable)
    │
    ├── core/                         ── SHARED CORE UTILITIES ──
    │   ├── __init__.py
    │   ├── logger.py                 MayimLogger (wraps QgsMessageLog)
    │   ├── settings_manager.py       SettingsManager (wraps QSettings)
    │   ├── event_bus.py              EventBus (pub/sub messaging)
    │   ├── layer_utils.py            LayerUtils (layer helpers)
    │   ├── geometry_utils.py         GeometryUtils (geometry helpers)
    │   ├── crs_manager.py            CRSManager (CRS utilities)
    │   ├── file_io.py                FileIO (read/write utilities)
    │   ├── validation_utils.py       ValidationUtils (input validation)
    │   ├── i18n_manager.py           I18n (translation support)
    │   └── plugin_manager.py         Plugin lifecycle helpers
    │
    ├── categories/                   ── TOOL CATEGORIES ──
    │   ├── __init__.py               Imports all categories
    │   ├── base_category.py          BaseCategory (abstract base class)
    │   ├── category_registry.py      CategoryRegistry (central registry)
    │   │
    │   ├── hydrology/                Category: Hydrology Tools
    │   │   ├── __init__.py           Self-registers HydrologyCategory
    │   │   ├── category.py           HydrologyCategory descriptor
    │   │   ├── tools/
    │   │   │   ├── __init__.py
    │   │   │   ├── dem_hydrological_screening.py
    │   │   │   ├── dem_hydrological_smoothing.py
    │   │   │   ├── dem_depression_analysis.py
    │   │   │   ├── dem_hydrological_filling.py
    │   │   │   ├── dem_gradient_resolution.py
    │   │   │   ├── dem_hydrography_enforcement.py
    │   │   │   ├── dem_conditioning_workflow.py
    │   │   │   ├── dem_d8_flow_direction.py      ← v0.3.0
    │   │   │   └── dem_d8_flow_accumulation.py   ← v0.3.0
    │   │   └── ui/
    │   │
    │   ├── rainfall/                 Category: Rainfall Analysis Tools
    │   │   ├── __init__.py
    │   │   └── category.py
    │   │
    │   ├── data/                     Category: Data Tools
    │   │   ├── __init__.py
    │   │   └── category.py
    │   │
    │   └── geometry/                 Category: Geometry Tools (empty)
    │       ├── __init__.py
    │       └── category.py
    │
    ├── processing/                   ── PROCESSING FRAMEWORK ──
    │   ├── __init__.py
    │   ├── provider.py               MayimToolsProvider
    │   └── algorithms/
    │       ├── __init__.py
    │       └── base_algorithm.py     MayimBaseAlgorithm (abstract base)
    │
    ├── design_rainfall/              ── DESIGN RAINFALL MODULE ──
    │   ├── __init__.py
    │   ├── core.py
    │   ├── report.py
    │   ├── design_rainfall_algorithm.py
    │   └── data/                     ← excluded from Git (.gitignore)
    │       └── design_rainfall.gpkg  (139 MB — not version controlled)
    │
    ├── huff_curves/                  ── HUFF CURVES MODULE ──
    │   ├── __init__.py
    │   ├── huff_curves_algorithm.py
    │   └── huffrain/
    │
    ├── grib_to_csv/                  ── GRIB TO CSV MODULE ──
    │   ├── __init__.py
    │   ├── core.py
    │   └── grib_to_csv_algorithm.py
    │
    ├── hydrology/                    ── HYDROLOGY DOMAIN LIBRARY ──
    │   ├── depression/
    │   ├── enforcement/
    │   ├── gradient/
    │   └── hydrography/
    │
    ├── contract/                     ── MAYIM MANIFEST ──
    │   └── manifest.py               MayimManifest (provenance contract)
    │
    ├── ui/                           ── UI COMPONENTS ──
    │   ├── __init__.py
    │   ├── main_toolbar.py
    │   ├── main_menu.py
    │   ├── dock_widget.py
    │   ├── about_dialog.py
    │   ├── base_dialog.py
    │   └── styles/
    │       ├── mayim_light.qss
    │       └── mayim_dark.qss
    │
    ├── icons/
    ├── tests/
    ├── docs/
    ├── i18n/
    └── scripts/
        ├── deploy.py
        └── compile_resources.py
