═══════════════════════════════════════════════════════════════════════════════

MAYIM TOOLS — AI CONTEXT & PROJECT INSTRUCTION DOCUMENT

Version: 2.0
Date: 2025
Plugin Version: 0.3.0

═══════════════════════════════════════════════════════════════════════════════

SECTION 1 — WHO I AM

─────────────────────────────────────────────────────────────────────────────

I am a civil/water engineer and QGIS user developing a custom QGIS 4+ plugin
called "Mayim Tools". I am not a professional software developer — I have
intermediate technical knowledge and am learning as I go, with AI assistance.

Name:           Chris
GitHub user:    chrismayim
Organisation:   Mayim Consulting
Repository:     https://github.com/chrismayim/mayim-tools

SECTION 2 — WHAT MAYIM TOOLS IS

─────────────────────────────────────────────────────────────────────────────

Mayim Tools is a publicly redistributable QGIS 4+ plugin designed for
engineering and geospatial analysis. The name "Mayim" comes from the Hebrew
word for "water" (מַיִם), reflecting its initial focus on hydrological and
water-related engineering tools.

The plugin provides a categorised suite of processing tools accessible via:

  • The QGIS Processing Toolbox
  • The QGIS Graphical Modeler
  • The QGIS Python Console (processing.run())
  • A custom Dock Panel (category/tool browser)
  • A custom Plugin Menu (Plugins > Mayim Tools)
  • A custom Toolbar

Repository:   https://github.com/chrismayim/mayim-tools
License:      GNU General Public License v2.0 or later (GPL-2.0+)
Status:       Active development — v0.3.0

SECTION 3 — TECHNOLOGY STACK

─────────────────────────────────────────────────────────────────────────────

Host Platform:    QGIS 4.0.3 (Norrköping)
Language:         Python 3.12.13
GUI Framework:    Qt6 / PyQt6
GIS API:          PyQGIS (QGIS Python API)
Raster I/O:       rasterio 1.5.1 (available and verified)
Array processing: NumPy
IDE:              VS Code
Version Control:  Git + GitHub
OS:               Windows 11

Python Path:      C:\Program Files\QGIS 4.0.3\apps\Python312\python.exe
                  NOTE: Do NOT use bin\python.exe (causes PYTHONHOME errors)

Plugins Folder:   C:\Users\caets\AppData\Roaming\QGIS\QGIS4\profiles\
                  default\python\plugins\
                  NOTE: QGIS4 not QGIS3 — both versions installed on machine

Project Folder:   D:\Dropbox\MAYIM\6 TEGNIESE DATA EN DOKUMENTE\QGIS\
                  QGIS Plug-ins\Mayim Tools\

Code Quality:
  Linter:         ruff (via ruff.toml)
  Formatter:      black (88 char line length)
  Type checker:   mypy
  Testing:        pytest

SECTION 4 — PROJECT FOLDER STRUCTURE

─────────────────────────────────────────────────────────────────────────────

mayim-tools/                          ← GitHub repository root
│
├── .gitignore                        (excludes *.gpkg, large data files)
├── .pre-commit-config.yaml           (disabled — use ruff/black directly)
├── .vscode/settings.json
├── ruff.toml                         (ruff configuration)
├── LICENSE                           (GPL-2.0+)
├── README.md
├── CHANGELOG.md
├── INTEGRATION.md                    (architecture reference)
├── AI_INTEGRATION_INSTRUCTIONS.md   (this file — share with AI each session)
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
    │   │   └── tools/
    │   │       ├── __init__.py
    │   │       ├── dem_hydrological_screening.py
    │   │       ├── dem_hydrological_smoothing.py
    │   │       ├── dem_depression_analysis.py
    │   │       ├── dem_hydrological_filling.py
    │   │       ├── dem_gradient_resolution.py
    │   │       ├── dem_hydrography_enforcement.py
    │   │       ├── dem_conditioning_workflow.py
    │   │       ├── dem_d8_flow_direction.py      ← added v0.3.0
    │   │       └── dem_d8_flow_accumulation.py   ← added v0.3.0
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
    │       └── design_rainfall.gpkg  (139 MB — NOT version controlled)
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
    │   ├── depression/               (classification, detection, features,
    │   │                              hierarchy)
    │   ├── enforcement/              (breaching, depitting, enforcement,
    │   │                              filling)
    │   ├── gradient/                 (flat_detection, flat_regions,
    │   │                              gradient_resolution)
    │   └── hydrography/              (divergence, enforcement, topology,
    │                                  validation)
    │
    ├── contract/
    │   └── manifest.py               MayimManifest (provenance contract)
    │
    ├── ui/                           ── UI COMPONENTS ──
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

SECTION 5 — REGISTERED ALGORITHMS (v0.3.0)

─────────────────────────────────────────────────────────────────────────────

Provider ID: "mayimtools"
Total algorithms registered: 12

Verify with:
    from qgis.core import QgsApplication
    p = QgsApplication.processingRegistry().providerById("mayimtools")
    for a in p.algorithms(): print(a.id())

5.1 HYDROLOGY TOOLS (groupId: "hydrology")

    mayimtools:demhydrologicalscreening   DEM Hydrological Screening
    mayimtools:demhydrologicalsmoothing   DEM Hydrological Smoothing
    mayimtools:demdepressionanalysis      DEM Depression Analysis
    mayimtools:demhydrologicalfilling     DEM Hydrological Filling
    mayimtools:demgradientresolution      DEM Gradient Resolution
    mayimtools:demhydrographyenforcement  DEM Hydrography Enforcement
    mayimtools:demconditioningworkflow    DEM Conditioning Workflow
    mayimtools:d8flowdirection            D8 Flow Direction     ← v0.3.0
    mayimtools:d8flowaccumulation         D8 Flow Accumulation  ← v0.3.0

5.2 RAINFALL ANALYSIS TOOLS (groupId: "rainfall")

    mayimtools:design_rainfall_point      Design Rainfall at Point(s)
    mayimtools:huff_curves                Huff Curves from CSV

5.3 DATA TOOLS (groupId: "data")

    mayimtools:grib_to_csv                Convert GRIB to CSV

SECTION 6 — ARCHITECTURE DESIGN PATTERNS

─────────────────────────────────────────────────────────────────────────────

6.1 BASE ALGORITHM PATTERN

All tools inherit from MayimBaseAlgorithm:

    from mayim_tools.processing.algorithms.base_algorithm import (
        MayimBaseAlgorithm,
    )

    class MyTool(MayimBaseAlgorithm):

        def name(self) -> str:
            return "mytool"

        def displayName(self) -> str:
            return "My Tool"

        def group(self) -> str:
            return "Hydrology Tools"

        def groupId(self) -> str:
            return "hydrology"

        def shortHelpString(self) -> str:
            return "Tool description here."

        def createInstance(self) -> "MyTool":
            return MyTool()

        def initAlgorithm(self, config: dict | None = None) -> None:
            # Add parameters here
            pass

        def processAlgorithm(
            self,
            parameters: dict,
            context: QgsProcessingContext,
            feedback: QgsProcessingFeedback,
        ) -> dict:
            # Tool logic here
            return {}

6.2 CATEGORY REGISTRATION PATTERN — CRITICAL

Each tool in category.py MUST be wrapped in its OWN try/except block.
A single shared try/except block will cause ALL tools to disappear if
any one import fails. This was a critical bug fixed in v0.3.0.

    def get_algorithms(self) -> list:
        algorithms: list = []

        try:
            from mayim_tools.categories.hydrology.tools.my_tool import (
                MyTool,
            )
            algorithms.append(MyTool())
            MayimLogger.info("Registered: My Tool")
        except Exception as e:  # noqa: BLE001
            MayimLogger.critical(f"Failed to register My Tool: {e}")

        # Each additional tool gets its own try/except block

        return algorithms

6.3 OUTPUT STRATEGY — STANDARD PATTERN (v0.3.0+)

New tools use explicit file path parameters — NOT folder destinations.
This matches QGIS/GDAL native tool conventions.

    # Raster output
    self.addParameter(
        QgsProcessingParameterRasterDestination(
            name="OUTPUT_RASTER",
            description="Output raster",
        )
    )

    # Text report output
    self.addParameter(
        QgsProcessingParameterFileDestination(
            name="OUTPUT_REPORT",
            description="Summary report",
            fileFilter="Text files (*.txt)",
        )
    )

    # Provenance JSON — auto-derived, not a user parameter
    self.addOutput(
        QgsProcessingOutputFile(
            name="OUTPUT_PROVENANCE",
            description="Provenance JSON (auto-named alongside report)",
        )
    )

    # In processAlgorithm — derive provenance path from report path:
    report_path = Path(
        self.parameterAsFileOutput(parameters, "OUTPUT_REPORT", context)
    )
    provenance_path = report_path.with_suffix(".json")

6.4 STANDARD processAlgorithm ERROR HANDLING PATTERN

    try:
        # all tool logic here

    except QgsProcessingException:
        raise
    except Exception as e:  # noqa: BLE001
        MayimLogger.critical(f"Tool Name failed: {e}")
        raise QgsProcessingException(
            f"Tool Name encountered an unexpected error: {e}"
        )

6.5 STANDARD REPORT STRUCTURE

Every tool writes a plain-text report with this structure:

    ═══ (72 chars)
      MAYIM TOOLS — <Tool Name>
      <Report Title>
    ═══ (72 chars)

    Run timestamp (UTC) : ...

    ── Inputs ───────────
    ── Properties ───────
    ── Cell Statistics ──
    ── Algorithm ────────
    ── Outputs ──────────
    ── Warnings ─────────
    ── Quality Assurance ─
    ═══ End of report. ═══

6.6 STANDARD PROVENANCE JSON STRUCTURE

Every tool writes a provenance JSON with this structure:

    {
        "tool": "ClassName",
        "processing_id": "mayimtools:toolid",
        "run_timestamp_utc": "...",
        "inputs": {
            "source": "...",
            ...
        },
        "raster_properties": {
            "n_rows": ...,
            "n_cols": ...,
            "cell_width": ...,
            "cell_height": ...,
            "crs": "...",
            "nodata_input": ...,
            "transform": [...]
        },
        "cell_statistics": {
            "total_cells": ...,
            "valid_cells": ...,
            "nodata_cells": ...,
            ...
        },
        "algorithm": {
            "method": "...",
            ...
        },
        "warnings": [...],
        "outputs": {
            "raster": "...",
            "report": "...",
            "provenance": "..."
        }
    }

6.7 CORE UTILITIES — USAGE

All core utilities are static classes — import and call directly.
Never instantiate them.

    from mayim_tools.core.logger import MayimLogger
    from mayim_tools.core.validation_utils import ValidationUtils
    from mayim_tools.core.layer_utils import LayerUtils
    from mayim_tools.core.crs_manager import CRSManager
    from mayim_tools.core.settings_manager import SettingsManager

    MayimLogger.info("message")        ← general information
    MayimLogger.warning("message")     ← non-critical issues
    MayimLogger.critical("message")    ← errors
    MayimLogger.success("message")     ← successful operations

    ValidationUtils.is_valid_vector_layer(layer)
    ValidationUtils.is_positive_number(value)
    ValidationUtils.is_in_range(value, min_val, max_val)

    LayerUtils.get_vector_layers()
    LayerUtils.get_layer_by_name("name")

    CRSManager.from_epsg(32735)
    CRSManager.project_crs()

    SettingsManager.set("key", value)
    SettingsManager.get("key", default)

6.8 RASTER I/O PATTERN

All new tools use rasterio for raster reading and writing:

    import rasterio
    import numpy as np

    # Reading
    with rasterio.open(source_path) as ds:
        array = ds.read(1).astype(np.float64)
        profile = ds.profile.copy()
        nodata_value = ds.nodata
        cell_width = float(ds.res[0])
        cell_height = float(ds.res[1])
        crs = ds.crs
        transform = ds.transform

    # Writing
    out_profile = profile.copy()
    out_profile.update(
        dtype=np.int32,
        count=1,
        nodata=-1,
        compress="deflate",
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(output_array, 1)

SECTION 7 — D8 HYDROLOGY CHAIN (v0.3.0)

─────────────────────────────────────────────────────────────────────────────

The following tools form the complete D8 hydrological processing chain.
Each tool feeds directly into the next.

    Stage 1   demhydrologicalscreening    DEM quality screening
    Stage 2   demhydrologicalsmoothing    Gaussian/median smoothing
    Stage 3   demdepressionanalysis       Depression detection & classification
    Stage 4   demhydrologicalfilling      Depression filling
    Stage 5   demgradientresolution       Flat cell gradient assignment
    Stage 6   demhydrographyenforcement   Hydrography stream burning
    Stage 7   demconditioningworkflow     Orchestrates Stages 1–6
    Stage 8   d8flowdirection             D8 flow direction raster
    Stage 9   d8flowaccumulation          D8 flow accumulation raster
    Stage 10  (planned)                   Stream network extraction
    Stage 11  (planned)                   Catchment delineation

7.1 D8 FLOW DIRECTION TOOL DETAILS

    File:           dem_d8_flow_direction.py
    Class:          D8FlowDirection
    Processing ID:  mayimtools:d8flowdirection

    Inputs:
      INPUT_DEM           QgsProcessingParameterRasterLayer
      USE_ESRI_ENCODING   QgsProcessingParameterBoolean (default: False)

    Outputs:
      OUTPUT_RASTER       QgsProcessingParameterRasterDestination
      OUTPUT_REPORT       QgsProcessingParameterFileDestination (.txt)
      OUTPUT_PROVENANCE   QgsProcessingOutputFile (.json, auto-derived)

    Encoding:
      Standard scheme  →  flat/NoData = 0
      ESRI scheme      →  flat/NoData = 255
      Direction codes  →  E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128

    Algorithm:
      Pure NumPy steepest-descent neighbour search.
      Diagonal gradients distance-corrected by √(dx²+dy²).
      Flat cells assigned _FLAT_SENTINEL (-1) internally,
      then re-encoded to scheme value before writing.

    Output dtype:   int16
    Compression:    DEFLATE
    NoData output:  0 (Standard) or 255 (ESRI)

    References:
      O'Callaghan and Mark (1984)
      Garbrecht and Martz (1997)

7.2 D8 FLOW ACCUMULATION TOOL DETAILS

    File:           dem_d8_flow_accumulation.py
    Class:          D8FlowAccumulation
    Processing ID:  mayimtools:d8flowaccumulation

    Inputs:
      INPUT_FLOW_DIRECTION  QgsProcessingParameterRasterLayer

    Outputs:
      OUTPUT_RASTER         QgsProcessingParameterRasterDestination
      OUTPUT_REPORT         QgsProcessingParameterFileDestination (.txt)
      OUTPUT_PROVENANCE     QgsProcessingOutputFile (.json, auto-derived)

    Algorithm:
      Topological sort (Kahn's algorithm).
      Valid cells identified by np.isin(array, [1,2,4,8,16,32,64,128]).
      Headwater cells (in_degree == 0) seed the queue.
      Each cell passes its accumulated count to its downstream neighbour.
      Unprocessed cells after sort completion indicate flow direction
      cycles and are reported as a warning.

    Output dtype:   int32
    Compression:    DEFLATE
    NoData output:  -1
    Min valid value: 1 (every valid cell counts itself)
    Weighting:      Unweighted (each cell contributes 1)

    References:
      Kahn (1962)
      O'Callaghan and Mark (1984)

SECTION 8 — KNOWN ISSUES & DECISIONS

─────────────────────────────────────────────────────────────────────────────

8.1 pre-commit DISABLED

    pre-commit hooks are disabled due to a conflict between QGIS's Python
    environment and pre-commit's virtual environment creation.
    Use mayim-format and mayim-lint manually instead.

8.2 resources_rc.py IS A STUB

    pyrcc6 is not available in QGIS 4.0.3. resources_rc.py is a manually
    created stub. Icons are loaded from file paths. Do not attempt to
    compile it.

8.3 ICONS ARE PLACEHOLDERS

    Current icons are simple 32x32 coloured PNG files. Replace with
    proper SVG/PNG icons before public release.

8.4 DEPLOY PATH IS QGIS4 NOT QGIS3

    C:\Users\caets\AppData\Roaming\QGIS\QGIS4\profiles\default\...
    Both QGIS 3.44 and QGIS 4.0.3 are installed. Always target QGIS4.

8.5 PYTHON PATH

    Use: C:\Program Files\QGIS 4.0.3\apps\Python312\python.exe
    NOT: C:\Program Files\QGIS 4.0.3\bin\python.exe

8.6 LARGE DATA FILES EXCLUDED FROM GIT

    mayim_tools/design_rainfall/data/design_rainfall.gpkg (139 MB)
    was removed from Git history using git filter-branch and is now
    listed in .gitignore. Do not commit large .gpkg or raster files.
    The .gitignore excludes:
        mayim_tools/design_rainfall/data/*.gpkg
        mayim_tools/design_rainfall/data/*.db
        mayim_tools/design_rainfall/data/*.sqlite

8.7 PRE-EXISTING RUFF WARNINGS

    115 ruff warnings exist across pre-existing files. These are
    non-blocking and do not affect plugin operation. They are scheduled
    for a dedicated cleanup commit. The two new tool files
    (dem_d8_flow_direction.py, dem_d8_flow_accumulation.py) are
    clean with zero ruff errors.

    Key pre-existing issues to fix in cleanup session:
      DTZ003 / DTZ005  datetime without timezone (multiple files)
      ISC004           implicit string concatenation in lists
      BLE001           broad except (add # noqa: BLE001)
      B025             duplicate exception in try/except
      RUF012           mutable class attribute defaults

8.8 LINE ENDINGS

    Run once to suppress LF/CRLF warnings on Windows:
        git config --global core.autocrlf true

SECTION 9 — CURRENT STATUS (v0.3.0)

─────────────────────────────────────────────────────────────────────────────

COMPLETED:
  ✅ Full plugin architecture designed and implemented
  ✅ Core framework (9 shared utility modules)
  ✅ Category system (registry + 4 categories registered)
  ✅ Processing Provider registered with QGIS
  ✅ Full UI (toolbar, menu, dock panel, dialogs)
  ✅ Test suite scaffolded
  ✅ Deploy script working
  ✅ Plugin loading cleanly in QGIS 4.0.3
  ✅ GitHub repository live and up to date
  ✅ DEM conditioning chain complete (Stages 1–7, 7 tools)
  ✅ D8 Flow Direction tool (mayimtools:d8flowdirection)
  ✅ D8 Flow Accumulation tool (mayimtools:d8flowaccumulation)
  ✅ HydrologyCategory refactored — isolated try/except per tool
  ✅ Output strategy standardised — explicit file paths, not folders
  ✅ Large data file removed from Git history

IN PROGRESS:
  🔄 Pre-existing ruff warning cleanup (115 warnings, non-blocking)

PENDING:
  ⏳ Stream network extraction tool
  ⏳ Catchment delineation tool
  ⏳ Rational method peak flow (Q = C.i.A)
  ⏳ Time of concentration tool
  ⏳ Real icons (placeholders in place)
  ⏳ GitHub Actions CI/CD pipeline
  ⏳ Public release on QGIS Plugin Repository
  ⏳ Documentation (Sphinx/ReadTheDocs)
  ⏳ Translations (i18n)

NEXT RECOMMENDED STEPS (in order):
  1. Pre-existing ruff cleanup (dedicated commit)
  2. Stream Network Extraction tool
     — threshold flow accumulation raster at user-defined cell count
     — output stream raster + optional vector polyline
  3. Catchment Delineation tool
  4. Rational Method Peak Flow tool (Q = C.i.A)

SECTION 10 — DEVELOPER WORKFLOW

─────────────────────────────────────────────────────────────────────────────

10.1 POWERSHELL SHORTCUTS (defined in PowerShell profile)

    mayim              Navigate to project folder
    qpy                Run QGIS Python 3.12
    qpip               pip install packages
    qruff              Run ruff linter
    qblack             Run black formatter
    mayim-lint         Lint all plugin files
    mayim-format       Format all plugin files
    mayim-deploy       Deploy plugin to QGIS plugins folder
    qtest              Run pytest test suite
    gs                 git status
    ga                 git add .
    gp                 git push
    gl                 git log
    gcommit "msg"      git add + commit + push in one command

10.2 DAILY DEVELOPMENT WORKFLOW

    1. mayim                          navigate to project
    2. (make code changes in VS Code)
    3. mayim-format                   auto-fix formatting
    4. mayim-lint                     check for issues
    5. py_compile check               syntax check (optional but fast)
    6. mayim-deploy                   deploy to QGIS
    7. (reload plugin in QGIS)
    8. verify in QGIS console         check algorithm count
    9. gcommit "description"          commit and push

10.3 SYNTAX CHECK COMMAND

    & "C:\Program Files\QGIS 4.0.3\apps\Python312\python.exe" `
      -m py_compile `
      ".\mayim_tools\categories\hydrology\tools\<filename>.py"

    No output = clean. Any output = syntax error with line number.

10.4 DEPLOYING THE PLUGIN

    mayim-deploy

    Copies mayim_tools/ to:
    C:\Users\caets\AppData\Roaming\QGIS\QGIS4\profiles\default\
    python\plugins\mayim_tools\

10.5 RELOADING IN QGIS

    Option A: Plugins > Manage and Install Plugins >
              Installed > Mayim Tools > uncheck > check
    Option B: Use Plugin Reloader plugin
    Option C: Restart QGIS (cleanest)

10.6 VERIFYING IN QGIS PYTHON CONSOLE

    # Check provider is registered
    from qgis.core import QgsApplication
    p = QgsApplication.processingRegistry().providerById("mayimtools")
    print(p.name() if p else "Not found")

    # Check all algorithms
    if p:
        algos = p.algorithms()
        print(f"Algorithms registered: {len(algos)}")
        for a in algos:
            print(f"  {a.id()}")

    # Check a specific tool imports cleanly
    try:
        from mayim_tools.categories.hydrology.tools.dem_d8_flow_direction import (
            D8FlowDirection,
        )
        print(f"OK: {D8FlowDirection().name()}")
    except Exception as e:
        import traceback
        traceback.print_exc()

    # Check category returns algorithms
    from mayim_tools.categories.hydrology.category import HydrologyCategory
    cat = HydrologyCategory()
    algos = cat.get_algorithms()
    print(f"Algorithms returned: {len(algos)}")
    for a in algos:
        print(f"  {a.name()}")

SECTION 11 — CODE CONVENTIONS

─────────────────────────────────────────────────────────────────────────────

11.1 GENERAL

  - All files start with: # -*- coding: utf-8 -*-
  - All classes and methods have docstrings
  - Type hints used throughout
  - Line length: 88 characters (black default)
  - Indentation: 4 spaces (no tabs)
  - from __future__ import annotations at top of every tool file

11.2 NAMING CONVENTIONS

  - Classes:      PascalCase        (D8FlowDirection, HydrologyCategory)
  - Functions:    snake_case        (get_algorithms, process_algorithm)
  - Constants:    UPPER_SNAKE_CASE  (_OUTPUT_NODATA, _FLAT_SENTINEL)
  - Files:        snake_case        (dem_d8_flow_direction.py)
  - Categories:   snake_case id     ("hydrology", "geometry")
  - Processing:   lowercase id      (d8flowdirection, d8flowaccumulation)

11.3 IMPORTS ORDER (ruff enforced)

  1. Standard library imports
     (json, pathlib, datetime, collections, typing)
  2. Third-party imports
     (numpy, rasterio, qgis, PyQt6)
  3. Local mayim_tools imports
     (MayimBaseAlgorithm, MayimLogger, etc.)

11.4 LOGGING CONVENTION

  Use MayimLogger — never print() in production code.
  MayimLogger is a static class — never instantiate it.

    MayimLogger.info("message")       ← general information
    MayimLogger.warning("message")    ← non-critical issues
    MayimLogger.critical("message")   ← errors
    MayimLogger.success("message")    ← successful operations

11.5 ERROR HANDLING CONVENTION

  All processAlgorithm methods use this exact pattern:

    try:
        # all logic here

    except QgsProcessingException:
        raise
    except Exception as e:  # noqa: BLE001
        MayimLogger.critical(f"Tool Name failed: {e}")
        raise QgsProcessingException(
            f"Tool Name encountered an unexpected error: {e}"
        )

11.6 RUFF SUPPRESSION CONVENTIONS

  Only suppress ruff warnings where intentional and documented:

    except Exception as e:  # noqa: BLE001
    ← Used in: processAlgorithm, get_algorithms, UI event handlers
    ← Reason: plugin resilience — must catch all unexpected errors

  Do NOT suppress other rules without discussion.

11.7 DATETIME CONVENTION

  Always use timezone-aware datetimes:

    from datetime import datetime, timezone
    run_timestamp = datetime.now(tz=timezone.utc).isoformat(
        timespec="seconds"
    )

  Never use:
    datetime.utcnow()      ← deprecated, flagged by DTZ003
    datetime.now()         ← no timezone, flagged by DTZ005

SECTION 12 — VERSIONING POLICY

─────────────────────────────────────────────────────────────────────────────

Format: MAJOR.MINOR.PATCH (e.g. 0.3.0)

  MAJOR  Breaking changes — tool renames, output format changes
  MINOR  New features — new tools, new categories
  PATCH  Bug fixes — small tweaks, performance, UI fixes

Current version: 0.3.0
Experimental:    True (QGIS Plugin Repository pre-release flag)

Roadmap:
  0.1.0  ✅ Architecture complete, rainfall and data tools
  0.2.0  ✅ Full DEM conditioning chain (7 tools)
  0.3.0  ✅ Current — D8 Flow Direction + D8 Flow Accumulation
  0.3.x  🔜 Next patch — bug fixes if required
  0.4.0  🔜 Planned — stream network extraction, catchment delineation
  0.5.0  🔜 Planned — rational method, time of concentration
  1.0.0  🎯 Target — public release on QGIS Plugin Repository

Files to update on every version change:
  1. mayim_tools/metadata.txt          → version=X.X.X
  2. mayim_tools/ui/about_dialog.py    → PLUGIN_VERSION = "X.X.X"
  3. CHANGELOG.md                      → add new version section
  4. AI_INTEGRATION_INSTRUCTIONS.md   → update version references

SECTION 13 — INSTRUCTIONS FOR AI ASSISTANT

─────────────────────────────────────────────────────────────────────────────

When assisting with Mayim Tools development, please:

1.  ALWAYS follow the established architecture patterns in this document.
    Do not suggest alternative structures — the architecture is fixed.

2.  ALWAYS use MayimBaseAlgorithm as the base for new tools.
    Never create standalone QgsProcessingAlgorithm subclasses.

3.  ALWAYS register new tools in category.py get_algorithms() using an
    INDIVIDUAL try/except block per tool (Section 6.2).
    NEVER use a single shared try/except for multiple tool imports.

4.  ALWAYS use the core utilities (MayimLogger, ValidationUtils, etc.)
    Never use print() or raw QgsMessageLog calls in tool code.

5.  ALWAYS include complete docstrings on all classes and methods.

6.  ALWAYS use type hints on all function signatures.

7.  ALWAYS wrap processAlgorithm logic in try/except per Section 11.5.

8.  ALWAYS use the output strategy in Section 6.3:
    - QgsProcessingParameterRasterDestination for raster outputs
    - QgsProcessingParameterFileDestination for report outputs
    - QgsProcessingOutputFile for provenance JSON (auto-derived)
    NEVER use QgsProcessingParameterFolderDestination.

9.  ALWAYS use rasterio for raster I/O (Section 6.8).
    rasterio 1.5.1 is verified available in QGIS 4.0.3.

10. ALWAYS use timezone-aware datetimes (Section 11.7).
    datetime.now(tz=timezone.utc) — never utcnow() or now().

11. ALWAYS write a plain-text report and provenance JSON for every
    new tool, following the structures in Sections 6.5 and 6.6.

12. WHEN adding a new tool, deliver ALL of the following:
    a. Complete tool file (copy-paste ready, no partial snippets)
    b. Updated category.py showing exactly where to add the new block
    c. Step-by-step instructions specifying which file to open,
       what to replace or add, and in what order
    d. Deploy reminder: mayim-deploy
    e. Reload reminder: Plugins > Manage > Mayim Tools > uncheck > check
    f. Verify reminder: QGIS Python Console algorithm count check
    g. Commit reminder: gcommit "description"

13. WHEN generating code:
    - Line length: 88 characters (black)
    - Python target: 3.12
    - All files start with: # -*- coding: utf-8 -*-
    - from __future__ import annotations at top of every tool file
    - Provide COMPLETE files — never partial snippets
    - Specify exactly which file to open and what to replace

14. BEFORE deployment, always remind the user to run:
    a. mayim-format    ← auto-fix formatting
    b. mayim-lint      ← check for issues
    c. py_compile      ← syntax check
    d. mayim-deploy    ← deploy
    e. reload in QGIS  ← verify live
    f. gcommit         ← commit only after verified

15. THE USER IS NOT A PROFESSIONAL DEVELOPER:
    - Explain concepts clearly and simply
    - Provide complete, copy-paste ready code
    - Point out when files need full replacement vs partial edit
    - Always specify which file to open and where to paste code
    - Anticipate common mistakes and warn proactively
    - Never assume context from previous sessions — always re-read
      this document at the start of each new session

16. KNOWN ENVIRONMENT QUIRKS (never forget these):
    - Python: C:\Program Files\QGIS 4.0.3\apps\Python312\python.exe
    - QGIS profile: QGIS4 (not QGIS3)
    - Paths with spaces must be quoted in PowerShell
    - pre-commit is disabled — use mayim-format and mayim-lint
    - resources_rc.py is a stub — do not compile it
    - rasterio 1.5.1 is available — use it for all raster I/O
    - Large files (>100 MB) must not be committed to Git

17. RUFF WARNINGS:
    - 115 pre-existing warnings exist — these are non-blocking
    - New tool files must have ZERO ruff errors before deployment
    - BLE001 (broad except) is suppressed with # noqa: BLE001
      where intentional — this is correct and expected

SECTION 14 — QUICK REFERENCE — NEW TOOL CHECKLIST

─────────────────────────────────────────────────────────────────────────────

Use this checklist when building any new tool:

  BEFORE CODING
  □ Confirm tool name, processing ID, category, inputs, outputs
  □ Confirm methodology and any applicable standards/references
  □ Confirm whether standalone or workflow-chained

  FILE CREATION
  □ Create: mayim_tools/categories/<category>/tools/<tool_name>.py
  □ Header: # -*- coding: utf-8 -*-
  □ from __future__ import annotations
  □ Complete module docstring (methodology, encoding, references, IP)
  □ Class inherits MayimBaseAlgorithm
  □ All 7 identity methods implemented
  □ initAlgorithm — parameters follow Section 6.3 output strategy
  □ processAlgorithm — try/except per Section 11.5
  □ All steps logged via feedback.pushInfo and MayimLogger
  □ Progress updated via feedback.setProgress (0 → 100)
  □ Cancellation checked via feedback.isCanceled()
  □ Plain-text report written (Section 6.5 structure)
  □ Provenance JSON written (Section 6.6 structure)
  □ createInstance() returns correct class

  REGISTRATION
  □ Add individual try/except block to category.py get_algorithms()
  □ Import uses exact class name from the tool file

  QUALITY CHECKS
  □ mayim-format — passes cleanly
  □ mayim-lint — zero errors in new file
  □ py_compile — no output (syntax clean)

  DEPLOYMENT
  □ mayim-deploy
  □ Reload plugin in QGIS
  □ Verify algorithm count in QGIS Python Console
  □ Confirm new tool appears in Processing Toolbox

  COMMIT
  □ gcommit "feat: add <Tool Name> tool to <Category> Tools"

═══════════════════════════════════════════════════════════════════════════════

END OF AI INTEGRATION INSTRUCTIONS

Version 2.0 — 2025 — Mayim Tools v0.3.0

Update this document whenever:
  - A new tool is added (Section 5, 7, 9)
  - A new architectural pattern is established (Section 6)
  - A known issue is resolved or discovered (Section 8)
  - The plugin version changes (Sections 1, 9, 12)
  - A new environment quirk is discovered (Section 13.16)

═══════════════════════════════════════════════════════════════════════════════
