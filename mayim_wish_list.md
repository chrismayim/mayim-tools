# Mayim Tools: list of Updates and New Tools


## 0. Road to version 1.0.0 (agreed 2026-11-02)

### A. Must do before 1.0.0
- [x] Open checks: Delineation report on real catchments; live GRIB to CSV run; Import RAS 2D as shapefiles (0.27.7); Design Storm Ensembles day scaling (0.27.7)
- [ ] Real-project test of every tool, especially those never run on real data (Storm Library, Prepare HEC-HMS Project, Prepare SWMM Project, satellite rainfall extractors, soil and land cover extracts): a one-page checklist per tool
- [ ] Passwords and API keys (IMERG Earthdata password, ERA5 CDS key) into the QGIS password store instead of plain text parameters
- [ ] User manual: chapters for Storm Library, Prepare HEC-HMS Project and Prepare SWMM Project; update the chapters changed since (Delineation, Time of Concentration, Design Rainfall, DEM Hydrological Processing, Design Storm Ensembles, Import RAS 2D)
- [ ] metadata.txt: drop experimental, move the version history out of about= into a change log, rewrite about= and description= to cover every tool group
- [ ] README rewritten
- [ ] Decide how 1.0.0 is distributed (own machine, zip for colleagues, or the QGIS plugin repository)

### B. Should do (code tidy-up from the 2026-10-11 audit)
- [ ] Cancel button honoured in the 12 tools that ignore it (slow ones first)
- [ ] Duplicated helpers into shared modules; no tool reaching into another tool's internals
- [ ] The three rainfall reports onto the shared report module
- [ ] Help text to the standard in all 37 tools
- [ ] A version number in every tool
- [ ] Automatic test run on GitHub (CI)

### C. After 1.0.0
- See sections 2 and 3 below



## 1. Tools under development and testing





## 2.Updates/refinements to existing tools

### 2.1 Catchment delineation

Work on Time of Concentration methods, aligned with HEC-HMS output tool: look to incorporate all methods, as well as a proper combined catchment flow method (see SANRAL spreadsheets as well)

### 2.2 Revisit rainfall time-series processing tools

Review and clean up. Allow for calibration to rain gauge?

### 2.3 Design Rainfall Estimation (South Africa)

~~Add additional output for clean DDF file~~ (done, 0.27.2)

### 2.4 SANRAL Deterministic methods

Spatially incorporate regional mapping
Tc used in calculation - see item 2.1 for Tc refinement
~~Add standard output tables to report~~ (done, 0.25.1)



## 3. New tool wishlist

### 3.1 Add road embankments and river channels to terrain for modelling in RAS and HMS

- As explained above
