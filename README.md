# Mayim Tools

A modular, publicly redistributable QGIS 4+ plugin for engineering,
hydrological and geospatial analysis.

**Current version:** 0.3.0  
**Status:** Development / Technical Preview  
**Primary platform:** QGIS 4.0+  
**Validated environment:** QGIS 4.0.3, Python 3.12.13, Qt 6.11.1

Mayim Tools provides a categorised collection of Processing Toolbox
algorithms designed to support transparent, reproducible and auditable
engineering workflows.

> Mayim Tools does not replace professional engineering judgement,
> independent quality assurance, site-specific investigation or applicable
> design standards and regulations.

---

## Tool Categories

### Hydrology Tools

Hydrology Tools provide a staged DEM conditioning workflow:

1. **DEM Hydrological Screening**
2. **DEM Hydrological Smoothing**
3. **DEM Depression Analysis**
4. **DEM Hydrological Filling**
5. **DEM Gradient Resolution**
6. **DEM Hydrography Enforcement**
7. **DEM Conditioning Workflow**

The individual tools can be run independently. The workflow tool orchestrates
Stages 1–5 automatically and runs Stage 6 optionally when hydrography and
flow-evidence inputs are available.

### Rainfall Analysis Tools

Rainfall Analysis Tools provide regional design-rainfall and temporal
rainfall-distribution analysis:

- **Design Rainfall at Point(s)**
- **Huff Curves from CSV**

The Design Rainfall tool implements the Smithers and Schulze regional L-moment
design-rainfall methodology described in WRC Report K5/1060 (2002).

Huff Curves from CSV derives dimensionless storm temporal-distribution
curves from a rainfall time series.

### Data Tools

Data Tools provide data conversion and preparation functions:

- **Convert GRIB to CSV**

The GRIB conversion tool uses `xarray`, `cfgrib` and `eccodes` to process
GRIB1 and GRIB2 data, including ERA5-style meteorological products.

---

## Processing IDs

The following Processing IDs are currently registered by the Mayim Tools
provider:

```text
mayimtools:demhydrologicalscreening
mayimtools:demhydrologicalsmoothing
mayimtools:demdepressionanalysis
mayimtools:demhydrologicalfilling
mayimtools:demgradientresolution
mayimtools:demhydrographyenforcement
mayimtools:demconditioningworkflow
