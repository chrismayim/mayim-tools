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

Hydrology Tools provide a complete staged DEM conditioning and D8
hydrological analysis workflow:

**DEM Conditioning Chain**

1. **DEM Hydrological Screening** — quality assessment and metadata
   recording for input DEMs
2. **DEM Hydrological Smoothing** — Gaussian and median smoothing to
   reduce surface noise
3. **DEM Depression Analysis** — detection, classification and
   hierarchical analysis of topographic depressions
4. **DEM Hydrological Filling** — filling of depressions to ensure
   continuous flow paths
5. **DEM Gradient Resolution** — assignment of artificial gradients
   to flat areas to enforce drainage
6. **DEM Hydrography Enforcement** — stream burning using authoritative
   hydrography and flow-evidence data
7. **DEM Conditioning Workflow** — orchestrates Stages 1–5
   automatically; Stage 6 runs optionally when hydrography and
   flow-evidence inputs are available

**D8 Flow Analysis Chain**

8. **D8 Flow Direction** — computes the D8 single-flow-direction
   raster from a hydrologically corrected DEM using steepest-descent
   neighbour analysis; supports Standard and ESRI encoding schemes
9. **D8 Flow Accumulation** — computes the upstream contributing cell
   count for each valid cell using a topological sort (Kahn's
   algorithm); single-pass, no iteration required

The DEM conditioning tools (Stages 1–7) can be run independently or
via the workflow tool. The D8 tools (Stages 8–9) operate on the
conditioned DEM output.

### Rainfall Analysis Tools

Rainfall Analysis Tools provide regional design-rainfall and temporal
rainfall-distribution analysis:

- **Design Rainfall at Point(s)** — implements the Smithers and Schulze
  regional L-moment design-rainfall methodology described in WRC Report
  K5/1060 (2002)
- **Huff Curves from CSV** — derives dimensionless storm
  temporal-distribution curves from a rainfall time series

### Data Tools

Data Tools provide data conversion and preparation functions:

- **Convert GRIB to CSV** — processes GRIB1 and GRIB2 data, including
  ERA5-style meteorological products, using `xarray`, `cfgrib` and
  `eccodes`

---

## Processing IDs

The following Processing IDs are currently registered by the Mayim Tools
provider (`mayimtools`):

```text
mayimtools:demhydrologicalscreening
mayimtools:demhydrologicalsmoothing
mayimtools:demdepressionanalysis
mayimtools:demhydrologicalfilling
mayimtools:demgradientresolution
mayimtools:demhydrographyenforcement
mayimtools:demconditioningworkflow
mayimtools:d8flowdirection
mayimtools:d8flowaccumulation
mayimtools:design_rainfall_point
mayimtools:huff_curves
mayimtools:grib_to_csv
