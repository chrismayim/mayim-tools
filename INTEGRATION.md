# Mayim Tools Integration Guide

## Scope

Mayim Tools is a QGIS Processing plugin containing engineering and
geospatial tools organised into functional categories.

Version 0.3.0 contains:

- Hydrology Tools.
- Rainfall Analysis Tools.
- Data Tools.

## Repository Structure

```text
Mayim Tools/
├── mayim_tools/
│   ├── categories/
│   │   ├── geometry/
│   │   ├── hydrology/
│   │   ├── rainfall/
│   │   └── data/
│   ├── design_rainfall/
│   ├── huff_curves/
│   ├── grib_to_csv/
│   ├── processing/
│   ├── core/
│   ├── icons/
│   └── metadata.txt
├── README.md
├── CHANGELOG.md
└── mayim_tools/scripts/
