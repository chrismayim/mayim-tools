# Mayim Tools

Mayim Tools is a QGIS Processing plugin developed by Mayim Consulting.

## Status

This repository was rebuilt from a clean, empty state in September 2026.
The plugin currently registers a single Processing provider with **no
algorithms**. Tool categories are being redesigned and added back
deliberately, one at a time, starting with hydrology.

## Requirements

- QGIS 4.0 or later
- Python 3.12 (bundled with QGIS)

## Development setup

Clone the repository directly into your QGIS profile plugins folder, or
develop elsewhere and deploy with a copy step (see below).

### Formatting and linting

```powershell
mayim-format
mayim-lint
