# Mayim Tools – third-party data

The Mayim Tools code is licensed under GPL-3 (see LICENSE). The datasets
listed below are **not** covered by that licence. Their copyright stays
with their owners, and their own terms apply. Each tool lists its sources
in its help text and run log, and the tools that write a Word report also
list them in the report.

Checked November 2026 against the providers' catalogue pages.

## Bundled with the plugin

| Data | Used by | Source and terms |
|---|---|---|
| `rainfall/design_rainfall/data/design_rainfall.gpkg` (national 1 arc-minute design rainfall grid and the station list) | Design Rainfall Estimation (South Africa) | Extracted from the Water Research Commission's publicly available *Design Rainfall Estimation in South Africa* program. Cite Smithers, J.C. and Schulze, R.E. (2002), *Design Rainfall and Flood Estimation in South Africa*, WRC Report No. K5/1060, Water Research Commission, Pretoria. Copyright remains with the WRC and the authors. No separate redistribution permission has been obtained. |
| `hydrology/dws_wr2012/data/DWS_Hydrological_Stations_*.csv` | Extract: DWS and WR2012 data | Department of Water and Sanitation (South Africa) hydrological station list, retrieved 20 September 2026 from the DWS public services (gia.dws.gov.za). |

## Downloaded by the tools at run time

| Dataset | Tool | Citation | Licence |
|---|---|---|---|
| CHIRPS v3.0 | Extract: CHIRPS precipitation | Climate Hazards Center (2025), CHIRPS3 Data Repository, doi:10.15780/G2JQ0P | Public domain |
| CMORPH CDR v1 | Extract: CMORPH precipitation | Xie et al. (2017), J. Hydrometeorology 18, 1617–1641, doi:10.1175/JHM-D-16-0168.1; NOAA NCEI dataset doi:10.25921/w9va-q159 | CC0 1.0; NOAA CDRs have no restrictions on use |
| ERA5 / ERA5-Land | Extract: ERA5 precipitation | Hersbach et al. (2023), doi:10.24381/cds.adbb2d47; Muñoz Sabater (2019), doi:10.24381/cds.e2161bac; Hersbach et al. (2020), QJRMS 146, 1999–2049, doi:10.1002/qj.3803 | CC BY 4.0. Required attribution: "Contains modified Copernicus Climate Change Service information [year]. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains." |
| GPM IMERG Final V07 | Extract: IMERG precipitation | Huffman et al. (2023), GES DISC, doi:10.5067/GPM/IMERG/3B-HH/07 | NASA open data, CC BY 4.0 |
| MERRA-2 M2T1NXFLX | Extract: MERRA2 precipitation | GMAO (2015), GES DISC, doi:10.5067/7MCPBJ41Y0K6; Gelaro et al. (2017), J. Climate 30, 5419–5454, doi:10.1175/JCLI-D-16-0758.1 | NASA open data |
| PERSIANN-CDR, PERSIANN-CCS-CDR | Extract: PERSIANN CDR precipitation | Sorooshian et al. (2014), NOAA NCEI, doi:10.7289/V51V5BWQ; Ashouri et al. (2015), BAMS 96, 69–83, doi:10.1175/BAMS-D-13-00068.1; Sadeghi et al. (2021), Scientific Data 8, 157, doi:10.1038/s41597-021-00940-9 | PERSIANN-CDR: no restrictions on use. PERSIANN-CCS-CDR: free from the CHRS Data Portal |
| SoilGrids, OpenLandMap, iSDAsoil, land cover products, WR2012 | Soil and land cover extracts, DWS and WR2012 | Listed in each tool's help text and report | As stated by each tool |
| OpenStreetMap basemap | Report maps | © OpenStreetMap contributors | ODbL |
