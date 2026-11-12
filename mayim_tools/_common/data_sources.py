"""
Citations and licences of the third-party datasets the tools read.

Each entry is shown in the tool's help (DATA SOURCE section) and written
to the run log, so the source and its terms travel with every extract.
DATA_NOTICE.md in the plugin folder lists the same information.

Checked November 2026 against the providers' catalogue pages.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSource:
    name: str
    citations: tuple[str, ...]
    licence: str
    attribution: str = ""


SOURCES = {
    "chirps": DataSource(
        name="CHIRPS version 3.0 (Climate Hazards Center, UC Santa Barbara)",
        citations=(
            "Climate Hazards Center (2025). Climate Hazards Center InfraRed "
            "Precipitation with Stations version 3. CHIRPS3 Data Repository. "
            "doi:10.15780/G2JQ0P",
        ),
        licence="Public domain (copyright waived by the Climate Hazards Center).",
    ),
    "cmorph": DataSource(
        name="CMORPH Climate Data Record, version 1 (NOAA CPC / NCEI)",
        citations=(
            "Xie, P., Joyce, R., Wu, S., Yoo, S.-H., Yarosh, Y., Sun, F. and "
            "Lin, R. (2017). Reprocessed, bias-corrected CMORPH global "
            "high-resolution precipitation estimates from 1998. Journal of "
            "Hydrometeorology 18, 1617-1641. doi:10.1175/JHM-D-16-0168.1",
            "NOAA NCEI. NOAA Climate Data Record (CDR) of CPC Morphing "
            "Technique (CMORPH) High Resolution Global Precipitation "
            "Estimates, Version 1. doi:10.25921/w9va-q159",
        ),
        licence="CC0 1.0 (public domain); NOAA CDRs carry no restrictions on use.",
    ),
    "era5": DataSource(
        name="ERA5 / ERA5-Land reanalysis (ECMWF, Copernicus Climate Change Service)",
        citations=(
            "Hersbach, H. et al. (2023). ERA5 hourly data on single levels "
            "from 1940 to present. Copernicus Climate Change Service (C3S) "
            "Climate Data Store (CDS). doi:10.24381/cds.adbb2d47",
            "Munoz Sabater, J. (2019). ERA5-Land hourly data from 1950 to "
            "present. Copernicus Climate Change Service (C3S) Climate Data "
            "Store (CDS). doi:10.24381/cds.e2161bac",
            "Hersbach, H. et al. (2020). The ERA5 global reanalysis. "
            "Quarterly Journal of the Royal Meteorological Society 146, "
            "1999-2049. doi:10.1002/qj.3803",
        ),
        licence="CC BY 4.0 (Copernicus licence for C3S data).",
        attribution=(
            "Contains modified Copernicus Climate Change Service information "
            "{year}. Neither the European Commission nor ECMWF is responsible "
            "for any use that may be made of the Copernicus information or "
            "data it contains."
        ),
    ),
    "imerg": DataSource(
        name="GPM IMERG Final Run V07, half-hourly 0.1 degree (NASA GES DISC)",
        citations=(
            "Huffman, G.J., Stocker, E.F., Bolvin, D.T., Nelkin, E.J. and "
            "Tan, J. (2023). GPM IMERG Final Precipitation L3 Half Hourly "
            "0.1 degree x 0.1 degree V07. Greenbelt, MD, Goddard Earth "
            "Sciences Data and Information Services Center (GES DISC). "
            "doi:10.5067/GPM/IMERG/3B-HH/07",
        ),
        licence="NASA open data, CC BY 4.0: free to use, cite the dataset.",
    ),
    "merra2": DataSource(
        name="MERRA-2 hourly surface flux diagnostics, PRECTOTCORR (NASA GMAO)",
        citations=(
            "Global Modeling and Assimilation Office (GMAO) (2015). MERRA-2 "
            "tavg1_2d_flx_Nx: 2d,1-Hourly,Time-Averaged,Single-Level,"
            "Assimilation,Surface Flux Diagnostics V5.12.4. Greenbelt, MD, "
            "GES DISC. doi:10.5067/7MCPBJ41Y0K6",
            "Gelaro, R. et al. (2017). The Modern-Era Retrospective Analysis "
            "for Research and Applications, Version 2 (MERRA-2). Journal of "
            "Climate 30, 5419-5454. doi:10.1175/JCLI-D-16-0758.1",
        ),
        licence="NASA open data: free to use, cite the dataset.",
    ),
    "persiann": DataSource(
        name="PERSIANN-CDR (daily) and PERSIANN-CCS-CDR (3-hourly) (CHRS, UC Irvine)",
        citations=(
            "Sorooshian, S., Hsu, K., Braithwaite, D., Ashouri, H. and NOAA "
            "CDR Program (2014). NOAA Climate Data Record (CDR) of "
            "Precipitation Estimation from Remotely Sensed Information using "
            "Artificial Neural Networks (PERSIANN-CDR), Version 1 Revision 1. "
            "NOAA NCEI. doi:10.7289/V51V5BWQ",
            "Ashouri, H. et al. (2015). PERSIANN-CDR: Daily precipitation "
            "climate data record from multisatellite observations for "
            "hydrological and climate studies. Bulletin of the American "
            "Meteorological Society 96, 69-83. doi:10.1175/BAMS-D-13-00068.1",
            "Sadeghi, M. et al. (2021). PERSIANN-CCS-CDR, a 3-hourly 0.04 "
            "degree global precipitation climate data record for heavy "
            "precipitation studies. Scientific Data 8, 157. "
            "doi:10.1038/s41597-021-00940-9",
        ),
        licence=(
            "PERSIANN-CDR: NOAA CDR, no restrictions on use. PERSIANN-CCS-CDR: "
            "free from the CHRS Data Portal (https://chrsdata.eng.uci.edu); "
            "cite Sadeghi et al. (2021)."
        ),
    ),
    "wrc_design_rainfall": DataSource(
        name=(
            "Design rainfall grid of Smithers and Schulze (2002), extracted "
            "from the Water Research Commission's publicly available Design "
            "Rainfall Estimation in South Africa program"
        ),
        citations=(
            "Smithers, J.C. and Schulze, R.E. (2002). Design Rainfall and "
            "Flood Estimation in South Africa. WRC Report No. K5/1060. Water "
            "Research Commission, Pretoria.",
        ),
        licence=(
            "Copyright remains with the Water Research Commission and the "
            "authors; the grid is not covered by the plugin's GPL-3 licence. "
            "Use it with acknowledgement of the source above."
        ),
    ),
}


def data_source_help(key: str) -> str:
    """The DATA SOURCE paragraph of a tool's help text."""
    src = SOURCES[key]
    lines = [f"DATA SOURCE:\t{src.name}.", "Cite:"]
    lines += [f"  {c}" for c in src.citations]
    lines.append(f"Licence: {src.licence}")
    if src.attribution:
        lines.append("Required attribution: " + src.attribution.format(year="<year>"))
    return "\n".join(lines)


def log_data_source(feedback, key: str, year: int | None = None) -> None:
    """Write the dataset citation and licence to the run log."""
    src = SOURCES[key]
    feedback.pushInfo(f"Data source: {src.name}")
    for c in src.citations:
        feedback.pushInfo(f"  Cite: {c}")
    feedback.pushInfo(f"  Licence: {src.licence}")
    if src.attribution:
        if year is None:
            from datetime import date

            year = date.today().year
        feedback.pushInfo("  Attribution: " + src.attribution.format(year=year))
