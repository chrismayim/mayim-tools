"""Pedotransfer functions (one module per method) and the method registry.

Every method takes harmonised inputs in the same units and returns the same
parameter set, so the Monte Carlo engine can run any member of the
ensemble on any draw. Inputs (numpy arrays, any matching shape):

    sand, silt, clay   % by weight of the fine earth, USDA limits, sum 100
    om                 organic matter, % by weight
    oc_pct             organic carbon, % by weight
    ph, cec            pH (H2O) and CEC, cmol(+)/kg (NaN if unknown)
    bulk_density       fine-earth bulk density, g/cm3 (NaN if unknown)
    gravel             coarse fragments > 2 mm, % by volume (NaN if unknown)

Outputs (dict of arrays): see PARAMETERS.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import saxton_rawls, toth2015


@dataclass(frozen=True)
class Parameter:
    code: str
    label: str
    units: str
    log: bool  # summarise / split variance on log10 scale
    decimals: int


PARAMETERS: tuple[Parameter, ...] = (
    Parameter("theta_s", "Saturated water content (porosity)", "m3/m3", False, 3),
    Parameter("theta_fc", "Field capacity (33 kPa)", "m3/m3", False, 3),
    Parameter("theta_wp", "Wilting point (1500 kPa)", "m3/m3", False, 3),
    Parameter("paw", "Plant-available water (33-1500 kPa)", "m3/m3", False, 3),
    Parameter("ksat", "Saturated hydraulic conductivity", "mm/h", True, 2),
    Parameter("psi_b", "Air-entry (bubbling) suction", "mm", True, 0),
    Parameter("lambda", "Pore-size distribution index", "-", False, 3),
    Parameter("psi_f", "Green-Ampt wetting-front suction", "mm", True, 0),
    Parameter("theta_r", "Residual water content (van Genuchten)", "m3/m3", False, 3),
    Parameter("alpha", "van Genuchten alpha", "1/cm", True, 4),
    Parameter("n_vg", "van Genuchten n", "-", False, 3),
)
PARAMETER_BY_CODE = {p.code: p for p in PARAMETERS}


@dataclass(frozen=True)
class Method:
    code: str
    name: str
    citation: str
    family: str
    needs: tuple[str, ...]  # input variables the method requires
    function: object  # callable(inputs: dict, options: dict) -> dict
    defines: tuple[str, ...]  # parameter codes the method returns
    available: bool = True  # False until verified against the primary source
    region: str = ""


METHODS: tuple[Method, ...] = (
    Method(
        "SR2006",
        "Saxton & Rawls (2006)",
        saxton_rawls.CITATION,
        "Campbell / Brooks-Corey retention; Ksat",
        ("sand", "clay", "om"),
        saxton_rawls.run,
        ("theta_s", "theta_fc", "theta_wp", "paw", "ksat", "psi_b", "lambda", "psi_f"),
        True,
        "USA (USDA/NRCS National Soil Characterization database, A horizons)",
    ),
    Method(
        "TOTH2015",
        "Tóth et al. (2015) / HiHydroSoil",
        toth2015.CITATION,
        "Mualem-van Genuchten retention; Ksat",
        ("silt", "clay", "oc_pct", "bulk_density"),
        toth2015.run,
        (
            "theta_s", "theta_fc", "theta_wp", "paw", "ksat", "psi_f",
            "theta_r", "alpha", "n_vg",
        ),
        toth2015.VERIFIED,
        "Europe (EU-HYDI database)",
    ),
)  # fmt: skip
METHOD_BY_CODE = {m.code: m for m in METHODS}
AVAILABLE_METHODS = tuple(m for m in METHODS if m.available)


def active_parameters(method_codes) -> tuple[Parameter, ...]:
    """Parameters defined by at least one of the selected methods (in the
    order of PARAMETERS)."""
    defined = set()
    for code in method_codes:
        defined.update(METHOD_BY_CODE[code].defines)
    return tuple(p for p in PARAMETERS if p.code in defined)
