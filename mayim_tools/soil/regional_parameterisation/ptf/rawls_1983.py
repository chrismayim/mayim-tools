"""Rawls, Brakensiek & Miller (1983) Green-Ampt parameters by USDA texture
class - used as a reference check, not as an ensemble member.

Values as tabulated in Chow, Maidment & Mays (1988), Applied Hydrology,
Table 4.3.1 (from Rawls, Brakensiek & Miller, 1983): total porosity,
effective porosity, wetting-front suction head (cm) and hydraulic
conductivity K (cm/h). The class table has no Silt row (too few samples).
"""

from __future__ import annotations

import numpy as np

CITATION = (
    "Rawls, W.J., Brakensiek, D.L. and Miller, N. (1983). Green-Ampt "
    "infiltration parameters from soils data. Journal of Hydraulic "
    "Engineering 109(1): 62-70. Values as tabulated in Chow, V.T., "
    "Maidment, D.R. and Mays, L.W. (1988). Applied Hydrology, Table 4.3.1."
)

# class code -> (porosity, effective porosity, suction psi_f cm, K cm/h)
RAWLS_1983: dict[int, tuple[float, float, float, float]] = {
    1: (0.437, 0.417, 4.95, 11.78),
    2: (0.437, 0.401, 6.13, 2.99),
    3: (0.453, 0.412, 11.01, 1.09),
    4: (0.463, 0.434, 8.89, 0.34),
    5: (0.501, 0.486, 16.68, 0.65),
    7: (0.398, 0.330, 21.85, 0.15),
    8: (0.464, 0.309, 20.88, 0.10),
    9: (0.471, 0.432, 27.30, 0.10),
    10: (0.430, 0.321, 23.90, 0.06),
    11: (0.479, 0.423, 29.22, 0.05),
    12: (0.475, 0.385, 31.63, 0.03),
}


def lookup(class_codes) -> dict:
    """Arrays of porosity, effective porosity, psi_f (mm) and K (mm/h) for
    USDA class codes; NaN for unclassified cells and for Silt (no row)."""
    codes = np.asarray(class_codes)
    out = {k: np.full(codes.shape, np.nan) for k in ("porosity", "eff_porosity")}
    out["psi_f"] = np.full(codes.shape, np.nan)
    out["k"] = np.full(codes.shape, np.nan)
    for code, (phi, phi_e, psi_cm, k_cm_h) in RAWLS_1983.items():
        sel = codes == code
        out["porosity"][sel] = phi
        out["eff_porosity"][sel] = phi_e
        out["psi_f"][sel] = psi_cm * 10.0
        out["k"][sel] = k_cm_h * 10.0
    return out
