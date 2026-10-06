"""Mualem-van Genuchten helpers shared by the van Genuchten methods
(no QGIS, no GDAL).

Retention (van Genuchten, 1980), h = suction head (cm, positive), m = 1 - 1/n:
    theta(h) = theta_r + (theta_s - theta_r) / (1 + (alpha h)^n)^m

Green-Ampt wetting-front suction = the effective capillary drive of the
Mualem conductivity curve (L = 0.5), in the closed form of Morel-Seytoux et
al. (1996):
    Hc = (1/alpha) (0.046 m + 2.07 m^2 + 19.5 m^3) / (1 + 4.7 m + 16 m^2)
The test suite checks this against direct numerical integration of
Kr(h) from 0 to infinity (agreement within 2 % for n = 1.09-2.68).
"""

from __future__ import annotations

import numpy as np

CITATION_VG = (
    "van Genuchten, M.Th. (1980). A closed-form equation for predicting the "
    "hydraulic conductivity of unsaturated soils. Soil Science Society of "
    "America Journal 44: 892-898."
)
CITATION_MS96 = (
    "Morel-Seytoux, H.J., Meyer, P.D., Nachabe, M., Touma, J., van Genuchten, "
    "M.Th. and Lenhard, R.J. (1996). Parameter equivalence for the Brooks-Corey "
    "and van Genuchten soil characteristics: preserving the effective capillary "
    "drive. Water Resources Research 32(5): 1251-1258."
)

KPA_TO_CM = 10.197162  # 1 kPa of suction = 10.197 cm of water
H33_CM = 33.0 * KPA_TO_CM
H1500_CM = 1500.0 * KPA_TO_CM


def theta_at(h_cm, theta_r, theta_s, alpha, n):
    """Water content (m3/m3) at suction head h (cm)."""
    n = np.asarray(n, dtype=np.float64)
    m = 1.0 - 1.0 / n
    with np.errstate(invalid="ignore", over="ignore"):
        se = (1.0 + (np.asarray(alpha) * h_cm) ** n) ** (-m)
    return theta_r + (theta_s - theta_r) * se


def capillary_drive_cm(alpha, n):
    """Effective capillary drive (cm), Morel-Seytoux et al. (1996)."""
    n = np.asarray(n, dtype=np.float64)
    m = 1.0 - 1.0 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return (1.0 / np.asarray(alpha)) * (
            (0.046 * m + 2.07 * m**2 + 19.5 * m**3) / (1.0 + 4.7 * m + 16.0 * m**2)
        )


def mualem_kr(h_cm, alpha, n, tortuosity=0.5):
    """Mualem relative conductivity at suction h (for tests and checks)."""
    n = np.asarray(n, dtype=np.float64)
    m = 1.0 - 1.0 / n
    se = (1.0 + (alpha * h_cm) ** n) ** (-m)
    return se**tortuosity * (1.0 - (1.0 - se ** (1.0 / m)) ** m) ** 2
