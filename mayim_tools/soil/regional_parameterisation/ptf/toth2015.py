"""Tóth et al. (2015) continuous European pedotransfer functions - the
method behind HiHydroSoil v2.0 (Simons et al., 2020). No QGIS, no GDAL.

Inputs: USDA silt and clay (%), organic carbon OC (%), bulk density BD
(g/cm3), topsoil flag T/S (topsoil = 1, subsoil = 0), pH (H2O) and CEC
(cmol(+)/kg) for Ksat.

    theta_s     = f(BD, clay, silt)
    log10 alpha = f(BD, OC, T/S, clay, silt)           alpha in 1/cm
    log10 (n-1) = f(BD, T/S, clay, silt, OC)
    theta_r     = regression tree on sand
    log10 Ksat  = f(pH, T/S, clay, silt, CEC)          Ksat in cm/day

Water contents at 33 and 1500 kPa follow from the van Genuchten curve; the
Green-Ampt suction is the effective capillary drive (van_genuchten.py).

VERIFICATION: every coefficient was checked digit by digit against the
HiHydroSoil v2.0 report (Simons et al., 2020, section "Pedotransfer
functions", pp. 7-8), which this method replicates. Units as stated there:
OC in %, BD in g/cm3, pH in water, CEC in cmol(c)/kg (= meq/100 g); T/S = 1
for topsoil (0-30 cm), 0 otherwise. The report cites the functions as Tóth et
al. (2014) - the online-first date of the 2015 paper.

Differences from the HiHydroSoil product (documented in the report):
- field capacity here is theta at 33 kPa (as for every method in this tool);
  HiHydroSoil publishes pF2 (10 kPa) as field capacity;
- for its 0-30 cm aggregate HiHydroSoil averages Ksat harmonically over the
  standard depths; here the inputs are averaged to the layer first and the
  Ksat equation is applied to the layer average.
"""

from __future__ import annotations

import numpy as np

from . import van_genuchten as vg

VERIFIED = True

CITATION = (
    "Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G. and Tóth, G. "
    "(2015). New generation of hydraulic pedotransfer functions for Europe. "
    "European Journal of Soil Science 66: 226-238. doi:10.1111/ejss.12192"
)
CITATION_HIHYDROSOIL = (
    "Simons, G.W.H., Koster, R. and Droogers, P. (2020). HiHydroSoil v2.0 - "
    "A high resolution soil map of global hydraulic properties. FutureWater "
    "Report 213, Wageningen."
)

# (intercept, {input: coefficient}) - as printed in the HiHydroSoil v2.0 report
COEFFICIENTS = {
    "theta_s": (0.83080, {"bd": -0.28217, "clay": 0.0002728, "silt": 0.000187}),
    "log10_alpha": (
        -0.43348,
        {"bd": -0.41729, "oc": -0.04762, "ts": 0.21810, "clay": -0.01581,
         "silt": -0.01207},
    ),  # fmt: skip
    "log10_n_minus_1": (
        0.22236,
        {"bd": -0.30189, "ts": -0.05558, "clay": -0.005306, "silt": -0.003084,
         "oc": -0.01072},
    ),  # fmt: skip
    "log10_ksat_cm_day": (
        0.40220,
        {"ph": 0.26122, "ts": 0.44565, "clay": -0.02329, "silt": -0.01265,
         "cec": -0.01038},
    ),  # fmt: skip
}
THETA_R_SAND_SPLIT = 2.0  # sand (%) >= split -> 0.041, else 0.179
THETA_R_VALUES = (0.041, 0.179)

# Validity flags (bit mask)
FLAG_NO_KSAT_INPUTS = 32  # pH or CEC missing - Ksat not computed
FLAG_LABELS = {
    FLAG_NO_KSAT_INPUTS: "pH or CEC missing: Tóth et al. (2015) Ksat not computed",
}


def _linear(name, x: dict):
    a, coefs = COEFFICIENTS[name]
    out = a
    for key, b in coefs.items():
        out = out + b * x[key]
    return out


def toth2015(sand, silt, clay, oc_pct, bulk_density, topsoil, ph=None, cec=None):
    """Arrays (or scalars) in, dict of arrays out (see module docstring)."""
    x = {
        "silt": np.asarray(silt, dtype=np.float64),
        "clay": np.asarray(clay, dtype=np.float64),
        "oc": np.asarray(oc_pct, dtype=np.float64),
        "bd": np.asarray(bulk_density, dtype=np.float64),
        "ts": 1.0 if topsoil else 0.0,
    }
    sand = np.asarray(sand, dtype=np.float64)
    theta_s = _linear("theta_s", x)
    alpha = 10.0 ** _linear("log10_alpha", x)
    n = 1.0 + 10.0 ** _linear("log10_n_minus_1", x)
    theta_r = np.where(sand >= THETA_R_SAND_SPLIT, *THETA_R_VALUES)
    theta_r = np.where(np.isfinite(sand), theta_r, np.nan)
    shape = np.broadcast(sand, x["silt"], x["clay"], x["oc"], x["bd"]).shape
    flags = np.zeros(shape, dtype=np.int16)
    if ph is not None and cec is not None:
        x["ph"] = np.asarray(ph, dtype=np.float64)
        x["cec"] = np.asarray(cec, dtype=np.float64)
        ksat = np.broadcast_to(
            10.0 ** _linear("log10_ksat_cm_day", x) * 10.0 / 24.0, shape
        )  # cm/day -> mm/h
        missing = np.broadcast_to(
            ~(np.isfinite(x["ph"]) & np.isfinite(x["cec"])), shape
        )
    else:
        ksat = np.full(flags.shape, np.nan)
        missing = np.ones(flags.shape, dtype=bool)
    flags |= np.where(missing, FLAG_NO_KSAT_INPUTS, 0).astype(np.int16)
    t33 = vg.theta_at(vg.H33_CM, theta_r, theta_s, alpha, n)
    t1500 = vg.theta_at(vg.H1500_CM, theta_r, theta_s, alpha, n)
    return {
        "theta_s": theta_s,
        "theta_r": theta_r,
        "alpha": alpha,
        "n_vg": n,
        "theta_fc": t33,
        "theta_wp": t1500,
        "paw": t33 - t1500,
        "ksat": ksat,
        "psi_f": vg.capillary_drive_cm(alpha, n) * 10.0,  # mm
        "flags": flags,
    }


def run(inputs: dict, options: dict) -> dict:
    """Ensemble interface (see ptf/__init__.py)."""
    return toth2015(
        inputs["sand"],
        inputs["silt"],
        inputs["clay"],
        inputs["oc_pct"],
        inputs["bulk_density"],
        options.get("topsoil", False),
        ph=inputs.get("ph"),
        cec=inputs.get("cec"),
    )
