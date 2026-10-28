"""
Loss, transform and routing methods beyond SCS / Clark / Muskingum for
Prepare HEC-HMS project - zero QGIS dependency.

Phase 1 (2026-10-28):

- loss: Initial and Constant, Deficit and Constant, Green and Ampt;
- transform: Snyder (standard);
- routing: Lag and K, Kinematic Wave (trapezoid).

Each estimated parameter records where it came from (``params["sources"]``):
a field on the input layer (``attrs``), the Regional Soil Parameterisation
rasters (zonal statistics over the sub-catchment), the curve number, the
'Catchments: 1. Delineation' geometry, or a default. The basin-file keywords and
units were checked by computing and re-saving generated projects in
HEC-HMS 4.14 (metric: mm, mm/h, m3/m3, hours for Snyder Tp and Lag and K).

Soil parameters (Regional Soil Parameterisation, 0-30 cm, P50 bands named
``<code>_0-30cm_P50 (<units>)``): ksat (mm/h), psi_f (mm), theta_s,
theta_fc, theta_wp (m3/m3). Ksat is averaged geometrically over the
sub-catchment, the others arithmetically.
"""

from __future__ import annotations

import math
import os

import numpy as np

LOSS_SCS = "scs"
LOSS_IC = "initial_constant"
LOSS_DC = "deficit_constant"
LOSS_GA = "green_ampt"
LOSS_LABELS = {
    LOSS_SCS: "SCS curve number",
    LOSS_IC: "Initial and Constant",
    LOSS_DC: "Deficit and Constant",
    LOSS_GA: "Green and Ampt",
}

TRANSFORM_SNYDER = "snyder"
ROUTE_LAGK = "lag_and_k"
ROUTE_KW = "kinematic_wave"

# HEC-HMS 4.13 / 4.14 reject a Snyder standard lag below 0.1 h ("Invalid
# Snyder lag", ERROR 41941; found by computing, independent of the time step)
SNYDER_TP_MIN_H = 0.1
SNYDER_TP_LAG = "lag"  # Tp = 0.6 Tc (the SCS lag)
SNYDER_TP_CT = "ct"  # Tp = 0.75 Ct (L Lca)^0.3, L and Lca in km

ANTECEDENT_MOIST = "moist"  # initial water content = field capacity
ANTECEDENT_DRY = "dry"  # initial water content = wilting point

SOIL_LAYER = "0-30cm"
SOIL_CODES = ("ksat", "psi_f", "theta_s", "theta_fc", "theta_wp")
SOIL_UNITS = {
    "ksat": "mm/h",
    "psi_f": "mm",
    "theta_s": "m3/m3",
    "theta_fc": "m3/m3",
    "theta_wp": "m3/m3",
}
# Field names on the sub-catchment layer that override the estimates.
SUB_FIELDS = {
    "il_mm": "initial loss (mm)",
    "cl_mmh": "constant loss rate (mm/h)",
    "id_mm": "initial deficit (mm)",
    "md_mm": "maximum deficit (mm)",
    "ga_ks_mmh": "Green and Ampt hydraulic conductivity (mm/h)",
    "ga_psi_mm": "Green and Ampt wetting front suction (mm)",
    "ga_ts": "Green and Ampt saturated content (m3/m3)",
    "ga_ti": "Green and Ampt initial content (m3/m3)",
    "snyder_tp_h": "Snyder standard lag Tp (h)",
    "snyder_cp": "Snyder peaking coefficient Cp",
}
REACH_FIELDS = {
    "lagk_lag_h": "Lag and K: lag (h)",
    "lagk_k_h": "Lag and K: K (h)",
    "kw_n": "kinematic wave Manning n",
    "kw_width_m": "kinematic wave bottom width (m)",
}


class MethodError(ValueError):
    """A parameter that cannot be estimated."""


# ---------------------------------------------------------------- soil rasters


def find_soil_rasters(folder: str, describe=None) -> tuple[dict, list[str]]:
    """{code: (path, band number)} for the 0-30 cm P50 bands of the Regional
    Soil Parameterisation rasters in ``folder`` (recognised by their band
    descriptions, so project-numbered names work; the newest file wins).
    ``describe(path) -> [band descriptions]`` (GDAL by default)."""
    if not folder or not os.path.isdir(folder):
        raise MethodError(f"Soil raster folder not found: {folder}")
    describe = describe or _band_descriptions
    found: dict[str, list[tuple[float, str, int]]] = {}
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        path = os.path.join(folder, name)
        try:
            descs = describe(path)
        except Exception:  # noqa: BLE001 - unreadable file: skip it
            continue
        for i, d in enumerate(descs, start=1):
            key = (d or "").strip().split(" ")[0]
            for code in SOIL_CODES:
                if key == f"{code}_{SOIL_LAYER}_P50":
                    found.setdefault(code, []).append((os.path.getmtime(path), path, i))
    out, notes = {}, []
    for code, cands in found.items():
        cands.sort(reverse=True)
        out[code] = (cands[0][1], cands[0][2])
        if len(cands) > 1:
            notes.append(
                f"{len(cands)} {code} rasters in the soil folder; the newest, "
                f"{os.path.basename(cands[0][1])}, was used."
            )
    return out, notes


def _band_descriptions(path: str) -> list[str]:
    from osgeo import gdal

    gdal.UseExceptions()
    ds = gdal.Open(path)
    return [ds.GetRasterBand(b).GetDescription() for b in range(1, ds.RasterCount + 1)]


def zonal_values(
    path: str, band_no: int, zones_wkt: list[str], geometric: bool = False
) -> list[float | None]:
    """Mean of the raster cells in each zone (zones in the raster's CRS),
    geometric for log-normal parameters such as Ksat; the cell under the
    centroid when no cell centre falls inside. None where there is no
    value."""
    from osgeo import gdal, ogr

    gdal.UseExceptions()
    ds = gdal.Open(path)
    band = ds.GetRasterBand(band_no)
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    srs = ds.GetSpatialRef()
    mem_drv = gdal.GetDriverByName("MEM")
    ogr_drv = ogr.GetDriverByName("Memory")
    out: list[float | None] = []
    for wkt in zones_wkt:
        g = ogr.CreateGeometryFromWkt(wkt)
        minx, maxx, miny, maxy = g.GetEnvelope()
        c0 = max(0, int(math.floor((minx - gt[0]) / gt[1])) - 1)
        c1 = min(ds.RasterXSize, int(math.ceil((maxx - gt[0]) / gt[1])) + 1)
        r0 = max(0, int(math.floor((maxy - gt[3]) / gt[5])) - 1)
        r1 = min(ds.RasterYSize, int(math.ceil((miny - gt[3]) / gt[5])) + 1)
        if c1 <= c0 or r1 <= r0:
            out.append(None)
            continue
        arr = band.ReadAsArray(c0, r0, c1 - c0, r1 - r0).astype(float)
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        mem = mem_drv.Create("", c1 - c0, r1 - r0, 1, gdal.GDT_Byte)
        mem.SetGeoTransform(
            (gt[0] + c0 * gt[1], gt[1], 0, gt[3] + r0 * gt[5], 0, gt[5])
        )
        mem.SetProjection(ds.GetProjection())
        vds = ogr_drv.CreateDataSource("zone")
        lyr = vds.CreateLayer("z", srs, ogr.wkbMultiPolygon)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        lyr.CreateFeature(f)
        gdal.RasterizeLayer(mem, [1], lyr, burn_values=[1])
        vals = arr[mem.GetRasterBand(1).ReadAsArray() == 1]
        vals = vals[np.isfinite(vals)]
        if geometric:
            vals = vals[vals > 0]
        if vals.size == 0:
            c = g.Centroid()
            col = int((c.GetX() - gt[0]) / gt[1]) - c0
            row = int((c.GetY() - gt[3]) / gt[5]) - r0
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                v = arr[row, col]
                if np.isfinite(v) and (v > 0 or not geometric):
                    vals = np.array([v])
        if vals.size == 0:
            out.append(None)
        elif geometric:
            out.append(float(np.exp(np.mean(np.log(vals)))))
        else:
            out.append(float(np.mean(vals)))
    return out


# ---------------------------------------------------------------- settings


class MethodSettings:
    """Method choices and the defaults behind the estimates."""

    def __init__(
        self,
        loss: str = LOSS_SCS,
        ksat_factor: float = 1.0,
        antecedent: str = ANTECEDENT_MOIST,
        soil_depth_mm: float = 300.0,
        initial_loss_mm: float = 10.0,
        constant_rate_mmh: float = 5.0,
        snyder_tp: str = SNYDER_TP_LAG,
        snyder_ct: float = 2.0,
        snyder_cp: float = 0.6,
        lagk_lag_share: float = 0.5,
    ):
        if loss not in LOSS_LABELS:
            raise MethodError(f"Unknown loss method '{loss}'.")
        self.loss = loss
        self.ksat_factor = float(ksat_factor)
        self.antecedent = antecedent
        self.soil_depth_mm = float(soil_depth_mm)
        self.initial_loss_mm = float(initial_loss_mm)
        self.constant_rate_mmh = float(constant_rate_mmh)
        self.snyder_tp = snyder_tp
        self.snyder_ct = float(snyder_ct)
        self.snyder_cp = float(snyder_cp)
        self.lagk_lag_share = min(max(float(lagk_lag_share), 0.0), 1.0)


def field_value(attrs: dict | None, key: str) -> float | None:
    if not attrs:
        return None
    v = attrs.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _soil(soil: dict | None, code: str) -> float | None:
    if not soil:
        return None
    v = soil.get(code)
    return float(v) if v is not None and math.isfinite(v) else None


def initial_content_code(ms: MethodSettings) -> str:
    return "theta_wp" if ms.antecedent == ANTECEDENT_DRY else "theta_fc"


def antecedent_label(ms: MethodSettings) -> str:
    return (
        "wilting point (dry start)"
        if ms.antecedent == ANTECEDENT_DRY
        else "field capacity (moist start)"
    )


# ---------------------------------------------------------------- loss


def _constant_rate(name, attrs, soil, ms, src) -> float:
    v = field_value(attrs, "cl_mmh")
    if v is not None:
        src["constant_rate_mmh"] = "field cl_mmh"
        return max(v, 0.0)
    k = _soil(soil, "ksat")
    if k is not None:
        src["constant_rate_mmh"] = "Ksat (soil rasters, geometric mean)" + (
            f" x {ms.ksat_factor:g}" if ms.ksat_factor != 1 else ""
        )
        return k * ms.ksat_factor
    src["constant_rate_mmh"] = "default"
    return ms.constant_rate_mmh


def loss_parameters(
    name: str,
    attrs: dict | None,
    soil: dict | None,
    cn: float | None,
    ia_mm: float | None,
    s_mm: float | None,
    ms: MethodSettings,
) -> tuple[dict, dict, list[str]]:
    """(parameters, {parameter: source}, warnings) of the chosen loss method
    for one sub-basin (SCS is handled by the core)."""
    p: dict = {}
    src: dict = {}
    warns: list[str] = []
    if ms.loss == LOSS_IC:
        v = field_value(attrs, "il_mm")
        if v is not None:
            p["initial_loss_mm"], src["initial_loss_mm"] = max(v, 0.0), "field il_mm"
        elif ia_mm is not None:
            p["initial_loss_mm"], src["initial_loss_mm"] = ia_mm, "Ia from the CN"
        else:
            p["initial_loss_mm"], src["initial_loss_mm"] = ms.initial_loss_mm, "default"
        p["constant_rate_mmh"] = _constant_rate(name, attrs, soil, ms, src)
    elif ms.loss == LOSS_DC:
        md = field_value(attrs, "md_mm")
        ts, twp = _soil(soil, "theta_s"), _soil(soil, "theta_wp")
        if md is not None:
            src["max_deficit_mm"] = "field md_mm"
        elif ts is not None and twp is not None:
            md = max(ts - twp, 0.0) * ms.soil_depth_mm
            src["max_deficit_mm"] = (
                f"(theta_s - theta_wp) x {ms.soil_depth_mm:g} mm (soil rasters)"
            )
        elif s_mm is not None:
            md = s_mm
            src["max_deficit_mm"] = "S from the CN"
        else:
            raise MethodError(
                f"{name}: no maximum deficit (give the soil rasters, the CN or a "
                "md_mm field)."
            )
        p["max_deficit_mm"] = md
        v = field_value(attrs, "id_mm")
        ti = _soil(soil, initial_content_code(ms))
        if v is not None:
            p["initial_deficit_mm"], src["initial_deficit_mm"] = v, "field id_mm"
        elif ts is not None and ti is not None:
            p["initial_deficit_mm"] = max(ts - ti, 0.0) * ms.soil_depth_mm
            src["initial_deficit_mm"] = (
                f"(theta_s - theta_i) x {ms.soil_depth_mm:g} mm, "
                f"{antecedent_label(ms)}"
            )
        elif ia_mm is not None:
            p["initial_deficit_mm"], src["initial_deficit_mm"] = ia_mm, "Ia from the CN"
        else:
            p["initial_deficit_mm"], src["initial_deficit_mm"] = (
                ms.initial_loss_mm,
                "default",
            )
        if p["initial_deficit_mm"] > md:
            warns.append(f"{name}: initial deficit limited to the maximum deficit.")
            p["initial_deficit_mm"] = md
        p["constant_rate_mmh"] = _constant_rate(name, attrs, soil, ms, src)
        p["recovery_factor"] = 1.0
    elif ms.loss == LOSS_GA:
        spec = (
            ("ga_ks_mmh", "ksat", "Ksat"),
            ("ga_psi_mm", "psi_f", "wetting front suction psi_f"),
            ("ga_ts", "theta_s", "saturated content theta_s"),
            ("ga_ti", initial_content_code(ms), "initial content"),
        )
        for fld, code, label in spec:
            v = field_value(attrs, fld)
            if v is not None:
                p[fld], src[fld] = v, f"field {fld}"
                continue
            v = _soil(soil, code)
            if v is None:
                raise MethodError(
                    f"{name}: no {label} for Green and Ampt (give the soil raster "
                    f"folder or a {fld} field)."
                )
            if fld == "ga_ks_mmh":
                v *= ms.ksat_factor
                src[fld] = "Ksat (soil rasters, geometric mean)" + (
                    f" x {ms.ksat_factor:g}" if ms.ksat_factor != 1 else ""
                )
            elif fld == "ga_ti":
                src[fld] = f"{code} (soil rasters), {antecedent_label(ms)}"
            else:
                src[fld] = f"{code} (soil rasters)"
            p[fld] = v
        if not 0 < p["ga_ts"] <= 1:
            raise MethodError(f"{name}: saturated content {p['ga_ts']:g} is not 0-1.")
        if p["ga_ti"] >= p["ga_ts"]:
            warns.append(
                f"{name}: initial content limited to 0.99 x the saturated content."
            )
            p["ga_ti"] = 0.99 * p["ga_ts"]
        p["ga_ti"] = max(p["ga_ti"], 0.0)
    return p, src, warns


def loss_lines(loss: str, p: dict, num) -> list[tuple[str, object]]:
    """Basin-file key/value lines after 'Percent Impervious Area'."""
    if loss == LOSS_IC:
        return [
            ("Initial Loss", num(p["initial_loss_mm"], 3)),
            ("Constant Loss Rate", num(p["constant_rate_mmh"], 4)),
        ]
    if loss == LOSS_DC:
        return [
            ("Initial Deficit", num(p["initial_deficit_mm"], 3)),
            ("Maximum Deficit", num(p["max_deficit_mm"], 3)),
            ("Percolation Rate", num(p["constant_rate_mmh"], 4)),
            ("Recovery Factor", num(p["recovery_factor"], 3)),
        ]
    if loss == LOSS_GA:
        return [
            ("Initial Variable", "Initial Content"),
            ("Initial Content", num(p["ga_ti"], 4)),
            ("Saturated Content", num(p["ga_ts"], 4)),
            ("Capillarity Method", "SUCTION"),
            ("Wetting Front Suction", num(p["ga_psi_mm"], 3)),
            ("Hydraulic Conductivity", num(p["ga_ks_mmh"], 4)),
        ]
    raise MethodError(f"No basin-file lines for loss method '{loss}'.")


HMS_LOSS_NAMES = {
    LOSS_SCS: "SCS",
    LOSS_IC: "Initial+Constant",
    LOSS_DC: "Deficit Constant",
    LOSS_GA: "Green and Ampt",
}


# ---------------------------------------------------------------- transform


def snyder_parameters(
    name: str, attrs: dict | None, tc_min: float, ms: MethodSettings
) -> tuple[dict, dict, list[str]]:
    """Snyder standard lag Tp (h) and peaking coefficient Cp."""
    p, src, warns = {}, {}, []
    v = field_value(attrs, "snyder_tp_h")
    if v is not None and v > 0:
        p["snyder_tp_h"], src["snyder_tp_h"] = v, "field snyder_tp_h"
    elif ms.snyder_tp == SNYDER_TP_CT:
        lfp, lca = field_value(attrs, "lfp_km"), field_value(attrs, "lca_km")
        if lfp and lca and lfp > 0 and lca > 0:
            p["snyder_tp_h"] = 0.75 * ms.snyder_ct * (lfp * lca) ** 0.3
            src["snyder_tp_h"] = (
                f"0.75 Ct (L Lca)^0.3, Ct = {ms.snyder_ct:g} (lfp_km, lca_km)"
            )
        else:
            p["snyder_tp_h"] = 0.6 * tc_min / 60.0
            src["snyder_tp_h"] = "0.6 Tc (no lfp_km / lca_km fields)"
            warns.append(f"{name}: no lfp_km / lca_km; Snyder Tp = 0.6 Tc used.")
    else:
        p["snyder_tp_h"] = 0.6 * tc_min / 60.0
        src["snyder_tp_h"] = "0.6 Tc"
    if p["snyder_tp_h"] < SNYDER_TP_MIN_H:
        warns.append(
            f"{name}: Snyder Tp {p['snyder_tp_h'] * 60:.1f} min raised to the "
            f"HEC-HMS minimum of {SNYDER_TP_MIN_H * 60:g} min."
        )
        src["snyder_tp_h"] += f", raised to {SNYDER_TP_MIN_H:g} h (HEC-HMS minimum)"
        p["snyder_tp_h"] = SNYDER_TP_MIN_H
    v = field_value(attrs, "snyder_cp")
    if v is not None and 0 < v <= 1:
        p["snyder_cp"], src["snyder_cp"] = v, "field snyder_cp"
    else:
        p["snyder_cp"], src["snyder_cp"] = ms.snyder_cp, "default"
    return p, src, warns


# ---------------------------------------------------------------- routing


def lagk_parameters(
    attrs: dict | None, travel_min: float, ms: MethodSettings
) -> tuple[dict, dict]:
    """Constant lag and K (h): fields, else the travel time split by the lag
    share (lag = share x T, K = (1 - share) x T)."""
    p, src = {}, {}
    t_h = travel_min / 60.0
    v = field_value(attrs, "lagk_lag_h")
    if v is not None and v >= 0:
        p["lagk_lag_h"], src["lagk_lag_h"] = v, "field lagk_lag_h"
    else:
        p["lagk_lag_h"] = ms.lagk_lag_share * t_h
        src["lagk_lag_h"] = f"{ms.lagk_lag_share:g} x travel time"
    v = field_value(attrs, "lagk_k_h")
    if v is not None and v >= 0:
        p["lagk_k_h"], src["lagk_k_h"] = v, "field lagk_k_h"
    else:
        p["lagk_k_h"] = (1.0 - ms.lagk_lag_share) * t_h
        src["lagk_k_h"] = f"{1.0 - ms.lagk_lag_share:g} x travel time"
    return p, src


def kw_subreaches(length_m: float, celerity_ms: float, dt_min: float) -> int:
    """Sub-reaches so that each is about one time step of travel (dx ~ c dt),
    at least 1 and at most 100."""
    if not length_m or celerity_ms <= 0 or dt_min <= 0:
        return 1
    return int(min(100, max(1, round(length_m / (celerity_ms * dt_min * 60.0)))))
