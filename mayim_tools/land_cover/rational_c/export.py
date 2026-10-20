"""
CSV outputs of "Run-off Coefficients (SANRAL Deterministic)" (no QGIS dependency): the C
CSV read by SANRAL Deterministic Peak Flows, the catchment attribute rows
(also used for the shapefile) and the sectioned results CSV. CSVs are UTF-8
with BOM; numbers have no thousands separators.
"""

from __future__ import annotations

import csv
import math

from . import cells as cl
from . import lookup as lk
from . import tables as tb
from .core import METHODS, TOOL_NAME, TOOL_VERSION, CatchmentResult, RunResult

C_COLUMNS = ("catchment_id", "alpha", "c1", "ft_class", "beta", "c2", "gamma", "c3")
SHORT = ("vlei", "flat", "hill", "steep")
PERIOD_FIELDS = [(t, f"C_{t}") for t in tb.RETURN_PERIODS] + [("PMF", "C_PMF")]

# field -> definition (all fields of the catchment layer, at most 10 characters)
FIELDS = {
    "catch_id": "Catchment ID",
    "A_km2": "Catchment area (km2)",
    "A_eff_km2": "Effective area (km2): the area less the ineffective classes",
    "A_inef_km2": "Ineffective area (km2): pans, sewage ponds, flooded pits, ocean",
    "alpha": "Rural fraction of the effective area",
    "beta": "Urban fraction of the effective area",
    "gamma": "Lakes and dams fraction of the effective area",
    **{
        f"Cs{i + 1}_pct": f"Slope class {i + 1} ({tb.SLOPE_CLASSES[i]}), % of rural area"
        for i in range(4)
    },
    **{
        f"Cp{i + 1}_pct": f"Permeability class {i + 1} ({tb.PERMEABILITY_CLASSES[i]}), % of rural area"
        for i in range(4)
    },
    **{
        f"Cv{i + 1}_pct": f"Vegetation class {i + 1} ({tb.VEGETATION_CLASSES[i]}), % of rural area"
        for i in range(4)
    },
    "Cs": "Surface slope component (Table 3.7, share-weighted)",
    "Cp": "Permeability component (Table 3.7, share-weighted)",
    "Cv": "Vegetation component (Table 3.7, share-weighted)",
    "C1": "Rural run-off coefficient C1 = Cs + Cp + Cv",
    "C1_P05": "C1 lower estimate (P05, less run-off)",
    "C1_P95": "C1 upper estimate (P95, more run-off)",
    "soil_unc": "Share of the C1 uncertainty from the soil groups (%)",
    "D_pct": "Dolomite, % of the rural area",
    "D_factor": "Weighted dolomite factor sum(Dfactor x Cs%)",
    "C1D": "C1 after the dolomite reduction: C1 (1 - D%) + C1 D% D_factor",
    "ft_class": "Table 3.8 row: steep_impermeable or flat_permeable",
    "ft_score": "Ft score: 0.5 (hilly + steep share) + 0.5 (semi-permeable + impermeable share)",
    "ft_src": "Ft class from the score or the override field",
    "C2": "Urban run-off coefficient C2 (area-weighted)",
    "C2_P05": "C2 lower end of the Table 3.7 ranges",
    "C2_P95": "C2 upper end of the Table 3.7 ranges",
    "C3": "Lakes and dams run-off coefficient C3",
    "MAP_mm": "Mean annual precipitation (mm)",
    "MAP_band": "Table 3.7 MAP band (mm)",
    **{
        f: f"C_T = alpha C1D Ft + beta C2 + gamma C3, {t}-year"
        for t, f in PERIOD_FIELDS[:-1]
    },
    "C_PMF": "C for the PMF: C1 = Cs + Cp,max + Cv,max, C2 = 1, Ft = 1",
    "n_flags": "Number of flags",
    "flags": "Flags (see the report)",
}


def _r(v, nd=4):
    if v is None:
        return None
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return round(float(v), nd)


def attributes(cr: CatchmentResult) -> dict:
    k = cr.coeff
    sh = cr.shares
    row = {
        "catch_id": cr.cid,
        "A_km2": _r(cr.area_km2, 4),
        "A_eff_km2": _r(cr.a_eff_km2, 4),
        "A_inef_km2": _r(cr.a_ineff_km2, 4),
    }
    if k is None:
        return {f: row.get(f) for f in FIELDS} | {
            "n_flags": len(cr.flags),
            "flags": "; ".join(cr.flags)[:254],
        }
    row |= {"alpha": _r(k.alpha), "beta": _r(k.beta), "gamma": _r(k.gamma)}
    for pre, vals in (
        ("Cs", sh.slope if sh else None),
        ("Cp", sh.perm if sh else None),
        ("Cv", sh.veg if sh else None),
    ):
        for i in range(4):
            row[f"{pre}{i + 1}_pct"] = _r(100 * vals[i], 2) if vals else None
    rural = k.alpha > 0 and sh is not None
    row |= {
        "Cs": _r(k.cs) if rural else None,
        "Cp": _r(k.cp) if rural else None,
        "Cv": _r(k.cv) if rural else None,
        "C1": _r(k.c1) if rural else None,
        "C1_P05": _r(k.c1_p05) if rural else None,
        "C1_P95": _r(k.c1_p95) if rural else None,
        "soil_unc": _r(100 * k.soil_share, 1) if rural else None,
        "D_pct": _r(100 * k.d_fraction, 2) if rural else None,
        "D_factor": _r(k.d_factor) if rural else None,
        "C1D": _r(k.c1d) if rural else None,
        "ft_class": k.ft_class,
        "ft_score": _r(k.ft_score, 3) if rural else None,
        "ft_src": k.ft_source,
        "C2": _r(k.c2) if k.beta > 0 else None,
        "C2_P05": _r(k.c2_p05) if k.beta > 0 else None,
        "C2_P95": _r(k.c2_p95) if k.beta > 0 else None,
        "C3": _r(k.c3, 3),
        "MAP_mm": _r(k.map_mm, 1),
        "MAP_band": tb.MAP_BANDS[k.band],
    }
    for t, f in PERIOD_FIELDS:
        row[f] = _r(k.c_t.get(t))
    row["n_flags"] = len(cr.flags)
    row["flags"] = "; ".join(cr.flags)[:254]
    return row


def c_rows(r: RunResult) -> list[list]:
    """Rows of the C CSV (SANRAL Deterministic Peak Flows): c1 is C1D."""
    rows = []
    for cr in r.catchments:
        k = cr.coeff
        if k is None:
            continue
        rows.append(
            [
                cr.cid,
                _r(k.alpha),
                _r(k.c1d) if k.alpha > 0 else 0,
                k.ft_class,
                _r(k.beta),
                _r(k.c2) if k.beta > 0 else 0,
                _r(k.gamma),
                _r(k.c3, 3) if k.gamma > 0 else 0,
            ]
        )
    return rows


def write_c_csv(path: str, r: RunResult) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(C_COLUMNS)
        for row in c_rows(r):
            w.writerow(["" if v is None else v for v in row])


def _v(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if not math.isfinite(v) else f"{v:.6g}"
    return v


def write_results_csv(path: str, r: RunResult, inputs: list | None = None) -> None:
    """Sectioned CSV: run, inputs, abbreviations, catchment values, C_T,
    class areas, flags, lookup used and the source tables."""
    s = r.settings
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["[run]"])
        w.writerow(["Tool", f"{TOOL_NAME} {TOOL_VERSION}"])
        w.writerow(["Run time", r.run_time_utc])
        w.writerow(["Method", "SANRAL (2013) Drainage Manual 6th ed., Section 3.5.1.2"])
        w.writerow([])
        w.writerow(["[inputs and settings]"])
        for k, v in inputs or []:
            w.writerow([k, _v(v)])
        w.writerow(["Analysis cell (m)", s.cell_m])
        w.writerow(["Soil-group method", METHODS.get(r.soil.method, r.soil.method)])
        w.writerow(["Urban C2 within the Table 3.7 range", s.position])
        w.writerow(
            ["Wetlands as impermeable", "yes" if s.wetlands_impermeable else "no"]
        )
        w.writerow(["C3", s.c3])
        w.writerow(["Ft on C2", "yes" if s.ft_on_c2 else "no"])
        w.writerow(["DEM resampling to the analysis cell", r.dem_resampling])
        w.writerow([])
        w.writerow(["[abbreviations]"])
        for k, v in FIELDS.items():
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["[catchments]"])
        rows = [attributes(cr) for cr in r.catchments]
        w.writerow(list(FIELDS))
        for row in rows:
            w.writerow([_v(row.get(k)) for k in FIELDS])
        w.writerow([])
        w.writerow(["[C_T borderline Ft: the other Table 3.8 row]"])
        w.writerow(["catch_id", "ft_class"] + [f for _, f in PERIOD_FIELDS[:-1]])
        for cr in r.catchments:
            k = cr.coeff
            if k and k.c_t_other_ft:
                other = (
                    "flat_permeable"
                    if k.ft_class == "steep_impermeable"
                    else "steep_impermeable"
                )
                w.writerow(
                    [cr.cid, other]
                    + [_v(_r(k.c_t_other_ft[t])) for t, _ in PERIOD_FIELDS[:-1]]
                )
        w.writerow([])
        w.writerow(["[slope classes, all cells: analysis cell and native DEM]"])
        w.writerow(["catch_id", "cell (m)"] + [f"{c} (%)" for c in tb.SLOPE_CLASSES])
        for cr in r.catchments:
            w.writerow([cr.cid, s.cell_m] + [_v(_r(100 * x, 2)) for x in cr.slope_all])
            if cr.slope_native:
                w.writerow(
                    [cr.cid, _r(cr.native_res, 2)]
                    + [_v(_r(100 * x, 2)) for x in cr.slope_native]
                )
        w.writerow([])
        w.writerow(["[land-cover classes (km2)]"])
        w.writerow(["catch_id", "code", "class", "role", "area (km2)"])
        for cr in r.catchments:
            scale = cr.area_km2 / cr.cells if cr.cells else 0.0
            for code, n in sorted(cr.class_cells.items()):
                rule = r.lookup.rule(code)
                w.writerow(
                    [
                        cr.cid,
                        code,
                        rule.name if rule else "-",
                        rule.role if rule else "not in lookup",
                        _v(_r(n * scale, 4)),
                    ]
                )
        w.writerow([])
        w.writerow(["[urban rows (% of urban area)]"])
        w.writerow(["catch_id", "Table 3.7 row", "share (%)"])
        for cr in r.catchments:
            for key, v in sorted(cr.urban_rows.items()):
                label = tb.URBAN_ROWS[key][0] if key in tb.URBAN_ROWS else "Lawns"
                w.writerow([cr.cid, label, _v(_r(100 * v, 2))])
        w.writerow([])
        w.writerow(["[permeability ranges (% of rural area): P05 and P95 soil groups]"])
        w.writerow(
            ["catch_id", "estimate"] + [f"{c} (%)" for c in tb.PERMEABILITY_CLASSES]
        )
        for cr in r.catchments:
            if cr.shares:
                for lab, vals in (
                    ("design", cr.shares.perm),
                    ("P05", cr.shares.perm_lo),
                    ("P95", cr.shares.perm_hi),
                ):
                    w.writerow([cr.cid, lab] + [_v(_r(100 * x, 2)) for x in vals])
        w.writerow([])
        w.writerow(["[vegetation ranges (% of rural area): lower and upper classes]"])
        w.writerow(
            ["catch_id", "estimate"] + [f"{c} (%)" for c in tb.VEGETATION_CLASSES]
        )
        for cr in r.catchments:
            if cr.shares:
                for lab, vals in (
                    ("design", cr.shares.veg),
                    ("lower", cr.shares.veg_lo),
                    ("upper", cr.shares.veg_hi),
                ):
                    w.writerow([cr.cid, lab] + [_v(_r(100 * x, 2)) for x in vals])
        w.writerow([])
        w.writerow(["[checks]"])
        w.writerow(
            [
                "catch_id",
                "rural cells with slope (%)",
                "rural cells with soil group (%)",
                "rock / wetlands forced impermeable (% rural)",
                "SANLC wetlands (% rural)",
                "texture agreement (%)",
            ]
        )
        for cr in r.catchments:
            w.writerow(
                [
                    cr.cid,
                    _v(_r(100 * cr.slope_known, 1)),
                    _v(_r(100 * cr.perm_known, 1)),
                    _v(_r(100 * cr.forced_impermeable, 1)),
                    _v(_r(100 * cr.wetland_share, 1)),
                    _v(
                        _r(100 * cr.texture_agree, 1)
                        if cr.texture_agree is not None
                        else None
                    ),
                ]
            )
        w.writerow([])
        w.writerow(["[flags]"])
        w.writerow(["catch_id", "flag"])
        for cr in r.catchments:
            for fl in cr.flags:
                w.writerow([cr.cid, fl])
        for warn in r.warnings:
            w.writerow(["run", warn])
        w.writerow([])
        w.writerow(["[lookup used]"])
        w.writerow(lk.CSV_COLUMNS)
        for rule in sorted(r.lookup.rules.values(), key=lambda x: x.code):
            w.writerow(
                [
                    rule.code,
                    rule.name,
                    rule.role,
                    lk._veg_key(rule.cv),
                    lk._veg_key(rule.cv_low),
                    lk._veg_key(rule.cv_high),
                    rule.urban or "",
                    rule.flag,
                ]
            )
        w.writerow([])
        w.writerow([f"[source tables: {tb.SOURCE}]"])
        w.writerow(["component", "class"] + [f"MAP {b} mm" for b in tb.MAP_BANDS])
        for name, classes, table in (
            ("Cs", tb.SLOPE_CLASSES, tb.CS),
            ("Cp", tb.PERMEABILITY_CLASSES, tb.CP),
            ("Cv", tb.VEGETATION_CLASSES, tb.CV),
        ):
            for c, vals in zip(classes, table, strict=True):
                w.writerow([name, c, *vals])
        w.writerow(["C2", "use", "low", "high"])
        for _k, (label, lo, hi) in tb.URBAN_ROWS.items():
            w.writerow(["C2", label, lo, hi])
        w.writerow(["Dolomite factor"] + list(tb.SLOPE_CLASSES))
        w.writerow([""] + list(tb.DOLOMITE_FACTORS))
        w.writerow(
            ["Ft (Table 3.8)", "return period (years)"] + list(tb.RETURN_PERIODS)
        )
        for key, vals in tb.FT.items():
            w.writerow(["Ft", tb.FT_LABELS[key], *vals])
        w.writerow(["Permeability half-classes", *cl.PERM2_LABELS.values()])
