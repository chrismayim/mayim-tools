"""Sectioned CSV and results.json for the SANRAL Deterministic Peak Flows
tool (no QGIS dependency). Numbers use a decimal point and no thousands
separators."""

from __future__ import annotations

import csv
import math

from mayim_tools._common.results_json import build_results, result_entry, write_results

from . import tables as tb
from .core import RunResult, field_names
from .methods import METHOD_ORDER, METHODS

REFERENCES = (
    "SANRAL (2013) Drainage Manual, 6th edition, Chapter 3: Flood calculations. "
    "The South African National Roads Agency SOC Ltd, Pretoria.",
    "SANRAL (2013) Drainage Manual Application Guide, worked examples 3.1 and 3.2.",
    "Kovacs Z P (1988) Regional maximum flood peaks in southern Africa. Technical "
    "Report TR137, Department of Water Affairs, Pretoria.",
    "Alexander W J R (2001) Flood risk reduction measures incorporating flood "
    "hydrology for southern Africa. SA National Committee on Large Dams, Pretoria.",
    "DWS (2025) A Guide to Flood Frequency Estimation Methods as applied in the "
    "DWS (Flood Studies 2025), Section 5.4: Catchment Parameter method (CAPA).",
    "Pietersen J P J, Gericke O J, Smithers J C & Woyessa Y E (2024) Estimation of "
    "Areal Reduction Factors in South Africa, Part 1. J. SAICE 66(4).",
)

ABBREVIATIONS = (
    ("A, Ae", "km2", "catchment area; effective area (A less ineffective areas)"),
    ("L", "km", "longest watercourse (flow path) length"),
    ("S1085", "m/m", "10-85 slope of the longest watercourse"),
    ("Lc", "km", "outlet to the point on L nearest the centroid"),
    ("SA", "m/m", "mean catchment slope (CAPA)"),
    ("Tc", "h", "time of concentration"),
    ("T", "years", "return period (AEP = 1/T)"),
    ("P", "mm", "point rainfall depth for duration Tc"),
    ("Pi, I", "mm/h", "point intensity P/Tc; catchment intensity Pi x ARF"),
    ("ARF", "%", "areal reduction factor"),
    ("C_T, Ft", "-", "runoff coefficient for T; rural C1 adjustment (Table 3.8)"),
    ("M, R", "mm, days", "2-year 1-day TR102 depth; days per year with thunder"),
    ("K_T, K_P", "-", "Midgley & Pitman (Table 3.17) / CAPA (Table 5-2) factors"),
    ("Q", "m3/s", "peak flow"),
)

INPUT_COLUMNS = (
    ("catchment_id", "-"),
    ("area_km2", "km2"),
    ("ineffective_km2", "km2"),
    ("effective_km2", "km2"),
    ("lfp_km", "km"),
    ("s1085", "m/m"),
    ("lca_km", "km"),
    ("s_mean", "m/m"),
    ("tc_min", "min"),
    ("map_mm", "mm"),
    ("sdf_basin", "-"),
    ("veld_zone", "-"),
    ("kovacs_region", "-"),
    ("arf_region", "-"),
    ("mode", "-"),
)


def fmt(value, nd=3):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        text = f"{value:.{nd}f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    return str(value)


def catchment_inputs(c) -> dict:
    return {
        "catchment_id": c.catchment_id,
        "area_km2": c.area_km2,
        "ineffective_km2": c.ineffective_km2 or 0.0,
        "effective_km2": c.effective_area,
        "lfp_km": c.lfp_km,
        "s1085": c.s1085,
        "lca_km": c.lca_km,
        "s_mean": c.s_mean,
        "tc_min": c.tc_min,
        "map_mm": c.map_mm,
        "sdf_basin": c.sdf_basin,
        "veld_zone": c.veld_zone,
        "kovacs_region": c.kovacs_region,
        "arf_region": c.arf_region,
        "mode": c.mode,
    }


def settings_rows(run: RunResult) -> list:
    s = run.settings
    rows = [
        ("Methods", ", ".join(METHODS[m][1] for m in METHOD_ORDER if m in s.methods)),
        ("Return periods (years)", " ".join(str(T) for T in s.return_periods)),
        ("Tc floor 15 min (Rational)", "yes" if s.tc_floor else "no"),
        ("Apply ARF (Rational)", "yes" if s.apply_arf else "no"),
        (
            "ARF method (Rational)",
            (
                "Pietersen et al (2024) for D >= 24 h, else Eq 3.10"
                if s.arf_method == "pietersen"
                else "DM Eq 3.10"
            ),
        ),
        ("SDF", "own Tc (Eq 3.17), basin TR102 station, Eq 3.19 ARF (DM p 3-47)"),
        ("Custom Kovacs K", fmt(s.k_custom)),
        ("Thunder days R override", fmt(s.r_days)),
    ]
    if run.ddf_info:
        rows.append(("DDF site", str(run.ddf_info.get("site") or "")))
        if run.ddf_info.get("fixed_day_durations"):
            rows.append(
                (
                    "DDF fixed-day conversion",
                    "; ".join(run.ddf_info["fixed_day_durations"]),
                )
            )
    return rows


def write_csv(path, run: RunResult, run_info: dict) -> None:
    names = field_names(run.settings)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SANRAL Deterministic Peak Flows"])
        w.writerow([])
        w.writerow(["# 1 Abbreviations"])
        w.writerow(["symbol", "unit", "meaning"])
        w.writerows(ABBREVIATIONS)
        w.writerow([])
        w.writerow(["# 2 Run"])
        for k, v in run_info.items():
            w.writerow([k, v])
        for k, v in settings_rows(run):
            w.writerow([k, v])
        w.writerow([])
        w.writerow(["# 3 Catchment inputs"])
        w.writerow([c for c, _ in INPUT_COLUMNS])
        w.writerow([u for _, u in INPUT_COLUMNS])
        for item in run.items:
            vals = catchment_inputs(item.catchment)
            w.writerow([fmt(vals[c], 4) for c, _ in INPUT_COLUMNS])
        w.writerow([])
        w.writerow(["# 4 Peak flows (m3/s)"])
        w.writerow(["catchment_id"] + names)
        for item in run.items:
            q = item.by_field()
            w.writerow(
                [item.catchment.catchment_id] + [fmt(q.get(n), 2) for n in names]
            )
        w.writerow([])
        w.writerow(["# 5 Derivations"])
        w.writerow(["catchment_id", "method", "T_years", "quantity", "value"])
        for item in run.items:
            for r in item.results:
                for key, value in r.derivation.items():
                    w.writerow(
                        [
                            item.catchment.catchment_id,
                            r.label,
                            fmt(r.return_period),
                            key,
                            fmt(value, 4) if isinstance(value, float) else value,
                        ]
                    )
        w.writerow([])
        w.writerow(["# 6 Notes and flags"])
        w.writerow(["catchment_id", "method", "T_years", "note"])
        for item in run.items:
            cid = item.catchment.catchment_id
            for n in item.notes:
                w.writerow([cid, "", "", n])
            for r in item.results:
                for flag in r.flags:
                    w.writerow([cid, r.label, fmt(r.return_period), flag])
        for msg in run.warnings:
            w.writerow(["", "", "", msg])
        w.writerow([])
        w.writerow(["# 7 Application and source notes"])
        for m in ("rational", "sdf", "midgley_pitman", "rmf", "capa"):
            w.writerow([m, tb.APPLICABILITY[m]])
        for note in tb.SOURCE_NOTES:
            w.writerow(["source", note])
        w.writerow([])
        w.writerow(["# 8 Field definitions (output layer)"])
        w.writerow(["field", "unit", "definition"])
        for n in names:
            w.writerow([n, "m3/s", field_definition(n)])
        w.writerow(["notes", "-", "skipped methods and flags, by method code"])
        w.writerow([])
        w.writerow(["# 9 References"])
        for ref in REFERENCES:
            w.writerow([ref])


def field_definition(name: str) -> str:
    parts = name.split("_")
    code = parts[1]
    label = next(v[1] for v in METHODS.values() if v[0] == code)
    if len(parts) == 2:
        return f"{label}: peak flow"
    return f"{label}: 1:{parts[2]} year peak flow"


def build_results_json(run: RunResult, tool: dict, inputs: dict) -> dict:
    items = []
    for item in run.items:
        items.append(
            {
                "id": item.catchment.catchment_id,
                "attributes": catchment_inputs(item.catchment),
                "notes": list(item.notes),
                "results": [
                    result_entry(
                        "peak_flow",
                        r.method,
                        r.label,
                        r.return_period,
                        r.q_m3s,
                        "m3/s",
                        r.flags,
                        r.derivation,
                    )
                    for r in item.results
                ],
            }
        )
    return build_results(
        tool,
        inputs,
        items,
        {"peak_flow": "m3/s"},
        warnings=run.warnings,
        references=REFERENCES,
    )


def write_results_json(path, run: RunResult, tool: dict, inputs: dict) -> None:
    write_results(path, build_results_json(run, tool, inputs))
