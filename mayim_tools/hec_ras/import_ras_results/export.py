"""
Import RAS result data - the peak-values CSV (no QGIS).

Sectioned (Mayim Tools output standard 4.6): [run], [structures],
[boundary conditions], [profile lines]; UTF-8 with BOM, a column key,
no thousands separators, missing values as "-".
"""

from __future__ import annotations

import csv
import math

from .core import PlanResults

PROFILE_COLUMNS = [
    ("name", "profile line name"),
    ("area_2d", "2D flow area"),
    ("q_peak_m3s", "peak flow across the line, m3/s (positive downstream)"),
    ("t_q_peak_h", "time of the peak flow, hours from the start of the run"),
    ("volume_m3", "volume passed across the line in the run, m3"),
    ("wse_max_m", "highest peak water level of the cells on the line, m"),
    ("wse_mean_m", "mean peak water level of those cells, m"),
    ("depth_max_m", "largest peak depth of those cells, m"),
    ("v_max_ms", "largest peak velocity of those cells, m/s"),
    ("faces", "number of mesh faces the line cuts"),
    ("note", "why a value is missing or approximate"),
]
PEAK_COLUMNS = [
    ("kind", "structure or boundary condition"),
    ("name", "HEC-RAS name"),
    ("area_2d", "2D flow area"),
    ("variable", "HEC-RAS variable"),
    ("unit", "unit"),
    ("peak", "largest value in the run (largest magnitude)"),
    ("time_h", "time of the peak, hours from the start of the run"),
]


def _v(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "-" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def profile_rows(plan: PlanResults, lines) -> list[dict]:
    rows = []
    for area in plan.areas:
        for i, r in sorted(area.profiles.items()):
            rows.append(
                {
                    "name": lines[i].name,
                    "area_2d": area.name,
                    "q_peak_m3s": r["q_peak"],
                    "t_q_peak_h": r["t_q_peak_h"],
                    "volume_m3": r["volume_m3"],
                    "wse_max_m": r["wse_max"],
                    "wse_mean_m": r["wse_mean"],
                    "depth_max_m": r["depth_max"],
                    "v_max_ms": r["v_max"],
                    "faces": r["faces"],
                    "note": r["note"],
                }
            )
    return rows


def write_peaks_csv(path: str, plan: PlanResults, lines, settings: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["# Column key"])
        for name, text in PEAK_COLUMNS + PROFILE_COLUMNS:
            w.writerow([f"# {name}", text])
        w.writerow(["[run]"])
        w.writerow(["item", "value"])
        w.writerow(["plan file", plan.path])
        w.writerow(["plan", plan.short_id or plan.plan_name or "-"])
        w.writerow(["flow", plan.flow_title or "-"])
        for key, value in settings.items():
            w.writerow([key, value])
        for key, value in plan.summary.items():
            if key != "Compute messages":
                w.writerow([key, value])
        w.writerow([])
        for section, kind in (
            ("[structures]", "SA/2D connection"),
            ("[boundary conditions]", "Boundary condition"),
        ):
            w.writerow([section])
            w.writerow([c for c, _ in PEAK_COLUMNS])
            items = plan.structures if kind == "SA/2D connection" else plan.boundaries
            for item in items:
                w.writerow(
                    [item["kind"], item["name"], item["area"], item["variable"],
                     item["unit"] or "-", _v(item["peak"]), _v(item["time_h"], 2)]
                )  # fmt: skip
            w.writerow([])
        w.writerow(["[profile lines]"])
        w.writerow([c for c, _ in PROFILE_COLUMNS])
        for row in profile_rows(plan, lines):
            w.writerow(
                [_v(row[c], 2 if c in ("t_q_peak_h", "volume_m3") else 3)
                 for c, _ in PROFILE_COLUMNS]
            )  # fmt: skip
