"""
Sectioned CSV of the HEC-HMS model set-up (and the computed peak flows) -
zero QGIS dependency. Numbers are written plainly, without thousands
separators.
"""

from __future__ import annotations

import csv
import math

from . import methods as mth
from .core import ROUTE_LAG, ROUTE_MC, ROUTE_MUSKINGUM, TRANSFORM_CLARK, ProjectResult
from .storms import SOURCE_LABELS, ari_label, duration_label

SUBBASIN_FIELDS = [
    ("name", "-", "HEC-HMS sub-basin name"),
    ("downstream", "-", "element it drains to"),
    ("area_km2", "km2", "area"),
    ("cn", "-", "SCS curve number used (AMC as in the CN raster)"),
    ("cn_mean", "-", "area-weighted mean CN of the raster cells"),
    ("cn_runoff", "-", "runoff-equivalent CN at the CN design depth"),
    ("cn_p05", "-", "5th percentile of the cell CNs"),
    ("cn_p95", "-", "95th percentile of the cell CNs"),
    ("s_mm", "mm", "potential maximum retention S = 25400/CN - 254"),
    ("ia_mm", "mm", "initial abstraction Ia = lambda S"),
    ("imperv_pct", "%", "directly connected impervious area"),
    ("loss", "-", "loss method"),
    ("initial_loss_mm", "mm", "Initial and Constant: initial loss"),
    ("constant_rate_mmh", "mm/h", "Initial / Deficit and Constant: constant rate"),
    ("initial_deficit_mm", "mm", "Deficit and Constant: initial deficit"),
    ("max_deficit_mm", "mm", "Deficit and Constant: maximum deficit"),
    ("ga_ks_mmh", "mm/h", "Green and Ampt: hydraulic conductivity"),
    ("ga_psi_mm", "mm", "Green and Ampt: wetting front suction"),
    ("ga_ts", "m3/m3", "Green and Ampt: saturated content"),
    ("ga_ti", "m3/m3", "Green and Ampt: initial content"),
    ("soil_ksat", "mm/h", "soil rasters: Ksat 0-30 cm (geometric mean)"),
    ("soil_psi_f", "mm", "soil rasters: wetting front suction 0-30 cm"),
    ("soil_theta_s", "m3/m3", "soil rasters: saturated content 0-30 cm"),
    ("soil_theta_fc", "m3/m3", "soil rasters: field capacity 0-30 cm"),
    ("soil_theta_wp", "m3/m3", "soil rasters: wilting point 0-30 cm"),
    ("tc_raw_min", "min", "time of concentration from the input"),
    ("tc_min", "min", "time of concentration used (after the minimum)"),
    ("lag_min", "min", "SCS lag = 0.6 Tc"),
    ("clark_tc_h", "h", "Clark time of concentration"),
    ("clark_r_h", "h", "Clark storage coefficient R"),
    ("snyder_tp_h", "h", "Snyder standard lag Tp"),
    ("snyder_cp", "-", "Snyder peaking coefficient Cp"),
    ("sources", "-", "where each estimated parameter came from"),
    ("lat", "deg", "centroid latitude"),
    ("lon", "deg", "centroid longitude"),
    ("flags", "-", "tc_floor: Tc raised to the minimum"),
]
REACH_FIELDS = [
    ("name", "-", "HEC-HMS reach name"),
    ("downstream", "-", "element it flows into"),
    ("method", "-", "routing method"),
    ("length_m", "m", "reach length"),
    ("slope", "m/m", "channel slope (Muskingum-Cunge)"),
    ("travel_min", "min", "travel time (Muskingum K, or length / velocity)"),
    ("k_source", "-", "where the travel time came from"),
    ("k_h", "h", "Muskingum K"),
    ("x", "-", "Muskingum X"),
    ("steps", "-", "Muskingum sub-reaches for the time step"),
    ("lag_min", "min", "Lag routing lag"),
    ("lagk_lag_h", "h", "Lag and K: constant lag"),
    ("lagk_k_h", "h", "Lag and K: constant K"),
    ("subreaches", "-", "kinematic wave sub-reaches (dx about celerity x time step)"),
    ("n", "-", "Manning n (Muskingum-Cunge / kinematic wave)"),
    ("width_m", "m", "bottom width (trapezoid)"),
    ("side_slope", "H:V", "side slope (trapezoid)"),
    ("celerity_ms", "m/s", "index celerity (Muskingum-Cunge / kinematic wave)"),
    ("sources", "-", "where each estimated parameter came from"),
    ("flags", "-", "lag_fallback: too short for Muskingum; min_slope: slope raised"),
]
STORM_FIELDS = [
    ("run", "-", "simulation run and meteorologic model name"),
    ("source", "-", "rainfall source"),
    ("ari_years", "years", "return period"),
    ("duration", "-", "storm duration"),
    ("total_mm", "mm", "storm depth"),
    (
        "peak_mmh",
        "mm/h",
        "peak intensity of the hyetograph (5-min for a frequency storm)",
    ),
    ("interval_min", "min", "hyetograph time step"),
    ("pattern", "-", "temporal pattern"),
    ("control", "-", "control specifications"),
    ("arf_storm", "-", "areal reduction factor at the storm duration"),
    (
        "arf_peak",
        "-",
        "ARF at the shortest reduced duration (nested / alternating block peak burst)",
    ),
    (
        "burst_min_pct",
        "%",
        "burst check: lowest ratio of a burst inside the storm to the areal DDF depth",
    ),
    ("burst_min_at", "-", "duration of that lowest ratio"),
    ("flags", "-", "extrap_short / extrap_long / extrap_T: DDF extrapolated"),
]
RESULT_FIELDS = [
    ("run", "-", "simulation run"),
    ("element", "-", "element"),
    ("kind", "-", "element type"),
    ("peak_m3s", "m3/s", "peak outflow"),
    ("peak_time", "-", "time of the peak (storm starts 01Jan2000 00:00)"),
    ("volume_m3", "m3", "outflow volume"),
    ("depth_mm", "mm", "outflow depth over the drainage area"),
    ("precip_mm", "mm", "precipitation total (sub-basins)"),
]


def _v(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.6g}" if abs(v) < 1e5 else f"{v:.1f}"
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {x}" for k, x in v.items())
    return v


def method_labels(r: ProjectResult) -> dict:
    s = r.settings
    routes = sorted({e.params.get("method") for e in r.network.of("Reach")} - {None})
    names = {
        ROUTE_MUSKINGUM: "Muskingum",
        ROUTE_LAG: "Lag",
        ROUTE_MC: "Muskingum-Cunge",
        mth.ROUTE_LAGK: "Lag and K",
        mth.ROUTE_KW: "Kinematic Wave",
    }
    ms = s.methods
    if ms.loss == mth.LOSS_SCS:
        loss = f"SCS curve number, Ia = {s.lambda_ia:g} S"
    else:
        loss = mth.LOSS_LABELS[ms.loss]
    if s.transform == TRANSFORM_CLARK:
        transform = f"Clark unit hydrograph, R/(Tc+R) = {s.clark_ratio:g}"
    elif s.transform == mth.TRANSFORM_SNYDER:
        transform = "Snyder unit hydrograph (standard)"
    else:
        transform = "SCS unit hydrograph, lag = 0.6 Tc"
    return {
        "loss": loss,
        "transform": transform,
        "routing": ", ".join(names.get(m, m) for m in routes) or "-",
        "cn": (
            f"runoff-equivalent CN at {s.cn_design_depth_mm:g} mm"
            if s.cn_method == "runoff"
            else "area-weighted mean CN"
        ),
    }


def subbasin_rows(r: ProjectResult) -> list[dict]:
    rows = []
    for e in r.network.of("Subbasin"):
        d = {"name": e.name, "downstream": e.downstream, "flags": e.flags}
        d.update(e.params)
        rows.append(d)
    return rows


def reach_rows(r: ProjectResult) -> list[dict]:
    rows = []
    for e in r.network.of("Reach"):
        d = {"name": e.name, "downstream": e.downstream, "flags": e.flags}
        d.update(e.params)
        rows.append(d)
    return rows


def storm_rows(r: ProjectResult) -> list[dict]:
    rows = []
    for run in r.runs:
        st = run.storm
        rows.append(
            {
                "run": run.name,
                "source": SOURCE_LABELS.get(st.source, st.source),
                "ari_years": st.ari,
                "duration": duration_label(st.duration_min),
                "total_mm": st.total_mm,
                "peak_mmh": st.peak_intensity_mmh,
                "interval_min": st.interval_min,
                "pattern": st.pattern,
                "control": run.control.name,
                "arf_storm": st.arf_total if st.arf_values else None,
                "arf_peak": st.arf_peak if len(st.arf_values) > 1 else None,
                "burst_min_pct": burst_min(st)[0],
                "burst_min_at": burst_min(st)[1],
                "flags": st.flags,
            }
        )
    return rows


def burst_min(st) -> tuple[float | None, str]:
    """(lowest burst ratio %, its duration label) or (None, '')."""
    vals = [(b[3], b[0]) for b in getattr(st, "burst", []) or [] if math.isfinite(b[3])]
    if not vals:
        return None, ""
    v, d = min(vals)
    return v, duration_label(d)


def arf_label(arf) -> str:
    return arf.label() if arf is not None else "None"


def arf_table(storms: list, arf) -> tuple[list[str], list[list]]:
    """ARF by duration (rows) and return period (columns) for the durations
    the storms use: the HEC-HMS nested durations up to the longest storm and
    every storm duration."""
    if arf is None or arf.is_unity:
        return [], []
    from .storms import HMS_FREQ_DURATIONS

    longest = max(float(st.duration_min) for st in storms)
    durs = sorted(
        {float(d) for d in HMS_FREQ_DURATIONS if d <= longest + 1e-9}
        | {float(st.duration_min) for st in storms}
    )
    aris = sorted({float(st.ari) for st in storms})
    rows = [
        [duration_label(d)] + [round(arf.factor(d, t), 4) for t in aris] for d in durs
    ]
    return ["duration"] + [ari_label(t) for t in aris], rows


def burst_table(named_storms: list) -> tuple[list[str], list[list]]:
    """Burst check per hyetograph storm: [name, pattern, ratio % at each
    duration]. ``named_storms`` = [(run name, Storm)]."""
    durs = sorted(
        {b[0] for _, st in named_storms for b in getattr(st, "burst", []) or []}
    )
    if not durs:
        return [], []
    rows = []
    for name, st in named_storms:
        if not getattr(st, "burst", None):
            continue
        by = {b[0]: b[3] for b in st.burst}
        rows.append(
            [name, st.pattern] + [round(by[d], 1) if d in by else None for d in durs]
        )
    return ["run", "pattern"] + [duration_label(d) for d in durs], rows


def write_csv(
    path: str, r: ProjectResult, inputs: list, results: list | None = None
) -> str:
    m = method_labels(r)
    s = r.settings
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        def section(title):
            w.writerow([])
            w.writerow([f"## {title}"])

        w.writerow(["# Prepare HEC-HMS project - model set-up"])
        section("Project")
        for k, v in [
            ("HEC-HMS project", s.project_name),
            ("Project folder", r.folder),
            ("Basin model", s.basin_name),
            ("HEC-HMS version", s.hms_version),
            ("Coordinate system", r.crs_name),
            ("Loss method", m["loss"]),
            ("Curve number", m["cn"]),
            ("Transform", m["transform"]),
            ("Routing", m["routing"]),
            ("Time step (min)", r.dt_min),
            ("Simulation after the storm (h)", r.tail_h),
            ("Storm start", s.start.strftime("%d %b %Y %H:%M")),
            ("Areal reduction", arf_label(r.arf)),
            ("ARF area", getattr(r.arf, "area_note", "") or "-"),
            ("Sub-basins", len(r.network.of("Subbasin"))),
            ("Reaches", len(r.network.of("Reach"))),
            ("Junctions", len(r.network.of("Junction"))),
            ("Sinks", len(r.network.of("Sink"))),
            ("Simulation runs", len(r.runs)),
            ("Ensemble analyses", len(r.ensembles)),
        ]:
            w.writerow([k, _v(v)])
        section("Inputs")
        for k, v in inputs:
            w.writerow([k, _v(v)])
        for title, fields, rows in (
            ("Sub-basins", SUBBASIN_FIELDS, subbasin_rows(r)),
            ("Reaches", REACH_FIELDS, reach_rows(r)),
            ("Design storms and runs", STORM_FIELDS, storm_rows(r)),
        ):
            section(title)
            w.writerow([f[0] for f in fields])
            for row in rows:
                w.writerow([_v(row.get(f[0])) for f in fields])
        write_arf_sections(
            w, section, r.storms, r.arf, [(run.name, run.storm) for run in r.runs]
        )
        section("Junctions and sinks")
        w.writerow(["name", "kind", "downstream", "x", "y"])
        for e in r.network.elements:
            if e.kind in ("Junction", "Sink"):
                w.writerow([e.name, e.kind, e.downstream or "", _v(e.x), _v(e.y)])
        section("Control specifications")
        w.writerow(["name", "start", "end", "time_step_min"])
        for c in r.controls:
            w.writerow(
                [
                    c.name,
                    c.start.strftime("%d %b %Y %H:%M"),
                    c.end.strftime("%d %b %Y %H:%M"),
                    _v(c.dt_min),
                ]
            )
        if r.ensembles:
            section("Ensemble analyses (HEC-HMS)")
            w.writerow(ENSEMBLE_HEADER + ["member_runs"])
            members = {
                e.name: "; ".join(run.name for run in e.runs) for e in r.ensembles
            }
            for row in ensemble_rows(r, results):
                w.writerow([_v(v) for v in row] + [members[row[0]]])
        if results:
            section("Computed peak flows (HEC-HMS)")
            w.writerow([f[0] for f in RESULT_FIELDS])
            for x in results:
                w.writerow([_v(getattr(x, f[0])) for f in RESULT_FIELDS])
        section("Warnings")
        for msg in r.warnings or ["None"]:
            w.writerow([msg])
        section("Field definitions")
        w.writerow(["section", "field", "unit", "description"])
        for title, fields in (
            ("Sub-basins", SUBBASIN_FIELDS),
            ("Reaches", REACH_FIELDS),
            ("Design storms and runs", STORM_FIELDS),
            ("Computed peak flows", RESULT_FIELDS),
        ):
            for name, unit, desc in fields:
                w.writerow([title, name, unit, desc])
    return path


def write_arf_sections(w, section, storms, arf, named_storms) -> None:
    """'Areal reduction factors' and 'Burst check' CSV sections."""
    head, rows = arf_table(storms, arf)
    if rows:
        section("Areal reduction factors (ARF by duration and return period)")
        w.writerow([f"# {arf_label(arf)}"])
        w.writerow(head)
        for row in rows:
            w.writerow([_v(v) for v in row])
    head, rows = burst_table(named_storms)
    if rows:
        section("Burst check (largest burst in the storm / areal DDF depth, %)")
        w.writerow(
            [
                "# near 100 %: the pattern is as intense as the areal DDF; well "
                "below: smoother than the DDF for this catchment (typical of "
                "gridded rainfall such as ERA5 on a small catchment)"
            ]
        )
        w.writerow(head)
        for row in rows:
            w.writerow([_v(v) for v in row])


def peak_table(r: ProjectResult, results: list) -> tuple[list[str], list[list]]:
    """Header and rows: one row per outlet / junction, one column per run
    (peak m3/s)."""
    runs = [run.name for run in r.runs]
    keep = [e.name for e in r.network.elements if e.kind in ("Sink", "Junction")]
    peaks = {(x.run, x.element): x.peak_m3s for x in results}
    rows = []
    for el in keep:
        rows.append([el] + [peaks.get((rn, el)) for rn in runs])
    return ["Element"] + runs, rows


def ensemble_rows(r: ProjectResult, results: list | None = None) -> list[list]:
    """[ensemble, method, return period, members, outlet, peak m3/s, critical
    run]: one row per ensemble and outlet (sink) with the largest computed
    peak of its members, or one row per ensemble without results."""
    from .core import ENSEMBLE_METHOD_NAMES

    peaks = {(x.run, x.element): x.peak_m3s for x in results or []}
    sinks = [e.name for e in r.network.of("Sink")]
    rows = []
    for ens in r.ensembles:
        head = [
            ens.name,
            ENSEMBLE_METHOD_NAMES.get(ens.source, ens.source),
            ari_label(ens.ari),
            len(ens.runs),
        ]
        if not results:
            rows.append(head + ["", None, ""])
            continue
        for el in sinks:
            vals = [
                (peaks[(run.name, el)], run.name)
                for run in ens.runs
                if peaks.get((run.name, el)) is not None
            ]
            if vals:
                v, rn = max(vals)
                rows.append(head + [el, v, rn])
    return rows


ENSEMBLE_HEADER = [
    "ensemble",
    "method",
    "return_period",
    "members",
    "outlet",
    "max_peak_m3s",
    "critical_run",
]


def storm_label(st) -> str:
    return f"{ari_label(st.ari)} {duration_label(st.duration_min)}"
