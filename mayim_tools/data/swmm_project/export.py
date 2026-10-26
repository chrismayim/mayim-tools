"""
Sectioned CSV of the SWMM model set-up and, when the storms were computed,
the peak flows and the critical storm per method and return period. Zero
QGIS dependency; UTF-8 with BOM, no thousands separators.
"""

from __future__ import annotations

import csv

from ..hec_hms_project.core import ENSEMBLE_METHOD_NAMES, make_ensembles
from ..hec_hms_project.export import arf_label, burst_min, write_arf_sections
from ..hec_hms_project.storms import ari_label, duration_label
from .core import SwmmModel


def _v(x, nd=4):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{round(x, nd):g}" if abs(x) < 1e15 else str(x)
    return x


def outlets(m: SwmmModel) -> list[str]:
    return [n.name for n in m.nodes if n.kind == "outfall"]


def peak_rows(m: SwmmModel, results) -> list[list]:
    """[run, series, node, peak, time, runoff err %, routing err %]."""
    rows = []
    for res in results or []:
        for node, (q, t) in res.peaks.items():
            rows.append(
                [
                    res.run,
                    res.series,
                    node,
                    q,
                    t,
                    res.runoff_error_pct,
                    res.routing_error_pct,
                ]
            )
    return rows


def critical_rows(m: SwmmModel, results) -> list[list]:
    """[method, return period, members, outlet, max peak, critical run] for
    each storm method and return period (the HEC-HMS ensemble grouping) and
    outlet."""
    if not results:
        return []
    by_run = {res.run: res for res in results}
    groups = make_ensembles(m.hms.runs)
    single = [run for run in m.hms.runs if not any(run in e.runs for e in groups)]
    rows = []
    for e in groups:
        for out in outlets(m):
            vals = [
                (by_run[r.name].peaks[out][0], r.name)
                for r in e.runs
                if r.name in by_run and out in by_run[r.name].peaks
            ]
            if vals:
                q, rn = max(vals)
                rows.append(
                    [
                        ENSEMBLE_METHOD_NAMES.get(e.source, e.source),
                        ari_label(e.ari),
                        len(e.runs),
                        out,
                        q,
                        rn,
                    ]
                )
    for run in single:
        res = by_run.get(run.name)
        for out in outlets(m):
            if res and out in res.peaks:
                rows.append(
                    [
                        ENSEMBLE_METHOD_NAMES.get(run.storm.source, run.storm.source),
                        ari_label(run.storm.ari),
                        1,
                        out,
                        res.peaks[out][0],
                        run.name,
                    ]
                )
    return rows


def write_csv(path: str, m: SwmmModel, inputs: list, results=None) -> str:
    s = m.settings
    nodes = m.node_map()
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)

        def section(title):
            w.writerow([])
            w.writerow([f"## {title}"])

        w.writerow(["# Prepare SWMM project - model set-up"])
        section("Model")
        for k, v in [
            ("SWMM input file", m.files.get("SWMM input file", "")),
            ("Coordinate system", m.crs_name),
            ("Flow routing", s.routing),
            ("Infiltration", "CURVE_NUMBER"),
            ("Rainfall step (min)", m.step_min),
            ("Simulation start", m.start.strftime("%d %b %Y %H:%M")),
            ("Simulation end", m.end.strftime("%d %b %Y %H:%M")),
            ("Subcatchments", len(m.subcatchments)),
            ("Junctions", sum(1 for n in m.nodes if n.kind == "junction")),
            ("Outfalls", sum(1 for n in m.nodes if n.kind == "outfall")),
            ("Conduits", len(m.conduits)),
            ("Design storms (time series)", len(m.series)),
            ("Rain gage time series in the file", m.series[0].name),
            ("Areal reduction", arf_label(m.hms.arf)),
            ("ARF area", getattr(m.hms.arf, "area_note", "") or "-"),
        ]:
            w.writerow([k, _v(v)])
        section("Inputs")
        for k, v in inputs:
            w.writerow([k, _v(v)])
        section("Subcatchments")
        w.writerow(
            [
                "name",
                "outlet",
                "area_ha",
                "imperv_pct",
                "width_m",
                "width_source",
                "slope_pc",
                "curve_number",
                "n_imperv",
                "n_perv",
                "dstore_imperv_mm",
                "dstore_perv_mm",
            ]
        )
        for x in m.subcatchments:
            w.writerow(
                [
                    x.name,
                    x.outlet,
                    _v(x.area_ha),
                    _v(x.imperv_pct),
                    _v(x.width_m, 2),
                    x.width_note,
                    _v(x.slope_pc),
                    _v(x.cn),
                    s.n_imperv,
                    s.n_perv,
                    s.ds_imperv_mm,
                    s.ds_perv_mm,
                ]
            )
        section("Junctions and outfalls")
        w.writerow(
            [
                "name",
                "type",
                "invert_m",
                "dem_invert_m",
                "max_depth_m",
                "x",
                "y",
                "flags",
            ]
        )
        for n in m.nodes:
            w.writerow(
                [
                    n.name,
                    n.kind,
                    _v(n.invert, 3),
                    _v(n.dem_invert, 3),
                    _v(n.max_depth, 3),
                    _v(n.x, 3),
                    _v(n.y, 3),
                    ";".join(n.flags),
                ]
            )
        section("Conduits")
        w.writerow(
            [
                "name",
                "from",
                "to",
                "length_m",
                "slope_m_m",
                "manning_n",
                "section",
                "depth_m",
                "bottom_width_m",
                "side_slope",
            ]
        )
        for c in m.conduits:
            w.writerow(
                [
                    c.name,
                    c.from_node,
                    c.to_node,
                    _v(c.length_m, 2),
                    _v(c.slope(nodes), 5),
                    c.n,
                    c.shape.lower(),
                    _v(c.depth_m, 3),
                    _v(c.bottom_width_m if c.shape != "IRREGULAR" else None),
                    _v(c.side_slope if c.shape != "IRREGULAR" else None),
                ]
            )
        section("Design storms (rainfall time series)")
        w.writerow(
            [
                "time_series",
                "run",
                "method",
                "return_period",
                "duration",
                "peak_pct",
                "depth_mm",
                "pattern",
                "arf_storm",
                "arf_peak",
                "burst_min_pct",
                "burst_min_at",
            ]
        )
        for ser in m.series:
            st = ser.storm
            w.writerow(
                [
                    ser.name,
                    ser.run_name,
                    ENSEMBLE_METHOD_NAMES.get(st.source, st.source),
                    ari_label(st.ari),
                    duration_label(st.duration_min),
                    _v(st.peak_pct),
                    _v(ser.cumulative_mm[-1], 2),
                    st.pattern,
                    _v(st.arf_total if st.arf_values else None),
                    _v(st.arf_peak if len(st.arf_values) > 1 else None),
                    _v(burst_min(st)[0], 3),
                    burst_min(st)[1],
                ]
            )
        write_arf_sections(
            w,
            section,
            [ser.storm for ser in m.series],
            m.hms.arf,
            [(ser.run_name, ser.storm) for ser in m.series],
        )
        if results:
            section("Critical storm per method and return period (SWMM)")
            w.writerow(
                [
                    "method",
                    "return_period",
                    "storms",
                    "outfall",
                    "max_peak_m3s",
                    "critical_storm",
                ]
            )
            for row in critical_rows(m, results):
                w.writerow([_v(v) for v in row])
            section("Peak node inflows (SWMM)")
            w.writerow(
                [
                    "run",
                    "time_series",
                    "node",
                    "peak_m3s",
                    "time_of_peak",
                    "runoff_continuity_pct",
                    "routing_continuity_pct",
                ]
            )
            for row in peak_rows(m, results):
                w.writerow([_v(v) for v in row])
        section("Warnings")
        for msg in m.warnings or ["None"]:
            w.writerow([msg])
    return path
