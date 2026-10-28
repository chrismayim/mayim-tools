"""Fields and the CSV output of the 'Catchments: 2. Time of Concentration'
tool (no QGIS)."""

from __future__ import annotations

import csv
import math

from . import tables as tb

# Parameter fields the user fills (name, type, definition). Types: "D"
# double, "S" string.
PARAM_FIELDS = [
    ("ov_len_m", "D", "overland (sheet) flow length from the divide, m"),
    ("ov_r", "D", "Kerby roughness r of the overland flow (DM Table 3.10)"),
    ("sc_type", "S", "shallow concentrated flow surface (NEH 630 Table 15-3)"),
    ("ch_start_m", "D", "channel start on the flow path, m from the outlet"),
    ("tc_method", "S", "Tc method for tc_sel_mn (empty: the run default)"),
    ("tc_user_mn", "D", "user Tc when tc_method is user, min"),
]
# Result fields the tool writes.
RESULT_FIELDS = [
    ("tc_kirp_mn", "D", "Tc Kirpich (L, s_avg), min"),
    ("tc_usbr_mn", "D", "Tc USBR / SANRAL DM Eq 3.12 (L, s_1085), min"),
    ("tc_bw_mn", "D", "Tc Bransby-Williams (L, A, s_ea), min"),
    ("l_ov_m", "D", "overland flow length used, m"),
    ("s_ov", "D", "overland flow slope, m/m"),
    ("tc_ov_mn", "D", "overland flow time, Kerby DM Eq 3.11, min"),
    ("l_sc_m", "D", "shallow concentrated flow length, m"),
    ("s_sc", "D", "shallow concentrated flow slope, m/m"),
    ("v_sc_ms", "D", "shallow concentrated flow velocity V = k S^0.5, m/s"),
    ("tc_sc_mn", "D", "shallow concentrated flow time, min"),
    ("l_ch_m", "D", "channel length, m"),
    ("s_ch", "D", "channel 10-85 slope, m/m"),
    ("tc_ch_mn", "D", "channel flow time, DM Eq 3.12, min"),
    ("tc_comb_mn", "D", "combined Tc = overland + shallow + channel, min"),
    ("tc_sel_mn", "D", "selected Tc (tc_method), min"),
    ("tc_sel_by", "S", "method of the selected Tc"),
    ("sug_r", "D", "suggested Kerby r from the land cover (overland part)"),
    ("sug_sc", "S", "suggested shallow flow surface from the land cover"),
    ("lc_ov", "I", "majority land-cover class on the overland part"),
    ("lc_ov_pc", "D", "share of that class on the overland part, %"),
    ("lc_sc", "I", "majority land-cover class on the shallow flow part"),
    ("lc_sc_pc", "D", "share of that class on the shallow flow part, %"),
    ("tc_status", "S", "missing parameters and flags"),
]
ROUND = {"s_ov": 5, "s_sc": 5, "s_ch": 5, "v_sc_ms": 3, "sug_r": 2}


def rounded(name, value):
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, ROUND.get(name, 2))
    return value


def status_text(c) -> str:
    items = []
    if c.missing:
        items.append("missing: " + ", ".join(c.missing))
    items += [f for f in c.flags if f not in items]
    return "; ".join(items)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return str(v)


def write_csv(path, items, run_rows, classification=None) -> None:
    """Sectioned CSV (UTF-8 with BOM): column key, run, then one row per
    catchment with the parameters and results."""
    params = [f[0] for f in PARAM_FIELDS]
    results = [f[0] for f in RESULT_FIELDS]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# Column key"])
        w.writerow(["# catchment_id", "catchment ID"])
        for name, _, text in PARAM_FIELDS + RESULT_FIELDS:
            w.writerow([f"# {name}", text])
        w.writerow([])
        w.writerow(["[run]"])
        for k, v in run_rows:
            w.writerow([k, _fmt(v)])
        w.writerow([])
        w.writerow(["[catchments]"])
        w.writerow(["catchment_id", "area_km2", "lfp_km", "s_avg", "s_1085",
                    "s_ea", "ch_head_m"] + params + results)  # fmt: skip
        for c in items:
            p = c.params
            w.writerow(
                [c.catchment_id, _fmt(c.area_km2), _fmt(c.lfp_km), _fmt(c.s_avg),
                 _fmt(c.s_1085), _fmt(c.s_ea), _fmt(c.ch_head_m)]
                + [_fmt(getattr(p, n)) for n in params]
                + [_fmt(rounded(n, c.values.get(n))) if n != "tc_status"
                   else status_text(c) for n in results]
            )  # fmt: skip
        w.writerow([])
        w.writerow(["[Kerby r options]"])
        for r, text in tb.KERBY_R.items():
            w.writerow([r, text])
        w.writerow([])
        w.writerow(["[shallow concentrated flow surfaces]", "description", "k (m/s)"])
        for code, (text, k) in tb.SC_TYPES.items():
            w.writerow([code, text, f"{k:.3f}"])
        if classification:
            w.writerow([])
            w.writerow(["[land-cover suggestions]", "class", "r", "sc_type"])
            for code, (r, sc) in sorted(
                tb.LAND_COVER_CLASSIFICATIONS[classification][1].items()
            ):
                w.writerow([classification, code, r, sc])
