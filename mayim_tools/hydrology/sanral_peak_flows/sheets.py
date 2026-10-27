"""Word (.docx) calculation sheets for the SANRAL Deterministic Peak Flows
tool: one sheet per catchment and method, laid out after the standard flood
calculation forms of the SANRAL Drainage Manual (2013) Application Guide,
Appendix 3C (forms R2, R3, S and E). CAPA (DWS 2025) has no Drainage Manual
form; its sheet follows the same layout.

Each sheet has the form's header (catchment, project, calculated by, date),
the physical characteristics, the form's component tables (rural C1
breakdown, time of concentration, TR102 n-day rainfall) and the per-return-
period rows of the form, ending in the peak flow. Values the tool does not
have are shown as '-'.

No QGIS dependency; python-docx is imported lazily. Numbers use a decimal
point and no thousands separators."""

from __future__ import annotations

import datetime as _dt
import math

from mayim_tools._common.docx_report import Doc
from mayim_tools.land_cover.rational_c import tables as rc

from . import tables as tb
from .core import RunResult
from .export import REFERENCES
from .methods import METHOD_ORDER, sdf_tc_hours
from .report import num, q_text

TITLE = "Peak Flow Calculation Sheets (SANRAL)"

# sheet id -> (form code, form title); the empirical sheet holds three methods
SHEETS = {
    "rational_alt2": ("R2", "Rational method (Alternative 2)"),
    "rational_alt3": ("R3", "Rational method (Alternative 3)"),
    "sdf": ("S", "Standard Design Flood method"),
    "empirical": ("E", "Empirical methods"),
    "capa": ("CAPA", "Catchment Parameter method (CAPA)"),
}
EMPIRICAL = ("midgley_pitman", "rmf", "rmf_ratio")

# Rural C1 classes and factors of DM Table 3.7 (AG Table 3C.1) per MAP band,
# shared with the Rational C tool.
SLOPE_CLASSES = rc.SLOPE_CLASSES
PERM_CLASSES = rc.PERMEABILITY_CLASSES
VEG_CLASSES = rc.VEGETATION_CLASSES
CS, CP, CV = rc.CS, rc.CP, rc.CV
MAP_BANDS = tuple(f"{b} mm" for b in rc.MAP_BANDS)

HERSHFIELD = "Pt,T = 1.13 (0.41 + 0.64 ln T) (-0.11 + 0.27 ln t) (0.79 M^0.69 R^0.20)"
SELECTION_CRITERIA = (
    ("TC < 6 hours", f"Modified Hershfield equation: {HERSHFIELD}"),
    (
        "6 hours <= TC < 24 hours",
        "Linear interpolation between the modified Hershfield point rainfall "
        "and the 1-day point rainfall from TR102",
    ),
    ("TC >= 24 hours", "Linear interpolation between n-day point rainfall from TR102"),
)

LEGENDS = {
    "rational_alt2": (
        ("C1 classes and factors", "DM Table 3.7 (AG Table 3C.1)"),
        ("Dolomite factor", "AG Table 3C.4"),
        ("Ft", "DM Table 3.8 (AG Table 3C.5); 1.0 at 1:200"),
        ("Tc", "DM Eq 3.11 (overland flow) or Eq 3.12 (defined watercourse)"),
        ("R (days of thunder)", "DM Figure 3.9"),
        ("M and the TR102 station", "TR102 (Adamson 1981); DM Figure 3.10"),
        ("Point rainfall", "AG Table 3C.6 (modified Hershfield, DM Eq 3.9)"),
        ("ARF", "DM Eq 3.10, or Pietersen et al (2024) for D >= 24 h"),
    ),
    "rational_alt3": (
        ("C1 classes and factors", "DM Table 3.7 (AG Table 3C.1)"),
        ("Dolomite factor", "AG Table 3C.4"),
        ("Ft", "DM Table 3.8 (AG Table 3C.5); 1.0 at 1:200"),
        ("Tc", "DM Eq 3.11 (overland flow) or Eq 3.12 (defined watercourse)"),
        ("Point rainfall", "Design rainfall DDF table (Smithers and Schulze 2003)"),
        ("ARF", "DM Eq 3.10, or Pietersen et al (2024) for D >= 24 h"),
    ),
    "sdf": (
        ("SDF basin", "DM Figure 3.30"),
        ("C2, C100, M and R", "AG Table 3B.1"),
        ("Tc", "DM Eq 3.17"),
        ("Point rainfall", "AG Table 3C.12 (modified Hershfield, DM Eq 3.18)"),
        ("ARF", "DM Eq 3.19"),
        ("Y_T", "DM Section 3.5.3"),
    ),
    "empirical": (
        ("Veld type", "DM Figure 3.20"),
        ("K_T", "DM Table 3.17 (AG Table 3C.13)"),
        ("Kovacs region", "DM Figure 3.37"),
        ("Q_RMF", "DM Table 3.18 (AG Table 3C.14); TR137"),
        ("Q_T/Q_RMF", "AG Table 3D.1"),
    ),
    "capa": (
        ("M", "DWS (2025) Section 5.4"),
        ("Q50%", "DWS (2025) Figure 5-2 (digitised)"),
        ("K_P", "DWS (2025) Table 5-2, linear in MAP"),
    ),
}  # fmt: skip


def _per_t_widths(n: int) -> list[float]:
    first = 6.5
    return [first] + [(16.5 - first) / max(n, 1)] * n


def _by_t(results, method):
    return {r.return_period: r for r in results if r.method == method}


def _row(label, by_t, rps, key=None, nd=2, fn=None):
    cells = [label]
    for T in rps:
        r = by_t.get(T)
        if r is None:
            cells.append("-")
        elif fn is not None:
            cells.append(fn(r))
        else:
            cells.append(num(r.derivation.get(key), nd))
    return cells


def _band(map_mm):
    if map_mm is None or not math.isfinite(map_mm) or map_mm <= 0:
        return None
    if map_mm < 600.0:
        return 0
    return 1 if map_mm <= 900.0 else 2


def _kv_table(doc, caption, pairs):
    """Physical characteristics as two item/value column pairs."""
    pairs = list(pairs)
    half = (len(pairs) + 1) // 2
    left, right = pairs[:half], pairs[half:]
    rows = []
    for i in range(half):
        a = left[i]
        b = right[i] if i < len(right) else ("", "")
        rows.append([a[0], a[1], b[0], b[1]])
    doc.caption("Table", caption)
    doc.table(
        ["Item", "Value", "Item", "Value"], rows, widths_cm=[5.0, 3.25, 5.0, 3.25]
    )


def _header(doc, cid, form, title, context):
    doc.caption("Table", f"Form {form}: {title}, catchment {cid}")
    doc.table(
        ["Description of catchment", "River detail", "Calculated by", "Date"],
        [
            [
                f"Catchment {cid}",
                context.get("project_name") or "-",
                context.get("calculated_by") or "Mayim Consulting (Mayim Tools)",
                _dt.date.today().isoformat(),
            ]
        ],
        widths_cm=[4.5, 4.5, 4.5, 3.0],
    )


def _legend(doc, sid, cid):
    doc.caption(
        "Table", f"Legend: references for form {SHEETS[sid][0]}, catchment {cid}"
    )
    doc.table(
        ["Item", "Reference"], [list(r) for r in LEGENDS[sid]], widths_cm=[5.5, 11.0]
    )


def _notes(doc, results, extra=()):
    items = list(extra)
    for r in results:
        for f in r.flags:
            if f not in items:
                items.append(f)
    if items:
        doc.label("Notes")
        for n in items:
            doc.bullet(n)


def _tr102_table(doc, station, cid, form):
    rps = tb.RETURN_PERIODS
    rows = [
        [f"{d} day" if d == 1 else f"{d} days"]
        + [num(station.depths.get(d, {}).get(T), 1) for T in rps]
        for d in sorted(station.depths)
    ]
    doc.caption(
        "Table", f"TR102 n-day point rainfall (mm), form {form}, catchment {cid}"
    )
    doc.table(
        ["Duration (days)"] + [str(T) for T in rps],
        rows,
        widths_cm=_per_t_widths(len(rps)),
    )


def _rural_table(doc, coeffs, map_mm, cid, form):
    """The form's rural C1 table (% of the rural area, Table 3.7 factor and
    the contribution) when the C CSV carries the breakdown."""
    d = coeffs.detail or {}
    band = _band(d.get("map_mm", map_mm))
    has_pct = any(f"cs{i}_pct" in d for i in range(1, 5))
    if not has_pct and not any(k in d for k in ("cs", "cp", "cv")):
        doc.para(
            f"C1 = {num(coeffs.c1, 3)} was supplied in the C CSV without the Cs, Cp "
            "and Cv breakdown, so the rural classes are not shown. The C CSV of the "
            "Rational C tool carries the breakdown."
        )
        return
    rows, groups = [], []
    for name, pre, classes, table in (
        ("Surface slope (Cs)", "cs", SLOPE_CLASSES, CS),
        ("Permeability (Cp)", "cp", PERM_CLASSES, CP),
        ("Vegetation (Cv)", "cv", VEG_CLASSES, CV),
    ):
        groups.append(len(rows))
        rows.append([name, "", "", ""])
        total_pct = 0.0
        for i, label in enumerate(classes):
            pct = d.get(f"{pre}{i + 1}_pct")
            factor = None if band is None else table[i][band]
            part = None if pct is None or factor is None else pct / 100.0 * factor
            total_pct += pct or 0.0
            rows.append([label, num(pct, 1), num(factor, 2), num(part, 4)])
        rows.append(
            ["Total", num(total_pct, 1) if has_pct else "-", "-", num(d.get(pre), 4)]
        )
    rows.append(["C1 = Cs + Cp + Cv", "", "", num(d.get("c1_base"), 4)])
    band_text = f", MAP band {MAP_BANDS[band]}" if band is not None else ""
    doc.caption(
        "Table",
        f"Rural run-off coefficient C1{band_text}, form {form}, catchment {cid}",
    )
    doc.table(
        ["Rural", "% of rural area", "Factor", "Component"],
        rows,
        widths_cm=[7.0, 3.2, 3.0, 3.3],
        group_rows=groups,
    )


# ------------------------------------------------------------------ sheets


def _rational_sheet(doc, sid, item, run, rps, context):
    c = item.catchment
    cid = c.catchment_id
    form, title = SHEETS[sid]
    results = [r for r in item.results if r.method == sid]
    by_t = _by_t(results, sid)
    k = item.coeffs
    d = (k.detail if k else None) or {}
    first = results[0].derivation
    station = item.stations.get(sid)
    _header(doc, cid, form, title, context)

    pairs = [
        ("Size of catchment, Ae (km²)", num(c.effective_area, 3)),
        ("Longest watercourse, L (km)", num(c.lfp_km, 3)),
        ("Average slope, S1085 (m/m)", num(c.s1085, 5)),
        ("Dolomite area, D (%)", num(d.get("d_pct"), 1)),
        ("Dolomite factor, sum(Dfactor x Cs%)", num(d.get("d_factor"), 3)),
        ("Mean annual precipitation, MAP (mm)", num(d.get("map_mm", c.map_mm), 0)),
    ]
    if sid == "rational_alt2" and station is not None:
        pairs += [
            ("2-year 1-day rainfall, M (mm)", num(station.m_mm, 1)),
            ("Days of thunder per year, R", num(station.r_days, 0)),
            ("Weather Service station", station.label),
        ]
    else:
        site = (run.ddf_info or {}).get("site")
        pairs.append(("Design rainfall (DDF) site", str(site) if site else "DDF table"))
    if k is not None:
        pairs += [
            ("Rural, α (-)", num(k.alpha, 3)),
            ("Urban, β (-)", num(k.beta, 3)),
            ("Lakes, γ (-)", num(k.gamma, 3)),
            ("Urban run-off coefficient, C2 (-)", num(k.c2, 3) if k.beta else "-"),
            ("Lakes run-off coefficient, C3 (-)", num(k.c3, 3) if k.gamma else "-"),
            ("Ft class (Table 3.8)", k.ft_class.replace("_", " and ")),
        ]
    _kv_table(doc, f"Physical characteristics, form {form}, catchment {cid}", pairs)

    if k is not None:
        _rural_table(doc, k, c.map_mm, cid, form)

    # time of concentration
    tc_rows = [
        [
            "Tc supplied on the catchment layer",
            f"{num(c.tc_min, 1)} min",
            f"{num(first.get('Tc_h'), 3)} h",
        ]
    ]
    if c.lfp_km and c.s1085 and c.s1085 > 0:
        tc_rows.append(
            [
                "Defined watercourse, Eq 3.12: (0.87 L² / 1000 Sav)^0.385",
                f"{num(60 * sdf_tc_hours(c.lfp_km, c.s1085), 1)} min",
                f"{num(sdf_tc_hours(c.lfp_km, c.s1085), 3)} h",
            ]
        )
    tc_rows.append(["Tc used", f"{num(60 * first.get('Tc_used_h', 0), 1)} min",
                    f"{num(first.get('Tc_used_h'), 3)} h"])  # fmt: skip
    doc.caption("Table", f"Time of concentration, form {form}, catchment {cid}")
    doc.table(["Time of concentration (TC)", "Minutes", "Hours"], tc_rows,
              widths_cm=[10.5, 3.0, 3.0])  # fmt: skip

    def c1_base(_r):
        if k is None or not k.alpha:
            return "-"
        return num(d.get("c1_base"), 3)

    def c1d(_r):
        return num(k.c1, 3) if k is not None and k.alpha else "-"

    def c1t(r):
        if k is None or not k.alpha:
            return "-"
        return num(k.c1 * r.derivation["Ft"], 3)

    rows = [
        ["Run-off coefficient"],
        _row("Run-off coefficient, C1 (= Cs + Cp + Cv)", by_t, rps, fn=c1_base),
        _row("Adjusted for dolomitic areas, C1D", by_t, rps, fn=c1d),
        _row("Adjustment factor for initial saturation, Ft", by_t, rps, "Ft", 2),
        _row("Adjusted run-off coefficient, C1T (= C1D x Ft)", by_t, rps, fn=c1t),
        _row("Combined run-off coefficient, CT (= αC1T + βC2 + γC3)", by_t, rps, "C_T", 3),
        ["Rainfall"],
        _row("Point precipitation (mm), PT", by_t, rps, "P_mm", 1),
        _row("Point intensity (mm/h), PiT (= PT / TC)", by_t, rps, "Pi_mmh", 2),
        _row("Area reduction factor (%), ARFT", by_t, rps, "ARF_pct", 1),
        _row("Average intensity (mm/h), IT (= PiT x ARFT)", by_t, rps, "I_mmh", 2),
        ["Peak flow"],
        _row("Peak flow (m³/s), QT (= CT IT A / 3.6)", by_t, rps, fn=lambda r: q_text(r.q_m3s)),
    ]  # fmt: skip
    _per_t(
        doc,
        rows,
        rps,
        f"Run-off coefficient, rainfall and peak flow, form {form}, catchment {cid}",
    )

    extra = []
    sources = {}
    for T in rps:
        r = by_t.get(T)
        if r is not None:
            how = str(r.derivation.get("rain_source", "")).split("; ")[-1]
            sources.setdefault(how, []).append(T)
    if sid == "rational_alt2" and sources:
        extra += [
            f"Point rainfall, {', '.join(f'1:{T}' for T in ts)} years: {how}"
            for how, ts in sources.items()
        ]
    arf = first.get("ARF_method")
    if arf:
        extra.append(f"ARF: {arf}")
    if d and k is not None and k.alpha and d.get("c1_base") is not None:
        extra.append("C1D = C1 (1 - D%) + C1 D% sum(Dfactor x Cs%)")
    _notes(doc, results, extra)

    if sid == "rational_alt2" and station is not None:
        _tr102_table(doc, station, cid, form)
        doc.caption(
            "Table",
            f"Point rainfall selection criteria (AG Table 3C.6), catchment {cid}",
        )
        doc.table(["Selection criteria", "Calculation method"],
                  [list(r) for r in SELECTION_CRITERIA], widths_cm=[4.0, 12.5])  # fmt: skip
    _legend(doc, sid, cid)


def _per_t(doc, rows, rps, caption):
    header = ["Return period (years), T"] + [str(T) for T in rps]
    groups = [i for i, r in enumerate(rows) if len(r) == 1]
    rows = [r + [""] * (len(header) - len(r)) if len(r) == 1 else r for r in rows]
    doc.caption("Table", caption)
    doc.table(header, rows, widths_cm=_per_t_widths(len(rps)), group_rows=groups)


def _sdf_sheet(doc, item, rps, context):
    c = item.catchment
    cid = c.catchment_id
    form, title = SHEETS["sdf"]
    results = [r for r in item.results if r.method == "sdf"]
    by_t = _by_t(results, "sdf")
    first = results[0].derivation
    station = item.stations.get("sdf")
    _header(doc, cid, form, title, context)
    tc = first.get("Tc_h")
    pairs = [
        ("Size of catchment, Ae (km²)", num(c.effective_area, 3)),
        ("Longest watercourse, L (km)", num(c.lfp_km, 3)),
        ("Average slope, S1085 (m/m)", num(c.s1085, 5)),
        ("SDF basin", num(first.get("basin"), 0)),
        ("2-year 1-day rainfall, M (mm)", num(first.get("M_mm"), 1)),
        ("TC = (0.87 L² / 1000 Sav)^0.385 (h)", num(tc, 3)),
        ("Time of concentration, t = 60 TC (min)", num(None if tc is None else 60 * tc, 1)),
        ("Days of thunder per year, R", num(first.get("R_days"), 0)),
        ("Weather Service station", station.label if station else "-"),
        ("Calibration factor C2 (2-year) (%)", num(100 * first["C2"], 0)),
        ("Calibration factor C100 (100-year) (%)", num(100 * first["C100"], 0)),
    ]  # fmt: skip
    _kv_table(doc, f"Physical characteristics, form {form}, catchment {cid}", pairs)
    if station is not None:
        _tr102_table(doc, station, cid, form)
    rows = [
        ["Rainfall"],
        _row("Point precipitation depth (mm), Pt,T", by_t, rps, "P_mm", 1),
        _row("Area reduction factor (%), ARF = (90000 - 12800 ln A + 9830 ln t)^0.4",
             by_t, rps, "ARF_pct", 1),
        _row("Average intensity (mm/h), IT (= Pt,T x ARF / TC)", by_t, rps, "I_mmh", 2),
        ["Run-off coefficients"],
        _row("Return period factor, YT", by_t, rps, "Y_T", 2),
        _row("Run-off coefficient, CT = C2/100 + YT/2.33 (C100/100 - C2/100)",
             by_t, rps, "C_T", 3),
        ["Peak flow"],
        _row("Peak flow (m³/s), QT (= CT IT A / 3.6)", by_t, rps, fn=lambda r: q_text(r.q_m3s)),
    ]  # fmt: skip
    _per_t(
        doc,
        rows,
        rps,
        f"Rainfall, run-off coefficient and peak flow, form {form}, catchment {cid}",
    )
    sources = {}
    for T in rps:
        r = by_t.get(T)
        if r is not None:
            how = str(r.derivation.get("rain_source", "")).split("; ")[-1]
            sources.setdefault(how, []).append(T)
    extra = [
        f"Point rainfall, {', '.join(f'1:{T}' for T in ts)} years: {how}"
        for how, ts in sources.items()
    ]
    _notes(doc, results, extra)
    doc.caption(
        "Table", f"Point rainfall selection criteria (AG Table 3C.12), catchment {cid}"
    )
    doc.table(["Criteria", "Calculation method"], [list(r) for r in SELECTION_CRITERIA],
              widths_cm=[4.0, 12.5])  # fmt: skip
    _legend(doc, "sdf", cid)


def _empirical_sheet(doc, item, rps, run, context):
    c = item.catchment
    cid = c.catchment_id
    form, title = SHEETS["empirical"]
    results = [r for r in item.results if r.method in EMPIRICAL]
    _header(doc, cid, form, title, context)
    mp = _by_t(results, "midgley_pitman")
    rmf_r = next((r for r in results if r.method == "rmf"), None)
    ratios = _by_t(results, "rmf_ratio")
    c_param = next(iter(mp.values())).derivation.get("C_param") if mp else None
    if c_param is None and c.lfp_km and c.lca_km and c.s1085:
        c_param = c.effective_area * math.sqrt(c.s1085) / (c.lfp_km * c.lca_km)
    zone = c.veld_zone
    pairs = [
        ("Size of catchment, Ae (km²)", num(c.effective_area, 3)),
        ("Longest watercourse, L (km)", num(c.lfp_km, 3)),
        ("Distance to centroid, Lc (km)", num(c.lca_km, 3)),
        ("Average slope, S1085 (m/m)", num(c.s1085, 5)),
        ("Mean annual precipitation, MAP (mm)", num(c.map_mm, 0)),
        ("Veld type",
         f"{zone} {tb.VELD_ZONES[zone][0]}" if zone in tb.VELD_ZONES else (zone or "-")),
        ("Catchment parameter, C = A S^0.5 / (L Lc)", num(c_param, 5)),
        ("Kovacs region", c.kovacs_region or "-"),
    ]  # fmt: skip
    if run.settings.k_custom is not None:
        pairs.append(("Custom K (Francou-Rodier)", num(run.settings.k_custom, 2)))
    _kv_table(doc, f"Physical characteristics, form {form}, catchment {cid}", pairs)

    if mp:
        ts = [T for T in tb.MP_RETURN_PERIODS if T in mp]
        rows = [
            _row("Constant, KT", mp, ts, "K_T", 2),
            _row("Peak flow (m³/s), QT = 0.0377 KT P A^0.6 C^0.2", mp, ts,
                 fn=lambda r: q_text(r.q_m3s)),
        ]  # fmt: skip
        doc.label("Midgley and Pitman method")
        _per_t(
            doc, rows, ts, f"Midgley and Pitman method, form {form}, catchment {cid}"
        )
    if rmf_r is not None:
        dv = rmf_r.derivation
        doc.label("Regional maximum flood (Kovacs)")
        doc.caption("Table", f"Regional maximum flood, form {form}, catchment {cid}")
        doc.table(
            ["Item", "Value"],
            [
                ["Kovacs region", str(dv.get("region") or "-")],
                ["RMF K value", num(dv.get("K"), 1)],
                ["Equation", str(dv.get("equation") or "-")],
                ["Regional maximum flood, QRMF (m³/s)", q_text(rmf_r.q_m3s)],
            ],
            widths_cm=[8.0, 8.5],
        )
    if ratios:
        ts = [T for T in tb.RMF_RATIO_RETURN_PERIODS if T in ratios]
        rows = [
            _row("Ratio, QT/QRMF", ratios, ts, "ratio", 3),
            _row("Peak flow (m³/s), QT based on the ratios", ratios, ts,
                 fn=lambda r: q_text(r.q_m3s)),
        ]  # fmt: skip
        doc.label("Peak flows from the QT/QRMF ratios")
        _per_t(doc, rows, ts, f"QT/QRMF ratios, form {form}, catchment {cid}")
    _notes(doc, results)
    _legend(doc, "empirical", cid)


def _capa_sheet(doc, item, rps, context):
    c = item.catchment
    cid = c.catchment_id
    form, title = SHEETS["capa"]
    results = [r for r in item.results if r.method == "capa"]
    by_t = _by_t(results, "capa")
    first = results[0].derivation
    _header(doc, cid, form, title, context)
    pairs = [
        ("Size of catchment, Ae (km²)", num(c.effective_area, 3)),
        ("Longest watercourse, L (km)", num(c.lfp_km, 3)),
        ("Mean catchment slope, SA (m/m)", num(c.s_mean, 4)),
        ("Mean annual precipitation, MAP (mm)", num(c.map_mm, 0)),
        ("Catchment index, M = MAP (100 SA Ae^0.5 / L)^0.5", num(first.get("M"), 0)),
        ("Mean annual flood, Q50% (m³/s)", q_text(first.get("Q50_m3s"))),
    ]
    _kv_table(doc, f"Physical characteristics, CAPA, catchment {cid}", pairs)
    ts = [T for T in rps if T in by_t]
    if ts:
        rows = [
            _row("Frequency factor, KP", by_t, ts, "K_P", 2),
            _row("Peak flow (m³/s), QP = KP Q50%", by_t, ts, fn=lambda r: q_text(r.q_m3s)),
        ]  # fmt: skip
        _per_t(doc, rows, ts, f"Peak flows, CAPA, catchment {cid}")
    _notes(doc, results)
    _legend(doc, "capa", cid)


def sheets_for(item, methods) -> list[str]:
    """Sheet ids for a catchment, in method order, for methods with results."""
    have = {r.method for r in item.results}
    out = []
    for m in METHOD_ORDER:
        if m not in methods or m not in have:
            continue
        sid = "empirical" if m in EMPIRICAL else m
        if sid not in out:
            out.append(sid)
    return out


def write_sheets(path, run: RunResult, run_info: dict, context=None) -> list[str]:
    """Writes the calculation sheets and returns notes. ``context`` gives
    project_number, project_name and software."""
    context = context or {}
    notes: list[str] = []
    rps = tuple(run.settings.return_periods)
    doc = Doc(house_style=True)
    doc.title_page(
        TITLE,
        "Calculation sheets for the design peak flows of each catchment, one per "
        "method, laid out after the standard flood calculation forms of the SANRAL "
        "Drainage Manual (2013) Application Guide, Appendix 3C. They accompany the "
        "Peak Flow Annexure, which compares the methods.",
        [
            ("Project number", context.get("project_number") or "-"),
            ("Project / site", context.get("project_name") or "-"),
            ("Prepared by", "Mayim Consulting"),
            (
                "Tool and version",
                f"{run_info.get('tool', 'SANRAL Deterministic Peak Flows')}, "
                f"version {run_info.get('version', '-')}",
            ),
            ("Software", context.get("software") or "-"),
            ("Report date", _dt.date.today().isoformat()),
            ("Document status", "Draft"),
        ],
        subtitle=context.get("project_name") or None,
    )

    doc.heading("Summary", 1)
    plan = [(item, sheets_for(item, run.settings.methods)) for item in run.items]
    n = sum(len(s) for _, s in plan)
    doc.para(
        f"{n} calculation sheet(s) for {len(run.items)} catchment(s). Each sheet "
        "follows the rows of its Drainage Manual form: the physical "
        "characteristics, the component tables and the values per return period "
        "down to the peak flow. Values the tool does not have are shown as '-'."
    )
    doc.caption("Table", "Calculation sheets")
    doc.table(
        ["Catchment", "Form", "Method"],
        [
            [item.catchment.catchment_id, SHEETS[s][0], SHEETS[s][1]]
            for item, sids in plan
            for s in sids
        ]
        or [["-", "-", "No method gave a result"]],
        widths_cm=[4.0, 2.5, 10.0],
    )

    doc.heading("Calculation sheets", 1)
    for item, sids in plan:
        cid = item.catchment.catchment_id
        for sid in sids:
            doc.doc.add_page_break()
            doc.heading(f"Catchment {cid}: form {SHEETS[sid][0]}, {SHEETS[sid][1]}", 2)
            try:
                if sid in ("rational_alt2", "rational_alt3"):
                    _rational_sheet(doc, sid, item, run, rps, context)
                elif sid == "sdf":
                    _sdf_sheet(doc, item, rps, context)
                elif sid == "empirical":
                    _empirical_sheet(doc, item, rps, run, context)
                else:
                    _capa_sheet(doc, item, rps, context)
            except Exception as e:  # one sheet must not stop the others
                msg = (
                    f"Catchment {cid}, form {SHEETS[sid][0]}: sheet not complete ({e})"
                )
                doc.para(msg)
                notes.append(msg)
        skipped = [n for n in item.notes if "not calculated" in n]
        if skipped:
            doc.label(f"Catchment {cid}: methods not calculated")
            for s in skipped:
                doc.bullet(s)

    doc.doc.add_page_break()
    doc.heading("Notes and limitations", 1)
    for text in (
        "The sheets show the tool's calculation; the design value is the "
        "engineer's choice from the comparison in the Peak Flow Annexure.",
        "Rational: Tc is the value on the catchment layer (raised to 15 min when "
        "the floor is on); the Eq 3.12 value is shown as a check. The rural "
        "C1 classes are shown only when the C CSV carries the breakdown (the "
        "Rational C tool writes it); the urban C2 is the area-weighted value, "
        "without the per-use rows of the form.",
        "The forms' 'Max' (PMF) column is not calculated; return periods are "
        "those chosen for the run.",
        "Forms R1 (Alternative 1), U (unit hydrograph) and C (SCS-SA) are not "
        "part of this tool. CAPA (DWS 2025) has no Drainage Manual form; its "
        "sheet follows the same layout.",
    ):
        doc.bullet(text)
    msgs = list(dict.fromkeys(list(run.warnings) + notes))
    if msgs:
        doc.label("Processing messages")
        for m in msgs:
            doc.bullet(m)

    doc.heading("References", 1)
    for ref in REFERENCES:
        doc.para(ref.replace(" & ", " and "))
    doc.doc.save(str(path))
    return notes
