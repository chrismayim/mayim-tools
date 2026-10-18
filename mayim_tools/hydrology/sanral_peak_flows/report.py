"""Word (.docx) annexure for the SANRAL Deterministic Peak Flows tool (no
QGIS dependency; python-docx and matplotlib are imported lazily through
mayim_tools._common.docx_report). Numbers use a decimal comma and no
thousands separators."""

from __future__ import annotations

import math

from mayim_tools._common.docx_report import (
    C_AQUA,
    C_BLUE,
    C_INK_2,
    C_ORANGE,
    Doc,
    new_figure,
    png,
    style_axes,
)

from . import tables as tb
from .core import RunResult
from .export import REFERENCES, catchment_inputs, settings_rows
from .methods import METHOD_ORDER, METHODS

SHORT_LABELS = {
    "rational_alt2": "Rational Alt 2",
    "rational_alt3": "Rational Alt 3",
    "sdf": "SDF",
    "midgley_pitman": "Midgley & Pitman",
    "rmf": "RMF (Kovacs)",
    "rmf_ratio": "RMF ratios",
    "capa": "CAPA",
}

CHART_COLOURS = (C_BLUE, C_ORANGE, C_AQUA, "#8a5cc2", "#c2477a", "#7a8a2e", C_INK_2)

METHOD_TEXT = {
    "rational_alt2": (
        "Q_T = C_T I_T A / 3,6 (DM Section 3.5.1). Point rainfall for duration Tc "
        "from the modified Hershfield equation (Eq 3.9) up to 6 h, interpolated to "
        "the TR102 1-day depth between 6 and 24 h and between TR102 n-day depths "
        "beyond; a Hershfield depth above the 1-day depth is reduced to it."
    ),
    "rational_alt3": (
        "Q_T = C_T I_T A / 3,6 with the point rainfall for duration Tc read from "
        "the design-rainfall DDF table (Design Rainfall Estimation, Smithers & "
        "Schulze 2003), the Manual's recommended Alternative 3."
    ),
    "sdf": (
        "Standard Design Flood (Alexander 2002; DM Section 3.5.3): Tc by Eq 3.17, "
        "point rainfall from the basin's representative TR102 station (Hershfield "
        "up to 6 h, TR102 interpolation beyond), ARF by Eq 3.19 and the calibrated "
        "C_T = C2 + Y_T / 2,33 (C100 - C2). Used exactly as published (DM p 3-47)."
    ),
    "midgley_pitman": (
        "Q_T = 0,0377 K_T P A^0,6 C^0,2 with C = A S^0,5 / (L Lc) (DM Eq 3.34 - "
        "3.35) and K_T for the generalised veld-type zone (Table 3.17), 1:10 to "
        "1:100 years."
    ),
    "rmf": (
        "Regional maximum flood (Kovacs 1988, TR137; DM Table 3.18) for the "
        "Kovacs region at the site, from the transition- or flood-zone equation "
        "for the effective area (Francou-Rodier, DM Eq 3.36, with a custom K)."
    ),
    "rmf_ratio": (
        "1:50, 1:100 and 1:200 year peaks as Q_T/Q_RMF fractions of the RMF "
        "(DM Appendix 3D, Table 3D.1), interpolated linearly in Ae."
    ),
    "capa": (
        "Catchment Parameter method (DWS 2025 Section 5.4): M = MAP (100 SA "
        "Ae^0,5 / L)^0,5, the mean annual flood Q50% from the M diagram (Figure "
        "5-2, digitised) and Q_P = K_P Q50% (Table 5-2)."
    ),
}

DERIVATION_LABELS = {
    "Tc_h": ("Tc (h)", 3),
    "Tc_used_h": ("Tc used (h)", 3),
    "P_mm": ("P (mm)", 1),
    "Pi_mmh": ("Pi (mm/h)", 2),
    "ARF_pct": ("ARF (%)", 1),
    "I_mmh": ("I (mm/h)", 2),
    "Ft": ("Ft", 2),
    "C_T": ("C_T", 4),
    "Y_T": ("Y_T", 2),
    "K_T": ("K_T", 2),
    "C_param": ("C", 5),
    "ratio": ("Q_T/Q_RMF", 3),
    "M": ("M", 0),
    "Q50_m3s": ("Q50% (m3/s)", 2),
    "K_P": ("K_P", 2),
}


def num(value, nd=2) -> str:
    """Decimal comma, no thousands separators."""
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "-"
        return f"{value:.{nd}f}".replace(".", ",")
    return str(value)


def q_text(value) -> str:
    if value is None:
        return "-"
    nd = 0 if value >= 100 else (1 if value >= 10 else 2)
    return num(float(value), nd)


def _results_table(doc, item, rps):
    by = {(r.method, r.return_period): r for r in item.results}
    methods = [m for m in METHOD_ORDER if any(r.method == m for r in item.results)]
    header = ["Method"] + [f"1:{T}" for T in rps] + ["RMF"]
    rows = []
    for m in methods:
        if m == "rmf":
            r = by.get(("rmf", None))
            rows.append(
                [SHORT_LABELS[m]] + ["-"] * len(rps) + [q_text(r.q_m3s if r else None)]
            )
            continue
        rows.append(
            [SHORT_LABELS[m]]
            + [q_text(by[(m, T)].q_m3s) if (m, T) in by else "-" for T in rps]
            + ["-"]
        )
    width = (16.0 - 3.4) / (len(rps) + 1)
    doc.table(header, rows, widths_cm=[3.4] + [width] * (len(rps) + 1))


def _derivation_tables(doc, item):
    for m in METHOD_ORDER:
        results = [r for r in item.results if r.method == m and r.return_period]
        if not results:
            continue
        keys = [k for k in DERIVATION_LABELS if k in results[0].derivation]
        if not keys:
            continue
        doc.label(METHODS[m][1])
        first = results[0].derivation
        for k in ("rain_source", "ARF_method", "veld_zone", "region", "basin"):
            if k in first:
                doc.para(
                    str(first[k]), bold_lead=k.replace("_", " ").capitalize() + ":"
                )
        header = ["T (years)"] + [DERIVATION_LABELS[k][0] for k in keys] + ["Q (m3/s)"]
        rows = []
        for r in results:
            rows.append(
                [str(r.return_period)]
                + [num(r.derivation.get(k), DERIVATION_LABELS[k][1]) for k in keys]
                + [q_text(r.q_m3s)]
            )
        doc.table(header, rows)


def _chart(item, rps):
    show = [T for T in (50, 100) if T in rps] or list(rps)[-2:]
    methods = [
        m
        for m in METHOD_ORDER
        if m != "rmf"
        and any(
            r.method == m and r.return_period in show and r.q_m3s for r in item.results
        )
    ]
    if not methods:
        return None
    fig = new_figure(1.2 + 0.9 * len(methods))
    ax = fig.add_subplot(1, 1, 1)
    height = 0.8 / len(show)
    for j, T in enumerate(show):
        vals = []
        for m in methods:
            r = next(
                (x for x in item.results if x.method == m and x.return_period == T),
                None,
            )
            vals.append(r.q_m3s if r and r.q_m3s else 0.0)
        ys = [i + j * height for i in range(len(methods))]
        ax.barh(
            ys, vals, height=height * 0.9, color=CHART_COLOURS[j], label=f"1:{T} year"
        )
    ax.set_yticks([i + height * (len(show) - 1) / 2 for i in range(len(methods))])
    ax.set_yticklabels([SHORT_LABELS[m] for m in methods], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Peak flow (m³/s)")
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False)
    return fig


def write_report(path, run: RunResult, run_info: dict) -> None:
    doc = Doc()
    rps = tuple(run.settings.return_periods)
    doc.heading("Deterministic design peak flows", 1)
    doc.para(
        "Design peak flows were calculated for each catchment with the "
        "deterministic and empirical methods of the SANRAL Drainage Manual (2013) "
        "Chapter 3 and, where selected, the DWS (2025) Catchment Parameter method. "
        "Each catchment is treated as an independent design point; no flood routing "
        "is applied. The Manual (Section 3.2.2) requires the results of the methods "
        "to be compared - without weighting - and the most appropriate value selected; "
        "the SDF method is always to be included."
    )

    doc.heading("Methods", 2)
    for m in METHOD_ORDER:
        if m in run.settings.methods:
            doc.para(METHOD_TEXT[m], bold_lead=METHODS[m][1] + ".")

    doc.heading("Settings", 2)
    doc.table(["Setting", "Value"], [[k, v] for k, v in settings_rows(run)])

    doc.heading("Catchment inputs", 2)
    header = [
        "Catchment",
        "A (km²)",
        "Ae (km²)",
        "L (km)",
        "S1085 (m/m)",
        "Lc (km)",
        "Tc (min)",
        "MAP (mm)",
        "Regions",
    ]
    rows = []
    for item in run.items:
        v = catchment_inputs(item.catchment)
        regions = " / ".join(
            str(x)
            for x in (
                f"SDF {v['sdf_basin']}" if v["sdf_basin"] else None,
                f"veld {v['veld_zone']}" if v["veld_zone"] else None,
                v["kovacs_region"],
                f"ARF {v['arf_region']}" if v["arf_region"] else None,
            )
            if x
        )
        rows.append(
            [
                v["catchment_id"],
                num(v["area_km2"], 3),
                num(v["effective_km2"], 3),
                num(v["lfp_km"], 3),
                num(v["s1085"], 5),
                num(v["lca_km"], 3),
                num(v["tc_min"], 1),
                num(v["map_mm"], 0),
                regions or "-",
            ]
        )
    doc.caption("Table", "Catchment inputs")
    doc.table(header, rows)

    for item in run.items:
        cid = item.catchment.catchment_id
        doc.heading(f"Catchment {cid}", 2)
        doc.caption("Table", f"Peak flows (m³/s), catchment {cid}")
        _results_table(doc, item, rps)
        fig = _chart(item, rps)
        if fig is not None:
            doc.picture(png(fig), f"Comparison of methods, catchment {cid}")
        doc.heading("Parameter derivations", 3)
        _derivation_tables(doc, item)
        notes = item.notes_text()
        if notes:
            doc.heading("Notes", 3)
            for n in notes.split("; "):
                doc.bullet(n)

    doc.heading("Range of application", 2)
    for m in ("rational", "sdf", "midgley_pitman", "rmf", "capa"):
        doc.bullet(tb.APPLICABILITY[m])
    doc.heading("Source notes", 2)
    for note in tb.SOURCE_NOTES:
        doc.bullet(note)
    for msg in run.warnings:
        doc.bullet(msg)
    doc.heading("References", 2)
    for ref in REFERENCES:
        doc.para(ref)
    doc.para(
        f"{run_info.get('tool', '')} {run_info.get('version', '')}, "
        f"{run_info.get('run_time', '')}",
        style=None,
    )
    doc.doc.save(path)
