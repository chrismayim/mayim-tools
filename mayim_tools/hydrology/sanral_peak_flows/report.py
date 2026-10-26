"""Word (.docx) annexure for the SANRAL Deterministic Peak Flows tool, in the
Mayim report standard: title page, summary, tool and method, inputs and
settings, parameters and calculations, results per catchment, outputs,
notes and limitations, references (house text styles, 9 pt tables with
#ADADCC headers, numbered captions, 16.5 cm house-style charts).

No QGIS dependency; python-docx and matplotlib are imported lazily. Numbers
use a decimal point and no thousands separators."""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path

from mayim_tools._common.docx_report import Doc

from . import tables as tb
from .core import RunResult
from .export import REFERENCES, catchment_inputs, settings_rows
from .methods import METHOD_ORDER, METHODS

TITLE = "Deterministic Design Peak Flows (SANRAL)"
TEXT_WIDTH_CM = 16.5

SHORT_LABELS = {
    "rational_alt2": "Rational Alt 2",
    "rational_alt3": "Rational Alt 3",
    "sdf": "SDF",
    "midgley_pitman": "Midgley and Pitman",
    "rmf": "RMF (Kovacs)",
    "rmf_ratio": "RMF ratios",
    "capa": "CAPA",
}

# House line / fill colours (yellow only for fills, so skipped for bars too).
BAR_COLOURS = ("#0000FF", "#FF0000", "#00CC00", "#FFC000", "#FF00FF", "#00FFFF")

METHOD_TEXT = {
    "rational_alt2": (
        "Q_T = C_T I_T A / 3.6 (DM Section 3.5.1). Point rainfall for duration Tc "
        "from the modified Hershfield equation (Eq 3.9) up to 6 h, interpolated to "
        "the TR102 1-day depth between 6 and 24 h and between TR102 n-day depths "
        "beyond; a Hershfield depth above the 1-day depth is reduced to it."
    ),
    "rational_alt3": (
        "Q_T = C_T I_T A / 3.6 with the point rainfall for duration Tc read from "
        "the design-rainfall DDF table (Design Rainfall Estimation, Smithers and "
        "Schulze 2003), the Manual's recommended Alternative 3."
    ),
    "sdf": (
        "Standard Design Flood (Alexander 2001; DM Section 3.5.3): Tc by Eq 3.17, "
        "point rainfall from the basin's representative TR102 station (Hershfield "
        "up to 6 h, TR102 interpolation beyond), ARF by Eq 3.19 and the calibrated "
        "C_T = C2 + Y_T / 2.33 (C100 - C2). Used exactly as published (DM p 3-47)."
    ),
    "midgley_pitman": (
        "Q_T = 0.0377 K_T P A^0.6 C^0.2 with C = A S^0.5 / (L Lc) (DM Eq 3.34 and "
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
        "Ae^0.5 / L)^0.5, the mean annual flood Q50% from the M diagram (Figure "
        "5-2, digitised) and Q_P = K_P Q50% (Table 5-2)."
    ),
}

EQUATIONS = (
    ("Rational peak flow Q_T (m³/s)", "C_T I_T A / 3.6", "DM Section 3.5.1"),
    (
        "Runoff coefficient C_T (-)",
        "α C1 Ft + β C2 + γ C3",
        "Rural, urban and lakes components; Ft from DM Table 3.8 (1.0 at 1:200)",
    ),
    (
        "Point rainfall P (mm)",
        "Hershfield / TR102 or DDF",
        "Alternative 2: modified Hershfield (DM Eq 3.9) and TR102; Alternative 3: "
        "the design rainfall DDF table",
    ),
    ("Intensity I (mm/h)", "P ARF / Tc", "Catchment intensity for duration Tc"),
    (
        "Areal reduction factor ARF (%)",
        "DM Eq 3.10",
        "Or Pietersen et al. (2024) for durations of 24 h and longer",
    ),
    ("SDF runoff coefficient C_T (-)", "C2 + Y_T / 2.33 (C100 - C2)", "DM Section 3.5.3"),
    (
        "Midgley and Pitman Q_T (m³/s)",
        "0.0377 K_T P A^0.6 C^0.2",
        "C = A S^0.5 / (L Lc); DM Eq 3.34 and 3.35, Table 3.17",
    ),
    ("RMF Q_RMF (m³/s)", "Kovacs zone equations", "Kovacs (1988) TR137; DM Table 3.18"),
    ("CAPA index M (-)", "MAP (100 SA Ae^0.5 / L)^0.5", "DWS (2025) Section 5.4"),
    ("CAPA Q_P (m³/s)", "K_P Q50%", "Q50% from the M diagram; K_P from DWS Table 5-2"),
)  # fmt: skip

DERIVATION_LABELS = {
    "Tc_h": ("Tc (h)", 3),
    "Tc_used_h": ("Tc used (h)", 3),
    "P_mm": ("P (mm)", 1),
    "Pi_mmh": ("Pi (mm/h)", 2),
    "ARF_pct": ("ARF (%)", 1),
    "I_mmh": ("I (mm/h)", 2),
    "Ft": ("Ft (-)", 2),
    "C_T": ("C_T (-)", 4),
    "Y_T": ("Y_T (-)", 2),
    "K_T": ("K_T (-)", 2),
    "C_param": ("C (-)", 5),
    "ratio": ("Q_T/Q_RMF (-)", 3),
    "M": ("M (-)", 0),
    "Q50_m3s": ("Q50% (m³/s)", 2),
    "K_P": ("K_P (-)", 2),
}


def num(value, nd=2) -> str:
    """Decimal point, no thousands separators; '-' when missing."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float | int):
        x = float(value)
        if not math.isfinite(x):
            return "-"
        return f"{x:.{nd}f}"
    return str(value)


def q_text(value) -> str:
    if value is None:
        return "-"
    nd = 0 if value >= 100 else (1 if value >= 10 else 2)
    return num(float(value), nd)


def _even(n_cols: int, first: float) -> list[float]:
    rest = (TEXT_WIDTH_CM - first) / max(n_cols - 1, 1)
    return [first] + [rest] * (n_cols - 1)


def _txt(v) -> str:
    return "-" if v is None or str(v).strip() == "" else str(v)


# ---------------------------------------------------------------- chart


def chart_figure(item, rps):
    """(PNG stream, style warnings) of the 1:50 and 1:100 year peak flows per
    method as horizontal bars in the house chart style, or (None, []) when
    there is nothing to draw. Raises ImportError without matplotlib."""
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullLocator, ScalarFormatter

    from mayim_tools._common.report_figures import ChartStyle

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
        return None, []
    style = ChartStyle()
    with style.context():
        fig = style.figure(max(5.0, 2.2 + 0.9 * len(methods)))
        ax = fig.subplots()
        height = 0.8 / len(show)
        handles = []
        for j, T in enumerate(show):
            vals = []
            for m in methods:
                r = next(
                    (x for x in item.results if x.method == m and x.return_period == T),
                    None,
                )
                vals.append(r.q_m3s if r and r.q_m3s else 0.0)
            ys = [i + j * height for i in range(len(methods))]
            colour = BAR_COLOURS[j % len(BAR_COLOURS)]
            ax.barh(
                ys, vals, height=height * 0.9, color=colour, edgecolor="#000000",
                linewidth=0.75,
            )  # fmt: skip
            handles.append(
                Patch(
                    facecolor=colour, edgecolor="#000000", lw=0.75, label=f"1:{T} year"
                )
            )
        ax.set_yticks([i + height * (len(show) - 1) / 2 for i in range(len(methods))])
        ax.set_yticklabels([SHORT_LABELS[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlabel("Peak flow (m³/s)")
        style.axes(ax)
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(False, axis="y", which="both")
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
        legend = fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=len(handles),
            frameon=False,
            prop=style.font_props(style="italic"),
        )
        for text in legend.get_texts():
            text.set_color("#000000")
        stream = style.png(fig)
    return stream, list(style.warnings)


# ---------------------------------------------------------------- tables


def _results_table(doc, item, rps, cid):
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
    doc.caption("Table", f"Peak flows (m³/s), catchment {cid}")
    doc.table(header, rows, widths_cm=_even(len(header), 3.4))


def _derivation_tables(doc, item, cid):
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
        header = ["T (years)"] + [DERIVATION_LABELS[k][0] for k in keys] + ["Q (m³/s)"]
        rows = [
            [str(r.return_period)]
            + [num(r.derivation.get(k), DERIVATION_LABELS[k][1]) for k in keys]
            + [q_text(r.q_m3s)]
            for r in results
        ]
        doc.caption("Table", f"{SHORT_LABELS[m]} derivation, catchment {cid}")
        doc.table(header, rows, widths_cm=_even(len(header), 1.8))


def _range(item, T):
    vals = [
        r.q_m3s
        for r in item.results
        if r.return_period == T and r.q_m3s is not None and r.method != "rmf"
    ]
    if not vals:
        return "-"
    lo, hi = min(vals), max(vals)
    return q_text(lo) if lo == hi else f"{q_text(lo)} to {q_text(hi)}"


# ---------------------------------------------------------------- report


def write_report(path, run: RunResult, run_info: dict, context=None) -> list[str]:
    """Writes the annexure and returns notes (chart font substitutions,
    charts that could not be drawn). ``context`` gives project_number,
    project_name, software, outputs [(label, path)] and warnings."""
    context = context or {}
    notes: list[str] = []
    rps = tuple(run.settings.return_periods)
    doc = Doc(house_style=True)

    doc.title_page(
        TITLE,
        "This annexure gives design peak flows for each catchment by the "
        "deterministic and empirical methods of the SANRAL Drainage Manual (2013) "
        "Chapter 3 and the DWS (2025) Catchment Parameter method, for comparison "
        "and selection of the design flood by the engineer.",
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

    # ------------------------------------------------------------ summary
    doc.heading("Summary", 1)
    doc.para(
        f"Design peak flows were calculated for {len(run.items)} catchment(s) with "
        f"{len(run.settings.methods)} method(s) and return periods of "
        f"{', '.join(f'1:{T}' for T in rps)} years. Each catchment is an independent "
        "design point; no flood routing is applied. The Manual (Section 3.2.2) asks "
        "for the methods to be compared, not weighted, and for the SDF always to be "
        "included; the design value is the engineer's choice."
    )
    show = [T for T in (50, 100) if T in rps] or list(rps)[-2:]
    rows = []
    for item in run.items:
        v = catchment_inputs(item.catchment)
        rows.append(
            [v["catchment_id"], num(v["area_km2"], 3), num(v["tc_min"], 1)]
            + [_range(item, T) for T in show]
        )
    header = ["Catchment", "Area (km²)", "Tc (min)"] + [
        f"1:{T} year, range of methods (m³/s)" for T in show
    ]
    doc.caption("Table", "Summary of peak flows")
    doc.table(header, rows, widths_cm=[3.0, 2.2, 2.0] + [9.3 / len(show)] * len(show))

    # ------------------------------------------------------------ method
    doc.heading("Tool and method", 1)
    for m in METHOD_ORDER:
        if m in run.settings.methods:
            doc.numbered(f"{METHODS[m][1]}: {METHOD_TEXT[m]}")
    doc.numbered(
        "Methods outside their range of application are still calculated and "
        "flagged in the notes; a method with missing inputs is skipped for that "
        "catchment only."
    )

    # ------------------------------------------------------------ inputs
    doc.heading("Inputs and settings", 1)
    files = [
        ("Catchment layer", run_info.get("catchment layer")),
        ("DDF CSV", run_info.get("DDF CSV")),
        ("Runoff coefficient CSV", run_info.get("C CSV")),
        ("TR102 CSV", run_info.get("TR102 CSV")),
    ]
    doc.caption("Table", "Inputs and settings")
    doc.table(
        ["Item", "Value"],
        [[a, _txt(b)] for a, b in files]
        + [[a, _txt(b)] for a, b in settings_rows(run)],
        widths_cm=[6.0, 10.5],
    )
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
    doc.table(header, rows, widths_cm=[2.2, 1.5, 1.5, 1.4, 1.7, 1.4, 1.4, 1.4, 4.0])

    # ------------------------------------------------------------ parameters
    doc.heading("Parameters and calculations", 1)
    doc.caption("Table", "Parameters and equations")
    doc.table(
        ["Parameter", "Equation", "Definition (source)"],
        [list(r) for r in EQUATIONS],
        widths_cm=[5.0, 4.2, 7.3],
    )

    # ------------------------------------------------------------ results
    doc.heading("Results", 1)
    for item in run.items:
        cid = item.catchment.catchment_id
        doc.heading(f"Catchment {cid}", 2)
        _results_table(doc, item, rps, cid)
        try:
            stream, warns = chart_figure(item, rps)
            notes.extend(warns)
        except ImportError:
            stream = None
            notes.append("Charts not drawn: matplotlib is not available.")
        if stream is not None:
            doc.picture(
                stream, f"1:50 and 1:100 year peak flows by method, catchment {cid}"
            )
        doc.heading("Parameter derivations", 3)
        _derivation_tables(doc, item, cid)
        cnotes = item.notes_text()
        if cnotes:
            doc.label("Notes")
            for n in cnotes.split("; "):
                doc.bullet(n)

    # ------------------------------------------------------------ outputs
    outputs = context.get("outputs") or []
    if outputs:
        doc.heading("Outputs", 1)
        doc.caption("Table", "Files written")
        doc.table(
            ["Output", "File"],
            [[a, Path(b).name if b else "-"] for a, b in outputs],
            widths_cm=[6.0, 10.5],
        )
        folders = sorted({str(Path(b).parent) for _, b in outputs if b})
        if folders:
            doc.para("Folder(s): " + "; ".join(folders))

    # ------------------------------------------------------------ notes
    doc.heading("Notes and limitations", 1)
    doc.label("Range of application")
    for m in ("rational", "sdf", "midgley_pitman", "rmf", "capa"):
        doc.bullet(tb.APPLICABILITY[m])
    doc.label("Source notes")
    for note in tb.SOURCE_NOTES:
        doc.bullet(note)
    msgs = list(
        dict.fromkeys(list(run.warnings) + list(context.get("warnings", [])) + notes)
    )
    if msgs:
        doc.label("Processing messages")
        for msg in msgs:
            doc.bullet(msg)

    doc.heading("References", 1)
    for ref in REFERENCES:
        doc.para(ref.replace(" & ", " and "))
    doc.doc.save(str(path))
    return notes
