"""Word (.docx) report of the 'Catchments: 2. Time of Concentration' tool in
the Mayim report standard: title page, summary, tool and method, inputs and settings,
parameters and calculations, results, outputs, notes and limitations,
references (house text styles, 9 pt tables, numbered captions, 16.5 cm
house-style charts).

No QGIS dependency; python-docx and matplotlib are imported lazily. Numbers
use a decimal point and no thousands separators."""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path

import numpy as np

from mayim_tools._common.docx_report import Doc

from . import tables as tb
from .core import storm_guidance
from .export import status_text

TITLE = "Time of Concentration"
MAX_FIGURES = 30
COLOURS = {"overland": "#FF0000", "shallow": "#00CC00", "channel": "#0000FF"}


def num(value, nd=1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float | int) and not isinstance(value, bool):
        x = float(value)
        return "-" if not math.isfinite(x) else f"{x:.{nd}f}"
    return str(value)


def pct(value, nd=2) -> str:
    return "-" if value is None else num(100.0 * value, nd)


def _sc_label(code):
    return tb.SC_TYPES[code][0] if code in tb.SC_TYPES else (code or "-")


def profile_figure(c):
    """(PNG stream, warnings) of the flow-path profile with the three parts,
    or (None, []) without a profile."""
    if c.x is None or len(c.x) < 2:
        return None, []
    from mayim_tools._common.report_figures import ChartStyle

    v = c.values
    x, z = c.x, c.z
    big_l = float(x[-1])
    ch = v.get("l_ch_m") or 0.0
    ov = v.get("l_ov_m") or 0.0
    parts = [
        ("Channel", "channel", 0.0, ch),
        ("Shallow concentrated flow", "shallow", ch, big_l - ov),
        ("Overland flow", "overland", big_l - ov, big_l),
    ]
    style = ChartStyle()
    with style.context():
        fig = style.figure(7.0)
        ax = fig.subplots()
        for label, key, a, b in parts:
            if b - a <= 0:
                continue
            # the part's own vertices plus its interpolated ends, so the parts
            # join up
            m = (x > a) & (x < b)
            xs = np.concatenate([[a], x[m], [b]])
            zs = np.interp(xs, x, z)
            ax.plot(xs / 1000.0, zs, color=COLOURS[key], lw=1.5, label=label)
        for xb in sorted({ch, big_l - ov}):
            if 0 < xb < big_l:
                ax.axvline(xb / 1000.0, color="#000000", lw=0.75, ls="--")
        ax.set_xlabel("Distance from the outlet (km)")
        ax.set_ylabel("Elevation (m)")
        ax.set_xlim(0, big_l / 1000.0)
        style.axes(ax)
        style.legend(ax, loc="upper left")
        stream = style.png(fig)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return stream, list(style.warnings)


def write_report(path, items, run_info: dict, context=None) -> list[str]:
    """Writes the report and returns notes. ``context`` gives
    project_number, project_name, software, outputs [(label, path)],
    warnings, figures (bool) and settings (Settings)."""
    context = context or {}
    s = context.get("settings")
    notes: list[str] = []
    doc = Doc(house_style=True)
    doc.title_page(
        TITLE,
        "This annexure gives the time of concentration of each catchment by "
        "three empirical formulae on the whole longest flow path and by a "
        "combined method that splits the path into overland, shallow "
        "concentrated and channel flow with parameters set per catchment, and "
        "the Tc selected for design.",
        [
            ("Project number", context.get("project_number") or "-"),
            ("Project / site", context.get("project_name") or "-"),
            ("Prepared by", "Mayim Consulting"),
            ("Tool and version",
             f"{run_info.get('tool', 'Catchments: 2. Time of Concentration')}, version "
             f"{run_info.get('version', '-')}"),
            ("Software", context.get("software") or "-"),
            ("Report date", _dt.date.today().isoformat()),
            ("Document status", "Draft"),
        ],  # fmt: skip
        subtitle=context.get("project_name") or None,
    )

    # ------------------------------------------------------------ summary
    n_missing = sum(1 for c in items if c.missing)
    doc.heading("Summary", 1)
    doc.para(
        f"Times of concentration were calculated for {len(items)} catchment(s). "
        "The selected Tc (last column) is the method chosen per catchment in "
        "tc_method"
        + (f" or, where none was chosen, the run default ({s.default_method})"
           if s else "")
        + "."
        + (f" {n_missing} catchment(s) still need parameters for the combined "
           "method (see Inputs and settings)." if n_missing else "")
    )  # fmt: skip
    doc.caption("Table", "Times of concentration (min)")
    doc.table(
        ["Catchment", "A (km²)", "L (km)", "Kirpich", "USBR", "Bransby-Williams",
         "Combined", "Selected", "Method"],
        [[c.catchment_id, num(c.area_km2, 3), num(c.lfp_km, 3),
          num(c.values.get("tc_kirp_mn")), num(c.values.get("tc_usbr_mn")),
          num(c.values.get("tc_bw_mn")), num(c.values.get("tc_comb_mn")),
          num(c.values.get("tc_sel_mn")), c.values.get("tc_sel_by") or "-"]
         for c in items],
        widths_cm=[2.3, 1.5, 1.4, 1.5, 1.4, 2.2, 2.0, 1.9, 2.3],
    )  # fmt: skip

    # ------------------------------------------------------------ method
    doc.heading("Tool and method", 1)
    for text in (
        "Kirpich, USBR (SANRAL Drainage Manual Eq 3.12) and Bransby-Williams "
        "are applied to the whole longest flow path L with its average, 10-85 "
        "and equal-area slopes respectively. They treat the whole path as one "
        "flow type. Eq 3.12 is the Kirpich formula in other units; the two "
        "differ only in the slope used.",
        "The combined method follows the Drainage Manual (Section 3.4: in "
        "small catchments Tc is the sum of the overland and channel flow "
        "times) and the NRCS velocity method (NEH 630 Chapter 15). The path is "
        "split, from the divide down, into overland (sheet) flow over the "
        "first L_ov metres (Kerby, DM Eq 3.11), shallow concentrated flow to "
        "the channel start (V = k S^0.5, NEH 630 Table 15-3) and channel flow "
        "from the channel start to the outlet (DM Eq 3.12 with the 10-85 "
        "slope of the channel part).",
        "The channel start is where the upstream area along the path first "
        "reaches the stream threshold used for the delineation (Catchment "
        "Delineation's ch_head_m), unless a value is entered in ch_start_m. "
        "Each part takes its slope from the DEM along the flow path.",
        "The parameters (overland length, Kerby r, shallow-flow surface, "
        "channel start, method) are fields of the catchment layer and are set "
        "per catchment by the engineer. Where a land-cover raster is given, "
        "the majority class along the overland and the shallow-flow parts "
        "gives a suggested r and surface, reported for checking.",
    ):
        doc.numbered(text)

    # ------------------------------------------------------------ inputs
    doc.heading("Inputs and settings", 1)
    rows = [[k, num(v) if isinstance(v, float) else (str(v) if v not in (None, "")
             else "-")] for k, v in run_info.get("rows", [])]  # fmt: skip
    doc.caption("Table", "Inputs and settings")
    doc.table(["Item", "Value"], rows, widths_cm=[6.0, 10.5])
    doc.caption("Table", "Parameters per catchment")
    doc.table(
        ["Catchment", "L_ov (m)", "r", "Suggested r", "Shallow-flow surface",
         "Suggested surface", "Channel start (m)", "tc_method"],
        [[c.catchment_id, num(c.params.ov_len_m, 0), num(c.params.ov_r, 2),
          num(c.values.get("sug_r"), 2), _sc_label(c.params.sc_type),
          _sc_label(c.values.get("sug_sc")) if c.values.get("sug_sc") else "-",
          num(c.params.ch_start_m, 0), c.params.tc_method or "-"]
         for c in items],
        widths_cm=[2.3, 1.3, 1.0, 1.6, 3.3, 3.0, 1.8, 2.2],
    )  # fmt: skip
    missing = [(c.catchment_id, ", ".join(c.missing)) for c in items if c.missing]
    if missing:
        doc.label("Parameters still needed")
        for cid, what in missing:
            doc.bullet(f"{cid}: {what}")

    # ------------------------------------------------------------ parameters
    doc.heading("Parameters and calculations", 1)
    doc.caption("Table", "Equations")
    doc.table(
        ["Quantity", "Equation", "Source"],
        [
            ["Kirpich Tc (min)", "0.0195 L^0.77 S_avg^-0.385 (L in m)",
             "Kirpich (1940)"],
            ["USBR Tc (h)", "(0.87 L² / (1000 S_1085))^0.385 (L in km)",
             "USBR (1973); DM Eq 3.12"],
            ["Bransby-Williams Tc (min)",
             "58 L / (A^0.1 S_e^0.2) (L km, A km², S_e m/km)",
             "Bransby-Williams (1922)"],
            ["Overland flow time (h)", "0.604 (r L_ov / S_ov^0.5)^0.467 (L in km)",
             "Kerby (1959); DM Eq 3.11"],
            ["Shallow flow velocity (m/s)", "V = k S_sc^0.5",
             "NEH 630 Ch 15, Table 15-3 (k converted to m/s)"],
            ["Shallow flow time (min)", "L_sc / (60 V)", "NEH 630 Ch 15"],
            ["Channel flow time (h)",
             "(0.87 L_ch² / (1000 S_ch))^0.385 (L in km)",
             "DM Eq 3.12; S_ch the 10-85 slope of the channel part"],
            ["Combined Tc (min)", "t_ov + t_sc + t_ch",
             "DM Section 3.4; NEH 630 Ch 15"],
        ],
        widths_cm=[4.2, 6.8, 5.5],
    )  # fmt: skip
    doc.caption("Table", "Kerby roughness r")
    doc.table(["r", "Surface (source)"],
              [[f"{r:g}", text] for r, text in tb.KERBY_R.items()],
              widths_cm=[2.0, 14.5])  # fmt: skip
    doc.caption("Table", "Shallow concentrated flow surfaces (NEH 630 Table 15-3)")
    doc.table(["Code", "Surface", "k (m/s)"],
              [[code, text, f"{k:.3f}"] for code, (text, k) in tb.SC_TYPES.items()],
              widths_cm=[2.5, 11.0, 3.0])  # fmt: skip

    # ------------------------------------------------------------ results
    doc.heading("Results", 1)
    doc.caption("Table", "Overland flow (Kerby)")
    doc.table(
        ["Catchment", "L_ov (m)", "S_ov (%)", "r", "t_ov (min)"],
        [[c.catchment_id, num(c.values.get("l_ov_m"), 0), pct(c.values.get("s_ov")),
          num(c.params.ov_r, 2), num(c.values.get("tc_ov_mn"))] for c in items],
        widths_cm=[3.5, 3.0, 3.0, 3.0, 4.0],
    )  # fmt: skip
    doc.caption("Table", "Shallow concentrated flow (NRCS)")
    doc.table(
        ["Catchment", "L_sc (m)", "S_sc (%)", "Surface", "V (m/s)", "t_sc (min)"],
        [[c.catchment_id, num(c.values.get("l_sc_m"), 0), pct(c.values.get("s_sc")),
          c.params.sc_type or "-", num(c.values.get("v_sc_ms"), 2),
          num(c.values.get("tc_sc_mn"))] for c in items],
        widths_cm=[3.0, 2.5, 2.5, 3.0, 2.5, 3.0],
    )  # fmt: skip
    doc.caption("Table", "Channel flow (DM Eq 3.12)")
    doc.table(
        ["Catchment", "L_ch (m)", "S_ch, 10-85 (%)", "t_ch (min)", "Combined Tc (min)"],
        [[c.catchment_id, num(c.values.get("l_ch_m"), 0), pct(c.values.get("s_ch")),
          num(c.values.get("tc_ch_mn")), num(c.values.get("tc_comb_mn"))]
         for c in items],
        widths_cm=[3.5, 3.0, 3.5, 3.0, 3.5],
    )  # fmt: skip

    if context.get("figures", True):
        drawn = 0
        for c in items:
            if drawn >= MAX_FIGURES:
                notes.append(
                    f"Profile figures drawn for the first {MAX_FIGURES} catchments "
                    "only."
                )
                break
            try:
                stream, warns = profile_figure(c)
            except ImportError:
                notes.append("Figures not drawn: matplotlib is not available.")
                break
            notes.extend(w for w in warns if w not in notes)
            if stream is not None:
                doc.picture(
                    stream,
                    f"Longest flow path of catchment {c.catchment_id} with the "
                    "overland, shallow concentrated and channel parts",
                )
                drawn += 1

    guide = [g for g in (storm_guidance(c.values) for c in items) if g]
    if guide:
        lo = min(g[0] for g in guide)
        hi = max(g[1] for g in guide)
        doc.label("Design storm duration and time step")
        doc.para(
            "Design storm durations should bracket the critical duration; as a "
            f"starting range, test from about {0.5 * lo:.0f} min to "
            f"{2.0 * hi:.0f} min. For hydrograph models, use a computational time "
            "step of no more than about 0.1 to 0.2 x Tc, i.e. "
            f"{0.1 * lo:.1f} to {0.2 * lo:.1f} min for the shortest Tc here "
            "(compare the unit hydrograph guidance of USDA-NRCS 2007, dt <= 0.29 "
            "x lag)."
        )

    # ------------------------------------------------------------ outputs
    outputs = context.get("outputs") or []
    if outputs:
        doc.heading("Outputs", 1)
        doc.caption("Table", "Files and layers written")
        doc.table(
            ["Output", "File"],
            [[a, Path(b).name if b else "-"] for a, b in outputs],
            widths_cm=[6.0, 10.5],
        )

    # ------------------------------------------------------------ notes
    doc.heading("Notes and limitations", 1)
    for text in (
        "The parameters are the engineer's: the tool does not choose them. "
        "Land-cover suggestions are a first guess from the majority class "
        "along each part of the path and should be checked against site "
        "conditions or imagery.",
        "Kerby was derived for overland flow up to about 366 m (1200 ft); NRCS "
        "notes that sheet flow seldom runs more than about 30 m (100 ft) on "
        "natural ground before it concentrates.",
        f"Slopes below {100 * tb.S_MIN:.1f} % (overland and shallow flow) and "
        f"{100 * tb.S_MIN_CHANNEL:.2f} % (channel) are raised to those values "
        "and flagged; very flat DEM stretches otherwise give unrealistic "
        "times.",
        "The channel start depends on the stream threshold of the "
        "delineation; check it against mapped or visible channels and enter "
        "ch_start_m where they differ.",
        "Tc is calculated along the longest flow path, not the path of longest "
        "travel time. Where a shorter path is much slower (e.g. a long flat "
        "overland stretch) check it separately.",
    ):
        doc.bullet(text)
    flagged = [(c.catchment_id, status_text(c)) for c in items if status_text(c)]
    if flagged:
        doc.label("Catchment flags")
        for cid, text in flagged:
            doc.bullet(f"{cid}: {text}")
    msgs = list(dict.fromkeys(list(context.get("warnings", [])) + notes))
    if msgs:
        doc.label("Processing messages")
        for m in msgs:
            doc.bullet(m)

    doc.heading("References", 1)
    for ref in tb.REFERENCES:
        doc.para(ref.replace(" & ", " and "))
    doc.doc.save(str(path))
    return notes
