"""Word (.docx) report for Adjust Sub-daily Rainfall to DDF, in the Mayim
report standard: title page, summary, tool and method, inputs and settings,
parameters and calculations, results, outputs, notes and limitations, and
references (house text styles, 9 pt tables with #ADADCC headers, numbered
captions, 16.5 cm house-style figure).

Requires python-docx (soft import) so core.py and the CSV outputs never
depend on it. Numbers have no thousands separators.
"""

from __future__ import annotations

import datetime as _dt
import math

REFERENCES = [
    "Ball, J., Babister, M., Nathan, R., Weeks, W., Weinmann, E., Retallick, M. "
    "and Testoni, I. (eds) (2019). Australian Rainfall and Runoff: A Guide to "
    "Flood Estimation, Books 2 and 4. Commonwealth of Australia (Geoscience "
    "Australia).",
    "Efron, B. and Tibshirani, R.J. (1993). An Introduction to the Bootstrap. "
    "Chapman and Hall, New York.",
    "Guerreiro, S.B., et al. (2024). Unravelling the complex interplay between "
    "daily and sub-daily rainfall extremes in different climates. Weather and "
    "Climate Extremes.",
    "Hosking, J.R.M. and Wallis, J.R. (1997). Regional Frequency Analysis: An "
    "Approach Based on L-Moments. Cambridge University Press.",
    "Lavers, D.A., et al. (2022). An evaluation of ERA5 precipitation for climate "
    "monitoring. Quarterly Journal of the Royal Meteorological Society 148: "
    "3152-3165.",
    "Maraun, D. (2013). Bias correction, quantile mapping, and downscaling: "
    "revisiting the inflation issue. Journal of Climate 26(6): 2137-2143.",
    "Reder, A., et al. (2022). Characterizing extreme values of precipitation at "
    "very high resolution: an experiment over twenty European cities. Weather and "
    "Climate Extremes.",
    "Smithers, J.C. and Schulze, R.E. (2003). Design Rainfall and Flood Estimation "
    "in South Africa. WRC Report K5/1060, Water Research Commission, Pretoria.",
    "Weiss, L.L. (1964). Ratio of true to fixed-interval maximum rainfall. Journal "
    "of the Hydraulics Division, ASCE 90(1): 77-82.",
    "Wu, G., Lv, P., Mao, Y. and Wang, K. (2024). ERA5 precipitation over China: "
    "better relative hourly and daily distribution than absolute values. Journal "
    "of Climate 37(5): 1581-1596.",
]
TITLE = "Adjust Sub-daily Rainfall to DDF"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"] + [
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
TEXT_WIDTH_CM = 16.5


def _f(v, nd=2):
    """Number with ``nd`` decimals and no thousands separator; '-' when
    missing."""
    if v is None or v == "":
        return "-"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(x) or math.isinf(x):
        return "-"
    return f"{x:.{nd}f}"


def _txt(v) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "-"
    return str(v)


def _list(v) -> str:
    """A comma-separated list with a space after each comma (so it is never
    read as a thousands separator)."""
    if v is None or v == "":
        return "-"
    if isinstance(v, list | tuple):
        return ", ".join(f"{x:g}" if isinstance(x, float) else str(x) for x in v)
    return ", ".join(
        p.strip() for p in str(v).replace(";", ",").split(",") if p.strip()
    )


def _even(n_cols: int, first: float) -> list[float]:
    rest = (TEXT_WIDTH_CM - first) / max(n_cols - 1, 1)
    return [first] + [rest] * (n_cols - 1)


def _role(r):
    return {"calibration": "fitted", "validation": "check"}.get(r, r)


def write_docx(
    result,
    path,
    chart_path=None,
    inputs=None,
    chart=None,
    context=None,
):
    """Writes the report and returns a list of notes (figures that could not
    be added). ``chart`` is a PNG stream (house-style figure); without it
    ``chart_path`` (a PNG file) is used, or the figure is drawn here.
    ``context`` gives project_number, project_name, software, settings
    (extra [(item, value)] rows), outputs [(label, path)] and warnings."""
    from mayim_tools._common.docx_report import Doc

    inputs = inputs or {}
    context = context or {}
    notes: list[str] = []
    md = result.metadata
    ap = result.adjust_params
    mi = result.map_info or {}
    site = inputs.get("site") or md.get("input_site") or md.get("reference_site")
    tier3 = "INDICATIVE" in (result.status_stamp or "")

    d = Doc(house_style=True)
    purpose = (
        "This report documents the adjustment of a sub-daily gridded rainfall "
        "series so that its annual-maximum depths at every tested duration match "
        "a reference depth-duration-frequency (DDF) table, while the timing and "
        "sequence of the recorded storms are kept, for use in design and "
        "continuous simulation."
    )
    if tier3:
        purpose += f" {result.status_stamp}."
    d.title_page(
        TITLE,
        purpose,
        [
            ("Project number", context.get("project_number") or "-"),
            ("Project / site", context.get("project_name") or site or "-"),
            ("Prepared by", "Mayim Consulting"),
            (
                "Tool and version",
                f"{TITLE}, version {context.get('tool_version') or '-'}",
            ),
            ("Software", context.get("software") or "-"),
            ("Report date", _dt.date.today().isoformat()),
            ("Document status", "Draft"),
        ],
        subtitle=context.get("project_name") or None,
    )

    # ------------------------------------------------------------ summary
    d.heading("Summary", 1)
    d.para(
        f"The rainfall series{f' for {site}' if site else ''} "
        f"({md.get('record_start')} to {md.get('record_end')}, "
        f"{md.get('n_years_used')} complete years at a {_f(md.get('native_step_min'), 0)} "
        "min step) was adjusted to the reference DDF. Before adjustment: "
        f"{md.get('overall_verdict') or '-'}. After adjustment: "
        f"{result.verdict_after or 'not adjusted'}."
    )
    if tier3:
        d.para(f"{result.status_stamp}: the reference is a gridded product only.")
    d.caption("Table", "Summary of the adjustment")
    d.table(
        ["Item", "Value"],
        [
            ["Reference tier", _txt(md.get("reference_tier"))],
            ["Mean annual precipitation, input (mm)", _f(mi.get("map_input_mm"), 1)],
            [
                "Target mean annual precipitation (mm)",
                f"{_f(mi.get('map_target_mm'), 1)} ({mi.get('map_target_source', '-')})",
            ],
            ["Mean annual precipitation, adjusted (mm)", _f(mi.get("map_output_mm"), 1)],
            ["Bulk exponent γ (-)", _f(mi.get("transfer_gamma"), 3)],
            [
                "Anchor self-check (pass 1)",
                f"{'passed' if result.self_check_passed else 'WARNING'}, maximum "
                f"deviation {_f(100 * (md.get('self_check_max_deviation') or 0), 1)} %",
            ],
            [
                "Verdict, validation durations only",
                _txt(ap.get("verdict_after_validation_durations")),
            ],
            ["Storms identified / reshaped", f"{_txt(md.get('n_events'))} / "
             f"{_txt(ap.get('n_storms_adjusted'))}"],
        ],  # fmt: skip
        [7.0, 9.5],
    )

    # ------------------------------------------------------------ method
    d.heading("Tool and method", 1)
    anchor_h = (md.get("anchor_duration_min") or 1440) / 60
    for step in (
        [
            "The series is regularised to its native time step; missing data stay "
            "missing (never zero) and years below the completeness threshold are left "
            "out of every duration. A time-zone offset, when given, is applied before "
            "calendar years and months are assigned.",
            "Annual-maximum series (AMS) are extracted with sliding windows at each "
            "duration and fitted with the chosen distribution; return periods are "
            "annual-maximum values (AEP = 1/T), as in the reference table. Maxima from "
            "fixed-interval data are multiplied by the Weiss (1964) factor.",
            "Storms are separated by the inter-event time definition (IETD) and a "
            "minimum storm depth.",
            f"Pass 1, magnitude and annual total: the product's own AMS distribution G "
            f"is fitted at the anchor duration ({_f(anchor_h, 0)} h). Each storm's "
            "anchor depth x is mapped to a target depth f(x) and the whole storm is "
            "multiplied by f(x)/x, which keeps its timing. In the annual-maximum range "
            f"(x ≥ x0, the product's depth at T0 = {_f(mi.get('transfer_t0_yr'), 1)} "
            "years) f(x) is the reference depth at the storm's own return period in G. "
            "Smaller storms follow f(x) = f(x0)(x/x0)^γ, with γ solved so that the "
            "adjusted series reproduces the target mean annual precipitation (MAP); "
            "γ > 1 moves the annual total out of light rain into fewer, heavier storms "
            "(Lavers et al. 2022; Wu et al. 2024).",
            "Pass 2, within-storm shape: working down a ladder of calibration durations, "
            "the peak window of each storm at that duration is multiplied by α and the "
            "rest of its parent window by β, so the parent total is unchanged (mass is "
            "conserved, dry steps stay dry, timing is kept). The coefficients of each "
            "rung are fitted so that the adjusted series' implied DDF matches the "
            "reference at that duration.",
            "Durations between the rungs and longer than the anchor are not fitted; "
            "they are the independent validation. The ratio F of implied to reference "
            "depth is reported before and after, with 5-95 % intervals from "
            f"{_txt(md.get('n_bootstrap'))} bootstrap resamples of whole years (Efron "
            "and Tibshirani 1993).",
        ]
        + (
            [
                f"Stochastic ensemble: {ap.get('ensemble_n_realisations')} realisations "
                "multiply each storm's α at every rung by an independent mean-one "
                f"lognormal factor (σ = {_f(ap.get('ensemble_sigma'), 2)}); the "
                "coefficients are re-fitted so that the ensemble-median implied DDF "
                "matches the reference."
            ]
            if ap.get("ensemble_n_realisations")
            else []
        )
        + (
            [
                "Storm library: normalised patterns of the adjusted storms with their "
                "AEP band, shape (Early / Middle / Late) and an embedded-burst flag "
                "(Ball et al. 2019), for Monte Carlo design storms."
            ]
            if md.get("n_library_patterns")
            else []
        )
    ):
        d.numbered(step)

    # ------------------------------------------------------------ inputs
    d.heading("Inputs and settings", 1)
    conversions = "; ".join(
        x
        for x in (
            md.get("reference_fixed_day_conversions"),
            (
                f"dropped: {md.get('reference_durations_dropped')}"
                if md.get("reference_durations_dropped")
                else ""
            ),
        )
        if x
    )
    rows = [
        ["Rainfall series", _txt(inputs.get("series"))],
        ["Site", _txt(site)],
        ["Record analysed", f"{md.get('record_start')} to {md.get('record_end')}"],
        ["Native time step (min)", _f(md.get("native_step_min"), 0)],
        ["Complete years used", _txt(md.get("n_years_used"))],
        ["Years left out (completeness)", _list(md.get("years_excluded")) if md.get("years_excluded") else "none"],
        ["Minimum year completeness (-)", _f(md.get("min_completeness"), 2)],
        ["Time-zone offset applied (h)", _f(md.get("timezone_offset_h"), 1)],
        ["Reference DDF", _txt(inputs.get("ddf"))],
        ["Reference tier", _txt(md.get("reference_tier"))],
        ["Reference already areal (ARF applied)", _txt(md.get("reference_is_areal"))],
        ["Reference durations (min)", _list(md.get("reference_durations_min"))],
        ["Reference return periods (years)", _list(md.get("reference_return_periods_yr"))],
        ["Reference conversions", conversions or "none"],
        ["Target MAP source", _txt(mi.get("map_target_source"))],
        ["Anchor duration (h)", _f(anchor_h, 0)],
        ["Calibration durations (min)", _list(ap.get("calibration_durations_min"))],
        ["Validation durations (min)", _list(ap.get("validation_durations_min"))],
        ["Peak concentration varies with return period", _txt(md.get("rarity_dependent"))],
        ["Distribution / fitting", _txt(md.get("distribution"))],
        ["Fixed-interval correction", _txt(md.get("fixed_interval_correction"))],
        ["IETD (h)", _f(md.get("ietd_hours"), 1)],
        ["Minimum storm depth (mm)", _f(md.get("min_event_depth_mm"), 1)],
        ["Tolerance |F - 1| (%)", _f(md.get("tolerance_pct"), 0)],
        ["Bootstrap replicates", _txt(md.get("n_bootstrap"))],
        ["Ensemble realisations / σ", f"{_txt(ap.get('ensemble_n_realisations') or 0)} / "
         f"{_f(ap.get('ensemble_sigma'), 2) if ap.get('ensemble_n_realisations') else '-'}"],
        ["Random seed", _txt(md.get("random_seed"))],
    ] + [[a, _txt(b)] for a, b in context.get("settings", [])]  # fmt: skip
    d.caption("Table", "Inputs and settings")
    d.table(["Item", "Value"], rows, [7.0, 9.5])

    # ------------------------------------------------------------ parameters
    d.heading("Parameters and calculations", 1)
    d.caption("Table", "Parameters and definitions")
    d.table(
        ["Parameter", "Value", "Definition / equation (source)"],
        [
            ["Weiss factor W (-)", "1/(1 - 1/(8n))",
             "True to fixed-interval maximum for an n-step window (Weiss 1964)"],
            ["Ratio F (-)", "implied / reference",
             "DDF depth implied by the series over the reference depth"],
            ["Product distribution G", _txt(md.get("distribution")),
             "AMS fit of the product at the anchor duration (Hosking and Wallis 1997)"],
            ["Transfer start x0 (mm)", _f(mi.get("transfer_x0_mm"), 1),
             f"Product anchor depth at T0 = {_f(mi.get('transfer_t0_yr'), 1)} years"],
            ["Bulk exponent γ (-)", _f(mi.get("transfer_gamma"), 3),
             "f(x) = f(x0)(x/x0)^γ below x0; solved for the target MAP"],
            ["Peak factor α (-)", "exp(a + c(ln T - ln 10))",
             "Pass-2 multiplier of the peak window; a, c fitted per rung"],
            ["Remainder factor β (-)", "(P - αC)/(P - C)",
             "Keeps the parent window total P (C = peak window depth)"],
            ["Tolerance (%)", _f(md.get("tolerance_pct"), 0),
             "Consistent when |F - 1| is within the tolerance"],
            ["Bootstrap interval", "5-95 %",
             f"{_txt(md.get('n_bootstrap'))} resamples of whole years (Efron and Tibshirani 1993)"],
        ],  # fmt: skip
        [4.6, 3.4, 8.5],
    )
    d.para(
        "Verdicts: 'consistent' when the median F is within the tolerance of 1, "
        "'inconclusive' when outside it but the interval still includes 1, and "
        "'flatter' or 'peakier' otherwise."
    )

    # ------------------------------------------------------------ results
    d.heading("Results", 1)
    d.heading("Mean annual precipitation and rainfall frequency", 2)
    d.caption("Table", "Mean annual precipitation and rainfall frequency")
    d.table(
        ["Item", "Value"],
        [
            ["MAP of the input series, complete years (mm)", _f(mi.get("map_input_mm"), 1)],
            ["Target MAP (mm)", f"{_f(mi.get('map_target_mm'), 1)} "
             f"(source: {mi.get('map_target_source', '-')})"],
            ["MAP of the adjusted series (mm)", _f(mi.get("map_output_mm"), 1)],
            ["Bulk exponent γ (-)", _f(mi.get("transfer_gamma"), 3)],
            ["Transfer start x0 (mm)", _f(mi.get("transfer_x0_mm"), 1)],
            [f"Wet steps per year (> {_f(mi.get('wet_step_threshold_mm'), 2)} mm), "
             "input / adjusted",
             f"{_f(mi.get('wet_steps_per_year_input'), 0)} / "
             f"{_f(mi.get('wet_steps_per_year_adjusted'), 0)}"],
        ],  # fmt: skip
        [9.5, 7.0],
    )
    if result.monthly_rows:
        d.caption("Table", "Mean monthly rainfall, input and adjusted (seasonal check)")
        d.table(
            ["Month", "Input (mm)", "Adjusted (mm)", "Ratio (-)"],
            [
                [
                    MONTHS[r["month"] - 1],
                    _f(r["input_mean_mm"], 1),
                    _f(r["adjusted_mean_mm"], 1),
                    _f(r["ratio"], 2),
                ]
                for r in result.monthly_rows
            ],
            [4.0, 4.2, 4.2, 4.1],
        )

    d.heading("Consistency with the reference DDF", 2)
    stream = chart
    if stream is None and chart_path:
        try:
            with open(chart_path, "rb") as fh:
                import io

                stream = io.BytesIO(fh.read())
        except OSError as exc:
            notes.append(f"Chart file could not be read ({exc}).")
    if stream is None:
        try:
            from .export import flatness_figure

            stream, style_notes = flatness_figure(result)
            notes.extend(style_notes)
        except ImportError:
            notes.append("matplotlib is not available: the chart was not drawn.")
        except Exception as exc:  # noqa: BLE001 - the report must still be written
            notes.append(f"The chart could not be drawn ({exc}).")
    if stream is not None:
        caption = (
            "Ratio F of implied to reference DDF depth against duration, before "
            "(left) and after (right) adjustment; shaded bands are the 5-95 % "
            "bootstrap intervals and the grey band the tolerance"
        )
        if tier3:
            caption += f" ({result.status_stamp})"
        try:
            d.picture(stream, caption)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"The chart could not be added ({exc}).")

    aris = sorted({r["return_period_yr"] for r in result.after_rows})
    durs = sorted({r["duration_min"] for r in result.after_rows})
    look = {(r["duration_min"], r["return_period_yr"]): r for r in result.after_rows}
    if aris and durs:
        rp_head = [f"1:{t:g} (-)" for t in aris]
        rows = []
        for dur in durs:
            first = look[(dur, aris[0])]
            rows.append(
                [first["duration"], _role(first["role"])]
                + [
                    f"{_f(look[(dur, t)]['F_before'])} / {_f(look[(dur, t)]['F_after'])}"
                    for t in aris
                ]
            )
        d.caption(
            "Table",
            "Ratio F (implied / reference), before / after adjustment, by duration "
            "and return period (fitted = calibration rung, check = validation)",
        )
        d.table(
            ["Duration", "Role"] + rp_head,
            rows,
            [2.0, 1.6] + [(TEXT_WIDTH_CM - 3.6) / len(aris)] * len(aris),
        )
        rows = [
            [look[(dur, aris[0])]["duration"]]
            + [
                f"{_f(look[(dur, t)]['reference_depth_mm'], 1)} / "
                f"{_f(look[(dur, t)]['adjusted_implied_depth_mm'], 1)}"
                for t in aris
            ]
            for dur in durs
        ]
        d.caption("Table", "Depth (mm), reference / adjusted series")
        d.table(
            ["Duration"] + [f"1:{t:g} (mm)" for t in aris],
            rows,
            _even(len(aris) + 1, 2.4),
        )

    rung_rows = []
    for k in sorted(ap):
        if k.startswith("rung_") and k.endswith("_a"):
            name = k[len("rung_") : -2]
            rung_rows.append(
                [
                    name,
                    _f(ap[k], 3),
                    _f(ap.get(f"rung_{name}_c"), 3),
                    _f(ap.get(f"rung_{name}_rmse_log"), 4),
                ]
            )
    if rung_rows:
        d.caption("Table", "Fitted pass-2 coefficients per calibration rung")
        d.table(["Rung", "a (-)", "c (-)", "RMSE of ln depth (-)"], rung_rows,
                [4.0, 4.0, 4.0, 4.5])  # fmt: skip

    if result.ensemble_rows and aris:
        el = {
            (r["duration_min"], r["return_period_yr"]): r for r in result.ensemble_rows
        }
        erows = [
            [look[(dur, aris[0])]["duration"]]
            + [
                (
                    f"{_f(el[(dur, t)]['F_ensemble_p05'])} - {_f(el[(dur, t)]['F_ensemble_p95'])}"
                    if (dur, t) in el
                    else "-"
                )
                for t in aris
            ]
            for dur in durs
        ]
        d.caption("Table", "Ensemble spread of F (5-95 % across realisations)")
        d.table(
            ["Duration"] + [f"1:{t:g} (-)" for t in aris],
            erows,
            _even(len(aris) + 1, 2.4),
        )

    d.heading("Diagnostics", 2)
    diag = [
        ["Storms identified", _txt(md.get("n_events"))],
        ["Storms touching missing data (not reshaped)", _txt(md.get("n_events_touching_missing"))],
        ["Storms in the bulk segment (below T0)", _txt(md.get("n_events_bulk_segment"))],
        ["Storms beyond the reference range (extrapolated)",
         _txt(md.get("n_events_beyond_reference_range"))],
        ["Storms reshaped in pass 2", _txt(ap.get("n_storms_adjusted"))],
        ["Storm library patterns", _txt(md.get("n_library_patterns"))],
    ]  # fmt: skip
    d.caption("Table", "Diagnostics")
    d.table(["Item", "Value"], diag, [9.5, 7.0])

    # ------------------------------------------------------------ outputs
    outputs = context.get("outputs") or []
    if outputs:
        d.heading("Outputs", 1)
        d.caption("Table", "Files written")
        d.table(["Output", "File"], [[a, b] for a, b in outputs], [5.0, 11.5])

    # ------------------------------------------------------------ notes
    d.heading("Notes and limitations", 1)
    for txt in (
        "The adjusted series is a statistical reconstruction for design and "
        "continuous-simulation use. It reproduces the reference DDF in the "
        "annual-maximum sense; individual storms are not claimed to be the true "
        "historical point rainfall.",
        "No detail finer than the input's native time step is created.",
        "The target MAP should describe the same period as the series. A long-term "
        "MAP from a different period (e.g. the gauge period behind a published DDF) "
        "embeds any trend between the two periods.",
        "The reference DDF is taken as correct. Its own uncertainty (e.g. the Design "
        "Rainfall lower and upper bounds, given in the consistency CSV) is not "
        "propagated into the adjustment.",
        "Part of the pre-adjustment deficit is the legitimate point-versus-area "
        "difference of a grid cell. If the adjusted series is to represent a "
        "catchment rather than a point, supply an areally reduced reference DDF.",
        "Validation durations are an internal consistency check; independent gauge "
        "records, where available, remain the preferred verification.",
    ):
        d.bullet(txt)
    msgs = list(
        dict.fromkeys(list(result.warnings) + list(context.get("warnings", [])) + notes)
    )
    if msgs:
        d.label("Processing messages")
        for m in msgs:
            d.bullet(m)

    d.heading("References", 1)
    for ref in REFERENCES:
        d.para(ref)
    d.doc.save(str(path))
    return notes
