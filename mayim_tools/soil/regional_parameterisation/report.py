"""Word (.docx) report for Regional Soil Parameterisation (no QGIS), in the
Mayim report and figure standard: house text styles (Doc(house_style=True)),
16.5 cm figures in the chart style, and the default annexure structure
(summary, tool and method, inputs and settings, parameters and calculations,
results, outputs, notes and limitations, references).

Needs python-docx and matplotlib (imported lazily by the shared helpers).
Figures use matplotlib's object API (no pyplot), which is safe on QGIS's
Processing worker thread. The USDA texture layout maps are rendered by QGIS
when the wrapper passes ``map_fn`` (texture_map.py); otherwise they are drawn
with matplotlib. A figure that fails is skipped with a note, so the report
itself is always written.
"""

from __future__ import annotations

import math
import os

import numpy as np

from mayim_tools._common.docx_report import (
    Doc,
)
from mayim_tools._common.report_maps import (
    OSM_LICENCE_NOTE,
    PLAN_DPI,
    PLAN_PLOT_WIDTH_CM,
    figure_rendered_map,
)

from . import figures
from . import fill as fill_mod
from .core import (
    OUT_METADATA_CSV,
    OUT_REPORT,
    TOOL_NAME,
    TOOL_VERSION,
    pair_label,
    product_pairs,
)
from .design import DEPTH_DESCRIPTION
from .ptf import (
    METHOD_BY_CODE,
    PARAMETER_BY_CODE,
    PARAMETERS,
    rawls_1983,
    saxton_rawls,
    toth2015,
)
from .texture import (
    CLASS_NAME,
    ISO_TO_USDA_SILT_FACTOR,
    TEXTURE_COLOURS,
    class_label,
)

REPORT_TITLE = "Regional Soil Parameterisation"
KEY_PARAMS = ("theta_s", "theta_fc", "theta_wp", "paw", "ksat", "psi_f")
SOURCE_COLOURS = {
    fill_mod.SOURCE_OWN: "#00CC00",
    fill_mod.SOURCE_OTHER: "#FFC000",
    fill_mod.SOURCE_SG2017: "#0000FF",
    fill_mod.SOURCE_NEIGHBOUR: "#FF00FF",
}
CONF_COLOURS = {1: "#00CC00", 2: "#FFFF00", 3: "#FFC000", 4: "#FF0000"}
KSAT_SHORT = {1: "x/÷ ≤ 2", 2: "x/÷ 2-4", 3: "x/÷ 4-10", 4: "x/÷ > 10"}
TEX_SHORT = {1: "≥ 80 %", 2: "60-80 %", 3: "40-60 %", 4: "< 40 %"}

REFERENCES = [
    "Aitchison, J. (1986). The Statistical Analysis of Compositional Data. "
    "Chapman and Hall, London.",
    "Bouwer, H. (1966). Rapid field measurement of air entry value and "
    "hydraulic conductivity of soil as significant parameters in flow system "
    "analysis. Water Resources Research 2(4): 729-738.",
    "Chow, V.T., Maidment, D.R. and Mays, L.W. (1988). Applied Hydrology. "
    "McGraw-Hill, New York.",
    "Fatichi, S., Or, D., Walko, R., et al. (2020). Soil structure is an "
    "important omission in Earth System Models. Nature Communications 11: 522.",
    "Hengl, T. and Gupta, S. (2019). Soil water content (volumetric %) for "
    "33 kPa and 1500 kPa suctions predicted at 6 standard depths at 250 m. "
    "Zenodo, doi:10.5281/zenodo.2784001.",
    "Hengl, T., Consoli, D., Tian, X., et al. (2026). OpenLandMap-soildb: "
    "global soil information at 30 m spatial resolution for 2000-2022+. "
    "Earth System Science Data 18: 989.",
    "Hengl, T., Mendes de Jesus, J., Heuvelink, G.B.M., et al. (2017). "
    "SoilGrids250m: Global gridded soil information based on machine "
    "learning. PLoS ONE 12(2): e0169748.",
    "Minasny, B. and McBratney, A.B. (2001). The Australian soil texture "
    "boomerang: a comparison of the Australian and USDA/FAO soil particle-"
    "size classification systems. Australian Journal of Soil Research 39: "
    "1443-1451.",
    "Nemes, A., Wosten, J.H.M., Lilly, A. and Oude Voshaar, J.H. (1999). "
    "Evaluation of different procedures to interpolate particle-size "
    "distributions to achieve compatibility within soil databases. Geoderma "
    "90: 187-202.",
    "Poggio, L., de Sousa, L.M., Batjes, N.H., et al. (2021). SoilGrids 2.0: "
    "producing soil information for the globe with quantified spatial "
    "uncertainty. SOIL 7: 217-240.",
    "Rawls, W.J., Brakensiek, D.L. and Miller, N. (1983). Green-Ampt "
    "infiltration parameters from soils data. Journal of Hydraulic "
    "Engineering 109(1): 62-70.",
    "Rossman, L.A. and Huber, W.C. (2016). Storm Water Management Model "
    "Reference Manual Volume I - Hydrology (Revised). EPA/600/R-15/162A. US "
    "Environmental Protection Agency, Cincinnati.",
    "Rawls, W.J., Brakensiek, D.L. and Saxton, K.E. (1982). Estimation of "
    "soil water properties. Transactions of the ASAE 25(5): 1316-1320.",
    "Saxton, K.E. and Rawls, W.J. (2006). Soil water characteristic "
    "estimates by texture and organic matter for hydrologic solutions. Soil "
    "Science Society of America Journal 70: 1569-1578.",
    "Morel-Seytoux, H.J., Meyer, P.D., Nachabe, M., Touma, J., van Genuchten, "
    "M.Th. and Lenhard, R.J. (1996). Parameter equivalence for the Brooks-Corey "
    "and van Genuchten soil characteristics: preserving the effective capillary "
    "drive. Water Resources Research 32(5): 1251-1258.",
    "Simons, G.W.H., Koster, R. and Droogers, P. (2020). HiHydroSoil v2.0 - A "
    "high resolution soil map of global hydraulic properties. FutureWater "
    "Report 213, Wageningen.",
    "Tóth, B., Weynants, M., Nemes, A., Makó, A., Bilas, G. and Tóth, G. "
    "(2015). New generation of hydraulic pedotransfer functions for Europe. "
    "European Journal of Soil Science 66: 226-238.",
    "van Genuchten, M.Th. (1980). A closed-form equation for predicting the "
    "hydraulic conductivity of unsaturated soils. Soil Science Society of "
    "America Journal 44: 892-898.",
    "Soil Science Division Staff (2017). Soil Survey Manual. USDA Handbook "
    "18. Government Printing Office, Washington, D.C.",
    "Loots, I. and Smithers, J.C. (2020). Design Flood Estimation in Urban "
    "Areas in South Africa: Preliminary Results from Tshwane Case Studies. WRC "
    "Report No. 2747/1/20, Water Research Commission, Pretoria.",
]
REFERENCES.sort(key=lambda r: r.lower())

UNIT_TEXT = {"m3/m3": "m³/m³", "km2": "km²", "g/cm3": "g/cm³", "1/cm": "1/cm"}


def _u(units: str) -> str:
    """Units with proper symbols for the report (m³/m³, km², g/cm³)."""
    return UNIT_TEXT.get(units, units)


# Table 12 style labels (symbol and unit) for the design table.
DESIGN_LABEL = {
    "ksat": "Ksat (mm/h)",
    "psi_f": "Wetting-front suction head ψf (mm)",
    "theta_s": "θs (m³/m³)",
    "theta_fc": "θ33 (m³/m³)",
    "theta_wp": "θ1500 (m³/m³)",
    "paw": "PAW (m³/m³)",
    "ima_dry": "Initial deficit, dry start (m³/m³)",
    "ima_moist": "Initial deficit, moist start (m³/m³)",
    "t_dry": "Drying time T (days)",
}
DEPTH_LABEL = {
    "ima_dry": "Initial deficit, dry start (mm)",
    "ima_moist": "Initial deficit, moist start (mm)",
}


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------


def _f(v, nd=2) -> str:
    if v is None:
        return "-"
    try:
        if not math.isfinite(float(v)):
            return "-"
    except (TypeError, ValueError):
        return str(v)
    return f"{float(v):.{nd}f}"


def _sig(v, n=3) -> str:
    """Significant figures for values spanning orders of magnitude."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(v):
        return "-"
    if v == 0:
        return "0"
    digits = max(0, n - 1 - int(math.floor(math.log10(abs(v)))))
    return f"{v:.{digits}f}"


def _pct(v) -> str:
    return _f(v, 0) if v is not None else "-"


def _row(result, zone, lab, code):
    for r in result.zone_rows:
        if r["Zone"] == zone and r["Layer"] == lab and r["Parameter"] == code:
            return r
    return None


def _value_range(r, code) -> str:
    if r is None:
        return "-"
    nd = PARAMETER_BY_CODE[code].decimals
    fmt = _sig if PARAMETER_BY_CODE[code].log else (lambda v: _f(v, nd))
    return f"{fmt(r['Median of P50'])} ({fmt(r['Median cell P5'])}-{fmt(r['Median cell P95'])})"


# ----------------------------------------------------------------------
# Figures (publication style, see figures.py)
# ----------------------------------------------------------------------

_continuous_maps = figures.continuous_maps
_categorical_maps = figures.categorical_maps
_texture_triangle = figures.texture_triangle
_scatter_panels = figures.scatter_panels
_profiles = figures.profiles
_box_by_product = figures.box_by_member
_variance_bars = figures.variance_bars


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

CONF_TEXT = {
    "High": "the 90 % range is within about a factor 2 (Ksat, suctions) or "
    "±15 % (water contents), and products and methods agree",
    "Medium": "the range is within about a factor 4 or ±30 %, and products and "
    "methods differ moderately",
    "Low": "the range is wider, or products and methods disagree strongly",
}


def _fmt_design(code, v) -> str:
    prm = PARAMETER_BY_CODE[code]
    return _sig(v, 3) if prm.log else _f(v, prm.decimals)


MAX_DESIGN_FIGURES = 6


def _plain_confidence_text(result) -> str:
    """Plain-language explanation of the confidence rating for readers
    without a soils or statistics background; follows the run settings."""
    n_prod = len(result.products)
    n_meth = len(result.settings.mc.methods)
    draws = result.settings.mc.draws
    sources = []
    if n_prod > 1:
        sources.append("two different soil maps")
    if n_meth > 1:
        sources.append("two different published calculation methods")
    if sources:
        using = " It also uses " + " and ".join(sources) + "."
    else:
        using = ""
    agree = []
    if n_prod > 1:
        agree.append("the two maps")
    if n_meth > 1:
        agree.append("the two methods")
    if agree:
        second = (
            f" Second, do {' and '.join(agree)} roughly agree when each is used "
            "on its own?"
        )
        test = "If the range is narrow and they agree closely"
        medium = "If both are moderate"
    else:
        second = ""
        test = "If the range is narrow"
        medium = "If it is moderate"
    causes = ["the soil maps themselves"]
    if n_prod > 1:
        causes.append("the disagreement between the two maps")
    if n_meth > 1:
        causes.append("the choice of method")
    cause_text = (
        ", ".join(causes[:-1]) + " or " + causes[-1] if len(causes) > 1 else causes[0]
    )
    if len(causes) > 1:
        source_text = (
            f"The tool also reports which part of the uncertainty is largest: "
            f"{cause_text}."
        )
    else:
        source_text = (
            "With one map and one method, all of this uncertainty comes from "
            "the soil map itself."
        )
    return (
        "The soil maps used here are estimates, and each comes with a range of "
        "plausible values rather than a single exact number. So instead of "
        "calculating each soil parameter once, the tool calculates it "
        f"{draws} times for every map cell. Each time it picks slightly "
        "different but realistic soil properties from within the ranges the "
        f"maps give.{using} This produces a spread of possible answers for "
        "every location. The value in the middle of that spread is the "
        '"candidate value". The low and high ends, leaving out the most '
        'extreme 5 % on each side, are the "sensitivity range". Confidence is '
        "then judged with simple questions. First, how wide is that range? For "
        "example, is the highest likely value of the water infiltration rate "
        "less than about four times the lowest, or more than a hundred times?"
        f"{second} {test}, the value is rated High. {medium}, Medium. "
        f"Otherwise, Low. {source_text} That shows the best way to improve the "
        "estimate, such as soil sampling or infiltration tests on site. A Low "
        "rating does not mean the value is wrong. It means the available "
        "information cannot pin it down, so the value should be tested over "
        "its whole range or checked in the field."
    )


PROPER_NAMES = ("Green-Ampt", "Brooks-Corey", "van Genuchten", "Ksat")


def _lower_first(name: str) -> str:
    """Lower-case a label's first letter mid-sentence, except proper names
    and acronyms."""
    if name.startswith(PROPER_NAMES) or name[1:2].isupper():
        return name
    return name[0].lower() + name[1:]


def _join(items, word="and") -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def _design_description(r) -> str:
    """Description column: how the value relates to the soil parameters and
    the SWMM Green-Ampt inputs, plus its main source of uncertainty."""
    text = r.get("Description", "")
    share = r.get("Share of variance (%)")
    source = r.get("Main uncertainty source")
    if source and share is not None and math.isfinite(float(share)):
        text += f" Main uncertainty: {source} ({float(share):.0f} %)."
    return text.strip()


def _design_table_rows(result, zrows):
    """(rows, group row indices) of the design table for one zone."""
    table, groups = [], []
    for lab in [lab for lab, _, _ in result.settings.layers]:
        lrows = [r for r in zrows if r["Layer"] == lab]
        if not lrows:
            continue
        groups.append(len(table))
        table.append([f"Layer {lab}", "", "", "", "", ""])
        for r in lrows:
            code = r["Parameter"]
            table.append(
                [
                    DESIGN_LABEL.get(
                        code, f"{figures.SYMBOL.get(code, code)} ({_u(r['Units'])})"
                    ),
                    _fmt_design(code, r["Candidate value"]),
                    f"{_fmt_design(code, r['Sensitivity low (P5)'])} - "
                    f"{_fmt_design(code, r['Sensitivity high (P95)'])}",
                    f"{_fmt_design(code, r['Product x method min'])} - "
                    f"{_fmt_design(code, r['Product x method max'])}",
                    r["Confidence"],
                    _design_description(r),
                ]
            )
            if code in DEPTH_LABEL and "Candidate depth (mm)" in r:
                table.append(
                    [
                        DEPTH_LABEL[code],
                        _f(r["Candidate depth (mm)"], 1),
                        f"{_f(r['Sensitivity low depth (mm)'], 1)} - "
                        f"{_f(r['Sensitivity high depth (mm)'], 1)}",
                        f"{_f(r['Product x method min depth (mm)'], 1)} - "
                        f"{_f(r['Product x method max depth (mm)'], 1)}",
                        r["Confidence"],
                        DEPTH_DESCRIPTION.format(mm=r["Layer thickness (mm)"]),
                    ]
                )
    return table, groups


def _design_section(d, result, notes, num: str) -> None:
    """Candidate design parameters with the Green-Ampt (SWMM) values,
    confidence, recommendations, limitations and risks."""
    from .design import ACTION_TEXT, EFFECTIVE_K_FACTOR, SOURCE_TEXT

    rows = result.design_rows
    d.heading(f"{num} Candidate design parameters", 3)
    d.para(
        "The table gives one candidate value per zone, layer and parameter for "
        "the engineer to confirm. Ksat, the wetting-front suction and the "
        "drying time are areal geometric means of the cell medians (they are "
        "log-normally distributed); water contents and initial deficits are "
        "arithmetic means. The sensitivity range aggregates the cell P5 and P95 "
        "maps in the same way, i.e. as if the whole zone sat at its lower or "
        "upper bound together - a wide, deliberately cautious range for model "
        "sensitivity runs. The product x method range shows the spread between "
        "the individual ensemble members (each product's central soil run "
        "through each method). The values are estimates from global soil maps "
        "and published methods; they are not prescribed by any design standard."
    )
    d.para(
        "The Green-Ampt infiltration inputs of SWMM (Rossman & Huber, 2016) "
        "follow from the soil parameters: Conductivity = Ksat; Suction head = "
        "ψf; Initial deficit = θs − θi, the porosity less the starting water "
        "content - θ1500 for a dry start and θ33 for a moist start - given both "
        "as a fraction and as a depth of water over the layer; and Drying time "
        "T = 15.78 / √Ksat days with Ksat in mm/h (the SI form in WRC Report "
        "2747/1/20). The initial deficit and drying time are computed on every "
        "Monte Carlo draw, so their ranges are consistent with the soil "
        "parameters. The Description column relates each value to its source "
        "parameters and gives its main source of uncertainty."
    )
    if not rows:
        d.para("No candidate values could be computed.")
        return
    for zone in result.zone_names:
        zrows = [r for r in rows if r["Zone"] == zone]
        if not zrows:
            continue
        table, groups = _design_table_rows(result, zrows)
        d.caption("Table", f"{zone}: candidate design parameters")
        d.table(
            [
                "Parameter",
                "Candidate",
                "Sensitivity range (P5-P95)",
                "Product x method range",
                "Confidence",
                "Description",
            ],
            table,
            widths_cm=[2.9, 1.9, 2.3, 2.3, 2.0, 5.1],
            group_rows=groups,
        )
        ks = [
            r
            for r in zrows
            if r["Parameter"] == "ksat" and "Effective K (0.5 Ksat)" in r
        ]
        if ks:
            d.para(
                "Effective conductivity for Green-Ampt models that use a wetted-zone "
                f"K (about {EFFECTIVE_K_FACTOR:g} x Ksat; Bouwer, 1966): "
                + "; ".join(
                    f"{r['Layer']} {_sig(r['Effective K (0.5 Ksat)'], 3)} mm/h"
                    for r in ks
                )
                + ". Use either Ksat or the effective K, as the model expects - not both."
            )
        if result.zone_names.index(zone) < MAX_DESIGN_FIGURES:
            _figure(
                d,
                lambda zone=zone: figures.design_forest(result, zone),
                f"Candidate design values for {zone}: candidate (diamond, "
                "coloured by confidence), sensitivity range (line) and each "
                "product x method member",
                notes,
            )
    if len(result.zone_names) > MAX_DESIGN_FIGURES:
        d.para(
            f"Design figures are drawn for the first {MAX_DESIGN_FIGURES} zones; "
            "all zones are in the tables and in the design-parameters CSV."
        )
    d.label("Confidence")
    d.para(
        "Each candidate is rated High, Medium or Low from two numbers: the "
        "width of its sensitivity range and the spread between the ensemble "
        "members. For Ksat, the suctions and the drying time the range is "
        "measured as the factor U = √(P95/P5) and the spread as the ratio of the "
        "largest to the smallest member; for water contents and initial "
        "deficits both are measured relative to the candidate."
    )
    d.caption("Table", "Confidence ratings")
    d.table(
        ["Confidence", "Meaning"],
        [[k, v] for k, v in CONF_TEXT.items()],
        widths_cm=[2.5, 14.0],
    )
    counts = {lvl: sum(1 for r in rows if r["Confidence"] == lvl) for lvl in CONF_TEXT}
    d.para(
        f"Of {len(rows)} candidate values, {counts['High']} are rated High, "
        f"{counts['Medium']} Medium and {counts['Low']} Low."
    )

    d.label("Recommendations")
    by_param = {}
    for r in rows:
        by_param.setdefault(r["Parameter"], []).append(r)
    groups = {}  # (levels, source) -> parameter names, in table order
    for code, prs in by_param.items():
        levels = tuple(
            sorted({r["Confidence"] for r in prs}, key=["High", "Medium", "Low"].index)
        )
        sources = {}
        for r in prs:
            src = r["Main uncertainty source"]
            sources[src] = sources.get(src, 0) + 1
        source = max(sources, key=sources.get)
        key = ("High",) if levels == ("High",) else (levels, source)
        groups.setdefault(key, []).append(PARAMETER_BY_CODE[code].label)
    for key, names in groups.items():
        listed = _join([names[0]] + [_lower_first(n) for n in names[1:]])
        if key == ("High",):
            d.bullet(f"{listed}: high confidence; use the candidate values.")
            continue
        levels, source = key
        verb = "is" if len(names) == 1 else "are"
        d.bullet(
            f"{listed} {verb} rated {_join([lvl.lower() for lvl in levels], 'or')}, "
            f"mainly because of {SOURCE_TEXT[source]}. Run the model at both "
            f"sensitivity bounds; the range is best reduced by {ACTION_TEXT[source]}."
        )
    d.bullet(
        "Choose between products and methods deliberately when their values "
        "differ (product x method range); state the choice and the reason in "
        "the design report."
    )
    d.bullet(
        "Choose the initial deficit for the antecedent condition of the design "
        "event: the dry-start value suits a storm after a long dry spell, the "
        "moist-start value a storm on soil near field capacity. The drying time "
        "governs how quickly the deficit recovers between storms in continuous "
        "simulation."
    )
    d.bullet(
        "Where the design standard prescribes loss parameters, use those; "
        "report these values as supporting evidence or for sensitivity testing."
    )
    d.bullet(
        "Confirm Ksat with infiltration tests (double-ring or tension "
        "infiltrometer) at representative sites before final design, and "
        "calibrate on observed runoff events where records exist."
    )

    d.label("Limitations of the candidate values")
    for text in (
        "Zone values average over the soils inside each zone; a zone with "
        "contrasting soils needs subdivision or the cell rasters.",
        "Texture-based methods describe the soil matrix. Macropores, root "
        "channels and cracks raise field infiltration; crusting, compaction and "
        "sealing lower it. Neither is represented.",
        "Built-up land is filled from older or neighbouring predictions and "
        "does not describe sealed surfaces or fill material.",
        "The initial deficits bracket the antecedent state (wilting point and "
        "field capacity); the actual starting water content of a design event "
        "lies between or outside them and is a modelling choice.",
        "The confidence rating describes the spread of the available data and "
        "methods, not their accuracy: a narrow range can still be biased.",
    ):
        d.bullet(text)

    d.label("Risks")
    for text in (
        "Flood estimation: taking Ksat from the upper end of the range, or the "
        "dry-start initial deficit, lowers runoff and can under-design "
        "drainage. For flood estimation, test the lower Ksat bound and the "
        "moist start too.",
        "Infiltration and retention design (soakaways, infiltration basins, "
        "on-site retention): taking Ksat from the upper end of the range "
        "over-estimates disposal capacity. Base such designs on site tests.",
        "Product choice: the soil products can describe different soils (see "
        "the product comparison); picking one without checking can bias every "
        "parameter in the same direction.",
        "Method choice: the methods can differ by a factor of several in the "
        "Green-Ampt suction; a single method hides this spread.",
        "Scale: a 30 m or 250 m map value is not a point measurement; field "
        "tests at a few points will scatter widely around it.",
    ):
        d.bullet(text)


def _short_source(text: str) -> str:
    if text.startswith("OpenLandMap"):
        return "OpenLandMap 250 m"
    if text.startswith("SoilGrids"):
        return "SoilGrids wv"
    return text


def core_density_warning() -> str:
    from .core import DENSITY_WARNING

    return DENSITY_WARNING


def _product_medians_table(d, result, lab) -> None:
    """Each product's central-soil medians next to the pooled ensemble, so
    product disagreement is visible beside the pooled range."""
    zone = result.zone_names[0]
    cols = []
    for p in result.products:
        for m in result.settings.mc.methods:
            cols.append(
                (
                    f"{SHORT_PRODUCT.get(p.name, p.name)} ({m})",
                    f"{SHORT_CODE.get(p.name, p.name)} central ({m})",
                )
            )
    if not cols:
        return
    rows = []
    for code in KEY_PARAMS:
        r = _row(result, zone, lab, code)
        if r is None:
            continue
        prm = PARAMETER_BY_CODE[code]
        fmt = (
            (lambda v: _sig(v, 3))
            if prm.log
            else (lambda v, nd=prm.decimals: _f(v, nd))
        )
        rows.append(
            [f"{figures.SYMBOL.get(code, code)} ({_u(prm.units)})"]
            + [fmt(r.get(key)) for _, key in cols]
            + [fmt(r["Median of P50"])]
        )
    d.caption(
        "Table",
        f"{zone}, {lab}: median of each product's central soil and of the pooled "
        "ensemble",
    )
    w = 13.5 / (len(cols) + 1)
    d.table(
        ["Parameter"] + [c for c, _ in cols] + ["Pooled P50"],
        rows,
        widths_cm=[3.0] + [w] * (len(cols) + 1),
    )


SHORT_PRODUCT = {
    "SoilGrids 2.0": "SoilGrids",
    "OpenLandMap-soildb": "OpenLandMap",
    "iSDAsoil": "iSDAsoil",
}
SHORT_CODE = {"SoilGrids 2.0": "SG", "OpenLandMap-soildb": "OLM", "iSDAsoil": "iSDA"}


def _and(items) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _figure(d, builder, caption, notes):
    try:
        d.picture(builder(), caption)
    except Exception as exc:  # noqa: BLE001 - a figure must never stop the report
        notes.append(f"Figure '{caption}' could not be drawn ({exc}).")


def _texture_map_figure(d, result, lab, map_fn, notes, osm_used) -> None:
    """USDA texture layout map of one layer: rendered by QGIS with the
    layer style when ``map_fn`` is given, otherwise drawn with matplotlib."""
    caption = f"USDA soil texture class, {lab} (mode of the Monte Carlo draws)"
    overlays = list(getattr(map_fn, "overlays", []) or [])
    if map_fn is not None:
        caption = caption[:-1] + ", 70 % transparent)"
    if overlays:
        caption += ", with " + " and ".join(overlays)
    elif result.settings.zones:
        caption += ", with the zones as the study area"
    if map_fn is not None:
        try:
            pm = map_fn(result, lab, PLAN_PLOT_WIDTH_CM, PLAN_DPI)
        except Exception as exc:  # noqa: BLE001 - the report must still be written
            pm = None
            notes.append(
                f"Texture map {lab}: the QGIS layout map could not be rendered "
                f"({exc}); a simplified map was drawn instead."
            )
        if pm is not None:
            try:
                stream = figure_rendered_map(pm, figures.chart_style())
                if pm.attribution:
                    caption += " (background: OpenStreetMap)"
                    osm_used.append(True)
                d.picture(stream, caption)
                return
            except Exception as exc:  # noqa: BLE001
                notes.append(f"Texture map {lab} could not be composed ({exc}).")
    labels = {c: class_label(c) for c in TEXTURE_COLOURS}
    _figure(
        d,
        lambda: _categorical_maps(
            result,
            [(result.texture_class[lab], lab)],
            TEXTURE_COLOURS,
            labels,
            height_cm=11.0,
        ),
        caption,
        notes,
    )


def _summary_table(result, labs, top, area) -> list[list[str]]:
    s = result.settings
    g = result.grid
    rows = [
        [
            "Study area",
            f"{len(result.zone_names)} zone{'s' if len(result.zone_names) != 1 else ''}"
            f" ({', '.join(result.zone_names)}), {_f(area, 3)} km²",
        ],
        ["Depth layers", ", ".join(labs)],
        ["Soil products", ", ".join(p.name for p in result.products)],
        [
            "Methods",
            ", ".join(METHOD_BY_CODE[m].name for m in s.mc.methods),
        ],
        ["Monte Carlo draws", f"{s.mc.draws} per cell and product"],
        [
            "Processing grid",
            f"{g.width} x {g.height} cells at {_sig(g.res, 3)} map units",
        ],
    ]
    zone = result.zone_names[0]
    for t in result.zone_texture:
        if t["Layer"] == top and t["Zone"] == zone:
            rows.append(
                [
                    f"Dominant USDA texture ({zone}, {top})",
                    f"{t['Dominant class']} ({_pct(t['Dominant share (%)'])} % of cells)",
                ]
            )
    for code in ("ksat", "psi_f", "ima_dry", "ima_moist", "t_dry"):
        r = next(
            (
                x
                for x in result.design_rows
                if x["Zone"] == zone and x["Layer"] == top and x["Parameter"] == code
            ),
            None,
        )
        if r is None:
            continue
        value = (
            f"{_fmt_design(code, r['Candidate value'])} "
            f"({_fmt_design(code, r['Sensitivity low (P5)'])} - "
            f"{_fmt_design(code, r['Sensitivity high (P95)'])}); "
            f"confidence {r['Confidence'].lower()}"
        )
        if code in DEPTH_LABEL and "Candidate depth (mm)" in r:
            value = (
                f"{_fmt_design(code, r['Candidate value'])} "
                f"({_f(r['Candidate depth (mm)'], 1)} mm); confidence "
                f"{r['Confidence'].lower()}"
            )
        rows.append([f"{DESIGN_LABEL[code]}, {zone}, {top}", value])
    rows.append(["Warnings", str(len(result.warnings)) if result.warnings else "none"])
    return rows


def write_report(result, path: str, map_fn=None) -> None:
    d = Doc(house_style=True)
    s = result.settings
    mc = s.mc
    g = result.grid
    notes: list[str] = []
    osm_used: list[bool] = []
    labs = [lab for lab, _, _ in s.layers if lab in result.texture_class]
    top = labs[0] if labs else s.layers[0][0]
    products = result.products
    names = [p.name for p in products]
    both = len(products) >= 2
    has_isda = "iSDAsoil" in names
    methods = [METHOD_BY_CODE[m] for m in mc.methods]
    area = float((result.zone_raster > 0).sum()) * result.cell_area_km2
    tex_style = figures.chart_style()
    notes += tex_style.warnings

    d.heading(REPORT_TITLE, 1)

    # 1. Summary ---------------------------------------------------------
    d.heading("1. Summary", 2)
    d.para(
        "This annexure documents hydraulic soil parameters estimated for "
        f"{len(result.zone_names)} zone{'s' if len(result.zone_names) != 1 else ''} "
        f"covering {_f(area, 3)} km², for the depth layers "
        f"{', '.join(labs)}. The inputs are soil maps from "
        f"{_and(names)}, harmonised to these layers. Parameters were "
        f"estimated with {', '.join(m.name for m in methods)} and their "
        f"uncertainty was propagated by Monte Carlo simulation ({mc.draws} draws "
        f"per cell and product). The USDA texture class, candidate design "
        "values per zone - including the Green-Ampt (SWMM) initial deficit and "
        "drying time - and a confidence rating are reported. Rawls, Brakensiek "
        "& Miller (1983) texture-class values are used as an independent "
        "reference check. The tool is guideline-neutral: the values are "
        "estimates from published methods, and the user decides how they are "
        "applied under the applicable design standard."
    )
    d.caption("Table", "Summary")
    d.table(
        ["Item", "Value"],
        _summary_table(result, labs, top, area),
        widths_cm=[6.2, 10.3],
    )
    d.caption(
        "Table",
        f"Key parameters for {top}: median of the cell medians "
        "(median cell P5-P95 in brackets)",
    )
    hdr = ["Zone"] + [
        f"{figures.SYMBOL.get(c, c)} ({_u(PARAMETER_BY_CODE[c].units)})"
        for c in KEY_PARAMS
    ]
    rows = []
    for zone in result.zone_names:
        rows.append(
            [zone] + [_value_range(_row(result, zone, top, c), c) for c in KEY_PARAMS]
        )
    d.table(hdr, rows, widths_cm=[2.4] + [14.1 / len(KEY_PARAMS)] * len(KEY_PARAMS))
    _product_medians_table(d, result, top)
    d.label("Confidence")
    for t in result.zone_texture:
        if t["Layer"] != top:
            continue
        d.bullet(
            f"{t['Zone']} ({top}): dominant texture class {t['Dominant class']} "
            f"({_pct(t['Dominant share (%)'])} % of cells). The median cell's Ksat "
            f"is uncertain by a factor of {_sig(t['Median Ksat factor (x/÷)'], 2)} "
            "either way (90 % range = median x/÷ that factor); "
            f"{_pct(t['Ksat class 1 (%)'])} % of cells are within x/÷2 and "
            f"{_pct(t['Ksat class 4 (%)'])} % are uncertain by more than x/÷10. "
            f"The texture class holds in at least 80 % of the draws in "
            f"{_pct(t['Texture confidence 1 (%)'])} % of cells."
        )
    d.label("Main caveats")
    d.bullet(
        "Pedotransfer functions describe the soil matrix only. Soil structure, "
        "macropores, crusts and compaction are not represented; field-saturated "
        "conductivity under natural vegetation is often several times higher, "
        "and lower on compacted or crusted soils (Fatichi et al., 2020)."
    )
    d.bullet(
        "Global soil maps have modest skill at site level; the input "
        "uncertainty is propagated, but the ensemble spread is a lower bound "
        "of the true uncertainty. Local measurements override these values."
    )
    for c in result.comparison:
        if c["Layer"] == top:
            d.bullet(
                f"Products {c['Pair']} ({top}): mean clay difference "
                f"{_f(c['Mean clay difference B - A (%)'], 1)} %, sand "
                f"{_f(c['Mean sand difference B - A (%)'], 1)} %; same texture "
                f"class in {_pct(c['Texture class agreement (%)'])} % of cells. The "
                "disagreement is part of the reported uncertainty."
            )
    for r in result.reference_rows:
        if r["Layer"] == top and r["Zone"] == result.zone_names[0]:
            d.bullet(
                f"Against the Rawls et al. (1983) class values ({top}), the median "
                f"Ksat ratio (tool / Rawls K) is {_sig(r['Median Ksat ratio (tool/Rawls)'], 2)} "
                f"and the median suction ratio {_sig(r['Median psi_f ratio (tool/Rawls)'], 2)}. "
                "Rawls K is a Green-Ampt conductivity, about half of Ksat."
            )
    filled = []
    for p in products:
        src = result.sources.get(p.name, {}).get(top)
        if src is None:
            continue
        inside = result.zone_raster > 0
        n = int((src[inside] > 0).sum())
        if n:
            f = 100.0 * (src[inside] > fill_mod.SOURCE_OWN).sum() / n
            if f >= 1:
                filled.append(f"{p.name} {_f(f, 0)} %")
    if filled:
        d.bullet(
            "Share of cells gap-filled (not the product's own prediction): "
            + "; ".join(filled)
            + ". See section 2.4."
        )

    # 2. Tool and method ---------------------------------------------------
    d.heading("2. Tool and method", 2)
    d.para(
        "The tool turns global soil property maps into the soil hydraulic "
        "parameters needed for infiltration and soil-water modelling. Every "
        "value is reported as a median with a 90 % range (P5-P95), and the "
        "sources of uncertainty are separated. The steps are:"
    )
    for step in (
        "Read each soil product's extraction folder and select the bands by "
        "their descriptions.",
        "Average every input to the target depth layers (thickness weighting "
        "of intervals, trapezoidal rule between depth points).",
        "Convert texture to USDA particle-size limits and estimate organic "
        "matter from organic carbon.",
        "Fill gaps from the other product, then SoilGrids 2017, then the "
        "nearest valid cells; map the source of every cell.",
        "Draw soils per cell from each product's uncertainty (Monte Carlo) and "
        "run every selected pedotransfer method on every draw.",
        "Derive the Green-Ampt (SWMM) initial deficit and drying time on every "
        "draw from the water contents and Ksat.",
        "Pool the draws into P5, P50 and P95 per cell, the USDA texture class "
        "(mode of the draws) and the confidence classes.",
        "Summarise per zone, split the variance by source, and derive one "
        "candidate design value per zone, layer and parameter with a "
        "sensitivity range and a confidence rating.",
        "Check against mapped water contents and the Rawls et al. (1983) "
        "class values, and write the selected outputs.",
    ):
        d.numbered(step)
    d.heading("2.1 Depth harmonisation", 3)
    d.para(
        "All inputs are averaged to the target layers. Interval products "
        "(SoilGrids 2.0: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm; "
        "OpenLandMap: 0-30, 30-60, 60-100 cm"
        + ("; iSDAsoil: 0-20, 20-50 cm" if has_isda else "")
        + ") are combined by thickness weighting:"
    )
    d.para(
        "x(L) = Σ w_i x_i,   w_i = overlap of interval i with layer L / thickness of L"
    )
    d.para(
        "Depth-point products (SoilGrids 2017 at 0, 5, 15, 30, 60, 100 and "
        "200 cm; OpenLandMap water contents at 0, 30, 60 and 100 cm) are "
        "averaged with the trapezoidal rule between the points that bound the "
        "layer:"
    )
    d.para("x(L) = Σ (z_k+1 - z_k)(x_k + x_k+1) / 2 / (bottom - top)")
    if has_isda:
        d.para(
            "iSDAsoil is mapped to 50 cm only. Its 20-50 cm value stands in for "
            "50-60 cm in the 30-60 cm layer, and it takes no part in the 60-100 "
            "cm layer, which rests on the other product(s). It is not gap-filled "
            "there, so it never enters that layer as a copy of another product."
        )
    d.para(
        "Quantiles (SoilGrids Q0.05, Q0.95; OpenLandMap P16, P84) are averaged "
        "in the same way. This treats the intervals as fully correlated with "
        "depth, which keeps the layer uncertainty conservative (not reduced "
        "by averaging)."
    )
    d.heading("2.2 Texture systems", 3)
    d.para(
        "The pedotransfer functions use USDA particle-size limits (clay < 2 "
        "µm, silt 2-50 µm, sand 50 µm-2 mm), as do SoilGrids and iSDAsoil. "
        "OpenLandMap uses ISO 11277 limits (silt 2-63 µm). OpenLandMap texture "
        "is converted by log-linear interpolation of the cumulative "
        "particle-size distribution between 2 and 63 µm (Nemes et al., 1999; "
        "Minasny & McBratney, 2001):"
    )
    d.para(
        f"silt_USDA = k · silt_ISO,  k = ln(50/2) / ln(63/2) = {ISO_TO_USDA_SILT_FACTOR:.4f};  "
        "clay unchanged;  sand_USDA = 100 - clay - silt_USDA"
    )
    d.para(
        "About 7 % of the ISO silt is moved to sand. The conversion is applied "
        "to every Monte Carlo draw. Its own error (the true shape of the "
        "distribution between 50 and 63 µm) is not included in the uncertainty. "
        "The USDA texture class of a cell is the class most of its draws fall "
        "in (Soil Science Division Staff, 2017)."
    )
    d.heading("2.3 Organic matter", 3)
    d.para(
        f"Organic matter is estimated from soil organic carbon: OM (%) = SOC "
        f"(g/kg) / 10 x {mc.om_factor:g}. The van Bemmelen factor 1.724 assumes "
        "58 % carbon in organic matter; values of 1.9-2.0 are often more "
        "realistic for surface soils, so the factor is a stated choice."
    )
    d.heading("2.4 Gap filling", 3)
    d.para(
        "SoilGrids 2.0 has no predictions for built-up land, inland water, "
        "glaciers and bare surfaces (ESA CCI land cover 2015); OpenLandMap "
        "leaves deserts and permanent ice unmapped. Missing cells are filled "
        "per variable and layer in this order: (1) the other product, with its "
        "own uncertainty and texture converted between USDA and ISO limits; "
        "(2) SoilGrids 2017 means (which map built-up land), with the median "
        "uncertainty of the product's own cells in that layer; (3) the nearest "
        f"valid cells within {_f(s.fill_radius_m, 0)} m (inverse-distance "
        "weighting of the centre values and uncertainties). The source of every "
        "cell is mapped in section 3 and in the quality raster."
    )
    for n in [n for p in products for n in p.notes]:
        d.bullet(n)
    d.heading("2.5 Uncertainty", 3)
    d.para(
        "Inputs are converted to per-cell distributions on the same footing. "
        "Texture fractions and organic carbon are treated on a log scale, "
        "bulk density and coarse fragments on a linear scale. For SoilGrids "
        "the centre is the median Q0.5 and the 90 % interval Q0.05-Q0.95 gives "
        "separate lower and upper standard deviations, σ_lo = (T(Q0.5) - "
        "T(Q0.05)) / 1.645 and σ_hi = (T(Q0.95) - T(Q0.5)) / 1.645 (a split "
        "normal). For OpenLandMap the centre is the 30 m mean and the 120 m "
        "68 % interval gives σ = (T(P84) - T(P16)) / (2 x 0.994) on both sides."
        + (
            " For iSDAsoil the centre is the 30 m mean and the published standard "
            "deviation gives a 90 % interval mean ∓ 1.645 sd (for organic carbon "
            "and stone content in ln(1 + x)), converted as for SoilGrids. That "
            "standard deviation is the spread of the model's learners, not a "
            "calibrated prediction interval, and is likely narrower than the "
            "true error."
            if has_isda
            else ""
        )
    )
    d.para(
        "Monte Carlo: in each cell, each product contributes "
        f"{mc.draws} draws (random seed {mc.seed}, so a re-run gives identical "
        "results). Sand, silt and clay are drawn independently in log space "
        "and closed to 100 % - a logistic-normal distribution on the simplex "
        "(Aitchison, 1986) - so every draw is a real soil; adding marginal "
        "quantiles (e.g. Q0.05 sand with Q0.05 clay) would not be. Every "
        "method runs on every draw, and the pooled draws give the P5, P50 and "
        "P95 per cell. Products are weighted equally."
    )
    d.para(
        "Variance split (law of total variance, log10 scale for Ksat, the "
        "suctions and the drying time): input data = mean variance within each "
        "method-product group; product and method = variance of the group "
        "means across products or methods; interaction = the remainder. "
        "Correlations between inputs and between neighbouring cells are not "
        "modelled, and the products' own interval estimates are taken at face "
        "value."
    )
    d.para(
        "Confidence per cell is graded rather than pass/fail. Ksat: the "
        "uncertainty factor F = √(P95/P5), so the 90 % range is the median "
        "multiplied or divided by F; classes F ≤ 2 (the range stays within one "
        "NRCS Ksat class, which are about a factor 4 apart), 2-4, 4-10 and > 10. "
        "Texture: the share of the draws that fall in the most frequent USDA "
        "class; classes ≥ 80 %, 60-80 %, 40-60 % and < 40 %."
    )
    if mc.density:
        d.para(core_density_warning(), bold_lead="Density adjustment:")
    d.heading("2.6 How confidence is judged", 3)
    d.para(_plain_confidence_text(result))

    # 3. Inputs and settings ------------------------------------------------
    d.heading("3. Inputs and settings", 2)
    d.caption("Table", "Soil products used")
    d.table(
        [
            "Product",
            "Extraction",
            "Access date",
            "Texture limits",
            "Input interval",
            "Licence",
        ],
        [
            [
                p.name,
                p.run.get("Tool", "-"),
                p.run.get("Access date", "-"),
                p.texture_system,
                p.interval,
                p.run.get("Licence", "-"),
            ]
            for p in products
        ],
        widths_cm=[2.6, 3.4, 2.2, 2.4, 3.2, 2.7],
    )
    for p in products:
        if p.run.get("Citation"):
            d.para(p.run["Citation"], bold_lead=f"{p.name}:")
    if any(p.water_source for p in products):
        d.para(
            "Mapped water contents used only for the checks in section 5.7: "
            + "; ".join(p.water_source for p in products if p.water_source)
            + "."
        )
    d.caption("Table", "Run settings")
    d.table(
        ["Setting", "Value"],
        [
            ["Tool", f"{TOOL_NAME} (Mayim Tools) v{TOOL_VERSION}"],
            ["Run time", result.run_time_utc],
            ["Products", ", ".join(names)],
            [
                "Processing grid",
                f"{g.width} x {g.height} cells of {_sig(g.res, 3)} map units (the "
                "finest input grid unless a resolution was set), nearest-neighbour "
                "resampling",
            ],
            ["Depth layers", ", ".join(labs)],
            ["Methods", ", ".join(m.name for m in methods)],
            ["Monte Carlo draws per product", str(mc.draws)],
            ["Random seed", str(mc.seed)],
            ["OM factor (OM = SOC x factor)", f"{mc.om_factor:g}"],
            ["Density adjustment", "yes" if mc.density else "no"],
            ["Gravel correction", "yes" if mc.gravel else "no"],
            ["Neighbour fill radius (m)", _f(s.fill_radius_m, 0)],
            ["Zones", ", ".join(result.zone_names)],
            [
                "Only cells inside the zones",
                "yes" if (s.only_zones and s.zones) else "no",
            ],
        ],
        widths_cm=[6.2, 10.3],
    )
    d.para(
        "Coarser inputs repeat in blocks on the processing grid; the effective "
        "resolution is that of the coarsest input used."
    )
    src_rows = []
    for p in products:
        for lab in labs:
            src = result.sources.get(p.name, {}).get(lab)
            if src is None:
                continue
            inside = src[result.zone_raster > 0]
            n = inside.size or 1
            src_rows.append(
                [p.name, lab]
                + [_f(100.0 * (inside == c).sum() / n, 1) for c in (1, 2, 3, 4, 0)]
            )
    if src_rows:
        d.caption(
            "Table",
            "Source of the clay input per product and layer (% of cells in the zones)",
        )
        d.table(
            [
                "Product",
                "Layer",
                "Own",
                "Other product",
                "SoilGrids 2017",
                "Neighbours",
                "No data",
            ],
            src_rows,
            widths_cm=[3.3, 2.2, 1.8, 2.4, 2.4, 2.4, 2.0],
        )
    panels = [
        (result.sources[p.name][top], p.name)
        for p in products
        if top in result.sources.get(p.name, {})
    ]
    if panels:
        _figure(
            d,
            lambda: _categorical_maps(
                result,
                panels,
                SOURCE_COLOURS,
                {k: v for k, v in fill_mod.SOURCE_LABELS.items()},
            ),
            f"Input source per cell, {top}",
            notes,
        )

    # 4. Parameters and calculations ----------------------------------------
    d.heading("4. Parameters and calculations", 2)
    d.caption("Table", "Parameters: symbol, definition and source")
    d.table(
        ["Parameter (symbol, unit)", "Definition / equation", "Source"],
        _definition_rows(mc.methods),
        widths_cm=[5.0, 7.7, 3.8],
    )
    sec = 0
    if "SR2006" in mc.methods:
        sec += 1
        _saxton_rawls_section(d, f"4.{sec}")
    if "TOTH2015" in mc.methods:
        sec += 1
        _toth_section(d, f"4.{sec}")
    sec += 1
    _green_ampt_section(d, f"4.{sec}")
    sec += 1
    _rawls_section(d, f"4.{sec}")
    sec += 1
    d.heading(f"4.{sec} Validity flags", 3)
    d.caption("Table", "Validity flags (bit mask in the quality raster)")
    d.table(
        ["Bit", "Meaning"],
        [
            [str(k), v]
            for k, v in {**saxton_rawls.FLAG_LABELS, **toth2015.FLAG_LABELS}.items()
        ],
        widths_cm=[1.6, 14.9],
    )
    d.para(
        "Flags are evaluated on each product's central soil and combined. "
        "With a single method the method share of the uncertainty is zero by "
        "construction; select both methods to see the method spread."
    )

    # 5. Results ------------------------------------------------------------
    d.heading("5. Results", 2)
    d.heading("5.1 USDA soil texture", 3)
    d.para(
        "The maps show the USDA texture class of each cell per depth layer (the "
        "class most Monte Carlo draws fall in), in the Mayim texture layer "
        "style. The table gives the dominant class per zone and layer and the "
        "ensemble median texture."
    )
    for lab in labs:
        _texture_map_figure(d, result, lab, map_fn, notes, osm_used)
    notes += list(getattr(map_fn, "messages", []) or [])
    tex_rows = [
        [
            t["Zone"],
            t["Layer"],
            t["Dominant class"],
            _pct(t["Dominant share (%)"]),
            str(t["Classes present"]),
            _f(t["Sand (%)"], 1),
            _f(t["Silt (%)"], 1),
            _f(t["Clay (%)"], 1),
            _f(t["OM (%)"], 2),
        ]
        for t in result.zone_texture
    ]
    d.caption("Table", "Texture (USDA limits, ensemble medians) per zone and layer")
    d.table(
        [
            "Zone",
            "Layer",
            "Dominant class",
            "Share (%)",
            "Classes",
            "Sand (%)",
            "Silt (%)",
            "Clay (%)",
            "OM (%)",
        ],
        tex_rows,
        widths_cm=[2.4, 1.8, 2.6, 1.5, 1.6, 1.6, 1.6, 1.6, 1.8],
    )
    for lab in labs[:1]:
        _figure(
            d,
            lambda lab=lab: _texture_triangle(result, lab),
            f"Soils of the area on the USDA texture triangle, {lab}",
            notes,
        )

    d.heading("5.2 Parameter maps", 3)
    med_maps = [
        (result.stats["ksat"][top][1], "Ksat P50", "viridis", True, "mm/h"),
        (result.stats["psi_f"][top][1], "ψf P50", "magma_r", True, "mm"),
        (result.stats["theta_s"][top][1], "θs P50", "Blues", False, "m³/m³"),
    ]
    _figure(d, lambda: _continuous_maps(result, med_maps), f"Median maps, {top}", notes)
    with np.errstate(invalid="ignore", divide="ignore"):
        rng_maps = [
            (
                result.stats["ksat"][top][2] / result.stats["ksat"][top][0],
                "Ksat P95/P5",
                "YlOrRd",
                True,
                "ratio",
            ),
            (result.stats["theta_fc"][top][1], "θ33 P50", "Blues", False, "m³/m³"),
            (result.stats["paw"][top][1], "PAW P50", "Greens", False, "m³/m³"),
        ]
    _figure(
        d,
        lambda: _continuous_maps(result, rng_maps),
        f"Uncertainty of Ksat and water-holding maps, {top}",
        notes,
    )
    if "ima_dry" in result.stats and "t_dry" in result.stats:
        ga_maps = [
            (
                result.stats["ima_dry"][top][1],
                "Initial deficit, dry",
                "Oranges",
                False,
                "m³/m³",
            ),
            (
                result.stats["ima_moist"][top][1],
                "Initial deficit, moist",
                "Oranges",
                False,
                "m³/m³",
            ),
            (result.stats["t_dry"][top][1], "Drying time", "viridis_r", True, "days"),
        ]
        _figure(
            d,
            lambda: _continuous_maps(result, ga_maps),
            f"Green-Ampt (SWMM) initial deficit and drying time, {top} (P50)",
            notes,
        )

    d.heading("5.3 Zone statistics", 3)
    for zone in result.zone_names:
        d.caption("Table", f"{zone}: median of cell medians, with median cell P5-P95")
        hdr = ["Parameter", "Units"] + labs
        rows = []
        for prm in PARAMETERS:
            if prm.code not in result.stats:
                continue
            rows.append(
                [prm.label, _u(prm.units)]
                + [
                    _value_range(_row(result, zone, lab, prm.code), prm.code)
                    for lab in labs
                ]
            )
        lw = (16.5 - 5.2 - 1.6) / max(len(labs), 1)
        d.table(hdr, rows, widths_cm=[5.2, 1.6] + [lw] * len(labs))
    _figure(
        d,
        lambda: _profiles(result, result.zone_names[0]),
        f"Depth profiles, {result.zone_names[0]} (median and median cell P5-P95)",
        notes,
    )
    _figure(
        d,
        lambda: _box_by_product(result, top),
        f"Distribution over cells per product and method (central soils) and the pooled median, {top}",
        notes,
    )

    d.heading("5.4 Confidence", 3)
    _figure(
        d,
        lambda: _categorical_maps(
            result,
            [(result.ksat_class[lab], lab) for lab in labs],
            CONF_COLOURS,
            KSAT_SHORT,
        ),
        "Ksat uncertainty class (90 % range = median x/÷ factor)",
        notes,
    )
    _figure(
        d,
        lambda: _categorical_maps(
            result,
            [(result.texture_conf[lab], lab) for lab in labs],
            CONF_COLOURS,
            TEX_SHORT,
        ),
        "Texture class confidence (share of draws in the most frequent class)",
        notes,
    )
    d.caption("Table", "Confidence per zone and layer (% of cells)")
    d.table(
        [
            "Zone",
            "Layer",
            "Ksat factor (median)",
            "Ksat x/÷ ≤ 2",
            "x/÷ 2-4",
            "x/÷ 4-10",
            "x/÷ > 10",
            "Texture ≥ 80 %",
            "Texture < 40 %",
        ],
        [
            [
                t["Zone"],
                t["Layer"],
                _sig(t["Median Ksat factor (x/÷)"], 2),
                _pct(t["Ksat class 1 (%)"]),
                _pct(t["Ksat class 2 (%)"]),
                _pct(t["Ksat class 3 (%)"]),
                _pct(t["Ksat class 4 (%)"]),
                _pct(t["Texture confidence 1 (%)"]),
                _pct(t["Texture confidence 4 (%)"]),
            ]
            for t in result.zone_texture
        ],
        widths_cm=[2.4, 1.8, 1.9, 1.7, 1.6, 1.6, 1.6, 1.9, 2.0],
    )
    _figure(
        d,
        lambda: _variance_bars(result, 1, top),
        f"Sources of uncertainty, {result.zone_names[0]}, {top}",
        notes,
    )

    sub = 5
    if both and result.comparison:
        d.heading("5.5 Product comparison", 3)
        sub = 6
        _product_comparison_section(d, result, top, notes)

    _design_section(d, result, notes, f"5.{sub}")
    _checks_section(d, result, top, notes, f"5.{sub + 1}")

    # 6. Outputs ------------------------------------------------------------
    d.heading("6. Outputs", 2)
    d.para(
        "The selected outputs were written to the folders below (file names "
        "<project number>_<description>). Each parameter raster holds the P50, "
        "P05 and P95 for every layer; select bands by their description (e.g. "
        "ksat_0-30cm_P50 (mm/h)), never by band number. The texture rasters "
        "hold the USDA class code (1-12) with the colour table of the layer "
        "style; the run details are also stored in the raster metadata."
    )
    out_rows = [
        [os.path.basename(f), os.path.dirname(f) or "-", desc]
        for f, desc in result.files
    ]
    meta = s.path(OUT_METADATA_CSV)
    if meta:
        out_rows.append(
            [
                os.path.basename(meta),
                os.path.dirname(meta) or "-",
                "Run metadata (settings, inputs, checks, warnings)",
            ]
        )
    rep = s.path(OUT_REPORT) or path
    out_rows.append([os.path.basename(rep), os.path.dirname(rep) or "-", "This report"])
    d.caption("Table", "Output files")
    d.table(["File", "Folder", "Content"], out_rows, widths_cm=[5.2, 5.6, 5.7])
    d.caption("Table", "Texture class codes (USDA)")
    d.table(
        ["Code", "Class", "Colour (layer style)"],
        [[str(c), class_label(c), TEXTURE_COLOURS[c]] for c in sorted(TEXTURE_COLOURS)],
        widths_cm=[1.6, 9.0, 5.9],
    )

    # 7. Notes and limitations ----------------------------------------------
    d.heading("7. Notes and limitations", 2)
    for text in (
        "Soil structure and macropores are not represented by texture-based "
        "methods; field infiltration can differ by an order of magnitude.",
        "Global soil maps are trained mostly on agricultural and national "
        "survey data; their site-level accuracy is modest, and smoothing "
        "under-represents very sandy and very clayey soils.",
        "Restrictive layers other than bedrock (plinthite, duplex contrasts, "
        "hardpans) and water tables are not mapped by these products.",
        "The ensemble spread is a lower bound: shared blind spots of the "
        "methods and products are not captured, and inputs are treated as "
        "independent.",
        "Built-up land is filled from older or neighbouring predictions; "
        "sealed surfaces and fill material are not soil and need separate "
        "treatment.",
        "Use local data where available: field infiltration tests and profile "
        "descriptions take precedence over these estimates.",
    ):
        d.bullet(text)
    if osm_used:
        d.para(OSM_LICENCE_NOTE)
    warn = list(result.warnings) + list(result.notes) + notes
    d.label("Processing warnings and notes from this run")
    if warn:
        for w in warn:
            d.bullet(w)
    else:
        d.para("None.")

    # 8. References ---------------------------------------------------------
    d.heading("8. References", 2)
    for ref in REFERENCES:
        d.para(ref)
    d.doc.save(path)


def _definition_rows(method_codes) -> list[list[str]]:
    """Parameters and calculations table (symbol, definition, source)."""
    sr = "SR2006" in method_codes
    toth = "TOTH2015" in method_codes
    both = " / ".join(
        x
        for x, on in (("Saxton & Rawls (2006)", sr), ("Tóth et al. (2015)", toth))
        if on
    )
    rows = [
        [
            "Saturated water content θs (m³/m³)",
            "Water content at saturation (total porosity of the matrix).",
            both,
        ],
        ["Field capacity θ33 (m³/m³)", "Water content at 33 kPa suction.", both],
        ["Wilting point θ1500 (m³/m³)", "Water content at 1500 kPa suction.", both],
        ["Plant-available water PAW (m³/m³)", "PAW = θ33 − θ1500.", both],
        [
            "Saturated hydraulic conductivity Ksat (mm/h)",
            "Conductivity of the saturated soil matrix. SWMM 'Conductivity'.",
            both,
        ],
        [
            "Wetting-front suction head ψf (mm)",
            "Brooks-Corey: ψf = (2 + 3λ)/(1 + 3λ) · ψb/2; van Genuchten: "
            "effective capillary drive (see the Green-Ampt sub-section). SWMM 'Suction head'.",
            "Rawls & Brakensiek (1983); Morel-Seytoux et al. (1996)",
        ],
        [
            "Initial deficit, dry start IMD (m³/m³ and mm)",
            "IMD = θs − θ1500; depth (mm) = IMD x layer thickness (mm). SWMM "
            "'Initial deficit'.",
            "Rossman & Huber (2016); Rawls et al. (1983)",
        ],
        [
            "Initial deficit, moist start IMD (m³/m³ and mm)",
            "IMD = θs − θ33; depth (mm) = IMD x layer thickness (mm).",
            "Rossman & Huber (2016); Rawls et al. (1983)",
        ],
        [
            "Drying time T (days)",
            "T = 15.78 / √Ksat, Ksat in mm/h: time for a saturated soil to "
            "drain to its initial deficit between storms.",
            "Rossman & Huber (2016); WRC Report 2747/1/20 (Eq. 8.2)",
        ],
    ]
    if sr:
        rows += [
            [
                "Air-entry suction ψb (mm)",
                "Bubbling pressure of the Campbell / Brooks-Corey curve.",
                "Saxton & Rawls (2006)",
            ],
            [
                "Pore-size distribution index λ (-)",
                "λ = 1 / B; slope of the log retention curve.",
                "Saxton & Rawls (2006)",
            ],
        ]
    if toth:
        rows += [
            [
                "van Genuchten θr, α (1/cm), n (-)",
                "θ(h) = θr + (θs − θr) / [1 + (αh)^n]^(1 − 1/n).",
                "van Genuchten (1980); Tóth et al. (2015)",
            ]
        ]
    rows += [
        [
            "USDA texture class (code 1-12)",
            "Class of the sand, silt and clay fractions (USDA limits); the mode "
            "of the Monte Carlo draws.",
            "Soil Science Division Staff (2017)",
        ],
        [
            "Ksat uncertainty factor F (x/÷)",
            "F = √(P95/P5); the 90 % range is the median x/÷ F.",
            "This tool",
        ],
    ]
    return rows


def _saxton_rawls_section(d, num) -> None:
    d.heading(f"{num} Saxton & Rawls (2006)", 3)
    d.para(
        "Calibration data: A-horizon samples of the USDA/NRCS National Soil "
        "Characterization database; 2149 samples reduced to 1722 by excluding "
        "bulk density below 1.0 or above 1.8 g/cm³, organic matter above 8 % "
        "and clay above 60 %. Reported fit: θ1500 R² = 0.86, θ33 R² = 0.63, "
        "θS-33 R² = 0.36, air-entry tension R² = 0.78 (standard error 2.9 kPa). "
        "Inputs: sand S and clay C (decimal, USDA), organic matter OM (%). "
        "Water contents are decimal volume fractions."
    )
    eq = [
        ("1", "θ1500t = -0.024S + 0.487C + 0.006OM + 0.005(S·OM) - 0.013(C·OM) + 0.068(S·C) + 0.031;  θ1500 = θ1500t + (0.14θ1500t - 0.02)"),
        ("2", "θ33t = -0.251S + 0.195C + 0.011OM + 0.006(S·OM) - 0.027(C·OM) + 0.452(S·C) + 0.299;  θ33 = θ33t + (1.283θ33t² - 0.374θ33t - 0.015)"),
        ("3", "θ(S-33)t = 0.278S + 0.034C + 0.022OM - 0.018(S·OM) - 0.027(C·OM) - 0.584(S·C) + 0.078;  θS-33 = θ(S-33)t + (0.636θ(S-33)t - 0.107)"),
        ("4", "ψet = -21.67S - 27.93C - 81.97θS-33 + 71.12(S·θS-33) + 8.29(C·θS-33) + 14.05(S·C) + 27.16;  ψe = ψet + (0.02ψet² - 0.113ψet - 0.70)  [kPa]"),
        ("5", "θS = θ33 + θS-33 - 0.097S + 0.043"),
        ("6", "ρN = (1 - θS) · 2.65  [g/cm³]"),
        ("7-10", "Density adjustment (optional): DF = ρ/ρN limited to 0.9-1.3; θS-DF = 1 - ρN·DF/2.65; θ33-DF = θ33 - 0.2(θS - θS-DF)"),
        ("14-15", "B = [ln 1500 - ln 33] / [ln θ33 - ln θ1500];  A = exp(ln 33 + B ln θ33)"),
        ("16", "Ksat = 1930 (θS - θ33)^(3 - λ)  [mm/h]"),
        ("18", "λ = 1 / B"),
        ("19-22", "Gravel (optional): Rv = αRw / [1 - Rw(1 - α)], α = ρ/2.65; Kb/Ks = (1 - Rw) / [1 - Rw(1 - 3α/2)]; PAWB = PAW(1 - Rv)"),
    ]  # fmt: skip
    d.caption(
        "Table",
        "Saxton & Rawls (2006) equations used (numbers as in the paper's Table 1)",
    )
    d.table(["Eq.", "Equation"], [list(e) for e in eq], widths_cm=[1.6, 14.9])
    d.para(
        "Verification: the implementation reproduces the paper's Table 3 "
        "(twelve texture-class examples at 2.5 % OM) exactly for wilting "
        "point, field capacity, saturation, plant-available water, Ksat and "
        "normal density; this is part of the automated test suite."
    )
    d.para(
        f"Air-entry tension: Eq. 4 returns values near or below zero for sands "
        f"(-0.96 kPa for the paper's sand example). ψe is therefore bounded below "
        f"at {saxton_rawls.YE_MIN_KPA:g} kPa, the geometric-mean bubbling "
        "pressure of sand (7.26 cm; Rawls, Brakensiek & Saxton, 1982); such "
        "cells carry validity flag 16."
    )


def _toth_section(d, num) -> None:
    d.heading(f"{num} Tóth et al. (2015) / HiHydroSoil v2.0", 3)
    d.para(
        "European continuous pedotransfer functions (Tóth et al., 2015), "
        "calibrated on the EU-HYDI database of European soils. They are the "
        "method behind the global HiHydroSoil v2.0 maps (Simons et al., "
        "2020), which applied them to SoilGrids; this tool applies them to "
        "the harmonised inputs and adds the Monte Carlo uncertainty. The "
        "retention curve is Mualem-van Genuchten (van Genuchten, 1980). "
        "Inputs: clay and silt (%, USDA limits), organic carbon OC (%), "
        "bulk density BD (g/cm³), pH in water and CEC (cmol(c)/kg); T/S = 1 "
        "for topsoil (0-30 cm) and 0 below."
    )
    teq = [
        ("θr", "0.041 if sand >= 2 %, else 0.179  [m³/m³]"),
        ("θs", "0.83080 - 0.28217 BD + 0.0002728 Cl + 0.000187 Si  [m³/m³]"),
        ("α", "log10 α = -0.43348 - 0.41729 BD - 0.04762 OC + 0.21810 T/S - 0.01581 Cl - 0.01207 Si  [1/cm]"),
        ("n", "log10 (n - 1) = 0.22236 - 0.30189 BD - 0.05558 T/S - 0.005306 Cl - 0.003084 Si - 0.01072 OC"),
        ("Ksat", "log10 Ksat = 0.40220 + 0.26122 pH + 0.44565 T/S - 0.02329 Cl - 0.01265 Si - 0.01038 CEC  [cm/day]"),
        ("θ(h)", "θr + (θs - θr) / [1 + (α h)^n]^(1 - 1/n);  h = 336.5 cm (33 kPa), 15296 cm (1500 kPa)"),
    ]  # fmt: skip
    d.caption("Table", "Tóth et al. (2015) equations as used in HiHydroSoil v2.0")
    d.table(["Output", "Equation"], [list(e) for e in teq], widths_cm=[1.6, 14.9])
    d.para(
        "Verification: every coefficient was checked against the "
        "HiHydroSoil v2.0 report (Simons et al., 2020, pp. 7-8). Two "
        "deliberate differences from the HiHydroSoil product: field "
        "capacity is θ at 33 kPa here (HiHydroSoil publishes pF2, about 10 "
        "kPa, which gives wetter values), and the 0-30 cm Ksat is computed "
        "from layer-averaged inputs (HiHydroSoil averages the depth values "
        "harmonically). Ksat needs CEC at pH 7, which only SoilGrids "
        "provides; the OpenLandMap and iSDAsoil members use the SoilGrids "
        "CEC when its folder is given (iSDAsoil maps effective CEC, which is "
        "not used). Calibration region: Europe. Applied elsewhere, the "
        "functions extrapolate, and their spread against the other method is "
        "part of the reported uncertainty."
    )


def _green_ampt_section(d, num) -> None:
    d.heading(f"{num} Green-Ampt (SWMM) parameters", 3)
    d.para(
        "For Brooks-Corey type methods the wetting-front suction head follows "
        "from the air-entry (bubbling) suction ψb and the pore-size distribution "
        "index λ (Rawls & Brakensiek, 1983; Rawls et al., 1993):"
    )
    d.para("ψf = (2 + 3λ) / (1 + 3λ) · ψb / 2")
    d.para(
        "with ψb = ψe of Saxton & Rawls (kPa x 101.97 = mm). For van Genuchten "
        "methods (Tóth et al.) ψf is the effective capillary drive of the "
        "Mualem conductivity curve, in the closed form of Morel-Seytoux et al. "
        "(1996): ψf = (1/α)(0.046m + 2.07m² + 19.5m³)/(1 + 4.7m + 16m²), m = "
        "1 - 1/n; the test suite checks it against numerical integration "
        "(within 2 %). The two definitions differ: the Brooks-Corey form "
        "follows the air-entry suction and gives larger values."
    )
    d.para(
        "The Green-Ampt moisture deficit is the porosity less the starting "
        "water content, IMD = θs − θi (Rawls et al., 1983). Two bounding starts "
        "are given: dry, θi = θ1500 (wilting point), and moist, θi = θ33 (field "
        "capacity). As a depth of water over a layer of thickness D, IMD (mm) "
        "= IMD x D (mm), e.g. x 300 mm for the 0-30 cm layer. Saxton & Rawls "
        "have no residual water content, so the effective porosity equals θS "
        "for this method."
    )
    d.para(
        "SWMM recovers the deficit between storms over the drying time T, the "
        "number of days for a fully saturated soil to dry to its initial "
        "deficit. In SI units (Rossman & Huber, 2016; WRC Report 2747/1/20, "
        "Eq. 8.2):"
    )
    d.para("T = 15.78 / √Ksat   [days, Ksat in mm/h]")
    d.para(
        "For example, Ksat = 25 mm/h gives T = 3.2 days. Some models use an "
        "effective conductivity of about 0.5 Ksat for the wetted zone "
        "(Bouwer, 1966); the Ksat outputs are not reduced."
    )


def _rawls_section(d, num) -> None:
    d.heading(f"{num} Reference check: Rawls, Brakensiek & Miller (1983)", 3)
    d.para(
        "Green-Ampt parameters tabulated per USDA texture class (as reproduced "
        "in Chow et al., 1988, Table 4.3.1, and in WRC Report 2747/1/20, Table "
        "6.6) are compared with the tool's medians in the checks. They are "
        "class means of a large US data set, not an ensemble member. The "
        "tabulated K is a Green-Ampt hydraulic conductivity, roughly half of "
        "the saturated conductivity, and the table has no Silt class."
    )
    d.caption("Table", "Rawls, Brakensiek & Miller (1983) class values")
    d.table(
        ["Class", "Porosity", "Effective porosity", "ψf (mm)", "K (mm/h)"],
        [
            [
                CLASS_NAME[c],
                _f(v[0], 3),
                _f(v[1], 3),
                _f(v[2] * 10, 1),
                _f(v[3] * 10, 1),
            ]
            for c, v in rawls_1983.RAWLS_1983.items()
        ],
        widths_cm=[4.5, 3.0, 3.0, 3.0, 3.0],
    )


def _product_comparison_section(d, result, top, notes) -> None:
    d.para(
        "The products are used as equal ensemble members. Each pair is "
        "compared on the central soils (USDA limits) where both have their "
        "own prediction; differences are the second product minus the first."
    )
    hdr = [
        "Pair (B - A)",
        "Layer",
        "Cells",
        "Clay A (%)",
        "Clay B (%)",
        "Sand A (%)",
        "Sand B (%)",
        "Same class (%)",
        "Ksat ratio B/A",
        "Ksat differs > x4 (%)",
    ]
    d.caption("Table", "Product comparison (medians over cells)")
    d.table(
        hdr,
        [
            [
                c["Pair"],
                c["Layer"],
                str(c["Cells"]),
                _f(c["Clay A (%)"], 1),
                _f(c["Clay B (%)"], 1),
                _f(c["Sand A (%)"], 1),
                _f(c["Sand B (%)"], 1),
                _pct(c["Texture class agreement (%)"]),
                _sig(c["Median Ksat ratio B / A"], 2),
                _pct(c["Ksat differs by more than x4 (%)"]),
            ]
            for c in result.comparison
        ],
        widths_cm=[2.5, 1.6, 1.4, 1.5, 1.5, 1.5, 1.5, 1.6, 1.6, 1.8],
    )
    for a_name, b_name in product_pairs(result):
        label = pair_label(a_name, b_name)
        sa, sb = SHORT_PRODUCT[a_name], SHORT_PRODUCT[b_name]
        ca = SHORT_CODE[a_name]
        cb = SHORT_CODE[b_name]
        if top not in result.central.get(a_name, {}) or top not in result.central.get(
            b_name, {}
        ):
            continue
        a, b = result.central[a_name][top], result.central[b_name][top]
        _figure(
            d,
            lambda a=a, b=b, ca=ca, cb=cb: _continuous_maps(
                result,
                [
                    (b["clay"] - a["clay"], f"Clay {cb} - {ca}", "RdBu_r", False, "%"),
                    (b["sand"] - a["sand"], f"Sand {cb} - {ca}", "RdBu_r", False, "%"),
                    (
                        np.log10(np.maximum(b["ksat"], 1e-6))
                        - np.log10(np.maximum(a["ksat"], 1e-6)),
                        f"log10 Ksat {cb}/{ca}",
                        "RdBu_r",
                        False,
                        "-",
                    ),
                ],
            ),
            f"Product differences {label}, {top} (central soils, USDA limits)",
            notes,
        )
        smp = result.comparison_samples.get((label, top))
        if smp:
            _figure(
                d,
                lambda smp=smp, sa=sa, sb=sb: _scatter_panels(
                    [
                        (*smp["clay"], f"Clay {sa} (%)", f"Clay {sb} (%)", False),
                        (*smp["sand"], f"Sand {sa} (%)", f"Sand {sb} (%)", False),
                        (*smp["ksat"], f"Ksat {sa} (mm/h)", f"Ksat {sb} (mm/h)", True),
                    ]
                ),
                f"Cell-by-cell comparison {label}, {top} (sample of cells)",
                notes,
            )


def _checks_section(d, result, top, notes, num) -> None:
    d.heading(f"{num} Checks", 3)
    d.label("Mapped water contents")
    if result.checks:
        d.para(
            "The pedotransfer estimates of field capacity and wilting point are "
            "compared with independently mapped water contents. These maps are "
            "themselves predictions, so agreement is a consistency check, not a "
            "validation."
        )
        d.caption("Table", "PTF estimates (ensemble P50) against mapped water contents")
        d.table(
            [
                "Source",
                "Layer",
                "Quantity",
                "Cells",
                "Mapped (m³/m³)",
                "PTF (m³/m³)",
                "Bias",
                "RMSE",
                "r",
            ],
            [
                [
                    _short_source(c["Source"]),
                    c["Layer"],
                    "θ33" if c["Quantity"].startswith("Field") else "θ1500",
                    str(c["Cells"]),
                    _f(c["Mapped median (m3/m3)"], 3),
                    _f(c["PTF median (m3/m3)"], 3),
                    _f(c["Bias PTF - mapped (m3/m3)"], 3),
                    _f(c["RMSE (m3/m3)"], 3),
                    _f(c["Correlation r"], 2),
                ]
                for c in result.checks
            ],
            widths_cm=[2.8, 1.8, 1.6, 1.5, 2.0, 2.0, 1.6, 1.6, 1.6],
        )
        panels = []
        for key, (obs, pred) in result.check_samples.items():
            if key[1] == top and len(panels) < 2:
                q = "θ33" if key[2] == "fc" else "θ1500"
                panels.append(
                    (obs, pred, f"Mapped {q} (m³/m³)", f"PTF {q} (m³/m³)", False)
                )
        if panels:
            _figure(
                d,
                lambda: _scatter_panels(panels),
                f"PTF against mapped water contents, {top}",
                notes,
            )
    else:
        d.para(
            "No mapped water contents were in the input folders (SoilGrids "
            "wv0033/wv1500 or the OpenLandMap 250 m water content), so this "
            "check was not made."
        )
    d.label("Rawls, Brakensiek & Miller (1983)")
    d.caption(
        "Table", "Tool medians against the class values of the dominant texture class"
    )
    d.table(
        [
            "Zone",
            "Layer",
            "Class",
            "Rawls φ",
            "Tool θs",
            "Rawls ψf (mm)",
            "Tool ψf (mm)",
            "Rawls K (mm/h)",
            "Tool Ksat (mm/h)",
            "Ksat ratio",
        ],
        [
            [
                r["Zone"],
                r["Layer"],
                r["Dominant class"],
                _f(r["Rawls porosity"], 3),
                _f(r["Tool theta_s"], 3),
                _f(r["Rawls psi_f (mm)"], 0),
                _f(r["Tool psi_f (mm)"], 0),
                _sig(r["Rawls K (mm/h)"]),
                _sig(r["Tool Ksat (mm/h)"]),
                _sig(r["Median Ksat ratio (tool/Rawls)"], 2),
            ]
            for r in result.reference_rows
        ],
        widths_cm=[2.2, 1.6, 2.0, 1.4, 1.4, 1.6, 1.6, 1.6, 1.6, 1.5],
    )
    d.para(
        "The Ksat ratio is the median over cells of the tool's Ksat divided by "
        "the class K of each cell's texture class. A ratio near 2 is expected "
        "because the tabulated K is about half of Ksat; much larger or smaller "
        "ratios show where the methods differ and deserve attention."
    )
