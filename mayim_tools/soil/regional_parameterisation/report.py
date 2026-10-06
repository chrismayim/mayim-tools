"""Word (.docx) report for Regional soil parameterisation (no QGIS).

Needs python-docx and matplotlib (imported lazily by the shared helpers).
Figures use matplotlib's object API (no pyplot), which is safe on QGIS's
Processing worker thread. A figure that fails is skipped with a note, so
the report itself is always written.
"""

from __future__ import annotations

import math

import numpy as np

from mayim_tools._common.docx_report import (
    C_AQUA,
    C_BLUE,
    C_INK,
    C_INK_2,
    C_ORANGE,
    Doc,
    new_figure,
    png,
    style_axes,
)

from . import fill as fill_mod
from .ptf import (
    METHOD_BY_CODE,
    PARAMETER_BY_CODE,
    PARAMETERS,
    rawls_1983,
    saxton_rawls,
    toth2015,
)
from .texture import (
    CLASS_ABBR,
    CLASS_NAME,
    ISO_TO_USDA_SILT_FACTOR,
    TEXTURE_CLASSES,
    ternary_xy,
    usda_class,
)
from .uncertainty import VARIANCE_PARTS

REPORT_TITLE = "Regional Soil Parameterisation"
KEY_PARAMS = ("theta_s", "theta_fc", "theta_wp", "paw", "ksat", "psi_f")
SOURCE_COLOURS = {
    fill_mod.SOURCE_OWN: "#2a78d6",
    fill_mod.SOURCE_OTHER: "#eb6834",
    fill_mod.SOURCE_SG2017: "#1baf7a",
    fill_mod.SOURCE_NEIGHBOUR: "#b07ad6",
}
CONF_COLOURS = {1: "#1baf7a", 2: "#5bc0de", 3: "#f0ad4e", 4: "#d9534f"}
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
]


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
# Figures
# ----------------------------------------------------------------------


def _extent(grid):
    return (grid.xmin, grid.xmax, grid.ymin, grid.ymax)


def _log_ticks(axis, lo, hi):
    """Three to five readable ticks on a log axis (no minor labels)."""
    from matplotlib.ticker import (
        FixedLocator,
        FuncFormatter,
        NullFormatter,
        NullLocator,
    )

    lo, hi = max(lo, 1e-6), max(hi, lo * 1.0001)
    ticks = np.geomspace(lo, hi, 4)
    ticks = sorted({float(_sig(t, 2)) for t in ticks})
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: _sig(v, 2)))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def _map_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(C_INK_2)
        side.set_linewidth(0.5)
    ax.set_aspect("equal")


def _continuous_maps(result, panels, height_cm=7.0):
    """panels: [(array, title, cmap, log, label)] - up to 3 side by side."""
    from matplotlib.colors import LogNorm

    fig = new_figure(height_cm)
    n = len(panels)
    for i, (arr, title, cmap, log, label) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, n, i)
        v = arr[np.isfinite(arr)]
        if v.size == 0:
            ax.set_title(title + " (no data)", fontsize=8)
            _map_axes(ax)
            continue
        lo, hi = np.percentile(v, (2, 98))
        if log:
            lo = max(lo, 1e-3)
            hi = max(hi, lo * 1.01)
            norm = LogNorm(vmin=lo, vmax=hi)
            im = ax.imshow(
                arr,
                extent=_extent(result.grid),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
        else:
            if hi <= lo:
                hi = lo + 1e-6
            im = ax.imshow(
                arr,
                extent=_extent(result.grid),
                cmap=cmap,
                vmin=lo,
                vmax=hi,
                interpolation="nearest",
            )
        _map_axes(ax)
        ax.set_title(title, fontsize=8, color=C_INK)
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, orientation="horizontal")
        if log:
            _log_ticks(cb.ax.xaxis, lo, hi)
        cb.ax.tick_params(labelsize=6)
        cb.set_label(label, fontsize=7)
    fig.tight_layout()
    return png(fig)


def _categorical_maps(result, panels, colours, labels, height_cm=7.0):
    """panels: [(int array, title)]; colours/labels: code -> colour/label."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    codes = sorted(colours)
    cmap = ListedColormap([colours[c] for c in codes])
    norm = BoundaryNorm([c - 0.5 for c in codes] + [codes[-1] + 0.5], cmap.N)
    fig = new_figure(height_cm)
    n = len(panels)
    present = set()
    for i, (arr, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, n, i)
        a = np.where(np.isin(arr, codes), arr, np.nan).astype(float)
        present |= {int(c) for c in np.unique(a[np.isfinite(a)])}
        ax.imshow(
            a,
            extent=_extent(result.grid),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        _map_axes(ax)
        ax.set_title(title, fontsize=8, color=C_INK)
    handles = [Patch(color=colours[c], label=labels[c]) for c in codes if c in present]
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(4, len(handles)),
            fontsize=7,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return png(fig)


def _class_colours():
    from mayim_tools.soil._common.export import class_colour

    return {c: class_colour(c) for c, _, _ in TEXTURE_CLASSES}


def _texture_triangle(result, lab):
    """USDA triangle (class areas from the tool's own classifier) with the
    area's median soils and each product's central soils."""
    from matplotlib.colors import ListedColormap

    fig = new_figure(11.0)
    ax = fig.add_subplot(1, 1, 1)
    step = 0.5
    g = np.arange(0, 100 + step / 2, step)
    sa, cl = np.meshgrid(g, g)
    si = 100 - sa - cl
    inside = si >= -1e-9
    codes = np.where(inside, usda_class(sa, np.maximum(si, 0), cl), 0).astype(float)
    codes[~inside] = np.nan
    x, y = ternary_xy(sa, cl)
    cols = _class_colours()
    cmap = ListedColormap(["#ffffff"] + [cols[c] for c in range(1, 13)])
    ax.scatter(
        x[inside].ravel(),
        y[inside].ravel(),
        c=codes[inside].ravel(),
        cmap=cmap,
        vmin=-0.5,
        vmax=12.5,
        s=1.2,
        marker="s",
        alpha=0.18,
        linewidths=0,
        rasterized=True,
    )
    for c in range(1, 13):
        sel = codes == c
        if sel.any():
            ax.text(
                x[sel].mean(),
                y[sel].mean(),
                CLASS_ABBR[c],
                fontsize=7,
                ha="center",
                va="center",
                color=C_INK_2,
            )
    # outline and axes ticks
    tri = np.array([[0, 0], [100, 0], [50, 50 * math.sqrt(3)], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color=C_INK, linewidth=0.8)
    rng = np.random.default_rng(3)
    markers = []
    for pname, colour in zip(
        [p.name for p in result.products], (C_BLUE, C_ORANGE), strict=False
    ):
        c = result.central.get(pname, {}).get(lab)
        if not c:
            continue
        ok = np.flatnonzero(
            np.isfinite(c["clay"].ravel()) & (result.zone_raster.ravel() > 0)
        )
        if ok.size > 1500:
            ok = rng.choice(ok, 1500, replace=False)
        px, py = ternary_xy(c["sand"].ravel()[ok], c["clay"].ravel()[ok])
        ax.scatter(
            px,
            py,
            s=3,
            color=colour,
            alpha=0.35,
            linewidths=0,
            label=f"{pname} (central soil)",
        )
        markers.append(pname)
    med = result.inputs_p50
    ok = np.flatnonzero(
        np.isfinite(med["clay"][lab].ravel()) & (result.zone_raster.ravel() > 0)
    )
    if ok.size > 1500:
        ok = rng.choice(ok, 1500, replace=False)
    px, py = ternary_xy(med["sand"][lab].ravel()[ok], med["clay"][lab].ravel()[ok])
    ax.scatter(
        px, py, s=3, color=C_INK, alpha=0.5, linewidths=0, label="Ensemble median"
    )
    for v in (20, 40, 60, 80):
        # bottom edge (clay 0): sand = v
        xs, ys = ternary_xy(v, 0)
        ax.text(xs, ys - 3.5, f"{v}", fontsize=6, ha="center", color=C_INK_2)
        # left edge (silt 0): clay = v
        xc, yc = ternary_xy(100 - v, v)
        ax.text(xc - 2, yc, f"{v}", fontsize=6, ha="right", va="center", color=C_INK_2)
        # right edge (sand 0): silt = v
        xr, yr = ternary_xy(0, 100 - v)
        ax.text(xr + 2, yr, f"{v}", fontsize=6, ha="left", va="center", color=C_INK_2)
    ax.text(50, -9, "Sand (%)", fontsize=7, ha="center", color=C_INK)
    ax.text(14, 46, "Clay (%)", fontsize=7, rotation=60, ha="center", color=C_INK)
    ax.text(86, 46, "Silt (%)", fontsize=7, rotation=-60, ha="center", color=C_INK)
    ax.set_xlim(-8, 108)
    ax.set_ylim(-12, 92)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=7, frameon=False, markerscale=3)
    fig.tight_layout()
    return png(fig)


def _scatter_panels(panels, height_cm=6.5):
    """panels: [(x, y, xlabel, ylabel, log)] with a 1:1 line."""
    fig = new_figure(height_cm)
    n = len(panels)
    for i, (x, y, xl, yl, log) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, n, i)
        ok = np.isfinite(x) & np.isfinite(y)
        if log:
            ok &= (x > 0) & (y > 0)
        x, y = x[ok], y[ok]
        ax.scatter(x, y, s=3, color=C_BLUE, alpha=0.35, linewidths=0)
        if x.size:
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], color=C_INK_2, linewidth=0.7, linestyle="--")
        if log and x.size:
            ax.set_xscale("log")
            ax.set_yscale("log")
            lo_, hi_ = min(x.min(), y.min()), max(x.max(), y.max())
            _log_ticks(ax.xaxis, lo_, hi_)
            _log_ticks(ax.yaxis, lo_, hi_)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        style_axes(ax)
    fig.tight_layout()
    return png(fig)


def _profiles(result, zone):
    fig = new_figure(7.0)
    codes = ("theta_s", "theta_fc", "theta_wp", "ksat", "psi_f")
    labs = [
        lab for lab, top, bot in result.settings.layers if lab in result.texture_class
    ]
    mids = {lab: (top + bot) / 2 for lab, top, bot in result.settings.layers}
    for i, code in enumerate(codes, start=1):
        ax = fig.add_subplot(1, len(codes), i)
        prm = PARAMETER_BY_CODE[code]
        ys, med, lo, hi = [], [], [], []
        for lab in labs:
            r = _row(result, zone, lab, code)
            if r is None:
                continue
            ys.append(mids[lab])
            med.append(r["Median of P50"])
            lo.append(r["Median cell P5"])
            hi.append(r["Median cell P95"])
        if ys:
            ax.fill_betweenx(ys, lo, hi, color=C_BLUE, alpha=0.18, linewidth=0)
            ax.plot(med, ys, color=C_BLUE, marker="o", markersize=3, linewidth=1)
        ax.invert_yaxis()
        if prm.log and ys:
            ax.set_xscale("log")
            _log_ticks(ax.xaxis, min(lo), max(hi))
        ax.set_xlabel(f"{code} ({prm.units})")
        if i == 1:
            ax.set_ylabel("Depth (cm, layer middle)")
        style_axes(ax)
    fig.tight_layout()
    return png(fig)


def _box_by_product(result, lab):
    """Distribution over cells of each product x method central soil and of
    the pooled ensemble median, for Ksat, the Green-Ampt suction and field
    capacity."""
    fig = new_figure(7.5)
    zone_mask = result.zone_raster > 0
    short = {"SoilGrids 2.0": "SG", "OpenLandMap-soildb": "OLM"}
    mshort = {"SR2006": "S&R", "TOTH2015": "Tóth"}
    for i, code in enumerate(("ksat", "psi_f", "theta_fc"), start=1):
        ax = fig.add_subplot(1, 3, i)
        data, names = [], []
        for p in result.products:
            by = result.central.get(p.name, {}).get(lab, {}).get("by_method", {})
            for mcode, vals in by.items():
                if code not in vals:
                    continue
                v = vals[code][zone_mask]
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                data.append(v)
                names.append(f"{short.get(p.name, p.name)}\n{mshort.get(mcode, mcode)}")
        v = result.stats[code][lab][1][zone_mask]
        data.append(v[np.isfinite(v)])
        names.append("Pooled")
        data = [d if d.size else np.array([np.nan]) for d in data]
        ax.boxplot(data, showfliers=False, widths=0.5, medianprops={"color": C_ORANGE})
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names)
        prm = PARAMETER_BY_CODE[code]
        if prm.log:
            ax.set_yscale("log")
            allv = np.concatenate(
                [x[np.isfinite(x) & (x > 0)] for x in data] or [np.array([1.0])]
            )
            if allv.size:
                _log_ticks(ax.yaxis, np.percentile(allv, 1), np.percentile(allv, 99))
        ax.set_ylabel(f"{code} ({prm.units})")
        ax.tick_params(axis="x", labelsize=6)
        style_axes(ax)
    fig.tight_layout()
    return png(fig)


def _variance_bars(result, zone_index, lab):
    fig = new_figure(6.0)
    ax = fig.add_subplot(1, 1, 1)
    per = result.variance.get(zone_index, {}).get(lab, {})
    codes = [p.code for p in PARAMETERS if p.code in per]
    colours = {
        "input": C_BLUE,
        "product": C_ORANGE,
        "method": C_AQUA,
        "interaction": "#999999",
    }
    left = np.zeros(len(codes))
    for part in VARIANCE_PARTS:
        vals = []
        for code in codes:
            tot = per[code].get("total", 0.0)
            vals.append(per[code].get(part, 0.0) / tot if tot > 0 else 0.0)
        vals = np.array(vals)
        ax.barh(codes, vals, left=left, color=colours[part], label=part)
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of variance")
    ax.invert_yaxis()
    ax.legend(
        fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0)
    )
    style_axes(ax)
    fig.tight_layout()
    return png(fig)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------


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
            [f"{code} ({prm.units})"]
            + [fmt(r.get(key)) for _, key in cols]
            + [fmt(r["Median of P50"])]
        )
    d.caption(
        "Table",
        f"{zone}, {lab}: median of each product's central soil and of the pooled "
        "ensemble",
    )
    d.table(["Parameter"] + [c for c, _ in cols] + ["Pooled P50"], rows)


SHORT_PRODUCT = {"SoilGrids 2.0": "SoilGrids", "OpenLandMap-soildb": "OpenLandMap"}
SHORT_CODE = {"SoilGrids 2.0": "SG", "OpenLandMap-soildb": "OLM"}


def _figure(d, builder, caption, notes):
    try:
        d.picture(builder(), caption)
    except Exception as exc:  # noqa: BLE001 - a figure must never stop the report
        notes.append(f"Figure '{caption}' could not be drawn ({exc}).")


def write_report(result, path: str) -> None:
    d = Doc()
    s = result.settings
    mc = s.mc
    g = result.grid
    notes: list[str] = []
    labs = [lab for lab, _, _ in s.layers if lab in result.texture_class]
    top = labs[0] if labs else s.layers[0][0]
    products = result.products
    names = [p.name for p in products]
    both = len(products) == 2
    methods = [METHOD_BY_CODE[m] for m in mc.methods]
    area = float((result.zone_raster > 0).sum()) * result.cell_area_km2

    d.heading(REPORT_TITLE, 1)

    # 1. Summary ---------------------------------------------------------
    d.heading("1. Summary", 2)
    d.para(
        "This report documents hydraulic soil parameters estimated for "
        f"{len(result.zone_names)} zone{'s' if len(result.zone_names) != 1 else ''} "
        f"covering {_f(area, 1)} km2, for the depth layers "
        f"{', '.join(labs)}. The inputs are global soil maps from "
        f"{' and '.join(names)}, harmonised to these layers on a "
        f"{_f(g.res, 0) if g.res >= 1 else _sig(g.res)} (map units) grid. Parameters were "
        f"estimated with {', '.join(m.name for m in methods)} and their "
        f"uncertainty was propagated by Monte Carlo simulation ({mc.draws} draws "
        f"per cell and product). Rawls, Brakensiek & Miller (1983) texture-class "
        "values are used as an independent reference check."
    )
    d.caption(
        "Table",
        f"Key parameters for {top}: median of the cell medians "
        "(median cell P5-P95 in brackets)",
    )
    hdr = ["Zone"] + [f"{c} ({PARAMETER_BY_CODE[c].units})" for c in KEY_PARAMS]
    rows = []
    for zone in result.zone_names:
        rows.append(
            [zone] + [_value_range(_row(result, zone, top, c), c) for c in KEY_PARAMS]
        )
    d.table(hdr, rows)
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
                f"The two products disagree ({top}): mean clay difference "
                f"OpenLandMap - SoilGrids {_f(c['Mean clay difference OLM - SG (%)'], 1)} %, "
                f"sand {_f(c['Mean sand difference OLM - SG (%)'], 1)} %; same texture "
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
            + ". See section 4.4."
        )

    # 2. Purpose and scope ---------------------------------------------
    d.heading("2. Purpose and scope", 2)
    d.para(
        "The tool turns global soil property maps into the soil hydraulic "
        "parameters needed for infiltration and soil-water modelling: water "
        "contents at saturation, field capacity (33 kPa) and wilting point "
        "(1500 kPa), plant-available water, saturated hydraulic conductivity "
        "(Ksat), Brooks-Corey retention parameters and the Green-Ampt "
        "wetting-front suction. Every value is reported as a median with a "
        "90 % range (P5-P95), and the sources of uncertainty are separated."
    )
    d.para(
        "The tool does not assign hydrologic soil groups or curve numbers and "
        "does not set initial moisture conditions; those are separate steps "
        "that use these outputs. It is guideline-neutral: the values are "
        "estimates from published methods, and the user decides how they are "
        "applied under the applicable design standard."
    )

    # 3. Data -------------------------------------------------------------
    d.heading("3. Data", 2)
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
    )
    for p in products:
        if p.run.get("Citation"):
            d.para(p.run["Citation"], bold_lead=f"{p.name}:")
    if any(p.water_source for p in products):
        d.para(
            "Mapped water contents used only for the checks in section 8: "
            + "; ".join(p.water_source for p in products if p.water_source)
            + "."
        )
    d.para(
        f"Processing grid: {g.width} x {g.height} cells of {_sig(g.res, 3)} map "
        "units (the finest input grid unless a resolution was set), nearest-"
        "neighbour resampling. Coarser inputs repeat in blocks on this grid; "
        "the effective resolution is that of the coarsest input used."
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

    # 4. Data preparation -------------------------------------------------
    d.heading("4. Data preparation", 2)
    d.heading("4.1 Depth harmonisation", 3)
    d.para(
        "All inputs are averaged to the target layers. Interval products "
        "(SoilGrids 2.0: 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm; "
        "OpenLandMap: 0-30, 30-60, 60-100 cm) are combined by thickness "
        "weighting:"
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
    d.para(
        "Quantiles (SoilGrids Q0.05, Q0.95; OpenLandMap P16, P84) are averaged "
        "in the same way. This treats the intervals as fully correlated with "
        "depth, which keeps the layer uncertainty conservative (not reduced "
        "by averaging)."
    )
    d.heading("4.2 Texture systems", 3)
    d.para(
        "The pedotransfer functions use USDA particle-size limits (clay < 2 "
        "µm, silt 2-50 µm, sand 50 µm-2 mm), as do SoilGrids. OpenLandMap uses "
        "ISO 11277 limits (silt 2-63 µm). OpenLandMap texture is converted by "
        "log-linear interpolation of the cumulative particle-size "
        "distribution between 2 and 63 µm (Nemes et al., 1999; Minasny & "
        "McBratney, 2001):"
    )
    d.para(
        f"silt_USDA = k · silt_ISO,  k = ln(50/2) / ln(63/2) = {ISO_TO_USDA_SILT_FACTOR:.4f};  "
        "clay unchanged;  sand_USDA = 100 - clay - silt_USDA"
    )
    d.para(
        "About 7 % of the ISO silt is moved to sand. The conversion is applied "
        "to every Monte Carlo draw. Its own error (the true shape of the "
        "distribution between 50 and 63 µm) is not included in the uncertainty."
    )
    d.heading("4.3 Organic matter", 3)
    d.para(
        f"Organic matter is estimated from soil organic carbon: OM (%) = SOC "
        f"(g/kg) / 10 x {mc.om_factor:g}. The van Bemmelen factor 1.724 assumes "
        "58 % carbon in organic matter; values of 1.9-2.0 are often more "
        "realistic for surface soils, so the factor is a stated choice."
    )
    d.heading("4.4 Gap filling", 3)
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
        "cell is mapped in section 3 and in rsp_quality.tif."
    )
    fill_notes = [n for p in products for n in p.notes]
    for n in fill_notes:
        d.bullet(n)
    if both:
        d.heading("4.5 Product comparison", 3)
        d.para(
            "Both products were supplied and are used as equal ensemble members. "
            "They are compared on their central soils (USDA limits) where both "
            "have their own prediction."
        )
        sg, olm = names
        hdr = [
            "Layer",
            "Cells",
            "Clay SG (%)",
            "Clay OLM (%)",
            "Sand SG (%)",
            "Sand OLM (%)",
            "Same class (%)",
            "Ksat ratio OLM/SG",
            "Ksat differs > x4 (%)",
        ]
        d.caption("Table", "Product comparison (medians over cells)")
        d.table(
            hdr,
            [
                [
                    c["Layer"],
                    str(c["Cells"]),
                    _f(c["Clay SG (%)"], 1),
                    _f(c["Clay OLM (%)"], 1),
                    _f(c["Sand SG (%)"], 1),
                    _f(c["Sand OLM (%)"], 1),
                    _pct(c["Texture class agreement (%)"]),
                    _sig(c["Median Ksat ratio OLM / SG"], 2),
                    _pct(c["Ksat differs by more than x4 (%)"]),
                ]
                for c in result.comparison
            ],
        )
        if top in result.central.get(sg, {}):
            a, b = result.central[sg][top], result.central[olm][top]
            _figure(
                d,
                lambda: _continuous_maps(
                    result,
                    [
                        (b["clay"] - a["clay"], "Clay OLM - SG", "RdBu_r", False, "%"),
                        (b["sand"] - a["sand"], "Sand OLM - SG", "RdBu_r", False, "%"),
                        (
                            np.log10(np.maximum(b["ksat"], 1e-6))
                            - np.log10(np.maximum(a["ksat"], 1e-6)),
                            "log10 Ksat OLM/SG",
                            "RdBu_r",
                            False,
                            "-",
                        ),
                    ],
                ),
                f"Product differences, {top} (central soils, USDA limits)",
                notes,
            )
        smp = result.comparison_samples.get(top)
        if smp:
            _figure(
                d,
                lambda: _scatter_panels(
                    [
                        (
                            *smp["clay"],
                            "Clay SoilGrids (%)",
                            "Clay OpenLandMap (%)",
                            False,
                        ),
                        (
                            *smp["sand"],
                            "Sand SoilGrids (%)",
                            "Sand OpenLandMap (%)",
                            False,
                        ),
                        (
                            *smp["ksat"],
                            "Ksat SoilGrids (mm/h)",
                            "Ksat OpenLandMap (mm/h)",
                            True,
                        ),
                    ]
                ),
                f"Cell-by-cell product comparison, {top} (sample of cells)",
                notes,
            )

    # 5. Methods ------------------------------------------------------------
    d.heading("5. Methods", 2)
    d.para(
        "Each method is a pedotransfer function: a published regression from "
        "laboratory-measured soils that predicts hydraulic properties from "
        "texture, organic matter and, for some methods, bulk density. Every "
        "method runs on every Monte Carlo draw. A method is used outside its "
        "calibration range only with a flag (validity flags, rsp_quality.tif)."
    )
    sec = 0
    if "SR2006" in mc.methods:
        sec += 1
        d.heading(f"5.{sec} Saxton & Rawls (2006)", 3)
        d.para(
            "Calibration data: A-horizon samples of the USDA/NRCS National Soil "
            "Characterization database; 2149 samples reduced to 1722 by excluding "
            "bulk density below 1.0 or above 1.8 g/cm3, organic matter above 8 % "
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
            ("6", "ρN = (1 - θS) · 2.65  [g/cm3]"),
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
        d.table(["Eq.", "Equation"], [list(e) for e in eq], widths_cm=[1.6, 14.4])
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
    if "TOTH2015" in mc.methods:
        sec += 1
        d.heading(f"5.{sec} Tóth et al. (2015) / HiHydroSoil v2.0", 3)
        d.para(
            "European continuous pedotransfer functions (Tóth et al., 2015), "
            "calibrated on the EU-HYDI database of European soils. They are the "
            "method behind the global HiHydroSoil v2.0 maps (Simons et al., "
            "2020), which applied them to SoilGrids; this tool applies them to "
            "the harmonised inputs and adds the Monte Carlo uncertainty. The "
            "retention curve is Mualem-van Genuchten (van Genuchten, 1980). "
            "Inputs: clay and silt (%, USDA limits), organic carbon OC (%), "
            "bulk density BD (g/cm3), pH in water and CEC (cmol(c)/kg); T/S = 1 "
            "for topsoil (0-30 cm) and 0 below."
        )
        teq = [
            ("θr", "0.041 if sand >= 2 %, else 0.179  [m3/m3]"),
            ("θs", "0.83080 - 0.28217 BD + 0.0002728 Cl + 0.000187 Si  [m3/m3]"),
            ("α", "log10 α = -0.43348 - 0.41729 BD - 0.04762 OC + 0.21810 T/S - 0.01581 Cl - 0.01207 Si  [1/cm]"),
            ("n", "log10 (n - 1) = 0.22236 - 0.30189 BD - 0.05558 T/S - 0.005306 Cl - 0.003084 Si - 0.01072 OC"),
            ("Ksat", "log10 Ksat = 0.40220 + 0.26122 pH + 0.44565 T/S - 0.02329 Cl - 0.01265 Si - 0.01038 CEC  [cm/day]"),
            ("θ(h)", "θr + (θs - θr) / [1 + (α h)^n]^(1 - 1/n);  h = 336.5 cm (33 kPa), 15296 cm (1500 kPa)"),
        ]  # fmt: skip
        d.caption("Table", "Tóth et al. (2015) equations as used in HiHydroSoil v2.0")
        d.table(["Output", "Equation"], [list(e) for e in teq], widths_cm=[1.6, 14.4])
        d.para(
            "Verification: every coefficient was checked against the "
            "HiHydroSoil v2.0 report (Simons et al., 2020, pp. 7-8). Two "
            "deliberate differences from the HiHydroSoil product: field "
            "capacity is θ at 33 kPa here (HiHydroSoil publishes pF2, about 10 "
            "kPa, which gives wetter values), and the 0-30 cm Ksat is computed "
            "from layer-averaged inputs (HiHydroSoil averages the depth values "
            "harmonically). Ksat needs CEC, which only SoilGrids provides; "
            "with both products the OpenLandMap member uses the SoilGrids CEC. "
            "Calibration region: Europe. Applied elsewhere, the functions "
            "extrapolate, and their spread against the other method is part of "
            "the reported uncertainty."
        )
    sec += 1
    d.heading(f"5.{sec} Green-Ampt wetting-front suction", 3)
    d.para(
        "For Brooks-Corey type methods the wetting-front suction follows from "
        "the air-entry (bubbling) suction ψb and the pore-size distribution "
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
        "follows the air-entry suction and gives larger values. The Green-Ampt "
        "moisture deficit, Δθ = effective porosity - initial water content, "
        "depends on antecedent conditions and is set in the infiltration "
        "model, not here. Saxton & Rawls have no residual water content, so "
        "the effective porosity equals θS for this method. Some models use an "
        "effective conductivity of about 0.5 Ksat for the wetted zone "
        "(Bouwer, 1966); the Ksat outputs are not reduced."
    )
    sec += 1
    d.heading(f"5.{sec} Reference check: Rawls, Brakensiek & Miller (1983)", 3)
    d.para(
        "Green-Ampt parameters tabulated per USDA texture class (as reproduced "
        "in Chow et al., 1988, Table 4.3.1) are compared with the tool's "
        "medians in section 8. They are class means of a large US data set, "
        "not an ensemble member. The tabulated K is a Green-Ampt hydraulic "
        "conductivity, roughly half of the saturated conductivity, and the "
        "table has no Silt class."
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
    )
    sec += 1
    d.heading(f"5.{sec} Validity flags", 3)
    d.table(
        ["Bit", "Meaning"],
        [
            [str(k), v]
            for k, v in {**saxton_rawls.FLAG_LABELS, **toth2015.FLAG_LABELS}.items()
        ],
        widths_cm=[1.6, 14.4],
    )
    d.para(
        "Flags are evaluated on each product's central soil and combined. "
        "With a single method the method share of the uncertainty is zero by "
        "construction; select both methods to see the method spread."
    )

    # 6. Uncertainty ------------------------------------------------------------
    d.heading("6. Uncertainty", 2)
    d.para(
        "Inputs are converted to per-cell distributions on the same footing. "
        "Texture fractions and organic carbon are treated on a log scale, "
        "bulk density and coarse fragments on a linear scale. For SoilGrids "
        "the centre is the median Q0.5 and the 90 % interval Q0.05-Q0.95 gives "
        "separate lower and upper standard deviations, σ_lo = (T(Q0.5) - "
        "T(Q0.05)) / 1.645 and σ_hi = (T(Q0.95) - T(Q0.5)) / 1.645 (a split "
        "normal). For OpenLandMap the centre is the 30 m mean and the 120 m "
        "68 % interval gives σ = (T(P84) - T(P16)) / (2 x 0.994) on both sides."
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
        "Variance split (law of total variance, log10 scale for Ksat and the "
        "suctions): input data = mean variance within each method-product "
        "group; product and method = variance of the group means across "
        "products or methods; interaction = the remainder. Correlations "
        "between inputs and between neighbouring cells are not modelled, and "
        "the products' own interval estimates are taken at face value."
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

    # 7. Results ------------------------------------------------------------
    d.heading("7. Results", 2)
    med_maps = [
        (result.stats["ksat"][top][1], "Ksat P50", "viridis", True, "mm/h"),
        (result.stats["psi_f"][top][1], "ψf P50", "magma_r", True, "mm"),
        (result.stats["theta_s"][top][1], "θS P50", "Blues", False, "m3/m3"),
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
            (result.stats["theta_fc"][top][1], "θ33 P50", "Blues", False, "m3/m3"),
            (result.stats["paw"][top][1], "PAW P50", "Greens", False, "m3/m3"),
        ]
    _figure(
        d,
        lambda: _continuous_maps(result, rng_maps),
        f"Uncertainty of Ksat and water-holding maps, {top}",
        notes,
    )
    for zone in result.zone_names:
        d.caption("Table", f"{zone}: median of cell medians, with median cell P5-P95")
        hdr = ["Parameter", "Units"] + labs
        rows = []
        for prm in PARAMETERS:
            rows.append(
                [prm.label, prm.units]
                + [
                    _value_range(_row(result, zone, lab, prm.code), prm.code)
                    for lab in labs
                ]
            )
        d.table(hdr, rows)
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
    for lab in labs[:1]:
        _figure(
            d,
            lambda lab=lab: _texture_triangle(result, lab),
            f"Soils of the area on the USDA texture triangle, {lab}",
            notes,
        )
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
    )
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
    )
    _figure(
        d,
        lambda: _variance_bars(result, 1, top),
        f"Sources of uncertainty, {result.zone_names[0]}, {top}",
        notes,
    )

    # 8. Checks -------------------------------------------------------------
    d.heading("8. Checks", 2)
    d.heading("8.1 Mapped water contents", 3)
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
                "Mapped",
                "PTF",
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
        )
        panels = []
        for key, (obs, pred) in result.check_samples.items():
            if key[1] == top and len(panels) < 2:
                q = "θ33" if key[2] == "fc" else "θ1500"
                panels.append(
                    (obs, pred, f"Mapped {q} (m3/m3)", f"PTF {q} (m3/m3)", False)
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
    d.heading("8.2 Rawls, Brakensiek & Miller (1983)", 3)
    d.caption(
        "Table", "Tool medians against the class values of the dominant texture class"
    )
    d.table(
        [
            "Zone",
            "Layer",
            "Class",
            "Rawls φ",
            "Tool θS",
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
    )
    d.para(
        "The Ksat ratio is the median over cells of the tool's Ksat divided by "
        "the class K of each cell's texture class. A ratio near 2 is expected "
        "because the tabulated K is about half of Ksat; much larger or smaller "
        "ratios show where the methods differ and deserve attention."
    )

    # 9. Limitations ----------------------------------------------------------
    d.heading("9. Limitations", 2)
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

    # 10. References --------------------------------------------------------
    d.heading("10. References", 2)
    for ref in REFERENCES:
        d.para(ref)

    # Annex -------------------------------------------------------------------
    d.heading("Annex A. Run settings", 2)
    d.table(
        ["Item", "Value"],
        [
            ["Tool", "Regional soil parameterisation (Mayim Tools)"],
            ["Run time", result.run_time_utc],
            ["Products", ", ".join(names)],
            ["Grid", f"{g.width} x {g.height} cells at {_sig(g.res, 3)}"],
            ["Depth layers", ", ".join(labs)],
            ["Methods", ", ".join(m.name for m in methods)],
            ["Monte Carlo draws per product", str(mc.draws)],
            ["Random seed", str(mc.seed)],
            ["OM factor", f"{mc.om_factor:g}"],
            ["Density adjustment", "yes" if mc.density else "no"],
            ["Gravel correction", "yes" if mc.gravel else "no"],
            ["Neighbour fill radius (m)", _f(s.fill_radius_m, 0)],
            ["Zones", ", ".join(result.zone_names)],
        ],
        widths_cm=[5.5, 10.5],
    )
    d.heading("Annex B. Output files", 2)
    import os

    d.table(
        ["File", "Content"],
        [[os.path.basename(f), desc] for f, desc in result.files]
        + [["rsp_metadata.csv", "Run metadata (settings, inputs, checks, warnings)"]],
        widths_cm=[6.0, 10.0],
    )
    d.para(
        "Each parameter file holds the P50, P05 and P95 for every layer; select "
        "bands by their description (e.g. ksat_0-30cm_P50 (mm/h)), never by "
        "band number."
    )
    warn = list(result.warnings) + notes
    if warn:
        d.heading("Annex C. Warnings", 2)
        for w in warn:
            d.bullet(w)
    d.doc.save(path)
