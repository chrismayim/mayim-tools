"""Word (.docx) report for the stream network tool - zero QGIS dependency.
python-docx and matplotlib are imported lazily (see
hydrology/_common/docx_report.py). Numbers are written without thousands
separators."""

from __future__ import annotations

import io
import math
from collections.abc import Sequence

import numpy as np

from mayim_tools._common.docx_report import (
    C_BLUE,
    C_INK,
    C_INK_2,
    C_ORANGE,
    Doc,
    new_figure,
    png,
    style_axes,
)

from .core import NetworkResult
from .export import FIT_FIELDS, NETWORK_FIELDS, ORDER_FIELDS, REACH_FIELDS

REPORT_TITLE = "Stream Network Extraction and Classification"
MAX_NETWORK_SECTIONS = 5

REFERENCES = [
    "Chow, V.T. (1959). Open-Channel Hydraulics. McGraw-Hill, New York.",
    "Hack, J.T. (1957). Studies of longitudinal stream profiles in Virginia "
    "and Maryland. U.S. Geological Survey Professional Paper 294-B.",
    "Horton, R.E. (1932). Drainage-basin characteristics. Transactions of "
    "the American Geophysical Union, 13, 350-361.",
    "Horton, R.E. (1945). Erosional development of streams and their "
    "drainage basins. Geological Society of America Bulletin, 56, 275-370.",
    "McCarthy, G.T. (1938). The unit hydrograph and flood routing. "
    "Conference of the North Atlantic Division, U.S. Army Corps of Engineers.",
    "O'Callaghan, J.F. & Mark, D.M. (1984). The extraction of drainage "
    "networks from digital elevation data. Computer Vision, Graphics, and "
    "Image Processing, 28, 323-344.",
    "Rodriguez-Iturbe, I. & Rinaldo, A. (1997). Fractal River Basins: Chance "
    "and Self-Organization. Cambridge University Press.",
    "Shreve, R.L. (1966). Statistical law of stream numbers. Journal of "
    "Geology, 74, 17-37.",
    "Strahler, A.N. (1957). Quantitative analysis of watershed "
    "geomorphology. Transactions of the American Geophysical Union, 38(6), "
    "913-920.",
    "Strahler, A.N. (1964). Quantitative geomorphology of drainage basins "
    "and channel networks. In V.T. Chow (ed.), Handbook of Applied "
    "Hydrology, Section 4-II. McGraw-Hill, New York.",
    "Tarboton, D.G., Bras, R.L. & Rodriguez-Iturbe, I. (1991). On the "
    "extraction of channel networks from digital elevation data. "
    "Hydrological Processes, 5, 81-100.",
    "USACE-HEC (2000). Hydrologic Modeling System HEC-HMS Technical "
    "Reference Manual. U.S. Army Corps of Engineers, Hydrologic Engineering "
    "Center, Davis, CA.",
]


def fmt(value, digits=2, unit=""):
    """Plain number formatting - no thousands separators."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value}{unit}"
    return f"{value:.{digits}f}{unit}"


def _order_colors(max_order: int):
    from matplotlib import colormaps

    cmap = colormaps["Blues"]
    if max_order <= 1:
        return {1: cmap(0.85)}
    return {
        u: cmap(0.5 + 0.5 * (u - 1) / (max_order - 1)) for u in range(1, max_order + 1)
    }


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def describe_network(net: dict, fit: dict | None, reaches: Sequence[dict]) -> list[str]:
    """Guideline-neutral description paragraphs for one network."""
    paras = [
        f"Network {net['network_id']} drains {fmt(net['area_km2'], 3)} km² to its "
        f"outlet at E {fmt(net['outlet_x'], 1)}, N {fmt(net['outlet_y'], 1)}. It has "
        f"{net['n_reaches']} reaches fed by {net['n_heads']} channel heads (Shreve "
        f"magnitude {net['magnitude']}), reaches Strahler order "
        f"{net['max_strahler']}, and holds {fmt(net['total_length_km'], 2)} km of "
        f"channel: a drainage density of {fmt(net['drainage_density'], 2)} km/km² "
        f"and a stream frequency of {fmt(net['stream_frequency'], 2)} heads per km²."
    ]
    if fit and fit.get("rb_fit") is not None:
        rb = fit["rb_fit"]
        text = (
            f"Horton's laws. The bifurcation ratio is {fmt(rb, 2)} from the "
            f"regression (R² {fmt(fit['rb_r2'], 2)}) and {fmt(fit['rb_mean'], 2)} "
            f"as a mean of consecutive orders; the length ratio is "
            f"{fmt(fit['rl_fit'], 2)} and the area ratio {fmt(fit['ra_fit'], 2)}. "
            "Natural networks typically give a bifurcation ratio of about 3 to 5, "
            "a length ratio of 1.5 to 3.5 and an area ratio of 3 to 6 "
            "(Rodriguez-Iturbe & Rinaldo 1997)"
        )
        if rb > 5:
            text += (
                "; the higher bifurcation ratio here suggests an elongated "
                "network or structural (geological) control."
            )
        elif rb < 3:
            text += (
                "; the lower bifurcation ratio here suggests a compact network "
                "that concentrates flow quickly."
            )
        else:
            text += "; this network sits within that range."
        if fit["max_order"] < 3:
            text += (
                " With fewer than three orders the ratios rest on very few "
                "streams and are indicative only."
            )
        paras.append(text)
    if net.get("hack_h") is not None:
        h = net["hack_h"]
        paras.append(
            f"Hack's law. Fitting main-stream length against upstream area over "
            f"the reaches gives L = {fmt(net['hack_c'], 2)} A^{fmt(h, 2)} "
            f"(L km, A km², R² {fmt(net['hack_r2'], 2)}). Hack (1957) found an "
            "exponent of about 0.6, and values of 0.5 to 0.6 are common; "
            + (
                "the larger exponent here means the network lengthens faster "
                "than it widens as it grows."
                if h > 0.65
                else (
                    "a smaller exponent means a network that widens as it grows."
                    if h < 0.45
                    else "this network follows that typical scaling."
                )
            )
        )
    paras.append(
        f"Main stem. The longest stream is {fmt(net['main_length_km'], 3)} km long "
        f"with {fmt(net['main_relief_m'], 1)} m of fall: an average slope of "
        f"{fmt(_pct(net['main_slope_avg']), 2)} % and a 10-85 slope of "
        f"{fmt(_pct(net['main_slope_1085']), 2)} %. Its sinuosity is "
        f"{fmt(net['main_sinuosity'], 2)} (D8 paths run a few per cent longer than "
        f"the true channel), and the summed reach travel time along it is "
        f"{fmt(net['main_tt_min'], 1)} min with the routing settings used."
    )
    return paras


def _pct(v):
    return None if v is None else 100.0 * v


def modelling_points(net: dict, reaches: Sequence[dict], settings) -> list[str]:
    mine = [r for r in reaches if r["network_id"] == net["network_id"]]
    pts = []
    short = [r for r in mine if r.get("dt_check", "").startswith("dt above")]
    if mine:
        pts.append(
            f"Of the {len(mine)} reaches, {len(short)} have a travel time shorter "
            f"than the {fmt(settings.time_step_min, 1)} min time step can route "
            "with Muskingum without numerical attenuation. In a node-link model, "
            "merge short reaches, use lag or no routing for them, or shorten the "
            "time step (USACE-HEC 2000)."
        )
    pts.append(
        "Routing velocities come from "
        + (
            f"a constant {fmt(settings.velocity_ms, 2)} m/s"
            if settings.method == "constant"
            else f"Manning's equation with n = {fmt(settings.manning_n, 3)} and a "
            f"hydraulic radius of {fmt(settings.hydraulic_radius_m, 2)} m"
        )
        + ". Muskingum K equals the reach travel time and is sensitive to that "
        "velocity: confirm it against surveyed sections, observed hydrographs or "
        "the governing guideline before design use."
    )
    pts.append(
        "Use the reach table to set up sub-catchment outlets at confluences of "
        "higher-order streams, where flows combine, and at points of interest; "
        "upstream areas give the drainage area at each node directly."
    )
    pts.append(
        "Check the network against imagery and mapped watercourses. In flat "
        "or urban areas a DEM-derived network can follow roads, drains or "
        "artefacts, and structures such as culverts are not represented."
    )
    return pts


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_drop_test(result: NetworkResult) -> io.BytesIO | None:
    rows = [r for r in result.drop_test if r["t_statistic"] is not None]
    if len(rows) < 2:
        return None
    fig = new_figure(7.0)
    ax = fig.add_subplot(1, 1, 1)
    x = [r["threshold_km2"] for r in rows]
    t = [abs(r["t_statistic"]) for r in rows]
    ax.axhspan(0, 2, color=C_BLUE, alpha=0.08, linewidth=0)
    ax.axhline(2, color=C_INK_2, linewidth=0.8, linestyle="--")
    ax.plot(x, t, color=C_BLUE, linewidth=1.6, marker="o", markersize=4)
    thr = result.settings.get("threshold_km2")
    if thr:
        ax.axvline(thr, color=C_ORANGE, linewidth=1.2, label="Threshold used")
    ax.set_xscale("log")
    ax.set_xlabel("Stream threshold (km²)")
    ax.set_ylabel("|t| first-order vs higher-order drops")
    ax.text(
        0.99, 0.04, "|t| < 2 passes", transform=ax.transAxes, ha="right",
        fontsize=7, color=C_INK_2,
    )  # fmt: skip
    style_axes(ax)
    if thr:
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    return png(fig)


def figure_network_map(result: NetworkResult, network_id: str | None = None):
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    reaches = [
        r for r in result.reaches if network_id is None or r["network_id"] == network_id
    ]
    if not reaches:
        return None
    max_u = max(r["strahler"] for r in reaches)
    colors = _order_colors(max_u)
    fig = new_figure(13.0)
    ax = fig.add_subplot(1, 1, 1)
    for u in range(1, max_u + 1):
        segs = [r["coords"] for r in reaches if r["strahler"] == u]
        if segs:
            ax.add_collection(
                LineCollection(
                    segs,
                    colors=[colors[u]],
                    linewidths=0.6 + 0.6 * (u - 1),
                    capstyle="round",
                )
            )
    outs = [n for n in result.nodes if n["node_type"] == "outlet"]
    if network_id is not None:
        outs = [n for n in outs if n["network_id"] == network_id]
    ax.plot(
        [n["x"] for n in outs], [n["y"] for n in outs], "v", color=C_INK,
        markersize=7, linestyle="none",
    )  # fmt: skip
    handles = [
        Line2D(
            [0], [0], color=colors[u], linewidth=0.6 + 0.6 * (u - 1), label=f"Order {u}"
        )
        for u in range(1, max_u + 1)
    ]
    handles.append(
        Line2D([0], [0], marker="v", color=C_INK, linestyle="none", label="Outlet")
    )
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    style_axes(ax)
    ax.ticklabel_format(useOffset=False, style="plain")
    return png(fig)


def figure_horton(order_rows: Sequence[dict]) -> io.BytesIO | None:
    if len(order_rows) < 2:
        return None
    fig = new_figure(5.5)
    u = [r["order"] for r in order_rows]
    panels = [
        ("n_streams", "Number of streams N_u"),
        ("mean_length_km", "Mean length (km)"),
        ("mean_area_km2", "Mean area (km²)"),
    ]
    for i, (key, label) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i)
        v = [r[key] for r in order_rows]
        ok = [(a, b) for a, b in zip(u, v, strict=True) if b and b > 0]
        if len(ok) >= 2:
            xs, ys = zip(*ok, strict=True)
            ax.plot(xs, ys, "o", color=C_BLUE, markersize=5)
            b, a = np.polyfit(xs, np.log10(ys), 1)
            xx = np.array([min(xs), max(xs)])
            ax.plot(xx, 10 ** (a + b * xx), color=C_ORANGE, linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xticks(u)
        ax.set_xlabel("Strahler order u")
        ax.set_title(label, fontsize=8, color=C_INK)
        style_axes(ax)
    fig.subplots_adjust(wspace=0.45)
    return png(fig)


def figure_profile(result: NetworkResult, network_id: str) -> io.BytesIO | None:
    rows = [p for p in result.profiles.get(network_id, []) if p["z_m"] is not None]
    if len(rows) < 2:
        return None
    fig = new_figure(7.0)
    ax = fig.add_subplot(1, 1, 1)
    d = np.array([p["dist_from_outlet_m"] for p in rows]) / 1000.0
    z = np.array([p["z_m"] for p in rows])
    o = np.argsort(d)
    ax.plot(d[o], z[o], color=C_BLUE, linewidth=1.6)
    ax.set_xlim(0, d.max())
    ax.set_xlabel("Distance upstream from outlet (km)")
    ax.set_ylabel("Elevation (m)")
    style_axes(ax)
    return png(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report_docx(path: str, result: NetworkResult, context: dict) -> None:
    d = Doc()
    s = result.settings
    routing = s["routing"]
    try:
        import matplotlib  # noqa: F401

        figures = True
    except ImportError:
        figures = False

    nets = sorted(result.networks, key=lambda n: -n["area_km2"])
    fits = {f["network_id"]: f for f in result.horton_fits}

    d.heading(REPORT_TITLE, 1)

    # 1. Summary
    d.heading("1. Summary", 2)
    d.para(
        f"This annexure documents the extraction and classification of the "
        f"stream network from a DEM with a {s['cell_size'][0]:g} m x "
        f"{s['cell_size'][1]:g} m cell size, using the "
        f"{context.get('tool', 'Stream Network')} tool (Mayim Tools, version "
        f"{context.get('version', '-')}). Streams start where the upstream area "
        f"reaches {fmt(s['threshold_km2'], 4)} km²"
        + (
            f"; the constant-drop test gives {fmt(result.drop_test_threshold_km2, 4)} "
            "km² as the smallest consistent threshold"
            if result.drop_test_threshold_km2 is not None
            else ""
        )
        + f". {len(nets)} network(s) with {len(result.reaches)} reaches were "
        "extracted. Every reach is classified by Strahler, Shreve, Horton and Hack "
        "ordering and carries geometry, upstream area and channel-routing inputs."
    )
    d.caption("Table", "Summary of drainage networks")
    d.table(
        ["Network", "Area (km²)", "Reaches", "Max order", "Magnitude",
         "Length (km)", "Dd (km/km²)", "Main stem (km)"],  # fmt: skip
        [
            [
                n["network_id"],
                fmt(n["area_km2"], 3),
                str(n["n_reaches"]),
                str(n["max_strahler"]),
                str(n["magnitude"]),
                fmt(n["total_length_km"], 2),
                fmt(n["drainage_density"], 2),
                fmt(n["main_length_km"], 2),
            ]
            for n in nets
        ],
    )
    if figures:
        stream = figure_network_map(result)
        if stream is not None:
            d.picture(stream, "Stream network classified by Strahler order")

    # 2. Method
    d.heading("2. Tool and method", 2)
    for step in [
        "Flow follows the D8 (eight-direction) pointer (O'Callaghan & Mark "
        "1984); flow accumulation is computed in topological order.",
        "With pour points, each is snapped to the highest flow accumulation "
        f"within {s['snap_radius_cells']} cells and the network is everything "
        "upstream of them; intermediate pour points also split reaches. "
        "Without pour points, every drainage network on the raster is "
        "extracted, one per outlet cell (raster edge, NoData edge or pit).",
        "A cell is a stream cell when its upstream area reaches the stream "
        "threshold. A constant-drop test (Tarboton et al. 1991) over a range of "
        "thresholds checks that choice (section 4).",
        "Stream cells are split into reaches at every confluence and pour "
        "point. Each reach is ordered by the Strahler (1957), Shreve (1966), "
        "Horton (1945) and Hack (1957) systems.",
        "Reaches of the same Strahler order are chained into Strahler streams "
        "to give Horton's stream statistics and ratios per network.",
        "Each reach gets a routing velocity, travel time and Muskingum "
        "parameters (McCarthy 1938) for node-link hydrological models.",
    ]:
        d.numbered(step)

    # 3. Inputs
    d.heading("3. Inputs and settings", 2)
    d.caption("Table", "Inputs and processing settings")
    d.table(
        ["Item", "Value"],
        [
            ["DEM", context.get("dem_path", "-")],
            ["D8 pointer", context.get("pointer_path", "-")],
            ["Pointer encoding", context.get("encoding", "-")],
            ["Coordinate reference system", context.get("crs", "-")],
            ["Cell size", f"{s['cell_size'][0]:g} m x {s['cell_size'][1]:g} m"],
            ["Grid", f"{s['grid'][0]} columns x {s['grid'][1]} rows"],
            ["Pour points used", str(s["n_pour_points"])],
            ["Stream threshold", f"{fmt(s['threshold_km2'], 4)} km²"],
            [
                "Drop-test range",
                (
                    "not run"
                    if not s["drop_test_range"]
                    else f"{s['drop_test_range'][0]:g} to {s['drop_test_range'][1]:g} "
                    f"km², {s['drop_test_range'][2]} steps"
                ),
            ],
            ["Minimum network area", f"{s['min_network_km2']:g} km²"],
            [
                "Routing velocity",
                (
                    f"constant {routing.velocity_ms:g} m/s"
                    if routing.method == "constant"
                    else f"Manning, n = {routing.manning_n:g}, R = "
                    f"{routing.hydraulic_radius_m:g} m, minimum slope "
                    f"{routing.min_slope:g} m/m"
                ),
            ],
            ["Muskingum X", f"{routing.muskingum_x:g}"],
            ["Computational time step", f"{routing.time_step_min:g} min"],
            ["Run", context.get("run_time", "-")],
        ],
        widths_cm=[5.0, 11.0],
    )

    # 4. Drop test
    d.heading("4. Stream threshold and constant-drop test", 2)
    d.para(
        "Horton's law of stream drops says Strahler streams of every order "
        "fall by about the same amount on average. Tarboton et al. (1991) use "
        "this to find the smallest support area that still gives a "
        "geomorphologically consistent network: below it, first-order "
        "'streams' extend into hillslopes and their mean drop differs "
        "significantly from that of higher-order streams. For each threshold "
        "the mean drop of first-order streams is compared with that of all "
        "higher-order streams with a two-sample t-test; the smallest threshold "
        "with |t| < 2 passes."
    )
    if result.drop_test:
        d.caption("Table", "Constant-drop test")
        d.table(
            ["Threshold (km²)", "Max order", "N first", "N higher",
             "Mean drop 1st (m)", "Mean drop higher (m)", "t", "Passes"],  # fmt: skip
            [
                [
                    fmt(r["threshold_km2"], 4),
                    str(r["max_order"]),
                    str(r["n_first_order"]),
                    str(r["n_higher_order"]),
                    fmt(r["mean_drop_first_m"], 2),
                    fmt(r["mean_drop_higher_m"], 2),
                    fmt(r["t_statistic"], 2),
                    "yes" if r["passes"] else "no",
                ]
                for r in result.drop_test
            ],
        )
        if figures:
            stream = figure_drop_test(result)
            if stream is not None:
                d.picture(stream, "Constant-drop test: |t| against stream threshold")
        thr, rec = s["threshold_km2"], result.drop_test_threshold_km2
        if rec is None:
            d.para(
                "No threshold in the tested range passed. Widen the range or "
                "check the DEM conditioning before relying on the network's "
                "first-order streams."
            )
        elif thr < rec:
            d.para(
                f"The threshold used ({fmt(thr, 4)} km²) is below the smallest "
                f"passing threshold ({fmt(rec, 4)} km²), so first-order streams "
                "probably extend further upslope than the channels do. This can "
                "be acceptable for drainage design, where small overland flow "
                "paths matter, but treat first-order statistics with care."
            )
        else:
            d.para(
                f"The threshold used ({fmt(thr, 4)} km²) is at or above the "
                f"smallest passing threshold ({fmt(rec, 4)} km²)."
            )
    else:
        d.para("The drop test was not run.")

    # 5. Ordering systems
    d.heading("5. Stream ordering systems", 2)
    for lead, text in [
        ("Strahler order.", "Headwater reaches are order 1; where two reaches of "
         "the same order meet, the order increases by one; otherwise the higher "
         "order continues (Strahler 1957)."),
        ("Shreve magnitude.", "The number of channel heads upstream: headwater "
         "reaches have magnitude 1 and magnitudes add at every confluence "
         "(Shreve 1966). It scales roughly with drainage area and discharge."),
        ("Horton order.", "Strahler orders re-assigned so that the main stream "
         "carries the order of the downstream reach all the way to its head; at "
         "each confluence the main stream is the higher-order branch, then the "
         "longer one (Horton 1945)."),
        ("Hack order.", "The main stem (longest stream to the outlet) is order "
         "1, streams flowing into it are order 2, their tributaries 3, and so on "
         "(Hack 1957)."),
    ]:  # fmt: skip
        d.para(text, bold_lead=lead)

    # 6. Networks
    d.heading("6. Network results", 2)
    if len(nets) > MAX_NETWORK_SECTIONS:
        d.para(
            f"The {MAX_NETWORK_SECTIONS} largest of {len(nets)} networks are "
            "described below; all networks are listed in the CSV output."
        )
    for i, net in enumerate(nets[:MAX_NETWORK_SECTIONS], start=1):
        nid = net["network_id"]
        d.heading(f"6.{i} Network {nid}", 3)
        for paragraph in describe_network(net, fits.get(nid), result.reaches):
            lead, sep, rest = paragraph.partition(". ")
            if lead in ("Horton's laws", "Hack's law", "Main stem"):
                d.para(rest, bold_lead=lead + ".")
            else:
                d.para(paragraph)
        rows = [r for r in result.order_stats if r["network_id"] == nid]
        if rows:
            d.caption("Table", f"Horton statistics by Strahler order, network {nid}")
            d.table(
                ["Order", "N_u", "Mean length (km)", "Mean area (km²)",
                 "Mean slope (%)", "R_b", "R_L", "R_A"],  # fmt: skip
                [
                    [
                        str(r["order"]),
                        str(r["n_streams"]),
                        fmt(r["mean_length_km"], 3),
                        fmt(r["mean_area_km2"], 3),
                        fmt(_pct(r["mean_slope"]), 2),
                        fmt(r["rb_next"], 2),
                        fmt(r["rl_next"], 2),
                        fmt(r["ra_next"], 2),
                    ]
                    for r in rows
                ],
            )
        mine = [r for r in result.reaches if r["network_id"] == nid]
        max_u = max(r["strahler"] for r in mine)
        route_rows = []
        for u in range(1, max_u + 1):
            ru = [r for r in mine if r["strahler"] == u]
            if not ru:
                continue
            slopes = [r["slope"] for r in ru if r["slope"] is not None]
            vs = [r["v_ms"] for r in ru if r["v_ms"] is not None]
            route_rows.append(
                [
                    str(u),
                    str(len(ru)),
                    fmt(sum(r["length_m"] for r in ru) / 1000.0, 3),
                    fmt(float(np.mean([r["length_m"] for r in ru])), 1),
                    fmt(_pct(float(np.mean(slopes))) if slopes else None, 2),
                    fmt(float(np.mean(vs)) if vs else None, 2),
                    fmt(float(np.mean([r["tt_min"] or 0 for r in ru])), 2),
                ]
            )
        d.caption(
            "Table", f"Reaches and routing inputs by Strahler order, network {nid}"
        )
        d.table(
            ["Order", "Reaches", "Length (km)", "Mean reach (m)", "Mean slope (%)",
             "Mean v (m/s)", "Mean travel time (min)"],  # fmt: skip
            route_rows,
        )
        if figures:
            for stream, cap in (
                (figure_network_map(result, nid), "stream network by Strahler order"),
                (
                    figure_horton(rows),
                    "Horton plots (points: per-order means; line: log-linear fit)",
                ),
                (figure_profile(result, nid), "main-stem longitudinal profile"),
            ):
                if stream is not None:
                    d.picture(stream, f"Network {nid}: {cap}")
        d.label("Hydrological assessment and modelling considerations")
        for p in modelling_points(net, result.reaches, routing):
            d.bullet(p)

    # 7. Definitions
    d.heading("7. Output fields", 2)
    d.caption("Table", "Reach attributes (vector layer and CSV section 6)")
    d.table(
        ["Field", "Unit", "Definition"],
        [[n, u, desc] for n, _t, u, desc in REACH_FIELDS],
        widths_cm=[3.0, 1.8, 11.2],
    )
    d.caption("Table", "Network, Horton and ratio fields (CSV sections 3 to 5)")
    d.table(
        ["Field", "Unit", "Definition"],
        [[n, u, desc] for n, u, desc in NETWORK_FIELDS + ORDER_FIELDS + FIT_FIELDS],
        widths_cm=[3.4, 1.8, 10.8],
    )

    # 8. Outputs
    outputs = context.get("outputs", [])
    if outputs:
        d.heading("8. Outputs", 2)
        d.caption("Table", "Files produced by the tool")
        d.table(["Output", "Location"], [list(o) for o in outputs], widths_cm=[5, 11])

    # 9. Notes
    d.heading("9. Notes and limitations", 2)
    for note in [
        "The network is only as good as the DEM and its conditioning. In flat "
        "areas small elevation errors can redirect streams.",
        "A single area threshold assumes channel initiation does not vary with "
        "slope, geology or land cover across the area.",
        "D8 routing cannot split flow, so braided, distributary and fan "
        "channels are reduced to single paths.",
        "Routing parameters are first estimates from reach geometry and the "
        "chosen velocity settings, not calibrated values.",
    ]:
        d.bullet(note)
    if result.warnings:
        d.para("Processing messages recorded during this run:")
        for w in result.warnings:
            d.bullet(w)

    d.heading("10. References", 2)
    for ref in REFERENCES:
        d.para(ref)
    d.doc.save(path)
