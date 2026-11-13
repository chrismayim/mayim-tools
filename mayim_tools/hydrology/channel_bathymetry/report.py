"""Word (.docx) report for DEM Channel Bathymetry, in the Mayim report and
figure standard (Doc(house_style=True), title page, ChartStyle figures at
16.5 cm, legends below the plot, no thousands separators).

Zero QGIS dependency; python-docx and matplotlib are imported lazily.
Figures use matplotlib's object API (no pyplot), safe on QGIS's
Processing worker thread.
"""

from __future__ import annotations

import io

import numpy as np

from mayim_tools._common.docx_report import Doc
from mayim_tools._common.report_figures import CHART_COLOURS, CHART_INK, ChartStyle
from mayim_tools.hydrology.catchment_delineation.report import (
    _north_arrow,
    _plan_height_cm,
    _scale_bar,
)
from mayim_tools.hydrology.stream_network.report import title_details

from .core import BASIS_TEXT, CAP_TEXT, SCENARIOS, ChannelResult, _sample, _to_rc

REPORT_TITLE = "DEM Channel Bathymetry"
C_RED, C_AMBER, C_YELLOW, C_GREEN, C_BLUE, C_CYAN, C_MAGENTA = CHART_COLOURS
FILL_ALPHA = 0.12

REFERENCES = [
    "Bureš, L., Sychová, P., Máča, P., Roub, R. & Marval, S. (2019). River "
    "bathymetry model based on floodplain topography. Water, 11(6), 1287.",
    "Dey, S. (2016). Role of river bathymetry in hydraulic modeling of river "
    "channels. MS thesis, Purdue University.",
    "Dingman, S.L. (2007). Analytical derivation of at-a-station "
    "hydraulic-geometry relations. Journal of Hydrology, 334, 17-27.",
    "Hollister, J.W., Milstead, W.B. & Urrutia, M.A. (2011). Predicting maximum "
    "lake depth from surrounding topography. PLoS ONE, 6(9), e25764.",
    "Merwade, V. & Maidment, D. (2005). River Channel Morphology Model. Center "
    "for Research in Water Resources, University of Texas at Austin.",
    "Ottevanger, W., Blanckaert, K. & Uijttewaal, W.S.J. (2012). Processes "
    "governing the flow redistribution in sharp river bends. Geomorphology, "
    "163-164, 45-55.",
]


def _f(value, fmt: str) -> str:
    try:
        return "-" if value is None or not np.isfinite(value) else fmt.format(value)
    except TypeError:
        return "-"


def describe(result: ChannelResult) -> str:
    sm = result.summary()
    return (
        f"Along {sm['length_m']:.0f} m of centreline, {sm['stations_with_water']} "
        f"of {sm['stations']} stations found water {_f(sm['width_min_m'], '{:.1f}')} "
        f"to {_f(sm['width_max_m'], '{:.1f}')} m wide. The estimated channel holds "
        f"{sm['volume_central_m3']:.0f} m³ below the surveyed water surface "
        f"(low {sm['volume_low_m3']:.0f} m³, high {sm['volume_high_m3']:.0f} m³) "
        f"and is up to {_f(sm['max_depth_central_m'], '{:.2f}')} m deep "
        f"(low {_f(sm['max_depth_low_m'], '{:.2f}')} m, high "
        f"{_f(sm['max_depth_high_m'], '{:.2f}')} m)."
    )


def parameter_rows(result: ChannelResult) -> list[list[str]]:
    s = result.settings
    sm = result.summary()
    return [
        ["Station spacing, Δs (m)", f"{s.spacing:g}",
         "Distance between the sections along each centreline."],
        ["Search half-width, H (m)", f"{s.search:g}",
         "How far each section reaches on either side of the centreline."],
        ["Flat-water tolerance, δ (m)", f"{s.flat_tol:g}",
         "A cell is water when |z - z_ref| ≤ δ, z_ref being the running median "
         "of the DEM under the centreline (never rising downstream)."],
        ["Maximum water slope (m/m)", f"{s.max_water_slope:g}",
         "Water cells must be flatter than this (smaller of the forward and "
         "backward differences, so cells on the edge of the flat count)."],
        ["Bank-top break slope (m/m)", f"{s.break_slope:g}",
         "Walking out from the water, the bank top is the first cell above the "
         "water where the ground flattens below this slope."],
        ["Bank band, b (m)", f"{s.band:g}",
         "Bank slopes are fitted to the bank cells from the water's edge up to "
         "b above the water (at least two cells, never past the bank top)."],
        ["Bank slopes, s_L and s_R (m/m)", "per station",
         "Least-squares line z = c + s·x through the band cells; where the line "
         "crosses the water level is taken as the water's edge. Running median "
         f"over {s.smooth_n} stations."],
        ["Depth from the banks, D (m)", "per station",
         "D = s_L s_R W / (2 (s_L + s_R)): two half-parabolas, each meeting its "
         "bank at that bank's slope and meeting each other level at the "
         "deepest point (s W / 4 when s_L = s_R)."],
        ["Deepest point, a (m)", "per station",
         "a = W s_R / (s_L + s_R) from the left edge, so the deepest point lies "
         f"toward the steeper bank; kept within {s.thalweg_limit:g} W to "
         f"{1 - s.thalweg_limit:g} W."],
        ["Low and high beds", f"{s.window_widths:g} W window",
         "The same formula with the 25th and 75th percentiles of each bank's "
         "slopes over a moving window of this many widths."],
        ["Minimum width-to-depth ratio, (W/D)_min", f"{s.min_ratio:g}",
         f"Cap D ≤ W / (W/D)_min. Stations limited: {sm['capped_ratio']}."],
        ["Maximum depth, D_max (m)",
         "-" if not s.max_depth else f"{s.max_depth:g}",
         f"Optional cap D ≤ D_max. Stations limited: {sm['capped_max']}."],
        ["Bend scour factor, A (-)", f"{s.scour:g}",
         "Stations with one bank slope: depth × (1 - A κ n), κ = centreline "
         "curvature, n = offset from the channel centre (Ikeda et al. 1981 and "
         "Odgaard 1981, as cited by Ottevanger et al. 2012); |A κ W/2| ≤ 0.8."],
        ["Fallback width-to-depth ratios", (
            "-" if not s.fallback_ratios
            else "/".join(f"{v:g}" for v in s.fallback_ratios)),
         "Shallow / central / deep, used only where no station on a "
         "centreline has a bank slope."],
    ]  # fmt: skip


def _station_arrays(result: ChannelResult, line: int):
    group = [st for st in result.stations if st.line == line and st.depth]
    ch = np.array([st.chainage for st in group])
    wse = np.array([st.wse for st in group])
    d = {sc: np.array([st.depth[sc] for st in group]) for sc in SCENARIOS}
    return group, ch, wse, d


def figure_long_section(result: ChannelResult, line: int, style) -> io.BytesIO | None:
    group, ch, wse, d = _station_arrays(result, line)
    if not group:
        return None
    with style.context():
        fig = style.figure(8.0)
        ax = fig.add_subplot(1, 1, 1)
        ax.fill_between(
            ch,
            wse - d["high"],
            wse - d["low"],
            color=C_RED,
            alpha=FILL_ALPHA,
            linewidth=0,
            label="Bed, low to high",
        )
        ax.plot(ch, wse, color=C_BLUE, linewidth=1.5, label="Water surface (DEM)")
        ax.plot(ch, wse - d["central"], color=C_RED, linewidth=1.5,
                label="Deepest bed level, central")  # fmt: skip
        tops = np.array([np.nanmin([st.height_left, st.height_right]) for st in group])
        ax.plot(ch, wse + tops, color=CHART_INK, linewidth=1.0, linestyle="--",
                label="Lower bank top (natural ground)")  # fmt: skip
        ax.set_xlabel("Chainage (m)")
        ax.set_ylabel("Elevation (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        style.axes(ax)
        style.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)
        return style.png(fig)


def figure_width_depth(result: ChannelResult, line: int, style) -> io.BytesIO | None:
    group, ch, _, d = _station_arrays(result, line)
    if not group:
        return None
    with style.context():
        fig = style.figure(8.0)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(ch, [st.width for st in group], color=C_BLUE, linewidth=1.5,
                label="Water width W (m)")  # fmt: skip
        ax.set_xlabel("Chainage (m)")
        ax.set_ylabel("Width (m)")
        ax2 = ax.twinx()
        ax2.fill_between(ch, d["low"], d["high"], color=C_RED, alpha=FILL_ALPHA,
                         linewidth=0)  # fmt: skip
        ax2.plot(ch, d["central"], color=C_RED, linewidth=1.5)
        capped = [i for i, st in enumerate(group) if st.cap]
        if capped:
            ax2.plot(ch[capped], d["central"][capped], linestyle="none", marker="o",
                     markersize=3, color=CHART_INK)  # fmt: skip
        ax2.set_ylabel("Depth (m)")
        ax2.set_ylim(bottom=0)
        ax.set_ylim(bottom=0)
        ax.plot([], [], color=C_RED, linewidth=1.5, label="Depth D, central (m)")
        ax.fill_between([], [], [], color=C_RED, alpha=FILL_ALPHA,
                        label="Depth, low to high")  # fmt: skip
        if capped:
            ax.plot([], [], linestyle="none", marker="o", markersize=3,
                    color=CHART_INK, label="Depth set by a cap")  # fmt: skip
        style.axes(ax)
        style.axes(ax2, grid=False)
        style.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2)
        return style.png(fig)


def section_profile(result: ChannelResult, dem: np.ndarray, st, half: float):
    cell = 0.5 * (abs(result.window_gt[1]) + abs(result.window_gt[5]))
    off = np.arange(-half, half + cell / 4, cell / 2)
    pts = np.column_stack([st.x + off * st.nx, st.y + off * st.ny])
    r, c = _to_rc(pts, result.window_gt)
    z, _ = _sample(np.asarray(dem, float), r, c, np.nan)
    beds = {sc: _sample(result.beds[sc], r, c, np.nan)[0] for sc in SCENARIOS}
    return off, z, beds


def figure_sections(result: ChannelResult, dem, style, n: int = 3) -> io.BytesIO | None:
    picks = [st for st in result.stations if st.depth and st.width > 0]
    if not picks:
        return None
    idx = np.unique(np.linspace(0, len(picks) - 1, n + 2).round().astype(int)[1:-1])
    chosen = [picks[i] for i in idx] or picks[:1]
    with style.context():
        fig = style.figure(6.0 * len(chosen) + 1.5)
        axes = [fig.add_subplot(len(chosen), 1, i + 1) for i in range(len(chosen))]
        for ax, st in zip(axes, chosen, strict=True):
            tops = [v for v in (st.top_left, st.top_right) if np.isfinite(v)]
            half = max([abs(v) for v in tops] + [st.width]) * 1.3 + 5
            off, z, beds = section_profile(result, dem, st, half)
            # looking downstream: left bank on the left
            x = -off
            ax.plot(x, z, color=CHART_INK, linewidth=1.0, linestyle="--",
                    label="DEM (water surface and banks)")  # fmt: skip
            ax.fill_between(x, beds["high"], beds["low"], color=C_RED,
                            alpha=FILL_ALPHA, linewidth=0,
                            label="Bed, low to high")  # fmt: skip
            ax.plot(
                x, beds["central"], color=C_RED, linewidth=1.5, label="Bed, central"
            )
            ax.axhline(st.wse, color=C_BLUE, linewidth=1.0, label="Water level")
            ax.set_title(
                f"Line {st.line + 1}, chainage {st.chainage:.0f} m (looking downstream)"
            )
            ax.set_ylabel("Elevation (m)")
            ax.ticklabel_format(useOffset=False, style="plain")
            style.axes(ax)
        axes[-1].set_xlabel("Offset from the centreline, left bank negative (m)")
        style.legend(axes[-1], loc="upper center", bbox_to_anchor=(0.5, -0.35),
                     ncols=2)  # fmt: skip
        return style.png(fig)


def figure_plan(result: ChannelResult, style) -> io.BytesIO | None:
    from matplotlib.colors import LinearSegmentedColormap

    depth = result.depth["central"]
    rows, cols = np.nonzero(np.isfinite(depth))
    if rows.size == 0:
        return None
    gt = result.window_gt
    xs = gt[0] + (np.r_[cols.min(), cols.max() + 1]) * gt[1]
    ys = gt[3] + (np.r_[rows.min(), rows.max() + 1]) * gt[5]
    pad = 0.05 * max(np.ptp(xs), np.ptp(ys), 1.0)
    extent = (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad)
    nr, nc = depth.shape
    img = (gt[0], gt[0] + nc * gt[1], gt[3] + nr * gt[5], gt[3])
    with style.context():
        fig = style.figure(_plan_height_cm(extent, 2.6))
        ax = fig.add_subplot(1, 1, 1)
        cmap = LinearSegmentedColormap.from_list("depth", ["#CCE5FF", C_BLUE])
        im = ax.imshow(np.ma.masked_invalid(depth), extent=img, cmap=cmap, vmin=0,
                       interpolation="nearest")  # fmt: skip
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Depth, central (m)", fontproperties=style.font_props("bold"))
        for line in result.lines:
            ax.plot(line[:, 0], line[:, 1], color=CHART_INK, linewidth=0.75,
                    linestyle="--")  # fmt: skip
        ax.plot([], [], color=CHART_INK, linewidth=0.75, linestyle="--",
                label="Centreline")  # fmt: skip
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(useOffset=False, style="plain")
        _scale_bar(ax, extent)
        _north_arrow(ax)
        style.axes(ax, grid=False)
        style.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncols=2)
        return style.png(fig)


def write_report_docx(
    path: str,
    result: ChannelResult,
    context: dict,
    dem: np.ndarray,
    include_figures: bool = True,
) -> list[str]:
    """Writes the report; returns messages for the log."""
    d = Doc(house_style=True)
    ctx = context or {}
    messages: list[str] = []
    style = None
    if include_figures:
        try:
            style = ChartStyle()
            messages.extend(style.warnings)
        except ImportError:
            include_figures = False
    s = result.settings
    sm = result.summary()
    dem_info = ctx.get("dem", {})

    d.title_page(
        REPORT_TITLE,
        purpose=(
            "This report estimates the river channel below the water surface "
            "recorded by the digital elevation model, from the slopes of the "
            "banks, and gives the bed DEM, depths and the assumptions behind "
            "them."
        ),
        details=title_details(ctx),
        subtitle=ctx.get("project_name") or None,
    )

    # 1. Summary
    d.heading("1. Summary", 2)
    d.para(
        "The digital elevation model (DEM) shows the water surface of the river "
        "on the day of the survey, not its bed. A channel was cut below that "
        f"surface with the {ctx.get('tool', REPORT_TITLE)} tool (Mayim Tools, "
        f"version {ctx.get('version', '-')}), from the slopes of the banks just "
        "above the water. No measured depths or flows were used."
    )
    d.para(describe(result))
    d.caption("Table", "Summary of the estimated channel")
    d.table(
        ["Bed", "Volume (m³)", "Max depth (m)", "Mean station depth (m)"],
        [
            [
                sc.capitalize(),
                f"{sm[f'volume_{sc}_m3']:.0f}",
                _f(sm[f"max_depth_{sc}_m"], "{:.2f}"),
                _f(sm[f"mean_station_depth_{sc}_m"], "{:.2f}"),
            ]
            for sc in SCENARIOS
        ],
        widths_cm=[3.5, 4.0, 4.0, 5.0],
    )

    # 2. Method
    d.heading("2. Tool and method", 2)
    for step in [
        f"Stations were set every {s.spacing:g} m along each centreline (drawn "
        "downstream; a line drawn upstream is reversed), each with a section at "
        f"right angles reaching {s.search:g} m to either side.",
        (
            "The water surface was taken from the water polygon given."
            if ctx.get("water_polygon")
            else "The flat water surface was found as the cells within "
            f"{s.flat_tol:g} m of the running median of the DEM under the "
            f"centreline, flatter than {s.max_water_slope:g} m/m and connected "
            "to the centreline."
        ),
        "On each section the water width and level were read, and the slope of "
        "each bank was fitted to the bank cells from the water's edge up to "
        f"{s.band:g} m above the water (or the first two cells of a steep bank), "
        "stopping at the bank top. The fitted line also places the water's edge "
        "between cells.",
        "Below the water the channel was drawn as two half-parabolas, each "
        "meeting its bank at that bank's slope and meeting the other level at "
        "the deepest point. This gives the depth D = s_L s_R W / (2 (s_L + s_R)) "
        "and moves the deepest point toward the steeper bank, as at the outer "
        "bank of a bend.",
        f"The depth was capped at W / {s.min_ratio:g}"
        + (f" and at {s.max_depth:g} m" if s.max_depth else "")
        + ". Low and high beds use the 25th and 75th percentiles of the bank "
        "slopes along the reach. Stations with one bank slope used it for both "
        "sides with a tilt toward the outer bank of the bend; stations with "
        "none took the depth-to-width ratio of their neighbours.",
        "For every water cell, the distances to the water's edge along the local "
        "section line set its place across the channel, so the bed meets the "
        "banks at the water's edge with no step. The bed was written into a "
        "copy of the DEM where it lies below the DEM.",
    ]:
        d.numbered(step)

    # 3. Inputs
    d.heading("3. Inputs and settings", 2)
    d.caption("Table", "Inputs")
    d.table(
        ["Item", "Value"],
        [
            ["DEM", dem_info.get("path", "-")],
            ["Coordinate reference system", dem_info.get("crs", "-")],
            ["DEM cell size",
             f"{dem_info.get('cell_x', 0):g} m x {dem_info.get('cell_y', 0):g} m"],
            ["Centrelines", ctx.get("centrelines", "-")],
            ["Water polygon", ctx.get("water_polygon") or "None (found from the DEM)"],
            ["Project number", ctx.get("project_number") or "-"],
            ["Run", ctx.get("run_time", "-")],
        ],
        widths_cm=[5.0, 11.5],
    )  # fmt: skip

    # 4. Parameters
    d.heading("4. Parameters and calculations", 2)
    d.caption("Table", "Parameters, values and definitions")
    d.table(
        ["Parameter (symbol, unit)", "Value", "Definition / equation (source)"],
        parameter_rows(result),
        widths_cm=[5.2, 2.6, 8.7],
    )

    # 5. Results
    d.heading("5. Results", 2)
    d.para(describe(result))
    d.caption("Table", "How the depth was set at the stations")
    rows = [
        [BASIS_TEXT[k], str(sm[f"basis_{k}"])] for k in BASIS_TEXT if sm[f"basis_{k}"]
    ]
    rows += [[f"Depth limited by the {CAP_TEXT[k]}", str(sm[f"capped_{k}"])]
             for k in ("ratio", "max") if sm[f"capped_{k}"]]  # fmt: skip
    d.table(["Basis", "Stations"], rows, widths_cm=[12.5, 4.0])
    if include_figures and style is not None:
        plan = figure_plan(result, style)
        if plan is not None:
            d.picture(
                plan, "Estimated depth below the surveyed water surface (central bed)"
            )
        for li in range(len(result.lines)):
            label = f" - centreline {li + 1}" if len(result.lines) > 1 else ""
            fig = figure_long_section(result, li, style)
            if fig is not None:
                d.picture(fig, f"Long section{label}: water surface, bed and bank tops")
            fig = figure_width_depth(result, li, style)
            if fig is not None:
                d.picture(
                    fig, f"Water width and estimated depth along the channel{label}"
                )
        fig = figure_sections(result, dem, style)
        if fig is not None:
            d.picture(fig, "Sample cross-sections: DEM and estimated bed")

    # 6. Outputs
    outputs = ctx.get("outputs", [])
    if outputs:
        d.heading("6. Outputs", 2)
        d.caption("Table", "Files produced by the tool")
        d.table(
            ["Output", "Location"], [list(o) for o in outputs], widths_cm=[5.0, 11.5]
        )

    # 7. Notes
    d.heading("7. Notes and limitations", 2)
    for note in [
        "No depth was measured and no flow was used: the depth rests on the "
        "slopes of the banks just above the water, taken to continue below it. "
        "Banks steeper than the bed (cut, vegetated or protected banks) give "
        "too deep a channel, and gentle banks over a deep channel too shallow "
        "a one. The caps limit the first case; check the station flags.",
        "The bank line is fitted over a band above the water, so on a smoothly "
        "curving bank it is a little steeper than the bank right at the "
        "water's edge, which makes the depth a little larger.",
        "Photogrammetry DEMs often show a noisy or false surface over water; "
        "give a water polygon for them. Lidar banks next to hydro-flattened "
        "water may be interpolated, which smooths the bank slopes.",
        "Only the channel below the surveyed water surface is estimated; bars, "
        "banks and the floodplain are taken from the DEM as they are.",
        "This is an estimate for hydraulic modelling where no survey exists. "
        "Where the result matters, check it against a few measured depths.",
    ]:
        d.bullet(note)
    warnings = list(ctx.get("warnings", [])) + messages
    if warnings:
        d.para("Processing messages recorded during this run:")
        for w in dict.fromkeys(warnings):
            d.bullet(w)

    # 8. References
    d.heading("8. References", 2)
    for ref in sorted(REFERENCES):
        d.para(ref)
    d.doc.save(path)
    return messages
