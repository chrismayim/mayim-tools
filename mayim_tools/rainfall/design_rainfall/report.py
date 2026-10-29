"""Report-ready output generation from PointEstimates.

Two output formats:
    - CSV: flat duration x return-period table, for import into
      spreadsheets or other design workflows; and a DDF table CSV with
      the depths only (write_ddf_csv).
    - DOCX: a Word annexure in the Mayim report and figure standard
      (Doc(house_style=True)): summary, method, inputs, parameters, depth
      tables per site with a layout map of the point of interest, the
      snapped grid point and the nearest rainfall stations, the nearest
      stations (reference only), outputs, notes and references.

Zero QGIS dependency: the layout map is rendered by QGIS in the wrapper
(qgis_map.make_plan_map, a PlanMapFn); without it a simplified location
plot is drawn with matplotlib. python-docx and matplotlib are imported
lazily. No thousands separators anywhere in the report.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .core import PointEstimate

if TYPE_CHECKING:
    from mayim_tools._common.report_maps import PlanMap


def _duration_minutes(label: str) -> float:
    """Sort key: converts a duration label back to minutes so rows can
    be ordered chronologically (5 min < ... < 1 day < ... < 7 day)."""
    if label.endswith(" min"):
        return float(label[:-4])
    if label.endswith(" h"):
        return float(label[:-2]) * 60
    if label.endswith(" day"):
        return float(label[:-4]) * 1440
    return float("inf")


def _sorted_durations(results) -> list[str]:
    """Unique duration labels, chronologically ordered."""
    seen = {}
    for r in results:
        seen.setdefault(r.duration_label, _duration_minutes(r.duration_label))
    return sorted(seen, key=lambda d: seen[d])


def _sorted_return_periods(results) -> list[int]:
    return sorted(set(r.return_period for r in results))


def _write_stations_block(
    writer, stations_df, with_site_column: bool = False, site_label: str | None = None
) -> None:
    """Appends every field from a stations GeoDataFrame (as produced by
    DesignRainfallEngine.nearest_stations) - same data as the 'nearest
    rainfall stations' output layer, for reference/comparison only, not
    used in the grid-based calculation."""
    if stations_df is None or stations_df.empty:
        return
    cols = [c for c in stations_df.columns if c != "geometry"]
    writer.writerow((["Site"] if with_site_column else []) + cols)
    for _, row in stations_df.iterrows():
        writer.writerow(
            (([site_label] if with_site_column else [])) + [row[c] for c in cols]
        )


def write_csv(estimate: PointEstimate, output_path: str | Path, stations=None) -> None:
    """Wide format: one row per duration, one Depth/Lower/Upper column
    triplet per return period. If a stations GeoDataFrame is supplied
    (nearest daily rainfall stations, reference only), it's appended at
    the bottom of the file."""
    durations = _sorted_durations(estimate.results)
    return_periods = _sorted_return_periods(estimate.results)
    lookup = {(r.duration_label, r.return_period): r for r in estimate.results}

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Latitude", "Longitude", "MAP (mm)", "Altitude (m)", "Cluster"]
        )
        writer.writerow(
            [
                estimate.latitude,
                estimate.longitude,
                estimate.map_mm,
                estimate.altitude_m,
                estimate.cluster,
            ]
        )
        writer.writerow([])

        header = ["Duration"]
        for rt in return_periods:
            header += [f"{rt}yr Depth (mm)", f"{rt}yr Lower (mm)", f"{rt}yr Upper (mm)"]
        writer.writerow(header)

        for dur in durations:
            row = [dur]
            for rt in return_periods:
                r = lookup.get((dur, rt))
                if r:
                    row += [
                        r.depth,
                        r.lower if r.lower is not None else "",
                        r.upper if r.upper is not None else "",
                    ]
                else:
                    row += ["", "", ""]
            writer.writerow(row)

        if stations is not None and not stations.empty:
            writer.writerow([])
            writer.writerow(
                [
                    "Nearest Rainfall Stations "
                    "(reference only - not used in the grid-based calculation)"
                ]
            )
            _write_stations_block(writer, stations)


def write_ddf_csv(estimates: dict[str, PointEstimate], output_path: str | Path) -> None:
    """DDF table only: one row per duration, one '<T>yr Depth (mm)' column
    per return period - the depth columns of write_csv without the lower /
    upper bounds, the site header and the nearest stations. With several
    sites a leading Site column identifies each site's rows (the return
    period columns are the union across the sites)."""
    all_results = [r for est in estimates.values() for r in est.results]
    return_periods = _sorted_return_periods(all_results)
    multi = len(estimates) > 1

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = (["Site"] if multi else []) + ["Duration"]
        header += [f"{rt}yr Depth (mm)" for rt in return_periods]
        writer.writerow(header)
        for site, est in estimates.items():
            lookup = {(r.duration_label, r.return_period): r for r in est.results}
            for dur in _sorted_durations(est.results):
                row = ([site] if multi else []) + [dur]
                for rt in return_periods:
                    r = lookup.get((dur, rt))
                    row.append(r.depth if r else "")
                writer.writerow(row)


# ----------------------------------------------------------------------
# Multi-site variants (for vector point layer input)
# ----------------------------------------------------------------------


def write_csv_multi(
    estimates: dict[str, PointEstimate],
    output_path: str | Path,
    stations_by_site: dict | None = None,
) -> None:
    """One combined CSV: a Site column identifies which point each row
    belongs to. Same wide layout as write_csv - one row per site+duration,
    one Depth/Lower/Upper column triplet per return period. The return
    period column set is the union across all sites, so every site's rows
    line up under the same headers even if individual sites happen to be
    missing a particular return period. If stations_by_site is supplied
    (dict[site_label, GeoDataFrame]), the nearest stations for every site
    are appended at the bottom of the file."""
    all_results = [r for est in estimates.values() for r in est.results]
    return_periods = _sorted_return_periods(all_results)

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = [
            "Site",
            "Latitude",
            "Longitude",
            "MAP (mm)",
            "Altitude (m)",
            "Cluster",
            "Duration",
        ]
        for rt in return_periods:
            header += [f"{rt}yr Depth (mm)", f"{rt}yr Lower (mm)", f"{rt}yr Upper (mm)"]
        writer.writerow(header)

        for site, est in estimates.items():
            durations = _sorted_durations(est.results)
            lookup = {(r.duration_label, r.return_period): r for r in est.results}
            for dur in durations:
                row = [
                    site,
                    est.latitude,
                    est.longitude,
                    est.map_mm,
                    est.altitude_m,
                    est.cluster,
                    dur,
                ]
                for rt in return_periods:
                    r = lookup.get((dur, rt))
                    if r:
                        row += [
                            r.depth,
                            r.lower if r.lower is not None else "",
                            r.upper if r.upper is not None else "",
                        ]
                    else:
                        row += ["", "", ""]
                writer.writerow(row)

        if stations_by_site and any(not df.empty for df in stations_by_site.values()):
            writer.writerow([])
            writer.writerow(
                [
                    "Nearest Rainfall Stations "
                    "(reference only - not used in the grid-based calculation)"
                ]
            )
            first = True
            for site, df in stations_by_site.items():
                if df is None or df.empty:
                    continue
                if first:
                    cols = [c for c in df.columns if c != "geometry"]
                    writer.writerow(["Site"] + cols)
                    first = False
                for _, row in df.iterrows():
                    writer.writerow([site] + [row[c] for c in cols])


# ----------------------------------------------------------------------
# Word report (Mayim report and figure standard)
# ----------------------------------------------------------------------

REPORT_TITLE = "Design Rainfall Estimation (South Africa)"
STATIONS_HEADING = (
    "Nearest Rainfall Stations "
    "(reference only - not used in the grid-based calculation)"
)
DEFAULT_METHOD_NOTE = (
    "regional L-moment design rainfall method of Smithers and Schulze (2002)"
)

# (site label, plot width in cm, dpi) -> PlanMap or None, supplied by the
# QGIS wrapper (qgis_map.make_plan_map); without it a simplified location
# plot is drawn with matplotlib.
PlanMapFn = Callable[[str, float, float], "PlanMap | None"]

REFERENCES = [
    "Hosking, J.R.M. & Wallis, J.R. (1997). Regional Frequency Analysis: An "
    "Approach Based on L-Moments. Cambridge University Press, Cambridge.",
    "Smithers, J.C. & Schulze, R.E. (2002). Design Rainfall and Flood "
    "Estimation in South Africa. WRC Report No. K5/1060. Water Research "
    "Commission, Pretoria.",
]


# layout map extent around the projected points
MIN_HALF_SIZE_M = 750.0  # smallest map half-width / half-height
PAD_FRACTION = 0.18  # padding around the points (room for station labels)
ASPECT_RANGE = (0.6, 1.0)  # map height / width


def map_extent(points_m, pad=PAD_FRACTION, min_half=MIN_HALF_SIZE_M):
    """Extent (xmin, ymin, xmax, ymax) in metres around projected points,
    padded, at least ``min_half`` either side of the centre and with a
    height / width ratio inside ASPECT_RANGE (used by qgis_map)."""
    xs = [p[0] for p in points_m]
    ys = [p[1] for p in points_m]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    half_w = max((max(xs) - min(xs)) / 2.0 * (1 + 2 * pad), min_half)
    half_h = max((max(ys) - min(ys)) / 2.0 * (1 + 2 * pad), min_half)
    lo, hi = ASPECT_RANGE
    if half_h / half_w < lo:
        half_h = half_w * lo
    elif half_h / half_w > hi:
        half_w = half_h / hi
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _fmt(value, fmt: str = "{:.1f}") -> str:
    """Formats a number without thousands separators; '-' when missing."""
    if value is None:
        return "-"
    try:
        if value != value:  # NaN
            return "-"
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def _text(value) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text and text.lower() not in ("nan", "none") else "-"


def _rt_label(rt: int) -> str:
    return f"1:{rt} yr"


def _duration_range(durations: list[str]) -> str:
    if not durations:
        return "-"
    if len(durations) == 1:
        return durations[0]
    return f"{durations[0]} to {durations[-1]}"


def _join(items: list[str]) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _depth_table_rows(estimate: PointEstimate, attr: str):
    """(header, rows) of a duration x return period table of one result
    attribute ('depth', 'lower' or 'upper')."""
    durations = _sorted_durations(estimate.results)
    return_periods = _sorted_return_periods(estimate.results)
    lookup = {(r.duration_label, r.return_period): r for r in estimate.results}
    header = ["Duration"] + [_rt_label(rt) for rt in return_periods]
    rows = []
    for dur in durations:
        row = [dur]
        for rt in return_periods:
            r = lookup.get((dur, rt))
            row.append(_fmt(getattr(r, attr) if r else None))
        rows.append(row)
    return header, rows


def _depth_widths(n_rt: int) -> list[float]:
    first = 2.5
    rest = (16.5 - first) / max(n_rt, 1)
    return [first] + [rest] * n_rt


def describe_site(site: str, est: PointEstimate) -> str:
    """One-paragraph description of the query point and the grid point used."""
    if est.requested_latitude is not None and est.requested_longitude is not None:
        where = (
            f"{site} (latitude {est.requested_latitude:.5f}°, longitude "
            f"{est.requested_longitude:.5f}°) was matched to the nearest grid "
            f"point at latitude {est.latitude:.5f}°, longitude {est.longitude:.5f}°"
        )
        if est.distance_km is not None:
            where += f", {est.distance_km:.2f} km away"
    else:
        where = (
            f"{site} was estimated at the grid point at latitude "
            f"{est.latitude:.5f}°, longitude {est.longitude:.5f}°"
        )
    return (
        f"{where}. At the grid point the mean annual precipitation (MAP) is "
        f"{est.map_mm:.0f} mm and the altitude {est.altitude_m:.0f} m; the "
        f"site falls in growth curve cluster {est.cluster}, short-duration "
        f"cluster {est.s_cluster} and multi-day region {est.av_cluster}."
    )


def _headline(est: PointEstimate) -> str | None:
    """Sentence giving the 1-day depth (or the longest duration) at the
    highest return period, with its 90 % interval."""
    if not est.results:
        return None
    durations = _sorted_durations(est.results)
    dur = "1 day" if "1 day" in durations else durations[-1]
    rt = max(r.return_period for r in est.results if r.duration_label == dur)
    r = next(
        r for r in est.results if r.duration_label == dur and r.return_period == rt
    )
    text = f"The {dur}, 1:{rt} year design rainfall depth is {r.depth:.1f} mm"
    if r.lower is not None and r.upper is not None:
        text += f" (90 % interval {r.lower:.1f} to {r.upper:.1f} mm)"
    return text + "."


def parameter_rows(est: PointEstimate | None) -> list[list[str]]:
    """Parameter | Value | Definition rows. Values are given when a single
    site is reported (``est``), otherwise '-' (values per site are in the
    results)."""

    def v(value, fmt):
        return _fmt(value, fmt) if est is not None else "-"

    e = est
    return [
        [
            "Mean annual precipitation, MAP (mm)",
            v(e and e.map_mm, "{:.0f}"),
            "MAP at the grid point, from the national 1 arc-minute grid "
            "(Smithers & Schulze, 2002).",
        ],
        [
            "Altitude, z (m)",
            v(e and e.altitude_m, "{:.0f}"),
            "Altitude of the grid point (Smithers & Schulze, 2002).",
        ],
        [
            "Growth curve cluster, C (-)",
            v(e and e.cluster, "{:d}"),
            "Homogeneous region of the 1-day growth curve used for every "
            "duration (Smithers & Schulze, 2002).",
        ],
        [
            "Short-duration cluster, Cs (-)",
            v(e and e.s_cluster, "{:d}"),
            "Region of the 24 h : 1-day ratio and the short-duration L1 "
            "regressions (Smithers & Schulze, 2002).",
        ],
        [
            "Multi-day region, Cr (-)",
            v(e and e.av_cluster, "{:d}"),
            "Region of the 2 to 7 day L1 regressions (Smithers & Schulze, 2002).",
        ],
        [
            "1-day index value, L1,1d (mm)",
            v(e and e.l1_1day, "{:.2f}"),
            "Adjusted mean of the annual maximum 1-day rainfall (L-moment "
            "L1) at the grid point (Smithers & Schulze, 2002).",
        ],
        [
            "24 h : 1-day ratio, R24 (-)",
            v(e and e.ratio_24h, "{:.3f}"),
            "Median ratio of 24 h to 1-day annual maxima for the short-"
            "duration cluster; L1,24h = R24 x L1,1d (Smithers & Schulze, 2002).",
        ],
        [
            "Short-duration index value, L1,D (mm)",
            "-",
            "L1,D = aD x L1,24h + bD for D = 5 min to 24 h, with aD and bD "
            "per cluster and duration (Smithers & Schulze, 2002, Eq. 16).",
        ],
        [
            "Multi-day index value, L1,D (mm)",
            "-",
            "L1,D = ψD x L1,1d + ωD for D = 2 to 7 days, with ψD = θ + τ D^σ "
            "and ωD = υ + κ D^ρ per region (Smithers & Schulze, 2002, "
            "Eq. 12 to 14).",
        ],
        [
            "Design rainfall depth, PD,T (mm)",
            "-",
            "PD,T = GCT x L1,D, with GCT the 1-day growth curve factor of "
            "cluster C for return period T (Smithers & Schulze, 2002, Eq. 33).",
        ],
        [
            "90 % upper bound, PU (mm)",
            "-",
            "PU = GCUT x L1,D x (1 + PiD / 100), with PiD the prediction "
            "interval of L1,D (%) at the grid point (Smithers & Schulze, "
            "2002, Eq. 34 to 37).",
        ],
        [
            "90 % lower bound, PL (mm)",
            "-",
            "PL = GCLT x L1,D x (1 - PiD / 100) (Smithers & Schulze, 2002, "
            "Eq. 34 to 37).",
        ],
    ]


def station_rows(stations) -> list[list[str]]:
    """Rows of the nearest-stations table from a stations GeoDataFrame."""
    rows = []
    if stations is None or stations.empty:
        return rows
    for _, s in stations.iterrows():
        fyr, lyr = s.get("FYR"), s.get("LYR")
        record = (
            f"{_fmt(fyr, '{:.0f}')} to {_fmt(lyr, '{:.0f}')}"
            if fyr is not None and lyr is not None
            else "-"
        )
        rows.append(
            [
                _text(s.get("FSTN_NO")),
                _text(s.get("STN_NAME")),
                _fmt(s.get("DIST_KM"), "{:.2f}"),
                _fmt(s.get("MAP"), "{:.0f}"),
                _fmt(s.get("ALTITUDE"), "{:.0f}"),
                record,
                _fmt(s.get("REC_LEN"), "{:.0f}"),
            ]
        )
    return rows


STATION_HEADER = [
    "Station no.",
    "Station name",
    "Distance (km)",
    "MAP (mm)",
    "Altitude (m)",
    "Record",
    "Length (years)",
]
STATION_WIDTHS = [2.4, 5.0, 1.7, 1.4, 1.7, 2.4, 1.9]


def station_1day_rows(stations, return_periods: list[int]):
    """(header, rows) of the 1-day design depths published for the stations
    (median of each return period, DRE dataset)."""
    header = ["Station name"] + [_rt_label(rt) for rt in return_periods]
    rows = []
    if stations is None or stations.empty:
        return header, rows
    for _, s in stations.iterrows():
        row = [_text(s.get("STN_NAME"))]
        for rt in return_periods:
            row.append(_fmt(s.get(f"D1_M{rt}")))
        rows.append(row)
    return header, rows


def figure_location_simple(site: str, est: PointEstimate, stations, style):
    """Fallback layout figure (no QGIS): the point of interest, grid point
    and stations in latitude / longitude, in the chart house style."""
    from mayim_tools._common.report_figures import CHART_COLOURS, CHART_INK

    amber, green, blue = CHART_COLOURS[1], CHART_COLOURS[3], CHART_COLOURS[4]
    with style.context():
        fig = style.figure(11.0)
        ax = fig.add_subplot(1, 1, 1)
        if stations is not None and not stations.empty:
            ax.scatter(
                stations.geometry.x,
                stations.geometry.y,
                marker="^",
                s=30,
                color=blue,
                edgecolor=CHART_INK,
                linewidth=0.5,
                label="Nearest daily rainfall stations (reference only)",
                zorder=3,
            )
            for _, s in stations.iterrows():
                ax.annotate(
                    _text(s.get("STN_NAME")),
                    (s.geometry.x, s.geometry.y),
                    xytext=(4, 3),
                    textcoords="offset points",
                )
        ax.scatter(
            [est.longitude],
            [est.latitude],
            marker="X",
            s=40,
            color=green,
            edgecolor=CHART_INK,
            linewidth=0.5,
            label="Snapped grid point",
            zorder=4,
        )
        if est.requested_latitude is not None:
            ax.scatter(
                [est.requested_longitude],
                [est.requested_latitude],
                marker="*",
                s=90,
                color=amber,
                edgecolor=CHART_INK,
                linewidth=0.5,
                label="Point of interest",
                zorder=5,
            )
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.margins(0.15)
        style.axes(ax)
        style.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=3)
        return style.png(fig)


def figure_layout(site, est, stations, style, plan_map, messages):
    """The site's layout map: rendered by ``plan_map`` (QGIS, layer QML
    styles, OpenStreetMap) when given, else the simplified figure. Returns
    (PNG stream, used OSM background)."""
    from mayim_tools._common.report_maps import (
        PLAN_DPI,
        PLAN_PLOT_WIDTH_CM,
        figure_rendered_map,
    )

    if plan_map is not None:
        try:
            pm = plan_map(site, PLAN_PLOT_WIDTH_CM, PLAN_DPI)
        except Exception as e:  # the report must still be written
            pm = None
            messages.append(
                f"{site}: the QGIS layout map could not be rendered ({e}); a "
                "simplified location plot was drawn instead."
            )
        if pm is not None:
            return figure_rendered_map(pm, style), bool(pm.attribution)
    return figure_location_simple(site, est, stations, style), False


def title_purpose(n_sites: int) -> str:
    """Short purpose and goal statement for the title page."""
    sites = "the site" if n_sites == 1 else f"{n_sites} sites"
    return (
        "This report gives design rainfall depths, with their 90 % "
        f"prediction intervals, for {sites}, estimated with the regional "
        "L-moment method of Smithers and Schulze (2002) from the national "
        "design rainfall grid, as input to stormwater and flood design."
    )


def title_details(ctx: dict, estimates: dict) -> list[tuple[str, str]]:
    """Report details for the title page ("-" where not available)."""
    from datetime import date

    site = ctx.get("project_name")
    if not site and len(estimates) == 1:
        site = next(iter(estimates))
    tool = ctx.get("tool", REPORT_TITLE)
    version = ctx.get("version")
    return [
        ("Project number", ctx.get("project_number") or "-"),
        ("Project / site", site or "-"),
        ("Prepared by", ctx.get("prepared_by") or "Mayim Consulting"),
        ("Tool and version", f"{tool}, version {version}" if version else tool),
        ("Software", ctx.get("software") or "-"),
        ("Report date", ctx.get("report_date") or date.today().isoformat()),
        ("Document status", ctx.get("status") or "Draft"),
    ]


def write_report_docx(
    path: str | Path,
    estimates: dict[str, PointEstimate],
    context: dict | None = None,
    stations_by_site: dict | None = None,
    plan_map: PlanMapFn | None = None,
    include_figures: bool = True,
) -> list[str]:
    """Writes the Word report (Mayim report and figure standard) and returns
    messages for the log (font substitutions, map problems).

    ``context`` describes the run: tool, version, software, project_number,
    project_name, status, run_time, input,
    num_stations, outputs [(label, location)], warnings and map_messages
    (a list the plan_map fills while rendering, read after the maps).
    ``stations_by_site`` maps a site label to its nearest stations
    (GeoDataFrame from DesignRainfallEngine.nearest_stations).
    ``plan_map`` renders each site's layout map in QGIS."""
    from mayim_tools._common.docx_report import Doc
    from mayim_tools._common.report_figures import ChartStyle
    from mayim_tools._common.report_maps import OSM_LICENCE_NOTE

    ctx = context or {}
    stations_by_site = stations_by_site or {}
    messages: list[str] = []
    style = None
    if include_figures:
        try:
            style = ChartStyle()
            messages.extend(style.warnings)
        except ImportError:
            include_figures = False

    all_results = [r for est in estimates.values() for r in est.results]
    durations = _sorted_durations(all_results)
    return_periods = _sorted_return_periods(all_results)
    single = len(estimates) == 1
    n_sites = len(estimates)
    num_stations = ctx.get("num_stations")
    if num_stations is None:
        num_stations = max(
            (len(df) for df in stations_by_site.values() if df is not None),
            default=0,
        )
    has_stations = any(
        df is not None and not df.empty for df in stations_by_site.values()
    )

    d = Doc(house_style=True)
    d.title_page(
        REPORT_TITLE,
        purpose=title_purpose(n_sites),
        details=title_details(ctx, estimates),
        subtitle=ctx.get("project_name") or None,
    )

    # 1. Summary ------------------------------------------------------------
    d.heading("1. Summary", 2)
    d.para(
        f"This annexure gives design rainfall depths for {n_sites} "
        f"site{'s' if n_sites != 1 else ''}, for durations of "
        f"{_duration_range(durations)} and return periods of "
        f"{_join([str(rt) for rt in return_periods])} years. The depths were "
        f"estimated with the {DEFAULT_METHOD_NOTE} at the nearest point of "
        "the national 1 arc-minute design rainfall grid, using the "
        f"{ctx.get('tool', REPORT_TITLE)} tool (version "
        f"{ctx.get('version', '-')}, Mayim Tools). Median depths are given "
        "with their 90 % prediction intervals."
    )
    if single:
        site, est = next(iter(estimates.items()))
        headline = _headline(est)
        if headline:
            d.para(headline)
    d.caption("Table", "Summary of the sites and the grid points used")
    d.table(
        [
            "Site",
            "Latitude (°)",
            "Longitude (°)",
            "Distance to grid point (km)",
            "MAP (mm)",
            "Altitude (m)",
            "Clusters C / Cs / Cr",
        ],
        [
            [
                site,
                _fmt(
                    (
                        est.requested_latitude
                        if est.requested_latitude is not None
                        else est.latitude
                    ),
                    "{:.5f}",
                ),
                _fmt(
                    (
                        est.requested_longitude
                        if est.requested_longitude is not None
                        else est.longitude
                    ),
                    "{:.5f}",
                ),
                _fmt(est.distance_km, "{:.2f}"),
                _fmt(est.map_mm, "{:.0f}"),
                _fmt(est.altitude_m, "{:.0f}"),
                f"{est.cluster} / {est.s_cluster} / {est.av_cluster}",
            ]
            for site, est in estimates.items()
        ],
        widths_cm=[3.4, 2.1, 2.1, 2.3, 1.8, 1.9, 2.9],
    )

    # 2. Tool and method ----------------------------------------------------
    d.heading("2. Tool and method", 2)
    d.para(
        "The design rainfall depths follow the regional L-moment method of "
        "Smithers and Schulze (2002), in which the index value (the mean of "
        "the annual maxima, L-moment L1) is estimated for each duration and "
        "scaled by a regional growth curve (Hosking & Wallis, 1997). The "
        "processing steps were:"
    )
    steps = [
        "Each site was matched to the nearest point of the national 1 "
        "arc-minute grid, which holds the MAP, altitude, the adjusted 1-day "
        "index value L1,1d, the cluster numbers and the prediction-interval "
        "percentages of each duration.",
        "1-day: the index value is L1,1d at the grid point.",
        "Short durations (5 min to 24 h): the 24 h index value is "
        "L1,24h = R24 x L1,1d, with R24 the median 24 h : 1-day ratio of the "
        "short-duration cluster; the index value of each duration then "
        "follows from the cluster's linear regression on L1,24h.",
        "2 to 7 days: the index value follows from L1,1d with the power-law "
        "regressions of the multi-day region.",
        "The median design depth is the index value times the 1-day growth "
        "curve factor of the site's cluster for each return period; the same "
        "growth curve is used for every duration.",
        "The 90 % upper and lower bounds combine the bounds of the growth "
        "curve with the prediction interval of the index value at the grid "
        "point.",
    ]
    if has_stations:
        steps.append(
            f"The {num_stations} nearest daily rainfall stations of the "
            "design rainfall database were listed per site with their 1-day "
            "design depths, for reference only. They are not used in the "
            "grid-based calculation."
        )
    for step in steps:
        d.numbered(step)

    # 3. Inputs and settings ------------------------------------------------
    d.heading("3. Inputs and settings", 2)
    d.caption("Table", "Inputs and processing settings")
    d.table(
        ["Item", "Value"],
        [
            ["Project number", ctx.get("project_number") or "-"],
            ["Project / site", ctx.get("project_name") or "-"],
            ["Input", ctx.get("input") or "-"],
            ["Sites", str(n_sites)],
            ["Durations", ", ".join(durations) or "-"],
            [
                "Return periods (years)",
                ", ".join(str(rt) for rt in return_periods) or "-",
            ],
            [
                "Nearest rainfall stations listed",
                f"{num_stations} per site (reference only)" if num_stations else "None",
            ],
            [
                "Design rainfall dataset",
                ctx.get("dataset")
                or "Design Rainfall Estimation in South Africa (rainfall3) "
                "national grid and station database",
            ],
            ["Run", ctx.get("run_time") or "-"],
        ],
        widths_cm=[5.0, 11.5],
    )

    # 4. Parameters and calculations ----------------------------------------
    d.heading("4. Parameters and calculations", 2)
    if not single:
        d.para(
            "The values of the grid-point parameters for each site are given "
            "in the summary table and in Section 5."
        )
    d.caption("Table", "Design rainfall parameters, values and definitions")
    d.table(
        ["Parameter (symbol, unit)", "Value", "Definition / equation (source)"],
        parameter_rows(next(iter(estimates.values())) if single else None),
        widths_cm=[6.2, 2.8, 7.5],
    )

    # 5. Results -------------------------------------------------------------
    d.heading("5. Results", 2)
    used_osm = False
    for i, (site, est) in enumerate(estimates.items(), start=1):
        d.heading(f"5.{i} {site}", 3)
        d.para(describe_site(site, est))
        headline = _headline(est)
        if headline and not single:
            d.para(headline)
        widths = _depth_widths(len(_sorted_return_periods(est.results)))
        for attr, title in (
            ("depth", "Median design rainfall depths (mm)"),
            ("lower", "90 % lower bound of the design rainfall depths (mm)"),
            ("upper", "90 % upper bound of the design rainfall depths (mm)"),
        ):
            header, rows = _depth_table_rows(est, attr)
            d.caption("Table", f"{site}: {title}")
            d.table(header, rows, widths_cm=widths)
        if include_figures and style is not None:
            stations = stations_by_site.get(site)
            stream, osm = figure_layout(site, est, stations, style, plan_map, messages)
            used_osm = used_osm or osm
            content = "point of interest, snapped grid point"
            if stations is not None and not stations.empty:
                content += " and nearest daily rainfall stations"
            d.picture(
                stream,
                f"{site}: layout of the {content}"
                + (" (background: OpenStreetMap)" if osm else ""),
            )

    # 6. Nearest rainfall stations -------------------------------------------
    section = 6
    if has_stations:
        d.heading(f"{section}. {STATIONS_HEADING}", 2)
        d.para(
            "The nearest daily rainfall stations of the design rainfall "
            "database are listed for comparison with the grid-based "
            "estimates. They are reference information only and are NOT used "
            "in the grid-based calculation. Their locations are shown on the "
            "layout maps in Section 5. The 1-day depths are the median design "
            "values published for each station in the database."
        )
        for site, stations in stations_by_site.items():
            if stations is None or stations.empty:
                continue
            if not single:
                d.label(site)
            d.caption("Table", f"{site}: nearest daily rainfall stations")
            d.table(STATION_HEADER, station_rows(stations), widths_cm=STATION_WIDTHS)
            header, rows = station_1day_rows(stations, return_periods)
            first = 5.0
            rest = (16.5 - first) / max(len(return_periods), 1)
            d.caption(
                "Table",
                f"{site}: 1-day median design rainfall depths at the nearest "
                "stations (mm)",
            )
            d.table(header, rows, widths_cm=[first] + [rest] * len(return_periods))
        section += 1

    # 7. Outputs ---------------------------------------------------------------
    outputs = ctx.get("outputs") or []
    if outputs:
        d.heading(f"{section}. Outputs", 2)
        d.caption("Table", "Outputs produced by the tool")
        d.table(
            ["Output", "Location"], [list(o) for o in outputs], widths_cm=[5.0, 11.5]
        )
        section += 1

    # 8. Notes and limitations -------------------------------------------------
    d.heading(f"{section}. Notes and limitations", 2)
    notes = [
        "Estimates are taken at the nearest point of the 1 arc-minute grid "
        "(about 1.5 to 1.8 km apart) without interpolation between grid "
        "points.",
        "The same 1-day growth curve is used for every duration, as in the "
        "source method; the bounds are 90 % prediction intervals, not "
        "confidence limits on an individual storm.",
        "Depths are point rainfall. Areal reduction factors are not applied "
        "and must be applied separately for catchment-average rainfall.",
        "The nearest rainfall stations are listed for reference only and do "
        "not influence the grid-based estimates.",
    ]
    if used_osm:
        notes.append(OSM_LICENCE_NOTE)
    for note in notes:
        d.bullet(note)
    warnings = (
        list(ctx.get("warnings") or []) + list(ctx.get("map_messages") or []) + messages
    )
    if warnings:
        d.para("Processing messages recorded during this run:")
        for w in dict.fromkeys(warnings):
            d.bullet(w)
    section += 1

    # 9. References -------------------------------------------------------------
    d.heading(f"{section}. References", 2)
    for ref in sorted(REFERENCES):
        d.para(ref)

    d.doc.save(str(path))
    return messages


def write_docx(
    estimate: PointEstimate,
    output_path: str | Path,
    site_name: str = "Site of interest",
    stations=None,
    **kwargs,
) -> list[str]:
    """Single-site report (kept for existing callers)."""
    return write_report_docx(
        output_path,
        {site_name: estimate},
        stations_by_site={site_name: stations} if stations is not None else None,
        **kwargs,
    )


def write_docx_multi(
    estimates: dict[str, PointEstimate], output_path: str | Path, **kwargs
) -> list[str]:
    """Multi-site report (kept for existing callers)."""
    return write_report_docx(output_path, estimates, **kwargs)
