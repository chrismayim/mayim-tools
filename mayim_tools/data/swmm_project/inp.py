"""
EPA SWMM 5 input file writer - zero QGIS dependency.

Plain SWMM 5.1/5.2 sections, written in the subset Autodesk Storm and
Sanitary Analysis 2024 itself exports (checked by importing generated files
into SSA 2024): the map sections ([MAP], [COORDINATES], [VERTICES],
[Polygons], [SYMBOLS]) are needed for SSA to draw the elements, and the rain
gage is CUMULATIVE with a time series of cumulative depths starting at 0, as
SSA writes it (EPA SWMM gives the same results as with intensities).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .core import Series, SwmmModel

NL = "\r\n"
GAGE = "RG1"


def _d(t: datetime) -> str:
    return t.strftime("%m/%d/%Y")


def _t(t: datetime) -> str:
    return t.strftime("%H:%M:%S")


def _hm(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60}:{m % 60:02d}"


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _n(v, nd=4) -> str:
    x = round(float(v), nd)
    if x == int(x):
        return str(int(x))
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def _row(*vals, w=17) -> str:
    return "  " + " ".join(str(v).ljust(w - 1) for v in vals).rstrip()


def _section(name: str, header: list[str] | None = None) -> list[str]:
    out = ["", f"[{name}]"]
    if header:
        out += [";;" + header[0]] + [";;" + h for h in header[1:]]
    return out


def options(m: SwmmModel) -> list[str]:
    s = m.settings
    o = [
        ("FLOW_UNITS", "CMS"),
        ("INFILTRATION", "CURVE_NUMBER"),
        ("FLOW_ROUTING", s.routing),
        ("LINK_OFFSETS", "DEPTH"),
        ("MIN_SLOPE", "0"),
        ("ALLOW_PONDING", "NO"),
        ("SKIP_STEADY_STATE", "NO"),
        ("START_DATE", _d(m.start)),
        ("START_TIME", _t(m.start)),
        ("REPORT_START_DATE", _d(m.start)),
        ("REPORT_START_TIME", _t(m.start)),
        ("END_DATE", _d(m.end)),
        ("END_TIME", _t(m.end)),
        ("SWEEP_START", "01/01"),
        ("SWEEP_END", "12/31"),
        ("DRY_DAYS", "0"),
        ("REPORT_STEP", _hms(s.report_step_min * 60)),
        ("WET_STEP", _hms(min(60.0, m.step_min * 60))),
        ("DRY_STEP", "01:00:00"),
        ("ROUTING_STEP", _hms(s.routing_step_s)),
        ("INERTIAL_DAMPING", "PARTIAL"),
        ("NORMAL_FLOW_LIMITED", "BOTH"),
        ("FORCE_MAIN_EQUATION", "H-W"),
        ("VARIABLE_STEP", "0.75"),
        ("LENGTHENING_STEP", "10"),
        ("MIN_SURFAREA", "1.14"),
    ]
    return ["[OPTIONS]"] + [f" {k:<22}{v}" for k, v in o]


def timeseries_rows(ser: Series, start: datetime) -> list[str]:
    rows = [_row(ser.name, _d(start), "0:00", "0")]
    for t, c in zip(ser.times_min[1:], ser.cumulative_mm[1:], strict=True):
        rows.append(_row(ser.name, "", _hm(t), _n(c, 4)))
    return rows


def inp_text(m: SwmmModel, gage_series: str | None = None) -> str:
    """The whole .inp; ``gage_series`` is the storm the rain gage uses
    (default: the first)."""
    s = m.settings
    gs = gage_series or m.series[0].name
    L: list[str] = ["[TITLE]", s.title, ""]
    L += options(m)

    L += _section(
        "RAINGAGES",
        ["Name             Type       Intrvl Catch  Source     Name"],
    )
    L.append(_row(GAGE, "CUMULATIVE", _hm(m.step_min), "1.0", "TIMESERIES", gs, w=11))

    L += _section(
        "SUBCATCHMENTS",
        [
            "Name             Raingage         Outlet           Area(ha) %Imperv  Width    Slope%   CurbLen"
        ],
    )
    for x in m.subcatchments:
        L.append(
            _row(
                x.name,
                GAGE,
                x.outlet,
                _n(x.area_ha, 4),
                _n(x.imperv_pct, 2),
                _n(x.width_m, 2),
                _n(x.slope_pc, 3),
                "0",
            )
        )
    L += _section(
        "SUBAREAS",
        [
            "Subcatchment     N-Imperv   N-Perv     S-Imperv   S-Perv     PctZero    RouteTo"
        ],
    )
    for x in m.subcatchments:
        L.append(
            _row(
                x.name,
                _n(s.n_imperv),
                _n(s.n_perv),
                _n(s.ds_imperv_mm),
                _n(s.ds_perv_mm),
                _n(s.pct_zero),
                "OUTLET",
                w=11,
            )
        )
    L += _section("INFILTRATION", ["Subcatchment     CurveNum   HydCon     DryTime"])
    for x in m.subcatchments:
        L.append(_row(x.name, _n(x.cn, 2), "0.5", _n(s.dry_days), w=11))

    L += _section(
        "JUNCTIONS",
        ["Name             Elevation  MaxDepth   InitDepth  SurDepth   Aponded"],
    )
    for n in m.nodes:
        if n.kind == "junction":
            L.append(
                _row(n.name, _n(n.invert, 3), _n(n.max_depth, 3), "0", "0", "0", w=11)
            )
    L += _section("OUTFALLS", ["Name             Elevation  Type       Gated"])
    for n in m.nodes:
        if n.kind == "outfall":
            L.append(_row(n.name, _n(n.invert, 3), "NORMAL", "NO", w=11))

    L += _section(
        "CONDUITS",
        [
            "Name             From Node        To Node          Length     Roughness  InOffset   OutOffset  InitFlow   MaxFlow"
        ],
    )
    for c in m.conduits:
        L.append(
            _row(
                c.name,
                c.from_node,
                c.to_node,
                _n(c.length_m, 2),
                _n(c.n),
                "0",
                "0",
                "0",
                "0",
            )
        )
    L += _section(
        "XSECTIONS",
        [
            "Link             Shape        Geom1            Geom2      Geom3      Geom4      Barrels"
        ],
    )
    for c in m.conduits:
        if c.shape == "IRREGULAR":
            L.append(_row(c.name, "IRREGULAR", c.name, "0", "0", "0", "1", w=13))
        else:
            L.append(
                _row(
                    c.name,
                    "TRAPEZOIDAL",
                    _n(c.depth_m, 3),
                    _n(c.bottom_width_m, 3),
                    _n(c.side_slope, 3),
                    _n(c.side_slope, 3),
                    "1",
                    w=13,
                )
            )
    tr = [c for c in m.conduits if c.transect is not None]
    if tr:
        L += _section("TRANSECTS", ["Transect data in HEC-2 format"])
        for c in tr:
            t = c.transect
            L.append(f"NC {_n(s.overbank_n)} {_n(s.overbank_n)} {_n(c.n)}")
            L.append(
                f"X1 {t.name} {len(t.stations)} {_n(t.left_bank, 2)} "
                f"{_n(t.right_bank, 2)} 0.0 0.0 0.0 0.0 0.0"
            )
            pairs = [
                f"{_n(z, 3)} {_n(st, 2)}"
                for st, z in zip(t.stations, t.elevations, strict=True)
            ]
            for i in range(0, len(pairs), 5):
                L.append("GR " + " ".join(pairs[i : i + 5]))

    L += _section(
        "TIMESERIES", ["Name             Date       Time       Value (cumulative mm)"]
    )
    for ser in m.series:
        L += timeseries_rows(ser, m.start)
        L.append(";")

    L += _section("REPORT")
    L += [
        " INPUT     NO",
        " CONTROLS  NO",
        " SUBCATCHMENTS ALL",
        " NODES ALL",
        " LINKS ALL",
    ]

    xs = [n.x for n in m.nodes] + [x.x for x in m.subcatchments]
    ys = [n.y for n in m.nodes] + [x.y for x in m.subcatchments]
    for x in m.subcatchments:
        xs += [p[0] for p in x.ring]
        ys += [p[1] for p in x.ring]
    pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    L += _section("MAP")
    L.append(
        f" DIMENSIONS {min(xs) - pad:.3f} {min(ys) - pad:.3f} "
        f"{max(xs) + pad:.3f} {max(ys) + pad:.3f}"
    )
    L.append(" Units      Meters")
    L += _section("COORDINATES", ["Node             X-Coord            Y-Coord"])
    for n in m.nodes:
        L.append(_row(n.name, f"{n.x:.3f}", f"{n.y:.3f}", w=19))
    L += _section("VERTICES", ["Link             X-Coord            Y-Coord"])
    for c in m.conduits:
        for p in c.vertices:
            L.append(_row(c.name, f"{p[0]:.3f}", f"{p[1]:.3f}", w=19))
    L += _section("Polygons", ["Subcatchment     X-Coord            Y-Coord"])
    for x in m.subcatchments:
        ring = x.ring
        if len(ring) < 3:
            h = (x.area_ha * 1e4) ** 0.5 / 2
            ring = [
                (x.x - h, x.y - h),
                (x.x + h, x.y - h),
                (x.x + h, x.y + h),
                (x.x - h, x.y + h),
            ]
        elif ring[0] == ring[-1]:
            ring = ring[:-1]
        for p in ring:
            L.append(_row(x.name, f"{p[0]:.3f}", f"{p[1]:.3f}", w=19))
    L += _section("SYMBOLS", ["Gage             X-Coord            Y-Coord"])
    L.append(_row(GAGE, f"{min(xs):.3f}", f"{max(ys):.3f}", w=19))
    return NL.join(L) + NL


def write_inp(path, m: SwmmModel, gage_series: str | None = None) -> str:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(inp_text(m, gage_series))
    return str(path)


def simulation_hours(m: SwmmModel) -> float:
    return (m.end - m.start) / timedelta(hours=1)
