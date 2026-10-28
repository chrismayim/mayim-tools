"""
Prepare SWMM project - model assembly, zero QGIS dependency.

The network (sub-catchments -> junctions at the stream nodes -> reaches ->
outlets), the curve numbers, the design storms, the time step and the
run-out time come from the Prepare HEC-HMS project core (``assemble``), so
both tools build the same model from the same GIS inputs. This module turns
that into the elements of an EPA SWMM 5 model:

- subcatchments: area (ha), outlet, % impervious, width, slope, Manning n and
  depression storage, Curve Number infiltration;
- junctions at the stream nodes and outfalls at the outlets, with inverts from
  the DEM (lowest cell near the node), forced to fall downstream by at least
  the minimum slope;
- conduits along the reaches: a trapezoid, or an irregular section cut across
  the reach from the DEM (HEC-2 style transect);
- one cumulative rainfall time series per design storm, on a common time step,
  with one rain gage that points to the first storm (switch the gage's time
  series to run another storm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import reduce

import numpy as np

from ..hec_hms_project import storms as stm

TOOL_NAME = "Prepare SWMM project"
TOOL_VERSION = "0.2.1"

ROUTE_DYNWAVE = "DYNWAVE"
ROUTE_KINWAVE = "KINWAVE"
XS_TRAPEZOID = "trapezoid"
XS_TRANSECT = "transect"

WIDTH_FLOW = "w_flow_m"
WIDTH_CHANNEL = "w_chan_m"
WIDTH_GUO = "w_gu_m"
WIDTH_SQRT = "sqrt_area"
WIDTH_FIELDS = (WIDTH_FLOW, WIDTH_CHANNEL, WIDTH_GUO)


class SwmmProjectError(ValueError):
    """A problem the user can fix (inputs or settings)."""


@dataclass
class SwmmSettings:
    title: str = "Prepared with Mayim Tools"
    routing: str = ROUTE_DYNWAVE
    # subcatchments
    width_source: str = WIDTH_FLOW  # field name, or sqrt_area
    default_slope_pc: float = 2.0
    n_imperv: float = 0.015
    n_perv: float = 0.15
    ds_imperv_mm: float = 1.5
    ds_perv_mm: float = 5.0
    pct_zero: float = 25.0
    dry_days: float = 7.0
    # channels
    xs_method: str = XS_TRAPEZOID
    channel_n: float = 0.04
    bottom_width_m: float = 5.0
    channel_depth_m: float = 2.0
    side_slope: float = 2.0  # H:V
    transect_half_width_m: float = 50.0
    transect_points: int = 41
    overbank_n: float = 0.06
    min_slope: float = 0.001  # m/m
    min_length_m: float = 10.0
    node_search_m: float = 15.0
    base_elevation_m: float = 100.0  # without a DEM
    # time
    hyetograph_step_min: float = 5.0  # frequency storms
    routing_step_s: float = 5.0
    # run-out after the longest storm (h); None: 2 x the HEC-HMS run-out, at least
    # 6 h (the SWMM non-linear reservoirs drain more slowly than unit hydrographs)
    tail_h: float | None = None
    report_step_min: float = 1.0


@dataclass
class Subcatchment:
    name: str
    outlet: str
    area_ha: float
    imperv_pct: float
    width_m: float
    slope_pc: float
    cn: float
    x: float
    y: float
    ring: list[tuple[float, float]] = field(default_factory=list)
    width_note: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass
class Node:
    name: str
    kind: str  # junction / outfall
    x: float
    y: float
    invert: float
    max_depth: float
    dem_invert: float | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class Transect:
    name: str
    stations: list[float]
    elevations: list[float]
    left_bank: float
    right_bank: float

    @property
    def depth(self) -> float:
        return max(self.elevations[0], self.elevations[-1]) - min(self.elevations)


@dataclass
class Conduit:
    name: str
    from_node: str
    to_node: str
    length_m: float
    n: float
    vertices: list[tuple[float, float]]
    shape: str  # TRAPEZOIDAL / IRREGULAR
    depth_m: float
    bottom_width_m: float = 0.0
    side_slope: float = 0.0
    transect: Transect | None = None
    flags: list[str] = field(default_factory=list)
    points: list = field(default_factory=list)  # reach line, downstream
    source: str = ""  # GIS reach name

    def slope(self, nodes: dict[str, Node]) -> float:
        return (nodes[self.from_node].invert - nodes[self.to_node].invert) / max(
            self.length_m, 1e-9
        )


@dataclass
class Series:
    name: str  # SWMM time series name (no spaces)
    run_name: str
    storm: object
    times_min: list[float]
    cumulative_mm: list[float]


@dataclass
class SwmmModel:
    settings: SwmmSettings
    hms: object  # hms.ProjectResult (network, storms, runs, ensembles, dt, tail)
    subcatchments: list[Subcatchment]
    nodes: list[Node]
    conduits: list[Conduit]
    series: list[Series]
    step_min: float
    start: datetime
    end: datetime
    warnings: list[str] = field(default_factory=list)
    crs_name: str = ""
    folder: str = ""
    files: dict = field(default_factory=dict)

    def node_map(self) -> dict[str, Node]:
        return {n.name: n for n in self.nodes}


def swmm_id(text: str) -> str:
    """A SWMM object name: no spaces (SWMM splits input lines on blanks)."""
    out = "_".join(str(text).split())
    if not out:
        raise SwmmProjectError(f"'{text}' cannot be used as a SWMM name.")
    return out


# ---------------------------------------------------------------- storms


def storm_cumulative(
    st, step_min: float, block_min: float | None = None
) -> tuple[list[float], list[float]]:
    """Cumulative depth (mm) of a design storm at ``step_min`` intervals from
    0 to its end. Hyetographs: their increments. Frequency storms: the nested
    (alternating block) cumulative curve of their DDF depths, with P % of
    every burst before the peak; the peak is moved by less than one block
    (``block_min``, the frequency storm step) so the shortest burst fills one
    rainfall block exactly instead of being split over two."""
    d = float(st.duration_min)
    if st.is_hyetograph:
        tt = np.arange(len(st.increments) + 1) * st.interval_min
        cc = np.r_[0.0, np.cumsum(st.increments)]
        end = d
    else:
        b = float(block_min or step_min)
        p = float(st.peak_pct) / 100.0
        tt, cc = stm.nested_cumulative(st.depths, d, st.peak_pct)
        k = math.ceil((p * d - p * b) / b - 1e-9)
        shift = k * b + p * b - p * d  # 0 <= shift < b
        tt = np.asarray(tt, dtype=float) + shift
        tt, cc = np.r_[0.0, tt], np.r_[0.0, cc]
        end = d + shift
    n = int(math.ceil(end / step_min - 1e-9))
    t = [i * step_min for i in range(n + 1)]
    c = np.interp(t, tt, cc)
    c = np.maximum.accumulate(c)
    return t, [float(v) for v in c]


def common_step(storms: list, freq_step: float) -> float:
    """Common rainfall step (min) for all storms: the greatest common divisor
    of their steps, so every storm's blocks fall on the gage interval."""
    steps = []
    for st in storms:
        s = st.interval_min if st.is_hyetograph else freq_step
        if abs(s - round(s)) > 1e-6 or s < 1:
            raise SwmmProjectError(
                f"Storm {st.name}: a {s:g} min step is not whole minutes."
            )
        steps.append(int(round(s)))
    return float(reduce(math.gcd, steps))


# ---------------------------------------------------------------- geometry


def largest_ring(wkb: bytes | None) -> list[tuple[float, float]]:
    """Outer ring of the largest polygon of a (multi)polygon WKB."""
    if not wkb:
        return []
    try:
        from osgeo import ogr
    except ImportError:
        return []
    g = ogr.CreateGeometryFromWkb(bytes(wkb))
    if g is None:
        return []
    polys = [g.GetGeometryRef(i) for i in range(g.GetGeometryCount())]
    if g.GetGeometryType() in (ogr.wkbPolygon, ogr.wkbPolygon25D):
        polys = [g]
    best = max(polys, key=lambda p: p.GetArea(), default=None)
    if best is None:
        return []
    ring = best.GetGeometryRef(0)
    return [(ring.GetX(i), ring.GetY(i)) for i in range(ring.GetPointCount())]


def line_points(wkb: bytes | None) -> list[tuple[float, float]]:
    if not wkb:
        return []
    try:
        from osgeo import ogr
    except ImportError:
        return []
    g = ogr.CreateGeometryFromWkb(bytes(wkb))
    if g is None:
        return []
    if g.GetGeometryCount() > 0 and g.GetGeometryType() not in (
        ogr.wkbLineString,
        ogr.wkbLineString25D,
    ):
        g = max(
            (g.GetGeometryRef(i) for i in range(g.GetGeometryCount())),
            key=lambda x: x.Length(),
        )
    return [(g.GetX(i), g.GetY(i)) for i in range(g.GetPointCount())]


class DemSampler:
    """Reads a DEM band once (in the model CRS) for node inverts and
    transects. ``array`` row 0 is the top; ``gt`` the GDAL geotransform."""

    def __init__(self, array: np.ndarray, gt, nodata=None):
        self.a = np.asarray(array, dtype=float)
        if nodata is not None:
            self.a[self.a == nodata] = np.nan
        self.gt = gt

    @classmethod
    def from_file(cls, path: str, dst_wkt: str = "", bounds=None, buffer_m=500.0):
        from osgeo import gdal

        gdal.UseExceptions()
        opts = {"format": "MEM", "dstNodata": -9999.0, "outputType": gdal.GDT_Float32}
        if dst_wkt:
            opts["dstSRS"] = dst_wkt
        if bounds:
            x0, y0, x1, y1 = bounds
            opts["outputBounds"] = (
                x0 - buffer_m,
                y0 - buffer_m,
                x1 + buffer_m,
                y1 + buffer_m,
            )
        ds = gdal.Warp("", path, options=gdal.WarpOptions(**opts))
        arr = ds.GetRasterBand(1).ReadAsArray()
        out = cls(arr, ds.GetGeoTransform(), -9999.0)
        ds = None
        return out

    def _rc(self, x, y):
        gt = self.gt
        return (y - gt[3]) / gt[5], (x - gt[0]) / gt[1]

    def value(self, x: float, y: float) -> float | None:
        r, c = self._rc(x, y)
        r, c = int(math.floor(r)), int(math.floor(c))
        if 0 <= r < self.a.shape[0] and 0 <= c < self.a.shape[1]:
            v = self.a[r, c]
            return None if np.isnan(v) else float(v)
        return None

    def lowest_near(self, x: float, y: float, radius_m: float) -> float | None:
        r, c = self._rc(x, y)
        cell = abs(self.gt[1])
        k = max(1, int(math.ceil(radius_m / cell)))
        r0, c0 = int(math.floor(r)), int(math.floor(c))
        win = self.a[
            max(0, r0 - k) : min(self.a.shape[0], r0 + k + 1),
            max(0, c0 - k) : min(self.a.shape[1], c0 + k + 1),
        ]
        if win.size == 0 or np.all(np.isnan(win)):
            return None
        return float(np.nanmin(win))


def _point_along(pts, frac):
    seg = [
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(pts[:-1], pts[1:], strict=True)
    ]
    total = sum(seg)
    target = frac * total
    acc = 0.0
    for (a, b), s in zip(zip(pts[:-1], pts[1:], strict=True), seg, strict=True):
        if acc + s >= target and s > 0:
            f = (target - acc) / s
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), (
                (b[0] - a[0]) / s,
                (b[1] - a[1]) / s,
            )
        acc += s
    a, b = pts[-2], pts[-1]
    s = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
    return b, ((b[0] - a[0]) / s, (b[1] - a[1]) / s)


def cut_transect(
    dem: DemSampler,
    name: str,
    pts,
    half_width: float,
    n: int,
    bottom_w: float,
    z: float,
    depth: float,
) -> Transect | None:
    """Cross-section perpendicular to the reach at its middle, looking
    downstream (station 0 at the left end). Banks where the trapezoid of the
    default channel would meet the ground."""
    if len(pts) < 2:
        return None
    (cx, cy), (ux, uy) = _point_along(pts, 0.5)
    lx, ly = -uy, ux  # left of the flow direction
    sts, zs = [], []
    for i in range(n):
        s = -half_width + 2 * half_width * i / (n - 1)
        v = dem.value(cx - s * lx, cy - s * ly)  # station increases to the right
        if v is None:
            return None
        sts.append(s + half_width)
        zs.append(v)
    zmin = min(zs)
    imin = zs.index(zmin)
    half = bottom_w / 2 + z * depth
    left = max(0.0, sts[imin] - half)
    right = min(sts[-1], sts[imin] + half)
    return Transect(name, sts, zs, left, right)


# ---------------------------------------------------------------- model


def build_model(
    r,
    s: SwmmSettings,
    sub_extra: dict[str, dict] | None = None,
    dem: DemSampler | None = None,
    reach_points: dict[str, list] | None = None,
    reach_slopes: dict[str, float] | None = None,
) -> SwmmModel:
    """SWMM model from an assembled HEC-HMS ProjectResult ``r``.
    ``sub_extra``: {sub-catchment name: {'w_flow_m': .., 's_mean_pc': ..}}
    ('Catchments: 1. Delineation' fields); ``reach_points``: {reach source name: xy
    downstream}; ``reach_slopes``: {reach source name: slope m/m} (inverts
    without a DEM)."""
    sub_extra = sub_extra or {}
    reach_points = reach_points or {}
    reach_slopes = reach_slopes or {}
    warnings: list[str] = []
    net = r.network

    # ---- subcatchments
    subs = []
    for e in net.of("Subbasin"):
        p = e.params
        ex = sub_extra.get(e.source, {})
        a_m2 = p["area_km2"] * 1e6
        note = ""
        w = None
        if s.width_source in WIDTH_FIELDS:
            w = ex.get(s.width_source)
            note = s.width_source
        if not w or not math.isfinite(w) or w <= 0:
            w = math.sqrt(a_m2)
            note = (
                "sqrt(area)"
                if s.width_source == WIDTH_SQRT
                else (f"sqrt(area) - no {s.width_source}")
            )
        slope = ex.get("s_mean_pc")
        flags = []
        if slope is None or not math.isfinite(slope) or slope <= 0:
            slope = s.default_slope_pc
            flags.append("default_slope")
        subs.append(
            Subcatchment(
                name=swmm_id(e.name),
                outlet=swmm_id(e.downstream),
                area_ha=a_m2 / 1e4,
                imperv_pct=float(p.get("imperv_pct") or 0.0),
                width_m=float(w),
                slope_pc=float(slope),
                cn=float(p["cn"]),
                x=e.x,
                y=e.y,
                ring=largest_ring(e.wkb),
                width_note=note,
                flags=flags,
            )
        )
    nsqrt = sum(1 for x in subs if x.width_note.startswith("sqrt(area) - no"))
    if nsqrt:
        warnings.append(
            f"{nsqrt} sub-catchment(s) have no '{s.width_source}' field value; their "
            "width is sqrt(area) (the 'Catchments: 1. Delineation' widths are better)."
        )
    if any("default_slope" in x.flags for x in subs):
        warnings.append(
            f"Sub-catchments without a mean slope (s_mean_pc) use {s.default_slope_pc:g} %."
        )

    # ---- conduits (topology)
    reaches = net.of("Reach")
    up_of = {}
    for e in net.elements:
        if e.kind == "Junction" and e.downstream:
            up_of[e.downstream] = e.name
    conduits = []
    for e in reaches:
        frm = up_of.get(e.name)
        if frm is None:
            raise SwmmProjectError(
                f"Reach {e.name} has no junction at its upstream end."
            )
        length = float(e.params.get("length_m") or 0.0)
        if length < s.min_length_m:
            warnings.append(
                f"Conduit {e.name}: length {length:.1f} m raised to {s.min_length_m:g} m."
            )
            length = s.min_length_m
        pts = reach_points.get(e.source) or line_points(e.wkb)
        verts = [tuple(p) for p in pts[1:-1]] if len(pts) > 2 else []
        conduits.append(
            Conduit(
                name=swmm_id(e.name),
                from_node=swmm_id(frm),
                to_node=swmm_id(e.downstream),
                length_m=length,
                n=s.channel_n,
                vertices=verts,
                shape="TRAPEZOIDAL",
                depth_m=s.channel_depth_m,
                bottom_width_m=s.bottom_width_m,
                side_slope=s.side_slope,
                points=pts,
                source=e.source,
            )
        )

    # ---- nodes and inverts
    nodes: dict[str, Node] = {}
    for e in net.elements:
        if e.kind not in ("Junction", "Sink"):
            continue
        nm = swmm_id(e.name)
        dem_inv = dem.lowest_near(e.x, e.y, s.node_search_m) if dem else None
        nodes[nm] = Node(
            nm,
            "junction" if e.kind == "Junction" else "outfall",
            e.x,
            e.y,
            dem_inv if dem_inv is not None else float("nan"),
            s.channel_depth_m,
            dem_invert=dem_inv,
        )
    if dem is None or any(math.isnan(n.invert) for n in nodes.values()):
        if dem is not None:
            warnings.append(
                "Some nodes are outside the DEM; their inverts follow the reach slopes."
            )
        _inverts_from_slopes(nodes, conduits, reach_slopes, s, warnings)

    # downstream fall >= minimum slope (process upstream reaches first)
    order = _upstream_first(conduits)
    lowered = 0
    for c in order:
        up, dn = nodes[c.from_node], nodes[c.to_node]
        need = up.invert - s.min_slope * c.length_m
        if dn.invert > need + 1e-9:
            dn.invert = need
            dn.flags.append("lowered")
            lowered += 1
    if lowered:
        warnings.append(
            f"{lowered} node invert(s) were lowered below the DEM so every conduit falls "
            f"at least {s.min_slope:g} m/m downstream (check the DEM drainage)."
        )

    # ---- cross-sections
    if s.xs_method == XS_TRANSECT:
        if dem is None:
            warnings.append(
                "No DEM: trapezoidal sections used instead of DEM transects."
            )
        else:
            nfail = 0
            for c in conduits:
                t = cut_transect(
                    dem,
                    c.name,
                    c.points,
                    s.transect_half_width_m,
                    s.transect_points,
                    s.bottom_width_m,
                    s.side_slope,
                    s.channel_depth_m,
                )
                if t is None or t.depth <= 0.05:
                    nfail += 1
                    continue
                c.shape = "IRREGULAR"
                c.transect = t
                c.depth_m = t.depth
            if nfail:
                warnings.append(
                    f"{nfail} reach(es) could not be cut from the DEM (too short, flat or "
                    "outside it) and keep the trapezoid."
                )
    for c in conduits:
        for nm in (c.from_node, c.to_node):
            nodes[nm].max_depth = max(nodes[nm].max_depth, c.depth_m)

    # ---- storms
    storms = [run.storm for run in r.runs]
    step = common_step(storms, s.hyetograph_step_min)
    series = []
    for run in r.runs:
        t, c = storm_cumulative(run.storm, step, s.hyetograph_step_min)
        series.append(Series(swmm_id(run.storm.name), run.name, run.storm, t, c))
    names = [x.name for x in series]
    if len(set(names)) != len(names):
        raise SwmmProjectError("Two design storms have the same SWMM time series name.")
    start = r.settings.start
    longest = max(float(st.duration_min) for st in storms)
    tail = s.tail_h if s.tail_h else max(2.0 * r.tail_h, 6.0)
    end = start + timedelta(minutes=longest + tail * 60.0)
    return SwmmModel(
        settings=s,
        hms=r,
        subcatchments=subs,
        nodes=list(nodes.values()),
        conduits=conduits,
        series=series,
        step_min=step,
        start=start,
        end=end,
        warnings=warnings,
    )


def _upstream_first(conduits: list[Conduit]) -> list[Conduit]:
    """Conduits ordered so each comes after every conduit flowing into its
    upstream node."""
    into: dict[str, list[Conduit]] = {}
    for c in conduits:
        into.setdefault(c.to_node, []).append(c)
    done: set[str] = set()
    out: list[Conduit] = []

    def visit(c: Conduit, stack: set):
        if c.name in done:
            return
        if c.name in stack:
            raise SwmmProjectError(f"The network has a loop at conduit {c.name}.")
        stack.add(c.name)
        for u in into.get(c.from_node, []):
            visit(u, stack)
        stack.discard(c.name)
        done.add(c.name)
        out.append(c)

    for c in conduits:
        visit(c, set())
    return out


def _inverts_from_slopes(nodes, conduits, slopes, s: SwmmSettings, warnings) -> None:
    """Without a DEM: outfalls at the base elevation, each upstream node at
    its downstream node + slope x length (reach slope field, else the
    minimum slope)."""
    by_to: dict[str, list[Conduit]] = {}
    for c in conduits:
        by_to.setdefault(c.to_node, []).append(c)
    nodefault = 0
    for n in nodes.values():
        if n.kind == "outfall" and math.isnan(n.invert):
            n.invert = s.base_elevation_m
    changed = True
    while changed:
        changed = False
        for c in conduits:
            dn, up = nodes[c.to_node], nodes[c.from_node]
            if not math.isnan(dn.invert) and math.isnan(up.invert):
                sl = slopes.get(c.source)
                if sl is None or not math.isfinite(sl) or sl <= 0:
                    sl = s.min_slope
                    nodefault += 1
                up.invert = dn.invert + sl * c.length_m
                changed = True
    for n in nodes.values():
        if math.isnan(n.invert):
            n.invert = s.base_elevation_m
    warnings.append(
        f"No DEM inverts: outlets at {s.base_elevation_m:g} m and nodes raised by the "
        "reach slopes (relative elevations only)."
    )
    if nodefault:
        warnings.append(
            f"{nodefault} reach(es) without a slope use {s.min_slope:g} m/m."
        )
