"""
Prepare HEC-HMS project - core logic, zero QGIS dependency.

Builds a HEC-HMS 4.14 (or 4.13) project (basin model, meteorologic models, control
specifications, simulation runs, precipitation gages, terrain and the
basin geometry) from GIS outputs:

- sub-catchments -> Subbasin elements (area; SCS curve number, Initial and
  Constant, Deficit and Constant or Green and Ampt loss; SCS unit
  hydrograph, Clark or Snyder transform from Tc - see methods.py);
- stream reaches -> Reach elements (Muskingum, Lag, Muskingum-Cunge, Lag
  and K or Kinematic Wave);
- stream nodes -> Junction elements; a node with no reach leaving it is
  the outlet of its system and becomes a Sink.

The topology follows the naming of Catchment Delineation (incremental mode)
and Stream Network: stream A1 drains to node A1 (the outlet), stream A2 to
node A2 at the top of A1, a branch A3A1 to the confluence node at the top
of A3 (A4); a sub-catchment's ``node`` field names the node it drains to.
So reach R starts at the node named after its stem plus the next number
(top of A3 is A4) and ends at the top node of the stream it flows into
(``ds_name``), or at its own name at an outlet.

HEC-HMS needs element names to be unique across element types, so reaches,
junctions and sinks get a prefix (R-A1, J-A2, Outlet-A1 by default).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from . import methods as mth

TOOL_NAME = "Prepare HEC-HMS project"
TOOL_VERSION = "0.5.0"
HMS_VERSION = "4.14"  # default; 4.13 is also supported
HMS_VERSIONS = ["4.14", "4.13"]

LOSS_SCS = "scs"
TRANSFORM_SCS = "scs"
TRANSFORM_CLARK = "clark"
TRANSFORM_SNYDER = mth.TRANSFORM_SNYDER
ROUTE_LAGK = mth.ROUTE_LAGK
ROUTE_KW = mth.ROUTE_KW
ROUTE_MUSKINGUM = "muskingum"
ROUTE_LAG = "lag"
ROUTE_MC = "muskingum_cunge"
CN_MEAN = "mean"
CN_RUNOFF = "runoff"

# Control time steps (min) the automatic choice picks from.
TIME_STEPS = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
AUTO_DT_MAX = 15.0
SCS_DT_LAG_RATIO = 0.29  # HEC-HMS: time step <= 0.29 x lag (SCS UH)

_TRAILING_NUMBER = re.compile(r"^(.*?)(\d+)$")
_BAD_NAME = re.compile(r"[/\\:,\r\n\t]+")


class HmsProjectError(ValueError):
    """Input that cannot be turned into a HEC-HMS project."""


# ---------------------------------------------------------------- inputs


@dataclass
class SubIn:
    """One sub-catchment, in the basin CRS (metres)."""

    name: str
    area_km2: float
    cx: float
    cy: float
    node: str | None = None
    tc_min: float | None = None
    cn: float | None = None
    cn_stats: dict | None = None
    imperv_pct: float = 0.0
    lat: float | None = None
    lon: float | None = None
    x_out: float | None = None
    y_out: float | None = None
    wkb: bytes | None = None
    sub_type: str = ""
    bank: str = ""
    attrs: dict | None = None  # numeric fields of the layer (parameter overrides)
    soil: dict | None = None  # zonal soil parameters (methods.SOIL_CODES)


@dataclass
class ReachIn:
    """One stream reach; ``xy`` runs downstream."""

    name: str
    ds_name: str | None = None
    xy: list[tuple[float, float]] = field(default_factory=list)
    length_m: float | None = None
    slope: float | None = None
    musk_k_h: float | None = None
    musk_x: float | None = None
    tt_min: float | None = None
    wkb: bytes | None = None
    attrs: dict | None = None  # numeric fields of the layer (parameter overrides)


@dataclass
class NodeIn:
    name: str
    x: float
    y: float


@dataclass
class Settings:
    project_name: str = "HMS Project"
    hms_version: str = HMS_VERSION
    basin_name: str = "Existing"
    description: str = ""
    subbasin_prefix: str = ""
    reach_prefix: str = "R-"
    junction_prefix: str = "J-"
    sink_prefix: str = "Outlet-"
    # loss
    lambda_ia: float = 0.1
    cn_method: str = CN_MEAN
    cn_design_depth_mm: float = 50.0
    # transform
    transform: str = TRANSFORM_SCS
    tc_floor_min: float = 5.0
    clark_ratio: float = 0.5  # R / (Tc + R)
    # routing
    routing: str = ROUTE_MUSKINGUM
    velocity_ms: float = 1.5
    musk_x: float = 0.2
    mc_n: float = 0.04
    mc_width_m: float = 5.0
    mc_side_slope: float = 2.0
    min_slope: float = 0.0005
    # time
    dt_min: float | None = None  # None: automatic
    tail_h: float | None = None  # None: automatic
    start: datetime = datetime(2000, 1, 1, 0, 0)
    time_zone: str = "UTC"
    # loss / transform / routing method details (methods.py)
    methods: mth.MethodSettings = field(default_factory=mth.MethodSettings)
    # rainfall
    storm_area_km2: float | None = None
    # one HEC-HMS ensemble analysis per storm method and return period; the
    # runs then share one control specification (HEC-HMS ensemble members
    # must have the same time window)
    ensembles: bool = True


# ---------------------------------------------------------------- elements


@dataclass
class Element:
    kind: str  # Subbasin, Reach, Junction, Sink
    name: str
    source: str
    x: float
    y: float
    downstream: str | None = None
    from_xy: tuple[float, float] | None = None
    params: dict = field(default_factory=dict)
    description: str = ""
    wkb: bytes | None = None
    flags: list[str] = field(default_factory=list)


@dataclass
class Network:
    elements: list[Element]
    warnings: list[str]

    def of(self, kind: str) -> list[Element]:
        return [e for e in self.elements if e.kind == kind]

    def by_name(self) -> dict[str, Element]:
        return {e.name: e for e in self.elements}


def hms_name(text: str) -> str:
    """An element / model name HEC-HMS can store (no slashes, colons or
    commas, which break its files or DSS pathnames)."""
    out = _BAD_NAME.sub("-", str(text).strip())
    out = re.sub(r"\s+", " ", out).strip(" -")
    if not out:
        raise HmsProjectError(f"'{text}' cannot be used as a HEC-HMS name.")
    return out


def file_stem(text: str) -> str:
    """A file name stem for HEC-HMS component files. HEC-HMS itself keeps
    only letters, digits, '_', '+' and '.' in the files it names; anything
    else (spaces, hyphens) becomes '_'."""
    out = re.sub(r"[^A-Za-z0-9._+]+", "_", str(text).strip())
    out = re.sub(r"_+", "_", out).strip("._")
    if not out:
        raise HmsProjectError(f"'{text}' cannot be used as a file name.")
    return out


def project_stem(text: str) -> str:
    """The HEC-HMS project name, also the stem of the .hms file. HEC-HMS
    rewrites every character other than A-Z, a-z and 0-9 in the project name
    to '_' and then looks for <that>.hms, .run, .gage ... - a project named
    'HEC-HMS_Model' therefore opens empty. Only [A-Za-z0-9_] is used."""
    out = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip())
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        raise HmsProjectError(f"'{text}' cannot be used as a HEC-HMS project name.")
    return out


def folder_stem(text: str) -> str:
    """Project folder name (Windows-safe; hyphens are kept)."""
    out = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(text).strip())
    out = re.sub(r"_+", "_", out).strip("._")
    if not out:
        raise HmsProjectError(f"'{text}' cannot be used as a folder name.")
    return out


def top_node(stream: str) -> str | None:
    """Node at the upstream end of a stream: stem + next number (A3 -> A4,
    A3A1 -> A3A2)."""
    m = _TRAILING_NUMBER.match(str(stream))
    if not m:
        return None
    return f"{m.group(1)}{int(m.group(2)) + 1}"


def node_below(reach: ReachIn) -> str:
    """Node at the downstream end of a reach: the top node of the stream it
    flows into, or its own name at an outlet."""
    if reach.ds_name:
        n = top_node(reach.ds_name)
        if n is None:
            raise HmsProjectError(
                f"Stream '{reach.ds_name}' (downstream of {reach.name}) does not "
                "follow the stem + number naming (A1, A3A2 ...)."
            )
        return n
    return reach.name


def natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


# ---------------------------------------------------------------- SCS CN


def potential_retention(cn: float) -> float:
    """S (mm) = 25400 / CN - 254."""
    return 25400.0 / cn - 254.0


def scs_runoff(p_mm, cn, lam: float):
    """Direct runoff Q (mm) of the SCS curve number method, elementwise."""
    p = np.asarray(p_mm, dtype=float)
    s = 25400.0 / np.asarray(cn, dtype=float) - 254.0
    ia = lam * s
    q = np.where(p > ia, (p - ia) ** 2 / np.maximum(p - ia + s, 1e-12), 0.0)
    return q


def runoff_equivalent_cn(cn_cells: np.ndarray, p_mm: float, lam: float) -> float | None:
    """The single CN giving the zone's mean runoff at depth ``p_mm``."""
    cn_cells = np.asarray(cn_cells, dtype=float)
    cn_cells = cn_cells[np.isfinite(cn_cells)]
    if cn_cells.size == 0:
        return None
    q_mean = float(scs_runoff(p_mm, np.clip(cn_cells, 1.0, 100.0), lam).mean())
    if q_mean <= 0:
        return None
    lo, hi = 1.0, 99.99
    if float(scs_runoff(p_mm, hi, lam)) < q_mean:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(scs_runoff(p_mm, mid, lam)) < q_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def zone_cn(cn_cells: np.ndarray, method: str, p_mm: float, lam: float) -> dict:
    """Curve number statistics of one zone's cells."""
    v = np.asarray(cn_cells, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return {"n_cells": 0}
    mean = float(v.mean())
    req = runoff_equivalent_cn(v, p_mm, lam)
    out = {
        "n_cells": int(v.size),
        "cn_mean": mean,
        "cn_runoff": req,
        "cn_p05": float(np.percentile(v, 5)),
        "cn_p95": float(np.percentile(v, 95)),
        "cn_min": float(v.min()),
        "cn_max": float(v.max()),
    }
    out["cn"] = req if (method == CN_RUNOFF and req is not None) else mean
    return out


def zonal_cn_from_raster(
    path: str,
    zones_wkt: list[str],
    method: str = CN_MEAN,
    p_mm: float = 50.0,
    lam: float = 0.1,
    band_description: str = "cn_design",
) -> tuple[list[dict], str]:
    """Curve number per zone from a CN raster (zones in the raster's CRS).
    The band whose description is ``band_description`` is used (the SCS
    Curve Numbers tool's design band), else band 1. A zone too small to
    hold a cell centre takes the cell under its centroid."""
    from osgeo import gdal, ogr

    gdal.UseExceptions()
    ds = gdal.Open(path)
    if ds is None:
        raise HmsProjectError(f"Cannot open the curve number raster {path}.")
    band_no = 1
    for b in range(1, ds.RasterCount + 1):
        if (ds.GetRasterBand(b).GetDescription() or "").strip().lower() == (
            band_description.lower()
        ):
            band_no = b
            break
    band = ds.GetRasterBand(band_no)
    label = f"band {band_no}" + (
        f" ({band.GetDescription()})" if band.GetDescription() else ""
    )
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    srs = ds.GetSpatialRef()
    mem_drv = gdal.GetDriverByName("MEM")
    ogr_drv = ogr.GetDriverByName("Memory")
    out = []
    # each zone on its own window, so nested (overlapping) catchments work
    for wkt in zones_wkt:
        g = ogr.CreateGeometryFromWkt(wkt)
        minx, maxx, miny, maxy = g.GetEnvelope()
        c0 = max(0, int(math.floor((minx - gt[0]) / gt[1])) - 1)
        c1 = min(ds.RasterXSize, int(math.ceil((maxx - gt[0]) / gt[1])) + 1)
        r0 = max(0, int(math.floor((maxy - gt[3]) / gt[5])) - 1)
        r1 = min(ds.RasterYSize, int(math.ceil((miny - gt[3]) / gt[5])) + 1)
        if c1 <= c0 or r1 <= r0:
            out.append({"n_cells": 0})
            continue
        arr = band.ReadAsArray(c0, r0, c1 - c0, r1 - r0).astype(float)
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[(arr <= 0) | (arr > 100)] = np.nan
        mem = mem_drv.Create("", c1 - c0, r1 - r0, 1, gdal.GDT_Byte)
        mem.SetGeoTransform(
            (gt[0] + c0 * gt[1], gt[1], 0, gt[3] + r0 * gt[5], 0, gt[5])
        )
        mem.SetProjection(ds.GetProjection())
        vds = ogr_drv.CreateDataSource("zone")
        lyr = vds.CreateLayer("z", srs, ogr.wkbMultiPolygon)
        f = ogr.Feature(lyr.GetLayerDefn())
        f.SetGeometry(g)
        lyr.CreateFeature(f)
        gdal.RasterizeLayer(mem, [1], lyr, burn_values=[1])
        mask = mem.GetRasterBand(1).ReadAsArray() == 1
        st = zone_cn(arr[mask], method, p_mm, lam)
        if st["n_cells"] == 0:
            c = g.Centroid()
            col = int((c.GetX() - gt[0]) / gt[1]) - c0
            row = int((c.GetY() - gt[3]) / gt[5]) - r0
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                st = zone_cn(np.array([arr[row, col]]), method, p_mm, lam)
                st["centroid_only"] = True
        out.append(st)
    if all(st["n_cells"] == 0 for st in out):
        raise HmsProjectError("The sub-catchments do not overlap the CN raster.")
    return out, label


# ---------------------------------------------------------------- network


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_network(
    subs: list[SubIn],
    reaches: list[ReachIn],
    nodes: list[NodeIn],
    s: Settings,
) -> Network:
    """Elements and links from the GIS inputs (parameters added later)."""
    warnings: list[str] = []
    if not subs:
        raise HmsProjectError("There are no sub-catchments.")
    names = [x.name for x in subs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise HmsProjectError(f"Sub-catchment names occur more than once: {dup}")
    rnames = [r.name for r in reaches]
    dup = sorted({n for n in rnames if rnames.count(n) > 1})
    if dup:
        raise HmsProjectError(f"Reach names occur more than once: {dup}")

    node_xy: dict[str, tuple[float, float]] = {n.name: (n.x, n.y) for n in nodes}
    starts: dict[str, ReachIn] = {}
    ends: dict[str, str] = {}
    for r in reaches:
        t = top_node(r.name)
        if t is None:
            raise HmsProjectError(
                f"Reach '{r.name}' does not follow the stem + number naming "
                "(A1, A3A2 ...) of Catchment Delineation / Stream Network."
            )
        if r.ds_name and r.ds_name not in rnames:
            warnings.append(
                f"Reach {r.name} flows into '{r.ds_name}', which is not in the "
                "reach layer; it ends at that stream's top node."
            )
        if t in starts:
            raise HmsProjectError(
                f"Reaches {starts[t].name} and {r.name} both start at node {t}."
            )
        starts[t] = r
        ends[r.name] = node_below(r)
        if r.xy:
            node_xy.setdefault(t, r.xy[0])
            node_xy.setdefault(ends[r.name], r.xy[-1])

    # nodes referenced by sub-catchments
    for x in subs:
        if x.node and x.node not in node_xy:
            if x.x_out is not None and x.y_out is not None:
                node_xy[x.node] = (x.x_out, x.y_out)

    elements: list[Element] = []
    used: dict[str, str] = {}

    def claim(name: str, what: str) -> str:
        n = hms_name(name)
        if n in used:
            raise HmsProjectError(
                f"HEC-HMS name '{n}' is used by {used[n]} and {what}; element names "
                "must be unique across sub-basins, reaches, junctions and sinks - "
                "set different prefixes."
            )
        used[n] = what
        return n

    node_names = set(ends.values()) | set(starts)
    node_names |= {x.node for x in subs if x.node}
    node_el: dict[str, str] = {}
    for nn in sorted(node_names, key=natural_key):
        if nn in starts:
            node_el[nn] = claim(s.junction_prefix + nn, f"junction {nn}")
        else:
            node_el[nn] = claim(s.sink_prefix + nn, f"sink {nn}")

    reach_el = {
        r.name: claim(s.reach_prefix + r.name, f"reach {r.name}") for r in reaches
    }

    for x in subs:
        nm = claim(s.subbasin_prefix + x.name, f"sub-catchment {x.name}")
        if x.node:
            ds = node_el[x.node]
        else:
            ds = claim(s.sink_prefix + x.name, f"sink of {x.name}")
            node_xy[x.name + "\x00sink"] = (
                (x.x_out, x.y_out) if x.x_out is not None else (x.cx, x.cy)
            )
            elements.append(
                Element(
                    "Sink",
                    ds,
                    x.name,
                    *node_xy[x.name + "\x00sink"],
                    description=f"Outlet of sub-catchment {x.name}",
                )
            )
        desc = "Sub-catchment " + x.name
        if x.sub_type:
            desc += f" ({x.sub_type}" + (f", bank {x.bank}" if x.bank else "") + ")"
        elements.append(
            Element("Subbasin", nm, x.name, x.cx, x.cy, ds, description=desc, wkb=x.wkb)
        )

    for nn in sorted(node_names, key=natural_key):
        xy = node_xy.get(nn)
        if xy is None:
            pts = [(x.cx, x.cy) for x in subs if x.node == nn]
            if not pts:
                raise HmsProjectError(f"No position for node {nn}.")
            xy = (
                float(np.mean([p[0] for p in pts])),
                float(np.mean([p[1] for p in pts])),
            )
            warnings.append(f"Node {nn}: no position given; placed at its sub-basins.")
        if nn in starts:
            elements.append(
                Element(
                    "Junction",
                    node_el[nn],
                    nn,
                    xy[0],
                    xy[1],
                    reach_el[starts[nn].name],
                    description=f"Stream node {nn}",
                )
            )
        else:
            elements.append(
                Element(
                    "Sink",
                    node_el[nn],
                    nn,
                    xy[0],
                    xy[1],
                    description=f"Outlet node {nn}",
                )
            )
        node_xy[nn] = xy

    for r in reaches:
        down = ends[r.name]
        up = top_node(r.name)
        elements.append(
            Element(
                "Reach",
                reach_el[r.name],
                r.name,
                node_xy[down][0],
                node_xy[down][1],
                node_el[down],
                from_xy=node_xy.get(up),
                description=f"Stream {r.name}",
                wkb=r.wkb,
            )
        )

    net = Network(elements, warnings)
    _order(net)
    return net


def _order(net: Network) -> None:
    """Check every element reaches a sink; sort upstream first."""
    by = net.by_name()
    depth: dict[str, int] = {}
    for e in net.elements:
        seen, cur, n = set(), e, 0
        while cur.downstream is not None:
            if cur.name in seen:
                raise HmsProjectError(f"The network loops at element {cur.name}.")
            seen.add(cur.name)
            nxt = by.get(cur.downstream)
            if nxt is None:
                raise HmsProjectError(
                    f"{e.kind} {e.name} drains to '{cur.downstream}', which does not exist."
                )
            cur, n = nxt, n + 1
        depth[e.name] = n
    rank = {"Subbasin": 0, "Junction": 1, "Reach": 2, "Sink": 3}
    net.elements.sort(
        key=lambda e: (-depth[e.name], natural_key(e.source), rank[e.kind])
    )


def system_areas(
    subs: list[SubIn], reaches: list[ReachIn], nodes: list[NodeIn], s: Settings
) -> dict[str, float]:
    """Total sub-catchment area (km2) draining to each outlet (sink), largest
    first - the area for the areal reduction factor."""
    net = build_network(subs, reaches, nodes, s)
    by = net.by_name()
    by_sub = {x.name: x for x in subs}
    out: dict[str, float] = {}
    for e in net.of("Subbasin"):
        cur = e
        while cur.downstream is not None:
            cur = by[cur.downstream]
        out[cur.name] = out.get(cur.name, 0.0) + by_sub[e.source].area_km2
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- parameters


def choose_dt(min_lag_min: float, gage_intervals: list[float]) -> float:
    """Largest standard time step <= 0.29 x the shortest lag (and no
    longer than the finest gage interval), capped at 15 min."""
    limit = min(AUTO_DT_MAX, SCS_DT_LAG_RATIO * min_lag_min)
    if gage_intervals:
        limit = min(limit, min(gage_intervals))
    ok = [t for t in TIME_STEPS if t <= limit + 1e-9]
    return float(ok[-1]) if ok else 1.0


def muskingum_steps(k_min: float, x: float, dt: float) -> int | None:
    """Sub-reaches n with 1/(2(1-X)) <= K/(n dt) <= 1/(2X), nearest K/dt;
    None when the reach is too short for Muskingum at this time step."""
    if k_min <= 0 or dt <= 0:
        return None
    lo = max(1, math.ceil(2.0 * x * k_min / dt - 1e-9)) if x > 0 else 1
    hi = math.floor(2.0 * (1.0 - x) * k_min / dt + 1e-9)
    if hi < lo:
        return None
    n = int(round(k_min / dt))
    return min(max(n, lo), hi)


def subbasin_parameters(e: Element, x: SubIn, s: Settings) -> list[str]:
    """Loss and transform parameters; returns warnings."""
    warns = []
    p = e.params
    p["area_km2"] = x.area_km2
    ms = s.methods
    p["loss"] = ms.loss
    sources = p.setdefault("sources", {})
    has_cn = x.cn is not None and 0 < x.cn <= 100
    if not has_cn and ms.loss == mth.LOSS_SCS:
        raise HmsProjectError(f"Sub-catchment {x.name} has no curve number.")
    if has_cn:
        cn = min(max(float(x.cn), 30.0), 99.0)
        if cn != x.cn:
            warns.append(f"{x.name}: CN {x.cn:.1f} limited to {cn:g}.")
        p["cn"] = round(cn, 1)
        p["s_mm"] = potential_retention(p["cn"])
        p["ia_mm"] = s.lambda_ia * p["s_mm"]
    p["imperv_pct"] = min(max(float(x.imperv_pct or 0.0), 0.0), 100.0)
    if x.cn_stats:
        p.update({k: v for k, v in x.cn_stats.items() if k != "cn"})
    for code, v in (x.soil or {}).items():
        if v is not None:
            p[f"soil_{code}"] = v
    if ms.loss != mth.LOSS_SCS:
        try:
            lp, src, w = mth.loss_parameters(
                x.name, x.attrs, x.soil, p.get("cn"), p.get("ia_mm"), p.get("s_mm"), ms
            )
        except mth.MethodError as err:
            raise HmsProjectError(str(err)) from err
        p.update(lp)
        sources.update(src)
        warns += w
    tc = x.tc_min
    if tc is None or not math.isfinite(tc) or tc <= 0:
        raise HmsProjectError(f"Sub-catchment {x.name} has no time of concentration.")
    p["tc_raw_min"] = float(tc)
    if tc < s.tc_floor_min:
        e.flags.append("tc_floor")
        tc = s.tc_floor_min
    p["tc_min"] = float(tc)
    if s.transform == TRANSFORM_CLARK:
        ratio = min(max(s.clark_ratio, 0.01), 0.99)
        p["clark_tc_h"] = tc / 60.0
        p["clark_r_h"] = ratio / (1.0 - ratio) * tc / 60.0
        p["lag_min"] = 0.6 * tc
    elif s.transform == TRANSFORM_SNYDER:
        sp, src, w = mth.snyder_parameters(x.name, x.attrs, tc, ms)
        p.update(sp)
        sources.update(src)
        warns += w
        p["lag_min"] = sp["snyder_tp_h"] * 60.0
    else:
        p["lag_min"] = 0.6 * tc
    return warns


def reach_parameters(e: Element, r: ReachIn, s: Settings, dt: float) -> list[str]:
    warns = []
    p = e.params
    length = r.length_m
    if (length is None or length <= 0) and len(r.xy) > 1:
        length = sum(_dist(a, b) for a, b in zip(r.xy[:-1], r.xy[1:], strict=True))
    p["length_m"] = length
    v = s.velocity_ms
    k_min = None
    if r.musk_k_h is not None and r.musk_k_h > 0:
        k_min = r.musk_k_h * 60.0
        p["k_source"] = "field"
    elif r.tt_min is not None and r.tt_min > 0:
        k_min = r.tt_min
        p["k_source"] = "travel time field"
    elif length:
        k_min = length / v / 60.0
        p["k_source"] = f"length / {v:g} m/s"
    p["travel_min"] = k_min
    method = s.routing
    if method in (ROUTE_MC, ROUTE_KW):
        slope = r.slope
        if slope is None or not math.isfinite(slope):
            raise HmsProjectError(
                f"Reach {r.name} has no slope (needed for Muskingum-Cunge)."
            )
        if slope < s.min_slope:
            e.flags.append("min_slope")
            slope = s.min_slope
        if not length:
            raise HmsProjectError(f"Reach {r.name} has no length.")
        sources = p.setdefault("sources", {})
        n_val, width = s.mc_n, s.mc_width_m
        if method == ROUTE_KW:
            fn, fw = mth.field_value(r.attrs, "kw_n"), mth.field_value(
                r.attrs, "kw_width_m"
            )
            if fn is not None and fn > 0:
                n_val, sources["n"] = fn, "field kw_n"
            if fw is not None and fw >= 0:
                width, sources["width_m"] = fw, "field kw_width_m"
        p.update(
            method=method,
            slope=slope,
            n=n_val,
            width_m=width,
            side_slope=s.mc_side_slope,
            celerity_ms=v,
        )
        if method == ROUTE_KW:
            p["subreaches"] = mth.kw_subreaches(length, v, dt)
        return warns
    if k_min is None:
        raise HmsProjectError(f"Reach {r.name} has no travel time or length.")
    if method == ROUTE_LAGK:
        lp, src = mth.lagk_parameters(r.attrs, k_min, s.methods)
        p.update(method=ROUTE_LAGK, **lp)
        p.setdefault("sources", {}).update(src)
        return warns
    if method == ROUTE_MUSKINGUM:
        x = r.musk_x if (r.musk_x is not None and 0 <= r.musk_x <= 0.5) else s.musk_x
        n = muskingum_steps(k_min, x, dt)
        if n is not None:
            p.update(method=ROUTE_MUSKINGUM, k_h=k_min / 60.0, x=x, steps=n)
            return warns
        e.flags.append("lag_fallback")
        warns.append(
            f"Reach {r.name}: K = {k_min:.1f} min is too short for Muskingum at a "
            f"{dt:g} min time step; Lag routing used."
        )
    p.update(method=ROUTE_LAG, lag_min=round(k_min, 2))
    return warns


def path_times(net: Network) -> dict[str, float]:
    """Sub-basin lag + routing time down to the sink (min)."""
    by = net.by_name()
    out = {}
    for e in net.of("Subbasin"):
        t, cur = e.params.get("lag_min", 0.0), by.get(e.downstream)
        while cur is not None:
            if cur.kind == "Reach":
                t += cur.params.get("travel_min") or 0.0
            cur = by.get(cur.downstream) if cur.downstream else None
        out[e.name] = t
    return out


def auto_tail_hours(net: Network) -> float:
    """Simulation time after the storm: 5 x lag plus routing, rounded up to
    the hour (at least 1 h)."""
    by = net.by_name()
    worst = 0.0
    for e in net.of("Subbasin"):
        t = 5.0 * e.params.get("lag_min", 0.0)
        cur = by.get(e.downstream)
        while cur is not None:
            if cur.kind == "Reach":
                t += cur.params.get("travel_min") or 0.0
            cur = by.get(cur.downstream) if cur.downstream else None
        worst = max(worst, t)
    return float(max(1, math.ceil(worst / 60.0)))


# ---------------------------------------------------------------- project


@dataclass
class Control:
    name: str
    duration_min: float
    start: datetime
    end: datetime
    dt_min: float


@dataclass
class Run:
    name: str
    storm: object
    control: Control
    met_name: str
    gage_name: str | None
    dss_file: str


@dataclass
class Ensemble:
    """HEC-HMS ensemble analysis: the simulation runs of one storm method and
    return period (all durations and peak positions / patterns)."""

    name: str
    source: str
    ari: float
    runs: list[Run]


@dataclass
class ProjectResult:
    settings: Settings
    network: Network
    storms: list
    controls: list[Control]
    runs: list[Run]
    dt_min: float
    tail_h: float
    folder: str = ""
    files: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    crs_name: str = ""
    ensembles: list[Ensemble] = field(default_factory=list)
    arf: object = None  # rainfall._common.arf.ArfModel used for the storms


ENSEMBLE_METHOD_NAMES = {
    "ddf": "Frequency",
    "alternating_block": "Alternating Block",
    "huff": "Huff",
    "ensemble": "Storm ensemble",
}


def make_ensembles(runs: list[Run]) -> list[Ensemble]:
    """One ensemble analysis per storm method and return period, in the order
    the runs were made; methods / return periods with a single run get none
    (an ensemble of one run adds nothing)."""
    from .storms import ari_label

    groups: dict[tuple[str, float], list[Run]] = {}
    for run in runs:
        groups.setdefault((run.storm.source, float(run.storm.ari)), []).append(run)
    out = []
    for (src, ari), members in groups.items():
        if len(members) < 2:
            continue
        label = ENSEMBLE_METHOD_NAMES.get(src, src)
        out.append(Ensemble(hms_name(f"{label} {ari_label(ari)}"), src, ari, members))
    return out


def assemble(
    subs: list[SubIn],
    reaches: list[ReachIn],
    nodes: list[NodeIn],
    storms: list,
    s: Settings,
) -> ProjectResult:
    """Network, parameters, controls and runs (no files written)."""
    net = build_network(subs, reaches, nodes, s)
    warnings = list(net.warnings)
    by_sub = {x.name: x for x in subs}
    for e in net.of("Subbasin"):
        warnings += subbasin_parameters(e, by_sub[e.source], s)
    min_lag = min(e.params["lag_min"] for e in net.of("Subbasin"))
    gage_steps = [st.interval_min for st in storms if st.is_hyetograph]
    dt = float(s.dt_min) if s.dt_min else choose_dt(min_lag, gage_steps)
    if s.transform == TRANSFORM_SCS:
        short = [
            e.name
            for e in net.of("Subbasin")
            if e.params["lag_min"] * SCS_DT_LAG_RATIO < dt - 1e-9
        ]
        if short:
            warnings.append(
                f"The {dt:g} min time step is longer than 0.29 x lag for "
                f"{len(short)} sub-basin(s) ({', '.join(short[:8])}"
                + (" ..." if len(short) > 8 else "")
                + "); HEC-HMS will warn. Use a shorter time step."
            )
    by_reach = {r.name: r for r in reaches}
    for e in net.of("Reach"):
        warnings += reach_parameters(e, by_reach[e.source], s, dt)
    tail = float(s.tail_h) if s.tail_h else auto_tail_hours(net)

    if not storms:
        raise HmsProjectError("No design storms were made; check the rainfall inputs.")
    names = [st.name for st in storms]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise HmsProjectError(f"Storm names occur more than once: {dup}")
    from .storms import duration_label

    groups = {(st.source, float(st.ari)) for st in storms}
    shared = s.ensembles and len(groups) < len(storms)
    controls: dict[float, Control] = {}
    longest = max(float(st.duration_min) for st in storms)
    runs = []
    for st in storms:
        d = longest if shared else float(st.duration_min)
        if d not in controls:
            end = s.start + timedelta(minutes=d + tail * 60.0)
            label = "Design storms" if shared else f"{duration_label(d)} storm"
            controls[d] = Control(label, d, s.start, end, dt)
        runs.append(
            Run(
                name=hms_name(st.name),
                storm=st,
                control=controls[d],
                met_name=hms_name(st.name),
                gage_name=hms_name(st.name) if st.is_hyetograph else None,
                dss_file=file_stem(st.name) + ".dss",
            )
        )
    return ProjectResult(
        settings=s,
        network=net,
        storms=storms,
        controls=sorted(controls.values(), key=lambda c: c.duration_min),
        runs=runs,
        dt_min=dt,
        tail_h=tail,
        warnings=warnings,
        ensembles=make_ensembles(runs) if shared else [],
    )
