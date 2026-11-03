"""
LandXML 1.2 reader for the Import LandXML Surface tool (zero QGIS
dependency; standard library and NumPy only).

Reads, from a LandXML file exported by Civil 3D or another survey/design
package:

- Surfaces: the TIN (Definition Pnts / Faces; faces flagged i="1" are
  hidden and left out), the source breaklines and boundaries;
- CgPoints (survey / COGO points), with their point groups;
- Alignments: the horizontal geometry (CoordGeom lines, arcs and spirals)
  as densified polylines;
- PlanFeatures (Civil 3D feature lines) as 3D polylines;
- Parcels (Civil 3D sites export their parcels) as polygons;
- PipeNetworks: structures (points) and pipes (lines between structures).

LandXML stores coordinates as northing, easting (, elevation); every
coordinate here is returned as (x = easting, y = northing[, z]).

Corridors are not a LandXML element of their own: Civil 3D exports a
corridor's surfaces, alignments and feature lines, which are read above.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

ARC_STEP_DEG = 2.0  # arc densification: at most this angle per segment
ARC_STEP_LENGTH = 2.0  # ... and at most this length (file units)
SPIRAL_STEPS = 50  # points per spiral


@dataclass
class Surface:
    name: str
    desc: str
    xyz: np.ndarray  # (n, 3) x, y, z
    triangles: np.ndarray  # (m, 3) indices into xyz
    point_ids: list = field(default_factory=list)
    breaklines: list = field(default_factory=list)  # [(desc, (k, 3) array)]
    boundaries: list = field(default_factory=list)  # [(name, type, (k, 3))]
    hidden_faces: int = 0
    area_2d: float | None = None  # from the file (Definition area2DSurf)
    area_3d: float | None = None

    @property
    def elev_range(self) -> tuple[float, float]:
        return float(self.xyz[:, 2].min()), float(self.xyz[:, 2].max())

    @property
    def extent(self) -> tuple[float, float, float, float]:
        x, y = self.xyz[:, 0], self.xyz[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    def computed_area_2d(self) -> float:
        p = self.xyz[self.triangles]
        a, b, c = p[:, 0, :2], p[:, 1, :2], p[:, 2, :2]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
            c[:, 0] - a[:, 0]
        )
        return float(np.abs(cross).sum() / 2.0)

    def computed_area_3d(self) -> float:
        p = self.xyz[self.triangles]
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        return float(np.linalg.norm(cross, axis=1).sum() / 2.0)

    def outline_edges(self) -> np.ndarray:
        """(k, 2) vertex index pairs of the TIN's boundary edges (edges used
        by one triangle only): the outer boundary and any holes."""
        t = self.triangles
        edges = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
        uniq, counts = np.unique(edges, axis=0, return_counts=True)
        return uniq[counts == 1]


@dataclass
class CgPoint:
    name: str
    x: float
    y: float
    z: float | None
    code: str = ""
    desc: str = ""
    groups: list = field(default_factory=list)


@dataclass
class Polyline:
    """A named line (alignment, feature line, pipe, breakline) or polygon
    ring (parcel): coordinates (k, 2) or (k, 3) and plain attributes."""

    name: str
    coords: np.ndarray
    attributes: dict = field(default_factory=dict)


@dataclass
class Structure:
    name: str
    x: float
    y: float
    attributes: dict = field(default_factory=dict)


@dataclass
class LandXmlData:
    path: str
    version: str = ""
    linear_unit: str = ""
    imperial: bool = False
    epsg: int | None = None
    crs_wkt: str = ""
    crs_desc: str = ""
    application: str = ""
    project: str = ""
    surfaces: list = field(default_factory=list)
    points: list = field(default_factory=list)  # CgPoint
    alignments: list = field(default_factory=list)  # Polyline
    feature_lines: list = field(default_factory=list)  # Polyline (3D)
    parcels: list = field(default_factory=list)  # Polyline (closed ring)
    structures: list = field(default_factory=list)  # Structure
    pipes: list = field(default_factory=list)  # Polyline
    warnings: list = field(default_factory=list)

    def counts(self) -> dict:
        return {
            "Surfaces": len(self.surfaces),
            "Survey points": len(self.points),
            "Alignments": len(self.alignments),
            "Feature lines": len(self.feature_lines),
            "Parcels": len(self.parcels),
            "Pipe network structures": len(self.structures),
            "Pipe network pipes": len(self.pipes),
        }


# ------------------------------------------------------------------ helpers


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _iter(el, name):
    return (e for e in el.iter() if _local(e.tag) == name)


def _floats(text) -> list[float]:
    return [float(v) for v in (text or "").split()]


def _ne(text) -> tuple:
    """LandXML 'N E [Z]' -> (x, y[, z])."""
    v = _floats(text)
    if len(v) < 2:
        raise ValueError(f"coordinate with fewer than two values: {text!r}")
    return (v[1], v[0], v[2]) if len(v) >= 3 else (v[1], v[0])


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _list3d(text, width: int = 3) -> np.ndarray:
    """PntList3D ('N E Z ...', width 3) or PntList2D ('N E ...', width 2) ->
    (k, width) array of x, y[, z]."""
    v = np.array(_floats(text), dtype=float)
    if v.size % width:
        raise ValueError(f"point list length {v.size} is not a multiple of {width}")
    arr = v.reshape(-1, width)
    out = arr.copy()
    out[:, 0], out[:, 1] = arr[:, 1], arr[:, 0]
    return out


def _pnt_list(el):
    """The element's PntList3D (k, 3) or PntList2D (k, 2) as x, y[, z], or
    None."""
    for tag, width in (("PntList3D", 3), ("PntList2D", 2)):
        pl = _child(el, tag)
        if pl is not None and (pl.text or "").strip():
            return _list3d(pl.text, width)
    return None


# ------------------------------------------------------------- coord geom


def _arc_points(start, centre, end, clockwise: bool) -> list:
    sx, sy = start[0] - centre[0], start[1] - centre[1]
    ex, ey = end[0] - centre[0], end[1] - centre[1]
    r = math.hypot(sx, sy)
    a0, a1 = math.atan2(sy, sx), math.atan2(ey, ex)
    sweep = a1 - a0
    if clockwise:
        while sweep >= 0:
            sweep -= 2 * math.pi
    else:
        while sweep <= 0:
            sweep += 2 * math.pi
    if math.isclose(abs(sweep), 2 * math.pi) and math.hypot(sx - ex, sy - ey) > 1e-9:
        sweep = 0.0
    n = max(
        2,
        int(math.ceil(abs(math.degrees(sweep)) / ARC_STEP_DEG)),
        int(math.ceil(abs(sweep) * r / ARC_STEP_LENGTH)),
    )
    z0 = start[2] if len(start) > 2 else None
    z1 = end[2] if len(end) > 2 else None
    pts = []
    for i in range(n + 1):
        t = i / n
        a = a0 + sweep * t
        p = (centre[0] + r * math.cos(a), centre[1] + r * math.sin(a))
        if z0 is not None and z1 is not None:
            p = (*p, z0 + (z1 - z0) * t)
        pts.append(p)
    pts[0], pts[-1] = tuple(start), tuple(end)
    return pts


def _spiral_points(el, start, end) -> list:
    """Clothoid from start to end: curvature varies linearly from
    1/radiusStart to 1/radiusEnd (INF = straight) over the length; the
    integrated curve is fitted onto the exact start and end points."""
    length = _num(el.get("length"))
    r0 = _num(el.get("radiusStart"))
    r1 = _num(el.get("radiusEnd"))
    k0 = 0.0 if r0 in (None, 0.0) or math.isinf(r0) else 1.0 / r0
    k1 = 0.0 if r1 in (None, 0.0) or math.isinf(r1) else 1.0 / r1
    sign = -1.0 if (el.get("rot") or "cw").lower() == "cw" else 1.0
    if not length or length <= 0:
        return [tuple(start), tuple(end)]
    s = np.linspace(0.0, length, SPIRAL_STEPS + 1)
    theta = sign * (k0 * s + (k1 - k0) * s**2 / (2 * length))
    ds = length / SPIRAL_STEPS
    mid = (theta[:-1] + theta[1:]) / 2
    x = np.concatenate([[0.0], np.cumsum(np.cos(mid) * ds)])
    y = np.concatenate([[0.0], np.cumsum(np.sin(mid) * ds)])
    # similarity transform: computed chord -> actual chord
    cx, cy = x[-1], y[-1]
    dx, dy = end[0] - start[0], end[1] - start[1]
    chord = math.hypot(cx, cy)
    if chord < 1e-9:
        return [tuple(start), tuple(end)]
    scale = math.hypot(dx, dy) / chord
    rot = math.atan2(dy, dx) - math.atan2(cy, cx)
    c, sn = math.cos(rot) * scale, math.sin(rot) * scale
    px = start[0] + c * x - sn * y
    py = start[1] + sn * x + c * y
    pts = [(float(a), float(b)) for a, b in zip(px, py, strict=True)]
    if len(start) > 2 and len(end) > 2:
        pts = [
            (p[0], p[1], start[2] + (end[2] - start[2]) * i / SPIRAL_STEPS)
            for i, p in enumerate(pts)
        ]
    pts[0], pts[-1] = tuple(start), tuple(end)
    return pts


def coord_geom_points(cg, warnings: list | None = None, label: str = "") -> list:
    """Densified points of a CoordGeom (Line, Curve, Spiral, IrregularLine)
    in order, as (x, y[, z]) tuples."""
    pts: list = []

    def add(seq):
        for p in seq:
            if pts and math.dist(pts[-1][:2], p[:2]) < 1e-9:
                continue
            pts.append(tuple(p))

    for el in cg:
        kind = _local(el.tag)
        start, end = _child(el, "Start"), _child(el, "End")
        if kind == "Line" and start is not None and end is not None:
            add([_ne(start.text), _ne(end.text)])
        elif kind == "Curve" and start is not None and end is not None:
            centre = _child(el, "Center")
            if centre is None:
                add([_ne(start.text), _ne(end.text)])
                continue
            add(
                _arc_points(
                    _ne(start.text),
                    _ne(centre.text),
                    _ne(end.text),
                    (el.get("rot") or "ccw").lower() == "cw",
                )
            )
        elif kind == "Spiral" and start is not None and end is not None:
            add(_spiral_points(el, _ne(start.text), _ne(end.text)))
        elif kind == "IrregularLine":
            pl = _pnt_list(el)
            if pl is not None:
                add([tuple(p) for p in pl])
        elif kind in ("Chain",):
            if warnings is not None:
                warnings.append(f"{label}: a Chain element was skipped.")
    return pts


# ----------------------------------------------------------------- readers


def _read_surface(el, warnings) -> Surface | None:
    name = el.get("name") or "Surface"
    definition = _child(el, "Definition")
    if definition is None:
        warnings.append(f"Surface {name}: no Definition (TIN) - skipped.")
        return None
    if (definition.get("surfType") or "TIN").upper() != "TIN":
        warnings.append(
            f"Surface {name}: surface type {definition.get('surfType')} is not "
            "a TIN - skipped."
        )
        return None
    pnts, faces = _child(definition, "Pnts"), _child(definition, "Faces")
    if pnts is None or faces is None:
        warnings.append(f"Surface {name}: no points or faces - skipped.")
        return None
    ids, coords = [], []
    for p in _children(pnts, "P"):
        ids.append(p.get("id"))
        c = _ne(p.text)
        coords.append(c if len(c) == 3 else (*c, float("nan")))
    index = {k: i for i, k in enumerate(ids)}
    tris, hidden, bad = [], 0, 0
    for f in _children(faces, "F"):
        if f.get("i") == "1":
            hidden += 1
            continue
        refs = (f.text or "").split()
        try:
            tris.append([index[r] for r in refs[:3]])
        except KeyError:
            bad += 1
    if bad:
        warnings.append(
            f"Surface {name}: {bad} face(s) refer to missing points - skipped."
        )
    if not tris:
        warnings.append(f"Surface {name}: no visible faces - skipped.")
        return None
    surf = Surface(
        name=name,
        desc=el.get("desc") or "",
        xyz=np.array(coords, dtype=float),
        triangles=np.array(tris, dtype=np.int64),
        point_ids=ids,
        hidden_faces=hidden,
        area_2d=_num(definition.get("area2DSurf")),
        area_3d=_num(definition.get("area3DSurf")),
    )
    source = _child(el, "SourceData")
    if source is not None:
        for bl in _iter(source, "Breakline"):
            pl = _pnt_list(bl)
            if pl is not None and len(pl) >= 2:
                surf.breaklines.append((bl.get("desc") or bl.get("name") or "", pl))
        for bd in _iter(source, "Boundary"):
            pl = _pnt_list(bd)
            if pl is not None and len(pl) >= 3:
                surf.boundaries.append(
                    (bd.get("name") or "", bd.get("bndType") or "", pl)
                )
    return surf


def _read_points(root, warnings) -> list:
    """CgPoint elements with coordinates; point groups are the named
    CgPoints collections that contain a point or refer to it (pntRef)."""
    points: dict[str, CgPoint] = {}
    for cp in _iter(root, "CgPoint"):
        text = (cp.text or "").strip()
        if not text:
            continue
        try:
            c = _ne(text)
        except ValueError:
            continue
        key = cp.get("name") or cp.get("oID") or str(len(points) + 1)
        if key not in points:
            points[key] = CgPoint(
                name=key,
                x=c[0],
                y=c[1],
                z=c[2] if len(c) > 2 else None,
                code=cp.get("code") or "",
                desc=cp.get("desc") or "",
            )
    for group in _iter(root, "CgPoints"):
        gname = group.get("name") or ""
        if not gname:
            continue
        for cp in _children(group, "CgPoint"):
            key = cp.get("pntRef") or cp.get("name") or cp.get("oID")
            pt = points.get(key)
            if pt is not None and gname not in pt.groups:
                pt.groups.append(gname)
    return list(points.values())


def _read_alignments(root, warnings) -> list:
    out = []
    for al in _iter(root, "Alignment"):
        cg = _child(al, "CoordGeom")
        if cg is None:
            continue
        name = al.get("name") or f"Alignment {len(out) + 1}"
        pts = coord_geom_points(cg, warnings, f"Alignment {name}")
        if len(pts) < 2:
            continue
        coords = np.array([p[:2] for p in pts], dtype=float)
        out.append(
            Polyline(
                name,
                coords,
                {
                    "length": _num(al.get("length")),
                    "sta_start": _num(al.get("staStart")),
                    "desc": al.get("desc") or "",
                },
            )
        )
    return out


def _read_feature_lines(root, warnings) -> list:
    out = []
    for group in _iter(root, "PlanFeatures"):
        gname = group.get("name") or ""
        for pf in _children(group, "PlanFeature"):
            cg = _child(pf, "CoordGeom")
            if cg is None:
                continue
            name = pf.get("name") or f"Feature line {len(out) + 1}"
            pts = coord_geom_points(cg, warnings, f"Feature line {name}")
            if len(pts) < 2:
                continue
            width = 3 if all(len(p) > 2 for p in pts) else 2
            coords = np.array([p[:width] for p in pts], dtype=float)
            out.append(
                Polyline(name, coords, {"group": gname, "desc": pf.get("desc") or ""})
            )
    return out


def _read_parcels(root, warnings) -> list:
    out = []
    for parcels in _iter(root, "Parcels"):
        site = parcels.get("name") or ""
        for pc in _children(parcels, "Parcel"):
            cg = _child(pc, "CoordGeom")
            if cg is None:
                continue
            name = pc.get("name") or f"Parcel {len(out) + 1}"
            pts = coord_geom_points(cg, warnings, f"Parcel {name}")
            if len(pts) < 3:
                continue
            coords = np.array([p[:2] for p in pts], dtype=float)
            if math.dist(coords[0], coords[-1]) > 1e-6:
                coords = np.vstack([coords, coords[:1]])
            out.append(
                Polyline(
                    name,
                    coords,
                    {
                        "site": site,
                        "area": _num(pc.get("area")),
                        "type": pc.get("parcelType") or "",
                        "desc": pc.get("desc") or "",
                    },
                )
            )
    return out


def _read_pipe_networks(root, warnings) -> tuple[list, list]:
    structs, pipes = [], []
    for net in _iter(root, "PipeNetwork"):
        nname = net.get("name") or ""
        centres: dict[str, tuple] = {}
        inverts: dict[tuple, float] = {}
        s_el = _child(net, "Structs")
        for st in _children(s_el, "Struct") if s_el is not None else []:
            centre = _child(st, "Center")
            if centre is None or not centre.text:
                continue
            c = _ne(centre.text)
            sname = st.get("name") or f"S{len(structs) + 1}"
            centres[sname] = c
            for inv in _children(st, "Invert"):
                elev = _num(inv.get("elev"))
                if elev is not None and inv.get("refPipe"):
                    inverts[(sname, inv.get("refPipe"))] = elev
            structs.append(
                Structure(
                    sname,
                    c[0],
                    c[1],
                    {
                        "network": nname,
                        "rim_m": _num(st.get("elevRim")),
                        "sump_m": _num(st.get("elevSump")),
                        "desc": st.get("desc") or "",
                    },
                )
            )
        p_el = _child(net, "Pipes")
        for pp in _children(p_el, "Pipe") if p_el is not None else []:
            a, b = pp.get("refStart"), pp.get("refEnd")
            pname = pp.get("name") or f"P{len(pipes) + 1}"
            if a not in centres or b not in centres:
                warnings.append(
                    f"Pipe {pname}: start or end structure not found - skipped."
                )
                continue
            size = None
            shape = ""
            for kind in ("CircPipe", "RectPipe", "EggPipe", "ElliPipe", "Channel"):
                k = _child(pp, kind)
                if k is not None:
                    shape = kind
                    size = _num(k.get("diameter")) or _num(k.get("span"))
                    break
            pipes.append(
                Polyline(
                    pname,
                    np.array([centres[a][:2], centres[b][:2]], dtype=float),
                    {
                        "network": nname,
                        "from": a,
                        "to": b,
                        "shape": shape,
                        "size": size,
                        "length": _num(pp.get("length")),
                        "slope": _num(pp.get("slope")),
                        "inv_start": inverts.get((a, pname)),
                        "inv_end": inverts.get((b, pname)),
                        "desc": pp.get("desc") or "",
                    },
                )
            )
    return structs, pipes


def read_landxml(path: str) -> LandXmlData:
    """Reads every supported element of a LandXML file."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"Not a readable XML file: {e}") from e
    if _local(root.tag) != "LandXML":
        raise ValueError("Not a LandXML file (the root element is not LandXML).")
    data = LandXmlData(path=str(path), version=root.get("version") or "")
    units = _child(root, "Units")
    if units is not None and len(units):
        u = units[0]
        data.imperial = _local(u.tag) == "Imperial"
        data.linear_unit = u.get("linearUnit") or ""
    cs = _child(root, "CoordinateSystem")
    if cs is not None:
        try:
            data.epsg = int(cs.get("epsgCode")) if cs.get("epsgCode") else None
        except ValueError:
            data.epsg = None
        data.crs_wkt = cs.get("ogcWktCode") or ""
        data.crs_desc = cs.get("desc") or cs.get("horizontalCoordinateSystemName") or ""
    app = _child(root, "Application")
    if app is not None:
        data.application = " ".join(
            x for x in (app.get("name"), app.get("version")) if x
        )
    proj = _child(root, "Project")
    if proj is not None:
        data.project = proj.get("name") or ""
    w = data.warnings
    for s in _iter(root, "Surface"):
        surf = _read_surface(s, w)
        if surf is not None:
            data.surfaces.append(surf)
    data.points = _read_points(root, w)
    data.alignments = _read_alignments(root, w)
    data.feature_lines = _read_feature_lines(root, w)
    data.parcels = _read_parcels(root, w)
    data.structures, data.pipes = _read_pipe_networks(root, w)
    if data.imperial:
        w.append(
            f"The file uses imperial units ({data.linear_unit}); coordinates, "
            "elevations and the cell size are in those units."
        )
    return data
