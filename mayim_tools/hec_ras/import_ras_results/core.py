"""
Import RAS result data - reading HEC-RAS 2D plan results (no QGIS).

A computed plan HDF file (<project>.p##.hdf) holds a copy of the geometry
and the unsteady results. Confirmed 2026-11-13 against project 2625038
(HEC-RAS 7.0.2):

- ``Results/Unsteady/Output/Output Blocks/Base Output/Summary Output/2D Flow
  Areas/<area>/Maximum Water Surface`` (2, cells): max WSE and its time
  (days);
- ``.../Unsteady Time Series/2D Flow Areas/<area>/Water Surface`` (times,
  cells) and ``Face Velocity`` (times, faces, normal velocity, m/s);
- ``.../Unsteady Time Series/Time`` (days);
- ``.../Unsteady Time Series/2D Flow Areas/<area>/2D Hyd Conn/<name>/
  Structure Variables`` and ``.../Boundary Conditions/<name>``;
- geometry: ``Faces NormalUnitVector and Length``, ``Cells Face and
  Orientation Info/Values``, ``FacePoints Cell Info / Index Values``;
- root attribute ``Projection`` (WKT) and ``Results/Unsteady/Summary``.

Cell velocities are reconstructed at every output time from the normal
velocities on the cell's faces, by least squares weighted by face length
(the cell-centred reconstruction RAS Mapper's velocity maps rest on), so
the peak velocity and the peak depth x velocity of a cell come from the
same moments in the flood.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mayim_tools.hec_ras._common.h5io import open_h5
from mayim_tools.hec_ras.import_ras_2d_data.mesh_2d.core import (
    list_2d_flow_area_names,
    read_2d_flow_area_cells,
    read_2d_flow_area_settings,
)

TOOL_VERSION = "1.1.0"

BASE = "Results/Unsteady/Output/Output Blocks/Base Output"
SUMMARY = f"{BASE}/Summary Output/2D Flow Areas"
SERIES = f"{BASE}/Unsteady Time Series"
AREAS = "Geometry/2D Flow Areas"
TIME_CHUNK = 200  # output times per block in the velocity reconstruction
# what the separate-process HDF5 reader copies (h5py cannot load in QGIS)
RESULT_ROOTS = (
    "Geometry", "Plan Data", f"{BASE}/Summary Output",
    f"{BASE}/Unsteady Time Series", "Results/Unsteady/Summary", "Results/Summary",
)  # fmt: skip


class ResultsError(ValueError):
    """The file holds no 2D results that can be read."""


def _text(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return _text(value.reshape(-1)[0])
    return str(value).strip()


@dataclass
class AreaResults:
    name: str
    centers: np.ndarray  # (n, 2) real cells
    polygons: list  # [[(x, y), ...]] per real cell
    min_elev: np.ndarray  # (n,)
    max_wse: np.ndarray  # (n,)
    t_max_wse_h: np.ndarray  # (n,) hours from the start
    max_depth: np.ndarray  # (n,) max WSE - cell minimum elevation
    max_velocity: np.ndarray  # (n,) peak speed (m/s)
    t_max_velocity_h: np.ndarray
    max_dv: np.ndarray  # (n,) peak of depth x speed (m2/s)
    arrival_h: np.ndarray  # (n,) first time wetter than the threshold (NaN: never)
    duration_h: np.ndarray  # (n,) hours wetter than the threshold
    facepoints: np.ndarray  # (m, 2) facepoint coordinates
    cell_facepoints: list  # per real cell: facepoint indices in ring order
    facepoint_cells: list  # per facepoint: adjacent real cells
    profiles: dict = field(default_factory=dict)  # profile line index -> results
    face_cells: np.ndarray | None = None  # (faces, 2) cell on each side
    face_facepoints: np.ndarray | None = None  # (faces, 2) end facepoints
    face_min_elev: np.ndarray | None = None  # (faces,) lowest terrain on the face


@dataclass
class PlanResults:
    path: str
    plan_name: str
    short_id: str
    flow_title: str
    projection_wkt: str
    terrain_file: str
    times_h: np.ndarray
    areas: list = field(default_factory=list)
    structures: list = field(default_factory=list)  # [dict]
    boundaries: list = field(default_factory=list)  # [dict]
    summary: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def _attrs(obj) -> dict:
    try:
        return {k: obj.attrs[k] for k in obj.attrs}
    except Exception:
        return {}


def _plan_info(f) -> dict:
    out = {}
    for path in ("Plan Data/Plan Information",):
        if path in f:
            out.update(_attrs(f[path]))
    return out


def reconstruct_cell_velocity(
    face_velocity: np.ndarray,
    normals: np.ndarray,
    lengths: np.ndarray,
    cell_faces: list,
) -> np.ndarray:
    """(times, cells, 2) cell velocity vectors from face-normal velocities:
    for each cell, V minimises sum_f L_f (n_f . V - u_f)^2 over its faces.
    ``face_velocity`` is (times, faces); ``cell_faces`` lists each cell's
    face indices. Cells with faces in only one direction get the velocity
    along that direction."""
    n_cells = len(cell_faces)
    pair_cell = np.concatenate(
        [np.full(len(fs), i) for i, fs in enumerate(cell_faces)]
    ).astype(np.int64)
    pair_face = np.concatenate([np.asarray(fs, np.int64) for fs in cell_faces])
    w = lengths[pair_face]
    nx, ny = normals[pair_face, 0], normals[pair_face, 1]
    a = np.zeros((n_cells, 2, 2))
    np.add.at(a[:, 0, 0], pair_cell, w * nx * nx)
    np.add.at(a[:, 0, 1], pair_cell, w * nx * ny)
    np.add.at(a[:, 1, 1], pair_cell, w * ny * ny)
    a[:, 1, 0] = a[:, 0, 1]
    # regularise near-singular cells (all faces parallel) slightly
    trace = a[:, 0, 0] + a[:, 1, 1]
    a[:, 0, 0] += 1e-9 * trace + 1e-12
    a[:, 1, 1] += 1e-9 * trace + 1e-12
    inv = np.linalg.inv(a)
    times = face_velocity.shape[0]
    out = np.empty((times, n_cells, 2))
    for t0 in range(0, times, TIME_CHUNK):
        u = face_velocity[t0 : t0 + TIME_CHUNK][:, pair_face] * w  # (t, pairs)
        b = np.zeros((u.shape[0], n_cells, 2))
        for k, comp in enumerate((nx, ny)):
            np.add.at(b[:, :, k], (slice(None), pair_cell), u * comp)
        out[t0 : t0 + TIME_CHUNK] = np.einsum("cij,tcj->tci", inv, b)
    return out


def _opt(geo, name, dtype):
    return geo[name][()].astype(dtype) if name in geo else None


def _cell_faces(info, values, n_cells) -> list:
    return [values[s : s + c, 0].astype(np.int64) for s, c in info[:n_cells]]


def _read_area(
    f, name: str, path: str, n_real: int | None, times_h, threshold,
    lines=None,
):  # fmt: skip
    geo = f[f"{AREAS}/{name}"]
    sm = f"{SUMMARY}/{name}"
    ts = f"{SERIES}/2D Flow Areas/{name}"
    if f"{sm}/Maximum Water Surface" not in f:
        raise ResultsError(f"No results for 2D flow area '{name}'.")
    max_ws = f[f"{sm}/Maximum Water Surface"][()]
    n_all = max_ws.shape[1]
    n = n_real or n_all
    cells = read_2d_flow_area_cells(path, name)  # real cells only
    n = min(n, len(cells))
    min_elev = geo["Cells Minimum Elevation"][()][:n].astype(float)
    max_wse = max_ws[0, :n].astype(float)
    t_max = max_ws[1, :n].astype(float) * 24.0
    max_depth = np.maximum(max_wse - min_elev, 0.0)
    max_depth[~np.isfinite(min_elev)] = np.nan

    # time series: velocity, depth x velocity, arrival, duration
    profiles = {}
    wse_all = f[f"{ts}/Water Surface"][()].astype(float)
    wse_t = wse_all[:, :n]
    depth_t = np.maximum(wse_t - min_elev[None, :], 0.0)
    fv_path = f"{ts}/Face Velocity"
    if fv_path in f:
        normals = geo["Faces NormalUnitVector and Length"][()]
        cell_faces = _cell_faces(
            geo["Cells Face and Orientation Info"][()],
            geo["Cells Face and Orientation Values"][()],
            n,
        )
        vel = reconstruct_cell_velocity(
            f[fv_path][()].astype(float), normals[:, :2], normals[:, 2], cell_faces
        )
        speed = np.hypot(vel[..., 0], vel[..., 1])
        speed[depth_t <= threshold] = 0.0  # dry cells carry no flow
        k = np.argmax(speed, axis=0)
        max_v = speed[k, np.arange(n)]
        t_v = times_h[k]
        max_dv = np.max(depth_t * speed, axis=0)
    else:
        max_v = np.full(n, np.nan)
        t_v = np.full(n, np.nan)
        max_dv = np.full(n, np.nan)
    wet = depth_t > threshold
    any_wet = wet.any(axis=0)
    arrival = np.where(any_wet, times_h[np.argmax(wet, axis=0)], np.nan)
    if len(times_h) > 1:
        dt = np.gradient(times_h)
        duration = (wet * dt[:, None]).sum(axis=0)
    else:
        duration = np.zeros(n)

    if lines:
        face_cells = geo["Faces Cell Indexes"][()].astype(np.int64)
        times_s = times_h * 3600.0
        vol = cell_volumes(geo, wse_all, n)
        bc_flows, resid = boundary_cell_flows(
            f, ts, face_cells, n, times_s, vol.sum(axis=1)
        )
        profiles = profile_results(
            lines, geo, n, vol, max_wse, min_elev,
            max_v if fv_path in f else None, times_h, threshold, bc_flows, resid,
        )  # fmt: skip

    fp_info = geo["FacePoints Cell Info"][()]
    fp_vals = geo["FacePoints Cell Index Values"][()]
    fp_cells = [
        [int(c) for c in fp_vals[s : s + cnt] if 0 <= c < n] for s, cnt in fp_info
    ]
    cfp = geo["Cells FacePoint Indexes"][()][:n]
    cell_fps = [[int(i) for i in row if i >= 0] for row in cfp]
    return AreaResults(
        name=name,
        centers=np.array([c.center for c in cells[:n]], float),
        polygons=[c.polygon for c in cells[:n]],
        min_elev=min_elev,
        max_wse=max_wse,
        t_max_wse_h=t_max,
        max_depth=max_depth,
        max_velocity=max_v,
        t_max_velocity_h=t_v,
        max_dv=max_dv,
        arrival_h=arrival,
        duration_h=duration,
        facepoints=geo["FacePoints Coordinate"][()].astype(float),
        cell_facepoints=cell_fps,
        facepoint_cells=fp_cells,
        face_cells=_opt(geo, "Faces Cell Indexes", np.int64),
        face_facepoints=_opt(geo, "Faces FacePoint Indexes", np.int64),
        face_min_elev=_opt(geo, "Faces Minimum Elevation", float),
        profiles=profiles,
    )


def _series_peaks(dataset, times_h, columns) -> list[tuple[str, float, float]]:
    data = dataset[()]
    if data.ndim == 1:
        data = data[:, None]
    out = []
    for j, col in enumerate(columns[: data.shape[1]]):
        v = data[:, j].astype(float)
        if not np.isfinite(v).any():
            continue
        k = int(np.nanargmax(np.abs(v)))
        out.append((col, float(v[k]), float(times_h[k])))
    return out


def _structures(f, area: str, times_h) -> list[dict]:
    base = f"{SERIES}/2D Flow Areas/{area}/2D Hyd Conn"
    out = []
    if base not in f:
        return out
    group = f[base]
    for name in group.keys():
        path = f"{base}/{name}/Structure Variables"
        if path not in f:
            continue
        ds = f[path]
        cols = [_text(r[0]) for r in _attrs(ds).get("Variable_Unit", [])]
        units = [_text(r[1]) for r in _attrs(ds).get("Variable_Unit", [])]
        for (col, peak, t), unit in zip(
            _series_peaks(ds, times_h, cols), units, strict=False
        ):
            out.append(
                {"kind": "SA/2D connection", "name": name, "area": area,
                 "variable": col, "unit": unit, "peak": peak, "time_h": t}
            )  # fmt: skip
    return out


def _boundaries(f, area: str, times_h) -> list[dict]:
    base = f"{SERIES}/2D Flow Areas/{area}/Boundary Conditions"
    out = []
    if base not in f:
        return out
    for name in f[base].keys():
        if " - " in name:  # per face / per cell detail
            continue
        ds = f[f"{base}/{name}"]
        attrs = _attrs(ds)
        cols = [_text(c) for c in attrs.get("Columns", [])] or ["Value"]
        for col, peak, t in _series_peaks(ds, times_h, cols):
            unit = _text(attrs.get(col, b""))
            out.append(
                {"kind": "Boundary condition", "name": name, "area": area,
                 "variable": col, "unit": unit, "peak": peak, "time_h": t}
            )  # fmt: skip
    return out


def read_plan_results(
    path: str,
    depth_threshold: float = 0.001,
    profile_lines: list | None = None,
) -> PlanResults:
    """Everything the tool writes, from one computed plan HDF file, with
    peak flow and stage across ``profile_lines`` (``ProfileLine``)."""
    names = list_2d_flow_area_names(path)
    settings = read_2d_flow_area_settings(path)
    with open_h5(path, roots=RESULT_ROOTS) as f:
        if f"{SERIES}/Time" not in f:
            raise ResultsError(
                "This file holds no unsteady 2D results. Choose a computed plan "
                "file (<project>.p##.hdf)."
            )
        times_h = f[f"{SERIES}/Time"][()].astype(float) * 24.0 + 0.0  # no -0.0
        info = _plan_info(f)
        root = _attrs(f)
        geo_attrs = _attrs(f["Geometry"]) if "Geometry" in f else {}
        plan = PlanResults(
            path=path,
            plan_name=_text(info.get("Plan Name", b"")),
            short_id=_text(info.get("Plan ShortID", b"")),
            flow_title=_text(info.get("Flow Title", b"")),
            projection_wkt=_text(root.get("Projection", b"")),
            terrain_file=_text(geo_attrs.get("Terrain Filename", b"")),
            times_h=times_h,
        )
        for name in names:
            n_real = settings.get(name, {}).get("cell_count")
            try:
                plan.areas.append(
                    _read_area(
                        f,
                        name,
                        path,
                        n_real,
                        times_h,
                        depth_threshold,
                        profile_lines,
                    )  # fmt: skip
                )
            except ResultsError as e:
                plan.warnings.append(str(e))
            plan.structures += _structures(f, name, times_h)
            plan.boundaries += _boundaries(f, name, times_h)
        if "Results/Unsteady/Summary" in f:
            s = _attrs(f["Results/Unsteady/Summary"])
            plan.summary = {k: _text(v) for k, v in s.items()}
        for name in names:
            comp = f"{SERIES}/2D Flow Areas/{name}/Computations"
            for key in ("Volume Error", "Volume", "Max Courant"):
                if f"{comp}/{key}" in f:
                    v = f[f"{comp}/{key}"][()].astype(float).reshape(-1)
                    if key == "Max Courant":
                        plan.summary[f"{name}: max Courant"] = f"{np.nanmax(v):.3g}"
                    else:
                        plan.summary[f"{name}: {key.lower()} at the end (m3)"] = (
                            f"{v[-1]:.4g}"
                        )
        msg = "Results/Summary/Compute Messages (text)"
        if msg in f:
            plan.summary["Compute messages"] = _text(f[msg][()])
    if not plan.areas:
        raise ResultsError("No 2D flow area results were found in this file.")
    return plan


# --------------------------------------------------------------------------
# Profile lines: peak flow and stage across lines drawn on the map
# --------------------------------------------------------------------------
#
# The flow across a line comes from the mass balance of the mesh upstream of
# it: Q(t) = inflow into that part - d(volume stored in it)/dt. The line
# splits the wet cells in two along the faces it cuts (a face is cut when the
# segment joining its two cell centres crosses the line); the volume comes
# from each cell's volume-elevation table at its water level, and the inflow
# from the boundary condition series of the cells upstream. HEC-RAS
# conserves mass to its reported volume error, so this matches its own
# boundary flows (checked on project 2625038: peak at every line within 0.4%
# of the inflow peak, attenuated and later downstream). Summing face
# velocity x face area instead was off by a factor of 0.8-2, because the
# face's water level is not stored. Water moved through a structure
# (SA/2D connection) is counted when it passes the line.


@dataclass
class ProfileLine:
    """A line to report peak flow and stage across. Draw it from the left
    bank to the right bank looking downstream (RAS Mapper's convention):
    positive flow then runs downstream. It must cross the whole flooded
    width."""

    name: str
    coords: list  # [(x, y), ...]


def face_crossings(coords, centers, face_cells, n_real):
    """(faces, sign) of the faces whose centre-to-centre segment crosses the
    polyline; sign +1 where cell 0 lies upstream (on the left looking
    downstream), so that a positive face velocity is downstream flow."""
    pts = np.asarray(coords, float)
    c0, c1 = face_cells[:, 0], face_cells[:, 1]
    real = (c0 >= 0) & (c1 >= 0) & (c0 < n_real) & (c1 < n_real)
    idx = np.flatnonzero(real)
    a = centers[c0[idx]]
    b = centers[c1[idx]]
    found = np.zeros(len(idx), bool)
    sign = np.zeros(len(idx))
    for p, q in zip(pts[:-1], pts[1:], strict=True):
        d = q - p
        if not np.any(d):
            continue

        def side(x, p=p, d=d):
            return d[0] * (x[:, 1] - p[1]) - d[1] * (x[:, 0] - p[0])

        sa, sb = side(a), side(b)  # >0: left of the line's direction
        e = b - a
        ca = e[:, 0] * (p[1] - a[:, 1]) - e[:, 1] * (p[0] - a[:, 0])
        cb = e[:, 0] * (q[1] - a[:, 1]) - e[:, 1] * (q[0] - a[:, 0])
        hit = (sa * sb < 0) & (ca * cb <= 0) & ~found
        # the normal (-dy, dx) points left of the drawn direction; drawn left
        # bank -> right bank looking downstream, left of it is downstream
        sign[hit] = np.where(sa[hit] < 0, 1.0, -1.0)
        found |= hit
    return idx[found], sign[found]


def _components(n, a, b) -> np.ndarray:
    """Connected-component label per node of the graph with edges a-b."""
    try:
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components

        g = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
        return connected_components(g, directed=False)[1]
    except ImportError:  # pragma: no cover - QGIS ships SciPy
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x, y in zip(a.tolist(), b.tolist(), strict=True):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        return np.array([find(x) for x in range(n)])


def cell_volumes(geo, wse_t, n) -> np.ndarray:
    """(times, n) water volume in each real cell (volume-elevation tables,
    extended above the top with the cell's plan area)."""
    info = geo["Cells Volume Elevation Info"][()]
    vals = geo["Cells Volume Elevation Values"][()].astype(float)
    area = geo["Cells Surface Area"][()].astype(float)
    out = np.zeros((wse_t.shape[0], n))
    for c in range(n):
        s0, cnt = info[c]
        if cnt < 1:
            continue
        z, v = vals[s0 : s0 + cnt, 0], vals[s0 : s0 + cnt, 1]
        lv = wse_t[:, c]
        vol = np.interp(lv, z, v, left=0.0)
        above = lv > z[-1]
        vol[above] += (lv[above] - z[-1]) * area[c]
        out[:, c] = vol
    return out


def boundary_cell_flows(f, ts: str, face_cells, n, times_s, total_volume):
    """[(cells, series)] of the boundary flows into the area, signed into
    the area, and the residual of the whole area's mass balance relative to
    the peak flow. The sign of each boundary series is chosen so that the
    whole area balances (HEC-RAS reports inflows and outflows positive)."""
    base = f"{ts}/Boundary Conditions"
    items = []
    if base in f:
        for name in f[base].keys():
            ds = f[f"{base}/{name}"]
            attrs = _attrs(ds)
            if name.endswith(" - Flow per Cell") and "Cells" in attrs:
                cells = np.asarray(attrs["Cells"], np.int64)
            elif name.endswith(" - Flow per Face") and "Faces" in attrs:
                fc = face_cells[np.asarray(attrs["Faces"], np.int64)]
                cells = np.where((fc[:, 0] >= 0) & (fc[:, 0] < n), fc[:, 0], fc[:, 1])
            else:
                continue
            data = ds[()].astype(float)
            if data.ndim == 1:
                data = data[:, None]
            items.append([cells[: data.shape[1]], data[:, : len(cells)]])
    dvdt = np.gradient(total_volume, times_s) if len(times_s) > 1 else total_volume * 0
    peak = max([np.abs(d.sum(axis=1)).max() for _, d in items] + [1e-9])
    best, best_res = None, np.inf
    m = min(len(items), 10)
    for combo in range(2**m):
        signs = [(-1.0 if combo >> i & 1 else 1.0) for i in range(m)]
        net = np.zeros(len(times_s))
        for sg, (_, d) in zip(signs, items, strict=False):
            net += sg * d.sum(axis=1)
        res = float(np.abs(net - dvdt).max())
        if res < best_res:
            best, best_res = signs, res
    out = []
    for i, (cells, data) in enumerate(items):
        sg = best[i] if best and i < len(best) else 1.0
        out.append((cells, sg * data))
    return out, best_res / peak


def profile_results(
    lines, geo, n, vol, max_wse, min_elev, speed_max, times_h, threshold,
    bc_flows, balance_residual,
):  # fmt: skip
    """Per line index: peak flow, its time, volume passed and the stage."""
    centers = geo["Cells Center Coordinate"][()].astype(float)
    face_cells = geo["Faces Cell Indexes"][()].astype(np.int64)
    times_s = times_h * 3600.0
    ever_wet = np.nan_to_num(max_wse - min_elev) > threshold
    res = {}
    c0, c1 = face_cells[:, 0], face_cells[:, 1]
    real = (c0 >= 0) & (c1 >= 0) & (c0 < n) & (c1 < n)
    wet_edge = np.zeros(len(face_cells), bool)
    wet_edge[real] = ever_wet[c0[real]] & ever_wet[c1[real]]
    for i, line in enumerate(lines):
        faces, sign = face_crossings(line.coords, centers, face_cells, n)
        wet = wet_edge[faces]
        faces, sign = faces[wet], sign[wet]
        if not len(faces):
            continue
        cut = np.zeros(len(face_cells), bool)
        cut[faces] = True
        keep = wet_edge & ~cut
        label = _components(n, c0[keep], c1[keep])
        up_cells = np.where(sign > 0, c0[faces], c1[faces])
        dn_cells = np.where(sign > 0, c1[faces], c0[faces])
        up_lab, dn_lab = set(label[up_cells]), set(label[dn_cells])
        cells = np.unique(np.r_[up_cells, dn_cells])
        depth = max_wse[cells] - min_elev[cells]
        r = {
            "wse_max": float(max_wse[cells].max()),
            "wse_mean": float(max_wse[cells].mean()),
            "depth_max": float(depth.max()),
            "v_max": float(speed_max[cells].max()) if speed_max is not None else np.nan,
            "faces": int(len(faces)),
            "q_peak": np.nan,
            "t_q_peak_h": np.nan,
            "volume_m3": np.nan,
            "note": "",
        }
        if up_lab & dn_lab:
            r["note"] = (
                "The line does not cross the whole flooded width, so water can "
                "pass around its ends; extend it to dry ground on both sides."
            )
        else:
            up = np.isin(label, list(up_lab)) & ever_wet
            inflow = np.zeros(len(times_h))
            for bcells, data in bc_flows:
                sel = up[np.clip(bcells, 0, n - 1)] & (bcells < n)
                if sel.any():
                    inflow += data[:, sel].sum(axis=1)
            v_up = vol[:, up].sum(axis=1)
            q = inflow - (np.gradient(v_up, times_s) if len(times_s) > 1 else 0.0)
            k = int(np.argmax(np.abs(q)))
            r["q_peak"] = float(q[k])
            r["t_q_peak_h"] = float(times_h[k])
            r["volume_m3"] = float(np.trapezoid(q, times_s)) if hasattr(
                np, "trapezoid") else float(np.trapz(q, times_s))  # fmt: skip
            r["series"] = q
            if balance_residual > 0.05:
                r["note"] = (
                    f"The 2D area's own mass balance is out by "
                    f"{balance_residual:.0%} of the peak flow (rain on the mesh, "
                    "infiltration or flow to another area are not read), so "
                    "this flow is approximate."
                )
        res[i] = r
    return res


# --------------------------------------------------------------------------
# Files next to the plan: the terrain and RAS Mapper's profile lines
# --------------------------------------------------------------------------


def _project_file(plan_path: str, ext: str):
    """<project>.<ext> next to a <project>.p##.hdf plan file."""
    from pathlib import Path

    p = Path(plan_path)
    stem = p.name.split(".")[0]
    cand = p.with_name(f"{stem}{ext}")
    return cand if cand.exists() else None


def _rel(plan_path: str, filename: str):
    from pathlib import Path

    filename = filename.replace("\\", "/")
    if filename.startswith("./"):
        filename = filename[2:]
    return Path(plan_path).parent / filename


def find_terrain(plan: PlanResults):
    """The terrain raster the plan's geometry uses: the geometry's
    ``Terrain Filename`` (a RAS terrain .hdf, next to which RAS Mapper keeps
    a .vrt of its GeoTIFFs), or None."""
    if not plan.terrain_file:
        return None
    hdf = _rel(plan.path, plan.terrain_file)
    for cand in (hdf.with_suffix(".vrt"), hdf.with_suffix(".tif"), hdf):
        if cand.exists() and cand.suffix.lower() != ".hdf":
            return str(cand)
    return None


def rasmap_profile_lines(plan_path: str):
    """Path of RAS Mapper's 'Profile Lines' shapefile from <project>.rasmap,
    when it exists."""
    import re

    rasmap = _project_file(plan_path, ".rasmap")
    if rasmap is None:
        return None
    try:
        text = rasmap.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r'<Layer Name="Profile Lines"[^>]*Filename="([^"]+)"', text)
    if not m:
        return None
    path = _rel(plan_path, m.group(1))
    return str(path) if path.exists() else None
