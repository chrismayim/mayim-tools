"""
Network sub-catchments (Catchment Delineation, incremental mode) - pure
NumPy, no QGIS dependency.

The pour point(s) / pour line(s) mark only the final outlet of each system.
The stream network upstream of them (Stream Network core, with the optional
maximum stream length and its stream breakpoints) defines the sub-catchments:

- one STREAM sub-catchment per reach (the stretch of stream between two
  nodes): every cell whose flow first meets the stream on that reach. It
  drains to the node at the reach's lower end and is named after the
  reach (A1, A2, A3A1 ...), which is also the name of that node;
- one CHANNEL HEAD sub-catchment per channel head: the area that drains to
  the channel head itself (about the stream threshold area). It is named
  after the channel head node, the next number up the stem (A6 above A5);
- optionally, each stream sub-catchment is split along the stream
  centreline into its left bank (a) and right bank (b), looking downstream
  (A1a, A1b). The sub-catchment is delineated whole first, then cut by its
  stream centreline (through the stream cell centres, extended at both ends
  to the boundary): the cells on each side of the line form the pieces, and
  each piece takes the side of the line it lies on. Cells on the line go with
  the left bank. Pieces smaller than the split tolerance (a minimum area and
  a minimum mean width, in cells) join the neighbouring piece they share the
  longest boundary with; pieces on the same side stay one (multi-part)
  sub-catchment; if a whole side fails the tolerance the stream is not
  split. Each bank keeps its own centroid and longest flow path, which runs
  down the stream to the node. Channel head sub-catchments are not split.
  The QGIS wrapper cuts the sub-catchment polygon along the same line, so
  the boundary between the banks follows the centreline exactly.

Names follow the Stream Network core (name_network): each system is a stem
A, B, C ... by area, numbered upstream from its outlet; branches take the
name of the reach they flow into plus a letter (A3A1 ...).

Each sub-catchment records the node it drains to (``node``) and the stream
that node discharges into (``ds_stream``, empty at the outlet), so the
sub-catchments can be linked into a hydrological model (sub-basin ->
junction ``node`` -> reach ``ds_stream``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

import numpy as np

from mayim_tools.hydrology._common.d8_network import (
    build_downstream_index,
    cell_center,
    check_geotransform,
    flow_accumulation_cells,
    nodata_mask,
    pointer_to_direction_index,
    resolve_terminals,
    step_lengths,
)

from .core import (
    CatchmentError,
    DelineationResult,
    _Grid,
    catchment_dem,
    catchment_from_cells,
    fill_holes,
    group_polygons,
    ring_to_map,
    simplify_ring,
)

DEFAULT_MIN_SPLIT_CELLS = 10
DEFAULT_MIN_SPLIT_WIDTH_CELLS = 2.0
SUB_STREAM = "stream"
SUB_HEAD = "channel head"
BANK_LEFT, BANK_RIGHT = "a", "b"


def natural_key(name: str):
    """Sort key that orders A2 before A10, A3A1 before A4, and a stream's
    banks (A1a, A1b) straight after it, before its branches (A1A1)."""
    bank = ""
    if name and name[-1] in (BANK_LEFT, BANK_RIGHT):
        name, bank = name[:-1], name[-1]
    parts = [
        (0, int(t)) if t.isdigit() else (1, t) for t in re.findall(r"\d+|\D+", name)
    ]
    return parts, bank


def build_grid(
    pointer, pointer_nodata, dem, dem_nodata, geotransform, esri_style,
    stream_threshold_km2, warn,
):  # fmt: skip
    sx, sy = check_geotransform(geotransform)
    pointer = np.asarray(pointer)
    dem = np.asarray(dem, dtype=np.float64)
    if pointer.shape != dem.shape:
        raise CatchmentError("D8 pointer and DEM are not on the same grid.")
    rows, cols = pointer.shape
    direction_index, _ = pointer_to_direction_index(pointer, pointer_nodata, esri_style)
    valid = (direction_index != -2).ravel()
    dem_flat = np.where(nodata_mask(dem, dem_nodata), np.nan, dem).ravel()
    nxt = build_downstream_index(direction_index)
    steps = step_lengths(direction_index, sx, sy)
    steps[nxt == np.arange(nxt.size)] = 0.0
    acc, _ = flow_accumulation_cells(nxt, valid)
    grid = _Grid(
        geotransform=geotransform,
        rows=rows,
        cols=cols,
        sx=sx,
        sy=sy,
        nxt=nxt,
        steps=steps,
        acc=acc,
        dem_flat=dem_flat,
        stream_threshold_km2=stream_threshold_km2,
        warn=warn,
    )
    return grid, valid


def delineate_network_subcatchments(
    pointer: np.ndarray,
    pointer_nodata,
    dem: np.ndarray,
    dem_nodata,
    geotransform,
    network,
    outlet_cells: Sequence[tuple[str, Sequence[int]]],
    esri_style: bool = False,
    stream_threshold_km2: float = 0.1,
    split_banks: bool = False,
    min_split_cells: int = DEFAULT_MIN_SPLIT_CELLS,
    min_split_width_cells: float = DEFAULT_MIN_SPLIT_WIDTH_CELLS,
    progress: Callable[[float, str], None] | None = None,
    is_canceled: Callable[[], bool] | None = None,
) -> DelineationResult:
    """Sub-catchments of the stream network ``network`` (a Stream Network
    NetworkResult extracted upstream of ``outlet_cells``, with named
    reaches). See the module docstring."""
    result = DelineationResult()
    warn = result.warnings.append

    def report(f, msg):
        if progress is not None:
            progress(f, msg)

    def canceled():
        return is_canceled is not None and is_canceled()

    report(0.02, "Preparing the flow grid")
    grid, valid = build_grid(
        pointer, pointer_nodata, dem, dem_nodata, geotransform, esri_style,
        stream_threshold_km2, warn,
    )  # fmt: skip
    nxt, cols = grid.nxt, grid.cols
    n = nxt.size
    idx = np.arange(n, dtype=np.int64)
    reaches = [r for r in network.reaches if r.get("name")]
    if not reaches:
        raise CatchmentError("The stream network has no reaches to divide.")
    by_id = {r["link_id"]: r for r in reaches}
    kids: dict[int, list[dict]] = {r["link_id"]: [] for r in reaches}
    for r in reaches:
        if r["ds_link"] in kids:
            kids[r["ds_link"]].append(r)

    # --- seeds: stream cells by sub-catchment --------------------------------
    report(0.15, "Assigning cells to the stream network")
    subs: list[dict] = []  # index i -> label i + 1
    seed = np.zeros(n, dtype=np.int32)
    stream_of: dict[int, int] = {}  # link_id -> label of its stream catchment
    for r in reaches:
        cells = np.asarray(r["_cells"], dtype=np.int64)
        own = cells if r["ds_link"] is None else cells[:-1]
        head_name = r.get("head_name")
        if head_name and own.size == 1:
            warn(
                f"Stream {r['name']} is a single step below its channel head, so "
                f"the channel head sub-catchment {head_name} is part of the "
                f"{r['name']} sub-catchment."
            )
        elif head_name:
            subs.append(
                {
                    "name": head_name,
                    "sub_type": SUB_HEAD,
                    "reach": r,
                    "seed": own[:1],
                }
            )
            seed[own[0]] = len(subs)
            own = own[1:]
        subs.append({"name": r["name"], "sub_type": SUB_STREAM, "reach": r})
        subs[-1]["seed"] = own[-1:]
        seed[own] = len(subs)
        stream_of[r["link_id"]] = len(subs)

    # --- label every cell by the first stream cell it meets -------------------
    report(0.30, "Labelling sub-catchments")
    is_outlet = np.zeros(n, dtype=bool)
    for _name, cells in outlet_cells or ():
        cells = np.asarray(cells, dtype=np.int64)
        is_outlet[cells[(cells >= 0) & (cells < n)]] = True
    stops = (seed > 0) | is_outlet
    nxt_s = nxt.copy()
    nxt_s[stops] = idx[stops]
    terminal, _, unresolved = resolve_terminals(nxt_s)
    label = seed[terminal]
    ok = valid & ~unresolved
    label[~ok] = 0

    # cells that reach a pour line away from the stream go to the nearest
    # system outlet reach
    orphan = ok & (label == 0) & is_outlet[terminal]
    if orphan.any():
        roots = [
            (int(r["_end_cell"]), stream_of[r["link_id"]])
            for r in reaches
            if r["ds_link"] is None and r["link_id"] in stream_of
        ]
        if roots:
            ends = np.array([e for e, _ in roots])
            ex, ey = ends % cols, ends // cols
            for t in np.unique(terminal[orphan]):
                d2 = (ex - t % cols) ** 2 + (ey - t // cols) ** 2
                label[orphan & (terminal == t)] = roots[int(np.argmin(d2))][1]
    if canceled():
        return result

    # --- one catchment per sub-catchment (and bank) ----------------------------
    labelled = np.flatnonzero(label)
    order = labelled[np.argsort(label[labelled], kind="stable")]
    bounds = np.searchsorted(label[order], np.arange(1, len(subs) + 2))
    any_sub = label > 0
    pieces = []
    n_split = n_unsplit = 0
    if split_banks:
        report(0.40, "Splitting along the stream centreline")
    for i, sub in enumerate(subs, start=1):
        cells = order[bounds[i - 1] : bounds[i]]
        if cells.size == 0:
            continue
        split = None
        if split_banks and sub["sub_type"] == SUB_STREAM:
            line = centreline_cells(sub["reach"], kids)
            split = split_along_centreline(
                grid, cells, line, any_sub, min_split_cells, min_split_width_cells
            )
            if split is None:
                n_unsplit += 1
            else:
                n_split += 1
        if split is not None:
            for letter, part in split["banks"]:
                pieces.append(
                    (sub, f"{sub['name']}{letter}", letter, part, cells, split)
                )
        else:
            pieces.append((sub, sub["name"], None, cells, None, None))
    if split_banks and n_unsplit:
        warn(
            f"{n_unsplit} of {n_split + n_unsplit} stream sub-catchment(s) were not "
            "split: one bank was below the split tolerance "
            f"({min_split_cells} cells, mean width {min_split_width_cells:g} cells)."
        )
    pieces.sort(key=lambda p: natural_key(p[1]))

    windows = []
    for k, (sub, name, letter, cells, route, split) in enumerate(pieces):
        if canceled():
            return result
        report(0.45 + 0.5 * k / max(len(pieces), 1), f"Sub-catchment {name}")
        r = sub["reach"]
        if sub["sub_type"] == SUB_HEAD:
            node, stream, ds_stream = sub["name"], None, r["name"]
        else:
            node, stream, ds_stream = node_below(r, by_id), r["name"], r.get("ds_name")
        extra = {
            "sub_type": sub["sub_type"],
            "system": _system_letters(r.get("stem")),
            "stream": stream,
            "bank": letter,
            "node": node,
            "ds_stream": ds_stream,
        }
        catchment, window = catchment_from_cells(
            grid,
            name,
            "point",
            -1,
            cells,
            np.asarray(sub["seed"], dtype=np.int64),
            mode="incremental",
            keep_out=any_sub,
            route_cells=route,
            info={"ds_ids": ds_stream or ""},
            extra=extra,
        )
        catchment.detail["network_sub"] = extra
        if split is not None:  # for the polygon cut in the QGIS wrapper
            catchment.detail["split"] = {
                "line": split["line_xy"],
                "whole": split["whole_polygons"],
                "side_map": split["side_map"],
                "window": split["window"],
                "pieces": split["n_pieces"],
                "merged": split["n_merged"],
            }
        result.catchments.append(catchment)
        windows.append(window)

    report(0.96, "Building catchment DEM")
    catchment_dem(
        result, windows, grid.dem_flat, (grid.rows, grid.cols), dem_nodata,
        geotransform,
    )  # fmt: skip
    result.outlet_cells = list(outlet_cells or [])
    report(1.0, "Done")
    return result


def centreline_cells(reach: dict, kids: dict) -> np.ndarray:
    """Flat cells of a reach's centreline, upstream first: the stream cell
    above the reach (on the main inflow, if any), the reach's cells and the
    node below it."""
    cells = np.asarray(reach["_cells"], dtype=np.int64)
    above = sorted(
        kids.get(reach["link_id"], []), key=lambda k: -(k.get("us_km2") or 0.0)
    )
    if above and len(above[0]["_cells"]) > 1:
        cells = np.concatenate([[int(np.asarray(above[0]["_cells"])[-2])], cells])
    return cells


def _extend(xy: np.ndarray, inside, cell: float, max_steps: int = 400) -> np.ndarray:
    """Extends a polyline straight past both ends until each end leaves the
    area (``inside(x, y)`` is False)."""
    pts = [tuple(p) for p in xy]
    for end in (0, 1):
        seq = pts if end else pts[::-1]
        (x0, y0), (x1, y1) = seq[-2], seq[-1]
        d = math.hypot(x1 - x0, y1 - y0)
        if d == 0:
            continue
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        x, y, k = x1, y1, 0
        while inside(x, y) and k < max_steps:
            x, y, k = x + 0.5 * cell * ux, y + 0.5 * cell * uy, k + 1
        if k:
            x, y = x + cell * ux, y + cell * uy  # clear of the boundary
            seq.append((x, y))
        pts = seq if end else seq[::-1]
    return np.asarray(pts, dtype=np.float64)


def split_along_centreline(
    grid,
    cells: np.ndarray,
    line_cells: np.ndarray,
    any_sub: np.ndarray,
    min_cells: int = DEFAULT_MIN_SPLIT_CELLS,
    min_width_cells: float = DEFAULT_MIN_SPLIT_WIDTH_CELLS,
):
    """Splits the sub-catchment ``cells`` (flat) along its stream centreline
    (``line_cells``, upstream first). Returns None when it is not split (one
    side below the tolerance), else a dict with 'banks' [(letter, cells)],
    'line_xy' (the extended centreline), 'whole_polygons' (outline of the
    whole sub-catchment, map rings), 'side_map' (window of 0 outside, 1 left,
    2 right, 3 = on the line, left bank), 'window' (row0, col0) and the
    number of pieces before and after the tolerance merge."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    gt, cols = grid.geotransform, grid.cols
    cell = min(grid.sx, grid.sy)
    rr, cc = cells // cols, cells % cols
    r0, c0 = int(rr.min()) - 1, int(cc.min()) - 1
    h, w = int(rr.max()) - r0 + 2, int(cc.max()) - c0 + 2
    mask = np.zeros((h, w), dtype=bool)
    mask[rr - r0, cc - c0] = True

    def to_local(x, y):
        r = int(math.floor((y - gt[3]) / gt[5])) - r0
        c = int(math.floor((x - gt[0]) / gt[1])) - c0
        return r, c

    def inside(x, y):
        r, c = to_local(x, y)
        return 0 <= r < h and 0 <= c < w and mask[r, c]

    lx, ly = cell_center(line_cells // cols, line_cells % cols, gt)
    xy = _extend(np.column_stack([lx, ly]), inside, cell)

    # line samples every quarter cell: position, segment and chainage
    samples, seg_of, chain = [], [], []
    total = 0.0
    for i in range(len(xy) - 1):
        (x0, y0), (x1, y1) = xy[i], xy[i + 1]
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(1, math.ceil(d / (0.25 * cell)))
        t = np.arange(n) / n
        samples.append(np.column_stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]))
        seg_of.append(np.full(n, i))
        chain.append(total + t * d)
        total += d
    samples.append(xy[-1:])
    seg_of.append([len(xy) - 2])
    chain.append([total])
    samples = np.vstack(samples)
    seg_of = np.concatenate(seg_of).astype(int)
    chain = np.concatenate(chain)

    on_line = np.zeros((h, w), dtype=bool)
    sr = np.floor((samples[:, 1] - gt[3]) / gt[5]).astype(int) - r0
    sc = np.floor((samples[:, 0] - gt[0]) / gt[1]).astype(int) - c0
    ok = (sr >= 0) & (sr < h) & (sc >= 0) & (sc < w)
    on_line[sr[ok], sc[ok]] = True
    on_line &= mask

    # pieces: 4-connected areas the (8-connected) line separates
    regions, n_reg = ndimage.label(mask & ~on_line)
    if n_reg == 0:
        return None
    pr, pc = np.nonzero(regions)
    px, py = cell_center(pr + r0, pc + c0, gt)
    tree = cKDTree(samples)
    _, near = tree.query(np.column_stack([px, py]))
    seg = seg_of[near]
    ax, ay = xy[seg, 0], xy[seg, 1]
    bx, by = xy[seg + 1, 0], xy[seg + 1, 1]
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    lab = regions[pr, pc]
    left_votes = np.bincount(
        lab, weights=(cross > 0).astype(float), minlength=n_reg + 1
    )
    size = np.bincount(lab, minlength=n_reg + 1).astype(float)
    side = np.where(left_votes >= 0.5 * size, 1, 2)  # 1 left (a), 2 right (b)
    s = chain[near]
    lo = np.full(n_reg + 1, np.inf)
    hi = np.full(n_reg + 1, -np.inf)
    np.minimum.at(lo, lab, s)
    np.maximum.at(hi, lab, s)

    # contact between pieces: side by side, or across one line cell
    contact: dict[tuple[int, int], int] = {}
    pad = np.pad(regions, 2)
    pline = np.pad(on_line, 2)
    for dr, dc in ((0, 1), (1, 0)):
        a = pad[2 : 2 + h, 2 : 2 + w]
        b1 = pad[2 + dr : 2 + dr + h, 2 + dc : 2 + dc + w]
        b2 = pad[2 + 2 * dr : 2 + 2 * dr + h, 2 + 2 * dc : 2 + 2 * dc + w]
        mid = pline[2 + dr : 2 + dr + h, 2 + dc : 2 + dc + w]
        for other, sel in ((b1, (a > 0) & (b1 > 0) & (a != b1)),
                           (b2, (a > 0) & (b2 > 0) & (a != b2) & mid)):  # fmt: skip
            for i, j in zip(a[sel].tolist(), other[sel].tolist(), strict=True):
                key = (min(i, j), max(i, j))
                contact[key] = contact.get(key, 0) + 1

    # tolerance: merge small or thin pieces into their best neighbour
    parent = list(range(n_reg + 1))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cell_area = grid.sx * grid.sy
    min_width_m = min_width_cells * cell

    def fails(i):
        length = max(hi[i] - lo[i], cell)
        area = size[i] * cell_area
        return size[i] < min_cells or area / length < min_width_m

    n_merged = 0
    while True:
        roots = sorted({root(i) for i in range(1, n_reg + 1)}, key=lambda i: size[i])
        failing = [i for i in roots if fails(i)]
        if not failing or len(roots) == 1:
            break
        i = failing[0]
        best, best_n = None, 0
        for (p, q), k in contact.items():
            rp, rq = root(p), root(q)
            if rp == rq or i not in (rp, rq):
                continue
            j = rq if rp == i else rp
            if k > best_n:
                best, best_n = j, k
        if best is None:  # an isolated part: keep it with its own side's largest
            same = [j for j in roots if j != i and side[j] == side[i]]
            other = [j for j in roots if j != i]
            best = max(same or other, key=lambda j: size[j])
        parent[i] = best
        size[best] += size[i]
        lo[best] = min(lo[best], lo[i])
        hi[best] = max(hi[best], hi[i])
        n_merged += 1

    final_side = np.array([side[root(i)] for i in range(n_reg + 1)])
    side_map = np.zeros((h, w), dtype=np.int8)
    side_map[pr, pc] = final_side[lab]
    side_map[on_line] = 3
    left = (side_map == 1) | (side_map == 3)
    right = side_map == 2
    if not (side_map == 1).any() or not right.any():
        return None

    def flat(m):
        r, c = np.nonzero(m)
        return (r + r0) * cols + (c + c0)

    # outline of the whole sub-catchment (the wrapper cuts it along the line)
    keep = (
        any_sub[
            ((np.arange(h)[:, None] + r0).clip(0, grid.rows - 1)) * cols
            + (np.arange(w)[None, :] + c0).clip(0, cols - 1)
        ]
        & ~mask
    )
    _filled, rings_idx = fill_holes(mask, keep)
    whole = [
        [ring_to_map(simplify_ring(r), gt, r0, c0) for r in poly]
        for poly in group_polygons(rings_idx)
    ]
    return {
        "banks": [(BANK_LEFT, flat(left)), (BANK_RIGHT, flat(right))],
        "line_xy": [tuple(map(float, p)) for p in xy],
        "whole_polygons": whole,
        "side_map": side_map,
        "window": (r0, c0),
        "n_pieces": int(n_reg),
        "n_merged": int(n_merged),
    }


def node_below(reach: dict, by_id: dict) -> str:
    """Name of the node at the lower end of ``reach``: the reach's own name
    on the stem (and at the outlet), but for a branch the name of the
    confluence it joins, i.e. the main-stem reach above that confluence."""
    ds = by_id.get(reach["ds_link"]) if reach["ds_link"] is not None else None
    if ds is None:
        return reach["name"]
    return f"{ds['stem']}{ds['seg'] + 1}"


def _system_letters(stem: str | None) -> str | None:
    """Leading letters of a stem name: the system (A for A3A, AB for AB2C)."""
    if not stem:
        return None
    m = re.match(r"[A-Z]+", stem)
    return m.group(0) if m else None
