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
  (A1a, A1b). Every cell takes the bank on which its flow path joins the
  stream; the stream cells themselves go with the left bank. Each bank keeps
  its own centroid and longest flow path, which runs down the stream to the
  node. Channel head sub-catchments are not split.

Names follow the Stream Network core (name_network): each system is a stem
A, B, C ... by area, numbered upstream from its outlet; branches take the
name of the reach they flow into plus a letter (A3A1 ...).

Each sub-catchment records the node it drains to (``node``) and the stream
that node discharges into (``ds_stream``, empty at the outlet), so the
sub-catchments can be linked into a hydrological model (sub-basin ->
junction ``node`` -> reach ``ds_stream``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from mayim_tools.hydrology._common.d8_network import (
    build_downstream_index,
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
)

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


def bank_side(e, s, up, dn, cols, sx, sy) -> int:
    """1 (left bank, looking downstream) or 2 (right bank) for a cell ``e``
    flowing into stream cell ``s``; ``up`` and ``dn`` are the stream cells
    above and below ``s`` (flat indices). Uses the stream direction through
    ``s`` (up -> dn), then s -> dn alone if the cell lies on that line."""

    def xy(i):
        return (i % cols) * sx, -(i // cols) * sy

    ex, ey = xy(e)
    sx_, sy_ = xy(s)
    px, py = ex - sx_, ey - sy_
    for a, b in ((up, dn), (s, dn), (up, s)):
        ax, ay = xy(a)
        bx, by = xy(b)
        tx, ty = bx - ax, by - ay
        cross = tx * py - ty * px
        if cross > 0:
            return 1
        if cross < 0:
            return 2
    return 1


def bank_sides(e, s, up, dn, cols, sx, sy) -> np.ndarray:
    """Vectorised bank_side for arrays of flat indices."""
    e, s, up, dn = (np.asarray(a, dtype=np.int64) for a in (e, s, up, dn))

    def xy(i):
        return (i % cols) * sx, -(i // cols) * sy

    ex, ey = xy(e)
    cx, cy = xy(s)
    px, py = ex - cx, ey - cy
    out = np.zeros(e.size, dtype=np.int8)
    for a, b in ((up, dn), (s, dn), (up, s)):
        ax, ay = xy(a)
        bx, by = xy(b)
        cross = (bx - ax) * py - (by - ay) * px
        todo = out == 0
        out[todo & (cross > 0)] = 1
        out[todo & (cross < 0)] = 2
    out[out == 0] = 1
    return out


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
    up_cell = np.full(n, -1, dtype=np.int64)
    dn_cell = np.full(n, -1, dtype=np.int64)
    for r in reaches:
        cells = np.asarray(r["_cells"], dtype=np.int64)
        own = cells if r["ds_link"] is None else cells[:-1]
        for i in range(own.size):  # stream direction through each own cell
            c = int(cells[i])
            if i > 0:
                up_cell[c] = cells[i - 1]
            else:
                above = sorted(
                    kids[r["link_id"]], key=lambda k: -(k.get("us_km2") or 0.0)
                )
                up_cell[c] = (
                    int(np.asarray(above[0]["_cells"])[-2])
                    if above and len(above[0]["_cells"]) > 1
                    else c
                )
            dn_cell[c] = cells[i + 1] if i + 1 < cells.size else nxt[c]
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

    # --- banks ------------------------------------------------------------------
    bank = np.zeros(n, dtype=np.int8)
    stream_label = np.zeros(len(subs) + 1, dtype=bool)
    for i, sub in enumerate(subs, start=1):
        stream_label[i] = sub["sub_type"] == SUB_STREAM
    if split_banks:
        report(0.40, "Splitting along the stream centreline")
        down = nxt
        entry = (
            ok
            & (seed == 0)
            & (down != idx)
            & (seed[down] > 0)
            & stream_label[seed[down]]
        )
        side_of = np.zeros(n, dtype=np.int8)
        ent = np.flatnonzero(entry)
        side_of[ent] = bank_sides(
            ent, down[ent], up_cell[down[ent]], dn_cell[down[ent]], cols,
            grid.sx, grid.sy,
        )  # fmt: skip
        nxt_e = nxt_s.copy()
        nxt_e[entry] = idx[entry]
        term2, _, _ = resolve_terminals(nxt_e)
        bank = side_of[term2]
        bank[(seed > 0)] = 1  # stream cells go with the left bank
        bank[(bank == 0) & ok & (label > 0)] = 1
        bank[~stream_label[label]] = 0

    # --- one catchment per sub-catchment (and bank) ----------------------------
    labelled = np.flatnonzero(label)
    order = labelled[np.argsort(label[labelled], kind="stable")]
    bounds = np.searchsorted(label[order], np.arange(1, len(subs) + 2))
    any_sub = label > 0
    pieces = []
    for i, sub in enumerate(subs, start=1):
        cells = order[bounds[i - 1] : bounds[i]]
        if cells.size == 0:
            continue
        if split_banks and sub["sub_type"] == SUB_STREAM:
            for side, letter in ((1, BANK_LEFT), (2, BANK_RIGHT)):
                part = cells[bank[cells] == side]
                if part.size:
                    pieces.append((sub, f"{sub['name']}{letter}", letter, part, cells))
        else:
            pieces.append((sub, sub["name"], None, cells, None))
    pieces.sort(key=lambda p: natural_key(p[1]))

    windows = []
    for k, (sub, name, letter, cells, route) in enumerate(pieces):
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
