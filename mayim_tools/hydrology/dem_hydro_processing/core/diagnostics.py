"""Post-correction diagnostics: where does the corrected DEM still fail D8?

A cell is "D8-resolvable" when it has at least one strictly lower valid
8-neighbour, or it sits on the raster edge / next to NoData (a legitimate
outlet). Any other cell is a pit or flat-interior cell: flow accumulation
stops there. This module finds those cells, clusters them, and maps each
cluster back to the Stage 1 feature(s) that contain it in the ORIGINAL
DEM, so the cause (protected / never selected / resolved but still pit)
is visible instead of guessed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .depression_hierarchy import build_depression_hierarchy
from .priority_flood import _NEIGHBOR_OFFSETS


def find_unresolvable_cells(dem: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Boolean mask of valid, non-outlet cells with no strictly lower neighbour.
    Compared in float32, the precision the corrected DEM is actually stored in."""
    z = np.asarray(dem, dtype=np.float32)
    rows, cols = z.shape
    valid = (
        np.isfinite(z)
        if nodata is None
        else (np.isfinite(z) & (z != np.float32(nodata)))
    )
    pad = np.full((rows + 2, cols + 2), np.nan, dtype=np.float32)
    pad[1:-1, 1:-1] = np.where(valid, z, np.nan)
    has_lower = np.zeros((rows, cols), dtype=bool)
    outlet = np.zeros((rows, cols), dtype=bool)
    for dr, dc in _NEIGHBOR_OFFSETS:
        nb = pad[1 + dr : 1 + dr + rows, 1 + dc : 1 + dc + cols]
        outlet |= np.isnan(nb)  # off-raster or NoData neighbour => drains out
        with np.errstate(invalid="ignore"):
            has_lower |= nb < z
    return valid & ~outlet & ~has_lower


@dataclass
class PitCluster:
    cluster_id: int
    num_cells: int
    row: int  # lowest cell of the cluster
    col: int
    elevation: float
    original_changed: bool  # did correction touch any cell of this cluster?
    feature_chain: list = field(
        default_factory=list
    )  # [(feature_id, classification, protected_by, resolution)]


def diagnose(
    original: np.ndarray,
    corrected: np.ndarray,
    nodata: float | None,
    report_csv: str | None = None,
) -> list[PitCluster]:
    mask = find_unresolvable_cells(corrected, nodata)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    h = build_depression_hierarchy(np.asarray(original, dtype=np.float64), nodata)
    rows_by_id: dict[int, dict] = {}
    if report_csv:
        with open(report_csv, newline="", encoding="utf-8-sig") as fh:
            for rec in csv.DictReader(fh):
                rows_by_id[int(rec["feature_id"])] = rec
    changed = ~np.isclose(
        np.asarray(original, np.float32),
        np.asarray(corrected, np.float32),
        atol=0,
        rtol=0,
    )
    out = []
    for cid in range(1, n + 1):
        cells = np.argwhere(labels == cid)
        elev = np.asarray(corrected, np.float32)[cells[:, 0], cells[:, 1]]
        k = int(np.argmin(elev))
        r, c = int(cells[k, 0]), int(cells[k, 1])
        chain, node_id, seen = [], int(h.leaf_label[r, c]), set()
        while node_id is not None and node_id >= 0 and node_id not in seen:
            seen.add(node_id)
            node = h.nodes[node_id]
            if node.is_ocean:
                break
            rec = rows_by_id.get(node_id)
            if rec:
                chain.append(
                    (
                        node_id,
                        rec.get("classification", ""),
                        rec.get("protected_by", ""),
                        rec.get("resolution", ""),
                    )
                )
            else:
                chain.append((node_id, "(not in report)", "", ""))
            node_id = node.parent
        out.append(
            PitCluster(
                cid,
                len(cells),
                r,
                c,
                float(elev[k]),
                bool(changed[cells[:, 0], cells[:, 1]].any()),
                chain,
            )
        )
    out.sort(key=lambda p: -p.num_cells)
    return out


def format_diagnosis(clusters: list[PitCluster], max_chain: int = 4) -> str:
    if not clusters:
        return "No unresolvable interior cells: the corrected DEM is D8-resolvable everywhere."
    lines = [
        f"{len(clusters)} unresolved pit/flat cluster(s) in the corrected DEM:",
        "cluster, cells, row, col, x(col)_y(row), elevation, touched_by_correction, feature chain (leaf -> root)",
    ]
    for p in clusters:
        chain = " > ".join(
            f"{fid}[{cls}|prot={prot or '-'}|res={res or '-'}]"
            for fid, cls, prot, res in p.feature_chain[:max_chain]
        )
        lines.append(
            f"{p.cluster_id}, {p.num_cells}, {p.row}, {p.col}, {p.col}x{p.row}, "
            f"{p.elevation:.4f}, {p.original_changed}, {chain}"
        )
    return "\n".join(lines)
