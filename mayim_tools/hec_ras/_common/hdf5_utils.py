"""Low-level, reusable helpers for reading HEC-RAS 6.x/7.0 HDF5 geometry
and plan-output files with h5py directly - no ras-commander / rashdf
dependency (see the project's architecture decision: those libraries
either don't cover Reference Lines/Points at all, or are a volatile
dependency for a tool meant to be stable).

Zero QGIS dependency by design, so every function here is independently
unit-testable against a small synthetic HDF5 fixture (see
tests/test_hdf5_utils.py), without QGIS, without a real HEC-RAS project,
and without any proprietary test data.

CONFIRMED vs UNVERIFIED
------------------------
The "Polyline Info / Polyline Parts / Polyline Points" triple decoded by
``read_polyline_features`` is the storage convention CONFIRMED against a
real project for ``Geometry/Reference Lines/`` (see
claude/hec_ras_automation_plugin_research.md), and is documented
elsewhere (rashdf, HEC-RAS's own HDF5 schema notes) as the same
convention used for 2D Flow Area breaklines, boundary condition lines,
and mesh perimeters. It has NOT yet been re-confirmed byte-for-byte
against this project's own files for every one of those other geometry
types - each new reader built on top of this module should sanity-check
its first real file and flag any column-order surprises here.

The point-dataset convention used by ``read_point_features`` (an
``Attributes`` table plus a plain ``(N, 2)`` ``Points`` dataset) is
CONFIRMED 2026-09-22 against a real project for Reference Points, IC
Points, Pipe Nodes, and Pump Stations - all four use exactly this
shape, just under different group names.

``read_polygon_features`` is a thin, semantically-named wrapper around
``read_polyline_features``: HEC-RAS's ``Polygon Info/Polygon
Parts/Polygon Points`` triple (used for 2D Flow Area perimeters,
refinement regions, and reference areas) is byte-for-byte the same
``[point_start, point_count, part_start, part_count]`` encoding as the
``Polyline`` triple, confirmed 2026-09-22 against real refinement
region and reference area data (both pre-closed rings, first point
repeated as the last). Use whichever wrapper name matches the geometry
being decoded; both call the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolylineFeature:
    """One polyline feature decoded from a Polyline Info/Parts/Points
    triple. ``parts`` is a list of parts (almost always length 1 for
    Reference Lines/breaklines, but the encoding supports more), each
    part a list of (x, y) tuples in order."""

    attributes: dict
    parts: list[list[tuple[float, float]]] = field(default_factory=list)

    @property
    def name(self) -> str | None:
        """Convenience accessor - RAS attribute tables almost always
        carry a 'Name' field, but not guaranteed for every geometry
        type, so callers should not assume it's non-None."""
        for key in ("Name", "name"):
            if key in self.attributes:
                return self.attributes[key]
        return None


@dataclass
class PointFeature:
    """One point feature: its attribute row plus a single (x, y)."""

    attributes: dict
    xy: tuple[float, float]

    @property
    def name(self) -> str | None:
        for key in ("Name", "name"):
            if key in self.attributes:
                return self.attributes[key]
        return None


def decode_value(value):
    """Decode a single value out of an h5py structured-array row:
    bytes -> stripped str, numpy scalar -> plain Python scalar,
    anything else passed through unchanged."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if hasattr(value, "item"):
        # numpy scalar (int64, float32, etc.)
        return value.item()
    return value


def read_attribute_table(h5file, dataset_path: str) -> list[dict]:
    """Read an HDF5 compound/structured dataset (RAS's near-universal
    'Attributes' table convention) into a list of plain-Python dicts,
    one per row, with byte-string fields decoded to str.

    Returns an empty list if the dataset does not exist, rather than
    raising - callers decide whether an absent table is an error for
    their specific geometry type.
    """
    if dataset_path not in h5file:
        return []

    dataset = h5file[dataset_path]
    field_names = dataset.dtype.names
    if not field_names:
        raise ValueError(
            f"'{dataset_path}' is not a compound/structured dataset "
            f"(no field names) - can't read it as an attribute table."
        )

    rows = []
    for raw_row in dataset[()]:
        rows.append({name: decode_value(raw_row[name]) for name in field_names})
    return rows


def read_polyline_features(
    h5file,
    group_path: str,
    attributes_name: str = "Attributes",
    info_name: str = "Polyline Info",
    parts_name: str = "Polyline Parts",
    points_name: str = "Polyline Points",
) -> list[PolylineFeature]:
    """Decode a RAS 'Polyline Info / Polyline Parts / Polyline Points'
    triple into a list of PolylineFeature objects, one per row of
    Polyline Info, each carrying its matching Attributes row.

    Expected shapes (RAS's documented convention, confirmed for
    Geometry/Reference Lines/ - see module docstring):
      - Polyline Info:   (N, 4) int - [point_start, point_count,
                          part_start, part_count] per feature
      - Polyline Parts:  (M, 2) int - [point_start, point_count] per
                          part (M total parts across all N features)
      - Polyline Points: (P, 2) float - X, Y per point (P total points
                          across all parts)
      - Attributes:      (N,) compound dataset, one row per feature,
                          same order as Polyline Info

    IMPORTANT (2026-09-23 bugfix): a Polyline Parts row's own
    ``point_start`` column is RELATIVE to its owning feature's own
    Polyline Info ``point_start`` - it is NOT already an absolute index
    into Polyline Points. Confirmed against the real M2627 file: for
    every feature after the first, using the Parts row's point_start
    directly as an absolute Points index reads back the SAME leading
    slice of Polyline Points for many different features (whichever
    ones happen to share the same point_count), while the true,
    non-overlapping, gap-free reconstruction - Info's own point_start
    PLUS the Parts row's (relative) point_start - exactly tiles the
    whole Polyline Points array with zero gaps or overlaps. This bug
    was invisible in every previous confirmation of this reader (single
    -feature groups like Reference Lines/Reference Areas/Refinement
    Regions) because the first (and only) feature's point_start is
    always 0, so "absolute" and "relative to 0" happen to coincide -
    it only shows up with 2+ features in one group, which is exactly
    what Chris reported: 11 breaklines with a correct attribute table
    but wrong/duplicated/misplaced geometry once a 2nd, 3rd, ... feature
    existed to expose the difference.

    Raises KeyError if group_path itself doesn't exist, and ValueError
    if the Info/Parts/Points datasets are missing or an index range
    runs past the end of an array (a strong signal the column-order
    assumption above doesn't hold for this geometry type and needs
    re-checking against the real file).
    """
    if group_path not in h5file:
        raise KeyError(f"'{group_path}' not found in this HDF5 file.")

    group = h5file[group_path]
    for required in (info_name, parts_name, points_name):
        if required not in group:
            raise ValueError(
                f"'{group_path}/{required}' not found - this geometry "
                f"type may not use the Polyline Info/Parts/Points "
                f"convention, or the dataset names differ here."
            )

    info = group[info_name][()]
    parts_table = group[parts_name][()]
    points_table = group[points_name][()]
    attributes = read_attribute_table(h5file, f"{group_path}/{attributes_name}")

    n_points = len(points_table)
    n_parts = len(parts_table)

    features: list[PolylineFeature] = []
    for feature_index, row in enumerate(info):
        point_start, point_count, part_start, part_count = (int(v) for v in row[:4])

        if part_start + part_count > n_parts:
            raise ValueError(
                f"{group_path}/{info_name} row {feature_index}: part "
                f"range [{part_start}:{part_start + part_count}] runs "
                f"past Polyline Parts length {n_parts} - re-check "
                f"column order against the real file."
            )

        parts: list[list[tuple[float, float]]] = []
        for part_row in parts_table[part_start : part_start + part_count]:
            # part_row's own point_start is RELATIVE to this feature's
            # Polyline Info point_start - see the docstring note above.
            relative_p_start, p_count = int(part_row[0]), int(part_row[1])
            absolute_p_start = point_start + relative_p_start
            if absolute_p_start + p_count > n_points:
                raise ValueError(
                    f"{group_path}/{parts_name}: point range "
                    f"[{absolute_p_start}:{absolute_p_start + p_count}] "
                    f"(feature point_start {point_start} + part-relative "
                    f"start {relative_p_start}) runs past Polyline Points "
                    f"length {n_points}."
                )
            coords = [
                (float(x), float(y))
                for x, y in points_table[absolute_p_start : absolute_p_start + p_count]
            ]
            parts.append(coords)

        # Fall back to using Polyline Info's own point range directly if
        # a feature has zero parts recorded but a non-zero point range -
        # defensive only; not expected for the confirmed Reference Lines
        # schema, kept here in case another geometry type encodes it
        # differently.
        if not parts and point_count > 0:
            coords = [
                (float(x), float(y))
                for x, y in points_table[point_start : point_start + point_count]
            ]
            parts.append(coords)

        feature_attrs = (
            attributes[feature_index] if feature_index < len(attributes) else {}
        )
        features.append(PolylineFeature(attributes=feature_attrs, parts=parts))

    return features


def read_polygon_features(
    h5file,
    group_path: str,
    attributes_name: str = "Attributes",
    info_name: str = "Polygon Info",
    parts_name: str = "Polygon Parts",
    points_name: str = "Polygon Points",
) -> list[PolylineFeature]:
    """Decode a RAS 'Polygon Info/Parts/Points' triple - see module
    docstring: structurally identical to read_polyline_features (same
    [point_start, point_count, part_start, part_count] convention,
    just named for polygon-ring datasets). Each returned feature's
    ``parts`` is its list of closed rings (first point repeated as the
    last, confirmed against real data) - the caller decides whether to
    build a QGIS polygon or line geometry from them.

    Reuses read_polyline_features directly - same KeyError/ValueError
    behaviour on a missing group or an out-of-range index.
    """
    return read_polyline_features(
        h5file,
        group_path,
        attributes_name=attributes_name,
        info_name=info_name,
        parts_name=parts_name,
        points_name=points_name,
    )


def read_point_features(
    h5file,
    group_path: str,
    attributes_name: str = "Attributes",
    points_name: str = "Points",
) -> list[PointFeature]:
    """Decode a simple RAS point-feature group: an Attributes table plus
    a (N, 2) Points dataset of X,Y coordinates, one row per feature in
    the same order.

    UNVERIFIED against a real file as of 2026-09-22 - see module
    docstring. Reference Points may live at this shape under
    'Geometry/Reference Points/', or may instead be folded into the
    Reference Lines group's own Attributes table (RAS sometimes stores
    line and point reference locations together) - the first real
    project checked should confirm which, and this function's
    group_path/points_name defaults adjusted if needed.
    """
    if group_path not in h5file:
        raise KeyError(f"'{group_path}' not found in this HDF5 file.")

    group = h5file[group_path]
    if points_name not in group:
        raise ValueError(
            f"'{group_path}/{points_name}' not found - confirm the "
            f"actual dataset name/shape for point features against a "
            f"real file before relying on this reader."
        )

    points = group[points_name][()]
    attributes = read_attribute_table(h5file, f"{group_path}/{attributes_name}")

    features: list[PointFeature] = []
    for i, (x, y) in enumerate(points):
        feature_attrs = attributes[i] if i < len(attributes) else {}
        features.append(PointFeature(attributes=feature_attrs, xy=(float(x), float(y))))
    return features
