"""2D hydraulic structure geometry reader (bridges, culverts, weirs/gates,
inline and lateral structures, SA/2D and 2D/2D connections).

Confirmed 2026-09-22 against a real project (M2627_model.g02.hdf): HEC-RAS
stores every hydraulic structure attached to a 2D model - bridges,
culverts, gated/ungated weirs, inline structures, lateral structures, and
plain SA/2D or 2D/2D connections - in ONE group, 'Geometry/Structures',
not one group per structure type. The structure's centerline geometry
uses the same storage convention as Reference Lines/breaklines, just
under the names 'Centerline Info/Parts/Points' instead of 'Polyline
Info/Parts/Points' - confirmed byte-for-byte against the real file (a
single Bridge connection: Centerline Info [[0, 2, 0, 1]], Centerline
Parts [[0, 2]], 2 Centerline Points), so this is a direct reuse of
core/hdf5_utils.read_polyline_features via its configurable dataset
names, no new decoding logic needed.

What distinguishes a bridge from a culvert from a plain connection is
NOT the group location but the Attributes row:
  - 'Type'        - almost always 'Connection' for a 2D model (vs.
                     'River' for a 1D cross-section structure, which
                     this reader does not cover - those live under
                     Geometry/Cross Sections, not Structures).
  - 'Connection'  - the structure's own type, e.g. 'Bridge', 'Culvert',
                     'Inline Structure', 'Lateral Structure', or blank
                     for a plain SA/2D connection with no structure.
  - 'Mode'        - what hydraulic elements the connection contains,
                     e.g. 'Weir/Gate/Culverts'.
  - 'Culvert Groups' / 'Gate Groups' - counts of culvert/gate groups
                     defined on the connection (0 if none).
  - 'US SA/2D' / 'DS SA/2D' - the Storage Area or 2D Flow Area on each
                     side of the connection (both the same area name
                     for a structure that sits on one area's perimeter,
                     e.g. a bridge over a break line).
This reader does not attempt to interpret those fields into a single
canonical "structure type" - it exposes the raw Attributes row (all ~90
columns HEC-RAS stores per structure: weir geometry, bridge coefficients,
HTAB settings, cross-section station/elevation bank markers, etc.) and
leaves classification/styling to the caller/UI, same as the boundary
condition lines reader's 'Type' field.

Zero QGIS dependency - pure h5py/Python, independently testable (see
tests/test_structures_core.py).
"""

from __future__ import annotations

import h5py

from mayim_tools.hec_ras._common.hdf5_utils import (
    PolylineFeature,
    read_polyline_features,
)

# Confirmed 2026-09-22 against a real project's geometry HDF5.
STRUCTURES_GROUP = "Geometry/Structures"

# Confirmed 2026-09-23 against a real project (M2627_model.g02.hdf) once
# Chris pointed out that the "culvert" checklist item should be the
# CULVERT CENTRELINE (one per barrel), not the parent SA/2D connection's
# own centerline. HEC-RAS stores culvert geometry as a separate, FLAT,
# model-wide sub-group of Structures - not nested per-connection:
#   Geometry/Structures/Culvert Groups            - one row per culvert
#     group (a culvert group can have >1 identical barrels), with
#     'Structure ID' linking back to a Culvert Groups position (see
#     read_culvert_barrels docstring) and a 'Barrels' count.
#   Geometry/Structures/Culvert Groups/Barrels     - one row per BARREL
#     (the actual pipe/box run), with its own 'Centerline Info/Parts/
#     Points' geometry (same convention as Structures' own Centerline),
#     plus 'Structure ID' and 'Culvert Group ID' fields to trace a
#     barrel back to its owning connection/group.
# Confirmed against real data: one culvert group ('Group #1', Circular,
# 1 barrel) on the one weir/culvert connection in this project, with the
# barrel's own 2-point centerline distinct from (not a duplicate of) the
# parent connection's centerline.
CULVERT_BARRELS_GROUP = "Geometry/Structures/Culvert Groups/Barrels"


def read_structures(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every 2D hydraulic structure (bridge, culvert, weir/gate,
    inline/lateral structure, or plain SA/2D or 2D/2D connection)
    defined in a HEC-RAS geometry HDF5 file, as its centerline geometry
    plus full Attributes row.

    Returns an empty list if the project has no structures defined at
    all, rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if STRUCTURES_GROUP not in h5file:
            return []
        return read_polyline_features(
            h5file,
            STRUCTURES_GROUP,
            info_name="Centerline Info",
            parts_name="Centerline Parts",
            points_name="Centerline Points",
        )


# --- Structure-subtype classification -------------------------------
#
# Confirmed 2026-09-23 against a real project once Chris added a second
# connection (a plain SA/2D connection with a culvert group) alongside
# the existing bridge. Chris asked for these as SEPARATE checklist
# items/output layers, so the functions below partition the single
# read_structures() result by its Attributes fields rather than reading
# any new HDF5 location - there is still only one group
# ('Geometry/Structures') and one read per file; this is a filter, not
# a new decode.
#
# Classification rules, in order (a structure matches the first rule it
# satisfies):
#   1. Connection == 'Bridge' and Mode == 'Bridge Opening (2D)'
#        -> "Bridge 2D" (RAS's newer 2D-mesh bridge-opening method).
#   2. Connection == 'Bridge' (any other Mode, e.g. the traditional
#      Energy/Momentum/Yarnell/WSPro methods using the Bridge
#      Coefficient Attributes table)
#        -> "Bridge 1D".
#   3. Mode == 'Linear Routing'
#        -> "Linear Routing".
#   4. Everything else with Type == 'Connection' (in practice, Mode ==
#      'Weir/Gate/Culverts' - RAS's generic label for a plain SA/2D or
#      2D/2D connection using the weir equation plus optional gates
#      and/or culverts)
#        -> "Weir".
# "Culvert" is NOT a fifth mutually-exclusive category - per Chris
# ("culvert (subset of weir)"), it's the SUBSET of Weir connections
# that also have at least one culvert group defined ('Culvert Groups'
# > 0), so a culvert-bearing connection appears in BOTH the Weir output
# and the Culvert output.
#
# Confirmed against real data: the one Bridge in the test project has
# Mode 'Bridge Opening (2D)' -> classified Bridge 2D; the one plain
# connection has Mode 'Weir/Gate/Culverts' and Culvert Groups=1 ->
# classified Weir, and also appears in Culvert. No Bridge 1D or Linear
# Routing example exists in this project yet, so those two functions
# are unexercised against real data (they'll simply return [] until a
# project has one, same as every other not-yet-populated element type
# in this plugin).


def _is_bridge(feature: PolylineFeature) -> bool:
    return feature.attributes.get("Connection") == "Bridge"


def _is_bridge_2d(feature: PolylineFeature) -> bool:
    mode = feature.attributes.get("Mode")
    return _is_bridge(feature) and mode == "Bridge Opening (2D)"


def _is_linear_routing(feature: PolylineFeature) -> bool:
    mode = feature.attributes.get("Mode")
    return not _is_bridge(feature) and mode == "Linear Routing"


def _is_weir(feature: PolylineFeature) -> bool:
    return not _is_bridge(feature) and not _is_linear_routing(feature)


def _has_culvert(feature: PolylineFeature) -> bool:
    try:
        return int(feature.attributes.get("Culvert Groups") or 0) > 0
    except (TypeError, ValueError):
        return False


def read_bridge_2d_structures(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Structures classified as a 2D bridge opening (Connection ==
    'Bridge', Mode == 'Bridge Opening (2D)'). See classification notes
    above."""
    return [f for f in read_structures(geometry_hdf5_path) if _is_bridge_2d(f)]


def read_bridge_1d_structures(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Structures classified as a traditional 1D-method bridge
    (Connection == 'Bridge', any Mode other than 'Bridge Opening
    (2D)'). See classification notes above."""
    structures = read_structures(geometry_hdf5_path)
    return [f for f in structures if _is_bridge(f) and not _is_bridge_2d(f)]


def read_weir_structures(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Structures classified as a plain SA/2D or 2D/2D weir/gate/
    culvert connection (not a Bridge, not Linear Routing). Includes
    connections with a culvert group - see read_culvert_structures for
    that subset. See classification notes above."""
    return [f for f in read_structures(geometry_hdf5_path) if _is_weir(f)]


def read_linear_routing_structures(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Structures classified as a Linear Routing connection (Mode ==
    'Linear Routing'). See classification notes above."""
    return [f for f in read_structures(geometry_hdf5_path) if _is_linear_routing(f)]


def read_culvert_structures(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """The subset of Weir connections that also have at least one
    culvert group defined ('Culvert Groups' > 0) - a SUBSET of
    read_weir_structures, not a mutually exclusive category. See
    classification notes above.

    NOTE: this returns the parent CONNECTION's own centerline, one
    feature per culvert-bearing connection - not the individual barrel
    centerlines. For the "culvert" checklist item/output layer, use
    read_culvert_barrels instead (2026-09-23: Chris confirmed the
    culvert output should be each barrel's own centreline, not the
    connection's). Kept here since "which connections have a culvert"
    is still a useful, distinct query in its own right.
    """
    structures = read_structures(geometry_hdf5_path)
    return [f for f in structures if _is_weir(f) and _has_culvert(f)]


def read_culvert_barrels(geometry_hdf5_path: str) -> list[PolylineFeature]:
    """Read every culvert BARREL's own centerline geometry (2026-09-23)
    - one feature per physical pipe/box run, not one per SA/2D
    connection. This is what the "culvert" checklist item/output layer
    now uses, per Chris: "The SA_2D_connection_-_culvert should be the
    Culvert Centreline and not the SA/2D connection line. So it will be
    the centreline of each barrel."

    Each feature's attributes carry the barrel's own row from
    Geometry/Structures/Culvert Groups/Barrels/Attributes, including
    'Structure ID' and 'Culvert Group ID' fields a caller can use to
    trace a barrel back to its owning connection/culvert group if
    needed (not resolved to a name here, since Structures/Attributes
    and Culvert Groups/Attributes have not been confirmed against a
    project with more than one culvert-bearing connection yet - the
    exact 'Structure ID' numbering scheme, e.g. whether it's a global
    Structures row index or a counter local to culvert-bearing
    connections only, is unconfirmed with only one real example to
    check against).

    Returns an empty list if the project has no culvert barrels defined
    at all, rather than raising.
    """
    with h5py.File(geometry_hdf5_path, "r") as h5file:
        if CULVERT_BARRELS_GROUP not in h5file:
            return []
        return read_polyline_features(
            h5file,
            CULVERT_BARRELS_GROUP,
            info_name="Centerline Info",
            parts_name="Centerline Parts",
            points_name="Centerline Points",
        )
