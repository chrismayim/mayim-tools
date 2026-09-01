"""
Mayim Tools - Hydrography Topology Preparation
===============================================

Stage 7B native topology preparation for DEM hydrography enforcement.

This module analyses vector hydrography before any terrain modification
takes place. It identifies network connectivity, endpoint relationships,
intersections, duplicate segments, disconnected components and feature
priorities.

The module does not:

    - Modify the DEM.
    - Modify source geometries.
    - Rasterise hydrography.
    - Burn hydrography into the DEM.
    - Calculate burn depth.
    - Resolve DEM/vector divergence.

Those operations belong to later Stage 7 components.

Methodological basis
--------------------
The topology-preparation design follows the revised Mayim Tools DEM
Hydrological Conditioning methodology. It is intended to support the
topology-aware and scale-aware hydrography-enforcement approach described
in the methodology, including prioritisation by stream order or
upstream-contributing area where those attributes are available.

Generic geometry operations use the public Shapely geometry interface.
Shapely is treated as generic geospatial infrastructure, not as a
hydrography-enforcement algorithm.

IP status
---------
Original Mayim topology-analysis implementation.

No WhiteboxTools, RichDEM or TauDEM runtime implementation is used.
This module does not copy or wrap a third-party hydrography-enforcement
algorithm. The Mayim-specific topology decisions, conflict reporting
and feature-priority logic remain in this module.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import hypot, isfinite
from typing import Any


def prepare_hydrography_topology(
    geometries: Iterable[Any],
    feature_ids: Iterable[Any] | None = None,
    stream_order: Mapping[Any, Any] | None = None,
    upstream_area: Mapping[Any, Any] | None = None,
    endpoint_tolerance: float = 0.0,
) -> dict[str, Any]:
    """
    Prepare a deterministic topology summary for line hydrography.

    Features are analysed using their endpoints. Two features are
    considered connected when an endpoint from one feature is within
    ``endpoint_tolerance`` of an endpoint from another feature.

    Geometric intersections between line features are also reported.
    Intersections do not automatically create endpoint connectivity,
    because crossing lines may represent bridges, overpasses or
    unrelated networks. Such cases are reported for later review.

    Parameters
    ----------
    geometries:
        Iterable of Shapely-like line geometries. Each geometry must
        provide:

            is_empty
            is_valid
            geom_type
            coords, for LineString-like objects
            geoms, for MultiLineString-like objects

    feature_ids:
        Optional feature identifiers in the same order as
        ``geometries``. If omitted, identifiers ``1, 2, 3, ...`` are
        assigned deterministically.

    stream_order:
        Optional mapping from feature ID to stream order. Higher
        stream-order values receive higher priority.

    upstream_area:
        Optional mapping from feature ID to contributing area. Larger
        upstream areas receive higher priority.

    endpoint_tolerance:
        Non-negative tolerance in geometry coordinate units for endpoint
        connectivity.

    Returns
    -------
    dict
        A serialisable topology summary containing:

            valid
            errors
            warnings
            feature_count
            valid_feature_count
            invalid_feature_count
            node_count
            edge_count
            component_count
            components
            endpoints
            intersections
            duplicate_segments
            topology_conflicts
            feature_priorities

    Raises
    ------
    ValueError
        If feature IDs do not match the number of geometries or if the
        endpoint tolerance is invalid.

    Notes
    -----
    This function does not prove that a hydrography network represents
    the true drainage system. It only prepares structural evidence for
    subsequent analysis.
    """
    if endpoint_tolerance < 0.0 or not isfinite(
        float(endpoint_tolerance)
    ):
        raise ValueError(
            "endpoint_tolerance must be finite and non-negative."
        )

    geometry_values = list(geometries) if geometries is not None else []

    if feature_ids is None:
        ids = list(range(1, len(geometry_values) + 1))
    else:
        ids = list(feature_ids)

    if len(ids) != len(geometry_values):
        raise ValueError(
            "feature_ids must contain one identifier per geometry."
        )

    errors: list[str] = []
    warnings: list[str] = []
    topology_conflicts: list[dict[str, Any]] = []

    valid_features: list[dict[str, Any]] = []
    invalid_feature_count = 0

    for index, (feature_id, geometry) in enumerate(
        zip(ids, geometry_values),
        start=1,
    ):
        validation = _validate_line_geometry(
            geometry=geometry,
            feature_id=feature_id,
            position=index,
        )

        if not validation["valid"]:
            invalid_feature_count += 1
            errors.extend(validation["errors"])
            continue

        line_parts = _extract_line_parts(geometry)

        if not line_parts:
            invalid_feature_count += 1
            errors.append(
                f"Feature {feature_id} contains no usable line parts."
            )
            continue

        endpoints = _feature_endpoints(line_parts)

        valid_features.append(
            {
                "feature_id": feature_id,
                "geometry": geometry,
                "line_parts": line_parts,
                "endpoints": endpoints,
                "stream_order": _numeric_attribute(
                    stream_order,
                    feature_id,
                ),
                "upstream_area": _numeric_attribute(
                    upstream_area,
                    feature_id,
                ),
            }
        )

    valid_feature_count = len(valid_features)

    if invalid_feature_count:
        warnings.append(
            f"{invalid_feature_count} feature(s) were excluded from "
            "topology preparation because they were invalid or "
            "not usable as line geometry."
        )

    if valid_feature_count == 0:
        warnings.append(
            "No valid line features are available for topology "
            "preparation."
        )

    nodes, endpoint_connections = _build_endpoint_nodes(
        valid_features=valid_features,
        tolerance=float(endpoint_tolerance),
    )

    components = _build_components(
        valid_features=valid_features,
        endpoint_connections=endpoint_connections,
    )

    intersections = _find_intersections(valid_features)

    duplicate_segments = _find_duplicate_segments(valid_features)

    if duplicate_segments:
        topology_conflicts.extend(
            {
                "type": "duplicate_segment",
                "feature_ids": duplicate,
                "severity": "review_required",
            }
            for duplicate in duplicate_segments
        )

    crossing_intersections = [
        intersection
        for intersection in intersections
        if intersection["intersection_type"] == "crossing"
    ]

    if crossing_intersections:
        topology_conflicts.extend(
            {
                "type": "crossing_intersection",
                "feature_ids": item["feature_ids"],
                "coordinate": item["coordinate"],
                "severity": "review_required",
            }
            for item in crossing_intersections
        )

        warnings.append(
            f"{len(crossing_intersections)} crossing intersection(s) "
            "were detected. Crossings are not automatically interpreted "
            "as network connections."
        )

    priorities = _calculate_feature_priorities(
        valid_features=valid_features,
    )

    if endpoint_tolerance == 0.0:
        disconnected_endpoints = _count_unconnected_endpoints(
            valid_features=valid_features,
            endpoint_connections=endpoint_connections,
        )

        if disconnected_endpoints:
            warnings.append(
                f"{disconnected_endpoints} endpoint(s) have no exact "
                "connection to another hydrography endpoint."
            )

    components_output = [
        {
            "component_id": component["component_id"],
            "feature_ids": component["feature_ids"],
            "feature_count": len(component["feature_ids"]),
            "node_ids": component["node_ids"],
        }
        for component in components
    ]

    valid = (
        valid_feature_count > 0
        and not errors
    )

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "feature_count": len(geometry_values),
        "valid_feature_count": valid_feature_count,
        "invalid_feature_count": invalid_feature_count,
        "node_count": len(nodes),
        "edge_count": _count_network_edges(
            valid_features=valid_features,
            endpoint_connections=endpoint_connections,
        ),
        "component_count": len(components_output),
        "components": components_output,
        "endpoints": _serialise_endpoints(valid_features),
        "endpoint_connections": endpoint_connections,
        "intersections": intersections,
        "duplicate_segments": duplicate_segments,
        "topology_conflicts": topology_conflicts,
        "feature_priorities": priorities,
    }


def _validate_line_geometry(
    geometry: Any,
    feature_id: Any,
    position: int,
) -> dict[str, Any]:
    """
    Validate one geometry for topology preparation.

    :param geometry: Geometry-like object.
    :param feature_id: Identifier assigned to the feature.
    :param position: One-based input position.
    :returns: Validation result dictionary.
    """
    errors: list[str] = []

    if geometry is None:
        errors.append(
            f"Feature {feature_id} at position {position} is null."
        )
        return {
            "valid": False,
            "errors": errors,
        }

    if bool(getattr(geometry, "is_empty", False)):
        errors.append(
            f"Feature {feature_id} is empty."
        )
        return {
            "valid": False,
            "errors": errors,
        }

    if not bool(getattr(geometry, "is_valid", False)):
        errors.append(
            f"Feature {feature_id} is geometrically invalid."
        )
        return {
            "valid": False,
            "errors": errors,
        }

    geometry_type = str(
        getattr(geometry, "geom_type", "")
    ).lower()

    if geometry_type not in {
        "line",
        "linestring",
        "multilinestring",
    }:
        errors.append(
            f"Feature {feature_id} is not a line geometry: "
            f"{geometry_type or 'unknown'}."
        )
        return {
            "valid": False,
            "errors": errors,
        }

    try:
        line_parts = _extract_line_parts(geometry)
    except (AttributeError, TypeError):
        errors.append(
            f"Feature {feature_id} does not expose usable line "
            "coordinates."
        )
        return {
            "valid": False,
            "errors": errors,
        }

    if not line_parts:
        errors.append(
            f"Feature {feature_id} has no usable coordinates."
        )

    return {
        "valid": not errors,
        "errors": errors,
    }


def _extract_line_parts(
    geometry: Any,
) -> list[list[tuple[float, float]]]:
    """
    Extract coordinate sequences from LineString-like geometry.

    :param geometry: LineString or MultiLineString-like object.
    :returns: List of coordinate sequences.
    """
    geometry_type = str(
        getattr(geometry, "geom_type", "")
    ).lower()

    if geometry_type in {"line", "linestring"}:
        coordinates = list(geometry.coords)
        return [_normalise_coordinates(coordinates)]

    if geometry_type == "multilinestring":
        return [
            _normalise_coordinates(list(part.coords))
            for part in geometry.geoms
            if len(part.coords) >= 2
        ]

    return []


def _normalise_coordinates(
    coordinates: list[Any],
) -> list[tuple[float, float]]:
    """
    Convert coordinate values to two-dimensional float tuples.
    """
    normalised = []

    for coordinate in coordinates:
        if len(coordinate) < 2:
            continue

        x = float(coordinate[0])
        y = float(coordinate[1])

        if isfinite(x) and isfinite(y):
            normalised.append((x, y))

    return normalised


def _feature_endpoints(
    line_parts: list[list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    """
    Return the endpoints for a feature's line parts.
    """
    endpoints = []

    for part_index, coordinates in enumerate(line_parts):
        if len(coordinates) < 2:
            continue

        endpoints.append(
            {
                "part_index": part_index,
                "position": "start",
                "coordinate": [
                    float(coordinates[0][0]),
                    float(coordinates[0][1]),
                ],
            }
        )
        endpoints.append(
            {
                "part_index": part_index,
                "position": "end",
                "coordinate": [
                    float(coordinates[-1][0]),
                    float(coordinates[-1][1]),
                ],
            }
        )

    return endpoints


def _build_endpoint_nodes(
    valid_features: list[dict[str, Any]],
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Cluster feature endpoints into deterministic network nodes.

    :param valid_features: Valid feature records.
    :param tolerance: Endpoint clustering tolerance.
    :returns: Node records and endpoint connection records.
    """
    nodes: list[dict[str, Any]] = []
    endpoint_connections: list[dict[str, Any]] = []

    for feature in valid_features:
        feature_id = feature["feature_id"]

        for endpoint in feature["endpoints"]:
            coordinate = tuple(endpoint["coordinate"])
            node_id = _find_matching_node(
                nodes=nodes,
                coordinate=coordinate,
                tolerance=tolerance,
            )

            if node_id is None:
                node_id = len(nodes) + 1
                nodes.append(
                    {
                        "node_id": node_id,
                        "coordinate": [
                            float(coordinate[0]),
                            float(coordinate[1]),
                        ],
                        "endpoint_refs": [],
                    }
                )

            nodes[node_id - 1]["endpoint_refs"].append(
                {
                    "feature_id": feature_id,
                    "part_index": endpoint["part_index"],
                    "position": endpoint["position"],
                }
            )

            endpoint_connections.append(
                {
                    "feature_id": feature_id,
                    "part_index": endpoint["part_index"],
                    "position": endpoint["position"],
                    "node_id": node_id,
                }
            )

    return nodes, endpoint_connections


def _find_matching_node(
    nodes: list[dict[str, Any]],
    coordinate: tuple[float, float],
    tolerance: float,
) -> int | None:
    """
    Find the first existing node within tolerance.

    Nodes are examined in creation order, producing deterministic
    results for repeated runs.
    """
    for node in nodes:
        node_coordinate = node["coordinate"]
        distance = hypot(
            coordinate[0] - node_coordinate[0],
            coordinate[1] - node_coordinate[1],
        )

        if distance <= tolerance:
            return int(node["node_id"])

    return None


def _build_components(
    valid_features: list[dict[str, Any]],
    endpoint_connections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build connected feature components from shared endpoint nodes.
    """
    feature_ids = [
        feature["feature_id"]
        for feature in valid_features
    ]

    adjacency: dict[Any, set[Any]] = {
        feature_id: set()
        for feature_id in feature_ids
    }

    by_node: dict[int, list[Any]] = defaultdict(list)

    for connection in endpoint_connections:
        by_node[connection["node_id"]].append(
            connection["feature_id"]
        )

    for connected_features in by_node.values():
        unique_features = list(dict.fromkeys(connected_features))

        for feature_id in unique_features:
            adjacency[feature_id].update(
                other
                for other in unique_features
                if other != feature_id
            )

    components = []
    visited: set[Any] = set()

    for feature_id in sorted(
        feature_ids,
        key=_stable_identifier,
    ):
        if feature_id in visited:
            continue

        component_features = []
        queue = [feature_id]
        visited.add(feature_id)

        while queue:
            current = queue.pop(0)
            component_features.append(current)

            for neighbour in sorted(
                adjacency[current],
                key=_stable_identifier,
            ):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        component_features.sort(key=_stable_identifier)

        node_ids = sorted(
            {
                connection["node_id"]
                for connection in endpoint_connections
                if connection["feature_id"] in component_features
            }
        )

        components.append(
            {
                "component_id": len(components) + 1,
                "feature_ids": component_features,
                "node_ids": node_ids,
            }
        )

    return components


def _find_intersections(
    valid_features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Identify pairwise line intersections.

    Crossings are reported separately from endpoint connections.
    """
    intersections = []

    for first_index, first in enumerate(valid_features):
        for second in valid_features[first_index + 1:]:
            first_geometry = first["geometry"]
            second_geometry = second["geometry"]

            try:
                if not first_geometry.intersects(second_geometry):
                    continue

                intersection = first_geometry.intersection(
                    second_geometry
                )
            except (AttributeError, TypeError):
                continue

            if bool(getattr(intersection, "is_empty", True)):
                continue

            intersection_type = str(
                getattr(intersection, "geom_type", "")
            ).lower()

            coordinate = _representative_coordinate(
                intersection
            )

            intersections.append(
                {
                    "feature_ids": [
                        first["feature_id"],
                        second["feature_id"],
                    ],
                    "intersection_type": _intersection_type(
                        intersection_type
                    ),
                    "geometry_type": intersection_type,
                    "coordinate": coordinate,
                               }
            )

    return intersections


def _intersection_type(
    geometry_type: str,
) -> str:
    """
    Classify an intersection geometry.

    Point and multipoint intersections are treated as crossings.
    Line and multiline intersections are treated as overlaps.
    Other intersection types are reported as unknown.
    """
    if geometry_type in {"point", "multipoint"}:
        return "crossing"

    if geometry_type in {
        "line",
        "linestring",
        "multiline",
        "multilinestring",
    }:
        return "overlap"

    return "unknown"


def _representative_coordinate(
    geometry: Any,
) -> list[float] | None:
    """
    Return a representative coordinate for an intersection geometry.

    Point-like intersections use their x/y values. For other geometry
    types, the centroid is used where available.
    """
    geometry_type = str(
        getattr(geometry, "geom_type", "")
    ).lower()

    if geometry_type == "point":
        return [
            float(geometry.x),
            float(geometry.y),
        ]

    centroid = getattr(geometry, "centroid", None)

    if centroid is not None:
        try:
            return [
                float(centroid.x),
                float(centroid.y),
            ]
        except (AttributeError, TypeError, ValueError):
            return None

    return None


def _find_duplicate_segments(
    valid_features: list[dict[str, Any]],
) -> list[list[Any]]:
    """
    Identify duplicate or geometrically equivalent line features.

    The comparison uses the normalised coordinate sequence of each
    feature. A segment and its reversed representation are treated as
    duplicates.

    This function reports duplicates only. It does not remove or
    modify source features.
    """
    coordinate_index: dict[tuple, list[Any]] = defaultdict(list)

    for feature in valid_features:
        feature_id = feature["feature_id"]
        line_parts = feature["line_parts"]

        for part_index, coordinates in enumerate(line_parts):
            forward = tuple(
                (
                    round(float(x), 12),
                    round(float(y), 12),
                )
                for x, y in coordinates
            )

            reverse = tuple(reversed(forward))
            canonical = min(forward, reverse)

            coordinate_index[
                (part_index, canonical)
            ].append(feature_id)

    duplicates = []

    for feature_ids in coordinate_index.values():
        unique_feature_ids = list(dict.fromkeys(feature_ids))

        if len(unique_feature_ids) > 1:
            duplicates.append(
                sorted(
                    unique_feature_ids,
                    key=_stable_identifier,
                )
            )

    return duplicates


def _numeric_attribute(
    mapping: Mapping[Any, Any] | None,
    feature_id: Any,
) -> float | None:
    """
    Read a finite numeric feature attribute.

    Missing, non-numeric and non-finite values are returned as None.
    """
    if mapping is None:
        return None

    value = mapping.get(feature_id)

    if value is None:
        value = mapping.get(str(feature_id))

    if value is None:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(numeric_value):
        return None

    return numeric_value


def _calculate_feature_priorities(
    valid_features: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Calculate deterministic feature priorities.

    Priority is based on available stream order and upstream area:

        1. Higher stream order.
        2. Larger upstream area.
        3. Stable feature identifier as a tie-breaker.

    Missing attributes receive a value of zero and do not cause an
    exception. The result is a dictionary keyed by string feature ID
    for JSON serialisation.
    """
    priority_records = []

    for feature in valid_features:
        feature_id = feature["feature_id"]
        stream_order = feature["stream_order"]
        upstream_area = feature["upstream_area"]

        priority_records.append(
            {
                "feature_id": feature_id,
                "stream_order": (
                    float(stream_order)
                    if stream_order is not None
                    else 0.0
                ),
                "upstream_area": (
                    float(upstream_area)
                    if upstream_area is not None
                    else 0.0
                ),
            }
        )

    ordered = sorted(
        priority_records,
        key=lambda record: (
            -record["stream_order"],
            -record["upstream_area"],
            _stable_identifier(record["feature_id"]),
        ),
    )

    priorities = {}

    for rank, record in enumerate(ordered, start=1):
        feature_id = record["feature_id"]

        priorities[str(feature_id)] = {
            "feature_id": feature_id,
            "priority_rank": rank,
            "stream_order": record["stream_order"],
            "upstream_area": record["upstream_area"],
        }

    return priorities


def _count_unconnected_endpoints(
    valid_features: list[dict[str, Any]],
    endpoint_connections: list[dict[str, Any]],
) -> int:
    """
    Count endpoints that occur at nodes containing only one endpoint.

    A single endpoint is not necessarily an error: it may represent a
    legitimate source or outlet. The count is therefore used as a
    warning and not as an automatic rejection.
    """
    endpoint_counts: dict[int, int] = defaultdict(int)

    for connection in endpoint_connections:
        endpoint_counts[connection["node_id"]] += 1

    return sum(
        1
        for count in endpoint_counts.values()
        if count == 1
    )


def _count_network_edges(
    valid_features: list[dict[str, Any]],
    endpoint_connections: list[dict[str, Any]],
) -> int:
    """
    Count valid line features represented as network edges.

    Each valid feature contributes one logical edge. A MultiLineString
    remains one source feature but may contain multiple line parts;
    the current topology summary counts it as one feature edge.
    """
    del valid_features
    return len(
        {
            connection["feature_id"]
            for connection in endpoint_connections
        }
    )


def _serialise_endpoints(
    valid_features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return deterministic endpoint records suitable for JSON output.
    """
    serialised = []

    for feature in sorted(
        valid_features,
        key=lambda item: _stable_identifier(item["feature_id"]),
    ):
        for endpoint in feature["endpoints"]:
            serialised.append(
                {
                    "feature_id": feature["feature_id"],
                    "part_index": endpoint["part_index"],
                    "position": endpoint["position"],
                    "coordinate": [
                        float(endpoint["coordinate"][0]),
                        float(endpoint["coordinate"][1]),
                    ],
                }
            )

    return serialised


def _stable_identifier(
    value: Any,
) -> tuple[str, str]:
    """
    Return a deterministic sorting key for arbitrary feature IDs.

    Numeric identifiers sort numerically before non-numeric identifiers.
    """
    try:
        numeric_value = float(value)

        if isfinite(numeric_value):
            return "0", f"{numeric_value:030.12f}"
    except (TypeError, ValueError):
        pass

    return "1", str(value)

