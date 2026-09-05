"""
Mayim Tools - Hydrography Validation
====================================

Stage 7A native validation for vector hydrography supplied for optional
DEM hydrography enforcement.

This module performs validation only. It does not modify the DEM or
hydrography and does not perform stream burning.

Checks include:

    - Null and empty geometries.
    - Geometry validity.
    - Line geometry type.
    - CRS availability and equality.
    - Intersection with the DEM extent.

IP status
---------
Original Mayim validation implementation using Python standard-library
components and Shapely geometry objects supplied by the caller.

No WhiteboxTools, RichDEM, TauDEM or other third-party hydrological
implementation is used.

The validator is based on the Stage 7 requirements of the revised
Mayim Tools DEM Hydrological Conditioning methodology.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def validate_hydrography(
    geometries: Iterable[Any],
    dem_crs: Any,
    hydrography_crs: Any,
    dem_bounds: Any,
) -> dict[str, Any]:
    """
    Validate vector hydrography against a DEM extent and CRS.

    This function does not reproject, repair, clip or modify any
    geometry. It reports issues for the calling tool or analyst to
    address.

    Parameters
    ----------
    geometries:
        Iterable of geometry objects. Shapely-like objects are expected
        to provide ``is_empty``, ``is_valid``, ``geom_type`` and
        ``intersects()`` attributes or methods.

    dem_crs:
        DEM CRS object or CRS identifier.

    hydrography_crs:
        Hydrography CRS object or CRS identifier.

    dem_bounds:
        DEM extent object. It must either provide ``left``, ``bottom``,
        ``right`` and ``top`` attributes, or be a four-item sequence in
        that order.

    Returns
    -------
    dict
        Validation report containing validity, warnings, errors and
        feature counts.

    Notes
    -----
    A CRS comparison is only a preliminary equality check. It does not
    prove that two independently supplied CRS labels are correct.
    """
    errors: list[str] = []
    warnings: list[str] = []

    geometry_count = 0
    invalid_geometry_count = 0
    empty_geometry_count = 0
    non_line_count = 0
    outside_dem_count = 0
    intersecting_dem_count = 0

    if geometries is None:
        errors.append("Hydrography geometries are missing.")
        geometry_values: Iterable[Any] = []
    else:
        try:
            geometry_values = list(geometries)
        except TypeError as error:
            errors.append(f"Hydrography geometries are not iterable: {error}")
            geometry_values = []

    bounds = _normalise_bounds(dem_bounds)

    if bounds is None:
        errors.append("DEM bounds could not be interpreted.")

    crs_match = _crs_equal(
        dem_crs,
        hydrography_crs,
    )

    if dem_crs is None:
        errors.append("DEM CRS is missing.")

    if hydrography_crs is None:
        errors.append("Hydrography CRS is missing.")

    if dem_crs is not None and hydrography_crs is not None and not crs_match:
        errors.append(
            "DEM CRS and hydrography CRS do not match. "
            "Reprojection is required before enforcement."
        )

    for geometry in geometry_values:
        geometry_count += 1

        if geometry is None:
            invalid_geometry_count += 1
            errors.append(f"Hydrography feature {geometry_count} is null.")
            continue

        if bool(getattr(geometry, "is_empty", False)):
            empty_geometry_count += 1
            invalid_geometry_count += 1
            errors.append(f"Hydrography feature {geometry_count} is empty.")
            continue

        if not bool(getattr(geometry, "is_valid", False)):
            invalid_geometry_count += 1
            errors.append(f"Hydrography feature {geometry_count} is invalid.")
            continue

        geometry_type = str(getattr(geometry, "geom_type", "")).lower()

        if geometry_type not in {
            "line",
            "linestring",
            "multilinestring",
        }:
            non_line_count += 1
            errors.append(
                f"Hydrography feature {geometry_count} is not a "
                f"line geometry: {geometry_type or 'unknown'}."
            )
            continue

        if bounds is not None:
            intersects = _geometry_intersects_bounds(
                geometry,
                bounds,
            )

            if intersects:
                intersecting_dem_count += 1
            else:
                outside_dem_count += 1
                warnings.append(
                    f"Hydrography feature {geometry_count} does not "
                    "intersect the DEM extent."
                )

    if geometry_count == 0:
        warnings.append("No hydrography geometries were supplied.")

    if outside_dem_count > 0:
        warnings.append(
            f"{outside_dem_count} hydrography feature(s) lie " "outside the DEM extent."
        )

    if geometry_count > 0 and intersecting_dem_count == 0:
        warnings.append("No hydrography features intersect the DEM extent.")

    valid = not errors and geometry_count > 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "geometry_count": geometry_count,
        "invalid_geometry_count": invalid_geometry_count,
        "empty_geometry_count": empty_geometry_count,
        "non_line_count": non_line_count,
        "outside_dem_count": outside_dem_count,
        "intersecting_dem_count": intersecting_dem_count,
        "crs_match": crs_match,
    }


def _normalise_bounds(
    dem_bounds: Any,
) -> tuple[float, float, float, float] | None:
    """
    Convert DEM bounds to left, bottom, right, top order.

    :param dem_bounds: Bounds object or four-item sequence.
    :returns: Normalised bounds tuple, or None if invalid.
    """
    if dem_bounds is None:
        return None

    try:
        values = (
            float(dem_bounds.left),
            float(dem_bounds.bottom),
            float(dem_bounds.right),
            float(dem_bounds.top),
        )
    except (AttributeError, TypeError, ValueError):
        try:
            if len(dem_bounds) != 4:
                return None

            values = tuple(float(value) for value in dem_bounds)
        except (TypeError, ValueError):
            return None

    left, bottom, right, top = values

    if left > right or bottom > top:
        return None

    return left, bottom, right, top


def _crs_equal(
    dem_crs: Any,
    hydrography_crs: Any,
) -> bool:
    """
    Compare two CRS values conservatively.

    Where CRS objects provide ``isGeographic`` or ``authid`` methods,
    their string representation is used as a stable comparison value.
    """
    if dem_crs is None or hydrography_crs is None:
        return False

    if dem_crs == hydrography_crs:
        return True

    dem_value = _crs_value(dem_crs)
    hydrography_value = _crs_value(hydrography_crs)

    return dem_value == hydrography_value


def _crs_value(crs: Any) -> str:
    """
    Obtain a comparable CRS representation.
    """
    for method_name in (
        "authid",
        "to_string",
        "toWkt",
    ):
        method = getattr(crs, method_name, None)

        if callable(method):
            try:
                value = method()
                if value:
                    return str(value)
            except Exception:  # noqa: BLE001, S112
                continue

    return str(crs)


def _geometry_intersects_bounds(
    geometry: Any,
    bounds: tuple[float, float, float, float],
) -> bool:
    """
    Test geometry intersection with a rectangular DEM extent.

    Shapely is used only through the geometry object's public
    intersection interface supplied by the caller.
    """
    left, bottom, right, top = bounds

    try:
        from shapely.geometry import box

        return bool(geometry.intersects(box(left, bottom, right, top)))
    except ImportError as error:
        raise RuntimeError(
            "Shapely is required for geometry-boundary validation."
        ) from error
