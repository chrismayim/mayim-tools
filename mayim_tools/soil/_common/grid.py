"""Area-of-interest checks and the output grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import SoilDataError

EARTH_RADIUS_KM = 6371.0088


BUFFER_CELLS = 2


def area_km2(bounds_ll: tuple[float, float, float, float]) -> float:
    """Area of a lon/lat box on a sphere (km2)."""
    lon0, lat0, lon1, lat1 = bounds_ll
    if lon1 <= lon0 or lat1 <= lat0:
        return 0.0
    dlon = math.radians(lon1 - lon0)
    band = math.sin(math.radians(lat1)) - math.sin(math.radians(lat0))
    return EARTH_RADIUS_KM**2 * dlon * band


def check_area(
    bounds_ll: tuple[float, float, float, float], max_area_km2: float
) -> float:
    """Return the area of interest in km2; raise if above the limit."""
    lon0, lat0, lon1, lat1 = bounds_ll
    if not (-180 <= lon0 < lon1 <= 180 and -90 <= lat0 < lat1 <= 90):
        raise SoilDataError(
            "The area of interest has an invalid or empty extent "
            f"(lon {lon0:.4f} to {lon1:.4f}, lat {lat0:.4f} to {lat1:.4f})."
        )
    area = area_km2(bounds_ll)
    if max_area_km2 > 0 and area > max_area_km2:
        raise SoilDataError(
            f"The area of interest is {area:.0f} km2 (bounding box), which is "
            f"above the maximum of {max_area_km2:.0f} km2. Reduce the area, or "
            "raise 'Maximum area per run (km2)' under Advanced parameters."
        )
    return area


@dataclass(frozen=True)
class TargetGrid:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    res: float
    crs_wkt: str

    @property
    def width(self) -> int:
        return int(round((self.xmax - self.xmin) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.ymax - self.ymin) / self.res))

    @property
    def geotransform(self) -> tuple[float, float, float, float, float, float]:
        return (self.xmin, self.res, 0.0, self.ymax, 0.0, -self.res)


def make_grid(
    bounds: tuple[float, float, float, float],
    res: float,
    crs_wkt: str,
    buffer_cells: int = BUFFER_CELLS,
) -> TargetGrid:
    """Output grid: AOI bounds buffered by ``buffer_cells`` and snapped
    outward to whole multiples of ``res`` (stable alignment between runs)."""
    if res <= 0:
        raise SoilDataError("Output resolution must be greater than zero.")
    xmin, ymin, xmax, ymax = bounds
    if xmax <= xmin or ymax <= ymin:
        raise SoilDataError("The area of interest has an empty extent.")
    pad = buffer_cells * res
    xmin = math.floor((xmin - pad) / res) * res
    ymin = math.floor((ymin - pad) / res) * res
    xmax = math.ceil((xmax + pad) / res) * res
    ymax = math.ceil((ymax + pad) / res) * res
    return TargetGrid(xmin, ymin, xmax, ymax, res, crs_wkt)


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
