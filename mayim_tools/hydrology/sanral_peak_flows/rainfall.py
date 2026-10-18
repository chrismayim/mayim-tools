"""Point rainfall and areal reduction for the SANRAL methods (no QGIS
dependency).

- Modified Hershfield equation (DM Eq 3.9 / 3.18) with TR102 n-day depths,
  following DM Step 6 (p 3-46) and Section 3.5.1.3.2 (p 3-23).
- Areal reduction factor: DM Eq 3.10 / 3.19 (UK FSR via Alexander) and,
  optionally, Pietersen et al (2024) Equations 1-2 for durations >= 24 h.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .tables import PIETERSEN_ARF, RETURN_PERIODS, SDF_BASINS, TR102_BASIN_STATIONS

TR102_DAYS = (1, 2, 3, 7)


@dataclass
class TR102Station:
    """Representative TR102 station: n-day depths per return period, plus
    M (2-year 1-day depth) and R (days per year with thunder)."""

    label: str
    depths: dict  # {days: {T: mm}}
    m_mm: float
    r_days: float
    source: str

    def depth(self, days: int, T: int) -> float:
        return float(self.depths[days][T])


def basin_station(basin: int, r_override: float | None = None) -> TR102Station:
    """The SDF basin's representative station (DM Tables 3B.1 and 3B.2)."""
    if basin not in SDF_BASINS:
        raise ValueError(f"SDF basin {basin} is not 1 - 29.")
    info = SDF_BASINS[basin]
    row = TR102_BASIN_STATIONS[basin]
    depths = {
        d: dict(zip(RETURN_PERIODS, row["depths"][d], strict=True)) for d in TR102_DAYS
    }
    return TR102Station(
        label=f"SDF basin {basin}: {row['station']} {row['name']}",
        depths=depths,
        m_mm=float(info[2]),
        r_days=float(r_override if r_override is not None else info[3]),
        source="DM Tables 3B.1 and 3B.2",
    )


def hershfield_depth(t_min: float, T: float, m_mm: float, r_days: float) -> float:
    """Modified (recalibrated) Hershfield point depth, DM Eq 3.9 / 3.18:
    P = 1.13 (0.41 + 0.64 ln T)(-0.11 + 0.27 ln t)(0.79 M^0.69 R^0.20),
    t in minutes. Reproduces AG examples 3.1 and 3.2 exactly."""
    return (
        1.13
        * (0.41 + 0.64 * math.log(T))
        * (-0.11 + 0.27 * math.log(t_min))
        * (0.79 * m_mm**0.69 * r_days**0.20)
    )


def _interp(x, x0, x1, y0, y1):
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


def tr102_point_depth(
    tc_h: float, T: int, station: TR102Station, cap_at_1day: bool
) -> tuple[float, str, list[str]]:
    """Point depth for duration Tc (hours) from Hershfield and TR102.

    <= 6 h: Hershfield; 6 - 24 h: linear between Hershfield (6 h) and the
    TR102 1-day depth; > 24 h: linear between TR102 n-day depths (1, 2, 3,
    7 days), held at the 7-day depth beyond 7 days (flagged).
    cap_at_1day (Rational Alternative 2, DM p 3-23): a Hershfield value above
    the TR102 1-day depth is reduced to it when Tc < 24 h."""
    flags: list[str] = []
    t_min = tc_h * 60.0
    one_day = station.depth(1, T)
    if tc_h <= 6.0:
        depth = hershfield_depth(t_min, T, station.m_mm, station.r_days)
        how = "Hershfield"
    elif tc_h <= 24.0:
        h6 = hershfield_depth(360.0, T, station.m_mm, station.r_days)
        depth = _interp(tc_h, 6.0, 24.0, h6, one_day)
        how = "Hershfield 6 h - TR102 1 day interpolation"
    else:
        hours = [d * 24.0 for d in TR102_DAYS]
        values = [station.depth(d, T) for d in TR102_DAYS]
        if tc_h >= hours[-1]:
            depth = values[-1]
            how = "TR102 7 day"
            if tc_h > hours[-1]:
                flags.append("Tc exceeds 7 days: TR102 7-day depth used")
        else:
            i = max(j for j in range(len(hours)) if hours[j] <= tc_h)
            depth = _interp(tc_h, hours[i], hours[i + 1], values[i], values[i + 1])
            how = f"TR102 {TR102_DAYS[i]} - {TR102_DAYS[i + 1]} day interpolation"
    if cap_at_1day and tc_h < 24.0 and depth > one_day:
        flags.append(
            f"Hershfield depth {depth:.1f} mm reduced to the TR102 1-day "
            f"depth {one_day:.1f} mm (DM p 3-23)"
        )
        depth = one_day
        how += ", capped at 1 day"
    return depth, how, flags


def arf_eq310(area_km2: float, duration_h: float) -> float:
    """DM Eq 3.10 / 3.19: ARF (%) = (90000 - 12800 ln A + 9830 ln(60 Tc))^0.4,
    limited to 100 %."""
    base = 90000.0 - 12800.0 * math.log(area_km2) + 9830.0 * math.log(60.0 * duration_h)
    if base <= 0:
        return 0.0
    return min(100.0, base**0.4)


def arf_pietersen(region: int, area_km2: float, duration_h: float, T: float) -> float:
    """Pietersen et al (2024) Eq 1-2, ARF (%) limited to 0 - 100 %. Valid for
    24 <= D <= 168 h, A <= 30 000 km2 and T <= 200 years."""
    a, b, c, x1, x2, x3, x4, x5, x6, x7 = PIETERSEN_ARF[region]
    lg = math.log10
    ld, lt, la = lg(duration_h / 24.0), lg(T), lg(area_km2)
    x = x1 * ld**2 + x2 * ld - x3 * lt**2 + x4 * lt - x5 * la**2 - x6 * la + x7
    return max(0.0, min(100.0, a * x * x + b * x - c))


def areal_reduction(
    method: str,
    area_km2: float,
    duration_h: float,
    T: int,
    region: int | None,
) -> tuple[float, str, list[str]]:
    """ARF (%) by the chosen method, with the Eq 3.10 fallback outside the
    Pietersen range. Returns (arf, method label, flags)."""
    flags: list[str] = []
    if method == "pietersen":
        problems = []
        if duration_h < 24.0:
            problems.append("D < 24 h")
        if duration_h > 168.0:
            problems.append("D > 168 h")
        if area_km2 > 30000.0:
            problems.append("A > 30 000 km2")
        if region not in PIETERSEN_ARF:
            problems.append("no ARF region")
        if not problems:
            return (
                arf_pietersen(region, area_km2, duration_h, T),
                f"Pietersen et al (2024), region {region}",
                flags,
            )
        flags.append(
            "Pietersen ARF not applicable (" + ", ".join(problems) + "): Eq 3.10 used"
        )
    return arf_eq310(area_km2, duration_h), "DM Eq 3.10", flags
