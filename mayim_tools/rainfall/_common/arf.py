"""
Areal reduction factors (ARF) - zero QGIS dependency.

One place for every tool that turns point design rainfall into catchment
(areal) rainfall: Prepare HEC-HMS project, Prepare SWMM project and DDF to
Alternating Block Design Hyetographs (the SANRAL Deterministic Peak Flows
tool uses the same SANRAL equations from its own rainfall module).

Methods (ARF as a fraction 0-1; D the duration, A the area in km2):

- SANRAL Drainage Manual Eq 3.10 / 3.19 (Alexander's fit to the UK Flood
  Studies Report curves): ARF % = (90000 - 12800 ln A + 9830 ln(60 D_h))^0.4,
  limited to 100 %.
- Pietersen et al. (2024) regional ARFs (South Africa, regions 1-5) for
  24 h <= D <= 168 h, A <= 30 000 km2, T <= 200 years; Eq 3.10 is used
  outside that range (flagged).
- UK Flood Estimation Handbook (1999) Vol 2 / NERC (1975) FSR, Keers and
  Wescott (1977): ARF = 1 - b D_h^-a with a, b by area band (as in the UKFE
  R package of the UK Environment Agency).
- User table: ARF per duration (and optionally per area) from a CSV,
  interpolated linearly in ln D (and ln A); held at the end values outside
  the table (flagged).
- Constant factor, or none (the DDF is already areal, e.g. unadjusted
  gridded rainfall).

Design Rainfall SA (Smithers and Schulze 2003) and gauge DDFs are point
rainfall even where they come on a grid (each cell holds the point value
at that location), so they need an ARF; so does gridded rainfall adjusted
to a point DDF.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

METHOD_EQ310 = "eq310"
METHOD_PIETERSEN = "pietersen"
METHOD_FEH = "feh"
METHOD_TABLE = "table"
METHOD_CONSTANT = "constant"
METHOD_NONE = "none"

METHODS = [
    (METHOD_EQ310, "SANRAL Drainage Manual Eq 3.10 (UK FSR via Alexander)"),
    (
        METHOD_PIETERSEN,
        "Pietersen et al. (2024) regional ARF for D >= 24 h "
        "(South Africa; Eq 3.10 below)",
    ),
    (METHOD_FEH, "UK Flood Estimation Handbook (FSR, Keers and Wescott 1977)"),
    (METHOD_TABLE, "User ARF table (CSV)"),
    (METHOD_CONSTANT, "Constant factor"),
    (METHOD_NONE, "None (the DDF is already areal)"),
]
METHOD_LABELS = dict(METHODS)
SHORT_LABELS = {
    METHOD_EQ310: "SANRAL DM Eq 3.10",
    METHOD_PIETERSEN: "Pietersen et al. (2024)",
    METHOD_FEH: "UK FEH (Keers and Wescott 1977)",
    METHOD_TABLE: "User ARF table",
    METHOD_CONSTANT: "Constant",
    METHOD_NONE: "None",
}


class ArfError(ValueError):
    """An ARF input that cannot be used."""


# ---------------------------------------------------------------- equations


def arf_eq310(area_km2: float, duration_min: float) -> float:
    """SANRAL DM Eq 3.10 / 3.19 as a fraction (0-1)."""
    from mayim_tools.hydrology.sanral_peak_flows.rainfall import arf_eq310 as pct

    return pct(area_km2, duration_min / 60.0) / 100.0


def arf_pietersen(
    region: int, area_km2: float, duration_min: float, ari: float
) -> float:
    """Pietersen et al. (2024) Eq 1-2 as a fraction (no range checks)."""
    from mayim_tools.hydrology.sanral_peak_flows.rainfall import arf_pietersen as pct

    return pct(region, area_km2, duration_min / 60.0, ari) / 100.0


def arf_feh(area_km2: float, duration_min: float) -> float:
    """FEH Vol 2 / FSR ARF = 1 - b D^-a (D in hours, A in km2), Keers and
    Wescott (1977), limited to 0-1."""
    a_km2 = float(area_km2)
    d_h = float(duration_min) / 60.0
    ln_a = math.log(a_km2)
    if a_km2 <= 20:
        a = 0.4 - 0.0208 * math.log(4.6 - ln_a)
        b = 0.0394 * a_km2**0.354
    elif a_km2 < 100:
        a = 0.4 - 0.00382 * (4.6 - ln_a) ** 2
        b = 0.0394 * a_km2**0.354
    elif a_km2 < 500:
        a = 0.4 - 0.00382 * (4.6 - ln_a) ** 2
        b = 0.0627 * a_km2**0.254
    elif a_km2 < 1000:
        a = 0.4 - 0.0208 * math.log(ln_a - 4.6)
        b = 0.0627 * a_km2**0.254
    else:
        a = 0.4 - 0.0208 * math.log(ln_a - 4.6)
        b = 0.1050 * a_km2**0.180
    return max(0.0, min(1.0, 1.0 - b * d_h ** (-a)))


# ---------------------------------------------------------------- user table


@dataclass
class ArfTable:
    """ARF per duration (min) and optionally per area (km2)."""

    durations: list[float]
    areas: list[float] | None  # None: one ARF row for any area
    values: list[list[float]]  # [area index][duration index], fraction

    @classmethod
    def from_csv(cls, path: str) -> ArfTable:
        """CSV with a duration column (DurationMin, Duration_min, Duration),
        an ARF column (fraction or %) and optionally an area column
        (AreaKm2, Area_km2, Area). Durations may carry units ('1h')."""
        from mayim_tools.rainfall._common.ddf import parse_duration_to_minutes

        with open(path, encoding="utf-8-sig", newline="") as f:
            text = f.read()
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = [r for r in csv.reader(text.splitlines(), dialect) if any(r)]
        rows = [r for r in rows if not r[0].lstrip().startswith("#")]
        if len(rows) < 2:
            raise ArfError("The ARF table has no data rows.")
        low = [
            h.strip().lower().replace(" ", "").replace("(", "_").rstrip(")")
            for h in rows[0]
        ]

        def col(names):
            for n in names:
                if n in low:
                    return low.index(n)
            return None

        i_d = col(["durationmin", "duration_min", "duration", "d_min", "d"])
        i_f = col(["arf", "arf_pct", "arf%", "arf_%", "factor"])
        i_a = col(["areakm2", "area_km2", "area", "a_km2"])
        if i_d is None or i_f is None:
            raise ArfError(
                "The ARF table needs a duration column (DurationMin) and an ARF "
                "column (ARF); an area column (AreaKm2) is optional."
            )
        pts: dict[float, dict[float, float]] = {}
        for r in rows[1:]:
            r = r + [""] * (len(low) - len(r))
            try:
                d = float(parse_duration_to_minutes(r[i_d].strip()))
                v = float(r[i_f])
                a = float(r[i_a]) if i_a is not None and r[i_a].strip() else -1.0
            except (ValueError, TypeError):
                continue
            pts.setdefault(a, {})[d] = v
        if not pts:
            raise ArfError("The ARF table has no numeric rows.")
        allv = [v for row in pts.values() for v in row.values()]
        scale = 100.0 if max(allv) > 1.5 else 1.0
        durations = sorted({d for row in pts.values() for d in row})
        areas = sorted(pts)
        if any(a <= 0 for a in areas) and len(areas) > 1:
            raise ArfError("Give an area on every row of the ARF table, or on none.")
        values = []
        for a in areas:
            row = pts[a]
            missing = [d for d in durations if d not in row]
            if missing:
                raise ArfError(
                    f"The ARF table has no value for {missing[0]:g} min"
                    + (f" at {a:g} km2." if a > 0 else ".")
                )
            values.append([row[d] / scale for d in durations])
        if any(not 0 < v <= 1.0 + 1e-9 for vs in values for v in vs):
            raise ArfError("ARF values must lie between 0 and 1 (or 0 and 100 %).")
        return cls(durations, areas if areas[0] > 0 else None, values)

    @staticmethod
    def _interp(x: float, xs: list[float], ys: list[float]) -> tuple[float, bool]:
        if len(xs) == 1:
            return ys[0], not math.isclose(x, xs[0])
        if x <= xs[0]:
            return ys[0], x < xs[0] * (1 - 1e-9)
        if x >= xs[-1]:
            return ys[-1], x > xs[-1] * (1 + 1e-9)
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                w = (math.log(x) - math.log(xs[i])) / (
                    math.log(xs[i + 1]) - math.log(xs[i])
                )
                return ys[i] + w * (ys[i + 1] - ys[i]), False
        return ys[-1], True

    def factor(self, area_km2: float, duration_min: float) -> tuple[float, bool]:
        """(ARF, outside the table)."""
        by_area = []
        out = False
        for row in self.values:
            v, o = self._interp(duration_min, self.durations, row)
            by_area.append(v)
            out = out or o
        if self.areas is None:
            return by_area[0], out
        v, o = self._interp(area_km2, self.areas, by_area)
        return v, out or o


# ---------------------------------------------------------------- model


@dataclass
class ArfModel:
    """The ARF of one catchment: ``factor(duration_min, ari)`` -> fraction.
    Calling the model returns the factor only."""

    method: str = METHOD_NONE
    area_km2: float | None = None
    constant: float = 1.0
    region: int | None = None
    table: ArfTable | None = None
    area_note: str = ""
    flags: set = field(default_factory=set)

    def __post_init__(self):
        if self.method not in METHOD_LABELS:
            raise ArfError(f"Unknown ARF method '{self.method}'.")
        if self.needs_area and not (self.area_km2 and self.area_km2 > 0):
            raise ArfError("The ARF method needs the catchment area.")
        if self.method == METHOD_TABLE and self.table is None:
            raise ArfError("Give the ARF table (CSV).")
        if self.method == METHOD_CONSTANT and not 0 < self.constant <= 1.0:
            raise ArfError("The constant ARF must lie between 0 and 1.")
        if self.method == METHOD_PIETERSEN:
            from mayim_tools.hydrology.sanral_peak_flows.tables import PIETERSEN_ARF

            if self.region not in PIETERSEN_ARF:
                raise ArfError(
                    "Choose the Pietersen et al. (2024) ARF region (1 - 5) for "
                    "the Pietersen ARF."
                )

    @property
    def needs_area(self) -> bool:
        return self.method in (METHOD_EQ310, METHOD_PIETERSEN, METHOD_FEH) or (
            self.method == METHOD_TABLE
            and self.table is not None
            and self.table.areas is not None
        )

    @property
    def is_unity(self) -> bool:
        return self.method == METHOD_NONE or (
            self.method == METHOD_CONSTANT and self.constant == 1.0
        )

    @property
    def varies_with_duration(self) -> bool:
        return self.method not in (METHOD_NONE, METHOD_CONSTANT)

    def factor(self, duration_min: float, ari: float | None = None) -> float:
        m, a, d = self.method, self.area_km2, float(duration_min)
        if m == METHOD_NONE:
            return 1.0
        if m == METHOD_CONSTANT:
            return float(self.constant)
        if m == METHOD_EQ310:
            return arf_eq310(a, d)
        if m == METHOD_FEH:
            if a > 10000:
                self.flags.add("feh_area")
            return arf_feh(a, d)
        if m == METHOD_TABLE:
            v, out = self.table.factor(a or 0.0, d)
            if out:
                self.flags.add("table_extrap")
            return v
        # Pietersen, with Eq 3.10 outside its range
        t = float(ari) if ari else 2.0
        if 1440.0 - 1e-6 <= d <= 10080.0 + 1e-6 and a <= 30000 and t <= 200:
            return arf_pietersen(self.region, a, d, t)
        self.flags.add("pietersen_fallback")
        return arf_eq310(a, d)

    __call__ = factor

    def label(self) -> str:
        """'SANRAL DM Eq 3.10, A = 45.2 km2' etc."""
        m = self.method
        if m == METHOD_NONE:
            return "None (DDF already areal)"
        if m == METHOD_CONSTANT:
            return f"Constant {self.constant:g}"
        txt = SHORT_LABELS[m]
        if m == METHOD_PIETERSEN:
            txt += f", region {self.region} (Eq 3.10 below 24 h)"
        if self.area_km2:
            txt += f", A = {self.area_km2:.4g} km2"
        return txt

    def notes(self) -> list[str]:
        out = []
        if "pietersen_fallback" in self.flags:
            out.append(
                "Pietersen et al. (2024) applies to 24 h - 7 days, A <= 30000 km2 "
                "and T <= 200 years; SANRAL Eq 3.10 was used outside that range."
            )
        if "feh_area" in self.flags:
            out.append("The FEH ARF was derived for areas up to 10000 km2.")
        if "table_extrap" in self.flags:
            out.append(
                "Some durations / areas lie outside the ARF table; its end values "
                "were used."
            )
        return out


def burst_check(
    increments: list[float],
    interval_min: float,
    point_depth,
    arf,
    ari: float,
    durations: list[float],
) -> list[tuple[float, float, float, float]]:
    """For each duration d (a multiple of the step, shorter than the storm),
    (d, largest d-duration depth inside the storm, areal DDF depth at d
    (point x ARF), ratio %). ``point_depth(d, ari)`` gives the point DDF
    depth; ``arf(d, ari)`` the ARF."""
    n = len(increments)
    total_d = n * interval_min
    cum = [0.0]
    for v in increments:
        cum.append(cum[-1] + v)
    out = []
    for d in durations:
        k = d / interval_min
        if d >= total_d - 1e-9 or abs(k - round(k)) > 1e-6 or round(k) < 1:
            continue
        k = int(round(k))
        mx = max(cum[i + k] - cum[i] for i in range(n - k + 1))
        ref = point_depth(d, ari) * arf(d, ari)
        out.append((d, mx, ref, 100.0 * mx / ref if ref > 0 else float("nan")))
    return out
