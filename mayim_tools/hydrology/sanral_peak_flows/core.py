"""SANRAL Deterministic Peak Flows - orchestration (no QGIS dependency).

One CatchmentInput per catchment feature; each is an independent design
point (its own area and Tc, also for the ARF). The methods are in
methods.py, the rainfall and ARF in rainfall.py and the transcribed tables
in tables.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import tables as tb
from .methods import (
    METHOD_ORDER,
    MethodResult,
    RunoffCoefficients,
    capa,
    midgley_pitman,
    rational,
    rmf,
    rmf_ratios,
    sdf,
)
from .rainfall import TR102Station, basin_station

TOOL_VERSION = "0.2.0"


@dataclass
class CatchmentInput:
    catchment_id: str
    area_km2: float
    lfp_km: float | None = None
    s1085: float | None = None  # m/m
    lca_km: float | None = None
    s_mean: float | None = None  # mean catchment slope, m/m
    tc_min: float | None = None  # Tc for the Rational method
    ineffective_km2: float = 0.0
    mode: str | None = None  # Catchment Delineation 'total' / 'incremental'
    map_mm: float | None = None
    sdf_basin: int | None = None
    veld_zone: str | None = None
    kovacs_region: str | None = None
    arf_region: int | None = None

    @property
    def effective_area(self) -> float:
        """Ae = gross area less ineffective areas (pans, marshes): TR137 p 7,
        DM Section 3.5.1.5. Used by every method."""
        return self.area_km2 - (self.ineffective_km2 or 0.0)


@dataclass
class Settings:
    methods: tuple = METHOD_ORDER
    return_periods: tuple = tb.RETURN_PERIODS
    tc_floor: bool = True
    apply_arf: bool = True
    arf_method: str = "eq310"  # or "pietersen"
    k_custom: float | None = None
    r_days: float | None = None  # thunder days override for Alternative 2


@dataclass
class CatchmentResult:
    catchment: CatchmentInput
    results: list = field(default_factory=list)  # MethodResult
    notes: list = field(default_factory=list)  # catchment-level notes

    def by_field(self) -> dict:
        return {r.field_name: r.q_m3s for r in self.results}

    def notes_text(self) -> str:
        items = list(self.notes)
        for r in self.results:
            for f in r.flags:
                tag = f"{r.code}: {f}"
                if tag not in items:
                    items.append(tag)
        return "; ".join(items)


@dataclass
class RunResult:
    items: list
    warnings: list
    settings: Settings
    ddf_info: dict | None = None


def _skip(cr: CatchmentResult, method: str, reason: str) -> None:
    cr.notes.append(f"{method}: not calculated - {reason}")


def run_catchment(
    c: CatchmentInput,
    s: Settings,
    *,
    ddf=None,
    c_values: dict[str, RunoffCoefficients] | None = None,
    tr102_override: TR102Station | None = None,
) -> CatchmentResult:
    cr = CatchmentResult(c)
    a = c.effective_area
    if a <= 0:
        cr.notes.append("effective area <= 0: nothing calculated")
        return cr
    if c.ineffective_km2:
        cr.notes.append(f"Ae = {a:.3f} km2 ({c.ineffective_km2:.3f} km2 ineffective)")
    if (c.mode or "").lower().startswith("incr"):
        cr.notes.append(
            "incremental catchment: peaks are for its local area only, not the "
            "flow at its outlet from the whole upstream catchment"
        )
    rps = tuple(s.return_periods)
    wanted = set(s.methods)

    station = None
    if c.sdf_basin is not None:
        station = basin_station(c.sdf_basin, s.r_days)

    # Rational
    for alt, mid in ((2, "rational_alt2"), (3, "rational_alt3")):
        if mid not in wanted:
            continue
        coeffs = (c_values or {}).get(c.catchment_id)
        if coeffs is None:
            _skip(cr, mid, "no row in the C CSV")
            continue
        if c.tc_min is None or c.tc_min <= 0:
            _skip(cr, mid, "no Tc")
            continue
        rain_station = tr102_override or station
        if alt == 2 and rain_station is None:
            _skip(cr, mid, "no SDF basin or TR102 station")
            continue
        if alt == 3 and ddf is None:
            _skip(cr, mid, "no DDF table")
            continue
        cr.results += rational(
            alt,
            a,
            c.tc_min / 60.0,
            coeffs,
            rps,
            tc_floor=s.tc_floor,
            apply_arf=s.apply_arf,
            arf_method=s.arf_method,
            arf_region=c.arf_region,
            ddf=ddf,
            station=rain_station,
        )

    if "sdf" in wanted:
        if station is None:
            _skip(cr, "sdf", "no SDF basin")
        elif not (c.lfp_km and c.s1085 and c.s1085 > 0):
            _skip(cr, "sdf", "no watercourse length or 10-85 slope")
        else:
            cr.results += sdf(a, c.lfp_km, c.s1085, c.sdf_basin, station, rps)

    if "midgley_pitman" in wanted:
        missing = [
            n
            for n, v in (
                ("MAP", c.map_mm),
                ("L", c.lfp_km),
                ("Lc", c.lca_km),
                ("10-85 slope", c.s1085),
                ("veld zone", c.veld_zone),
            )
            if not v
        ]
        if missing:
            _skip(cr, "midgley_pitman", "missing " + ", ".join(missing))
        else:
            cr.results += [
                r
                for r in midgley_pitman(
                    a, c.map_mm, c.lfp_km, c.lca_km, c.s1085, c.veld_zone
                )
                if r.return_period in rps
            ]

    if wanted & {"rmf", "rmf_ratio"}:
        if c.kovacs_region is None and s.k_custom is None:
            _skip(cr, "rmf", "no Kovacs region or K")
        else:
            q = rmf(a, c.kovacs_region, s.k_custom)
            if "rmf" in wanted:
                cr.results.append(q)
            if "rmf_ratio" in wanted:
                if s.k_custom is not None:
                    _skip(cr, "rmf_ratio", "Table 3D.1 ratios need a Kovacs region")
                else:
                    cr.results += rmf_ratios(a, c.kovacs_region, q.q_m3s, rps)

    if "capa" in wanted:
        missing = [
            n
            for n, v in (("MAP", c.map_mm), ("L", c.lfp_km), ("mean slope", c.s_mean))
            if not v
        ]
        if missing:
            _skip(cr, "capa", "missing " + ", ".join(missing))
        else:
            cr.results += capa(a, c.map_mm, c.s_mean, c.lfp_km, rps)

    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    cr.results.sort(key=lambda r: (order[r.method], r.return_period or 10**6))
    return cr


def run(
    catchments,
    settings: Settings,
    *,
    ddf=None,
    ddf_info: dict | None = None,
    c_values=None,
    tr102_override=None,
    feedback=None,
) -> RunResult:
    items, warnings = [], []
    n = len(catchments)
    for i, c in enumerate(catchments):
        if feedback is not None:
            if feedback.isCanceled():
                break
            feedback.setProgress(100.0 * i / max(n, 1))
        items.append(
            run_catchment(
                c, settings, ddf=ddf, c_values=c_values, tr102_override=tr102_override
            )
        )
    if c_values is not None:
        unused = set(c_values) - {c.catchment_id for c in catchments}
        if unused:
            warnings.append(
                "C CSV rows with no matching catchment: " + ", ".join(sorted(unused))
            )
    return RunResult(items, warnings, settings, ddf_info)


def field_names(settings: Settings) -> list[str]:
    """Output-layer field names for the chosen methods and return periods."""
    from .methods import METHODS

    rps = settings.return_periods
    allowed = {
        "rational_alt2": rps,
        "rational_alt3": rps,
        "sdf": rps,
        "midgley_pitman": [T for T in rps if T in tb.MP_RETURN_PERIODS],
        "rmf": [None],
        "rmf_ratio": [T for T in rps if T in tb.RMF_RATIO_RETURN_PERIODS],
        "capa": [T for T in rps if T == 2 or T in tb.CAPA_KP_RETURN_PERIODS],
    }
    names = []
    for m in METHOD_ORDER:
        if m not in settings.methods:
            continue
        code = METHODS[m][0]
        for T in allowed[m]:
            names.append(f"Q_{code}" if T is None else f"Q_{code}_{T}")
    return names


__all__ = [
    "CatchmentInput",
    "CatchmentResult",
    "MethodResult",
    "RunResult",
    "RunoffCoefficients",
    "Settings",
    "TOOL_VERSION",
    "field_names",
    "run",
    "run_catchment",
]
