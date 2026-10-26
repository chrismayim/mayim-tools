"""
Design storms for the HEC-HMS project - zero QGIS dependency.

Four rainfall sources, each giving one Storm per return period and duration:

- DDF table -> the HEC-HMS "Frequency Based Hypothetical" storm. HEC-HMS
  builds the nested (alternating block) storm itself from the depths at its
  standard durations, so only depths are written (no DSS records).
- Alternating Block hyetographs (the CSV of "DDF to Alternating Block Design
  Hyetographs") -> the incremental depths exactly as that tool wrote them.
- Huff curves (the curve CSV of "Precipitation data to Huff Curves") x the
  DDF depth for the storm duration.
- Design storm ensembles (the ensemble CSV of "Design Storm Ensembles") x the
  DDF depth; the AEP band follows from the return period.

The last three are written as precipitation gages (incremental depth time
series in a DSS file) and used with the "Specified Average" method.

DDF depths between table durations are interpolated log-log (a power law
per segment, as in the Alternating Block tool) and linearly in ln(T)
between return periods; below the shortest table duration the first
segment's power law is extended (flagged), as in that tool.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field

import numpy as np

# HEC-HMS 4.13 / 4.14 frequency storm durations (min) - the "Depth <d>:" keys.
HMS_FREQ_DURATIONS = [
    5,
    10,
    15,
    30,
    60,
    120,
    180,
    360,
    720,
    1440,
    2880,
    4320,
    5760,
    10080,
    14400,
]
PEAK_POSITIONS = [25, 33, 50, 67, 75]
# Alternating Block CSV labels its peak ratios 25/33/50/66/75 %.
AB_PEAK_LABEL = {25: "25", 33: "33", 50: "50", 67: "66", 75: "75"}
HUFF_STATS = ["P10", "P25", "P50", "P75", "P90", "Mean"]
SHAPES = ["Early", "Middle", "Late"]
# DSS standard intervals (min) a gage can use.
DSS_INTERVALS = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, 120, 180, 240, 360, 480]
DSS_INTERVALS += [720, 1440]

SOURCE_DDF = "ddf"
SOURCE_AB = "alternating_block"
SOURCE_HUFF = "huff"
SOURCE_ENSEMBLE = "ensemble"
SOURCE_LABELS = {
    SOURCE_DDF: "DDF - HEC-HMS frequency storm",
    SOURCE_AB: "Alternating Block hyetographs",
    SOURCE_HUFF: "Huff curves x DDF depth",
    SOURCE_ENSEMBLE: "Design storm ensembles x DDF depth",
}


class StormError(ValueError):
    """A rainfall input that cannot be used."""


@dataclass
class Storm:
    """One design storm. ``depths`` (HMS duration -> mm) for a frequency
    storm; ``increments`` (mm per ``interval_min``) for a hyetograph."""

    source: str
    ari: float
    duration_min: float
    name: str
    description: str
    peak_pct: float | None = None
    depths: dict[int, float] | None = None
    interval_min: float | None = None
    increments: list[float] | None = None
    pattern: str = ""
    flags: list[str] = field(default_factory=list)
    # areal reduction: {duration min: ARF} as applied (empty: none)
    arf_values: dict[float, float] = field(default_factory=dict)
    # burst check: [(duration min, max depth in storm, areal DDF depth, %)]
    burst: list = field(default_factory=list)

    @property
    def arf_total(self) -> float:
        """ARF at the storm duration (1 without reduction)."""
        if not self.arf_values:
            return 1.0
        k = min(self.arf_values, key=lambda d: abs(d - self.duration_min))
        return self.arf_values[k]

    @property
    def arf_peak(self) -> float:
        """ARF at the shortest reduced duration (the peak burst)."""
        if not self.arf_values:
            return 1.0
        return self.arf_values[min(self.arf_values)]

    @property
    def total_mm(self) -> float:
        if self.increments is not None:
            return float(sum(self.increments))
        if self.depths:
            d = [v for k, v in self.depths.items() if k <= self.duration_min]
            return float(max(d)) if d else 0.0
        return 0.0

    @property
    def peak_intensity_mmh(self) -> float | None:
        if self.increments and self.interval_min:
            return max(self.increments) * 60.0 / self.interval_min
        if self.depths:
            first = self.depths.get(min(self.depths))
            return first * 60.0 / min(self.depths) if first else None
        return None

    @property
    def is_hyetograph(self) -> bool:
        return self.increments is not None


# ---------------------------------------------------------------- labels


def duration_label(minutes: float) -> str:
    """'15min', '1h', '1.5h', '24h'."""
    m = float(minutes)
    if m < 60:
        return f"{m:g}min"
    return f"{m / 60:g}h"


def ari_label(ari: float) -> str:
    return f"1in{ari:g}"


def parse_number_list(text: str | None) -> list[float]:
    """'2, 5; 10 50' -> [2, 5, 10, 50]; blank -> []."""
    if not text or not str(text).strip():
        return []
    out = []
    for part in re.split(r"[,;\s]+", str(text).strip()):
        if part:
            try:
                out.append(float(part))
            except ValueError as e:
                raise StormError(f"'{part}' is not a number.") from e
    return out


def parse_duration_list(text: str | None) -> list[float]:
    """Durations in minutes, or with units: '60, 3h, 1440, 2 days'."""
    from mayim_tools.rainfall._common.ddf import parse_duration_to_minutes

    if not text or not str(text).strip():
        return []
    out = []
    joined = re.sub(r"(\d)\s+(?=[a-zA-Z])", r"\1", str(text).strip())
    for part in re.split(r"[,;\s]+", joined):
        if not part:
            continue
        try:
            out.append(float(parse_duration_to_minutes(part)))
        except ValueError as e:
            raise StormError(f"'{part}' is not a duration.") from e
    return out


def arf_function(arf):
    """``arf`` as a function (duration_min, ari) -> factor: a number (the
    same factor for every depth), an ArfModel, or any callable."""
    if callable(arf):
        return arf
    value = 1.0 if arf is None else float(arf)
    return lambda _d, _t=None: value


def _arf_text(values: dict[float, float]) -> str:
    """', ARF 0.912' or ', ARF 0.912 (6h) - 0.781 (5min)' for descriptions."""
    if not values or all(abs(v - 1.0) < 1e-12 for v in values.values()):
        return ""
    lo, hi = min(values), max(values)
    if abs(values[lo] - values[hi]) < 5e-4:
        return f", ARF {values[hi]:.3f}"
    return (
        f", ARF {values[hi]:.3f} ({duration_label(hi)}) - "
        f"{values[lo]:.3f} ({duration_label(lo)})"
    )


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------- DDF


class DdfCurves:
    """Depth at any duration and return period from a DDF table read with
    ``read_ddf_csv`` (central estimates)."""

    def __init__(self, long_df, info: dict | None = None):
        from mayim_tools.rainfall._common.ddf import DDFTable

        self.table = DDFTable(long_df)
        self.info = info or {}
        self.durations = [float(d) for d in self.table.durations]
        self.aris = [float(a) for a in self.table.aris]

    @classmethod
    def from_csv(cls, path: str, site: str | None = None):
        from mayim_tools.rainfall._common.ddf import read_ddf_csv

        long_df, info = read_ddf_csv(path, site=site or None)
        return cls(long_df, info)

    def _at_table_duration(self, d: float, ari: float) -> tuple[float, bool]:
        depth, flag = self.table.depth_at_ari(d, ari)
        return float(depth), flag != "interp"

    def depth(self, duration_min: float, ari: float) -> tuple[float, list[str]]:
        """(depth mm, flags). Flags: 'extrap_T' outside the table's return
        periods, 'extrap_short' below its shortest duration, 'extrap_long'
        above its longest."""
        flags: list[str] = []
        ds = self.durations
        t = float(duration_min)
        if len(ds) == 1:
            d0, ex = self._at_table_duration(ds[0], ari)
            if ex:
                flags.append("extrap_T")
            if not _close(t, ds[0]):
                raise StormError("The DDF table has only one duration.")
            return d0, flags
        if t < ds[0] or t > ds[-1]:
            pair = (ds[0], ds[1]) if t < ds[0] else (ds[-2], ds[-1])
            flags.append("extrap_short" if t < ds[0] else "extrap_long")
        else:
            i = int(np.searchsorted(ds, t, side="right")) - 1
            i = min(max(i, 0), len(ds) - 2)
            pair = (ds[i], ds[i + 1])
        t1, t2 = pair
        d1, e1 = self._at_table_duration(t1, ari)
        d2, e2 = self._at_table_duration(t2, ari)
        if e1 or e2:
            flags.append("extrap_T")
        if _close(t, t1):
            return d1, flags
        if _close(t, t2):
            return d2, flags
        if d1 <= 0 or d2 <= 0:
            depth = d1 + (d2 - d1) * (t - t1) / (t2 - t1)
        else:
            b = math.log(d2 / d1) / math.log(t2 / t1)
            depth = d1 * (t / t1) ** b
        return float(depth), flags


def frequency_storm(
    curves: DdfCurves,
    ari: float,
    duration_min: float,
    peak_pct: float,
    arf=1.0,
) -> Storm:
    """HEC-HMS frequency storm: depths at the HMS durations up to the storm
    duration (which must be one of them). ``arf`` (a number or an ArfModel)
    reduces every nested depth by the ARF of its own duration, so the storm
    follows the areal DDF curve."""
    f_arf = arf_function(arf)
    arfs: dict[float, float] = {}
    if not any(_close(duration_min, d) for d in HMS_FREQ_DURATIONS):
        raise StormError(
            f"A frequency storm duration must be one of the HEC-HMS durations "
            f"({', '.join(duration_label(d) for d in HMS_FREQ_DURATIONS)}); "
            f"{duration_label(duration_min)} is not."
        )
    depths: dict[int, float] = {}
    flags: list[str] = []
    for d in HMS_FREQ_DURATIONS:
        if d > duration_min + 1e-9:
            break
        v, f = curves.depth(d, ari)
        arfs[float(d)] = float(f_arf(d, ari))
        depths[d] = v * arfs[float(d)]
        flags.extend(f)
    # a depth-duration curve must not decrease
    running = 0.0
    for d in sorted(depths):
        if depths[d] < running:
            depths[d] = running
            flags.append("monotonic")
        running = depths[d]
    name = f"Freq {ari_label(ari)} {duration_label(duration_min)} P{peak_pct:g}"
    return Storm(
        source=SOURCE_DDF,
        ari=ari,
        duration_min=float(duration_min),
        name=name,
        description=(
            f"HEC-HMS frequency storm from the DDF table, {ari_label(ari)} year, "
            f"{duration_label(duration_min)}, peak at {peak_pct:g} %" + _arf_text(arfs)
        ),
        peak_pct=float(peak_pct),
        depths=depths,
        pattern=f"Nested, peak {peak_pct:g} %",
        flags=sorted(set(flags)),
        arf_values=arfs if any(abs(v - 1) > 1e-12 for v in arfs.values()) else {},
    )


def nested_cumulative(depths: dict, duration_min: float, peak_pct: float):
    """Cumulative depth curve of the nested (frequency) storm: the depth of
    every duration k is centred on the peak with ``peak_pct`` of it before
    the peak. Returns (times in min, cumulative mm)."""
    p = peak_pct / 100.0
    ks = sorted(k for k in depths if k <= duration_min + 1e-9)
    d_total = depths[ks[-1]]
    peak = p * duration_min
    pts = {0.0: 0.0, float(duration_min): d_total, peak: p * d_total}
    for k in ks:
        pts[peak - p * k] = p * (d_total - depths[k])
        pts[peak + (1 - p) * k] = p * d_total + (1 - p) * depths[k]
    t = sorted(pts)
    return t, [pts[x] for x in t]


# ---------------------------------------------------------------- CSV readers


def _read_rows(path: str) -> list[list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    # column-key lines ('# column: meaning') above the header are skipped
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [
        r for r in csv.reader(text.splitlines(), dialect) if any(c.strip() for c in r)
    ]


def _num(v) -> float | None:
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


_AB_COL = re.compile(r"^\s*(\d+)%_([\d.]+)min\s*$")


def read_alternating_block_csv(path: str) -> list[dict]:
    """Rows of the Alternating Block incremental depth CSV:
    [{ari, duration_min, interval_min, patterns: {'50': [mm...]}}]."""
    rows = _read_rows(path)
    if not rows:
        raise StormError("The Alternating Block CSV is empty.")
    header = [c.strip() for c in rows[0]]
    low = [h.lower() for h in header]
    try:
        i_t = low.index("returnperiodyears")
        i_d = low.index("targetdurationmin")
    except ValueError as e:
        raise StormError(
            "Not an Alternating Block hyetograph CSV (expected ReturnPeriodYears, "
            "TargetDurationMin and '50%_5min'-style columns)."
        ) from e
    cols: dict[str, list[tuple[int, float]]] = {}
    for j, h in enumerate(header):
        m = _AB_COL.match(h)
        if m:
            cols.setdefault(m.group(1), []).append((j, float(m.group(2))))
    if not cols:
        raise StormError("No '<peak>%_<time>min' columns in the Alternating Block CSV.")
    first = next(iter(cols.values()))
    interval = first[0][1]
    if "intensity" in " ".join(low):
        raise StormError("Give the incremental depth CSV, not the intensity CSV.")
    # rows reduced by the Alternating Block tool carry ArfMethod / ArfAreaKm2
    i_m = low.index("arfmethod") if "arfmethod" in low else None
    out = []
    for r in rows[1:]:
        r = r + [""] * (len(header) - len(r))
        t, dur = _num(r[i_t]), _num(r[i_d])
        if t is None or dur is None:
            continue
        pats = {}
        for key, idx in cols.items():
            vals = [_num(r[j]) for j, _ in idx]
            vals = [v for v in vals if v is not None]
            if vals:
                pats[key] = vals
        out.append(
            {
                "ari": t,
                "duration_min": dur,
                "interval_min": interval,
                "patterns": pats,
                "arf_method": r[i_m].strip() if i_m is not None else "",
            }
        )
    if not out:
        raise StormError("The Alternating Block CSV has no data rows.")
    return out


def read_huff_csv(path: str) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """{(quartile, stat): (x 0-1, cumulative fraction 0-1)} from the Huff
    curves CSV (Quartile, X, ..., Mean, P10 ... P90)."""
    rows = _read_rows(path)
    if not rows:
        raise StormError("The Huff curve CSV is empty.")
    header = [c.strip() for c in rows[0]]
    low = [h.lower() for h in header]
    if "quartile" not in low or "x" not in low:
        raise StormError(
            "Not a Huff curve CSV (expected Quartile, X, Mean and P10-P90 columns); "
            "give the 'Huff curves' output of Precipitation data to Huff Curves."
        )
    iq, ix = low.index("quartile"), low.index("x")
    stats = {h: j for j, h in enumerate(header) if h in HUFF_STATS}
    data: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for r in rows[1:]:
        r = r + [""] * (len(header) - len(r))
        q = _num(r[iq])
        x = _num(r[ix])
        if q is None or x is None:
            continue
        for s, j in stats.items():
            y = _num(r[j])
            if y is not None:
                data.setdefault((int(q), s), []).append((x, y))
    out = {}
    for key, pts in data.items():
        pts.sort()
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        if x.max() > 1.5:  # percent
            x = x / 100.0
        if y.max() > 1.5:
            y = y / 100.0
        if x[0] > 0:
            x, y = np.r_[0.0, x], np.r_[0.0, y]
        if x[-1] < 1:
            x, y = np.r_[x, 1.0], np.r_[y, 1.0]
        y = np.maximum.accumulate(np.clip(y, 0.0, 1.0))
        out[key] = (x, y)
    if not out:
        raise StormError("The Huff curve CSV has no curves.")
    return out


def read_ensemble_csv(path: str) -> list[dict]:
    """Rows of the Design Storm Ensembles CSV:
    [{duration_min, interval_min, band, shape, increments_pct}]."""
    rows = _read_rows(path)
    if not rows:
        raise StormError("The ensemble CSV is empty.")
    header = [c.strip() for c in rows[0]]
    low = [h.lower() for h in header]
    need = ["duration", "timestep", "aep", "shape"]
    if not all(n in low for n in need):
        raise StormError(
            "Not a Design Storm Ensembles CSV (expected Duration, TimeStep, AEP, "
            "Shape and Increment_1 ... columns)."
        )
    idx = {n: low.index(n) for n in need}
    inc_cols = [j for j, h in enumerate(low) if h.startswith("increment_")]
    out = []
    for r in rows[1:]:
        r = r + [""] * (len(header) - len(r))
        d, ts = _num(r[idx["duration"]]), _num(r[idx["timestep"]])
        if d is None or ts is None:
            continue
        inc = [_num(r[j]) for j in inc_cols]
        inc = [v for v in inc if v is not None]
        out.append(
            {
                "duration_min": d,
                "interval_min": ts,
                "band": r[idx["aep"]].strip().lower(),
                "shape": r[idx["shape"]].strip().capitalize(),
                "increments_pct": inc,
            }
        )
    return out


# ---------------------------------------------------------------- hyetographs


def _check_interval(interval_min: float, what: str) -> None:
    if not any(_close(interval_min, v) for v in DSS_INTERVALS):
        raise StormError(
            f"{what}: a {interval_min:g} min time step is not a DSS interval "
            f"({', '.join(f'{v:g}' for v in DSS_INTERVALS)} min)."
        )


def alternating_block_storms(
    rows: list[dict],
    aris: list[float],
    durations: list[float],
    peak_pct,
    arf=1.0,
) -> tuple[list[Storm], list[str]]:
    """Storms from the Alternating Block CSV rows for the chosen peak
    position(s) (one value or a list); empty ``aris`` / ``durations`` take
    every row. ``arf`` (a number or an ArfModel) is applied per cumulative
    duration (see ``areal_alternating_block``); rows the Alternating Block
    tool already reduced (ARF columns) are used as they are."""
    f_arf = arf_function(arf)
    peaks = list(peak_pct) if isinstance(peak_pct, (list, tuple)) else [peak_pct]
    keys = [AB_PEAK_LABEL.get(int(round(p)), f"{p:g}") for p in peaks]
    warnings = []
    out = []
    for r in rows:
        if aris and not any(_close(r["ari"], a) for a in aris):
            continue
        if durations and not any(_close(r["duration_min"], d) for d in durations):
            continue
        _check_interval(r["interval_min"], "Alternating Block CSV")
        n = int(round(r["duration_min"] / r["interval_min"]))
        for key in keys:
            vals = r["patterns"].get(key)
            if not vals:
                warnings.append(
                    f"Alternating Block {ari_label(r['ari'])} "
                    f"{duration_label(r['duration_min'])}: no {key} % peak column."
                )
                continue
            vals = vals[: max(n, 1)]
            if r.get("arf_method"):
                arfs = {}
                note = f", areal ({r['arf_method']}) in the CSV"
            else:
                vals, arfs = areal_alternating_block(
                    vals, r["interval_min"], lambda d, t=r["ari"]: f_arf(d, t)
                )
                note = _arf_text(arfs)
            out.append(
                Storm(
                    source=SOURCE_AB,
                    ari=r["ari"],
                    duration_min=r["duration_min"],
                    name=(
                        f"AB {ari_label(r['ari'])} "
                        f"{duration_label(r['duration_min'])} P{key}"
                    ),
                    description=(
                        f"Alternating Block hyetograph, {ari_label(r['ari'])} year, "
                        f"{duration_label(r['duration_min'])}, peak at {key} %" + note
                    ),
                    peak_pct=float(key),
                    interval_min=r["interval_min"],
                    increments=vals,
                    pattern=f"Alternating block, peak {key} %",
                    arf_values=arfs,
                )
            )
    if (aris or durations) and not out:
        warnings.append(
            "No Alternating Block rows match the return periods / durations."
        )
    return out, warnings


def areal_alternating_block(
    increments: list[float], interval_min: float, arf
) -> tuple[list[float], dict[float, float]]:
    """Alternating block hyetograph reduced to areal rainfall.

    The k largest blocks of an alternating block storm add up to the point
    DDF depth for k x step. Each of those cumulative depths is multiplied by
    the ARF of its own duration, ``arf(k x step)``, the reduced curve is
    differenced back into blocks, and block k goes back to the position of
    the k-th largest original block (the peak position is kept). Returns
    (increments, {duration: ARF}); a constant ARF scales every block."""
    vals = [max(0.0, float(v)) for v in increments]
    n = len(vals)
    if n == 0:
        return vals, {}
    order = sorted(range(n), key=lambda i: -vals[i])
    arfs = {
        (k + 1) * interval_min: float(arf((k + 1) * interval_min)) for k in range(n)
    }
    if all(abs(v - 1.0) < 1e-12 for v in arfs.values()):
        return vals, {}
    cum, out, prev = 0.0, [0.0] * n, 0.0
    for k, i in enumerate(order):
        cum += vals[i]
        reduced = cum * arfs[(k + 1) * interval_min]
        out[i] = max(0.0, reduced - prev)
        prev = max(prev, reduced)
    return out, arfs


def huff_quartile_for(duration_min: float) -> int:
    """Huff (1967): first-quartile storms are mostly up to 6 h, second 6-12 h,
    third 12-24 h, fourth longer."""
    if duration_min <= 360:
        return 1
    if duration_min <= 720:
        return 2
    if duration_min <= 1440:
        return 3
    return 4


def _pattern_increments(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    t = np.arange(0, n + 1) / n
    cum = np.interp(t, x, y)
    cum = (cum - cum[0]) / (cum[-1] - cum[0]) if cum[-1] > cum[0] else t
    return np.clip(np.diff(cum), 0.0, None)


def huff_storms(
    curves: DdfCurves,
    huff: dict,
    aris: list[float],
    durations: list[float],
    interval_min: float,
    quartile: int | None,
    stat: str,
    arf=1.0,
) -> tuple[list[Storm], list[str]]:
    """Huff mass curve x DDF depth x ARF at the storm duration; ``quartile``
    None picks it by duration."""
    f_arf = arf_function(arf)
    _check_interval(interval_min, "Huff storm time step")
    out, warnings = [], []
    for d in durations:
        n = d / interval_min
        if abs(n - round(n)) > 1e-6 or round(n) < 2:
            raise StormError(
                f"Huff storm {duration_label(d)}: the duration must be a multiple "
                f"(at least 2x) of the {interval_min:g} min time step."
            )
        n = int(round(n))
        q = quartile or huff_quartile_for(d)
        curve = huff.get((q, stat))
        if curve is None:
            warnings.append(
                f"Huff curve Q{q} {stat} is not in the Huff CSV; "
                f"{duration_label(d)} skipped."
            )
            continue
        frac = _pattern_increments(curve[0], curve[1], n)
        for t in aris:
            depth, flags = curves.depth(d, t)
            a = float(f_arf(d, t))
            depth *= a
            arfs = {float(d): a} if abs(a - 1.0) > 1e-12 else {}
            out.append(
                Storm(
                    source=SOURCE_HUFF,
                    ari=t,
                    duration_min=d,
                    name=f"Huff Q{q} {ari_label(t)} {duration_label(d)}",
                    description=(
                        f"Huff quartile {q} {stat} curve x DDF depth "
                        f"{depth:.1f} mm, {ari_label(t)} year, {duration_label(d)}"
                        + _arf_text(arfs)
                    ),
                    interval_min=interval_min,
                    increments=list(frac * depth),
                    pattern=f"Huff Q{q} {stat}",
                    flags=flags,
                    arf_values=arfs,
                )
            )
    return out, warnings


def ensemble_band(ari: float) -> str:
    from mayim_tools.rainfall._common.ddf import aep_band

    return aep_band(100.0 / ari)


def ensemble_storms(
    curves: DdfCurves,
    rows: list[dict],
    aris: list[float],
    durations: list[float],
    shapes: list[str],
    arf=1.0,
) -> tuple[list[Storm], list[str]]:
    """Ensemble pattern (percent increments) for the return period's AEP band
    and each chosen shape x DDF depth x ARF at the storm duration."""
    f_arf = arf_function(arf)
    out, warnings = [], []
    for d in durations:
        for t in aris:
            band = ensemble_band(t)
            for shape in shapes:
                row = next(
                    (
                        r
                        for r in rows
                        if _close(r["duration_min"], d)
                        and r["band"] == band
                        and r["shape"] == shape
                    ),
                    None,
                )
                if row is None or not row["increments_pct"]:
                    warnings.append(
                        f"Ensemble {duration_label(d)} {band} {shape}: no pattern in "
                        "the ensemble CSV (bin left blank there); skipped."
                    )
                    continue
                _check_interval(row["interval_min"], "Ensemble time step")
                pct = np.clip(np.array(row["increments_pct"], dtype=float), 0, None)
                if pct.sum() <= 0:
                    continue
                depth, flags = curves.depth(d, t)
                a = float(f_arf(d, t))
                depth *= a
                arfs = {float(d): a} if abs(a - 1.0) > 1e-12 else {}
                out.append(
                    Storm(
                        source=SOURCE_ENSEMBLE,
                        ari=t,
                        duration_min=d,
                        name=f"Ens {ari_label(t)} {duration_label(d)} {shape}",
                        description=(
                            f"Design storm ensemble pattern ({band} AEP band, {shape}) "
                            f"x DDF depth {depth:.1f} mm, {ari_label(t)} year, "
                            f"{duration_label(d)}" + _arf_text(arfs)
                        ),
                        interval_min=row["interval_min"],
                        increments=list(pct / pct.sum() * depth),
                        pattern=f"Ensemble {band} {shape}",
                        flags=flags,
                        arf_values=arfs,
                    )
                )
    return out, warnings


# ---------------------------------------------------------------- burst check

BURST_DURATIONS = [15, 30, 60, 120, 180, 360, 720, 1440, 2880, 4320]


def add_burst_checks(storms: list[Storm], curves: DdfCurves | None, arf=1.0) -> None:
    """Fill ``storm.burst`` for every hyetograph: the largest 15 min, 30 min,
    1 h ... depth inside the storm as a % of the areal DDF depth (point DDF
    x ARF) for that duration. Near 100 %: the pattern is as intense as the
    DDF for this catchment; well below: the pattern is smoother (typical of
    gridded rainfall such as ERA5 on a small catchment). Frequency storms
    follow the areal DDF by construction (100 %)."""
    if curves is None:
        return
    from mayim_tools.rainfall._common.arf import burst_check

    f_arf = arf_function(arf)
    for st in storms:
        if not st.is_hyetograph or not st.interval_min:
            continue
        ari = st.ari
        try:
            st.burst = burst_check(
                st.increments,
                st.interval_min,
                lambda d, t: curves.depth(d, t)[0],
                f_arf,
                ari,
                BURST_DURATIONS,
            )
        except StormError:
            st.burst = []
