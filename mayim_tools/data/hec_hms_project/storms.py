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
    arf: float = 1.0,
) -> Storm:
    """HEC-HMS frequency storm: depths at the HMS durations up to the storm
    duration (which must be one of them)."""
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
        depths[d] = v * arf
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
            f"{duration_label(duration_min)}, peak at {peak_pct:g} %"
            + (f", ARF {arf:g}" if arf != 1.0 else "")
        ),
        peak_pct=float(peak_pct),
        depths=depths,
        pattern=f"Nested, peak {peak_pct:g} %",
        flags=sorted(set(flags)),
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
            {"ari": t, "duration_min": dur, "interval_min": interval, "patterns": pats}
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
    arf: float = 1.0,
) -> tuple[list[Storm], list[str]]:
    """Storms from the Alternating Block CSV rows for the chosen peak
    position(s) (one value or a list); empty ``aris`` / ``durations`` take
    every row."""
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
            vals = [v * arf for v in vals[: max(n, 1)]]
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
                        f"{duration_label(r['duration_min'])}, peak at {key} %"
                        + (f", ARF {arf:g}" if arf != 1.0 else "")
                    ),
                    peak_pct=float(key),
                    interval_min=r["interval_min"],
                    increments=vals,
                    pattern=f"Alternating block, peak {key} %",
                )
            )
    if (aris or durations) and not out:
        warnings.append(
            "No Alternating Block rows match the return periods / durations."
        )
    return out, warnings


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
    arf: float = 1.0,
) -> tuple[list[Storm], list[str]]:
    """Huff mass curve x DDF depth; ``quartile`` None picks it by duration."""
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
            depth *= arf
            out.append(
                Storm(
                    source=SOURCE_HUFF,
                    ari=t,
                    duration_min=d,
                    name=f"Huff Q{q} {ari_label(t)} {duration_label(d)}",
                    description=(
                        f"Huff quartile {q} {stat} curve x DDF depth "
                        f"{depth:.1f} mm, {ari_label(t)} year, {duration_label(d)}"
                        + (f", ARF {arf:g}" if arf != 1.0 else "")
                    ),
                    interval_min=interval_min,
                    increments=list(frac * depth),
                    pattern=f"Huff Q{q} {stat}",
                    flags=flags,
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
    arf: float = 1.0,
) -> tuple[list[Storm], list[str]]:
    """Ensemble pattern (percent increments) for the return period's AEP band
    and each chosen shape x DDF depth."""
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
                depth *= arf
                out.append(
                    Storm(
                        source=SOURCE_ENSEMBLE,
                        ari=t,
                        duration_min=d,
                        name=f"Ens {ari_label(t)} {duration_label(d)} {shape}",
                        description=(
                            f"Design storm ensemble pattern ({band} AEP band, {shape}) "
                            f"x DDF depth {depth:.1f} mm, {ari_label(t)} year, "
                            f"{duration_label(d)}"
                            + (f", ARF {arf:g}" if arf != 1.0 else "")
                        ),
                        interval_min=row["interval_min"],
                        increments=list(pct / pct.sum() * depth),
                        pattern=f"Ensemble {band} {shape}",
                        flags=flags,
                    )
                )
    return out, warnings
