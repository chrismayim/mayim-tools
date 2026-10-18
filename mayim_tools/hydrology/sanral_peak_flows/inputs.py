"""Readers for the tool's CSV inputs (no QGIS dependency): the per-catchment
runoff-coefficient CSV and an optional TR102 station CSV."""

from __future__ import annotations

import csv

from .methods import RunoffCoefficients
from .rainfall import TR102_DAYS, TR102Station
from .tables import RETURN_PERIODS

C_COLUMNS = ("catchment_id", "alpha", "c1", "ft_class", "beta", "c2", "gamma", "c3")
FT_CLASS_ALIASES = {
    "steep_impermeable": "steep_impermeable",
    "steep": "steep_impermeable",
    "impermeable": "steep_impermeable",
    "flat_permeable": "flat_permeable",
    "flat": "flat_permeable",
    "permeable": "flat_permeable",
}


class InputError(ValueError):
    pass


def _sniff_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [
        r for r in csv.reader(text.splitlines(), dialect) if any(c.strip() for c in r)
    ]


def _num(value, name, row_id, default=None):
    text = str(value).strip().replace(",", ".") if value is not None else ""
    if text == "":
        if default is None:
            raise InputError(f"C CSV: {name} is empty for catchment {row_id}.")
        return default
    try:
        return float(text)
    except ValueError as exc:
        raise InputError(
            f"C CSV: {name} '{value}' is not a number ({row_id})."
        ) from exc


def read_c_csv(path) -> dict[str, RunoffCoefficients]:
    """Read the runoff-coefficient CSV: one row per catchment with columns
    catchment_id, alpha, C1, Ft_class (steep_impermeable / flat_permeable),
    beta, C2, gamma, C3 (case-insensitive; beta/C2/gamma/C3 may be empty
    for a rural catchment). Returns {catchment_id: RunoffCoefficients}."""
    rows = _sniff_rows(path)
    if not rows:
        raise InputError("C CSV is empty.")
    header = [c.strip().lower() for c in rows[0]]
    missing = [
        c for c in ("catchment_id", "alpha", "c1", "ft_class") if c not in header
    ]
    if missing:
        raise InputError(
            "C CSV is missing column(s): "
            + ", ".join(missing)
            + ". Expected: "
            + ", ".join(C_COLUMNS)
        )
    ix = {c: header.index(c) for c in C_COLUMNS if c in header}
    out = {}
    for r in rows[1:]:
        r = r + [""] * (len(header) - len(r))
        cid = r[ix["catchment_id"]].strip()
        if not cid:
            continue
        cls = FT_CLASS_ALIASES.get(r[ix["ft_class"]].strip().lower())
        if cls is None:
            raise InputError(
                f"C CSV: Ft_class '{r[ix['ft_class']]}' for {cid} must be "
                "steep_impermeable or flat_permeable (DM Table 3.8)."
            )

        def get(name, default=0.0, _r=r, _cid=cid):
            if name not in ix:
                return default
            return _num(_r[ix[name]], name, _cid, default)

        out[cid] = RunoffCoefficients(
            alpha=_num(r[ix["alpha"]], "alpha", cid),
            c1=_num(r[ix["c1"]], "C1", cid),
            ft_class=cls,
            beta=get("beta"),
            c2=get("c2"),
            gamma=get("gamma"),
            c3=get("c3"),
        )
    return out


def read_tr102_csv(path, r_days: float | None) -> TR102Station:
    """Read a TR102 station CSV (Rational Alternative 2 override): optional
    'key,value' lines (Station, R) followed by a header 'Duration (days),2,5,
    10,20,50,100,200' and rows for 1, 2, 3 and 7 days. M is the 2-year 1-day
    depth. R from the file is used unless r_days is given."""
    rows = _sniff_rows(path)
    meta, header_i = {}, None
    for i, r in enumerate(rows):
        first = r[0].strip().lower()
        if first.startswith("duration"):
            header_i = i
            break
        if len(r) >= 2:
            meta[first] = r[1].strip()
    if header_i is None:
        raise InputError("TR102 CSV: no 'Duration (days)' header row found.")
    header = [c.strip() for c in rows[header_i]]
    try:
        periods = [int(float(c.lower().replace("yr", "").strip())) for c in header[1:]]
    except ValueError as exc:
        raise InputError("TR102 CSV: return-period headers must be numbers.") from exc
    if any(T not in periods for T in RETURN_PERIODS):
        raise InputError("TR102 CSV needs columns 2, 5, 10, 20, 50, 100 and 200.")
    depths = {}
    for r in rows[header_i + 1 :]:
        days = int(float(r[0]))
        depths[days] = {T: float(r[1 + periods.index(T)]) for T in RETURN_PERIODS}
    missing = [d for d in TR102_DAYS if d not in depths]
    if missing:
        raise InputError(f"TR102 CSV is missing duration(s) {missing} days.")
    if r_days is None:
        if "r" not in meta:
            raise InputError("TR102 CSV: give R (thunder days) in the file or tool.")
        r_days = float(meta["r"].replace(",", "."))
    label = meta.get("station", "TR102 station CSV")
    return TR102Station(
        label=f"TR102 station: {label}",
        depths=depths,
        m_mm=depths[1][2],
        r_days=float(r_days),
        source=str(path),
    )
