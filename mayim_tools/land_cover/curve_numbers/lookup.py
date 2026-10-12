"""Land-cover to curve-number lookup: defaults, CSV in and out, checks.

CSV layout (one row per class and case; the tool writes it on every run
as cn_lookup_used.csv, which is also the file to edit and feed back in):

    Code, Class, Case, A, A/B, B, B/C, C, C/D, D, Group override, Table row, Note

Case is low (less runoff), design or high (more runoff). Blank
intermediate columns (A/B, B/C, C/D) are interpolated as the mean of the
neighbouring groups. A class given only as "design" has no condition
range. "Group override" (A-D, or blank) applies one group to the class
whatever the soil map says, e.g. D for wetlands.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

from mayim_tools.soil._common.errors import SoilDataError

from . import tables as tb

CASES = ("low", "design", "high")
COLUMNS = ["Code", "Class", "Case", *tb.GROUPS, "Group override", "Table row", "Note"]


@dataclass
class ClassEntry:
    code: int
    name: str
    values: dict  # case -> np.ndarray(7) (NaN where missing)
    override: str = ""
    rows: dict = field(default_factory=dict)  # case -> row spec / text
    note: str = ""


@dataclass
class Lookup:
    name: str
    table: str  # "scs_sa", "tr55" or "custom"
    classes: dict  # code -> ClassEntry
    source: str = ""
    notes: list = field(default_factory=list)


def _interpolate(vals: np.ndarray, where: str) -> np.ndarray:
    v = np.array(vals, dtype=float)
    for i in (1, 3, 5):  # intermediates
        if (
            not math.isfinite(v[i])
            and math.isfinite(v[i - 1])
            and math.isfinite(v[i + 1])
        ):
            v[i] = 0.5 * (v[i - 1] + v[i + 1])
    return np.round(v, 1)


def _check(entry: ClassEntry, where: str) -> None:
    for case, v in entry.values.items():
        need = range(7) if not entry.override else [tb.GROUPS.index(entry.override)]
        missing = [tb.GROUPS[i] for i in need if not math.isfinite(v[i])]
        if missing:
            raise SoilDataError(
                f"{where}: class {entry.code} ({entry.name}), {case}: no curve "
                f"number for group(s) {', '.join(missing)}."
            )
        bad = [x for x in v if math.isfinite(x) and not 0 <= x <= 100]
        if bad:
            raise SoilDataError(
                f"{where}: class {entry.code} ({entry.name}), {case}: curve "
                "numbers must lie between 0 and 100."
            )


def default_lookup(key: str) -> Lookup:
    if key not in tb.DEFAULT_LOOKUPS:
        raise SoilDataError(f"Unknown built-in lookup '{key}'.")
    name, table, classes = tb.DEFAULT_LOOKUPS[key]
    out = {}
    for c in classes:
        specs = {"low": c.low, "design": c.design, "high": c.high}
        values = {
            case: _interpolate(
                np.array([np.nan if x is None else x for x in tb.row_values(s)]), name
            )
            for case, s in specs.items()
        }
        e = ClassEntry(c.code, c.name, values, c.override, specs, c.note)
        _check(e, name)
        out[c.code] = e
    return Lookup(name, table, out, source=tb.TABLES[table][1])


def _num(text):
    text = (text or "").strip()
    if not text:
        return math.nan
    return float(text.replace(",", "."))


def read_lookup_csv(path: str) -> Lookup:
    """Read a lookup CSV (layout in the module docstring). Header names are
    matched ignoring case and spaces."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    except OSError as exc:
        raise SoilDataError(f"Cannot read the lookup CSV {path}: {exc}") from exc
    rows = [r for r in rows if any(c.strip() for c in r) and not r[0].startswith("#")]
    if not rows:
        raise SoilDataError(f"The lookup CSV {path} is empty.")
    norm = [h.strip().lower().replace(" ", "") for h in rows[0]]

    def col(name, required=True):
        key = name.lower().replace(" ", "")
        if key in norm:
            return norm.index(key)
        if required:
            raise SoilDataError(f"The lookup CSV has no '{name}' column.")
        return None

    i_code, i_case = col("Code"), col("Case", required=False)
    i_name = col("Class", required=False)
    i_groups = {g: col(g, required=g in tb.BASIC) for g in tb.GROUPS}
    i_over = col("Group override", required=False)
    i_row = col("Table row", required=False)
    i_note = col("Note", required=False)

    def cell(r, i):
        return r[i].strip() if i is not None and i < len(r) else ""

    data: dict[int, dict] = {}
    notes = []
    for n, r in enumerate(rows[1:], start=2):
        try:
            code = int(float(cell(r, i_code)))
        except ValueError as exc:
            raise SoilDataError(
                f"Lookup CSV line {n}: code '{cell(r, i_code)}' is not a number."
            ) from exc
        case = (cell(r, i_case) or "design").lower()
        if case not in CASES:
            raise SoilDataError(
                f"Lookup CSV line {n}: case must be low, design or high, not '{case}'."
            )
        try:
            vals = np.array([_num(cell(r, i_groups[g])) for g in tb.GROUPS])
        except ValueError as exc:
            raise SoilDataError(
                f"Lookup CSV line {n}: a curve number is not a number."
            ) from exc
        d = data.setdefault(
            code,
            {
                "name": cell(r, i_name) or str(code),
                "values": {},
                "rows": {},
                "override": "",
                "note": "",
            },
        )
        if case in d["values"]:
            raise SoilDataError(
                f"Lookup CSV line {n}: class {code} has two '{case}' rows."
            )
        d["values"][case] = _interpolate(vals, path)
        d["rows"][case] = cell(r, i_row)
        over = cell(r, i_over).upper()
        if over:
            if over not in tb.BASIC:
                raise SoilDataError(
                    f"Lookup CSV line {n}: group override must be A, B, C or D."
                )
            d["override"] = over
        d["note"] = d["note"] or cell(r, i_note)
    out = {}
    for code, d in data.items():
        if "design" not in d["values"]:
            raise SoilDataError(f"Lookup CSV: class {code} has no 'design' row.")
        for case in ("low", "high"):
            if case not in d["values"]:
                d["values"][case] = d["values"]["design"].copy()
                d["rows"][case] = d["rows"]["design"]
                notes.append(
                    f"class {code} ({d['name']}) has no '{case}' row: no condition range"
                )
        e = ClassEntry(
            code, d["name"], d["values"], d["override"], d["rows"], d["note"]
        )
        _check(e, "Lookup CSV")
        out[code] = e
    return Lookup("Custom lookup", "custom", out, source=path, notes=notes)


def write_lookup_csv(lookup: Lookup, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                f"# {lookup.name}. Low = less runoff, high = more runoff. "
                "Edit and use as a custom lookup."
            ]
        )
        w.writerow(COLUMNS)
        for code in sorted(lookup.classes):
            e = lookup.classes[code]
            for case in CASES:
                v = e.values[case]
                w.writerow(
                    [code, e.name, case]
                    + ["" if not math.isfinite(x) else _fmt(x) for x in v]
                    + [
                        e.override,
                        e.rows.get(case, ""),
                        e.note if case == "design" else "",
                    ]
                )


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def write_reference_table(table: str, path: str) -> None:
    """The full CN table (all rows) for building custom lookups."""
    rows = tb.SCS_SA_ROWS if table == "scs_sa" else tb.TR55_ROWS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# {tb.TABLES[table][1]}"])
        w.writerow(["Table row", "Land cover", "Description", *tb.GROUPS])
        for r in rows:
            w.writerow(
                [r.key, r.cover, r.description]
                + ["" if v is None else _fmt(v) for v in r.values]
            )
