"""Machine-readable results file shared by Mayim Tools (zero QGIS
dependency): schema "mayim.results/1".

Any tool can write one so that a later tool (e.g. the planned Compile
Hydrology Report) can read results without parsing CSVs:

{
  "schema": "mayim.results/1",
  "tool": {"id", "name", "version", "run_time"},
  "inputs": {"files": {...}, "parameters": {...}},
  "units": {"<quantity>": "<unit>"},
  "items": [
    {"id": "<feature id>",
     "attributes": {...},
     "notes": ["..."],
     "results": [
        {"quantity": "peak_flow", "method": "<id>", "method_label": "...",
         "return_period_years": 50, "aep_percent": 2.0,
         "value": 123.4, "unit": "m3/s",
         "flags": ["..."], "derivation": {...}}]}],
  "warnings": ["..."],
  "references": ["..."]
}

return_period_years / aep_percent are null for results without a return
period (e.g. a regional maximum flood); aep_percent = 100 / T
(annual-maximum convention used throughout the suite). Non-finite
numbers are written as null.
"""

from __future__ import annotations

import json
import math

SCHEMA = "mayim.results/1"


def _clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def result_entry(
    quantity,
    method,
    method_label,
    return_period,
    value,
    unit,
    flags=(),
    derivation=None,
) -> dict:
    return {
        "quantity": quantity,
        "method": method,
        "method_label": method_label,
        "return_period_years": return_period,
        "aep_percent": (100.0 / return_period) if return_period else None,
        "value": value,
        "unit": unit,
        "flags": list(flags),
        "derivation": dict(derivation or {}),
    }


def build_results(tool, inputs, items, units, warnings=(), references=()) -> dict:
    return _clean(
        {
            "schema": SCHEMA,
            "tool": dict(tool),
            "inputs": dict(inputs),
            "units": dict(units),
            "items": list(items),
            "warnings": list(warnings),
            "references": list(references),
        }
    )


def write_results(path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_results(path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} results file.")
    return data
