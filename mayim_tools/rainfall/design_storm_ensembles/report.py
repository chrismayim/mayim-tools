"""Output generation for Design Storm Ensembles - zero QGIS dependency,
like core.py.

CSVs follow the Mayim output standard: UTF-8 with a byte-order mark (so
Excel opens them cleanly) and a column key above the header as
'# column: meaning' lines (read with pandas ``comment='#'``; the
Prepare HEC-HMS / SWMM project readers skip them too)."""

import csv

from mayim_tools.rainfall._common.ddf import (
    AEP_BAND_FREQUENT_MIN,
    AEP_BAND_INTERMEDIATE_MIN,
)

CSV_ENCODING = "utf-8-sig"
KEY_PREFIX = "# "

_BANDS = (
    f"AEP band: frequent (AEP {AEP_BAND_FREQUENT_MIN:g} % or more) / intermediate "
    f"({AEP_BAND_INTERMEDIATE_MIN:g} to {AEP_BAND_FREQUENT_MIN:g} %) / rare "
    f"(less than {AEP_BAND_INTERMEDIATE_MIN:g} %)"
)
_SHAPE = "third of the storm that holds the peak burst: Early / Middle / Late"

ENSEMBLE_KEY = {
    "Duration": "storm duration (min)",
    "TimeStep": "time step of the pattern (min)",
    "AEP": _BANDS,
    "Shape": _SHAPE,
    "Increment_1 ...": (
        "rainfall in time step 1 ... n as a percentage of the storm total "
        "(each row sums to 100); a row with no increments is a bin with too "
        "few storms - see the bin summary"
    ),
}

CATALOGUE_KEY = {
    "duration_minutes": "duration of the extracted storm window (min)",
    "event_id": "number of the independent storm event in the input series",
    "start": "start of the storm window (date and time as in the input)",
    "end": "end of the storm window (date and time as in the input)",
    "depth_mm": "rainfall depth in the window (mm)",
    "aep_percent": "annual exceedance probability of that depth (%)",
    "aep_band": _BANDS,
    "aep_method": (
        "how the AEP was found: ddf = from the DDF table / empirical = Weibull "
        "rank in the sample (depth outside the DDF table's range)"
    ),
    "shape": _SHAPE,
}

BIN_SUMMARY_KEY = {
    "Duration": "storm duration (min)",
    "TimeStep": "time step of the pattern (min)",
    "AEP": _BANDS,
    "Shape": _SHAPE,
    "n_storms": "storms used for the pattern (including pooled storms)",
    "n_storms_own_duration": "storms of this bin's own duration",
    "status": (
        "how the pattern was built: direct_from_data / direct_from_data_pooled "
        "(neighbouring durations added) / monte_carlo_disaggregated (time step "
        "finer than the input series) / insufficient_data / "
        "insufficient_data_even_pooled / insufficient_data_no_anchor (left blank)"
    ),
    "pooled_from_durations": "durations (min) whose storms were pooled in",
}

CATALOGUE_COLUMNS = list(CATALOGUE_KEY)
BIN_SUMMARY_COLUMNS = list(BIN_SUMMARY_KEY)


def _write_key(f, key, columns):
    f.write(f"{KEY_PREFIX}Column key\n")
    for col, meaning in key.items():
        if col in columns or col.endswith("..."):
            f.write(f"{KEY_PREFIX}{col}: {meaning}\n")


def write_ensemble_csv(ensemble_rows, path):
    """Main ensemble CSV: Duration, TimeStep, AEP, Shape, then one
    column per increment (padded with blanks for rows with fewer
    increments than the widest row)."""
    max_increments = max((len(r["Increments"]) for r in ensemble_rows), default=0)
    header = ["Duration", "TimeStep", "AEP", "Shape"] + [
        f"Increment_{i + 1}" for i in range(max_increments)
    ]
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        _write_key(f, ENSEMBLE_KEY, header)
        writer = csv.writer(f)
        writer.writerow(header)
        for r in ensemble_rows:
            row = [r["Duration"], r["TimeStep"], r["AEP"], r["Shape"]]
            row += r["Increments"]
            row += [""] * (max_increments - len(r["Increments"]))
            writer.writerow(row)


def _write_dicts(rows, path, default_columns, key):
    columns = list(rows[0].keys()) if rows else list(default_columns)
    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        _write_key(f, key, columns)
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_catalogue_csv(catalogue_rows, path):
    _write_dicts(catalogue_rows, path, CATALOGUE_COLUMNS, CATALOGUE_KEY)


def write_bin_summary_csv(bin_summary_rows, path):
    _write_dicts(bin_summary_rows, path, BIN_SUMMARY_COLUMNS, BIN_SUMMARY_KEY)
