"""
Core logic for extracting ERA5 / ERA5-Land precipitation at a single
point over a date range, via the Copernicus Climate Data Store (CDS).

ARCHITECTURE: unlike IMERG (Giovanni bulk API) or CHIRPS (direct per-
timestep COG reads), a single CDS request can span many months or
years in one call - a full ~75-year record needs only a handful of
requests, not thousands. This module chunks by YEAR as a deliberate
safety margin against undocumented server-side size/field-count
limits (the exact limit isn't published anywhere found during
research), not because it's strictly required - see fetch_era5_point().

ONE TOOL FOR BOTH PRODUCTS: ERA5 and ERA5-Land share the exact same
CDS access mechanism (same auth, same request/response pattern, same
official cdsapi client) - the only difference is the dataset
identifier ('reanalysis-era5-single-levels' vs 'reanalysis-era5-land')
and, critically, their accumulation convention (see below). This is a
genuine "one core, one parameter" case, unlike combining IMERG/CHIRPS/
ERA5 into one tool, which was rejected because those three have
fundamentally different access mechanisms that would tangle unrelated
logic together.

WHY GRIB, NOT NETCDF: multiple current (late 2024/2025) ECMWF forum
threads report their new GRIB-to-NetCDF conversion pipeline dropping
or corrupting precipitation data specifically ("Missing Variables
(total_precipitation...)", "netcdf only downloaded one time step
while netcdf_legacy correctly downloaded four"). Requesting GRIB
directly avoids that conversion step entirely, and reuses the same
cfgrib/xarray reading approach already built and validated for
grib_to_csv. 'download_format': 'unarchived' is set explicitly since
CDS has been reported to sometimes wrap single files in a zip anyway
regardless of this setting - read_grib_response() defends against
that directly rather than assuming it never happens.

WHY GRIB (De)ACCUMULATION NEEDS CARE: this section describes the
history of two rebuilds; see deaccumulate()'s own docstring for a
THIRD, more fundamental correction that supersedes the "needs
differencing" premise below entirely - as delivered by this tool's
specific CDS request (queried by valid time, not forecast step), each
step's own value is already the correct final hourly figure, and no
differencing against a neighbouring step happens any more. The
history below is kept for context (the reference_time/valid_time
handling it describes is still accurate), not because de-accumulation
is still believed necessary.

ERA5 precipitation is not a
simple per-hour value - it's a forecast accumulation that resets
periodically, and must be de-accumulated (differenced between
consecutive hours within each forecast cycle) to get true hourly
totals. A real discrepancy was found via a live comparison against
the same request retrieved directly from the CDS website: a
systematic undercount plus a missing first value. This went through
two rebuilds, not one - worth recording both, since the first one was
itself a mistake. The first rebuild concluded, from secondary ECMWF
forum/GitHub sources, that ERA5 resets once daily at 00 UTC rather
than twice daily at 06/18 UTC, and changed the grouping accordingly.
That conclusion was wrong: a real downloaded GRIB file's own structure
was shared directly afterward (time, step, latitude, longitude, tp,
valid_time columns) and showed a clear 06:00 reference time with
hourly lead-time steps - confirming the ORIGINAL twice-daily
06/18 UTC grouping-by-reference_time was conceptually correct all
along. Changing it on the basis of indirect web evidence, without
checking a real file's actual structure first, should not have
happened - reverted back to grouping by reference_time, now confirmed
against real data rather than assumed from either direction.

The genuine, still-real bugs this second rebuild fixes: extract_point_series()
previously paired up "time" and "step" values via a manual nested loop
with individual .sel(time=t, step=s) calls, silently skipping any
combination that raised an exception - replaced with xarray's own
da.to_dataframe(), which handles the real underlying indexing
correctly rather than re-implementing it by hand. And deaccumulate()
previously computed its own "valid_time" as reference_time + step_hours
rather than reading the file's own authoritative valid_time field
(confirmed present in the real file inspected, and directly pointed
out as "the correct one" to use) - it's now read directly, with the
old computation kept only as a defensive fallback for a file that
genuinely lacks one.

Negative values after deaccumulation are a documented, currently-
discussed artifact (GRIB's lossy value packing can make consecutive
accumulations non-monotonic by a tiny amount) - small negatives are
clipped to zero; anything beyond a small tolerance is NOT silently
clipped, since a large negative would indicate a real bug, not
packing noise - see NEGATIVE_CLIP_TOLERANCE_MM.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

DATASETS = {
    "era5": "reanalysis-era5-single-levels",
    "era5_land": "reanalysis-era5-land",
}

# The "fast access" alternative - a separate, ARCO-based CDS dataset
# purpose-built for retrieving a long single-point time series in one
# request, rather than the year-chunked bulk-grid retrieval DATASETS
# above uses. Confirmed directly via real live testing (not assumed):
# a 76-year (1950-2025) single request for one point completed in 40
# seconds, versus multi-hour runs for a 3-year request against the
# standard dataset. Also confirmed directly: the response is already
# de-accumulated (each hour's value is independent, not a running
# cumulative total - verified from real output showing a non-monotonic
# sequence and a genuine mid-day zero, the same kind of check that
# caught the standard dataset's own de-accumulation bug), and its
# values match a real CDS website reference exactly for the same
# hours. ECMWF's own documentation for this dataset states plainly:
# "this is an experimental catalogue entry ... not recommended for
# use [in] operational systems" - offered here as a selectable,
# clearly-labelled alternative alongside the standard method, not a
# replacement for it, so that caution is respected rather than quietly
# built past.
TIMESERIES_DATASETS = {
    "era5": "reanalysis-era5-single-levels-timeseries",
    "era5_land": "reanalysis-era5-land-timeseries",  # per ECMWF's own announcement of this parallel dataset - not independently live-tested by this tool (only the plain era5 path has been)
}

# Both products' records now extend back to ~1950 (ERA5-Land's full
# back-extension to 1950 completed 2021; ERA5 has an earlier
# "preliminary" extension to 1940 generally treated as lower
# confidence - defaulting the full-record start to the more
# conservative, fully-confirmed 1950 for both products).
ERA5_RECORD_START = date(1950, 1, 1)

VARIABLE = "total_precipitation"

# GRIB packing can make consecutive accumulations non-monotonic by a
# tiny amount (documented ECMWF forum discussion, not this module's
# own bug) - clip only within this tolerance; anything beyond it is
# left as a genuine negative value and flagged, not silently hidden,
# since a large negative would indicate a real problem worth seeing.
NEGATIVE_CLIP_TOLERANCE_MM = 0.05

DEFAULT_AREA_BUFFER_DEG = 0.15  # bounding box half-width around the point, in degrees


@dataclass
class Era5Result:
    dataframe: pd.DataFrame  # columns: ValidTime, PrecipitationMM
    warnings: list = field(default_factory=list)
    product: str = ""
    anomalies: pd.DataFrame | None = (
        None  # genuinely negative raw hourly values, full detail - see deaccumulate()
    )
    requested_lat: float | None = None  # the point actually asked for
    requested_lon: float | None = None
    matched_lat: float | None = None  # nearest-gridpoint coordinate actually used
    matched_lon: float | None = None
    distance_km: float | None = (
        None  # requested point -> matched grid cell, see _approx_distance_km
    )


def point_to_area(
    lat: float, lon: float, buffer_deg: float = DEFAULT_AREA_BUFFER_DEG
) -> list:
    """CDS requests use an [N, W, S, E] bounding box, not a literal
    point - returns the smallest reasonable box around the target
    coordinate. The buffer is deliberately larger than one grid cell
    so the nearest-point selection downstream always has real
    neighbouring cells to choose from regardless of grid alignment."""
    return [lat + buffer_deg, lon - buffer_deg, lat - buffer_deg, lon + buffer_deg]


def build_request(product: str, years: list, area: list) -> tuple:
    """Returns (dataset_name, request_dict) for one CDS request
    spanning the given years (all months/days/hours within them)."""
    if product not in DATASETS:
        raise ValueError(
            f"Unknown product {product!r} - expected one of {list(DATASETS)}"
        )

    request = {
        "product_type": ["reanalysis"] if product == "era5" else None,
        "variable": [VARIABLE],
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "data_format": "grib",
        "download_format": "unarchived",
    }
    # era5-land's request schema doesn't take product_type (it has only
    # one - a plain reanalysis rerun) - only include it for plain era5.
    if request["product_type"] is None:
        del request["product_type"]

    return DATASETS[product], request


def chunk_years(start: date, end: date, years_per_chunk: int = 1) -> list:
    """Splits [start.year, end.year] into chunks of years_per_chunk
    years each - a deliberate safety margin against undocumented CDS
    request size/field-count limits (the exact limit isn't published
    anywhere found during research). Pure function, fully testable
    without network access."""
    if start > end:
        raise ValueError(f"start ({start}) is after end ({end})")
    years = list(range(start.year, end.year + 1))
    return [
        years[i : i + years_per_chunk] for i in range(0, len(years), years_per_chunk)
    ]


def is_zip_file(path: str | Path) -> bool:
    return zipfile.is_zipfile(path)


def extract_grib_from_zip(zip_path: str | Path, extract_dir: str | Path) -> str:
    """CDS has been reported (ECMWF forum, 2025) to sometimes wrap a
    single-file response in a zip despite 'download_format':
    'unarchived' being set - defends against that rather than
    assuming it never happens. Returns the path to the extracted GRIB
    file; raises if the zip doesn't contain exactly one data file."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(
                f"Expected exactly one file in the CDS response zip, found {len(names)}: {names}"
            )
        zf.extract(names[0], extract_dir)
        return str(Path(extract_dir) / names[0])


# ----------------------------------------------------------------------
# Fast access method: the ARCO/timeseries CDS dataset
# ----------------------------------------------------------------------


def build_timeseries_request(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variable: str = VARIABLE,
    data_format: str = "csv",
) -> dict:
    """Builds the request body for the fast-access timeseries dataset.
    Confirmed directly by real live testing to be accepted by CDS
    without a validation error - not just a documented convention.
    A single request covers the ENTIRE requested date range (no
    year-chunking needed, unlike the standard dataset - confirmed
    directly: a 76-year single request completed in 40 seconds, so
    there's no scale-driven need to split it up)."""
    return {
        "variable": [variable],
        "location": {"latitude": lat, "longitude": lon},
        "date": [f"{start_date}/{end_date}"],
        "data_format": data_format,
    }


def extract_csv_from_zip(zip_path: str | Path, extract_dir: str | Path) -> str:
    """CDS wraps the timeseries dataset's response in a zip even when
    'data_format': 'csv' is explicitly requested - confirmed directly
    by real live testing (the same behaviour already documented for
    the standard dataset's GRIB responses, evidently not specific to
    GRIB). Same logic as extract_grib_from_zip, kept as a separate,
    clearly-named function rather than reused directly, so this path's
    own behaviour isn't implicitly coupled to a function whose name
    and docstring describe GRIB specifically. Returns the path to the
    extracted CSV file; raises if the zip doesn't contain exactly one
    data file."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(
                f"Expected exactly one file in the CDS response zip, found {len(names)}: {names}"
            )
        zf.extract(names[0], extract_dir)
        return str(Path(extract_dir) / names[0])


def read_timeseries_csv_response(path: str, lat: float, lon: float) -> pd.DataFrame:
    """Reads the fast-access timeseries dataset's own CSV response
    format - confirmed directly from two real live test runs, not
    assumed from documentation: columns 'valid_time', 'tp',
    'latitude', 'longitude' (the last two being the MATCHED grid
    cell's own coordinates, which can differ from the requested point
    - the dataset does its own nearest-neighbour matching). 'tp' is in
    metres, the same convention as the standard GRIB retrieval -
    converted to mm here for consistency with the rest of this tool's
    output.

    Each row's value is already an independent, correct per-hour
    figure - confirmed directly (not assumed to match the standard
    dataset's now-corrected behaviour just because it seemed
    plausible): real output showed a non-monotonic sequence within a
    single day and a genuine zero sandwiched between nonzero hours,
    both impossible for a true running cumulative total. No
    de-accumulation is applied here, matching that finding.

    NOT yet independently confirmed: how a genuine gap in the source
    record is represented in this format (no real gap has been seen
    in testing so far). Handled defensively - relies on pandas' own
    standard empty-cell-as-NaN behaviour - rather than assumed absent.
    lat/lon (the originally REQUESTED point, not the matched cell) are
    accepted for signature consistency with every other read_fn in
    this suite and to let the caller compare requested vs matched
    coordinates; not used to filter or select rows here, since the
    file already contains only the one matched cell's data.
    """
    df = pd.read_csv(path)

    expected_cols = {"valid_time", "tp", "latitude", "longitude"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Unexpected columns in timeseries CSV response: {list(df.columns)} - "
            f"expected at least {expected_cols}. The dataset's response format may "
            f"have changed since this was last confirmed."
        )

    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["precipitation_mm"] = pd.to_numeric(df["tp"], errors="coerce") * 1000.0

    return df[["valid_time", "precipitation_mm", "latitude", "longitude"]]


def deaccumulate(df: pd.DataFrame) -> pd.DataFrame:
    """Finalises raw per-step precipitation into the output
    'precipitation_mm' column. Kept under its original name for
    minimal disruption to calling code, but THIS FUNCTION NO LONGER
    DE-ACCUMULATES ANYTHING - see below for why, since the name is
    now a genuine misnomer worth understanding, not just tolerating.

    REBUILT A THIRD TIME after a live comparison against the CDS
    website STILL showed a major discrepancy (a ~13x undercount) even
    after the previous two rebuilds - and this time the actual root
    cause was found, by reconstructing the values this function was
    about to discard from a real run's own anomalies output and
    checking them directly against the same hours' true values on the
    CDS website, rather than reasoning about ERA5's accumulation
    convention in the abstract again.

    The finding: for a real anomaly row (reference_time=06:00,
    step=8, accumulated_mm=0.6466, precipitation_mm_raw=-1.193), the
    PREVIOUS step's accumulated_mm implied by that arithmetic
    (0.6466 - (-1.193) = 1.8396) is not some intermediate figure -
    it is, to four decimal places, exactly the CDS website's own
    reported hourly value for that same hour (1.8396378). This was
    confirmed independently across three separate reference-time
    groups from three different days, not a single coincidence.

    In other words: each (reference_time, step)'s own accumulated_mm,
    exactly as delivered by this tool's CDS request (queried by valid
    `time`, not forecast `step`/lead-time), IS ALREADY the correct,
    final hourly precipitation value - not a cumulative total that
    needs differencing against the previous step at all. The previous
    (now removed) differencing logic was taking two independently
    correct hourly values and subtracting one from the other, which
    has no physical meaning - explaining every symptom observed
    across all three rebuilds at once: systematic undercounting
    (differencing two positive numbers is smaller than either alone,
    often dramatically so), large spurious negatives (whenever the
    second hour's true value was smaller than the first's), and why
    some hours matched the website almost exactly anyway - precisely
    the hours immediately following a DRY hour, where the wrongly-
    subtracted previous value happened to be zero, so the wrong
    formula accidentally produced the right answer. Confirmed directly
    against the pattern in a real side-by-side comparison: every close
    match sat right after a dry hour; every large divergence sat
    during a run of consecutive wet hours.

    This directly contradicts extensive documentation found earlier
    (ECMWF's own forum, GitHub issues) describing ERA5 tp as a
    forecast accumulation requiring de-accumulation. The most likely
    reconciliation, though not independently confirmed by a citation -
    this is inferred from the data itself, which is the appropriate
    thing to trust here given how the previous two rebuilds went wrong
    trusting secondary sources over real evidence: that documentation
    most likely describes raw MARS-style forecast retrieval (querying
    by lead-time step), a different access path from what this tool
    actually uses - cdsapi's standard retrieve() against
    reanalysis-era5-single-levels, queried by valid `time`. CDS's own
    backend appears to resolve each requested valid hour to its own
    already-correct value before returning it.

    Retained from the previous version: negative-value detection
    (a genuinely negative per-hour tp value would now indicate a real
    GRIB precision artifact or data issue, not a mis-differencing
    artifact - expected to be rare, unlike the ~7.5-11% rates seen
    under the old, incorrect differencing logic) and missing-raw-value
    tracking. reference_time/step_hours/valid_time are retained as
    audit columns; reference_time is no longer used for grouping,
    since no differencing occurs.

    Returns (df, large_negative_rows) - large_negative_rows carries
    full context (reference_time, step_hours, valid_time,
    accumulated_mm, precipitation_mm_raw) for whatever, now rare,
    negative values remain.
    """
    n_missing_raw = int(df["accumulated_mm"].isna().sum())

    df = df.copy()
    df = df.sort_values(["reference_time", "valid_time"]).reset_index(drop=True)
    df["precipitation_mm_raw"] = df["accumulated_mm"]
    df.attrs["n_missing_raw"] = n_missing_raw

    negative_mask = df["precipitation_mm_raw"] < 0
    small_negative = negative_mask & (
        df["precipitation_mm_raw"] >= -NEGATIVE_CLIP_TOLERANCE_MM
    )
    large_negative = negative_mask & (
        df["precipitation_mm_raw"] < -NEGATIVE_CLIP_TOLERANCE_MM
    )

    df["precipitation_mm"] = df["precipitation_mm_raw"]
    df.loc[small_negative, "precipitation_mm"] = 0.0
    # large_negative values are left as-is (not clipped, not dropped) -
    # a real anomaly the caller should see, not one this function hides.

    large_negative_rows = df.loc[
        large_negative,
        [
            "reference_time",
            "step_hours",
            "valid_time",
            "accumulated_mm",
            "precipitation_mm_raw",
        ],
    ].copy()

    return df, large_negative_rows


def extract_point_series(ds, lat: float, lon: float) -> pd.DataFrame:
    """Extracts the nearest-gridpoint accumulated-precipitation series
    from an ALREADY-OPENED xarray Dataset (as cfgrib would produce for
    ERA5/ERA5-Land). Separated from read_grib_response() specifically
    so this extraction logic is testable against a synthetic in-memory
    Dataset, without needing a real GRIB file. Returns columns:
    reference_time, step_hours, valid_time, accumulated_mm.

    REBUILT after a real discrepancy was found between this tool's
    output and the same request retrieved independently from the CDS
    website. A real downloaded file's own structure was inspected
    directly (shared as a table: time, step, latitude, longitude, tp,
    number, surface, valid_time columns) and showed two things this
    version corrects:

    1. The file carries its own authoritative 'valid_time' coordinate
       (= time + step, but computed by cfgrib itself, not by this
       tool). The previous version never read it - it recomputed
       reference_time + step_hours by hand and trusted that instead.
       Even though the arithmetic checks out in the sample inspected,
       trusting a self-derived value over the file's own authoritative
       field was an avoidable risk, not a necessary one. This version
       reads 'valid_time' directly when the dataset provides it, and
       only falls back to computing it from time+step if it's
       genuinely absent.
    2. The previous version paired up 'time' and 'step' values via a
       manual nested loop with individual .sel(time=t, step=s) calls,
       silently swallowing any combination that raised an exception
       (try/except: continue). If the underlying data isn't a
       perfectly dense time-x-step grid - plausible for a multi-year,
       multi-chunk request - this could silently skip real data or,
       worse, is simply more exposed to subtle indexing mistakes than
       necessary. This version uses xarray's own da.to_dataframe(),
       which handles the real underlying indexing correctly rather
       than re-implementing it by hand - the same, more robust
       approach used to inspect the real file that caught this in the
       first place.
    """
    var_name = None
    for candidate in ("tp", "precipitation", VARIABLE):
        if candidate in ds.data_vars:
            var_name = candidate
            break
    if var_name is None:
        raise ValueError(
            f"Could not find a precipitation variable in the dataset. "
            f"Available: {list(ds.data_vars)}"
        )

    da = ds[var_name].sel(latitude=lat, longitude=lon, method="nearest")

    if "time" not in da.coords:
        raise ValueError(
            f"Expected a 'time' (reference time) coordinate, found: {list(da.coords)}"
        )

    df = da.to_dataframe().reset_index()

    rename_map = {var_name: "accumulated_mm", "time": "reference_time"}
    df = df.rename(columns=rename_map)

    if "step" in df.columns:
        df["step_hours"] = pd.to_timedelta(df["step"]).dt.total_seconds() / 3600
    else:
        df["step_hours"] = 0.0

    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"])
    else:
        # defensive fallback only - the real file inspected during
        # development DID carry its own valid_time; this covers a
        # differently-structured file that genuinely lacks one
        df["valid_time"] = pd.to_datetime(df["reference_time"]) + pd.to_timedelta(
            df["step_hours"], unit="h"
        )

    df["reference_time"] = pd.to_datetime(df["reference_time"])
    # ERA5's 'tp' is in metres - convert to mm (standard convention throughout this suite)
    df["accumulated_mm"] = df["accumulated_mm"].astype(float) * 1000

    # capture the ACTUAL nearest-gridpoint coordinate .sel(..., method="nearest")
    # matched, before it's dropped by the keep_cols filter below - carried via
    # .attrs (same pattern deaccumulate() already uses for n_missing_raw) so the
    # Standard method's caller (fetch_era5_point) can report/expose it exactly
    # like the Fast method already does, without changing this function's
    # documented column contract.
    matched_lat = float(df["latitude"].iloc[0]) if len(df) else None
    matched_lon = float(df["longitude"].iloc[0]) if len(df) else None

    keep_cols = ["reference_time", "step_hours", "valid_time", "accumulated_mm"]
    result = df[keep_cols].reset_index(drop=True)
    result.attrs["matched_lat"] = matched_lat
    result.attrs["matched_lon"] = matched_lon
    return result


def read_grib_response(grib_path: str | Path, lat: float, lon: float) -> pd.DataFrame:
    """Reads a downloaded ERA5/ERA5-Land GRIB file and delegates to
    extract_point_series() for the actual extraction. Uses cfgrib/
    xarray, the same approach already built and validated for
    grib_to_csv."""
    import xarray as xr

    with xr.open_dataset(grib_path, engine="cfgrib") as ds:
        return extract_point_series(ds, lat, lon)


def fetch_era5_point(
    lat: float,
    lon: float,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    product: str = "era5",
    years_per_chunk: int = 1,
    cds_retrieve_fn: Callable | None = None,
    read_fn: Callable = read_grib_response,
    progress_callback: Callable[[int], None] | None = None,
    work_dir: str | Path = ".",
    api_key: str | None = None,
    save_credentials: bool = True,
    credentials_path: str | Path | None = None,
) -> Era5Result:
    """Orchestrates fetching every year-chunk in [start_date, end_date]
    for the chosen product, submitting one CDS request per chunk
    (each request itself spans many months/years' worth of hourly
    data in one call - a fundamentally different scale problem than
    IMERG/CHIRPS's per-timestep-file model). cds_retrieve_fn and
    read_fn are injectable so the orchestration logic (chunking,
    per-chunk error isolation, deaccumulation, DataFrame assembly) can
    be fully unit-tested without any real CDS access or credentials.

    cds_retrieve_fn(dataset, request, target_path) -> None: performs
    one CDS retrieve-and-download. Defaults to a real cdsapi-based
    implementation (see _default_cds_retrieve) reading credentials
    from ~/.cdsapirc via the official cdsapi client - the same
    file-based pattern used for IMERG's Earthdata login. If api_key is
    provided, it's SAVED to ~/.cdsapirc (see save_cds_credentials)
    before being used for this run, so a key entered once via a QGIS
    parameter is picked up automatically on every future run without
    needing to be re-entered or ever appearing in a QGIS project file
    or Processing history - the key lives only in the standard
    cdsapi config file, never in this tool's own parameters/state.
    Set save_credentials=False to use an api_key for this run only,
    without persisting it.
    """
    if product not in DATASETS:
        raise ValueError(
            f"Unknown product {product!r} - expected one of {list(DATASETS)}"
        )

    start = _to_date(start_date) if start_date else ERA5_RECORD_START
    end = _to_date(end_date) if end_date else date.today()

    year_chunks = chunk_years(start, end, years_per_chunk)
    area = point_to_area(lat, lon)

    if api_key and save_credentials:
        save_cds_credentials(api_key, config_path=credentials_path)
        # the key is now in ~/.cdsapirc - let the default cdsapi.Client()
        # pick it up from there rather than passing it through explicitly,
        # so this run and every future run go through the identical path
        api_key = None

    if cds_retrieve_fn is None:

        def cds_retrieve_fn(dataset, request, target_path, _api_key=api_key):
            _default_cds_retrieve(dataset, request, target_path, api_key=_api_key)

    warnings = []
    all_rows = []
    all_anomalies = []
    n = len(year_chunks)
    total_missing_raw = 0
    matched_lat = None
    matched_lon = None

    for i, years in enumerate(year_chunks):
        dataset, request = build_request(product, years, area)
        target_path = Path(work_dir) / f"era5_{product}_{years[0]}_{years[-1]}.grib"

        try:
            cds_retrieve_fn(dataset, request, target_path)

            actual_path = target_path
            if is_zip_file(target_path):
                actual_path = extract_grib_from_zip(target_path, Path(work_dir))

            raw_df = read_fn(actual_path, lat, lon)
            # the matched grid cell is the same for every chunk (same lat/lon,
            # same underlying grid) - only need to capture it once
            if matched_lat is None:
                matched_lat = raw_df.attrs.get("matched_lat")
                matched_lon = raw_df.attrs.get("matched_lon")
            deacc_df, anomaly_rows = deaccumulate(raw_df)
            total_missing_raw += deacc_df.attrs.get("n_missing_raw", 0)

            # keep only rows within the actually-requested date range
            # (a year-chunk request returns the WHOLE year(s), even if
            # start/end only cover part of the first/last one) - applied
            # to the anomalies too, not just the main output. Found via
            # a real discrepancy a user caught by independently counting
            # negative values in the final CSV (12387 anomalies reported
            # vs 11521 actually present): anomaly_rows was previously
            # collected BEFORE this trim, so anomalies occurring in the
            # "extra" untrimmed portion of a chunk (e.g. Feb-Dec of a
            # final year when only Jan 1 was actually requested) were
            # counted in the anomaly warning/audit CSV despite never
            # appearing anywhere in the actual output - the audit output
            # must describe what's actually in the CSV, not a superset
            # of it that happened to pass through de-accumulation.
            mask = (deacc_df["valid_time"].dt.date >= start) & (
                deacc_df["valid_time"].dt.date <= end
            )

            if len(anomaly_rows):
                anomaly_mask = (anomaly_rows["valid_time"].dt.date >= start) & (
                    anomaly_rows["valid_time"].dt.date <= end
                )
                anomaly_rows = anomaly_rows.loc[anomaly_mask]
                if len(anomaly_rows):
                    all_anomalies.append(anomaly_rows)

            chunk_df = deacc_df.loc[mask, ["valid_time", "precipitation_mm"]]
            all_rows.append(chunk_df)

        except Exception as e:
            warnings.append(f"{years[0]}-{years[-1]}: failed ({type(e).__name__}: {e})")

        if progress_callback:
            progress_callback(int(100 * (i + 1) / max(n, 1)))

    distance_km = None
    if matched_lat is not None and matched_lon is not None:
        distance_km = _approx_distance_km(lat, lon, matched_lat, matched_lon)
        warnings.append(
            f"Requested point ({lat:.4f}, {lon:.4f}) was matched to the nearest grid cell "
            f"({matched_lat:.4f}, {matched_lon:.4f}), approximately {distance_km:.1f} km away."
        )

    anomalies_df = (
        pd.concat(all_anomalies, ignore_index=True)
        if all_anomalies
        else pd.DataFrame(
            columns=[
                "reference_time",
                "step_hours",
                "valid_time",
                "accumulated_mm",
                "precipitation_mm_raw",
            ]
        )
    )

    if total_missing_raw:
        warnings.append(
            f"{total_missing_raw} raw precipitation value(s) were genuinely missing "
            f"(NaN) in the source GRIB data - written as an empty cell in the final output "
            f"for that hour only (not zero or dropped, and no longer affecting neighbouring "
            f"hours, since values are no longer differenced against each other). This is a "
            f"gap in the source data itself, not something this tool can fill in."
        )

    if len(anomalies_df):
        vals = anomalies_df["precipitation_mm_raw"]
        warnings.append(
            f"{len(anomalies_df)} value(s) were negative beyond the GRIB-packing tolerance "
            f"({NEGATIVE_CLIP_TOLERANCE_MM} mm) - left as-is rather than silently clipped, since "
            f"this may indicate a real problem rather than packing noise. A genuinely negative raw "
            f"hourly precipitation value from the source data is expected to be rare - if this count "
            f"is high, it's worth investigating rather than assuming packing noise. "
            f"Magnitude range: {vals.min():.3f} to {vals.max():.3f} mm, median {vals.median():.3f} mm "
            f"- see the anomalies detail output for the full list with timestamps."
        )

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined = combined.rename(
            columns={"valid_time": "ValidTime", "precipitation_mm": "PrecipitationMM"}
        )
        combined = combined.sort_values("ValidTime").reset_index(drop=True)
        n_before_dedup = len(combined)
        combined = combined.drop_duplicates(subset="ValidTime").reset_index(drop=True)
        n_duplicates = n_before_dedup - len(combined)
    else:
        combined = pd.DataFrame(columns=["ValidTime", "PrecipitationMM"])
        n_duplicates = 0

    if n_duplicates:
        warnings.append(
            f"{n_duplicates} duplicate ValidTime timestamp(s) were found across chunk results and "
            f"dropped (keeping one row per timestamp) - found on a real live run via a discrepancy "
            f"between the tool's own anomaly count and a user's independent count of negative values "
            f"in the final CSV; the exact mechanism producing these duplicates (e.g. an overlap "
            f"between adjacent year-chunks, or between ERA5's two daily forecast cycles at a "
            f"specific step) is not yet confirmed - this count is reported so a pattern (e.g. "
            f"clustering at year-chunk boundaries) can be investigated from here."
        )

    # keep the anomalies audit output consistent with what's actually in the
    # final CSV - a real live run found these were previously reported
    # inconsistently (the anomalies count included duplicates the main
    # output had already dropped), which a user's own independent count
    # of negative values caught directly
    if len(anomalies_df):
        n_anomalies_before_dedup = len(anomalies_df)
        anomalies_df = anomalies_df.drop_duplicates(subset="valid_time").reset_index(
            drop=True
        )
        n_anomaly_duplicates = n_anomalies_before_dedup - len(anomalies_df)
        if n_anomaly_duplicates:
            warnings.append(
                f"{n_anomaly_duplicates} duplicate anomaly row(s) were also found and dropped from "
                f"the anomalies audit output, for the same reason."
            )

    return Era5Result(
        dataframe=combined,
        warnings=warnings,
        product=product,
        anomalies=anomalies_df,
        requested_lat=lat,
        requested_lon=lon,
        matched_lat=matched_lat,
        matched_lon=matched_lon,
        distance_km=distance_km,
    )


def fetch_era5_point_timeseries(
    lat: float,
    lon: float,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    product: str = "era5",
    cds_retrieve_fn: Callable | None = None,
    read_fn: Callable = read_timeseries_csv_response,
    work_dir: str | Path = ".",
    api_key: str | None = None,
    save_credentials: bool = True,
    credentials_path: str | Path | None = None,
) -> Era5Result:
    """The FAST access method - see TIMESERIES_DATASETS' own comment
    for the full evidence trail (confirmed live: 76 years in ~40
    seconds, already-correct per-hour values needing no
    de-accumulation, exact agreement with a real CDS website
    reference). A genuinely simpler orchestration than
    fetch_era5_point: ONE request covers the entire date range - no
    year-chunking, no per-chunk error isolation, no reference_time/
    step bookkeeping, since this dataset exposes none of that
    complexity to begin with.

    Still carries a real, ECMWF-stated caveat (see TIMESERIES_DATASETS)
    - offered as a selectable alternative, not a silent replacement.

    cds_retrieve_fn/read_fn are injectable for the same reason as
    fetch_era5_point: full unit-testability without real CDS access.
    """
    if product not in TIMESERIES_DATASETS:
        raise ValueError(
            f"Unknown product {product!r} - expected one of {list(TIMESERIES_DATASETS)}"
        )

    start = _to_date(start_date) if start_date else ERA5_RECORD_START
    end = _to_date(end_date) if end_date else date.today()

    warnings = []

    if api_key and save_credentials:
        save_cds_credentials(api_key, config_path=credentials_path)
        api_key = None

    if cds_retrieve_fn is None:

        def cds_retrieve_fn(dataset, request, target_path, _api_key=api_key):
            _default_cds_retrieve(dataset, request, target_path, api_key=_api_key)

    dataset = TIMESERIES_DATASETS[product]
    request = build_timeseries_request(lat, lon, start.isoformat(), end.isoformat())

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    target_path = work_dir / "era5_timeseries_response.download"

    cds_retrieve_fn(dataset, request, target_path)

    actual_path = str(target_path)
    if is_zip_file(target_path):
        actual_path = extract_csv_from_zip(target_path, work_dir)

    df = read_fn(actual_path, lat, lon)

    matched_lat = None
    matched_lon = None
    distance_km = None
    if len(df):
        matched_lat = float(df["latitude"].iloc[0])
        matched_lon = float(df["longitude"].iloc[0])
        distance_km = _approx_distance_km(lat, lon, matched_lat, matched_lon)
        warnings.append(
            f"Requested point ({lat:.4f}, {lon:.4f}) was matched to the nearest grid cell "
            f"({matched_lat:.4f}, {matched_lon:.4f}), approximately {distance_km:.1f} km away."
        )

    n_missing = int(df["precipitation_mm"].isna().sum()) if len(df) else 0
    if n_missing:
        warnings.append(
            f"{n_missing} row(s) had a missing precipitation value - written as an empty cell, "
            f"not zero or dropped. This dataset's own missing-value representation has not been "
            f"independently confirmed against a real gap before now - worth checking which "
            f"timestamps before treating the output as complete."
        )

    if len(df):
        negative_mask = df["precipitation_mm"] < 0
        small_negative = negative_mask & (
            df["precipitation_mm"] >= -NEGATIVE_CLIP_TOLERANCE_MM
        )
        large_negative = negative_mask & ~small_negative
        anomalies_df = df.loc[
            large_negative, ["valid_time", "precipitation_mm", "latitude", "longitude"]
        ].copy()
        df.loc[small_negative, "precipitation_mm"] = 0.0
    else:
        anomalies_df = pd.DataFrame(
            columns=["valid_time", "precipitation_mm", "latitude", "longitude"]
        )

    if len(anomalies_df):
        warnings.append(
            f"{len(anomalies_df)} value(s) were negative beyond the {NEGATIVE_CLIP_TOLERANCE_MM} mm "
            f"tolerance - left as-is rather than silently clipped. This is genuinely unexpected for "
            f"this access method (its values are already correct per-hour figures, not a "
            f"differenced result prone to this) - worth investigating if this count is nonzero."
        )

    if len(df):
        combined = df[["valid_time", "precipitation_mm"]].rename(
            columns={"valid_time": "ValidTime", "precipitation_mm": "PrecipitationMM"}
        )
    else:
        combined = pd.DataFrame(columns=["ValidTime", "PrecipitationMM"])
    combined = combined.sort_values("ValidTime").reset_index(drop=True)

    return Era5Result(
        dataframe=combined,
        warnings=warnings,
        product=product,
        anomalies=anomalies_df,
        requested_lat=lat,
        requested_lon=lon,
        matched_lat=matched_lat,
        matched_lon=matched_lon,
        distance_km=distance_km,
    )


def _approx_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """A simple planar approximation, adequate for reporting a small
    (sub-grid-cell) nearest-neighbour offset - not intended for
    anything requiring genuine geodesic accuracy."""
    import math

    lat_km = (lat2 - lat1) * 111.0
    lon_km = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(lat_km, lon_km)


CDSAPIRC_URL = "https://cds.climate.copernicus.eu/api"


def save_cds_credentials(api_key: str, config_path: str | Path | None = None) -> Path:
    """Writes ~/.cdsapirc with the given API key, in the exact 2-line
    format the official cdsapi client expects - so a key entered once
    via a QGIS parameter is picked up automatically on every future
    run without needing to be re-entered (get_cds_client()'s default
    cdsapi.Client() call reads this same file).

    config_path is overridable (used by tests to write to a temp
    location instead of the real home directory - never the real
    ~/.cdsapirc during a test run). Always writes exactly these two
    lines - .cdsapirc has no other legitimate content for this use
    case, so a clean overwrite is simpler and more robust than trying
    to selectively patch an existing file.

    The key is written to disk here, never logged or echoed anywhere
    in this module or the QGIS algorithm - callers should confirm the
    save happened without printing the key value itself.
    """
    path = Path(config_path) if config_path else Path.home() / ".cdsapirc"
    path.write_text(f"url: {CDSAPIRC_URL}\nkey: {api_key}\n")
    return path


def get_cds_client(api_key: str | None = None):
    """Initializes the official ECMWF cdsapi client. Defaults to
    reading ~/.cdsapirc (cdsapi.Client()'s own standard behaviour) -
    the same file-based credential pattern used for IMERG's Earthdata
    Login, so credentials never need to be typed into a QGIS parameter
    or saved in a QGIS project file. An explicit api_key overrides
    this for a single run if provided (e.g. a QGIS parameter override),
    but the default and recommended path is the config file."""
    import cdsapi

    if api_key:
        return cdsapi.Client(url=CDSAPIRC_URL, key=api_key)
    try:
        return cdsapi.Client()
    except Exception as e:
        raise RuntimeError(
            "Could not initialize the CDS client. Configure ~/.cdsapirc with your "
            "Personal Access Token from https://cds.climate.copernicus.eu/profile "
            "(format: 'url: https://cds.climate.copernicus.eu/api' then "
            "'key: <TOKEN>' on the next line), or provide an API key directly via "
            f"this tool's optional CDS API Key parameter. Underlying error: {e}"
        ) from e


def _default_cds_retrieve(
    dataset: str, request: dict, target_path: Path, api_key: str | None = None
) -> None:
    """Real CDS retrieval via the official cdsapi client for the
    SUBMIT step only - the actual file DOWNLOAD is now done manually
    with `requests`, not via cdsapi's own Result.download()/
    _download(). See module docstring below for why.

    HISTORY, kept because it explains why this looks the way it does:
    a first live attempt used the chained
    `client.retrieve(dataset, request).download(target)` form and
    failed with "'NoneType' object has no attribute 'write'" after the
    request itself completed successfully (confirming the request
    construction - dataset, variable, area, dates, format - was
    correct; only the download step failed). A second attempt switched
    to the 3-argument direct form, `client.retrieve(dataset, request,
    target)`, based on multiple independently-sourced current working
    examples. A second live run, after another full ~12.5-minute queue
    wait, failed with the IDENTICAL error. Tracing cdsapi's own source
    directly (not just documentation examples) showed why the second
    fix could never have worked: `Client.retrieve(..., target)`
    internally just calls `result.download(target)`, which calls
    `result._download(...)` - the EXACT SAME code path the first
    (chained) attempt already went through. Both calling conventions
    hit the same internal download implementation, so switching
    between them changed nothing about what actually runs - whatever
    the real bug is, it lives inside cdsapi's own download internals
    (possibly version-specific - Chris's installed cdsapi version was
    never confirmed to match the one inspected while building this),
    not in how this code calls it.

    FIX: stop depending on cdsapi's own download() entirely. Submit
    the request via `client.retrieve(dataset, request)` with NO target
    (so cdsapi never attempts its own download), then download the
    result's file manually via a plain `requests.get(...)` against
    `result.location` - the same URL cdsapi's own downloader would
    have used, confirmed by reading Result._download()'s source
    directly, just with the download loop fully under this code's own
    control instead of depending on cdsapi's internals working
    correctly on whatever version happens to be installed.
    """
    client = get_cds_client(api_key)
    result = client.retrieve(dataset, request)
    _download_result_manually(result, target_path)


def _download_result_manually(
    result, target_path: Path, chunk_size: int = 1024 * 1024
) -> None:
    """Downloads a completed CDS request's result file directly via
    `requests`, bypassing cdsapi's own Result.download()/_download()
    entirely - see _default_cds_retrieve's docstring for why. Streams
    to disk in chunks rather than loading the whole file into memory
    (ERA5 GRIB responses can be tens of MB even for a small area/short
    date range). Raises a clear, specific error - naming the actual
    byte-count mismatch or HTTP status - rather than letting a bare
    AttributeError surface if anything about the response is
    unexpected, so a future failure here is immediately diagnosable
    from the QGIS log alone."""
    import requests

    url = result.location
    expected_size = getattr(result, "content_length", None)

    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    total = 0
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    if expected_size and total != expected_size:
        raise RuntimeError(
            f"Download incomplete: got {total} byte(s), expected {expected_size} byte(s) "
            f"from {url}. The request itself succeeded (this is a transfer issue, not a "
            f"request-construction one) - retrying the same run is the appropriate next step."
        )


def _to_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()
