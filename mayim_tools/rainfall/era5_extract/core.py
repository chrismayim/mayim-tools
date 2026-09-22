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
and, historically, their accumulation convention (see deaccumulate()'s
own docstring for why that no longer matters for this tool's specific
request pattern).

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

THE DEACCUMULATION HISTORY - kept because it explains the current
design, not because de-accumulation is still believed necessary. ERA5
precipitation was long assumed to be a forecast accumulation requiring
differencing between consecutive forecast steps to recover true hourly
totals. A live comparison against the same request retrieved directly
from the CDS website went through two genuine rebuilds before the real
root cause was found (see deaccumulate()'s own docstring for the full
evidence trail): for THIS tool's specific CDS request (queried by
valid time via cdsapi's standard retrieve(), not by forecast lead-time
step), each returned value is already the correct, final hourly
figure - no differencing against a neighbouring step is needed or
correct. The function is kept under its original name for minimal
disruption to calling code, despite no longer de-accumulating
anything.

Negative values are a documented, currently-discussed GRIB packing
artifact (lossy value packing can make a genuinely-zero or small value
come back as a tiny negative) - small negatives are clipped to zero;
anything beyond a small tolerance is left as-is and flagged, since a
large negative likely indicates a real problem, not packing noise -
see NEGATIVE_CLIP_TOLERANCE_MM.
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

# Bounding box half-width around the point, in degrees - CDS requests
# use an [N, W, S, E] box, not a literal point.
DEFAULT_AREA_BUFFER_DEG = 0.15

CDSAPIRC_URL = "https://cds.climate.copernicus.eu/api"


@dataclass
class Era5Result:
    dataframe: pd.DataFrame  # columns: ValidTime, PrecipitationMM
    warnings: list = field(default_factory=list)
    product: str = ""
    anomalies: pd.DataFrame | None = (
        None  # genuinely negative raw hourly values, full detail
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


def is_zip_file(path) -> bool:
    return zipfile.is_zipfile(path)


def extract_grib_from_zip(zip_path, extract_dir) -> str:
    """CDS has been reported (ECMWF forum, 2025) to sometimes wrap a
    single-file response in a zip despite 'download_format':
    'unarchived' being set - defends against that rather than
    assuming it never happens. Returns the path to the extracted GRIB
    file; raises if the zip doesn't contain exactly one data file."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(
                f"Expected exactly one file in the CDS response zip, found "
                f"{len(names)}: {names}"
            )
        zf.extract(names[0], extract_dir)
        return str(Path(extract_dir) / names[0])


def deaccumulate(df: pd.DataFrame) -> tuple:
    """Finalises raw per-step precipitation into the output
    'precipitation_mm' column. Kept under its original name for
    minimal disruption to calling code, but THIS FUNCTION NO LONGER
    DE-ACCUMULATES ANYTHING - the name is now a genuine misnomer worth
    understanding, not just tolerating.

    REBUILT A THIRD TIME after a live comparison against the CDS
    website STILL showed a major discrepancy (a ~13x undercount) even
    after two previous rebuilds - and this time the actual root cause
    was found, by reconstructing the values this function was about to
    discard from a real run's own anomalies output and checking them
    directly against the same hours' true values on the CDS website,
    rather than reasoning about ERA5's accumulation convention in the
    abstract again.

    The finding: for a real anomaly row (reference_time=06:00, step=8,
    accumulated_mm=0.6466, precipitation_mm_raw=-1.193 under the OLD,
    now-removed differencing logic), the PREVIOUS step's accumulated_mm
    implied by that arithmetic (0.6466 - (-1.193) = 1.8396) is not some
    intermediate figure - it is, to four decimal places, exactly the
    CDS website's own reported hourly value for that same hour
    (1.8396378). Confirmed independently across three separate
    reference-time groups from three different days.

    In other words: each (reference_time, step)'s own accumulated_mm,
    exactly as delivered by this tool's CDS request (queried by valid
    time, not forecast step/lead-time), IS ALREADY the correct, final
    hourly precipitation value - not a cumulative total needing
    differencing against the previous step at all. The previous (now
    removed) differencing logic was taking two independently correct
    hourly values and subtracting one from the other, which has no
    physical meaning - explaining every symptom observed across all
    three rebuilds at once: systematic undercounting, large spurious
    negatives, and why some hours matched the website almost exactly
    anyway (precisely the hours immediately following a DRY hour,
    where the wrongly-subtracted previous value happened to be zero,
    so the wrong formula accidentally produced the right answer).

    This directly contradicts extensive documentation found earlier
    (ECMWF's own forum, GitHub issues) describing ERA5 tp as a forecast
    accumulation requiring de-accumulation. The most likely
    reconciliation (inferred from the data itself, not independently
    confirmed by a citation): that documentation most likely describes
    raw MARS-style forecast retrieval (querying by lead-time step), a
    different access path from what this tool actually uses - cdsapi's
    standard retrieve() against reanalysis-era5-single-levels, queried
    by valid time. CDS's own backend appears to resolve each requested
    valid hour to its own already-correct value before returning it.

    Retained: negative-value detection (a genuinely negative per-hour
    tp value would now indicate a real GRIB precision artifact, not a
    mis-differencing artifact - expected to be rare) and missing-raw-
    value tracking. reference_time/step_hours/valid_time are retained
    as audit columns; reference_time is no longer used for grouping,
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

    Trusts the dataset's own authoritative 'valid_time' coordinate
    (confirmed present in a real downloaded file, and confirmed as
    "the correct one" to use) rather than recomputing it as
    reference_time + step_hours by hand - falling back to that
    computation only if the dataset genuinely lacks a valid_time field.
    Uses xarray's own da.to_dataframe() to pair up every (time, step)
    combination correctly, rather than a hand-rolled nested loop with
    individual .sel() calls that could silently skip real data on a
    non-dense time-x-step grid.
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
    df = df.rename(columns={var_name: "accumulated_mm", "time": "reference_time"})

    if "step" in df.columns:
        df["step_hours"] = pd.to_timedelta(df["step"]).dt.total_seconds() / 3600
    else:
        df["step_hours"] = 0.0

    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"])
    else:
        # Defensive fallback only - a real file inspected during
        # development DID carry its own valid_time; this covers a
        # differently-structured file that genuinely lacks one.
        df["valid_time"] = pd.to_datetime(df["reference_time"]) + pd.to_timedelta(
            df["step_hours"], unit="h"
        )

    df["reference_time"] = pd.to_datetime(df["reference_time"])
    # ERA5's 'tp' is in metres - convert to mm (standard convention
    # throughout this suite).
    df["accumulated_mm"] = df["accumulated_mm"].astype(float) * 1000

    keep_cols = ["reference_time", "step_hours", "valid_time", "accumulated_mm"]
    return df[keep_cols].reset_index(drop=True)


def read_grib_response(grib_path, lat: float, lon: float) -> pd.DataFrame:
    """Reads a downloaded ERA5/ERA5-Land GRIB file and delegates to
    extract_point_series() for the actual extraction. Uses cfgrib/
    xarray, the same approach already built and validated for
    grib_to_csv."""
    import xarray as xr

    with xr.open_dataset(grib_path, engine="cfgrib") as ds:
        return extract_point_series(ds, lat, lon)


def save_cds_credentials(api_key: str, config_path=None) -> Path:
    """Writes ~/.cdsapirc with the given API key, in the exact 2-line
    format the official cdsapi client expects - so a key entered once
    via a QGIS parameter is picked up automatically on every future
    run without needing to be re-entered (get_cds_client()'s default
    cdsapi.Client() call reads this same file).

    config_path is overridable (used by tests to write to a temp
    location instead of the real home directory - never the real
    ~/.cdsapirc during a test run).

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
    Login. An explicit api_key overrides this for a single run if
    provided, but the default and recommended path is the config
    file."""
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


def default_cds_retrieve(
    dataset: str, request: dict, target_path, api_key: str | None = None
) -> None:
    """Real CDS retrieval via the official cdsapi client for the
    SUBMIT step only - the actual file DOWNLOAD is done manually with
    requests, not via cdsapi's own Result.download()/_download().

    HISTORY, kept because it explains why this looks the way it does:
    two separate live attempts (a chained .download() call, then the
    3-argument retrieve(..., target) form) both failed identically
    with "'NoneType' object has no attribute 'write'". Tracing cdsapi's
    own source directly showed both calling conventions internally
    reach the exact same download code - so switching between them
    could never have fixed anything. The real bug lives inside
    cdsapi's own download internals (possibly version-specific).

    FIX: stop depending on cdsapi's own download() entirely. Submit
    the request via client.retrieve(dataset, request) with NO target
    (so cdsapi never attempts its own download), then download the
    result's file manually via download_result_manually() - the same
    URL cdsapi's own downloader would have used, just with the
    download loop fully under this code's own control.
    """
    client = get_cds_client(api_key)
    result = client.retrieve(dataset, request)
    download_result_manually(result, target_path)


def download_result_manually(
    result, target_path, chunk_size: int = 1024 * 1024
) -> None:
    """Downloads a completed CDS request's result file directly via
    requests, bypassing cdsapi's own Result.download()/_download()
    entirely - see default_cds_retrieve()'s docstring for why. Streams
    to disk in chunks rather than loading the whole file into memory
    (ERA5 GRIB responses can be tens of MB even for a small area/short
    date range). Raises a clear, specific error naming the actual
    byte-count mismatch or HTTP status, so a future failure here is
    immediately diagnosable from the QGIS log alone."""
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
            f"Download incomplete: got {total} byte(s), expected "
            f"{expected_size} byte(s) from {url}. The request itself succeeded "
            f"(this is a transfer issue, not a request-construction one) - "
            f"retrying the same run is the appropriate next step."
        )


def to_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def fetch_era5_point(
    lat: float,
    lon: float,
    start_date=None,
    end_date=None,
    product: str = "era5",
    years_per_chunk: int = 1,
    cds_retrieve_fn: Callable | None = None,
    read_fn: Callable = read_grib_response,
    progress_callback: Callable[[int], None] | None = None,
    work_dir=".",
    api_key: str | None = None,
    save_credentials: bool = True,
    credentials_path=None,
) -> Era5Result:
    """Orchestrates fetching every year-chunk in [start_date, end_date]
    for the chosen product, submitting one CDS request per chunk.
    cds_retrieve_fn and read_fn are injectable so the orchestration
    logic (chunking, per-chunk error isolation, deaccumulation,
    DataFrame assembly) can be fully unit-tested without any real CDS
    access or credentials.

    If api_key is provided, it's SAVED to ~/.cdsapirc (see
    save_cds_credentials) before being used for this run, so a key
    entered once via a QGIS parameter is picked up automatically on
    every future run - the key lives only in the standard cdsapi
    config file, never in this tool's own parameters/state. Set
    save_credentials=False to use an api_key for this run only,
    without persisting it.
    """
    if product not in DATASETS:
        raise ValueError(
            f"Unknown product {product!r} - expected one of {list(DATASETS)}"
        )

    start = to_date(start_date) if start_date else ERA5_RECORD_START
    end = to_date(end_date) if end_date else date.today()
    year_chunks = chunk_years(start, end, years_per_chunk)
    area = point_to_area(lat, lon)

    if api_key and save_credentials:
        save_cds_credentials(api_key, config_path=credentials_path)
        # The key is now in ~/.cdsapirc - let the default cdsapi.Client()
        # pick it up from there rather than passing it through explicitly,
        # so this run and every future run go through the identical path.
        api_key = None

    if cds_retrieve_fn is None:

        def _default_cds_retrieve_fn(dataset, request, target_path, apikey=api_key):
            # `apikey` below is a confirmed ruff false positive on this exact
            # pattern (isolated via bisection: an AST-equivalent reconstruction
            # of this same function does not trigger it). `apikey` is this
            # function's own parameter, defined on the line above. Runtime
            # correctness is verified by
            # test_fetch_era5_point_default_cds_retrieve_fn_passes_api_key.
            default_cds_retrieve(
                dataset,
                request,
                target_path,
                api_key=apikey,  # noqa: F821
            )

        cds_retrieve_fn = _default_cds_retrieve_fn

    warnings = []
    all_rows = []
    all_anomalies = []
    n = len(year_chunks)
    total_missing_raw = 0

    for i, years in enumerate(year_chunks):
        dataset, request = build_request(product, years, area)
        target_path = Path(work_dir) / f"era5_{product}_{years[0]}_{years[-1]}.grib"
        try:
            cds_retrieve_fn(dataset, request, target_path)
            actual_path = target_path
            if is_zip_file(target_path):
                actual_path = extract_grib_from_zip(target_path, Path(work_dir))
            raw_df = read_fn(actual_path, lat, lon)
            deacc_df, anomaly_rows = deaccumulate(raw_df)
            total_missing_raw += deacc_df.attrs.get("n_missing_raw", 0)

            # Keep only rows within the actually-requested date range - a
            # year-chunk request returns the WHOLE year(s), even if
            # start/end only cover part of the first/last one. Applied to
            # the anomalies too, not just the main output - the audit
            # output must describe what's actually in the CSV, not a
            # superset of it that happened to pass through de-
            # accumulation as a side effect of year-based chunking.
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
            f"{total_missing_raw} raw precipitation value(s) were genuinely "
            f"missing (NaN) in the source GRIB data - written as an empty cell "
            f"in the final output for that hour only (not zero or dropped, and "
            f"no longer affecting neighbouring hours, since values are no "
            f"longer differenced against each other). This is a gap in the "
            f"source data itself, not something this tool can fill in."
        )

    if len(anomalies_df):
        vals = anomalies_df["precipitation_mm_raw"]
        warnings.append(
            f"{len(anomalies_df)} value(s) were negative beyond the GRIB-packing "
            f"tolerance ({NEGATIVE_CLIP_TOLERANCE_MM} mm) - left as-is rather "
            f"than silently clipped, since this may indicate a real problem "
            f"rather than packing noise. A genuinely negative raw hourly "
            f"precipitation value from the source data is expected to be rare "
            f"- if this count is high, it's worth investigating rather than "
            f"assuming packing noise. Magnitude range: {vals.min():.3f} to "
            f"{vals.max():.3f} mm, median {vals.median():.3f} mm - see the "
            f"anomalies detail output for the full list with timestamps."
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
            f"{n_duplicates} duplicate ValidTime timestamp(s) were found across "
            f"chunk results and dropped (keeping one row per timestamp) - the "
            f"exact mechanism producing these duplicates (e.g. an overlap "
            f"between adjacent year-chunks) is not yet confirmed - this count "
            f"is reported so a pattern (e.g. clustering at year-chunk "
            f"boundaries) can be investigated from here."
        )

    # Keep the anomalies audit output consistent with what's actually in
    # the final CSV.
    if len(anomalies_df):
        n_anomalies_before_dedup = len(anomalies_df)
        anomalies_df = anomalies_df.drop_duplicates(subset="valid_time").reset_index(
            drop=True
        )
        n_anomaly_duplicates = n_anomalies_before_dedup - len(anomalies_df)
        if n_anomaly_duplicates:
            warnings.append(
                f"{n_anomaly_duplicates} duplicate anomaly row(s) were also found "
                f"and dropped from the anomalies audit output, for the same reason."
            )

    return Era5Result(
        dataframe=combined, warnings=warnings, product=product, anomalies=anomalies_df
    )
