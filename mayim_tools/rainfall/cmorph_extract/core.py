"""
Core logic for extracting NOAA CMORPH CDR (bias-adjusted, Climate Data
Record quality) precipitation at a single point.

ARCHITECTURE: unlike the Giovanni-based tools (imerg_extract,
merra2_extract), CMORPH is a NOAA product, not NASA/GES DISC, and has
no confirmed bulk time-series API. Access is via NOAA's Open Data
Dissemination (NODD) Program, which hosts CMORPH CDR as public,
UNAUTHENTICATED files on Azure Blob Storage - confirmed directly, not
assumed: a live, unauthenticated fetch of
https://noaacdr.blob.core.windows.net/precip-cmorph?restype=container&comp=list
returned a real Azure Blob List XML response with real filenames, no
login prompt. This is architecturally closer to chirps_extract (direct
per-file HTTP reads, no auth, concurrent fetching) than to the
Giovanni-based tools.

URL/FILENAME PATTERN - confirmed directly from that real listing, not
guessed (unlike CHIRPS's daily URL pattern, which needed a live-run
correction the first time):
    https://noaacdr.blob.core.windows.net/precip-cmorph/data/30min/8km/
        {year}/{month:02d}/{day:02d}/CMORPH_V1.0_ADJ_8km-30min_{YYYYMMDDHH}.nc
One file per HOUR (both an 00 and 01 entry were visible as separate
files for the same day), holding whatever timesteps are inside for
that hour - see below.

WHAT'S STILL UNCONFIRMED, honestly: no network access to this domain
was available from the environment this was built in (outside the
sandbox's allowed domain list, the same limitation every other real
endpoint in this project has had), so two things remain unconfirmed
until the first live run:
1. Whether each hourly file holds ONE timestep or TWO (the ":00" and
   ":30" half-hours implied by "30min" resolution) - handled
   defensively by reading however many time steps are actually
   present in the opened file, not assuming either count.
2. The exact NetCDF variable name for precipitation - handled
   defensively by trying several plausible candidate names, the same
   approach already used for IMERG's V06/V07 naming difference.

UNITS: CMORPH CDR's own documentation states the output variable is
"precipitation rate in mm/hour" for both the native 30-minute tier and
the hourly tier - each recorded value is already a rate in mm/hr for
whatever interval it represents, not a value needing further
deaccumulation (similar to MERRA-2, unlike ERA5). Output column is
PrecipitationMMHR, matching the naming convention already used for
IMERG and MERRA-2 - no separate depth column, matching Chris's explicit
preference on the MERRA-2 tool (leave the rate column as-is; a depth
conversion would need multiplying by the interval in hours - 0.5 for
the native 30-min tier - if ever wanted, not built in by default).

SCALE: one file per hour across the full 1998-present record is
roughly 245,000 files - the same order of magnitude as IMERG's granule
count, and handled the same way: concurrent fetching (ThreadPoolExecutor),
not a naive serial loop, mirroring the pattern already proven for
chirps_extract and imerg_extract's granule method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import requests

import functools
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

RECORD_START = date(1998, 1, 1)  # confirmed from multiple NOAA/NCEI sources

BASE_URL = "https://noaacdr.blob.core.windows.net/precip-cmorph/data/30min/8km"
# ts = YYYYMMDDHH, confirmed from a real listing
FILENAME_TEMPLATE = "CMORPH_V1.0_ADJ_8km-30min_{ts}.nc"

VARIABLE_NAME_CANDIDATES = (
    "cmorph",
    "precipitation",
    "precip",
    "rain",
    "cdr_precip",
)  # not confirmed - defensive
LAT_COORD_CANDIDATES = ("lat", "latitude")
LON_COORD_CANDIDATES = ("lon", "longitude")

DEFAULT_MAX_WORKERS = (
    8  # matches chirps_extract's/imerg_extract's default and reasoning
)


@dataclass
class FetchResult:
    dataframe: pd.DataFrame  # columns: Timestamp, PrecipitationMMHR
    warnings: list
    method: str = "direct_file"
    # matched-grid-cell reporting - see read_hour_file()/fetch_cmorph_timeseries()
    requested_lat: float | None = None
    requested_lon: float | None = None
    matched_lat: float | None = None
    matched_lon: float | None = None
    distance_km: float | None = None


def _approx_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """A simple planar approximation - good enough for reporting how far a
    requested point sits from the grid cell it was snapped to, not a
    geodesic-precision distance. Duplicated per-tool rather than shared,
    matching this suite's existing pattern of each plugin being
    independently self-contained."""
    import math

    lat_km = (lat2 - lat1) * 111.0
    lon_km = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(lat_km, lon_km)


def build_hour_url(dt: datetime) -> str:
    """Constructs the exact file URL for a given hour, using the
    pattern confirmed directly from a real Azure Blob List response -
    not guessed."""
    ts = dt.strftime("%Y%m%d%H")
    return (
        f"{BASE_URL}/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/"
        f"{FILENAME_TEMPLATE.format(ts=ts)}"
    )


def hourly_range(start: date, end: date) -> list:
    """Every hour timestamp from start 00:00 to end 23:00 inclusive.
    Pure function - fully unit-testable without network access."""
    if start > end:
        raise ValueError(f"start ({start}) is after end ({end})")
    current = datetime(start.year, start.month, start.day, 0)
    stop = datetime(end.year, end.month, end.day, 23)
    hours = []
    while current <= stop:
        hours.append(current)
        current += timedelta(hours=1)
    return hours


def find_variable_name(ds, candidates=VARIABLE_NAME_CANDIDATES) -> str:
    """Defensive lookup - the real CMORPH CDR variable name was not
    confirmed against a live file when this was built (no network
    access to the source domain from the build environment). Tries
    each candidate in turn; falls back to the first data variable in
    the dataset if none of the named candidates match, on the
    reasoning that a single-variable-per-file product (which CMORPH
    CDR's per-hour-per-tier file structure strongly suggests) likely
    has exactly one real data variable regardless of its exact name."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    data_var_names = list(ds.data_vars)
    if len(data_var_names) == 1:
        return data_var_names[0]
    raise ValueError(
        f"Could not identify the precipitation variable - none of {candidates} found, "
        f"and the dataset has {len(data_var_names)} data variables (not exactly 1 to "
        f"fall back on): {data_var_names}"
    )


def find_coord_name(ds, candidates) -> str:
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(
        f"Could not identify a coordinate matching any of {candidates}. "
        f"Available: {list(ds.coords)}"
    )


def read_hour_file(path_or_buffer, lat: float, lon: float) -> pd.DataFrame:
    """Reads one downloaded CMORPH hourly file, extracts the nearest-
    gridpoint value at EVERY timestep actually present in the file -
    deliberately not assuming exactly 1 or exactly 2 (the "30min"
    resolution tag implies 2 per hour-labelled file, but this was
    never confirmed against a live file - see module docstring).
    Returns columns: Timestamp, PrecipitationMMHR.
    """
    import xarray as xr

    with xr.open_dataset(path_or_buffer) as ds:
        var_name = find_variable_name(ds)
        lat_name = find_coord_name(ds, LAT_COORD_CANDIDATES)
        lon_name = find_coord_name(ds, LON_COORD_CANDIDATES)

        da = ds[var_name].sel(**{lat_name: lat, lon_name: lon}, method="nearest")
        matched_lat = float(da[lat_name].values)
        matched_lon = float(da[lon_name].values)

        time_coord = None
        for candidate in ("time", "Time"):
            if candidate in da.coords or candidate in da.dims:
                time_coord = candidate
                break

        rows = []
        if time_coord is not None and da[time_coord].ndim > 0:
            for t in da[time_coord].values:
                value = float(da.sel(**{time_coord: t}).values)
                rows.append({"Timestamp": pd.Timestamp(t), "PrecipitationMMHR": value})
        else:
            # single-timestep file (or no explicit time dimension) -
            # fall back to the file's own reference time if present,
            # otherwise this row's timestamp must come from the caller
            # (the hour this file was requested for)
            t = da[time_coord].values if time_coord is not None else None
            value = float(da.values)
            rows.append(
                {
                    "Timestamp": pd.Timestamp(t) if t is not None else None,
                    "PrecipitationMMHR": value,
                }
            )

    out = pd.DataFrame(rows)
    out.attrs["matched_lat"] = matched_lat
    out.attrs["matched_lon"] = matched_lon
    return out


def _build_session(pool_size: int) -> requests.Session:
    """Builds a requests.Session with a connection pool sized for the
    actual concurrency in use, reused across ALL hourly fetches.

    Found via a real performance question (how to cut a ~4-hour, 5-year
    run's time down): every fetch was previously a bare requests.get()
    call, opening a brand new TCP connection and negotiating a fresh
    TLS handshake for EVERY one of the ~43,872 files in a full-record
    run, even though nearly all of them hit the exact same host
    (noaacdr.blob.core.windows.net). A shared Session with a properly
    sized HTTPAdapter connection pool lets requests reuse already-
    established, already-negotiated connections via HTTP keep-alive
    instead - requests.Session is documented as thread-safe for
    concurrent use across threads (each thread's request checks out a
    pooled connection rather than mutating shared session state), so
    this is safe to share across the ThreadPoolExecutor's workers.

    pool_maxsize is set to the actual worker count, not left at
    requests' own default of 10 - with more concurrent workers than
    pooled connections, workers would otherwise queue and block
    waiting for a free connection, silently capping the real
    concurrency below what max_workers asked for.
    """
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_one_hour_file(
    url: str,
    lat: float,
    lon: float,
    fallback_timestamp: datetime,
    timeout: int = 60,
    session=None,
) -> pd.DataFrame:
    """Downloads one hourly file to a temp location and reads it -
    isolated into its own function so tests can substitute a fake
    version without touching the orchestration logic. No
    authentication - confirmed public access via NOAA's NODD program.

    session, if given, is a shared requests.Session reused across all
    hourly fetches for connection pooling (see _build_session) - falls
    back to a plain requests.get() when not provided, so existing
    callers/tests that don't pass one keep working unchanged."""
    import tempfile

    import requests

    getter = session.get if session is not None else requests.get
    resp = getter(url, timeout=timeout)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        f.write(resp.content)
        tmp_path = f.name

    try:
        df = read_hour_file(tmp_path, lat=lat, lon=lon)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if df["Timestamp"].isna().any():
        df["Timestamp"] = df["Timestamp"].fillna(pd.Timestamp(fallback_timestamp))
    return df


def fetch_cmorph_timeseries(
    lat: float,
    lon: float,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
    progress_callback: Callable[[int], None] | None = None,
    fetch_fn: Callable = fetch_one_hour_file,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> FetchResult:
    """Orchestrates CONCURRENT per-hour file fetches across the
    requested date range, with retry-with-backoff per file and
    per-file error isolation - the same pattern already proven for
    chirps_extract's per-date fetches and imerg_extract's granule
    method, adapted here since CMORPH has no bulk time-series API.

    fetch_fn is injectable so this orchestration logic can be fully
    unit-tested with a fake network call. max_workers=1 falls back to
    fully serial fetching.

    When fetch_fn is left as the default (a real run, not a test), a
    single shared requests.Session (see _build_session) is created
    once and reused for every hourly fetch, cutting per-request
    connection/TLS overhead - see fetch_one_hour_file's docstring.
    Injected fakes (used throughout the test suite) are left exactly
    as before, still called as fetch_fn(url, lat, lon, dt) with no
    session argument, so no existing test needs to change.
    """
    start = _to_date(start_date) if start_date else RECORD_START
    end = _to_date(end_date) if end_date else date.today()

    hours = hourly_range(start, end)
    warnings = []
    frames = []
    completed = 0
    completed_lock = threading.Lock()
    matched_lock = threading.Lock()
    matched = {"lat": None, "lon": None}

    if fetch_fn is fetch_one_hour_file:
        session = _build_session(pool_size=max(max_workers, 1))
        fetch_fn = functools.partial(fetch_one_hour_file, session=session)

    def _fetch_one(dt):
        url = build_hour_url(dt)
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                df = fetch_fn(url, lat, lon, dt)
                return df, []
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(retry_backoff_s * attempt)
        return None, [
            f"{dt.isoformat()}: failed after {max_retries} attempts "
            f"({url}): {last_error}"
        ]

    def _maybe_capture_matched(df):
        # captures the matched grid-cell coordinate exactly once, from
        # whichever fetch completes first - only real read_hour_file()
        # results carry these attrs (see that function); a fake fetch_fn
        # without them (the whole existing test suite) simply never
        # populates this, leaving matched_lat/matched_lon as None,
        # unchanged from before this feature existed.
        m_lat = df.attrs.get("matched_lat")
        m_lon = df.attrs.get("matched_lon")
        if m_lat is None:
            return
        with matched_lock:
            if matched["lat"] is None:
                matched["lat"] = m_lat
                matched["lon"] = m_lon

    if max_workers <= 1:
        results = [_fetch_one(dt) for dt in hours]
        for df, w in results:
            if df is not None:
                frames.append(df)
                _maybe_capture_matched(df)
            warnings.extend(w)
            completed += 1
            if progress_callback:
                progress_callback(int(100 * completed / max(len(hours), 1)))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, dt): dt for dt in hours}
            for future in as_completed(futures):
                df, w = future.result()
                if df is not None:
                    frames.append(df)
                    _maybe_capture_matched(df)
                warnings.extend(w)
                with completed_lock:
                    completed += 1
                    pct = int(100 * completed / max(len(hours), 1))
                if progress_callback:
                    progress_callback(pct)

    if not frames:
        combined = pd.DataFrame(columns=["Timestamp", "PrecipitationMMHR"])
    else:
        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.drop_duplicates(subset="Timestamp")
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

    matched_lat, matched_lon = matched["lat"], matched["lon"]
    distance_km = None
    if matched_lat is not None:
        distance_km = _approx_distance_km(lat, lon, matched_lat, matched_lon)
        grid_warning = (
            f"Requested point ({lat}, {lon}) was matched to the nearest grid cell "
            f"({matched_lat}, {matched_lon}), approximately {distance_km:.2f} km away."
        )
        warnings.insert(0, grid_warning)

    n_expected_files = len(hours)
    n_failed = sum(1 for w in warnings if "failed after" in w)
    if n_failed:
        warnings.append(
            f"{n_failed} of {n_expected_files} hourly file(s) failed after "
            f"{max_retries} attempts each - see individual entries above for which "
            f"hours and why."
        )

    n_missing_values = (
        int(combined["PrecipitationMMHR"].isna().sum()) if len(combined) else 0
    )
    if n_missing_values:
        pct = 100 * n_missing_values / len(combined)
        warnings.append(
            f"{n_missing_values} of {len(combined)} timesteps ({pct:.2f}%) had no "
            f"value at this point - the file for that hour was fetched and read "
            f"successfully, but the specific grid cell's value was missing/masked in "
            f"the source data (written as an empty CSV cell, not zero or dropped). "
            f"This is a plausible real characteristic of a satellite-derived "
            f"precipitation product (occasional genuine data gaps), not necessarily "
            f"a bug - but not confirmed as such either; worth checking whether the "
            f"missing timestamps cluster around specific dates (suggesting a real "
            f"satellite/processing gap) or look scattered/random before treating "
            f"this as expected."
        )

    return FetchResult(
        dataframe=combined,
        warnings=warnings,
        method="direct_file",
        requested_lat=lat,
        requested_lon=lon,
        matched_lat=matched_lat,
        matched_lon=matched_lon,
        distance_km=distance_km,
    )


def _to_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()
