"""
Core logic for extracting PERSIANN precipitation at a single point, in
either of two products selectable by the caller:

- "daily" (PERSIANN-CDR): 0.25 deg, daily, 1983-01-01 to present.
  Accessed via NOAA NCEI's own ERDDAP server (griddap protocol) -
  confirmed directly: dataset ID `cdr_persiann_by_time_lon_lat`,
  variable `precipitation`, time_coverage 1983-01-01 to (at research
  time) 2025-12-31, read off the dataset's own live ERDDAP metadata
  page. A single query can return the full requested time range in
  one request (chunked here only as a conservative safety margin, the
  same reasoning as era5_extract's year-chunking) - genuinely fast
  compared to the 3-hourly product below.

- "3hourly" (PERSIANN-CCS-CDR): 0.04 deg (~4km), 3-hourly, 1983-01-01
  to present. Accessed via CHRS's own HTTPS file server - confirmed
  directly via a live, real directory listing (NOT the FTP path
  mentioned in older papers): plain HTTPS at
  persiann.eng.uci.edu/CHRSdata/PCCSCDR/3hrly/, filename pattern
  PCCSCDR3h{YYMMDDHH}.bin.gz confirmed against real listed files, all
  files in ONE FLAT DIRECTORY (no year/month subdirectories, unlike
  CMORPH). Files are GZIP-COMPRESSED flat binary float32 arrays (3000
  rows x 9000 cols, 0.04 deg, 60N-60S, longitude 0-360) - confirmed
  directly from CHRS's own PCCSCDR_readme.txt. No bulk time-series API
  exists for this product, so - like CMORPH - this uses direct
  concurrent per-file fetching, one file per 3-hour period. Because
  the files are gzip-compressed, HTTP Range requests cannot be used to
  read a single point without downloading the whole file first (gzip's
  compressed-byte-to-uncompressed-byte mapping isn't fixed) - each
  file must be downloaded whole and decompressed, though at 2.5-9MB
  real observed sizes this is still modest per file.

BOTH products use a 0-360 degree longitude convention (confirmed
directly for both from their own metadata/documentation) - a standard
-180/180 input longitude is converted internally.

CONNECTION REUSE: applied here from the start (not added later as a
fix, the way it was for cmorph_extract) - a shared, pooled
requests.Session is used for the 3-hourly product's per-file fetches,
for the same reason: nearly every request hits the same host, so
reusing already-negotiated connections avoids paying a fresh TCP+TLS
handshake per file across what can be well over 100,000 files for a
full-record extraction.

WHAT'S NOT CONFIRMED: the exact PERSIANN-CDR ERDDAP CSV/JSON row
count/format was inferred from ERDDAP's own general, standardized
JSON response structure (documented consistently across every ERDDAP
installation, not specific to NCEI's), not confirmed against a live
response from this specific dataset when this was built - handled
defensively (parses the JSON structure directly rather than assuming
a fixed row-skip count, which would be the fragile alternative). The
exact fill/missing-value convention for PERSIANN-CCS-CDR's raw binary
format was not confirmed either - see find_missing_value_candidates.
"""

from __future__ import annotations

import functools
import gzip
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import requests

RECORD_START = date(
    1983, 1, 1
)  # confirmed for both products from NCEI/CHRS documentation

# --- Daily PERSIANN-CDR (direct Azure NODD file access - see module note below) ---
# Confirmed directly: this container is listed by name on Microsoft's own
# NODD/Azure CDR page, alongside CMORPH's (precip-cmorph), which is
# already proven reliable in this suite.
AZURE_DAILY_BASE_URL = "https://noaacdr.blob.core.windows.net/precip-persiann"
AZURE_DAILY_FILENAME_RE = re.compile(
    r"PERSIANN-CDR_v\d+r\d+_(\d{8})_c\d{8}\.nc$"
)  # confirmed directly against a real blob listing
# "precipitation" confirmed from this dataset's own ERDDAP metadata
# (cdr_variable=precipitation); others included defensively.
PERSIANN_DAILY_VARIABLE_CANDIDATES = (
    "precipitation",
    "precip",
    "prcp",
)

# --- 3-Hourly PERSIANN-CCS-CDR (direct file access) ---
CCS_BASE_URL = "https://persiann.eng.uci.edu/CHRSdata/PCCSCDR/3hrly"
CCS_FILENAME_TEMPLATE = (
    "PCCSCDR3h{ts}.bin.gz"  # confirmed directly from a real directory listing
)
CCS_GRID_ROWS = 3000
CCS_GRID_COLS = 9000
CCS_GRID_RES_DEG = 0.04
CCS_GRID_TOP_LAT = (
    59.98  # centre of the northernmost row - confirmed from CHRS's own readme
)
CCS_GRID_LEFT_LON = (
    0.02  # centre of the westernmost column - confirmed from CHRS's own readme
)

DEFAULT_MAX_WORKERS = 8

# NOTE ON THE DAILY PRODUCT'S ACCESS METHOD: this was originally built
# against NOAA NCEI's own ERDDAP server, since a single griddap query
# can in principle answer a whole date range in one request - see this
# module's git history / dev/README.md for that version. Two live
# attempts (with a raised timeout and a proper browser User-Agent
# added between them) both failed identically: a read timeout at the
# FULL configured timeout, on the smallest possible query (a single
# day, single point). Two increasingly generous fixes producing the
# exact same full-timeout failure is a real, repeated pattern, not an
# occasional blip - rather than continue tuning parameters against an
# approach that has now failed consistently, this switches to direct
# per-file access via NOAA's Open Data Dissemination (NODD) program on
# Azure Blob Storage - the same proven-reliable architecture already
# working for cmorph_extract (confirmed: "Precipitation - PERSIANN"
# (container precip-persiann) is listed directly alongside CMORPH's
# own container on Microsoft's own NODD/Azure CDR page).
#
# Confirmed directly against a real, live blob listing before writing
# this code (not guessed): files sit under `data/{year}/`, one file
# per day, named `PERSIANN-CDR_v01r01_{YYYYMMDD}_c{creation_date}.nc`.
# The `_c{creation_date}` suffix is NOT predictable from the data date
# alone (e.g. every file in the 1983 listing checked so far carries
# `_c20140523`, a batch reprocessing date, not a per-file one - and
# there is no guarantee every year's batch date is uniform) - this
# rules out constructing the exact filename directly the way CMORPH's
# hourly files could be. Instead, each requested YEAR's folder is
# listed ONCE (Azure's Blob List API, filtered by prefix), and the
# real filenames returned are used to build a date-to-URL lookup -
# amortising the "unpredictable suffix" problem across a whole year's
# ~365 files per listing call, not one extra request per day.


@dataclass
class FetchResult:
    dataframe: pd.DataFrame  # columns: Date/Timestamp, PrecipitationMM
    warnings: list
    product: str = ""


def to_0_360(lon: float) -> float:
    """Both PERSIANN products use a 0-360 degree longitude convention,
    confirmed directly for each from their own metadata/documentation
    - not a shared assumption carried over unverified from one to the
    other."""
    return lon % 360.0


def _build_session(pool_size: int) -> requests.Session:
    """Shared, pooled connection for per-file fetches (both PERSIANN
    products use this) - applied here from the start, unlike
    cmorph_extract where this was added only after a live run exposed
    the cost of a fresh connection per file. See module docstring."""
    import requests
    from requests.adapters import HTTPAdapter

    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ----------------------------------------------------------------------
# Daily PERSIANN-CDR via direct Azure NODD file access
# ----------------------------------------------------------------------


def build_year_list_url(year: int) -> str:
    """Azure Blob List API query for one year's worth of daily files,
    filtered by prefix so the response covers only that year (~365
    files), not the whole container's multi-decade history."""
    return f"{AZURE_DAILY_BASE_URL}?restype=container&comp=list&prefix=data/{year}/"


def parse_azure_blob_list(xml_text: str) -> tuple:
    """Parses an Azure Blob List XML response into a {YYYYMMDD: url}
    dict, matching filenames against the pattern confirmed directly
    from a real listing. Returns (date_to_url, warnings).

    Checks for a NextMarker element (Azure's pagination-continuation
    signal) and warns if present, rather than silently returning a
    truncated result - every year's listing actually inspected while
    building this (~365 files) fit in a single call comfortably under
    Azure's per-call limit, so pagination was never exercised for
    real, but a warning here is cheap insurance against a future year
    that doesn't fit."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    date_to_url = {}
    for blob in root.iter("Blob"):
        name_el = blob.find("Name")
        url_el = blob.find("Url")
        if name_el is None or url_el is None or name_el.text is None:
            continue
        match = AZURE_DAILY_FILENAME_RE.search(name_el.text)
        if match:
            date_to_url[match.group(1)] = url_el.text

    warnings = []
    next_marker = root.find("NextMarker")
    if next_marker is not None and (next_marker.text or "").strip():
        warnings.append(
            f"Azure blob listing was paginated (NextMarker present) - "
            f"only {len(date_to_url)} file(s) were parsed from this "
            f"response, and some dates for this year may be missing "
            f"as a result. Pagination is not implemented (every year "
            f"checked while building this tool fit in a single "
            f"listing call)."
        )

    return date_to_url, warnings


def fetch_year_listing(year: int, session=None, timeout: int = 60) -> tuple:
    """Fetches and parses one year's blob listing - isolated into its
    own function so tests can substitute a fake version without
    touching the orchestration logic."""
    import requests

    getter = session.get if session is not None else requests.get
    resp = getter(build_year_list_url(year), timeout=timeout)
    resp.raise_for_status()
    return parse_azure_blob_list(resp.text)


def read_persiann_daily_file(raw_bytes: bytes, lat: float, lon: float) -> float:
    """Reads a single PERSIANN-CDR daily NetCDF file (downloaded from
    Azure), extracts the nearest-gridpoint value. Variable name
    'precipitation' confirmed directly from this dataset's own ERDDAP
    metadata; other candidates included defensively in case this
    per-file NetCDF's internal naming differs from the ERDDAP-exposed
    name (never directly confirmed against an opened file, the same
    honest gap this plugin already had for the 3-hourly product's
    missing-value convention)."""
    import tempfile

    import xarray as xr

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        f.write(raw_bytes)
        tmp_path = f.name

    try:
        with xr.open_dataset(tmp_path) as ds:
            var_name = None
            for candidate in PERSIANN_DAILY_VARIABLE_CANDIDATES:
                if candidate in ds.data_vars:
                    var_name = candidate
                    break
            if var_name is None:
                data_vars = list(ds.data_vars)
                if len(data_vars) == 1:
                    var_name = data_vars[0]
                else:
                    raise ValueError(
                        f"Could not identify the precipitation "
                        f"variable - none of "
                        f"{PERSIANN_DAILY_VARIABLE_CANDIDATES} found, "
                        f"and {len(data_vars)} data variables "
                        f"present (not exactly 1 to fall back on): "
                        f"{data_vars}"
                    )

            lat_name = "lat" if "lat" in ds.coords else "latitude"
            lon_name = "lon" if "lon" in ds.coords else "longitude"
            lon_360 = to_0_360(lon)

            da = ds[var_name].sel(
                **{lat_name: lat, lon_name: lon_360}, method="nearest"
            )
            value = float(np.asarray(da.values).squeeze())
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return value


def fetch_one_daily_file(
    url: str, lat: float, lon: float, fallback_date, timeout: int = 60, session=None
) -> pd.DataFrame:
    """Downloads and reads one daily PERSIANN-CDR file - isolated into
    its own function so tests can substitute a fake version. session,
    if given, is a shared requests.Session for connection pooling."""
    import requests

    getter = session.get if session is not None else requests.get
    resp = getter(url, timeout=timeout)
    resp.raise_for_status()

    value = read_persiann_daily_file(resp.content, lat=lat, lon=lon)
    return pd.DataFrame(
        {"Date": [pd.Timestamp(fallback_date)], "PrecipitationMM": [value]}
    )


def fetch_persiann_daily(
    lat: float,
    lon: float,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
    progress_callback: Callable[[int], None] | None = None,
    fetch_fn: Callable = fetch_one_daily_file,
    listing_fn: Callable = fetch_year_listing,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> FetchResult:
    """Orchestrates the daily product's extraction: lists each
    requested YEAR's folder once (not once per day), builds a
    date-to-URL lookup from the real filenames returned, then fetches
    each requested day's file CONCURRENTLY - the same architecture
    already proven for cmorph_extract and this plugin's own 3-hourly
    product, applied here after two live ERDDAP failures. See the
    module-level note above for the full reasoning.

    fetch_fn/listing_fn are injectable so this orchestration logic can
    be fully unit-tested with fake network calls. max_workers=1 falls
    back to fully serial fetching.
    """
    start = _to_date(start_date) if start_date else RECORD_START
    end = _to_date(end_date) if end_date else date.today()

    warnings = []

    if fetch_fn is fetch_one_daily_file:
        session = _build_session(pool_size=max(max_workers, 1))
        fetch_fn = functools.partial(fetch_one_daily_file, session=session)
        listing_fn = functools.partial(fetch_year_listing, session=session)

    all_dates = []
    cur = start
    while cur <= end:
        all_dates.append(cur)
        cur += timedelta(days=1)

    years_needed = sorted(set(d.year for d in all_dates))
    date_to_url = {}
    for year in years_needed:
        try:
            year_dict, year_warnings = listing_fn(year)
            date_to_url.update(year_dict)
            warnings.extend(year_warnings)
        except Exception as e:
            warnings.append(f"Could not list year {year} (Azure blob listing): {e}")

    dates_to_fetch = [d for d in all_dates if d.strftime("%Y%m%d") in date_to_url]
    missing_dates = [d for d in all_dates if d.strftime("%Y%m%d") not in date_to_url]
    if missing_dates:
        warnings.append(
            f"{len(missing_dates)} of {len(all_dates)} requested "
            f"date(s) had no matching file in the Azure listing "
            f"(first: {missing_dates[0].isoformat()}) - a genuine "
            f"gap in the archive, a listing that failed to fetch "
            f"for that year (see above), or a date outside the "
            f"archive's actual coverage."
        )

    frames = []
    completed = 0
    completed_lock = threading.Lock()

    def _fetch_one(d):
        url = date_to_url[d.strftime("%Y%m%d")]
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return fetch_fn(url, lat, lon, d), []
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(retry_backoff_s * attempt)
        return None, [
            f"{d.isoformat()}: failed after {max_retries} attempts "
            f"({url}): {last_error}"
        ]

    if max_workers <= 1:
        results = [_fetch_one(d) for d in dates_to_fetch]
        for df, w in results:
            if df is not None:
                frames.append(df)
            warnings.extend(w)
            completed += 1
            if progress_callback:
                progress_callback(int(100 * completed / max(len(dates_to_fetch), 1)))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, d): d for d in dates_to_fetch}
            for future in as_completed(futures):
                df, w = future.result()
                if df is not None:
                    frames.append(df)
                warnings.extend(w)
                with completed_lock:
                    completed += 1
                    pct = int(100 * completed / max(len(dates_to_fetch), 1))
                if progress_callback:
                    progress_callback(pct)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.drop_duplicates(subset="Date")
            .sort_values("Date")
            .reset_index(drop=True)
        )
    else:
        combined = pd.DataFrame(columns=["Date", "PrecipitationMM"])

    n_failed = sum(1 for w in warnings if "failed after" in w)
    if n_failed:
        warnings.append(
            f"{n_failed} of {len(dates_to_fetch)} daily file(s) "
            f"failed after {max_retries} attempts each - see "
            f"individual entries above."
        )

    n_missing = int(combined["PrecipitationMM"].isna().sum()) if len(combined) else 0
    if n_missing:
        pct = 100 * n_missing / len(combined)
        warnings.append(
            f"{n_missing} of {len(combined)} day(s) ({pct:.2f}%) had "
            f"no value at this point - written as an empty cell, "
            f"not zero or dropped."
        )

    return FetchResult(
        dataframe=combined, warnings=warnings, product="persiann_cdr_daily"
    )


# ----------------------------------------------------------------------
# 3-Hourly PERSIANN-CCS-CDR via direct gzip-binary file access
# ----------------------------------------------------------------------


def build_ccs_url(dt: datetime) -> str:
    """Constructs the exact file URL for a given 3-hour period, using
    the pattern confirmed directly from a real directory listing -
    all files sit in ONE FLAT directory (no year/month subdirectories,
    unlike CMORPH's per-day folder structure)."""
    yy = dt.year % 100
    ts = f"{yy:02d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}"
    return f"{CCS_BASE_URL}/{CCS_FILENAME_TEMPLATE.format(ts=ts)}"


def three_hourly_range(start: date, end: date) -> list:
    """Every 3-hour period timestamp (00, 03, 06, ..., 21) from start
    to end inclusive. Pure function - fully unit-testable."""
    if start > end:
        raise ValueError(f"start ({start}) is after end ({end})")
    current = datetime(start.year, start.month, start.day, 0)
    stop = datetime(end.year, end.month, end.day, 21)
    periods = []
    while current <= stop:
        periods.append(current)
        current += timedelta(hours=3)
    return periods


def read_ccs_file(raw_bytes: bytes, lat: float, lon: float) -> float:
    """Decompresses and reads a single value from a PERSIANN-CCS-CDR
    flat binary file - grid layout confirmed directly from CHRS's own
    PCCSCDR_readme.txt: 3000 rows x 9000 cols, 0.04 deg resolution,
    row-major, first row centred at 59.98N, first column centred at
    0.02E (0-360 convention). No file header/metadata - the exact
    byte offset for any point is computed directly from this
    documented layout, not read from the file itself."""
    import numpy as np

    decompressed = gzip.decompress(raw_bytes)
    expected_bytes = CCS_GRID_ROWS * CCS_GRID_COLS * 4
    if len(decompressed) != expected_bytes:
        raise ValueError(
            f"Decompressed file is {len(decompressed)} bytes, "
            f"expected exactly {expected_bytes} "
            f"({CCS_GRID_ROWS}x{CCS_GRID_COLS} float32) - the grid "
            f"layout may not match what was documented, or the "
            f"file is corrupt/truncated."
        )

    grid = np.frombuffer(decompressed, dtype="<f4").reshape(
        CCS_GRID_ROWS, CCS_GRID_COLS
    )

    lon_360 = to_0_360(lon)
    row = round((CCS_GRID_TOP_LAT - lat) / CCS_GRID_RES_DEG)
    col = round((lon_360 - CCS_GRID_LEFT_LON) / CCS_GRID_RES_DEG)
    row = min(max(row, 0), CCS_GRID_ROWS - 1)
    col = min(max(col, 0), CCS_GRID_COLS - 1)

    return float(grid[row, col])


MISSING_VALUE_CANDIDATES = (
    -9999.0,
    -999.0,
    -1.0,
)  # not confirmed - defensive, see module docstring


def fetch_one_ccs_file(
    url: str,
    lat: float,
    lon: float,
    fallback_timestamp: datetime,
    timeout: int = 60,
    session=None,
) -> pd.DataFrame:
    """Downloads and reads one 3-hourly CCS-CDR file - isolated into
    its own function so tests can substitute a fake version. session,
    if given, is a shared requests.Session for connection pooling
    (see _build_session) - falls back to plain requests.get() when
    not provided, so tests injecting a fake fetch_fn are unaffected."""
    import requests

    getter = session.get if session is not None else requests.get
    resp = getter(url, timeout=timeout)
    resp.raise_for_status()

    value = read_ccs_file(resp.content, lat=lat, lon=lon)
    if value in MISSING_VALUE_CANDIDATES:
        value = float("nan")

    return pd.DataFrame(
        {"Timestamp": [pd.Timestamp(fallback_timestamp)], "PrecipitationMM": [value]}
    )


def fetch_persiann_3hourly(
    lat: float,
    lon: float,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
    progress_callback: Callable[[int], None] | None = None,
    fetch_fn: Callable = fetch_one_ccs_file,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> FetchResult:
    """Orchestrates CONCURRENT per-3-hour-file fetches - the same
    architecture already proven for cmorph_extract, since this product
    has no bulk time-series API either. A full-record (1983-present)
    extraction involves on the order of 125,000 individual files -
    comparable in scale to CMORPH's ~245,000, though each file is
    somewhat larger (2.5-9MB observed vs CMORPH's ~1.8MB)."""
    start = _to_date(start_date) if start_date else RECORD_START
    end = _to_date(end_date) if end_date else date.today()

    periods = three_hourly_range(start, end)
    warnings = []
    frames = []
    completed = 0
    completed_lock = threading.Lock()

    if fetch_fn is fetch_one_ccs_file:
        session = _build_session(pool_size=max(max_workers, 1))
        fetch_fn = functools.partial(fetch_one_ccs_file, session=session)

    def _fetch_one(dt):
        url = build_ccs_url(dt)
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                return fetch_fn(url, lat, lon, dt), []
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(retry_backoff_s * attempt)
        return None, [
            f"{dt.isoformat()}: failed after {max_retries} attempts "
            f"({url}): {last_error}"
        ]

    if max_workers <= 1:
        results = [_fetch_one(dt) for dt in periods]
        for df, w in results:
            if df is not None:
                frames.append(df)
            warnings.extend(w)
            completed += 1
            if progress_callback:
                progress_callback(int(100 * completed / max(len(periods), 1)))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, dt): dt for dt in periods}
            for future in as_completed(futures):
                df, w = future.result()
                if df is not None:
                    frames.append(df)
                warnings.extend(w)
                with completed_lock:
                    completed += 1
                    pct = int(100 * completed / max(len(periods), 1))
                if progress_callback:
                    progress_callback(pct)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.drop_duplicates(subset="Timestamp")
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
    else:
        combined = pd.DataFrame(columns=["Timestamp", "PrecipitationMM"])

    n_failed = sum(1 for w in warnings if "failed after" in w)
    if n_failed:
        warnings.append(
            f"{n_failed} of {len(periods)} 3-hourly file(s) failed after {max_retries} "
            f"attempts each - see individual entries above."
        )

    n_missing = int(combined["PrecipitationMM"].isna().sum()) if len(combined) else 0
    if n_missing:
        pct = 100 * n_missing / len(combined)
        warnings.append(
            f"{n_missing} of {len(combined)} timesteps ({pct:.2f}%) "
            f"had a value matching one of the candidate missing/fill "
            f"values {MISSING_VALUE_CANDIDATES} (not confirmed "
            f"against real CHRS documentation - a defensive guess) "
            f"- written as an empty cell."
        )

    return FetchResult(
        dataframe=combined, warnings=warnings, product="persiann_ccs_cdr_3hourly"
    )


def _to_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()
