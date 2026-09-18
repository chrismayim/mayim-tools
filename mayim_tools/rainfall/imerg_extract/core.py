"""
Core logic for extracting an IMERG Final Run half-hourly precipitation
time series at a single point.

TWO METHODS, because the "full available record" is a genuinely hard
scale problem:

    IMERG Final Run half-hourly has run continuously since 2000-06-01
    at 48 granules/day - roughly 455,000 individual files for the full
    record. Downloading and opening each one individually would take
    somewhere on the order of many hours to days for the full record -
    the same class of mistake as the original GRIB-to-CSV tool's GDAL
    approach before it was rebuilt around eccodes.

1. fetch_giovanni_timeseries() - PRIMARY/RECOMMENDED. Uses NASA GES
   DISC's official "Giovanni in the Cloud: Time Series Service" REST
   API, purpose-built for exactly this - single-point, long-time-
   series extraction - with no need to touch individual granules at
   all. Chunks the requested date range into manageable pieces
   (default: 1 year) as a robustness measure against unknown server-
   side response-size limits.

   PERFORMANCE (aligned with chirps_extract's v0.2 optimization):
   chunks are fetched CONCURRENTLY, not serially - each chunk is an
   independent HTTP request to Giovanni, exactly the same latency-
   bound, embarrassingly-parallel shape as CHIRPS's per-date-file
   fetches. Same ThreadPoolExecutor approach, same reasoning: network
   wait time is what's being overlapped, not request processing
   itself. Default max_workers=8, matching CHIRPS's default and
   the same "moderate, not maximal" reasoning - fast enough to matter,
   conservative enough to stay a reasonable citizen of a shared NASA
   service.

2. fetch_granule_based_timeseries() - FALLBACK, for short windows only.
   Mirrors the original sample script's approach (earthaccess search
   + download + xr.open_dataset(..., group="Grid") + nearest-point
   selection) but on individual half-hourly granules. Refuses to run
   above a granule-count threshold unless explicitly forced - NOT
   suitable for the full multi-decade record.

CREDENTIALS (aligned with era5_extract's v0.2 "enter once, remembered"
pattern): get_earthdata_token()/the granule-based path both read
~/.netrc by default. save_earthdata_credentials() lets a QGIS
parameter entry be written to .netrc automatically on first use, the
same UX as ERA5's CDS API Key parameter - BUT .netrc is a shared,
multi-service file by convention (unlike .cdsapirc, which exists
solely for one purpose), so saving here must preserve any OTHER
machine entries already present, only adding/replacing the
urs.earthdata.nasa.gov entry specifically - never a blind overwrite.

IMPORTANT - things verified vs assumed (see dev/README.md for detail
and sources): the Giovanni API endpoint, its single-point restriction,
the 2000-06-01 record start, and the V07 "precipitation" field rename
are all confirmed against current NASA/GES DISC documentation. The
EXACT Giovanni variable-ID string for V07 and the EXACT response CSV
structure are NOT independently verified against a live response -
both are implemented defensively (configurable variable ID; tolerant
CSV parsing) specifically so a wrong assumption is a small, obvious
fix rather than a deep rewrite.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

RECORD_START = "2000-06-01"  # confirmed: IMERG Final Run's actual record start
DEFAULT_GIOVANNI_VARIABLE = (
    "GPM_3IMERGHH_07_precipitation"  # ASSUMED - see module docstring
)
DEFAULT_SHORT_NAME = "GPM_3IMERGHH"
DEFAULT_VERSION = "07"
# V07 renamed precipitationCal -> precipitation; try the new name first,
# fall back to the old one for compatibility with V06 granules.
VARIABLE_NAME_CANDIDATES = ("precipitation", "precipitationCal")
GIOVANNI_TIMESERIES_URL = "https://api.giovanni.earthdata.nasa.gov/timeseries"

# Granule-based fallback: refuse by default above this many granules
# (roughly 2 months at 48/day) unless explicitly forced.
GRANULE_COUNT_WARN_THRESHOLD = 3000

DEFAULT_MAX_WORKERS = 8  # matches chirps_extract's default and reasoning

EARTHDATA_NETRC_MACHINE = "urs.earthdata.nasa.gov"


@dataclass
class FetchResult:
    dataframe: pd.DataFrame  # columns: Timestamp, PrecipitationMMHR
    warnings: list
    method: str


def _split_date_range(start: str, end: str, chunk_months: int) -> list:
    """Splits [start, end] into contiguous, non-overlapping chunks of
    approximately chunk_months each. Pure function - no I/O - fully
    unit-testable without network access."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    if start_dt > end_dt:
        raise ValueError(f"start ({start}) is after end ({end})")

    chunks = []
    cur = start_dt
    while cur <= end_dt:
        nxt = min(
            cur + pd.DateOffset(months=chunk_months) - pd.Timedelta(seconds=1), end_dt
        )
        chunks.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt + pd.Timedelta(seconds=1)
    return chunks


def _parse_giovanni_csv(text: str) -> pd.DataFrame:
    """Parser for the Giovanni time-series CSV response.

    REDESIGNED after a second real live response revealed the
    previous approach's hypothesis was wrong, not just incomplete.
    The actual response has a METADATA PREAMBLE before the real data,
    formatted as plain (uncommented, so '#'-stripping doesn't touch
    it) 'key,value' pairs - prod_name, doi, param_short_name,
    param_name, unit, undef, begin_time, end_time, lat, lon, and
    possibly others not yet seen. A single pd.read_csv() call over the
    whole blob (the original approach) misreads the FIRST metadata
    line as the table header and the REST of the metadata as data rows
    under that same 2-column shape - which is what produced the
    'prod_name, GPM_3IMERGHH.07' columns error both times, even though
    the two failures had genuinely different root causes (the first
    fix, promoting a timestamp-like index to a column, solved a
    different problem than this one and evidently never even
    triggered for this actual response shape).

    Rather than hardcode the exact metadata field names/count (fragile
    - Giovanni could add or reorder fields), this scans line-by-line
    for the FIRST line whose first comma-separated field parses as a
    valid timestamp - that reliably marks where the real data begins,
    regardless of exactly how much metadata precedes it or what it
    contains. Everything from that line onward is parsed directly
    (not via pd.read_csv, which is the wrong tool once the blob mixes
    two different row shapes) - each data line is split on commas,
    the first field taken as the timestamp and the LAST field taken
    as the value (tolerating either a plain 'timestamp,value' shape or
    a 'timestamp,label,value' shape with an extra column in between,
    since the exact data-row shape past the metadata preamble hasn't
    been confirmed against a real response yet either - see
    dev/README.md).
    """
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]

    data_start_idx = None
    for i, line in enumerate(lines):
        first_field = line.split(",")[0].strip()
        if not first_field:
            continue
        try:
            pd.Timestamp(first_field)
            data_start_idx = i
            break
        except (ValueError, TypeError):
            continue

    if data_start_idx is None:
        raise ValueError(
            "No timestamped data rows found in the Giovanni response - the entire response "
            f"may be metadata, or use a format not yet seen. Raw response (first 800 chars): "
            f"{text[:800]!r}"
        )

    rows = []
    malformed = 0
    for line in lines[data_start_idx:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            malformed += 1
            continue
        try:
            ts = pd.Timestamp(parts[0])
            val = float(parts[-1])
            rows.append({"Timestamp": ts, "PrecipitationMMHR": val})
        except (ValueError, TypeError):
            malformed += 1
            continue

    if not rows:
        raise ValueError(
            "Found what looked like the start of a data section but no rows parsed "
            f"successfully. Raw response (first 800 chars): {text[:800]!r}"
        )

    out = pd.DataFrame(rows)
    if malformed:
        # not fatal - a handful of trailing/odd lines (e.g. a footer)
        # shouldn't block otherwise-good data - but worth being able
        # to see this happened, so it's attached rather than silent.
        out.attrs["malformed_line_count"] = malformed
    return out


def fetch_giovanni_chunk(
    lat: float,
    lon: float,
    start_iso: str,
    end_iso: str,
    variable: str,
    token: str,
    timeout: int = 180,
) -> pd.DataFrame:
    """Single Giovanni API call for one date-range chunk. Isolated
    into its own function so tests can substitute a fake version of
    this exact call without touching the orchestration logic."""
    import requests

    params = {
        "data": variable,
        "location": f"[{round(lat, 4)},{round(lon, 4)}]",
        "time": f"{start_iso}/{end_iso}",
    }
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        GIOVANNI_TIMESERIES_URL, params=params, headers=headers, timeout=timeout
    )
    resp.raise_for_status()
    return _parse_giovanni_csv(resp.text)


def save_earthdata_credentials(
    username: str, password: str, config_path: str | Path | None = None
) -> list:
    """Writes/updates a netrc-format file with an entry for
    urs.earthdata.nasa.gov, so a username/password entered once via a
    QGIS parameter is picked up automatically on every future run -
    the same UX as era5_extract's CDS API Key parameter.

    WINDOWS FILENAME QUIRK (found via a real live-run failure, not
    anticipated in advance): Python's own netrc module - and by
    extension earthaccess/requests, which rely on it - defaults to
    looking for `_netrc` (leading underscore) on Windows, NOT `.netrc`
    (leading dot), a genuine documented platform difference. A version
    of this function that only wrote `.netrc` correctly created the
    file, but on Windows nothing was actually reading it - the earlier
    live-run error ("No .netrc found at ...\\_netrc") revealed this
    directly. Fixed by writing BOTH `.netrc` and, on Windows
    specifically, `_netrc` as well - harmless on non-Windows (the
    second write is skipped), and covers whichever convention any
    given library on Windows happens to check, rather than betting on
    one.

    UNLIKE .cdsapirc (which exists solely for one purpose), netrc
    files are shared, multi-service files by convention - a user may
    already have entries for other machines in them. This function
    preserves any such existing entries in EACH file it touches,
    replacing ONLY the urs.earthdata.nasa.gov block if one already
    exists there, or appending a new one if not - never a blind
    overwrite of the whole file.

    config_path is overridable (used by tests to write to a temp
    location instead of the real home directory) - when given
    explicitly, only that one exact path is written (no Windows-
    specific second file), so tests stay precise about what they're
    checking. Returns the list of paths actually written.
    """
    if config_path:
        return [_write_netrc_entry(Path(config_path), username, password)]

    import platform

    paths = [Path.home() / ".netrc"]
    if platform.system() == "Windows":
        paths.append(Path.home() / "_netrc")

    return [_write_netrc_entry(p, username, password) for p in paths]


def _write_netrc_entry(path: Path, username: str, password: str) -> Path:
    """Writes/updates one netrc-format file at the given path with an
    entry for urs.earthdata.nasa.gov, preserving any other existing
    machine entries in that same file - see save_earthdata_credentials()
    for why this matters."""
    entry = (
        f"machine {EARTHDATA_NETRC_MACHINE}\nlogin {username}\npassword {password}\n"
    )

    if path.exists():
        existing = path.read_text()
        blocks = _split_netrc_blocks(existing)
        blocks = [
            b
            for b in blocks
            if not b.strip().startswith(f"machine {EARTHDATA_NETRC_MACHINE}")
        ]
        blocks.append(entry.strip())
        new_content = "\n".join(b.strip() for b in blocks if b.strip()) + "\n"
    else:
        new_content = entry

    path.write_text(new_content)
    return path


def _split_netrc_blocks(content: str) -> list:
    """Splits a .netrc file's content into per-machine blocks, each
    starting with a 'machine <name>' line. Used only by
    save_earthdata_credentials() to preserve unrelated entries."""
    blocks = []
    current = []
    for line in content.splitlines():
        if line.strip().startswith("machine") and current:
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def get_earthdata_token(config_path: str | Path | None = None) -> str:
    """Authenticates via earthaccess (tries a stored .netrc first, then
    environment variables - no interactive prompt, since this runs
    inside a non-interactive QGIS Processing algorithm) and returns a
    bearer token for the Giovanni API."""
    try:
        import earthaccess
    except ImportError as missing_dependency:
        raise RuntimeError(
            "The 'earthaccess' package is required. Install it into QGIS's Python: "
            "python -m pip install earthaccess"
        ) from missing_dependency

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError(
            "Could not authenticate with Earthdata Login. Enter your Earthdata username/"
            "password into this tool's parameters once (saved automatically for future "
            "runs to ~/.netrc, and on Windows also to ~/_netrc - Python's own netrc "
            "handling looks for that underscore-prefixed name on Windows specifically, "
            "not the dot-prefixed one), or set up the file by hand yourself (machine "
            "urs.earthdata.nasa.gov). Also confirm 'NASA GESDISC DATA ARCHIVE' is an "
            "authorized application on your Earthdata profile: "
            "https://urs.earthdata.nasa.gov/profile"
        )
    token = earthaccess.get_edl_token()
    return token["access_token"] if isinstance(token, dict) else token


def _extend_bare_date_to_end_of_day(end_date: str) -> str:
    """If end_date parses to exactly midnight (00:00:00) - i.e. it was
    given as a bare date like '2005-03-05' with no time component,
    which is how this parameter is used in practice - extends it to
    23:59:59 of that same day, so the Giovanni request covers the
    WHOLE day rather than stopping at its very first instant.

    Found via a real discrepancy: for START_DATE=2005-03-01,
    END_DATE=2005-03-05, the granule method (via earthaccess's date-
    only temporal query, which NASA's CMR treats as inclusive of the
    whole end day) correctly returned data through 2005-03-05 23:30,
    but Giovanni - given the SAME bare end date - took it literally as
    the timestamp 2005-03-05T00:00:00 and stopped there, missing the
    entire day of March 5th except its first instant. Confirmed
    exactly: 193 rows for that range is precisely 4 days x 48 half-
    hours + 1 (midnight-to-midnight inclusive), not the 240 rows a
    full 5-day span would produce.

    An end_date that already carries an explicit non-midnight time
    (e.g. '2005-03-05T12:00:00') is left exactly as given - this only
    adjusts the bare-date case, not a deliberately precise timestamp.
    """
    ts = pd.Timestamp(end_date)
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
        ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts.isoformat()


def fetch_giovanni_timeseries(
    lat: float,
    lon: float,
    start_date: str | None = None,
    end_date: str | None = None,
    variable: str = DEFAULT_GIOVANNI_VARIABLE,
    chunk_months: int = 12,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
    token: str | None = None,
    progress_callback: Callable[[int], None] | None = None,
    fetch_chunk_fn: Callable = fetch_giovanni_chunk,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> FetchResult:
    """Orchestrates CONCURRENT Giovanni API calls across the full
    requested date range (see module docstring PERFORMANCE section),
    with retry-with-backoff per chunk and gap reporting (compares the
    returned row count against the theoretically expected count for a
    gap-free 30-minute record, without ever inventing values for
    missing periods).

    fetch_chunk_fn is injectable so this orchestration logic can be
    fully unit-tested with a fake network call. max_workers=1 falls
    back to fully serial fetching.
    """
    start_date = start_date or RECORD_START
    end_date = end_date or datetime.now(UTC).strftime("%Y-%m-%d")
    end_date = _extend_bare_date_to_end_of_day(end_date)
    token = token or get_earthdata_token()

    chunks = _split_date_range(start_date, end_date, chunk_months)
    warnings = []
    frames = []
    completed = 0
    completed_lock = threading.Lock()

    def _fetch_one_chunk(chunk):
        chunk_start, chunk_end = chunk
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                df = fetch_chunk_fn(lat, lon, chunk_start, chunk_end, variable, token)
                expected = (
                    int(
                        (
                            pd.Timestamp(chunk_end) - pd.Timestamp(chunk_start)
                        ).total_seconds()
                        // 1800
                    )
                    + 1
                )
                chunk_warnings = []
                if len(df) < expected * 0.95:
                    chunk_warnings.append(
                        f"Chunk {chunk_start[:10]} to {chunk_end[:10]}: got {len(df)} rows, "
                        f"expected ~{expected} for a gap-free record - some data may be missing "
                        "for this period (this is reported, not filled in)."
                    )
                return df, chunk_warnings
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(retry_backoff_s * attempt)
        return None, [
            f"Chunk {chunk_start[:10]} to {chunk_end[:10]} failed after {max_retries} attempts: {last_error}"
        ]

    if max_workers <= 1:
        results = [_fetch_one_chunk(c) for c in chunks]
        for df, chunk_warnings in results:
            if df is not None:
                frames.append(df)
            warnings.extend(chunk_warnings)
            completed += 1
            if progress_callback:
                progress_callback(int(100 * completed / max(len(chunks), 1)))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one_chunk, c): c for c in chunks}
            for future in as_completed(futures):
                df, chunk_warnings = future.result()
                if df is not None:
                    frames.append(df)
                warnings.extend(chunk_warnings)
                with completed_lock:
                    completed += 1
                    pct = int(100 * completed / max(len(chunks), 1))
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

    # sort warnings by their leading chunk-start date for a readable
    # log even though completion order under concurrency is not
    warnings_sorted = sorted(
        warnings, key=lambda w: w.split(" ")[1] if w.startswith("Chunk") else w
    )

    return FetchResult(dataframe=combined, warnings=warnings_sorted, method="giovanni")


# ---------------------------------------------------------------------
# Fallback: granule-based extraction, mirroring the original sample
# script's approach, but using xr.open_dataset (singular - see below
# for why, not open_mfdataset) since each granule is opened one at a
# time, not merged across files.
# ---------------------------------------------------------------------


def estimate_granule_count(start_date: str, end_date: str) -> int:
    """48 granules/day for IMERG half-hourly."""
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).total_seconds() / 86400
    return int(days * 48) + 1


def _to_pandas_timestamp(value) -> pd.Timestamp:
    """Converts a time value that may be a numpy datetime64/pandas-
    compatible value OR a cftime calendar object (DatetimeJulian,
    DatetimeNoLeap, Datetime360Day, etc.) into a plain pd.Timestamp.

    Found via a real live failure: IMERG's HDF5 granule files declare
    a Julian calendar for their time variable's metadata, so xarray
    decodes it as cftime.DatetimeJulian rather than a numpy datetime64
    - pd.Timestamp() cannot parse a cftime object directly ("Cannot
    convert input ... of type cftime._cftime.DatetimeJulian to
    Timestamp"), confirmed by reproducing the exact error with a
    synthetic cftime.DatetimeJulian(2005, 3, 1) before writing this
    fix. Every cftime calendar subtype exposes the same stable
    year/month/day/hour/minute/second component attributes regardless
    of which specific calendar it represents, so building a plain
    Gregorian pd.Timestamp from those components - rather than relying
    on any cftime-to-pandas conversion method whose availability might
    vary by cftime/pandas version - works universally. For IMERG's
    actual date range (2000-present), the "Julian" label on the
    source file is a metadata detail only; the underlying values are
    real calendar dates that convert cleanly this way.
    """
    if hasattr(value, "year") and hasattr(value, "month"):
        return pd.Timestamp(
            year=value.year,
            month=value.month,
            day=value.day,
            hour=value.hour,
            minute=value.minute,
            second=value.second,
        )
    return pd.Timestamp(value)


def read_point_from_dataset(
    ds, lat: float, lon: float, variable_candidates=VARIABLE_NAME_CANDIDATES
) -> float:
    """Nearest-neighbour point extraction from an opened IMERG xarray
    Dataset, trying each candidate variable name in order (handles the
    V06 precipitationCal -> V07 precipitation rename transparently)."""
    for var in variable_candidates:
        if var in ds.data_vars:
            value = ds[var].sel(lat=lat, lon=lon, method="nearest")
            return float(value.values.squeeze())
    raise ValueError(
        f"None of the candidate variable names {variable_candidates} found. "
        f"Available: {list(ds.data_vars)}"
    )


def _check_netcdf_backend_available() -> None:
    """Fails fast, before any download starts, if xarray has no
    working NetCDF/HDF5 backend installed. Found the hard way: a live
    25-minute run downloaded all 240 granules for a 5-day window
    successfully, then failed to OPEN every single one of them with
    'found the following matches with the input file in xarray's IO
    backends: [netcdf4, h5netcdf]. But their dependencies may not be
    installed.' xarray itself does NOT bundle a NetCDF/HDF5 reader -
    it needs a separate package (netCDF4 or h5netcdf) that was never
    included in this suite's setup instructions, since every other
    plugin's file-reading need (GRIB via cfgrib, GeoTIFF via rasterio)
    happened to need a different backend. Checking this BEFORE the
    granule loop starts turns a 25-minute wasted run into an
    immediate, clear error - the download step for each granule is
    the expensive part; there's no reason to pay that cost 240 times
    over before discovering the file can't be opened at all.
    """
    try:
        import netCDF4  # noqa: F401

        return
    except ImportError:
        pass
    try:
        import h5netcdf  # noqa: F401

        return
    except ImportError:
        pass
    raise RuntimeError(
        "Neither 'netCDF4' nor 'h5netcdf' is installed - xarray needs one of these "
        "to actually read IMERG's granule files (xarray itself doesn't bundle a "
        "NetCDF/HDF5 reader). Install one into QGIS's Python: "
        "python -m pip install netCDF4"
    )


def fetch_granule_based_timeseries(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    short_name: str = DEFAULT_SHORT_NAME,
    version: str = DEFAULT_VERSION,
    variable_candidates=VARIABLE_NAME_CANDIDATES,
    force: bool = False,
    progress_callback: Callable[[int], None] | None = None,
) -> FetchResult:
    """Fallback method - see module docstring. Refuses to run above
    GRANULE_COUNT_WARN_THRESHOLD granules unless force=True."""
    n_expected = estimate_granule_count(start_date, end_date)
    if n_expected > GRANULE_COUNT_WARN_THRESHOLD and not force:
        raise ValueError(
            f"This date range needs an estimated {n_expected:,} individual granule downloads "
            f"(threshold: {GRANULE_COUNT_WARN_THRESHOLD:,}). This method is not suitable for "
            "long ranges - use the Giovanni API method instead, or pass force=True if you "
            "really want to proceed (it will be slow)."
        )

    try:
        import earthaccess
        import xarray as xr
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency for the granule-based method: {e}"
        ) from e

    _check_netcdf_backend_available()

    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError(
            "Could not authenticate with Earthdata Login (see get_earthdata_token's error for detail)."
        )

    results = earthaccess.search_data(
        short_name=short_name,
        version=version,
        temporal=(start_date, end_date),
        bounding_box=(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05),
    )

    warnings = []
    rows = []
    n = len(results)
    for i, granule in enumerate(results):
        try:
            files = earthaccess.download([granule], local_path=".")
            # open_dataset (singular), not open_mfdataset - each iteration
            # handles exactly one file already (earthaccess.download() is
            # called with a single-element list), so there's no actual
            # multi-file concatenation happening here. open_mfdataset
            # unconditionally requires 'dask' for its lazy multi-file
            # concat machinery, even with only one file - confirmed via a
            # live failure ("chunk manager 'dask' is not available")
            # after installing netCDF4 fixed the previous backend gap.
            # open_dataset needs no such dependency and is the correct
            # function for this single-file-per-call usage pattern.
            with xr.open_dataset(files[0], group="Grid") as ds:
                value = read_point_from_dataset(ds, lat, lon, variable_candidates)
                timestamp = _to_pandas_timestamp(ds["time"].values[0])
                rows.append({"Timestamp": timestamp, "PrecipitationMMHR": value})
        except Exception as e:
            warnings.append(f"Granule {i} failed: {e}")
        if progress_callback:
            progress_callback(int(100 * (i + 1) / max(n, 1)))

    df = (
        pd.DataFrame(rows).sort_values("Timestamp").reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["Timestamp", "PrecipitationMMHR"])
    )

    return FetchResult(dataframe=df, warnings=warnings, method="granule")
