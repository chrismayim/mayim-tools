"""
Core logic for extracting MERRA-2 bias-corrected precipitation
(PRECTOTCORR) at a single point, via NASA GES DISC's Giovanni Time
Series API - the same service already built and validated for the
IMERG Point Extractor plugin, confirmed to also support MERRA-2
directly from Giovanni's own live WMS service listing (not a
secondhand source): variable ID `M2T1NXFLX_5_12_4_PRECTOTCORR`,
date range 1980-01-01 to (at research time) mid-2026, actively updated.

WHY PRECTOTCORR, NOT PRECTOT: MERRA-2's flux collection carries two
precipitation fields. PRECTOT is the raw atmospheric model's own
precipitation (not bias-corrected, not what the land-surface
component is actually forced with). PRECTOTCORR is a bias-corrected
product - the one actually used to force MERRA-2's own land surface,
and the one used throughout the peer-reviewed literature for
hydrological work (e.g. Great Lakes Basin streamflow studies comparing
ERA5 TP against MERRA-2 PRECTOTCORR specifically, not PRECTOT). This
tool extracts PRECTOTCORR only - PRECTOT is not exposed as an option,
the same deliberate-default philosophy used for plain ERA5 over
ERA5-Land in era5_extract.

REGIONAL CAVEAT WORTH KNOWING, not just a generic disclaimer: per
GMAO's own MERRA-2 file specification (Office Note 9), PRECTOTCORR's
gauge-based correction "tapers back to MERRA-2 model generated
precipitation poleward of 42.5 degrees latitude, and is completely
MERRA-2 precipitation poleward of 62.5 degrees. Also, over continental
Africa, the observations change to the CMAP gauge-satellite product,
due to limitations in the available gauge observations." For a West
African site specifically, PRECTOTCORR's correction source is CMAP
(a coarser gauge-satellite blend), not the primary correction used
over e.g. North America - still a genuine bias correction over the
raw model field, just from a different, coarser-resolution source.

UNITS - simpler than ERA5, no deaccumulation needed: MERRA-2
precipitation is a TIME-AVERAGED RATE in kg/m^2/s, numerically
identical to mm/s (confirmed directly from NASA's own Earthdata
Forum: "mm/hour => precip*3600"). Each hourly value already
represents that hour's average rate - multiplying by 3600 gives both
the rate in mm/hr AND the actual depth for that hour (since the
interval IS one hour, rate x 1hr = depth), unlike ERA5's cumulative-
since-cycle-start convention which genuinely needed differencing.

TIMESTAMPS: MERRA-2 hourly values are stamped at the CENTER of each
hour, starting from 00:30 UTC (00:30, 01:30, ..., 23:30), not the top
of the hour - confirmed directly from GES DISC's own collection
documentation. Whatever timestamp Giovanni's response actually
returns is preserved exactly, not silently re-aligned to the hour.

ARCHITECTURE: deliberately mirrors imerg_extract's Giovanni
orchestration closely (concurrent chunked fetching, per-chunk retry
with backoff, the same metadata-preamble-tolerant CSV parsing logic
that was hard-won for IMERG's Giovanni responses) - since this is the
SAME Giovanni service, response-format surprises found there are
assumed likely to recur here too, so the same defensive parsing is
applied from the start rather than re-discovering them the hard way a
second time. NOT independently confirmed against a live MERRA-2
Giovanni response, though - see dev/README.md.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from mayim_tools._common.credentials import earthdata_login

RECORD_START = "1980-01-01"  # confirmed directly from Giovanni's live WMS listing
# confirmed directly from Giovanni's live WMS listing
DEFAULT_GIOVANNI_VARIABLE = "M2T1NXFLX_5_12_4_PRECTOTCORR"
GIOVANNI_TIMESERIES_URL = "https://api.giovanni.earthdata.nasa.gov/timeseries"

MM_PER_SECOND_TO_MM_PER_HOUR = (
    3600.0  # kg/m^2/s == mm/s; x3600 = mm/hr (confirmed via NASA Earthdata Forum)
)

DEFAULT_MAX_WORKERS = (
    8  # matches chirps_extract's/imerg_extract's default and reasoning
)


@dataclass
class FetchResult:
    dataframe: pd.DataFrame  # columns: Timestamp, PrecipitationMMHR
    warnings: list
    method: str = "giovanni"


def _split_date_range(start: str, end: str, chunk_months: int) -> list:
    """Splits [start, end] into contiguous, non-overlapping chunks of
    approximately chunk_months each. Pure function - no I/O - fully
    unit-testable without network access. Identical logic to
    imerg_extract's own _split_date_range - kept as a local copy
    rather than a cross-plugin import, matching this suite's existing
    pattern of each plugin being independently self-contained."""
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


def _extend_bare_date_to_end_of_day(end_date: str) -> str:
    """If end_date parses to exactly midnight (a bare date like
    '2005-03-05', the normal way this parameter is used), extends it
    to 23:59:59 of that same day - otherwise Giovanni takes it
    literally as that exact instant and stops there, excluding the
    rest of the day. Confirmed as a real, live discrepancy while
    building imerg_extract (comparing Giovanni's results against an
    independent extraction method for the identical date range) -
    applied here proactively from the start rather than waiting to
    rediscover the same issue against MERRA-2's Giovanni endpoint."""
    ts = pd.Timestamp(end_date)
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
        ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return ts.isoformat()


def _parse_giovanni_csv(text: str) -> pd.DataFrame:
    """Parser for the Giovanni time-series CSV response.

    Uses the SAME metadata-preamble-tolerant approach hard-won for
    imerg_extract's identical Giovanni service: the real IMERG
    response turned out to be a metadata PREAMBLE of plain,
    uncommented 'key,value' lines (prod_name, doi, param_short_name,
    unit, begin_time, lat, lon, ...) before the actual data - NOT a
    simple two-column table, and NOT reliably parseable with
    pd.read_csv() over the whole blob (which misreads the first
    metadata line as a table header). Rather than assume MERRA-2's
    Giovanni response looks the same and risk being wrong again, this
    scans line-by-line for the first line whose first comma-separated
    field parses as a valid timestamp - that reliably marks where real
    data begins regardless of exactly what metadata precedes it,
    working correctly whether the preamble looks identical to IMERG's,
    similar, or different. NOT independently confirmed against a live
    MERRA-2 response - see dev/README.md.
    """
    lines = [
        raw_line
        for raw_line in text.splitlines()
        if not raw_line.lstrip().startswith("#")
    ]

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
            "No timestamped data rows found in the Giovanni response - the "
            "entire response may be metadata, or use a format not yet seen. "
            f"Raw response (first 800 chars): {text[:800]!r}"
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
            rows.append({"Timestamp": ts, "value_mm_per_s": val})
        except (ValueError, TypeError):
            malformed += 1
            continue

    if not rows:
        raise ValueError(
            "Found what looked like the start of a data section but no rows parsed "
            f"successfully. Raw response (first 800 chars): {text[:800]!r}"
        )

    out = pd.DataFrame(rows)
    out["PrecipitationMMHR"] = out["value_mm_per_s"] * MM_PER_SECOND_TO_MM_PER_HOUR
    out = out.drop(columns=["value_mm_per_s"])
    if malformed:
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


def get_earthdata_token() -> str:
    """Log in to NASA Earthdata (see _common.credentials.earthdata_login:
    the tool's QGIS authentication configuration first, then ~/.netrc)
    and return a bearer token for the Giovanni API."""
    import earthaccess

    earthdata_login()
    token = earthaccess.get_edl_token()
    return token["access_token"] if isinstance(token, dict) else token


def fetch_merra2_timeseries(
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
    requested date range, with retry-with-backoff per chunk. Deliberately
    mirrors imerg_extract's fetch_giovanni_timeseries() structure -
    same concurrency model, same chunk-level error isolation, same
    bare-end-date extension - since it's the same underlying service
    and the same lessons are expected to apply.

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
                        // 3600
                    )
                    + 1
                )
                chunk_warnings = []
                if len(df) < expected * 0.95:
                    gap_msg = (
                        f"Chunk {chunk_start[:10]} to {chunk_end[:10]}: "
                        f"got {len(df)} rows, expected ~{expected} for a "
                        "gap-free hourly record - some data may be missing "
                        "for this period (reported, not filled in)."
                    )
                    chunk_warnings.append(gap_msg)
                return df, chunk_warnings
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(retry_backoff_s * attempt)
        failure_msg = (
            f"Chunk {chunk_start[:10]} to {chunk_end[:10]} failed after "
            f"{max_retries} attempts: {last_error}"
        )
        return None, [failure_msg]

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

    warnings_sorted = sorted(
        warnings, key=lambda w: w.split(" ")[1] if w.startswith("Chunk") else w
    )

    return FetchResult(dataframe=combined, warnings=warnings_sorted, method="giovanni")
