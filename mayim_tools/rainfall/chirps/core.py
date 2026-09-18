"""
Core logic for extracting CHIRPS v3 precipitation at a single point
over a date range.

ARCHITECTURE DECISION - why this is simpler than the IMERG plugin:
CHIRPS is fundamentally a PENTAD (5-day) and MONTHLY product - daily
data is a *derived* product (two variants, see below), not CHIRPS's
native resolution. This matters a lot for scale: a full 1981-present
pentad record is roughly 45 years x 73 pentads/year =~ 3,300 files -
two orders of magnitude smaller than IMERG's ~455,000 half-hourly
granules. Combined with CHIRPS being served as individual COG-capable
GeoTIFFs on a fully public, unauthenticated HTTP(S) server, a direct
per-timestep read using GDAL's HTTP range-request support (`/vsicurl/`)
is efficient enough on its own. No Giovanni-style bulk API is needed.

WHY NOT GOOGLE EARTH ENGINE: CHIRPS v3 is also available via GEE
(`UCSB-CHC/CHIRPS/V3/PENTAD` etc.), which would make point-time-series
extraction trivial. Not used here because Earth Engine's terms state
it's "free to use for research, education, and nonprofit use" -
Mayim Consulting is a commercial consultancy, and Google introduced
commercial Earth Engine pricing in 2024. Using GEE's free tier for
commercial consulting work risks a licensing/ToS problem that direct
access to CHIRPS's own public-domain data avoids entirely.

WHY NOT CLIMATESERV: SERVIR/USGS's ClimateSERV API was investigated
as a possible simpler alternative. Rejected: it is (1) polygon-based,
not point-based - a point gets buffered into a small polygon and an
AREA STATISTIC is returned, not an exact value at the coordinate, a
real methodological difference for point-based design work; (2) an
asynchronous job-queue API (submit -> poll -> retrieve), which is
MORE moving parts to build and test, not fewer; and (3) of unconfirmed
CHIRPS v3 support (the API predates v3's January 2025 release).

WHAT'S VERIFIED VS INFERRED (see dev/README.md for full detail and
sources):
- VERIFIED: the server is public, unauthenticated, and the MONTHLY
  URL pattern is confirmed exactly via CHC's own documented wget
  example: .../v3.0/monthly/global/tifs/chirps-v3.0.YYYY.MM.tif
- VERIFIED: the DAILY URL pattern (both 'rnl' and 'sat' variants) is
  confirmed exactly by reading the actual source code of the
  ropensci/chirps R package's .chirps_raw_urls() function - a real,
  working, actively-maintained client for this exact API:
  .../v3.0/daily/final/{type}/{year}/chirps-v3.0.{type}.{date}.tif
  (This corrected an earlier inferred pattern that was wrong in three
  ways: missing the "final/" quality-level directory, missing the
  per-year subdirectory, and missing the type embedded in the
  filename itself - found via a live QGIS run returning a 404, fixed
  by reading the R package's actual source rather than guessing again.)
- INFERRED, still NOT independently verified: the PENTAD filename
  pattern. Directory existence is confirmed, but the exact filename
  was not found in any reference client's source code. Deliberately
  NOT "improved" by borrowing daily's final/year-subdirectory
  structure: the confirmed MONTHLY pattern has neither of those,
  proving CHC's convention isn't uniform across products - applying
  daily's structure to pentad would be an unjustified guess, not a
  real fix.

PERFORMANCE (added after a live run showed ~1.46s/file, projecting to
~6 hours for a 40-year daily extraction):
1. GDAL/vsicurl tuning (see _VSICURL_ENV below) - skips unnecessary
   per-file round-trips (existence-check HEAD request, sidecar-file
   probing for .aux.xml/.ovr files that don't exist here). Pure
   configuration, no logic change, no risk.
2. Concurrent fetching (see fetch_chirps_timeseries's max_workers) -
   each date's file is completely independent of every other date's,
   so this is textbook latency-bound work that parallelizes well.
   Default of 8 concurrent requests is a deliberately moderate choice
   - fast enough to matter, conservative enough to stay a good citizen
   of a public data server rather than hammering it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

CHIRPS_RECORD_START = date(1981, 1, 1)  # confirmed: CHIRPS v3's record start

# Per-product default start date, used ONLY when the user leaves the
# start date blank (an explicit user-specified start is always
# respected, even earlier than this). CHIRPS pentad/monthly/daily_rnl
# genuinely cover the full 1981-present record. daily_sat does NOT:
# it's built from NASA IMERG, whose standard record starts in 2000,
# with a separate ongoing effort to extend it back to 1998 ("efforts
# to extend the starting year to 1998 are ongoing" - NCAR Climate Data
# Guide) - CHIRPS v3's 'sat' variant uses that extended record. ERA5
# (behind 'rnl') has no such limit - its own reanalysis record starts
# in 1940, comfortably covering all of CHIRPS's span. Confirmed by a
# live run: daily_rnl correctly returned data from 1981, daily_sat
# only from 1998 - not a bug, a genuine data-availability limit in the
# satellite record daily_sat depends on. Defaulting daily_sat's blank-
# start to 1998 avoids ~17 years of guaranteed-404 requests (and the
# very noisy warning log that comes with them) on every full-record
# daily_sat run, without blocking an explicit earlier date if someone
# deliberately wants to confirm this limit for themselves.
IMERG_SAT_RECORD_START = date(1998, 1, 1)

_DEFAULT_START_BY_PRODUCT = {
    "pentad": CHIRPS_RECORD_START,
    "monthly": CHIRPS_RECORD_START,
    "daily_rnl": CHIRPS_RECORD_START,
    "daily_sat": IMERG_SAT_RECORD_START,
}
BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0"

# VERIFIED exactly via CHC's own wget documentation example.
_MONTHLY_URL_TEMPLATE = (
    BASE_URL + "/monthly/global/tifs/chirps-v3.0.{year:04d}.{month:02d}.tif"
)

# INFERRED by extending the confirmed monthly convention - directory
# existence confirmed, exact filename NOT independently verified. This
# remains the ONE unconfirmed pattern in this module - see dev/README.md.
_PENTAD_URL_TEMPLATE = (
    BASE_URL + "/pentads/global/tifs/chirps-v3.0.{year:04d}.{month:02d}.{pentad:d}.tif"
)

# CONFIRMED via the actual source code of the ropensci/chirps R package's
# .chirps_raw_urls() function (fetched and read directly, not just its
# documentation).
_DAILY_URL_TEMPLATES = {
    "rnl": BASE_URL + "/daily/final/rnl/{year:04d}/"
    "chirps-v3.0.rnl.{year:04d}.{month:02d}.{day:02d}.tif",
    "sat": BASE_URL + "/daily/final/sat/{year:04d}/"
    "chirps-v3.0.sat.{year:04d}.{month:02d}.{day:02d}.tif",
}

PRODUCTS = ("pentad", "daily_rnl", "daily_sat", "monthly")

DEFAULT_MAX_WORKERS = 8

# GDAL/vsicurl tuning to cut per-file round-trips:
#   CPL_VSIL_CURL_USE_HEAD=NO           skip the existence-check HEAD
#                                       request GDAL does by default
#                                       before reading - one whole
#                                       round-trip saved per file.
#   GDAL_DISABLE_READDIR_ON_OPEN        stop GDAL probing for sidecar
#     =EMPTY_DIR                       files (.aux.xml, .ovr) that
#                                       don't exist for these files -
#                                       saves 1-2 more round-trips.
#   GDAL_HTTP_VERSION=2                 enables HTTP/2, allowing
#                                       connection/session reuse
#                                       across requests instead of a
#                                       fresh TCP+TLS handshake each
#                                       time.
#   GDAL_HTTP_MULTIPLEX=YES             lets multiple range requests
#                                       share one HTTP/2 connection.
# Not independently verified against the real CHC server from this
# environment (data.chc.ucsb.edu is not reachable from here - see
# dev/README.md) - these are standard, well-documented GDAL vsicurl
# options, not guesses, but the actual speedup on the real server
# should be confirmed empirically on first use after this change.
VSICURL_ENV_OPTIONS = {
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MULTIPLEX": "YES",
}


@dataclass
class ChirpsResult:
    dataframe: pd.DataFrame  # columns: Date, PrecipitationMM
    warnings: list = field(default_factory=list)
    product: str = ""
    urls_attempted_sample: list = field(
        default_factory=list
    )  # first few URLs, for diagnosing a wrong pattern


def _pentad_periods(start: date, end: date):
    """Yields (period_start_date, year, month, pentad_index[1-6]) for
    every pentad overlapping [start, end]. CHIRPS pentads split each
    month into six ~5-day periods: 1-5, 6-10, 11-15, 16-20, 21-25,
    26-end-of-month (the last pentad is 3-6 days depending on month
    length, not a fixed 5)."""
    cur_month = date(start.year, start.month, 1)
    while cur_month <= end:
        for p in range(1, 7):
            day0 = 26 if p == 6 else (p - 1) * 5 + 1
            period_start = date(cur_month.year, cur_month.month, day0)
            if p < 6:
                period_end = date(cur_month.year, cur_month.month, day0 + 4)
            else:
                period_end = _month_add(cur_month, 1) - timedelta(days=1)
            if period_end >= start and period_start <= end:
                yield period_start, cur_month.year, cur_month.month, p
        cur_month = _month_add(cur_month, 1)


def _month_add(d: date, n: int) -> date:
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    return date(year, month, 1)


def _daily_dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _monthly_periods(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        yield cur
        cur = _month_add(cur, 1)


def build_url(product: str, d: date, pentad_index: int | None = None) -> str:
    """Constructs the CHIRPS v3 URL for one time period. product:
    'pentad', 'daily_rnl', 'daily_sat', or 'monthly'."""
    if product == "monthly":
        return _MONTHLY_URL_TEMPLATE.format(year=d.year, month=d.month)
    if product == "pentad":
        if pentad_index is None:
            raise ValueError("pentad_index is required for product='pentad'")
        return _PENTAD_URL_TEMPLATE.format(
            year=d.year, month=d.month, pentad=pentad_index
        )
    if product == "daily_rnl":
        return _DAILY_URL_TEMPLATES["rnl"].format(year=d.year, month=d.month, day=d.day)
    if product == "daily_sat":
        return _DAILY_URL_TEMPLATES["sat"].format(year=d.year, month=d.month, day=d.day)
    raise ValueError(f"Unknown product {product!r} - expected one of {PRODUCTS}")


# rasterio.Env() is not documented as thread-safe for concurrent use
# from multiple threads simultaneously manipulating the SAME env stack.
# GDAL's own config option mechanism is process-global (thread-local in
# recent GDAL, but this varies by version) - to avoid relying on that
# subtlety, each worker thread sets the options once via a lock-guarded
# one-time initialization rather than wrapping every single read in its
# own Env() context (which would also add per-call overhead).
_env_lock = threading.Lock()
_env_applied = False


def _ensure_vsicurl_env():
    global _env_applied
    if _env_applied:
        return
    with _env_lock:
        if _env_applied:
            return
        import rasterio.env

        for key, value in VSICURL_ENV_OPTIONS.items():
            rasterio.env.set_gdal_config(key, value)
        _env_applied = True


def read_point_value(url: str, lat: float, lon: float) -> float:
    """Reads a single pixel value from a remote GeoTIFF/COG via GDAL's
    HTTP range-request support (/vsicurl/) - fetches only the file
    header and the tile/strip containing the target pixel, not the
    whole global raster. Applies the VSICURL_ENV_OPTIONS tuning once
    per process (see module docstring's PERFORMANCE section) rather
    than per-call.

    The out-of-bounds check wraps the actual read in a try/except as a
    safety net, not just a row/col range comparison beforehand - see
    tests/test_core.py for why.
    """
    import rasterio

    _ensure_vsicurl_env()

    vsi_url = url if url.startswith("/vsicurl/") else f"/vsicurl/{url}"
    with rasterio.open(vsi_url) as src:
        row, col = src.index(lon, lat)
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            raise ValueError(
                f"Point ({lat}, {lon}) is outside the raster's extent."
            ) from None
        try:
            window = ((row, row + 1), (col, col + 1))
            value = src.read(1, window=window)[0, 0]
        except IndexError:
            raise ValueError(
                f"Point ({lat}, {lon}) is outside the raster's extent."
            ) from None
        nodata = src.nodata
        if nodata is not None and value == nodata:
            return float("nan")
        return float(value)


def fetch_chirps_timeseries(
    lat: float,
    lon: float,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    product: str = "pentad",
    progress_callback: Callable[[int], None] | None = None,
    read_fn: Callable = read_point_value,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> ChirpsResult:
    """Orchestrates fetching every period in [start_date, end_date] for
    the chosen product, CONCURRENTLY (see module docstring's
    PERFORMANCE section) - each date's file is completely independent
    of every other date's, so this is textbook latency-bound work that
    parallelizes well. read_fn is injectable so the orchestration
    logic (concurrency, retry/error isolation per-file, progress
    reporting, DataFrame assembly, sort-back-into-date-order) can be
    fully unit-tested without any real network access.

    start_date/end_date: None on either end means the full available
    record for that product. For pentad/monthly/daily_rnl that's
    1981-01-01 to today; for daily_sat it's 1998-01-01 to today (see
    IMERG_SAT_RECORD_START - daily_sat depends on IMERG, which doesn't
    have data before 1998). Accepts date objects or 'YYYY-MM-DD'
    strings. An explicit start_date is always respected even if
    earlier than the product's real data availability - you'll just
    get 404 warnings for the dates with no data, same as before this
    default was added.

    max_workers: number of concurrent requests in flight. Default (8)
    is a deliberately moderate choice - fast enough to matter,
    conservative enough to stay a good citizen of a public data server.
    Set to 1 to fall back to fully serial fetching (e.g. for debugging).
    """
    if product not in PRODUCTS:
        raise ValueError(f"Unknown product {product!r} - expected one of {PRODUCTS}")

    start = (
        _to_date(start_date)
        if start_date
        else _DEFAULT_START_BY_PRODUCT.get(product, CHIRPS_RECORD_START)
    )
    end = _to_date(end_date) if end_date else date.today()
    if start > end:
        raise ValueError(f"start_date ({start}) is after end_date ({end})")

    if product == "pentad":
        periods = [(d, y, m, p) for d, y, m, p in _pentad_periods(start, end)]
        urls_and_dates = [
            (build_url("pentad", d, pentad_index=p), d) for d, y, m, p in periods
        ]
    elif product == "monthly":
        periods = list(_monthly_periods(start, end))
        urls_and_dates = [(build_url("monthly", d), d) for d in periods]
    else:  # daily_rnl / daily_sat
        periods = list(_daily_dates(start, end))
        urls_and_dates = [(build_url(product, d), d) for d in periods]

    warnings = []
    rows = []
    n = len(urls_and_dates)
    completed = 0
    completed_lock = threading.Lock()

    def _fetch_one(url_and_date):
        url, period_date = url_and_date
        try:
            value = read_fn(url, lat, lon)
            return (period_date, value, None)
        except Exception as e:
            return (
                period_date,
                None,
                f"{period_date}: failed ({type(e).__name__}: {e}) - URL: {url}",
            )

    if max_workers <= 1:
        # serial fallback (also avoids ThreadPoolExecutor overhead for
        # tiny requests, e.g. short single-month test ranges)
        results = [_fetch_one(ud) for ud in urls_and_dates]
        for period_date, value, warning in results:
            if warning:
                warnings.append(warning)
            else:
                rows.append({"Date": period_date, "PrecipitationMM": value})
            completed += 1
            if progress_callback:
                progress_callback(int(100 * completed / max(n, 1)))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_one, ud): ud for ud in urls_and_dates}
            for future in as_completed(futures):
                period_date, value, warning = future.result()
                if warning:
                    warnings.append(warning)
                else:
                    rows.append({"Date": period_date, "PrecipitationMM": value})
                with completed_lock:
                    completed += 1
                    pct = int(100 * completed / max(n, 1))
                if progress_callback:
                    progress_callback(pct)

    df = (
        pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["Date", "PrecipitationMM"])
    )

    if warnings and len(warnings) == n:
        warnings.append(
            "Every request failed - this strongly suggests the URL pattern for this "
            "product is wrong (see dev/README.md: the pentad pattern is inferred, not "
            "independently verified). Check the attempted URL above against the "
            "actual CHIRPS v3 directory structure."
        )

    # sort warnings by date for a readable log even though completion
    # order under concurrency is nondeterministic
    warnings_sorted = sorted(warnings, key=lambda w: w.split(":")[0])

    return ChirpsResult(
        dataframe=df,
        warnings=warnings_sorted,
        product=product,
        urls_attempted_sample=[u for u, _ in urls_and_dates[:3]],
    )


def _to_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()
