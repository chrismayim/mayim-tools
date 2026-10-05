"""Access-route fallback and the parallel job runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from .errors import SoilDataError


@dataclass(frozen=True)
class Source:
    route: str  # "Tiles" | "VRT" | "2017 archive"
    path: str | tuple[str, ...]


def read_with_fallback(sources: Sequence[Source], reader: Callable, *args):
    """Try each source in turn. Returns (result, route, errors).
    Raises SoilDataError listing every attempt if all fail."""
    errors: list[str] = []
    for i, src in enumerate(sources):
        try:
            out = reader(src.path, *args)
        except Exception as exc:  # noqa: BLE001 - every failure is reported
            errors.append(f"{src.route}: {exc}")
            continue
        last = i == len(sources) - 1
        if src.route == "Tiles" and not last and _all_nan(out):
            # Tiles that exist but cover a different area give no data and
            # no error; confirm against the next route before accepting.
            errors.append(f"{src.route}: no data in the area")
            continue
        return out, src.route, errors
    raise SoilDataError("All access routes failed - " + " | ".join(errors))


def _all_nan(values) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return arr.size > 0 and not np.isfinite(arr).any()


def run_jobs(
    keys: Sequence,
    fn: Callable,
    workers: int,
    cancel_fn: Callable,
    on_done: Callable,
) -> dict:
    """Run fn(key) for every key, ``workers`` at a time. Results by key.
    Cancellation is checked as each job finishes; pending jobs are dropped."""
    results: dict = {}
    if workers <= 1 or len(keys) <= 1:
        for key in keys:
            if cancel_fn():
                raise InterruptedError("Cancelled by user")
            results[key] = fn(key)
            on_done(key)
        return results
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(fn, key): key for key in keys}
    try:
        for fut in as_completed(futures):
            key = futures[fut]
            results[key] = fut.result()
            on_done(key)
            if cancel_fn():
                raise InterruptedError("Cancelled by user")
    finally:
        for fut in futures:
            fut.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    return results


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------


def _noop(*_args, **_kwargs):
    return None


def _never_cancelled() -> bool:
    return False


class _Progress:
    def __init__(self, total: int, progress_fn: Callable):
        self.total = max(total, 1)
        self.done = 0
        self.progress_fn = progress_fn

    def step(self, message: str) -> None:
        self.done += 1
        self.progress_fn(min(self.done / self.total, 0.999), message)
