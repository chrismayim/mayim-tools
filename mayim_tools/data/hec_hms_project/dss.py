"""
Precipitation gage records in a HEC-DSS file, written with HEC's own
``hecdss`` Python package (MIT licence, bundles the HEC-DSS 7 library for
Windows and Linux). Install it in QGIS's Python once:

    python -m pip install hecdss

Only the hyetograph storms (Alternating Block, Huff, ensembles) need it;
HEC-HMS frequency storms are stored in the meteorologic files.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

INSTALL_HINT = (
    "The hyetograph storms are written as precipitation gages in a HEC-DSS "
    "file, which needs HEC's hecdss package in QGIS's Python. Install it once "
    "from the OSGeo4W Shell (or with QGIS's python.exe): python -m pip install "
    "hecdss - then run the tool again. Frequency storms from the DDF table do "
    "not need it."
)


class DssUnavailableError(RuntimeError):
    pass


def interval_part(minutes: float) -> str:
    """DSS E part for an interval: 5MIN, 1HOUR, 1DAY."""
    m = int(round(minutes))
    if m < 60:
        return f"{m}MIN"
    if m < 1440:
        return f"{m // 60}HOUR"
    return f"{m // 1440}DAY"


def hecdss_available() -> bool:
    try:
        import hecdss  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable
        return False
    return True


def write_gages(
    path: str | Path, series: list[tuple[str, datetime, float, list]]
) -> int:
    """Writes incremental precipitation (mm) records. ``series`` items are
    (pathname, time of the first value, interval in minutes, values).
    Returns the number of records written."""
    try:
        from hecdss import HecDss, RegularTimeSeries
    except Exception as e:  # noqa: BLE001
        raise DssUnavailableError(INSTALL_HINT) from e
    import numpy as np

    try:
        HecDss.set_global_debug_level(1)
    except Exception:  # noqa: BLE001 - older hecdss versions
        pass
    path = Path(path)
    if path.exists():
        path.unlink()
    n = 0
    with HecDss(str(path)) as dss:
        for pathname, first, interval, values in series:
            ts = RegularTimeSeries()
            ts.id = pathname
            ts.values = np.asarray(values, dtype=float)
            ts.times = [
                first + timedelta(minutes=interval * i) for i in range(len(values))
            ]
            ts.units = "MM"
            ts.data_type = "PER-CUM"
            ts.interval = int(round(interval * 60))
            status = dss.put(ts)
            if status != 0:
                raise RuntimeError(
                    f"HEC-DSS could not store {pathname} (status {status})."
                )
            n += 1
    return n


def read_gage(path: str | Path, pathname: str):
    """(times, values) of a record - used by the tests."""
    from hecdss import HecDss

    with HecDss(str(path)) as dss:
        cat = dss.get_catalog()
        want = pathname.strip("/").split("/")
        for p in cat.uncondensed_paths:
            parts = p.strip("/").split("/")
            if parts[0] == want[0] and parts[1].upper() == want[1].upper():
                ts = dss.get(p)
                return list(ts.times), list(ts.values)
    raise KeyError(pathname)
