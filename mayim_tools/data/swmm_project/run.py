"""
Runs every design storm of a SWMM model in the EPA SWMM 5 engine (the
swmm-toolkit package: python -m pip install swmm-toolkit) and reads the peak
node inflows and continuity errors from the reports. Zero QGIS dependency.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .inp import write_inp

INSTALL_HINT = (
    "Computing the storms needs the EPA SWMM engine for Python. In the OSGeo4W Shell "
    "(or QGIS's Python) run:  python -m pip install swmm-toolkit"
)


class SwmmUnavailableError(RuntimeError):
    pass


@dataclass
class RunResult:
    run: str
    series: str
    peaks: dict = field(default_factory=dict)  # node -> (peak m3/s, "days hh:mm")
    runoff_error_pct: float | None = None
    routing_error_pct: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def solver():
    try:
        from swmm.toolkit import solver as sv
    except ImportError as e:
        raise SwmmUnavailableError(INSTALL_HINT) from e
    return sv


def engine_version() -> str:
    try:
        sv = solver()
        return str(sv.swmm_version_info())
    except Exception:  # noqa: BLE001
        return "?"


_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def parse_report(text: str) -> tuple[dict, float | None, float | None, list, list]:
    """(peaks {node: (max total inflow, time)}, runoff continuity %, routing
    continuity %, errors, warnings) from a SWMM 5 report."""
    errors = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("ERROR")]
    warns = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("WARNING")]
    cont = re.findall(r"Continuity Error \(%\) \.+\s*(" + _NUM + ")", text)
    runoff = float(cont[0]) if len(cont) > 0 else None
    routing = float(cont[1]) if len(cont) > 1 else None
    peaks = {}
    i = text.find("Node Inflow Summary")
    if i >= 0:
        block = text[i:].split("\n")
        dashes = 0
        for ln in block[1:]:
            if ln.strip().startswith("---"):
                dashes += 1
                if dashes >= 3:
                    break
                continue
            if dashes == 2 and ln.strip():
                parts = ln.split()
                if len(parts) >= 6:
                    try:
                        peaks[parts[0]] = (float(parts[3]), f"{parts[4]} {parts[5]}")
                    except ValueError:
                        continue
            elif dashes == 2 and not ln.strip():
                break
    return peaks, runoff, routing, errors, warns


def _run_quiet(sv, inp: str, rpt: str, out: str) -> None:
    """swmm_run without its console progress lines: open, step to the end,
    write the report, close."""
    sv.swmm_open(inp, rpt, out)
    try:
        sv.swmm_start(1)
        try:
            while sv.swmm_step() != 0:
                pass
        finally:
            sv.swmm_end()
            sv.swmm_report()
    finally:
        sv.swmm_close()


def run_all(
    model, workdir: str | None = None, log=print, feedback=None
) -> list[RunResult]:
    """Runs each storm (the rain gage switched to its time series)."""
    sv = solver()
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="mayim_swmm_"))
    tmp.mkdir(parents=True, exist_ok=True)
    out = []
    for i, ser in enumerate(model.series):
        if feedback is not None and feedback.isCanceled():
            break
        stem = f"run_{i + 1:03d}"
        inp = tmp / f"{stem}.inp"
        rpt = tmp / f"{stem}.rpt"
        bin_ = tmp / f"{stem}.out"
        write_inp(inp, model, ser.name)
        res = RunResult(ser.run_name, ser.name)
        try:
            _run_quiet(sv, str(inp), str(rpt), str(bin_))
        except Exception as e:  # noqa: BLE001 - reported per run
            res.errors.append(str(e))
        if rpt.exists():
            text = rpt.read_text(errors="replace")
            res.peaks, res.runoff_error_pct, res.routing_error_pct, errs, warns = (
                parse_report(text)
            )
            res.errors += errs
            res.warnings += warns
        try:
            bin_.unlink(missing_ok=True)
        except OSError:
            pass
        out.append(res)
        if feedback is not None:
            feedback.setProgress(100.0 * (i + 1) / len(model.series))
    return out
