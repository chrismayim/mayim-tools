"""
Running the simulations in HEC-HMS and reading back the peak flows -
zero QGIS dependency.

HEC-HMS computes the runs through its Jython scripting (HEC-HMS.cmd -s
script.py, run from the HEC-HMS program folder). Each computed run leaves
results/RUN_<run>.results (XML) with the statistics of every element.
"""

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .core import file_stem


@dataclass
class ElementResult:
    run: str
    element: str
    kind: str
    peak_m3s: float | None
    peak_time: str
    volume_m3: float | None
    depth_mm: float | None
    precip_mm: float | None


def find_hms(hms_dir: str) -> str | None:
    """Path of HEC-HMS.cmd in ``hms_dir`` (or None)."""
    if not hms_dir:
        return None
    p = Path(hms_dir) / "HEC-HMS.cmd"
    return str(p) if p.is_file() else None


def run_hms(
    hms_dir: str, project_folder: str, timeout_s: int = 7200
) -> tuple[int, str]:
    """Computes every run with the project's run_all_simulations.py.
    Returns (exit code, console output)."""
    cmd = find_hms(hms_dir)
    if cmd is None:
        raise FileNotFoundError(f"HEC-HMS.cmd was not found in {hms_dir}.")
    script = str(Path(project_folder) / "run_all_simulations.py")
    env = dict(os.environ, MAYIM_HMS_FOLDER=str(project_folder))
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        args = ["cmd.exe", "/c", "HEC-HMS.cmd", "-s", script]
    else:
        args = ["sh", cmd, "-s", script]
    proc = subprocess.run(
        args,
        cwd=hms_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        **kwargs,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_run(path: str | Path, run: str) -> list[ElementResult]:
    root = ET.parse(str(path)).getroot()
    out = []
    for be in root.iter("BasinElement"):
        st = {m.get("type"): m.get("value") for m in be.iter("StatisticMeasure")}
        out.append(
            ElementResult(
                run=run,
                element=be.get("name") or "",
                kind=be.get("type") or "",
                peak_m3s=_f(st.get("Outflow Maximum")),
                peak_time=st.get("Outflow Maximum Time") or "",
                volume_m3=_f(st.get("Outflow Volume")),
                depth_mm=_f(st.get("Outflow Depth")),
                precip_mm=_f(st.get("Precipitation Total")),
            )
        )
    return out


def read_results(project_folder: str, run_names: list[str]) -> tuple[list, list[str]]:
    """Element results of every run that has a results file, and the runs
    that have none (not computed, or failed)."""
    folder = Path(project_folder) / "results"
    rows, missing = [], []
    for run in run_names:
        candidates = [
            folder / f"RUN_{run}.results",
            folder / f"RUN_{file_stem(run)}.results",
        ]
        p = next((c for c in candidates if c.is_file()), None)
        if p is None:
            missing.append(run)
            continue
        rows.extend(read_run(p, run))
    return rows, missing


def run_log_errors(project_folder: str, dss_file: str) -> list[str]:
    """ERROR lines of a run's log file."""
    log = Path(project_folder) / (dss_file[:-4] + ".log")
    if not log.is_file():
        return []
    with open(log, encoding="utf-8", errors="replace") as f:
        return [ln.strip() for ln in f if ln.startswith("ERROR")]
