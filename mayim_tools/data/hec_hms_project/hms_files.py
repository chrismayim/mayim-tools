"""
HEC-HMS 4.14 / 4.13 project files - zero QGIS dependency.

The text layouts follow what HEC-HMS 4.13 and 4.14 write when they save a
project (checked by opening, computing and re-saving generated projects
with HEC-HMS 4.13 and 4.14; the two write the same layouts and differ only
in the Version lines). Files use Windows line endings and UTF-8.

- <project>.hms        project: DSS file, basin, meteorologic and control
                       components
- <basin>.basin        sub-basins, reaches, junctions, sinks; coordinate
                       system and terrain; map view
- <basin>.sqlite       sub-basin polygons and reach lines (OGR SQLite
                       layout HEC-HMS uses: subbasin2d, reach2d ...)
- <met>.met            one per storm: Frequency Based Hypothetical, or
                       Specified Average with the storm's gage
- <control>.control    one per storm duration
- <project>.run        one simulation run per storm
- <project>.gage       precipitation gages (External DSS)
- <project>.terrain    terrain data (terrain/<name>/00/elevation.tif)
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from .core import (
    HMS_VERSION,
    ROUTE_MC,
    ROUTE_MUSKINGUM,
    TRANSFORM_CLARK,
    ProjectResult,
    file_stem,
)
from .storms import HMS_FREQ_DURATIONS

IND = "     "
CN_GRID_NAME = "Curve Numbers"
NL = "\r\n"
SEP = "\\"  # Windows file path separator, as on the user's machine
MET_METHODS_OFF = [
    ("Air Temperature Method", "None"),
    ("Atmospheric Pressure Method", "None"),
    ("Dew Point Method", "None"),
    ("Wind Speed Method", "None"),
    ("Shortwave Radiation Method", "None"),
    ("Longwave Radiation Method", "None"),
    ("Snowmelt Method", "None"),
    ("Evapotranspiration Method", "No Evapotranspiration"),
]


def _stamp(now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now()
    return f"{now.day} {now.strftime('%B %Y')}", now.strftime("%H:%M:%S")


def hms_date(t: datetime) -> str:
    """HEC-HMS control date: '1 January 2000'."""
    return f"{t.day} {t.strftime('%B %Y')}"


def hms_time(t: datetime) -> str:
    return t.strftime("%H:%M")


def num(v, nd: int = 6) -> str:
    """Plain number without thousands separators or trailing zeros."""
    if v is None:
        return ""
    x = round(float(v), nd)
    if x == int(x):
        return str(int(x))
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


class Block:
    def __init__(self, header: str):
        self.lines = [header]

    def kv(self, key: str, value) -> Block:
        self.lines.append(f"{IND}{key}: {value}")
        return self

    def blank(self) -> Block:
        self.lines.append("")
        return self

    def raw(self, line: str) -> Block:
        self.lines.append(line)
        return self

    def text(self) -> str:
        return NL.join(self.lines + ["End:", ""])


def _join(blocks: list[Block]) -> str:
    return NL.join(b.text() for b in blocks)


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return str(path)


# ---------------------------------------------------------------- names


class ProjectNames:
    """File names of the project components."""

    def __init__(self, r: ProjectResult, stem: str):
        self.stem = stem
        self.version = r.settings.hms_version
        self.basin_file = file_stem(r.settings.basin_name) + ".basin"
        self.sqlite_file = file_stem(r.settings.basin_name) + ".sqlite"
        self.met_files = {
            run.met_name: file_stem(run.met_name) + ".met" for run in r.runs
        }
        self.control_files = {
            c.name: file_stem(c.name) + ".control" for c in r.controls
        }
        self.rain_dss = file_stem(stem + "_design_storms") + ".dss"


# ---------------------------------------------------------------- project


def project_text(r: ProjectResult, n: ProjectNames) -> str:
    s = r.settings
    d, t = _stamp()
    blocks = [
        Block(f"Project: {s.project_name}")
        .kv("Description", s.description or "Prepared with Mayim Tools")
        .kv("Version", n.version)
        .kv("Filepath Separator", SEP)
        .kv("DSS File Name", f"{n.stem}.dss")
        .kv("Time Zone ID", s.time_zone)
    ]
    for run in r.runs:
        blocks.append(
            Block(f"Precipitation: {run.met_name}")
            .kv("Filename", n.met_files[run.met_name])
            .kv("Description", run.storm.description)
            .kv("Last Modified Date", d)
            .kv("Last Modified Time", t[:5])
        )
    blocks.append(
        Block(f"Basin: {s.basin_name}")
        .kv("Filename", n.basin_file)
        .kv("Description", s.description or "Prepared with Mayim Tools")
        .kv("Last Modified Date", d)
        .kv("Last Modified Time", t[:5])
    )
    for c in r.controls:
        blocks.append(
            Block(f"Control: {c.name}")
            .kv("FileName", n.control_files[c.name])
            .kv("Description", control_description(c))
        )
    return _join(blocks)


# ---------------------------------------------------------------- basin


def basin_text(
    r: ProjectResult,
    n: ProjectNames,
    crs_wkt: str = "",
    terrain_name: str | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    map_layers: list[tuple[str, str]] | None = None,
) -> str:
    """``map_layers``: [(display name, shapefile path relative to the project
    folder)] shown in the basin map (HEC-HMS map layers are vector only)."""
    s = r.settings
    d, t = _stamp()
    blocks = [
        Block(f"Basin: {s.basin_name}")
        .kv("Description", s.description or "Prepared with Mayim Tools")
        .kv("Last Modified Date", d)
        .kv("Last Modified Time", t)
        .kv("Version", s.hms_version)
        .kv("Filepath Separator", SEP)
        .kv("Unit System", "Metric")
        .kv("Missing Flow To Zero", "No")
        .kv("Enable Flow Ratio", "No")
        .kv("Compute Local Flow At Junctions", "No")
        .kv("Unregulated Output Required", "No")
        .blank()
        .kv("Enable Sediment Routing", "No")
    ]
    for e in r.network.elements:
        b = Block(f"{e.kind}: {e.name}")
        if e.description:
            b.kv("Description", e.description)
        b.kv("Last Modified Date", d).kv("Last Modified Time", t)
        p = e.params
        if e.kind == "Subbasin":
            if p.get("lat") is not None and p.get("lon") is not None:
                b.kv("Latitude Degrees", f"{p['lat']:.12f}")
                b.kv("Longitude Degrees", f"{p['lon']:.12f}")
            b.kv("Canvas X", num(e.x, 3)).kv("Canvas Y", num(e.y, 3))
            b.kv("Area", num(p["area_km2"], 6)).kv("Downstream", e.downstream)
            b.blank().kv("Discretization", "None").kv("File", n.sqlite_file)
            b.blank().kv("Canopy", "None")
            b.kv("Allow Simultaneous Precip Et", "No").kv("Plant Uptake Method", "None")
            b.blank().kv("Surface", "None")
            b.blank().kv("LossRate", "SCS")
            b.kv("Percent Impervious Area", num(p["imperv_pct"], 2))
            b.kv("Curve Number", num(p["cn"], 2))
            b.kv("Initial Abstraction", num(p["ia_mm"], 3))
            b.blank()
            if s.transform == TRANSFORM_CLARK:
                b.kv("Transform", "Clark").kv("Clark Method", "Specified")
                b.kv("Time of Concentration", num(p["clark_tc_h"], 4))
                b.kv("Storage Coefficient", num(p["clark_r_h"], 4))
                b.kv("Time Area Method", "Default")
            else:
                b.kv("Transform", "SCS").kv("Lag", num(p["lag_min"], 3))
                b.kv("Unitgraph Type", "STANDARD")
            b.blank().kv("Baseflow", "None")
        elif e.kind == "Reach":
            b.kv("Canvas X", num(e.x, 3)).kv("Canvas Y", num(e.y, 3))
            if e.from_xy:
                b.kv("From Canvas X", num(e.from_xy[0], 3))
                b.kv("From Canvas Y", num(e.from_xy[1], 3))
            b.kv("Downstream", e.downstream).blank()
            m = p.get("method")
            if m == ROUTE_MUSKINGUM:
                b.kv("Route", "Muskingum").kv("Initial Variable", "Combined Inflow")
                b.kv("Muskingum K", num(p["k_h"], 4)).kv("Muskingum x", num(p["x"], 3))
                b.kv("Muskingum Steps", int(p["steps"]))
            elif m == ROUTE_MC:
                b.kv("Route", "Muskingum Cunge").kv("Channel", "Trapezoid")
                b.kv("Length", num(p["length_m"], 2))
                b.kv("Energy Slope", num(p["slope"], 6))
                b.kv("Mannings n", num(p["n"], 4))
                b.kv("Bottom Width", num(p["width_m"], 3))
                b.kv("Side Slope", num(p["side_slope"], 3))
                b.kv("Initial Variable", "Combined Inflow")
                b.kv("Space-Time Method", "Automatic DX and DT")
                b.kv("Index Parameter Type", "Index Celerity")
                b.kv("Index Celerity", num(p["celerity_ms"], 3))
                b.kv("Maximum Depth Iterations", 20)
                b.kv("Maximum Route Step Iterations", 30)
            else:
                b.kv("Route", "Lag").kv("Initial Variable", "Combined Inflow")
                b.kv("Lag", num(p["lag_min"], 2))
            b.kv("Channel Loss", "None")
        elif e.kind == "Junction":
            b.kv("Canvas X", num(e.x, 3)).kv("Canvas Y", num(e.y, 3))
            b.kv("Downstream", e.downstream)
        else:  # Sink
            b.kv("Canvas X", num(e.x, 3)).kv("Canvas Y", num(e.y, 3))
        blocks.append(b)

    layers = Block("Basin Layer Properties:")
    for nm in ("Icons", "Subbasins"):
        layers.raw(f"{IND}Element Layer:").raw(f"{IND}     Name: {nm}")
        layers.raw(f"{IND}     Layer shown: Yes").raw(f"{IND}End Layer:")
    for display, rel in map_layers or []:
        layers.raw(f"{IND}Map Layer:")
        layers.raw(f"{IND}     Filename: {rel.replace('/', SEP)}")
        layers.raw(f"{IND}     Layer:0")
        layers.raw(f"{IND}     Display name: {display}")
        layers.raw(f"{IND}     Minimum Scale: Undefined")
        layers.raw(f"{IND}     Maximum Scale: Undefined")
        layers.raw(f"{IND}     Layer shown: Yes")
        layers.raw(f"{IND}End Layer:")
    blocks.append(layers)

    spatial = Block("Basin Spatial Properties:")
    if crs_wkt:
        spatial.kv("Coordinate System", crs_wkt)
    if terrain_name:
        spatial.kv("Terrain", terrain_name)
    blocks.append(spatial)

    sch = Block("Basin Schematic Properties:")
    if crs_wkt:
        sch.kv("Coordinate System", crs_wkt)
    if bounds is None:
        xs = [e.x for e in r.network.elements]
        ys = [e.y for e in r.network.elements]
        bounds = (min(xs), min(ys), max(xs), max(ys))
    w, so, ea, no = bounds
    pad = 0.05 * max(ea - w, no - so, 1.0)
    for key, val in (
        ("Last View N", no + pad),
        ("Last View S", so - pad),
        ("Last View W", w - pad),
        ("Last View E", ea + pad),
        ("Maximum View N", no + pad),
        ("Maximum View S", so - pad),
        ("Maximum View W", w - pad),
        ("Maximum View E", ea + pad),
    ):
        sch.kv(key, num(val, 3))
    for key, val in (
        ("Extent Method", "Elements"),
        ("Buffer", "0"),
        ("Draw Icons", "Yes"),
        ("Draw Icon Labels", "Name"),
        ("Draw Map Objects", "Yes"),
        ("Draw Gridlines", "No"),
        ("Draw Flow Direction", "No"),
        ("Draw HillShade Layer", "Yes" if terrain_name else "No"),
        ("Draw Elevation Layer", "Yes" if terrain_name else "No"),
        ("Elevation Layer Color Palette", "Default"),
        ("Ignore Elevation Color Ramp Scale", "No"),
        ("Use Interpolated Color Ramp for Elevation Layer", "Yes"),
        ("Color Ramp Opacity Level for Elevation Layer", "33.0"),
        ("Fix Element Locations", "No"),
        ("Fix Hydrologic Order", "No"),
    ):
        sch.kv(key, val)
    blocks.append(sch)
    return _join(blocks)


# ---------------------------------------------------------------- met / gage


def _met_header(
    name: str, description: str, method: str, basin: str, version: str = HMS_VERSION
) -> Block:
    d, t = _stamp()
    b = (
        Block(f"Meteorology: {name}")
        .kv("Description", description)
        .kv("Last Modified Date", d)
        .kv("Last Modified Time", t)
        .kv("Version", version)
        .kv("Unit System", "Metric")
        .kv("Set Missing Data to Default", "Yes")
        .kv("Precipitation Method", method)
    )
    for k, v in MET_METHODS_OFF:
        b.kv(k, v)
    b.kv("Use Basin Model", basin)
    return b


def met_text(r: ProjectResult, run) -> str:
    s = r.settings
    st = run.storm
    d, t = _stamp()
    subs = [e.name for e in r.network.of("Subbasin")]
    if st.is_hyetograph:
        blocks = [
            _met_header(
                run.met_name,
                st.description,
                "Specified Average",
                s.basin_name,
                s.hms_version,
            ),
            Block("Precip Method Parameters: Specified Average")
            .kv("Last Modified Date", d)
            .kv("Last Modified Time", t)
            .kv("Allow Depth Override", "No"),
        ]
        for sb in subs:
            blocks.append(
                Block(f"Subbasin: {sb}")
                .kv("Last Modified Date", d)
                .kv("Last Modified Time", t)
                .kv("Gage", run.gage_name)
            )
        return _join(blocks)
    area = s.storm_area_km2 or sum(
        e.params["area_km2"] for e in r.network.of("Subbasin")
    )
    p = (
        Block("Precip Method Parameters: Frequency Based Hypothetical")
        .kv("Last Modified Date", d)
        .kv("Last Modified Time", t)
        .kv("Storm Type", "Undefined")
        .kv("Single Hypothetical Storm Size", "Yes")
        .kv("Uniform Depth Duration Curve", "Yes")
        .kv("User Specified Storm Area", "Yes")
        .kv("Storm Size", num(area, 4))
        .kv("Re-sort Storm Symmetrically", "No")
        .kv("Total Duration", int(round(st.duration_min)))
        .kv("Time Interval", intensity_duration(run.control.dt_min))
        .kv("Percent of Duration Before Peak Rainfall", num(st.peak_pct))
    )
    if _version_at_least(s.hms_version, "4.14"):
        # 4.14 interpolates between the depths Log (default), Semi-Log or Linear;
        # Log matches the log-log DDF interpolation used for the depths.
        p.kv("Frequency Depths Interpolation Method", "Log")
    p.kv("Depth-Area Reduction Method", "No Reduction")
    for dur in HMS_FREQ_DURATIONS:
        v = st.depths.get(dur)
        p.kv(f"Depth {dur}", f"{v:.6f}" if v is not None else "")
    return _join(
        [
            _met_header(
                run.met_name,
                st.description,
                "Frequency Based Hypothetical",
                s.basin_name,
                s.hms_version,
            ),
            p,
        ]
    )


def _version_at_least(version: str, minimum: str) -> bool:
    def parts(v):
        return tuple(int(x) for x in str(v).split(".") if x.isdigit())

    return parts(version) >= parts(minimum)


def intensity_duration(dt_min: float) -> int:
    """Frequency storm intensity duration: the shortest HEC-HMS depth
    duration not shorter than the time step (5 min at the least)."""
    return next((d for d in HMS_FREQ_DURATIONS if d >= dt_min - 1e-9), 60)


def gage_pathname(gage_name: str, start: datetime, interval_min: float) -> str:
    from .dss import interval_part

    return f"//{gage_name.upper()}/PRECIP-INC/{start.strftime('%d%b%Y').upper()}/" + (
        f"{interval_part(interval_min)}/MAYIM/"
    )


def gage_text(r: ProjectResult, n: ProjectNames) -> str:
    d, t = _stamp()
    blocks = [
        Block(f"Gage Manager: {n.stem}")
        .kv("Gage Manager", n.stem)
        .kv("Version", n.version)
        .kv("Filepath Separator", SEP)
    ]
    for run in r.runs:
        if not run.gage_name:
            continue
        st = run.storm
        start = run.control.start
        end = run.control.end
        b = (
            Block(f"Gage: {run.gage_name}")
            .kv("Gage", run.gage_name)
            .kv("Gage Type", "Precipitation")
            .kv("Description", st.description)
            .kv("Last Modified Date", d)
            .kv("Last Modified Time", t)
            .kv("Reference Height Units", "Meters")
            .kv("Reference Height", "10")
            .kv("Data Source Type", "External DSS")
            .kv("Filename", n.rain_dss)
            .kv("Pathname", gage_pathname(run.gage_name, start, st.interval_min))
            .raw(f"{IND}Variant: Variant-1")
            .raw(f"{IND}  Start Time: {hms_date(start)}, {hms_time(start)}")
            .raw(f"{IND}  End Time: {hms_date(end)}, {hms_time(end)}")
            .raw(f"{IND}End Variant: Variant-1")
        )
        blocks.append(b)
    return _join(blocks)


def gage_series(run) -> tuple[datetime, float, list[float]]:
    """(time of the first value, interval, values) covering the whole
    control window (zeros after the storm), values at interval ends."""
    st = run.storm
    dt = st.interval_min
    start = run.control.start + timedelta(minutes=dt)
    span = (run.control.end - run.control.start).total_seconds() / 60.0
    n = int(round(span / dt))
    vals = list(st.increments) + [0.0] * max(0, n - len(st.increments))
    return start, dt, [float(v) for v in vals]


# ---------------------------------------------------------------- control / run


def control_description(c) -> str:
    from .storms import duration_label

    span = (c.end - c.start).total_seconds() / 3600.0
    if c.name == "Design storms":
        return (
            f"All design storms (up to {duration_label(c.duration_min)}), "
            f"{num(span, 2)} h window"
        )
    return f"Design storm of {duration_label(c.duration_min)}, {num(span, 2)} h window"


def control_text(c, version: str = HMS_VERSION) -> str:
    d, t = _stamp()
    return (
        Block(f"Control: {c.name}")
        .kv("Description", control_description(c))
        .kv("Last Modified Date", d)
        .kv("Last Modified Time", t)
        .kv("Version", version)
        .kv("Start Date", hms_date(c.start))
        .kv("Start Time", hms_time(c.start))
        .kv("End Date", hms_date(c.end))
        .kv("End Time", hms_time(c.end))
        .kv("Time Interval", num(c.dt_min))
        .text()
    )


def run_text(r: ProjectResult) -> str:
    d, t = _stamp()
    blocks = []
    for run in r.runs:
        blocks.append(
            Block(f"Run: {run.name}")
            .kv("Description", run.storm.description)
            .kv("Log File", run.dss_file[:-4] + ".log")
            .kv("DSS File", run.dss_file)
            .kv("Is Save Spatial Results", "No")
            .kv("Last Modified Date", d)
            .kv("Last Modified Time", t)
            .kv("Basin", r.settings.basin_name)
            .kv("Precip", run.met_name)
            .kv("Control", run.control.name)
            .kv("Save State Type", "None")
            .kv("Time-Series Output", "Save All")
            .kv("Time Series Results Manager Start", "")
            .kv("Time Series Results Manager End", "")
        )
    text = _join(blocks)
    return text.replace(f"Manager Start: {NL}", f"Manager Start:{NL}").replace(
        f"Manager End: {NL}", f"Manager End:{NL}"
    )


def terrain_text(n: ProjectNames, terrain_name: str, description: str) -> str:
    d, t = _stamp()
    return _join(
        [
            Block(f"Terrain Data Manager: {n.stem}")
            .kv("Version", n.version)
            .kv("Filepath Separator", SEP),
            Block(f"Terrain Data: {terrain_name}")
            .kv("Description", description)
            .kv("Terrain Directory", f"terrain{SEP}{file_stem(terrain_name)}")
            .kv("Vertical Units", "Meters")
            .kv("Last Modified Date", d)
            .kv("Last Modified Time", t),
        ]
    )


def terrain_relpath(terrain_name: str) -> str:
    """Elevation file HEC-HMS reads: <terrain directory>/00/elevation.tif."""
    return f"terrain/{file_stem(terrain_name)}/00/elevation.tif"


# ---------------------------------------------------------------- grid data


def grid_text(n: ProjectNames, grids: list[tuple[str, str, str, str]]) -> str:
    """<project>.grid: grid data entries [(name, grid type, GeoTIFF path
    relative to the project folder, description)], e.g. the curve number
    grid of the Gridded SCS loss. HEC-HMS needs the Last Modified date and
    time of every entry (it fails to open the project without them)."""
    d, t = _stamp()
    blocks = [
        Block(f"Grid Manager: {n.stem}")
        .kv("Grid Manager", n.stem)
        .kv("Version", n.version)
        .kv("Filepath Separator", SEP)
    ]
    for name, gtype, rel, desc in grids:
        blocks.append(
            Block(f"Grid: {name}")
            .kv("Grid", name)
            .kv("Grid Type", gtype)
            .kv("Description", desc)
            .kv("Last Modified Date", d)
            .kv("Last Modified Time", t)
            .kv("Data Source Type", "GeoTIFF")
            .kv("Filename", rel.replace("/", SEP))
        )
    return _join(blocks)


# ---------------------------------------------------------------- ensembles


def hms_component_file(name: str, ext: str) -> str:
    """File name HEC-HMS gives a component of this name: every character
    other than A-Z, a-z, 0-9, '_', '+' and '.' becomes '_' (not collapsed)."""
    return re.sub(r"[^A-Za-z0-9_+.]", "_", name) + ext


def ensemble_text(ens, r: ProjectResult) -> str:
    """ensemble/<name>.ens: an HEC-HMS ensemble analysis of simulation runs,
    with every element's outflow selected as an analysis point (HEC-HMS
    refuses to compute an ensemble without one). The members must share the
    computation time interval - and, as found when computing, the time
    window - so the project then has one control specification."""
    d, t = _stamp()
    stem = hms_component_file(ens.name, "")
    lines = [
        f"Ensemble Analysis: {ens.name}",
        f"{IND}Description: {len(ens.runs)} runs - all durations and peak positions "
        "/ patterns of this storm method and return period",
        f"{IND}Version: {r.settings.hms_version}",
        f"{IND}Log File: {stem}.log",
        f"{IND}DSS File: {stem}.dss",
        f"{IND}Last Modified Date: {d}",
        f"{IND}Last Modified Time: {t}",
        "",
        f"{IND}Ensemble Member Type: Simulation Run",
    ]
    lines += [f"{IND}     Ensemble Member: {run.name}" for run in ens.runs]
    lines += [
        "",
        f"{IND}Time-Series Output: Save All",
        f"{IND}Time Series Results Manager Start:",
        f"{IND}Time Series Results Manager End:",
    ]
    for e in r.network.elements:
        lines += [
            "",
            f"{IND}Analysis Point: {e.name}",
            f"{IND}Time Series Type: Outflow",
            f"{IND}End Analysis Point: ",
        ]
    lines += ["", f"End Ensemble Analysis: {ens.name}", ""]
    return NL.join(lines)


# ---------------------------------------------------------------- run-all script


def run_all_script(r: ProjectResult, n: ProjectNames, folder: str) -> str:
    runs = ",\n    ".join(repr(run.name) for run in r.runs)
    return (
        "# Computes every simulation run of this project in HEC-HMS (Jython).\n"
        "# Start it with run_all_simulations.cmd, or from a command prompt:\n"
        '#   "C:\\Program Files\\HEC\\HEC-HMS\\<version>\\HEC-HMS.cmd" -s '
        '"<this folder>\\run_all_simulations.py"\n'
        "# Written by Mayim Tools - Prepare HEC-HMS project.\n"
        "import os\n"
        "from hms.model.JythonHms import *\n"
        "\n"
        f"PROJECT = {r.settings.project_name!r}\n"
        f"FOLDER = os.environ.get('MAYIM_HMS_FOLDER') or {folder!r}\n"
        "RUNS = [\n"
        f"    {runs},\n"
        "]\n"
        "\n"
        "OpenProject(PROJECT, FOLDER)\n"
        "failed = []\n"
        "for run in RUNS:\n"
        "    try:\n"
        "        Compute(run)\n"
        "    except Exception as e:\n"
        "        failed.append(run)\n"
        "        print('Run %s failed: %s' % (run, e))\n"
        "print('%d of %d runs computed.' % (len(RUNS) - len(failed), len(RUNS)))\n"
        "Exit(1)\n"
    ).replace("\n", NL)


def run_all_cmd(hms_dir: str) -> str:
    return NL.join(
        [
            "@echo off",
            "rem Computes every simulation run of this HEC-HMS project.",
            "rem Written by Mayim Tools - Prepare HEC-HMS project.",
            'set "MAYIM_HMS_FOLDER=%~dp0"',
            'if "%MAYIM_HMS_FOLDER:~-1%"=="\\" set "MAYIM_HMS_FOLDER=%MAYIM_HMS_FOLDER:~0,-1%"',
            f'set "HMS_DIR={hms_dir}"',
            'if not exist "%HMS_DIR%\\HEC-HMS.cmd" (',
            "  echo HEC-HMS was not found in %HMS_DIR% - edit HMS_DIR in this file.",
            "  pause",
            "  exit /b 1",
            ")",
            'pushd "%HMS_DIR%"',
            'call HEC-HMS.cmd -s "%MAYIM_HMS_FOLDER%\\run_all_simulations.py"',
            "popd",
            "pause",
            "",
        ]
    )


# ---------------------------------------------------------------- SQLite geometry

_SQLITE_TABLES = {
    # name: (geometry type code, extra columns)
    "subbasin": (1, ""),
    "source": (1, ""),
    "junction": (1, ""),
    "reach": (2, ""),
    "reservoir": (1, ""),
    "diversion": (1, ""),
    "sink": (1, ""),
    "subbasin2d": (
        6,
        ", 'centroid_x' FLOAT, 'centroid_y' FLOAT, 'area_sqkm' FLOAT, "
        "'latitude' FLOAT, 'longitude' FLOAT",
    ),
    "reach2d": (2, ""),
    "reservoir2d": (6, ""),
}


def write_sqlite(
    path: Path, r: ProjectResult, srid: int, srtext: str, auth: str = "EPSG"
) -> str:
    """Basin geometry in the OGR SQLite layout of HEC-HMS 4.13 / 4.14 projects:
    sub-basin polygons in subbasin2d and reach lines in reach2d (the point
    tables stay empty, as in the HEC-HMS sample projects)."""
    path = Path(path)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE geometry_columns (     f_table_name VARCHAR,      "
            "f_geometry_column VARCHAR,      geometry_type INTEGER,      "
            "coord_dimension INTEGER,      srid INTEGER,     geometry_format VARCHAR )"
        )
        cur.execute(
            "CREATE TABLE spatial_ref_sys        (     srid INTEGER UNIQUE,     "
            "auth_name TEXT,     auth_srid TEXT,     srtext TEXT)"
        )
        cur.execute(
            "INSERT INTO spatial_ref_sys VALUES (?,?,?,?)",
            (srid, auth, str(srid), srtext),
        )
        for name, (gtype, extra) in _SQLITE_TABLES.items():
            cur.execute(
                f"CREATE TABLE '{name}' ( ogc_fid INTEGER PRIMARY KEY, "
                f"'GEOMETRY' BLOB, 'name' VARCHAR{extra})"
            )
            cur.execute(
                "INSERT INTO geometry_columns VALUES (?,?,?,?,?,?)",
                (name, "GEOMETRY", gtype, 2, srid, "WKB"),
            )
        for e in r.network.of("Subbasin"):
            p = e.params
            if e.wkb:
                cur.execute(
                    "INSERT INTO subbasin2d (GEOMETRY, name, centroid_x, centroid_y, "
                    "area_sqkm, latitude, longitude) VALUES (?,?,?,?,?,?,?)",
                    (
                        e.wkb,
                        e.name,
                        e.x,
                        e.y,
                        p.get("area_km2"),
                        p.get("lat"),
                        p.get("lon"),
                    ),
                )
        for e in r.network.of("Reach"):
            if e.wkb:
                cur.execute(
                    "INSERT INTO reach2d (GEOMETRY, name) VALUES (?,?)", (e.wkb, e.name)
                )
        con.commit()
    finally:
        con.close()
    return str(path)
