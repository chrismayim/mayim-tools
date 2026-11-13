"""Processing Toolbox algorithm: Import RAS result data.

Reads a computed HEC-RAS 2D plan (<project>.p##.hdf) and writes, for the
peak of the run (Chris's choices, 2026-11-13):

- result rasters on the terrain grid: maximum depth, maximum water level,
  maximum velocity, depth x velocity, flood hazard H1-H6, arrival time and
  duration of flooding; the water surface slopes between cells as in RAS
  Mapper (see rasters.py);
- the inundation boundary (polygons of the maximum flood extent);
- peak flow and stage across profile lines (RAS Mapper's Profile Lines, the
  geometry's Reference Lines, or any line layer), and the peak values of the
  structures (SA/2D connections) and boundary conditions, in a CSV.

HEC-RAS does not write the result rasters by default (RAS Mapper makes them
on demand), so they are computed here from the cell results and the
terrain. The reading and the calculations are in core.py and rasters.py
(no QGIS); this file is the QGIS wrapper.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    RASTER,
    SHAPEFILE,
    clean_name_part,
    output_file_name,
)
from mayim_tools.hydrology._common.grid_polygons import (
    group_polygons,
    ring_to_map,
    simplify_ring,
    trace_rings,
)
from mayim_tools.hydrology.catchment_delineation.qgis_outputs import save_shapefile
from mayim_tools.hydrology.inundation_boundary.qgis_outputs import (
    load_on_completion,
    memory_layer,
)
from mayim_tools.hydrology.terrain_storage.terrain_storage_algorithm import (
    _rescale_pseudocolor,
)

from .core import (
    TOOL_VERSION,
    ProfileLine,
    ResultsError,
    find_terrain,
    rasmap_profile_lines,
    read_plan_results,
)
from .export import profile_rows, write_peaks_csv
from .rasters import (
    HAZARD_CLASSES,
    METHOD_KEYS,
    METHODS,
    build_grids,
    write_grid,
)

# closest to RAS Mapper's own flood edge on project 2625038 (see help)
DEFAULT_METHOD = "faces_shallow"

STYLES_DIR = Path(__file__).resolve().parent / "styles"

_S = QMetaType.Type.QString
_I = QMetaType.Type.Int
_D = QMetaType.Type.Double

# key, label, file description (after <project>_RAS_<plan>_), kind, default,
# grid attribute, style, rescale the style's ramp to the data
OUTPUTS = [
    ("OUT_DEPTH", "Maximum depth (raster)", "Max_Depth", RASTER, True,
     "depth", "RAS_Depth.qml", True),
    ("OUT_WSE", "Maximum water level (raster)", "Max_WSE", RASTER, True,
     "wse", "RAS_WSE.qml", True),
    ("OUT_VEL", "Maximum velocity (raster)", "Max_Velocity", RASTER, True,
     "velocity", "RAS_Velocity_cont_70pct.qml", True),
    ("OUT_DV", "Maximum depth x velocity (raster)", "Max_DxV", RASTER, True,
     "dv", "RAS_Velocity_cont_70pct.qml", True),
    ("OUT_HAZARD", "Flood hazard H1-H6 (raster)", "Hazard_H1_H6", RASTER, True,
     "hazard", "RAS_ARR_Flood_Hazard_70pct.qml", False),
    ("OUT_ARRIVAL", "Arrival time (raster, hours)", "Arrival_Time_h", RASTER,
     True, "arrival_h", "RAS_Velocity_cont_70pct.qml", True),
    ("OUT_DURATION", "Duration of flooding (raster, hours)", "Duration_h",
     RASTER, True, "duration_h", "RAS_Velocity_cont_70pct.qml", True),
    ("OUT_BOUNDARY", "Inundation boundary (polygons)", "Inundation_Boundary",
     SHAPEFILE, True, None, None, False),
    ("OUT_PROFILES", "Peak flow and stage at the profile lines (lines)",
     "Profile_Lines_Peaks", SHAPEFILE, True, None, "RAS_profile_lines.qml",
     False),
    ("OUT_CSV", "Peak values: structures, boundary conditions, profile lines "
     "(CSV)", "Peak_Values", CSV, True, None, None, False),
]  # fmt: skip
OUTPUT_BY_KEY = {o[0]: o for o in OUTPUTS}

FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
}

BOUNDARY_FIELDS = [
    ("id", _I, "polygon number"),
    ("area_m2", _D, "flooded area, m2"),
    ("plan", _S, "HEC-RAS plan"),
]
PROFILE_FIELDS = [
    ("Name", _S, "profile line name"),
    ("Area_2D", _S, "2D flow area"),
    ("Q_peak", _D, "peak flow across the line, m3/s (positive downstream)"),
    ("T_Qpeak_h", _D, "time of the peak flow, hours from the start"),
    ("Vol_m3", _D, "volume passed across the line, m3"),
    ("WSE_max", _D, "highest peak water level on the line, m"),
    ("WSE_mean", _D, "mean peak water level on the line, m"),
    ("D_max", _D, "largest peak depth on the line, m"),
    ("V_max", _D, "largest peak velocity on the line, m/s"),
    ("Faces", _I, "mesh faces cut by the line"),
    ("Note", _S, "why a value is missing or approximate"),
]
_PROFILE_MAP = {
    "Name": "name", "Area_2D": "area_2d", "Q_peak": "q_peak_m3s",
    "T_Qpeak_h": "t_q_peak_h", "Vol_m3": "volume_m3", "WSE_max": "wse_max_m",
    "WSE_mean": "wse_mean_m", "D_max": "depth_max_m", "V_max": "v_max_ms",
    "Faces": "faces", "Note": "note",
}  # fmt: skip


def plan_label(plan) -> str:
    """File-name part for the plan: its short id, e.g. '2D_flow - 2pct_AEP'
    -> '2D_flow_2pct_AEP'."""
    text = plan.short_id or plan.flow_title or plan.plan_name or Path(plan.path).stem
    text = text.replace(" - ", "_")
    return clean_name_part(text)


def output_name(project_number: str, plan: str, description: str, kind: str):
    return output_file_name(project_number, f"RAS_{plan}_{description}", kind)


def _software_version() -> str:
    parts = []
    metadata = Path(__file__).resolve().parents[2] / "metadata.txt"
    try:
        for line in metadata.read_text(encoding="utf-8").splitlines():
            if line.startswith("version="):
                parts.append(f"Mayim Tools {line.split('=', 1)[1].strip()}")
                break
    except OSError:
        pass
    parts.append(f"QGIS {Qgis.version().split('-')[0]}")
    return ", ".join(parts)


def _num(value, digits=3):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return round(value, digits) if np.isfinite(value) else None


class _Styler(QgsProcessingLayerPostProcessorInterface):
    """Loads a QML style once the layer is in the project; a pseudocolour
    ramp is stretched to this run's range."""

    def __init__(self, style: str, raster_range=None, aliases=None):
        super().__init__()
        self.style = str(STYLES_DIR / style)
        self.raster_range = raster_range
        self.aliases = aliases or {}

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            layer.loadNamedStyle(self.style)
            if self.raster_range is not None:
                _rescale_pseudocolor(layer, *self.raster_range)
            for name, alias in self.aliases.items():
                idx = layer.fields().indexOf(name)
                if idx >= 0:
                    layer.setFieldAlias(idx, alias)
            layer.triggerRepaint()
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style not applied to {layer.name()}: {e}")


class ImportRasResultsAlgorithm(QgsProcessingAlgorithm):
    PLAN_HDF = "PLAN_HDF"
    TERRAIN = "TERRAIN"
    CRS = "CRS"
    PROFILE_LINES = "PROFILE_LINES"
    NAME_FIELD = "NAME_FIELD"
    THRESHOLD = "THRESHOLD"
    METHOD = "METHOD"
    PROJECT_NUMBER = "PROJECT_NUMBER"

    def createInstance(self):
        return ImportRasResultsAlgorithm()

    def name(self):
        return "import_ras_results"

    def displayName(self):
        return "Import RAS result data"

    def group(self):
        return "HEC-RAS Tools"

    def groupId(self):
        return "hec_ras_tools"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def shortHelpString(self):
        p = "<project number>_RAS_<plan>_"
        files = "\n".join(
            f"  {label}: {p}{desc}{'.tif' if kind == RASTER else '.shp' if kind == SHAPEFILE else '.csv'}"  # noqa: E501
            for _, label, desc, kind, *_ in OUTPUTS
        )
        hazards = "\n".join(f"  {text}" for text in HAZARD_CLASSES.values())
        return (
            f"Import RAS result data (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tReads the results of a computed HEC-RAS 2D plan "
            "(<project>.p##.hdf, HEC-RAS 6.x/7.0) and writes the peak "
            "(maximum) result rasters, the inundation boundary, and the peak "
            "flow and stage at profile lines, structures and boundary "
            "conditions. HEC-RAS does not write result rasters by default, so "
            "they are computed here from the cell results on the terrain grid, "
            "as RAS Mapper does.\n"
            "\n"
            "WATER SURFACE:\tThe five RAS Mapper renderings. Corner levels "
            "come from the wet cells around the corner (1) or from the faces "
            "meeting there (2-4); inside a cell the level is interpolated "
            "linearly between the cell centre, its corners and (2-4) its face "
            "centres.\n"
            "  1. Sloping (cell corners).\n"
            "  2. Sloping (cell corners + face centres): face level = mean of "
            "the two cells (the wet cell where only one side is wet and the "
            "water reaches the face).\n"
            "  3. As 2, with depth-weighted faces (RAS Mapper's 'precip mode'): "
            "the deeper cell counts more at the face.\n"
            "  4. As 2, with shallow water reducing to horizontal (default): "
            "the sloping surface is blended to the cell's own level by how much "
            "of the cell's terrain the water covers - fully covered cells "
            "slope, just-wet cells are flat.\n"
            "  5. Horizontal: one level per cell.\n"
            "On project 2625038 (0.5 m terrain) the distance from RAS Mapper's "
            "own flood edge was: 1 and 3 about 0.9 m median, 3.7 m at 90%; 2 "
            "0.8 m and 3.7 m; 4 and 5 0.5 m and 1.5 m. 1-3 give too small a "
            "flood on steep, shallow flow, because a corner averages in the "
            "lower cells downhill. Depth = water level - terrain, where above "
            "the depth threshold.\n"
            "\n"
            "VELOCITY AND HAZARD:\tCell velocities are rebuilt at every output "
            "time from the face velocities (least squares, weighted by face "
            "length). Depth x velocity and the hazard use the cell's peak depth "
            "and peak velocity, as a RAS Mapper calculated layer on the 'Max' "
            "profiles does. Flood hazard classes (Smith, Davey and Cox 2014; "
            "ARR 2019 Book 6 Chapter 7):\n"
            f"{hazards}\n"
            "\n"
            "ARRIVAL AND DURATION:\tHours from the start of the run until the "
            "cell is first deeper than the threshold, and the hours it stays "
            "so (at the plan's output interval).\n"
            "\n"
            "PROFILE LINES:\tThe peak flow across a line comes from the mass "
            "balance of the mesh upstream of it (inflow minus the change in "
            "stored volume), which matches HEC-RAS's own boundary flows. Lines "
            "are taken from the 'Profile lines' layer when one is given, "
            "otherwise from RAS Mapper's Profile Lines (<project>.rasmap) and "
            "the geometry's Reference Lines. Draw a line from the left bank to "
            "the right bank looking downstream (positive flow is downstream) "
            "and across the whole flooded width, onto dry ground; a line that "
            "water can pass around gets no flow and a note. Flow through a "
            "structure is included where it passes the line. The stage is the "
            "peak water level of the cells on the line (highest and mean).\n"
            "\n"
            "PARAMETERS:\n"
            "  Plan results HDF file: <project>.p##.hdf of a computed plan.\n"
            "  Terrain (optional): default the terrain the geometry uses "
            "(the .vrt next to the RAS terrain .hdf).\n"
            "  CRS (optional): default the projection in the plan file.\n"
            "  Profile lines (optional) and their name field.\n"
            "  Depth threshold: depths at or below it count as dry "
            "(default 0.001 m, as RAS Mapper).\n"
            "\n"
            "OUTPUTS (tick the ones wanted; <plan> is the plan's short id, "
            "e.g. 2D_flow_2pct_AEP):\n"
            f"{files}\n"
            "\n"
            "PROFILE LINE FIELDS:\t"
            + "; ".join(f"{n} = {d}" for n, _, d in PROFILE_FIELDS)
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.PLAN_HDF, "Plan results HDF file (<project>.p##.hdf)",
                extension="hdf",
            )
        )  # fmt: skip
        self.addParameter(
            QgsProcessingParameterFile(
                self.TERRAIN,
                "Terrain raster (optional; default the geometry's terrain)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.CRS, "CRS (optional; default the plan's projection)",
                optional=True,
            )
        )  # fmt: skip
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.PROFILE_LINES,
                "Profile lines (optional; default RAS Mapper's profile lines "
                "and the reference lines)",
                types=[QgsProcessing.SourceType.TypeVectorLine],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.NAME_FIELD, "Profile line name field (optional)",
                optional=True,
            )
        )  # fmt: skip
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METHOD,
                "Water surface rendering (as in RAS Mapper)",
                options=[f"{i + 1}. {METHODS[k]}" for i, k in enumerate(METHOD_KEYS)],
                defaultValue=METHOD_KEYS.index(DEFAULT_METHOD),
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.THRESHOLD, "Depth threshold (m)",
                type=QgsProcessingParameterNumber.Type.Double, minValue=0.0,
                defaultValue=0.001,
            )
        )  # fmt: skip
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER, "Project number (file name prefix, optional)",
                optional=True,
            )
        )  # fmt: skip
        for key, label, _, _, default, *_ in OUTPUTS:
            self.addParameter(
                QgsProcessingParameterBoolean(
                    key, f"Output: {label}", defaultValue=default
                )
            )
        for name, label in FOLDER_PARAMS.values():
            self.addParameter(
                QgsProcessingParameterFolderDestination(name, label, optional=True)
            )

    # ------------------------------------------------------------------

    def _profile_lines(self, parameters, context, crs, plan_path, feedback):
        """[ProfileLine] in the plan's CRS and where they came from."""
        source = self.parameterAsSource(parameters, self.PROFILE_LINES, context)
        name_field = (
            self.parameterAsString(parameters, self.NAME_FIELD, context) or ""
        ).strip()
        layers = []
        if source is not None:
            layers.append((source, "the profile lines layer"))
        else:
            path = rasmap_profile_lines(plan_path)
            if path:
                layer = QgsVectorLayer(path, "Profile Lines", "ogr")
                if layer.isValid():
                    layers.append((layer, f"RAS Mapper's profile lines ({path})"))
        lines = []
        used = []
        for src, label in layers:
            transform = None
            src_crs = src.sourceCrs() if hasattr(src, "sourceCrs") else src.crs()
            if src_crs.isValid() and crs.isValid() and src_crs != crs:
                transform = QgsCoordinateTransform(
                    src_crs, crs, context.transformContext()
                )
            names = [f.name() for f in src.fields()]
            field = name_field if name_field in names else next(
                (n for n in names if n.lower() in ("name", "profilenam", "id")), None
            )  # fmt: skip
            n0 = len(lines)
            for k, feat in enumerate(src.getFeatures(), start=1):
                geom = QgsGeometry(feat.geometry())
                if geom.isNull() or geom.isEmpty():
                    continue
                if transform is not None:
                    geom.transform(transform)
                parts = (
                    geom.asMultiPolyline() if geom.isMultipart()
                    else [geom.asPolyline()]
                )  # fmt: skip
                base = str(feat[field]) if field else f"Line {k}"
                for j, part in enumerate(parts):
                    if len(part) < 2:
                        continue
                    name = base if len(parts) == 1 else f"{base} ({j + 1})"
                    lines.append(ProfileLine(name, [(p.x(), p.y()) for p in part]))
            if len(lines) > n0:
                used.append(f"{len(lines) - n0} from {label}")
        if source is None:
            try:
                from mayim_tools.hec_ras.import_ras_2d_data.reference_lines_points.core import (  # noqa: E501
                    read_reference_lines,
                )

                refs = read_reference_lines(plan_path)
            except Exception as e:
                refs = []
                feedback.pushWarning(f"Reference lines not read: {e}")
            n0 = len(lines)
            for ref in refs:
                for part in ref.parts:
                    if len(part) >= 2:
                        lines.append(
                            ProfileLine(
                                str(ref.attributes.get("Name", "Reference line")),
                                [tuple(map(float, p)) for p in part],
                            )
                        )
            if len(lines) > n0:
                used.append(f"{len(lines) - n0} reference line(s) from the geometry")
        return lines, used

    def processAlgorithm(self, parameters, context, feedback):
        from mayim_tools.hec_ras._common import h5io

        plan_path = self.parameterAsFile(parameters, self.PLAN_HDF, context)
        if not plan_path or not os.path.exists(plan_path):
            raise QgsProcessingException("Choose the plan results HDF file.")
        project_number = clean_name_part(
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        )
        threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)
        method = METHOD_KEYS[self.parameterAsEnum(parameters, self.METHOD, context)]
        selected = [
            o for o in OUTPUTS if self.parameterAsBoolean(parameters, o[0], context)
        ]
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {o[3] for o in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        want = {o[0] for o in selected}

        h5io.NOTES.clear()
        if h5io.backend() != "h5py":
            feedback.pushInfo(
                f"h5py cannot load inside QGIS ({h5io.H5PY_ERROR}); the HDF5 "
                "file is read with h5py in a separate Python process."
                if h5io.backend() == "h5py-external"
                else "h5py cannot load inside QGIS; the HDF5 file is read "
                "through GDAL's HDF5 driver."
            )
        if h5io.backend() not in ("h5py", "h5py-external"):
            raise QgsProcessingException(
                "The plan results need h5py (in QGIS or in a separate Python "
                "with h5py); GDAL's HDF5 reader cannot read the result tables."
            )

        # plan CRS first: profile lines are transformed into it
        crs = self.parameterAsCrs(parameters, self.CRS, context)
        feedback.setProgressText("Reading the plan results")
        try:
            from .core import RESULT_ROOTS, _attrs, _text, open_h5

            with open_h5(plan_path, roots=RESULT_ROOTS) as f:
                wkt = _text(_attrs(f).get("Projection", b""))
        except Exception:
            wkt = ""
        if not crs.isValid() and wkt:
            crs = QgsCoordinateReferenceSystem.fromWkt(wkt)
        if not crs.isValid():
            feedback.pushWarning(
                "The plan file has no projection and no CRS was given; the "
                "outputs have no CRS."
            )

        lines = []
        if {"OUT_PROFILES", "OUT_CSV"} & want:
            lines, used = self._profile_lines(
                parameters, context, crs, plan_path, feedback
            )
            if used:
                feedback.pushInfo("Profile lines: " + "; ".join(used))
            elif "OUT_PROFILES" in want:
                feedback.pushWarning(
                    "No profile lines found (no layer given, no RAS Mapper "
                    "profile lines and no reference lines): the profile line "
                    "output is skipped."
                )
        try:
            plan = read_plan_results(plan_path, threshold, lines or None)
        except ResultsError as e:
            raise QgsProcessingException(str(e)) from e
        for w in plan.warnings:
            feedback.pushWarning(w)
        label = plan_label(plan)
        feedback.pushInfo(
            f"Plan '{plan.short_id or plan.plan_name}' (flow '{plan.flow_title}'): "
            f"{len(plan.areas)} 2D flow area(s), "
            f"{sum(len(a.centers) for a in plan.areas)} cells, "
            f"{len(plan.times_h)} output times over "
            f"{plan.times_h[-1] - plan.times_h[0]:.2f} h"
        )
        feedback.setProgress(30)

        self._post_processors = []
        outputs = {name: folders[kind] for kind, (name, _) in FOLDER_PARAMS.items()
                   if kind in folders}  # fmt: skip
        written = []

        def path_for(key):
            o = OUTPUT_BY_KEY[key]
            return str(
                Path(folders[o[3]]) / output_name(project_number, label, o[2], o[3])
            )

        def failed(key, path, error):
            feedback.reportError(
                f"{OUTPUT_BY_KEY[key][1]}: {Path(path).name} could not be written "
                f"({error}). If it is open in QGIS, remove it from the project and "
                "run again.",
                fatalError=False,
            )

        def clear(path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    raise RuntimeError("the existing file is in use") from e

        grids = None
        need_grids = any(o[3] == RASTER for o in selected) or "OUT_BOUNDARY" in want
        if need_grids:
            terrain = self.parameterAsFile(parameters, self.TERRAIN, context)
            if not terrain:
                terrain = find_terrain(plan)
            if not terrain or not os.path.exists(terrain):
                raise QgsProcessingException(
                    "The terrain was not found next to the plan "
                    f"({plan.terrain_file or 'no terrain in the geometry'}); choose "
                    "the terrain raster."
                )
            feedback.pushInfo(f"Terrain: {terrain}")
            feedback.setProgressText("Computing the result rasters")
            try:
                grids = build_grids(plan, terrain, method, threshold)
            except ValueError as e:
                raise QgsProcessingException(str(e)) from e
            wet = np.isfinite(grids.depth)
            cell_area = abs(grids.gt[1] * grids.gt[5])
            feedback.pushInfo(
                f"Water surface: {METHODS[method]}; flooded area "
                f"{wet.sum() * cell_area:.0f} m2; maximum depth "
                f"{np.nanmax(grids.depth) if wet.any() else 0:.3f} m"
            )
            projection = crs.toWkt() if crs.isValid() else grids.projection
        feedback.setProgress(60)

        for key, _lab, _d, kind, _df, attr, style, rescale in OUTPUTS:
            if key not in want or kind != RASTER:
                continue
            path = path_for(key)
            data = getattr(grids, attr)
            try:
                clear(path)
                write_grid(path, data, grids.gt, projection,
                           "uint8" if attr == "hazard" else "float32")  # fmt: skip
            except Exception as e:
                failed(key, path, e)
                continue
            rng = None
            if rescale:
                ok = np.isfinite(data)
                if ok.any():
                    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
                    rng = (0.0 if attr != "wse" else lo, hi if hi > lo else lo + 1)
            load_on_completion(context, path, Path(path).stem,
                               _Styler(style, rng), self._post_processors)  # fmt: skip
            outputs[key] = path
            written.append(path)
        feedback.setProgress(75)

        if "OUT_BOUNDARY" in want:
            path = path_for("OUT_BOUNDARY")
            try:
                layer = self._boundary_layer(grids, crs, plan)
                path = save_shapefile(layer, path, context.transformContext())
                style = (
                    "DRN_inundation_2pct.qml" if "2pct" in label.lower()
                    else "DRN_inundation_1pct.qml"
                )  # fmt: skip
                load_on_completion(context, path, Path(path).stem, _Styler(style),
                                   self._post_processors)  # fmt: skip
                outputs["OUT_BOUNDARY"] = path
                written.append(path)
            except Exception as e:
                failed("OUT_BOUNDARY", path, e)

        if "OUT_PROFILES" in want and lines:
            path = path_for("OUT_PROFILES")
            try:
                layer = self._profile_layer(plan, lines, crs)
                path = save_shapefile(layer, path, context.transformContext())
                load_on_completion(
                    context, path, Path(path).stem,
                    _Styler("RAS_profile_lines.qml",
                            aliases={n: d for n, _, d in PROFILE_FIELDS}),
                    self._post_processors,
                )  # fmt: skip
                outputs["OUT_PROFILES"] = path
                written.append(path)
            except Exception as e:
                failed("OUT_PROFILES", path, e)
        for area in plan.areas:
            for i, r in sorted(area.profiles.items()):
                if r["note"]:
                    feedback.pushWarning(f"Profile line '{lines[i].name}': {r['note']}")
                else:
                    feedback.pushInfo(
                        f"Profile line '{lines[i].name}': peak flow "
                        f"{r['q_peak']:.3f} m3/s at {r['t_q_peak_h']:.2f} h, "
                        f"highest water level {r['wse_max']:.2f} m"
                    )
        if lines:
            hit = {i for a in plan.areas for i in a.profiles}
            for i, line in enumerate(lines):
                if i not in hit:
                    feedback.pushWarning(
                        f"Profile line '{line.name}' does not cross any wet cell."
                    )

        if "OUT_CSV" in want:
            path = path_for("OUT_CSV")
            try:
                write_peaks_csv(
                    path, plan, lines,
                    {"tool": f"{self.displayName()} {TOOL_VERSION}",
                     "software": _software_version(),
                     "run date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "water surface": METHODS[method],
                     "depth threshold (m)": f"{threshold:g}"},
                )  # fmt: skip
                outputs["OUT_CSV"] = path
                written.append(path)
            except Exception as e:
                failed("OUT_CSV", path, e)
        for s in plan.structures:
            if "flow" in s["variable"].lower():
                feedback.pushInfo(
                    f"{s['name']}: peak {s['variable']} {s['peak']:.3f} {s['unit']} "
                    f"at {s['time_h']:.2f} h"
                )
        for note in h5io.NOTES:
            if "incomplete" in note or "failed" in note:
                feedback.pushWarning(note)
        h5io.clear_cache()
        if not written:
            raise QgsProcessingException("No output could be written.")
        for path in written:
            feedback.pushInfo(f"Written: {path}")
        feedback.setProgress(100)
        return outputs

    # ------------------------------------------------------------------

    @staticmethod
    def _boundary_layer(grids, crs, plan):
        wet = np.isfinite(grids.depth)
        rows = []
        for i, poly in enumerate(group_polygons(trace_rings(wet)), start=1):
            rings = [
                [QgsPointXY(x, y) for x, y in ring_to_map(simplify_ring(r), grids.gt)]
                for r in poly
            ]
            geom = QgsGeometry.fromPolygonXY(rings)
            if not geom.isGeosValid():
                geom = geom.makeValid()
            geom.convertToMultiType()
            rows.append(
                (geom, {"id": i, "area_m2": round(geom.area(), 2),
                        "plan": plan.short_id or plan.plan_name})
            )  # fmt: skip
        return memory_layer(
            "Inundation boundary", BOUNDARY_FIELDS, Qgis.WkbType.MultiPolygon, crs, rows
        )

    @staticmethod
    def _profile_layer(plan, lines, crs):
        rows = []
        by_line = {}
        for row, (_area, i) in zip(
            profile_rows(plan, lines),
            [(a, i) for a in plan.areas for i in sorted(a.profiles)],
            strict=True,
        ):
            by_line.setdefault(i, []).append(row)
        for i, line in enumerate(lines):
            geom = QgsGeometry.fromPolylineXY(
                [QgsPointXY(x, y) for x, y in line.coords]
            )
            missing = {"name": line.name, "note": "does not cross any wet cell"}
            for row in by_line.get(i, [missing]):
                values = {}
                for field, key in _PROFILE_MAP.items():
                    v = row.get(key)
                    if field in ("Name", "Area_2D", "Note"):
                        values[field] = (str(v) if v else "")[:254]
                    elif field == "Faces":
                        values[field] = int(v) if v is not None else None
                    else:
                        values[field] = _num(v)
                rows.append((QgsGeometry(geom), values))
        return memory_layer(
            "Profile lines", PROFILE_FIELDS, Qgis.WkbType.LineString, crs, rows
        )
