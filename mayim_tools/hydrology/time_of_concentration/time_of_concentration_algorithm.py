"""Processing Toolbox algorithm: Time of Concentration.

Catchment polygons (Catchment Delineation output) + their longest flow paths
+ the DEM in -> Kirpich, USBR (DM Eq 3.12) and Bransby-Williams Tc, the
combined overland + shallow concentrated + channel Tc and the selected Tc
per catchment.

The parameters are fields of the catchment layer, which is updated in
place: run, fill in the parameters (the attribute form has drop-down lists
for r, the surface and the method), run again. The CSV and Word report
follow the Mayim output standard (tick boxes, CSV and Word folders, files
named <project number>_<description>).

Thin wrapper - the logic is in core.py (no QGIS; see
tests/test_time_of_concentration_core.py).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsEditorWidgetSetup,
    QgsField,
    QgsFieldConstraints,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorLayer,
    QgsVectorDataProvider,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    WORD,
    OutputSpec,
    clean_name_part,
)
from mayim_tools.hydrology._common.depression import (
    polygon_window,
    raster_info,
    read_window,
)

from . import tables as tb
from .core import (
    TOOL_VERSION,
    CatchmentTc,
    Params,
    Settings,
    calculate,
    fill_parameters,
    part_bounds,
    profile_from_line,
    sample_cells,
)
from .export import PARAM_FIELDS, RESULT_FIELDS, rounded, status_text, write_csv

OUTPUT_SPECS = [
    OutputSpec("OUTPUT_CSV", "Tc calculations (CSV)", "Time of Concentration", CSV),
    OutputSpec("OUTPUT_DOCX", "Tc report (Word)", "Time of Concentration Report", WORD),
]  # fmt: skip
SPEC_BY_KEY = {s.key: s for s in OUTPUT_SPECS}
FOLDER_PARAMS = {
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}
METHOD_KEYS = list(tb.TC_METHODS)[:4]  # run default: not "user"
CLASS_KEYS = list(tb.LAND_COVER_CLASSIFICATIONS)
TYPES = {"D": QMetaType.Type.Double, "S": QMetaType.Type.QString,
         "I": QMetaType.Type.Int}  # fmt: skip
ALIASES = {
    "ov_len_m": "Overland flow length (m)",
    "ov_r": "Kerby r (overland)",
    "sc_type": "Shallow concentrated flow surface",
    "ch_start_m": "Channel start from the outlet (m)",
    "tc_method": "Tc method for the selected Tc",
    "tc_user_mn": "User Tc (min)",
}


def _advanced(param):
    param.setFlags(param.flags() | Qgis.ProcessingParameterFlag.Advanced)
    return param


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
    parts.append(f"QGIS {Qgis.version()}")
    return ", ".join(parts)


def _num(value):
    if value is None:
        return None
    try:
        text = str(value).strip().replace(",", ".")
        if text in ("", "NULL", "None"):
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _text(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text in ("", "NULL", "None") else text


def apply_form(layer) -> None:
    """Attribute form for entering the parameters: drop-down lists for r,
    the surface and the method, aliases, required-field warnings and the
    result fields read-only."""
    fields = layer.fields()

    def value_map(name, pairs):
        i = fields.indexOf(name)
        if i >= 0:
            layer.setEditorWidgetSetup(
                i, QgsEditorWidgetSetup("ValueMap", {"map": [{d: v} for d, v in pairs]})
            )

    value_map("ov_r", [(f"{r:g} - {text}", str(r)) for r, text in tb.KERBY_R.items()])
    value_map("sc_type", [(f"{text} (k {k:.2f} m/s)", code)
                          for code, (text, k) in tb.SC_TYPES.items()])  # fmt: skip
    value_map("tc_method", [(text, code) for code, text in tb.TC_METHODS.items()])
    config = layer.editFormConfig()
    for name, alias in ALIASES.items():
        i = fields.indexOf(name)
        if i >= 0:
            layer.setFieldAlias(i, alias)
    for name in ("ov_len_m", "ov_r", "sc_type"):
        i = fields.indexOf(name)
        if i >= 0:
            layer.setFieldConstraint(
                i,
                QgsFieldConstraints.Constraint.ConstraintNotNull,
                QgsFieldConstraints.ConstraintStrength.ConstraintStrengthSoft,
            )
    params = {f[0] for f in PARAM_FIELDS}
    for i in range(fields.count()):
        if fields.at(i).name() not in params:
            config.setReadOnly(i, True)
    layer.setEditFormConfig(config)


def save_layer_style(layer) -> bool:
    """Saves the style (with the form) as the layer's default .qml, so the
    form comes back when the shapefile is opened again."""
    if layer.providerType() != "ogr":
        return False
    try:
        result = layer.saveDefaultStyle(layer.StyleCategory.AllStyleCategories)
    except TypeError:
        result = layer.saveDefaultStyle()
    return bool(result[1]) if isinstance(result, tuple) else bool(result)


class TimeOfConcentrationAlgorithm(QgsProcessingAlgorithm):
    CATCHMENTS = "CATCHMENTS"
    ID_FIELD = "ID_FIELD"
    AREA_FIELD = "AREA_FIELD"
    LFP_FIELD = "LFP_FIELD"
    SAVG_FIELD = "SAVG_FIELD"
    S1085_FIELD = "S1085_FIELD"
    SEA_FIELD = "SEA_FIELD"
    CH_FIELD = "CH_FIELD"
    FLOWPATHS = "FLOWPATHS"
    FP_ID_FIELD = "FP_ID_FIELD"
    DEM = "DEM"
    LAND_COVER = "LAND_COVER"
    CLASSIFICATION = "CLASSIFICATION"
    FILL_LC = "FILL_LC"
    DEFAULT_OV = "DEFAULT_OV"
    DEFAULT_METHOD = "DEFAULT_METHOD"
    FIGURES = "FIGURES"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def flags(self):
        # edits the catchment layer in place, so it runs on the main thread
        return super().flags() | Qgis.ProcessingAlgorithmFlag.NoThreading

    def createInstance(self):
        return TimeOfConcentrationAlgorithm()

    def name(self):
        return "time_of_concentration"

    def displayName(self):
        return "Time of Concentration"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        files = "\n".join(
            f"  {s.label}: <project number>_{clean_name_part(s.description)}"
            f"{EXTENSIONS[s.kind]}"
            for s in OUTPUT_SPECS
        )
        r_opts = "\n".join(f"  {r:g}: {t}" for r, t in tb.KERBY_R.items())
        sc_opts = "\n".join(
            f"  {c}: {t} (k = {k:.2f} m/s)" for c, (t, k) in tb.SC_TYPES.items()
        )
        return (
            f"Time of Concentration (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tTc per catchment by Kirpich, USBR (SANRAL DM Eq 3.12) and "
            "Bransby-Williams on the whole longest flow path, and by a combined "
            "method: overland flow (Kerby, DM Eq 3.11) + shallow concentrated "
            "flow (NRCS, V = k S^0.5) + channel flow (DM Eq 3.12), with the "
            "parameters set per catchment. tc_sel_mn is the Tc chosen per "
            "catchment (tc_method), the default Tc field of SANRAL Deterministic "
            "Peak Flows, Prepare HEC-HMS project and Prepare SWMM project.\n"
            "\n"
            "INPUTS:\tThe Catchment Delineation catchments (lfp_km, s_avg, "
            "s_1085, s_ea, ch_head_m) and longest flow paths, and the DEM they "
            "were delineated from. Optional land cover (SANLC or ESA "
            "WorldCover) for suggested r and surface.\n"
            "\n"
            "WORKFLOW:\tThe catchment layer is updated in place (save or cancel "
            "its edits first). Run 1 adds the parameter and result fields, "
            "fills ch_start_m from the measured channel start and an empty "
            "ov_len_m with the default length, and calculates the empirical Tc. "
            "Then fill in ov_r and sc_type per catchment - open the attribute "
            "table in form view to step through the catchments, or select "
            "several and edit them together - and run again. Catchments still "
            "missing parameters are listed (tc_status) and get no combined Tc.\n"
            "\n"
            "PARAMETERS:\tov_len_m overland length from the divide (cut at the "
            "channel start; Kerby's data went to 366 m, NRCS sheet flow seldom "
            "exceeds about 30 m); ov_r Kerby r; sc_type shallow-flow surface; "
            "ch_start_m channel start (m from the outlet); tc_method combined, "
            "usbr, kirpich, bransby_williams or user (with tc_user_mn).\n"
            "Kerby r:\n" + r_opts + "\nShallow-flow surfaces:\n" + sc_opts + "\n"
            "\n"
            "OUTPUTS:\tThe updated catchment layer (with an attribute form, "
            "saved as its default style), and the ticked files:\n" + files + "\n"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.CATCHMENTS,
                "Catchments (updated in place)",
                [QgsProcessing.SourceType.TypeVectorPolygon],
            )
        )
        num = Qgis.ProcessingFieldParameterDataType.Numeric
        anyt = Qgis.ProcessingFieldParameterDataType.Any
        for key, label, default, kind, optional in (
            (self.ID_FIELD, "Catchment ID field", "outlet_id", anyt, False),
            (self.AREA_FIELD, "Area field (km²)", "area_km2", num, False),
            (self.LFP_FIELD, "Longest flow path length field (km)", "lfp_km", num,
             False),
            (self.SAVG_FIELD, "Average slope field (m/m)", "s_avg", num, True),
            (self.S1085_FIELD, "10-85 slope field (m/m)", "s_1085", num, True),
            (self.SEA_FIELD, "Equal-area slope field (m/m)", "s_ea", num, True),
            (self.CH_FIELD, "Measured channel start field (m from the outlet)",
             "ch_head_m", num, True),
        ):  # fmt: skip
            self.addParameter(
                QgsProcessingParameterField(
                    key, label, default, self.CATCHMENTS, kind, optional=optional
                )
            )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.FLOWPATHS,
                "Longest flow paths (Catchment Delineation)",
                [QgsProcessing.SourceType.TypeVectorLine],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FP_ID_FIELD, "Flow path catchment ID field", "outlet_id",
                self.FLOWPATHS,
            )
        )  # fmt: skip
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DEM, "DEM (the one the catchments were delineated from)"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.LAND_COVER, "Land cover (optional, for suggestions)",
                optional=True,
            )
        )  # fmt: skip
        self.addParameter(
            QgsProcessingParameterEnum(
                self.CLASSIFICATION,
                "Land-cover classification",
                [v[0] for v in tb.LAND_COVER_CLASSIFICATIONS.values()],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.FILL_LC,
                "Fill empty r and surface from the land-cover suggestion",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DEFAULT_OV,
                "Overland length for empty ov_len_m (m; 0 = leave empty)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=100.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DEFAULT_METHOD,
                "Method for the selected Tc where tc_method is empty",
                [tb.TC_METHODS[k] for k in METHOD_KEYS],
                defaultValue=0,
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.FIGURES, "Profile figures in the report", defaultValue=True
                )
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix, optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project name (report title page, optional)",
                optional=True,
            )
        )
        for spec in OUTPUT_SPECS:
            self.addParameter(
                QgsProcessingParameterBoolean(
                    spec.key, f"Output: {spec.label}", defaultValue=spec.default
                )
            )
        for name, label in FOLDER_PARAMS.values():
            self.addParameter(
                QgsProcessingParameterFolderDestination(name, label, optional=True)
            )

    # ------------------------------------------------------------------
    def _field(self, parameters, key, context, fields):
        name = self.parameterAsString(parameters, key, context)
        return name if name and fields.indexOf(name) >= 0 else None

    def _ensure_fields(self, layer):
        provider = layer.dataProvider()
        caps = provider.capabilities()
        need = QgsVectorDataProvider.Capability
        if not (caps & need.AddAttributes and caps & need.ChangeAttributeValues):
            raise QgsProcessingException(
                "The catchment layer cannot be edited (add fields / change "
                "values). Save it as a shapefile or GeoPackage first."
            )
        names = {f.name() for f in layer.fields()}
        new = []
        for name, kind, _ in PARAM_FIELDS + RESULT_FIELDS:
            if name in names:
                continue
            length = 254 if name == "tc_status" else (40 if kind == "S" else 20)
            prec = 0 if kind in ("S", "I") else 6
            new.append(QgsField(name, TYPES[kind], len=length, prec=prec))
        if new:
            if not provider.addAttributes(new):
                raise QgsProcessingException("Could not add fields to the layer.")
            layer.updateFields()
        return [f.name() for f in new]

    def _read_paths(self, layer, id_field, dem_crs, context):
        """{catchment id: [(x, y), ...]} in the DEM CRS, divide -> outlet."""
        tr = None
        if layer.crs() != dem_crs:
            tr = QgsCoordinateTransform(
                layer.crs(), dem_crs, context.transformContext()
            )
        out = {}
        for f in layer.getFeatures():
            geom = f.geometry()
            if geom is None or geom.isEmpty():
                continue
            if tr is not None:
                geom.transform(tr)
            lines = (
                geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
            )
            pts = [(p.x(), p.y()) for line in lines for p in line]
            if len(pts) >= 2:
                out[str(f[id_field])] = pts
        return out

    @staticmethod
    def _sample_raster(path, pts):
        shape, gt, _, nodata = raster_info(path)
        window = polygon_window([[pts]], gt, shape, pad=2)
        if window is None:
            return np.full(len(pts), np.nan)
        array = read_window(path, window)
        gt_w = list(gt)
        gt_w[0] = gt[0] + window[1] * gt[1]
        gt_w[3] = gt[3] + window[0] * gt[5]
        xy = np.asarray(pts, dtype=float)
        return sample_cells(array, gt_w, nodata, xy[:, 0], xy[:, 1])

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        layer = self.parameterAsVectorLayer(parameters, self.CATCHMENTS, context)
        if layer is None:
            raise QgsProcessingException("No catchment layer.")
        if layer.isEditable():
            raise QgsProcessingException(
                "The catchment layer is in edit mode: save or cancel its edits "
                "first (the tool writes to it)."
            )
        project_number = (
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        ).strip()
        project_name = (
            self.parameterAsString(parameters, self.PROJECT_NAME, context) or ""
        ).strip()
        selected = [
            s
            for s in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, s.key, context)
        ]
        folders = {}
        for kind in {s.kind for s in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(
                    f"Choose the {label[0].lower() + label[1:]}."
                )
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {s.key: s.path(folders, project_number) for s in selected}

        fields = layer.fields()
        f = {
            k: self._field(parameters, k, context, fields)
            for k in (self.ID_FIELD, self.AREA_FIELD, self.LFP_FIELD, self.SAVG_FIELD,
                      self.S1085_FIELD, self.SEA_FIELD, self.CH_FIELD)
        }  # fmt: skip
        for key, what in ((self.ID_FIELD, "ID"), (self.AREA_FIELD, "area"),
                          (self.LFP_FIELD, "flow path length")):  # fmt: skip
            if f[key] is None:
                raise QgsProcessingException(
                    f"The {what} field was not found on the catchment layer."
                )
        dem = self.parameterAsRasterLayer(parameters, self.DEM, context)
        if dem is None:
            raise QgsProcessingException("A DEM is required.")
        if dem.crs().isGeographic():
            raise QgsProcessingException(
                "The DEM is in a geographic CRS: lengths need a projected CRS "
                "in metres."
            )
        fp_layer = self.parameterAsVectorLayer(parameters, self.FLOWPATHS, context)
        fp_id = self.parameterAsString(parameters, self.FP_ID_FIELD, context)
        if fp_layer is None or fp_layer.fields().indexOf(fp_id) < 0:
            raise QgsProcessingException(
                "The flow path layer or its ID field was not found."
            )
        lc = self.parameterAsRasterLayer(parameters, self.LAND_COVER, context)
        settings = Settings(
            default_ov_len_m=self.parameterAsDouble(
                parameters, self.DEFAULT_OV, context),
            fill_from_land_cover=self.parameterAsBoolean(
                parameters, self.FILL_LC, context),
            default_method=METHOD_KEYS[
                self.parameterAsEnum(parameters, self.DEFAULT_METHOD, context)
            ],
            classification=CLASS_KEYS[
                self.parameterAsEnum(parameters, self.CLASSIFICATION, context)
            ],
        )  # fmt: skip
        if settings.fill_from_land_cover and lc is None:
            feedback.pushWarning(
                "No land cover given: r and the surface are not filled."
            )

        added = self._ensure_fields(layer)
        if added:
            feedback.pushInfo(f"Fields added to the layer: {', '.join(added)}")
        fields = layer.fields()
        fpaths = self._read_paths(fp_layer, fp_id, dem.crs(), context)
        dem_path = dem.source()
        lc_tr = None
        if lc is not None and lc.crs() != dem.crs():
            lc_tr = QgsCoordinateTransform(
                dem.crs(), lc.crs(), context.transformContext()
            )

        items, fids, fill_notes = [], [], {}
        feats = list(layer.getFeatures())
        n = len(feats)
        for i, feat in enumerate(feats):
            if feedback.isCanceled():
                break
            feedback.setProgress(80.0 * i / max(n, 1))

            def get(key, _feat=feat):
                return None if f[key] is None else _num(_feat[f[key]])

            cid = str(feat[f[self.ID_FIELD]])
            params = Params(
                ov_len_m=_num(feat["ov_len_m"]),
                ov_r=_num(feat["ov_r"]),
                sc_type=_text(feat["sc_type"]),
                ch_start_m=_num(feat["ch_start_m"]),
                tc_method=_text(feat["tc_method"]),
                tc_user_mn=_num(feat["tc_user_mn"]),
            )
            c = CatchmentTc(
                cid, get(self.AREA_FIELD), get(self.LFP_FIELD),
                s_avg=get(self.SAVG_FIELD), s_1085=get(self.S1085_FIELD),
                s_ea=get(self.SEA_FIELD), ch_head_m=get(self.CH_FIELD),
                params=params,
            )  # fmt: skip
            pts = fpaths.get(cid)
            vx = None
            if pts is None:
                c.flags.append("no longest flow path with this ID")
            else:
                try:
                    zs = self._sample_raster(dem_path, pts)
                except Exception as e:
                    raise QgsProcessingException(f"Could not read the DEM: {e}") from e
                c.x, c.z = profile_from_line(pts, zs)
                xy = np.asarray(pts, dtype=float)
                seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                vx = cum[-1] - cum  # distance from the outlet per vertex
                if len(c.x) < len(pts):
                    c.flags.append(
                        f"{len(pts) - len(c.x)} flow path vertices have no DEM value"
                    )
            # first the lengths, then the land cover along the parts
            notes = fill_parameters(
                c, Settings(default_ov_len_m=settings.default_ov_len_m)
            )
            if lc is not None and vx is not None:
                bounds = part_bounds(c)
                for part, attr in (
                    ("overland", "lc_overland"),
                    ("shallow", "lc_shallow"),
                ):
                    a, b = bounds[part]
                    sel = [p for p, xv in zip(pts, vx, strict=True) if a <= xv <= b]
                    if not sel or b <= a:
                        continue
                    if lc_tr is not None:
                        sel = [
                            (q.x(), q.y())
                            for q in (lc_tr.transform(QgsPointXY(*p)) for p in sel)
                        ]
                    vals = self._sample_raster(lc.source(), sel)
                    setattr(c, attr, [int(v) for v in vals if np.isfinite(v)])
            notes += fill_parameters(c, settings)
            calculate(c, settings)
            if notes:
                fill_notes[cid] = notes
            items.append(c)
            fids.append(feat.id())

        # write back to the layer
        changes = {}
        param_names = [p[0] for p in PARAM_FIELDS]
        for fid, c in zip(fids, items, strict=True):
            attrs = {}
            for name in param_names:
                value = getattr(c.params, name)
                i = fields.indexOf(name)
                if i >= 0:
                    attrs[i] = value
            for name, _, _ in RESULT_FIELDS:
                i = fields.indexOf(name)
                if i < 0:
                    continue
                attrs[i] = (
                    (status_text(c)[:254] or None)
                    if name == "tc_status"
                    else (rounded(name, c.values.get(name)))
                )
            changes[fid] = attrs
        if not layer.dataProvider().changeAttributeValues(changes):
            raise QgsProcessingException(
                "Could not write the values to the catchment layer. If it is "
                "open in another program, close it and run again."
            )
        layer.reload()
        apply_form(layer)
        if save_layer_style(layer):
            feedback.pushInfo("Attribute form saved as the layer's default style.")
        layer.triggerRepaint()

        for cid, notes in fill_notes.items():
            feedback.pushInfo(f"{cid}: " + "; ".join(notes))
        todo = [c for c in items if c.missing]
        if todo:
            feedback.pushWarning(
                f"{len(todo)} catchment(s) still need parameters for the combined "
                "Tc: " + "; ".join(f"{c.catchment_id} ({', '.join(c.missing)})"
                                   for c in todo[:20])
                + (" ..." if len(todo) > 20 else "")
                + ". Fill them in the attribute form and run again."
            )  # fmt: skip

        run_rows = [
            ("Tool", f"{self.displayName()} {TOOL_VERSION}"),
            ("Run time", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Catchment layer", layer.source()),
            ("Longest flow paths", fp_layer.source()),
            ("DEM", dem_path),
            ("Land cover", lc.source() if lc is not None else ""),
            ("Land-cover classification",
             tb.LAND_COVER_CLASSIFICATIONS[settings.classification][0]
             if lc is not None else ""),
            ("Fill empty r and surface from the land cover",
             "yes" if settings.fill_from_land_cover else "no"),
            ("Overland length for empty ov_len_m (m)", settings.default_ov_len_m),
            ("Method where tc_method is empty", settings.default_method),
        ]  # fmt: skip
        outputs = {self.CATCHMENTS: layer.id()}
        written = [("Catchments (parameters and Tc fields)", layer.source())]

        def save(key, writer):
            path = paths[key]
            try:
                writer(path)
            except ImportError as e:
                feedback.pushWarning(
                    f"{SPEC_BY_KEY[key].label}: not written - needs python-docx "
                    f"and matplotlib in QGIS's Python ({e})."
                )
                return
            except Exception as e:
                feedback.reportError(
                    f"{SPEC_BY_KEY[key].label}: {Path(path).name} could not be "
                    f"written ({e}). If it is open in Excel or Word, close it and "
                    "run again.",
                    fatalError=False,
                )
                return
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))
            feedback.pushInfo(f"Written: {path}")

        if "OUTPUT_CSV" in paths:
            save(
                "OUTPUT_CSV",
                lambda p: write_csv(
                    p, items, run_rows, settings.classification if lc else None
                ),
            )
        if "OUTPUT_DOCX" in paths:
            feedback.setProgressText("Writing the Word report")
            docx_path = paths["OUTPUT_DOCX"]

            def write_docx(p):
                from .report import write_report

                for note in write_report(
                    p,
                    items,
                    {"tool": self.displayName(), "version": TOOL_VERSION,
                     "rows": run_rows},
                    context={
                        "project_number": project_number,
                        "project_name": project_name,
                        "software": _software_version(),
                        "outputs": written + [("This report", docx_path)],
                        "settings": settings,
                        "figures": self.parameterAsBoolean(
                            parameters, self.FIGURES, context
                        ),
                    },
                ):  # fmt: skip
                    feedback.pushInfo(note)

            save("OUTPUT_DOCX", write_docx)

        for c in items:
            v = c.values
            feedback.pushInfo(
                f"{c.catchment_id}: USBR {rounded('x', v.get('tc_usbr_mn'))}, "
                f"combined {rounded('x', v.get('tc_comb_mn'))}, selected "
                f"{rounded('x', v.get('tc_sel_mn'))} min ({v.get('tc_sel_by')})"
            )
        return outputs
