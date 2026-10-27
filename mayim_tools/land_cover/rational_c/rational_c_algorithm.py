"""Processing Toolbox algorithm: Run-off Coefficients (SANRAL Deterministic).

Catchment polygons + DEM + SANLC land cover + hydrologic soil groups + MAP
-> the run-off coefficient C of the Rational method per catchment, after
the SANRAL Drainage Manual (2013) Table 3.7 and Table 3.8: a catchment
layer, the C CSV read by SANRAL Deterministic Peak Flows, a sectioned
results CSV, class rasters and a Word report. Thin wrapper - the logic is in
core.py, cells.py and coefficients.py (tests/test_rational_c_core.py).
"""

import os
import tempfile
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMemoryProviderUtils,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import CSV, RASTER, SHAPEFILE, WORD, OutputSpec
from mayim_tools.hydrology.catchment_delineation.qgis_outputs import (
    STYLE_CATCHMENTS,
    load_on_completion,
    save_shapefile,
)

from . import cells as cl
from . import core
from . import export as ex
from . import lookup as lk

POSITION_OPTIONS = [
    ("upper", "Upper end (as in the Application Guide example)"),
    ("middle", "Middle of the range"),
    ("lower", "Lower end"),
]
VEG_KEYS = {k: i for i, k in enumerate(lk.VEG_KEYS)}

OUTPUT_SPECS = [
    OutputSpec("OUTPUT_LAYER", "Catchments with run-off coefficients", "Runoff Coefficients", SHAPEFILE),
    OutputSpec("OUTPUT_C_CSV", "C CSV for SANRAL Deterministic Peak Flows", "C Values", CSV),
    OutputSpec("OUTPUT_RESULTS", "Results (sectioned CSV)", "Runoff Coefficient Results", CSV),
    OutputSpec("OUTPUT_LOOKUP", "Land-cover lookup used (editable CSV)", "C Lookup", CSV, False),
    OutputSpec("OUTPUT_ROLE", "Raster: land-cover role", "C Land Cover Role", RASTER),
    OutputSpec("OUTPUT_SLOPE", "Raster: slope class (Cs)", "C Slope Class", RASTER, False),
    OutputSpec("OUTPUT_PERM", "Raster: permeability class (Cp)", "C Permeability Class", RASTER, False),
    OutputSpec("OUTPUT_VEG", "Raster: vegetation class (Cv)", "C Vegetation Class", RASTER, False),
    OutputSpec("OUTPUT_C1", "Raster: C1 rural", "C1 Rural", RASTER, False),
    OutputSpec("OUTPUT_C2", "Raster: C2 urban", "C2 Urban", RASTER, False),
    OutputSpec("OUTPUT_C50", "Raster: C 1:50 year", "C 50 Year", RASTER),
    OutputSpec("OUTPUT_DOCX", "Report (Word)", "Runoff Coefficient Report", WORD),
]  # fmt: skip
SPEC_BY_KEY = {s.key: s for s in OUTPUT_SPECS}
RASTER_KEYS = {
    "OUTPUT_ROLE": "role",
    "OUTPUT_SLOPE": "slope_class",
    "OUTPUT_PERM": "perm_class",
    "OUTPUT_VEG": "veg_class",
    "OUTPUT_C1": "c1",
    "OUTPUT_C2": "c2",
    "OUTPUT_C50": "c50",
}
MAP_KEYS = ("role", "slope_class", "perm_class", "veg_class", "c1", "c50")
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}
STRING_FIELDS = {"catch_id", "ft_class", "ft_src", "MAP_band", "flags"}
INT_FIELDS = {"n_flags"}
STRING_LENGTHS = {"catch_id": 80, "ft_class": 20, "ft_src": 10, "MAP_band": 12}


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


def _advanced(param):
    param.setFlags(param.flags() | Qgis.ProcessingParameterFlag.Advanced)
    return param


class RationalCAlgorithm(QgsProcessingAlgorithm):
    CATCHMENTS = "CATCHMENTS"
    ID_FIELD = "ID_FIELD"
    MAP = "MAP"
    MAP_FIELD = "MAP_FIELD"
    FT_FIELD = "FT_FIELD"
    DEM = "DEM"
    LANDCOVER = "LANDCOVER"
    HSG = "HSG"
    TEXTURE = "TEXTURE"
    DOLOMITE = "DOLOMITE"
    OVERRIDES = "OVERRIDES"
    CELL = "CELL"
    POSITION = "POSITION"
    USE_PROB = "USE_PROB"
    WETLANDS_IMP = "WETLANDS_IMP"
    C3 = "C3"
    FT_ON_C2 = "FT_ON_C2"
    LOOKUP_CSV = "LOOKUP_CSV"
    NATIVE_SLOPE = "NATIVE_SLOPE"
    REPORT_DETAIL = "REPORT_DETAIL"
    REPORT_BASEMAP = "REPORT_BASEMAP"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"
    # output tick boxes (same keys as the result dictionary)
    OUTPUT_LAYER = "OUTPUT_LAYER"
    OUTPUT_C_CSV = "OUTPUT_C_CSV"
    OUTPUT_RESULTS = "OUTPUT_RESULTS"
    OUTPUT_LOOKUP = "OUTPUT_LOOKUP"
    OUTPUT_DOCX = "OUTPUT_DOCX"

    def __init__(self):
        super().__init__()
        self._post_processors = []

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return RationalCAlgorithm()

    def name(self):
        return "rational_c"

    def displayName(self):
        return "Run-off Coefficients (SANRAL Deterministic)"

    def group(self):
        return "Land Cover Tools"

    def groupId(self):
        return "land_cover_tools"

    def shortHelpString(self):
        fields = "\n".join(f"  {k}: {v}" for k, v in ex.FIELDS.items())
        return (
            f"Run-off Coefficients (SANRAL Deterministic) (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tWorks out the run-off coefficient C of the Rational method "
            "for each catchment with the SANRAL Drainage Manual (2013), Section "
            "3.5.1.2: C1 = Cs + Cp + Cv from the area shares of the slope, "
            "permeability and vegetation classes of Table 3.7 (as on the manual's "
            "calculation form), the dolomite reduction, the Ft class of Table 3.8, "
            "the urban C2 and the lakes C3. Writes the C CSV that SANRAL "
            "Deterministic Peak Flows reads.\n"
            "\n"
            "INPUTS:\tCatchments: polygons in a projected CRS (e.g. the Catchment "
            "Delineation catchments, ID field outlet_id). DEM. Land cover: SANLC "
            "2018 / 2020 / 2022 (73 classes; 0 = no data). Soil groups: "
            "hsg_neh630_recommended.tif (or hsg_scssa_recommended.tif) from "
            "Hydrological Soil Groups (HSG); the matching hsg_*_probability.tif beside it "
            "is found automatically and gives the soil range. MAP (mm), or a MAP "
            "field. Optional: Ft class field (steep_impermeable / flat_permeable), "
            "dolomite polygons, site override polygons, soil texture raster "
            "(Regional Soil Parameterisation, USDA codes 1-12) for a cross-check.\n"
            "\n"
            "METHOD:\tAll rasters go onto a 20 m analysis grid (land cover and "
            "soil by nearest neighbour, DEM averaged). Slope: Horn, %; classes "
            "< 3, 3-10, 10-30, >= 30 %. Permeability: A very permeable, B "
            "permeable, C semi-permeable, D (and A/D, B/D, C/D) impermeable; SCS-SA "
            "A/B, B/C, C/D take the average of the neighbouring classes; natural "
            "rock is impermeable. Vegetation and urban rows from the SANLC lookup. "
            "Ineffective classes (ocean, wet and dry pans, sewage ponds, flooded "
            "mine pits) are left out of the effective area. Ft class: score = half "
            "the hilly + steep share plus half the semi-permeable + impermeable "
            "share; 0.5 or more is steep and impermeable.\n"
            "\n"
            "OVERRIDES:\tPolygon layer with any of the fields 'sanlc' (land-cover "
            "code), 'cv_class' (thick_bush, light_bush, grassland or "
            "no_vegetation), 'urban_row' (a Table 3.7 row key: houses, flats, "
            "light_industry, heavy_industry, city_centre, suburban, streets, "
            "lawn, ...) and 'c2' (a fixed C2).\n"
            "\n"
            "LOOKUP:\tThe SANLC lookup (role, vegetation class with a lower and "
            "upper class, urban row) can be written out as a CSV, edited and read "
            "back as a custom lookup.\n"
            "\n"
            "OUTPUTS:\tSelect with the tick boxes; files are named <project "
            "number>_<description> in the raster, shapefile, CSV and Word folders. "
            "The C CSV has catchment_id, alpha, c1 (= C1D, after dolomite), "
            "ft_class, beta, c2, gamma, c3, followed by the C1 breakdown (class "
            "shares cs1_pct - cv4_pct, cs, cp, cv, c1_base before dolomite, d_pct, "
            "d_factor, map_mm) for the SANRAL calculation sheets.\n"
            "\n"
            f"CATCHMENT LAYER FIELDS:\n{fields}\n"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.CATCHMENTS, "Catchments", [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ID_FIELD,
                "Catchment ID field",
                defaultValue="outlet_id",
                parentLayerParameterName=self.CATCHMENTS,
                optional=True,
            )
        )
        self.addParameter(QgsProcessingParameterRasterLayer(self.DEM, "DEM"))
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.LANDCOVER, "Land cover (SANLC)")
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.HSG, "Hydrological soil groups raster (hsg_*_recommended.tif)"
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAP,
                "Mean annual precipitation MAP (mm)",
                QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.MAP_FIELD,
                "MAP field (mm, per catchment; optional)",
                parentLayerParameterName=self.CATCHMENTS,
                type=QgsProcessingParameterField.Numeric,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FT_FIELD,
                "Ft class override field (steep_impermeable / flat_permeable; optional)",
                parentLayerParameterName=self.CATCHMENTS,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.DOLOMITE,
                "Dolomite polygons (optional)",
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.OVERRIDES,
                "Site override polygons (optional; fields sanlc, cv_class, urban_row, c2)",
                [QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.TEXTURE,
                "Soil texture raster for the cross-check (USDA codes 1-12; optional)",
                optional=True,
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    self.CELL,
                    "Analysis cell (m)",
                    QgsProcessingParameterNumber.Double,
                    defaultValue=core.DEFAULT_CELL_M,
                    minValue=1.0,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterEnum(
                    self.POSITION,
                    "Urban C2 within the Table 3.7 range",
                    options=[label for _, label in POSITION_OPTIONS],
                    defaultValue=0,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.USE_PROB,
                    "Use the soil-group probabilities for the C1 range",
                    defaultValue=True,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.WETLANDS_IMP,
                    "Treat wetlands (SANLC 22, 23, 73) as impermeable",
                    defaultValue=False,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterNumber(
                    self.C3,
                    "C3 for lakes and dams",
                    QgsProcessingParameterNumber.Double,
                    defaultValue=1.0,
                    minValue=0.0,
                    maxValue=1.0,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.FT_ON_C2, "Apply Ft to C2 (Table 3.8)", defaultValue=False
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterFile(
                    self.LOOKUP_CSV,
                    "Custom land-cover lookup CSV (optional)",
                    extension="csv",
                    optional=True,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.NATIVE_SLOPE,
                    "Report the slope classes at the DEM's own cell size",
                    defaultValue=True,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.REPORT_DETAIL,
                    "Report: calculation form for every catchment",
                    defaultValue=False,
                )
            )
        )
        self.addParameter(
            _advanced(
                QgsProcessingParameterBoolean(
                    self.REPORT_BASEMAP,
                    "Report maps: OpenStreetMap background",
                    defaultValue=True,
                )
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix and report, optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NAME,
                "Project / site name (report title page, optional)",
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

    def _polygons(self, source, crs, context, feedback, what):
        """[(feature, wkt in crs)] of a polygon source."""
        tr = None
        if source.sourceCrs() != crs:
            tr = QgsCoordinateTransform(
                source.sourceCrs(), crs, context.transformContext()
            )
        out = []
        for f in source.getFeatures():
            g = QgsGeometry(f.geometry())
            if g.isNull() or g.isEmpty():
                continue
            if tr is not None:
                g.transform(tr)
            if not g.isGeosValid():
                g = g.makeValid()
            out.append((f, g.asWkt()))
        if not out:
            feedback.pushWarning(f"{what}: no polygons.")
        return out

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        source = self.parameterAsSource(parameters, self.CATCHMENTS, context)
        if source is None:
            raise QgsProcessingException("Choose the catchment layer.")
        crs = source.sourceCrs()
        if crs.isGeographic():
            raise QgsProcessingException(
                "The catchments are in a geographic CRS (degrees). Reproject them to "
                "a projected CRS in metres (e.g. the DEM's) first."
            )
        dem = self.parameterAsRasterLayer(parameters, self.DEM, context)
        lc = self.parameterAsRasterLayer(parameters, self.LANDCOVER, context)
        hsg = self.parameterAsRasterLayer(parameters, self.HSG, context)
        tex = self.parameterAsRasterLayer(parameters, self.TEXTURE, context)
        id_field = self.parameterAsString(parameters, self.ID_FIELD, context)
        map_field = self.parameterAsString(parameters, self.MAP_FIELD, context)
        ft_field = self.parameterAsString(parameters, self.FT_FIELD, context)
        map_mm = self.parameterAsDouble(parameters, self.MAP, context)
        project_number = self.parameterAsString(
            parameters, self.PROJECT_NUMBER, context
        )
        project_name = self.parameterAsString(
            parameters, self.PROJECT_NAME, context
        ).strip()
        names = source.fields().names()
        if id_field and id_field not in names:
            feedback.pushWarning(f"ID field '{id_field}' not found: feature ids used.")
            id_field = ""

        selected = [
            s
            for s in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, s.key, context)
        ]
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {s.kind for s in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {s.key: s.path(folders, project_number) for s in selected}
        want = set(paths)

        # catchments
        cats = []
        seen = set()
        for f, wkt in self._polygons(source, crs, context, feedback, "Catchments"):
            cid = (
                str(f[id_field])
                if id_field and f[id_field] not in (None, "")
                else str(f.id())
            )
            if cid in seen:
                feedback.pushWarning(f"Catchment ID '{cid}' occurs more than once.")
                cid = f"{cid}_{f.id()}"
            seen.add(cid)
            m = None
            if map_field:
                try:
                    m = float(f[map_field])
                except (TypeError, ValueError):
                    m = None
                if m is not None and m <= 0:
                    m = None
            ft = None
            if ft_field:
                raw = str(f[ft_field] or "").strip().lower()
                if raw and raw not in ("null", "none"):
                    from mayim_tools.hydrology.sanral_peak_flows.inputs import (
                        FT_CLASS_ALIASES,
                    )

                    ft = FT_CLASS_ALIASES.get(raw)
                    if ft is None:
                        feedback.pushWarning(
                            f"{cid}: Ft class '{raw}' not recognised; the score is used."
                        )
            cats.append(core.Catchment(cid, wkt, m, ft))
        if not cats:
            raise QgsProcessingException("The catchment layer has no polygons.")
        if map_mm <= 0 and any(c.map_mm is None for c in cats):
            raise QgsProcessingException(
                "Give the MAP (mm), or a MAP field with a value for every catchment."
            )

        dolomite = []
        dsrc = self.parameterAsSource(parameters, self.DOLOMITE, context)
        if dsrc is not None:
            dolomite = [
                w for _, w in self._polygons(dsrc, crs, context, feedback, "Dolomite")
            ]
        overrides = []
        osrc = self.parameterAsSource(parameters, self.OVERRIDES, context)
        if osrc is not None:
            onames = {n.lower(): n for n in osrc.fields().names()}
            for f, wkt in self._polygons(osrc, crs, context, feedback, "Overrides"):
                overrides.append(self._override(f, wkt, onames, feedback))

        lookup_csv = self.parameterAsFile(parameters, self.LOOKUP_CSV, context)
        position = POSITION_OPTIONS[
            self.parameterAsEnum(parameters, self.POSITION, context)
        ][0]
        tmp = tempfile.mkdtemp(prefix="mayim_rc_")
        rasters = {}
        for key, rkey in RASTER_KEYS.items():
            if key in want:
                rasters[rkey] = paths[key]
            elif self.OUTPUT_DOCX in want and rkey in MAP_KEYS:
                rasters[rkey] = os.path.join(tmp, f"{rkey}.tif")
        s = core.RunSettings(
            catchments=cats,
            crs_wkt=crs.toWkt(),
            dem_path=dem.source(),
            landcover_path=lc.source(),
            hsg_path=hsg.source(),
            map_mm=map_mm,
            cell_m=self.parameterAsDouble(parameters, self.CELL, context),
            lookup_key="custom" if lookup_csv else "sanlc",
            lookup_csv=lookup_csv or "",
            position=position,
            wetlands_impermeable=self.parameterAsBoolean(
                parameters, self.WETLANDS_IMP, context
            ),
            c3=self.parameterAsDouble(parameters, self.C3, context),
            ft_on_c2=self.parameterAsBoolean(parameters, self.FT_ON_C2, context),
            use_probability=self.parameterAsBoolean(parameters, self.USE_PROB, context),
            texture_path=tex.source() if tex is not None else "",
            dolomite=dolomite,
            overrides=overrides,
            native_slope=self.parameterAsBoolean(
                parameters, self.NATIVE_SLOPE, context
            ),
            rasters=rasters,
        )

        def progress(fraction, message):
            feedback.setProgress(int(100 * fraction))
            feedback.setProgressText(message)

        try:
            r = core.run(s, progress, feedback.pushInfo, feedback.isCanceled)
        except InterruptedError:
            return {}
        except core.RationalCError as e:
            raise QgsProcessingException(str(e)) from e
        except Exception as e:
            raise QgsProcessingException(
                f"Run-off Coefficients (SANRAL Deterministic) failed: {e}"
            ) from e

        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        written = []

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))

        inputs = [
            ("Catchments", f"{source.sourceName()} ({len(cats)} polygons; ID field {id_field or 'feature id'})"),
            ("CRS", f"{crs.authid()} - {crs.description()}"),
            ("DEM", dem.source()),
            ("Land cover", lc.source()),
            ("Soil groups", hsg.source()),
            ("MAP field", map_field or "-"),
            ("Ft override field", ft_field or "-"),
            ("Dolomite layer", dsrc.sourceName() if dsrc is not None else "-"),
            ("Override layer", osrc.sourceName() if osrc is not None else "-"),
            ("Soil texture", tex.source() if tex is not None else "-"),
            ("Custom lookup", lookup_csv or "-"),
        ]  # fmt: skip

        layer = self._layer(r, cats, crs)
        if self.OUTPUT_LAYER in want:
            try:
                path = save_shapefile(
                    layer, paths[self.OUTPUT_LAYER], context.transformContext()
                )
                load_on_completion(
                    context,
                    path,
                    Path(path).stem,
                    STYLE_CATCHMENTS,
                    self._post_processors,
                )
                done(self.OUTPUT_LAYER, path)
            except Exception as e:
                feedback.reportError(str(e), fatalError=False)
        for key, writer in (
            (self.OUTPUT_C_CSV, lambda p: ex.write_c_csv(p, r)),
            (self.OUTPUT_RESULTS, lambda p: ex.write_results_csv(p, r, inputs)),
            (self.OUTPUT_LOOKUP, lambda p: lk.write_lookup_csv(p, r.lookup)),
        ):
            if key in want:
                try:
                    writer(paths[key])
                    done(key, paths[key])
                except OSError as e:
                    feedback.reportError(
                        f"{Path(paths[key]).name} could not be written ({e}); it may be "
                        "open in another program.",
                        fatalError=False,
                    )
        from .qgis_map import RASTER_STYLES, STYLES

        for key, rkey in RASTER_KEYS.items():
            if key in want and os.path.exists(paths[key]):
                qml = STYLES / RASTER_STYLES[rkey]
                self._load_raster(context, paths[key], qml)
                done(key, paths[key])

        if self.OUTPUT_DOCX in want:
            feedback.setProgressText("Writing report")
            messages = []
            map_fn = None
            try:
                from .qgis_map import make_map_fn

                map_fn = make_map_fn(
                    r,
                    layer,
                    crs,
                    context.transformContext(),
                    self.parameterAsBoolean(parameters, self.REPORT_BASEMAP, context),
                    messages,
                )
            except Exception as e:  # noqa: BLE001
                messages.append(f"QGIS maps unavailable ({e}); simple maps drawn.")
            path = paths[self.OUTPUT_DOCX]
            try:
                from .report import write_report

                notes = write_report(
                    path,
                    r,
                    {
                        "project_number": project_number,
                        "project_name": project_name,
                        "software": _software_version(),
                        "inputs": inputs,
                        "outputs": written + [("This report", path)],
                        "warnings": messages,
                        "detail": self.parameterAsBoolean(
                            parameters, self.REPORT_DETAIL, context
                        ),
                    },
                    map_fn,
                )
                for m in dict.fromkeys(notes + messages):
                    feedback.pushWarning(m)
                done(self.OUTPUT_DOCX, path)
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs python-docx and matplotlib in QGIS's Python "
                    "(python -m pip install python-docx). All other outputs were written."
                )
            except Exception as e:  # noqa: BLE001 - the results must not be lost
                feedback.pushWarning(f"The Word report could not be written: {e}")

        for w in r.warnings:
            feedback.pushWarning(w)
        for cr in r.catchments:
            k = cr.coeff
            if k is None:
                feedback.pushWarning(
                    f"{cr.cid}: no coefficients ({'; '.join(cr.flags)})"
                )
                continue
            feedback.pushInfo(
                f"{cr.cid}: A_eff {cr.a_eff_km2:.4f} km2, alpha {k.alpha:.3f}, "
                f"C1D {k.c1d:.3f}, {k.ft_class}, beta {k.beta:.3f}, C2 {k.c2:.3f}, "
                f"gamma {k.gamma:.3f}; C50 {k.c_t[50]:.3f}"
            )
        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        return outputs

    # ------------------------------------------------------------------

    def _override(self, f, wkt, onames, feedback):
        def val(name):
            n = onames.get(name)
            if n is None:
                return None
            v = f[n]
            if v is None or str(v).strip() in ("", "NULL"):
                return None
            return v

        o = core.Override(wkt)
        v = val("sanlc")
        if v is not None:
            o.lc_code = int(float(v))
        v = val("cv_class")
        if v is not None:
            key = str(v).strip().lower()
            if key.isdigit() and 1 <= int(key) <= 4:
                o.cv = int(key) - 1
            elif key in VEG_KEYS:
                o.cv = VEG_KEYS[key]
            else:
                feedback.pushWarning(f"Override {f.id()}: cv_class '{v}' ignored.")
        v = val("urban_row")
        if v is not None:
            key = str(v).strip().lower()
            if key in cl.URBAN_INDEX:
                o.urban = key
            else:
                feedback.pushWarning(f"Override {f.id()}: urban_row '{v}' ignored.")
        v = val("c2")
        if v is not None:
            o.c2 = float(v)
        return o

    def _layer(self, r, cats, crs):
        fields = QgsFields()
        for name in ex.FIELDS:
            t = (
                QMetaType.Type.QString
                if name in STRING_FIELDS
                else QMetaType.Type.Int if name in INT_FIELDS else QMetaType.Type.Double
            )
            field = QgsField(name, t)
            if name in STRING_FIELDS:
                field.setLength(STRING_LENGTHS.get(name, 254))
            fields.append(field)
        layer = QgsMemoryProviderUtils.createMemoryLayer(
            "Runoff coefficients", fields, Qgis.WkbType.MultiPolygon, crs
        )
        feats = []
        for c, cr in zip(cats, r.catchments, strict=True):
            feat = QgsFeature(fields)
            g = QgsGeometry.fromWkt(c.wkt)
            g.convertToMultiType()
            feat.setGeometry(g)
            row = ex.attributes(cr)
            for name in ex.FIELDS:
                v = row.get(name)
                feat[name] = v if v is not None else None
            feats.append(feat)
        layer.dataProvider().addFeatures(feats)
        layer.updateExtents()
        return layer

    def _load_raster(self, context, path, qml):
        name = Path(path).stem
        details = QgsProcessingContext.LayerDetails(name, context.project(), name)
        if qml.is_file():
            proc = AbsoluteStyle(str(qml))
            details.setPostProcessor(proc)
            self._post_processors.append(proc)
        context.addLayerToLoadOnCompletion(path, details)


class AbsoluteStyle(QgsProcessingLayerPostProcessorInterface):
    """Applies a QML (absolute path) once Processing has loaded the layer."""

    def __init__(self, qml: str):
        super().__init__()
        self.qml = qml

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            layer.loadNamedStyle(self.qml)
            layer.triggerRepaint()
        except Exception as e:  # noqa: BLE001 - styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style not applied: {e}")
