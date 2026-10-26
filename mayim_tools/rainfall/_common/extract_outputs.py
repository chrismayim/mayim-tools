"""Shared outputs of the point-extraction tools (Extract: CHIRPS, CMORPH,
IMERG, MERRA2, PERSIANN CDR and ERA5 precipitation), in the Mayim output
standard:

- a tick box per output and one folder per kind (CSV, shapefile);
- files named ``<project number>_<description>`` with underscores;
- the time-series CSV in UTF-8 with a byte-order mark (no '#' key: its
  headings are plain and the rainfall tools read it directly);
- the matched grid cell and query point layers written as shapefiles and
  loaded with the tool's QML style;
- an output that cannot be written (open in Excel or QGIS) is reported and
  the other outputs are still written. Because an extraction can take
  hours, locked files are also flagged before the download starts.
"""

from __future__ import annotations

import os
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMemoryProviderUtils,
    QgsPointXY,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterString,
)
from qgis.PyQt.QtCore import QMetaType

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    SHAPEFILE,
    OutputSpec,
    clean_name_part,
)
from mayim_tools.hydrology.catchment_delineation.qgis_outputs import save_shapefile

CSV_ENCODING = "utf-8-sig"
WGS84 = QgsCoordinateReferenceSystem("EPSG:4326")

PROJECT_NUMBER = "PROJECT_NUMBER"
OUTPUT_CSV = "OUTPUT_CSV"
OUTPUT_MATCHED_POINTS = "OUTPUT_MATCHED_POINTS"
OUTPUT_QUERY_POINTS = "OUTPUT_QUERY_POINTS"

FOLDER_PARAMS = {
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
}

MATCHED_FIELDS_HELP = (
    "site: site name; req_lat / req_lon: requested point; match_lat / "
    "match_lon: centre of the grid cell the data was read from; dist_km: "
    "distance between them (km)"
)


def extract_specs(product: str, extra=(), matched: bool = True) -> list[OutputSpec]:
    """The standard outputs of an extraction tool for ``product`` (e.g.
    'CHIRPS'): the precipitation CSV, any ``extra`` specs, the matched grid
    cell layer (when the product has one) and the query point layer."""
    specs = [
        OutputSpec(
            OUTPUT_CSV,
            "Precipitation time series (CSV)",
            f"{product} Precipitation",
            CSV,
        )
    ]
    specs += list(extra)
    if matched:
        specs.append(
            OutputSpec(
                OUTPUT_MATCHED_POINTS,
                "Matched grid cell location (shapefile)",
                f"{product} Grid Cell",
                SHAPEFILE,
            )
        )
    specs.append(
        OutputSpec(
            OUTPUT_QUERY_POINTS,
            "Points of interest (shapefile)",
            f"{product} POI",
            SHAPEFILE,
        )
    )
    return specs


def files_help(specs) -> str:
    """Help-text paragraph on the outputs and their file names."""
    files = "\n".join(
        f"  {s.label}: <project number>_{clean_name_part(s.description)}"
        f"{EXTENSIONS[s.kind]}"
        for s in specs
    )
    return (
        "OUTPUTS:\tTick the outputs wanted (all by default) and choose the CSV "
        "and shapefile folders. Files are named <project number>_<description> "
        "with underscores (just <description> without a project number) and "
        "replace files of the same name. A file that is open in Excel or QGIS "
        "cannot be replaced: it is flagged before the download starts and, if "
        "still open at the end, reported while the other outputs are written. "
        "The CSV is UTF-8 with a byte-order mark (Excel opens it cleanly) and "
        "is read directly by Precipitation data to DDF, Huff Curves, Design "
        "Storm Ensembles and Adjust Sub-daily Rainfall to DDF. The point "
        "layers load with the Mayim styles. Matched grid cell fields: "
        f"{MATCHED_FIELDS_HELP}.\n" + files
    )


def add_output_parameters(alg, specs) -> None:
    """Project number, one tick box per output and the folder parameters."""
    alg.addParameter(
        QgsProcessingParameterString(
            PROJECT_NUMBER,
            "Project number (file name prefix, optional)",
            optional=True,
        )
    )
    for spec in specs:
        alg.addParameter(
            QgsProcessingParameterBoolean(
                spec.key, f"Output: {spec.label}", defaultValue=spec.default
            )
        )
    kinds = {s.kind for s in specs}
    for kind, (name, label) in FOLDER_PARAMS.items():
        if kind in kinds:
            alg.addParameter(
                QgsProcessingParameterFolderDestination(name, label, optional=True)
            )


def _is_locked(path: str) -> bool:
    """True when an existing file cannot be opened for writing (e.g. open in
    Excel). Opening in append mode leaves the file unchanged."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "a"):
            return False
    except OSError:
        return True


class ExtractOutputs:
    """Resolves the selected outputs and their paths before the extraction,
    and writes them afterwards with locked-file reporting."""

    def __init__(self, alg, specs, parameters, context, feedback, styles_dir):
        self.specs = {s.key: s for s in specs}
        self.context = context
        self.feedback = feedback
        self.styles_dir = Path(styles_dir)
        self.keep_alive = []
        project_number = (
            alg.parameterAsString(parameters, PROJECT_NUMBER, context) or ""
        ).strip()
        selected = [
            s for s in specs if alg.parameterAsBoolean(parameters, s.key, context)
        ]
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        self.folders = {}
        for kind in {s.kind for s in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = alg.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(
                    f"Choose the {label[0].lower() + label[1:]}."
                )
            os.makedirs(folder, exist_ok=True)
            self.folders[kind] = folder
        self.paths = {s.key: s.path(self.folders, project_number) for s in selected}
        self.outputs = {
            name: self.folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in self.folders
        }
        self.written = []
        for key, path in self.paths.items():
            if self.specs[key].kind == CSV and _is_locked(path):
                feedback.reportError(
                    f"{Path(path).name} is open in another program - close it "
                    "before the extraction finishes, or it cannot be replaced.",
                    fatalError=False,
                )

    def wants(self, key) -> bool:
        return key in self.paths

    def _failed(self, key, error):
        self.feedback.reportError(
            f"{self.specs[key].label}: {Path(self.paths[key]).name} could not be "
            f"written ({error}). If it is open in Excel or QGIS, close it and run "
            "again.",
            fatalError=False,
        )

    def _done(self, key, path, message=None):
        self.outputs[key] = path
        self.written.append(path)
        self.feedback.pushInfo(message or f"Written: {path}")

    def write_csv(self, key, writer):
        """``writer(path)`` writes the file and may return a message."""
        if key not in self.paths:
            return None
        path = self.paths[key]
        try:
            message = writer(path)
        except Exception as e:  # locked or not writable
            self._failed(key, e)
            return None
        self._done(key, path, message if isinstance(message, str) else None)
        return path

    def write_layer(self, key, layer, style_filename):
        """Writes a memory point layer to its shapefile and loads it with the
        given QML style from the tool's styles folder."""
        if key not in self.paths:
            return None
        if layer is None or layer.featureCount() == 0:
            self.feedback.pushWarning(f"{self.specs[key].label}: nothing to write.")
            return None
        try:
            path = save_shapefile(
                layer, self.paths[key], self.context.transformContext()
            )
        except Exception as e:
            self._failed(key, e)
            return None
        name = Path(path).stem
        details = QgsProcessingContext.LayerDetails(name, self.context.project(), name)
        style = self.styles_dir / style_filename
        if style.is_file():
            processor = _StylePostProcessor(style)
            details.setPostProcessor(processor)
            self.keep_alive.append(processor)
        self.context.addLayerToLoadOnCompletion(path, details)
        self._done(key, path)
        return path

    def finish(self) -> dict:
        if not self.written:
            raise QgsProcessingException(
                "No output could be written - see the errors above."
            )
        return self.outputs


class _StylePostProcessor(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, style_path: Path):
        super().__init__()
        self.style_path = str(style_path)

    def postProcessLayer(self, layer, context, feedback):
        layer.loadNamedStyle(self.style_path)
        layer.triggerRepaint()


def matched_points_layer(results_by_site: dict):
    """One point per site at the centre of the grid cell the data was read
    from (``result.matched_lat/matched_lon``), with the requested point and
    the distance. Sites without a match are left out."""
    rows = [
        (label, r)
        for label, r in results_by_site.items()
        if getattr(r, "matched_lat", None) is not None
        and getattr(r, "matched_lon", None) is not None
    ]
    if not rows:
        return None
    fields = QgsFields()
    for name, kind in (
        ("site", QMetaType.Type.QString),
        ("req_lat", QMetaType.Type.Double),
        ("req_lon", QMetaType.Type.Double),
        ("match_lat", QMetaType.Type.Double),
        ("match_lon", QMetaType.Type.Double),
        ("dist_km", QMetaType.Type.Double),
    ):
        fields.append(QgsField(name, kind))
    layer = QgsMemoryProviderUtils.createMemoryLayer(
        "Matched grid cell", fields, Qgis.WkbType.Point, WGS84
    )
    feats = []
    for label, r in rows:
        feat = QgsFeature(fields)
        feat.setGeometry(
            QgsGeometry.fromPointXY(QgsPointXY(r.matched_lon, r.matched_lat))
        )
        feat.setAttributes(
            [
                str(label),
                getattr(r, "requested_lat", None),
                getattr(r, "requested_lon", None),
                r.matched_lat,
                r.matched_lon,
                getattr(r, "distance_km", None),
            ]
        )
        feats.append(feat)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer


def poi_layer(sites: list):
    """One point per requested site: [(label, lat, lon)]."""
    if not sites:
        return None
    fields = QgsFields()
    fields.append(QgsField("site", QMetaType.Type.QString))
    fields.append(QgsField("latitude", QMetaType.Type.Double))
    fields.append(QgsField("longitude", QMetaType.Type.Double))
    layer = QgsMemoryProviderUtils.createMemoryLayer(
        "Points of interest", fields, Qgis.WkbType.Point, WGS84
    )
    feats = []
    for label, lat, lon in sites:
        feat = QgsFeature(fields)
        feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(lon, lat)))
        feat.setAttributes([str(label), lat, lon])
        feats.append(feat)
    layer.dataProvider().addFeatures(feats)
    layer.updateExtents()
    return layer


def write_dataframe_csv(df, path) -> None:
    """A pandas time series as UTF-8 with BOM, no index."""
    df.to_csv(path, index=False, encoding=CSV_ENCODING)
