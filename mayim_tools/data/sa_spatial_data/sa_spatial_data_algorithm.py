"""Processing Toolbox algorithm: Extract: South African spatial data.

Downloads, for a study area (polygon layer or map extent, optional buffer),
the current vector data from the DFFE geoportal (CSG cadastre, MDB
boundaries, protected areas) and the Department of Agriculture's AGIS
provincial services (water, catchments, soils, land types, vegetation,
biodiversity), as shapefiles in the chosen CRS, with a CSV of the sources,
feature counts and data dates. Features that touch the area are kept whole.
The catalogue and the REST reading are in core.py.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsSingleSymbolRenderer,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import clean_name_part, output_file_name
from mayim_tools.hydrology.dws_wr2012 import core as rest
from mayim_tools.hydrology.dws_wr2012.qgis_layers import (
    WGS84,
    add_features,
    make_fetch,
    memory_layer,
    save_shapefile,
)

from . import core

# outline colours per group (no fills, so layers do not hide each other)
GROUP_COLOURS = {
    "Cadastre": "#000000",
    "Admin boundaries": "#FF00FF",
    "Water and catchments": "#0000FF",
    "Environment and agriculture": "#00CC00",
}


class _Styler(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, colour, width):
        super().__init__()
        self.colour, self.width = colour, width

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            gtype = layer.geometryType()
            if gtype == Qgis.GeometryType.Polygon:
                sym = QgsFillSymbol.createSimple(
                    {
                        "style": "no",
                        "outline_color": self.colour,
                        "outline_width": str(self.width),
                        "outline_width_unit": "MM",
                    }
                )
            elif gtype == Qgis.GeometryType.Line:
                sym = QgsLineSymbol.createSimple(
                    {"line_color": self.colour, "line_width": str(self.width)}
                )
            else:
                sym = QgsMarkerSymbol.createSimple({"color": self.colour, "size": "2"})
            layer.setRenderer(QgsSingleSymbolRenderer(sym))
            layer.triggerRepaint()
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style not applied to {layer.name()}: {e}")


class SaSpatialDataAlgorithm(QgsProcessingAlgorithm):
    AREA = "AREA"
    EXTENT = "EXTENT"
    BUFFER = "BUFFER"
    LAYERS = "LAYERS"
    OUTPUT_CRS = "OUTPUT_CRS"
    MAX_FEATURES = "MAX_FEATURES"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    FOLDER_SHP = "FOLDER_SHP"
    FOLDER_CSV = "FOLDER_CSV"
    WRITE_CSV = "WRITE_CSV"
    LOAD = "LOAD"

    def __init__(self):
        super().__init__()
        self._post_processors = []

    def createInstance(self):
        return SaSpatialDataAlgorithm()

    def name(self):
        return "sa_spatial_data_extract"

    def displayName(self):
        return "Extract: South African spatial data"

    def group(self):
        return "Data Tools"

    def groupId(self):
        return "data_tools"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def shortHelpString(self):
        rows = []
        for group in core.GROUPS:
            rows.append(f"  {group}:")
            for i in core.CATALOGUE:
                if i.group == group:
                    src = f"{core.SOURCE_NAMES[i.source]} " + (
                        f"({i.service.split('/')[1]})"
                        if i.source == "dffe"
                        else "(provincial service)"
                    )
                    rows.append(
                        f"    {i.label}{' [default]' if i.default else ''} - "
                        f"<project number>_{clean_name_part(i.description)}.shp; "
                        f"{src}"
                    )
        return (
            f"Extract: South African spatial data (version {core.TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tDownloads the current vector data for a study area from "
            "the government map services, as shapefiles: the Chief Surveyor-"
            "General's cadastre and the Municipal Demarcation Board boundaries "
            "and protected areas (DFFE geoportal), and water, catchment, soil, "
            "land-type, vegetation and biodiversity layers (Department of "
            "Agriculture AGIS, from the service of each province the area "
            "touches). These services show as images in QGIS's browser, but "
            "their layers are vector features that the tool reads directly.\n"
            "\n"
            "AREA:\tA polygon layer (use 'Selected features only' for one "
            "site) or a map extent, with an optional buffer in metres. Every "
            "feature that touches the area is downloaded whole, not clipped, "
            "so farm and erf boundaries stay complete.\n"
            "\n"
            "LAYERS:\n" + "\n".join(rows) + "\n"
            "\n"
            "CURRENCY:\tThe data is as the services hold it on the day of the "
            "run. The sources CSV records the download date and, where the "
            "layer has one (the SG cadastre's DATE_STAMP), the latest date in "
            "the downloaded features. Rasters on these services (rainfall, "
            "land capability, NLC, hillshade) are images only and are not "
            "offered.\n"
            "\n"
            "LIMITS:\t'Maximum features per layer' (default 20000) stops a "
            "layer that would be very large (e.g. erven over a whole metro); "
            "it is left out with a warning. A service that does not answer is "
            "reported and the other layers are still written.\n"
            "\n"
            f"TERMS:\t{core.TERMS}\n"
            "\n"
            "OUTPUTS:\tOne shapefile per layer in the vector folder, in the "
            "output CRS (default the area's CRS), field names shortened to 10 "
            "characters; <project number>_Spatial_Data_Sources.csv in the CSV "
            "folder (layer, custodian, service URL, features, download date, "
            "data date, warnings)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.AREA,
                "Study area (polygons)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterExtent(
                self.EXTENT,
                "or a map extent (when no study area is given)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUFFER,
                "Buffer around the area (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.LAYERS,
                "Layers to download",
                options=core.option_labels(),
                allowMultiple=True,
                defaultValue=core.default_indices(),
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.OUTPUT_CRS,
                "Output CRS (optional; default the area's CRS)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_FEATURES,
                "Maximum features per layer (0 = no limit)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=0,
                defaultValue=20000,
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
            QgsProcessingParameterBoolean(
                self.WRITE_CSV, "Output: sources CSV", defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD, "Load the layers into the project", defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.FOLDER_SHP, "Output folder for vector layers (shapefiles)"
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.FOLDER_CSV, "Output folder for CSV files", optional=True
            )
        )

    # ------------------------------------------------------------------

    def _area(self, parameters, context, feedback):
        """(area geometry in WGS 84, area CRS)."""
        source = self.parameterAsSource(parameters, self.AREA, context)
        buffer_m = self.parameterAsDouble(parameters, self.BUFFER, context)
        if source is not None:
            crs = source.sourceCrs()
            parts = [
                QgsGeometry(f.geometry())
                for f in source.getFeatures()
                if f.hasGeometry()
            ]
            if not parts:
                raise QgsProcessingException("The study area has no polygons.")
            geom = QgsGeometry.unaryUnion(parts)
        else:
            rect = self.parameterAsExtent(parameters, self.EXTENT, context)
            if rect is None or rect.isNull() or rect.isEmpty():
                raise QgsProcessingException("Give a study area layer or a map extent.")
            crs = self.parameterAsExtentCrs(parameters, self.EXTENT, context)
            if not crs.isValid():
                crs = context.project().crs() if context.project() else WGS84
            geom = QgsGeometry.fromRect(rect)
        if buffer_m > 0:
            if crs.isGeographic():
                # buffer in a local metric CRS (UTM zone of the centroid)
                c = geom.centroid().asPoint()
                to_wgs = QgsCoordinateTransform(crs, WGS84, context.transformContext())
                lon = to_wgs.transform(c).x()
                zone = int((lon + 180) // 6) + 1
                utm = QgsCoordinateReferenceSystem(f"EPSG:{32700 + zone}")
                t = QgsCoordinateTransform(crs, utm, context.transformContext())
                g = QgsGeometry(geom)
                g.transform(t)
                g = g.buffer(buffer_m, 16)
                g.transform(t, Qgis.TransformDirection.Reverse)
                geom = g
            else:
                geom = geom.buffer(buffer_m, 16)
        g84 = QgsGeometry(geom)
        if crs != WGS84:
            g84.transform(
                QgsCoordinateTransform(crs, WGS84, context.transformContext())
            )
        return g84, crs

    def processAlgorithm(self, parameters, context, feedback):
        self._post_processors = []
        area84, area_crs = self._area(parameters, context, feedback)
        bb = area84.boundingBox()
        bbox = (bb.xMinimum(), bb.yMinimum(), bb.xMaximum(), bb.yMaximum())
        out_crs = self.parameterAsCrs(parameters, self.OUTPUT_CRS, context)
        if not out_crs.isValid():
            out_crs = area_crs
        indices = self.parameterAsEnums(parameters, self.LAYERS, context)
        items = [core.CATALOGUE[i] for i in sorted(indices)]
        if not items:
            raise QgsProcessingException("Tick at least one layer.")
        project_number = clean_name_part(
            self.parameterAsString(parameters, self.PROJECT_NUMBER, context) or ""
        )
        shp_folder = self.parameterAsFileOutput(parameters, self.FOLDER_SHP, context)
        if not shp_folder:
            raise QgsProcessingException("Choose the output folder for vector layers.")
        os.makedirs(shp_folder, exist_ok=True)
        write_csv = self.parameterAsBoolean(parameters, self.WRITE_CSV, context)
        csv_folder = self.parameterAsFileOutput(parameters, self.FOLDER_CSV, context)
        if write_csv and not csv_folder:
            csv_folder = shp_folder
        load = self.parameterAsBoolean(parameters, self.LOAD, context)
        max_features = self.parameterAsInt(parameters, self.MAX_FEATURES, context)
        feedback.pushInfo(
            f"Area: {bbox[0]:.4f} to {bbox[2]:.4f} E, {bbox[1]:.4f} to "
            f"{bbox[3]:.4f} (WGS 84); output CRS {out_crs.authid()}."
        )
        fetch = make_fetch(feedback)
        services: dict[str, dict] = {}

        def service_json(url):
            if url not in services:
                services[url] = rest.check_response(fetch(url, {"f": "json"}))
            return services[url]

        provinces: list[str] = []
        if any(i.source == "nda" for i in items):
            feedback.setProgressText("Finding the provinces of the area")
            try:
                _, feats, _, _ = core.fetch_features(core.PROVINCE_LAYER, bbox, fetch)
                values = []
                for wkt, attrs in feats:
                    if QgsGeometry.fromWkt(wkt).intersects(area84):
                        values += [v for v in attrs.values() if isinstance(v, str)]
                provinces = core.provinces_in(values)
            except rest.RestError as e:
                feedback.reportError(
                    f"The province boundaries did not answer ({e}); the "
                    "Department of Agriculture layers are left out.",
                    fatalError=False,
                )
            if provinces:
                feedback.pushInfo(
                    "Provinces: " + ", ".join(p.replace("_", " ") for p in provinces)
                )

        today = datetime.now().strftime("%Y-%m-%d")
        rows = []
        outputs = {}
        total = len(items)
        for n, item in enumerate(items):
            if feedback.isCanceled():
                break
            feedback.setProgressText(f"{item.group}: {item.label}")
            feedback.setProgress(100.0 * n / total)
            urls = (
                [core.service_url(item, p) for p in provinces]
                if item.source == "nda"
                else [core.service_url(item)]
            )
            if not urls:
                rows.append(self._row(item, "", 0, today, "", "no province found"))
                continue
            fields: dict[str, str] = {}
            feats = []
            kind = ""
            notes = []
            used_urls = []
            for svc in urls:
                try:
                    ids = core.resolve_layer_ids(
                        service_json(svc), item.names, item.fallback_ids
                    )
                except rest.RestError as e:
                    notes.append(f"{svc} did not answer ({e})")
                    continue
                if not ids:
                    notes.append(f"{svc}: no layer named {' / '.join(item.names)}")
                for lid, lname in ids:
                    url = f"{svc}/{lid}"
                    try:
                        f_fields, f_feats, f_kind, warns = core.fetch_features(
                            url, bbox, fetch, max_features, feedback.isCanceled
                        )
                    except core.TooManyFeaturesError as e:
                        notes.append(
                            f"{lname}: {e.count} features touch the area's "
                            f"bounding box (limit {e.limit}) - left out; use a "
                            "smaller area or raise the limit"
                        )
                        continue
                    except rest.RestError as e:
                        notes.append(f"{url} did not answer ({e})")
                        continue
                    used_urls.append(url)
                    kind = kind or f_kind
                    notes += [f"{lname}: {w}" for w in warns]
                    for name, k in f_fields:
                        fields.setdefault(name, k)
                    merged = item.merge
                    for wkt, attrs in f_feats:
                        g = QgsGeometry.fromWkt(wkt)
                        if not g.isEmpty() and g.intersects(area84):
                            if merged:
                                attrs = dict(attrs, layer=_short(lname))
                            feats.append((g, attrs))
            for note in notes:
                feedback.pushWarning(f"{item.label}: {note}")
            if item.merge:
                fields.setdefault("layer", "string")
            data_date = core.latest_date([(None, a) for _, a in feats])
            if not feats or not kind:
                if not notes:
                    feedback.pushInfo(f"{item.label}: no features in the area.")
                rows.append(
                    self._row(
                        item,
                        "; ".join(used_urls) or "; ".join(urls),
                        0,
                        today,
                        data_date,
                        "; ".join(notes),
                    )
                )
                continue
            field_list = list(fields.items())
            layer = memory_layer(item.label, kind, field_list)
            count = add_features(layer, field_list, feats, kind)
            path = os.path.join(
                shp_folder,
                output_file_name(project_number, item.description, "shapefile"),
            )
            try:
                save_shapefile(layer, path, out_crs, context.transformContext())
            except RuntimeError as e:
                feedback.reportError(str(e), fatalError=False)
                rows.append(
                    self._row(
                        item,
                        "; ".join(used_urls),
                        count,
                        today,
                        data_date,
                        f"not written: {e}",
                    )
                )
                continue
            feedback.pushInfo(
                f"{item.label}: {count} features"
                + (f", latest data date {data_date}" if data_date else "")
                + f" -> {path}"
            )
            outputs[item.key] = path
            rows.append(
                self._row(
                    item,
                    "; ".join(used_urls),
                    count,
                    today,
                    data_date,
                    "; ".join(notes),
                )
            )
            if load:
                self._load(context, path, item)

        if write_csv:
            os.makedirs(csv_folder, exist_ok=True)
            cpath = os.path.join(
                csv_folder,
                output_file_name(project_number, "Spatial Data Sources", "csv"),
            )
            try:
                self._write_csv(cpath, rows, bbox, out_crs, today)
                outputs["CSV"] = cpath
                feedback.pushInfo(f"Sources: {cpath}")
            except OSError as e:
                feedback.reportError(
                    f"{Path(cpath).name} could not be written ({e}); close it and "
                    "run again.",
                    fatalError=False,
                )
        if not any(k != "CSV" for k in outputs):
            feedback.reportError(
                "No layer had features in the area or none could be downloaded.",
                fatalError=False,
            )
        feedback.setProgress(100)
        return outputs

    def _load(self, context, path, item):
        name = Path(path).stem
        context.addLayerToLoadOnCompletion(
            path,
            QgsProcessingContext.LayerDetails(name, context.project(), name),
        )
        styler = _Styler(
            GROUP_COLOURS.get(item.group, "#000000"),
            0.5 if item.key in ("parent_farm", "provinces") else 0.26,
        )
        self._post_processors.append(styler)
        context.layerToLoadOnCompletionDetails(path).setPostProcessor(styler)

    @staticmethod
    def _row(item, url, count, today, data_date, notes):
        return {
            "group": item.group,
            "layer": item.label,
            "custodian": core.CUSTODIANS.get(
                item.group, core.SOURCE_NAMES[item.source]
            ),
            "service": url,
            "features": count,
            "downloaded": today,
            "data_date": data_date or "-",
            "notes": notes or "",
        }

    @staticmethod
    def _write_csv(path, rows, bbox, crs, today):
        key = [
            ("group", "layer group"),
            ("layer", "layer"),
            ("custodian", "data custodian / publisher"),
            ("service", "ArcGIS REST layer URL(s) read"),
            ("features", "features that touch the area (kept whole)"),
            ("downloaded", "download date"),
            (
                "data_date",
                "latest date in the features' date field, where there is one",
            ),
            ("notes", "warnings"),
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["# Column key"])
            for name, text in key:
                w.writerow([f"# {name}", text])
            w.writerow(["[run]"])
            w.writerow(["item", "value"])
            w.writerow(
                ["tool", f"Extract: South African spatial data {core.TOOL_VERSION}"]
            )
            w.writerow(["run date", today])
            w.writerow(
                ["area bounding box (WGS 84)", " ".join(f"{v:.6f}" for v in bbox)]
            )
            w.writerow(["output CRS", crs.authid()])
            w.writerow(["terms", core.TERMS])
            w.writerow([])
            w.writerow(["[layers]"])
            w.writerow([k for k, _ in key])
            for r in rows:
                w.writerow([r[k] for k, _ in key])


def _short(name: str) -> str:
    """'ENV_DB.LDR.National_Park' -> 'National Park'."""
    return name.split(".")[-1].replace("_", " ").replace("  ", " ").strip()
