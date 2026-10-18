"""Processing Toolbox algorithm (Flood Extent Terrain Profile): DEM +
section line(s) drawn on the map, optional
water-surface elevation rasters (one per event) and a property polygon in
-> report-ready section figures (16.5 cm wide), the sampled profiles, the
flood-to-property measurements and map layers out.

Thin wrapper - all real logic lives in core.py and figure.py (zero QGIS
dependency, independently testable; see tests/test_terrain_profile_core.py).
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from qgis.core import (
    Qgis,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsLineSymbol,
    QgsMapRendererCustomPainterJob,
    QgsMapSettings,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterGeometry,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProject,
    QgsRectangle,
    QgsRendererCategory,
    QgsTextFormat,
    QgsVectorFileWriter,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QMetaType, QSize
from qgis.PyQt.QtGui import QColor, QFont, QIcon, QImage, QPainter

from mayim_tools.hydrology._common.d8_network import GridError
from mayim_tools.hydrology._common.depression import (
    polygon_window,
    raster_info,
    read_window,
    window_geotransform,
)

from .core import (
    DEFAULT_EDGE_EXTENSION_CELLS,
    DEFAULT_SEARCH_RADIUS_M,
    EDGE_WATERLINE,
    MEASUREMENT_FIELDS,
    STATUS_DRY,
    Grid,
    build_section,
    measure,
    nice_step,
    plan_distance,
    point_at,
    section_labels,
)
from .export import write_measurements_csv, write_profile_csv

TOOL_VERSION = "0.3.2"
DISPLAY_NAME = "Flood Extent Terrain Profile"

FORMAT_OPTIONS = ["PNG (300 dpi)", "SVG (text kept as Arial)", "PDF"]
FORMAT_EXT = ["png", "svg", "pdf"]

_S = QMetaType.Type.QString
_D = QMetaType.Type.Double

SECTION_FIELDS = [
    ("section", _S, "section label"),
    ("start_lbl", _S, "label at the start of the line (chainage 0)"),
    ("end_lbl", _S, "label at the end of the line"),
    ("length_m", _D, "length of the section, m"),
    ("figure", _S, "figure file (first format)"),
]

POINT_FIELDS = [
    ("section", _S, "section label"),
    ("kind", _S, "section end / chainage / property boundary / flood edge"),
    ("event", _S, "water-surface (event) label, for flood edges"),
    ("chainage_m", _D, "chainage along the section, m"),
    ("elev_m", _D, "ground (or water) level at the point, m"),
    ("label", _S, "map label"),
]

PLAN_FIELDS = [
    ("event", _S, "water-surface (event) label"),
    ("status", _S, "property dry / property inundated / none in radius"),
    ("dist_m", _D, "shortest plan distance property to flooded cell, m"),
    ("accuracy_m", _D, "+/- accuracy (half a WSE cell), m"),
    ("wet_area_m2", _D, "flooded area inside the property, m2"),
]

OWN_LAYER_STEMS = ("section_lines", "section_points", "plan_distance")

KIND_END = "section end"
KIND_CHAINAGE = "chainage"
KIND_BOUNDARY = "property boundary"
KIND_EDGE = "flood edge"
PRIME = "′"


def _fields(specs) -> QgsFields:
    fields = QgsFields()
    for name, qtype, _ in specs:
        fields.append(QgsField(name, qtype))
    return fields


def _safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text).strip("_") or "section"


def _is_null(value) -> bool:
    return value is None or str(value) in ("", "NULL")


class _LayerStyler(QgsProcessingLayerPostProcessorInterface):
    """Builds the output symbology in code once the layer is loaded."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    @staticmethod
    def _labels(layer, field, size=10.0, colour="#1f1e1c", placement=None):
        settings = QgsPalLayerSettings()
        settings.fieldName = field
        settings.enabled = True
        fmt = QgsTextFormat()
        fmt.setFont(QFont("Arial Narrow"))
        fmt.setSize(size)
        fmt.setColor(QColor(colour))
        buffer = fmt.buffer()
        buffer.setEnabled(True)
        buffer.setSize(1.0)
        buffer.setColor(QColor("white"))
        fmt.setBuffer(buffer)
        settings.setFormat(fmt)
        if placement is not None:
            settings.placement = placement
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)

    def postProcessLayer(self, layer, context, feedback):
        try:
            if self.kind == "sections":
                layer.renderer().setSymbol(
                    QgsLineSymbol.createSimple({"color": "#1f1e1c", "width": "0.6"})
                )
            elif self.kind == "points":
                styles = {
                    KIND_END: ("square", "#1f1e1c", "3.0"),
                    KIND_CHAINAGE: ("line", "#1f1e1c", "2.6"),
                    KIND_BOUNDARY: ("circle", "#d9531e", "2.8"),
                    KIND_EDGE: ("triangle", "#2a78d6", "2.8"),
                }
                cats = []
                for value, (shape, colour, size) in styles.items():
                    sym = QgsMarkerSymbol.createSimple(
                        {
                            "name": shape,
                            "color": colour,
                            "outline_color": colour if shape == "line" else "white",
                            "size": size,
                        }
                    )
                    cats.append(QgsRendererCategory(value, sym, value))
                layer.setRenderer(QgsCategorizedSymbolRenderer("kind", cats))
                self._labels(layer, "label", size=10.0)
            elif self.kind == "plan":
                layer.renderer().setSymbol(
                    QgsLineSymbol.createSimple(
                        {"color": "#2a78d6", "width": "0.5", "line_style": "dash"}
                    )
                )
                self._labels(layer, "label", colour="#0f3f8c")
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style skipped ({self.kind}): {e}")
        layer.triggerRepaint()


class TerrainProfileAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    SECTION_LINE = "SECTION_LINE"
    WSE_RASTERS = "WSE_RASTERS"
    WSE_LABELS = "WSE_LABELS"
    PROPERTY = "PROPERTY"
    PROPERTY_LABEL = "PROPERTY_LABEL"
    VE = "VE"
    SAMPLE_INTERVAL = "SAMPLE_INTERVAL"
    SEARCH_RADIUS = "SEARCH_RADIUS"
    FORMATS = "FORMATS"
    PLAN_VIEW = "PLAN_VIEW"
    MIN_HEIGHT = "MIN_HEIGHT"
    MAX_HEIGHT = "MAX_HEIGHT"
    EDGE_EXTENSION = "EDGE_EXTENSION"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return TerrainProfileAlgorithm()

    def name(self):
        return "terrain_profile"

    def displayName(self):
        return DISPLAY_NAME

    def flags(self):
        # the plan view renders the project's map layers, which must happen
        # on the main thread
        return super().flags() | Qgis.ProcessingAlgorithmFlag.NoThreading

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        glossary = "\n".join(f"  {n}: {d}" for n, d in MEASUREMENT_FIELDS)
        return (
            f"{DISPLAY_NAME} (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tReport-ready cross-sections and long-sections through a "
            "DEM, with flood water surfaces and a development property, and "
            "the horizontal and vertical distances between the flood and the "
            "property.\n"
            "\n"
            "INPUTS:\tA DEM in a projected CRS in metres and the section "
            "line(s): press 'Draw section...', click the vertices on the map "
            "(snapping applies), right-click or Enter to finish - Backspace "
            "removes the last vertex, Esc cancels. Draw again for more sections; "
            "they are labelled A, B, C ... in drawing order (ends A and A′). "
            "Chainage runs from the first click, so draw each line in the "
            "direction it should read left to right. Optional: one or more "
            "water-surface elevation rasters (HEC-RAS WSE, DEM Inundation "
            "Boundary, ...) in the DEM's CRS, with labels such as '1:50, "
            "1:100' (otherwise the layer names); and the property polygon "
            "(all features together).\n"
            "\n"
            "FLOOD EDGE:\tGround is wet where the water surface is above it. "
            "WSE rasters hold values on wet cells only, so the water surface "
            "is carried 1.5 cells past the last wet cell and the waterline is "
            "placed where it meets the ground (as RAS Mapper does). Where the "
            "water surface ends before meeting the ground, the edge is the "
            "limit of the WSE data and its distance is marked ≈.\n"
            "\n"
            "MEASUREMENTS:\tProperty dry - the horizontal distance along the "
            "section from the property boundary crossing nearest the flood to "
            "the flood edge, and the ground level there minus the water level "
            "(vertical). Property inundated - the wet length inside the "
            "property and the maximum depth. Distances along a section are "
            "measured along the drawn line; the shortest plan distance from "
            "the property to the nearest flooded cell (within the search "
            "radius) is reported separately, to within half a WSE cell. The "
            "figure is dimensioned for the first water surface; the CSV has "
            "every event.\n"
            "\n"
            "FIGURES:\t16.5 cm wide, Arial (main text) and Arial Narrow (minor "
            "text), drawn to scale at the vertical exaggeration (default "
            "×10). The height follows from the exaggeration, kept between "
            "the minimum and maximum height: a flat section gets a wider "
            "elevation range, a steep one a smaller exaggeration (reported in "
            "the log). The profile is titled 'Section A\u2013A\u2032' below "
            "its legend. Saved as PNG (300 dpi) and SVG at their final size - "
            "insert at 100 %.\n"
            "\n"
            "PLAN VIEW:\tWith 'Also draw a plan view' (default), a second "
            "figure per section, section_A_plan.png, has a plan above the "
            "profile: the map layers visible in the project (in layer order), "
            "fitted to the section, with the section line, chainage ticks, A "
            "and A\u2032, the property outline, the flood edges, the shortest "
            "plan distance, a scale bar and a north arrow (grid north). The "
            "profile on its own is always saved too. Set the map up as it "
            "should appear (basemap, flood extent, cadastre) before running.\n"
            "\n"
            "OUTPUT FOLDER:\tEverything is written to the folder: the figures "
            "(section_A.png, section_A_plan.png ...), profiles.csv (every "
            "sample), measurements.csv, "
            "section_lines.gpkg and section_points.gpkg (both loaded into the "
            "map, styled) and, with a property and water surfaces, "
            "plan_distance.gpkg (shortest plan distance lines).\n"
            "\n"
            "MEASUREMENTS.CSV FIELDS:\n" + glossary
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(self.DEM, "DEM (projected, metres)")
        )
        section = QgsProcessingParameterGeometry(
            self.SECTION_LINE,
            "Section line(s) - draw on the map",
            geometryTypes=[Qgis.GeometryType.Line.value],
            allowMultipart=True,
        )
        try:  # draw-on-map panel (falls back to a WKT text box)
            from .section_widget import SectionLineWidgetWrapper

            if SectionLineWidgetWrapper is not None:
                section.setMetadata(
                    {"widget_wrapper": {"class": SectionLineWidgetWrapper}}
                )
        except Exception:  # noqa: BLE001 - never block the tool over the widget
            pass
        self.addParameter(section)
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.WSE_RASTERS,
                "Water-surface elevation raster(s), one per event (optional)",
                layerType=QgsProcessing.SourceType.TypeRaster,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.WSE_LABELS,
                "Event labels, comma-separated in the same order (optional)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.PROPERTY,
                "Property boundary polygon (optional)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PROPERTY_LABEL,
                "Property label on the figure",
                defaultValue="Property",
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.VE,
                "Vertical exaggeration",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=1.0,
                defaultValue=10.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FORMATS,
                "Figure formats",
                options=FORMAT_OPTIONS,
                allowMultiple=True,
                defaultValue=[0, 1],
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.PLAN_VIEW,
                "Also draw a plan view above the profile (visible map layers)",
                defaultValue=True,
            )
        )
        advanced = [
            QgsProcessingParameterNumber(
                self.SAMPLE_INTERVAL,
                "Sampling interval along the section (m; 0 = DEM cell size)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.0,
            ),
            QgsProcessingParameterNumber(
                self.SEARCH_RADIUS,
                "Search radius for the plan distance (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=1.0,
                defaultValue=DEFAULT_SEARCH_RADIUS_M,
            ),
            QgsProcessingParameterNumber(
                self.MIN_HEIGHT,
                "Minimum figure height (cm)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=3.0,
                defaultValue=6.0,
            ),
            QgsProcessingParameterNumber(
                self.MAX_HEIGHT,
                "Maximum figure height (cm)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=4.0,
                defaultValue=14.0,
            ),
            QgsProcessingParameterNumber(
                self.EDGE_EXTENSION,
                "Carry the water surface past the last wet cell by (cells)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=DEFAULT_EDGE_EXTENSION_CELLS,
            ),
        ]
        for p in advanced:
            p.setFlags(p.flags() | Qgis.ProcessingParameterFlag.Advanced)
            self.addParameter(p)
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER, "Output folder (figures, tables and layers)"
            )
        )

    # ------------------------------------------------------------------

    def _load(self, context, path, name, kind):
        """Loads a written layer into the project when the run ends, styled."""
        details = QgsProcessingContext.LayerDetails(name, context.project(), name)
        processor = _LayerStyler(kind)
        details.setPostProcessor(processor)
        self._post_processors.append(processor)  # Processing keeps a weak ref
        context.addLayerToLoadOnCompletion(path, details)

    @staticmethod
    def _transform(source, crs, context):
        if source is None or source.sourceCrs() == crs:
            return None
        return QgsCoordinateTransform(
            source.sourceCrs(), crs, context.transformContext()
        )

    def _read_property(self, source, crs, context):
        if source is None:
            return None
        transform = self._transform(source, crs, context)
        parts = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            geom.convertToStraightSegment()
            if not geom.isGeosValid():
                geom = geom.makeValid()
            parts.append(geom)
        if not parts:
            return None
        union = QgsGeometry.unaryUnion(parts)
        union.convertGeometryCollectionToSubclass(Qgis.GeometryType.Polygon)
        polys = union.asMultiPolygon() if union.isMultipart() else [union.asPolygon()]
        return [
            [[(p.x(), p.y()) for p in ring] for ring in poly if ring]
            for poly in polys
            if poly
        ]

    @staticmethod
    def _render_map(context, crs, extent, width_px, height_px, folder):
        """Renders the project's visible layers (in layer order, not this
        tool's own earlier outputs) for the extent; returns (RGBA array,
        extent used, layer names)."""
        from .figure import PLAN_DPI

        project = context.project() or QgsProject.instance()
        root = project.layerTreeRoot()
        own = {
            str((Path(folder) / f"{stem}.gpkg").resolve()).lower()
            for stem in OWN_LAYER_STEMS
        }
        layers = []
        for layer in root.layerOrder():
            node = root.findLayer(layer.id())
            if node is None or not node.isVisible():
                continue
            # skip only this tool's own layers from an earlier run (they are
            # drawn on top anyway); everything else visible is rendered
            source = layer.source().split("|")[0]
            try:
                resolved = Path(source).resolve()
                stem = re.sub(r"_\d{8}_\d{6}$", "", resolved.stem)
                key = str(resolved.with_name(stem + resolved.suffix)).lower()
                if key in own:
                    continue
            except (OSError, ValueError):
                pass
            layers.append(layer)
        settings = QgsMapSettings()
        settings.setLayers(layers)
        settings.setDestinationCrs(crs)
        settings.setTransformContext(project.transformContext())
        settings.setOutputSize(QSize(width_px, height_px))
        settings.setOutputDpi(PLAN_DPI)
        settings.setExtent(QgsRectangle(*extent))
        settings.setBackgroundColor(QColor("white"))
        settings.setFlag(Qgis.MapSettingsFlag.Antialiasing, True)
        settings.setFlag(Qgis.MapSettingsFlag.DrawLabeling, True)
        # wait for rasters, WMS / XYZ tiles and other remote sources to load
        settings.setFlag(Qgis.MapSettingsFlag.RenderBlocking, True)
        settings.setFlag(Qgis.MapSettingsFlag.UseAdvancedEffects, True)
        image = QImage(width_px, height_px, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor("white"))
        painter = QPainter(image)
        job = QgsMapRendererCustomPainterJob(settings, painter)
        job.renderSynchronously()
        painter.end()
        rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
        ptr = rgba.constBits()
        ptr.setsize(rgba.sizeInBytes())
        rows = np.frombuffer(ptr, np.uint8).reshape(rgba.height(), rgba.bytesPerLine())
        array = (
            rows[:, : rgba.width() * 4].reshape(rgba.height(), rgba.width(), 4).copy()
        )
        ve = settings.visibleExtent()
        used = (ve.xMinimum(), ve.yMinimum(), ve.xMaximum(), ve.yMaximum())
        return array, used, [layer.name() for layer in layers]

    def _read_drawn_sections(self, parameters, crs, context):
        """The drawn line(s) as [(label, [(x, y), ...]), ...] in the DEM CRS.
        Accepts a (referenced) geometry or 'CRS=<authid>;<WKT>' text."""
        raw = parameters.get(self.SECTION_LINE)
        src_crs = None
        if isinstance(raw, QgsGeometry):  # also QgsReferencedGeometry
            geom = QgsGeometry(raw)
            if hasattr(raw, "crs"):
                src_crs = raw.crs()
        else:
            text = "" if _is_null(raw) else str(raw).strip()
            match = re.match(r"^\s*(?:CRS=(.*?);)?(.*)$", text, re.S)
            if match.group(1):
                src_crs = QgsCoordinateReferenceSystem(match.group(1).strip())
            geom = QgsGeometry.fromWkt(match.group(2).strip())
        if geom is None or geom.isNull() or geom.isEmpty():
            raise QgsProcessingException(
                "Draw at least one section line on the map (Draw section...)."
            )
        if src_crs is None or not src_crs.isValid():
            project = context.project()
            src_crs = project.crs() if project is not None else crs
        if src_crs.isValid() and src_crs != crs:
            geom.transform(
                QgsCoordinateTransform(src_crs, crs, context.transformContext())
            )
        geom.convertToStraightSegment()
        lines = geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
        lines = [[(p.x(), p.y()) for p in ln] for ln in lines if len(ln) >= 2]
        if not lines:
            raise QgsProcessingException("The section line needs two or more vertices.")
        return list(zip(section_labels(len(lines)), lines, strict=True))

    @staticmethod
    def _free_path(path: Path, feedback) -> Path:
        """Overwrites an earlier output; when it is locked (still loaded in
        QGIS) a time-stamped name is used instead."""
        if not path.exists():
            return path
        try:
            path.unlink()
            return path
        except OSError:
            alt = path.with_name(
                f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}"
            )
            feedback.pushWarning(
                f"{path.name} is in use (loaded in QGIS?) - written as {alt.name}."
            )
            return alt

    def _write_gpkg(self, path, specs, wkb, crs, context, feedback, items):
        """Writes [(geometry, {field: value}), ...] to a GeoPackage."""
        path = self._free_path(path, feedback)
        fields = _fields(specs)
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = path.stem
        options.fileEncoding = "UTF-8"
        writer = QgsVectorFileWriter.create(
            str(path), fields, wkb, crs, context.transformContext(), options
        )
        if writer.hasError() != QgsVectorFileWriter.WriterError.NoError:
            raise QgsProcessingException(
                f"Could not write {path.name}: {writer.errorMessage()}"
            )
        for geom, attrs in items:
            f = QgsFeature(fields)
            f.setGeometry(geom)
            for key, value in attrs.items():
                f[key] = value
            writer.addFeature(f)
        del writer  # flushes and closes the file
        return str(path)

    @staticmethod
    def _read_grid(path, bbox, pad=3):
        """Grid for the window covering bbox (xmin, ymin, xmax, ymax), or
        None when the raster does not overlap it."""
        shape, gt, _, nodata = raster_info(path)
        x0, y0, x1, y1 = bbox
        ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        window = polygon_window([[ring]], gt, shape, pad=pad)
        if window is None:
            return None
        array = read_window(path, window)
        return Grid.from_raw(
            array, window_geotransform(gt, window[0], window[1]), nodata
        )

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        try:
            from mayim_tools._common.report_figures import save_report_figure

            from .figure import HouseFonts, draw_section, plan_view
        except ImportError as e:
            raise QgsProcessingException(
                f"The figures need matplotlib in QGIS's Python ({e})."
            ) from e

        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        if dem_layer is None:
            raise QgsProcessingException("A DEM is required.")
        crs = dem_layer.crs()
        if crs.isGeographic():
            raise QgsProcessingException(
                "The DEM is in a geographic CRS (degrees). Chainages and distances "
                "need a projected CRS in metres - reproject the DEM first."
            )
        wse_layers = (
            self.parameterAsLayerList(parameters, self.WSE_RASTERS, context) or []
        )
        labels_text = self.parameterAsString(parameters, self.WSE_LABELS, context) or ""
        prop_source = self.parameterAsSource(parameters, self.PROPERTY, context)
        prop_label = (
            self.parameterAsString(parameters, self.PROPERTY_LABEL, context)
            or "Property"
        )
        ve = self.parameterAsDouble(parameters, self.VE, context)
        interval = self.parameterAsDouble(parameters, self.SAMPLE_INTERVAL, context)
        radius = self.parameterAsDouble(parameters, self.SEARCH_RADIUS, context)
        min_h = self.parameterAsDouble(parameters, self.MIN_HEIGHT, context)
        max_h = self.parameterAsDouble(parameters, self.MAX_HEIGHT, context)
        if max_h < min_h:
            raise QgsProcessingException(
                "The maximum figure height is less than the minimum."
            )
        extension = self.parameterAsDouble(parameters, self.EDGE_EXTENSION, context)
        want_plan = self.parameterAsBool(parameters, self.PLAN_VIEW, context)
        formats = [
            FORMAT_EXT[i]
            for i in self.parameterAsEnums(parameters, self.FORMATS, context)
        ] or ["png"]
        folder = Path(self.parameterAsString(parameters, self.OUTPUT_FOLDER, context))
        folder.mkdir(parents=True, exist_ok=True)

        sections_in = self._read_drawn_sections(parameters, crs, context)
        polygons = self._read_property(prop_source, crs, context)
        if prop_source is not None and not polygons:
            feedback.pushWarning("The property layer has no polygon features.")

        # event labels
        given = (
            [t.strip() for t in labels_text.split(",")] if labels_text.strip() else []
        )
        events = []
        for i, layer in enumerate(wse_layers):
            if layer.crs() != crs:
                raise QgsProcessingException(
                    f"The water-surface raster '{layer.name()}' is in "
                    f"{layer.crs().authid()}, the DEM in {crs.authid()} - use "
                    "rasters in the same CRS."
                )
            label = given[i] if i < len(given) and given[i] else layer.name()
            events.append((label, layer))
        if given and len(given) != len(wse_layers):
            feedback.pushWarning(
                f"{len(given)} label(s) for {len(wse_layers)} water-surface "
                "raster(s) - layer names used where a label is missing."
            )

        fonts = HouseFonts()
        for w in fonts.warnings:
            feedback.pushWarning(w)
        note = f"Ground: {dem_layer.name()}"
        if events:
            note += "; water surface: " + ", ".join(lyr.name() for _, lyr in events)

        def read_all(bbox):
            """DEM and WSE grids for one window (each raster read only there)."""
            dem = self._read_grid(dem_layer.source(), bbox)
            grids = []
            for label, layer in events:
                grid = self._read_grid(layer.source(), bbox)
                if grid is not None:
                    grids.append((label, grid))
            return dem, grids

        # plan distance (per event, independent of the sections)
        plans = {}
        if polygons and events:
            px = [p[0] for poly in polygons for ring in poly for p in ring]
            py = [p[1] for poly in polygons for ring in poly for p in ring]
            box = (
                min(px) - radius,
                min(py) - radius,
                max(px) + radius,
                max(py) + radius,
            )
            try:
                dem_p, grids_p = read_all(box)
                if dem_p is not None:
                    for label, grid in grids_p:
                        plans[label] = plan_distance(
                            label, polygons, grid, dem_p.sample, radius
                        )
            except GridError as e:
                feedback.pushWarning(f"Plan distance skipped: {e}")

        sections, rows, figures = [], [], {}
        plan_layers_reported = False
        n = len(sections_in)
        for k, (label, xy) in enumerate(sections_in):
            if feedback.isCanceled():
                return {}
            feedback.setProgress(int(90 * k / n))
            try:
                xs = [p[0] for p in xy]
                ys = [p[1] for p in xy]
                dem, grids = read_all((min(xs), min(ys), max(xs), max(ys)))
                if dem is None:
                    raise GridError("the section does not overlap the DEM")
                sec = build_section(
                    label,
                    xy,
                    dem,
                    grids,
                    interval if interval > 0 else dem.cell_size,
                    polygons,
                    extension,
                )
            except GridError as e:
                feedback.pushWarning(f"Section {label} skipped: {e}")
                continue
            for w in sec.warnings:
                feedback.pushWarning(w)
            sec_rows = []
            for ev in sec.events:
                row = measure(sec, ev)
                pd = plans.get(ev.label)
                if pd is not None:
                    row["plan_dist_m"] = pd.distance_m
                    row["plan_status"] = pd.status
                    row["wet_area_in_prop_m2"] = pd.wet_area_in_property_m2
                sec_rows.append(row)
            dimension = sec_rows[0]["dimension"] if sec_rows else None
            fig, lay = draw_section(
                sec,
                dimension,
                ve=ve,
                min_height_cm=min_h,
                max_height_cm=max_h,
                property_label=prop_label,
                source_note=note,
                fonts=fonts,
            )
            paths = save_report_figure(
                fig, folder / f"section_{_safe_name(label)}", formats
            )
            figures[label] = paths
            if want_plan:
                try:
                    pv = plan_view(sec)
                    w, h = pv.pixels
                    pv.image, pv.extent, used = self._render_map(
                        context, crs, pv.extent, w, h, folder
                    )
                    if not used:
                        feedback.pushWarning(
                            "No map layers are visible - the plan view has a "
                            "blank background."
                        )
                    elif not plan_layers_reported:
                        feedback.pushInfo("Plan view layers: " + ", ".join(used))
                        plan_layers_reported = True
                    pv.polygons = polygons
                    pv.plan_lines = [
                        (pd.property_xy, pd.flood_xy, f"{lab}: {pd.distance_m:.1f} m")
                        for lab, pd in plans.items()
                        if pd.property_xy is not None
                    ]
                    fig2, _ = draw_section(
                        sec,
                        dimension,
                        ve=ve,
                        min_height_cm=min_h,
                        max_height_cm=max_h,
                        property_label=prop_label,
                        source_note=note,
                        fonts=fonts,
                        plan=pv,
                    )
                    plan_paths = save_report_figure(
                        fig2, folder / f"section_{_safe_name(label)}_plan", formats
                    )
                    feedback.pushInfo(
                        f"Section {label}: plan and profile (plan 1:{pv.scale:.0f}) "
                        f"-> {Path(plan_paths[0]).name}"
                    )
                except Exception as e:  # the plan must never lose the profile
                    feedback.pushWarning(f"Section {label}: plan view skipped - {e}")
            if lay.ve_reduced:
                feedback.pushWarning(
                    f"Section {label}: vertical exaggeration reduced to "
                    f"×{lay.ve:.1f} to keep the figure within {max_h:g} cm."
                )
            for r in sec_rows:
                feedback.pushInfo(self._summary(r))
            sections.append(sec)
            rows.extend(sec_rows)
            feedback.pushInfo(
                f"Section {label}: {sec.length:.1f} m, figure {lay.height_cm:.1f} cm "
                f"high -> {Path(paths[0]).name}"
            )
        if not sections:
            raise QgsProcessingException("No section could be drawn.")

        write_profile_csv(str(folder / "profiles.csv"), sections)
        if rows:
            write_measurements_csv(str(folder / "measurements.csv"), rows)
        for label, pd in plans.items():
            if pd.distance_m is None:
                feedback.pushInfo(f"{label}: {pd.status} ({radius:g} m).")
            elif pd.status == STATUS_DRY:
                feedback.pushInfo(
                    f"{label}: shortest plan distance property to flood "
                    f"{pd.distance_m:.1f} m (±{pd.accuracy_m:.1f} m)."
                )
            else:
                feedback.pushInfo(
                    f"{label}: property inundated, about "
                    f"{pd.wet_area_in_property_m2:.0f} m2 flooded inside it."
                )

        outputs = {self.OUTPUT_FOLDER: str(folder)}
        outputs.update(
            self._write_layers(context, feedback, folder, crs, sections, plans, figures)
        )
        feedback.setProgress(100)
        return outputs

    @staticmethod
    def _summary(r) -> str:
        head = f"Section {r['section']}, {r['event']}: {r['status']}"
        if r["horiz_dist_m"] is not None:
            return (
                f"{head} - flood edge {r['horiz_dist_m']:.1f} m from the boundary "
                f"along the section; ground at the boundary {r['vert_dist_m']:+.2f} m "
                "relative to the water level"
            )
        if r["wet_length_in_prop_m"]:
            return (
                f"{head} - {r['wet_length_in_prop_m']:.1f} m wet inside the property, "
                f"maximum depth {r['max_depth_in_prop_m']:.2f} m"
            )
        return head

    def _write_layers(self, context, feedback, folder, crs, sections, plans, figures):
        """Section lines and points (loaded into the map, styled) and the
        plan-distance lines, as GeoPackages in the output folder."""
        items = []
        for sec in sections:
            geom = QgsGeometry.fromPolylineXY(
                [QgsPointXY(x, y) for x, y in sec.vertices]
            )
            items.append(
                (
                    geom,
                    {
                        "section": sec.label,
                        "start_lbl": sec.label,
                        "end_lbl": sec.label + PRIME,
                        "length_m": round(sec.length, 2),
                        "figure": Path(figures[sec.label][0]).name,
                    },
                )
            )
        lines_path = self._write_gpkg(
            folder / "section_lines.gpkg",
            SECTION_FIELDS,
            Qgis.WkbType.LineString,
            crs,
            context,
            feedback,
            items,
        )
        self._load(context, lines_path, "Section lines", "sections")

        items = []

        def add(sec, kind, ch, elev, label, event=""):
            x, y = point_at(sec.vertices, ch)
            items.append(
                (
                    QgsGeometry.fromPointXY(QgsPointXY(float(x), float(y))),
                    {
                        "section": sec.label,
                        "kind": kind,
                        "event": event,
                        "chainage_m": round(float(ch), 3),
                        "elev_m": None if math.isnan(elev) else round(float(elev), 3),
                        "label": label,
                    },
                )
            )

        for sec in sections:
            add(sec, KIND_END, 0.0, sec.ground_at(0.0), sec.label)
            add(sec, KIND_END, sec.length, sec.ground_at(sec.length), sec.label + PRIME)
            step = nice_step(sec.length)
            c = step
            while c < sec.length - 0.25 * step:
                add(sec, KIND_CHAINAGE, c, sec.ground_at(c), f"{c:g}")
                c += step
            for c in sec.property_crossings:
                add(sec, KIND_BOUNDARY, c, sec.ground_at(c), "")
            for ev in sec.events:
                for w in ev.intervals:
                    for which in ("start", "end"):
                        ch, z, kind = w.edge(which)
                        if kind == EDGE_WATERLINE:
                            add(sec, KIND_EDGE, ch, z, "", ev.label)
        points_path = self._write_gpkg(
            folder / "section_points.gpkg",
            POINT_FIELDS,
            Qgis.WkbType.Point,
            crs,
            context,
            feedback,
            items,
        )
        self._load(context, points_path, "Section points", "points")

        if plans:
            plan_specs = PLAN_FIELDS + [("label", _S, "map label")]
            items = [
                (
                    QgsGeometry.fromPolylineXY(
                        [QgsPointXY(*pd.property_xy), QgsPointXY(*pd.flood_xy)]
                    ),
                    {
                        "event": label,
                        "status": pd.status,
                        "dist_m": round(pd.distance_m, 2),
                        "accuracy_m": round(pd.accuracy_m, 2),
                        "wet_area_m2": round(pd.wet_area_in_property_m2, 1),
                        "label": f"{label}: {pd.distance_m:.1f} m",
                    },
                )
                for label, pd in plans.items()
                if pd.property_xy is not None
            ]
            if items:
                self._write_gpkg(
                    folder / "plan_distance.gpkg",
                    plan_specs,
                    Qgis.WkbType.LineString,
                    crs,
                    context,
                    feedback,
                    items,
                )
        return {}
