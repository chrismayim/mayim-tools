"""Processing Toolbox algorithm: a DEM that shows a river's water surface and
the river's centreline in -> the DEM with an estimated channel cut below the
water, from the slopes of the banks (two half-parabolas), plus the depth,
water surface, stations, section lines, a station table and a Word report.

Outputs follow the Mayim report and figure standard: tick the outputs
wanted, one folder per kind, files named <project number>_<description>.

Thin wrapper - all real logic lives in core.py (no QGIS; see
tests/test_channel_bathymetry_core.py).
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsMarkerSymbol,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsSingleSymbolRenderer,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    EXTENSIONS,
    RASTER,
    SHAPEFILE,
    WORD,
    OutputSpec,
    clean_name_part,
)
from mayim_tools.hydrology._common.d8_network import GridError, write_raster
from mayim_tools.hydrology._common.depression import (
    coverage_fraction,
    polygon_window,
    raster_info,
    read_window,
    window_geotransform,
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
    STYLE_DEPTH,
    STYLES_DIR,
    _StylePostProcessor,
)

from .core import (
    BASIS_TEXT,
    CAP_TEXT,
    TOOL_VERSION,
    ChannelError,
    Settings,
    run_channel_bathymetry,
)
from .export import write_bed_dem, write_station_csv
from .report import write_report_docx

NODATA = -9999.0
STYLE_DEPTH_PATH = str(STYLES_DIR / STYLE_DEPTH)

_S = QMetaType.Type.QString
_I = QMetaType.Type.Int
_D = QMetaType.Type.Double

STATION_FIELDS = [
    ("line", _I, "centreline number"),
    ("chain_m", _D, "chainage downstream along the centreline, m"),
    ("wse_m", _D, "water level, m"),
    ("width_m", _D, "water width W, m"),
    ("s_left", _D, "left bank slope (looking downstream), m/m"),
    ("s_right", _D, "right bank slope, m/m"),
    ("d_banks", _D, "depth from the bank slopes before the caps, m"),
    ("d_low", _D, "depth - low (shallow) bed, m"),
    ("d_central", _D, "depth - central bed, m"),
    ("d_high", _D, "depth - high (deep) bed, m"),
    ("bed_m", _D, "lowest bed level - central bed, m"),
    ("thalweg", _D, "deepest point from the left edge / W"),
    ("h_left", _D, "left bank-top height above the water, m"),
    ("h_right", _D, "right bank-top height above the water, m"),
    ("basis", _S, "how the depth was set"),
    ("cap", _S, "cap that limited the depth"),
    ("flags", _S, "notes on the station"),
]
SECTION_FIELDS = [
    ("line", _I, "centreline number"),
    ("chain_m", _D, "chainage of the station, m"),
    ("wse_m", _D, "water level, m"),
    ("d_central", _D, "depth - central bed, m"),
    ("length_m", _D, "length of the section line, m"),
]
WATER_FIELDS = [
    ("id", _I, "polygon number"),
    ("area_m2", _D, "water-surface area, m2"),
    ("wse_min", _D, "lowest water level in the polygon, m"),
    ("wse_max", _D, "highest water level in the polygon, m"),
]

OUTPUT_SPECS = [
    OutputSpec("OUTPUT_BED", "Bed DEM - central (whole DEM, channel cut in)",
               "Channel Bed DEM", RASTER),
    OutputSpec("OUTPUT_BED_LOW", "Bed DEM - low (shallow) bed",
               "Channel Bed DEM Low", RASTER, False),
    OutputSpec("OUTPUT_BED_HIGH", "Bed DEM - high (deep) bed",
               "Channel Bed DEM High", RASTER, False),
    OutputSpec("OUTPUT_DEPTH", "Depth below the water (central bed)",
               "Channel Depth", RASTER),
    OutputSpec("OUTPUT_WATER", "Water surface (polygons)", "Channel Water Surface",
               SHAPEFILE),
    OutputSpec("OUTPUT_STATIONS", "Stations (points)", "Channel Stations", SHAPEFILE),
    OutputSpec("OUTPUT_SECTIONS", "Section lines (bank top to bank top)",
               "Channel Sections", SHAPEFILE, False),
    OutputSpec("OUTPUT_CSV", "Station table (CSV)", "Channel Profile", CSV),
    OutputSpec("OUTPUT_DOCX", "Channel bathymetry report (Word)",
               "Channel Bathymetry Report", WORD, False),
]  # fmt: skip
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}

SYMBOLS = {
    "water": lambda: QgsFillSymbol.createSimple(
        {"color": "0,0,255,60", "outline_color": "0,0,255,255",
         "outline_width": "0.26", "outline_width_unit": "MM"}
    ),
    "stations": lambda: QgsMarkerSymbol.createSimple(
        {"name": "circle", "color": "255,0,0,255", "outline_color": "0,0,0,255",
         "outline_width": "0.2", "size": "1.6", "size_unit": "MM"}
    ),
    "sections": lambda: QgsLineSymbol.createSimple(
        {"line_color": "0,0,0,255", "line_width": "0.26", "line_width_unit": "MM"}
    ),
}  # fmt: skip


class _VectorStyler(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802
        try:
            layer.setRenderer(QgsSingleSymbolRenderer(SYMBOLS[self.kind]()))
            layer.triggerRepaint()
        except Exception as e:  # styling must never fail the run
            if feedback is not None:
                feedback.pushWarning(f"Style not applied to {layer.name()}: {e}")


def _r(value, digits=3):
    return (
        None if value is None or not np.isfinite(value) else round(float(value), digits)
    )


class ChannelBathymetryAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    CENTRELINES = "CENTRELINES"
    WATER_POLYGON = "WATER_POLYGON"
    MAX_DEPTH = "MAX_DEPTH"
    MIN_RATIO = "MIN_RATIO"
    FALLBACK = "FALLBACK"
    SCOUR = "SCOUR"
    SPACING = "SPACING"
    SEARCH = "SEARCH"
    FLAT_TOL = "FLAT_TOL"
    MAX_WATER_SLOPE = "MAX_WATER_SLOPE"
    BREAK_SLOPE = "BREAK_SLOPE"
    BAND = "BAND"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return ChannelBathymetryAlgorithm()

    def name(self):
        return "channel_bathymetry"

    def displayName(self):
        return "DEM Channel Bathymetry"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        files = "\n".join(
            f"  {spec.label}: <project number>_"
            f"{clean_name_part(spec.description)}{EXTENSIONS[spec.kind]}"
            for spec in OUTPUT_SPECS
        )
        fields = "\n".join(f"  {n}: {d}" for n, _, d in STATION_FIELDS)
        return (
            f"DEM Channel Bathymetry (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tLidar and photogrammetry DEMs show a river's water "
            "surface on the survey day, not its bed. This tool cuts an "
            "estimated channel into the DEM below the water, from the slopes "
            "of the banks, for hydraulic models such as HEC-RAS. No measured "
            "depths or flows are needed.\n"
            "\n"
            "INPUTS:\tA DEM in a projected CRS in metres and the river "
            "centreline(s), drawn downstream (a line drawn upstream is "
            "reversed). The Stream Network output can be used. Optional: a "
            "water polygon, which replaces the flat-water detection - needed "
            "for photogrammetry DEMs, whose water surface is noisy.\n"
            "\n"
            "METHOD:\t1. Stations along each centreline, each with a section at "
            "right angles. 2. The flat water surface: cells within the "
            "tolerance of the running median of the DEM under the centreline, "
            "nearly flat and connected to it. 3. On each section the water "
            "width W and level, the bank tops and the slope of each bank, "
            "fitted to the bank cells up to the band above the water. 4. Below "
            "the water, two half-parabolas, each meeting its bank at that "
            "bank's slope: depth D = sL sR W / (2 (sL + sR)), deepest point "
            "toward the steeper bank. 5. Caps: W / D at least the minimum "
            "ratio, and D at most the maximum depth when given. Low and high "
            "beds from the spread of the bank slopes along the reach.\n"
            "\n"
            "RISK:\tThe bank above the water stands in for the bed below it. "
            "Banks steeper than the bed (cut, vegetated or protected banks) "
            "give too deep a channel - use the maximum depth or the ratio cap, "
            "and check the stations flagged 'capped'.\n"
            "\n"
            "OUTPUTS:\tTick the outputs wanted and choose a folder for each "
            "kind used. The bed DEM is a copy of the whole DEM with the channel "
            "cut in, ready for a HEC-RAS terrain.\n" + files + "\n"
            "\n"
            "STATION FIELDS:\n" + fields
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DEM, "DEM showing the water surface (projected, metres)"
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.CENTRELINES,
                "River centreline(s), drawn downstream",
                types=[QgsProcessing.SourceType.TypeVectorLine],
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.WATER_POLYGON,
                "Water polygon (optional; replaces the flat-water detection)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_DEPTH,
                "Maximum depth below the water (m, optional)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_RATIO,
                "Minimum width-to-depth ratio W/D (caps the depth)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=1.0,
                defaultValue=5.0,
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
                self.PROJECT_NAME, "Project / site name (report, optional)",
                optional=True,
            )
        )  # fmt: skip
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
        d = Settings()
        num = QgsProcessingParameterNumber.Type.Double
        for param in (
            QgsProcessingParameterString(
                self.FALLBACK,
                "Fallback width-to-depth ratios, shallow, central, deep (only "
                "where no bank slope is found on a centreline, e.g. 15, 10, 7)",
                optional=True,
            ),
            QgsProcessingParameterNumber(
                self.SCOUR,
                "Bend scour factor A (where one bank slope is missing; 0 = off)",
                type=num, minValue=0.0, defaultValue=d.scour,
            ),
            QgsProcessingParameterNumber(
                self.SPACING, "Station spacing (m)", type=num, minValue=0.1,
                defaultValue=d.spacing,
            ),
            QgsProcessingParameterNumber(
                self.SEARCH, "Search half-width of each section (m)", type=num,
                minValue=1.0, defaultValue=d.search,
            ),
            QgsProcessingParameterNumber(
                self.FLAT_TOL, "Flat-water tolerance (m)", type=num, minValue=0.0,
                defaultValue=d.flat_tol,
            ),
            QgsProcessingParameterNumber(
                self.MAX_WATER_SLOPE, "Maximum slope of water cells (m/m)", type=num,
                minValue=0.0, defaultValue=d.max_water_slope,
            ),
            QgsProcessingParameterNumber(
                self.BREAK_SLOPE, "Bank-top break slope (m/m)", type=num,
                minValue=0.0, defaultValue=d.break_slope,
            ),
            QgsProcessingParameterNumber(
                self.BAND, "Bank-slope band above the water (m)", type=num,
                minValue=0.01, defaultValue=d.band,
            ),
        ):  # fmt: skip
            param.setFlags(param.flags() | Qgis.ProcessingParameterFlag.Advanced)
            self.addParameter(param)

    # ------------------------------------------------------------------

    @staticmethod
    def _transform(source, crs, context):
        if source is None or source.sourceCrs() == crs:
            return None
        return QgsCoordinateTransform(
            source.sourceCrs(), crs, context.transformContext()
        )

    def _read_lines(self, source, crs, context):
        """Every part of every feature as an (n, 2) array."""
        transform = self._transform(source, crs, context)
        lines = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            geom.convertToStraightSegment()
            parts = (
                geom.asMultiPolyline() if geom.isMultipart() else [geom.asPolyline()]
            )
            for part in parts:
                if len(part) >= 2:
                    lines.append(np.array([(p.x(), p.y()) for p in part]))
        return lines

    def _read_polygons(self, source, crs, context):
        if source is None:
            return None
        transform = self._transform(source, crs, context)
        polys = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            geom.convertToStraightSegment()
            for poly in (
                geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]
            ):
                polys.append([[(p.x(), p.y()) for p in ring] for ring in poly if ring])
        return polys or None

    @staticmethod
    def _fallback(text):
        if not text or not text.strip():
            return None
        try:
            vals = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
        except ValueError as e:
            raise QgsProcessingException(
                "Fallback ratios: give three numbers, e.g. 15, 10, 7."
            ) from e
        if len(vals) != 3 or min(vals) <= 0:
            raise QgsProcessingException(
                "Fallback ratios: give three positive numbers (shallow, central, "
                "deep), e.g. 15, 10, 7."
            )
        return tuple(sorted(vals, reverse=True))

    @staticmethod
    def _water_layer(result, crs):
        rows = []
        polys = group_polygons(trace_rings(result.water))
        wse = result.wse
        for i, poly in enumerate(polys, start=1):
            rings = [
                [
                    QgsPointXY(x, y)
                    for x, y in ring_to_map(simplify_ring(r), result.window_gt)
                ]
                for r in poly
            ]
            geom = QgsGeometry.fromPolygonXY(rings)
            if not geom.isGeosValid():
                geom = geom.makeValid()
            geom.convertToMultiType()
            rows.append((geom, {"id": i, "area_m2": round(geom.area(), 2)}))
        # levels per polygon: water cells grouped by connected label
        from scipy import ndimage

        labels, n = ndimage.label(result.water, structure=np.ones((3, 3), bool))
        if rows and n:
            lows = ndimage.minimum(
                np.nan_to_num(wse, nan=np.inf), labels, range(1, n + 1)
            )
            highs = ndimage.maximum(
                np.nan_to_num(wse, nan=-np.inf), labels, range(1, n + 1)
            )
            gt = result.window_gt
            for geom, values in rows:
                p = geom.pointOnSurface().asPoint()
                r = int((p.y() - gt[3]) / gt[5])
                c = int((p.x() - gt[0]) / gt[1])
                if (
                    0 <= r < labels.shape[0]
                    and 0 <= c < labels.shape[1]
                    and labels[r, c]
                ):
                    k = labels[r, c] - 1
                    values["wse_min"] = _r(lows[k])
                    values["wse_max"] = _r(highs[k])
        return memory_layer(
            "Channel water surface", WATER_FIELDS, Qgis.WkbType.MultiPolygon, crs, rows
        )

    @staticmethod
    def _station_layer(result, crs):
        rows = []
        for st in result.stations:
            dep = st.depth or {}
            dc = dep.get("central", np.nan)
            rows.append((
                QgsGeometry.fromPointXY(QgsPointXY(st.x, st.y)),
                {
                    "line": st.line + 1, "chain_m": _r(st.chainage, 2),
                    "wse_m": _r(st.wse), "width_m": _r(st.width, 2),
                    "s_left": _r(st.sl, 4), "s_right": _r(st.sr, 4),
                    "d_banks": _r(st.depth_banks), "d_low": _r(dep.get("low", np.nan)),
                    "d_central": _r(dc), "d_high": _r(dep.get("high", np.nan)),
                    "bed_m": _r(st.wse - dc),
                    "thalweg": _r(st.split, 3) if dep else None,
                    "h_left": _r(st.height_left), "h_right": _r(st.height_right),
                    "basis": BASIS_TEXT[st.basis][:254], "cap": CAP_TEXT[st.cap],
                    "flags": "; ".join(dict.fromkeys(st.flags))[:254],
                },
            ))  # fmt: skip
        return memory_layer(
            "Channel stations", STATION_FIELDS, Qgis.WkbType.Point, crs, rows
        )

    @staticmethod
    def _section_layer(result, crs):
        rows = []
        for st in result.stations:
            if not st.width > 0:
                continue
            left = (
                st.top_left
                if np.isfinite(st.top_left)
                else st.edge_left + 0.25 * st.width
            )
            right = (
                st.top_right
                if np.isfinite(st.top_right)
                else st.edge_right - 0.25 * st.width
            )
            # cut lines run left to right looking downstream (HEC-RAS convention)
            pts = [
                QgsPointXY(st.x + o * st.nx, st.y + o * st.ny) for o in (left, right)
            ]
            geom = QgsGeometry.fromPolylineXY(pts)
            rows.append((geom, {
                "line": st.line + 1, "chain_m": _r(st.chainage, 2), "wse_m": _r(st.wse),
                "d_central": _r((st.depth or {}).get("central", np.nan)),
                "length_m": _r(geom.length(), 2),
            }))  # fmt: skip
        return memory_layer(
            "Channel sections", SECTION_FIELDS, Qgis.WkbType.LineString, crs, rows
        )

    # ------------------------------------------------------------------

    def processAlgorithm(self, parameters, context, feedback):
        self._post_processors = []
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        if dem_layer is None:
            raise QgsProcessingException("A DEM is required.")
        crs = dem_layer.crs()
        if crs.isGeographic():
            raise QgsProcessingException(
                "The DEM is in a geographic CRS (degrees). Reproject it to a "
                "projected CRS in metres first."
            )
        line_source = self.parameterAsSource(parameters, self.CENTRELINES, context)
        if line_source is None:
            raise QgsProcessingException("A centreline layer is required.")
        poly_source = self.parameterAsSource(parameters, self.WATER_POLYGON, context)
        max_depth = None
        if parameters.get(self.MAX_DEPTH) not in (None, ""):
            max_depth = self.parameterAsDouble(parameters, self.MAX_DEPTH, context)
            if max_depth <= 0:
                max_depth = None
        settings = Settings(
            spacing=self.parameterAsDouble(parameters, self.SPACING, context),
            search=self.parameterAsDouble(parameters, self.SEARCH, context),
            flat_tol=self.parameterAsDouble(parameters, self.FLAT_TOL, context),
            max_water_slope=self.parameterAsDouble(
                parameters, self.MAX_WATER_SLOPE, context
            ),
            break_slope=self.parameterAsDouble(parameters, self.BREAK_SLOPE, context),
            band=self.parameterAsDouble(parameters, self.BAND, context),
            min_ratio=self.parameterAsDouble(parameters, self.MIN_RATIO, context),
            max_depth=max_depth,
            fallback_ratios=self._fallback(
                self.parameterAsString(parameters, self.FALLBACK, context)
            ),
            scour=self.parameterAsDouble(parameters, self.SCOUR, context),
        )  # fmt: skip
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
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        folders = {}
        for kind in {spec.kind for spec in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {spec.key: spec.path(folders, project_number) for spec in selected}
        want = set(paths)

        try:
            shape, gt, projection, nodata = raster_info(dem_layer.source())
        except GridError as e:
            raise QgsProcessingException(str(e)) from e
        lines = self._read_lines(line_source, crs, context)
        if not lines:
            raise QgsProcessingException("The centreline layer has no lines.")
        cell = 0.5 * (abs(gt[1]) + abs(gt[5]))
        pad = 3 + math.ceil(settings.search / cell)
        window = polygon_window([[line.tolist() for line in lines]], gt, shape, pad=pad)
        if window is None:
            raise QgsProcessingException("The centrelines do not overlap the DEM.")
        n_cells = window[2] * window[3]
        feedback.pushInfo(
            f"{len(lines)} centreline part(s); DEM window {window[3]} x "
            f"{window[2]} cells"
        )
        if n_cells > 60_000_000:
            feedback.pushWarning(
                f"The window holds {n_cells} cells, which needs a lot of memory. "
                "Run the tool on shorter reaches if it fails."
            )
        window_gt = window_geotransform(gt, window[0], window[1])
        dem = read_window(dem_layer.source(), window).astype(float)
        user_mask = None
        polys = self._read_polygons(poly_source, crs, context)
        if polys is not None:
            user_mask = (
                coverage_fraction(polys, window_gt, dem.shape, supersample=1) >= 0.5
            )
        feedback.setProgress(10)
        feedback.setProgressText("Estimating the channel")
        try:
            result = run_channel_bathymetry(
                dem, nodata, window_gt, lines, settings, user_mask,
                is_canceled=feedback.isCanceled,
            )  # fmt: skip
        except ChannelError as e:
            if feedback.isCanceled():
                return {}
            raise QgsProcessingException(str(e)) from e
        for message in result.warnings:
            feedback.pushWarning(message)
        sm = result.summary()
        feedback.pushInfo(
            f"{sm['stations_with_water']} of {sm['stations']} stations found water; "
            f"volume below the water {sm['volume_central_m3']:.0f} m3 (low "
            f"{sm['volume_low_m3']:.0f}, high {sm['volume_high_m3']:.0f}); maximum "
            f"depth {sm['max_depth_central_m']:.2f} m"
        )
        n_cap = sm["capped_ratio"] + sm["capped_max"]
        if n_cap:
            feedback.pushWarning(
                f"{n_cap} station(s) had the depth limited by a cap "
                f"({sm['capped_ratio']} by the W/D ratio, {sm['capped_max']} by the "
                "maximum depth): the bank slopes there point to a deeper channel."
            )
        feedback.setProgress(60)

        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        written = []

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))

        def failed(key, path, error):
            feedback.reportError(
                f"{SPEC_BY_KEY[key].label}: {Path(path).name} could not be written "
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

        feedback.setProgressText("Writing rasters")
        for key, sc in (("OUTPUT_BED", "central"), ("OUTPUT_BED_LOW", "low"),
                        ("OUTPUT_BED_HIGH", "high")):  # fmt: skip
            if key not in want:
                continue
            try:
                clear(paths[key])
                write_bed_dem(dem_layer.source(), paths[key], window, result.beds[sc])
            except Exception as e:
                failed(key, paths[key], e)
                continue
            load_on_completion(context, paths[key], Path(paths[key]).stem, None,
                               self._post_processors)  # fmt: skip
            done(key, paths[key])
        if "OUTPUT_DEPTH" in want:
            path = paths["OUTPUT_DEPTH"]
            depth = result.depth["central"]
            ok = np.isfinite(depth)
            try:
                clear(path)
                write_raster(path, np.where(ok, depth, NODATA).astype("float32"),
                             window_gt, projection, NODATA)  # fmt: skip
                rng = (
                    (float(depth[ok].min()), float(depth[ok].max()))
                    if ok.any()
                    else None
                )
                load_on_completion(
                    context, path, Path(path).stem,
                    _StylePostProcessor(STYLE_DEPTH_PATH, raster_range=rng),
                    self._post_processors,
                )  # fmt: skip
                done("OUTPUT_DEPTH", path)
            except Exception as e:
                failed("OUTPUT_DEPTH", path, e)
        feedback.setProgress(75)

        feedback.setProgressText("Writing layers")
        builders = {
            "OUTPUT_WATER": (self._water_layer, "water"),
            "OUTPUT_STATIONS": (self._station_layer, "stations"),
            "OUTPUT_SECTIONS": (self._section_layer, "sections"),
        }
        for key, (build, kind) in builders.items():
            if key not in want:
                continue
            try:
                path = save_shapefile(
                    build(result, crs), paths[key], context.transformContext()
                )
            except Exception as e:
                failed(key, paths[key], e)
                continue
            load_on_completion(context, path, Path(path).stem, _VectorStyler(kind),
                               self._post_processors)  # fmt: skip
            done(key, path)
        if "OUTPUT_CSV" in want:
            try:
                write_station_csv(paths["OUTPUT_CSV"], result)
                done("OUTPUT_CSV", paths["OUTPUT_CSV"])
            except Exception as e:
                failed("OUTPUT_CSV", paths["OUTPUT_CSV"], e)
        feedback.setProgress(85)

        if "OUTPUT_DOCX" in want:
            path = paths["OUTPUT_DOCX"]
            feedback.setProgressText("Writing report")
            ctx = {
                "tool": self.displayName(),
                "version": TOOL_VERSION,
                "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "project_number": project_number or None,
                "project_name": project_name or None,
                "software": f"QGIS {Qgis.version()}",
                "dem": {
                    "path": dem_layer.source(),
                    "crs": f"{crs.authid()} - {crs.description()}",
                    "cell_x": abs(gt[1]),
                    "cell_y": abs(gt[5]),
                },
                "centrelines": (
                    f"{line_source.sourceName()} ({len(lines)} line part(s))"
                ),
                "water_polygon": (
                    None if poly_source is None else poly_source.sourceName()
                ),
                "outputs": written + [("This report", path)],
                "warnings": list(result.warnings),
            }
            dem_nan = np.where(np.isfinite(dem) & (dem != nodata), dem, np.nan)
            try:
                for message in write_report_docx(path, result, ctx, dem_nan) or []:
                    feedback.pushWarning(message)
                done("OUTPUT_DOCX", path)
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs the python-docx package. Install it from "
                    "the OSGeo4W Shell with: python -m pip install python-docx. All "
                    "other outputs were written."
                )
            except PermissionError as e:
                failed("OUTPUT_DOCX", path, e)
            except Exception as e:  # report failure must not lose the results
                feedback.pushWarning(f"The Word report could not be written: {e}")

        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        feedback.setProgress(100)
        return outputs
