"""Processing Toolbox algorithm: Extract: DWS and WR2012 data.

A study area in -> DWS layers (water management areas, drainage regions,
quinary catchments, rivers, dams, Reserve status), DWS monitoring station
locations and, from a local WR2012 copy, per-quaternary water-balance data
with an area-weighted summary and a Word report out.

Thin wrapper - the logic is in core.py (zero QGIS dependency, tested in
tests/test_dws_wr2012_core.py), the report in report.py and the QGIS
layer handling in qgis_layers.py.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime
from pathlib import Path

from qgis.analysis import QgsZonalStatistics
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsCoordinateTransformContext,
    QgsDistanceArea,
    QgsFeatureRequest,
    QgsGeometry,
    QgsPointXY,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsRasterLayer,
    QgsRectangle,
    QgsVectorLayer,
)
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import CSV, SHAPEFILE, WORD, OutputSpec
from mayim_tools._common.qgis_maps import (
    legend_entries,
    map_crs_and_extent,
    osm_has_tiles,
    osm_layer,
    render_map,
    utm_crs_for,
)
from mayim_tools._common.report_maps import (
    OSM_ATTRIBUTION,
    PLAN_DPI,
    PLAN_PLOT_WIDTH_CM,
    PlanMap,
)

from . import core
from .core import (
    DWS_LAYERS,
    GRA2_GRID_DIR,
    GRA2_GRIDS,
    STATION_FIELDS,
    STATION_TYPES,
    TOOL_VERSION,
    WMA_LAYERS,
    WMA_OPTIONS,
    WMA_VERSIONS,
    WR2012_FILES,
    WR_FIELDS,
    ExtractError,
    RestError,
)
from .qgis_layers import (
    WGS84,
    add_features,
    load_on_completion,
    make_fetch,
    memory_layer,
    present_station_types,
    save_shapefile,
    styled_clone,
)
from .report import ReportData, write_report_docx

# selectable outputs (key, dialog label, file-name description, folder kind,
# ticked by default)
LAYER_SPECS = [
    OutputSpec("OUT_WMA", "Water management areas", "DWS WMA", SHAPEFILE)
] + [
    OutputSpec(layer.key, layer.label, layer.description, SHAPEFILE, layer.default)
    for layer in DWS_LAYERS
]
OUTPUT_SPECS = LAYER_SPECS + [
    OutputSpec("OUT_STATIONS", "DWS monitoring stations", "DWS Stations", SHAPEFILE),
    OutputSpec(
        "OUT_WR_QUAT",
        "WR2012 quaternaries with water-balance data",
        "WR2012 Quaternaries",
        SHAPEFILE,
    ),
    OutputSpec(
        "OUT_WR_CSV", "WR2012 quaternary data (CSV)", "WR2012 Quaternary Data", CSV
    ),
    OutputSpec(
        "OUT_SUMMARY", "Water-balance summary (CSV)", "Water Balance Summary", CSV
    ),
    OutputSpec("OUT_STATIONS_CSV", "Station list (CSV)", "DWS Stations", CSV),
    OutputSpec("OUT_DOCX", "Report (Word)", "DWS WR2012 Data Report", WORD),
]
SPEC_BY_KEY = {s.key: s for s in OUTPUT_SPECS}
FOLDER_PARAMS = {
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}
DEFAULT_STATION_TYPES = [0, 1, 2, 3]  # river, reservoir, rainfall, evaporation
REPORT_LAYERS = ("OUT_WMA", "OUT_TERTIARY", "OUT_QUATERNARY")
MAP_STATION_LABELS = 25  # station codes on the report map up to this many


def _ellipsoid():
    d = QgsDistanceArea()
    d.setSourceCrs(WGS84, QgsCoordinateTransformContext())
    d.setEllipsoid("WGS84")
    return d


class DwsWr2012Algorithm(QgsProcessingAlgorithm):
    STUDY_AREA = "STUDY_AREA"
    BUFFER = "BUFFER"
    WMA_VERSION = "WMA_VERSION"
    WR2012_FOLDER = "WR2012_FOLDER"
    STATION_TYPES = "STATION_TYPES"
    STATION_BUFFER = "STATION_BUFFER"
    OPEN_ONLY = "OPEN_ONLY"
    OUTPUT_CRS = "OUTPUT_CRS"
    REPORT_BASEMAP = "REPORT_BASEMAP"
    PROJECT_NUMBER = "PROJECT_NUMBER"

    def __init__(self):
        super().__init__()
        self._post_processors = []

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return DwsWr2012Algorithm()

    def name(self):
        return "dws_wr2012_extract"

    def displayName(self):
        return "Extract: DWS and WR2012 data"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        return (
            f"Extract: DWS and WR2012 data (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tCollects the Department of Water and Sanitation (DWS) "
            "and WR2012 data needed for a water balance study of a site or "
            "catchment in South Africa, Lesotho or Eswatini.\n"
            "\n"
            "DWS LAYERS (internet):\tWater management areas (2023 by default, "
            "or 2012 / 2004), primary to quaternary drainage regions, quinary "
            "catchments, 1:50 000 rivers and dams, surface-water and "
            "groundwater Reserve status and EWR sites, from the DWS ArcGIS "
            "services (gia.dws.gov.za; DWS ArcGIS Online as fall-back). "
            "Catchments are kept whole; rivers, dams and sites are clipped "
            "to the study area.\n"
            "\n"
            "STATIONS:\tLocations of DWS river, reservoir, rainfall, "
            "evaporation, groundwater and tidal stations from the DWS "
            f"verified station lists ({core.STATION_LIST_DATE}), bundled "
            "with the tool, within the search distance of the study area. "
            "Station data are not downloaded.\n"
            "\n"
            "WR2012 (optional):\tPoint the tool at a local WR2012 'Coverages' "
            "folder to add, per quaternary intersecting the study area: MAP, "
            "naturalised MAR (mm and million m³), Pitman parameters, rainfall "
            "and evaporation zones, A-pan MAE, EIS and PES classes, TDS, "
            "sediment yield and GRA II recharge, baseflow and exploitation "
            "values, plus an area-weighted water-balance summary. Leave it "
            "empty to skip.\n"
            "\n"
            "OUTPUTS:\tTick the outputs wanted; they are written to the "
            "shapefile, CSV and Word folders as <project number>_<description> "
            "and the layers are loaded into the map."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.STUDY_AREA,
                "Study area (polygon, or points / lines with a buffer)",
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUFFER,
                "Buffer around the study area (m)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WMA_VERSION,
                "Water management area boundaries",
                options=WMA_OPTIONS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.WR2012_FOLDER,
                "WR2012 'Coverages' folder (optional - skipped when empty)",
                behavior=Qgis.ProcessingFileParameterBehavior.Folder,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STATION_TYPES,
                "DWS station types",
                options=[label for _, label in STATION_TYPES],
                allowMultiple=True,
                defaultValue=DEFAULT_STATION_TYPES,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.STATION_BUFFER,
                "Station search distance from the study area (km)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=10.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.OPEN_ONLY,
                "Open stations only (record to 2025 or later)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.OUTPUT_CRS, "CRS of the output layers", defaultValue="ProjectCrs"
            )
        )
        basemap = QgsProcessingParameterBoolean(
            self.REPORT_BASEMAP,
            "Word report: OpenStreetMap background on the layout map (needs internet)",
            defaultValue=True,
        )
        basemap.setFlags(basemap.flags() | Qgis.ProcessingParameterFlag.Advanced)
        self.addParameter(basemap)
        self.addParameter(
            QgsProcessingParameterString(
                self.PROJECT_NUMBER,
                "Project number (file name prefix, optional)",
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
    # study area
    # ------------------------------------------------------------------

    def _study_area(self, source, buffer_m, context):
        """(study area in WGS 84, its UTM CRS, the WGS84 -> UTM transform)."""
        to_wgs = QgsCoordinateTransform(
            source.sourceCrs(), WGS84, context.transformContext()
        )
        geoms = []
        for f in source.getFeatures():
            g = QgsGeometry(f.geometry())
            if g.isEmpty():
                continue
            g.transform(to_wgs)
            geoms.append(g)
        if not geoms:
            raise QgsProcessingException("The study area layer has no features.")
        union = QgsGeometry.unaryUnion(geoms)
        centre = union.boundingBox().center()
        utm = utm_crs_for(centre.x(), centre.y())
        fwd = QgsCoordinateTransform(WGS84, utm, context.transformContext())
        back = QgsCoordinateTransform(utm, WGS84, context.transformContext())
        if buffer_m > 0:
            g = QgsGeometry(union)
            g.transform(fwd)
            g = g.buffer(buffer_m, 24)
            g.transform(back)
            union = g
        if not union.isGeosValid():
            union = union.makeValid()
        return union, utm, fwd, back

    @staticmethod
    def _buffered(geom, metres, fwd, back):
        g = QgsGeometry(geom)
        g.transform(fwd)
        g = g.buffer(metres, 24)
        g.transform(back)
        return g

    # ------------------------------------------------------------------
    # DWS layers
    # ------------------------------------------------------------------

    def _download(self, layer_def, bbox, aoi, clip_geom, fetch, feedback):
        """(memory layer or None, source text, warnings). A source that
        fails is only reported (as information) when another one answers."""
        warnings, tried = [], []
        data = None
        for url in layer_def.urls:
            if feedback.isCanceled():
                return None, "cancelled", warnings
            try:
                feedback.pushInfo(f"{layer_def.label}: {url}")
                data = core.fetch_layer(
                    url,
                    bbox,
                    fetch,
                    is_cancelled=feedback.isCanceled,
                    max_offset=layer_def.max_offset,
                )
                break
            except RestError as e:
                tried.append(f"{layer_def.label}: {url} did not answer ({e}).")
                data = None
        if data is None:
            return None, "not available", tried
        for msg in tried:
            feedback.pushInfo(msg + " The next source was used.")
        warnings.extend(f"{layer_def.label}: {w}" for w in data.warnings)
        target = clip_geom if layer_def.clip else aoi
        engine = QgsGeometry.createGeometryEngine(target.constGet())
        engine.prepareGeometry()
        items = []
        for wkt, attrs in data.features:
            g = QgsGeometry.fromWkt(wkt)
            if g.isEmpty():
                continue
            if not g.isGeosValid():
                g = g.makeValid()
            if not engine.intersects(g.constGet()):
                continue
            if (
                layer_def.clip
                and layer_def.geometry != "point"
                and not engine.contains(g.constGet())
            ):
                g = g.intersection(clip_geom)
                if g.isEmpty():
                    continue
                g = _of_type(g, layer_def.geometry)
                if g is None:
                    continue
            items.append((g, attrs))
        layer = memory_layer(layer_def.description, layer_def.geometry, data.fields)
        add_features(layer, data.fields, items, layer_def.geometry)
        host = data.url.split("/")[2]
        return layer, f"{host} ({data.url.split('/rest/services/')[-1]})", warnings

    # ------------------------------------------------------------------
    # WR2012
    # ------------------------------------------------------------------

    def _wr2012(self, root, aoi, tc, feedback, messages):
        """(rows, quaternary memory layer) for the WR2012 quaternaries that
        intersect the study area."""
        tables = core.load_wr2012_tables(root, messages)
        quat_path = core.find_path(root, WR2012_FILES["quaternaries"])
        if quat_path is None:
            raise ExtractError(
                f"The WR2012 quaternaries ({WR2012_FILES['quaternaries']}) were "
                f"not found in {root}."
            )
        src = QgsVectorLayer(str(quat_path), "dquat", "ogr")
        if not src.isValid():
            raise ExtractError(f"{quat_path} could not be opened.")
        to_wgs = QgsCoordinateTransform(
            src.crs() if src.crs().isValid() else WGS84, WGS84, tc
        )
        da = _ellipsoid()
        request = QgsFeatureRequest().setFilterRect(
            to_wgs.transformBoundingBox(
                aoi.boundingBox(), Qgis.TransformDirection.Reverse
            )
        )
        qfield = next(
            (f.name() for f in src.fields() if f.name().upper() == "QUATERNARY"), None
        )
        if qfield is None:
            raise ExtractError("The WR2012 quaternary layer has no QUATERNARY field.")
        rows, geoms = [], {}
        for f in src.getFeatures(request):
            g = QgsGeometry(f.geometry())
            g.transform(to_wgs)
            if not g.isGeosValid():
                g = g.makeValid()
            if not g.intersects(aoi):
                continue
            q = core.norm_quat(f[qfield])
            if not q:
                continue
            inside = g.intersection(aoi)
            row = {
                "QUAT": q,
                "AREA_KM2": da.measureArea(g) / 1e6,
                "AOI_KM2": (
                    da.measureArea(inside) / 1e6 if not inside.isEmpty() else 0.0
                ),
            }
            row.update(tables.get(q, {}))
            if q in geoms:  # multi-part quaternary stored as two features
                prev = next(r for r in rows if r["QUAT"] == q)
                prev["AREA_KM2"] += row["AREA_KM2"]
                prev["AOI_KM2"] += row["AOI_KM2"]
                geoms[q] = geoms[q].combine(g)
                continue
            rows.append(row)
            geoms[q] = g
        if not rows:
            messages.append("No WR2012 quaternary intersects the study area.")
            return [], None
        feedback.pushInfo(f"WR2012: {len(rows)} quaternary catchment(s)")

        # zones: the zone covering the largest part of each quaternary
        for key, out_name, fields in (
            ("rain_zones", "RAINZONE", ("RAINZ",)),
            ("evap_zones", "EVAPZONE", ("EVAPZ",)),
            ("mae", "MAE_CLASS", ("EVAPORATIO",)),
        ):
            path = core.find_path(root, WR2012_FILES[key])
            if path is None:
                messages.append(f"WR2012 layer not found: {WR2012_FILES[key]}")
                continue
            zl = QgsVectorLayer(str(path), key, "ogr")
            fname = next((n for n in fields if zl.fields().indexOf(n) >= 0), None)
            if not zl.isValid() or fname is None:
                messages.append(
                    f"WR2012 layer {WR2012_FILES[key]} could not be opened (the "
                    "file appears to be damaged) - this zone is left out."
                )
                continue
            zt = QgsCoordinateTransform(
                zl.crs() if zl.crs().isValid() else WGS84, WGS84, tc
            )
            for row in rows:
                g = geoms[row["QUAT"]]
                req = QgsFeatureRequest().setFilterRect(
                    zt.transformBoundingBox(
                        g.boundingBox(), Qgis.TransformDirection.Reverse
                    )
                )
                pieces = []
                for zf in zl.getFeatures(req):
                    zg = QgsGeometry(zf.geometry())
                    zg.transform(zt)
                    if not zg.isGeosValid():
                        zg = zg.makeValid()
                    if zg.intersects(g):
                        pieces.append((zf[fname], zg.intersection(g).area()))
                value = core.dominant(pieces)
                row[out_name] = None if value is None else str(value).strip()

        # GRA II grids: mean of the cells inside each quaternary
        grid_root = core.find_path(root, GRA2_GRID_DIR)
        if grid_root is None:
            messages.append(f"GRA II grids not found: {GRA2_GRID_DIR}")
        else:
            stat = _mean_statistic()
            for grid, (out_name, factor) in GRA2_GRIDS.items():
                path = core.find_path(grid_root, grid)
                if path is None or not path.is_dir():
                    messages.append(f"GRA II grid not found: {grid}")
                    continue
                rl = QgsRasterLayer(str(path), grid, "gdal")
                if not rl.isValid():
                    messages.append(f"GRA II grid {grid} could not be opened.")
                    continue
                rt = QgsCoordinateTransform(WGS84, rl.crs(), tc)
                found = 0
                for row in rows:
                    g = QgsGeometry(geoms[row["QUAT"]])
                    g.transform(rt)
                    value = _zonal_mean(rl, g, stat)
                    row[out_name] = None if value is None else value * factor
                    found += value is not None
                if found < len(rows):
                    messages.append(
                        f"GRA II grid {grid}: no values for {len(rows) - found} of "
                        f"{len(rows)} quaternaries - the grid has no data there or "
                        "its blocks cannot be read (damaged file)."
                    )
        rows = [core.finish_row(r) for r in sorted(rows, key=lambda r: r["QUAT"])]

        fields = [
            (f.name, "string" if f.decimals is None else "double") for f in WR_FIELDS
        ]
        layer = memory_layer("WR2012 Quaternaries", "polygon", fields)
        add_features(layer, fields, [(geoms[r["QUAT"]], r) for r in rows], "polygon")
        return rows, layer

    # ------------------------------------------------------------------
    # stations
    # ------------------------------------------------------------------

    def _stations(self, types, open_only, search, aoi, fwd):
        rows = core.load_stations()
        aoi_m = QgsGeometry(aoi)
        aoi_m.transform(fwd)
        box = search.boundingBox()

        def keep(r):
            if not box.contains(QgsPointXY(r["LON"], r["LAT"])):
                return False
            return search.contains(
                QgsGeometry.fromPointXY(QgsPointXY(r["LON"], r["LAT"]))
            )

        sel = core.filter_stations(rows, types, open_only, keep)
        for r in sel:
            p = QgsGeometry.fromPointXY(QgsPointXY(r["LON"], r["LAT"]))
            p.transform(fwd)
            r["DIST_KM"] = aoi_m.distance(p) / 1000.0
        layer = memory_layer("DWS Stations", "point", STATION_FIELDS)
        add_features(
            layer,
            STATION_FIELDS,
            [(QgsGeometry.fromPointXY(QgsPointXY(r["LON"], r["LAT"])), r) for r in sel],
            "point",
        )
        return sel, layer

    # ------------------------------------------------------------------
    # report map
    # ------------------------------------------------------------------

    def _plan_map(self, aoi, layers, context, use_osm, messages):
        """Layout map: study area, stations, rivers, dams, quaternaries,
        WMA over OpenStreetMap; fitted to the intersected quaternaries."""
        order = [  # top first
            ("STUDY_AREA", "Study area"),
            ("OUT_STATIONS", "Station"),
            ("OUT_EWR_SITES", "EWR site"),
            ("OUT_DAMS", "Dam"),
            ("OUT_RIVERS", "River"),
            ("OUT_WR_QUAT", "WR2012 quaternary"),
            ("OUT_QUATERNARY", "Quaternary catchment (DWS)"),
            ("OUT_TERTIARY", "Tertiary drainage region"),
            ("OUT_WMA", "Water management area"),
        ]
        top_down, legend = [], []
        study = memory_layer("Study area", "polygon", [])
        if aoi.type() == Qgis.GeometryType.Polygon:
            add_features(study, [], [(aoi, {})], "polygon")
        for key, title in order:
            layer = study if key == "STUDY_AREA" else layers.get(key)
            if layer is None or layer.featureCount() == 0:
                continue
            clone = styled_clone(layer, key)
            # keep the map readable: one set of quaternary labels, and no
            # station codes when there are many stations
            if (key == "OUT_QUATERNARY" and "OUT_WR_QUAT" in layers) or (
                key == "OUT_STATIONS" and layer.featureCount() > MAP_STATION_LABELS
            ):
                clone.setLabelsEnabled(False)
            top_down.append(clone)
            present = present_station_types(layer) if key == "OUT_STATIONS" else None
            legend += legend_entries(clone, title, present)

        fit = layers.get("OUT_WR_QUAT") or layers.get("OUT_QUATERNARY")
        ext = QgsRectangle(aoi.boundingBox())
        if fit is not None and fit.featureCount():
            ext.combineExtentWith(fit.extent())
        if ext.width() < 0.02 or ext.height() < 0.02:  # about 2 km
            ext.grow(0.01)
        ext.scale(1.08)
        crs, extent = map_crs_and_extent(
            WGS84,
            (ext.xMinimum(), ext.yMinimum(), ext.xMaximum(), ext.yMaximum()),
            context.transformContext(),
        )
        width_px = int(round(PLAN_PLOT_WIDTH_CM / 2.54 * PLAN_DPI))
        aspect = (extent[3] - extent[1]) / max(extent[2] - extent[0], 1e-9)
        height_px = max(int(round(width_px * aspect)), 1)
        attribution = None
        if use_osm:
            osm = osm_layer()
            if osm is not None and osm_has_tiles(
                osm,
                extent,
                crs,
                context.transformContext(),
                width_px,
                height_px,
                PLAN_DPI,
            ):
                top_down.append(osm)
                attribution = OSM_ATTRIBUTION
            else:
                messages.append(
                    "OpenStreetMap background not added: no map tiles were "
                    "received (check the internet connection or proxy)."
                )
        image, rendered, errors = render_map(
            top_down,
            extent,
            crs,
            context.transformContext(),
            width_px,
            height_px,
            PLAN_DPI,
        )
        messages.extend(f"Layout map: {e}" for e in errors)
        return PlanMap(
            image=image, extent=rendered, legend=legend, attribution=attribution
        )

    # ------------------------------------------------------------------

    def processAlgorithm(self, parameters, context, feedback):
        self._post_processors = []
        source = self.parameterAsSource(parameters, self.STUDY_AREA, context)
        if source is None:
            raise QgsProcessingException("No study area layer provided.")
        buffer_m = self.parameterAsDouble(parameters, self.BUFFER, context)
        wma_version = WMA_VERSIONS[
            self.parameterAsEnum(parameters, self.WMA_VERSION, context)
        ]
        wr_root = (
            self.parameterAsFile(parameters, self.WR2012_FOLDER, context) or ""
        ).strip()
        type_idx = self.parameterAsEnums(parameters, self.STATION_TYPES, context)
        types = [STATION_TYPES[i][0] for i in type_idx]
        station_km = self.parameterAsDouble(parameters, self.STATION_BUFFER, context)
        open_only = self.parameterAsBoolean(parameters, self.OPEN_ONLY, context)
        out_crs = self.parameterAsCrs(parameters, self.OUTPUT_CRS, context)
        if not out_crs.isValid():
            out_crs = WGS84
        use_osm = self.parameterAsBoolean(parameters, self.REPORT_BASEMAP, context)
        project_number = self.parameterAsString(
            parameters, self.PROJECT_NUMBER, context
        )

        selected = [
            s
            for s in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, s.key, context)
        ]
        if not selected:
            raise QgsProcessingException("Tick at least one output.")
        want = {s.key for s in selected}
        wr_keys = {"OUT_WR_QUAT", "OUT_WR_CSV", "OUT_SUMMARY"}
        if want & wr_keys and not wr_root:
            feedback.pushWarning(
                "No WR2012 folder given - the WR2012 outputs are skipped."
            )
            want -= wr_keys
            selected = [s for s in selected if s.key in want]
        if wr_root and not os.path.isdir(wr_root):
            raise QgsProcessingException(f"The WR2012 folder does not exist: {wr_root}")
        station_keys = {"OUT_STATIONS", "OUT_STATIONS_CSV"}
        if want & station_keys and not types:
            feedback.pushWarning(
                "No station types ticked - the station outputs are skipped."
            )
            want -= station_keys
            selected = [s for s in selected if s.key in want]
        if not selected:
            raise QgsProcessingException("None of the ticked outputs can be produced.")
        folders = {}
        for kind in {s.kind for s in selected}:
            name, label = FOLDER_PARAMS[kind]
            folder = self.parameterAsFileOutput(parameters, name, context)
            if not folder:
                raise QgsProcessingException(f"Choose the {label.lower()}.")
            os.makedirs(folder, exist_ok=True)
            folders[kind] = folder
        paths = {s.key: s.path(folders, project_number) for s in selected}
        if "OUT_WMA" in paths:  # the file name carries the WMA version
            paths["OUT_WMA"] = OutputSpec(
                "OUT_WMA", "", WMA_LAYERS[wma_version].description, SHAPEFILE
            ).path(folders, project_number)
        report = "OUT_DOCX" in want
        messages: list[str] = []

        # study area ---------------------------------------------------------
        aoi, utm, fwd, back = self._study_area(source, buffer_m, context)
        da = _ellipsoid()
        aoi_km2 = (
            da.measureArea(aoi) / 1e6
            if aoi.type() == Qgis.GeometryType.Polygon
            else 0.0
        )
        bbox_r = aoi.boundingBox()
        if bbox_r.width() == 0 or bbox_r.height() == 0:
            bbox_r.grow(0.0005)
        bbox = (
            bbox_r.xMinimum(),
            bbox_r.yMinimum(),
            bbox_r.xMaximum(),
            bbox_r.yMaximum(),
        )
        clip_geom = aoi if aoi_km2 > 0 else QgsGeometry.fromRect(bbox_r)
        feedback.pushInfo(
            f"Study area: {aoi_km2:.3f} km², bounding box "
            f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} (WGS 84)"
        )
        if not (14.0 < bbox[0] < 34.0 and -36.0 < bbox[1] < -21.0):
            feedback.pushWarning(
                "The study area is outside South Africa, Lesotho and Eswatini - "
                "the DWS and WR2012 data cover those countries only."
            )

        # DWS layers ---------------------------------------------------------
        fetch = make_fetch(feedback)
        defs = [WMA_LAYERS[wma_version]] + DWS_LAYERS
        fetch_keys = {d.key for d in defs if d.key in want}
        if report:
            fetch_keys |= set(REPORT_LAYERS)
        layers, layer_log = {}, []
        clock = [time.monotonic()]

        def lap() -> str:
            now = time.monotonic()
            text = f"{now - clock[0]:.0f} s"
            clock[0] = now
            return text

        steps = len(fetch_keys) + 3
        step = 0
        for d in defs:
            if d.key not in fetch_keys:
                continue
            if feedback.isCanceled():
                return {}
            feedback.setProgressText(f"Downloading {d.label}")
            layer, src, warns = self._download(d, bbox, aoi, clip_geom, fetch, feedback)
            for w in warns:
                feedback.pushWarning(w)
            messages.extend(warns)
            if layer is not None:
                layers[d.key] = layer
                layer_log.append((d.label, layer.featureCount(), src))
                feedback.pushInfo(
                    f"{d.label}: {layer.featureCount()} feature(s) ({lap()})"
                )
            else:
                layer_log.append((d.label, 0, "not available"))
            step += 1
            feedback.setProgress(int(100 * step / steps))

        # stations -----------------------------------------------------------
        stations = []
        if types and (want & station_keys or report):
            feedback.setProgressText("Selecting DWS stations")
            search = self._buffered(aoi, max(station_km, 0.0) * 1000.0, fwd, back)
            stations, layers["OUT_STATIONS"] = self._stations(
                types, open_only, search, aoi, fwd
            )
            feedback.pushInfo(f"DWS stations: {len(stations)} ({lap()})")
        step += 1
        feedback.setProgress(int(100 * step / steps))

        # WR2012 -------------------------------------------------------------
        wr_rows, summary = None, []
        if wr_root and (want & wr_keys or report):
            feedback.setProgressText("Reading WR2012")
            try:
                wr_rows, wr_layer = self._wr2012(
                    wr_root, aoi, context.transformContext(), feedback, messages
                )
            except ExtractError as e:
                raise QgsProcessingException(str(e)) from e
            if wr_layer is not None:
                layers["OUT_WR_QUAT"] = wr_layer
            summary = core.water_balance_summary(wr_rows, aoi_km2)
            feedback.pushInfo(f"WR2012 done ({lap()})")
        step += 1
        feedback.setProgress(int(100 * step / steps))

        # write --------------------------------------------------------------
        outputs = {
            name: folders[kind]
            for kind, (name, _) in FOLDER_PARAMS.items()
            if kind in folders
        }
        written = []

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))

        feedback.setProgressText("Writing outputs")
        for spec in selected:
            if spec.kind != SHAPEFILE:
                continue
            layer = layers.get(spec.key)
            if layer is None:
                continue
            if layer.featureCount() == 0:
                feedback.pushWarning(f"{spec.label}: nothing inside the study area.")
                continue
            try:
                path = save_shapefile(
                    layer, paths[spec.key], out_crs, context.transformContext()
                )
            except RuntimeError as e:
                feedback.reportError(str(e), fatalError=False)
                continue
            load_on_completion(
                context, path, Path(path).stem, spec.key, self._post_processors
            )
            done(spec.key, path)
        if "OUT_WR_CSV" in want and wr_rows is not None:
            done("OUT_WR_CSV", core.write_wr2012_csv(paths["OUT_WR_CSV"], wr_rows))
        if "OUT_SUMMARY" in want and wr_rows is not None:
            done("OUT_SUMMARY", core.write_summary_csv(paths["OUT_SUMMARY"], summary))
        if "OUT_STATIONS_CSV" in want:
            done(
                "OUT_STATIONS_CSV",
                core.write_stations_csv(paths["OUT_STATIONS_CSV"], stations),
            )

        feedback.pushInfo(f"Layers and tables written ({lap()})")
        if report:
            feedback.setProgressText("Writing the Word report")
            plan = None
            try:
                plan = self._plan_map(aoi, layers, context, use_osm, messages)
            except Exception as e:  # the map must never stop the report
                messages.append(f"Layout map not drawn: {e}")
            wmas = []
            wl = layers.get("OUT_WMA")
            if wl is not None:
                id_f = _field(wl, ("WMA_ID", "WMA_NO", "WMA"))
                nm_f = _field(wl, ("WMA_NAME", "WMA_Name", "MapName", "NAME"))
                for f in wl.getFeatures():
                    name = str(f[nm_f]).replace("_", "-") if nm_f else ""
                    wmas.append((f[id_f] if id_f else "", name))
                wmas.sort(key=lambda t: str(t[0]))
            drainage = []
            for key, label, names in (
                ("OUT_PRIMARY", "Primary", ("PRIMARY", "PRIMARY_", "PRIMARY_DR")),
                ("OUT_SECONDARY", "Secondary", ("SECONDARY", "SECONDARY_")),
                ("OUT_TERTIARY", "Tertiary", ("TERTIARY", "TERTIARY_")),
                ("OUT_QUATERNARY", "Quaternary", ("QUATERNARY", "QUAT")),
            ):
                lyr = layers.get(key)
                fld = _field(lyr, names) if lyr is not None else None
                if fld:
                    codes = sorted(
                        {str(f[fld]).strip() for f in lyr.getFeatures()} - {"", "NULL"}
                    )
                    if codes:
                        drainage.append((label, ", ".join(codes)))
            data = ReportData(
                project_number=project_number,
                run_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
                study_area=source.sourceName() if hasattr(source, "sourceName") else "",
                study_area_km2=aoi_km2,
                buffer_m=buffer_m,
                bbox=bbox,
                wma_version=wma_version,
                inputs=[
                    (
                        "Study area",
                        f"{source.sourceName()} ({source.featureCount()} feature(s), "
                        f"{source.sourceCrs().authid()})",
                    ),
                    ("WR2012 folder", wr_root or "not given - WR2012 part skipped"),
                    ("DWS station table", f"{core.STATION_TABLE} (bundled)"),
                    ("DWS services", "gia.dws.gov.za; DWS ArcGIS Online (fall-back)"),
                ],
                settings=[
                    ("Buffer around the study area", f"{core.fmt(buffer_m)} m"),
                    (
                        "Water management areas",
                        WMA_OPTIONS[WMA_VERSIONS.index(wma_version)],
                    ),
                    (
                        "Station types",
                        ", ".join(STATION_TYPES[i][1] for i in type_idx) or "-",
                    ),
                    ("Station search distance", f"{core.fmt(station_km)} km"),
                    ("Open stations only", "Yes" if open_only else "No"),
                    ("Output CRS", out_crs.authid() or out_crs.description()),
                    ("Project number", project_number or "-"),
                ],
                layers=[r for r in layer_log],
                wmas=wmas,
                drainage=drainage,
                wr_rows=wr_rows,
                summary=summary,
                stations=stations,
                station_counts=core.station_counts(stations),
                station_types=types,
                station_buffer_km=station_km,
                open_only=open_only,
                outputs=written + [(SPEC_BY_KEY["OUT_DOCX"].label, paths["OUT_DOCX"])],
                warnings=list(dict.fromkeys(messages)),
                plan_map=plan,
            )
            try:
                for m in write_report_docx(paths["OUT_DOCX"], data):
                    feedback.pushWarning(m)
                done("OUT_DOCX", paths["OUT_DOCX"])
                feedback.pushInfo(f"Word report written ({lap()})")
            except PermissionError:
                feedback.reportError(
                    f"{Path(paths['OUT_DOCX']).name} could not be written - it is "
                    "probably open in Word. Close it and run again.",
                    fatalError=False,
                )
        for m in dict.fromkeys(messages):
            feedback.pushWarning(m)
        feedback.setProgress(100)
        return outputs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _field(layer, names) -> str | None:
    if layer is None:
        return None
    lower = {f.name().lower(): f.name() for f in layer.fields()}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _of_type(geom: QgsGeometry, kind: str):
    """Keeps only the parts of an intersection result of the wanted type."""
    want = {"polygon": Qgis.GeometryType.Polygon, "line": Qgis.GeometryType.Line}[kind]
    if geom.type() == want:
        return geom
    parts = [QgsGeometry(p.clone()) for p in geom.constParts()]
    parts = [p for p in parts if p.type() == want]
    if not parts:
        return None
    return QgsGeometry.collectGeometry(parts)


def _mean_statistic():
    enum = getattr(Qgis, "ZonalStatistic", None)
    if enum is not None:
        return enum.Mean
    return QgsZonalStatistics.Statistic.Mean


def _zonal_mean(raster, geom, stat):
    """Mean of the raster cells inside the polygon; the cell value at the
    centroid when the polygon holds no cell centre."""
    provider = raster.dataProvider()
    value = None
    try:
        result = QgsZonalStatistics.calculateStatistics(
            provider,
            geom,
            raster.rasterUnitsPerPixelX(),
            raster.rasterUnitsPerPixelY(),
            1,
            stat,
        )
        if result:
            value = next(iter(result.values()))
    except Exception:  # noqa: BLE001 - fall back to the centroid sample
        value = None
    try:
        value = None if value is None else float(value)
    except (TypeError, ValueError):
        value = None
    if value is None or math.isnan(value):
        c = geom.centroid().asPoint()
        sample, ok = provider.sample(QgsPointXY(c.x(), c.y()), 1)
        value = float(sample) if ok and not math.isnan(sample) else None
    return value
