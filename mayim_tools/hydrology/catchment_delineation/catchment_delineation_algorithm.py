"""Processing Toolbox algorithm: D8 pointer + DEM + pour point(s)/line(s)
in -> catchment polygon(s), catchment DEM, longest flow path(s),
snapped outlet(s) and a parameters report out.

Thin wrapper - all real logic lives in core.py (zero QGIS dependency,
independently testable; see tests/test_catchment_delineation_core.py).
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QIcon

from mayim_tools._common.output_files import (
    CSV,
    RASTER,
    SHAPEFILE,
    WORD,
    OutputSpec,
)
from mayim_tools.hydrology._common.contours import minor_major_contours
from mayim_tools.hydrology.stream_network.core import (
    StreamNetworkError,
    extract_stream_network,
)
from mayim_tools.hydrology.stream_network.export import write_sectioned_csv
from mayim_tools.hydrology.stream_network.report import (
    write_report_docx as write_network_report_docx,
)
from mayim_tools.hydrology.stream_network.stream_network_algorithm import (
    add_network_parameters,
    read_network_settings,
    write_main_stems,
    write_nodes,
    write_reaches,
)

from .core import (
    CatchmentError,
    Outlet,
    check_same_grid,
    delineate_catchments,
    read_raster,
    rings_to_multipolygon_wkt,
    write_raster,
)
from .export import write_hypsometric_csv, write_parameters_csv
from .qgis_outputs import (
    STYLE_CATCHMENTS,
    STYLE_CENTROIDS,
    STYLE_FLOWPATHS,
    STYLE_NODES,
    STYLE_OUTLETS,
    STYLE_STREAMS,
    MemoryLayers,
    copy_source_layer,
    load_on_completion,
    make_plan_map,
    save_shapefile,
    split_bank_geometries,
)
from .report import write_report_docx
from .subcatchments import delineate_network_subcatchments

TOOL_VERSION = "0.10.1"

MODE_OPTIONS = [
    "Total (full catchment per outlet; nested catchments overlap)",
    "Incremental (sub-catchments along the stream network; the pour point / "
    "line is the final outlet)",
]
MODE_KEYS = ["total", "incremental"]

_S = QMetaType.Type.QString
_I = QMetaType.Type.Int
_D = QMetaType.Type.Double

# (field name, type, description) - the description feeds the help text.
CATCHMENT_FIELDS = [
    ("outlet_id", _S, "outlet name (ID field, or feature id)"),
    ("src_fid", _I, "source feature id"),
    ("out_type", _S, "point or line"),
    ("mode", _S, "total or incremental"),
    ("ds_ids", _S, "outlet(s) or stream this catchment drains into"),
    ("sub_type", _S, "incremental: stream or channel head sub-catchment"),
    ("system", _S, "incremental: system (stem letter of the final outlet)"),
    ("stream", _S, "incremental: stream the sub-catchment drains through"),
    ("bank", _S, "incremental: a = left bank, b = right bank (looking downstream)"),
    ("node", _S, "incremental: stream node the sub-catchment drains to"),
    ("ds_stream", _S, "incremental: stream that node discharges into"),
    ("x_in", _D, "input point X (points only)"),
    ("y_in", _D, "input point Y (points only)"),
    ("x_out", _D, "outlet cell centre X (snapped)"),
    ("y_out", _D, "outlet cell centre Y (snapped)"),
    ("snap_m", _D, "snap distance, m (points only)"),
    ("n_outcells", _I, "number of outlet cells"),
    ("area_km2", _D, "contributing area, km2"),
    ("area_ha", _D, "contributing area, ha"),
    ("hole_ha", _D, "interior holes filled, ha"),
    ("poly_km2", _D, "polygon area incl. filled holes, km2"),
    ("perim_km", _D, "polygon perimeter (cell outline), km"),
    ("n_parts", _I, "polygon parts (diagonal contacts split)"),
    ("z_outlet", _D, "outlet elevation, m"),
    ("z_min", _D, "minimum elevation, m"),
    ("z_max", _D, "maximum elevation, m"),
    ("z_mean", _D, "mean elevation, m"),
    ("relief_m", _D, "relief z_max - z_min, m"),
    ("elev_gain", _D, "elevation gain H = z_max - z_outlet, m"),
    ("cx", _D, "centroid X"),
    ("cy", _D, "centroid Y"),
    ("lfp_km", _D, "longest flow path length, km"),
    ("lca_km", _D, "outlet to point on LFP nearest the centroid, km"),
    ("s_avg", _D, "LFP average slope (H/L), m/m"),
    ("s_1085", _D, "LFP 10-85 slope, m/m"),
    ("s_ea", _D, "LFP equal-area slope, m/m"),
    ("s_mean_pc", _D, "mean catchment slope (Horn), %"),
    ("form_f", _D, "form factor A/L^2"),
    ("circ_r", _D, "circularity ratio 4piA/P^2"),
    ("elong_r", _D, "elongation ratio 2sqrt(A/pi)/L"),
    ("gravel_kc", _D, "Gravelius compactness P/(2sqrt(piA))"),
    ("hyps_int", _D, "hypsometric integral"),
    ("strm_km", _D, "stream length above threshold, km"),
    ("dd_kmkm2", _D, "drainage density, km/km2"),
    ("lb_km", _D, "basin length (outlet to farthest point), km"),
    ("relief_r", _D, "relief ratio (z_max - z_outlet) / Lb"),
    ("melton_r", _D, "Melton ratio H / sqrt(A)"),
    ("rugged_n", _D, "ruggedness number H x Dd"),
    ("sinuosity", _D, "LFP length / straight-line distance"),
    ("strm_ord", _I, "highest Strahler stream order"),
    ("n_strm", _I, "number of stream segments"),
    ("bif_r", _D, "mean bifurcation ratio"),
    ("strm_freq", _D, "stream frequency, segments/km2"),
    ("lo_km", _D, "length of overland flow 1/(2 Dd), km"),
    ("c_maint", _D, "constant of channel maintenance 1/Dd, km2/km"),
    ("ch_head_m", _D, "channel start on the LFP, m from the outlet (blank: none)"),
    ("chan_len_m", _D, "SWMM width: waterway length L_w (stream, else LFP), m"),
    ("lo_mean_m", _D, "SWMM width: mean overland flow length to a stream, m"),
    ("w_flow_m", _D, "SWMM width W_f = A / mean overland flow length, m"),
    ("w_chan_m", _D, "SWMM width W_c = 2 L_w (L_w for one bank), m"),
    ("gu_x", _D, "Guo & Urbonas shape factor X = A / L_w^2 (max 4)"),
    ("gu_z", _D, "Guo & Urbonas skew factor Z (0.5-1; 1 for one bank)"),
    ("w_gu_m", _D, "SWMM width, Guo & Urbonas (2009), m"),
    ("w_ratio", _D, "width ratio W_g / W_f"),
]


# Selectable outputs: (key, dialog label, file-name description, folder kind,
# ticked by default), and the QML style each layer is loaded with.
OUTPUT_SPECS = [
    OutputSpec("OUTPUT_CATCHMENTS", "Catchments", "Catchments", SHAPEFILE),
    OutputSpec("OUTPUT_DEM", "Catchment DEM", "Catchment DEM", RASTER),
    OutputSpec(
        "OUTPUT_FLOWPATHS", "Longest flow paths", "Longest Flow Paths", SHAPEFILE
    ),
    OutputSpec("OUTPUT_OUTLETS", "Snapped outlets", "Outlets", SHAPEFILE),
    OutputSpec(
        "OUTPUT_CENTROIDS", "Catchment centroids", "Catchment Centroids", SHAPEFILE
    ),
    OutputSpec(
        "OUTPUT_STREAMS",
        "Stream network (Strahler, Shreve, Horton, Hack orders)",
        "Stream Network",
        SHAPEFILE,
    ),
    OutputSpec(
        "OUTPUT_STREAM_NODES",
        "Stream nodes (channel heads, confluences, outlets, breakpoints)",
        "Stream Nodes",
        SHAPEFILE,
        default=False,
    ),
    OutputSpec("OUTPUT_MAIN_STEMS", "Main stems", "Main Stems", SHAPEFILE, False),
    OutputSpec(
        "OUTPUT_REPORT", "Catchment parameters (CSV)", "Catchment Parameters", CSV
    ),
    OutputSpec(
        "OUTPUT_HYPSO", "Hypsometric curves (CSV)", "Hypsometric Curves", CSV, False
    ),
    OutputSpec(
        "OUTPUT_STREAM_CSV",
        "Stream network properties (CSV)",
        "Stream Network Properties",
        CSV,
        default=False,
    ),
    OutputSpec("OUTPUT_DOCX", "Catchment report (Word)", "Catchment Report", WORD),
    OutputSpec(
        "OUTPUT_STREAM_DOCX",
        "Stream network report (Word)",
        "Stream Network Report",
        WORD,
        default=False,
    ),
]
SPEC_BY_KEY = {spec.key: spec for spec in OUTPUT_SPECS}
LAYER_STYLES = {
    "OUTPUT_CATCHMENTS": STYLE_CATCHMENTS,
    "OUTPUT_FLOWPATHS": STYLE_FLOWPATHS,
    "OUTPUT_OUTLETS": STYLE_OUTLETS,
    "OUTPUT_CENTROIDS": STYLE_CENTROIDS,
    "OUTPUT_STREAMS": STYLE_STREAMS,
    "OUTPUT_STREAM_NODES": STYLE_NODES,
    "OUTPUT_MAIN_STEMS": None,
}
STREAM_OUTPUT_KEYS = (
    "OUTPUT_STREAMS",
    "OUTPUT_STREAM_NODES",
    "OUTPUT_MAIN_STEMS",
    "OUTPUT_STREAM_CSV",
    "OUTPUT_STREAM_DOCX",
)
FOLDER_PARAMS = {
    RASTER: ("FOLDER_RASTER", "Output folder for rasters (GeoTIFF)"),
    SHAPEFILE: ("FOLDER_SHP", "Output folder for vector layers (shapefiles)"),
    CSV: ("FOLDER_CSV", "Output folder for CSV files"),
    WORD: ("FOLDER_WORD", "Output folder for Word reports"),
}


def _software_version() -> str:
    """'Mayim Tools <version>, QGIS <version>' for the report title page."""
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


def source_name(source) -> str:
    return source.sourceName() if source is not None else "no"


def _fields(specs) -> QgsFields:
    fields = QgsFields()
    for name, qtype, _ in specs:
        fields.append(QgsField(name, qtype))
    return fields


class CatchmentDelineationAlgorithm(QgsProcessingAlgorithm):
    DEM = "DEM"
    POINTER = "POINTER"
    ESRI_STYLE = "ESRI_STYLE"
    OUTLETS = "OUTLETS"
    ID_FIELD = "ID_FIELD"
    SNAP_RADIUS = "SNAP_RADIUS"
    MODE = "MODE"
    STREAM_THRESHOLD = "STREAM_THRESHOLD"
    MAX_STREAM_LENGTH = "MAX_STREAM_LENGTH"
    MIN_FINAL_REACH = "MIN_FINAL_REACH"
    SPLIT_BANKS = "SPLIT_BANKS"
    REPORT_DETAIL = "REPORT_DETAIL"
    SPLIT_MIN_CELLS = "SPLIT_MIN_CELLS"
    MAX_OVERLAND = "MAX_OVERLAND"
    SPLIT_MIN_WIDTH = "SPLIT_MIN_WIDTH"
    REPORT_BASEMAP = "REPORT_BASEMAP"
    STUDY_AREA = "STUDY_AREA"
    SHOW_CONTOURS = "SHOW_CONTOURS"
    CONTOUR_INTERVAL = "CONTOUR_INTERVAL"
    PROJECT_NUMBER = "PROJECT_NUMBER"
    PROJECT_NAME = "PROJECT_NAME"
    # output selection tick boxes (same keys as the result dictionary)
    OUTPUT_CATCHMENTS = "OUTPUT_CATCHMENTS"
    OUTPUT_DEM = "OUTPUT_DEM"
    OUTPUT_FLOWPATHS = "OUTPUT_FLOWPATHS"
    OUTPUT_OUTLETS = "OUTPUT_OUTLETS"
    OUTPUT_STREAMS = "OUTPUT_STREAMS"
    OUTPUT_STREAM_NODES = "OUTPUT_STREAM_NODES"
    OUTPUT_MAIN_STEMS = "OUTPUT_MAIN_STEMS"
    OUTPUT_STREAM_CSV = "OUTPUT_STREAM_CSV"
    OUTPUT_STREAM_DOCX = "OUTPUT_STREAM_DOCX"
    OUTPUT_CENTROIDS = "OUTPUT_CENTROIDS"
    OUTPUT_REPORT = "OUTPUT_REPORT"
    OUTPUT_HYPSO = "OUTPUT_HYPSO"
    OUTPUT_DOCX = "OUTPUT_DOCX"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def createInstance(self):
        return CatchmentDelineationAlgorithm()

    def name(self):
        return "catchment_delineation"

    def displayName(self):
        return "Catchments: 1. Delineation"

    def group(self):
        return "Hydrological Tools"

    def groupId(self):
        return "hydrological_tools"

    def shortHelpString(self):
        glossary = "\n".join(f"  {n}: {d}" for n, _, d in CATCHMENT_FIELDS)
        return (
            f"Catchments: 1. Delineation (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tDelineates the catchment draining to each pour "
            "point or pour line, and writes the catchment polygon(s) "
            "(geometry repaired), a catchment DEM, the longest flow "
            "path(s), the snapped outlet(s) and a set of hydrological "
            "modelling parameters.\n"
            "\n"
            "INPUTS:\tA D8 pointer raster (e.g. from D8 Flow Direction) "
            "and the DEM it was built from - both on exactly the same "
            "grid, in a projected CRS in metres. The DEM should be "
            "hydrologically conditioned. Pour points are snapped to the "
            "highest flow accumulation within the search radius. For "
            "pour lines, every cell the line crosses is an outlet cell "
            "(no snapping), and the catchment is everything draining to "
            "any point on the line.\n"
            "\n"
            "MODES:\tTotal gives each outlet its full catchment, so "
            "nested catchments overlap. Incremental divides the catchment of "
            "the final outlet (the pour point or pour line) into "
            "sub-catchments along the stream network (useful for HEC-HMS): "
            "one per stream (the reach between two nodes, after any stream "
            "breakpoints), draining to the node at its lower end, and one per "
            "channel head. Each has its own centroid and longest flow path. "
            "Optionally, each stream sub-catchment is split along the stream "
            "centreline into its left bank (a) and right bank (b), looking "
            "downstream: the whole sub-catchment is cut along the stream "
            "centreline, extended at both ends to its boundary, and cells on "
            "the line go with the left bank. Split tolerance: a piece smaller "
            "than the minimum area or narrower than the minimum mean width "
            "(area / length along the stream), both in cells, joins the "
            "neighbouring piece it shares the longest boundary with; pieces on "
            "the same side stay one multi-part sub-catchment, and a stream "
            "whose whole bank fails is not split. The catchment DEM covers all "
            "catchments together.\n"
            "\n"
            "NAMES (incremental):\tThe main stem is A (further outlets B, C ... "
            "by area). Streams are numbered upstream from the outlet: A1, A2 "
            "... At each confluence the stem follows the larger upstream area; "
            "a branch is named after the stream it flows into plus a letter "
            "(a branch joining at the top of A3 is A3A, its streams A3A1, A3A2 "
            "...; a second one A3B). Each sub-catchment is named after its "
            "stream (A1, or A1a and A1b when split); a channel head "
            "sub-catchment is named after the channel head node, the next "
            "number up the stem (A6 above A5). Nodes are named after the "
            "stream draining to them: the outlet is A1 and the node at the top "
            "of A3 is A4. node names the node a sub-catchment drains to and "
            "ds_stream the stream that node discharges into (empty at the "
            "outlet), so the sub-catchments link into a model as sub-basin -> "
            "junction (node) -> reach (ds_stream).\n"
            "\n"
            "GEOMETRY:\tPolygons follow cell edges. Interior holes (cells "
            "inside the outline that do not drain to the outlet) are "
            "filled and reported as hole_ha. A large share of holes means "
            "the DEM needs conditioning. In incremental mode, a sub-"
            "catchment wholly enclosed by another stays a real hole, so "
            "polygons never overlap. Cells that touch only at a "
            "corner become separate parts of a valid MultiPolygon. Each "
            "polygon is passed through makeValid and checked. Perimeter "
            "is the stepped cell outline, so it runs longer than a "
            "smooth boundary, and circularity/Gravelius reflect that.\n"
            "\n"
            "SWMM WIDTH:\tThree estimates of the SWMM sub-catchment width, "
            "which cannot be measured (calibrate it where flows are recorded): "
            "w_flow_m = A / mean overland flow length (D8 flow length from every "
            "cell to the nearest stream cell; optionally capped, Advanced); "
            "w_chan_m = 2 x the waterway length, or 1 x for one bank of a "
            "stream; and w_gu_m by Guo & Urbonas (2009), W = L (1.5 - Z) "
            "(2.286 X - 0.286 X^2) with X = A / L^2 (max 4) and the skew Z = "
            "larger side area / A. The waterway is the stream through a stream "
            "sub-catchment and the longest flow path otherwise. w_ratio = "
            "w_gu_m / w_flow_m flags shapes far from a rectangle.\n"
            "\n"
            "TIME OF CONCENTRATION:\tNot calculated here. ch_head_m gives "
            "where the channel starts on the longest flow path (the most "
            "upstream path cell whose upstream area reaches the stream "
            "threshold, in m from the outlet). The 'Catchments: 2. Time of "
            "Concentration' tool takes the catchments, the longest flow "
            "paths and the DEM and calculates Tc per catchment.\n"
            "\n"
            "STREAM NETWORK:\tThe same stream outputs as the Stream Network "
            "tool, for the network upstream of the outlets: reaches split at "
            "every confluence and at intermediate outlets, each with Strahler, "
            "Shreve, Horton and Hack orders, geometry, upstream area and "
            "routing inputs (one layer, styled by Strahler order); nodes "
            "(channel heads, confluences, outlets and stream breakpoints); main "
            "stems; the sectioned "
            "stream network CSV (abbreviation key, drop test, networks, Horton "
            "statistics and ratios, reaches, nodes, profiles, definitions); and "
            "an optional stream network Word report. The constant-drop test "
            "and routing settings are under Advanced parameters. Each reach "
            "appears once; outlet_id is the first outlet downstream. The "
            "stream network is only computed when one of these outputs is "
            "requested.\n"
            "\n"
            "MAXIMUM STREAM LENGTH:\tOptional (0 = none). Each reach (the stretch "
            "between two nodes) is measured upstream from its lower end; at "
            "every maximum length a 'stream breakpoint' node is added and the "
            "reach is split there, but only where the rest of the reach is "
            "longer than the minimum final reach length (default 10 % of the "
            "maximum). E.g. a maximum of 500 m on a 1320 m reach gives "
            "breakpoints at 500 and 1000 m; with 10 %, a 540 m reach is not "
            "split. Split reaches keep their id on the lowest piece "
            "and record split_from and piece. Network statistics (Horton, "
            "Hack, main stems) are not affected.\n"
            "\n"
            "CENTROIDS:\tOptional point layer of each catchment's area "
            "centroid, with its elevation, whether it falls inside the "
            "catchment (inside = 0 for strongly curved shapes), and the point "
            "on the longest flow path nearest to it (lca_x, lca_y) that "
            "defines Lca.\n"
            "\n"
            "REPORT:\tThe Word (.docx) annexure in the Mayim report standard: "
            "title page (project number and project / site name), summary, "
            "tool and method, "
            "inputs, parameter definitions with sources, and for each "
            "catchment a description, general, shape and relief parameter "
            "tables, slope, aspect and stream-order tables, figures (layout "
            "map, longitudinal profile, hypsometric curve, elevation "
            "histogram), modelling considerations and references. The layout "
            "map is drawn by QGIS with the same layer styles as the outputs, "
            "over an OpenStreetMap background (needs internet; Advanced "
            "parameters), with the optional study area boundary and contours "
            "generated from the DEM (minor contours fine, major contours every "
            "5 x the minor interval and slightly thicker; untick to leave "
            "them off), a "
            "legend, scale bar and north arrow. It needs python-docx (and "
            "matplotlib for figures) in QGIS's Python.\n"
            "\n"
            "OUTPUTS:\tTick the outputs you need and choose a folder for each "
            "kind: rasters (GeoTIFF), vector layers (shapefiles), CSV files and "
            "Word reports. Files are named <project number>_<description>, "
            "with underscores instead of spaces (e.g. MC2025_014_Catchments."
            "shp), and existing files of the same name are overwritten (remove "
            "them from the QGIS project first). Layers are loaded into the "
            "project with the Mayim layer styles (DRN_watershed, "
            "DRN_longest_flowpath, DRN_outlets, DRN_centroids, "
            "DRN_strahler_stream_order, DRN_stream_nodes). Shapefile field "
            "names are limited to 10 characters, so links_to_out is written "
            "as links_to_o.\n"
            "\n"
            "ATTRIBUTES:\n" + glossary
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.POINTER, "D8 pointer raster (flow direction)"
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.DEM, "DEM (same grid as the pointer; conditioned)"
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.ESRI_STYLE,
                "Pointer uses ESRI encoding (default: WhiteboxTools)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.OUTLETS,
                "Pour point(s) or pour line(s)",
                types=[
                    QgsProcessing.SourceType.TypeVectorPoint,
                    QgsProcessing.SourceType.TypeVectorLine,
                ],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.ID_FIELD,
                "Outlet name field (optional)",
                parentLayerParameterName=self.OUTLETS,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SNAP_RADIUS,
                "Pour point snap radius (cells; 0 = no snapping)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=0,
                defaultValue=3,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE, "Catchment mode", options=MODE_OPTIONS, defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.STREAM_THRESHOLD,
                "Stream threshold (upstream area, km²)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_STREAM_LENGTH,
                "Maximum stream length (m; stream breakpoints, 0 = none)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=0.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_FINAL_REACH,
                "Minimum final reach length (% of the maximum stream length; a "
                "breakpoint is only placed if the rest of the stream is longer)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                maxValue=99.0,
                defaultValue=10.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SPLIT_BANKS,
                "Incremental mode: split the stream sub-catchments along the "
                "stream centreline (a = left bank, b = right bank)",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SPLIT_MIN_CELLS,
                "Centreline split tolerance: minimum piece area (cells)",
                type=QgsProcessingParameterNumber.Type.Integer,
                minValue=1,
                defaultValue=10,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SPLIT_MIN_WIDTH,
                "Centreline split tolerance: minimum mean piece width (cells)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=2.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.STUDY_AREA,
                "Study area boundary for the report layout map (optional)",
                types=[QgsProcessing.SourceType.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SHOW_CONTOURS,
                "Show contours on the report layout map",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CONTOUR_INTERVAL,
                "Minor contour interval for the report layout map (m; major "
                "contours every 5 x this; 0 = no contours)",
                type=QgsProcessingParameterNumber.Type.Double,
                minValue=0.0,
                defaultValue=1.0,
            )
        )
        add_network_parameters(self, include_min_network=False)
        overland_param = QgsProcessingParameterNumber(
            self.MAX_OVERLAND,
            "SWMM width: maximum overland flow length (m; 0 = no cap, e.g. 30 "
            "after NRCS, 150 after UDFCD)",
            type=QgsProcessingParameterNumber.Type.Double,
            minValue=0.0,
            defaultValue=0.0,
        )
        overland_param.setFlags(
            overland_param.flags() | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(overland_param)
        detail_param = QgsProcessingParameterBoolean(
            self.REPORT_DETAIL,
            "Word report, incremental mode: a full section for every "
            "sub-catchment (long; otherwise parameter tables)",
            defaultValue=False,
        )
        detail_param.setFlags(
            detail_param.flags() | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(detail_param)
        basemap_param = QgsProcessingParameterBoolean(
            self.REPORT_BASEMAP,
            "Word report: OpenStreetMap background on the layout map "
            "(needs internet)",
            defaultValue=True,
        )
        basemap_param.setFlags(
            basemap_param.flags() | Qgis.ProcessingParameterFlag.Advanced
        )
        self.addParameter(basemap_param)
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
                    spec.key,
                    f"Output: {spec.label}",
                    defaultValue=spec.default,
                )
            )
        for name, label in FOLDER_PARAMS.values():
            self.addParameter(
                QgsProcessingParameterFolderDestination(name, label, optional=True)
            )

    # ------------------------------------------------------------------

    def _read_outlets(self, source, id_field, target_crs, context, feedback):
        transform = None
        if source.sourceCrs() != target_crs:
            transform = QgsCoordinateTransform(
                source.sourceCrs(), target_crs, context.transformContext()
            )
        outlets = []
        for feature in source.getFeatures():
            geom = QgsGeometry(feature.geometry())
            if geom.isNull() or geom.isEmpty():
                continue
            if transform is not None:
                geom.transform(transform)
            value = feature[id_field] if id_field else None
            if value is None or str(value) in ("", "NULL"):
                value = feature.id()
            name = str(value)
            gtype = geom.type()
            if gtype == Qgis.GeometryType.Point:
                pts = geom.asMultiPoint() if geom.isMultipart() else [geom.asPoint()]
                parts = [[(p.x(), p.y())] for p in pts]
                outlets.append(Outlet(name, "point", parts, int(feature.id())))
            elif gtype == Qgis.GeometryType.Line:
                geom.convertToStraightSegment()
                lines = (
                    geom.asMultiPolyline()
                    if geom.isMultipart()
                    else [geom.asPolyline()]
                )
                parts = [[(p.x(), p.y()) for p in line] for line in lines if line]
                outlets.append(Outlet(name, "line", parts, int(feature.id())))
            else:
                feedback.pushWarning(
                    f"Feature {feature.id()} skipped: not a point or line."
                )
        return outlets

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        self._post_processors = []
        pointer_layer = self.parameterAsRasterLayer(parameters, self.POINTER, context)
        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        if pointer_layer is None or dem_layer is None:
            raise QgsProcessingException("Both a D8 pointer and a DEM are required.")
        crs = dem_layer.crs()
        if crs.isGeographic():
            raise QgsProcessingException(
                "The DEM is in a geographic CRS (degrees). Areas, lengths and "
                "slopes need a projected CRS in metres - reproject the DEM and "
                "pointer first."
            )
        if pointer_layer.crs() != crs:
            raise QgsProcessingException(
                "The D8 pointer and DEM have different CRSs - they must share "
                "the same grid."
            )

        esri_style = self.parameterAsBoolean(parameters, self.ESRI_STYLE, context)
        source = self.parameterAsSource(parameters, self.OUTLETS, context)
        if source is None:
            raise QgsProcessingException("No pour point/line layer provided.")
        id_field = self.parameterAsString(parameters, self.ID_FIELD, context)
        snap_radius = self.parameterAsInt(parameters, self.SNAP_RADIUS, context)
        mode = MODE_KEYS[self.parameterAsEnum(parameters, self.MODE, context)]
        threshold = self.parameterAsDouble(parameters, self.STREAM_THRESHOLD, context)
        use_basemap = self.parameterAsBoolean(parameters, self.REPORT_BASEMAP, context)
        study_source = self.parameterAsSource(parameters, self.STUDY_AREA, context)
        contour_interval = self.parameterAsDouble(
            parameters, self.CONTOUR_INTERVAL, context
        )
        if not self.parameterAsBoolean(parameters, self.SHOW_CONTOURS, context):
            contour_interval = 0.0
        max_stream_m = (
            self.parameterAsDouble(parameters, self.MAX_STREAM_LENGTH, context)
            if parameters.get(self.MAX_STREAM_LENGTH) is not None
            else 0.0
        )
        min_final_pct = self.parameterAsDouble(
            parameters, self.MIN_FINAL_REACH, context
        )
        split_banks = self.parameterAsBoolean(parameters, self.SPLIT_BANKS, context)
        split_min_cells = self.parameterAsInt(parameters, self.SPLIT_MIN_CELLS, context)
        max_overland = self.parameterAsDouble(parameters, self.MAX_OVERLAND, context)
        split_min_width = self.parameterAsDouble(
            parameters, self.SPLIT_MIN_WIDTH, context
        )
        project_name = self.parameterAsString(
            parameters, self.PROJECT_NAME, context
        ).strip()
        project_number = self.parameterAsString(
            parameters, self.PROJECT_NUMBER, context
        )
        drop_range, routing = read_network_settings(self, parameters, context)

        selected = [
            spec
            for spec in OUTPUT_SPECS
            if self.parameterAsBoolean(parameters, spec.key, context)
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
        want_streams = bool(want & set(STREAM_OUTPUT_KEYS))
        if split_banks and mode != "incremental":
            feedback.pushWarning(
                "Splitting along the stream centreline only applies in "
                "incremental mode."
            )
        if max_stream_m > 0 and not want_streams and mode != "incremental":
            feedback.pushWarning(
                "The maximum stream length only affects the stream network "
                "outputs (stream network, nodes, main stems, stream CSV and "
                "report); none of them is ticked."
            )

        outlets = self._read_outlets(source, id_field, crs, context, feedback)
        if not outlets:
            raise QgsProcessingException("The pour point/line layer has no features.")
        feedback.pushInfo(f"{len(outlets)} outlet(s), mode: {mode}")

        network = None
        network_stage = False
        try:
            feedback.setProgressText("Reading rasters")
            pointer, p_gt, _, p_nodata = read_raster(pointer_layer.source())
            dem, d_gt, projection, d_nodata = read_raster(dem_layer.source())
            check_same_grid(pointer.shape, p_gt, dem.shape, d_gt)
            self._geotransform = d_gt

            def progress(fraction, message):
                feedback.setProgress(int(100 * fraction))
                feedback.setProgressText(message)

            result = delineate_catchments(
                pointer,
                p_nodata,
                dem,
                d_nodata,
                d_gt,
                outlets,
                esri_style=esri_style,
                snap_radius_cells=snap_radius,
                mode="total" if mode == "incremental" else mode,
                stream_threshold_km2=threshold,
                progress=progress,
                is_canceled=feedback.isCanceled,
                max_overland_m=max_overland,
            )
            if mode == "incremental" and result.catchments:
                network_stage = True
                # the outlets are the final outlets: divide their catchments
                # along the stream network
                feedback.setProgressText("Extracting the stream network")
                network = extract_stream_network(
                    pointer,
                    p_nodata,
                    dem,
                    d_nodata,
                    d_gt,
                    esri_style=esri_style,
                    threshold_km2=threshold,
                    drop_test_range=drop_range if want_streams else None,
                    routing=routing,
                    is_canceled=feedback.isCanceled,
                    outlet_cells=result.outlet_cells,
                    max_reach_length_m=max_stream_m or None,
                    min_remainder=min_final_pct / 100.0,
                )
                warnings_total = [
                    w for w in result.warnings if not w.startswith("Catchment '")
                ]
                result = delineate_network_subcatchments(
                    pointer,
                    p_nodata,
                    dem,
                    d_nodata,
                    d_gt,
                    network,
                    result.outlet_cells,
                    esri_style=esri_style,
                    stream_threshold_km2=threshold,
                    split_banks=split_banks,
                    min_split_cells=split_min_cells,
                    min_split_width_cells=split_min_width,
                    max_overland_m=max_overland,
                    progress=progress,
                    is_canceled=feedback.isCanceled,
                )
                result.warnings[:0] = warnings_total
                network_stage = False
                feedback.pushInfo(
                    f"{len(result.catchments)} sub-catchment(s) on "
                    f"{len(network.reaches)} stream(s)"
                    + (" (split along the stream centreline)" if split_banks else "")
                )
        except CatchmentError as e:  # (the same class as StreamNetworkError)
            if network_stage:
                raise QgsProcessingException(
                    "Incremental mode divides the catchment along the stream "
                    f"network upstream of the outlet, which failed: {e}"
                ) from e
            raise QgsProcessingException(str(e)) from e
        except Exception as e:
            raise QgsProcessingException(f"Catchment delineation failed: {e}") from e

        if feedback.isCanceled():
            return {}
        for message in result.warnings:
            feedback.pushWarning(message)
        if not result.catchments:
            raise QgsProcessingException(
                "No catchment could be delineated - check that the outlets lie "
                "on the pointer/DEM grid."
            )

        # vector results as memory layers (saved below and used by the map)
        memory = MemoryLayers()
        self._write_catchments(memory, feedback, result, crs)
        self._write_flow_paths(memory, result, crs)
        self._write_outlets(memory, result, crs)
        self._write_centroids(memory, result, crs)

        if want_streams and network is None:
            feedback.setProgressText("Extracting the stream network")
            try:
                network = extract_stream_network(
                    pointer,
                    p_nodata,
                    dem,
                    d_nodata,
                    d_gt,
                    esri_style=esri_style,
                    threshold_km2=threshold,
                    drop_test_range=drop_range,
                    routing=routing,
                    is_canceled=feedback.isCanceled,
                    outlet_cells=result.outlet_cells,
                    max_reach_length_m=max_stream_m or None,
                    min_remainder=min_final_pct / 100.0,
                )
            except StreamNetworkError as e:
                feedback.pushWarning(f"Stream network not written: {e}")
        if network is not None:  # (incremental mode always has one)
            if max_stream_m > 0:
                feedback.pushInfo(
                    f"Stream breakpoints: {network.settings['n_breakpoints']} "
                    f"(maximum stream length {max_stream_m:g} m)."
                )
            for message in network.warnings:
                feedback.pushWarning(f"Stream network: {message}")
            write_reaches(
                memory,
                parameters,
                self.OUTPUT_STREAMS,
                context,
                network,
                crs,
                [],
            )
            write_nodes(
                memory, parameters, self.OUTPUT_STREAM_NODES, context, network, crs
            )
            write_main_stems(
                memory, parameters, self.OUTPUT_MAIN_STEMS, context, network, crs
            )

        outputs = {name: folders[kind] for kind, (name, _) in FOLDER_PARAMS.items()
                   if kind in folders}  # fmt: skip
        written = []  # (label, path) for the reports

        def done(key, path):
            outputs[key] = path
            written.append((SPEC_BY_KEY[key].label, path))

        # shapefiles, loaded into the project with the Mayim styles
        feedback.setProgressText("Writing layers")
        for key, qml in LAYER_STYLES.items():
            if key not in want:
                continue
            layer = memory.get(key)
            if layer is None:
                feedback.pushWarning(f"{SPEC_BY_KEY[key].label}: nothing to write.")
                continue
            try:
                path = save_shapefile(layer, paths[key], context.transformContext())
            except Exception as e:
                feedback.reportError(str(e), fatalError=False)
                continue
            load_on_completion(
                context, path, Path(path).stem, qml, self._post_processors
            )
            done(key, path)

        if self.OUTPUT_DEM in want:
            path = paths[self.OUTPUT_DEM]
            try:
                write_raster(
                    path,
                    result.dem_array,
                    result.dem_geotransform,
                    projection,
                    result.dem_nodata,
                )
                load_on_completion(
                    context, path, Path(path).stem, None, self._post_processors
                )
                done(self.OUTPUT_DEM, path)
            except Exception as e:
                feedback.reportError(
                    f"{Path(path).name} could not be written ({e}) - it is "
                    "probably open in QGIS; remove it from the project and run "
                    "again.",
                    fatalError=False,
                )

        def write_csv_output(key, writer):
            path = paths[key]
            try:
                writer(path)
            except Exception as e:
                feedback.reportError(
                    f"{SPEC_BY_KEY[key].label}: {Path(path).name} could not be "
                    f"written ({e}). If it is open in QGIS or Excel, close it and "
                    "run again.",
                    fatalError=False,
                )
                return
            done(key, path)

        if self.OUTPUT_REPORT in want:
            param_key = {name: desc for name, _, desc in CATCHMENT_FIELDS}
            write_csv_output(
                self.OUTPUT_REPORT,
                lambda p: write_parameters_csv(p, result.catchments, key=param_key),
            )
        if self.OUTPUT_HYPSO in want:
            write_csv_output(
                self.OUTPUT_HYPSO,
                lambda p: write_hypsometric_csv(p, result.catchments),
            )

        if network is not None and want_streams:
            net_context = {
                "tool": f"{self.displayName()} - stream network",
                "version": TOOL_VERSION,
                "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "dem_path": dem_layer.source(),
                "pointer_path": pointer_layer.source(),
                "encoding": "ESRI" if esri_style else "WhiteboxTools",
                "crs": f"{crs.authid()} - {crs.description()}",
                "software": _software_version(),
                "project_number": project_number,
                "project_name": project_name,
                "max_stream_length_m": max_stream_m,
                "min_final_reach_pct": min_final_pct,
            }
            if self.OUTPUT_STREAM_CSV in want:
                write_csv_output(
                    self.OUTPUT_STREAM_CSV,
                    lambda p: write_sectioned_csv(p, network, net_context),
                )
            if self.OUTPUT_STREAM_DOCX in want:
                path = paths[self.OUTPUT_STREAM_DOCX]
                net_context["outputs"] = [
                    (label, p)
                    for label, p in written
                    if label.startswith(("Stream", "Main"))
                ] + [("This report", path)]
                try:
                    write_network_report_docx(path, network, net_context)
                    done(self.OUTPUT_STREAM_DOCX, path)
                except ImportError:
                    feedback.pushWarning(
                        "The stream network Word report needs python-docx in "
                        "QGIS's Python (python -m pip install python-docx)."
                    )
                except Exception as e:
                    feedback.pushWarning(
                        f"The stream network Word report could not be written: {e}"
                    )

        if self.OUTPUT_DOCX in want:
            docx_path = paths[self.OUTPUT_DOCX]
            feedback.setProgressText("Writing report")
            map_messages = []
            study_layer = None
            if study_source is not None:
                study_layer = copy_source_layer(
                    study_source, "Study area", crs, context.transformContext()
                )

            surface = np.asarray(dem, dtype="float64").copy()  # NoData as NaN
            if d_nodata is not None:
                surface[surface == d_nodata] = np.nan

            def contours_for(extent):
                minor, major, warnings = minor_major_contours(
                    surface, d_gt, contour_interval, extent=extent
                )
                map_messages.extend(warnings)
                return minor, major

            plan_map = make_plan_map(
                crs,
                context.transformContext(),
                memory,
                {
                    "catchments": self.OUTPUT_CATCHMENTS,
                    "flowpaths": self.OUTPUT_FLOWPATHS,
                    "outlets": self.OUTPUT_OUTLETS,
                    "centroids": self.OUTPUT_CENTROIDS,
                    **(
                        {"nodes": self.OUTPUT_STREAM_NODES}
                        if self.OUTPUT_STREAM_NODES in want or mode == "incremental"
                        else {}
                    ),
                    **(
                        {"streams": self.OUTPUT_STREAMS}
                        if mode == "incremental" and network is not None
                        else {}
                    ),
                },
                study_area=study_layer,
                contours_for=contours_for,
                contour_interval=contour_interval,
                use_osm=use_basemap,
                messages=map_messages,
                label_catchments=mode == "incremental",
            )
            context_info = {
                "tool": self.displayName(),
                "version": TOOL_VERSION,
                "software": _software_version(),
                "project_number": project_number,
                "project_name": project_name,
                "study_area": source_name(study_source),
                "contour_interval": contour_interval,
                "max_stream_length_m": max_stream_m,
                "min_final_reach_pct": min_final_pct,
                "split_banks": split_banks,
                "split_min_cells": split_min_cells,
                "split_min_width": split_min_width,
                "max_overland_m": max_overland,
                "detail_sections": self.parameterAsBoolean(
                    parameters, self.REPORT_DETAIL, context
                ),
                "n_streams": len(network.reaches) if network else None,
                "n_breakpoints": (
                    network.settings.get("n_breakpoints") if network else None
                ),
                "run_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "dem": {
                    "path": dem_layer.source(),
                    "crs": f"{crs.authid()} - {crs.description()}",
                    "cell_x": abs(d_gt[1]),
                    "cell_y": abs(d_gt[5]),
                    "cols": dem.shape[1],
                    "rows": dem.shape[0],
                },
                "pointer": {
                    "path": pointer_layer.source(),
                    "encoding": "ESRI" if esri_style else "WhiteboxTools",
                },
                "outlets": {
                    "layer": source.sourceName(),
                    "n_features": len(outlets),
                    "id_field": id_field,
                },
                "snap_radius_cells": snap_radius,
                "mode": mode,
                "stream_threshold_km2": threshold,
                "outputs": written + [("This report", docx_path)],
                "warnings": result.warnings,
                "map_messages": map_messages,
            }
            try:
                messages = write_report_docx(
                    docx_path, result.catchments, context_info, plan_map=plan_map
                )
                for message in dict.fromkeys((messages or []) + map_messages):
                    feedback.pushWarning(message)
                done(self.OUTPUT_DOCX, docx_path)
            except ImportError:
                feedback.pushWarning(
                    "The Word report needs the python-docx package, which is not "
                    "installed in QGIS's Python. Install it from the OSGeo4W Shell "
                    "with: python -m pip install python-docx. All other outputs "
                    "were written."
                )
            except Exception as e:  # report failure must not lose the results
                feedback.pushWarning(f"The Word report could not be written: {e}")

        for label, path in written:
            feedback.pushInfo(f"{label}: {path}")
        for c in result.catchments:
            a = c.attributes
            feedback.pushInfo(
                f"{c.outlet_id}: {a['area_km2']:.4f} km2, LFP {a['lfp_km']:.3f} km"
            )
        return outputs

    # ------------------------------------------------------------------

    def _write_catchments(self, memory, feedback, result, crs):
        sink, dest_id = memory.parameterAsSink(
            None,
            self.OUTPUT_CATCHMENTS,
            None,
            _fields(CATCHMENT_FIELDS),
            Qgis.WkbType.MultiPolygon,
            crs,
        )
        if sink is None:
            raise QgsProcessingException("Could not create the catchments output.")
        cut_cache: dict = {}
        cut_failed = []
        for c in result.catchments:
            geom = None
            split = c.detail.get("split")
            if split is not None:  # a bank: cut the outline along the stream
                stream = c.attributes.get("stream")
                if stream not in cut_cache:
                    try:
                        cut_cache[stream] = split_bank_geometries(
                            split, self._geotransform
                        )
                    except Exception:
                        cut_cache[stream] = None
                    if cut_cache[stream] is None:
                        cut_failed.append(stream)
                cut = cut_cache[stream]
                if cut is not None:
                    geom = QgsGeometry(cut[c.attributes["bank"]])
                    c.attributes["poly_km2"] = round(geom.area() / 1e6, 6)
                    c.attributes["n_parts"] = geom.constGet().numGeometries()
            if geom is None:
                geom = QgsGeometry.fromWkt(rings_to_multipolygon_wkt(c.polygons))
            geom = geom.makeValid()
            if QgsWkbTypes.flatType(geom.wkbType()) not in (
                Qgis.WkbType.Polygon,
                Qgis.WkbType.MultiPolygon,
            ):
                geom.convertGeometryCollectionToSubclass(Qgis.GeometryType.Polygon)
            geom.convertToMultiType()
            if geom.isEmpty() or not geom.isGeosValid():
                feedback.pushWarning(
                    f"Catchment '{c.outlet_id}': polygon could not be repaired "
                    "to a valid geometry."
                )
            feat = QgsFeature(_fields(CATCHMENT_FIELDS))
            feat.setGeometry(geom)
            for name, _, _ in CATCHMENT_FIELDS:
                feat[name] = c.attributes.get(name)
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        if cut_failed:
            feedback.pushWarning(
                "The outline of stream sub-catchment(s) "
                f"{', '.join(cut_failed)} could not be cut along the centreline; "
                "their banks follow the cell edges instead."
            )
        return dest_id

    def _write_flow_paths(self, memory, result, crs):
        specs = [
            ("outlet_id", _S, ""),
            ("lfp_km", _D, ""),
            ("z_start", _D, ""),
            ("z_outlet", _D, ""),
            ("s_avg", _D, ""),
            ("s_1085", _D, ""),
            ("s_ea", _D, ""),
            ("ch_head_m", _D, ""),
        ]
        sink, dest_id = memory.parameterAsSink(
            None,
            self.OUTPUT_FLOWPATHS,
            None,
            _fields(specs),
            Qgis.WkbType.LineString,
            crs,
        )
        if sink is None:
            return None
        for c in result.catchments:
            if len(c.flow_path) < 2:
                continue
            a = c.attributes
            feat = QgsFeature(_fields(specs))
            feat.setGeometry(
                QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in c.flow_path])
            )
            # ends of the path itself (for a pour line the catchment outlet
            # elevation z_outlet is the lowest line cell, which can differ)
            prof_z = c.detail.get("profile_z_m")
            z_start = z_end = None
            if prof_z is not None and len(prof_z) and not np.isnan(prof_z[-1]):
                z_start = round(float(prof_z[-1]), 3)
            if prof_z is not None and len(prof_z) and not np.isnan(prof_z[0]):
                z_end = round(float(prof_z[0]), 3)
            feat["outlet_id"] = c.outlet_id
            feat["lfp_km"] = a["lfp_km"]
            feat["z_start"] = z_start
            feat["z_outlet"] = z_end
            feat["s_avg"] = a["s_avg"]
            feat["s_1085"] = a["s_1085"]
            feat["s_ea"] = a["s_ea"]
            feat["ch_head_m"] = a.get("ch_head_m")
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_outlets(self, memory, result, crs):
        specs = [
            ("outlet_id", _S, ""),
            ("out_type", _S, ""),
            ("x_in", _D, ""),
            ("y_in", _D, ""),
            ("snap_m", _D, ""),
            ("area_km2", _D, ""),
        ]
        sink, dest_id = memory.parameterAsSink(
            None,
            self.OUTPUT_OUTLETS,
            None,
            _fields(specs),
            Qgis.WkbType.Point,
            crs,
        )
        if sink is None:
            return None
        for c in result.catchments:
            if c.outlet_xy is None:
                continue
            feat = QgsFeature(_fields(specs))
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(*c.outlet_xy)))
            for name, _, _ in specs:
                feat[name] = c.attributes.get(name)
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id

    def _write_centroids(self, memory, result, crs):
        specs = [
            ("outlet_id", _S, ""),
            ("cx", _D, ""),
            ("cy", _D, ""),
            ("z_m", _D, ""),
            ("inside", _I, ""),
            ("area_km2", _D, ""),
            ("lca_km", _D, ""),
            ("lca_x", _D, ""),
            ("lca_y", _D, ""),
        ]
        sink, dest_id = memory.parameterAsSink(
            None,
            self.OUTPUT_CENTROIDS,
            None,
            _fields(specs),
            Qgis.WkbType.Point,
            crs,
        )
        if sink is None:
            return None
        for c in result.catchments:
            a = c.attributes
            lca = c.detail.get("lca_point") or (None, None)
            feat = QgsFeature(_fields(specs))
            feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(a["cx"], a["cy"])))
            feat["outlet_id"] = c.outlet_id
            feat["cx"] = a["cx"]
            feat["cy"] = a["cy"]
            feat["z_m"] = c.detail.get("centroid_z")
            feat["inside"] = int(bool(c.detail.get("centroid_inside")))
            feat["area_km2"] = a["area_km2"]
            feat["lca_km"] = a["lca_km"]
            feat["lca_x"] = lca[0]
            feat["lca_y"] = lca[1]
            sink.addFeature(feat, QgsFeatureSink.Flag.FastInsert)
        return dest_id
