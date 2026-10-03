"""Processing Toolbox algorithm: HEC-RAS geometry HDF5 in -> every
selected 2D geometry element type out, as GeoPackage layers written
into one output folder, in one tool run.

Ported 2026-09-29 from the standalone "Import RAS 2D Data" plugin
(package ``ras2d_import``) into the Mayim Tools suite, per Chris's
request: "Can we at this time include this plugin within the Mayim
Tools plugin suite please. It will sit under Mayim Tools -> HEC-RAS
Tools -> Import RAS 2D data." Only the QGIS-facing wrapper's package
paths, ``name()``/``displayName()``/``group()``/``groupId()``/
``icon()`` changed on the port - none of the reading, writing, or
styling logic below was touched. See
claude/hec_ras_automation_plugin_research.md for the standalone
plugin's full development history.

Two UI decisions drive this file's shape, both direct requests from
Chris after testing earlier builds of the standalone plugin:

1. (2026-09-22) "I want this to be one tool, not separate tools for
   each element type... a checklist in the tool for the user to select
   which elements to import (all selected by default)." -> one
   QgsProcessingParameterEnum(allowMultiple=True) checklist, not one
   tool/output per element.
2. (2026-09-23) "can we specify a folder, and then assign default
   names for all these layers" -> rather than one
   QgsProcessingParameterFeatureSink per element (18 of them, which
   would mean naming 18 outputs by hand in the dialog - the same
   problem as (1) one level down), this algorithm takes a SINGLE
   QgsProcessingParameterFolderDestination and writes one GeoPackage
   file per SELECTED element straight into it, named after Chris's own
   list with spaces (and, since a literal '/' can't appear in a
   filename, forward slashes too) replaced with underscores - e.g.
   "SA/2D connections - bridge 2D" -> "SA_2D_connections_-_bridge_2D.gpkg".
   Each written file is also registered with the QGIS project so it
   loads into the Layers panel automatically, same as a normal
   Processing sink output would.
3. (2026-09-23, bugfix) Chris reported that in real QGIS every line
   layer (and possibly polygon layers) came out with all its features
   collapsed into one shape / all features sharing one geometry. The
   first cut of (2) wrote features one at a time straight into a raw
   QgsVectorFileWriter obtained via QgsVectorFileWriter.create() (an
   incremental "open file, addFeature() in a loop, close on del"
   writer) - not the same code path as the previously-working,
   sink-based single-tool-per-element build, which always went through
   QgsProcessingParameterFeatureSink/parameterAsSink. Rather than keep
   guessing at what specifically differs in the incremental-writer
   path without a real QGIS/GDAL install available to reproduce
   against, every generic writer here was switched to the standard,
   heavily-used two-step QGIS idiom for "materialise features built
   from scratch, then save them": build an in-memory QgsVectorLayer,
   populate it in ONE batch call to dataProvider().addFeatures(...),
   then export the whole layer to GeoPackage with
   QgsVectorFileWriter.writeAsVectorFormatV3(). This sidesteps whatever
   was going wrong in the incremental per-feature writer path.
4. (2026-09-23, resolved) The real cause turned out to be neither (a)
   nor (b) above but a genuine geometry-decoding bug in
   _common/hdf5_utils.py's read_polyline_features (a Polyline/Centerline
   Parts point_start being treated as an absolute index into
   Points when it's actually relative to the feature's own Info
   point_start) - see the project doc's "Root cause found and fixed"
   section for the full story. _DistinctFeatureStyler (below) wasn't
   the fix, but stays in place as the fallback default style for any
   layer without a QML mapping (see point 5).
5. (2026-09-23) Chris supplied a saved .qml style file per element
   (his own "RAS_..." naming, in a folder on his machine) and asked
   for these to be applied automatically on import, replacing
   _DistinctFeatureStyler's generic styling for every element that has
   one. Originally exposed as an editable Processing parameter, then
   (same day) Chris asked for it to be hard-coded into the script
   instead - see QML_STYLES_DIR / _ELEMENT_QML_STYLES / _QmlFileStyler
   below. Only "2D Mesh Cells (polygons)" and "2D perimeter" have no
   QML mapping and keep _DistinctFeatureStyler's generic styling; every
   other element, including "2D computation points" (added in the same
   request), has one.

This file is a thin QGIS-facing wrapper only - it imports and calls the
zero-QGIS core.py reader functions in each element's own package
(mesh_2d/core.py, boundary_lines/core.py, reference_lines_points/core.py,
reference_areas/core.py, structures/core.py, initial_condition_points/core.py,
pump_stations/core.py, pipe_networks/core.py, all under this tool's own
package) plus the shared _common/hdf5_utils.py. None of the reading logic
is duplicated here.
"""

import random
from pathlib import Path

from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsPointXY,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsRendererCategory,
    QgsSingleSymbolRenderer,
    QgsVectorFileWriter,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor, QIcon

from mayim_tools.hec_ras.import_ras_2d_data.boundary_lines.core import (
    read_boundary_condition_lines,
)
from mayim_tools.hec_ras.import_ras_2d_data.initial_condition_points.core import (
    read_initial_condition_points,
)
from mayim_tools.hec_ras.import_ras_2d_data.mesh_2d.core import (
    list_2d_flow_area_names,
    read_2d_flow_area_cells,
    read_2d_flow_area_perimeter,
    read_break_lines,
    read_refinement_regions,
)
from mayim_tools.hec_ras.import_ras_2d_data.pipe_networks.core import (
    read_pipe_conduits,
    read_pipe_nodes,
)
from mayim_tools.hec_ras.import_ras_2d_data.pump_stations.core import (
    read_pump_stations,
)
from mayim_tools.hec_ras.import_ras_2d_data.reference_areas.core import (
    read_reference_areas,
)
from mayim_tools.hec_ras.import_ras_2d_data.reference_lines_points.core import (
    read_reference_lines,
    read_reference_points,
)
from mayim_tools.hec_ras.import_ras_2d_data.structures.core import (
    read_bridge_1d_structures,
    read_bridge_2d_structures,
    read_culvert_barrels,
    read_linear_routing_structures,
    read_weir_structures,
)

TOOL_VERSION = "0.4.5"

# --- Element catalogue -------------------------------------------------
#
# One row per checklist item. `label` is what's shown in the dialog;
# `slug` is the filename stem (Chris's own naming, spaces AND forward
# slashes replaced with underscores - a literal '/' can't appear in a
# Windows or GPKG filename, so this is a necessary deviation from "only
# replace spaces"). `kind` selects which of the three generic writers
# in this file builds the geometry: "line" (PolylineFeature.parts as a
# MultiLineString), "polygon" (PolylineFeature.parts as one polygon's
# rings), or "point" (PointFeature.xy). `reader` is the zero-argument-
# except-hdf5-path core.py function to call. Mesh-derived elements
# (perimeter/cell polygons/computation points) are handled separately
# in _load_mesh_elements since all three are built from one pass over
# each 2D Flow Area's cells, not a single reader call each.
#
# Order matches the order Chris listed the elements in, with two
# additions kept from the previous build (not in Chris's list, but
# already-working capability - see shortHelpString): 2D Mesh Cells
# (polygons), placed next to 2D computation points.

ELEMENT_PERIMETER = 0
ELEMENT_CELL_POLYGONS = 1
ELEMENT_CELL_POINTS = 2
ELEMENT_BREAK_LINES = 3
ELEMENT_REFINEMENT_REGIONS = 4
ELEMENT_BRIDGE_2D = 5
ELEMENT_BRIDGE_1D = 6
ELEMENT_WEIR = 7
ELEMENT_LINEAR_ROUTING = 8
ELEMENT_CULVERT = 9
ELEMENT_PUMP_STATIONS = 10
ELEMENT_BOUNDARY_LINES = 11
ELEMENT_IC_POINTS = 12
ELEMENT_REFERENCE_POINTS = 13
ELEMENT_REFERENCE_LINES = 14
ELEMENT_REFERENCE_AREAS = 15
ELEMENT_PIPE_NODES = 16
ELEMENT_PIPE_CONDUITS = 17

# Mesh elements handled by _load_mesh_elements (no single reader call).
_MESH_ELEMENTS = {ELEMENT_PERIMETER, ELEMENT_CELL_POLYGONS, ELEMENT_CELL_POINTS}

# Location of Chris's saved .qml style files. Hard-coded per his
# explicit 2026-09-23 request ("The default styles must be part of the
# script (hard-coded)") - not a Processing parameter, so styling is
# always applied with zero per-run configuration. If this path ever
# changes on Chris's machine, it needs a code change (and redeploy)
# here, not a dialog edit.
QML_STYLES_DIR = Path(
    r"D:\Dropbox\MAYIM\6 TEGNIESE DATA EN DOKUMENTE\QGIS\QGIS Layer styles"
)

# element_id -> QML file stem (no ".qml" extension), per Chris's
# mapping (2026-09-23, extended same day to add 2D computation
# points). Several elements intentionally share one style file (e.g.
# every SA/2D connection type except culvert uses "RAS_SA2D") - this
# is a many-to-one mapping, not a typo. Every element in
# _ELEMENT_CATALOGUE has an entry, plus ELEMENT_CELL_POINTS (a mesh
# element, handled separately by _load_mesh_elements - see its
# _register_output call). Only ELEMENT_PERIMETER and
# ELEMENT_CELL_POLYGONS have no mapping and keep
# _DistinctFeatureStyler's generic styling instead.
_ELEMENT_QML_STYLES = {
    ELEMENT_CELL_POINTS: "RAS_2D_comp_points",
    ELEMENT_BREAK_LINES: "RAS_2D_breaklines",
    ELEMENT_REFINEMENT_REGIONS: "RAS_2D_regions",
    ELEMENT_BRIDGE_2D: "RAS_SA2D",
    ELEMENT_BRIDGE_1D: "RAS_SA2D",
    ELEMENT_WEIR: "RAS_SA2D",
    ELEMENT_LINEAR_ROUTING: "RAS_SA2D",
    ELEMENT_CULVERT: "RAS_SA2D_culvert_CL",
    ELEMENT_PUMP_STATIONS: "RAS_pump_stations",
    ELEMENT_BOUNDARY_LINES: "RAS_boundary_conditions",
    ELEMENT_IC_POINTS: "RAS_IC_points",
    ELEMENT_REFERENCE_POINTS: "RAS_ref_points",
    ELEMENT_REFERENCE_LINES: "RAS_ref_line",
    ELEMENT_REFERENCE_AREAS: "RAS_ref_area",
    ELEMENT_PIPE_NODES: "RAS_pipe_network_nodes",
    ELEMENT_PIPE_CONDUITS: "RAS_pipe_network_conduit",
}

# label, slug, kind, reader - for every element NOT in _MESH_ELEMENTS.
_ELEMENT_CATALOGUE = {
    ELEMENT_BREAK_LINES: ("2D breaklines", "2D_breaklines", "line", read_break_lines),
    ELEMENT_REFINEMENT_REGIONS: (
        "2D refinement regions",
        "2D_refinement_regions",
        "polygon",
        read_refinement_regions,
    ),
    ELEMENT_BRIDGE_2D: (
        "SA/2D connections - bridge 2D",
        "SA_2D_connections_-_bridge_2D",
        "line",
        read_bridge_2d_structures,
    ),
    ELEMENT_BRIDGE_1D: (
        "SA/2D connections - bridge 1D",
        "SA_2D_connections_-_bridge_1D",
        "line",
        read_bridge_1d_structures,
    ),
    ELEMENT_WEIR: (
        "SA/2D connection - weir",
        "SA_2D_connection_-_weir",
        "line",
        read_weir_structures,
    ),
    ELEMENT_LINEAR_ROUTING: (
        "SA/2D connection - linear routing",
        "SA_2D_connection_-_linear_routing",
        "line",
        read_linear_routing_structures,
    ),
    ELEMENT_CULVERT: (
        "SA/2D connection - culvert (subset of weir)",
        "SA_2D_connection_-_culvert",
        "line",
        read_culvert_barrels,
    ),
    ELEMENT_PUMP_STATIONS: (
        "Pump stations",
        "Pump_stations",
        "point",
        read_pump_stations,
    ),
    ELEMENT_BOUNDARY_LINES: (
        "Boundary condition lines",
        "Boundary_condition_lines",
        "line",
        read_boundary_condition_lines,
    ),
    ELEMENT_IC_POINTS: (
        "Initial condition points",
        "Initial_condition_points",
        "point",
        read_initial_condition_points,
    ),
    ELEMENT_REFERENCE_POINTS: (
        "Reference points",
        "Reference_points",
        "point",
        read_reference_points,
    ),
    ELEMENT_REFERENCE_LINES: (
        "Reference lines",
        "Reference_lines",
        "line",
        read_reference_lines,
    ),
    ELEMENT_REFERENCE_AREAS: (
        "Reference areas",
        "Reference_areas",
        "polygon",
        read_reference_areas,
    ),
    ELEMENT_PIPE_NODES: (
        "Pipe networks - nodes",
        "Pipe_networks_-_nodes",
        "point",
        read_pipe_nodes,
    ),
    ELEMENT_PIPE_CONDUITS: (
        "Pipe networks - conduits",
        "Pipe_networks_-_conduits",
        "line",
        read_pipe_conduits,
    ),
}

# Index positions double as the checklist order shown to the user -
# keep this in sync with the constants above if ever reordered.
ELEMENT_LABELS = [
    "2D perimeter",
    "2D Mesh Cells (polygons)",
    "2D computation points",
    "2D breaklines",
    "2D refinement regions",
    "SA/2D connections - bridge 2D",
    "SA/2D connections - bridge 1D",
    "SA/2D connection - weir",
    "SA/2D connection - linear routing",
    "SA/2D connection - culvert (subset of weir)",
    "Pump stations",
    "Boundary condition lines",
    "Initial condition points",
    "Reference points",
    "Reference lines",
    "Reference areas",
    "Pipe networks - nodes",
    "Pipe networks - conduits",
]


class _DistinctFeatureStyler(QgsProcessingLayerPostProcessorInterface):
    """Default styling applied to every layer this tool writes, added
    2026-09-23 after Chris reported line (and possibly polygon) layers
    "looking combined into one shape" in real QGIS even though the
    attribute table correctly showed every feature. QGIS's own default
    single-symbol style can genuinely make this happen with no data
    bug at all: many touching/adjacent polygons (a tessellated 2D mesh)
    with a solid fill read as one blob with the cell boundaries
    invisible, and a chain of end-to-end line features (e.g.
    breaklines running along a channel bank) all drawn the same colour
    read as one continuous line. This class exists so every output
    layer is visually legible as "many separate features" on first
    load, not just correctly separate underneath.

    - Polygon layers get a single no-fill/visible-stroke symbol, which
      reveals every shared cell edge in a mesh instead of one solid
      colour field.
    - Line layers get a categorized renderer - one random colour per
      distinct value of ``category_field`` (falling back to feature id
      if that field is absent, has only one value, or the layer wasn't
      given one) - so neighbouring/touching segments are visibly
      different features. Capped at MAX_CATEGORIES to avoid an
      exploded legend on a very large line layer; beyond that, QGIS's
      own default style is left alone.
    - Point layers are left at QGIS's default style - not implicated
      in the reported symptom.

    NOTE: instances of this class must be kept alive (see
    ImportRas2dDataAlgorithm._post_processors) until QGIS actually
    calls postProcessLayer() after the algorithm returns - Processing
    only holds a weak reference to a LayerDetails' post-processor.
    """

    MAX_CATEGORIES = 60

    def __init__(self, category_field: str | None = None):
        super().__init__()
        self._category_field = category_field

    def postProcessLayer(self, layer, context, feedback):
        geometry_type = layer.geometryType()
        if geometry_type == QgsWkbTypes.GeometryType.PolygonGeometry:
            self._style_outline_only(layer)
        elif geometry_type == QgsWkbTypes.GeometryType.LineGeometry:
            self._style_categorized(layer)

    def _style_outline_only(self, layer):
        symbol = QgsFillSymbol.createSimple(
            {
                "color": "255,255,255,0",
                "outline_color": "30,30,30,255",
                "outline_width": "0.3",
                "style": "solid",
            }
        )
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        layer.triggerRepaint()

    def _style_categorized(self, layer):
        field_name = self._category_field
        if field_name and layer.fields().indexOf(field_name) < 0:
            field_name = None

        values = None
        if field_name:
            values = sorted(
                {feature[field_name] for feature in layer.getFeatures()}, key=str
            )
            if len(values) <= 1:
                values = None  # nothing to distinguish by - fall back to id

        if values is None:
            field_name = "$id"
            values = [feature.id() for feature in layer.getFeatures()]

        if len(values) > self.MAX_CATEGORIES:
            return  # leave QGIS's own default style in place

        categories = []
        for value in values:
            color = QColor(
                random.randint(30, 220),
                random.randint(30, 220),
                random.randint(30, 220),
            )
            symbol = QgsLineSymbol.createSimple(
                {"line_color": color.name(), "line_width": "0.6"}
            )
            categories.append(QgsRendererCategory(value, symbol, str(value)))

        layer.setRenderer(QgsCategorizedSymbolRenderer(field_name, categories))
        layer.triggerRepaint()


class _QmlFileStyler(QgsProcessingLayerPostProcessorInterface):
    """Applies one of Chris's saved .qml style files to a layer on
    load, added 2026-09-23 per his element-to-QML mapping (see
    _ELEMENT_QML_STYLES). Used in place of _DistinctFeatureStyler for
    every element that has a mapped style file.

    Never raises: a missing/unreadable QML file only produces a
    feedback warning and leaves the layer at QGIS's own default style,
    since one missing style file shouldn't fail the whole import.

    NOTE: like _DistinctFeatureStyler, instances must be kept alive
    (see ImportRas2dDataAlgorithm._post_processors) until QGIS calls
    postProcessLayer() after the algorithm returns.
    """

    def __init__(self, qml_path: str):
        super().__init__()
        self._qml_path = qml_path

    def postProcessLayer(self, layer, context, feedback):
        if not Path(self._qml_path).is_file():
            feedback.pushWarning(
                f"QML style file not found, left at default style: " f"{self._qml_path}"
            )
            return
        # QgsMapLayer.loadNamedStyle returns (message, resultFlag) in
        # PyQGIS - resultFlag is the C++ method's by-reference bool
        # out-param, folded into the Python return tuple.
        message, ok = layer.loadNamedStyle(self._qml_path)
        if not ok:
            feedback.pushWarning(
                f"Failed to apply QML style '{self._qml_path}': {message}"
            )
            return
        layer.triggerRepaint()


def _attribute_fields(attribute_dicts) -> QgsFields:
    """Build a QgsFields schema from the union of attribute keys seen
    across a list of plain attribute dicts, typed as strings."""
    fields = QgsFields()
    seen = set()
    for attributes in attribute_dicts:
        for key in attributes:
            if key not in seen:
                seen.add(key)
                fields.append(QgsField(key, QMetaType.Type.QString))
    return fields


class ImportRas2dDataAlgorithm(QgsProcessingAlgorithm):

    INPUT_GEOMETRY_HDF = "INPUT_GEOMETRY_HDF"
    CRS = "CRS"
    ELEMENTS = "ELEMENTS"
    OUTPUT_FOLDER = "OUTPUT_FOLDER"

    def createInstance(self):
        return ImportRas2dDataAlgorithm()

    def name(self):
        return "import_ras_2d_data"

    def displayName(self):
        return "Import RAS 2D data"

    def group(self):
        return "HEC-RAS Tools"

    def groupId(self):
        return "hec_ras_tools"

    def icon(self):
        return QIcon(
            str(Path(__file__).resolve().parents[2] / "icons" / "mayim_logo.png")
        )

    def shortHelpString(self):
        return (
            f"Import RAS 2D data (version {TOOL_VERSION})\n"
            "\n"
            "PURPOSE:\tReads every 2D geometry element defined in a "
            "HEC-RAS 6.x/7.0 model - mesh, breaklines, refinement "
            "regions, SA/2D connections (bridges, weirs, culverts, "
            "linear routing), pump stations, boundary condition lines, "
            "initial condition points, reference lines/points/areas, "
            "and pipe networks - as GeoPackage layers written into one "
            "output folder, in one tool run. Use the 'Elements to "
            "import' checklist to pick which of these to write; all "
            "are selected by default. Each selected element becomes "
            "its own <name>.gpkg file in the output folder and is "
            "loaded into the project automatically.\n"
            "\n"
            "NOTE:\t'2D Mesh Cells (polygons)' isn't one of Chris's "
            "originally-listed elements but is kept from the earlier "
            "build alongside '2D computation points' (the cell centre "
            "points) since it already worked - leave it unchecked if "
            "not wanted. SA/2D connections all live in one underlying "
            "HDF5 group and are split into the bridge 2D / bridge 1D / "
            "weir / linear routing / culvert outputs by their "
            "Connection/Mode/Culvert-Groups attributes, not by separate "
            "HDF5 locations. 'Bridge 2D'/'Bridge 1D'/'weir'/'linear "
            "routing' are each one feature per SA/2D CONNECTION, split "
            "by their Connection/Mode attributes. 'culvert' is "
            "different (2026-09-23, per Chris): it is each culvert "
            "BARREL's own centreline (Geometry/Structures/Culvert "
            "Groups/Barrels) - not the parent connection's centerline, "
            "and not a subset of the 'weir' output's features - so a "
            "weir/gate/culvert connection with 2 barrels contributes 1 "
            "feature to 'weir' and 2 features to 'culvert'. The "
            "Boundary Condition Lines 'Type' attribute is RAS's own "
            "External/Internal classification, not an inflow/outflow "
            "direction.\n"
            "\n"
            "PARAMETERS:\n"
            "  Geometry HDF5 file: the .hdf file next to your HEC-RAS "
            ".gXX geometry file (or a plan output HDF5, which embeds "
            "the same Geometry/ group).\n"
            "  CRS: the coordinate reference system your RAS project "
            "geometry is defined in - not yet auto-detected from the "
            "file, set this to match your project.\n"
            "  Elements to import: which geometry types to write.\n"
            "  Output folder: where the .gpkg file for each selected "
            "element is written, named after the element (spaces - and "
            "forward slashes, which can't appear in a filename - "
            "replaced with underscores, e.g. 'SA_2D_connections_-_"
            "bridge_2D.gpkg').\n"
            "\n"
            "STYLING:\tEach output layer is styled automatically on "
            "load. Most elements get one of Chris's saved .qml style "
            "files, loaded from a hard-coded folder on his machine "
            "(see QML_STYLES_DIR / _ELEMENT_QML_STYLES in this file) - "
            "not a dialog parameter, per his own request. Only '2D "
            "perimeter' and '2D Mesh Cells (polygons)' have no mapped "
            ".qml and instead get a generic outline-only/categorized-"
            "random-colour default. A missing individual .qml file "
            "only logs a warning and leaves that one layer unstyled - "
            "it never fails the run."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_GEOMETRY_HDF,
                "HEC-RAS geometry HDF5 file",
                extension="hdf",
            )
        )
        self.addParameter(
            QgsProcessingParameterCrs(
                self.CRS,
                "Project coordinate reference system",
                defaultValue="EPSG:4326",
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ELEMENTS,
                "Elements to import",
                options=ELEMENT_LABELS,
                allowMultiple=True,
                defaultValue=list(range(len(ELEMENT_LABELS))),
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.OUTPUT_FOLDER,
                "Output folder (one .gpkg per selected element)",
            )
        )
        # No QML-styles-folder parameter: styling is hard-coded (see
        # QML_STYLES_DIR above), per Chris's explicit request.

    def processAlgorithm(
        self, parameters, context: QgsProcessingContext, feedback: QgsProcessingFeedback
    ):
        hdf_path = self.parameterAsFile(parameters, self.INPUT_GEOMETRY_HDF, context)
        if not hdf_path:
            raise QgsProcessingException("No geometry HDF5 file provided.")

        crs: QgsCoordinateReferenceSystem = self.parameterAsCrs(
            parameters, self.CRS, context
        )
        output_folder = self.parameterAsString(parameters, self.OUTPUT_FOLDER, context)
        if not output_folder:
            raise QgsProcessingException("No output folder provided.")
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Keeps every post-processor (_DistinctFeatureStyler or
        # _QmlFileStyler) alive until QGIS actually calls
        # postProcessLayer() after this method returns - LayerDetails
        # only holds a weak reference to its post-processor, so
        # without this each styler would be garbage-collected before
        # it ever ran.
        self._post_processors = []

        selected = set(self.parameterAsEnums(parameters, self.ELEMENTS, context))
        if not selected:
            feedback.pushWarning("No elements selected - nothing to import.")

        written_paths = []

        mesh_elements_selected = selected & _MESH_ELEMENTS
        if mesh_elements_selected:
            written_paths.extend(
                self._load_mesh_elements(
                    hdf_path,
                    crs,
                    mesh_elements_selected,
                    output_folder,
                    context,
                    feedback,
                )
            )

        for element_id, (label, slug, kind, reader) in _ELEMENT_CATALOGUE.items():
            if element_id not in selected:
                continue
            features = reader(hdf_path)
            if kind == "line":
                path = self._write_line_features(
                    features,
                    label,
                    slug,
                    output_folder,
                    crs,
                    context,
                    feedback,
                    element_id,
                )
            elif kind == "polygon":
                path = self._write_polygon_features(
                    features,
                    label,
                    slug,
                    output_folder,
                    crs,
                    context,
                    feedback,
                    element_id,
                )
            else:
                path = self._write_point_features(
                    features,
                    label,
                    slug,
                    output_folder,
                    crs,
                    context,
                    feedback,
                    element_id,
                )
            written_paths.append(path)

        feedback.pushInfo(f"Wrote {len(written_paths)} layer(s) to {output_folder}")
        return {self.OUTPUT_FOLDER: output_folder}

    # --- Generic file writers -------------------------------------
    #
    # Build an in-memory QgsVectorLayer, hand it every feature in ONE
    # batch call to dataProvider().addFeatures(...), then export the
    # whole layer to GeoPackage in one call. This is the standard QGIS
    # idiom for "materialise features built from scratch and save
    # them" - see the 2026-09-23 bugfix note at the top of this file
    # for why it replaced a raw QgsVectorFileWriter.create() +
    # per-feature addFeature() loop.

    def _build_memory_layer(self, fields, wkb_type, crs):
        geom_name = QgsWkbTypes.displayString(wkb_type)
        layer = QgsVectorLayer(f"{geom_name}?crs={crs.authid()}", "temp", "memory")
        provider = layer.dataProvider()
        provider.addAttributes(fields)
        layer.updateFields()
        return layer

    def _export_layer(self, layer, out_path, context):
        layer.updateExtents()
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.fileEncoding = "UTF-8"
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, out_path, context.transformContext(), options
        )
        error = result[0]
        error_message = result[1] if len(result) > 1 else ""
        if error != QgsVectorFileWriter.WriterError.NoError:
            raise QgsProcessingException(
                f"Failed to write '{out_path}': {error_message}"
            )

    def _register_output(
        self, out_path, label, context, element_id=None, category_field=None
    ):
        qml_stem = (
            _ELEMENT_QML_STYLES.get(element_id) if element_id is not None else None
        )
        if qml_stem:
            qml_path = str(QML_STYLES_DIR / f"{qml_stem}.qml")
            styler = _QmlFileStyler(qml_path)
        else:
            styler = _DistinctFeatureStyler(category_field)
        details = QgsProcessingContext.LayerDetails(label, context.project())
        details.setPostProcessor(styler)
        self._post_processors.append(styler)
        context.addLayerToLoadOnCompletion(out_path, details)

    def _write_line_features(
        self, features, label, slug, output_folder, crs, context, feedback, element_id
    ):
        feedback.pushInfo(f"{label}: found {len(features)}")
        fields = _attribute_fields(f.attributes for f in features)
        out_path = str(Path(output_folder) / f"{slug}.gpkg")
        layer = self._build_memory_layer(fields, QgsWkbTypes.MultiLineString, crs)
        qgs_features = []
        for feature in features:
            qgs_feature = QgsFeature(layer.fields())
            for key, value in feature.attributes.items():
                qgs_feature[key] = str(value)
            qgs_feature.setGeometry(
                QgsGeometry.fromMultiPolylineXY(
                    [[QgsPointXY(x, y) for x, y in part] for part in feature.parts]
                )
            )
            qgs_features.append(qgs_feature)
        layer.dataProvider().addFeatures(qgs_features)
        self._export_layer(layer, out_path, context)
        self._register_output(
            out_path, label, context, element_id=element_id, category_field="Name"
        )
        return out_path

    def _write_polygon_features(
        self, features, label, slug, output_folder, crs, context, feedback, element_id
    ):
        feedback.pushInfo(f"{label}: found {len(features)}")
        fields = _attribute_fields(f.attributes for f in features)
        out_path = str(Path(output_folder) / f"{slug}.gpkg")
        layer = self._build_memory_layer(fields, QgsWkbTypes.Polygon, crs)
        qgs_features = []
        for feature in features:
            qgs_feature = QgsFeature(layer.fields())
            for key, value in feature.attributes.items():
                qgs_feature[key] = str(value)
            if feature.parts:  # a feature with zero rings keeps a null geometry
                qgs_feature.setGeometry(
                    QgsGeometry.fromPolygonXY(
                        [[QgsPointXY(x, y) for x, y in part] for part in feature.parts]
                    )
                )
            qgs_features.append(qgs_feature)
        layer.dataProvider().addFeatures(qgs_features)
        self._export_layer(layer, out_path, context)
        self._register_output(out_path, label, context, element_id=element_id)
        return out_path

    def _write_point_features(
        self, features, label, slug, output_folder, crs, context, feedback, element_id
    ):
        feedback.pushInfo(f"{label}: found {len(features)}")
        fields = _attribute_fields(f.attributes for f in features)
        out_path = str(Path(output_folder) / f"{slug}.gpkg")
        layer = self._build_memory_layer(fields, QgsWkbTypes.Point, crs)
        qgs_features = []
        for feature in features:
            qgs_feature = QgsFeature(layer.fields())
            for key, value in feature.attributes.items():
                qgs_feature[key] = str(value)
            x, y = feature.xy
            qgs_feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
            qgs_features.append(qgs_feature)
        layer.dataProvider().addFeatures(qgs_features)
        self._export_layer(layer, out_path, context)
        self._register_output(out_path, label, context, element_id=element_id)
        return out_path

    def _load_mesh_elements(
        self, hdf_path, crs, wanted, output_folder, context, feedback
    ):
        try:
            flow_area_names = list_2d_flow_area_names(hdf_path)
        except Exception as e:
            raise QgsProcessingException(f"Failed to read 2D Flow Areas: {e}") from e

        if not flow_area_names:
            feedback.pushWarning("No 2D Flow Areas found in this geometry file.")

        area_field = QgsFields()
        area_field.append(QgsField("Flow Area", QMetaType.Type.QString))

        # One in-memory layer + feature list per wanted mesh element -
        # features accumulate in these lists across all flow areas and
        # are only added to their layer (and exported) once, in one
        # batch each, at the end. See the class-level writer docstring
        # for why (2026-09-23 bugfix).
        layers = {}
        feature_lists = {}
        out_paths = {}
        labels = {}
        if ELEMENT_PERIMETER in wanted:
            out_paths[ELEMENT_PERIMETER] = str(
                Path(output_folder) / "2D_perimeter.gpkg"
            )
            labels[ELEMENT_PERIMETER] = "2D perimeter"
            layers[ELEMENT_PERIMETER] = self._build_memory_layer(
                area_field, QgsWkbTypes.Polygon, crs
            )
            feature_lists[ELEMENT_PERIMETER] = []
        if ELEMENT_CELL_POLYGONS in wanted:
            out_paths[ELEMENT_CELL_POLYGONS] = str(
                Path(output_folder) / "2D_Mesh_Cells_(polygons).gpkg"
            )
            labels[ELEMENT_CELL_POLYGONS] = "2D Mesh Cells (polygons)"
            layers[ELEMENT_CELL_POLYGONS] = self._build_memory_layer(
                area_field, QgsWkbTypes.Polygon, crs
            )
            feature_lists[ELEMENT_CELL_POLYGONS] = []
        if ELEMENT_CELL_POINTS in wanted:
            out_paths[ELEMENT_CELL_POINTS] = str(
                Path(output_folder) / "2D_computation_points.gpkg"
            )
            labels[ELEMENT_CELL_POINTS] = "2D computation points"
            layers[ELEMENT_CELL_POINTS] = self._build_memory_layer(
                area_field, QgsWkbTypes.Point, crs
            )
            feature_lists[ELEMENT_CELL_POINTS] = []

        total_cells = 0
        for area_name in flow_area_names:
            feedback.pushInfo(f"Reading 2D Flow Area: {area_name}")

            if ELEMENT_PERIMETER in layers:
                perimeter_ring = read_2d_flow_area_perimeter(hdf_path, area_name)
                perim_feature = QgsFeature(layers[ELEMENT_PERIMETER].fields())
                perim_feature["Flow Area"] = area_name
                perim_feature.setGeometry(
                    QgsGeometry.fromPolygonXY(
                        [[QgsPointXY(x, y) for x, y in perimeter_ring]]
                    )
                )
                feature_lists[ELEMENT_PERIMETER].append(perim_feature)

            if ELEMENT_CELL_POLYGONS in layers or ELEMENT_CELL_POINTS in layers:
                cells = read_2d_flow_area_cells(hdf_path, area_name)
                total_cells += len(cells)
                for cell in cells:
                    if ELEMENT_CELL_POLYGONS in layers:
                        poly_feature = QgsFeature(
                            layers[ELEMENT_CELL_POLYGONS].fields()
                        )
                        poly_feature["Flow Area"] = area_name
                        if cell.polygon:
                            poly_feature.setGeometry(
                                QgsGeometry.fromPolygonXY(
                                    [[QgsPointXY(x, y) for x, y in cell.polygon]]
                                )
                            )
                        feature_lists[ELEMENT_CELL_POLYGONS].append(poly_feature)
                    if ELEMENT_CELL_POINTS in layers:
                        point_feature = QgsFeature(layers[ELEMENT_CELL_POINTS].fields())
                        point_feature["Flow Area"] = area_name
                        point_feature.setGeometry(
                            QgsGeometry.fromPointXY(QgsPointXY(*cell.center))
                        )
                        feature_lists[ELEMENT_CELL_POINTS].append(point_feature)

        written_paths = []
        for element_id, layer in layers.items():
            layer.dataProvider().addFeatures(feature_lists[element_id])
            out_path = out_paths[element_id]
            self._export_layer(layer, out_path, context)
            # ELEMENT_CELL_POINTS has a QML mapping (RAS_2D_comp_points)
            # and gets _QmlFileStyler via _register_output below;
            # ELEMENT_PERIMETER/ELEMENT_CELL_POLYGONS have none and
            # fall back to _DistinctFeatureStyler's generic styling.
            self._register_output(
                out_path, labels[element_id], context, element_id=element_id
            )
            written_paths.append(out_path)

        feedback.pushInfo(
            f"{len(flow_area_names)} flow area(s)"
            + (f", {total_cells} cell(s) total" if total_cells else "")
        )
        return written_paths
