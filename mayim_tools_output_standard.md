# Mayim Tools Output Standard

*House standard for every output of the Mayim tools and analyses: Word reports, graphs and charts, maps, and the files and layers the tools write (version 2.4, October 2026). Previously called the Mayim Report, Figure and Graph Standard.*

*Version 2: title page added (section 1.4); report structure and checklist updated to match; font availability rule added (section 2.2). Version 2.1: minor and major contours, with an option to leave contours off (section 3.2). Version 2.2: gridlines drawn as semi-transparent black (section 2.3). Version 2.3: gridlines in solid greys, because semi-transparent black showed as solid black in some matplotlib versions (section 2.3); tables left-aligned with their left border on the text margin (section 1.3); A4 page with a 16.5 cm text width (section 1.0). Version 2.4: renamed the Mayim Tools Output Standard (file `mayim_tools_output_standard.md`); section 4 (output files and layers) expanded with the conventions settled while bringing every tool to the standard (October 2026); implementation reference and checklist updated to match.*

**How to use this document:** give it to Claude in any chat with the instruction "apply the Mayim Tools output standard". Every value below is a requirement unless it is marked as a default. If something is not covered here, follow the closest rule and say which choice was made.

---

## 1. Word reports (.docx)

### 1.0 Page

- **A4** portrait (21.0 × 29.7 cm).
- **Margins:** left and right 2.25 cm, top and bottom 2.5 cm, so the text width is **16.5 cm**, the width of every table and figure.
- Set the page explicitly: python-docx starts from US Letter with 3.17 cm side margins (a 15.24 cm text width), on which 16.5 cm tables and figures run past the right margin.

### 1.1 Text styles

| Element | Font | Size | Weight / style | Colour | Other |
|---|---|---|---|---|---|
| Normal (body) text | Arial | 10 pt | Regular | Black #000000 | Also used for bullets, numbered lists and table text (tables use 9 pt, see 1.3) |
| Headings (levels 1 and 2: main sections) | Arial | 12 pt | Bold, not italic | #0F0F36 | ALL CAPS (set as the font's All caps property, so the text itself keeps normal case) |
| Sub-headings (level 3 and below, plus run-in labels such as "Geomorphological description") | Arial | 10 pt | Bold, not italic | #0F0F36 | Normal case |
| Table and figure headings (captions) | Arial | 9 pt | Italic | Black | "Table n: text" above the table; "Figure n: text" below the figure |
| Report title (title page only, see 1.4) | Arial | 20 pt | Bold, not italic | #0F0F36 | Centre-aligned, normal case (not all caps) |

- Set the fonts on the **styles** (Normal, Title, Heading 1–4, Caption) rather than on individual runs. Set them for every script (ascii, hAnsi, eastAsia, cs) and remove the theme-font links, otherwise Word falls back to Calibri or Aptos.
- Remove theme colours from the heading styles so that #0F0F36 is used exactly.
- Keep each table heading with its table, and each figure with its caption (keep-with-next).
- Number tables and figures consecutively through the report.

### 1.2 Numbers and units

- **No thousands separators:** write 424627, not 424,627 or 424 627.
- Use a decimal point. Use a sensible, consistent number of decimals per quantity, e.g. areas in km² to 3–4 decimals, elevations to 2, slopes in % to 2–3, times to 1.
- Use SI units with proper symbols: km², m/m, %, m³/s and so on. Write units in brackets in labels and column headings, e.g. "Area (km²)".
- When a value is missing or cannot be computed, show "-", never "None", "nan" or a blank.

### 1.3 Tables

- Use the "Table Grid" style, with column widths set explicitly so the table fits the 16.5 cm text width (the widths add up to 16.5 cm or less).
- **Placement:** left-aligned, with the table's **left border on the text margin**, in line with the body text. This is the position of the table itself, not the alignment of the text in its cells. In Word's compatibility mode the table indent is measured to the cell text, so the table is indented by its left cell margin (108 twips, 0.19 cm, in Table Grid) to put the border on the margin.
- The title-page details table (1.4) stays centred.
- **Header row:** 9 pt bold, shaded #ADADCC.
- **Body:** 9 pt regular.
- **Group sub-heading rows** (when a table is split into groups): merged across the full width, 9 pt bold, shaded #F2F2F2.
- **Parameter tables:** use three columns, Parameter (with symbol and unit) | Value | Definition/equation (with source), at widths of about 6.2 / 2.8 / 7.5 cm.
- Leave one empty paragraph after each table.

### 1.4 Title page

Every report starts with a title page on its own (page 1), followed by a page break. When the report is used as an annexure, the title page is its cover.

Content, all centre-aligned, in this order:

| # | Element | Format | Content |
|---|---|---|---|
| 1 | Report title | Arial 20 pt bold, #0F0F36 ("Title" style), placed about one third of the way down the page | The tool or report name, e.g. "Design Rainfall Estimation (South Africa)" |
| 2 | Subtitle (optional) | Arial 12 pt bold, #0F0F36, normal case | The project or site name, when given |
| 3 | Purpose and goal | Arial 10 pt regular, black; at most 2–3 sentences (about 60 words) | What the report covers and what it is for, e.g. "This report gives design rainfall depths for the site, estimated with the regional L-moment method, as input to the stormwater design." |
| 4 | Report details | Borderless two-column table, Arial 10 pt, labels bold, about 5 + 7 cm wide, centred | See the list below |

Report details (show "-" for an item that is not available):

- Project number
- Project / site
- Prepared by: Mayim Consulting
- Tool and version, e.g. "Design Rainfall Estimation (South Africa), version 1.1"
- Software, e.g. "Mayim Tools 0.18.2, QGIS 4.0.3"
- Report date (the date the report was written, YYYY-MM-DD)
- Document status (default "Draft")

Rules:

- No page number, header or footer on the title page.
- The no-thousands-separators rule (1.2) applies here too.
- The title is not repeated as a heading on page 2; the report body starts with the Summary.

### 1.5 Report structure (default for tool annexures)

1. Title page (1.4), then a page break.
2. Summary, with a summary table.
3. Tool and method, as numbered steps.
4. Inputs and settings (table).
5. Parameters and calculations: a definitions table giving the symbol, equation and source.
6. Results: per item (e.g. per catchment), the description, then tables, then figures.
7. Outputs: the files produced and their locations.
8. Notes and limitations, including the processing warnings from the run.
9. References.

Write guideline-neutral content unless a specific standard (e.g. SANRAL, ARR) is the subject. Cite sources for equations.

---

## 2. Graphs and charts

### 2.1 Size and output

- Every figure is exactly **16.5 cm wide**. The height varies with the content.
- Save figures at their **exact size**, with no tight-bbox cropping, so Word places them at 16.5 cm without rescaling. Use PNG at 200 dpi, or 150 dpi for map renders.
- Use a white background.
- In matplotlib, use the object API (`Figure` and `FigureCanvasAgg`, not `pyplot`), which is thread-safe inside QGIS Processing. Use `layout="constrained"`.

### 2.2 Fonts (all 9 pt, black)

| Element | Font | Size | Style |
|---|---|---|---|
| Default text (tick labels, annotations, notes, scale-bar labels) | Arial Narrow | 9 pt | Regular, black |
| Legend entries | Arial Narrow | 9 pt | *Italic*, black |
| Headings, e.g. axis titles and figure titles | Arial Narrow | 9 pt | **Bold**, black |

- **Font availability:** report figures must use the house fonts (Arial; Arial Narrow for chart text). If a font is installed in Windows but not seen by the figure software, the tool must find it in the Windows font folders, including the per-user folder (`%LOCALAPPDATA%\Microsoft\Windows\Fonts`), before falling back to a substitute. Any substitution must be named in the run log.
- Only if the font cannot be found in any of those folders, substitute a metric-compatible font (Liberation Sans Narrow, then Arial) and report the substitution to the user.

### 2.3 Plot area, axes and grid

- **Plot area border:** black, 0.75 pt, on all four sides.
- **Ticks:** black, 0.75 pt wide, with minor ticks on.
- **Major gridlines:** solid, 0.75 pt, dark grey **#808080**.
- **Minor gridlines:** solid, 0.5 pt, light grey **#D9D9D9**.
- Use solid (opaque) greys, not semi-transparent black: in some matplotlib versions the grid alpha setting overrides the colour's own alpha, so semi-transparent black showed as solid black (version 2.2).
- Draw gridlines behind the data.
- Use plain number formatting, with no offset or scientific notation on coordinate or elevation axes, and no thousands separators.
- Show units in the axis titles: "Elevation (m)", "Distance upstream from outlet (km)".
- Legends have no frame on graphs, and are placed where they do not cover the data (e.g. matplotlib `loc="best"`). On maps they sit below the map (see 3.3).

### 2.4 Colours

Content colours, used in any order:

| # | Colour | Hex |
|---|---|---|
| 1 | Red | #FF0000 |
| 2 | Amber | #FFC000 |
| 3 | Yellow | #FFFF00 |
| 4 | Green | #00CC00 |
| 5 | Blue | #0000FF |
| 6 | Cyan | #00FFFF |
| 7 | Magenta | #FF00FF |

- Use yellow only for areas and fills. Thin lines on white are unreadable in yellow, so a line series that would be third uses green (#00CC00) instead.
- Reference and secondary lines, such as a 1:1 or uniform-slope line, are black, 0.75 pt, dashed.
- Main data lines are 1.5 pt. Fills under a curve use the series colour at about 12 % opacity.
- Bars are filled with the series colour and have a black 0.75 pt edge.

### 2.5 Special line conventions

- **Natural ground profiles** (long sections and terrain profiles) are black, 1 pt, dashed.
- Use dashed or dotted line styles to separate derived lines (e.g. slope lines) from measured data.

---

## 3. Maps and layout figures

### 3.1 Approach

- QGIS renders the map content (all layers, with their saved QML styles), so maps always match the layer styles in the GIS project.
- If the QML styles are not known or specified, request them from the user.
- The frame around the map is then composed in the chart style above: coordinate axes, legend, scale bar, north arrow and attribution.
- The map frame is about 14 cm wide inside the 16.5 cm figure. Its height follows the map extent.
- Render the map at 150 dpi so that symbol sizes in mm stay true to the QML styles.
- The axes show Easting (m) and Northing (m) in the projected CRS, in plain numbers.

### 3.2 Map content

- **Background:** the standard OpenStreetMap tiles (XYZ `https://tile.openstreetmap.org/{z}/{x}/{y}.png`).
  - Show the attribution "© OpenStreetMap contributors" on the map, bottom right.
  - Add a note in the report giving the licence (ODbL, openstreetmap.org/copyright).
  - If no tiles are received (e.g. offline), leave the background out and warn the user. Never show a blank background with an attribution.
- **Study area boundary** (optional input): black, 0.5 mm, dashed outline, no fill.
- **Contours:** generated from the DEM, in two weights so they do not overwhelm the map.
  - **Inputs:** a tick box "Show contours on the layout map" (default on) and the **minor contour interval** in metres (default 1 m). Unticking the box, or an interval of 0, leaves the contours off altogether.
  - **Major contours** are every fifth minor contour (major interval = 5 × minor interval, e.g. 1 m minor and 5 m major).
  - **Minor contours:** black, 0.15 mm, solid. **Major contours:** black, 0.35 mm, solid, drawn above the minor ones. A level is drawn once only: minor lines are not drawn where a major line falls.
  - **Legend:** two entries, e.g. "Minor contours (1 m interval)" and "Major contours (5 m interval)".
  - **Too dense:** if the minor contours would be too many for the map (e.g. a large area with much relief), draw the major contours only and say so in the run log and the report's processing messages; if even the major contours are too many, leave the contours off with a message suggesting a larger interval.
- **Layers** use the Mayim QML layer styles (folder `D:\Dropbox\MAYIM\6 TEGNIESE DATA EN DOKUMENTE\QGIS\QGIS Layer styles`). Copies are kept in each tool's `styles/` folder.

| Layer | Style file | Key field |
|---|---|---|
| Catchments / watersheds | DRN_watershed.qml | – |
| Catchment centroids | DRN_centroids.qml | – |
| Longest flow paths | DRN_longest_flowpath.qml | – |
| Outlets / pour points | DRN_outlets.qml | – |
| Streams by Strahler order | DRN_strahler_stream_order.qml | `strahler` |
| Stream nodes | DRN_stream_nodes.qml | `node_type` |

- **Drawing order** (top to bottom): centroids, outlets, stream nodes, flow paths, streams, catchments, study area, major contours, minor contours, OSM.

### 3.3 Map furniture

- **Legend:** below the map, 3 columns, Arial Narrow 9 pt italic, no frame.
  - Legend keys are rendered by QGIS from the layer styles. A key box is 2 × 1 font-size units, i.e. 18 × 9 pt.
  - Show only the classes present in the data, e.g. only the Strahler orders that occur.
- **Scale bar:** bottom left, on a white semi-transparent backing.
  - It has 4 alternating black and white segments with a 0.75 pt black edge.
  - Its length is the largest 1, 2 or 5 × 10ⁿ m that fits in a quarter of the map width.
  - Labels are 0, the half-way value and the full length, with the unit (m, or km at 1000 m and over).
- **North arrow:** top right, a black arrow with a bold "N" above it on a white backing.
- **No gridlines** on maps. The border and coordinate ticks follow 2.3.
- **Figure caption** (below): describe the content, e.g. "Catchment C1: layout of the catchment, stream network, longest flow path, outlet and centroid (background: OpenStreetMap)".

---

## 4. Output files and layers from tools

### 4.1 Selection and folders

- **Selection:** a tick box for each output, so the user selects which outputs to generate. Each output has a sensible default (the main layer, the calculations CSV and the report usually on; diagnostic or specialist outputs off). At least one output must be ticked.
- **Folders:** one per kind, chosen by the user: rasters (GeoTIFF .tif), vector layers (shapefile .shp), CSV (.csv) and Word reports (.docx). A tool that writes figure files (e.g. PNG sections) gives them their own figure folder. Only the folders of the ticked outputs are needed.
- **Project inputs:** an optional **project number** (the file-name prefix) and, for tools with a report, an optional **project / site name** for the title page.
- Check the ticked outputs and their folders **before** the processing starts, so a missing folder fails fast rather than after a long run.
- Supporting files go with the output they belong to: a layer file (.qlr) beside its rasters, a metadata CSV beside the rasters it describes, a results.json (see 4.4) in the CSV folder.

### 4.2 File names

- **Names:** `<project number>_<description>.<ext>`, with underscores throughout and no spaces, e.g. `MC2025_014_Catchments.shp` or `MC2025_014_Catchment_Report.docx`. The project number is optional; without it the name is just the description.
  - Characters that are not allowed in Windows file names are removed.
- Tools that convert an input file name their output after it: `<project number>_<input file>_<variable>.csv` (e.g. GRIB to CSV, NetCDF4 to CSV).
- Tools whose files already had established names (e.g. the soil and land-cover extracts) keep those names and add the `<project number>_` prefix. Tools that read such files find them with or without the prefix.
- Per-method or per-item files add the method or item before the description, e.g. `<project number>_HSG_NEH630_Recommended.tif`, `<project number>_Section_A_Plan.png`.
- The tool help lists, for every output, the exact file name it produces, built from the same definitions the code uses (so the help cannot drift from the behaviour).

### 4.3 Writing

- **Existing files** with the same name are overwritten.
- **Locked files:** if a file cannot be written (e.g. open in QGIS, Excel or Word), report it clearly, with the advice to close it and run again, and still write the other outputs. Stop with "No output could be written" only when nothing at all was written.
- Every written file is listed in the run log, and in the report's Outputs section (1.5).

### 4.4 Formats

- **Vector layers:** shapefiles. Exception: GeoPackage where the attribute names must stay longer than 10 characters, e.g. HEC-RAS attributes used by the RAS layer styles (Import RAS 2D data).
- **Rasters:** GeoTIFF, with a layer file (.qlr) that loads them with their styles where a tool writes several.
- **Results for the report compiler** (optional, off by default): `<project number>_<description>.json` in the CSV folder, schema `mayim.results/1` (tool, inputs, and per item every result with method, return period, value, unit, flags and derivation).
- **Text files for other programs** (e.g. HEC-HMS elevation-volume tables) stay plain text in the layout that program expects.
- **Processing outputs for model chaining:** tools whose result is mainly an input to the next Processing step (e.g. D8 Flow Direction and D8 Flow Accumulation) may keep Processing raster outputs. Single-file tools still declare their file as a Processing output, so it can be linked in models.

### 4.5 Layers

- **Loading:** written layers are loaded into the QGIS project with their QML style.
- **Shapefile field names:** at most 10 characters. Shorten longer ones explicitly and list them in the help text.
- **Long text fields** (e.g. notes) are cut at 254 characters by the shapefile format; the full text goes in the CSV and the report.
- **Layers updated in place:** where the user enters parameters per feature and runs the tool again (e.g. Catchments: 2. Time of Concentration), the tool may update the input layer itself instead of writing a new file:
  - it adds its parameter and result fields, keeps the values the user entered, and never overwrites them with defaults;
  - it fills an empty parameter only with a measured value or an explicit default, and says which in the run log;
  - it sets up the attribute form for entering the parameters: drop-down lists (value maps) for coded values, aliases, warnings for required fields that are empty, and the result fields read-only; the form is saved as the layer's default style (.qml next to the file);
  - it refuses a layer that is in edit mode, and lists the features that still need parameters.

### 4.6 CSV files

- UTF-8 with BOM (so Excel shows ² and ³ correctly).
- Where the headings are abbreviated, start with a **column key**: `# Column key` followed by one `# <column>, <definition>` line per column (read in pandas with `comment="#"`).
- Multi-part CSVs are **sectioned**: each part starts with a `[section name]` line, e.g. `[run]`, `[catchments]`.
- Decimal point, no thousands separators (1.2); missing values are empty or "-", never "None" or "nan".

---

## 5. Implementation reference (mayim_tools)

The standard is implemented in shared modules. Reuse them rather than re-coding the values.

| What | Where | How to use it |
|---|---|---|
| Report page, text styles, title page, tables, captions, 16.5 cm pictures | `mayim_tools/_common/docx_report.py` | `Doc(house_style=True)` sets the A4 page (1.0), the text styles and left-aligned tables on the text margin (`table_align = "left"`, the house-style default; `"center"` is still available); then `.title_page(title, purpose, details, subtitle)`, `.heading()`, `.para()`, `.label()`, `.bullet()`, `.numbered()`, `.caption()`, `.table()`, `.picture()` |
| Chart style (fonts, border, grids, palette, exact-size PNG) | `mayim_tools/_common/report_figures.py` | `style = ChartStyle()`; `with style.context(): fig = style.figure(height_cm)` … `style.axes(ax)`; `style.legend(ax)`; `style.png(fig)`. The constants are `CHART_SIZE = 9`, `CHART_COLOURS`, `CHART_GRID_MAJOR` (#808080, 0.75 pt) and `CHART_GRID_MINOR` (#D9D9D9, `CHART_GRID_MINOR_PT` 0.5 pt); `style.axes(ax)` draws the grid, so tools need no axis drawing-order workarounds |
| Map composition (legend with symbol images, scale bar, north arrow) | `mayim_tools/_common/report_maps.py` | `PlanMap`, `figure_rendered_map`, `scale_bar`, `north_arrow`, `image_legend`, `OSM_ATTRIBUTION`, `OSM_LICENCE_NOTE` |
| QGIS map render, legend keys, OSM background | `mayim_tools/_common/qgis_maps.py` | `render_map`, `legend_entries`, `osm_layer`, `osm_has_tiles`, `map_crs_and_extent` |
| Shapefile export, loading layers with styles | `hydrology/catchment_delineation/qgis_outputs.py` | `apply_style`, `save_shapefile`, `load_on_completion`, `make_plan_map` |
| Output names and folders | `mayim_tools/_common/output_files.py` | `OutputSpec(key, label, description, kind, default)` with `.path(folders, project_number)`; `EXTENSIONS`, `clean_name_part`, `output_file_name`, `shapefile_field_names`; `single_output_path()` for single-file tools named after their input |
| Extract tools (prefixed established names, guarded writes, readers) | `mayim_tools/soil/_common/extract_files.py` | `add_output_parameters`, `ExtractFiles` (`name()`, `path()`, `write()`, `raster_writer()`, `actual()`, `finish()`), `find_named_file()`, `FILES_HELP` |
| Layers updated in place with an attribute form | `hydrology/time_of_concentration/time_of_concentration_algorithm.py` | `apply_form(layer)` (value maps, aliases, soft not-null constraints, read-only results) and `save_layer_style(layer)`; the algorithm runs with `NoThreading` because it edits the input layer |
| DEM contours | `hydrology/_common/contours.py` | `dem_contours(dem, geotransform, interval, extent)`; minor and major contours are drawn from two calls (minor interval, then 5 × minor) with the major levels left out of the minor set |

### Minimal matplotlib equivalent (outside mayim_tools)

```python
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties
import matplotlib
from cycler import cycler

COLOURS = ["#FF0000", "#FFC000", "#FFFF00", "#00CC00", "#0000FF", "#00FFFF", "#FF00FF"]
RC = {"font.family": ["Arial Narrow"], "font.size": 9, "text.color": "black",
      "axes.labelsize": 9, "axes.labelweight": "bold", "axes.labelcolor": "black",
      "axes.edgecolor": "black", "axes.linewidth": 0.75,
      "axes.prop_cycle": cycler(color=COLOURS),
      "xtick.color": "black", "ytick.color": "black",
      "xtick.labelsize": 9, "ytick.labelsize": 9,
      "xtick.major.width": 0.75, "ytick.major.width": 0.75,
      "xtick.minor.width": 0.75, "ytick.minor.width": 0.75}

with matplotlib.rc_context(RC):
    fig = Figure(figsize=(16.5 / 2.54, 8.5 / 2.54), dpi=200, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    # ... plot ...
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("black"); s.set_linewidth(0.75)
    ax.minorticks_on(); ax.set_axisbelow(True)
    # solid greys (not semi-transparent black, which some versions draw black)
    ax.grid(True, which="minor", color="#D9D9D9", linewidth=0.5, linestyle="-")
    ax.grid(True, which="major", color="#808080", linewidth=0.75, linestyle="-")
    ax.legend(prop=FontProperties(family="Arial Narrow", size=9, style="italic"),
              frameon=False, loc="best")
    fig.savefig("figure.png", dpi=200, facecolor="white")  # no bbox_inches="tight"
```

### Minimal python-docx page and table placement (outside mayim_tools)

```python
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm

doc = Document()
for sec in doc.sections:  # A4, 16.5 cm text width
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.25)
    sec.top_margin = sec.bottom_margin = Cm(2.5)

t = doc.add_table(rows=1, cols=3)
t.style = "Table Grid"
t.alignment = WD_TABLE_ALIGNMENT.LEFT
ind = OxmlElement("w:tblInd")  # indent by the cell margin: border on the text margin
ind.set(qn("w:w"), "108"); ind.set(qn("w:type"), "dxa")
t._tbl.tblPr.append(ind)  # mayim_tools inserts it in schema order
```

---

## 6. Checklist before delivering

- [ ] A4 page, 2.25 cm side margins (16.5 cm text width).
- [ ] Title page: title Arial 20 pt bold, #0F0F36, centred; short purpose and goal; report details (project number, tool and version, date, status); page break after it.
- [ ] Body Arial 10 pt black; headings Arial 12 pt bold, #0F0F36, all caps; sub-headings Arial 10 pt bold, #0F0F36; captions Arial 9 pt italic.
- [ ] Tables left-aligned with the left border on the text margin, within 16.5 cm (title-page details table centred).
- [ ] No thousands separators anywhere: text, tables, axis labels or CSVs.
- [ ] Every figure is 16.5 cm wide, saved without cropping.
- [ ] Figure text is Arial Narrow 9 pt: regular for text, bold for axis titles, italic for legends.
- [ ] House fonts found (including the per-user Windows font folder); any substitution named in the run log.
- [ ] Black 0.75 pt plot border; major gridlines solid dark grey #808080 0.75 pt, minor gridlines solid light grey #D9D9D9 0.5 pt.
- [ ] Legends clear of the data.
- [ ] Content colours follow the palette order, with no thin yellow lines.
- [ ] Natural ground profiles are black, 1 pt, dashed.
- [ ] Maps:
  - [ ] QML layer styles applied.
  - [ ] OSM background with attribution, or a warning when no tiles are received.
  - [ ] Study area: black dashed 0.5 mm. Contours: minor black 0.15 mm, major (5 × minor interval) black 0.35 mm, or left off when the user unticks them.
  - [ ] Legend below the map, scale bar, north arrow.
- [ ] Output files:
  - [ ] A tick box per output with sensible defaults; a folder per kind; optional project number (and project name for reports).
  - [ ] Named `<project number>_<description>` with underscores; the help lists every file name.
  - [ ] Sorted into the raster, shapefile, CSV, Word (and figure) folders.
  - [ ] A locked file is reported and the other outputs are still written.
  - [ ] Loaded into the project with their styles; shapefile field names at most 10 characters.
  - [ ] CSVs UTF-8 with BOM, with a column key where headings are abbreviated.
  - [ ] Layers updated in place keep the user's entries, have an attribute form, and refuse edit mode.
