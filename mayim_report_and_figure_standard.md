# Mayim Report, Figure and Graph Standard

*House design standard for every Word report, graph, chart and map produced by Mayim tools and analyses (version 1, October 2026).*

**How to use this document:** give it to Claude in any chat with the instruction "apply the Mayim report and figure standard". Every value below is a requirement unless it is marked as a default. If something is not covered here, follow the closest rule and say which choice was made.

---
Font availability: Report figures must use the house fonts (Arial; Arial Narrow for chart text). If a font is installed in Windows but not seen by the figure software, the tool must find it in the Windows font folders, including the per-user folder (%LOCALAPPDATA%\Microsoft\Windows\Fonts), before falling back to a substitute; any substitution must be named in the run log.

## 1. Word reports (.docx)

### 1.1 Text styles

| Element | Font | Size | Weight / style | Colour | Other |
|---|---|---|---|---|---|
| Normal (body) text | Arial | 10 pt | Regular | Black #000000 | Also used for bullets, numbered lists and table text (tables use 9 pt, see 1.3) |
| Headings (levels 1 and 2: report title and main sections) | Arial | 12 pt | Bold, not italic | #0F0F36 | ALL CAPS (set as the font's All caps property, so the text itself keeps normal case) |
| Sub-headings (level 3 and below, plus run-in labels such as "Geomorphological description") | Arial | 10 pt | Bold, not italic | #0F0F36 | Normal case |
| Table and figure headings (captions) | Arial | 9 pt | Italic | Black | "Table n: text" above the table; "Figure n: text" below the figure |

- Set the fonts on the **styles** (Normal, Heading 1–4, Caption) rather than on individual runs. Set them for every script (ascii, hAnsi, eastAsia, cs) and remove the theme-font links, otherwise Word falls back to Calibri or Aptos.
- Remove theme colours from the heading styles so that #0F0F36 is used exactly.
- Keep each table heading with its table, and each figure with its caption (keep-with-next).
- Number tables and figures consecutively through the report.

### 1.2 Numbers and units

- **No thousands separators:** write 424627, not 424,627 or 424 627.
- Use a decimal point. Use a sensible, consistent number of decimals per quantity, e.g. areas in km² to 3–4 decimals, elevations to 2, slopes in % to 2–3, times to 1.
- Use SI units with proper symbols: km², m/m, %, m³/s and so on. Write units in brackets in labels and column headings, e.g. "Area (km²)".
- When a value is missing or cannot be computed, show "-", never "None", "nan" or a blank.

### 1.3 Tables

- Use the "Table Grid" style, centred on the page, with column widths set explicitly so the table fits the 16.5 cm text width.
- **Header row:** 9 pt bold, shaded #ADADCC.
- **Body:** 9 pt regular.
- **Group sub-heading rows** (when a table is split into groups): merged across the full width, 9 pt bold, shaded #F2F2F2.
- **Parameter tables:** use three columns, Parameter (with symbol and unit) | Value | Definition/equation (with source), at widths of about 6.2 / 2.8 / 7.5 cm.
- Leave one empty paragraph after each table.

### 1.4 Report structure (default for tool annexures)

1. Title (Heading 1). There is **no title block or cover page**, because reports are used as annexures.
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

- If Arial Narrow is not installed, substitute a metric-compatible font (Liberation Sans Narrow, then Arial) and report the substitution to the user.

### 2.3 Plot area, axes and grid

- **Plot area border:** black, 0.75 pt, on all four sides.
- **Ticks:** black, 0.75 pt wide, with minor ticks on.
- **Major gridlines:** solid, 0.75 pt, #BFBFBF.
- **Minor gridlines:** solid, 0.75 pt, #F2F2F2.
- Draw gridlines behind the data.
- Use plain number formatting, with no offset or scientific notation on coordinate or elevation axes, and no thousands separators.
- Show units in the axis titles: "Elevation (m)", "Distance upstream from outlet (km)".
- Legends have no frame on graphs. On maps they sit below the map (see 3.3).

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
- If QML styles or not known or specified, request it from the user.
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
- **Contours:** generated from the DEM at a user-set interval (default 1 m, 0 = none). Black, 0.2 mm, solid.
- **Layers** use the Mayim QML layer styles (folder `D:\Dropbox\MAYIM\6 TEGNIESE DATA EN DOKUMENTE\QGIS\QGIS Layer styles`). Copies are kept in each tool's `styles/` folder.

| Layer | Style file | Key field |
|---|---|---|
| Catchments / watersheds | DRN_watershed.qml | – |
| Catchment centroids | DRN_centroids.qml | – |
| Longest flow paths | DRN_longest_flowpath.qml | – |
| Outlets / pour points | DRN_outlets.qml | – |
| Streams by Strahler order | DRN_strahler_stream_order.qml | `strahler` |
| Stream nodes | DRN_stream_nodes.qml | `node_type` |

- **Drawing order** (top to bottom): centroids, outlets, stream nodes, flow paths, streams, catchments, study area, contours, OSM.

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

## 4. Output files from tools

- **Selection:** a tick box for each output, so the user selects which outputs to generate.
- **Folders:** one per kind, chosen by the user: rasters (GeoTIFF .tif), vector layers (shapefile .shp), CSV (.csv) and Word reports (.docx).
- **Names:** `<project number>_<layer description>.<ext>`, with underscores throughout and no spaces, e.g. `MC2025_014_Catchments.shp` or `MC2025_014_Catchment_Report.docx`. The project number is optional; without it the name is just the description.
  - Characters that are not allowed in Windows file names are removed.
- **Existing files** with the same name are overwritten. If a file is locked (open in QGIS), report it and still write the other outputs.
- **Loading:** written layers are loaded into the QGIS project with their QML style.
- **Shapefile field names:** at most 10 characters. Shorten longer ones explicitly and list them in the help text.
- **CSV:** UTF-8 with BOM (so Excel shows ² and ³ correctly). Where the headings are abbreviated, put an abbreviation key at the top.

---

## 5. Implementation reference (mayim_tools)

The standard is implemented in shared modules. Reuse them rather than re-coding the values.

| What | Where | How to use it |
|---|---|---|
| Report text styles, tables, captions, 16.5 cm pictures | `mayim_tools/_common/docx_report.py` | `Doc(house_style=True)`, then `.heading()`, `.para()`, `.label()`, `.bullet()`, `.numbered()`, `.caption()`, `.table()`, `.picture()` |
| Chart style (fonts, border, grids, palette, exact-size PNG) | `mayim_tools/_common/report_figures.py` | `style = ChartStyle()`; `with style.context(): fig = style.figure(height_cm)` … `style.axes(ax)`; `style.legend(ax)`; `style.png(fig)`. The constants are `CHART_SIZE = 9`, `CHART_COLOURS`, `CHART_GRID_MAJOR` and `CHART_GRID_MINOR` |
| Map composition (legend with symbol images, scale bar, north arrow) | `hydrology/catchment_delineation/report.py` | `PlanMap`, `_figure_plan_rendered`, `_scale_bar`, `_north_arrow`, `_image_legend` (candidates to move to `_common` when a second tool needs them) |
| QGIS map render, legend keys, QML loading, shapefile export | `hydrology/catchment_delineation/qgis_outputs.py` | `render_map`, `legend_entries`, `apply_style`, `save_shapefile`, `load_on_completion`, `make_plan_map` |
| Output names and folders | `mayim_tools/_common/output_files.py` | `OutputSpec`, `output_file_name`, `shapefile_field_names` |
| DEM contours | `hydrology/_common/contours.py` | `dem_contours(dem, geotransform, interval, extent)` |

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
    ax.grid(True, which="major", color="#BFBFBF", linewidth=0.75, linestyle="-")
    ax.grid(True, which="minor", color="#F2F2F2", linewidth=0.75, linestyle="-")
    ax.legend(prop=FontProperties(family="Arial Narrow", size=9, style="italic"),
              frameon=False)
    fig.savefig("figure.png", dpi=200, facecolor="white")  # no bbox_inches="tight"
```

---

## 6. Checklist before delivering

- [ ] Body Arial 10 pt black; headings Arial 12 pt bold, #0F0F36, all caps; sub-headings Arial 10 pt bold, #0F0F36; captions Arial 9 pt italic.
- [ ] No thousands separators anywhere: text, tables, axis labels or CSVs.
- [ ] Every figure is 16.5 cm wide, saved without cropping.
- [ ] Figure text is Arial Narrow 9 pt: regular for text, bold for axis titles, italic for legends.
- [ ] Black 0.75 pt plot border; major and minor gridlines in #BFBFBF and #F2F2F2.
- [ ] Content colours follow the palette order, with no thin yellow lines.
- [ ] Natural ground profiles are black, 1 pt, dashed.
- [ ] Maps:
  - [ ] QML layer styles applied.
  - [ ] OSM background with attribution, or a warning when no tiles are received.
  - [ ] Study area: black dashed 0.5 mm. Contours: black 0.2 mm.
  - [ ] Legend below the map, scale bar, north arrow.
- [ ] Output files:
  - [ ] Named `<project number>_<description>` with underscores.
  - [ ] Sorted into the raster, shapefile, CSV and Word folders.
  - [ ] Loaded into the project with their styles.
