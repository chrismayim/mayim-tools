"""
Word report (Mayim report standard v2.3) for the Import LandXML Surface
tool. No QGIS dependency: the layout maps come from a PlanMapFn supplied
by the QGIS wrapper (qgis_layers.make_plan_map). No thousands separators.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

REPORT_TITLE = "LandXML Survey Data Import"

REFERENCES = [
    "LandXML.org (2008). LandXML-1.2 schema. "
    "http://www.landxml.org/schema/LandXML-1.2/LandXML-1.2.xsd",
    "Autodesk (2026). Civil 3D: exporting LandXML data. Autodesk Help.",
    "Lawson, C.L. (1977). Software for C1 surface interpolation. In: Rice, "
    "J.R. (ed.), Mathematical Software III, Academic Press, 161-194 (linear "
    "interpolation on a triangulated irregular network).",
]


def num(value, nd=2) -> str:
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # NaN
        return "-"
    return f"{v:.{nd}f}"


def _even(n_cols, first):
    rest = (16.5 - first) / max(n_cols - 1, 1)
    return [first] + [rest] * (n_cols - 1)


def write_report(path, data, ctx, plan_map=None) -> list[str]:
    """Writes the report; returns messages for the log (font substitutions,
    maps that could not be drawn). ``ctx``: tool, version, software,
    project_number, project_name, crs, crs_source, cell_size,
    contour_interval, surfaces {name: {"dem": (rows, cols) or None}},
    outputs [(label, path)], warnings."""
    from mayim_tools._common.docx_report import Doc
    from mayim_tools._common.report_figures import ChartStyle
    from mayim_tools._common.report_maps import (
        OSM_LICENCE_NOTE,
        PLAN_DPI,
        PLAN_PLOT_WIDTH_CM,
        figure_rendered_map,
    )

    messages: list[str] = []
    style = ChartStyle()
    messages.extend(style.warnings)
    osm_used = False
    unit = "ft" if data.imperial else "m"
    doc = Doc(house_style=True)
    counts = {k: v for k, v in data.counts().items() if v}
    doc.title_page(
        REPORT_TITLE,
        "This report records the import of a LandXML survey file into QGIS: "
        "the surfaces as TIN meshes and DEM rasters, and the other survey "
        "elements as GIS layers, for use in hydrological and hydraulic "
        "modelling.",
        [
            ("Project number", ctx.get("project_number") or "-"),
            ("Project / site", ctx.get("project_name") or "-"),
            ("Prepared by", "Mayim Consulting"),
            (
                "Tool and version",
                f"{ctx.get('tool', '-')}, version {ctx.get('version', '-')}",
            ),
            ("Software", ctx.get("software") or "-"),
            ("Report date", ctx.get("date") or "-"),
            ("Document status", "Draft"),
        ],
        subtitle=ctx.get("project_name") or None,
    )

    # 1. Summary
    doc.heading("1. Summary", 2)
    parts = [f"{k.lower()} ({v})" for k, v in counts.items()]
    doc.para(
        f"The LandXML file {Path(data.path).name} was read with the "
        f"{ctx.get('tool', '-')} tool. It contains: "
        + (", ".join(parts) if parts else "no supported elements")
        + ". "
        + (
            f"Each surface was written as a TIN mesh and as a DEM raster with a "
            f"{ctx.get('cell_size', 0):g} {unit} cell size."
            if data.surfaces
            else ""
        )
    )
    doc.caption("Table", "Contents of the LandXML file")
    doc.table(
        ["Element", "Number"],
        [[k, str(v)] for k, v in data.counts().items()],
        widths_cm=[8.0, 8.5],
    )

    # 2. Tool and method
    doc.heading("2. Tool and method", 2)
    for step in (
        "The LandXML file is read element by element. LandXML gives coordinates "
        "as northing, easting (and elevation); they are written to the GIS "
        "layers as easting (x), northing (y) and elevation (z).",
        "Each TIN surface is taken as defined in the file (points and "
        "triangular faces); faces marked hidden (for example holes) are left "
        "out. The TIN is written as a 2DM mesh, which QGIS opens as a mesh "
        "layer, without any change.",
        "The DEM raster is sampled from the TIN by linear interpolation within "
        "each triangle at the centre of every cell, so it reproduces the TIN "
        "surface; cells outside the TIN are NoData.",
        "Contours (when selected) are traced from the DEM and smoothed; every "
        "fifth contour is a major contour.",
        "Alignments, feature lines and parcels are built from their coordinate "
        "geometry: lines, circular arcs (densified to at most 2 degrees and "
        "2 m per segment) and spirals (clothoids with linearly varying "
        "curvature, fitted to their start and end points). Pipes are drawn "
        "as straight lines between the centres of their structures.",
    ):
        doc.numbered(step)

    # 3. Inputs and settings
    doc.heading("3. Inputs and settings", 2)
    rows = [
        ["LandXML file", Path(data.path).name],
        ["LandXML version", data.version or "-"],
        ["Exported by", data.application or "-"],
        ["Source drawing", PureWindowsPath(data.project).name if data.project else "-"],
        [
            "Units",
            f"{'Imperial' if data.imperial else 'Metric'} ({data.linear_unit or '-'})",
        ],
        ["Coordinate system", f"{ctx.get('crs', '-')} ({ctx.get('crs_source', '-')})"],
        ["DEM cell size", f"{ctx.get('cell_size', 0):g} {unit}"],
        [
            "Contour interval",
            (
                (
                    f"{ctx['contour_interval']:g} {unit} (major every "
                    f"{5 * ctx['contour_interval']:g} {unit})"
                )
                if ctx.get("contour_interval")
                else "-"
            ),
        ],
    ]
    doc.caption("Table", "Inputs and settings")
    doc.table(["Item", "Value"], rows, widths_cm=[5.0, 11.5])

    # 4. Surfaces
    doc.heading("4. Surfaces", 2)
    if not data.surfaces:
        doc.para("The file contains no TIN surface.")
    for k, s in enumerate(data.surfaces, 1):
        doc.heading(f"4.{k} Surface {s.name}", 3)
        zmin, zmax = s.elev_range
        xmin, ymin, xmax, ymax = s.extent
        dem = (ctx.get("surfaces") or {}).get(s.name, {}).get("dem")
        rows = [
            ["Description", s.desc or "-"],
            ["TIN points", str(len(s.xyz))],
            [
                "TIN triangles",
                str(len(s.triangles))
                + (
                    f" ({s.hidden_faces} hidden faces left out)"
                    if s.hidden_faces
                    else ""
                ),
            ],
            ["Breaklines / boundaries", f"{len(s.breaklines)} / {len(s.boundaries)}"],
            [
                "Plan area (m²)" if not data.imperial else "Plan area (ft²)",
                num(s.computed_area_2d(), 1),
            ],
            [
                "Surface area (m²)" if not data.imperial else "Surface area (ft²)",
                num(s.computed_area_3d(), 1),
            ],
            [f"Elevation range ({unit})", f"{num(zmin)} to {num(zmax)}"],
            [f"Extent, easting ({unit})", f"{num(xmin)} to {num(xmax)}"],
            [f"Extent, northing ({unit})", f"{num(ymin)} to {num(ymax)}"],
            ["DEM grid (rows x columns)", f"{dem[0]} x {dem[1]}" if dem else "-"],
        ]
        doc.caption("Table", f"Surface {s.name}")
        doc.table(["Parameter", "Value"], rows, widths_cm=[6.0, 10.5])
        if plan_map is not None:
            try:
                pm = plan_map(s.name, PLAN_PLOT_WIDTH_CM, PLAN_DPI)
            except Exception as e:
                pm = None
                messages.append(f"Layout map of surface {s.name} not drawn ({e}).")
            if pm is not None:
                osm_used = osm_used or bool(pm.attribution)
                doc.picture(
                    figure_rendered_map(pm, style),
                    f"Surface {s.name}: elevation, TIN boundary and survey elements"
                    + (" (background: OpenStreetMap)" if pm.attribution else ""),
                )

    # 5. Other survey elements
    others = [
        ("Survey points", data.points),
        ("Alignments", data.alignments),
        ("Feature lines", data.feature_lines),
        ("Parcels", data.parcels),
        ("Pipe network structures", data.structures),
        ("Pipe network pipes", data.pipes),
    ]
    if any(items for _n, items in others):
        doc.heading("5. Other survey elements", 2)
        if data.points:
            groups: dict[str, int] = {}
            for p in data.points:
                for g in p.groups or ["(no group)"]:
                    groups[g] = groups.get(g, 0) + 1
            doc.caption("Table", "Survey points by point group")
            doc.table(
                ["Point group", "Points"],
                [[g, str(n)] for g, n in groups.items()],
                widths_cm=[10.0, 6.5],
            )
        if data.alignments:
            doc.caption("Table", "Alignments")
            doc.table(
                ["Alignment", f"Length ({unit})", "Start station"],
                [
                    [
                        a.name,
                        num(a.attributes.get("length")),
                        num(a.attributes.get("sta_start")),
                    ]
                    for a in data.alignments
                ],
                widths_cm=_even(3, 8.5),
            )
        if data.parcels:
            doc.caption("Table", "Parcels")
            doc.table(
                ["Parcel", "Site", f"Area ({unit}²)"],
                [
                    [
                        p.name,
                        p.attributes.get("site") or "-",
                        num(p.attributes.get("area"), 1),
                    ]
                    for p in data.parcels
                ],
                widths_cm=_even(3, 6.5),
            )
        if data.structures or data.pipes:
            nets: dict[str, list] = {}
            for s in data.structures:
                nets.setdefault(s.attributes.get("network", ""), [0, 0, 0.0])[0] += 1
            for p in data.pipes:
                row = nets.setdefault(p.attributes.get("network", ""), [0, 0, 0.0])
                row[1] += 1
                row[2] += p.attributes.get("length") or 0.0
            doc.caption("Table", "Pipe networks")
            doc.table(
                ["Pipe network", "Structures", "Pipes", f"Pipe length ({unit})"],
                [
                    [n or "-", str(v[0]), str(v[1]), num(v[2], 1)]
                    for n, v in nets.items()
                ],
                widths_cm=_even(4, 7.0),
            )
        if data.feature_lines:
            doc.para(
                f"{len(data.feature_lines)} feature line(s) were written as 3D "
                "lines."
            )
        if plan_map is not None and not data.surfaces:
            try:
                pm = plan_map(None, PLAN_PLOT_WIDTH_CM, PLAN_DPI)
            except Exception as e:
                pm = None
                messages.append(f"Layout map not drawn ({e}).")
            if pm is not None:
                osm_used = osm_used or bool(pm.attribution)
                doc.picture(
                    figure_rendered_map(pm, style),
                    "Survey elements"
                    + (" (background: OpenStreetMap)" if pm.attribution else ""),
                )

    # 6. Outputs
    n = 6 if any(items for _n, items in others) else 5
    outputs = ctx.get("outputs") or []
    doc.heading(f"{n}. Outputs", 2)
    if outputs:
        doc.caption("Table", "Files written")
        doc.table(
            ["Output", "File"],
            [[a, Path(b).name] for a, b in outputs],
            widths_cm=[6.0, 10.5],
        )
        folders = sorted({str(Path(b).parent) for _a, b in outputs})
        doc.para("Folder(s): " + "; ".join(folders))
    else:
        doc.para("No other files were written.")

    # 7. Notes
    doc.heading(f"{n + 1}. Notes and limitations", 2)
    for note in (
        "The DEM reproduces the TIN; it adds no detail between the survey "
        "points. Choose a cell size small enough to keep the breaklines.",
        "Alignment profiles (vertical geometry), cross-sections and corridor "
        "assemblies are not imported. A Civil 3D corridor is exported to "
        "LandXML as its surfaces, alignments and feature lines, which are.",
        "Spirals are reconstructed from their length and radii and fitted to "
        "their start and end points.",
        "Check that the layers line up with known features: LandXML gives "
        "northing before easting.",
    ):
        doc.bullet(note)
    if osm_used:
        doc.bullet(OSM_LICENCE_NOTE)
    msgs = list(dict.fromkeys(list(data.warnings) + list(ctx.get("warnings", []))))
    if msgs:
        doc.label("Processing messages")
        for m in msgs:
            doc.bullet(m)

    doc.heading(f"{n + 2}. References", 2)
    for ref in REFERENCES:
        doc.para(ref)
    doc.doc.save(str(path))
    return messages
