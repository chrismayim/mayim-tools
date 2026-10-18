"""
House style for report-ready figures (Mayim Consulting reports) - shared by
every tool that writes figures meant to be placed straight into a Word
report. Zero QGIS dependency; matplotlib's object API only (no pyplot), so
it is safe on QGIS's Processing worker thread.

- Width: exactly 16.5 cm (figures are saved without bbox cropping, so the
  file is the size it will be placed at in the report - no rescaling).
- Fonts: Arial for main text (axis titles, legend, labels and values),
  Arial Narrow for minor text (tick labels, notes). When a font is not
  installed, a metric-compatible substitute is used and reported.
"""

from __future__ import annotations

import os
from pathlib import Path

REPORT_FIG_WIDTH_CM = 16.5
MAIN_FONT = "Arial"
MINOR_FONT = "Arial Narrow"
MAIN_SIZE = 9.0  # pt
MINOR_SIZE = 8.0  # pt
PNG_DPI = 300

_FALLBACKS = {
    "Arial": ["Liberation Sans", "Arimo", "Helvetica", "DejaVu Sans"],
    "Arial Narrow": [
        "Liberation Sans Narrow",
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
    ],
}

CM = 1 / 2.54  # inches per cm


def _installed_families() -> set[str]:
    from matplotlib import font_manager

    return {f.name for f in font_manager.fontManager.ttflist}


def _font_dirs() -> list[str]:
    """System font folders, including the per-user Windows folder where
    Office and user-installed fonts (e.g. Arial Narrow) often live."""
    dirs = []
    for env, sub in (
        ("WINDIR", "Fonts"),
        ("LOCALAPPDATA", os.path.join("Microsoft", "Windows", "Fonts")),
    ):
        base = os.environ.get(env)
        if base and os.path.isdir(os.path.join(base, sub)):
            dirs.append(os.path.join(base, sub))
    return dirs


_REGISTER_TRIED: set[str] = set()


def _register_font(name: str) -> bool:
    """Adds the font files of family ``name`` to matplotlib when its font
    cache does not list them (a cache built before the font was installed,
    or a font in the per-user Windows font folder). True when found."""
    if name in _REGISTER_TRIED:
        return False
    _REGISTER_TRIED.add(name)
    try:
        from matplotlib import font_manager
        from matplotlib.ft2font import FT2Font
    except ImportError:  # pragma: no cover
        return False
    found = False
    files = set(font_manager.findSystemFonts(fontext="ttf"))
    for folder in _font_dirs():
        files.update(font_manager.findSystemFonts(fontpaths=[folder], fontext="ttf"))
    for path in sorted(files):
        try:
            if FT2Font(path).family_name != name:
                continue
            font_manager.fontManager.addfont(path)
            found = True
        except Exception:  # unreadable font file - skip it
            continue
    return found


def resolve_font(name: str) -> tuple[str, str | None]:
    """(family to use, warning or None)."""
    families = _installed_families()
    if name in families:
        return name, None
    if _register_font(name):
        return name, None
    for alt in _FALLBACKS.get(name, []):
        if alt in families:
            return alt, f"Font '{name}' is not installed - '{alt}' was used instead."
    return (
        "sans-serif",
        f"Font '{name}' is not installed - the default sans-serif font was used.",
    )


class FigureFonts:
    """Resolved main/minor fonts plus any substitution warnings."""

    def __init__(self, main: str = MAIN_FONT, minor: str = MINOR_FONT):
        self.main, w1 = resolve_font(main)
        self.minor, w2 = resolve_font(minor)
        self.warnings = [w for w in (w1, w2) if w]

    def main_kw(self, size: float = MAIN_SIZE, **kw) -> dict:
        return {"fontfamily": self.main, "fontsize": size, **kw}

    def minor_kw(self, size: float = MINOR_SIZE, **kw) -> dict:
        return {"fontfamily": self.minor, "fontsize": size, **kw}


def new_report_figure(height_cm: float, width_cm: float = REPORT_FIG_WIDTH_CM):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width_cm * CM, height_cm * CM), dpi=PNG_DPI)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("white")
    return fig


def save_report_figure(fig, path_stem: str | Path, formats=("png", "svg")) -> list[str]:
    """Saves the figure at its exact size (no tight cropping). SVG keeps
    text as text (font names, not outlines), so Word renders it in Arial."""
    import matplotlib

    written = []
    for fmt in formats:
        path = str(Path(path_stem).with_suffix("." + fmt))
        with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
            fig.savefig(path, format=fmt, dpi=PNG_DPI, facecolor="white")
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Chart house style - graphs and maps placed in the Word reports
# ---------------------------------------------------------------------------
#   all text       Arial Narrow 9 pt black (tick labels, notes)
#   legend         Arial Narrow 9 pt black italic
#   axis titles    Arial Narrow 9 pt bold
#   plot area      black 0.75 pt border on all four sides
#   gridlines      major solid 0.75 pt #BFBFBF, minor solid 0.75 pt #F2F2F2
#   content        CHART_COLOURS in this order
#   width          16.5 cm, saved at that exact size (height to suit content)

CHART_FONT = "Arial Narrow"
CHART_SIZE = 9.0  # pt
CHART_INK = "#000000"
CHART_COLOURS = [
    "#FF0000",
    "#FFC000",
    "#FFFF00",
    "#00CC00",
    "#0000FF",
    "#00FFFF",
    "#FF00FF",
]
CHART_LINE_PT = 0.75  # border and gridlines
CHART_GRID_MAJOR = "#BFBFBF"
CHART_GRID_MINOR = "#F2F2F2"
CHART_DPI = 200


class ChartStyle:
    """Applies the chart house style. Use ``with style.context():`` around
    building and saving a figure, then ``style.axes(ax)`` and
    ``style.legend(ax)`` on each axes."""

    def __init__(self, font: str = CHART_FONT):
        self.font, warning = resolve_font(font)
        self.warnings = [warning] if warning else []

    def rc(self) -> dict:
        from cycler import cycler

        return {
            "font.family": [self.font],
            "font.size": CHART_SIZE,
            "font.style": "normal",
            "text.color": CHART_INK,
            "axes.labelcolor": CHART_INK,
            "axes.labelsize": CHART_SIZE,
            "axes.labelweight": "bold",
            "axes.titlesize": CHART_SIZE,
            "axes.titleweight": "bold",
            "axes.edgecolor": CHART_INK,
            "axes.linewidth": CHART_LINE_PT,
            "axes.prop_cycle": cycler(color=CHART_COLOURS),
            "xtick.color": CHART_INK,
            "ytick.color": CHART_INK,
            "xtick.labelsize": CHART_SIZE,
            "ytick.labelsize": CHART_SIZE,
            "xtick.major.width": CHART_LINE_PT,
            "ytick.major.width": CHART_LINE_PT,
            "xtick.minor.width": CHART_LINE_PT,
            "ytick.minor.width": CHART_LINE_PT,
            "legend.fontsize": CHART_SIZE,
            "legend.labelcolor": CHART_INK,
        }

    def context(self):
        import matplotlib

        return matplotlib.rc_context(self.rc())

    def figure(self, height_cm: float, width_cm: float = REPORT_FIG_WIDTH_CM):
        fig = new_report_figure(height_cm, width_cm)
        fig.set_dpi(CHART_DPI)
        fig.set_layout_engine("constrained")
        return fig

    def font_props(self, weight: str = "normal", style: str = "normal"):
        from matplotlib.font_manager import FontProperties

        return FontProperties(
            family=self.font, size=CHART_SIZE, weight=weight, style=style
        )

    def axes(self, ax, grid: bool = True, minor: bool = True) -> None:
        """Border on all four sides, black ticks and text, bold axis
        titles, major/minor gridlines behind the content."""
        for side in ("top", "right", "bottom", "left"):
            spine = ax.spines[side]
            spine.set_visible(True)
            spine.set_color(CHART_INK)
            spine.set_linewidth(CHART_LINE_PT)
        ax.tick_params(
            which="both", colors=CHART_INK, labelcolor=CHART_INK, width=CHART_LINE_PT
        )
        ax.tick_params(which="major", labelsize=CHART_SIZE)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontproperties(self.font_props(weight="bold"))
            label.set_color(CHART_INK)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(self.font_props())
        for text in ax.texts:
            text.set_fontfamily(self.font)
            text.set_fontsize(CHART_SIZE)
            text.set_color(CHART_INK)
        ax.set_axisbelow(True)
        if grid:
            if minor:
                ax.minorticks_on()
                ax.grid(
                    True,
                    which="minor",
                    color=CHART_GRID_MINOR,
                    linewidth=CHART_LINE_PT,
                    linestyle="-",
                )
            ax.grid(
                True,
                which="major",
                color=CHART_GRID_MAJOR,
                linewidth=CHART_LINE_PT,
                linestyle="-",
            )
        else:
            ax.grid(False, which="both")

    def legend(self, ax, **kwargs):
        kwargs.setdefault("frameon", False)
        legend = ax.legend(prop=self.font_props(style="italic"), **kwargs)
        for text in legend.get_texts():
            text.set_color(CHART_INK)
        return legend

    def png(self, fig):
        """PNG at the figure's exact size (no cropping, so it is placed in
        Word at 16.5 cm without rescaling)."""
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=CHART_DPI, facecolor="white")
        buf.seek(0)
        return buf
