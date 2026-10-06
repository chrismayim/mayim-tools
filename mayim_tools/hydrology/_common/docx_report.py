"""Compatibility shim: the Word report helpers now live in
mayim_tools/_common/docx_report.py (shared by every tool group)."""

from mayim_tools._common.docx_report import (  # noqa: F401
    C_AQUA,
    C_BLUE,
    C_GRID,
    C_INK,
    C_INK_2,
    C_ORANGE,
    FIG_WIDTH_CM,
    Doc,
    format_value,
    new_figure,
    png,
    style_axes,
)
