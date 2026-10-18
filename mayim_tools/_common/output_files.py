"""
Output file naming and folder routing shared by tools that write their
results to user-chosen folders (raster, shapefile, CSV and Word folders)
instead of one file destination per output. Zero QGIS dependency.

File names are ``<project number>_<description>.<ext>`` with underscores
throughout, e.g. ``MC2025_014_Catchments.shp``; without a project number
just ``Catchments.shp``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# folder kinds
RASTER = "raster"
SHAPEFILE = "shapefile"
CSV = "csv"
WORD = "word"

EXTENSIONS = {RASTER: ".tif", SHAPEFILE: ".shp", CSV: ".csv", WORD: ".docx"}

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def clean_name_part(text: str) -> str:
    """Spaces (and runs of them) become single underscores; characters that
    Windows does not allow in file names are dropped; leading/trailing
    underscores and dots are trimmed."""
    text = _INVALID.sub("", str(text or "").strip())
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_.")


def output_file_name(project_number: str | None, description: str, kind: str) -> str:
    """File name (with extension) for an output of the given folder kind."""
    parts = [clean_name_part(project_number), clean_name_part(description)]
    stem = "_".join(p for p in parts if p)
    if not stem:
        raise ValueError("An output needs a description to be named.")
    return stem + EXTENSIONS[kind]


@dataclass(frozen=True)
class OutputSpec:
    """One selectable output: its key, the label shown in the dialog, the
    description used in the file name, its folder kind and whether it is
    ticked by default."""

    key: str
    label: str
    description: str
    kind: str
    default: bool = True

    def path(self, folders: dict[str, str], project_number: str | None) -> str:
        folder = folders.get(self.kind)
        if not folder:
            raise ValueError(f"No {self.kind} folder was given for '{self.label}'.")
        return str(
            Path(folder) / output_file_name(project_number, self.description, self.kind)
        )


def shapefile_field_names(names: list[str]) -> list[str]:
    """Unique shapefile (dBASE) field names of at most 10 characters: names
    that already fit are kept, longer ones are shortened and, if that
    collides, numbered."""
    used: set[str] = set()
    out = []
    for name in names:
        candidate = name[:10]
        k = 1
        while candidate.lower() in used:
            suffix = f"_{k}"
            candidate = name[: 10 - len(suffix)] + suffix
            k += 1
        used.add(candidate.lower())
        out.append(candidate)
    return out
