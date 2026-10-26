"""Output folders and file names of the extraction tools (Extract:
SoilGrids 2.0, OpenLandMap Soils, iSDAsoil and Land Cover), in the Mayim
output standard:

- area mode writes the rasters, the run's metadata CSV and the .qlr layer
  file to the raster folder (the metadata stays beside the rasters it
  documents; Regional Soil Parameterisation reads it from there);
- point mode writes the point CSV and its metadata to the CSV folder;
- every file is named ``<project number>_<name>`` (just ``<name>`` without
  a project number, so names are unchanged from earlier versions), where
  ``<name>`` is the tool's own file name, e.g.
  ``MC2025_014_soilgrids_clay_Q0.50.tif``;
- CSVs are UTF-8 with a byte-order mark;
- a file that cannot be replaced (open in QGIS or Excel) is reported and
  the other outputs are still written.

Downstream tools find these files by band description or by a
case-insensitive name ending, so the project-number prefix does not get in
their way (see :func:`find_named_file`).
"""

from __future__ import annotations

import os
from pathlib import Path

from mayim_tools._common.output_files import clean_name_part

CSV_ENCODING = "utf-8-sig"
PROJECT_NUMBER = "PROJECT_NUMBER"
FOLDER_RASTER = "FOLDER_RASTER"
FOLDER_CSV = "FOLDER_CSV"


def find_named_file(folder: str, name: str) -> str | None:
    """The file in ``folder`` called ``name`` or ``<anything>_<name>``
    (case-insensitive; the newest when several runs share the folder), or
    None. Lets readers find extraction outputs with or without a
    project-number prefix."""
    if not folder or not os.path.isdir(folder):
        return None
    target = name.lower()
    hits = []
    for entry in os.listdir(folder):
        low = entry.lower()
        if low == target or low.endswith("_" + target):
            path = os.path.join(folder, entry)
            if os.path.isfile(path):
                hits.append(path)
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def add_output_parameters(alg, area_label="rasters, metadata and layer file"):
    """Project number and the raster / CSV folders (QGIS parameters)."""
    from qgis.core import (
        QgsProcessingParameterFolderDestination,
        QgsProcessingParameterString,
    )

    alg.addParameter(
        QgsProcessingParameterString(
            PROJECT_NUMBER, "Project number (file name prefix, optional)",
            optional=True,
        )
    )  # fmt: skip
    alg.addParameter(
        QgsProcessingParameterFolderDestination(
            FOLDER_RASTER, f"Output folder for area mode ({area_label})",
            optional=True,
        )
    )  # fmt: skip
    alg.addParameter(
        QgsProcessingParameterFolderDestination(
            FOLDER_CSV, "Output folder for point mode (CSV files)", optional=True
        )
    )


FILES_HELP = (
    "FILES:\tArea mode writes to the raster folder, point mode to the CSV "
    "folder. Files are named <project number>_<name> (just <name> without a "
    "project number) and replace files of the same name; a file that is open "
    "in QGIS or Excel cannot be replaced and is reported, and the other "
    "outputs are still written. CSVs are UTF-8 with a byte-order mark. Later "
    "tools (Regional Soil Parameterisation, Hydrological Soil Groups) find the "
    "files with or without the project number."
)


class ExtractFiles:
    """Resolves the folder and names for one run and writes guarded."""

    def __init__(self, alg, parameters, context, feedback, area: bool):
        from qgis.core import QgsProcessingException

        self.feedback = feedback
        self.project_number = clean_name_part(
            alg.parameterAsString(parameters, PROJECT_NUMBER, context) or ""
        )
        key, label = (
            (FOLDER_RASTER, "output folder for area mode")
            if area
            else (FOLDER_CSV, "output folder for point mode")
        )
        folder = alg.parameterAsFileOutput(parameters, key, context)
        if not folder:
            raise QgsProcessingException(f"Choose the {label}.")
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.key = key
        self.written: list[str] = []
        self.failed: dict[str, str] = {}
        self._map: dict[str, str] = {}

    def name(self, name: str) -> str:
        return f"{self.project_number}_{name}" if self.project_number else name

    def path(self, name: str) -> str:
        return os.path.join(self.folder, self.name(name))

    def report(self, path: str, error) -> None:
        self.failed[path] = str(error)
        self.feedback.reportError(
            f"{Path(path).name} could not be written ({error}). If it is open in "
            "QGIS or Excel, close it and run again.",
            fatalError=False,
        )

    def write(self, path: str, writer) -> bool:
        """``writer(path)``; a failure is reported and False returned."""
        try:
            writer(path)
        except Exception as e:  # locked or not writable
            self.report(path, e)
            return False
        self.written.append(path)
        return True

    def raster_writer(self, write_fn):
        """Wraps the tool's ``write_fn(path, ...)``: the core names the file
        in the raster folder; the project number is added here and a file
        that cannot be written is reported instead of stopping the run."""

        def wrapped(path, *args, **kwargs):
            new = os.path.join(os.path.dirname(path), self.name(os.path.basename(path)))
            try:
                write_fn(new, *args, **kwargs)
            except Exception as e:  # locked (loaded in QGIS) or not writable
                self.report(new, e)
                return
            self._map[str(path)] = new
            self.written.append(new)

        return wrapped

    def actual(self, path) -> str | None:
        """Where the core's ``path`` was written (None if it failed)."""
        return self._map.get(str(path))

    def finish(self, result: dict) -> dict:
        from qgis.core import QgsProcessingException

        if not self.written:
            raise QgsProcessingException(
                "No output could be written - see the errors above."
            )
        result[self.key] = self.folder
        return result
