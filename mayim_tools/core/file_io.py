# -*- coding: utf-8 -*-
"""
Mayim Tools – File IO Utilities
Helpers for reading and writing common geospatial and data file formats.
"""

import csv
import json
from pathlib import Path
from typing import Optional

from qgis.core import (
    QgsCoordinateTransformContext,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from mayim_tools.core.logger import MayimLogger


class FileIO:
    """Static utility class for file input/output operations."""

    # ── Supported vector formats ──
    FORMATS = {
        "gpkg": "GPKG",
        "shp": "ESRI Shapefile",
        "geojson": "GeoJSON",
        "csv": "CSV",
    }

    @staticmethod
    def read_json(file_path: str) -> Optional[dict]:
        """
        Read a JSON file and return its contents as a dictionary.

        :param file_path: Full path to the JSON file
        :returns: Parsed dictionary or None on failure
        """
        path = Path(file_path)
        if not path.exists():
            MayimLogger.warning(f"File not found: {file_path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            MayimLogger.critical(f"Failed to read JSON: {e}")
            return None

    @staticmethod
    def write_json(file_path: str, data: dict) -> bool:
        """
        Write a dictionary to a JSON file.

        :param file_path: Full path to write to
        :param data: Dictionary to serialise
        :returns: True on success, False on failure
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            MayimLogger.critical(f"Failed to write JSON: {e}")
            return False

    @staticmethod
    def read_csv(file_path: str) -> Optional[list[dict]]:
        """
        Read a CSV file and return rows as a list of dictionaries.

        :param file_path: Full path to the CSV file
        :returns: List of row dictionaries or None on failure
        """
        path = Path(file_path)
        if not path.exists():
            MayimLogger.warning(f"File not found: {file_path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            MayimLogger.critical(f"Failed to read CSV: {e}")
            return None

    @staticmethod
    def save_vector_layer(
        layer: QgsVectorLayer,
        output_path: str,
        format_ext: str = "gpkg",
    ) -> bool:
        """
        Save a vector layer to disk in the specified format.

        :param layer: QgsVectorLayer to save
        :param output_path: Full output file path
        :param format_ext: File format extension (gpkg, shp, geojson)
        :returns: True on success, False on failure
        """
        driver = FileIO.FORMATS.get(format_ext.lower(), "GPKG")
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = driver
        options.fileEncoding = "UTF-8"

        error, msg, _, _ = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer,
            output_path,
            QgsCoordinateTransformContext(),
            options,
        )

        if error == QgsVectorFileWriter.NoError:
            MayimLogger.success(f"Layer saved to: {output_path}")
            return True
        else:
            MayimLogger.critical(f"Failed to save layer: {msg}")
            return False
