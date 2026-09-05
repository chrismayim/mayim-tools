"""
Mayim Tools - File I/O utilities.
"""

from __future__ import annotations

import csv
import json
from typing import Any, ClassVar

from mayim_tools.core.logger import MayimLogger


class FileIO:
    """
    Shared file read/write helpers.
    """

    FORMATS: ClassVar[dict[str, str]] = {
        "gpkg": "GPKG",
        "shp": "ESRI Shapefile",
        "geojson": "GeoJSON",
        "csv": "CSV",
    }

    @staticmethod
    def read_json(path: str) -> dict[str, Any] | list[Any] | None:
        """
        Read a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON file.

        Returns
        -------
        dict[str, Any] | list[Any] | None
            Parsed JSON object, or None on failure.
        """
        try:
            with open(path, encoding="utf-8") as file:
                return json.load(file)
        except Exception as error:  # noqa: BLE001
            MayimLogger.critical(f"Failed to read JSON: {error}")
            return None

    @staticmethod
    def write_json(
        path: str,
        data: dict[str, Any] | list[Any],
    ) -> bool:
        """
        Write a JSON file.

        Parameters
        ----------
        path : str
            Output path.
        data : dict[str, Any] | list[Any]
            JSON-serialisable object.

        Returns
        -------
        bool
            True on success, False on failure.
        """
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)
            return True
        except Exception as error:  # noqa: BLE001
            MayimLogger.critical(f"Failed to write JSON: {error}")
            return False

    @staticmethod
    def read_csv(path: str) -> list[dict[str, str]] | None:
        """
        Read a CSV file as a list of dictionaries.

        Parameters
        ----------
        path : str
            Path to the CSV file.

        Returns
        -------
        list[dict[str, str]] | None
            CSV rows as dictionaries, or None on failure.
        """
        try:
            with open(path, encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                return list(reader)
        except Exception as error:  # noqa: BLE001
            MayimLogger.critical(f"Failed to read CSV: {error}")
            return None
