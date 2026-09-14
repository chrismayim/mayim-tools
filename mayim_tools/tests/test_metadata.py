"""
Tests for the plugin metadata.txt file.

These tests intentionally avoid importing qgis, so they run in any
Python environment, including outside QGIS.
"""

from __future__ import annotations

import configparser
from pathlib import Path


def _metadata_path() -> Path:
    return Path(__file__).resolve().parents[1] / "metadata.txt"


def test_metadata_file_exists() -> None:
    """metadata.txt must exist alongside the plugin package."""
    assert _metadata_path().is_file()


def test_metadata_has_required_fields() -> None:
    """metadata.txt must define the fields QGIS requires."""
    parser = configparser.ConfigParser()
    parser.read(_metadata_path(), encoding="utf-8")

    assert parser.has_section("general")

    required_fields = ["name", "qgisMinimumVersion", "description", "version"]

    for field in required_fields:
        assert parser.has_option("general", field)
        assert parser.get("general", field).strip() != ""


def test_metadata_name_matches_plugin() -> None:
    """The declared plugin name must match the expected value."""
    parser = configparser.ConfigParser()
    parser.read(_metadata_path(), encoding="utf-8")

    assert parser.get("general", "name") == "Mayim Tools"
