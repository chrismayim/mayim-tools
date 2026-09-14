"""
Shared pytest fixtures and configuration for Mayim Tools tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]
