"""
Tests for the Mayim Tools Processing provider.

These tests require the qgis package. When run outside the QGIS Python
environment, they are skipped rather than failing collection.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qgis")


def test_provider_id() -> None:
    """The provider ID must be the fixed, lowercase identifier."""
    from mayim_tools.processing.provider import MayimToolsProvider

    provider = MayimToolsProvider()
    assert provider.id() == "mayimtools"


def test_provider_name() -> None:
    """The provider display name must be human readable."""
    from mayim_tools.processing.provider import MayimToolsProvider

    provider = MayimToolsProvider()
    assert provider.name() == "Mayim Tools"


def test_provider_has_no_algorithms_yet() -> None:
    """The clean-rebuild provider intentionally registers zero algorithms."""
    from mayim_tools.processing.provider import MayimToolsProvider

    provider = MayimToolsProvider()
    provider.loadAlgorithms()
    assert provider.algorithms() == []
