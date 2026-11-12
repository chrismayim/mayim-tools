"""Tests for _common.credentials (no QGIS needed for these)."""

from __future__ import annotations

import os

from mayim_tools._common.credentials import temporary_environment


def test_temporary_environment_sets_and_restores():
    os.environ.pop("MAYIM_TEST_A", None)
    os.environ["MAYIM_TEST_B"] = "old"
    with temporary_environment(MAYIM_TEST_A="1", MAYIM_TEST_B="new"):
        assert os.environ["MAYIM_TEST_A"] == "1"
        assert os.environ["MAYIM_TEST_B"] == "new"
    assert "MAYIM_TEST_A" not in os.environ
    assert os.environ["MAYIM_TEST_B"] == "old"
    del os.environ["MAYIM_TEST_B"]


def test_temporary_environment_restores_after_error():
    os.environ.pop("MAYIM_TEST_C", None)
    try:
        with temporary_environment(MAYIM_TEST_C="x"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "MAYIM_TEST_C" not in os.environ


def test_temporary_environment_empty_is_noop():
    before = dict(os.environ)
    with temporary_environment():
        pass
    assert dict(os.environ) == before
