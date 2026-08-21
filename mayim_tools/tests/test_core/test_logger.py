# -*- coding: utf-8 -*-
"""
Tests for mayim_tools.core.logger
"""

from unittest.mock import call, patch

import pytest

from mayim_tools.core.logger import MayimLogger


class TestMayimLogger:
    """Unit tests for the MayimLogger static class."""

    @patch("mayim_tools.core.logger.QgsMessageLog.logMessage")
    def test_info_logs_correct_tag(self, mock_log):
        """Info messages should be tagged with 'Mayim Tools'."""
        MayimLogger.info("Test info message")
        assert mock_log.called
        args = mock_log.call_args[0]
        assert args[1] == "Mayim Tools"

    @patch("mayim_tools.core.logger.QgsMessageLog.logMessage")
    def test_warning_logs_correct_tag(self, mock_log):
        """Warning messages should be tagged with 'Mayim Tools'."""
        MayimLogger.warning("Test warning message")
        assert mock_log.called
        args = mock_log.call_args[0]
        assert args[1] == "Mayim Tools"

    @patch("mayim_tools.core.logger.QgsMessageLog.logMessage")
    def test_critical_logs_correct_tag(self, mock_log):
        """Critical messages should be tagged with 'Mayim Tools'."""
        MayimLogger.critical("Test critical message")
        assert mock_log.called
        args = mock_log.call_args[0]
        assert args[1] == "Mayim Tools"

    @patch("mayim_tools.core.logger.QgsMessageLog.logMessage")
    def test_success_prepends_checkmark(self, mock_log):
        """Success messages should be prefixed with a checkmark emoji."""
        MayimLogger.success("Operation complete")
        args = mock_log.call_args[0]
        assert "✅" in args[0]
