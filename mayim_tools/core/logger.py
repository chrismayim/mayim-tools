"""
Mayim Tools – Centralised Logger
Wraps QgsMessageLog for consistent, tagged logging across all modules.
"""

from qgis.core import Qgis, QgsMessageLog


class MayimLogger:
    """Static logger class for Mayim Tools."""

    TAG = "Mayim Tools"

    @staticmethod
    def info(message: str) -> None:
        """Log an informational message."""
        QgsMessageLog.logMessage(message, MayimLogger.TAG, Qgis.MessageLevel.Info)

    @staticmethod
    def warning(message: str) -> None:
        """Log a warning message."""
        QgsMessageLog.logMessage(message, MayimLogger.TAG, Qgis.MessageLevel.Warning)

    @staticmethod
    def critical(message: str) -> None:
        """Log a critical error message."""
        QgsMessageLog.logMessage(message, MayimLogger.TAG, Qgis.MessageLevel.Critical)

    @staticmethod
    def success(message: str) -> None:
        """Log a success message (displayed as Info level)."""
        QgsMessageLog.logMessage(
            f"✅ {message}", MayimLogger.TAG, Qgis.MessageLevel.Info
        )
