# -*- coding: utf-8 -*-
"""
Mayim Tools – Base Algorithm
Abstract base class that all Mayim Tools processing algorithms must inherit.
Enforces a consistent interface and provides shared helper methods.
"""

from abc import abstractmethod

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingFeedback,
)

from mayim_tools.core.logger import MayimLogger


class MayimBaseAlgorithm(QgsProcessingAlgorithm):
    """
    Abstract base class for all Mayim Tools processing algorithms.

    Every tool in Mayim Tools inherits from this class.
    Subclasses MUST implement:
        - name()
        - displayName()
        - group()
        - groupId()
        - initAlgorithm()
        - processAlgorithm()
        - createInstance()

    Subclasses MAY override:
        - shortHelpString()
        - helpUrl()
        - icon()
        - tags()
    """

    # ── Abstract Methods — Must be implemented by every tool ── #

    @abstractmethod
    def name(self) -> str:
        """
        Unique algorithm identifier (lowercase, no spaces).
        Used in processing.run() calls.
        Example: 'catchmentdelineation'
        """
        raise NotImplementedError

    @abstractmethod
    def displayName(self) -> str:
        """
        Human-readable tool name shown in the Processing Toolbox.
        Example: 'Catchment Delineation'
        """
        raise NotImplementedError

    @abstractmethod
    def group(self) -> str:
        """
        Category group name shown in the Processing Toolbox.
        Example: 'Hydrology Tools'
        """
        raise NotImplementedError

    @abstractmethod
    def groupId(self) -> str:
        """
        Unique category group identifier (lowercase, no spaces).
        Example: 'hydrology'
        """
        raise NotImplementedError

    @abstractmethod
    def initAlgorithm(self, config=None) -> None:
        """
        Define all input and output parameters for this tool.
        Called by QGIS when building the Processing dialog.
        Use self.addParameter() to add inputs and outputs.
        """
        raise NotImplementedError

    @abstractmethod
    def processAlgorithm(
        self,
        parameters: dict,
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict:
        """
        Core processing logic for this tool.
        Must return a dictionary of output values.

        :param parameters: Input parameter values
        :param context: QGIS processing context
        :param feedback: Feedback object for progress and cancellation
        :returns: Dictionary of output key-value pairs
        """
        raise NotImplementedError

    @abstractmethod
    def createInstance(self):
        """
        Return a fresh instance of this algorithm class.
        Required by the QGIS Processing Framework.
        Always implement as: return self.__class__()
        """
        raise NotImplementedError

    # ── Optional Methods — Override as needed ── #

    def shortHelpString(self) -> str:
        """
        Short description shown in the Processing dialog help panel.
        Override in subclass to provide tool-specific help text.
        """
        return (
            f"<b>{self.displayName()}</b><br><br>"
            f"Part of the <b>Mayim Tools</b> plugin — {self.group()}.<br><br>"
            f"For full documentation, visit the Mayim Tools repository."
        )

    def helpUrl(self) -> str:
        """
        URL to full documentation for this tool.
        Override in subclass with a direct link to the tool's docs page.
        """
        return "https://github.com/chrismayim/mayim-tools"

    def tags(self) -> list[str]:
        """
        List of searchable tags for this algorithm.
        Override in subclass to add tool-specific tags.
        """
        return ["mayim", "tools", self.groupId()]

    # ── Shared Helper Methods — Available to all tools ── #

    def log(self, message: str, feedback: QgsProcessingFeedback = None) -> None:
        """
        Log a message to both QgsMessageLog and the Processing feedback panel.

        :param message: Message string to log
        :param feedback: Optional feedback object for Processing panel output
        """
        MayimLogger.info(message)
        if feedback:
            feedback.pushInfo(message)

    def log_warning(self, message: str, feedback: QgsProcessingFeedback = None) -> None:
        """
        Log a warning to both QgsMessageLog and the Processing feedback panel.

        :param message: Warning message string
        :param feedback: Optional feedback object for Processing panel output
        """
        MayimLogger.warning(message)
        if feedback:
            feedback.pushWarning(message)

    def is_cancelled(self, feedback: QgsProcessingFeedback) -> bool:
        """
        Check if the user has requested cancellation of the algorithm.
        Call this inside loops in processAlgorithm() to support cancellation.

        :param feedback: The feedback object from processAlgorithm()
        :returns: True if cancelled, False otherwise
        """
        return feedback.isCanceled()

    def set_progress(
        self,
        feedback: QgsProcessingFeedback,
        current: int,
        total: int,
    ) -> None:
        """
        Update the progress bar in the Processing dialog.

        :param feedback: The feedback object from processAlgorithm()
        :param current: Current iteration index
        :param total: Total number of iterations
        """
        if total > 0:
            feedback.setProgress(int((current / total) * 100))
