# -*- coding: utf-8 -*-
"""
Mayim Tools – Base Category
Abstract base class that all tool categories must inherit from.
Enforces a consistent interface for category discovery and registration.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from qgis.PyQt.QtGui import QIcon


class BaseCategory(ABC):
    """
    Abstract base class for all Mayim Tools categories.
    Every category must implement these properties and methods.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Unique identifier for this category.
        Used internally by the CategoryRegistry.
        Example: 'hydrology'
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable category name.
        Displayed in the toolbar, menu, and dock panel.
        Example: 'Hydrology Tools'
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Short description of what this category contains.
        Displayed in the About section and dock panel tooltip.
        """
        raise NotImplementedError

    @property
    def icon_path(self) -> str:
        """
        Absolute file path to the category icon.
        Override in subclass to provide a custom icon.
        """
        from mayim_tools.resources_rc import get_icon_path
        return get_icon_path("mayim_logo.png")

    @property
    def icon(self) -> QIcon:
        """Returns a QIcon for this category."""
        from mayim_tools.resources_rc import get_icon_path
        path = self.icon_path
        if path and Path(path).exists():
            return QIcon(path)
        return QIcon()

    @abstractmethod
    def get_algorithms(self) -> list:
        """
        Return a list of instantiated algorithm objects for this category.
        These are registered with the Processing Provider.

        :returns: List of QgsProcessingAlgorithm instances
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<MayimCategory: {self.id} — {self.name}>"
