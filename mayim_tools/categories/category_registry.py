"""
Mayim Tools - Category Registry.

Provides a central registry for all Mayim tool categories.
"""

from __future__ import annotations

from typing import ClassVar

from mayim_tools.categories.base_category import BaseCategory


class CategoryRegistry:
    """
    Central registry for tool categories.

    Categories are stored by unique category ID.
    """

    _categories: ClassVar[dict[str, BaseCategory]] = {}

    @classmethod
    def register(cls, category: BaseCategory) -> None:
        """
        Register a category instance.

        Parameters
        ----------
        category : BaseCategory
            Category instance to register.
        """
        cls._categories[category.id] = category

    @classmethod
    def get(cls, category_id: str) -> BaseCategory | None:
        """
        Return a category by ID.

        Parameters
        ----------
        category_id : str
            Category identifier.

        Returns
        -------
        BaseCategory | None
            Registered category or None if not found.
        """
        return cls._categories.get(category_id)

    @classmethod
    def get_all(cls) -> list[BaseCategory]:
        """
        Return all registered categories.

        Returns
        -------
        list[BaseCategory]
            Registered categories in insertion order.
        """
        return list(cls._categories.values())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered categories."""
        cls._categories.clear()
