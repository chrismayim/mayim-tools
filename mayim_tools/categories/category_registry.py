"""
Mayim Tools – Category Registry
Central registry for all tool categories.
Categories self-register at import time — no hard-coded lists needed.
"""

from mayim_tools.categories.base_category import BaseCategory
from mayim_tools.core.logger import MayimLogger


class CategoryRegistry:
    """
    Central registry that tracks all registered Mayim Tools categories.

    Usage:
        # In your category's __init__.py:
        CategoryRegistry.register(HydrologyCategory())

        # To retrieve all categories (e.g., in the provider):
        all_categories = CategoryRegistry.get_all()
    """

    _categories: dict[str, BaseCategory] = {}

    @classmethod
    def register(cls, category: BaseCategory) -> None:
        """
        Register a category instance.

        :param category: An instance of a BaseCategory subclass
        """
        if category.id in cls._categories:
            MayimLogger.warning(
                f"Category '{category.id}' is already registered. Skipping."
            )
            return
        cls._categories[category.id] = category
        MayimLogger.info(f"Category registered: {category.name}")

    @classmethod
    def unregister(cls, category_id: str) -> None:
        """
        Unregister a category by its ID.

        :param category_id: The unique category ID to remove
        """
        if category_id in cls._categories:
            del cls._categories[category_id]
            MayimLogger.info(f"Category unregistered: {category_id}")

    @classmethod
    def get(cls, category_id: str) -> BaseCategory:
        """
        Retrieve a single category by its ID.

        :param category_id: The unique category ID
        :returns: BaseCategory instance or None
        """
        return cls._categories.get(category_id, None)

    @classmethod
    def get_all(cls) -> list[BaseCategory]:
        """
        Retrieve all registered categories.

        :returns: List of all BaseCategory instances
        """
        return list(cls._categories.values())

    @classmethod
    def get_all_algorithms(cls) -> list:
        """
        Retrieve all algorithms from all registered categories.
        Used by the Processing Provider to load all tools at once.

        :returns: Flat list of all QgsProcessingAlgorithm instances
        """
        algorithms = []
        for category in cls._categories.values():
            algorithms.extend(category.get_algorithms())
        return algorithms

    @classmethod
    def is_registered(cls, category_id: str) -> bool:
        """
        Check if a category ID is already registered.

        :param category_id: The unique category ID to check
        :returns: True if registered, False otherwise
        """
        return category_id in cls._categories

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered categories.
        Called during plugin unload to prevent memory leaks.
        """
        cls._categories.clear()
        MayimLogger.info("Category registry cleared.")
