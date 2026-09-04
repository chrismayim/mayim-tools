"""
Mayim Tools - Categories Package Initialiser

Importing this package triggers all category self-registrations.
"""

from mayim_tools.categories.category_registry import CategoryRegistry

# Clear any previously registered categories before importing categories.
CategoryRegistry.clear()

# Import each category to trigger self-registration.
import mayim_tools.categories.data
import mayim_tools.categories.geometry
import mayim_tools.categories.hydrology
import mayim_tools.categories.rainfall  # noqa: F401
