"""
Mayim Tools – Geometry Category Initialiser
Automatically registers the Geometry category with the CategoryRegistry.
"""

from mayim_tools.categories.category_registry import CategoryRegistry
from mayim_tools.categories.geometry.category import GeometryCategory

# Self-register on import
CategoryRegistry.register(GeometryCategory())
