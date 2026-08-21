# -*- coding: utf-8 -*-
"""
Mayim Tools – Hydrology Category Initialiser
Automatically registers the Hydrology category with the CategoryRegistry
when this module is imported.
"""

from mayim_tools.categories.category_registry import CategoryRegistry
from mayim_tools.categories.hydrology.category import HydrologyCategory

# Self-register on import
CategoryRegistry.register(HydrologyCategory())
