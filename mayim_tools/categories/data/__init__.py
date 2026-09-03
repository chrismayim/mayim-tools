"""
Mayim Tools — Data Tools Category Initialiser

Importing this module triggers self-registration of the
Data Tools category with the CategoryRegistry.
"""

from __future__ import annotations

from mayim_tools.categories.category_registry import CategoryRegistry
from mayim_tools.categories.data.category import DataCategory

CategoryRegistry.register(DataCategory())
