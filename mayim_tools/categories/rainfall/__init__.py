"""
Mayim Tools — Rainfall Analysis Category Initialiser

Importing this module triggers self-registration of the
Rainfall Analysis category with the CategoryRegistry.
"""

from __future__ import annotations

from mayim_tools.categories.category_registry import CategoryRegistry
from mayim_tools.categories.rainfall.category import RainfallCategory

CategoryRegistry.register(RainfallCategory())
