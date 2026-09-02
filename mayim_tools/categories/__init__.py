# -*- coding: utf-8 -*-
"""
Mayim Tools - Categories Package Initialiser
Importing this package triggers all category self-registrations.
"""

from mayim_tools.categories.category_registry import CategoryRegistry

# ── Clear any previously registered categories first ──
# This prevents duplicate registrations on plugin reload
CategoryRegistry.clear()

# ── Import each category to trigger self-registration ──
import mayim_tools.categories.geometry   # noqa: F401
import mayim_tools.categories.hydrology  # noqa: F401
import mayim_tools.categories.rainfall   # noqa: F401
# import mayim_tools.categories.network  # noqa: F401
