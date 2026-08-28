"""
Mayim Tools - Native Stage 5 Enforcement Components
====================================================

Native, independently tested components for Stage 5 Selective Flow
Enforcement:

    - Single-cell de-pitting.
    - Confined Priority-Flood filling.
    - Constrained least-cost breaching.
    - Classification-driven selective enforcement.

No third-party hydrological package is used at runtime.
"""

from mayim_tools.hydrology.enforcement.breaching import (
    apply_breach_path,
    least_cost_breach,
)
from mayim_tools.hydrology.enforcement.depitting import (
    depit_single_cell,
)
from mayim_tools.hydrology.enforcement.enforcement import (
    enforce_selectively,
)
from mayim_tools.hydrology.enforcement.filling import (
    confined_priority_flood_fill,
)

__all__ = [
    "apply_breach_path",
    "confined_priority_flood_fill",
    "depit_single_cell",
    "enforce_selectively",
    "least_cost_breach",
]
