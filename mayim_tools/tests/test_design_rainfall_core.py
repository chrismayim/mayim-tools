from __future__ import annotations

import pandas as pd
import pytest

from mayim_tools.rainfall.design_rainfall.core import DesignRainfallEngine


def make_engine() -> DesignRainfallEngine:
    """Create an engine with an in-memory multiday regression table."""
    engine = object.__new__(DesignRainfallEngine)
    engine._multiday_regr = pd.DataFrame(
        [
            {
                "REGION": 7,
                "THETA": 1.0,
                "TAU": 0.5,
                "SIGMA": 1.0,
                "UPSILON": 2.0,
                "KAPPA": 0.25,
                "RHO": 1.0,
            }
        ]
    )
    return engine


def test_multiday_l1_regression() -> None:
    """Verify the 2–7-day L1 regression equation."""
    engine = make_engine()

    result = engine._multiday_l1(7, 3, 100.0)

    expected = (1.0 + 0.5 * (3**1.0)) * 100.0 + (2.0 + 0.25 * (3**1.0))

    assert result == pytest.approx(expected)


def test_depth_bounds_use_prediction_percentage() -> None:
    """Verify that prediction intervals are interpreted as percentages."""
    engine = make_engine()

    result = engine._depth_with_bounds(
        100.0,
        10.0,
        pd.Series({"GC2": 1.2, "GCU2": 1.3, "GCL2": 1.1}),
        [2],
    )

    _, median, lower, upper = result[0]

    assert median == pytest.approx(120.0)
    assert lower == pytest.approx(99.0)
    assert upper == pytest.approx(143.0)
