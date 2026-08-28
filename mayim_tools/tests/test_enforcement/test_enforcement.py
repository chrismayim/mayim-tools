"""Tests for native Stage 5 selective enforcement.

These tests validate the classification-driven orchestration layer.
They use native Mayim components only and do not call WhiteboxTools,
RichDEM or TauDEM.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.enforcement.enforcement import (
    ARTIFACT,
    DECISION_BREACHED,
    DECISION_DEPITTED,
    DECISION_FILLED,
    DECISION_REAL_PRESERVED,
    DECISION_REVIEW_PRESERVED,
    REAL_FEATURE,
    REVIEW_REQUIRED,
    enforce_selectively,
)


def make_record(
    classification: str,
    pit_row: int,
    pit_col: int,
    spill_elevation: float,
    area_cells: int,
) -> dict:
    """Create a valid depression record for testing."""
    return {
        "classification": classification,
        "pit_row": pit_row,
        "pit_col": pit_col,
        "spill_elevation": spill_elevation,
        "area_cells": area_cells,
    }


class TestEnforceSelectively:
    """Tests for enforce_selectively()."""

    def test_artifact_single_cell_is_depitted(self):
        """A one-cell artifact depression is de-pitted."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 8.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        mask = np.zeros_like(dem, dtype=bool)
        mask[1, 1] = True

        preserved, ready, codes, audits = enforce_selectively(
            dem=dem,
            depression_records={
                1: make_record(
                    classification=ARTIFACT,
                    pit_row=1,
                    pit_col=1,
                    spill_elevation=8.0,
                    area_cells=1,
                ),
            },
            depression_masks={1: mask},
            max_breach_length=20,
            max_breach_depth=20.0,
            nodata=-9999.0,
        )

        assert preserved[1, 1] == 8.0
        assert ready[1, 1] == 8.0
        assert codes[1, 1] == DECISION_DEPITTED
        assert len(audits) == 1
        assert audits[0]["decision"] == "depitted"

    def test_real_feature_is_preserved(self):
        """A REAL_FEATURE depression is not modified."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        original = dem.copy()
        mask = np.zeros_like(dem, dtype=bool)
        mask[1, 1] = True

        preserved, ready, codes, audits = enforce_selectively(
            dem=dem,
            depression_records={
                1: make_record(
                    classification=REAL_FEATURE,
                    pit_row=1,
                    pit_col=1,
                    spill_elevation=10.0,
                    area_cells=1,
                ),
            },
            depression_masks={1: mask},
            max_breach_length=20,
            max_breach_depth=20.0,
            nodata=-9999.0,
        )

        assert np.array_equal(preserved, original)
        assert np.array_equal(ready, original)
        assert codes[1, 1] == DECISION_REAL_PRESERVED
        assert audits[0]["modified"] is False
        assert audits[0]["decision"] == "preserved_real_feature"

    def test_review_required_is_preserved(self):
        """A REVIEW_REQUIRED depression is not modified."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        original = dem.copy()
        mask = np.zeros_like(dem, dtype=bool)
        mask[1, 1] = True

        preserved, ready, codes, audits = enforce_selectively(
            dem=dem,
            depression_records={
                1: make_record(
                    classification=REVIEW_REQUIRED,
                    pit_row=1,
                    pit_col=1,
                    spill_elevation=10.0,
                    area_cells=1,
                ),
            },
            depression_masks={1: mask},
            max_breach_length=20,
            max_breach_depth=20.0,
            nodata=-9999.0,
        )

        assert np.array_equal(preserved, original)
        assert np.array_equal(ready, original)
        assert codes[1, 1] == DECISION_REVIEW_PRESERVED
        assert audits[0]["modified"] is False
        assert audits[0]["decision"] == "preserved_pending_review"

    def test_larger_artifact_is_breached_when_path_is_admissible(self):
        """A larger artifact uses breaching when a valid path exists."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 8.0, 8.0, 8.0, 10.0],
                [10.0, 8.0, 2.0, 8.0, 10.0],
                [10.0, 8.0, 8.0, 8.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        mask = np.zeros_like(dem, dtype=bool)
        mask[1:4, 1:4] = True

        preserved, ready, codes, audits = enforce_selectively(
            dem=dem,
            depression_records={
                1: make_record(
                    classification=ARTIFACT,
                    pit_row=2,
                    pit_col=2,
                    spill_elevation=8.0,
                    area_cells=9,
                ),
            },
            depression_masks={1: mask},
            max_breach_length=20,
            max_breach_depth=20.0,
            nodata=-9999.0,
        )

        assert audits[0]["decision"] == "breached"
        assert audits[0]["decision_code"] == DECISION_BREACHED
        assert np.all(preserved >= dem)
        assert np.array_equal(preserved, ready)
        assert np.any(codes == DECISION_BREACHED)

    def test_breach_failure_uses_filling_fallback(self):
        """An inadmissible breach falls back to confined filling."""
        dem = np.array(
            [
                [100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 95.0, 95.0, 95.0, 100.0],
                [100.0, 95.0, 2.0, 95.0, 100.0],
                [100.0, 95.0, 95.0, 95.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0],
            ],
            dtype=np.float64,
        )

        mask = np.zeros_like(dem, dtype=bool)
        mask[1:4, 1:4] = True

        preserved, ready, codes, audits = enforce_selectively(
            dem=dem,
            depression_records={
                1: make_record(
                    classification=ARTIFACT,
                    pit_row=2,
                    pit_col=2,
                    spill_elevation=90.0,
                    area_cells=9,
                ),
            },
            depression_masks={1: mask},
            max_breach_length=20,
            max_breach_depth=1.0,
            nodata=-9999.0,
        )

        assert audits[0]["decision"] == "filled_fallback"
        assert audits[0]["decision_code"] == DECISION_FILLED
        assert preserved[2, 2] == 90.0
        assert ready[2, 2] == 90.0
        assert np.all(preserved >= dem)
        assert np.any(codes == DECISION_FILLED)

    def test_nodata_cells_remain_unchanged(self):
        """NoData cells are not modified by enforcement."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, -9999.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        original = dem.copy()
        mask = np.zeros_like(dem, dtype=bool)
        mask[1, 1] = True

        preserved, ready, codes, _ = enforce_selectively(
            dem=dem,
            depression_records={
                1: make_record(
                    classification=ARTIFACT,
                    pit_row=1,
                    pit_col=1,
                    spill_elevation=10.0,
                    area_cells=1,
                ),
            },
            depression_masks={1: mask},
            max_breach_length=20,
            max_breach_depth=20.0,
            nodata=-9999.0,
        )

        assert preserved[1, 2] == -9999.0
        assert ready[1, 2] == -9999.0
        assert codes[1, 2] == 255
        assert original[1, 2] == -9999.0

    def test_input_dem_is_not_modified(self):
        """The input DEM is never modified in place."""
        dem = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 2.0, 8.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )

        original = dem.copy()
        mask = np.zeros_like(dem, dtype=bool)
        mask[1, 1] = True

        enforce_selectively(
            dem=dem,
            depression_records={
                1: make_record(
                    classification=ARTIFACT,
                    pit_row=1,
                    pit_col=1,
                    spill_elevation=8.0,
                    area_cells=1,
                ),
            },
            depression_masks={1: mask},
            max_breach_length=20,
            max_breach_depth=20.0,
            nodata=-9999.0,
        )

        assert np.array_equal(dem, original)

    def test_missing_depression_mask_is_rejected(self):
        """Every depression record must have a matching mask."""
        dem = np.ones((3, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="Missing depression mask"):
            enforce_selectively(
                dem=dem,
                depression_records={
                    1: make_record(
                        classification=REAL_FEATURE,
                        pit_row=1,
                        pit_col=1,
                        spill_elevation=1.0,
                        area_cells=1,
                    ),
                },
                depression_masks={},
                max_breach_length=10,
                max_breach_depth=5.0,
                nodata=-9999.0,
            )

    def test_unknown_classification_is_rejected(self):
        """Unknown classification labels must not be silently modified."""
        dem = np.ones((3, 3), dtype=np.float64)
        mask = np.ones_like(dem, dtype=bool)

        with pytest.raises(ValueError, match="Unknown classification"):
            enforce_selectively(
                dem=dem,
                depression_records={
                    1: make_record(
                        classification="UNKNOWN",
                        pit_row=1,
                        pit_col=1,
                        spill_elevation=1.0,
                        area_cells=1,
                    ),
                },
                depression_masks={1: mask},
                max_breach_length=10,
                max_breach_depth=5.0,
                nodata=-9999.0,
            )
