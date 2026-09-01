"""
Tests for native Stage 7 hydrography/DEM divergence analysis.

These tests use Boolean raster masks and original synthetic data.
They do not use WhiteboxTools, RichDEM or TauDEM.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.hydrography.divergence import (
    analyse_hydrography_divergence,
)


class TestAnalyseHydrographyDivergence:
    """Tests for analyse_hydrography_divergence()."""

    def test_identical_masks_are_aligned(self):
        """Identical hydrography and flow masks are aligned."""
        hydrography_mask = np.zeros((5, 5), dtype=bool)
        dem_flow_mask = np.zeros((5, 5), dtype=bool)

        hydrography_mask[2, 1:4] = True
        dem_flow_mask[2, 1:4] = True

        result = analyse_hydrography_divergence(
            hydrography_mask=hydrography_mask,
            dem_flow_mask=dem_flow_mask,
            positional_tolerance_cells=0,
        )

        assert result["statistics"]["aligned_cells"] == 3
        assert result["statistics"]["material_divergence_cells"] == 0
        assert np.all(result["aligned_mask"][2, 1:4])
        assert not np.any(result["divergence_mask"])

    def test_small_offset_is_within_tolerance(self):
        """A small positional offset is tolerated."""
        hydrography_mask = np.zeros((7, 7), dtype=bool)
        dem_flow_mask = np.zeros((7, 7), dtype=bool)

        hydrography_mask[3, 1:6] = True
        dem_flow_mask[3, 2:7] = True

        result = analyse_hydrography_divergence(
            hydrography_mask=hydrography_mask,
            dem_flow_mask=dem_flow_mask,
            positional_tolerance_cells=1,
        )

        assert result["statistics"]["tolerated_cells"] > 0
        assert result["statistics"]["material_divergence_cells"] == 0
        assert not np.any(result["divergence_mask"])

    def test_large_offset_is_material_divergence(self):
        """A large positional offset is material divergence."""
        hydrography_mask = np.zeros((9, 9), dtype=bool)
        dem_flow_mask = np.zeros((9, 9), dtype=bool)

        hydrography_mask[2, 1:4] = True
        dem_flow_mask[6, 1:4] = True

        result = analyse_hydrography_divergence(
            hydrography_mask=hydrography_mask,
            dem_flow_mask=dem_flow_mask,
            positional_tolerance_cells=1,
        )

        assert result["statistics"]["material_divergence_cells"] > 0
        assert np.any(result["divergence_mask"])

    def test_hydrography_without_dem_flow_is_divergent(self):
        """Mapped hydrography without matching DEM flow is divergent."""
        hydrography_mask = np.zeros((5, 5), dtype=bool)
        dem_flow_mask = np.zeros((5, 5), dtype=bool)

        hydrography_mask[2, 1:4] = True

        result = analyse_hydrography_divergence(
            hydrography_mask=hydrography_mask,
            dem_flow_mask=dem_flow_mask,
            positional_tolerance_cells=0,
        )

        assert result["statistics"]["hydrography_only_cells"] == 3
        assert result["statistics"]["material_divergence_cells"] == 3

    def test_dem_flow_without_hydrography_is_reported(self):
        """DEM drainage without mapped hydrography is reported."""
        hydrography_mask = np.zeros((5, 5), dtype=bool)
        dem_flow_mask = np.zeros((5, 5), dtype=bool)

        dem_flow_mask[2, 1:4] = True

        result = analyse_hydrography_divergence(
            hydrography_mask=hydrography_mask,
            dem_flow_mask=dem_flow_mask,
            positional_tolerance_cells=0,
        )

        assert result["statistics"]["dem_only_cells"] == 3
        assert result["statistics"]["material_divergence_cells"] == 3

    def test_nodata_cells_are_not_assessed(self):
        """NoData cells are excluded from divergence assessment."""
        hydrography_mask = np.zeros((5, 5), dtype=bool)
        dem_flow_mask = np.zeros((5, 5), dtype=bool)
        nodata_mask = np.zeros((5, 5), dtype=bool)

        hydrography_mask[2, 1:4] = True
        dem_flow_mask[2, 1:4] = True
        nodata_mask[2, 2] = True

        result = analyse_hydrography_divergence(
            hydrography_mask=hydrography_mask,
            dem_flow_mask=dem_flow_mask,
            positional_tolerance_cells=0,
            nodata_mask=nodata_mask,
        )

        assert result["statistics"]["nodata_cells"] == 1
        assert result["aligned_mask"][2, 2] == 255
        assert result["divergence_mask"][2, 2] == 255

    def test_input_masks_are_not_modified(self):
        """Input masks remain unchanged."""
        hydrography_mask = np.zeros((5, 5), dtype=bool)
        dem_flow_mask = np.zeros((5, 5), dtype=bool)

        hydrography_mask[2, 1:4] = True
        dem_flow_mask[2, 2:5] = True

        original_hydrography = hydrography_mask.copy()
        original_flow = dem_flow_mask.copy()

        analyse_hydrography_divergence(
            hydrography_mask=hydrography_mask,
            dem_flow_mask=dem_flow_mask,
            positional_tolerance_cells=1,
        )

        assert np.array_equal(
            hydrography_mask,
            original_hydrography,
        )
        assert np.array_equal(
            dem_flow_mask,
            original_flow,
        )

    def test_masks_must_have_same_shape(self):
        """Input masks must have identical shapes."""
        hydrography_mask = np.zeros((5, 5), dtype=bool)
        dem_flow_mask = np.zeros((4, 4), dtype=bool)

        with pytest.raises(ValueError, match="same shape"):
            analyse_hydrography_divergence(
                hydrography_mask=hydrography_mask,
                dem_flow_mask=dem_flow_mask,
                positional_tolerance_cells=1,
            )

    def test_masks_must_be_boolean(self):
        """Input masks must use Boolean dtype."""
        hydrography_mask = np.zeros((5, 5), dtype=np.uint8)
        dem_flow_mask = np.zeros((5, 5), dtype=bool)

        with pytest.raises(ValueError, match="Boolean"):
            analyse_hydrography_divergence(
                hydrography_mask=hydrography_mask,
                dem_flow_mask=dem_flow_mask,
                positional_tolerance_cells=1,
            )

    def test_negative_tolerance_is_rejected(self):
        """Positional tolerance cannot be negative."""
        hydrography_mask = np.zeros((5, 5), dtype=bool)
        dem_flow_mask = np.zeros((5, 5), dtype=bool)

        with pytest.raises(ValueError, match="tolerance"):
            analyse_hydrography_divergence(
                hydrography_mask=hydrography_mask,
                dem_flow_mask=dem_flow_mask,
                positional_tolerance_cells=-1,
            )

    def test_results_are_deterministic(self):
        """Repeated analysis produces identical results."""
        hydrography_mask = np.zeros((7, 7), dtype=bool)
        dem_flow_mask = np.zeros((7, 7), dtype=bool)

        hydrography_mask[3, 1:6] = True
        dem_flow_mask[3, 2:7] = True

        first = analyse_hydrography_divergence(
            hydrography_mask=hydrography_mask,
            dem_flow_mask=dem_flow_mask,
            positional_tolerance_cells=1,
        )

        second = analyse_hydrography_divergence(
            hydrography_mask=hydrography_mask,
            dem_flow_mask=dem_flow_mask,
            positional_tolerance_cells=1,
        )

        assert np.array_equal(
            first["divergence_mask"],
            second["divergence_mask"],
        )
        assert first["statistics"] == second["statistics"]
        assert first["review_records"] == second["review_records"]
