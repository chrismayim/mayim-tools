"""
Tests for native connected flat-region identification.

No WhiteboxTools, RichDEM or TauDEM is used.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.gradient.flat_regions import (
    label_flat_regions,
)


class TestLabelFlatRegions:
    """Tests for label_flat_regions()."""

    def test_one_connected_region_gets_one_id(self):
        """A connected mask receives one region ID."""
        flat_mask = np.array(
            [
                [False, False, False, False],
                [False, True, True, False],
                [False, True, True, False],
                [False, False, False, False],
            ],
            dtype=bool,
        )

        region_ids, regions = label_flat_regions(
            flat_mask=flat_mask,
            connectivity=4,
        )

        assert set(regions.keys()) == {1}
        assert np.all(region_ids[flat_mask] == 1)
        assert np.all(region_ids[~flat_mask] == 0)
        assert regions[1]["cell_count"] == 4
        assert regions[1]["row_min"] == 1
        assert regions[1]["row_max"] == 2
        assert regions[1]["col_min"] == 1
        assert regions[1]["col_max"] == 2

    def test_two_disconnected_regions_get_different_ids(self):
        """Disconnected regions receive different IDs."""
        flat_mask = np.array(
            [
                [True, False, False, True],
                [True, False, False, True],
                [False, False, False, False],
                [False, True, True, False],
            ],
            dtype=bool,
        )

        region_ids, regions = label_flat_regions(
            flat_mask=flat_mask,
            connectivity=4,
        )

        assert len(regions) == 3
        assert region_ids[0, 0] == 1
        assert region_ids[0, 3] == 2
        assert region_ids[3, 1] == 3

    def test_diagonal_cells_connect_with_eight_connectivity(self):
        """Diagonal cells connect when connectivity is eight."""
        flat_mask = np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=bool,
        )

        region_ids, regions = label_flat_regions(
            flat_mask=flat_mask,
            connectivity=8,
        )

        assert len(regions) == 1
        assert np.all(region_ids[flat_mask] == 1)
        assert regions[1]["cell_count"] == 2

    def test_diagonal_cells_are_separate_with_four_connectivity(self):
        """Diagonal cells remain separate with four connectivity."""
        flat_mask = np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=bool,
        )

        region_ids, regions = label_flat_regions(
            flat_mask=flat_mask,
            connectivity=4,
        )

        assert len(regions) == 2
        assert region_ids[0, 0] != region_ids[1, 1]
        assert regions[1]["cell_count"] == 1
        assert regions[2]["cell_count"] == 1

    def test_non_flat_cells_receive_zero(self):
        """Non-flat cells receive region ID zero."""
        flat_mask = np.array(
            [
                [False, True, False],
                [False, True, False],
                [False, False, False],
            ],
            dtype=bool,
        )

        region_ids, _ = label_flat_regions(
            flat_mask=flat_mask,
            connectivity=4,
        )

        assert np.all(region_ids[~flat_mask] == 0)

    def test_empty_mask_returns_empty_metadata(self):
        """An empty mask returns no regions."""
        flat_mask = np.zeros(
            (4, 4),
            dtype=bool,
        )

        region_ids, regions = label_flat_regions(
            flat_mask=flat_mask,
            connectivity=8,
        )

        assert np.all(region_ids == 0)
        assert regions == {}

    def test_region_ids_are_deterministic(self):
        """Repeated calls produce identical region IDs."""
        flat_mask = np.array(
            [
                [True, False, False, True],
                [True, False, False, True],
                [False, False, False, False],
            ],
            dtype=bool,
        )

        first_ids, first_regions = label_flat_regions(flat_mask)
        second_ids, second_regions = label_flat_regions(flat_mask)

        assert np.array_equal(first_ids, second_ids)
        assert first_regions == second_regions

    def test_invalid_mask_shape_is_rejected(self):
        """Three-dimensional masks are rejected."""
        flat_mask = np.ones(
            (2, 2, 2),
            dtype=bool,
        )

        with pytest.raises(ValueError, match="two-dimensional"):
            label_flat_regions(flat_mask)

    def test_invalid_mask_dtype_is_rejected(self):
        """Non-Boolean masks are rejected."""
        flat_mask = np.ones(
            (2, 2),
            dtype=np.uint8,
        )

        with pytest.raises(ValueError, match="Boolean"):
            label_flat_regions(flat_mask)

    def test_invalid_connectivity_is_rejected(self):
        """Unsupported connectivity values are rejected."""
        flat_mask = np.ones(
            (2, 2),
            dtype=bool,
        )

        with pytest.raises(ValueError, match="connectivity"):
            label_flat_regions(
                flat_mask,
                connectivity=6,
            )
