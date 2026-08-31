"""Tests for native Stage 6 Garbrecht-Martz flat resolution.

These tests validate the current native flat-resolution component.
They do not use WhiteboxTools, RichDEM, TauDEM or another
third-party hydrological implementation.
"""

import numpy as np
import pytest

from mayim_tools.hydrology.gradient.flat_detection import detect_flats
from mayim_tools.hydrology.gradient.gradient_resolution import (
    resolve_flats,
)


def make_flat_test_dem() -> np.ndarray:
    """Return a synthetic flat with a higher and lower boundary."""
    return np.array(
        [
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 5.0, 5.0, 5.0, 10.0],
            [10.0, 5.0, 5.0, 5.0, 10.0],
            [10.0, 5.0, 5.0, 5.0, 3.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
        ],
        dtype=np.float64,
    )


class TestResolveFlats:
    """Tests for resolve_flats()."""

    def test_flat_cells_receive_a_correction(self):
        """Detected flat cells receive a non-zero correction."""
        dem = make_flat_test_dem()

        flat_mask, higher_boundary, lower_boundary = detect_flats(
            dem=dem,
            nodata=-9999.0,
        )

        result, audit = resolve_flats(
            dem=dem,
            flat_mask=flat_mask,
            higher_boundary=higher_boundary,
            lower_boundary=lower_boundary,
            cell_size=1.0,
            vertical_accuracy=0.1,
            nodata=-9999.0,
        )

        assert np.any(result[flat_mask] != dem[flat_mask])
        assert audit["modified_cells"] == int(np.sum(flat_mask))
        assert audit["method"] == "garbrecht_martz_flat_resolution"

    def test_non_flat_cells_remain_unchanged(self):
        """Cells outside the flat mask are not modified."""
        dem = make_flat_test_dem()

        flat_mask, higher_boundary, lower_boundary = detect_flats(
            dem=dem,
            nodata=-9999.0,
        )

        result, _ = resolve_flats(
            dem=dem,
            flat_mask=flat_mask,
            higher_boundary=higher_boundary,
            lower_boundary=lower_boundary,
            cell_size=1.0,
            vertical_accuracy=0.1,
            nodata=-9999.0,
        )

        assert np.array_equal(result[~flat_mask], dem[~flat_mask])

    def test_input_dem_is_not_modified(self):
        """The input DEM is not modified in place."""
        dem = make_flat_test_dem()
        original = dem.copy()

        flat_mask, higher_boundary, lower_boundary = detect_flats(
            dem=dem,
            nodata=-9999.0,
        )

        resolve_flats(
            dem=dem,
            flat_mask=flat_mask,
            higher_boundary=higher_boundary,
            lower_boundary=lower_boundary,
            cell_size=1.0,
            vertical_accuracy=0.1,
            nodata=-9999.0,
        )

        assert np.array_equal(dem, original)

    def test_empty_flat_mask_returns_unchanged_dem(self):
        """An empty flat mask produces an auditable no-op."""
        dem = np.array(
            [
                [10.0, 9.0, 8.0],
                [9.0, 8.0, 7.0],
                [8.0, 7.0, 6.0],
            ],
            dtype=np.float64,
        )

        empty_mask = np.zeros_like(dem, dtype=bool)

        result, audit = resolve_flats(
            dem=dem,
            flat_mask=empty_mask,
            higher_boundary=empty_mask,
            lower_boundary=empty_mask,
            cell_size=1.0,
            vertical_accuracy=0.1,
            nodata=-9999.0,
        )

        assert np.array_equal(result, dem)
        assert audit["modified_cells"] == 0
        assert audit["total_elevation_change"] == 0.0

    def test_flat_without_lower_boundary_is_rejected(self):
        """A flat with no outlet cannot be resolved."""
        dem = np.full(
            (5, 5),
            5.0,
            dtype=np.float64,
        )

        flat_mask = np.ones_like(dem, dtype=bool)
        flat_mask[0, :] = False
        flat_mask[-1, :] = False
        flat_mask[:, 0] = False
        flat_mask[:, -1] = False

        higher_boundary = np.zeros_like(dem, dtype=bool)
        lower_boundary = np.zeros_like(dem, dtype=bool)

        with pytest.raises(
            ValueError,
            match="no valid lower boundary",
        ):
            resolve_flats(
                dem=dem,
                flat_mask=flat_mask,
                higher_boundary=higher_boundary,
                lower_boundary=lower_boundary,
                cell_size=1.0,
                vertical_accuracy=0.1,
                nodata=-9999.0,
            )

    def test_invalid_cell_size_is_rejected(self):
        """Cell size must be positive."""
        dem = make_flat_test_dem()

        flat_mask, higher_boundary, lower_boundary = detect_flats(
            dem=dem,
            nodata=-9999.0,
        )

        with pytest.raises(
            ValueError,
            match="cell_size",
        ):
            resolve_flats(
                dem=dem,
                flat_mask=flat_mask,
                higher_boundary=higher_boundary,
                lower_boundary=lower_boundary,
                cell_size=0.0,
                vertical_accuracy=0.1,
                nodata=-9999.0,
            )
