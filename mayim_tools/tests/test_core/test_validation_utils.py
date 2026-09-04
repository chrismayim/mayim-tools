"""
Tests for mayim_tools.core.validation_utils
"""

from mayim_tools.core.validation_utils import ValidationUtils


class TestValidationUtils:
    """Unit tests for ValidationUtils static class."""

    # ── is_not_none ──
    def test_is_not_none_with_value(self):
        assert ValidationUtils.is_not_none("hello") is True

    def test_is_not_none_with_none(self):
        assert ValidationUtils.is_not_none(None) is False

    # ── is_not_empty_string ──
    def test_is_not_empty_string_valid(self):
        assert ValidationUtils.is_not_empty_string("Mayim") is True

    def test_is_not_empty_string_empty(self):
        assert ValidationUtils.is_not_empty_string("") is False

    def test_is_not_empty_string_whitespace(self):
        assert ValidationUtils.is_not_empty_string("   ") is False

    def test_is_not_empty_string_none(self):
        assert ValidationUtils.is_not_empty_string(None) is False

    # ── is_positive_number ──
    def test_is_positive_number_valid_int(self):
        assert ValidationUtils.is_positive_number(5) is True

    def test_is_positive_number_valid_float(self):
        assert ValidationUtils.is_positive_number(3.14) is True

    def test_is_positive_number_zero(self):
        assert ValidationUtils.is_positive_number(0) is False

    def test_is_positive_number_negative(self):
        assert ValidationUtils.is_positive_number(-1) is False

    def test_is_positive_number_string(self):
        assert ValidationUtils.is_positive_number("abc") is False

    # ── is_in_range ──
    def test_is_in_range_within(self):
        assert ValidationUtils.is_in_range(5.0, 0.0, 10.0) is True

    def test_is_in_range_at_boundary(self):
        assert ValidationUtils.is_in_range(0.0, 0.0, 10.0) is True
        assert ValidationUtils.is_in_range(10.0, 0.0, 10.0) is True

    def test_is_in_range_outside(self):
        assert ValidationUtils.is_in_range(11.0, 0.0, 10.0) is False

    def test_is_in_range_invalid_type(self):
        assert ValidationUtils.is_in_range("x", 0.0, 10.0) is False

    # ── is_valid_file_path ──
    def test_is_valid_file_path_nonexistent(self):
        assert (
            ValidationUtils.is_valid_file_path(
                "C:/nonexistent/path.gpkg", must_exist=True
            )
            is False
        )

    def test_is_valid_file_path_no_existence_check(self):
        assert (
            ValidationUtils.is_valid_file_path("C:/some/path.gpkg", must_exist=False)
            is True
        )

    def test_is_valid_file_path_empty(self):
        assert ValidationUtils.is_valid_file_path("") is False

    # ── is_valid_epsg ──
    def test_is_valid_epsg_wgs84(self):
        assert ValidationUtils.is_valid_epsg(4326) is True

    def test_is_valid_epsg_lo29(self):
        assert ValidationUtils.is_valid_epsg(22229) is True

    def test_is_valid_epsg_invalid_string(self):
        assert ValidationUtils.is_valid_epsg("abc") is False
