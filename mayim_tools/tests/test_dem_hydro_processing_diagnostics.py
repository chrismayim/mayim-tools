import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.diagnostics import (
    diagnose,
    find_unresolvable_cells,
)


def _slope():
    return np.add.outer(np.linspace(10, 0, 12), np.zeros(12)).astype(np.float64)


def test_clean_slope_has_no_pits():
    assert not find_unresolvable_cells(_slope()).any()


def test_single_pit_is_found():
    d = _slope()
    d[5, 5] -= 3.0
    m = find_unresolvable_cells(d)
    assert m[5, 5] and m.sum() == 1


def test_flat_plateau_interior_flagged():
    d = _slope()
    d[4:8, 4:8] = d[4, 4]
    m = find_unresolvable_cells(d)
    assert m.any()


def test_diagnose_reports_untouched_pit():
    d = _slope()
    d[5, 5] -= 3.0
    cl = diagnose(d, d.copy(), None)
    assert (
        len(cl) == 1
        and cl[0].original_changed is False
        and (cl[0].row, cl[0].col) == (5, 5)
    )


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
