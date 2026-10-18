import numpy as np

from mayim_tools.hydrology.dem_hydro_processing.core.diagnostics import (
    find_unresolvable_cells,
)
from mayim_tools.hydrology.dem_hydro_processing.core.drainage_enforcement import (
    enforce_drainage,
)


def _slope():
    return np.add.outer(np.linspace(10, 0, 14), np.zeros(14)).astype(np.float32)


def test_pit_is_resolved_and_float32_safe():
    d = _slope()
    d[6, 6] -= 3.0
    out, n = enforce_drainage(d)
    assert n > 0 and not find_unresolvable_cells(np.asarray(out, np.float32)).any()


def test_already_draining_dem_unchanged():
    d = _slope()
    out, n = enforce_drainage(d)
    assert n == 0 and np.array_equal(np.asarray(out, np.float32), d)


def test_protected_sink_is_left_alone():
    d = _slope()
    d[6, 6] -= 3.0
    out, _ = enforce_drainage(d, sink_cells=[(6, 6)])
    assert np.float32(out[6, 6]) == d[6, 6]


def test_large_flat_resolves():
    d = _slope()
    d[3:11, 3:11] = d[3, 3]
    out, _ = enforce_drainage(d)
    assert not find_unresolvable_cells(np.asarray(out, np.float32)).any()


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
