"""Unit tests for geometry/pump_stations/core.py against a synthetic
HDF5 fixture - no QGIS, no real HEC-RAS project needed.
"""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras.import_ras_2d_data.pump_stations.core import (  # noqa: E402
    read_pump_stations,
)


@pytest.fixture
def geometry_hdf_with_pump_station(tmp_path):
    path = tmp_path / "geometry.hdf"
    with h5py.File(path, "w") as f:
        group = f.create_group("Geometry/Pump Stations")
        attrs_dtype = np.dtype(
            [("Name", "S32"), ("Inlet SA/2D", "S16"), ("Pump Groups", "i4")]
        )
        attrs = np.array([("Pump Station #1", "Perimeter 1", 1)], dtype=attrs_dtype)
        group.create_dataset("Attributes", data=attrs)
        group.create_dataset("Points", data=np.array([[5.0, 6.0]], dtype="f8"))
    return path


def test_read_pump_stations(geometry_hdf_with_pump_station):
    stations = read_pump_stations(str(geometry_hdf_with_pump_station))
    assert len(stations) == 1
    assert stations[0].name == "Pump Station #1"
    assert stations[0].xy == (5.0, 6.0)
    assert stations[0].attributes["Pump Groups"] == 1


def test_read_pump_stations_absent_group_returns_empty(tmp_path):
    path = tmp_path / "no_pumps.hdf"
    with h5py.File(path, "w") as f:
        f.create_group("Geometry/2D Flow Areas")
    assert read_pump_stations(str(path)) == []
