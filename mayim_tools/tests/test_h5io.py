"""The fallback HDF5 readers give the same results as h5py. They are used
when h5py cannot load inside QGIS (a DLL clash with QGIS's own HDF5):
first h5py in a separate Python process, then GDAL's HDF5 driver."""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras._common import h5io  # noqa: E402


@pytest.fixture
def hdf(tmp_path):
    path = tmp_path / "g.hdf"
    with h5py.File(path, "w") as f:
        g = f.create_group("Geometry/2D Flow Areas")
        dt = np.dtype([("Name", "S16"), ("Cell Count", "<i4"), ("Mann", "<f4")])
        g.create_dataset(
            "Attributes", data=np.array([(b"Area 1 ", 10, 0.03), (b"B", 5, 0.05)], dt)
        )
        a = g.create_group("Area 1")
        a.create_dataset("Perimeter", data=np.arange(10.0).reshape(5, 2))
        a.create_dataset("Idx", data=np.array([[0, 1, -1], [2, 3, 4]], "i4"))
        f.create_dataset("Geometry/Names", data=np.array([b"a", b"bb"], "S4"))
        vlen = np.dtype(
            [("Desc", h5py.string_dtype()), ("N", "<i4"), ("XY", "<f8", (2,))]
        )
        f.create_dataset(
            "Geometry/Notes",
            data=np.array([("first note", 1, (1, 2)), ("x", 2, (3, 4))], vlen),
        )
        f.create_dataset("Results/Big", data=np.zeros(3))  # outside Geometry
    return str(path)


@pytest.fixture
def external(monkeypatch):
    """No h5py in this process: the separate-process reader is used."""
    monkeypatch.setattr(h5io, "_H5PY", False)
    monkeypatch.setattr(h5io, "_EXTERNAL", None)
    h5io.clear_cache()
    h5io.NOTES.clear()
    if h5io.backend() != "h5py-external":
        pytest.skip("no Python with h5py found for the separate process")
    yield
    h5io.clear_cache()


def test_external_copy_matches_h5py(hdf, external):
    with h5io.open_h5(hdf) as g, h5py.File(hdf, "r") as h:
        assert type(g).__name__ == "CopyH5File"
        assert "Geometry/2D Flow Areas/Area 1" in g
        assert "Geometry/Missing" not in g and "Results/Big" not in g
        areas = g["Geometry/2D Flow Areas"]
        assert areas.keys() == sorted(h["Geometry/2D Flow Areas"].keys())
        assert [k for k in areas.keys() if h5io.is_group(areas[k])] == ["Area 1"]
        for name in ("Perimeter", "Idx"):
            got = g["Geometry/2D Flow Areas/Area 1"][name][()]
            exp = h["Geometry/2D Flow Areas/Area 1"][name][()]
            np.testing.assert_array_equal(got, exp)
            assert got.dtype == exp.dtype
        rows = g["Geometry/2D Flow Areas/Attributes"][()]
        assert rows.dtype.names == ("Name", "Cell Count", "Mann")
        assert rows["Name"].tolist() == [b"Area 1 ", b"B"]
        assert rows["Cell Count"].tolist() == [10, 5]
        assert g["Geometry/Names"][()].tolist() == [b"a", b"bb"]
        notes = g["Geometry/Notes"][()]
        assert notes["Desc"].tolist() == [b"first note", b"x"]
        assert notes["XY"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert any("separate Python process" in n for n in h5io.NOTES)


def test_external_copy_is_made_once(hdf, external):
    calls = []
    real = h5io._run

    def counting(args, timeout):
        calls.append(args)
        return real(args, timeout)

    import unittest.mock

    with unittest.mock.patch.object(h5io, "_run", counting):
        for _ in range(3):
            with h5io.open_h5(hdf) as g:
                assert "Geometry/Names" in g
    assert len(calls) == 1


def test_readers_through_external_copy(hdf, external):
    from mayim_tools.hec_ras._common.hdf5_utils import read_attribute_table
    from mayim_tools.hec_ras.import_ras_2d_data.mesh_2d.core import (
        list_2d_flow_area_names,
        read_2d_flow_area_perimeter,
    )

    with h5io.open_h5(hdf) as f:
        rows = read_attribute_table(f, "Geometry/2D Flow Areas/Attributes")
    assert rows[0]["Name"] == "Area 1" and rows[1]["Cell Count"] == 5
    assert list_2d_flow_area_names(hdf) == ["Area 1"]
    assert read_2d_flow_area_perimeter(hdf, "Area 1")[1] == (2.0, 3.0)


def test_gdal_reader(hdf):
    pytest.importorskip("osgeo.gdal")
    h5io.NOTES.clear()
    with h5io.open_h5(hdf, force_gdal=True) as g, h5py.File(hdf, "r") as h:
        assert "Geometry/2D Flow Areas/Area 1" in g
        assert "Geometry/Missing" not in g
        areas = g["Geometry/2D Flow Areas"]
        assert [k for k in areas.keys() if h5io.is_group(areas[k])] == ["Area 1"]
        for name in ("Perimeter", "Idx"):
            got = g["Geometry/2D Flow Areas/Area 1"][name][()]
            exp = h["Geometry/2D Flow Areas/Area 1"][name][()]
            np.testing.assert_array_equal(got, exp)
        rows = g["Geometry/2D Flow Areas/Attributes"][()]
        assert rows["Cell Count"].tolist() == [10, 5]
        name = rows["Name"][0].strip()
        # GDAL 3.13 returns only the first character of fixed-length text in
        # tables; the reader then says so
        assert name == b"Area 1" or (
            name == b"A" and any("first character" in n for n in h5io.NOTES)
        )


class _Shape:
    shape = (2,)


@pytest.mark.parametrize(
    "values, expected",
    [
        (["Area 1", "B"], [b"Area 1", b"B"]),
        (np.array([65, 66], np.uint8), [b"A", b"B"]),  # GDAL 3.13
        (
            np.array([list(b"Area 1\0\0"), list(b"B\0\0\0\0\0\0\0")], np.uint8),
            [b"Area 1", b"B"],
        ),
    ],
)
def test_gdal_text_forms(values, expected):
    obj = h5io.GdalH5Dataset.__new__(h5io.GdalH5Dataset)
    h5io.GdalH5Dataset.shape = property(lambda self: (2,))
    try:
        assert obj._strings(values).tolist() == expected
    finally:
        del h5io.GdalH5Dataset.shape
