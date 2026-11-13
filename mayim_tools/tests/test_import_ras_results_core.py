"""Import RAS result data: reading a 2D plan's results, cell velocities,
profile line flows by mass balance, result rasters and the CSV - on a small
synthetic plan file in the HEC-RAS 7.0 layout (checked against project
2625038)."""

from __future__ import annotations

import math

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mayim_tools.hec_ras.import_ras_results import core  # noqa: E402
from mayim_tools.hec_ras.import_ras_results.rasters import (  # noqa: E402
    hazard_class,
)

N = 6  # real cells in a row, 10 m x 10 m, flow in +x
A = 100.0  # cell plan area
T = 61  # output times, 1 minute apart


def _bed(i):
    return 100.0 - 0.1 * i


def _series():
    t_s = np.arange(T) * 60.0
    q_in = 2.0 * np.sin(np.pi * np.clip(t_s / 1800.0, 0, 1)) ** 2  # m3/s
    # stored depth rises later and less downstream
    depth = np.zeros((T, N))
    for i in range(N):
        lag = 120.0 * i
        depth[:, i] = (
            0.3
            * (1 - 0.1 * i)
            * np.sin(np.pi * np.clip((t_s - lag) / 2400.0, 0, 1)) ** 2
        )
    wse = depth + np.array([_bed(i) for i in range(N)])[None, :]
    vol = A * depth
    dvdt = np.gradient(vol, t_s, axis=0)
    # flow across face i -> i+1: inflow minus storage upstream of it
    q_face = q_in[:, None] - np.cumsum(dvdt, axis=1)  # (T, N): face after cell i
    return t_s, q_in, depth, wse, q_face


@pytest.fixture
def plan_hdf(tmp_path):
    t_s, q_in, depth, wse, q_face = _series()
    path = tmp_path / "proj.p01.hdf"
    with h5py.File(path, "w") as f:
        f.attrs["Projection"] = b'PROJCS["test"]'
        g = f.create_group("Geometry")
        g.attrs["Terrain Filename"] = b".\\Terrain\\T.hdf"
        areas = g.create_group("2D Flow Areas")
        dt = np.dtype([("Name", "S16"), ("Cell Count", "<i4")])
        areas.create_dataset("Attributes", data=np.array([(b"Area 1", N)], dt))
        a = areas.create_group("Area 1")
        centers = [(10 * i + 5.0, 5.0) for i in range(N)] + [(10 * N + 5.0, 5.0)]
        a.create_dataset("Cells Center Coordinate", data=np.array(centers))
        fp = [(10.0 * i, 0.0) for i in range(N + 1)] + [
            (10.0 * i, 10.0) for i in range(N + 1)
        ]
        a.create_dataset("FacePoints Coordinate", data=np.array(fp))
        ring = [[i, i + 1, N + 1 + i + 1, N + 1 + i] for i in range(N)] + [
            [N, N, -1, -1]
        ]
        a.create_dataset("Cells FacePoint Indexes", data=np.array(ring, "i4"))
        a.create_dataset(
            "Cells Minimum Elevation",
            data=np.array([_bed(i) for i in range(N)] + [np.nan], "f4"),
        )
        a.create_dataset("Cells Surface Area", data=np.full(N + 1, A, "f4"))
        # faces: i -> i+1 for i < N-1, and the outflow face N-1 -> virtual N
        faces = [(i, i + 1) for i in range(N)]
        a.create_dataset("Faces Cell Indexes", data=np.array(faces, "i4"))
        a.create_dataset(
            "Faces NormalUnitVector and Length",
            data=np.array([(1.0, 0.0, 10.0)] * N, "f4"),
        )
        a.create_dataset(
            "Faces FacePoint Indexes",
            data=np.array([(i + 1, N + 1 + i + 1) for i in range(N)], "i4"),
        )
        cf_info, cf_vals = [], []
        for c in range(N):
            fs = ([c - 1] if c > 0 else []) + [c]
            cf_info.append((len(cf_vals), len(fs)))
            cf_vals += [(fi, 1) for fi in fs]
        cf_info.append((len(cf_vals), 1))
        cf_vals.append((N - 1, -1))
        a.create_dataset(
            "Cells Face and Orientation Info", data=np.array(cf_info, "i4")
        )
        a.create_dataset(
            "Cells Face and Orientation Values", data=np.array(cf_vals, "i4")
        )
        fpc_info, fpc_vals = [], []
        for p in range(2 * (N + 1)):
            i = p % (N + 1)
            cells = [c for c in (i - 1, i) if 0 <= c < N]
            fpc_info.append((len(fpc_vals), len(cells)))
            fpc_vals += cells
        a.create_dataset("FacePoints Cell Info", data=np.array(fpc_info, "i4"))
        a.create_dataset("FacePoints Cell Index Values", data=np.array(fpc_vals, "i4"))
        ve_info, ve_vals = [], []
        for c in range(N + 1):
            z = _bed(min(c, N - 1))
            ve_info.append((len(ve_vals), 2))
            ve_vals += [(z, 0.0), (z + 10.0, 10.0 * A)]
        a.create_dataset("Cells Volume Elevation Info", data=np.array(ve_info, "i4"))
        a.create_dataset("Cells Volume Elevation Values", data=np.array(ve_vals, "f4"))
        pi = f.create_group("Plan Data/Plan Information")
        pi.attrs["Plan ShortID"] = b"2D_flow - test"
        pi.attrs["Flow Title"] = b"test"
        base = "Results/Unsteady/Output/Output Blocks/Base Output"
        ts = f.create_group(f"{base}/Unsteady Time Series")
        ts.create_dataset("Time", data=t_s / 86400.0)
        ar = ts.create_group("2D Flow Areas/Area 1")
        wse_all = np.c_[wse, wse[:, -1]]  # virtual cell
        ar.create_dataset("Water Surface", data=wse_all.astype("f4"))
        d_face = np.maximum(depth, 0.01)
        ar.create_dataset("Face Velocity", data=(q_face / (10.0 * d_face)).astype("f4"))
        bc = ar.create_group("Boundary Conditions")
        inflow = bc.create_dataset("Inflow", data=np.c_[np.zeros(T), q_in])
        inflow.attrs["Columns"] = np.array([b"Stage", b"Flow"])
        per_cell = bc.create_dataset("Inflow - Flow per Cell", data=q_in[:, None])
        per_cell.attrs["Cells"] = np.array([0], "i4")
        out = bc.create_dataset("Outflow - Flow per Face", data=q_face[:, -1:])
        out.attrs["Faces"] = np.array([N - 1], "i4")
        sm = f.create_group(f"{base}/Summary Output/2D Flow Areas/Area 1")
        k = np.argmax(wse_all, axis=0)
        sm.create_dataset(
            "Maximum Water Surface",
            data=np.vstack([wse_all.max(axis=0), t_s[k] / 86400.0]).astype("f4"),
        )
    return str(path), q_face, t_s


def test_hazard_classes():
    d = np.array([0.1, 0.4, 0.6, 1.3, 0.2, 2.5, 0.1, 5.0, np.nan])
    v = np.array([0.5, 0.5, 0.1, 0.1, 2.5, 0.1, 5.0, 0.1, 1.0])
    assert hazard_class(d, v).tolist() == [1, 2, 3, 4, 5, 5, 6, 6, 0]


def test_cell_velocity_from_face_velocities():
    normals = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
    lengths = np.array([2.0, 1.0, 2.0, 1.0])
    vel = np.array([1.5, -0.5])
    u = (normals @ vel)[None, :]  # one time step, four faces
    out = core.reconstruct_cell_velocity(u, normals, lengths, [np.arange(4)])
    np.testing.assert_allclose(out[0, 0], vel, atol=1e-6)


def test_face_crossings_sign():
    centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    faces = np.array([[0, 1]])
    # drawn left bank -> right bank looking downstream (+x): from y=+5 to y=-5
    f, s = core.face_crossings([(5, 5), (5, -5)], centers, faces, 2)
    assert f.tolist() == [0] and s.tolist() == [1.0]
    f, s = core.face_crossings([(5, -5), (5, 5)], centers, faces, 2)
    assert s.tolist() == [-1.0]
    f, s = core.face_crossings([(20, 5), (20, -5)], centers, faces, 2)
    assert len(f) == 0


def test_plan_results_and_profile_flow(plan_hdf):
    path, q_face, t_s = plan_hdf
    lines = [
        core.ProfileLine("A", [(20.0, 20.0), (20.0, -10.0)]),  # face 1 -> 2
        core.ProfileLine("B", [(40.0, -10.0), (40.0, 20.0)]),  # drawn reversed
    ]
    plan = core.read_plan_results(path, profile_lines=lines)
    assert plan.short_id == "2D_flow - test" and plan.flow_title == "test"
    area = plan.areas[0]
    assert len(area.centers) == N  # the virtual cell is left out
    assert area.max_depth[0] == pytest.approx(0.3, abs=1e-3)
    assert np.isfinite(area.arrival_h).all()
    assert (area.duration_h > 0).all()
    a = area.profiles[0]
    k = int(np.argmax(np.abs(q_face[:, 1])))
    assert a["q_peak"] == pytest.approx(q_face[k, 1], rel=1e-3)
    assert a["t_q_peak_h"] == pytest.approx(t_s[k] / 3600.0)
    assert a["note"] == "" and a["faces"] == 1
    b = area.profiles[1]
    assert b["q_peak"] == pytest.approx(-q_face[:, 3].max(), rel=1e-3)
    assert b["wse_max"] == pytest.approx(area.max_wse[3:5].max())


def test_profile_line_that_water_can_pass_around(plan_hdf):
    path, _, _ = plan_hdf
    plan = core.read_plan_results(
        path, profile_lines=[core.ProfileLine("C", [(20.0, 20.0), (20.0, -10.0)])]
    )
    assert plan.areas[0].profiles[0]["note"] == ""
    # a line along the row crosses no centre-to-centre segment
    plan = core.read_plan_results(
        path, profile_lines=[core.ProfileLine("D", [(1.0, 5.0), (59.0, 5.0)])]
    )
    assert plan.areas[0].profiles == {}


def test_not_separating_line_gets_a_note():
    # a 2 x 2 block of wet cells; a line cutting only the lower pair's face
    centers = np.array([[5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]])
    faces = np.array([[0, 1], [2, 3], [0, 2], [1, 3]])

    class Geo(dict):
        pass

    geo = Geo({"Cells Center Coordinate": centers, "Faces Cell Indexes": faces})
    res = core.profile_results(
        [core.ProfileLine("E", [(10.0, 8.0), (10.0, 2.0)])], geo, 4,
        np.ones((3, 4)), np.full(4, 101.0), np.full(4, 100.0), np.ones(4),
        np.arange(3.0), 0.001, [], 0.0,
    )  # fmt: skip
    assert math.isnan(res[0]["q_peak"])
    assert "pass around" in res[0]["note"]


def test_result_grids_and_terrain(plan_hdf, tmp_path):
    gdal = pytest.importorskip("osgeo.gdal")
    from mayim_tools.hec_ras.import_ras_results.rasters import build_grids

    path, _, _ = plan_hdf
    terrain = tmp_path / "Terrain"
    terrain.mkdir()
    tif = terrain / "T.vrt"
    nx, ny = N * 10, 10
    x = np.arange(nx) + 0.5
    z = np.tile(np.array([_bed(int(xx // 10)) for xx in x]), (ny, 1))
    z[:, :] += 0.0
    tmp = str(terrain / "T.tif")
    ds = gdal.GetDriverByName("GTiff").Create(tmp, nx, ny, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, 10, 0, -1))
    ds.GetRasterBand(1).WriteArray(z)
    ds = None
    gdal.BuildVRT(str(tif), [tmp]).FlushCache()
    plan = core.read_plan_results(path)
    assert core.find_terrain(plan) == str(tif)
    for method in ("horizontal", "sloping"):
        g = build_grids(plan, str(tif), method=method)
        wet = np.isfinite(g.depth)
        assert wet.all()  # every cell is wet and flat-bottomed
        # flat-bottomed cells are fully under water: depth = cell depth there
        np.testing.assert_allclose(
            np.nanmax(g.depth), plan.areas[0].max_depth.max(), atol=0.05
        )
        assert set(np.unique(g.hazard)) <= {1, 2, 3, 4, 5, 6}


def test_rasmap_profile_lines(tmp_path):
    (tmp_path / "Features").mkdir()
    (tmp_path / "Features" / "Profile Lines.shp").write_bytes(b"")
    (tmp_path / "proj.rasmap").write_text(
        '<Layer Name="Profile Lines" Type="PolylineFeatureLayer" '
        'Filename=".\\Features\\Profile Lines.shp" />'
    )
    found = core.rasmap_profile_lines(str(tmp_path / "proj.p01.hdf"))
    assert found and found.endswith("Profile Lines.shp")
    assert core.rasmap_profile_lines(str(tmp_path / "other.p01.hdf")) is None


def test_peaks_csv(plan_hdf, tmp_path):
    from mayim_tools.hec_ras.import_ras_results.export import write_peaks_csv

    path, _, _ = plan_hdf
    lines = [core.ProfileLine("A", [(20.0, 20.0), (20.0, -10.0)])]
    plan = core.read_plan_results(path, profile_lines=lines)
    out = tmp_path / "p.csv"
    write_peaks_csv(str(out), plan, lines, {"water surface": "sloping"})
    text = out.read_text(encoding="utf-8-sig")
    for section in ("[run]", "[structures]", "[boundary conditions]",
                    "[profile lines]"):  # fmt: skip
        assert section in text
    assert "Inflow,Area 1,Flow" in text
    row = [r for r in text.splitlines() if r.startswith("A,Area 1,")][0]
    assert float(row.split(",")[2]) > 0
    assert "nan" not in text.lower().replace("unstable,nan", "")


def test_results_through_separate_process(plan_hdf, monkeypatch):
    """Inside QGIS h5py often cannot load: the results (and attributes) are
    then read through the separate-process copy, with the same values."""
    from mayim_tools.hec_ras._common import h5io

    path, _, _ = plan_hdf
    lines = [core.ProfileLine("A", [(20.0, 20.0), (20.0, -10.0)])]
    direct = core.read_plan_results(path, profile_lines=lines)
    monkeypatch.setattr(h5io, "_H5PY", False)
    monkeypatch.setattr(h5io, "_EXTERNAL", None)
    h5io.clear_cache()
    if h5io.backend() != "h5py-external":
        pytest.skip("no Python with h5py found for the separate process")
    try:
        copied = core.read_plan_results(path, profile_lines=lines)
    finally:
        h5io.clear_cache()
    assert copied.short_id == direct.short_id == "2D_flow - test"
    assert copied.projection_wkt == direct.projection_wkt
    np.testing.assert_allclose(copied.areas[0].max_wse, direct.areas[0].max_wse)
    assert copied.areas[0].profiles[0]["q_peak"] == pytest.approx(
        direct.areas[0].profiles[0]["q_peak"]
    )
    assert [b["name"] for b in copied.boundaries] == ["Inflow", "Inflow"]
